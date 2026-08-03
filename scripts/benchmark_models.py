"""
Benchmark every available local model against the DeepSeek API baseline on the
same contracts, the same retrieved context, and the same eval.

Retrieval is model-independent, so contexts are built ONCE and cached. Every
model then sees byte-identical input and the comparison isolates the LLM.

Usage:
  python scripts/benchmark_models.py --contracts 12 --list
  python scripts/benchmark_models.py --contracts 12                 # all local + deepseek
  python scripts/benchmark_models.py --contracts 12 --models mistral:7b-instruct-v0.3-q4_K_M
  python scripts/benchmark_models.py --contracts 12 --json-mode     # constrained decoding
  python scripts/benchmark_models.py --report                       # rebuild report from cache
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ingestion import load_contract, clean_text
from pipeline.chunker import chunk_contract
from pipeline.embedder import build_per_doc_index
from pipeline.retriever import retrieve_sla_chunks
from pipeline.extractor import extract_sla
from pipeline.llm import list_ollama_models
from pipeline.schemas import SLAClause
from evals.eval_runner import score_extraction
from config import DEEPSEEK_MODEL

BENCH_DIR = Path("output/benchmarks")
CONTEXTS_PATH = BENCH_DIR / "contexts.json"
RESULTS_PATH = BENCH_DIR / "results.json"
CORPUS = Path("data/cuad_raw")

# CUAD's own datasheet, not a contract — the eval excludes it, so do we.
EXCLUDE = {"cuad_0000"}

# Our largest retrieved context is ~23K tokens. Ollama defaults to 4096, which
# silently truncates the contract out of the prompt — the model then scores 0.0
# and looks incompetent when it simply never saw the text. Give every model the
# full prompt, clamped to whatever its own architecture allows.
TARGET_NUM_CTX = 24576

# Reasoning models emit chain-of-thought before the answer. At the pipeline's
# 1500-token cap they run out mid-thought and return nothing, which measures our
# budget rather than their ability — so they get a bigger allowance.
THINKING_MAX_TOKENS = 4000


# ── Contract sample ────────────────────────────────────────────────────────────

def pick_contracts(n: int) -> list[Path]:
    """Evenly spaced across the sorted corpus — deterministic, no cherry-picking."""
    files = sorted(f for f in CORPUS.glob("*.txt") if f.stem not in EXCLUDE)
    if not files:
        raise SystemExit(f"No contracts in {CORPUS}. Run scripts/download_cuad.py first.")
    if n >= len(files):
        return files
    stride = len(files) / n
    return [files[int(i * stride)] for i in range(n)]


def build_contexts(files: list[Path]) -> dict:
    """Run ingestion→chunk→embed→retrieve once per contract and cache the result."""
    cache = json.loads(CONTEXTS_PATH.read_text()) if CONTEXTS_PATH.exists() else {}
    todo = [f for f in files if f.stem not in cache]
    if not todo:
        print(f"Contexts: {len(files)} cached")
        return cache

    print(f"Building retrieval context for {len(todo)} contracts...")
    for i, f in enumerate(todo, 1):
        t0 = time.perf_counter()
        text = clean_text(load_contract(str(f)))
        chunks = chunk_contract(text)
        index = build_per_doc_index(f.stem, chunks)
        context = retrieve_sla_chunks(index)
        cache[f.stem] = {
            "file_path": str(f),
            "context": context,
            "source_chars": len(text),
            "chunks": len(chunks),
            "retrieval_s": round(time.perf_counter() - t0, 2),
        }
        print(f"  [{i}/{len(todo)}] {f.stem}: {len(chunks)} chunks -> "
              f"{len(context):,} chars ({cache[f.stem]['retrieval_s']}s)")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_PATH.write_text(json.dumps(cache, indent=2))
    return cache


# ── Benchmark ──────────────────────────────────────────────────────────────────

def run_model(model_id: str, provider: str, contexts: dict, contract_ids: list[str],
              results: dict, json_mode: bool, num_ctx: int | None = None,
              thinking: bool = False) -> None:
    key = f"{model_id}|json" if json_mode else model_id
    results.setdefault(key, {"provider": provider, "model": model_id,
                             "json_mode": json_mode, "num_ctx": num_ctx, "runs": {}})
    runs = results[key]["runs"]

    pending = [c for c in contract_ids if c not in runs]
    if not pending:
        print(f"  {key}: cached")
        return

    print(f"\n{'=' * 70}\n{key}  ({provider}){'  [json mode]' if json_mode else ''}\n{'=' * 70}")
    for i, cid in enumerate(pending, 1):
        entry = contexts[cid]
        t0 = time.perf_counter()
        res = extract_sla(cid, entry["file_path"], entry["context"],
                          provider=provider, model=model_id, json_mode=json_mode,
                          num_ctx=num_ctx,
                          max_tokens=THINKING_MAX_TOKENS if thinking else None)
        wall = time.perf_counter() - t0

        row = res.sla.model_dump()
        ev = score_extraction(cid, entry["file_path"], row)
        populated = sum(1 for v in row.values() if v is not None)

        runs[cid] = {
            "status": res.status,
            "score": round(ev["overall"], 4),
            "field_scores": {k: v for k, v in ev["scores"].items()},
            "populated": populated,
            "missed": ev["missed"],
            "tokens": res.tokens_used,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            # Ollama silently drops overflow; hitting the ceiling means the
            # contract was clipped, so the score reflects the window not the model
            "truncated": bool(res.num_ctx and res.tokens_used >= res.num_ctx - 8),
            "effective_num_ctx": res.num_ctx,
            "latency_s": round(wall, 2),
            "error": res.error,
        }
        flag = "OK " if res.status == "success" else ("PART" if res.status == "partial" else "FAIL")
        err = f"  {res.error[:70]}" if res.error else ""
        if runs[cid]["truncated"]:
            err += "  [TRUNCATED]"
        print(f"  [{i}/{len(pending)}] {cid}  {flag}  score={ev['overall']:.3f}  "
              f"fields={populated}/19  {wall:.1f}s{err}")
        save(results)


def save(results: dict) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


# ── Aggregation ────────────────────────────────────────────────────────────────

def summarise(results: dict, contexts: dict) -> list[dict]:
    rows = []
    for key, block in results.items():
        runs = block["runs"]
        if not runs:
            continue
        scores = [r["score"] for r in runs.values()]
        lat = [r["latency_s"] for r in runs.values()]
        ok = sum(1 for r in runs.values() if r["status"] == "success")
        failed = sum(1 for r in runs.values() if r["status"] == "failed")
        partial = sum(1 for r in runs.values() if r["status"] == "partial")
        n = len(runs)
        completion = [r["tokens"] for r in runs.values() if r["tokens"]]
        rows.append({
            "key": key,
            "model": block["model"],
            "provider": block["provider"],
            "json_mode": block.get("json_mode", False),
            "n": n,
            "avg_score": sum(scores) / n,
            "score_gte_07": sum(1 for s in scores if s >= 0.7),
            "success_rate": ok / n,
            "partial": partial,
            "failed": failed,
            "avg_fields": sum(r["populated"] for r in runs.values()) / n,
            "avg_latency": sum(lat) / n,
            "median_latency": sorted(lat)[n // 2],
            "avg_tokens": (sum(completion) / len(completion)) if completion else 0,
            "truncated": sum(1 for r in runs.values() if r.get("truncated")),
            "num_ctx": block.get("num_ctx") or 0,
        })
    return sorted(rows, key=lambda r: -r["avg_score"])


def per_field_table(results: dict) -> dict:
    """avg score per field per model, for the strengths/weaknesses section."""
    out = {}
    for key, block in results.items():
        acc = {}
        for r in block["runs"].values():
            for f, s in (r.get("field_scores") or {}).items():
                if s is not None:
                    acc.setdefault(f, []).append(s)
        out[key] = {f: sum(v) / len(v) for f, v in acc.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", type=int, default=12)
    ap.add_argument("--models", nargs="*", help="Specific model ids (default: all local)")
    ap.add_argument("--skip-deepseek", action="store_true")
    ap.add_argument("--json-mode", action="store_true", help="Constrained JSON decoding")
    ap.add_argument("--list", action="store_true", help="List local models and exit")
    ap.add_argument("--report", action="store_true", help="Rebuild report from cache only")
    args = ap.parse_args()

    if args.list:
        for m in list_ollama_models():
            tags = " ".join(t for t, on in
                            (("thinking", m["thinking"]), ("vision", m["vision"])) if on)
            print(f"{m['name']:<42} {m['size_gb']:>6.2f}GB  {m['params']:>6}  "
                  f"{m['quant']:<8} ctx={m['context_length']}  {tags}")
        return

    files = pick_contracts(args.contracts)
    contract_ids = [f.stem for f in files]
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}

    if args.report:
        contexts = json.loads(CONTEXTS_PATH.read_text())
        write_report(results, contexts, contract_ids)
        return

    contexts = build_contexts(files)

    catalog = {m["name"]: m for m in list_ollama_models()}
    local = args.models or [m["name"] for m in catalog.values() if not m["vision_only"]]
    targets = [(m, "ollama") for m in local]
    if not args.skip_deepseek:
        targets.append((DEEPSEEK_MODEL, "api"))

    def ctx_for(model_id: str) -> int | None:
        info = catalog.get(model_id)
        if not info:
            return None
        limit = info.get("context_length") or TARGET_NUM_CTX
        return min(TARGET_NUM_CTX, limit)

    print(f"\n{len(targets)} models x {len(contract_ids)} contracts")
    t0 = time.perf_counter()
    for model_id, provider in targets:
        try:
            nc = ctx_for(model_id) if provider == "ollama" else None
            thinking = bool(catalog.get(model_id, {}).get("thinking"))
            run_model(model_id, provider, contexts, contract_ids, results,
                      args.json_mode, num_ctx=nc, thinking=thinking)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  {model_id}: ABORTED — {type(e).__name__}: {e}")
        save(results)
    print(f"\nTotal benchmark wall time: {(time.perf_counter() - t0) / 60:.1f} min")

    write_report(results, contexts, contract_ids)


def write_report(results: dict, contexts: dict, contract_ids: list[str]) -> None:
    rows = summarise(results, contexts)
    fields = per_field_table(results)

    out = ["# Local Model Benchmark — SLA Extraction", ""]
    out.append(f"Contracts: {len(contract_ids)} | identical cached retrieval context per contract")
    out.append("")
    out.append("| Model | Provider | num_ctx | Score | >=0.7 | Success | Partial | Fail | Trunc | Fields | Median s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| `{r['key']}` | {r['provider']} | {r['num_ctx'] or '-'} | **{r['avg_score']:.3f}** | "
            f"{r['score_gte_07']}/{r['n']} | {r['success_rate']:.0%} | {r['partial']} | "
            f"{r['failed']} | {r['truncated']} | {r['avg_fields']:.1f} | "
            f"{r['median_latency']:.1f} |"
        )
    out.append("")

    baseline = next((r for r in rows if r["provider"] == "api"), None)
    if baseline:
        out.append(f"Baseline: `{baseline['key']}` at {baseline['avg_score']:.3f}, "
                   f"{baseline['median_latency']:.1f}s median")
        out.append("")
        out.append("| Model | Score vs baseline | Speed vs baseline |")
        out.append("|---|---|---|")
        for r in rows:
            if r["provider"] == "api":
                continue
            d = r["avg_score"] - baseline["avg_score"]
            spd = baseline["median_latency"] / r["median_latency"] if r["median_latency"] else 0
            out.append(f"| `{r['key']}` | {d:+.3f} ({d / baseline['avg_score']:+.0%}) | {spd:.2f}x |")
        out.append("")

    out.append("## Per-field average score")
    out.append("")
    all_fields = sorted({f for v in fields.values() for f in v})
    out.append("| Field | " + " | ".join(f"`{r['key'].split(':')[0]}`" for r in rows) + " |")
    out.append("|---|" + "---|" * len(rows))
    for f in all_fields:
        cells = " | ".join(f"{fields[r['key']].get(f, float('nan')):.2f}" for r in rows)
        out.append(f"| {f} | {cells} |")

    path = BENCH_DIR / "BENCHMARK_REPORT.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nReport: {path}")
    print("\n".join(out[:6 + len(rows)]))


if __name__ == "__main__":
    main()
