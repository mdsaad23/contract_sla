# 100-Contract Random Benchmark: Trimmed Roster + DeepSeek V4 + 32K Context

> Status: **planned, not yet implemented.** Extends the 12-contract benchmark documented in
> [`LOCAL_MODELS.md`](LOCAL_MODELS.md). Written 2026-08-03.

## Context

The existing benchmark ([`docs/LOCAL_MODELS.md`](LOCAL_MODELS.md)) covers 13 models on **12 evenly-spaced** contracts. At n=12 the noise floor is ±0.05, too wide to separate the mid-tier. n=100 drops it to ~±0.02 for ~4 h of wall clock instead of ~16 h for the full 510.

Four things force changes beyond raising `--contracts`:

1. **`deepseek-chat` is dead.** DeepSeek retired the `deepseek-chat` / `deepseek-reasoner` aliases on **2026-07-24**. `config.py:7` still points at it, so the API baseline is currently broken. The replacement `deepseek-v4-pro` **thinks by default at `high` effort**, which will silently eat the 1500-token output budget and return empty content — the same failure mode already documented for Ollama reasoning models. Thinking must be explicitly disabled.
2. **Sampling is deterministic stride, not random** (`scripts/benchmark_models.py:56-64`).
3. **24576 is too small — ~7% of contracts are being silently truncated right now.** Quantified below. This is measurement bug #1 recurring.
4. **The embedder is the weakest link and nobody has measured it.** Every model is capped by what retrieval hands it.

All new artifacts are written to **new paths**. Nothing existing is deleted or overwritten.

---

## Q: Is 24576 enough? — No. Here are the numbers.

Corpus, 510 contracts (`data/cuad_raw`, excluding `cuad_0000`):

| | KB |
|---|---|
| p25 | 16.0 |
| p50 | 31.7 |
| p75 | 64.2 |
| p90 | 118.7 |
| p95 | 160.2 |
| p99 | 263.0 |
| max | 297.8 |

Retrieval (`pipeline/retriever.py:41-55`) runs **19 fixed queries × top_k 5**, dedupes, and joins with **no length cap at all** (`:57`). So context size is bounded only by the 95-unique-chunk ceiling. From the 12 cached contexts, chunks average **1,214 chars** (SentenceSplitter breaks on sentence boundaries, so chunks land well under the 512-token nominal).

- **Hard ceiling:** 95 × 1,214 ≈ 115K chars ≈ **28.8K tokens**.
- **Exceeds 24576 tokens** (≈98K chars ≈ 81 chunks) once source ≥ ~145 KB → **~7% of the corpus**, i.e. **~7 of the 100 sampled contracts**.

What happens to those 7 today: `pipeline/llm.py:100-108` catches the prefill OOM and **halves the window to 12288**, cutting context in half for precisely the hardest contracts. The 12-contract sample topped out at 19K tokens and never triggered it — that was luck, not headroom.

**`num_ctx = 32768` covers 100% of the corpus** (32768 tokens ≈ 131K chars > the 115K ceiling). Zero truncation, no per-contract special cases.

---

## Q: Are smaller quantizations worth it? — Yes, and it's what makes 32K possible.

Budget **15.9 GiB** (AMD RX 9070 XT, ROCm, Windows 11). KV cache = `2 × layers × kv_heads × head_dim × bytes × num_ctx`. All three models are 8 kv-heads × 128 head_dim, so f16 costs 128 KiB/token at 32 layers and 160 KiB/token at 40.

| Model | Weights | KV @32768 | Total | Verdict |
|---|---|---|---|---|
| `llama3.1:8b-q8` (32L) | 8.5 GiB | 4.0 GiB f16 | 12.5 | ✅ f16, comfortable |
| `qwen3:14b` (40L, q4_K_M) | 9.3 GiB | 5.0 GiB f16 | 14.3 | ✅ f16, fits |
| `mistral-small:24b` **Q4_K_M** (40L) | 13.3 GiB | 5.0 f16 / 2.5 q8_0 | 18.3 ❌ / 16.3 ❌ | ❌ **does not fit at 32K in any config** |
| `mistral-small:24b` **IQ4_XS** | **11.7 GiB** | 2.5 GiB q8_0 | **14.7** | ✅ **this is the one** |

So the answer to both questions is the same move: **IQ4_XS is not a compromise here, it's the enabler.** Q4_K_M forces a choice between the 24B model and full context; IQ4_XS gives both with ~1.2 GiB to spare. Published numbers put IQ4_XS at 4.3 bpw vs Q4_K_M's 4.5, with perplexity parity or slightly better (imatrix-calibrated bit allocation) — at 24B, 4-bit damage is mild.

**Do not go below IQ4_XS.** Q3 is where structured-JSON extraction starts producing wrong field values, and the Qwen3.6 community reports flag IQ3 failing function-call JSON. There's no VRAM reason to.

Pull path — official Ollama `mistral-small` only publishes q4_K_M / q8_0 / fp16, so IQ4_XS comes from HF:

```powershell
ollama pull hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:IQ4_XS
```

Verify the tag resolves before planning around it. Also pull `mistral-small:24b` (Q4_K_M, text-only — **not** `mistral-small3.2`, which bundles a ~1 GB vision projector) as the fallback at 24576.

**Risks to measure, not assume:** I-quant dequantization is more complex and runs 5–20% slower; on ROCm/Windows the i-quant kernels are historically weaker than on CUDA. And `OLLAMA_KV_CACHE_TYPE=q8_0` only engages with `OLLAMA_FLASH_ATTENTION=1`, and Ollama **silently falls back to f16** on non-allowlisted architectures. Both are caught by the probe in Verification.

### Optional 4th run: higher quant on qwen3:14b

Quantization isn't monotonic in the existing results (q8 helped at 8B, hurt at 3B). `qwen3:14b-q5_K_M` ≈ 10.5 GiB + 5.0 f16 KV = 15.5 GiB — fits at 32K. Cheapest available shot at beating the current 0.807 local ceiling. +~1.6 h. Optional; skip if the 4 h budget is already tight.

---

## Model roster (researched 2026-08-03)

Everything else from the 2026 crop fails the filters — the field went sparse-MoE and multimodal, both wrong for 16 GB:

| Rejected | Why |
|---|---|
| `qwen3.6:27b` | 16–18 GiB weights; spills |
| `qwen3.6:35b-a3b` | 24 GiB Q4_K_M (18 IQ4_XS); vision + thinking tags |
| Qwen3.5 family (9B/27B/35B-A3B) | vision-language **and** hybrid-thinking with no non-thinking checkpoint; `enable_thinking:false` is per-request and build-dependent |
| `gemma4:26b-a4b` | 18.0 GiB — already measured spilling on this box |
| `gemma4:12b` / `:31b` | multimodal (image+video); 31b far over budget |
| Nemotron 3 Nano 30B-A3B | ~17–18 GiB; multimodal |
| `gpt-oss:20b` | fits (12.8 GiB MXFP4) but is a reasoning model — excluded |
| Mistral Small 4 | 64 GiB at Q4 |

`mistral-small:24b` (Mistral Small 3, 2501, Apache 2.0, 32K native context, text-only, no thinking mode) is the only candidate clearing every filter. Its **native context is 32768**, so `ctx_for()` clamps to exactly 32768 — no headroom above that.

---

## Q: Is the embedder good enough? — No.

Current: `all-MiniLM-L6-v2`, 384-dim, hardcoded at `pipeline/embedder.py:23`. (`config.py:10` holds the same string but is dead — nothing imports it.)

- **Trained at 128 tokens, capped at 256**, while `pipeline/chunker.py:6` emits 512-token chunks. Half of every chunk is outside the embedder's competence.
- Retrieval is dense-only: **no reranking, no similarity cutoff, no MMR**. Nothing downstream repairs a bad first-stage ranking.

**Recommend `BAAI/bge-base-en-v1.5`** — 768-dim, retrieval-trained, native 512 tokens, already in fastembed's supported model list so it drops straight into the existing ONNX path. ~2× embed time, irrelevant at ~5 min per 100 contracts.

It's a real fork — new scores won't be comparable to the MiniLM table in `LOCAL_MODELS.md`. Step 5 settles it empirically for ~$0.40 rather than by argument.

---

## Implementation

### 1. Non-destructive artifact paths — `scripts/benchmark_models.py`, `config.py`

**Nothing existing is touched.** Add a `--run-tag` flag (default `""` = current behaviour) that suffixes all three outputs:

```python
tag = f"_{args.run_tag}" if args.run_tag else ""
CONTEXTS_PATH = BENCH_DIR / f"contexts{tag}.json"
RESULTS_PATH  = BENCH_DIR / f"results{tag}.json"
REPORT_PATH   = BENCH_DIR / f"BENCHMARK_REPORT{tag}.md"
```

`CONTEXTS_PATH`/`RESULTS_PATH` are module constants at `:35-36` and `write_report` hardcodes the report path at `:307-308`; all three become locals threaded from `main()`. This run uses `--run-tag bge100`, writing `contexts_bge100.json` / `results_bge100.json` / `BENCHMARK_REPORT_bge100.md`. The existing `contexts.json` (12 MiniLM contracts), `results.json` (13 models) and `BENCHMARK_REPORT.md` stay exactly as they are and remain reusable.

Chroma likewise gets a **new** database, old one untouched:

- `config.py:29` → `CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")`
- run with `CHROMA_DB_PATH=./data/chroma_db_bge`

The existing 389 MB `data/chroma_db` keeps its 384-dim MiniLM collections. (`build_per_doc_index` at `:121-126` drops and recreates per contract with no dimension check, so mixing embedders in one DB would corrupt it — separate paths avoid that entirely.)

Embedder becomes env-selectable so no code edit is needed to switch back:

- `pipeline/embedder.py:23-24` → `LOCAL_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")`, `EMBED_DIM` looked up from the fastembed model metadata rather than hardcoded (it's currently declared and never used).
- Point `config.py:10` at the same env var or delete it — two sources of truth for one value is how this drifts.

### 2. Random sampling — `scripts/benchmark_models.py:56-64`

```python
def pick_contracts(n: int, seed: int) -> list[Path]:
    """Random sample, seeded — reproducible, no stride artefacts."""
    files = sorted(f for f in CORPUS.glob("*.txt") if f.stem not in EXCLUDE)
    if not files:
        raise SystemExit(f"No contracts in {CORPUS}. Run scripts/download_cuad.py first.")
    if n >= len(files):
        return files
    return sorted(random.Random(seed).sample(files, n))
```

Single call site at `:223` → `pick_contracts(args.contracts, args.seed)`; add `--seed` (default `42`). Keep the outer `sorted()` so run order is stable. Record the seed in the results file.

### 3. Context window — `scripts/benchmark_models.py:46`

`TARGET_NUM_CTX = 24576` → **32768**. `ctx_for()` (`:240-245`) already clamps to each model's architectural limit via `min()`, so llama3.1 (131072) and qwen3 (40960) take 32768, and mistral-small (32768) takes exactly its native max.

Env for the mistral run only:

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```

llama3.1 and qwen3 run f16 KV and need neither.

### 4. DeepSeek V4 Pro — `config.py:7`, `pipeline/llm.py:72`

- `DEEPSEEK_MODEL = "deepseek-v4-pro"`
- Add `extra_body={"thinking": {"type": "disabled"}}` to the OpenAI-SDK call.

Both required. `reasoning_effort="none"` alone does **not** disable thinking on V4 — the `thinking` block must be set explicitly. Leave `temperature=0` (`llm.py:73`); it's honoured in non-thinking mode but ignored if thinking leaks through, which is a useful tell.

Cost at V4 Pro rates ($0.435/M cache-miss in, $0.87/M out), ~7.8K in + ~600 out per contract: **~$0.40 per 100-contract pass**. Watch the announced peak-hour 2× multiplier (09:00–12:00 / 14:00–18:00 Beijing).

### 5. A/B the embedder before committing 4 h

Build both context sets over the same 100 contracts (`--run-tag minilm100` and `--run-tag bge100`) and run `deepseek-v4-pro` against each. ~40 min, ~$0.80. If bge-base doesn't move the score, run the locals on `minilm100` and stay comparable with the existing table. Because of step 1 both context sets are kept for later reuse either way.

### 6. Execution order — no change needed

Already correct: `main()` (`:249-259`) is model-outer / contract-inner, and `build_contexts` runs once up front (`:232`) before any model loads. One model loads, all 100 contracts run, then the next — no load/offload thrash. `save(results)` fires after every contract (`:148`), so every run is resumable.

Run each model as its own invocation so failures are isolated and progress is reportable:

```powershell
# contexts + API baseline
python scripts/benchmark_models.py --contracts 100 --seed 42 --run-tag bge100 --models

# locals, one at a time
python scripts/benchmark_models.py --contracts 100 --seed 42 --run-tag bge100 --skip-deepseek --models llama3.1:8b-q8
python scripts/benchmark_models.py --contracts 100 --seed 42 --run-tag bge100 --skip-deepseek --models qwen3:14b
$env:OLLAMA_FLASH_ATTENTION=1; $env:OLLAMA_KV_CACHE_TYPE="q8_0"
python scripts/benchmark_models.py --contracts 100 --seed 42 --run-tag bge100 --skip-deepseek --models hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:IQ4_XS

python scripts/benchmark_models.py --report --run-tag bge100
```

### 7. Estimated wall clock

| Step | Time |
|---|---|
| Context build ×2 (MiniLM + bge, 100 contracts) | ~10 min |
| `deepseek-v4-pro` ×2 (A/B) | ~40 min |
| `llama3.1:8b-q8` @32K | ~0.9 h |
| `qwen3:14b` @32K | ~1.6 h |
| `mistral-small` IQ4_XS @32K | ~1.1–1.4 h |
| **Total** | **~4.5 h** |
| *(optional `qwen3:14b-q5_K_M`)* | *+1.6 h* |

Slightly above the earlier 24K estimate: bigger KV means longer prefill, and the ~7% of contracts that used to get halved now run at full length.

---

## Verification

1. **VRAM probe — gates the whole mistral run.** Run one contract, on the **largest** contract in the sample (not the first), then `ollama ps`. Must read **100% GPU**. Any `x%/y% CPU/GPU` split means it spilled. Cross-check the Ollama server log for `KV self size` showing `q8_0`, not `f16` — if it says `f16`, flash attention was silently refused for this architecture and the q8_0 setting did nothing, in which case drop mistral to `num_ctx=24576` (11.7 + 3.75 f16 = 15.5 GiB, still fits at IQ4_XS) and record the reduced window.

2. **No truncation at 32K.** `truncated` must be 0 across all 100 runs, and the `num_ctx` recorded per run (`pipeline/extractor.py:50`) must be 32768 for every row — any 16384 means the halving backoff at `llm.py:100-108` fired and that contract's context was cut in half.

3. **Context histogram sanity.** After `build_contexts`, check the token distribution in `contexts_bge100.json`: p100 should land near ~28.8K and nothing should exceed it. If something does, the 95-chunk ceiling reasoning is wrong and `num_ctx` needs revisiting before the locals run.

4. **DeepSeek thinking is actually off.** After one contract, confirm no `reasoning_content` in the response and `completion_tokens` in the ~400–800 range, not pinned at 1500. Truncated or empty extractions mean the thinking block wasn't disabled.

5. **Sampling is random and reproducible.** Two `--seed 42` invocations select the identical 100 stems; `--seed 43` selects a different set; the sample must not look like an arithmetic sequence of `cuad_NNNN` ids.

6. **Old artifacts intact.** `output/benchmarks/contexts.json`, `results.json`, `BENCHMARK_REPORT.md` and `data/chroma_db/` unchanged (compare mtime/size before and after). New files are `*_bge100.*`, `*_minilm100.*` and `data/chroma_db_bge/`.

7. **End-to-end report.** `--report --run-tag bge100` regenerates `BENCHMARK_REPORT_bge100.md` from cache. Check `n=100` on every row and the recorded `num_ctx` per model. Then add a new section to `LOCAL_MODELS.md` — don't overwrite the existing table — flagging that these numbers use bge-base contexts at 32K and are not comparable to the 24K MiniLM run.

---

## Sources

- [Models & Pricing — DeepSeek API Docs](https://api-docs.deepseek.com/quick_start/pricing/) · [Thinking Mode — DeepSeek API Docs](https://api-docs.deepseek.com/guides/thinking_mode/)
- [ollama.com/library/mistral-small/tags](https://ollama.com/library/mistral-small/tags) · [mistral-small:24b](https://ollama.com/library/mistral-small:24b)
- [Choosing a GGUF model: K-quants, I-quants, and legacy formats](https://kaitchup.substack.com/p/choosing-a-gguf-model-k-quants-i) · [GGUF quantization: quality vs speed on consumer GPUs](https://dasroot.net/posts/2026/02/gguf-quantization-quality-speed-consumer-gpus/)
- [OLLAMA_KV_CACHE_TYPE: halve Ollama's KV cache memory](https://modelpiper.com/blog/ollama-kv-cache-quantization) · [Ollama #13337 — flash attention / KV quant architecture allowlist](https://github.com/ollama/ollama/issues/13337)
- [Gemma 4 model overview](https://ai.google.dev/gemma/docs/core) · [Qwen3.5 open-weights family — DeepLearning.AI](https://www.deeplearning.ai/the-batch/alibabas-latest-flagship-models-are-open-weights-moe-performers-in-sizes-from-less-than-1b-parameters) · [Qwen3.5-27B thinking cannot be disabled at deploy time](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/discussions/4) · [Qwen3.6 VRAM table](https://knightli.com/en/2026/05/01/qwen3-6-local-vram-quantization-table/)
- [FastEmbed supported models](https://github.com/qdrant/fastembed/blob/main/docs/examples/Supported_Models.ipynb) · [all-MiniLM-L6-v2 max_seq_length is 256, trained at 128](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/54)
