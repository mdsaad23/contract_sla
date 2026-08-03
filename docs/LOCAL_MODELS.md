# Local Models: Setup and Benchmark

Everything in this pipeline except the extraction call already ran locally
(PDF parsing, chunking, MiniLM embeddings, ChromaDB, eval). This document
covers making the last step local too, and what it costs you in accuracy.

**Headline:** the best local model scores **0.807** against DeepSeek V3.2's
**0.819** — a **1.5%** quality gap, for **6.9× the latency** and zero API
spend. Most local models are much worse than that, and the reasons are
specific and fixable.

---

## 1. Running it locally

The provider is a config switch, not a code change.

```bash
# once
ollama serve
ollama pull qwen3:14b-q4_K_M

# per run
LLM_PROVIDER=ollama LLM_MODEL=qwen3:14b-q4_K_M python scripts/run_pipeline.py
```

| Env var | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `api` | `api` (OpenAI-compatible) or `ollama` |
| `LLM_MODEL` | provider default | e.g. `qwen3:14b-q4_K_M` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_NUM_CTX` | `24576` | do not lower it — see §3 |
| `OLLAMA_TIMEOUT` | `900` | reasoning models on a loaded GPU are slow |

`pipeline/llm.py` talks to Ollama's native `/api/chat`, not its
OpenAI-compatibility shim, for one reason: the shim gives you no way to set
`num_ctx`. It also returns real token counts and load timings, which the
benchmark needs.

Hardware for every number below: AMD RX 9070 XT, 15.9 GiB VRAM, Ollama on
ROCm, Windows 11.

---

## 2. Benchmark method

**12 contracts × 13 models, and every model sees byte-identical input.**

Retrieval runs *once* and is cached to `output/benchmarks/contexts.json`.
Every model is then fed those exact context blocks. This isolates the LLM: no
model gets a luckier chunk than another, and re-running a model is
deterministic (`temperature=0`).

- Contracts: an evenly-spaced deterministic slice of CUAD, `cuad_0000`
  excluded (it's CUAD's own datasheet, not a contract). Context sizes span
  7K–76K characters, so the set includes contracts that overflow small windows.
- Scoring: the same `evals/eval_runner.py:score_extraction` the main pipeline
  uses — substring overlap against the source, keyword coverage, format
  validation. It is a **proxy** eval; CUAD ships no gold labels for these
  fields. Treat the numbers as comparative, not absolute.
- Context window: `min(24576, model's own context_length)`. 24576 covers the
  largest prompt (~23K tokens).
- Output budget: 2000 tokens, raised to 4000 for reasoning models.
- Reproduce: `python scripts/benchmark_models.py --report`

```bash
python scripts/benchmark_models.py --list              # what's installed
python scripts/benchmark_models.py --contracts 12      # full sweep
python scripts/benchmark_models.py --models qwen3:14b-q4_K_M
python scripts/benchmark_models.py --json-mode         # constrained decoding
```

Runs are saved after every call, so an interrupted sweep resumes.

---

## 3. Three measurement bugs found before any number was trustworthy

Each of these produced a plausible-looking but wrong ranking. They are the
most transferable part of this exercise.

**(a) Silent context truncation.** Ollama's default `num_ctx` is 4096; this
repo's default was 8192. The largest contract's prompt is ~23K tokens. Ollama
does not error on overflow — it truncates and answers anyway. The first
llama3.2:3b run scored **0.372** with one hard failure, and the three worst
contracts were exactly the three largest. Raising `num_ctx` to 24576:
**0.372 → 0.487, +31%, all three failures became successes.** Nothing about
the model changed.

**(b) OOM on prefill, not on load.** gemma4 returned HTTP 500 on all 12
contracts at both sizes, while loading fine at `num_ctx=24576` with a short
prompt. A 23K-token prefill allocates far more than the weights do. Fixed
with a halve-the-context retry loop in `_call_ollama`; the effective window is
recorded per run so a degraded run is visible rather than silent. The invalid
runs were archived to `results_prefix_oom.json` rather than deleted.

**(c) Reasoning consuming the entire output budget.** Newer Ollama returns
chain-of-thought in a separate `thinking` field. gemma4 filled `thinking` and
left `content` empty — it spent all 2000 tokens reasoning and never emitted
the answer. That surfaced as an empty-response error indistinguishable from a
parse failure, so `_ollama_once` now raises an explicit
`reasoning-only response: N tokens of thinking, no content`. Raising the
budget to 4000 lifted **qwen3:14b 0.724 → 0.807** but did not rescue
gemma4:12b and did not move deepseek-r1 at all.

Also worth noting: `_coerce()` in `pipeline/extractor.py` exists entirely
because of small local models. They drift from the schema in predictable ways
— echoing the `"EXACT QUOTE or null"` placeholder verbatim, emitting the
*string* `"null"`, wrapping values in `{"value": ...}`, returning a list of
candidates. Normalising these keeps a *placeholder answer* distinguishable
from a *parse failure*, which are different bugs.

---

## 4. Results

12 contracts each, identical retrieved context, temperature 0.

| # | Model | Score | vs DeepSeek | Size | Params | Quant | Median latency | ok / partial / fail | Score per GB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **deepseek-chat** (API) | **0.819** | — | — | ~685B MoE | — | 5.6s | 12 / 0 / 0 | — |
| 2 | **qwen3:14b-q4_K_M** | **0.807** | −1.5% | 9.3 GB | 14.8B | Q4_K_M | 38.4s | 12 / 0 / 0 | 0.087 |
| 3 | llama3.1:8b-instruct-q8_0 | 0.636 | −22.4% | 8.5 GB | 8.0B | Q8_0 | 26.1s | 11 / 0 / 1 | 0.074 |
| 4 | phi4:14b-q4_K_M | 0.626 | −23.6% | 9.1 GB | 14.7B | Q4_K_M | 14.4s | 10 / 0 / 2 | 0.069 |
| 5 | gemma4:26b-a4b-it-q4_K_M | 0.618 | −24.5% | 18.0 GB | 25.8B MoE | Q4_K_M | 100.1s | 9 / 0 / 3 | 0.034 |
| 6 | mistral:7b-instruct-v0.3-q4_K_M | 0.579 | −29.3% | 4.4 GB | 7.2B | Q4_K_M | 10.2s | 9 / 3 / 0 | 0.133 |
| 7 | llama3.1:8b-instruct-q4_K_M | 0.576 | −29.6% | 4.9 GB | 8.0B | Q4_K_M | 8.5s | 12 / 0 / 0 | 0.117 |
| 8 | qwen2.5-coder:7b-instruct-q4_K_M | 0.530 | −35.3% | 4.7 GB | 7.6B | Q4_K_M | 9.7s | 11 / 0 / 1 | 0.113 |
| 9 | **llama3.2:3b-instruct-q4_K_M** | 0.487 | −40.5% | 2.0 GB | 3.2B | Q4_K_M | 4.4s | 12 / 0 / 0 | **0.241** |
| 10 | granite4:7b-a1b-h | 0.468 | −42.8% | 4.2 GB | 6.9B MoE | Q4_K_M | 5.2s | 9 / 3 / 0 | 0.111 |
| 11 | llama3.2:3b-instruct-q8_0 | 0.410 | −49.9% | 3.4 GB | 3.2B | Q8_0 | 4.7s | 9 / 2 / 1 | 0.120 |
| 12 | deepseek-r1:8b-llama-distill-q4_K_M | 0.380 | −53.6% | 4.9 GB | 8.0B | Q4_K_M | 21.7s | 7 / 5 / 0 | 0.077 |
| 13 | gemma4:12b-it-q4_K_M | 0.113 | −86.2% | 7.6 GB | 11.9B | Q4_K_M | 83.6s | 2 / 0 / 10 | 0.015 |

`ok` = ≥2 fields populated and parsed; `partial` = parsed but nearly empty;
`fail` = no usable JSON. Latency is per contract, median, excluding model
load. `gemma4:26b` ran at 18 GB of weights on a 15.9 GiB card, so it spilled
to system RAM — its 100s is a memory-bandwidth number, not a model property.

---

## 5. Which are better, which are worse, and why

### qwen3:14b is the only local model that competes

0.807 vs 0.819 is inside the noise of a 12-contract proxy eval. Two things
set it apart:

- It is the only local model that gets **better on long contracts**: 0.908 on
  the four largest vs 0.756 on the rest. DeepSeek does the same (0.846 /
  0.806). Every other local model degrades — phi4 0.367/0.755,
  qwen2.5-coder 0.308/0.641, llama3.1-q8 0.467/0.720. **Long-context
  robustness, not raw score, is what separates the top tier.** A model that
  averages 0.63 by acing short contracts and collapsing on long ones is worse
  in production than the average suggests, because long contracts are where
  the interesting clauses live.
- Its reasoning is *usable* reasoning. It generates 1087 output tokens per
  contract to DeepSeek's 569, and given a 4000-token budget it thinks and then
  answers. Given 2000 it scored 0.724. Budget was the whole difference.

**Cost of switching:** 38.4s vs 5.6s per contract. Extrapolated to the full
510-contract corpus that's ~5.4 hours local vs ~48 minutes on the API — to
save $1.72. Local wins on *confidentiality*, not on cost or speed.

### The mid-tier (0.53–0.64) fails the same way

llama3.1:8b, phi4, mistral, qwen2.5-coder all land within 0.11 of each other,
and their losses are concentrated in the same four fields — the ones needing
*judgement about which span to quote* rather than pattern-matching:

| Field | deepseek | qwen3 | llama3.1-q8 | phi4 | mistral | llama3.2:3b |
|---|---|---|---|---|---|---|
| governing_law | 1.00 | 1.00 | 0.90 | 0.83 | 0.88 | 0.18 |
| penalty_currency | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| penalty_has_monetary | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| penalty_max_amount | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| termination_clause | 0.83 | 0.83 | 0.48 | 0.70 | 0.42 | 0.05 |
| renewal_terms | 0.73 | 0.87 | 0.50 | 0.47 | 0.47 | 0.00 |
| dispute_resolution | 0.63 | 0.50 | 0.30 | 0.25 | 0.30 | 0.05 |
| liability_cap | 0.57 | 0.80 | 0.40 | 0.43 | 0.26 | 0.09 |

Booleans, currencies and amounts are at parity across the whole field — those
are format-validated and effectively solved at 3B. The gap is entirely in
**long verbatim spans**: `termination_clause`, `renewal_terms`,
`dispute_resolution`, `liability_cap`. Small models find the right clause and
then quote it sloppily — paraphrasing, truncating mid-sentence, or merging two
clauses — which the substring-overlap signal correctly penalises.

*(Per-field averages come from the contracts where the field is present at
all; the sparse rows — `uptime_guarantee`, `service_credit_cap`,
`penalty_data_breach` — are 1–3 samples and shouldn't be read as rankings.
CUAD is affiliate and licensing agreements, not SaaS contracts, so uptime
fields are genuinely absent from most documents.)*

### phi4's "JSON bug" is a context bug

phi4's 2 failures are `cuad_0171` and `cuad_0298` — precisely the two
contracts whose prompts exceed its 16384-token limit. Its context window, not
its formatting, is the ceiling. Constrained decoding could not fix it, which
confirms the diagnosis: **0.626 → 0.606 with `--json-mode`.**

### Quantization does not help monotonically

| | Q4_K_M | Q8_0 | |
|---|---|---|---|
| llama3.1:8b | 0.576 | **0.636** | q8 wins, +10% |
| llama3.2:3b | **0.487** | 0.410 | q4 wins, +19% |

At 8B, the extra precision buys real accuracy. At 3B it *loses* — and the q8
run also produced 2 partials and 1 failure where the q4 run had 12 clean
successes. Doubling the weights of a small model does not make it better at a
task it was never big enough for; it just costs you 1.4 GB and 0.3s. Test the
tier you're actually deploying instead of assuming higher precision is safer.

### Reasoning models are wrong for fixed-schema extraction

deepseek-r1:8b (0.380, 5 partials, 21.7s) is the worst non-broken model and is
*slower* than the plain llama3.1:8b it's distilled from, which scores 0.576.
gemma4:12b in thinking mode is unusable: **0.113, 10 of 12 failures** — 7
reasoning-only, 3 malformed JSON — at 83.6s each.

The task is "find this span, copy it exactly." Chain-of-thought adds tokens,
latency and an extra failure mode (never reaching the answer) without adding
information. qwen3 is the exception that proves it: it's good *despite*
thinking, and only once you pay for a 4000-token budget.

### Constrained JSON decoding barely matters

| Model | default | `--json-mode` |
|---|---|---|
| mistral:7b | 0.579 | 0.579 |
| llama3.1:8b-q4 | 0.576 | 0.576 |
| qwen2.5-coder:7b | 0.530 | 0.513 |
| phi4:14b | 0.626 | 0.606 |

Two unchanged, two slightly worse. Grammar-constrained decoding guarantees
*parseable* output, but parseability was never the bottleneck — the remaining
errors are wrong or sloppy *content*. Forcing the grammar also constrains
token choice slightly, which is the plausible cause of the small drops. Prompt
`_coerce()` handles the drift more cheaply.

### The efficiency winner is llama3.2:3b

0.241 score-per-GB, 2.0 GB, 4.4s, 12/12 clean parses, no failures at any
contract size. It is genuinely bad at long verbatim quoting (0.05 on
`termination_clause`) but perfect on booleans, currency and max-amount. If
your query is "which contracts carry a real cash penalty," a 2 GB model
answers it on a laptop with no GPU.

---

## 6. What to actually run

| Constraint | Pick | Why |
|---|---|---|
| Best quality, cost no object | `deepseek-chat` | 0.819, 5.6s, $0.0034/contract |
| Data cannot leave the building | `qwen3:14b-q4_K_M` | 0.807 at 1.5% behind; needs ≥10 GB VRAM and a 4000-token output budget |
| Laptop / no GPU / triage only | `llama3.2:3b-instruct-q4_K_M` | 0.487 but flawless on the boolean+amount fields; 2 GB |
| Avoid | any reasoning distill, `gemma4:*` | reasoning burns the output budget for no gain on fixed-schema extraction |

`OLLAMA_NUM_CTX` now defaults to 24576 for exactly this reason. Lower it and
you will benchmark truncation instead of the model.

---

## 7. Caveats

- 12 contracts is a small sample. Differences under ~0.05 are not meaningful;
  the 0.819-vs-0.807 result should be read as "indistinguishable," not
  "DeepSeek wins."
- The eval is a proxy (overlap / keyword coverage / format), not gold labels.
  It rewards verbatim quoting, which is exactly what the prompt demands — but
  a model that paraphrases *correctly* is scored down.
- All timings are one GPU, one driver stack. `gemma4:26b`'s 100s in particular
  is an artifact of exceeding VRAM.
- DeepSeek was measured over the public API, so its latency includes network
  and queueing that the local models don't pay.

Raw data: `output/benchmarks/results.json`, `summary.json`, `contexts.json`.
