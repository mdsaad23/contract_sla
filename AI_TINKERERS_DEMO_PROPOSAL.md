# AI Tinkerers Dubai — Demo Proposal

Form answers for the SLA Extraction RAG Pipeline demo. Copy/paste each block into the matching form field.

---

## Your Email

```
justsaad95@gmail.com
```

---

## Format Preference

**Main Stage Demo**

Rationale: the pipeline runs a contract end-to-end in 6–8 seconds, so a live run, a live eval, and a live SQL query all fit inside a demo slot without dead air. The story is a debugging arc (eval said 0.28, the pipeline was fine) which lands better as a narrative to a room than as a repeated booth conversation.

*(If they push back on capacity, "Either format works for me" is a safe fallback — the demo degrades gracefully to a laptop-on-a-table version.)*

---

## Talk Title

**Primary:**
```
My Eval Said 0.28. My Benchmark Said Local Models Were Useless. Both Were Wrong.
```

**Alternates:**
```
My Eval Said 0.28. The Pipeline Was Fine.
510 Contracts, $1.72, and Every Bug Was in the Measurement
I Benchmarked 13 Local Models Against DeepSeek. Then I Found Three Bugs in My Benchmark.
The Gap Between a 9GB Local Model and a Frontier API Is 1.5%
```

---

## What did you build?

```
A RAG pipeline that reads commercial contracts and extracts 19 structured SLA
fields — uptime guarantees, liquidated damages, late-payment interest,
termination fees, liability caps — from raw PDFs into a queryable SQLite
database. I ran it over all 510 contracts in the CUAD dataset (real SEC-filed
commercial agreements) for a total API spend of $1.72, at 6–8 seconds per
contract on a consumer GPU.

Then I made the LLM call a config switch and benchmarked 13 local models
through Ollama against DeepSeek on identical retrieved context. Best local
model: 0.807 vs DeepSeek's 0.819. A 1.5% gap, on a consumer GPU, for $0.

Live, I'll show:

1. A contract going through the whole pipeline in one terminal — PDF text in,
   chunks, 19 retrieval queries against a per-document ChromaDB index, one
   DeepSeek call out, structured JSON in. Roughly 6 seconds, real API, no
   pre-recording.
2. The same contract again with `LLM_PROVIDER=ollama` — same retrieval, same
   prompt, a 9GB model on the GPU instead of an API. Slower, nearly identical
   output.
3. The retrieval layer with its lid off: which of the 19 queries actually
   surfaced the penalty clause, and the deduplicated context block the LLM
   sees. This is where the accuracy lives, not in the model.
4. The two measurement stories side by side — the eval runner scoring a
   contract with its three signals plus the diff that moved the reported score
   from 0.28 to 0.79 without touching extraction logic; and the benchmark
   result that jumped 0.372 → 0.487 the moment I stopped letting Ollama
   silently truncate my prompt.
5. SQL against the 510-row results DB: "which contracts have a real cash
   penalty, not just a service credit?" — 128 of 510. That query is the whole
   reason the schema has 19 fields instead of 8.

Repo, eval report, 13-model benchmark, failure log, and cost breakdown are all
public.
```

---

## What will another builder learn?

```
The main lesson: my first eval scored 0.28 and I nearly rewrote the retrieval
layer over it. Three of the four root causes were bugs in the eval itself —
the dataset's own datasheet was being scored as if it were a contract, my text
cleaner stripped apostrophes so "don't" in the source never matched "don t" in
the extraction, and the eval was still scoring an 8-field schema the pipeline
had already outgrown. The extraction was ~100% verbatim-accurate the whole
time. If your eval is younger than your pipeline, spot-check the eval before
you believe it. I'll show the actual scoring code and the diff.

The same thing happened again when I benchmarked local models, which is why
I think it's a pattern and not an anecdote. First sweep said small models were
hopeless. Three bugs, all in the measurement, none in the models:

• Ollama's default context is 4096 tokens. My largest contract prompt is
  ~23,000. Ollama does not error on overflow — it truncates and answers
  anyway. Raising num_ctx to 24576 took llama3.2:3b from 0.372 to 0.487 and
  turned all three of its failures into successes. Nothing about the model
  changed.
• gemma4 threw HTTP 500 on every contract while loading fine with a short
  prompt — a 23K-token prefill allocates far more than the weights do. That's
  an OOM wearing a server error.
• Reasoning models were returning nothing at all. Newer Ollama puts
  chain-of-thought in a separate `thinking` field; they were spending the
  entire output budget reasoning and never emitting the answer. Raising the
  budget to 4000 tokens took qwen3:14b from 0.724 to 0.807 — from mid-tier to
  matching the API.

And the result once the measurement was honest: qwen3:14b scores 0.807 against
DeepSeek's 0.819. If your data can't leave the building, that 1.5% is the
entire price of admission. You pay it in latency — 38s vs 6s per contract —
not in accuracy.

Three more that transfer:

• Profile before you optimise the thing you assume is slow. I blamed the LLM
  API for 31s/contract. It was the embedder loading its weights twice per
  contract because two functions each instantiated their own. A module-level
  singleton plus a PyTorch→ONNX swap took it to 3–8s — a 6–10× speedup where
  the API call was never the bottleneck. GPU acceleration on top added only
  1.6×; MiniLM is too small to saturate a GPU.

• "EXACT QUOTE or null" beats "extract the clause." Forcing verbatim quoting
  in the prompt does double duty: it cuts hallucination, and it makes the
  output automatically checkable — you can grep the extraction against the
  source. That one constraint is what made a proxy eval possible with zero
  labelled data.

• Schema design was harder than the engineering, and it's what makes the
  output useful. One generic penalty_clause string conflated service credits
  with cash damages and could not answer the only question anyone cared about.
  Splitting into five penalty types plus a penalty_has_monetary boolean cost
  +20% tokens and made the database queryable. Structure the schema around the
  question, not around what's easy for the model to emit.

What to avoid: don't reach for a frontier model on a fixed-schema extraction
task. DeepSeek V3.2 did this for $1.72; the same run on Claude Sonnet is ~$56,
GPT-4o ~$94, for negligible quality difference when the model's job is to fill
known fields from retrieved text. Spend the budget on retrieval instead.

And two counterintuitive ones from the benchmark, both of which cost me time:
q8 quantization beats q4 at 8B (0.636 vs 0.576) but *loses* at 3B (0.410 vs
0.487) — higher precision is not monotonically safer. And reasoning distills
are actively worse here: deepseek-r1:8b scores 0.380, below the plain
llama3.1:8b it's distilled from, and takes 2.5× longer to do it. The task is
"find this span, copy it exactly." Chain-of-thought adds tokens, latency, and
a failure mode, and no information.
```

---

## Technologies Used

```
Models
• DeepSeek V3.2 (deepseek-chat) — the only paid component. Reads the retrieved
  context and emits 19 structured JSON fields. Chosen over Claude/GPT-4o for
  ~33–55× lower cost on a fixed-schema extraction task. Total: $1.72 for 510
  contracts, 4.3M tokens.
• sentence-transformers/all-MiniLM-L6-v2 — 22.7M-param embedding model,
  384-dim, Apache 2.0. Runs locally, so contract text never leaves the machine
  during embedding and embeddings cost $0. Matters for regulated industries.
• Ollama (native /api/chat, not the OpenAI-compat shim) — 13 local models
  benchmarked on an RX 9070 XT: qwen3:14b, phi4:14b, llama3.1:8b at q4 and q8,
  llama3.2:3b at q4 and q8, mistral:7b, qwen2.5-coder:7b, granite4 and gemma4
  MoE, deepseek-r1:8b. The native endpoint because the compat shim gives you
  no way to set num_ctx, and num_ctx is the whole ballgame — see below.
  Best local result 0.807 vs DeepSeek's 0.819.

Retrieval & indexing
• ChromaDB (PersistentClient) — per-document vector index, one collection per
  contract. No cross-contract contamination, trivial resume logic.
• LlamaIndex SentenceSplitter — 512-token chunks, 64-token overlap, on NLTK
  sentence boundaries. A legal clause cut mid-sentence is meaningless.
• Custom multi-query retriever — 19 hand-written queries (performance SLAs,
  six penalty-specific, contract mechanics, plus phrase-level anchors like
  "this agreement shall be governed by the laws"), top-5 each, deduplicated
  into one context block. The phrase anchors catch boilerplate that semantic
  similarity alone misses.

Runtime & performance
• fastembed (ONNX Runtime) — replaced the PyTorch embedding backend; 2–3×
  faster on CPU. Needed a 40-line custom LlamaIndex BaseEmbedding adapter
  because the official fastembed integration pins fastembed<0.2.0.
• onnxruntime-directml — GPU inference on any DX12 GPU (validated on an AMD
  RX 9070 XT), auto-detected with CPU fallback. Chosen over CUDA/ROCm for
  vendor independence on Windows.

Data & plumbing
• CUAD (theatticusproject/cuad) via HuggingFace datasets — 511 real commercial
  contracts from SEC EDGAR. Loaded with VerificationMode.NO_CHECKS because the
  cached metadata expects 84,325 rows and the PDFs are 511.
• pdfplumber / pypdf — PDF text extraction.
• openai Python SDK pointed at base_url="https://api.deepseek.com" — DeepSeek
  is OpenAI-compatible, so no vendor-specific client and a one-line swap to
  test another provider.
• Pydantic — the 19-field schema, single source of truth.
• SQLite + JSON Lines + CSV — SQLite for querying, JSONL for streaming into
  pandas, CSV for the people who want a spreadsheet.
• Custom three-signal eval (no framework) — substring overlap against the
  source, keyword coverage to catch false-negative nulls, regex format
  validation. Built because CUAD ships no gold labels.
• scripts/benchmark_models.py — controlled model comparison. Retrieval runs
  once and is cached to disk, so all 13 models see byte-identical input and
  the LLM is the only variable. Resume-safe, temperature 0, per-model context
  clamped to min(24576, the model's own limit).
```

---

## Backing Numbers (for Q&A)

| Metric | Value |
|---|---|
| Contracts processed | 510 (CUAD, real SEC-filed commercial agreements) |
| Total API cost | $1.72 — $0.0034/contract, 4.29M tokens |
| Cost on Claude Sonnet 3.5 / GPT-4o | ~$56 / ~$94 for the same run |
| Throughput | 6–8s per contract (GPU), 2h 04m wall clock for the full run |
| Speedup from the singleton + ONNX fix | 31–50s → 3–8s per contract |
| Successful extractions (≥2 fields) | 456/510 (89.4%) |
| Partial | 51/510 (10.0%) |
| JSON parse failures | 4/510 (0.8%) |
| Avg eval score | 0.790 across 510 (0.866 on the 9-contract dev set) |
| Contracts ≥ 0.7 | 408/510 (80%) |
| Contracts with a real cash penalty | 128/510 (25%) |
| Service credits only / no penalty | 256 (50%) / 127 (25%) |
| RAG vs keyword baseline (F1) | 0.78 vs 0.56 |

**Local vs API benchmark** — 13 models × 12 contracts, identical cached retrieval, temperature 0:

| Model | Score | vs DeepSeek | Size | Median latency |
|---|---|---|---|---|
| deepseek-chat (API) | 0.819 | — | — | 5.6s |
| qwen3:14b-q4_K_M | **0.807** | −1.5% | 9.3 GB | 38.4s |
| llama3.1:8b-instruct-q8_0 | 0.636 | −22.4% | 8.5 GB | 26.1s |
| phi4:14b-q4_K_M | 0.626 | −23.6% | 9.1 GB | 14.4s |
| mistral:7b-instruct-v0.3 | 0.579 | −29.3% | 4.4 GB | 10.2s |
| llama3.1:8b-instruct-q4_K_M | 0.576 | −29.6% | 4.9 GB | 8.5s |
| llama3.2:3b-instruct-q4_K_M | 0.487 | −40.5% | 2.0 GB | 4.4s |
| deepseek-r1:8b-llama-distill | 0.380 | −53.6% | 4.9 GB | 21.7s |

Extra ammunition if they dig:
- Only qwen3 and DeepSeek get *better* on long contracts (0.908 / 0.846 on the four largest). Everything else degrades — phi4 0.367 vs 0.755. Long-context robustness, not average score, is what separates the top tier.
- phi4's two failures are exactly the two contracts exceeding its 16K context limit. A context bug wearing a JSON error message; `--json-mode` couldn't fix it, which confirms it.
- Constrained JSON decoding barely moves anything: mistral 0.579→0.579, llama3.1-q4 0.576→0.576, phi4 0.626→0.606. Parseability was never the bottleneck.
- Local models lose almost entirely on the four fields needing long verbatim spans (liability_cap, termination_clause, renewal_terms, dispute_resolution). Booleans, currency and max-amount are at parity even at 3B.

Honest caveats, if asked:
- The eval is a proxy, not gold labels — CUAD ships no answer key for these fields.
- Uptime/response-time coverage is low (3.7% / 7.0%) because CUAD is affiliate and licensing agreements, not SaaS contracts. Domain mismatch, not a retrieval failure.
- The 0.866 figure is a 9-contract dev set; 0.790 is the number that counts.
- The benchmark is 12 contracts. Differences under ~0.05 aren't meaningful — 0.819 vs 0.807 should be read as "indistinguishable," not "DeepSeek wins."
- DeepSeek's latency includes network; the local models don't pay that.

---

## 3-Part Demo Formula (rehearsal script)

**1. What you built** — "A RAG pipeline that reads 510 real commercial contracts and pulls out every penalty clause into a SQL table. Total API cost: one dollar seventy-two."

**2. How you built it** — "Local MiniLM embeddings on ONNX, per-contract ChromaDB index, 19 hand-written retrieval queries, one DeepSeek call per contract, SQLite out. The only thing I pay for is the last step — and then I made that a config switch and ran 13 local models against it."

**3. Builder takeaway** — "Twice on this project I nearly rewrote working code because my measurement was broken. My eval said 0.28 — the bugs were in the eval. My benchmark said local models were hopeless — Ollama was silently truncating a 23,000-token prompt down to 4,096. Check the instrument before you believe the reading. When mine was honest, a 9GB local model landed 1.5% behind a frontier API."

---

## Run-of-Show (~6 min)

| Time | Beat | On screen |
|---|---|---|
| 0:00 | Hook — one dollar seventy-two, 510 contracts | Terminal, cost line from the final run |
| 0:30 | The problem, in one sentence: penalty clauses locked in PDFs | A contract PDF, scrolled to a buried liquidated-damages clause |
| 1:00 | **Live run** on that contract, ~6s | `python scripts/run_pipeline.py <contract>` |
| 1:45 | Lid off the retrieval layer — which query surfaced the clause | Retrieved context block, dedup count |
| 2:30 | **The eval story** — 0.28, and where the bugs actually were | `evals/eval_runner.py` + the diff |
| 3:30 | **Same contract, local** — one env var, 9GB model on the GPU | `LLM_PROVIDER=ollama ...`, side-by-side JSON |
| 4:15 | **The benchmark story** — 13 models, and the num_ctx bug that hid the result | The results table, 0.372 → 0.487 |
| 5:15 | **The payoff query** — 128 of 510 have real cash penalties | `sqlite3 output/results.db` |
| 5:45 | Takeaways + repo link | — |

Cut for time, in this order: the perf story (double-loaded embedder, 31s → 6s), then the retrieval lid-off. The two measurement stories are the talk.

**Backup plan (no venue wifi / DeepSeek API down):** the local run needs no network at all, so the demo survives a dead venue connection — run the Ollama path live and replay a terminal recording for the API leg. Retrieval, eval and SQL are all offline against `output/results.db` regardless.

**Pre-flight checklist:**
- [ ] `.env` key valid, one throwaway run done on venue wifi
- [ ] `ollama serve` running, qwen3:14b-q4_K_M pulled and warmed (first call pays a load penalty)
- [ ] `OLLAMA_NUM_CTX=24576` exported — the whole point of the story
- [ ] `output/results.db` present and populated (510 rows)
- [ ] `output/benchmarks/BENCHMARK_REPORT.md` open in a second window
- [ ] Terminal font scaled for a projector; dark theme
- [ ] The demo contract picked in advance — one with an unambiguous cash penalty, and small enough that the local model answers in ~30s on stage
- [ ] `asciinema`/screen recording of both runs saved as fallback
- [ ] Repo public, README link short enough to read off a slide
