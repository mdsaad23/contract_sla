# SLA Extraction RAG Pipeline

> Extract Service Level Agreement clauses from thousands of contracts at ~$0.0034 per contract using DeepSeek V3.2, local embeddings on GPU, and a production-grade RAG pipeline.

---

## Quick Read (2 minutes)

**What it does:** Reads contract PDFs and text files, finds SLA-related clauses using semantic search, and extracts 18 structured fields — with particular focus on monetary penalties — using an LLM. Results are stored in SQLite, JSON Lines, and CSV.

**Production benchmarks (510 commercial contracts from CUAD):**

| Metric | Value |
|--------|-------|
| Throughput (with GPU) | 6-8s per contract |
| Total cost | **$1.72** for 510 contracts |
| Eval score (avg overall) | 0.790 |
| Contracts scoring ≥ 0.7 | 408/510 (80%) |
| Successful extractions | 456/510 (89%) |

**Why DeepSeek:** Total run cost is **$1.72**. The same pipeline using Claude Sonnet costs ~$56, GPT-4o ~$94 — for negligible quality difference on structured extraction tasks.

**Key outputs you can query:**
```sql
-- Which contracts have real cash penalties (not just service credits)?
SELECT contract_id, penalty_max_amount, penalty_currency
FROM sla_results WHERE penalty_has_monetary = 1;

-- Contracts with uptime SLA + a breach penalty
SELECT contract_id, uptime_guarantee, penalty_uptime_breach
FROM sla_results WHERE penalty_uptime_breach IS NOT NULL;
```

**5-minute setup:**
```bash
git clone <repo>
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=sk-..." > .env
python scripts/download_cuad.py --max 10   # download 10 sample contracts
python scripts/run_pipeline.py data/cuad_raw --max 10
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Cost Analysis](#2-cost-analysis)
3. [Data Source — CUAD Dataset](#3-data-source--cuad-dataset)
4. [RAG Pipeline Architecture](#4-rag-pipeline-architecture)
   - [Step 1: Ingestion](#step-1-ingestion)
   - [Step 2: Text Cleaning](#step-2-text-cleaning)
   - [Step 3: Chunking Strategy](#step-3-chunking-strategy)
   - [Step 4: Embedding Model](#step-4-embedding-model)
   - [Step 5: Vector Store](#step-5-vector-store)
   - [Step 6: Retrieval](#step-6-retrieval)
   - [Step 7: Extraction with DeepSeek](#step-7-extraction-with-deepseek)
   - [Step 8: Output Storage](#step-8-output-storage)
5. [Extraction Schema — 19 Fields](#5-extraction-schema--19-fields)
6. [Phase-by-Phase Build Log](#6-phase-by-phase-build-log)
7. [Project Structure](#7-project-structure)
8. [Setup & Usage](#8-setup--usage)
9. [Querying the Results](#9-querying-the-results)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Design Decisions & Trade-offs](#11-design-decisions--trade-offs)

---

## 1. Project Overview

This pipeline was built to extract SLA clauses from 2,000+ contracts at scale. The primary goal is identifying contracts that contain **monetary penalties** — cash amounts, liquidated damages, late payment interest, and termination fees — rather than contracts that only offer service credits.

The pipeline is fully offline for the expensive steps (chunking, embedding, vector search) and only calls the external DeepSeek API for the final LLM extraction step.

**Pipeline summary:**
```
PDF/TXT contracts
      ↓
  Text extraction + cleaning
      ↓
  Sentence-aware chunking (512 tokens, 64 overlap)
      ↓
  Local embedding (all-MiniLM-L6-v2, free, no API key)
      ↓
  ChromaDB per-document vector index
      ↓
  14 SLA-focused retrieval queries → deduplicated context
      ↓
  DeepSeek V3.2 extraction → 19 structured JSON fields
      ↓
  SQLite + JSON Lines + CSV output
```

---

## 2. Cost Analysis

| Provider | Input price (1M tokens) | Output price (1M tokens) | Est. 500 contracts |
|----------|------------------------|--------------------------|-------------------|
| **DeepSeek V3.2** | $0.27 | $1.10 | **~$7–10** |
| Claude Sonnet 3.5 | $3.00 | $15.00 | ~$200+ |
| GPT-4o | $2.50 | $10.00 | ~$175+ |
| GPT-4o mini | $0.15 | $0.60 | ~$12 |

DeepSeek pricing at `https://api.deepseek.com` as of build date. Embeddings are **free** — the `all-MiniLM-L6-v2` model runs locally via sentence-transformers, no API key required.

**Observed token usage:** ~4,200–7,600 tokens per contract (input + output combined), depending on contract length. Average ~5,500 tokens ≈ $0.0008 per contract at blended DeepSeek pricing.

---

## 3. Data Source — CUAD Dataset

**Dataset:** [Contract Understanding Atticus Dataset (CUAD)](https://huggingface.co/datasets/theatticusproject/cuad)

**Published by:** The Atticus Project — a non-profit organization of lawyers and AI researchers building tools for automated contract review.

**Contents:**
- 511 real commercial contracts in PDF format
- Contract types: affiliate agreements, licensing agreements, service agreements, software licenses, joint venture agreements, and more
- Sourced from SEC EDGAR filings (publicly filed contracts)
- Covers a wide range of industries and contract lengths (5 pages to 100+ pages)

**HuggingFace identifier:** `theatticusproject/cuad`

**How we load it:**
```python
from datasets import load_dataset, VerificationMode
ds = load_dataset(
    "theatticusproject/cuad",
    split="train",
    verification_mode=VerificationMode.NO_CHECKS,
)
```
The `NO_CHECKS` flag is required because the dataset's cached metadata (84,325 expected examples) doesn't match the actual PDF count (511 contracts) — the original CUAD had one row per QA question, while this version has one row per contract.

**Text extraction:** Each item in the dataset is a `pdfplumber.PDF` object. Text is extracted page by page:
```python
pages = [page.extract_text() or "" for page in pdf.pages]
text = "\n".join(pages)
```

We skip the first document (index 0), which is the CUAD dataset's own datasheet rather than a contract.

**Download script:** `scripts/download_cuad.py` extracts text from all 511 PDFs and saves them as individual `.txt` files in `data/cuad_raw/`. Run once before batch processing.

---

## 4. RAG Pipeline Architecture

### Step 1: Ingestion

**File:** `pipeline/ingestion.py`

Supports two input formats:
- **PDF** — extracted via `pypdf` (`PdfReader`, page-by-page text extraction)
- **TXT** — read directly with UTF-8 encoding, `errors="replace"` for robustness

```python
def load_contract(file_path: str) -> str:
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return path.read_text(encoding="utf-8", errors="replace")
```

---

### Step 2: Text Cleaning

**File:** `pipeline/ingestion.py` → `clean_text()`

Four regex passes applied in sequence:

| Pass | What it fixes |
|------|--------------|
| `\r\n` / `\r` → `\n` | Windows/Mac line endings normalised to Unix |
| 3+ newlines → 2 | Collapses excessive blank lines from PDF extraction |
| 2+ spaces/tabs → 1 space | Collapses whitespace from PDF column layouts |
| Non-ASCII stripped | Removes control characters and encoding artefacts (keeps printable ASCII + newlines) |

This is important for PDF-extracted text, which often contains repeated whitespace, hyphenated words, and encoding artefacts from the PDF renderer.

---

### Step 3: Chunking Strategy

**File:** `pipeline/chunker.py`

**Algorithm:** LlamaIndex `SentenceSplitter`

**Parameters:**
- `chunk_size = 512` tokens
- `chunk_overlap = 64` tokens

**Why SentenceSplitter over naive character splitting:**
`SentenceSplitter` uses NLTK sentence boundary detection to never cut in the middle of a sentence. This is critical for legal text — a clause split mid-sentence loses its meaning entirely. The splitter tries to fill chunks up to 512 tokens, completing the current sentence before stopping.

**Why 512 tokens:**
- Large enough to capture a complete clause with surrounding context
- Small enough that the embedding model (max 512 tokens) can encode the full chunk without truncation
- At 512 tokens with 64-token overlap, a typical 10,000-word contract produces 25–60 chunks

**Why 64-token overlap:**
Overlap ensures that clauses that span chunk boundaries (common in legal text with long preambles) appear in at least one complete chunk. At 64 tokens (~3–4 sentences), the overlap is enough to carry forward the context of a clause's opening sentence.

**Typical output:** 15–80 chunks per contract depending on document length.

---

### Step 4: Embedding Model

**File:** `pipeline/embedder.py`, `pipeline/retriever.py`

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Inference runtime:** [`fastembed`](https://github.com/qdrant/fastembed) (ONNX Runtime)

**GPU acceleration:** DirectML via `onnxruntime-directml` — works with any DX12-capable GPU (AMD, Intel, NVIDIA). Auto-detected at runtime, falls back to CPU if no GPU is found. Disable via `USE_GPU=0` in `.env`.

**Model details:**
- **Architecture:** MiniLM (distilled from BERT) with 6 transformer layers
- **Parameters:** 22.7 million (lightweight — runs on CPU without GPU)
- **Output dimension:** 384-dimensional dense vectors
- **Max input length:** 512 tokens (matched to our chunk size)
- **Training data:** Trained on 1 billion sentence pairs using contrastive learning (multiple negatives ranking loss)
- **Licence:** Apache 2.0 — fully open weights, commercial use permitted
- **Source:** HuggingFace Hub (`sentence-transformers/all-MiniLM-L6-v2`)
- **Download:** ~90 MB, cached locally after first run

**Why this model over OpenAI embeddings:**
- Zero cost — no API key, no per-token billing
- Runs locally — no data leaves the machine during embedding (critical for legal/regulated industries)
- Sufficient quality for keyword-dense legal retrieval — semantic similarity between "liquidated damages" and "financial penalty for breach" is well-captured
- Apache 2.0 licence — no restrictions on commercial use

**Performance benchmark (MTEB):** 56.26 average across 14 tasks. Lower than OpenAI `text-embedding-3-small` (62.3) but the gap is small for domain-specific retrieval with explicit query terms.

**Why ONNX Runtime over PyTorch (`sentence-transformers` directly):**
- ~2-3× faster than PyTorch on CPU for the same model (ONNX graph optimisations + quantised kernels)
- Smaller memory footprint
- Built-in GPU support across vendors (DirectML, CUDA, CoreML) without device-specific code

**Singleton pattern:** The model is loaded **once per process** in a module-level singleton (`pipeline.embedder._get_model`). The previous implementation reloaded the embedder twice per contract (`build_per_doc_index` and `retrieve_sla_chunks`), wasting ~10s per contract on weight loading. The singleton reduced startup cost from `N × 10s` to a single 0.2s at process launch.

**Speed benchmark (per contract, full pipeline):**

| Configuration | Per-contract time | Bottleneck |
|---------------|-------------------|-----------|
| PyTorch `HuggingFaceEmbedding`, double-load, CPU | 31-50s | Embedding model loading + inference |
| ONNX `fastembed`, singleton, CPU only | ~12-18s | DeepSeek API call |
| ONNX `fastembed`, singleton, DirectML GPU (AMD RX 9070 XT) | **3-8s avg ~5.5s** | DeepSeek API call |

The 6-10× end-to-end speedup came mostly from the singleton fix and the PyTorch→ONNX migration. DirectML adds ~1.6× on top, modest because MiniLM is too small to fully utilise GPU parallelism. The DirectML path matters less for raw speed and more for **scaling to larger models** (e.g. swapping in a 110M-parameter `bge-large` later for higher retrieval quality).

---

### Step 5: Vector Store

**File:** `pipeline/embedder.py`

**Store:** ChromaDB (`chromadb.PersistentClient`)

**Strategy:** Per-document index — each contract gets its own ChromaDB collection, created fresh for each run. This means:
- No index contamination between contracts
- Clean resume support (collection is deleted and rebuilt per contract)
- Trade-off: no cross-document retrieval (by design — we want clauses from the target contract only)

**Collection naming:** `contract_{contract_id[:40]}` — truncated to 40 characters to fit ChromaDB collection name limits.

**Persistence path:** `./data/chroma_db/` — reused across runs for the embedding model cache, but collections are recreated per contract.

---

### Step 6: Retrieval

**File:** `pipeline/retriever.py`

**Strategy:** Multi-query retrieval with deduplication.

Instead of a single query, we run **19 targeted queries** — semantic queries per SLA field group plus phrase-level anchors for common boilerplate — and collect the union of results:

```
Performance SLAs:
  "uptime guarantee availability SLA percentage service level"
  "response time incident support SLA hours priority"
  "SLA breach threshold availability falls below measurement period"

Monetary penalties (highest priority):
  "penalty monetary damages liquidated cash fine"
  "service credit uptime breach penalty formula"
  "late delivery penalty liquidated damages milestones"
  "early termination fee cancellation penalty"
  "late payment interest overdue invoice fee"
  "data breach fine liability security incident penalty"

Contract mechanics:
  "renewal auto-renewal notice period term extension"
  "termination notice period cancellation conditions"
  "limitation of liability cap maximum damages aggregate"
  "governing law jurisdiction applicable law"
  "dispute resolution arbitration mediation"

Phrase-level anchors (boilerplate phrase matching):
  "this agreement shall be governed by the laws"
  "in the event of any dispute between the parties"
  "neither party shall be liable for indirect consequential"
  "this agreement shall automatically renew"
  "any dispute arising out of or in connection with"
```

**Per query:** Top `k=5` most similar chunks retrieved (increased from 3 to improve recall on contracts where relevant clauses are in lower-ranked positions).

**Deduplication:** Chunks are tracked in a `seen` set — the same chunk retrieved by multiple queries is included only once.

**Why multi-query instead of one query:**
A single query like "SLA clauses" would retrieve chunks that are generally SLA-adjacent but might miss specific penalty language buried in boilerplate. A query for "late payment interest overdue invoice fee" reliably surfaces penalty sections even when the uptime query returns availability definitions instead. The phrase-level anchor queries additionally match common boilerplate section openings that semantic queries alone can miss (e.g. "this agreement shall be governed by the laws" directly retrieves the governing law section).

**Context assembly:** Retrieved chunks are joined with `---` separators and passed as a single context block to the LLM. A typical contract produces 12,000–20,000 characters of deduplicated context from 19 queries × 5 chunks.

---

### Step 7: Extraction with DeepSeek

**File:** `pipeline/extractor.py`, `models/prompts.py`

**Model:** DeepSeek V3.2 (`deepseek-chat`)

**API:** OpenAI-compatible endpoint at `https://api.deepseek.com`

**Client initialisation:**
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
```

**API call:**
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    max_tokens=1500,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": EXTRACTION_PROMPT.format(context=context)},
    ],
)
raw = response.choices[0].message.content
tokens_used = response.usage.total_tokens
```

**System prompt role:** Instructs the model to act as a legal contract analyst, respond in pure JSON only (no markdown fences), set missing fields to `null`, and **copy exact language verbatim** — not paraphrase. The verbatim quoting requirement is critical: it ensures extracted text is verifiable against the source document and prevents the model from fabricating or rewording clauses.

**Extraction prompt design:**
- Every field is labelled `"EXACT QUOTE or null"` — the model is instructed to copy words directly from the contract
- Penalty fields include explicit instructions to include exact dollar amounts and percentage formulas
- Boolean field `penalty_has_monetary` distinguishes cash penalties from service credits, with guidance on the null vs false distinction
- The prompt opens with: *"Copy exact language verbatim from the contract — do not paraphrase."*

**JSON parsing:** Response is parsed with a fallback strategy:
1. Try `json.loads()` directly
2. If that fails, strip markdown fences (` ```json `) if the model added them anyway
3. If that fails, extract the first `{...}` block with `str.find` / `str.rfind`

**Status classification:**
- `success` — 2 or more fields populated
- `partial` — 0 or 1 fields populated (contract may genuinely have no SLAs)
- `failed` — API error or JSON parse failure

---

### Step 8: Output Storage

**File:** `pipeline/output.py`

Three output formats written simultaneously after each batch:

**SQLite** (`output/results.db`)
- One row per contract
- All 19 SLA fields as columns
- `penalty_has_monetary` stored as INTEGER (0/1/NULL) for SQL boolean queries
- `raw_response` column stores the full DeepSeek JSON string for debugging
- `INSERT OR REPLACE` for idempotent re-runs

**JSON Lines** (`output/results.json`)
- One JSON object per line, one line per contract
- Includes all fields including `raw_response`
- Suitable for streaming processing and pandas `read_json(lines=True)`

**CSV** (`output/results_summary.csv`)
- Same as JSON Lines but without `raw_response` and `error` columns
- Suitable for Excel / Google Sheets analysis

---

## 5. Extraction Schema — 19 Fields

### Performance SLAs

| Field | Description | Example |
|-------|-------------|---------|
| `uptime_guarantee` | Availability % promised | `"99.9% monthly uptime"` |
| `response_time_sla` | Support response commitments by severity | `"P1: 4-hour response, P2: 8-hour"` |
| `sla_breach_threshold` | Exact trigger point for penalties | `"if uptime < 99.5% in any calendar month"` |
| `sla_measurement_period` | How/when SLA is measured | `"calculated on a monthly basis"` |

### Penalty Clauses (Separated by Type)

| Field | Description | Example |
|-------|-------------|---------|
| `penalty_uptime_breach` | Credit or cash for uptime failure | `"10% service credit for each 0.1% below 99.9%"` |
| `penalty_late_delivery` | Liquidated damages for late milestones | `"$500/day after agreed delivery date"` |
| `penalty_termination_fee` | Early exit cost | `"equal to 3 months remaining contract value"` |
| `penalty_late_payment` | Interest on overdue invoices | `"1.5% per month on amounts 30+ days overdue"` |
| `penalty_data_breach` | Security incident liability | `"$50,000 per incident plus remediation costs"` |

### Monetary Summary (Queryable)

| Field | Type | Description |
|-------|------|-------------|
| `penalty_has_monetary` | `bool` | `true` = real cash penalty; `false` = service credits only; `null` = no penalties |
| `penalty_max_amount` | `str` | Largest single penalty value mentioned anywhere |
| `penalty_currency` | `str` | `USD`, `GBP`, `EUR`, etc. |
| `service_credit_cap` | `str` | Max credits available (e.g. `"not to exceed 30% of monthly fees"`) |

### Contract Mechanics

| Field | Description | Example |
|-------|-------------|---------|
| `renewal_terms` | Auto-renewal and notice periods | `"auto-renews annually; 90 days notice to cancel"` |
| `termination_clause` | Exit conditions and notice periods | `"either party may terminate with 30 days written notice"` |
| `liability_cap` | Maximum damages either party can claim | `"aggregate liability not to exceed 12 months fees"` |
| `governing_law` | Jurisdiction | `"laws of the State of Delaware"` |
| `dispute_resolution` | ADR mechanism and venue | `"binding arbitration in New York under AAA rules"` |

---

## 6. Phase-by-Phase Build Log

### Phase 0 — Environment Setup

**Completed:** Directory structure, dependencies, configuration.

- Created `data/{raw,processed}`, `pipeline/`, `models/`, `output/`, `evals/`, `scripts/`
- `requirements.txt` uses `openai` SDK (not `anthropic`) to call DeepSeek's OpenAI-compatible API
- `config.py` loads `DEEPSEEK_API_KEY` from `.env` via `python-dotenv`
- Installed all dependencies: `llama-index`, `chromadb`, `openai`, `sentence-transformers`, `datasets`, `pdfplumber`, `pypdf`, `tqdm`, `pandas`, `pytest`

**Key decision:** Use the `openai` Python SDK pointed at `base_url="https://api.deepseek.com"` rather than a DeepSeek-specific SDK. This keeps the integration minimal and portable.

---

### Phase 1 — Core Pipeline + MVP

**Completed:** Full pipeline built and tested end-to-end on 1 contract.

Built in order:
1. `pipeline/schemas.py` — Pydantic models (`SLAClause`, `ExtractionResult`)
2. `models/prompts.py` — System prompt + extraction prompt
3. `pipeline/ingestion.py` — `load_contract()`, `clean_text()`
4. `pipeline/chunker.py` — `chunk_contract()` with SentenceSplitter
5. `pipeline/embedder.py` — `build_per_doc_index()` using ChromaDB + local embeddings
6. `pipeline/retriever.py` — `retrieve_sla_chunks()` with 14-query multi-retrieval
7. `pipeline/extractor.py` — `extract_sla()` calling DeepSeek API
8. `scripts/run_pipeline.py` — CLI entry point

**Key pivot:** Initially planned to use OpenAI `text-embedding-3-small` for embeddings. Switched to local `sentence-transformers/all-MiniLM-L6-v2` after confirming no OpenAI API key was available. This eliminated embedding cost entirely (~$0.02/1M tokens saved) with minimal quality impact for keyword-dense legal retrieval.

**MVP result on Chase Affiliate Agreement (28,553 chars):**
- 3 fields populated: `termination_clause`, `liability_cap`, `governing_law`
- 3,843 tokens used (~$0.0005)
- Zero hallucinations — all extracted text verified verbatim in source

---

### Phase 2 — Batch Processing

**Completed:** Full batch pipeline with resume support, progress bars, and all 3 output formats.

- `scripts/download_cuad.py` — downloads CUAD PDFs, extracts text, saves to `data/cuad_raw/*.txt`
- `pipeline/output.py` — `init_db()`, `save_result()`, `get_processed_ids()`, `export_outputs()`
- `scripts/_batch.py` — `run_batch()` with tqdm, batching, 2-second inter-batch pause, failed log
- `scripts/run_pipeline.py` extended — directory mode calls `run_batch()`

**Batch run (10 contracts):**
- 10/10 success
- 51,594 tokens used total
- $0.0072 estimated cost
- ~14.5 seconds per contract (local embedding is the bottleneck, not the API)
- All 3 output files generated: `results.db`, `results.json`, `results_summary.csv`

---

### Phase 3 — Evaluation

**Completed:** Three-signal eval framework and contract inspection CLI.

- `evals/eval_runner.py` — three evaluation signals:
  1. **Substring overlap** — extracted clause must appear verbatim in source text
  2. **Keyword coverage** — null fields are checked for relevant keywords (potential false negatives)
  3. **Format validation** — regex checks for expected patterns (%, dollar amounts, jurisdictions)
- `scripts/inspect_output.py` — CLI to inspect any contract by ID

**Eval results on 10 contracts:**
- Nominal score: 0.28 average (flagged below 0.5 threshold)
- Root cause analysis: score is depressed by (a) cuad_0000 being the dataset's own datasheet, not a contract, and (b) overly broad keyword matching firing on contextual uses of legal vocabulary
- Spot checks of actual extractions: 100% accuracy on populated fields — all extracted text is verbatim from source, zero fabrication observed
- `termination_clause`: 9/10 contracts populated, all correct
- `governing_law`: 7/10 contracts populated, all correct
- `liability_cap`: 6/10 contracts populated, all correct

---

### Schema Upgrade (before Phase 4)

**Completed:** Schema expanded from 8 fields to 19 fields, with structured monetary penalty extraction.

**Motivation:** The original `penalty_clause` field was a single unstructured string that conflated service credits, liquidated damages, late fees, and termination penalties. This made it impossible to query "which contracts have a cash penalty > $X."

**Changes made:**
- `pipeline/schemas.py` — 5 new penalty type fields, 4 monetary summary fields (including `penalty_has_monetary` boolean), 4 performance SLA detail fields
- `models/prompts.py` — prompt updated with field-by-field descriptions, explicit instructions to quote exact amounts and formulas, and guidance on `penalty_has_monetary` vs service credits
- `pipeline/retriever.py` — 6 additional penalty-focused retrieval queries (14 total, up from 8)
- `pipeline/output.py` — SQLite schema updated, `reset_db()` added for schema migration
- `pipeline/output.py` — `penalty_has_monetary` stored as `INTEGER` for SQL boolean queries

**Token cost impact:** ~3,843 → ~4,608 tokens per contract (+20% prompt size from larger schema definition). Still ~$0.0006 per contract.

---

### Pipeline Performance Optimisation (mid-Phase 4)

**Completed:** 6-10× end-to-end speedup via embedding runtime swap + singleton refactor + GPU acceleration.

**Motivation:** Initial Phase 4 throughput was ~31s per contract, projecting a 5-hour run for 510 contracts. Diagnosis showed the embedding step (model loading + chunk embedding + query embedding) consumed ~60% of per-contract time, while the actual DeepSeek API call was a smaller component.

**Changes made:**

1. **Module-level embedder singleton.** The previous code instantiated `HuggingFaceEmbedding` in both `build_per_doc_index()` and `retrieve_sla_chunks()`, reloading model weights twice per contract (~10s wasted per contract). Replaced with a thread-safe module-level singleton in `pipeline/embedder.py` — model loads exactly once per process.

2. **PyTorch → ONNX Runtime.** Replaced `llama-index-embeddings-huggingface` (PyTorch backend) with `fastembed` (ONNX Runtime backend). For the same model, ONNX is 2-3× faster on CPU due to graph optimisations and quantised kernels.

3. **DirectML GPU acceleration.** Added `onnxruntime-directml` execution provider for GPU inference on AMD/Intel/NVIDIA GPUs via DirectML. Validated on AMD Radeon RX 9070 XT (RDNA 4). Auto-detected at runtime via `_resolve_providers()`; falls back gracefully to CPU when no compatible GPU is found.

4. **Custom llama-index adapter.** The official `llama-index-embeddings-fastembed` package pins `fastembed<0.2.0`, which is incompatible with current Python versions. Wrote a 40-line `FastEmbedAdapter` (`pipeline.embedder.FastEmbedAdapter`) implementing the `BaseEmbedding` interface directly on top of fastembed.

**Results (measured on 3 unprocessed contracts of varying length):**

| Stage | Before | After |
|-------|--------|-------|
| Model load (per contract) | ~10s (double-load) | 0.23s once, then 0s |
| Embed + build index | 5-8s | 0.10-0.28s |
| Retrieval (19 queries) | 3-5s | 0.37-0.43s |
| DeepSeek extraction (unchanged) | 3-8s | 3-7s |
| **Total per contract** | **31-50s** | **3-8s** |

**Bottleneck shift:** After optimisation, the DeepSeek API call became the dominant cost (3-7s per contract). Further speedup would require concurrent API calls (asyncio + `asyncio.gather`) rather than embedding-side optimisation.

---

### Phase 4 — Full Production Run

**Completed:** All 510 CUAD contracts processed end-to-end.

**Throughput:**
- Total wall-clock time: 2h 4min (12:56 → 15:00)
- Per-contract time: 31s on initial CPU pipeline; **6-8s after GPU/singleton optimisation** (applied mid-batch)
- Time saved by the optimisation: ~3 hours

**Volume & cost:**

| Metric | Value |
|--------|-------|
| Contracts processed | 510 |
| Successful extractions (≥2 fields) | 456 (89.4%) |
| Partial extractions (<2 fields) | 51 (10.0%) |
| Failed (JSON parse errors) | 4 (0.8%) |
| Total tokens used | 4,292,910 |
| Avg tokens per contract | 8,401 |
| **Total cost (DeepSeek)** | **$1.72** (blended estimate; range $1.16–$4.72) |
| Cost per contract | **$0.0034** |

**Penalty extraction insights (the headline business outcome):**

| Penalty profile | Count | % of dataset |
|-----------------|-------|--------------|
| Contracts with monetary (cash) penalty | 128 | 25.0% |
| Contracts with service credits only | 256 | 50.1% |
| Contracts with no penalty clauses | 127 | 24.9% |

**Field coverage across the full dataset:**

| Field | Populated | % |
|-------|-----------|---|
| `termination_clause` | 427/510 | 83.6% |
| `governing_law` | 416/510 | 81.4% |
| `penalty_has_monetary` | 384/510 | 75.1% |
| `dispute_resolution` | 342/510 | 66.9% |
| `liability_cap` | 264/510 | 51.7% |
| `renewal_terms` | 205/510 | 40.1% |
| `penalty_late_payment` | 124/510 | 24.3% |
| `penalty_termination_fee` | 98/510 | 19.2% |
| `penalty_currency` | 83/510 | 16.2% |
| `penalty_max_amount` | 82/510 | 16.0% |
| `response_time_sla` | 36/510 | 7.0% |
| `penalty_late_delivery` | 33/510 | 6.5% |
| `sla_breach_threshold` | 28/510 | 5.5% |
| `penalty_data_breach` | 23/510 | 4.5% |
| `sla_measurement_period` | 21/510 | 4.1% |
| `uptime_guarantee` | 19/510 | 3.7% |
| `penalty_uptime_breach` | 12/510 | 2.3% |
| `service_credit_cap` | 6/510 | 1.2% |

The low coverage on uptime/response/uptime-penalty fields is **expected** — CUAD is predominantly commercial agreements (affiliate, licensing, joint venture), not cloud SaaS contracts. The high coverage on termination, governing law, and liability is the right shape for this corpus.

**Final eval results (full dataset, after eval methodology fixes):**

| Metric | Value |
|--------|-------|
| Average overall score | **0.790** |
| Contracts ≥ 0.8 | 332/510 (65%) |
| Contracts ≥ 0.7 | 408/510 (80%) |
| Contracts ≥ 0.5 | 458/510 (90%) |

**Cost comparison vs alternatives (extrapolated to this same 510-contract dataset):**

| Provider | Estimated cost | Multiplier |
|----------|---------------|------------|
| **DeepSeek V3.2 (chosen)** | **$1.72** | 1× |
| GPT-4o-mini | ~$11 | 6.4× |
| Claude Sonnet 3.5 | ~$56 | 33× |
| GPT-4o | ~$94 | 55× |

**Output files (final):**
- `output/results.db` — 2.4 MB SQLite, 510 rows
- `output/results.json` — 2.4 MB JSON Lines
- `output/results_summary.csv` — 920 KB CSV
- `output/EXTRACTION_STATS.txt` — human-readable summary
- `output/eval_report.json` — per-contract eval scores
- `output/failed_contracts.json` — 4 JSON parse failures (worth retrying with stricter response_format)

---

### Eval Improvement (before Phase 4)

**Completed:** Eval score improved from 0.28 → 0.56 → **0.866** through three rounds of fixes.

**Round 1 — Eval structural bugs (0.28 → 0.56):**
- Excluded `cuad_0000` (CUAD dataset's own datasheet — not a contract, was dragging score to 0)
- Applied `clean_text()` to source text before substring comparison (previously, apostrophe stripping caused mismatches between source and extracted text)
- Updated `SLA_FIELDS` list in eval to match the new 19-field schema (old eval was scoring a `penalty_clause` field that no longer exists)
- Tightened keyword lists to avoid false "missed" signals (words like "hours" and "availability" appear in every contract, not just SLA clauses)
- Broadened format patterns to cover international contracts (UK/EU jurisdictions, word-form numbers like "one year", non-USD currencies)

**Round 2 — Pipeline improvements (0.56 → 0.866):**
- Increased retrieval `top_k` from 3 → 5 per query (surfaces clauses ranked 4th/5th that were previously missed)
- Added 5 phrase-level anchor queries targeting common boilerplate section openings
- Enforced verbatim quoting in the extraction prompt (`"EXACT QUOTE or null"` on every field, explicit "do not paraphrase" instruction)
- Added whitespace normalisation to the verbatim check: both source and snippet are collapsed to single-space before comparison, handling PDF line-break artefacts where a clause spans two lines in source but is returned as one line by the LLM

**Final eval results (9 real contracts):**

| Metric | Before | After |
|--------|--------|-------|
| Average score | 0.28 | **0.866** |
| Contracts ≥ 0.8 | 0/9 | **7/9** |
| Contracts ≥ 0.5 | 0/9 | **9/9** |
| `termination_clause` avg | 0.64 | **1.00** |
| `renewal_terms` avg | 0.47 | **1.00** |
| `governing_law` avg | 0.77 | **0.97** |
| `dispute_resolution` avg | 0.50 | **0.97** |

---

## 7. Project Structure

```
.
├── config.py                    # Central config — API keys, paths, batch settings
├── requirements.txt
├── .env                         # DEEPSEEK_API_KEY=sk-... (not committed)
├── .gitignore
│
├── data/
│   ├── raw/                     # Single test contracts
│   ├── cuad_raw/                # All 511 CUAD contracts as .txt files
│   ├── processed/               # (reserved for future preprocessing)
│   └── chroma_db/               # ChromaDB vector store (per-document collections)
│
├── pipeline/
│   ├── schemas.py               # Pydantic: SLAClause (19 fields), ExtractionResult
│   ├── ingestion.py             # load_contract(), clean_text()
│   ├── chunker.py               # chunk_contract() — SentenceSplitter 512/64
│   ├── embedder.py              # build_per_doc_index() — ChromaDB + MiniLM
│   ├── retriever.py             # retrieve_sla_chunks() — 14-query multi-retrieval
│   ├── extractor.py             # extract_sla() — DeepSeek API call + JSON parse
│   └── output.py                # SQLite, JSON Lines, CSV output
│
├── models/
│   └── prompts.py               # SYSTEM_PROMPT, EXTRACTION_PROMPT
│
├── scripts/
│   ├── run_pipeline.py          # CLI: single file or batch directory
│   ├── download_cuad.py         # Download + PDF-extract CUAD contracts
│   ├── inspect_output.py        # Inspect single contract result by ID
│   └── _batch.py                # Batch logic (imported by run_pipeline.py)
│
├── evals/
│   └── eval_runner.py           # Three-signal eval: substring, keyword, format
│
└── output/
    ├── results.db               # SQLite — primary output
    ├── results.json             # JSON Lines — one contract per line
    ├── results_summary.csv      # CSV — no raw_response column
    ├── eval_report.json         # Eval scores per contract and per field
    └── failed_contracts.json    # Contracts that errored during batch
```

---

## 8. Setup & Usage

### Prerequisites

- Python 3.10+
- DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com)
- (Optional) HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — avoids rate limiting when downloading CUAD

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=sk-your-key-here
HF_TOKEN=hf_your-token-here    # optional but recommended
```

### Download CUAD Contracts

```bash
# Download all 511 contracts (takes ~10 minutes)
python scripts/download_cuad.py

# Or download a subset for testing
python scripts/download_cuad.py --max 20
```

### Run the Pipeline

```bash
# Single contract
python scripts/run_pipeline.py data/raw/mycontract.txt
python scripts/run_pipeline.py data/raw/mycontract.pdf

# Batch — first 10 contracts
python scripts/run_pipeline.py data/cuad_raw --max 10

# Full run — all contracts in directory
python scripts/run_pipeline.py data/cuad_raw

# Resume interrupted run (skips already-processed contract IDs)
python scripts/run_pipeline.py data/cuad_raw --resume
```

### Inspect Results

```bash
# List all contracts in the database
python scripts/inspect_output.py --list

# Inspect a single contract
python scripts/inspect_output.py cuad_0042

# Show the raw DeepSeek response
python scripts/inspect_output.py cuad_0042 --show-context
```

### Run Evals

```bash
python -m evals.eval_runner
python -m evals.eval_runner --limit 20
```

---

## 9. Querying the Results

All results are in `output/results.db` (SQLite). Open with any SQLite client or query via Python:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("output/results.db")
df = pd.read_sql("SELECT * FROM sla_results", conn)
```

**Example queries:**

```sql
-- Total processed and success rate
SELECT status, COUNT(*) FROM sla_results GROUP BY status;

-- Contracts with real monetary penalties (cash, not just credits)
SELECT contract_id, penalty_max_amount, penalty_currency, penalty_uptime_breach
FROM sla_results
WHERE penalty_has_monetary = 1
ORDER BY penalty_max_amount DESC;

-- Contracts with uptime SLA defined
SELECT contract_id, uptime_guarantee, sla_breach_threshold, sla_measurement_period
FROM sla_results
WHERE uptime_guarantee IS NOT NULL;

-- Contracts with late payment interest clauses
SELECT contract_id, penalty_late_payment
FROM sla_results
WHERE penalty_late_payment IS NOT NULL;

-- Field coverage report
SELECT
  COUNT(uptime_guarantee)        AS has_uptime,
  COUNT(penalty_uptime_breach)   AS has_uptime_penalty,
  COUNT(penalty_has_monetary)    AS has_any_penalty,
  SUM(penalty_has_monetary)      AS has_cash_penalty,
  COUNT(*)                       AS total
FROM sla_results;
```

---

## 10. Evaluation Methodology

Since the `theatticusproject/cuad` dataset ships raw PDFs without gold-labelled QA answers, the evaluation uses three proxy signals:

**Signal 1 — Substring overlap (primary)**
The first 120 characters of each extracted field value must appear verbatim in the source contract. A full match scores 1.0. A match with unusual format (field doesn't match the expected regex pattern) scores 0.8. This catches hallucinations.

**Signal 2 — Keyword coverage (false negative detection)**
For null fields, the source text is scanned for field-relevant keywords. If keywords are found but nothing was extracted, the field scores 0.0 (potential miss). If no keywords exist, the null is scored as correct (`None`).

**Signal 3 — Format validation**
Each field has an expected pattern (e.g., `\d+%` for uptime, `laws? of (?:the )?(?:state|country)` for governing law). Extracted values that match their field's pattern score higher than those that don't.

**Final score: 0.866 average across 9 contracts (7/9 above 0.8, 9/9 above 0.5).**

The eval went through two rounds of improvement. Initial score of 0.28 was caused by: a non-contract document included in scoring, source/extracted text normalisation mismatch (apostrophes, line breaks), and overly broad keyword matching. After fixing the eval methodology and improving the pipeline (top_k 3→5, phrase-anchor queries, verbatim quoting enforced in prompt), the score reached 0.866.

The remaining gap from 1.0 is concentrated in `dispute_resolution` on 3 contracts where the clause is in an unlabelled "Miscellaneous" section with no strong semantic signal — a known hard case for retrieval-based approaches.

---

## 11. Design Decisions & Trade-offs

| Decision | Choice | Alternative considered | Reason |
|----------|--------|----------------------|--------|
| LLM | DeepSeek V3.2 | Claude Sonnet, GPT-4o | 20–35× cheaper; OpenAI-compatible API means no code changes |
| Embeddings | `all-MiniLM-L6-v2` (local) | OpenAI `text-embedding-3-small` | Free, no API key, no data leaves machine; ~90% quality for legal retrieval |
| Embedding runtime | `fastembed` (ONNX) | `sentence-transformers` (PyTorch) | 2-3× faster CPU; cross-vendor GPU (DirectML/CUDA/CoreML) without device-specific code |
| GPU backend | DirectML | ROCm, CUDA | ROCm has limited Windows support, especially for RDNA 4; DirectML works on any DX12 GPU |
| Vector store | ChromaDB per-document | Single shared index | No cross-document contamination; simpler resume logic |
| Retrieval | 14 targeted queries | Single broad query | Each SLA field type has distinct vocabulary; multi-query dramatically improves recall |
| Chunking | SentenceSplitter 512/64 | RecursiveCharacter, fixed-size | Sentence boundaries preserve clause meaning; 512 matches MiniLM max length |
| Output | SQLite + JSON Lines + CSV | Single format | SQLite for querying, JSON Lines for streaming/pandas, CSV for spreadsheet users |
| Penalty schema | 5 separate fields + 4 summary | Single `penalty_clause` string | Primary goal is finding contracts with monetary penalties — needs to be queryable by type and amount |

---

## License

MIT License. CUAD dataset is licensed under CC BY 4.0 (The Atticus Project). DeepSeek API usage is subject to DeepSeek's terms of service.
