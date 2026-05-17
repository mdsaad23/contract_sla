# SLA Extraction RAG Pipeline — Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT CONTRACTS                             │
│                    (PDF or TXT files in data/)                       │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │   INGESTION (Step 1)   │
                    │  • PDF text extraction │
                    │  • TXT file loading    │
                    └────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │  TEXT CLEANING (Step 2)│
                    │  • Line ending norm.   │
                    │  • Whitespace collapse │
                    │  • Non-ASCII removal   │
                    └────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │  CHUNKING (Step 3)     │
                    │  • SentenceSplitter    │
                    │  • 512 tokens, 64 overlap
                    │  • Sentence boundaries │
                    └────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │  EMBEDDING (Step 4)    │
                    │  • all-MiniLM-L6-v2    │
                    │  • FastEmbed (ONNX)    │
                    │  • GPU accel (DirectML)│
                    │  • 384-dim vectors     │
                    └────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │ VECTOR STORE (Step 5)  │
                    │  • ChromaDB persistence│
                    │  • Per-document index  │
                    │  • No cross-contamina. │
                    └────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │ RETRIEVAL (Step 6)     │
                    │  • 19 targeted queries │
                    │  • Top-5 per query     │
                    │  • Deduplication       │
                    │  • 12K-20K chars ctx   │
                    └────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │ EXTRACTION (Step 7)    │
                    │  • DeepSeek V3.2 API   │
                    │  • 19-field schema     │
                    │  • Verbatim quoting    │
                    │  • JSON parse + retry  │
                    └────────────────────────┘
                                 ↓
        ┌───────────────────────┬────────────────────┐
        ↓                       ↓                    ↓
   ┌─────────────┐      ┌──────────────┐    ┌────────────────┐
   │   SQLite    │      │  JSON Lines  │    │      CSV       │
   │  results.db │      │ results.json │    │ results_summary│
   │   (primary) │      │  (streaming) │    │  (spreadsheet) │
   └─────────────┘      └──────────────┘    └────────────────┘
```

## Component Details

### 1. Ingestion (`pipeline/ingestion.py`)

**Purpose:** Load contract text from PDF or TXT files.

**Flow:**
- PDF: Extract page-by-page via `pypdf.PdfReader`
- TXT: Read directly with `encoding="utf-8", errors="replace"`

**Output:** Raw contract text (30K–200K characters)

---

### 2. Text Cleaning (`pipeline/ingestion.py`)

**Purpose:** Normalize text from PDF extraction artifacts.

**Operations:**
1. Line ending normalization (`\r\n` / `\r` → `\n`)
2. Excessive newline collapse (3+ → 2)
3. Whitespace collapse (2+ spaces/tabs → 1)
4. Non-ASCII removal (control characters, encoding artifacts)

**Why:** PDFs often have repeated whitespace, hyphenated words, and rendering artifacts that break clause boundaries.

---

### 3. Chunking (`pipeline/chunker.py`)

**Purpose:** Break contract into meaningful chunks for embedding.

**Algorithm:** `LlamaIndex.SentenceSplitter`
- **Chunk size:** 512 tokens (matches MiniLM max input length)
- **Overlap:** 64 tokens (~3-4 sentences)
- **Strategy:** Never cuts mid-sentence; completes current sentence before truncating

**Why 512 tokens:**
- Large enough to capture a clause + context
- Small enough that embedding model can encode without truncation
- Produces 25–80 chunks per typical 10K-word contract

**Output:** 15–80 chunks per contract, deduplicated

---

### 4. Embedding (`pipeline/embedder.py`)

**Purpose:** Convert text chunks into dense vectors for semantic search.

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture:** 6-layer distilled BERT (22.7M parameters)
- **Output:** 384-dimensional vectors
- **Training:** 1B sentence pairs, contrastive learning
- **License:** Apache 2.0 (commercial use OK)

**Runtime:** `fastembed` (ONNX Runtime)
- **CPU:** 2-3× faster than PyTorch
- **GPU:** DirectML (AMD/Intel/NVIDIA via DX12)
- **Pattern:** Module-level singleton (load once per process, reuse)

**Benchmarks:**
- Per-contract embedding: 0.10–0.28s (CPU) or <0.10s (GPU)
- Total startup: 0.23s (first load), then ~0ms overhead

---

### 5. Vector Store (`pipeline/embedder.py`)

**Purpose:** Index and retrieve semantically similar chunks.

**Store:** ChromaDB with persistence
- **Scope:** Per-document (one collection per contract)
- **Naming:** `contract_{contract_id[:40]}`
- **Persistence:** `./data/chroma_db/`

**Trade-offs:**
- ✅ No cross-document contamination
- ✅ Simple resume logic (recreate per contract)
- ❌ No cross-document retrieval (by design)

---

### 6. Retrieval (`pipeline/retriever.py`)

**Purpose:** Find SLA-relevant clauses for the LLM to analyze.

**Strategy:** Multi-query with deduplication

**Query groups (19 total):**

| Category | Count | Examples |
|----------|-------|----------|
| Performance SLAs | 3 | uptime, response time, breach threshold |
| Monetary penalties | 6 | liquidated damages, late fees, data breach |
| Contract mechanics | 6 | renewal, termination, liability, law, dispute |
| Phrase anchors | 4 | Boilerplate: "governed by", "in the event of", etc. |

**Per-query parameters:**
- `top_k=5` (increased from 3 for better recall)
- Deduplicated across all 19 queries
- Joined with `---` separators

**Output:** 12K–20K character context block

---

### 7. Extraction (`pipeline/extractor.py`)

**Purpose:** Parse SLA clauses into 19 structured fields using an LLM.

**Model:** DeepSeek V3.2
- **API:** OpenAI-compatible endpoint (`https://api.deepseek.com`)
- **Max tokens:** 1,500 response
- **Format:** Pure JSON only (no markdown)

**Extraction constraints:**
- Every field labelled `"EXACT QUOTE or null"`
- System prompt enforces verbatim quoting (no paraphrase)
- Missing fields → `null` (not empty string)
- Boolean `penalty_has_monetary` distinguishes cash from credits

**JSON parsing (fallback chain):**
1. Try `json.loads()` directly
2. Strip markdown fences if present
3. Extract first `{...}` block with string find/rfind

**Status classification:**
- `success` — 2+ fields populated
- `partial` — 0-1 fields populated
- `failed` — API error or parse failure

---

### 8. Output Storage (`pipeline/output.py`)

**Purpose:** Persist extraction results in multiple formats.

**SQLite** (`output/results.db`)
- One row per contract
- All 19 fields as columns
- `penalty_has_monetary` as INTEGER (0/1/NULL)
- `raw_response` for debugging
- `INSERT OR REPLACE` for idempotent re-runs

**JSON Lines** (`output/results.json`)
- One JSON object per line
- All fields including `raw_response`
- Streamable with pandas

**CSV** (`output/results_summary.csv`)
- Same as JSON Lines, minus `raw_response` and `error`
- Spreadsheet-friendly

---

## Performance Characteristics

### Speed (per contract)

| Phase | Configuration | Time |
|-------|---|---|
| Ingestion | Text extraction | 0.1–0.3s |
| Cleaning | Regex normalization | <0.01s |
| Chunking | SentenceSplitter (512/64) | 0.05–0.15s |
| Embedding (CPU) | PyTorch (old) | 10–15s |
| Embedding (CPU) | ONNX singleton | 0.10–0.28s |
| Embedding (GPU) | DirectML | <0.10s |
| Retrieval | 19 queries × 5 chunks | 0.37–0.43s |
| Extraction | DeepSeek API call | 3–7s |
| Output | SQLite/JSON/CSV write | 0.1s |
| **Total (CPU)** | | ~12–18s |
| **Total (GPU)** | | **3–8s** |

### Throughput

- **Without GPU:** ~6 contracts/minute (12–18s each)
- **With GPU (DirectML):** ~8 contracts/minute (3–8s each, bottleneck is API latency)
- **Full 510-contract run:** 2h 4min wall-clock with GPU

### Cost

| Phase | Cost |
|-------|------|
| Embeddings | **$0** (local model, no API key) |
| Retrieval | **$0** (local ChromaDB) |
| Extraction | **$0.0034** per contract (DeepSeek V3.2) |
| **Total per contract** | **$0.0034** |

510 contracts = **$1.72 total**

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **DeepSeek** over Claude/GPT-4o | 20–35× cheaper; OpenAI-compatible API |
| **Local embeddings** over OpenAI API | Free, no data leaves machine, sufficient quality |
| **ONNX Runtime** over PyTorch | 2-3× faster, cross-vendor GPU support |
| **DirectML** over CUDA/ROCm | Works on any DX12 GPU; better Windows support |
| **Per-document index** over shared | No contamination; clean resume logic |
| **19 targeted queries** over 1 broad | SLA fields have distinct vocabularies |
| **SentenceSplitter** over fixed-size | Preserves clause meaning at boundaries |
| **3 output formats** over 1 | SQLite for querying, JSON for streaming, CSV for sheets |
| **5 penalty fields** over 1 generic | Enables: "which contracts have cash penalty > $X?" |

---

## Error Handling

### Common Failure Modes

| Scenario | Handling |
|----------|----------|
| PDF corruption | `pypdf` falls back to empty text per page |
| Encoding issues | UTF-8 with `errors="replace"` → substitution chars |
| JSON parse error | Fallback chain: markdown strip → brute-force extraction |
| API timeout | Logged as `failed`; resume logic skips on re-run |
| Missing fields | Set to `null` (not empty string) |

### Resume Support

- Track processed contract IDs in SQLite
- `--resume` flag skips already-processed IDs
- Failed contracts logged separately in `failed_contracts.json`
- Safe to interrupt and re-run without data loss
