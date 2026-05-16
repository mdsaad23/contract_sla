# SLA Extraction RAG Pipeline — Weekend Project Plan

**Goal:** Extract SLA clauses from 2,000+ contracts using a per-document RAG pipeline, producing a structured, queryable output file.
**Stack:** Python, LlamaIndex / LangChain, ChromaDB, Anthropic/OpenAI API, SQLite + JSON
**Target:** Weekend build — Saturday MVP, Sunday hardening

---

## Project Structure

```
sla-extractor/
│
├── data/
│   ├── raw/                     # Original contract PDFs / text files
│   ├── processed/               # Cleaned text per contract
│   └── cuad_labels/             # CUAD gold labels for eval (optional)
│
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py             # Load + clean contract text
│   ├── chunker.py               # Document chunking strategy
│   ├── embedder.py              # Embedding model wrapper
│   ├── retriever.py             # Per-document retrieval logic
│   ├── extractor.py             # LLM extraction + structured output
│   └── aggregator.py            # Merge results across contracts
│
├── models/
│   ├── schemas.py               # Pydantic models for SLA fields
│   └── prompts.py               # All prompt templates
│
├── output/
│   ├── results.db               # SQLite — primary queryable store
│   ├── results.json             # JSON Lines — portable flat file
│   └── results_summary.csv      # Human-readable summary
│
├── evals/
│   ├── eval_runner.py           # Run evals against gold labels
│   ├── metrics.py               # Precision, recall, field accuracy
│   └── sample_contracts/        # 5-10 hand-labelled test contracts
│
├── scripts/
│   ├── run_pipeline.py          # Main entry point
│   ├── run_evals.py             # Standalone eval runner
│   └── inspect_output.py        # CLI to query results.db
│
├── config.py                    # All tuneable parameters
├── requirements.txt
└── README.md
```

---

## Output Schema (Pydantic)

Define this before writing any pipeline code. Everything else serves this shape.

```python
# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SLAClause(BaseModel):
    contract_id: str                          # Filename or UUID
    contract_name: Optional[str]              # Vendor / party name if extractable
    
    # Core SLA fields
    uptime_guarantee: Optional[str]           # e.g. "99.9%"
    response_time_p1: Optional[str]           # Priority 1 incident response
    response_time_p2: Optional[str]
    resolution_time_p1: Optional[str]
    resolution_time_p2: Optional[str]
    penalty_clause: Optional[str]             # Service credits / penalties
    exclusions: Optional[str]                 # What is excluded from SLA
    measurement_window: Optional[str]         # Monthly / quarterly / annual
    reporting_frequency: Optional[str]
    escalation_path: Optional[str]
    termination_for_breach: Optional[str]     # Right to terminate if SLA missed
    
    # Provenance
    source_chunks: list[str] = Field(default_factory=list)  # Raw chunks used
    confidence: Optional[str]                 # high / medium / low
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str]                      # Anything LLM flagged as ambiguous

class ExtractionResult(BaseModel):
    contract_id: str
    status: str                               # success / partial / failed
    sla: Optional[SLAClause]
    error: Optional[str]
```

---

## Phase 0 — Setup (Friday Evening, ~1 hour)

### 0.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
  llama-index \
  llama-index-vector-stores-chroma \
  chromadb \
  anthropic \
  openai \
  pydantic \
  pypdf \
  python-dotenv \
  tqdm \
  pandas \
  pytest
```

```bash
# .env
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...        # Optional - only if using OpenAI embeddings
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=claude-sonnet-4-20250514
```

### 0.2 Get Contract Data

**Primary — CUAD Dataset (recommended for weekend build)**

```bash
# Download via HuggingFace datasets
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('cuad', split='train')
ds.save_to_disk('./data/cuad_raw')
print(f'Loaded {len(ds)} contracts')
"
```

CUAD gives you 510 real contracts + 41 labelled clause types per contract.
The `sla_or_maintenance` category maps directly to your use case.
Use 10 contracts as a labelled eval set and run the pipeline on the rest.

**Supplementary — SEC EDGAR (messier, more realistic)**

```python
# scripts/fetch_edgar.py — pull material contracts (Exhibit 10)
import requests

def fetch_edgar_contracts(n=50):
    url = "https://efts.sec.gov/LATEST/search-index?q=%22exhibit+10%22&dateRange=custom&startdt=2020-01-01&enddt=2024-01-01&_source=file_date,period_of_report,entity_name,file_num,form_type&forms=8-K"
    # Iterate results, download filing index, pull Exhibit 10 document URLs
    pass
```

---

## Phase 1 — MVP (Saturday Morning, ~4 hours)

Goal: single contract in → structured SLA JSON out.

### 1.1 Ingestion

```python
# pipeline/ingestion.py
from pathlib import Path
from pypdf import PdfReader

def load_contract(path: str) -> dict:
    """Load a contract file and return {id, text, metadata}."""
    p = Path(path)
    if p.suffix == ".pdf":
        reader = PdfReader(str(p))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")
    
    return {
        "contract_id": p.stem,
        "path": str(p),
        "text": text,
        "char_count": len(text),
        "page_count": len(reader.pages) if p.suffix == ".pdf" else None
    }

def clean_text(text: str) -> str:
    """Remove boilerplate noise — headers, page numbers, repeated whitespace."""
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text.strip()
```

### 1.2 Chunking

The right chunking strategy is the most consequential decision in the pipeline.

```python
# pipeline/chunker.py
from llama_index.core.node_parser import SentenceSplitter

def chunk_contract(text: str, contract_id: str) -> list[dict]:
    """
    Chunk at 512 tokens with 64 token overlap.
    
    Why 512 / 64:
    - SLA clauses are typically 100-300 tokens — fits in one chunk
    - 64-token overlap ensures clause boundaries are not split
    - Small enough that per-doc ChromaDB stays lightweight
    """
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents([
        {"text": text, "metadata": {"contract_id": contract_id}}
    ])
    return [
        {
            "chunk_id": f"{contract_id}_{i}",
            "contract_id": contract_id,
            "text": node.get_content(),
            "chunk_index": i
        }
        for i, node in enumerate(nodes)
    ]
```

**Chunking decision table — choose based on your contract type:**

| Contract type | Chunk size | Overlap | Notes |
|---|---|---|---|
| Well-structured (numbered sections) | 256–512 | 32 | Sections are natural boundaries |
| Dense legal prose | 512–768 | 128 | More overlap to catch cross-para clauses |
| Scanned PDFs (noisy OCR) | 768–1024 | 128 | Larger chunks tolerate OCR errors |
| Mixed (CUAD) | 512 | 64 | Good default |

### 1.3 Per-Document Embedding + Retrieval

```python
# pipeline/embedder.py + retriever.py
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def build_per_doc_index(chunks: list[dict], contract_id: str) -> chromadb.Collection:
    """
    Create an isolated ChromaDB collection per contract.
    This is the core architectural decision — no cross-contract bleed.
    """
    client = chromadb.EphemeralClient()  # In-memory; switch to PersistentClient for caching
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
    
    collection = client.create_collection(
        name=f"contract_{contract_id}",
        embedding_function=ef
    )
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["chunk_id"] for c in chunks],
        metadatas=[{"contract_id": c["contract_id"], "chunk_index": c["chunk_index"]} for c in chunks]
    )
    return collection


SLA_QUERIES = [
    "service level agreement uptime availability percentage",
    "response time incident priority P1 P2",
    "resolution time fix deadline",
    "penalty credit service breach",
    "SLA exclusions exceptions maintenance window",
    "measurement reporting period monthly quarterly",
    "escalation process support contact",
    "termination right breach SLA failure"
]

def retrieve_sla_chunks(collection: chromadb.Collection, n_results: int = 3) -> list[str]:
    """
    Run multiple targeted queries against the per-doc index.
    Union the results — this ensures we catch all SLA fields.
    """
    seen = set()
    all_chunks = []
    
    for query in SLA_QUERIES:
        results = collection.query(query_texts=[query], n_results=n_results)
        for doc in results["documents"][0]:
            if doc not in seen:
                seen.add(doc)
                all_chunks.append(doc)
    
    return all_chunks
```

### 1.4 LLM Extraction

```python
# models/prompts.py
SYSTEM_PROMPT = """You are a contract analyst. Extract SLA information from the provided contract excerpts.
Be precise. If a field is not mentioned, return null — do not invent values.
Return confidence as:
- high: explicit numeric or date values stated clearly
- medium: implied or inferred from context
- low: uncertain, reconstructed from scattered references"""

EXTRACTION_PROMPT = """Extract all SLA-related information from these contract excerpts.

CONTRACT ID: {contract_id}

EXCERPTS:
{chunks}

Return a JSON object matching this exact schema:
{schema}

Rules:
- Use exact quoted values where possible (e.g. "99.9%" not "approximately 99%")
- For time values, normalise to the unit stated in the contract (hours, days, business days)
- If multiple SLA tiers exist, capture the most critical (P1 / highest severity)
- Set confidence to "low" if you are reconstructing from implied context"""
```

```python
# pipeline/extractor.py
import anthropic
import json
from models.schemas import SLAClause
from models.prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT

client = anthropic.Anthropic()

def extract_sla(contract_id: str, chunks: list[str]) -> SLAClause:
    schema_str = json.dumps(SLAClause.model_json_schema(), indent=2)
    chunk_text = "\n\n---\n\n".join(chunks)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(
                contract_id=contract_id,
                chunks=chunk_text,
                schema=schema_str
            )
        }]
    )
    
    raw = message.content[0].text
    
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    data = json.loads(raw.strip())
    data["contract_id"] = contract_id
    data["source_chunks"] = chunks
    
    return SLAClause(**data)
```

### 1.5 MVP Runner

```python
# scripts/run_pipeline.py (MVP — single contract)
from pipeline.ingestion import load_contract, clean_text
from pipeline.chunker import chunk_contract
from pipeline.embedder import build_per_doc_index
from pipeline.retriever import retrieve_sla_chunks
from pipeline.extractor import extract_sla
from models.schemas import ExtractionResult
import json
from datetime import datetime

def process_contract(path: str) -> ExtractionResult:
    try:
        doc = load_contract(path)
        doc["text"] = clean_text(doc["text"])
        chunks = chunk_contract(doc["text"], doc["contract_id"])
        collection = build_per_doc_index(chunks, doc["contract_id"])
        sla_chunks = retrieve_sla_chunks(collection)
        sla = extract_sla(doc["contract_id"], sla_chunks)
        return ExtractionResult(contract_id=doc["contract_id"], status="success", sla=sla)
    except Exception as e:
        return ExtractionResult(contract_id=doc["contract_id"], status="failed", error=str(e))

if __name__ == "__main__":
    result = process_contract("data/raw/sample_contract.pdf")
    print(json.dumps(result.model_dump(), indent=2, default=str))
```

**Saturday morning milestone:** One contract → valid JSON with populated SLA fields.

---

## Phase 2 — Scale to All Contracts (Saturday Afternoon, ~3 hours)

Goal: run the pipeline across all contracts with batching, error handling, and progress tracking.

### 2.1 Batch Runner with Resume Support

```python
# scripts/run_pipeline.py (full batch)
from pathlib import Path
from tqdm import tqdm
import json
import sqlite3
from pipeline.output import save_result, init_db

def run_batch(input_dir: str, max_workers: int = 4):
    contracts = list(Path(input_dir).glob("**/*.pdf")) + \
                list(Path(input_dir).glob("**/*.txt"))
    
    # Resume: skip already-processed contracts
    db = init_db("output/results.db")
    processed = get_processed_ids(db)
    contracts = [c for c in contracts if c.stem not in processed]
    
    print(f"Processing {len(contracts)} contracts ({len(processed)} already done)")
    
    results = []
    failed = []
    
    for contract_path in tqdm(contracts, desc="Extracting SLAs"):
        result = process_contract(str(contract_path))
        save_result(db, result)
        
        if result.status == "failed":
            failed.append({"id": result.contract_id, "error": result.error})
        else:
            results.append(result)
    
    print(f"\nDone. Success: {len(results)} | Failed: {len(failed)}")
    if failed:
        with open("output/failed_contracts.json", "w") as f:
            json.dump(failed, f, indent=2)
    
    export_outputs(db)
```

### 2.2 Output Storage

```python
# pipeline/output.py
import sqlite3
import json
import csv
from models.schemas import ExtractionResult

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sla_results (
            contract_id TEXT PRIMARY KEY,
            status TEXT,
            uptime_guarantee TEXT,
            response_time_p1 TEXT,
            response_time_p2 TEXT,
            resolution_time_p1 TEXT,
            resolution_time_p2 TEXT,
            penalty_clause TEXT,
            exclusions TEXT,
            measurement_window TEXT,
            reporting_frequency TEXT,
            escalation_path TEXT,
            termination_for_breach TEXT,
            confidence TEXT,
            notes TEXT,
            source_chunks TEXT,           -- JSON array stored as text
            extraction_timestamp TEXT,
            error TEXT
        )
    """)
    conn.commit()
    return conn

def save_result(conn: sqlite3.Connection, result: ExtractionResult):
    sla = result.sla
    conn.execute("""
        INSERT OR REPLACE INTO sla_results VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, (
        result.contract_id,
        result.status,
        sla.uptime_guarantee if sla else None,
        sla.response_time_p1 if sla else None,
        sla.response_time_p2 if sla else None,
        sla.resolution_time_p1 if sla else None,
        sla.resolution_time_p2 if sla else None,
        sla.penalty_clause if sla else None,
        sla.exclusions if sla else None,
        sla.measurement_window if sla else None,
        sla.reporting_frequency if sla else None,
        sla.escalation_path if sla else None,
        sla.termination_for_breach if sla else None,
        sla.confidence if sla else None,
        sla.notes if sla else None,
        json.dumps(sla.source_chunks) if sla else None,
        sla.extraction_timestamp.isoformat() if sla else None,
        result.error
    ))
    conn.commit()

def export_outputs(conn: sqlite3.Connection):
    """Export to JSON Lines and CSV for portability."""
    rows = conn.execute("SELECT * FROM sla_results").fetchall()
    cols = [d[0] for d in conn.execute("PRAGMA table_info(sla_results)").fetchall()]
    
    # JSON Lines
    with open("output/results.json", "w") as f:
        for row in rows:
            record = dict(zip(cols, row))
            f.write(json.dumps(record) + "\n")
    
    # CSV summary (human-readable, excludes raw chunks)
    skip = {"source_chunks"}
    csv_cols = [c for c in cols if c not in skip]
    with open("output/results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for row in rows:
            record = {k: v for k, v in zip(cols, row) if k not in skip}
            writer.writerow(record)
    
    print("Exported: output/results.json, output/results_summary.csv")
```

### 2.3 Cost + Rate Limit Management

```python
# config.py
import os

BATCH_SIZE = 10                  # Contracts per batch before a short pause
PAUSE_BETWEEN_BATCHES = 2        # Seconds
MAX_CHUNKS_PER_CONTRACT = 25     # Cap to control token spend
MAX_TOKENS_PER_CALL = 1500
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "claude-sonnet-4-20250514"

# Estimated cost for 2,000 contracts:
# Embeddings: 2000 contracts × ~100 chunks × 512 tokens = ~100M tokens ≈ $1.00
# LLM extraction: 2000 × ~6000 input tokens + 1500 output ≈ $25-40 total
# Total estimate: $30-45 for full run. Use --sample flag to test on 50 first.
```

**Saturday afternoon milestone:** 50 contracts processed, results in `results.db` and `results.json`.

---

## Phase 3 — Evals (Saturday Evening, ~2 hours)

Goal: know whether your extractions are correct before scaling to 2,000.

### 3.1 Eval Strategy

Use CUAD's gold labels as ground truth. CUAD includes a `sla_or_maintenance` label with highlighted spans per contract — compare your extracted values against those spans.

```python
# evals/eval_runner.py
from datasets import load_from_disk
import json

def load_cuad_labels(path="data/cuad_raw") -> dict:
    """Return {contract_id: {field: gold_value}} for SLA-relevant fields."""
    ds = load_from_disk(path)
    labels = {}
    for row in ds:
        cid = row["title"].replace(" ", "_")
        labels[cid] = {
            "sla_maintenance": row.get("sla_or_maintenance", []),
        }
    return labels

def score_extraction(predicted: str, gold_spans: list[str]) -> dict:
    """
    Simple token overlap score.
    For a production system, use ROUGE or BERTScore.
    """
    if not predicted or not gold_spans:
        return {"match": False, "score": 0.0}
    
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(" ".join(gold_spans).lower().split())
    
    if not gold_tokens:
        return {"match": True, "score": 1.0}  # No gold = nothing to miss
    
    overlap = pred_tokens & gold_tokens
    score = len(overlap) / len(gold_tokens)
    return {"match": score > 0.5, "score": round(score, 3)}

def run_eval(results_db_path: str, cuad_labels: dict):
    import sqlite3
    conn = sqlite3.connect(results_db_path)
    rows = conn.execute("SELECT contract_id, uptime_guarantee, response_time_p1 FROM sla_results WHERE status='success'").fetchall()
    
    scores = []
    for contract_id, uptime, rt_p1 in rows:
        gold = cuad_labels.get(contract_id, {})
        score = score_extraction(uptime, gold.get("sla_maintenance", []))
        scores.append(score["score"])
    
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nEval Results")
    print(f"Contracts evaluated: {len(scores)}")
    print(f"Average overlap score: {avg:.3f}")
    print(f"Matches (>50% overlap): {sum(1 for s in scores if s > 0.5)} / {len(scores)}")
```

### 3.2 Spot Check Script

For manual review — the most important eval you can do:

```python
# scripts/inspect_output.py
import sqlite3
import sys

def inspect(contract_id: str, db_path="output/results.db"):
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT * FROM sla_results WHERE contract_id=?", (contract_id,)
    ).fetchone()
    if not row:
        print(f"No result for {contract_id}")
        return
    cols = [d[0] for d in conn.execute("PRAGMA table_info(sla_results)").fetchall()]
    for col, val in zip(cols, row):
        if col != "source_chunks":
            print(f"{col:30}: {val}")

if __name__ == "__main__":
    inspect(sys.argv[1])
```

**Eval failure modes to watch for:**

| Symptom | Root cause | Fix |
|---|---|---|
| All fields null | SLA queries not hitting right chunks | Add more targeted queries to `SLA_QUERIES` |
| Wrong values (hallucination) | LLM inferring absent data | Tighten prompt: "return null if not explicitly stated" |
| Partial fields only | Contract uses non-standard SLA language | Add paraphrase variants to queries |
| Correct chunks, wrong parse | JSON schema mismatch | Check Pydantic field names match prompt schema |

---

## Phase 4 — Full Run + Hardening (Sunday, ~4 hours)

### 4.1 Validate Output

```python
# pipeline/validator.py
from models.schemas import SLAClause

CRITICAL_FIELDS = ["uptime_guarantee", "response_time_p1", "penalty_clause"]

def validate_result(sla: SLAClause) -> dict:
    issues = []
    
    # At least one critical field must be populated
    populated = [f for f in CRITICAL_FIELDS if getattr(sla, f)]
    if not populated:
        issues.append("No critical SLA fields extracted — possible non-SLA document")
    
    # Uptime should be a percentage
    if sla.uptime_guarantee:
        if "%" not in sla.uptime_guarantee:
            issues.append(f"uptime_guarantee '{sla.uptime_guarantee}' missing % — may be malformed")
    
    # Flag low confidence extractions for manual review
    if sla.confidence == "low":
        issues.append("Low confidence — recommend manual review")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "populated_fields": len([f for f in sla.model_fields if getattr(sla, f)])
    }
```

### 4.2 Retry Failed Contracts

```python
# scripts/retry_failed.py
import json
from scripts.run_pipeline import process_contract
from pipeline.output import save_result, init_db

def retry_failed(failed_log="output/failed_contracts.json", db_path="output/results.db"):
    with open(failed_log) as f:
        failed = json.load(f)
    
    db = init_db(db_path)
    print(f"Retrying {len(failed)} failed contracts...")
    
    for item in failed:
        result = process_contract(f"data/raw/{item['id']}.pdf")
        save_result(db, result)
        print(f"{'OK' if result.status == 'success' else 'FAIL'}: {item['id']}")
```

### 4.3 Final Output Files

After full run, you will have:

| File | Format | Use |
|---|---|---|
| `output/results.db` | SQLite | Query with any SQL client: `SELECT * WHERE uptime_guarantee IS NOT NULL` |
| `output/results.json` | JSON Lines | One JSON object per line — pipe to `jq`, load with pandas |
| `output/results_summary.csv` | CSV | Open in Excel / share with non-engineers |
| `output/failed_contracts.json` | JSON | Manual review list |

**Query examples for `results.db`:**

```sql
-- All contracts with uptime below 99.9%
SELECT contract_id, uptime_guarantee FROM sla_results
WHERE uptime_guarantee NOT LIKE '%99.9%' AND uptime_guarantee IS NOT NULL;

-- Contracts with penalty clauses
SELECT contract_id, penalty_clause FROM sla_results
WHERE penalty_clause IS NOT NULL;

-- Low confidence extractions (manual review candidates)
SELECT contract_id, confidence, notes FROM sla_results
WHERE confidence = 'low';

-- Completion rate
SELECT status, COUNT(*) FROM sla_results GROUP BY status;
```

---

## Weekend Timeline

| Time | Task | Deliverable |
|---|---|---|
| Fri evening | Setup env, download CUAD, define schema | Working Python env, 10 test contracts |
| Sat 9am | Phase 0–1: ingestion, chunking, per-doc index | Single contract → JSON output |
| Sat 1pm | Phase 1 complete: LLM extraction + output storage | `results.db` with 1 row |
| Sat 3pm | Phase 2: batch runner on 50 contracts | `results.json` with 50 rows |
| Sat 6pm | Phase 3: evals on CUAD test set | Eval score, identified failure modes |
| Sat 8pm | Prompt tuning based on eval results | Improved extraction quality |
| Sun 9am | Phase 4: full run on all contracts | All contracts processed |
| Sun 12pm | Validation + retry failed | Clean `results.db` |
| Sun 3pm | Final exports, query testing, README | Deliverable output files |
| Sun 5pm | LinkedIn post + code push | Done |

---

## Key Decisions Log

These are the architectural decisions made in this plan and why. Revisit if requirements change.

| Decision | Choice | Why |
|---|---|---|
| RAG architecture | Per-document isolation | Guarantees all contracts are processed; no cross-doc retrieval bleed |
| Vector store | ChromaDB (EphemeralClient) | No infra setup; swap to PersistentClient to cache embeddings across runs |
| Chunking | 512 tokens / 64 overlap | SLA clauses fit in one chunk; overlap catches boundary splits |
| Retrieval | Multi-query union | Single query misses field-specific language; 8 targeted queries covers all SLA fields |
| LLM | Claude Sonnet | Instruction following + JSON output reliability; cost-effective at scale |
| Output format | SQLite primary + JSON + CSV | SQLite for querying; JSON for portability; CSV for stakeholder sharing |
| Evals | CUAD gold labels + token overlap | Free ground truth; swap to BERTScore for production |

---

## Phase 5 — Future Enhancements (Post-Weekend)

If this becomes a production tool, the next investments in order of value:

1. **Semantic chunking** — split on section headers (SECTION 7, SCHEDULE A) rather than fixed token count. Reduces fragmentation of multi-clause SLA sections.
2. **Persistent vector cache** — switch to `chromadb.PersistentClient` so re-runs skip re-embedding contracts that haven't changed.
3. **BERTScore evals** — more semantically aware than token overlap. Critical if normalised SLA language diverges from gold labels.
4. **Multi-pass extraction** — first pass extracts, second pass validates/corrects. Reduces null fields on complex contracts.
5. **Contract type classifier** — route short simple NDAs through a faster/cheaper model, reserve Sonnet for dense MSAs and service agreements.
6. **REST API wrapper** — FastAPI endpoint to process new contracts on demand as they arrive.
