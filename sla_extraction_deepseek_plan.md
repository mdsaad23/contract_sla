# SLA Extraction RAG — DeepSeek Edition

**Goal:** Extract SLA clauses from 2,000+ contracts using DeepSeek V3.2 (10x cheaper than Claude)
**Stack:** Python, LlamaIndex / LangChain, ChromaDB, DeepSeek API, SQLite + JSON
**Cost Estimate:** ~$28 total (vs $450 with Claude)
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
│   ├── extractor.py             # DeepSeek extraction + structured output
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
├── .env                         # DEEPSEEK_API_KEY=sk-...
├── config.py                    # All tuneable parameters
├── requirements.txt
└── README.md
```

---

## .env Setup

**Your .env file should look like:**

```dotenv
DEEPSEEK_API_KEY=sk-...  (copy from platform.deepseek.com)
```

**Verify it works:**

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓ Key loaded' if os.getenv('DEEPSEEK_API_KEY') else '✗ Key not found')"
```

---

## Output Schema (Pydantic)

Same as before — no changes needed:

```python
# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SLAClause(BaseModel):
    contract_id: str
    contract_name: Optional[str]
    uptime_guarantee: Optional[str]
    response_time_p1: Optional[str]
    response_time_p2: Optional[str]
    resolution_time_p1: Optional[str]
    resolution_time_p2: Optional[str]
    penalty_clause: Optional[str]
    exclusions: Optional[str]
    measurement_window: Optional[str]
    reporting_frequency: Optional[str]
    escalation_path: Optional[str]
    termination_for_breach: Optional[str]
    source_chunks: list[str] = Field(default_factory=list)
    confidence: Optional[str]
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str]

class ExtractionResult(BaseModel):
    contract_id: str
    status: str
    sla: Optional[SLAClause]
    error: Optional[str]
```

---

## Phase 0 — Setup (Friday Evening, ~1 hour)

### 0.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install \
  llama-index \
  llama-index-vector-stores-chroma \
  chromadb \
  openai \
  pydantic \
  pypdf \
  python-dotenv \
  tqdm \
  pandas \
  pytest
```

**Note:** Using `openai` package, not `anthropic`. DeepSeek is OpenAI-compatible.

### 0.2 .env File

Create `.env` in project root:

```dotenv
DEEPSEEK_API_KEY=sk-...
```

Get key from: https://platform.deepseek.com

### 0.3 Get Contract Data

```bash
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('cuad', split='train')
ds.save_to_disk('./data/cuad_raw')
print(f'Loaded {len(ds)} contracts')
"
```

---

## Phase 1 — MVP (Saturday Morning, ~4 hours)

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

```python
# pipeline/chunker.py
from llama_index.core.node_parser import SentenceSplitter

def chunk_contract(text: str, contract_id: str) -> list[dict]:
    """
    Chunk at 512 tokens with 64 token overlap.
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

### 1.3 Per-Document Embedding + Retrieval

```python
# pipeline/embedder.py + retriever.py
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def build_per_doc_index(chunks: list[dict], contract_id: str) -> chromadb.Collection:
    """
    Create an isolated ChromaDB collection per contract.
    Uses OpenAI embeddings (compatible with DeepSeek).
    """
    client = chromadb.EphemeralClient()
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
    Union the results — ensures all SLA fields are retrieved.
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

### 1.4 LLM Extraction with DeepSeek

**CRITICAL: This is the only major code change from Claude version**

```python
# pipeline/extractor.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from models.schemas import SLAClause
from models.prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT

load_dotenv()

# Initialize DeepSeek client (OpenAI API compatible)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def extract_sla(contract_id: str, chunks: list[str]) -> SLAClause:
    """Extract SLA information using DeepSeek V3.2."""
    schema_str = json.dumps(SLAClause.model_json_schema(), indent=2)
    chunk_text = "\n\n---\n\n".join(chunks)
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    contract_id=contract_id,
                    chunks=chunk_text,
                    schema=schema_str
                )
            }
        ]
    )
    
    raw = response.choices[0].message.content
    
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

### 1.5 Config (Updated for DeepSeek)

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
EMBEDDING_MODEL = "text-embedding-3-small"

# Pipeline Configuration
BATCH_SIZE = 10
PAUSE_BETWEEN_BATCHES = 2
MAX_CHUNKS_PER_CONTRACT = 25
MAX_TOKENS_PER_CALL = 1500
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Cost Estimates
# Embeddings: 2000 contracts × 100 chunks × 512 tokens = 100M tokens ≈ $14
# LLM extraction: 2000 × 6000 input + 1500 output = $14 (input) + $14 (output)
# Total estimate: ~$28 (vs $450 with Claude)

print(f"Using DeepSeek API")
print(f"API Key loaded: {'✓' if DEEPSEEK_API_KEY else '✗'}")
```

### 1.6 MVP Runner

```python
# scripts/run_pipeline.py (MVP — single contract)
from pipeline.ingestion import load_contract, clean_text
from pipeline.chunker import chunk_contract
from pipeline.embedder import build_per_doc_index
from pipeline.retriever import retrieve_sla_chunks
from pipeline.extractor import extract_sla
from models.schemas import ExtractionResult
import json

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
    result = process_contract("data/raw/test.txt")
    print(json.dumps(result.model_dump(), indent=2, default=str))
```

**Saturday morning milestone:** One contract → valid JSON with populated SLA fields.

---

## Phase 2 — Scale to All Contracts (Saturday Afternoon, ~3 hours)

### 2.1 Batch Runner with Resume Support

```python
# scripts/run_pipeline.py (full batch)
from pathlib import Path
from tqdm import tqdm
import json
import sqlite3
from pipeline.output import save_result, init_db

def get_processed_ids(db) -> set:
    cursor = db.execute("SELECT contract_id FROM sla_results")
    return {row[0] for row in cursor.fetchall()}

def run_batch(input_dir: str, max_workers: int = 1):  # DeepSeek API: sequential is fine
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

### 2.2 Output Storage (Unchanged)

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
            source_chunks TEXT,
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
    
    # CSV summary
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

---

## Phase 3 — Evals (Saturday Evening, ~2 hours)

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
    """Token overlap score between predicted and gold."""
    if not predicted or not gold_spans:
        return {"match": False, "score": 0.0}
    
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(" ".join(gold_spans).lower().split())
    
    if not gold_tokens:
        return {"match": True, "score": 1.0}
    
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
    print(f"\nEval Results (DeepSeek)")
    print(f"Contracts evaluated: {len(scores)}")
    print(f"Average overlap score: {avg:.3f}")
    print(f"Matches (>50% overlap): {sum(1 for s in scores if s > 0.5)} / {len(scores)}")
```

---

## Phase 4 — Full Production Run (Sunday, ~4 hours)

Same as original plan — just uses DeepSeek instead of Claude.

**Full run command:**
```bash
python scripts/run_pipeline.py data/cuad_raw --max 500 --resume
```

**Estimated time:** 30–60 mins for 500 contracts
**Estimated cost:** ~$28 total

---

## Requirements.txt (Updated)

```
llama-index>=0.9.0
llama-index-vector-stores-chroma>=0.2.0
chromadb>=0.4.0
openai>=1.0.0          # ← Changed from 'anthropic'
pydantic>=2.0.0
pypdf>=3.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0
pandas>=1.5.0
pytest>=7.0.0
```

---

## Prompts (No Changes Needed)

DeepSeek V3.2 handles the same prompts as Claude. No modifications to `models/prompts.py` required.

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
- Use exact quoted values where possible
- For time values, normalise to the unit stated in the contract
- If multiple SLA tiers exist, capture the most critical (P1 / highest severity)
- Set confidence to "low" if you are reconstructing from implied context"""
```

---

## Cost Breakdown

For 2,000 CUAD contracts:

| Component | Tokens | Cost per 1M | Total |
|---|---|---|---|
| **Embeddings** | 100M | $0.05 | $5 |
| **LLM input** | 12M | $0.14 | $1.68 |
| **LLM output** | 3M | $0.28 | $0.84 |
| **Total** | 115M | — | **~$7.50** |

**vs Claude:** ~$150–200 for same work
**Savings:** 95%

---

## Quick Integration Notes

1. **API Key:** DeepSeek requires signup at https://platform.deepseek.com (free tier available)
2. **OpenAI SDK:** DeepSeek uses OpenAI's API format — seamless integration
3. **Base URL:** Must set `base_url="https://api.deepseek.com"` in OpenAI client
4. **Model name:** Use `"deepseek-chat"` (not v3.2 — that's internal)
5. **No embedding changes:** Still uses OpenAI's embedding model (text-embedding-3-small)

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `AuthenticationError` | Missing/wrong API key | Check .env file, copy key from platform.deepseek.com |
| `RateLimitError` | Hitting rate limits | Add `PAUSE_BETWEEN_BATCHES = 2` in config.py |
| `Invalid request` | Wrong model name | Use `"deepseek-chat"` not `"deepseek-v3.2"` |
| `Connection timeout` | Base URL wrong | Ensure `https://api.deepseek.com` is set |

---

## Summary of Changes from Claude Plan

✅ **What stays the same:**
- Per-document RAG architecture
- Chunking strategy (512 tokens / 64 overlap)
- Schema & output formats (SQLite, JSON, CSV)
- Evals & validation logic
- Project structure

🔄 **What changed:**
- `anthropic` → `openai` SDK
- `claude-sonnet-4-20250514` → `deepseek-chat`
- Client initialization (set base_url)
- `.env` variable name: `ANTHROPIC_API_KEY` → `DEEPSEEK_API_KEY`
- `message.content[0].text` → `response.choices[0].message.content`
- Cost: $150–200 → $7–10

✨ **New advantage:**
- 20x cheaper
- Chinese model (you wanted this)
- Same instruction-following quality for extraction

