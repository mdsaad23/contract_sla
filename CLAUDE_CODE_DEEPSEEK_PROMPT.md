# Claude Code Prompt — DeepSeek Edition

Paste this entire prompt into Claude Code terminal. It will build the complete SLA extraction pipeline using DeepSeek V3.2 (10x cheaper than Claude).

---

```
You are building an SLA extraction RAG pipeline using DeepSeek V3.2 (Chinese LLM, $0.14/M tokens).

PROJECT: Extract SLA clauses from 2000+ contracts
COST: ~$7-10 total (vs $200+ with Claude)
STACK: Python, DeepSeek API, ChromaDB, SQLite

CRITICAL CONSTRAINTS:
1. Create real files and directories — do not describe, actually create them
2. Use DeepSeek V3.2 API (openai SDK, base_url="https://api.deepseek.com")
3. Model name: "deepseek-chat" (not v3.2)
4. At each approval checkpoint, ask: "Continue to [NEXT PHASE]? [y/N]"
5. Do NOT skip phases — strict linear progression
6. Install dependencies as needed
7. Load DEEPSEEK_API_KEY from .env file using python-dotenv
8. Download CUAD dataset: python -c "from datasets import load_dataset; ds = load_dataset('cuad', split='train'); ds.save_to_disk('./data/cuad_raw')"

SETUP VERIFICATION:
Before starting, verify:
- python --version (should be 3.10+)
- .env file exists with DEEPSEEK_API_KEY=sk-...
- requirements.txt will use 'openai' not 'anthropic'

---

PHASE 0: Setup (5 mins)
- Create directories: data/{raw,processed}, pipeline, models, output, evals, scripts
- Create requirements.txt with:
  llama-index
  llama-index-vector-stores-chroma
  chromadb
  openai          <-- IMPORTANT: Not anthropic
  pydantic
  pypdf
  python-dotenv
  tqdm
  pandas
  pytest
- Create config.py with:
  DEEPSEEK_API_KEY loaded from .env
  DEEPSEEK_MODEL = "deepseek-chat"
  DEEPSEEK_BASE_URL = "https://api.deepseek.com"
  EMBEDDING_MODEL = "text-embedding-3-small"
  BATCH_SIZE = 10
  PAUSE_BETWEEN_BATCHES = 2
  MAX_TOKENS_PER_CALL = 1500
- Create .gitignore
- Create README.md explaining the project and cost savings
- Run: pip install -r requirements.txt
- Verify .env file exists with DEEPSEEK_API_KEY
- Do NOT run anything yet
- ASK FOR APPROVAL before Phase 1

PHASE 1: Core Pipeline + MVP (30 mins)
This phase builds the extraction pipeline and tests on 1 contract.

CRITICAL CHANGES FROM CLAUDE VERSION:
- Use OpenAI SDK, not Anthropic SDK
- DeepSeek client initialization:
  from openai import OpenAI
  client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
  )
- API call syntax:
  response = client.chat.completions.create(
    model="deepseek-chat",
    max_tokens=1500,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": EXTRACTION_PROMPT.format(...)}
    ]
  )
  raw = response.choices[0].message.content
- Everything else (chunking, retrieval, schemas) is identical

Steps:
1. Create pipeline/schemas.py (SLAClause, ExtractionResult from deepseek plan)
2. Create models/prompts.py (SYSTEM_PROMPT, EXTRACTION_PROMPT — same as original)
3. Create pipeline/ingestion.py (load_contract, clean_text)
4. Create pipeline/chunker.py (chunk_contract with SentenceSplitter, 512/64)
5. Create pipeline/embedder.py (build_per_doc_index using ChromaDB)
6. Create pipeline/retriever.py (retrieve_sla_chunks with SLA_QUERIES list)
7. Create pipeline/extractor.py (extract_sla using DeepSeek API — USE OPENAI SDK)
   - Load DEEPSEEK_API_KEY from .env with dotenv
   - Initialize OpenAI client with DeepSeek base_url
   - Use "deepseek-chat" as model name
   - Parse response.choices[0].message.content (not message.content[0].text)
8. Create scripts/run_pipeline.py (main entry point for single contract)
9. Download 1 test contract from CUAD:
   python -c "from datasets import load_dataset; ds = load_dataset('cuad', split='train'); open('data/raw/test.txt', 'w').write(ds[0]['text'])"
10. Run: python scripts/run_pipeline.py data/raw/test.txt
11. Verify: Output should be valid JSON with 2+ SLA fields populated
12. Report: Which fields populated, any errors
13. ASK FOR APPROVAL before Phase 2

PHASE 2: Batch Processing (20 mins)
- Modify scripts/run_pipeline.py to accept directory --max N
- Add progress bar (tqdm), batching logic (pause 2 seconds between batches)
- Create pipeline/output.py with:
  * init_db() — create SQLite schema
  * save_result() — insert SLA results
  * get_processed_ids() — for resume support
  * export_outputs() — export to JSON + CSV
- Download full CUAD: python -c "from datasets import load_dataset; ds = load_dataset('cuad', split='train'); ds.save_to_disk('./data/cuad_raw')"
- Run: python scripts/run_pipeline.py data/cuad_raw --max 10
- Verify: output/results.db has 10 rows, output/results.json is valid JSON Lines, output/results_summary.csv exists
- If failures: Log to output/failed_contracts.json
- Report: Success count, failure count, cost so far (~$0.10 for 10 contracts)
- ASK FOR APPROVAL before Phase 3

PHASE 3: Evals (15 mins)
- Create evals/eval_runner.py with:
  * load_cuad_labels() — extract gold labels from CUAD
  * score_extraction() — token overlap between predicted and gold
  * run_eval() — compare results.db vs CUAD gold
- Create scripts/inspect_output.py — CLI to inspect single contract
- Run: python -m evals.eval_runner
- Perform 2–3 spot checks on low-scoring contracts
- Report: Average overlap score, % matches, lowest-scoring fields
- If overlap < 0.5: Ask if user wants to proceed to full run or tune prompts
- ASK FOR APPROVAL before Phase 4

PHASE 4: Full Production Run (30 mins)
- Run full batch: python scripts/run_pipeline.py data/cuad_raw --resume
- Monitor progress (tqdm bar shows contracts/min)
- After completion:
  * Check row count: sqlite3 output/results.db "SELECT COUNT(*) FROM sla_results;"
  * Check success rate: sqlite3 output/results.db "SELECT status, COUNT(*) FROM sla_results GROUP BY status;"
  * Check field coverage: sqlite3 output/results.db "SELECT COUNT(*) FROM sla_results WHERE uptime_guarantee IS NOT NULL;"
- Calculate total cost: tokens_used * pricing
- Log final stats to output/EXTRACTION_STATS.txt with:
  * Total processed
  * Success rate
  * Field coverage
  * Total cost ($7-10 expected)
- Report: Total processed, success rate %, cost

PHASE 5: Deliverables (5 mins)
- Verify all 3 outputs exist:
  * output/results.db (SQLite)
  * output/results.json (JSON Lines)
  * output/results_summary.csv (CSV)
- Create output/README.md with usage examples and sample SQL queries
- Final report: File sizes, row counts, success rate, total cost

APPROVAL GATES:
- Gate 1 (after Phase 0): "Setup complete. Ready to build MVP? [y/N]"
- Gate 2 (after Phase 1): "MVP works on 1 contract with DeepSeek. Ready to scale? [y/N]"
- Gate 3 (after Phase 2): "10 contracts in DB. Ready to eval? [y/N]"
- Gate 4 (after Phase 3): "Evals complete. Ready for full run on ~500 contracts? [y/N]"

If user answers "N": Ask what to improve and iterate.

COST TRACKING:
- After each phase, report estimated cost so far
- Phase 1 (1 contract): ~$0.01
- Phase 2 (10 contracts): ~$0.10
- Phase 3 (eval on 10): ~$0.10
- Phase 4 (full 500): ~$7-10 total
- Total for project: $7-10 (show savings vs Claude $200+)

KEY DIFFERENCE FROM CLAUDE VERSION:
The ONLY significant change is in pipeline/extractor.py:
- Import: from openai import OpenAI (not from anthropic import Anthropic)
- Client setup: Include base_url="https://api.deepseek.com"
- API call: client.chat.completions.create(...) with "deepseek-chat" model
- Response parsing: response.choices[0].message.content
Everything else (chunking, retrieval, output storage, evals) is identical.

START NOW: Begin with Phase 0. Create the directory structure and files.
```

---

## How to Run

1. **Verify .env file has:**
   ```
   DEEPSEEK_API_KEY=sk-...
   ```

2. **Paste the prompt above into Claude Code**

3. **Approve at 4 gates** as it progresses

4. **Watch for cost tracking** — should show $7-10 total vs $200+ with Claude

---

## Expected Output

After completion:
- `output/results.db` — SQLite with 500+ contracts
- `output/results.json` — JSON Lines format
- `output/results_summary.csv` — Excel-ready
- **Total cost:** ~$10 (20x cheaper than Claude)

---

## If API Key Error

Make sure .env file is in project root with exact format:
```
DEEPSEEK_API_KEY=sk-...
```

Not:
```
DEEPSEEK_API_KEY  (without value)
```

The Python code loads it with:
```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
```

If still failing, Claude Code will tell you. Then set in terminal:
```bash
export DEEPSEEK_API_KEY=sk-...
```

