# Changelog

## [1.0.0] - 2025-05-17

### Initial Release

#### Phase 0 — Environment Setup
- Directory structure created: `data/`, `pipeline/`, `models/`, `output/`, `evals/`, `scripts/`
- Dependencies installed and locked in `requirements.txt`
- Configuration system via `config.py` and `.env`

#### Phase 1 — Core Pipeline MVP
- Built complete RAG pipeline: ingestion → chunking → embedding → retrieval → extraction
- Pydantic schemas for SLA extraction (8 fields initial)
- Multi-query retrieval strategy with 14 targeted SLA queries
- DeepSeek V3.2 integration via OpenAI-compatible API
- Single contract MVP tested successfully

#### Phase 2 — Batch Processing
- Batch pipeline with resume support
- Three output formats: SQLite, JSON Lines, CSV
- CUAD dataset download script with PDF text extraction
- Progress tracking and error logging

#### Phase 3 — Evaluation Framework
- Three-signal evaluation: substring overlap, keyword coverage, format validation
- Initial eval score: 0.28 (improved to 0.866 after fixes)
- Contract inspection CLI for debugging

#### Schema Upgrade (before Phase 4)
- Expanded from 8 to 19 fields
- Separated penalty types: uptime, late delivery, termination fee, late payment, data breach
- Added structured monetary penalty fields: `penalty_has_monetary`, `penalty_max_amount`, `penalty_currency`
- Updated retrieval with 6 additional penalty-focused queries

#### Pipeline Performance Optimisation (mid-Phase 4)
- **6-10× end-to-end speedup** through three changes:
  1. Module-level embedder singleton (eliminated double model loading)
  2. PyTorch → ONNX Runtime migration (2-3× faster)
  3. DirectML GPU acceleration (1.6× additional improvement on compatible GPUs)
- Per-contract time: 31-50s → **3-8s** (with GPU)

#### Phase 4 — Full Production Run
- **510 CUAD contracts processed end-to-end**
- **89.4% successful extractions** (456/510 with ≥2 fields)
- **Total cost: $1.72** ($0.0034 per contract)
- Average eval score: **0.790** (408/510 contracts ≥ 0.7)
- Field coverage analysis completed
- Cost comparison vs Claude Sonnet (33×), GPT-4o (55×)

#### Key Outcomes
- **128 contracts** identified with real monetary (cash) penalties
- **256 contracts** with service credits only
- **127 contracts** with no penalty clauses
- Top fields extracted: termination clause (83.6%), governing law (81.4%), penalty presence (75.1%)

### Deliverables
- Production-grade SQLite database with queryable monetary penalties
- JSON Lines output for streaming/pandas integration
- CSV export for spreadsheet analysis
- Comprehensive evaluation report
- Detailed extraction statistics
