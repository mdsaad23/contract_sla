import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# deepseek-chat/-reasoner aliases were retired 2026-07-24. V4 Pro thinks by
# default; pipeline/llm.py disables it explicitly (thinking eats max_tokens).
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# One key, every hosted model, and usage.cost returns the billed dollars per call.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Single source of truth for the embedder — pipeline/embedder.py imports this.
# Switching it invalidates existing Chroma collections (different dim), so pair
# it with a fresh CHROMA_DB_PATH.
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Local generation via Ollama. num_ctx must exceed our prompt (~900 tokens of
# schema + 4-6K tokens of retrieved context); Ollama's 4096 default truncates it.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# 24576 covers the largest CUAD prompt (~23K tokens). Ollama truncates silently
# past num_ctx rather than erroring, and _call_ollama halves this on OOM, so a
# too-high default degrades gracefully while a too-low one corrupts results.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "24576"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))

# Default generation backend: "api" (DeepSeek) or "ollama" (local)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "api")
LLM_MODEL = os.getenv("LLM_MODEL")  # None -> provider default

BATCH_SIZE = 10
PAUSE_BETWEEN_BATCHES = 2
# 19 fields of verbatim quotes. A few contracts carry multi-thousand-character
# arbitration and termination clauses; at 1500 they came back as a JSON object
# cut off mid-string and scored 0.000, measuring this budget rather than the
# model. Worst observed need is ~3.5K. Raising the ceiling is free — you pay for
# tokens generated — and no non-thinking local model has exceeded 600 here.
MAX_TOKENS_PER_CALL = 4000

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
SQLITE_DB_PATH = "./output/results.db"
OUTPUT_JSON_PATH = "./output/results.json"
OUTPUT_CSV_PATH = "./output/results_summary.csv"
FAILED_LOG_PATH = "./output/failed_contracts.json"
