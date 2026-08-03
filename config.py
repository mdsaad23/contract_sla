import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, free

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
MAX_TOKENS_PER_CALL = 1500

CHROMA_DB_PATH = "./data/chroma_db"
SQLITE_DB_PATH = "./output/results.db"
OUTPUT_JSON_PATH = "./output/results.json"
OUTPUT_CSV_PATH = "./output/results_summary.csv"
FAILED_LOG_PATH = "./output/failed_contracts.json"
