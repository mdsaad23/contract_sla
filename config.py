import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, free

BATCH_SIZE = 10
PAUSE_BETWEEN_BATCHES = 2
MAX_TOKENS_PER_CALL = 1500

CHROMA_DB_PATH = "./data/chroma_db"
SQLITE_DB_PATH = "./output/results.db"
OUTPUT_JSON_PATH = "./output/results.json"
OUTPUT_CSV_PATH = "./output/results_summary.csv"
FAILED_LOG_PATH = "./output/failed_contracts.json"
