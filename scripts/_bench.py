"""Benchmark end-to-end pipeline speed on 3 unprocessed contracts."""

import sys
import time
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pipeline.ingestion import load_contract, clean_text
from pipeline.chunker import chunk_contract
from pipeline.embedder import build_per_doc_index, get_active_provider
from pipeline.retriever import retrieve_sla_chunks
from pipeline.extractor import extract_sla

# Find 3 contracts not yet in the DB
conn = sqlite3.connect("output/results.db")
done = {r[0] for r in conn.execute("SELECT contract_id FROM sla_results").fetchall()}
conn.close()

all_files = sorted(Path("data/cuad_raw").glob("*.txt"))
candidates = [f for f in all_files if f.stem not in done][:3]
print(f"Benchmarking on: {[c.name for c in candidates]}")

# Warm up the embedder (load model)
print("Warming up embedder...")
t0 = time.perf_counter()
from pipeline.embedder import get_embed_model
get_embed_model()
print(f"Model load: {time.perf_counter()-t0:.2f}s | Provider: {get_active_provider()}\n")

for f in candidates:
    contract_id = f.stem
    print(f"--- {contract_id} ---")
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    text = clean_text(load_contract(str(f)))
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_contract(text)
    t_chunk = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = build_per_doc_index(contract_id, chunks)
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    context = retrieve_sla_chunks(index)
    t_retr = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = extract_sla(contract_id, str(f), context)
    t_extr = time.perf_counter() - t0

    t_all = time.perf_counter() - t_total
    populated = sum(1 for v in result.sla.model_dump().values() if v is not None)
    print(f"  load+clean:  {t_load:6.2f}s  ({len(text):,} chars)")
    print(f"  chunk:       {t_chunk:6.2f}s  ({len(chunks)} chunks)")
    print(f"  embed+index: {t_index:6.2f}s")
    print(f"  retrieve:    {t_retr:6.2f}s  ({len(context):,} chars context)")
    print(f"  extract:     {t_extr:6.2f}s  ({result.tokens_used} tokens, status={result.status}, {populated}/18 fields)")
    print(f"  TOTAL:       {t_all:6.2f}s")
    print()
