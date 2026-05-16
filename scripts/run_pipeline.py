"""
Main pipeline entry point.

Usage:
  Single contract:    python scripts/run_pipeline.py data/raw/test.txt
  Batch (Phase 2):    python scripts/run_pipeline.py data/cuad_raw --max 10
  Full run (Phase 4): python scripts/run_pipeline.py data/cuad_raw --resume
"""

import sys
import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ingestion import load_contract, clean_text
from pipeline.chunker import chunk_contract
from pipeline.embedder import build_per_doc_index
from pipeline.retriever import retrieve_sla_chunks
from pipeline.extractor import extract_sla


def run_single(file_path: str) -> dict:
    path = Path(file_path)
    contract_id = path.stem

    print(f"[1/5] Loading contract: {path.name}")
    raw_text = load_contract(file_path)
    text = clean_text(raw_text)
    print(f"      {len(text):,} chars")

    print(f"[2/5] Chunking...")
    chunks = chunk_contract(text)
    print(f"      {len(chunks)} chunks")

    print(f"[3/5] Building vector index...")
    index = build_per_doc_index(contract_id, chunks)

    print(f"[4/5] Retrieving SLA-relevant chunks...")
    context = retrieve_sla_chunks(index)
    print(f"      {len(context):,} chars of context")

    print(f"[5/5] Extracting SLA clauses with DeepSeek...")
    result = extract_sla(contract_id, file_path, context)

    return result.model_dump()


def main():
    parser = argparse.ArgumentParser(description="SLA Extraction Pipeline")
    parser.add_argument("input", help="Path to a contract file or dataset directory")
    parser.add_argument("--max", type=int, default=None, help="Max contracts to process (batch mode)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed contracts")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        result = run_single(str(input_path))
        print("\n" + "=" * 60)
        print("EXTRACTION RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2))

        sla = result.get("sla", {})
        populated = [k for k, v in sla.items() if v is not None]
        print(f"\nStatus: {result['status']}")
        total_fields = len(sla)
        print(f"Fields populated ({len(populated)}/{total_fields}): {', '.join(populated) if populated else 'none'}")
        print(f"Tokens used: {result.get('tokens_used', 0):,}")
        if result.get("error"):
            print(f"Error: {result['error']}")

    elif input_path.is_dir():
        from scripts._batch import run_batch
        run_batch(str(input_path), max_contracts=args.max, resume=args.resume)

    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
