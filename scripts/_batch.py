"""Batch processing logic (Phase 2). Imported by run_pipeline.py for directory inputs."""

import time
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ingestion import load_contract, clean_text
from pipeline.chunker import chunk_contract
from pipeline.embedder import build_per_doc_index
from pipeline.retriever import retrieve_sla_chunks
from pipeline.extractor import extract_sla
from pipeline.output import init_db, save_result, get_processed_ids, export_outputs
from config import BATCH_SIZE, PAUSE_BETWEEN_BATCHES, FAILED_LOG_PATH


def _collect_contracts(dataset_path: str, max_contracts=None):
    path = Path(dataset_path)
    contracts = []

    # HuggingFace Arrow dataset directory
    arrow_files = list(path.glob("**/*.arrow"))
    if arrow_files:
        try:
            from datasets import load_from_disk
            ds = load_from_disk(dataset_path)
            for i, item in enumerate(ds):
                if max_contracts and i >= max_contracts:
                    break
                contracts.append({"id": f"cuad_{i}", "text": item.get("text", ""), "source": "cuad"})
            return contracts
        except Exception as e:
            print(f"Warning: could not load Arrow dataset: {e}")

    # Plain text/pdf files
    for ext in ("*.txt", "*.pdf"):
        for f in sorted(path.glob(ext)):
            contracts.append({"id": f.stem, "path": str(f), "source": "file"})
            if max_contracts and len(contracts) >= max_contracts:
                break

    return contracts[:max_contracts] if max_contracts else contracts


def run_batch(dataset_path: str, max_contracts=None, resume=False):
    init_db()
    processed_ids = get_processed_ids() if resume else set()

    contracts = _collect_contracts(dataset_path, max_contracts)
    if not contracts:
        print("No contracts found.")
        return

    if resume and processed_ids:
        contracts = [c for c in contracts if c["id"] not in processed_ids]
        print(f"Resuming: {len(contracts)} contracts remaining")

    print(f"Processing {len(contracts)} contracts in batches of {BATCH_SIZE}...")

    failed = []
    success_count = 0
    total_tokens = 0

    for i, contract in enumerate(tqdm(contracts, desc="Contracts", unit="contract")):
        try:
            if "text" in contract:
                text = clean_text(contract["text"])
                file_path = f"cuad_dataset[{contract['id']}]"
            else:
                raw = load_contract(contract["path"])
                text = clean_text(raw)
                file_path = contract["path"]

            if len(text) < 100:
                failed.append({"id": contract["id"], "error": "text too short"})
                continue

            chunks = chunk_contract(text)
            index = build_per_doc_index(contract["id"], chunks)
            context = retrieve_sla_chunks(index)
            result = extract_sla(contract["id"], file_path, context)

            save_result(result)
            total_tokens += result.tokens_used

            if result.status != "failed":
                success_count += 1
            else:
                failed.append({"id": contract["id"], "error": result.error})

        except Exception as e:
            failed.append({"id": contract["id"], "error": str(e)})

        # Pause between batches
        if (i + 1) % BATCH_SIZE == 0 and (i + 1) < len(contracts):
            tqdm.write(f"Batch {(i+1)//BATCH_SIZE} complete — pausing {PAUSE_BETWEEN_BATCHES}s")
            time.sleep(PAUSE_BETWEEN_BATCHES)

    # Save failed log
    if failed:
        Path(FAILED_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_LOG_PATH, "w") as f:
            json.dump(failed, f, indent=2)

    export_outputs()

    cost_estimate = total_tokens / 1_000_000 * 0.14
    print(f"\n{'='*60}")
    print(f"Batch complete")
    print(f"  Success: {success_count}/{len(contracts)}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Tokens:  {total_tokens:,}")
    print(f"  Est. cost: ${cost_estimate:.4f}")
    print(f"  Outputs: output/results.db | results.json | results_summary.csv")
