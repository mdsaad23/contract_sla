"""
Download CUAD contracts, extract text from PDFs, save as .txt files.

Usage:
  python scripts/download_cuad.py          # all 511 contracts
  python scripts/download_cuad.py --max 10 # first 10 only
"""

import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("Warning: HF_TOKEN not set in .env — unauthenticated HF requests may be rate-limited")


def main():
    parser = argparse.ArgumentParser(description="Download and extract CUAD contracts")
    parser.add_argument("--max", type=int, default=None, help="Max contracts to extract")
    parser.add_argument("--out", default="data/cuad_raw", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CUAD dataset from HuggingFace...")
    from datasets import load_dataset, VerificationMode
    ds = load_dataset(
        "theatticusproject/cuad",
        split="train",
        verification_mode=VerificationMode.NO_CHECKS,
    )

    total = min(len(ds), args.max) if args.max else len(ds)
    print(f"Extracting text from {total} contracts -> {out_dir}/")

    skipped = 0
    written = 0

    for i in tqdm(range(total), desc="Extracting", unit="contract"):
        item = ds[i]
        pdf = item["pdf"]

        try:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages).strip()
        except Exception as e:
            tqdm.write(f"[{i}] PDF extract failed: {e}")
            skipped += 1
            continue

        if len(text) < 200:
            skipped += 1
            continue

        # Skip the dataset's own datasheet (index 0 is CUAD documentation, not a contract)
        if i == 0 and "atticus dataset" in text[:500].lower():
            tqdm.write(f"[{i}] Skipping CUAD datasheet")
            skipped += 1
            continue

        # Skip if first page looks like a dataset/paper document
        first_page_lower = text[:500].lower()
        if any(kw in first_page_lower for kw in ["datasheet for", "arxiv:", "abstract\n"]):
            tqdm.write(f"[{i}] Skipping non-contract document")
            skipped += 1
            continue

        out_path = out_dir / f"cuad_{i:04d}.txt"
        out_path.write_text(text, encoding="utf-8")
        written += 1

    print(f"\nDone: {written} contracts saved to {out_dir}/ ({skipped} skipped)")


if __name__ == "__main__":
    main()
