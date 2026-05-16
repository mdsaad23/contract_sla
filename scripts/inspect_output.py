"""
CLI to inspect a single contract's extraction result.

Usage:
  python scripts/inspect_output.py cuad_0001
  python scripts/inspect_output.py cuad_0001 --show-context
  python scripts/inspect_output.py --list
"""

import sys
import json
import sqlite3
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SQLITE_DB_PATH

SLA_FIELDS = [
    "uptime_guarantee", "response_time_sla", "penalty_clause", "renewal_terms",
    "termination_clause", "liability_cap", "governing_law", "dispute_resolution",
]


def list_contracts():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    rows = conn.execute(
        "SELECT contract_id, status, tokens_used FROM sla_results ORDER BY contract_id"
    ).fetchall()
    conn.close()

    populated = lambda cid: _count_populated(cid)
    print(f"{'Contract ID':<20} {'Status':<10} {'Tokens':>8} {'Fields':>7}")
    print("-" * 50)
    for cid, status, tokens in rows:
        n = _count_populated(cid)
        print(f"{cid:<20} {status:<10} {tokens or 0:>8,} {n:>7}/8")


def _count_populated(contract_id: str) -> int:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    row = conn.execute(
        f"SELECT {', '.join(SLA_FIELDS)} FROM sla_results WHERE contract_id=?",
        (contract_id,)
    ).fetchone()
    conn.close()
    if not row:
        return 0
    return sum(1 for v in row if v is not None)


def inspect(contract_id: str, show_context: bool = False):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM sla_results WHERE contract_id=?", (contract_id,)
    ).fetchone()
    conn.close()

    if not row:
        print(f"No result found for contract_id: {contract_id}")
        sys.exit(1)

    row = dict(row)
    print(f"\n{'='*60}")
    print(f"Contract: {row['contract_id']}")
    print(f"File:     {row['file_path']}")
    print(f"Status:   {row['status']}")
    print(f"Tokens:   {row.get('tokens_used', 0):,}")
    print(f"{'='*60}\n")

    for field in SLA_FIELDS:
        val = row.get(field)
        marker = "Y" if val else "-"
        print(f"  [{marker}] {field}")
        if val:
            print(f"       {val[:200]}{'...' if len(val) > 200 else ''}")
        print()

    if row.get("error"):
        print(f"ERROR: {row['error']}")

    if show_context and row.get("raw_response"):
        print("\nRaw DeepSeek response:")
        print("-" * 40)
        try:
            parsed = json.loads(row["raw_response"])
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(row["raw_response"])


def main():
    parser = argparse.ArgumentParser(description="Inspect SLA extraction output")
    parser.add_argument("contract_id", nargs="?", help="Contract ID to inspect")
    parser.add_argument("--list", action="store_true", help="List all contracts")
    parser.add_argument("--show-context", action="store_true", help="Show raw DeepSeek response")
    args = parser.parse_args()

    if args.list or not args.contract_id:
        list_contracts()
    else:
        inspect(args.contract_id, show_context=args.show_context)


if __name__ == "__main__":
    main()
