"""Compute and persist final Phase 4 statistics."""
from pathlib import Path
import sqlite3
import json

SLA_FIELDS = [
    "uptime_guarantee", "response_time_sla", "sla_breach_threshold", "sla_measurement_period",
    "penalty_uptime_breach", "penalty_late_delivery", "penalty_termination_fee",
    "penalty_late_payment", "penalty_data_breach",
    "penalty_has_monetary", "penalty_max_amount", "penalty_currency", "service_credit_cap",
    "renewal_terms", "termination_clause", "liability_cap", "governing_law", "dispute_resolution",
]

c = sqlite3.connect("output/results.db")

total = c.execute("SELECT COUNT(*) FROM sla_results").fetchone()[0]
status = dict(c.execute("SELECT status, COUNT(*) FROM sla_results GROUP BY status").fetchall())
tokens = c.execute("SELECT SUM(tokens_used) FROM sla_results").fetchone()[0]
first  = c.execute("SELECT MIN(created_at) FROM sla_results").fetchone()[0]
last   = c.execute("SELECT MAX(created_at) FROM sla_results").fetchone()[0]

# Field coverage
coverage = {}
for f in SLA_FIELDS:
    n = c.execute(f"SELECT COUNT({f}) FROM sla_results WHERE {f} IS NOT NULL").fetchone()[0]
    coverage[f] = n

# Penalty insights
penalty_monetary = c.execute("SELECT COUNT(*) FROM sla_results WHERE penalty_has_monetary = 1").fetchone()[0]
penalty_credits  = c.execute("SELECT COUNT(*) FROM sla_results WHERE penalty_has_monetary = 0").fetchone()[0]
penalty_none     = c.execute("SELECT COUNT(*) FROM sla_results WHERE penalty_has_monetary IS NULL").fetchone()[0]

# Cost calculation — DeepSeek pricing $0.27/$1.10 per M tokens (input/output)
# We don't have input/output split per call, use blended estimate ~ $0.40/M
cost_blended  = tokens * 0.40 / 1_000_000
cost_max      = tokens * 1.10 / 1_000_000   # worst-case all-output pricing
cost_min      = tokens * 0.27 / 1_000_000   # best-case all-input

# File sizes
db_size   = Path("output/results.db").stat().st_size
json_size = Path("output/results.json").stat().st_size if Path("output/results.json").exists() else 0
csv_size  = Path("output/results_summary.csv").stat().st_size if Path("output/results_summary.csv").exists() else 0
failed    = json.loads(Path("output/failed_contracts.json").read_text()) if Path("output/failed_contracts.json").exists() else []

c.close()

# Format report
report = f"""================================================================
SLA EXTRACTION PIPELINE — PHASE 4 FINAL STATS
================================================================

DATASET
  Source         : CUAD (theatticusproject/cuad)
  Available text : 510 contracts (1 PDF was the datasheet, skipped)

PROCESSING
  Total processed: {total}
  Status         : {status}
  Failures logged: {len(failed)}
  Time span      : {first} -> {last}

TOKEN USAGE & COST
  Total tokens   : {tokens:,}
  Avg per contract: {tokens / max(total,1):,.0f}
  Cost (min)     : ${cost_min:.4f}
  Cost (blended) : ${cost_blended:.4f}
  Cost (max)     : ${cost_max:.4f}

PENALTY EXTRACTION INSIGHTS
  Monetary penalty (cash): {penalty_monetary}  ({penalty_monetary/total*100:.1f}%)
  Service credits only   : {penalty_credits}   ({penalty_credits/total*100:.1f}%)
  No penalty clauses     : {penalty_none}      ({penalty_none/total*100:.1f}%)

FIELD COVERAGE
"""
for f in SLA_FIELDS:
    pct = coverage[f] / total * 100
    bar = "#" * int(pct / 2.5)
    report += f"  {f:<25} {coverage[f]:>4}/{total} ({pct:>5.1f}%)  {bar}\n"

report += f"""
OUTPUT FILES
  output/results.db          : {db_size:>10,} bytes
  output/results.json        : {json_size:>10,} bytes
  output/results_summary.csv : {csv_size:>10,} bytes
================================================================
"""

print(report)
Path("output/EXTRACTION_STATS.txt").write_text(report, encoding="utf-8")
print("Saved to output/EXTRACTION_STATS.txt")
