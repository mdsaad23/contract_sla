import re
import sys
import json
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.ingestion import clean_text

report = json.loads(Path("output/eval_report.json").read_text())

buckets = {1.0: 0, 0.8: 0, 0.6: 0, 0.4: 0, 0.0: 0, "null": 0}
reasons_06 = []
reasons_04 = []
missed_fields = {}

conn = sqlite3.connect("output/results.db")
conn.row_factory = sqlite3.Row

for ev in report["per_contract"]:
    cid = ev["contract_id"]
    row = conn.execute("SELECT * FROM sla_results WHERE contract_id=?", (cid,)).fetchone()
    if not row:
        continue
    row = dict(row)
    src_raw = ""
    fp = Path(row["file_path"])
    if fp.exists():
        src_raw = fp.read_text(encoding="utf-8", errors="replace")
    src_clean = clean_text(src_raw).lower()
    src_ws    = re.sub(r"\s+", " ", src_clean)

    for field, score in ev["scores"].items():
        if score is None:
            buckets["null"] += 1
        else:
            buckets[score] = buckets.get(score, 0) + 1
            val = row.get(field)
            if val and score == 0.6:
                snippet    = str(val)[:80].lower().strip()
                snippet_ws = re.sub(r"\s+", " ", snippet)
                ws_match   = snippet_ws in src_ws
                reasons_06.append(
                    f"{cid}.{field}: ws_match={ws_match} | {snippet[:50]}"
                )
            if val and score == 0.4:
                reasons_04.append(f"{cid}.{field}: {str(val)[:60]}")
            if score == 0.0:
                missed_fields[field] = missed_fields.get(field, 0) + 1

conn.close()

print("Score distribution across all field*contract pairs:")
for k in [1.0, 0.8, 0.6, 0.4, 0.0, "null"]:
    label = f"  {str(k):>5}"
    print(f"{label}: {buckets.get(k, 0)}")

print(f"\nSample 0.6 scores (ws_match=True => whitespace normalisation fix converts to 1.0):")
for r in reasons_06[:10]:
    print(f"  {r}")

print(f"\nSample 0.4 scores (extracted but unverifiable):")
for r in reasons_04[:8]:
    print(f"  {r}")

print(f"\nMissed field counts (keyword hit but null extraction):")
for f, n in sorted(missed_fields.items(), key=lambda x: -x[1]):
    print(f"  {f}: {n}")
