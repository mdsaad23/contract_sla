"""Compute the rate based on the most recent N contracts only."""
import sys
from datetime import datetime, timedelta
import sqlite3

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

c = sqlite3.connect("output/results.db")
rows = c.execute(
    f"SELECT contract_id, created_at FROM sla_results ORDER BY created_at DESC LIMIT {N}"
).fetchall()

if len(rows) < 2:
    print("Not enough data")
else:
    times = [datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S") for r in rows]
    latest, earliest = times[0], times[-1]
    span_sec = (latest - earliest).total_seconds()
    rate_per_sec = (len(rows) - 1) / span_sec
    sec_each = 1 / rate_per_sec
    print(f"Most recent {len(rows)} contracts:")
    print(f"  Time span : {span_sec/60:.1f} min")
    print(f"  Rate      : {rate_per_sec*60:.1f} contracts/min  ({sec_each:.1f}s each)")

    total_done = c.execute("SELECT COUNT(*) FROM sla_results").fetchone()[0]
    remaining = 510 - total_done
    eta_sec = remaining / rate_per_sec
    eta_time = datetime.now() + timedelta(seconds=eta_sec)
    print(f"  Remaining : {remaining} contracts")
    print(f"  ETA       : {eta_sec/60:.0f} min  (~{eta_time.strftime('%H:%M')} local time)")

c.close()
