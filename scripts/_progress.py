from datetime import datetime, timedelta
import sqlite3

c = sqlite3.connect("output/results.db")
rows = c.execute("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM sla_results").fetchone()
status = c.execute("SELECT status, COUNT(*) FROM sla_results GROUP BY status").fetchall()
total_done = rows[0]
first = datetime.strptime(rows[1], "%Y-%m-%d %H:%M:%S")
last  = datetime.strptime(rows[2], "%Y-%m-%d %H:%M:%S")

elapsed_sec = (last - first).total_seconds()
rate        = total_done / elapsed_sec
remaining   = 510 - total_done
eta_sec     = remaining / rate
eta_time    = datetime.now() + timedelta(seconds=eta_sec)

print(f"Done      : {total_done}/510  ({total_done/510*100:.1f}%)")
print(f"Status    : {dict(status)}")
print(f"Elapsed   : {elapsed_sec/60:.1f} min")
print(f"Rate      : {rate*60:.1f} contracts/min  ({1/rate:.0f}s each)")
print(f"Remaining : {remaining} contracts")
print(f"ETA       : {eta_sec/60:.0f} min  (~{eta_time.strftime('%H:%M')} local time)")
