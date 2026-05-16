import json
from pathlib import Path

report = json.loads(Path("output/eval_report.json").read_text())
contracts = report["per_contract"]

real = [c for c in contracts if c["contract_id"] != "cuad_0000"]
avg_real = sum(c["overall"] for c in real) / len(real)
high_conf = sum(1 for c in real if c["overall"] >= 0.5)

print(f"Without cuad_0000 (datasheet):")
print(f"  Avg overall score : {avg_real:.3f}")
print(f"  Contracts >= 0.5  : {high_conf}/{len(real)}")
print()
for c in sorted(real, key=lambda x: x["overall"], reverse=True):
    populated = sum(1 for s in c["scores"].values() if s and s > 0)
    missed    = sum(1 for s in c["scores"].values() if s == 0.0)
    print(f"  {c['contract_id']}: overall={c['overall']:.2f}  extracted={populated}  missed={missed}")
