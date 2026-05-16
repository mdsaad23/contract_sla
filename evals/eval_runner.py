"""
Eval runner for SLA extraction results.

Three complementary signals:
1. Substring overlap  — extracted text must appear in the cleaned+whitespace-
                        normalised source (handles PDF line-break artefacts)
2. Keyword coverage   — tight, field-specific keywords flag genuine misses
3. Format validation  — broad patterns covering US and international contracts

Run:
  python -m evals.eval_runner
  python -m evals.eval_runner --limit 5
"""

import re
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SQLITE_DB_PATH
from pipeline.ingestion import clean_text


# ── Field definitions ──────────────────────────────────────────────────────────

SLA_FIELDS = [
    "uptime_guarantee", "response_time_sla", "sla_breach_threshold",
    "sla_measurement_period",
    "penalty_uptime_breach", "penalty_late_delivery", "penalty_termination_fee",
    "penalty_late_payment", "penalty_data_breach",
    "penalty_has_monetary", "penalty_max_amount", "penalty_currency",
    "service_credit_cap",
    "renewal_terms", "termination_clause", "liability_cap",
    "governing_law", "dispute_resolution",
]

# Tight keywords — must strongly imply the field exists in the document.
# Deliberately specific to avoid false "missed" flags.
FIELD_KEYWORDS = {
    "uptime_guarantee":       ["uptime sla", "availability sla", "99.9", "99.5", "service level agreement"],
    "response_time_sla":      ["response time sla", "respond within", "priority 1", "p1 issue", "critical incident response"],
    "sla_breach_threshold":   ["falls below", "breach threshold", "sla is breached", "availability drops"],
    "sla_measurement_period": ["measured monthly", "calculated quarterly", "rolling 12", "measurement period"],
    "penalty_uptime_breach":  ["service credit", "uptime credit", "availability credit", "sla credit"],
    "penalty_late_delivery":  ["liquidated damage", "late delivery penalty", "delay penalty", "per day penalty"],
    "penalty_termination_fee":["early termination fee", "termination penalty", "cancellation fee"],
    "penalty_late_payment":   ["late payment interest", "overdue interest", "past due interest", "1.5% per month"],
    "penalty_data_breach":    ["data breach penalty", "security breach fine", "breach notification fine"],
    "penalty_has_monetary":   [],
    "penalty_max_amount":     [],
    "penalty_currency":       [],
    "service_credit_cap":     ["credit cap", "maximum credit", "credits shall not exceed", "credit limit"],
    "renewal_terms":          ["auto-renew", "automatically renew", "renewal notice", "notice of non-renewal"],
    "termination_clause":     ["terminat", "right to terminate", "notice of termination"],
    "liability_cap":          ["aggregate liability", "limitation of liability", "in no event", "shall not exceed"],
    "governing_law":          ["govern", "applicable law", "jurisdiction", "laws of"],
    "dispute_resolution":     ["arbitrat", "mediat", "dispute resolution", "legal action", "proceedings"],
}

# Broad patterns — cover US and international phrasing.
# Word-form numbers (one, two, twelve) handled for renewal/penalty fields.
_WORD_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirty|sixty|ninety)"
_CURRENCY = r"(?:\$|£|€|USD|GBP|EUR|CAD|AUD)"

FIELD_PATTERNS = {
    "uptime_guarantee":       r"\d{2,3}(?:\.\d+)?\s*%",
    "response_time_sla":      _WORD_NUM + r"\s*(?:hour|business day|day|minute|working day)",
    "sla_breach_threshold":   r"(?:below|less than|under|drops?\s+below)\s*\d|breach|falls?\s+below",
    "sla_measurement_period": r"(?:month|quarter|annual|year|rolling|calendar)",
    "penalty_uptime_breach":  r"(?:\d+(?:\.\d+)?\s*%|credit|" + _CURRENCY + r"|\bfee\b)",
    "penalty_late_delivery":  r"(?:" + _CURRENCY + r"|\d+\s*(?:per day|per week|%)|liquidated|damage)",
    "penalty_termination_fee":r"(?:" + _CURRENCY + r"|" + _WORD_NUM + r"\s*month|fee|equal to|remaining)",
    "penalty_late_payment":   r"(?:\d+(?:\.\d+)?\s*%|interest|per month|per annum|overdue|30\s*days)",
    "penalty_data_breach":    r"(?:" + _CURRENCY + r"|\d+|fine|indemnif|reimburse|notify)",
    "penalty_has_monetary":   r"",   # boolean — skip
    "penalty_max_amount":     r"(?:" + _CURRENCY + r"|\d[\d,]*(?:\.\d+)?(?:\s*(?:million|thousand|M|K))?|" + _WORD_NUM + r"\s*month)",
    "penalty_currency":       r"(?:USD|GBP|EUR|CAD|AUD|\$|£|€)",
    "service_credit_cap":     r"(?:\d+(?:\.\d+)?\s*%|not\s+to\s+exceed|maximum|cap)",
    "renewal_terms":          r"(?:" + _WORD_NUM + r"\s*(?:day|month|year)|auto.?renew|notice|perpetuity|term)",
    "termination_clause":     r"(?:terminat|cancel|notice|days?\s+(?:prior|written|advance))",
    "liability_cap":          r"(?:aggregate|in no event|shall not exceed|" + _CURRENCY + r"|\d[\d,]*|\d+\s*month|limitation|no liability)",
    "governing_law":          r"(?:laws?\s+of|govern|jurisdiction|applicable law|courts?\s+of)",
    "dispute_resolution":     r"(?:arbitrat|mediat|dispute|adr|conciliat|legal action|courts?\s+of|venue|proceedings)",
}

NON_CONTRACT_MARKERS = [
    "datasheet for contract understanding",
    "arxiv:",
    "abstract\nwe present",
    "i.motivation\na. who created",
]


# ── Source loading ──────────────────────────────────────────────────────────────

def _load_source(file_path: str):
    """Returns (clean_source, ws_source) — two normalisation levels."""
    p = Path(file_path)
    if not p.exists():
        return "", ""
    raw      = p.read_text(encoding="utf-8", errors="replace")
    cleaned  = clean_text(raw).lower()
    ws_norm  = re.sub(r"\s+", " ", cleaned)   # collapse ALL whitespace to single space
    return cleaned, ws_norm


def _is_non_contract(source_clean: str) -> bool:
    return any(m in source_clean[:1000] for m in NON_CONTRACT_MARKERS)


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_extraction(contract_id: str, file_path: str, row: dict) -> dict:
    src_clean, src_ws = _load_source(file_path)
    is_non_contract   = _is_non_contract(src_clean)

    scores = {}

    for field in SLA_FIELDS:
        value = row.get(field)

        # Boolean field: present = 1.0
        if field == "penalty_has_monetary":
            scores[field] = 1.0 if value is not None else None
            continue

        # Derived summary fields with no reliable verbatim/pattern check
        if field in ("penalty_max_amount", "penalty_currency"):
            scores[field] = 1.0 if value else None
            continue

        if value:
            snippet    = str(value)[:120].lower().strip()
            snippet_ws = re.sub(r"\s+", " ", snippet)

            # Two-level verbatim check: exact clean match OR whitespace-normalised match
            in_source = (snippet in src_clean) or (snippet_ws in src_ws) if src_clean else False

            pattern  = FIELD_PATTERNS.get(field, "")
            patt_ok  = bool(re.search(pattern, str(value), re.IGNORECASE)) if pattern else True

            if in_source and patt_ok:
                score = 1.0
            elif in_source:
                score = 0.8   # verbatim confirmed, pattern unusual
            elif patt_ok:
                score = 0.6   # looks right, verbatim check still failing
            else:
                score = 0.4   # extracted something, unverifiable
        else:
            keywords    = FIELD_KEYWORDS.get(field, [])
            keyword_hit = any(kw in src_clean for kw in keywords) if (src_clean and keywords) else False
            score       = 0.0 if keyword_hit else None

        scores[field] = score

    scoreable = [s for s in scores.values() if s is not None]
    overall   = sum(scoreable) / len(scoreable) if scoreable else 0.0
    missed    = sum(1 for s in scores.values() if s == 0.0)

    return {
        "contract_id":   contract_id,
        "is_non_contract": is_non_contract,
        "scores":        scores,
        "overall":       overall,
        "missed":        missed,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def load_results(limit: Optional[int] = None) -> list:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM sla_results ORDER BY contract_id"
    if limit:
        q += f" LIMIT {limit}"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_eval(limit: Optional[int] = None):
    rows = load_results(limit)
    if not rows:
        print("No results in DB. Run batch pipeline first.")
        return

    print(f"Evaluating {len(rows)} contracts...\n")

    all_evals   = []
    skipped_ids = []

    for row in rows:
        ev = score_extraction(row["contract_id"], row["file_path"], row)
        if ev["is_non_contract"]:
            skipped_ids.append(row["contract_id"])
        else:
            all_evals.append(ev)

    if skipped_ids:
        print(f"Excluded (non-contract documents): {', '.join(skipped_ids)}\n")

    if not all_evals:
        print("No contract results to evaluate.")
        return

    field_scores = {f: [] for f in SLA_FIELDS}
    field_missed = {f: 0   for f in SLA_FIELDS}
    field_absent = {f: 0   for f in SLA_FIELDS}

    for ev in all_evals:
        for field, score in ev["scores"].items():
            if score is None:
                field_absent[field] += 1
            elif score == 0.0:
                field_missed[field] += 1
            else:
                field_scores[field].append(score)

    n = len(all_evals)
    print(f"{'Field':<26} {'AvgScore':>8} {'Extracted':>9} {'Missed':>7} {'Absent':>7}")
    print("-" * 62)
    for field in SLA_FIELDS:
        sc  = field_scores[field]
        avg = sum(sc) / len(sc) if sc else 0.0
        print(f"{field:<26} {avg:>8.2f} {len(sc):>9} {field_missed[field]:>7} {field_absent[field]:>7}")

    overall_scores = [ev["overall"] for ev in all_evals]
    avg_overall    = sum(overall_scores) / len(overall_scores)
    gte_08         = sum(1 for s in overall_scores if s >= 0.8)
    gte_07         = sum(1 for s in overall_scores if s >= 0.7)
    gte_05         = sum(1 for s in overall_scores if s >= 0.5)

    print()
    print(f"Contracts evaluated   : {n}")
    print(f"Average overall score : {avg_overall:.3f}")
    print(f"Score >= 0.8          : {gte_08}/{n}")
    print(f"Score >= 0.7          : {gte_07}/{n}")
    print(f"Score >= 0.5          : {gte_05}/{n}")

    sorted_evals = sorted(all_evals, key=lambda e: e["overall"])
    print(f"\nLowest-scoring contracts:")
    for ev in sorted_evals[:3]:
        mf = [f for f, s in ev["scores"].items() if s == 0.0]
        print(f"  {ev['contract_id']}: {ev['overall']:.2f} | missed: {mf}")

    print(f"\nHighest-scoring contracts:")
    for ev in sorted_evals[-3:]:
        hf = [f for f, s in ev["scores"].items() if s and s >= 0.8]
        print(f"  {ev['contract_id']}: {ev['overall']:.2f} | high-conf: {hf}")

    report = {
        "n_contracts":            n,
        "excluded_non_contracts": skipped_ids,
        "avg_overall":            avg_overall,
        "score_gte_0_8":          gte_08,
        "score_gte_0_7":          gte_07,
        "score_gte_0_5":          gte_05,
        "per_field": {
            f: {
                "avg_score": sum(field_scores[f]) / len(field_scores[f]) if field_scores[f] else 0.0,
                "extracted": len(field_scores[f]),
                "missed":    field_missed[f],
                "absent":    field_absent[f],
            }
            for f in SLA_FIELDS
        },
        "per_contract": all_evals,
    }
    Path("output/eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved to output/eval_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_eval(limit=args.limit)
