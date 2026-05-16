import sqlite3
import json
import csv
from pathlib import Path
from pipeline.schemas import ExtractionResult
from config import SQLITE_DB_PATH, OUTPUT_JSON_PATH, OUTPUT_CSV_PATH

SLA_FIELDS = [
    # Performance SLAs
    "uptime_guarantee", "response_time_sla", "sla_breach_threshold", "sla_measurement_period",
    # Penalties by type
    "penalty_uptime_breach", "penalty_late_delivery", "penalty_termination_fee",
    "penalty_late_payment", "penalty_data_breach",
    # Monetary summary
    "penalty_has_monetary", "penalty_max_amount", "penalty_currency", "service_credit_cap",
    # Contract mechanics
    "renewal_terms", "termination_clause", "liability_cap", "governing_law", "dispute_resolution",
]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sla_results (
    contract_id             TEXT PRIMARY KEY,
    file_path               TEXT,
    status                  TEXT,
    tokens_used             INTEGER,

    uptime_guarantee        TEXT,
    response_time_sla       TEXT,
    sla_breach_threshold    TEXT,
    sla_measurement_period  TEXT,

    penalty_uptime_breach   TEXT,
    penalty_late_delivery   TEXT,
    penalty_termination_fee TEXT,
    penalty_late_payment    TEXT,
    penalty_data_breach     TEXT,

    penalty_has_monetary    INTEGER,
    penalty_max_amount      TEXT,
    penalty_currency        TEXT,
    service_credit_cap      TEXT,

    renewal_terms           TEXT,
    termination_clause      TEXT,
    liability_cap           TEXT,
    governing_law           TEXT,
    dispute_resolution      TEXT,

    raw_response            TEXT,
    error                   TEXT,
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _conn():
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(SQLITE_DB_PATH)


def init_db():
    with _conn() as conn:
        conn.execute(CREATE_TABLE)
        conn.commit()


def reset_db():
    with _conn() as conn:
        conn.execute("DROP TABLE IF EXISTS sla_results")
        conn.execute(CREATE_TABLE)
        conn.commit()


def save_result(result: ExtractionResult):
    sla = result.sla
    with _conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sla_results
               (contract_id, file_path, status, tokens_used,
                uptime_guarantee, response_time_sla, sla_breach_threshold, sla_measurement_period,
                penalty_uptime_breach, penalty_late_delivery, penalty_termination_fee,
                penalty_late_payment, penalty_data_breach,
                penalty_has_monetary, penalty_max_amount, penalty_currency, service_credit_cap,
                renewal_terms, termination_clause, liability_cap, governing_law, dispute_resolution,
                raw_response, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                result.contract_id, result.file_path, result.status, result.tokens_used,
                sla.uptime_guarantee, sla.response_time_sla,
                sla.sla_breach_threshold, sla.sla_measurement_period,
                sla.penalty_uptime_breach, sla.penalty_late_delivery,
                sla.penalty_termination_fee, sla.penalty_late_payment, sla.penalty_data_breach,
                int(sla.penalty_has_monetary) if sla.penalty_has_monetary is not None else None,
                sla.penalty_max_amount, sla.penalty_currency, sla.service_credit_cap,
                sla.renewal_terms, sla.termination_clause, sla.liability_cap,
                sla.governing_law, sla.dispute_resolution,
                result.raw_response, result.error,
            ),
        )
        conn.commit()


def get_processed_ids() -> set:
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT contract_id FROM sla_results").fetchall()
            return {r[0] for r in rows}
    except Exception:
        return set()


def export_outputs():
    Path(OUTPUT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)

    with _conn() as conn:
        rows = conn.execute("SELECT * FROM sla_results").fetchall()
        cols = [d[0] for d in conn.execute("SELECT * FROM sla_results LIMIT 0").description]

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            record = dict(zip(cols, row))
            f.write(json.dumps(record) + "\n")

    summary_cols = [c for c in cols if c not in ("raw_response", "error")]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        for row in rows:
            record = {k: v for k, v in zip(cols, row) if k in summary_cols}
            writer.writerow(record)
