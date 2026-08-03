# Local Model Benchmark — SLA Extraction

Contracts: 100 | identical cached retrieval context per contract

Seed `42` | embedder `sentence-transformers/all-MiniLM-L6-v2` | target num_ctx 32768

| Model | Provider | num_ctx | Score | >=0.7 | Success | Partial | Fail | Trunc | Fields | Median s |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-pro` | api | - | **0.775** | 73/100 | 88% | 12 | 0 | 0 | 4.5 | 7.6 |

Baseline: `deepseek-v4-pro` at 0.775, 7.6s median

| Model | Score vs baseline | Speed vs baseline |
|---|---|---|

## Per-field average score

| Field | `deepseek-v4-pro` |
|---|---|
| dispute_resolution | 0.82 |
| governing_law | 0.85 |
| liability_cap | 0.78 |
| penalty_currency | 1.00 |
| penalty_has_monetary | 1.00 |
| penalty_late_delivery | 0.37 |
| penalty_late_payment | 0.98 |
| penalty_max_amount | 1.00 |
| penalty_termination_fee | 0.90 |
| penalty_uptime_breach | 0.80 |
| renewal_terms | 0.92 |
| response_time_sla | 0.49 |
| service_credit_cap | 1.00 |
| sla_breach_threshold | 0.48 |
| sla_measurement_period | 0.50 |
| termination_clause | 0.91 |
| uptime_guarantee | 0.20 |