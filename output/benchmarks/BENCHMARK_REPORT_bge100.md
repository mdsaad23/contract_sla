# Local Model Benchmark — SLA Extraction

Contracts: 100 | identical cached retrieval context per contract

Seed `42` | embedder `BAAI/bge-base-en-v1.5` | target num_ctx 32768

| Model | Provider | num_ctx | Score | >=0.7 | Success | Partial | Fail | Trunc | Fields | Median s |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-pro` | api | - | **0.792** | 80/100 | 88% | 12 | 0 | 0 | 4.6 | 7.7 |

Baseline: `deepseek-v4-pro` at 0.792, 7.7s median

| Model | Score vs baseline | Speed vs baseline |
|---|---|---|

## Per-field average score

| Field | `deepseek-v4-pro` |
|---|---|
| dispute_resolution | 0.86 |
| governing_law | 0.91 |
| liability_cap | 0.75 |
| penalty_currency | 1.00 |
| penalty_data_breach | 0.80 |
| penalty_has_monetary | 1.00 |
| penalty_late_delivery | 0.37 |
| penalty_late_payment | 0.99 |
| penalty_max_amount | 1.00 |
| penalty_termination_fee | 0.90 |
| penalty_uptime_breach | 0.80 |
| renewal_terms | 0.94 |
| response_time_sla | 0.48 |
| service_credit_cap | 1.00 |
| sla_breach_threshold | 0.70 |
| sla_measurement_period | 0.56 |
| termination_clause | 0.92 |
| uptime_guarantee | 0.20 |