# Local Model Benchmark — SLA Extraction

Contracts: 100 | identical cached retrieval context per contract

Seed `42` | embedder `BAAI/bge-base-en-v1.5` | target num_ctx 32768

| Model | Provider | num_ctx | Score | >=0.7 | Success | Partial | Fail | Trunc | Fields | Median s |
|---|---|---|---|---|---|---|---|---|---|---|
| `hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:IQ4_XS` | ollama | 32768 | **0.822** | 80/100 | 91% | 9 | 0 | 0 | 4.9 | 17.8 |
| `deepseek-v4-pro` | api | - | **0.792** | 80/100 | 88% | 12 | 0 | 0 | 4.6 | 7.7 |
| `qwen3:14b-q4_K_M` | ollama | 32768 | **0.778** | 82/100 | 88% | 12 | 0 | 0 | 4.7 | 107.7 |
| `llama3.1:8b-instruct-q8_0` | ollama | 32768 | **0.750** | 66/100 | 92% | 7 | 1 | 0 | 4.5 | 8.7 |

Baseline: `deepseek-v4-pro` at 0.792, 7.7s median

| Model | Score vs baseline | Speed vs baseline |
|---|---|---|
| `hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:IQ4_XS` | +0.030 (+4%) | 0.43x |
| `qwen3:14b-q4_K_M` | -0.014 (-2%) | 0.07x |
| `llama3.1:8b-instruct-q8_0` | -0.042 (-5%) | 0.89x |

## Per-field average score

| Field | `hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF` | `deepseek-v4-pro` | `qwen3` | `llama3.1` |
|---|---|---|---|---|
| dispute_resolution | 0.63 | 0.86 | 0.69 | 0.66 |
| governing_law | 0.87 | 0.91 | 0.90 | 0.86 |
| liability_cap | 0.48 | 0.75 | 0.73 | 0.45 |
| penalty_currency | 1.00 | 1.00 | 1.00 | 1.00 |
| penalty_data_breach | 0.80 | 0.80 | nan | nan |
| penalty_has_monetary | 1.00 | 1.00 | 1.00 | 1.00 |
| penalty_late_delivery | 0.16 | 0.37 | 0.25 | 0.53 |
| penalty_late_payment | 0.88 | 0.99 | 0.97 | 0.78 |
| penalty_max_amount | 1.00 | 1.00 | 1.00 | 1.00 |
| penalty_termination_fee | 0.80 | 0.90 | 0.84 | 0.80 |
| penalty_uptime_breach | 1.00 | 0.80 | 0.70 | nan |
| renewal_terms | 0.91 | 0.94 | 0.89 | 0.71 |
| response_time_sla | 0.44 | 0.48 | 0.35 | 0.25 |
| service_credit_cap | 1.00 | 1.00 | 1.00 | 1.00 |
| sla_breach_threshold | 0.00 | 0.70 | 0.52 | 0.00 |
| sla_measurement_period | 0.33 | 0.56 | 0.44 | 0.00 |
| termination_clause | 0.89 | 0.92 | 0.90 | 0.63 |
| uptime_guarantee | 0.00 | 0.20 | 0.00 | 0.00 |