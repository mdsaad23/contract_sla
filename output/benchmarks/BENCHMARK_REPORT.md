# Local Model Benchmark — SLA Extraction

Contracts: 12 | identical cached retrieval context per contract

| Model | Provider | num_ctx | Score | >=0.7 | Success | Partial | Fail | Trunc | Fields | Median s |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-chat` | api | - | **0.819** | 10/12 | 100% | 0 | 0 | 0 | 4.5 | 5.6 |
| `qwen3:14b-q4_K_M` | ollama | 24576 | **0.807** | 9/12 | 100% | 0 | 0 | 0 | 4.2 | 38.4 |
| `llama3.1:8b-instruct-q8_0` | ollama | 24576 | **0.636** | 6/12 | 92% | 0 | 1 | 0 | 3.6 | 26.1 |
| `phi4:14b-q4_K_M` | ollama | 16384 | **0.626** | 7/12 | 83% | 0 | 2 | 0 | 3.2 | 14.4 |
| `gemma4:26b-a4b-it-q4_K_M` | ollama | 24576 | **0.618** | 8/12 | 75% | 0 | 3 | 0 | 3.0 | 100.1 |
| `phi4:14b-q4_K_M|json` | ollama | 16384 | **0.606** | 7/12 | 83% | 0 | 2 | 0 | 3.0 | 38.2 |
| `mistral:7b-instruct-v0.3-q4_K_M` | ollama | 24576 | **0.579** | 5/12 | 75% | 3 | 0 | 0 | 3.6 | 10.2 |
| `mistral:7b-instruct-v0.3-q4_K_M|json` | ollama | 24576 | **0.579** | 5/12 | 75% | 3 | 0 | 0 | 3.6 | 10.0 |
| `llama3.1:8b-instruct-q4_K_M` | ollama | 24576 | **0.576** | 4/12 | 100% | 0 | 0 | 0 | 3.2 | 8.5 |
| `llama3.1:8b-instruct-q4_K_M|json` | ollama | 24576 | **0.576** | 4/12 | 100% | 0 | 0 | 0 | 3.2 | 8.6 |
| `qwen2.5-coder:7b-instruct-q4_K_M` | ollama | 24576 | **0.530** | 4/12 | 92% | 0 | 1 | 0 | 2.8 | 9.7 |
| `qwen2.5-coder:7b-instruct-q4_K_M|json` | ollama | 24576 | **0.513** | 3/12 | 92% | 1 | 0 | 0 | 2.8 | 8.3 |
| `llama3.2:3b-instruct-q4_K_M` | ollama | 24576 | **0.487** | 1/12 | 100% | 0 | 0 | 0 | 5.8 | 4.4 |
| `granite4:7b-a1b-h` | ollama | 24576 | **0.468** | 2/12 | 75% | 3 | 0 | 0 | 4.1 | 5.2 |
| `llama3.2:3b-instruct-q8_0` | ollama | 24576 | **0.410** | 0/12 | 75% | 2 | 1 | 0 | 4.3 | 4.7 |
| `deepseek-r1:8b-llama-distill-q4_K_M` | ollama | 24576 | **0.380** | 1/12 | 58% | 5 | 0 | 0 | 2.6 | 21.7 |
| `gemma4:12b-it-q4_K_M` | ollama | 24576 | **0.113** | 1/12 | 17% | 0 | 10 | 0 | 0.5 | 83.6 |

Baseline: `deepseek-chat` at 0.819, 5.6s median

| Model | Score vs baseline | Speed vs baseline |
|---|---|---|
| `qwen3:14b-q4_K_M` | -0.012 (-1%) | 0.15x |
| `llama3.1:8b-instruct-q8_0` | -0.184 (-22%) | 0.22x |
| `phi4:14b-q4_K_M` | -0.193 (-24%) | 0.39x |
| `gemma4:26b-a4b-it-q4_K_M` | -0.201 (-25%) | 0.06x |
| `phi4:14b-q4_K_M|json` | -0.214 (-26%) | 0.15x |
| `mistral:7b-instruct-v0.3-q4_K_M` | -0.240 (-29%) | 0.55x |
| `mistral:7b-instruct-v0.3-q4_K_M|json` | -0.240 (-29%) | 0.56x |
| `llama3.1:8b-instruct-q4_K_M` | -0.243 (-30%) | 0.66x |
| `llama3.1:8b-instruct-q4_K_M|json` | -0.243 (-30%) | 0.65x |
| `qwen2.5-coder:7b-instruct-q4_K_M` | -0.289 (-35%) | 0.58x |
| `qwen2.5-coder:7b-instruct-q4_K_M|json` | -0.306 (-37%) | 0.68x |
| `llama3.2:3b-instruct-q4_K_M` | -0.332 (-40%) | 1.29x |
| `granite4:7b-a1b-h` | -0.351 (-43%) | 1.08x |
| `llama3.2:3b-instruct-q8_0` | -0.409 (-50%) | 1.20x |
| `deepseek-r1:8b-llama-distill-q4_K_M` | -0.439 (-54%) | 0.26x |
| `gemma4:12b-it-q4_K_M` | -0.706 (-86%) | 0.07x |

## Per-field average score

| Field | `deepseek-chat` | `qwen3` | `llama3.1` | `phi4` | `gemma4` | `phi4` | `mistral` | `mistral` | `llama3.1` | `llama3.1` | `qwen2.5-coder` | `qwen2.5-coder` | `llama3.2` | `granite4` | `llama3.2` | `deepseek-r1` | `gemma4` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dispute_resolution | 0.63 | 0.50 | 0.30 | 0.25 | 0.55 | 0.25 | 0.30 | 0.30 | 0.37 | 0.37 | 0.28 | 0.37 | 0.05 | 0.23 | 0.05 | 0.25 | 0.07 |
| governing_law | 1.00 | 1.00 | 0.90 | 0.83 | 0.75 | 0.83 | 0.88 | 0.88 | 1.00 | 1.00 | 0.92 | 0.92 | 0.18 | 0.32 | 0.25 | 0.33 | 0.17 |
| liability_cap | 0.57 | 0.80 | 0.40 | 0.43 | 0.40 | 0.43 | 0.26 | 0.40 | 0.40 | 0.40 | 0.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 |
| penalty_currency | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | nan | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | nan |
| penalty_data_breach | nan | 0.80 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 1.00 | nan | nan |
| penalty_has_monetary | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| penalty_late_delivery | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.72 | nan | 0.80 | 1.00 | nan |
| penalty_late_payment | 1.00 | 1.00 | 1.00 | nan | 1.00 | nan | nan | nan | nan | nan | 1.00 | 1.00 | 0.67 | nan | 0.60 | 0.60 | nan |
| penalty_max_amount | 1.00 | 1.00 | 1.00 | 1.00 | nan | nan | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | nan |
| penalty_termination_fee | nan | nan | 1.00 | nan | nan | nan | 0.90 | 0.90 | nan | nan | 0.60 | nan | 0.60 | nan | nan | nan | nan |
| penalty_uptime_breach | nan | nan | nan | nan | nan | nan | 0.60 | 0.60 | nan | nan | nan | nan | 0.67 | 0.60 | 0.60 | nan | nan |
| renewal_terms | 0.73 | 0.87 | 0.50 | 0.47 | 0.40 | 0.47 | 0.47 | 0.47 | 0.27 | 0.27 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.27 |
| response_time_sla | 1.00 | 1.00 | 0.00 | 0.00 | 0.60 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.40 | 0.00 | 0.00 | 0.00 |
| service_credit_cap | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.60 | 0.60 | 0.60 | 0.50 | nan |
| sla_breach_threshold | 0.80 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.40 | 0.60 | 0.40 | 0.60 | nan |
| sla_measurement_period | nan | nan | nan | nan | nan | nan | 0.60 | 0.60 | nan | nan | nan | nan | 0.60 | 0.60 | 0.60 | nan | nan |
| termination_clause | 0.83 | 0.83 | 0.48 | 0.70 | 0.67 | 0.63 | 0.42 | 0.33 | 0.08 | 0.08 | 0.33 | 0.17 | 0.05 | 0.05 | 0.00 | 0.13 | 0.08 |
| uptime_guarantee | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.60 | nan | nan | nan |