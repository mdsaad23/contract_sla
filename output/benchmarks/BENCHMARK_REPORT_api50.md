# Model Benchmark — SLA Extraction

Contracts: 50 | identical cached retrieval context per contract

Seed `42` | embedder `BAAI/bge-base-en-v1.5` | target num_ctx 32768

| Model | Provider | num_ctx | Score | >=0.7 | Success | Partial | Fail | Trunc | Fields | Median s |
|---|---|---|---|---|---|---|---|---|---|---|
| `anthropic/claude-opus-5.5` | openrouter | - | **0.868** | 44/50 | 92% | 4 | 0 | 0 | 6.0 | 10.2 |
| `deepseek/deepseek-v4-pro-0813` | openrouter | - | **0.833** | 42/50 | 92% | 4 | 0 | 0 | 4.8 | 5.8 |
| `openai/gpt-6-sol` | openrouter | - | **0.817** | 40/50 | 90% | 5 | 0 | 0 | 4.5 | 6.5 |
| `qwen/qwen3.8-max-0902` | openrouter | - | **0.799** | 40/50 | 90% | 4 | 1 | 0 | 4.7 | 29.4 |
| `openai/gpt-6-luna` | openrouter | - | **0.791** | 37/50 | 90% | 5 | 0 | 0 | 4.3 | 6.7 |
| `openai/gpt-6-astra:batch` | openrouter | - | **0.000** | 0/5 | 0% | 0 | 5 | 0 | 0.0 | 0.7 |

## Cost and tokens per contract

Captured per call: $6.44
OpenRouter ledger since run start: $6.28 (unattributed: $-0.15)

$ / contract counts every attempt, failed ones included.

## Summary

Efficiency is score per 1K tokens per contract (input + output).

| Model | Contracts | Failed | Score | Input tok | Output tok | Efficiency | Tokens / task | $ / task |
|---|---|---|---|---|---|---|---|---|
| `anthropic/claude-opus-5.5` | 50 | 0 | 0.868 | 649,770 | 50,396 | 0.062 | 14,003 | $0.0721 |
| `deepseek/deepseek-v4-pro-0813` | 50 | 0 | 0.833 | 406,021 | 30,395 | 0.095 | 8,728 | $0.0032 |
| `openai/gpt-6-sol` | 50 | 0 | 0.817 | 400,158 | 34,143 | 0.094 | 8,686 | $0.0268 |
| `qwen/qwen3.8-max-0902` | 50 | 1 | 0.799 | 414,737 | 71,497 | 0.082 | 9,725 | $0.0252 |
| `openai/gpt-6-luna` | 50 | 0 | 0.791 | 400,158 | 39,299 | 0.090 | 8,789 | $0.0014 |

| Model | Score | $ / contract | $ total | Input tok | Output tok | of which reasoning |
|---|---|---|---|---|---|---|
| `openai/gpt-6-luna` | 0.791 | $0.0014 | $0.07 | 8,003 | 786 | 186 |
| `deepseek/deepseek-v4-pro-0813` | 0.833 | $0.0032 | $0.16 | 8,120 | 608 | 0 |
| `qwen/qwen3.8-max-0902` | 0.799 | $0.0252 | $1.26 | 8,464 | 1,459 | 980 |
| `openai/gpt-6-sol` | 0.817 | $0.0268 | $1.34 | 8,003 | 683 | 79 |
| `anthropic/claude-opus-5.5` | 0.868 | $0.0721 | $3.61 | 12,995 | 1,008 | 0 |

## Per-field average score

| Field | `anthropic/claude-opus-5.5` | `deepseek/deepseek-v4-pro-0813` | `openai/gpt-6-sol` | `qwen/qwen3.8-max-0902` | `openai/gpt-6-luna` | `openai/gpt-6-astra` |
|---|---|---|---|---|---|---|
| dispute_resolution | 0.94 | 0.91 | 0.89 | 0.84 | 0.92 | 0.00 |
| governing_law | 0.96 | 0.93 | 0.93 | 0.91 | 0.93 | 0.00 |
| liability_cap | 0.85 | 0.77 | 0.63 | 0.73 | 0.49 | 0.00 |
| penalty_currency | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | nan |
| penalty_data_breach | 0.91 | 0.80 | 0.90 | 0.80 | 0.93 | nan |
| penalty_has_monetary | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | nan |
| penalty_late_delivery | 0.76 | 0.00 | 0.53 | 0.60 | 0.00 | nan |
| penalty_late_payment | 0.97 | 0.98 | 1.00 | 0.96 | 1.00 | nan |
| penalty_max_amount | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | nan |
| penalty_termination_fee | 0.88 | 0.97 | 0.94 | 1.00 | 0.97 | nan |
| penalty_uptime_breach | 0.90 | nan | nan | 0.80 | nan | nan |
| renewal_terms | 0.98 | 0.96 | 0.97 | 0.95 | 0.97 | nan |
| response_time_sla | 0.70 | 0.57 | 0.56 | 0.63 | 0.69 | 0.00 |
| service_credit_cap | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | nan |
| sla_breach_threshold | 0.82 | 0.80 | 0.80 | 0.80 | 0.80 | nan |
| sla_measurement_period | 0.89 | 1.00 | 0.93 | 1.00 | 0.93 | nan |
| termination_clause | 0.93 | 0.92 | 0.92 | 0.89 | 0.90 | 0.00 |
| uptime_guarantee | 0.27 | 0.27 | 0.27 | 0.27 | 0.00 | nan |