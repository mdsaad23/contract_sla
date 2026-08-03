# AI Tinkerers Dubai — 5-Minute Talk Script

**Event:** Saturday, August 8, 2026
**Slot:** 5 minutes speaking, live demo
**Title:** *My Eval Said 0.28. My Benchmark Said Local Models Were Useless. Both Were Wrong.*

This is the word-for-word delivery script for the confirmed 5-minute slot. It's
a trim of the 6-minute version in `AI_TINKERERS_DEMO_PROPOSAL.md` — same three
beats, one demo cut (retrieval lid-off) to make the clock. Read it once
out loud with a timer before Saturday; adjust the bracketed pacing notes to
your own cadence.

Per the AI Tinkerers demo guide: this is a live run, no pre-recorded video,
one slide only (for the benchmark table, since that's the one thing better
read than spoken), large terminal font, and the goal is sharing a debugging
story, not a product pitch.

---

## Timing at a glance (5:00 total)

| Time | Beat | Action |
|---|---|---|
| 0:00–0:25 | Hook | Say it, cost line already on screen |
| 0:25–0:50 | The problem | Show the buried clause in a contract PDF |
| 0:50–1:30 | Live run #1 | `python scripts/run_pipeline.py <contract>` (API) |
| 1:30–2:15 | Eval story | 0.28 → 0.79, show the diff |
| 2:15–3:00 | Live run #2 | Same contract, `LLM_PROVIDER=ollama`, side-by-side JSON |
| 3:00–3:45 | Benchmark story | One slide: 13-model table, the num_ctx bug |
| 3:45–4:30 | Payoff query | `sqlite3 output/results.db`, 128/510 |
| 4:30–5:00 | Takeaway + link | Close |

Hard cut list if you're running long, in order: shrink the eval-story diff
walkthrough to one sentence; skip narrating live run #2's JSON line-by-line
and just say "same output, 9GB model, $0"; drop the benchmark slide's
long-context caveat.

---

## 0:00–0:25 — Hook

**[Screen: terminal showing the final cost line from the 510-contract run]**

> "I ran a RAG pipeline over 510 real commercial contracts — SEC-filed
> agreements, the CUAD dataset — and pulled 19 structured fields out of each
> one: uptime guarantees, penalty clauses, termination fees. Total API cost
> for all 510: one dollar seventy-two."

*(Pacing note: don't explain RAG yet — the definition lands naturally in the
next beat.)*

---

## 0:25–0:50 — The problem, in one sentence

**[Screen: a contract PDF scrolled to a buried liquidated-damages clause]**

> "Contracts like this bury the numbers that matter — penalty amounts,
> caps, termination fees — in paragraphs of legal boilerplate. RAG just means:
> find the paragraph that actually answers the question, then have a model
> read only that paragraph. That's the whole idea."

---

## 0:50–1:30 — Live run #1 (real API call, ~6 seconds)

**[Screen: terminal, run the command live]**

```
python scripts/run_pipeline.py <contract>
```

> "Watch the terminal, not me. This is one contract, live, real API call. PDF
> in, 19 retrieval queries against a per-contract vector index, one DeepSeek
> call, structured JSON out."

*(Let it run — ~6 seconds of dead air is fine, it proves it's live.)*

> "Six seconds. Nineteen fields, each one traceable back to an exact quote
> in the source — that's a deliberate constraint: the model has to copy the
> clause verbatim or say null, never paraphrase. That's what makes the next
> part possible."

---

## 1:30–2:15 — The eval story

**[Screen: `evals/eval_runner.py` + the score-fixing diff]**

> "First time I scored this pipeline automatically, I got 0.28 out of 1. My
> instinct was: the retrieval is broken, rewrite it. Instead I checked the
> eval first. Three bugs — none in the pipeline. The dataset's own
> documentation file was being scored as if it were a contract. My text
> cleaner stripped apostrophes, so 'don't' in the source never matched 'don t'
> in my extraction. And the eval was still grading an 8-field schema the
> pipeline had already grown past to 19. Fix the eval, not the pipeline:
> score goes to 0.79. The extraction was right the entire time."

> "Lesson: if your eval is younger than your pipeline, check the eval before
> you believe the number."

---

## 2:15–3:00 — Live run #2: same contract, local model

**[Screen: terminal, run the command live]**

```
LLM_PROVIDER=ollama python scripts/run_pipeline.py <contract>
```

> "Same contract. One environment variable. Now it's a 9GB model running on
> my own GPU instead of an API call — zero network, zero dollars."

*(Let it run — slower this time, ~30-40s. Narrate while it runs rather than
standing silent.)*

> "While that finishes: this is the exact same retrieval, the exact same
> prompt. The only thing that changed is which model reads the context.
> [When done] Nearly identical output to the API run."

---

## 3:00–3:45 — The benchmark story

**[Screen: the one slide — 13-model results table]**

> "So I benchmarked 13 local models against the API on all the same
> contracts. First pass, the local models looked hopeless. Then I found the
> bug — Ollama defaults to a 4096-token context window. My longest contract
> prompt is 23,000 tokens. Ollama doesn't error on overflow, it silently
> truncates and answers anyway. One config change — raising that limit —
> took a 3-billion-parameter model's score from 0.372 to 0.487, and flipped
> every one of its failures into a pass. Nothing about the model changed.
> Only the measurement did."

> "Once the benchmark was honest: the best local model, a 9GB model called
> Qwen3, scored 0.807. The API scored 0.819. A one-and-a-half percent gap —
> for free, on a laptop GPU."

---

## 3:45–4:30 — The payoff query

**[Screen: terminal, run live]**

```
sqlite3 output/results.db
```

> "This is why the schema has 19 fields instead of 8 — so I can ask the
> question that actually matters: which of these contracts have a real cash
> penalty, not just a service credit? [run query] 128 out of 510. That's a
> business answer, not a text blob — because every field is queryable SQL."

---

## 4:30–5:00 — Takeaway + close

**[Screen: repo URL, large and readable]**

> "Twice on this project I almost rewrote working code because the thing
> measuring it was broken, not the thing itself. Check the instrument before
> you trust the reading. When mine was honest, a free 9GB model on a
> consumer GPU landed within two percent of a frontier API. Repo, eval
> report, and the full 13-model benchmark are all public — link's up now."

---

## Jargon, defined once, in-line (per demo guide)

Don't re-explain these once said — say the definition inline the first time
the term appears, then move on:

- **RAG** → "find the paragraph that answers the question, then have a model
  read only that paragraph" (said at 0:25).
- **Eval** → implied by "scored this pipeline automatically" (1:30) — don't
  stop to define "eval" as a word, the sentence does the work.
- **Local model / Ollama** → "a model running on my own GPU instead of an
  API call" (2:15) — covers both terms in one clause.
- **num_ctx / context window** → "how much text the model is allowed to
  read at once" — only needed if someone asks in Q&A; the script above
  already conveys it via "silently truncates."

---

## Pre-flight checklist (Saturday morning)

- [ ] `.env` key valid — one throwaway run done on venue wifi
- [ ] `ollama serve` running, target model pulled and warmed (first call
      pays a load penalty — don't let that happen on stage)
- [ ] `OLLAMA_NUM_CTX=24576` exported — this is the whole story, don't skip it
- [ ] `output/results.db` present and populated (510 rows)
- [ ] Demo contract picked in advance: unambiguous cash penalty, small
      enough the local model answers in ~30s on stage
- [ ] Terminal font scaled for a projector, dark theme, two windows ready
      (API run / local run) so no mid-demo env-var fumbling
- [ ] Benchmark slide (the one slide) open and ready to alt-tab to
- [ ] `asciinema`/screen recording of both runs saved as a fallback only —
      not to be used unless live genuinely fails
- [ ] Repo public, link short enough to read off a screen at 5:00
- [ ] Run the whole script once, out loud, with a stopwatch — if it's over
      5:00, cut from the "hard cut list" above, in order

---

## Backup plan

If the DeepSeek API is unreachable (dead venue wifi): skip straight to live
run #2 (fully local, no network needed) and narrate the eval/benchmark
story from the pre-generated report instead of re-running it. The SQL query
against `output/results.db` needs no network either way — that beat never
has to change.
