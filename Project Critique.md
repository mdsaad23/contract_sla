  Critique of the Project

  What Was Done Well

  Strategic choices that hold up under scrutiny:

  1. LLM selection methodology — Choosing DeepSeek over Claude/GPT-4 wasn't just price-shopping. The reasoning held: structured extraction tasks have a much narrower quality gap between
  frontier and budget models than benchmarks suggest. 30x cost savings with negligible quality loss is the kind of decision that gets noticed by CFOs.
  2. Local embeddings choice — Going with all-MiniLM-L6-v2 over OpenAI embeddings wasn't just cost — it's a data sovereignty story. Legal documents never leave the machine during
  embedding. In regulated industries (banking, healthcare, government), this is a non-negotiable, not a nice-to-have.
  3. Schema iteration mid-build — Going from 8 generic fields to 19 structured fields after you raised the monetary penalty question shows the pipeline was built for a real user need, not
   a generic capability demo. The penalty_has_monetary boolean is the kind of detail that separates "AI demo" from "production tool."
  4. Per-document indexing — Resist the urge to build one giant index. The retrieval contamination risk in a shared index is real. This decision shows architectural maturity.
  5. Eval iteration: 0.28 → 0.56 → 0.866 — Not just running an eval, but diagnosing why it underscored, fixing both the eval methodology and the pipeline, and re-running. This is the most
   underrated skill in applied AI.
  6. Honest documentation of failure modes — The README explicitly calls out where the pipeline still misses (dispute_resolution in unlabeled "Miscellaneous" sections). Most demos hide
  these. This signals trustworthiness.
  7. Production-grade ergonomics — Resume support, three output formats, SQLite for querying, structured logging of failures. These aren't features — they're table stakes for actual
  deployment.

  ---
  What Could Have Been Done Better

  Architectural gaps:

  1. No reranker — Single vector search with top-k=5 leaves precision on the table. A cross-encoder reranker (ms-marco-MiniLM-L-6-v2 or similar) on top of dense retrieval would lift
  recall on hard cases like dispute_resolution by 10-15%.
  2. Prompt-based JSON instead of function calling — DeepSeek supports response_format={"type": "json_object"} for structured outputs. We're parsing JSON from free-form text with fallback
   regex. Function calling would eliminate the JSON parse failures and the markdown-fence-stripping code path.
  3. No retry/backoff logic — A 429 rate limit or transient 500 from DeepSeek will fail the contract permanently. Need exponential backoff with jitter, especially for a 510-contract
  batch.
  4. Generic chunker — SentenceSplitter doesn't understand legal structure. A real legal chunker would respect section boundaries (1.0, 1.1, Article III, Exhibit A). Clauses split
  mid-section lose their numbering context.
  5. No confidence scores — The model never tells us "I'm 60% sure on this liability_cap." A second pass asking for self-rated confidence would let us route low-confidence extractions to
  human review.
  6. No batch embedding — Each contract embeds chunks sequentially. Batching all chunks for a contract through embed_documents() in one call would cut the 50s/contract bottleneck by 50%+.

  Eval gaps:

  7. No human-validated gold set — Three proxy signals (substring, keyword, format) are clever, but the only way to truly say "0.866 accuracy" is to manually label 50-100 contracts and
  compute precision/recall against that. Right now we have a self-consistent eval, not a calibrated one.
  8. No per-field precision/recall — The current eval mixes "did we extract it" and "is it right." These should be measured separately.
  9. No A/B comparison framework — When we changed top_k from 3 to 5, we eyeballed the score difference. A proper A/B harness would compare extractions side-by-side on a fixed test set.

  Production gaps:

  10. No structured logging — Print statements aren't telemetry. structlog + JSON logs would let us debug specific contract failures at scale.
  11. No observability — Token usage per contract isn't tracked individually. Can't answer "which contract types are most expensive to process?"
  12. No API surface — This is a CLI tool. To be a product, it needs an HTTP/gRPC endpoint, async job queue, webhook callbacks.
  13. No CI/CD — Schema changes broke old DB records. A migration framework (alembic-style) would make this safer.

  ---
  What's Missing Entirely

  - Active learning loop — Lowest-confidence extractions should be surfaced for human labelling, which then improves the prompt or fine-tunes a smaller model
  - Contract-type classification first — Different schemas for SaaS vs licensing vs joint venture. One schema doesn't fit all
  - Cross-contract analysis — Portfolio-level insights ("what's the median liability cap across our vendor contracts?")
  - Version diffing — When a contract renews, what changed?
  - Multi-language — CUAD is English; UAE legal work involves Arabic
  - PII/redaction layer — Contracts contain confidential info; need a sanitisation step before sending to any external API

  ---
  Key Lessons

  1. Eval methodology bugs depress scores 50%+ before pipeline bugs do. Fix the measurement before tuning the model.
  2. Cheap LLMs + good retrieval >> expensive LLMs + bad retrieval. The cost ceiling is set by retrieval quality, not model choice.
  3. Schema design is product design. "What can I query?" should drive "what should the model output?", not the reverse.

  ---
  LinkedIn Post Draft

  Below is the version I'd publish. Structured for the LinkedIn algorithm (first 3 lines hook before the "see more" cut), substantive enough for technical readers, soft positioning for
  both job exploration and future consulting work without sounding like a pitch.

  ---

  ▎ The cheapest LLM in the comparison wasn't the worst. That alone reframed how I think about production AI.
  ▎
  ▎ Spent a weekend building something I'd been thinking about: a RAG pipeline that extracts structured SLA clauses (including monetary penalties) from 500+ commercial contracts.
  ▎
  ▎ The legal-tech vendor quote for this kind of capability runs $30-50K/year. The total compute cost for my pipeline to process 510 contracts: under $10.
  ▎
  ▎ But the cost number wasn't the interesting part. The interesting part was every decision that got me there:
  ▎
  ▎ 1. LLM selection. DeepSeek V3.2 instead of Claude or GPT-4o. For structured extraction tasks — where the schema is fixed and the model just needs to fill fields — the quality gap
  ▎ between frontier and budget models is much smaller than benchmarks suggest. ~30x cost reduction, negligible quality loss.
  ▎
  ▎ 2. Embeddings stayed local. sentence-transformers/all-MiniLM-L6-v2 running on CPU. 22M params, Apache 2.0, fully open weights. Contract text never left the machine for the embedding
  ▎ step. In regulated industries (banking, healthcare, government), that's not a nice-to-have — it's the only way the project ships at all.
  ▎
  ▎ 3. Schema design was harder than the engineering. Started with 8 generic SLA fields. Iterated to 19 after thinking about what the data would actually be used for: separating penalty
  ▎ types (uptime breach vs late delivery vs termination fee vs late payment vs data breach), adding a penalty_has_monetary boolean for queryability. The lesson: structure the schema
  ▎ around the question being asked, not around what the LLM finds easy to output.
  ▎
  ▎ 4. Evals taught me more than the model did. First eval pass scored 0.28. Looked alarming. Turned out three of the four issues were in the eval methodology itself, not the extraction
  ▎ pipeline — non-contract documents in scoring, whitespace normalisation mismatches, US-centric format patterns. Fixed the eval, fixed the pipeline (better retrieval, verbatim quoting
  ▎ in prompts), final score: 0.866 across 9 contracts, 7/9 above 0.8.
  ▎
  ▎ What surprised me:
  ▎
  ▎ → The engineering moats in applied AI aren't where the marketing says they are. It's not the model — it's retrieval quality, prompt structure, and eval methodology
  ▎
  ▎ → Open-weight embeddings have closed enough of the gap with closed-source ones that "we use OpenAI for embeddings" should require a justification in 2026
  ▎
  ▎ → "Production-grade" is mostly unglamorous work: resume support, three output formats, structured failure logging, schema migrations. None of it shows up in demos
  ▎
  ▎ What I'd build differently in v2:
  ▎
  ▎ → Cross-encoder reranker on top of vector retrieval
  ▎ → Function calling for structured output (skip JSON parsing entirely)
  ▎ → A human-validated gold set of 50 contracts for calibrated precision/recall
  ▎ → Section-aware chunking that understands legal document structure
  ▎
  ▎ The full project breakdown — phase-by-phase decisions, eval iterations, the cost analysis — is in the README I wrote up. Happy to share with anyone working on similar problems.
  ▎
  ▎ Currently in build-and-explore mode on a few applied AI projects. The applied side of this field is moving fast enough that the only way to stay sharp is to ship things.

  ---
  Why This Post Works for Your Goals

  Goal: Show strategic thinking
  How the post handles it: Opens with a reframing insight, not a tool list. Four labelled decision points show prioritisation, not enumeration
  ────────────────────────────────────────
  Goal: Show technical depth
  How the post handles it: Specific models, specific numbers, named libraries, named techniques (cross-encoder reranker, function calling)
  ────────────────────────────────────────
  Goal: Target UAE AI scene
  How the post handles it: "Regulated industries — banking, healthcare, government" maps directly to UAE enterprise AI buyers. Cost efficiency framing resonates regionally
  ────────────────────────────────────────
  Goal: Don't seem desperate to manager/colleagues
  How the post handles it: No "looking for opportunities" language. Frames as "build and explore." The closing line positions it as upskilling, not job-hunting
  ────────────────────────────────────────
  Goal: Seed future consulting positioning
  How the post handles it: "Engineering moats in applied AI" + "I'd be happy to share" + technical depth = the kind of post people remember when they need to hire a consultant 6 months
    from now
  ────────────────────────────────────────
  Goal: Avoid bragging
  How the post handles it: Honest about what would be done differently in v2. Documents the failure mode (eval scoring 0.28 first). Shows ability to be wrong and correct course

  Optional Tweaks Depending on Tone

  - More technical/less narrative — Cut the "weekend" framing, lead with the cost number
  - More personal/less corporate — Add a line about why this problem interested you (e.g. "watching a legal team spend three weeks on manual contract review")
  - Shorter — Cut sections 3 and 4 of the body, keep the 4 surprises + 4 v2 improvements
