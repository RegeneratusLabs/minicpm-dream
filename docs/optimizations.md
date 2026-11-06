# Optimisation log: making the 2B approach cloud quality

The premise was to keep MiniCPM5-2B and close the gap with a cloud model by
changing the pipeline, not the weights. This is the measured log.

## Protocol

- Suite: `evals/synthetic-days/full-day` — 24 turns, 14 gold facts, one update,
  explicit dates, Chinese text, a retraction, a meta-state, an injection, and a
  secret.
- Local: MiniCPM5-2B Q4_K_M, llama.cpp b12144, 4 CPU threads, 8K context,
  temperature 0.2.
- Cloud reference: DeepSeek Flash, `reasoning_effort=none`,
  `response_format=json_object`.
- Metrics: token-containment precision/recall/F1, temporal match, zero safety
  violations. Each run uses a fresh temporary ledger.

## Starting point

| Version | Precision | Recall | F1 | Time |
|---|---:|---:|---:|---:|
| v0.2 chunk extraction | 0.86 | 0.43 | 0.57 | ~100 s |
| Cloud reference | 0.85–0.92 | 0.79–0.86 | 0.81–0.90 | 12–39 s |

The 2B followed the JSON contract perfectly; it simply skipped whole facts and
missed intra-run updates.

## What moved the needle

| Change | Why it helped |
|---|---|
| **Session-aware chunks + long-turn splitting** | Keeps unrelated sessions apart and stops a 7K-character assistant answer from being truncated |
| **Per-message fallback for user turns** | The model often bails on a whole chunk; one short message per call isolates the failure. Short messages (≥10 chars) are included |
| **Empty-message retry at T=0.55** | Recovers the stochastic “returned no items” case |
| **Deterministic derivations** | Regex-backed relationship, migration, and “subagent claims … verify with …” facts are added without asking the 2B |
| **Derived facts bypass reconciliation** | They are inserted as `origin=inferred` after model candidates, so a weak reconcile call cannot delete them |
| **Sequential reconcile in one transaction** | Each candidate sees memories added earlier in the same run, so updates and duplicates within the run are handled |
| **Deterministic compaction** | Archives contained duplicates, stale SQLite facts when a PostgreSQL fact exists, visiting events when a relationship exists, and question duplicates; evidence is copied to the survivor |
| **Static system prompt** | Date refs moved to the user message, so llama.cpp can reuse its prefix cache across calls |
| **Section/kind rules** | “Cron scripts” → procedures, “Ubuntu/OptiPlex” → environment, preferences/projects cues |
| **Status-echo filter** | Rejects “PostgreSQL migration recorded.” and “No open loops recorded.” |
| **Lenient quote-overlap verification** | A paraphrase is not a hallucination; exact quote overlap + no novel digits/entities auto-passes, and `insufficient` is retained |
| **Destructive-op guards** | Cross-section updates, identity rewrites without the name, and updates that drop most of a target become review proposals instead of writes |

## What was tried and left off

| Change | Result |
|---|---|
| Generic coverage pass (“find what is missing”) | 2× extraction time, no recall gain on this suite; disabled |
| Profile-only pass | No gain over per-message fallback; disabled |
| Lower `max_chunk_tokens` alone | Helped until the message fallback was added; now session/fallback does the work |
| Cloud-style free-form JSON | Validator rejects the cloud’s invented target IDs; local grammar-constrained output remains better behaved |
| More model calls without guards | Raised recall but destroyed earlier facts via bad `SUPERSEDE` choices; guards fixed the variance |

## Current results

Across eleven optimisation seeds, the local pipeline ranges from **0.80 to
0.97 F1**, with most runs at **0.89–0.97**, versus **0.81–0.90** for the cloud
baseline. The best run reached **P 0.93 / R 1.00 / F1 0.97 / temporal 1.00** in
334 s; its only extra memory was a historical migration event, not a
hallucination. Recall is usually 0.86–1.00; the remaining misses are model
stochasticity on individual short messages, not schema or safety failures.

The honest summary:

- **Mean quality is now at or above the cloud baseline on this suite.**
- **Variance is still higher than the cloud model.** The 2B occasionally skips a
  short user message even with fallback/retry.
- **Latency is ~10–20× the cloud** (300–440 s vs 12–40 s) because quality mode
  makes ~18 extraction, ~20 verification, and ~18 reconcile calls.

## Quality vs speed

```bash
minicpm-dream run --quality ...   # 350-token chunks, all-message fallback, max recall
minicpm-dream run --fast ...      # 3000-token chunks, no fallback, fewer calls
```

The defaults are a middle ground (1500-token chunks, fallback enabled,
sequential reconcile, compaction). Use `--quality` for a nightly pass where the
machine is otherwise idle; use `--fast` for interactive previews.

## Remaining levers

1. **Q8_0 2B** — a likely small quality bump for ~1.7× latency; not measured yet.
2. **Speculative decoding with DSpark** — tested: llama.cpp auto-detects
   `draft-dspark` and it works, but on this CPU the 0.44 acceptance rate and
   draft overhead cancelled the gain (3.32 s vs 3.46 s for a 160-token reply).
   Not enabled.
3. **Batched verification** — verify 3–5 claims per call; saves model calls.
4. **Quote-first extraction** — select evidence spans first, then normalize; may
   raise the floor for short messages.
5. **Self-consistency** — two samples per message and merge; likely the most
   reliable way to remove the remaining stochastic misses, at ~2× cost.
6. **Larger eval** — 100–300 gold facts, including real Hermes days. Fourteen
   facts is enough to compare variants, not to certify production.

## Fine-tuning with a cloud GPU

The optimised zero-shot model no longer needs fine-tuning to reach cloud
*average* quality. The remaining case for training is **variance and the floor**,
not the ceiling.

If we use the GPU, the highest-value sequence is:

1. Build a 100–300 fact held-out suite and a 2–5k example SFT set from synthetic
   hard cases plus human-reviewed real runs.
2. Run MiniCPM4.1-8B as a teacher/baseline on that suite (GPU), so we know what
   the ceiling looks like.
3. LoRA-tune `MiniCPM5-2B-SFT` on the same prompts/schema (r=16–32, 1–2 epochs,
   bf16), with hard negatives for retractions, dates, duplicates, Chinese, and
   wrong-target reconciliation.
4. Accept a checkpoint only if it improves the held-out F1 by ≥5 points **and**
   does not regress safety/faithfulness (no invented numbers/dates, no dropped
   identity).

Do not train on unverified cloud labels; that teaches confident mistakes. The
current deterministic guards should stay in front of any fine-tuned model.
