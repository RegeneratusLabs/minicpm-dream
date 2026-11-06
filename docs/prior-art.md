# Prior art and what we borrowed

This project exists because several strong systems solved parts of the problem,
and none of them combined a **local small model**, **write-time reconciliation**,
and a **reviewable Hermes integration**.

## hermes-dreaming

Source: [hermes-dreaming README](https://github.com/itpartypattaya/hermes-dreaming)
and the [Hermes marketplace entry](https://hermes-agent.nousresearch.com/marketplace/hermes-dreaming).

What it does:

- A deterministic Python pass over the day's human messages and the memory
  store. It calls **no LLM** and never writes memory itself; the agent applies
  proposals through the normal `memory` tool.
- Facts need corroboration from humans on distinct days. Cron prompts, subagent
  text, URLs, untrusted chats, and the agent's own threads never corroborate.
- OpenClaw-style scoring: relevance, frequency, diversity, recency,
  consolidation, richness, plus a corroboration gate.
- Matching against memory with aliases, stemmed containment, conflict detection
  when numbers differ, and a verbatim `old_text` anchor for `memory replace`.
- Screens secrets and prompt injection; expires dated events; deduplicates.
- A **wake gate**: a quiet night prints `{"wakeAgent": false}` and the model
  never starts. Fill level and memory pressure can wake it.
- A **loss guard** for mass deletion, a **diary** (`DREAMS.md`) with provenance,
  cooldowns (3/14/90/180 days), a reject list, and "every item gets an outcome".
- Caps: at most 6 memory changes a night, at most 3 new.

What we borrowed:

- the wake-gate idea and the "zero writes is success" stance;
- the loss-guard and char-budget awareness;
- cooldowns/rejects and the diary-with-provenance format;
- the insight that machine text must never corroborate a human fact;
- the review-first posture for a high-blast-radius store.

What we do differently:

- The semantic work is done by a **local MiniCPM model**, not by waking the
  host agent. This removes the big-model dependency for the nightly pass.
- We keep our own bitemporal ledger, so supersession is a first-class row, not
  a text edit.
- Proposals are exported as ready-to-run `memory` tool calls *and* as candidate
  files, so a Hermes agent can apply them in one review turn.
- We measure the small model on a structured eval instead of assuming a
  deterministic rule set is sufficient.

## OpenClaw dreaming

Source: [OpenClaw dreaming docs](https://github.com/openclaw/openclaw/blob/d8c11402/docs/concepts/dreaming.md).

OpenClaw frames background consolidation as three cooperative phases:

| Phase | Purpose | Durable write |
|---|---|---|
| Light | ingest, dedupe, stage recent material | no |
| Deep | score and promote durable candidates | yes (`MEMORY.md`) |
| REM | reflect on themes and recurring ideas | no |

Deep ranking uses six weighted signals: frequency 0.24, relevance 0.30, query
diversity 0.15, recency 0.15, consolidation 0.10, conceptual richness 0.06.
The Dream Diary is explicitly **not** a promotion source; only grounded snippets
are eligible.

What we borrowed: the phase vocabulary (our light/deep/REM map to
ingest+extract / reconcile+apply / synthesize), the "diary is not memory" rule,
and the weighted-salience idea. We use a simpler deterministic salience score
and let the model's confidence and evidence count contribute, rather than
recomputing retrieval statistics.

## Letta sleep-time compute

Source: [Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute),
paper [arXiv:2504.13171](https://arxiv.org/abs/2504.13171).

Letta runs a primary agent and a separate sleep-time agent. The sleep-time agent
owns memory-editing tools, works while the user is idle, and can be a stronger
model because latency does not matter. This moves memory management off the
interactive path and produces cleaner, more consolidated context than
incremental MemGPT-style edits.

What we borrowed: the batch/asynchronous stance, and the separation of
"conversation" from "memory maintenance". What we changed: instead of a second
general agent with tools, we use a narrow, non-agentic JSON interface. That is
what makes a 1–2B model viable (see below).

## mem0 and the write-time reconciliation question

Sources: [Mem0 paper](https://arxiv.org/abs/2504.19413),
[mem0 v2→v3 migration note](https://docs.mem0.ai/migration/platform-v2-to-v3).

The original mem0 pipeline extracts candidate facts and an LLM chooses among
`ADD`, `UPDATE`, `DELETE`, `NOOP`. The v3 platform deliberately moved to
**ADD-only**: "nothing is overwritten or deleted"; conflicting facts coexist
and retrieval sorts them out with recency/ranking.

That is a real trade-off, and this project takes the other side:

- a fact that changed should have one current row and a visible history;
- "which of these two facts won" should be recorded at write time, not
  re-derived on every read;
- supersession should be a flag with validity dates, not a deletion.

We implement bitemporal supersede (valid time + system time), keep the old row,
and never let the model silently delete. This is closer to Zep/Graphiti's
temporal model and to the "supersede, don't overwrite" pattern practitioners
converge on.

## The small-manager study that shaped the interface

Source: [Memory That Doesn't Expire](https://zenodo.org/records/21964738)
(August 2026), code `MasHasNoLife/yuna-ai`.

The study evaluates four memory strategies on LoCoMo with a fixed answerer and a
local extractor:

| Extractor | Overall judged accuracy | Temporal |
|---|---|---|
| qwen2.5 3B | 0.309 | 0.06 |
| qwen2.5 7B | 0.348 | 0.19 |
| qwen2.5 14B | 0.470 | 0.34 |

Findings that matter here:

1. **A typed protocol is what makes a small manager viable.** MemGPT reported
   that even 70B function-calling models failed as memory managers; a five-op
   fixed grammar is a much smaller ask.
2. **Absolute-date grounding at write time** is responsible for the entire
   temporal advantage: disabling it drops temporal accuracy from 0.364 to 0.081
   (4.5×).
3. **Temporal grounding breaks first** under compression; the study sets a
   minimum viable manager size around 7–14B for its protocol.
4. Retrieval-based memory is length-invariant, while full-context stuffing loses
   76% of accuracy once evidence falls outside the window.
5. Extraction precision can be very high (199/200 hand-verified faithful) when
   every memory is traceable to a source turn.
6. The paper's own future work is a periodic **consolidation/merge** pass to fix
   duplicate-family growth and materialize implied facts.

Our response to (3) is not to pretend a 2B is a 14B. It is to remove the hard
parts from the model:

- dates are resolved by code and offered as an enum (`d0..d4`);
- the model never invents target IDs;
- every claim is quote-grounded and entailment-checked;
- ambiguous merges become `DEFER`;
- the eval compares 1B/2B against 8B, and the default can change.

In other words: narrow the interface until the small model's job is mostly
*selection and phrasing*, and let deterministic code own identity, time,
budgets, and history.

## Other influences

- **MemGPT** ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)): memory
  tiers, and the negative result about small models driving general tools.
- **Generative Agents** ([arXiv:2304.03442](https://arxiv.org/abs/2304.03442)):
  recency/importance/relevance retrieval and periodic reflection.
- **MemoryBank** ([arXiv:2305.10250](https://arxiv.org/abs/2305.10250)):
  decay by strength and selective forgetting.
- **Hermes memory docs**
  ([Persistent Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)):
  2,200-char `MEMORY.md`, 1,375-char `USER.md`, `§` separators, `memory`
  add/replace/remove with `old_text`, frozen prompt snapshot, optional write
  approval.
- **Hermes cron docs**
  ([Scheduled Tasks](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)):
  `no_agent` script jobs and the `wakeAgent` gate. The docs warn that
  `state.db` is an internal schema; our gate therefore reads only our ledger.

## Sources

- https://github.com/itpartypattaya/hermes-dreaming
- https://hermes-agent.nousresearch.com/marketplace/hermes-dreaming
- https://github.com/openclaw/openclaw/blob/d8c11402/docs/concepts/dreaming.md
- https://www.letta.com/blog/sleep-time-compute
- https://arxiv.org/abs/2504.13171
- https://arxiv.org/abs/2504.19413
- https://docs.mem0.ai/migration/platform-v2-to-v3
- https://zenodo.org/records/21964738
- https://arxiv.org/abs/2310.08560
- https://arxiv.org/abs/2304.03442
- https://arxiv.org/abs/2305.10250
- https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
- https://hermes-agent.nousresearch.com/docs/user-guide/features/cron
