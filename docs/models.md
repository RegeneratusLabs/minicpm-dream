# Model stack

## Recommendation

**Default: MiniCPM5-2B, Q4_K_M GGUF, served by llama.cpp.**

It is the best capability-per-byte fit for a CPU-only or small-GPU box, and it is
the model this repo is tuned against.

| Role | Model | Size / format | Context | Why |
|---|---|---|---|---|
| Extractor + reconciler | **MiniCPM5-2B** | 2.52B / Q4_K_M 1.56 GB | 131,072 | 2B-class SOTA; strong instruction following, long context and tool use; standard `LlamaForCausalLM`; Apache-2.0 |
| Cheap verifier / triage | MiniCPM5-1B | 1.08B / Q4_K_M 0.69 GB | 131,072 | Same family, faster; useful as a second opinion or salience screen |
| Quality tier | MiniCPM4.1-8B | 8B / Q4_K_M 4.97 GB | 64K native, 128K YaRN | Trainable sparse attention + hybrid reasoning; best quality per run when a GPU is available |
| Million-token tier | MiniCPM-SALA | 9B hybrid sparse+linear | 1M+ | Whole-week/whole-corpus single pass on a 24 GB GPU |
| Embeddings | **MiniCPM-Embedding-Light** | 440M / safetensors 0.87 GB | 8,192 | 1024-dim, dense + sparse, bilingual; CPU-friendly |
| Reranker | **MiniCPM-Reranker-Light** | 1.2B / safetensors 2.72 GB | 8,192 | Cross-encoder; reranks the top candidates |

### Why not the alternatives

- **MiniCPM4-0.5B / BitCPM4-0.5B** — too weak for typed extraction and
  reconciliation; a 0.5B can classify, not reliably ground a fact in a quote.
  It could be a future salience gate, not the consolidator.
- **MiniCPM5-1B as default** — a good fallback if 2B latency is unacceptable,
  but it gives up the long-context and instruction-following margin that this
  task needs.
- **MiniCPM4.1-8B as default** — quality is better, but on a 6-core CPU a
  nightly pass becomes hours, and 5 GB of weights compete with the OS. Keep it
  as the GPU/quality tier and as the baseline that tells us whether 2B is good
  enough.
- **MiniCPM-SALA as default** — 1M context is attractive for whole-week passes,
  but it needs a serious GPU and custom kernels; not a distributable default.
- **MiniCPM-Embedding (full)** — 2.4B and capped at 512 tokens. The 440M
  `MiniCPM-Embedding-Light` is a much better fit for a nightly CPU pass and
  supports 8K-token memories. Both use the MiniCPM model license (free academic,
  commercial registration).
- **bge-small** — a fine MIT alternative, but not OpenBMB and capped at 512
  tokens; it is not wired into the current retrieval factory.

## Runtime

| Backend | Use | Notes |
|---|---|---|
| `llama_cpp` | local GGUF, CPU or CUDA | The tool starts/stops `llama-server`, uses `--jinja`, JSON-schema `response_format`, one parallel slot |
| `openai` | vLLM, SGLang, Ollama, LM Studio, remote GPU | Any OpenAI-compatible `/v1/chat/completions`; set `base_url` and `api_key_env` |
| `mock` | tests, CI, offline demo | Scripted JSON per task; no model download |

Recommended sampling for structured extraction:

```toml
[model]
temperature = 0.2
max_tokens = 1200
context = 8192
threads = 4
gpu_layers = 0
```

MiniCPM5's card recommends `temperature=1.0, top_p=0.95, min_p=0.0` for chat.
This tool deliberately runs cooler (0.1–0.3) because the task is extraction, not
creative generation; the schema constrains the shape, not the content. If you see
repetitive output, set `min_p=0.0` in llama.cpp and lower `repeat_penalty` rather
than raising temperature.

## Retrieval stack (OpenBMB)

`retrieval.embeddings = "openbmb"` and `retrieval.reranker = "openbmb"` are the
defaults. The pipeline:

1. keeps FTS5 BM25 as the lexical channel;
2. embeds memory text with MiniCPM-Embedding-Light and stores 1024-dim vectors in
   SQLite (`memory_vectors`), re-embedding only new/changed memories;
3. fuses lexical and dense ranks with reciprocal rank fusion;
4. reranks the top candidates with MiniCPM-Reranker-Light.

Measured on the reference 6-core CPU, float32 embedding and float16 reranker:

| Step | Time |
|---|---|
| load embedding model | ~7 s |
| embed one query | ~0.14 s |
| embed 3 memories | ~0.23 s |
| load reranker | ~7 s |
| rerank 3 documents (batch 2, 256 tokens) | ~3.1 s |

Both models are optional: `uv pip install -e ".[openbmb]"`. If the import or
weights are missing, retrieval falls back to FTS5 with a warning unless
`retrieval.strict = true`. Prefetch weights with
`minicpm-dream setup --download-embeddings`; build the index with
`minicpm-dream index` (or `--rebuild`).

The embedding model has a custom license: free for academic research, commercial
use requires registration with ModelBest/OpenBMB. The same applies to the
reranker. If you distribute a commercial product, complete that registration
first.

## Grammar-constrained output

llama.cpp converts a JSON Schema into a grammar at decode time. This guarantees
*parseable* output; it does not guarantee meaning. The pipeline therefore:

1. keeps schemas small (few optional fields, bounded strings);
2. describes every field in the prompt (the schema is invisible to the model);
3. validates semantics in Python (quotes, IDs, digits, dates, confidence);
4. retries once, then routes to `DEFER` rather than guessing.

Schemas: `src/minicpm_dream/schemas/{extract,verify,reconcile}.json`.

## Evaluation

`minicpm-dream eval --suite evals/synthetic-days/full-day --chunk-tokens 350 --max-items 8`
runs the full pipeline against a fixture with known gold facts and reports:

- extraction precision / recall / F1 (token containment match);
- schema validity (rejected items / attempted items);
- temporal accuracy (resolved `valid_from` matches gold);
- safety violations (secret-shaped active memories);
- applied / deferred counts and wall time.

Baselines on the reference machine (6-core i5-8500T, 15 GB RAM, no GPU,
llama.cpp b12144, 4 threads, 8192 context):

| Backend | Suite | Precision | Recall | F1 | Schema | Temporal | Safety | Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mock (plumbing only) | basic, 8 msgs / 3 facts | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | <1 s |
| MiniCPM5-2B Q4_K_M | basic, 8 msgs / 3 facts | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 49.4 s |
| MiniCPM5-2B Q4_K_M | full-day, 24 msgs / 14 facts | 0.86 | 0.43 | 0.57 | 1.0 | 0.67 | 0 | 100.7 s |
| deepseek-flash (cloud) | full-day, 24 msgs / 14 facts | 0.92 | 0.86 | 0.89 | 1.0 | 0.67 | 0 | 19.0 s |

The full-day suite is deliberately denser than `basic/`: 24 turns across three
days, 14 gold facts, an update, explicit dates, a Chinese turn, a retraction, a
meta-state, an injection, and a secret. It uses 350-token chunks and up to 8
items per call.

**Read this as a regression comparison, not a benchmark.** It is one synthetic
scenario. The local timing was measured while another `llama-server` was using
the CPU; the quality numbers are unaffected by that contention, but the wall
time is pessimistic. The 2B is stochastic at temperature 0.2: reruns move by
several points of recall.

The cloud comparison calls DeepSeek Flash (`deepseek-flash`) with
`reasoning_effort=none` and `response_format=json_object` (its API does not
support `json_schema`). Evidence arrays arrive as strings, not objects; the
validator normalizes that shape. Both models receive the same prompt and the
same deterministic validators.

What the real run actually did:

- extracted 3 durable memories and rejected the retraction ("no longer uses
  Ollama") as a new fact;
- resolved "tomorrow at 3pm" to `2026-11-06` and typed it as a task/open loop;
- rejected the meta-state ("the user is asking about lunch");
- dropped the turn containing an API key before prompting (redaction);
- produced 2 Hermes review actions (the open loop is diary-only);
- applied 3 ledger changes, 0 deferred, loss guard clean.

### Real Hermes memory validation

A sandbox copy of the live `MEMORY.md` / `USER.md`, plus a real 14-message
session (only four turns had non-empty text; the rest were tool calls), was run
through both stacks:

| | Cloud (deepseek-flash) | Local MiniCPM5-2B + OpenBMB |
|---|---|---|
| candidates | 16 | 4 |
| verified | 15 | 3 |
| applied to ledger | 10 | 3 |
| deferred | 5 | 0 |
| new `MEMORY.md` entries | 8 | 0 (only open loops/events) |
| wall time | 28 s | 264 s |

The live profile is essentially full: `MEMORY.md` 2,188/2,200 chars and
`USER.md` 1,374/1,375. With the real limits, the tool correctly generated no
Hermes actions and reported the overflow under `skipped`; the sandbox run above
used expanded 4,000/2,500 limits to exercise the write path. `review.json` now
carries a `pressure` block, and the cron gate wakes when memory is at or above
85% or entries were skipped for budget.

This is one tool-heavy session, not a representative day. It does show the
real-world failure mode: when memory is full, consolidation quality matters
more than extraction, because there is no room to add anything.

### Do we need fine-tuning?

Current evidence says **not for v1, and not yet**.

- The typed interface is not the bottleneck: schema validity is 1.0 and safety
  violations are 0 for both models. The 2B follows the JSON contract.
- On a small, concise day it matches the cloud model (3/3 facts). On a denser,
  long-message day it recalls 6/14 facts versus 12/14 for the cloud model. Those
  misses are mostly whole-chunk omissions and long assistant text, not malformed
  JSON.
- The local run's failures are addressable without gradient updates: split by
  session/day, use smaller chunks, extract per-message for long turns, add a
  second pass over missed chunks, and improve kind/section hints.
- Fine-tuning introduces new failure modes (overconfidence, hallucinated numbers)
  exactly where the current validators are strongest. It also needs a labelled
  corpus, a GPU, and a held-out suite larger than 14 facts.

The sensible order is:

1. **1B vs 2B vs 4.1-8B** on a 100–300 fact suite. A 4B/8B tier may close the
   gap more cheaply than training.
2. **Interface work**: session-aware chunking, per-message extraction for long
   assistant turns, a second "did we miss anything?" pass.
3. **Only then** LoRA on `MiniCPM5-2B-SFT` with reviewed examples, and only if the
   gap survives 1 and 2. Keep the safety validators unchanged and require the
   fine-tuned model to beat zero-shot on the same held-out suite.

If the product must match cloud recall on dense days, the first spend should be
a larger local model (4.1-8B) or an optional cloud escalation path, not a
fine-tune of the 2B.

### Failure modes found while wiring the 2B

These are why the interface looks the way it does:

1. **Thinking mode silently eats the budget.** MiniCPM5's template defaults to
   reasoning. Without `enable_thinking=false`, llama-server returned empty
   `content` and 1,200 reasoning tokens. Server flags alone were not enough;
   the request needs `chat_template_kwargs: {"enable_thinking": false}`. The
   backend now sends it (config `model.disable_thinking`), and an empty
   reasoning-only response raises a clear error instead of storing nothing.
2. **The model ignored `date_ref` and wrote "tomorrow".** Code now infers a
   unique date ref from the quote/text and rejects an unresolvable relative
   expression; the model cannot compute dates.
3. **The model stored a retraction as a preference.** A deterministic
   retraction detector rejects "no longer / stopped using / not true" items.
4. **The model called a reminder a preference.** Strong appointment/reminder
   cues force `kind=task` and `section=open_loops`.
5. **The model summarized the noise turn** ("the user mentioned dinner plans but
   no details were provided"). That is now caught by the meta-state filter.
6. **The model picked a date for undated facts.** A date ref that is not
   mentioned in the text or quote is ignored; only event/task entries fall back
   to the episode date.

When a model fails, prefer to fix the *interface* (smaller schema, explicit
field descriptions, fewer items per call) before fine-tuning.

### Live-day observation

A dry run over one real Hermes day (17 human/assistant turns, one 3,649-token
chunk) produced **0 candidates, 0 writes**, in 90 s. The prompt was dominated by
CPU prefill (~4.7K actual prompt tokens); generation was one JSON line.

That is a plausible quiet day, not proof of correct restraint. It could also be
under-extraction. The next suite must include both a deliberately quiet day
(expect 0) and a busy day with known facts (expect recall), and the pass should
be judged on both.

### Next measurements

- Expand from 14 to 100–300 gold facts with contradictions, corrections,
  multi-day recurrence, and multilingual turns.
- MiniCPM5-1B and MiniCPM4.1-8B on the same suite.
- Session-aware chunking and per-message extraction for long assistant turns.
- A held-out "quiet day" set to measure how often the pass invents work.
- Human review of 50–100 extracted memories to measure faithfulness, not just
  text overlap.

## Fine-tuning (deferred)

v1 is zero-shot. If the eval shows a specific gap, the training path is:

1. Generate synthetic days with a teacher model (or accepted real runs with
   human review), paired with the exact prompt and JSON schema.
2. Fine-tune `MiniCPM5-2B-SFT` or `MiniCPM5-2B-Base` with LoRA/TRL on
   extraction + reconciliation examples, keeping a held-out scenario set.
3. Compare against the zero-shot checkpoint on the same eval; only ship if the
   *safety* metrics do not regress (hallucinated numbers/dates are a hard fail).
4. Keep the training data schema in-repo, but never commit user transcripts.

Do not fine-tune on model-generated labels without a verification pass: that
teaches the model its own mistakes.

## Licensing

- MiniCPM5 / MiniCPM4.1 / MiniCPM-SALA weights: Apache-2.0.
- MiniCPM-Embedding-Light and MiniCPM-Reranker-Light: MiniCPM Model License
  (academic free; commercial registration required).
- llama.cpp: MIT.
- Code in this repo: MIT. See `NOTICE`.
