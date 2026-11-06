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
| Embeddings (optional) | BAAI/bge-small-en-v1.5 | 33M / ONNX | 512 | MIT, CPU-friendly, already common locally |

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
- **MiniCPM-Embedding** — 2.4B, 512-token input limit, and a separate model
  license that requires registration for commercial use. bge-small is smaller,
  MIT, and good enough for candidate retrieval.

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

## Grammar-constrained output

llama.cpp converts a JSON Schema into a grammar at decode time. This guarantees
*parseable* output; it does not guarantee meaning. The pipeline therefore:

1. keeps schemas small (few optional fields, bounded strings);
2. describes every field in the prompt (the schema is invisible to the model);
3. validates semantics in Python (quotes, IDs, digits, dates, confidence);
4. retries once, then routes to `DEFER` rather than guessing.

Schemas: `src/minicpm_dream/schemas/{extract,verify,reconcile}.json`.

## Evaluation

`minicpm-dream eval --suite evals/synthetic-days` runs the full pipeline against
a fixture with known gold facts and reports:

- extraction precision / recall / F1 (token containment match);
- schema validity (rejected items / attempted items);
- temporal accuracy (resolved `valid_from` matches gold);
- safety violations (secret-shaped active memories);
- applied / deferred counts and wall time.

Baselines on the reference machine (6-core i5-8500T, 15 GB RAM, no GPU,
llama.cpp b12144, 4 threads, 8192 context):

| Backend | Suite | Precision | Recall | F1 | Schema | Temporal | Safety | Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mock (plumbing only) | basic, 8 msgs | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | <1 s |
| MiniCPM5-2B Q4_K_M | basic, 8 msgs | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 49.4 s |

**These are one small synthetic scenario with three gold facts.** They prove the
model can follow the interface on this case; they are not a general benchmark,
not a LoCoMo score, and not evidence that 2B is sufficient for every day. Treat
the table as a regression baseline and expand the suite before making a model
decision.

What the real run actually did:

- extracted 3 durable memories and rejected the retraction ("no longer uses
  Ollama") as a new fact;
- resolved "tomorrow at 3pm" to `2026-11-06` and typed it as a task/open loop;
- rejected the meta-state ("the user is asking about lunch");
- dropped the turn containing an API key before prompting (redaction);
- produced 2 Hermes review actions (the open loop is diary-only);
- applied 3 ledger changes, 0 deferred, loss guard clean.

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

- 100–300 message synthetic days with 20–50 gold facts, contradictions,
  corrections, multi-day recurrence, and multilingual turns.
- MiniCPM5-1B and MiniCPM4.1-8B on the same suite.
- Real Hermes days (with consent, aggregated metrics only) to measure the
  false-positive rate on ordinary conversation.
- A held-out "quiet day" set to measure how often the pass invents work.

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
- bge-small-en-v1.5: MIT.
- llama.cpp: MIT.
- Code in this repo: MIT. See `NOTICE`.
