# Fine-tuning scaffold

This directory is for the **optional** 2B LoRA phase. Do not start here: the
optimized zero-shot pipeline already reaches cloud-level average quality on the
14-fact suite. Use training only after the expanded eval shows a persistent floor.

## 1. Generate a suite

Run from the repo root with the project virtualenv active.

```bash
minicpm-dream generate-eval --out evals/generated/gpu-eval --seed 42 --days 10 --facts 120
```

One 120-fact suite yields roughly 30 chunk-level examples. For 2–5k SFT records,
generate multiple seeds and pass `--suite` once per directory to
`build_dataset.py`.

The generator writes episodes, `gold.json`, `labels.json`, and notes. Labels keep
the verbatim quote and whether the fact is current, so they seed both evaluation
and training.

## 2. Build SFT data with production prompts

```bash
python training/build_dataset.py \
  --suite evals/generated/gpu-eval \
  --out training/data/gpu-eval \
  --valid-ratio 0.1
```

- `extract` examples use the exact runtime system/user prompt and JSON schema.
- `reconcile` examples use the exact reconcile prompt and target operation.
- The split is deterministic by record index.

## 3. Train LoRA (CUDA, ~24 GB)

```bash
python -m pip install -r training/requirements.txt
python training/train_lora.py \
  --model openbmb/MiniCPM5-2B-SFT \
  --train training/data/gpu-eval/train.jsonl \
  --valid training/data/gpu-eval/valid.jsonl \
  --output training/out/dream-lora
```

Use `MiniCPM5-2B-SFT`, not the RL/OPD release, for a clean SFT starting point.
Two epochs, r=16, bf16, batch 1 × grad-accum 8; that fits a 24 GB card.

## 4. Evaluate

Serve the adapter with vLLM (LoRA enabled), point an OpenAI-compatible config at
it, and run the same suite through `minicpm-dream eval`:

```bash
vllm serve openbmb/MiniCPM5-2B-SFT \
  --enable-lora --lora-modules dream=training/out/dream-lora \
  --port 8000
```

Create `training/eval-config.toml`:

```toml
[model]
backend = "openai"
base_url = "http://127.0.0.1:8000/v1"
name = "dream"
disable_thinking = false
structured_mode = "auto"

[retrieval]
embeddings = "off"
reranker = "off"
```

Then:

```bash
minicpm-dream eval --config training/eval-config.toml \
  --suite evals/generated/gpu-eval --chunk-tokens 350 --max-items 8
```

Compare against the zero-shot checkpoint on **exactly** the same suite. Keep the
deterministic validators and guards in front of the adapter.

## Acceptance gates

Ship the fine-tune only if all hold:

1. F1 improves by at least 5 points over zero-shot on the 100–300 fact suite.
2. Safety violations remain zero; no invented numbers, dates, names, or evidence.
3. Identity/relationship recall does not regress.
4. The adapter still emits schema-valid JSON through the runtime validators.

Do not train on unverified cloud labels. If you use a cloud teacher, run the
same quote/evidence validators over its outputs and keep only reviewed examples.

## Exporting to GGUF

LoRA adapters can be merged and converted with llama.cpp's
`convert_hf_to_gguf.py`, then quantized to Q4_K_M for the CPU runtime. Record the
conversion command and the exact adapter revision in the model card; the default
runtime still points at the upstream MiniCPM5-2B GGUF.
