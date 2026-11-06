#!/usr/bin/env python
"""LoRA SFT for MiniCPM5-2B on minicpm-dream extraction/reconcile JSONL.

Run on a CUDA machine with ~24 GB VRAM (2B bf16 + LoRA). This is intentionally a
plain transformers + peft loop, not a framework-specific recipe, so the exact
prompts and targets from training/build_dataset.py are preserved.

    python training/train_lora.py \
      --model openbmb/MiniCPM5-2B-SFT \
      --train training/data/train.jsonl \
      --valid training/data/valid.jsonl \
      --output training/out/dream-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class JsonlSftDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int):
        self.examples: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_attention_mask=False,
                )
                self.examples.append({"input_ids": encoded["input_ids"]})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B-SFT")
    parser.add_argument("--train", default="training/data/train.jsonl")
    parser.add_argument("--valid", default="training/data/valid.jsonl")
    parser.add_argument("--output", default="training/out/dream-lora")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_dataset = JsonlSftDataset(args.train, tokenizer, args.max_length)
    eval_dataset = (
        JsonlSftDataset(args.valid, tokenizer, args.max_length)
        if Path(args.valid).is_file() and Path(args.valid).stat().st_size > 0
        else None
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"adapter saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
