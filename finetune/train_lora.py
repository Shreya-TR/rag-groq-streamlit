import argparse
import json
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    train_file: str
    base_model: str
    output_dir: str
    epochs: float
    batch_size: int
    lr: float
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def _format_prompt(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()
    text = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{output_text}"
    )
    return {"text": text}


def train(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    ds = load_dataset("json", data_files=cfg.train_file, split="train")
    ds = ds.map(_format_prompt)

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = ds.map(tokenize, remove_columns=ds.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized,
        "data_collator": data_collator,
    }
    # transformers<5 accepted `tokenizer`; transformers>=5 removed it.
    try:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    summary = {
        "base_model": cfg.base_model,
        "train_file": cfg.train_file,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.lr,
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "global_steps": train_result.metrics.get("global_step"),
    }
    with open(os.path.join(cfg.output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="LoRA SFT training script.")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", default="finetune_outputs/lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    a = parser.parse_args()
    return TrainConfig(
        train_file=a.train_file,
        base_model=a.base_model,
        output_dir=a.output_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        max_length=a.max_length,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
