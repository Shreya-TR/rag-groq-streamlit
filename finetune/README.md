# Fine-Tuning Module (LoRA SFT)

This folder adds a real supervised fine-tuning (SFT) pipeline so you can report **actual epochs**.

## 1) Install fine-tuning dependencies

```bash
pip install -r requirements-finetune.txt
```

## 2) Prepare dataset

Input formats:
- `jsonl` with `instruction`, `input`, `output`
- `csv` with `question`, `answer` (converted by script)

Convert CSV to JSONL:

```bash
python finetune/prepare_sft_data.py --input_csv data/qa.csv --output_jsonl finetune/data/train.jsonl
```

## 3) Train LoRA adapter

```bash
python finetune/train_lora.py ^
  --train_file finetune/data/train.jsonl ^
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --output_dir finetune_outputs/tinyllama-lora ^
  --epochs 2 ^
  --batch_size 2 ^
  --lr 2e-4
```

After training, this file is generated:

- `finetune_outputs/tinyllama-lora/training_summary.json`

Use `epochs` from that file for form submission.

## 4) Quick inference with adapter

```bash
python finetune/infer_adapter.py ^
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --adapter_dir finetune_outputs/tinyllama-lora ^
  --prompt "Explain retrieval augmented generation."
```

## Notes

- GPU is strongly recommended.
- This module is independent from Streamlit Cloud runtime.
