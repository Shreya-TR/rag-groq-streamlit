# Enterprise Copilot + Analyst Multimodal RAG

Multimodal RAG application with:
- PDF/TXT ingestion
- Image OCR + vision captioning
- Audio transcription
- Dense retrieval (Jina embeddings) + FAISS fallback
- Reranking (Jina API + lexical fallback)
- Grounded answer generation + confidence guardrails
- Copilot Studio + Analyst Lab UI
- Persistent local index store (`.rag_store/`)

## Live Demo

`https://rag-groq-app-vzx48gfqckwgxgjribnqoc.streamlit.app/`

## Required Secrets

```toml
GROQ_API_KEY = "your_groq_key"
JINA_API_KEY = "your_jina_key"
```

## Run App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Fine-Tuning (Real Training Pipeline)

This repo now includes a real LoRA SFT pipeline under `finetune/`.

1) Install fine-tuning dependencies:

```bash
pip install -r requirements-finetune.txt
```

2) Use sample data or prepare your own:

```bash
python finetune/prepare_sft_data.py --input_csv data/qa.csv --output_jsonl finetune/data/train.jsonl
```

3) Train adapter (example: 2 epochs):

```bash
python finetune/train_lora.py \
  --train_file finetune/data/train.jsonl \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_dir finetune_outputs/tinyllama-lora \
  --epochs 2 \
  --batch_size 2 \
  --lr 2e-4
```

4) Epoch proof for form submission:
- `finetune_outputs/tinyllama-lora/training_summary.json`
- Use `epochs` from that file as your real trained epoch count.

## Project Structure

```text
rag-groq-streamlit/
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── requirements-dev.txt
├── requirements-finetune.txt
├── runtime.txt
├── README.md
├── finetune/
│   ├── __init__.py
│   ├── README.md
│   ├── prepare_sft_data.py
│   ├── train_lora.py
│   ├── infer_adapter.py
│   └── data/
│       └── sample_train.jsonl
├── rag/
│   ├── __init__.py
│   ├── chunking.py
│   ├── ingest.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── vision.py
│   ├── llm.py
│   ├── eval.py
│   ├── store.py
│   └── types.py
└── tests/
    ├── test_chunking.py
    ├── test_eval.py
    ├── test_finetune_prep.py
    ├── test_ingest_txt.py
    ├── test_retrieval_rerank.py
    └── test_store.py
```
