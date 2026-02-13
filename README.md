# Enterprise Copilot + Analyst Multimodal RAG

An end-to-end multimodal RAG system for question answering over:
- PDF and TXT documents
- Images (OCR + vision captioning)
- Audio (speech transcription)

This version includes dense retrieval with Jina embeddings, FAISS (with fallback), model/API reranking (with fallback), grounded generation, confidence-aware guardrails, persistent local index storage, and an analyst dashboard.

## Live Demo

`https://rag-groq-app-vzx48gfqckwgxgjribnqoc.streamlit.app/`

## Key Differentiators

- Dense embeddings via Jina v4 API (`JINA_API_KEY`)
- Vector retrieval with FAISS (`faiss-cpu` where supported) + numpy fallback
- Model-based reranking via Jina API + lexical fallback
- Multi-query retrieval and modality filtering (`all`, `text`, `image`, `audio`)
- Groq Vision image understanding and Groq ASR transcription
- Confidence and refusal behavior for low-confidence contexts
- Copilot Studio + Analyst Lab split workflow
- Benchmark pack with PASS/FAIL summary
- Stage-wise latency diagnostics:
  - ingest/index
  - query embedding
  - vector search
  - rerank
  - generation
- Structured JSON/TXT report exports with evidence metadata
- Persistent local index store (`.rag_store/`) for reload across restarts

## Project Structure

```text
rag-groq-streamlit/
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── runtime.txt
├── README.md
├── tests/
│   ├── test_chunking.py
│   ├── test_eval.py
│   ├── test_ingest_txt.py
│   └── test_retrieval_rerank.py
└── rag/
    ├── __init__.py
    ├── chunking.py
    ├── ingest.py
    ├── embeddings.py
    ├── retriever.py
    ├── reranker.py
    ├── vision.py
    ├── llm.py
    ├── eval.py
    ├── store.py
    └── types.py
```

## Required Secrets

Set in Streamlit Cloud -> `Secrets`:

```toml
GROQ_API_KEY = "your_groq_key"
JINA_API_KEY = "your_jina_key"
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Architecture Flow

1. Ingest files (`rag/ingest.py`)
2. Chunk content with overlap (`rag/chunking.py`)
3. Embed chunks (`rag/embeddings.py`)
4. Build and persist retrieval index (`rag/retriever.py`, `rag/store.py`)
5. Retrieve + rerank (`rag/retriever.py`, `rag/reranker.py`)
6. Generate grounded answer (`rag/llm.py`)
7. Show diagnostics/evaluation (`rag/eval.py`, `app.py`)

## Notes

- If `JINA_API_KEY` is missing, the system falls back to local hashed embeddings and lexical rerank.
- If `GROQ_API_KEY` is missing, generation uses fallback responses with top evidence.
- `faiss-cpu` is enabled where available; numpy fallback keeps the app functional.
