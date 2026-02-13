# Enterprise Copilot + Analyst Multimodal RAG

An end-to-end multimodal RAG system for question answering over:
- PDF text
- Images (OCR + vision captioning)
- Audio (speech transcription)

This version includes dense retrieval with Jina embeddings, FAISS (with fallback), reranking, grounded generation, confidence-aware guardrails, and an analyst dashboard.

## Live Demo

`https://rag-groq-app-vzx48gfqckwgxgjribnqoc.streamlit.app/`

<<<<<<< HEAD
`https://rag-groq-app-vzx48gfqckwgxgjribnqoc.streamlit.app/`
=======
## Key Differentiators
>>>>>>> 8b75768 (Implement enterprise copilot + analyst multimodal RAG architecture)

- Dense embeddings via Jina v4 API (`JINA_API_KEY`)
- Vector retrieval with FAISS (`faiss-cpu` when supported), numpy fallback otherwise
- Multi-query retrieval and modality filtering (`all`, `text`, `image`, `audio`)
- Reranking with retrieval + lexical hybrid score
- Groq Vision image understanding and Groq ASR transcription
- Confidence and refusal behavior for low-confidence contexts
- Copilot Studio + Analyst Lab split workflow
- Benchmark pack with PASS/FAIL summary
- Structured JSON/TXT report exports with evidence metadata

## Project Structure

```text
rag-groq-streamlit/
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── runtime.txt
├── README.md
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

## Architecture Flow

1. Ingest files (`rag/ingest.py`)
2. Chunk content with overlap (`rag/chunking.py`)
3. Embed chunks (`rag/embeddings.py`)
4. Build retrieval index (`rag/retriever.py`)
5. Retrieve + rerank (`rag/retriever.py`, `rag/reranker.py`)
6. Generate grounded answer (`rag/llm.py`)
7. Show diagnostics/evaluation (`rag/eval.py`, `app.py`)

## Notes

- If `JINA_API_KEY` is missing, the system falls back to local hashed embeddings.
- If `GROQ_API_KEY` is missing, generation uses a fallback response with top evidence.
- `faiss-cpu` is enabled where available; a numpy-based fallback keeps app functional.
