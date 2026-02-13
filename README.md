# Enterprise Multimodal RAG (Text + Vision + Audio)

This project is an end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system that enables question answering over:

- Text documents (PDF)
- Images (charts, screenshots, scanned notes)
- Audio files (speech transcription)

It combines:

- Context retrieval over chunked multimodal text
- Image understanding via OCR
- Audio understanding via Groq transcription
- Grounded answer generation using Groq LLMs
- A modern Streamlit user interface

---

## Live Demo

The application is deployed and accessible here:

`https://rag-groq-app-vzx48gfqckwgxgjribnqoc.streamlit.app/`

---

## Key Features

- Text-based RAG over uploaded PDF files
- Multimodal ingestion with image OCR and audio transcription
- Overlapping chunking for stronger retrieval context
- Multi-query retrieval (query expansion + merge)
- Retrieval scoring with source citations (`chunk id + score`)
- Grounded answer generation (context-first prompting)
- Document auto-summary after ingestion
- Session memory with recent chat history
- Evaluation tab with auto-generated benchmark questions
- Downloadable Q&A report from the UI
- Safe fallback behavior when `GROQ_API_KEY` is not set

---

## Architecture Overview

1. User uploads PDF/image/audio files
2. Text is extracted (PDF parser, OCR, audio transcription)
3. Text is chunked with overlap
4. Sparse retrieval index is built from chunks
5. User query is expanded into multiple retrieval variants
6. Top chunks are retrieved and scored
7. Groq LLM generates a context-grounded answer
8. UI shows answer, citations, and quality indicators

---

## Project Structure

```bash
rag-groq-streamlit/
├── app.py
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variable:

```bash
GROQ_API_KEY=your_key_here
```

3. Run the app:

```bash
streamlit run app.py
```

---

## Streamlit Cloud Configuration

- Main file path: `app.py`
- Add secret:

```toml
GROQ_API_KEY = "your_key_here"
```

- Reboot app after updating secrets or dependencies.

---

## Supported File Types

- PDF: `.pdf`
- Images: `.png`, `.jpg`, `.jpeg`
- Audio: `.wav`, `.mp3`, `.m4a`

---

## Notes

- If `GROQ_API_KEY` is missing, retrieval still runs and the app provides fallback responses.
- For best results, upload content-rich files and use domain-specific questions.
