# Advanced Multimodal RAG (Streamlit + Groq)

An end-to-end multimodal RAG app that can:
- Extract text from PDFs
- OCR text from images
- Transcribe audio files
- Chunk with overlap for better retrieval context
- Retrieve top relevant chunks
- Generate grounded answers with Groq LLM

## Project Structure

```text
rag-groq-streamlit/
├── app.py
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## Key Improvements

- Deployment-safe architecture for Streamlit Cloud
- Python 3.13-friendly dependency set
- Robust fallback when `GROQ_API_KEY` is missing
- Adjustable chunk size and overlap
- Retrieved source chunk viewer
- Downloadable Q&A report
- Cleaner submission-ready UI and metrics cards

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variable:
   ```bash
   GROQ_API_KEY=your_key_here
   ```

3. Run:
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Notes

- Main file path: `app.py`
- Add secret:
  - `GROQ_API_KEY = "your_key_here"`
- Reboot app after updating dependencies/secrets.

## Supported Input Types

- PDF: `.pdf`
- Images: `.png`, `.jpg`, `.jpeg`
- Audio: `.wav`, `.mp3`, `.m4a`

## Pipeline

1. Ingest files
2. Extract text (PDF/OCR/audio transcription)
3. Overlap chunking
4. Token-based retrieval index build
5. Retrieve top chunks for a query
6. Generate answer grounded in retrieved context
