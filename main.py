import os
import tempfile
from typing import Any, List, Tuple

import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

try:
    from groq import Groq
except ImportError:
    Groq = None

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def extract_pdf_text(path: str) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_image_text(path: str) -> str:
    return pytesseract.image_to_string(Image.open(path))


def extract_audio_text(path: str, model_size: str = "base") -> str:
    import whisper

    model = whisper.load_model(model_size)
    return model.transcribe(path).get("text", "").strip()


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 70) -> List[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    step = max(1, chunk_size - overlap)
    return [
        " ".join(words[i : i + chunk_size]).strip()
        for i in range(0, len(words), step)
        if " ".join(words[i : i + chunk_size]).strip()
    ]


def build_faiss_index(chunks: List[str]) -> Tuple[Any, List[str]]:
    import faiss

    if not chunks:
        raise ValueError("No chunks found to index.")
    embeddings = _get_embedder().encode(chunks, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def retrieve(query: str, index: Any, chunks: List[str], top_k: int = 3) -> List[str]:
    q_emb = _get_embedder().encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    top_k = min(top_k, len(chunks))
    _, idxs = index.search(q_emb, top_k)
    return [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]


def _resolve_client_and_model():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and Groq is not None:
        return Groq(api_key=api_key), "llama-3.3-70b-versatile", "groq"
    return None, None, None


def generate_answer(query: str, index: Any, chunks: List[str]) -> str:
    contexts = retrieve(query, index, chunks, top_k=4)
    if not contexts:
        return "I could not find relevant context for your question."

    context = "\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant for a RAG app.\n"
        "Answer only from the provided context. If context is insufficient, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    client, model, provider = _resolve_client_and_model()
    if provider == "groq":
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    return (
        "LLM is not configured. Set GROQ_API_KEY in environment/secrets to get generated answers.\n\n"
        "Top retrieved context:\n"
        f"{contexts[0]}"
    )


def save_uploaded_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name
