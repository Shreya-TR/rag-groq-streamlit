import os
import re
import tempfile
from collections import Counter
from typing import Any, Dict, List, Tuple

import pdfplumber
import pytesseract
from PIL import Image

try:
    from groq import Groq
except ImportError:
    Groq = None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _build_sparse_vectors(chunks: List[str]) -> Tuple[List[Dict[str, int]], List[str]]:
    vectors: List[Dict[str, int]] = []
    for chunk in chunks:
        tokens = _tokenize(chunk)
        vectors.append(dict(Counter(tokens)))
    return vectors, chunks


def _cosine_dict(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)
    mag_a = sum(v * v for v in a.values()) ** 0.5
    mag_b = sum(v * v for v in b.values()) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


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


def extract_audio_text(path: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return "Audio transcription unavailable (set GROQ_API_KEY)."

    client = Groq(api_key=api_key)
    with open(path, "rb") as audio_file:
        tr = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
        )

    text = getattr(tr, "text", "") or ""
    return text.strip()


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 70) -> List[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_faiss_index(chunks: List[str]) -> Tuple[Any, List[str]]:
    if not chunks:
        raise ValueError("No chunks found to index.")
    # Kept function name for app compatibility; internally this is a sparse-token index.
    return _build_sparse_vectors(chunks)


def retrieve_with_scores(query: str, index: Any, chunks: List[str], top_k: int = 3) -> List[Tuple[str, float, int]]:
    query_vec = dict(Counter(_tokenize(query)))
    scored: List[Tuple[float, int]] = []

    for i, chunk_vec in enumerate(index):
        score = _cosine_dict(query_vec, chunk_vec)
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    take_n = max(1, min(top_k, len(scored)))
    top_scored = [(score, idx) for score, idx in scored[:take_n] if score > 0]

    if not top_scored:
        fallback = []
        for idx in range(min(top_k, len(chunks))):
            fallback.append((chunks[idx], 0.0, idx))
        return fallback

    return [(chunks[idx], float(score), idx) for score, idx in top_scored]


def retrieve(query: str, index: Any, chunks: List[str], top_k: int = 3) -> List[str]:
    return [text for text, _, _ in retrieve_with_scores(query, index, chunks, top_k=top_k)]


def _resolve_client_and_model():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and Groq is not None:
        return Groq(api_key=api_key), "llama-3.3-70b-versatile", "groq"
    return None, None, None


def generate_answer(query: str, index: Any, chunks: List[str], history: List[Dict[str, str]] | None = None) -> str:
    contexts = retrieve(query, index, chunks, top_k=4)
    if not contexts:
        return "I could not find relevant context for your question."

    context = "\n\n".join(contexts)
    history_text = ""
    if history:
        last_turns = history[-4:]
        formatted = []
        for turn in last_turns:
            user_q = turn.get("question", "").strip()
            assistant_a = turn.get("answer", "").strip()
            if user_q or assistant_a:
                formatted.append(f"User: {user_q}\nAssistant: {assistant_a}")
        if formatted:
            history_text = "Conversation so far:\n" + "\n\n".join(formatted) + "\n\n"

    prompt = (
        "You are a helpful assistant for a RAG app.\n"
        "Answer only from the provided context. If context is insufficient, say so clearly.\n\n"
        f"{history_text}"
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


def summarize_document(chunks: List[str]) -> str:
    if not chunks:
        return "No content available for summary."

    preview = "\n\n".join(chunks[:4])
    client, model, provider = _resolve_client_and_model()
    if provider == "groq":
        prompt = (
            "Summarize this document in 5 concise bullets and then provide a 1-line abstract.\n\n"
            f"Content:\n{preview}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    return "Summary requires GROQ_API_KEY. Set it in Streamlit secrets to enable document summarization."


def save_uploaded_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name
