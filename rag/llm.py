import os
from typing import Dict, List

from config import LLM_MODEL, LOW_CONFIDENCE_THRESHOLD
from rag.types import AnswerResult, RetrievedChunk

try:
    from groq import Groq
except ImportError:
    Groq = None


def _client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return None
    return Groq(api_key=api_key)


def _confidence_from_scores(retrieved: List[RetrievedChunk]) -> float:
    if not retrieved:
        return 0.0
    values = [max(0.0, r.rerank_score) for r in retrieved]
    top = max(values)
    avg = sum(values) / len(values)
    spread = top - min(values)
    conf = (0.5 * top) + (0.35 * avg) + (0.15 * min(1.0, spread + 0.2))
    return max(0.0, min(1.0, conf))


def answer(
    query: str,
    contexts: List[RetrievedChunk],
    mode: str = "detailed",
    history: List[Dict[str, str]] | None = None,
    latency_ms: Dict[str, float] | None = None,
) -> AnswerResult:
    confidence = _confidence_from_scores(contexts)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        text = (
            "Insufficient context confidence to answer reliably.\n"
            "Please refine the question or upload more relevant documents."
        )
        return AnswerResult(
            answer=text,
            citations=[c.chunk_id for c in contexts],
            latency_ms=latency_ms or {},
            confidence=confidence,
            model_id=LLM_MODEL,
        )

    style = {
        "concise": "Give a concise answer in 3-5 sentences.",
        "detailed": "Give a structured detailed answer with short headings and bullets when useful.",
        "executive": "Give an executive summary with decisions, risks, and actions.",
    }.get(mode, "Give a structured detailed answer.")

    context_text = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in contexts])
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n\n".join([f"User: {h.get('question','')}\nAssistant: {h.get('answer','')}" for h in recent])

    prompt = (
        "You are an enterprise RAG assistant.\n"
        "Use only the provided context. If missing, say insufficient context.\n\n"
        f"{style}\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}"
    )

    client = _client()
    if client is None:
        fallback = contexts[0].text if contexts else "No context available."
        return AnswerResult(
            answer=f"LLM unavailable (set GROQ_API_KEY). Top context:\n{fallback}",
            citations=[c.chunk_id for c in contexts],
            latency_ms=latency_ms or {},
            confidence=confidence,
            model_id=LLM_MODEL,
        )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return AnswerResult(
        answer=(resp.choices[0].message.content or "").strip(),
        citations=[c.chunk_id for c in contexts],
        latency_ms=latency_ms or {},
        confidence=confidence,
        model_id=LLM_MODEL,
    )


def summarize(chunks: List[str]) -> str:
    if not chunks:
        return "No content available for summary."
    client = _client()
    if client is None:
        return "Summary requires GROQ_API_KEY."
    content = "\n\n".join(chunks[:4])
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Summarize in 5 bullets and one-line abstract.\n\n" + content,
            }
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_eval_set(chunks: List[str], n: int = 5) -> List[str]:
    if not chunks:
        return []
    client = _client()
    if client is None:
        return [
            "What is the main topic?",
            "What are the key points?",
            "What process is explained?",
            "What limitations are stated?",
            "What conclusion is supported?",
        ][:n]
    content = "\n\n".join(chunks[:5])
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Generate {n} evaluation questions answerable from this context. One per line.\n\n{content}",
            }
        ],
        temperature=0.2,
    )
    lines = [ln.strip("- ").strip() for ln in (resp.choices[0].message.content or "").splitlines()]
    return [ln for ln in lines if ln][:n]
