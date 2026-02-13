import os
import time
from typing import List

from config import RERANK_MODEL, TOP_K_RERANK
from rag.types import RetrievedChunk

try:
    import requests
except ImportError:
    requests = None


def _overlap_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def _api_rerank(query: str, retrieved: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk] | None:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key or not retrieved or requests is None:
        return None
    docs = [r.text for r in retrieved]
    payload = {"model": RERANK_MODEL, "query": query, "documents": docs, "top_n": min(top_k, len(docs))}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post("https://api.jina.ai/v1/rerank", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        out: List[RetrievedChunk] = []
        for row in results:
            idx = row.get("index")
            score = row.get("relevance_score", 0.0)
            if idx is None or idx < 0 or idx >= len(retrieved):
                continue
            item = retrieved[idx]
            item.rerank_score = float(score)
            out.append(item)
        return out[: max(1, min(top_k, len(out)))] if out else None
    except Exception:
        return None


def _fallback_rerank(query: str, retrieved: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
    rescored: List[RetrievedChunk] = []
    for item in retrieved:
        lexical = _overlap_score(query, item.text)
        item.rerank_score = float((0.7 * item.score) + (0.3 * lexical))
        rescored.append(item)

    rescored.sort(key=lambda x: x.rerank_score, reverse=True)
    return rescored[: max(1, min(top_k, len(rescored)))]


def rerank(query: str, retrieved: List[RetrievedChunk], top_k: int = TOP_K_RERANK, with_timings: bool = False):
    t0 = time.perf_counter()
    api_result = _api_rerank(query, retrieved, top_k)
    if api_result is not None:
        out = api_result
        mode = "jina_api"
    else:
        out = _fallback_rerank(query, retrieved, top_k)
        mode = "lexical_fallback"
    t_ms = round((time.perf_counter() - t0) * 1000, 2)
    if with_timings:
        return out, {"rerank_ms": t_ms, "rerank_mode": mode}
    return out
