from typing import List

from config import TOP_K_RERANK
from rag.types import RetrievedChunk


def _overlap_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def rerank(query: str, retrieved: List[RetrievedChunk], top_k: int = TOP_K_RERANK) -> List[RetrievedChunk]:
    rescored: List[RetrievedChunk] = []
    for item in retrieved:
        lexical = _overlap_score(query, item.text)
        item.rerank_score = float((0.7 * item.score) + (0.3 * lexical))
        rescored.append(item)

    rescored.sort(key=lambda x: x.rerank_score, reverse=True)
    return rescored[: max(1, min(top_k, len(rescored)))]
