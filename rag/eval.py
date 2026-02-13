from typing import Dict, List

from rag.types import RetrievedChunk


def retrieval_metrics(chunks: List[RetrievedChunk]) -> Dict[str, float]:
    if not chunks:
        return {"avg_score": 0.0, "score_spread": 0.0, "coverage": 0.0}
    scores = [c.rerank_score for c in chunks]
    avg_score = sum(scores) / len(scores)
    spread = max(scores) - min(scores)
    coverage = min(1.0, len(chunks) / 8.0)
    return {"avg_score": round(avg_score, 3), "score_spread": round(spread, 3), "coverage": round(coverage, 3)}


def pass_fail(confidence: float, threshold: float = 0.3) -> str:
    return "PASS" if confidence >= threshold else "FAIL"
