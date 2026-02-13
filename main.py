import time
from typing import Dict, List

from config import INDEX_STORE_DIR, TOP_K_RETRIEVE, TOP_K_RERANK
from rag.eval import retrieval_metrics
from rag.ingest import ingest_files as _ingest_files
from rag.llm import answer as _answer
from rag.llm import generate_eval_set as _generate_eval_set
from rag.llm import summarize as _summarize
from rag.reranker import rerank as _rerank
from rag.retriever import build_index as _build_index
from rag.retriever import retrieve as _retrieve
from rag.store import load_retrieval_index, save_retrieval_index
from rag.types import AnswerResult, DocumentChunk, RetrievalIndex, RetrievedChunk


_ACTIVE_INDEX: RetrievalIndex | None = None


def ingest_files(files) -> List[DocumentChunk]:
    chunks, _, _ = _ingest_files(files)
    return chunks


def build_index(chunks: List[DocumentChunk], with_latency: bool = False):
    global _ACTIVE_INDEX
    _ACTIVE_INDEX, idx_latency = _build_index(chunks, with_timings=True)
    save_retrieval_index(_ACTIVE_INDEX, INDEX_STORE_DIR)
    if with_latency:
        return _ACTIVE_INDEX, idx_latency
    return _ACTIVE_INDEX


def load_persisted_index() -> RetrievalIndex | None:
    global _ACTIVE_INDEX
    loaded = load_retrieval_index(INDEX_STORE_DIR)
    if loaded is not None:
        _ACTIVE_INDEX = loaded
    return _ACTIVE_INDEX


def retrieve(query: str, filters: str = "all", top_k: int = TOP_K_RETRIEVE) -> List[RetrievedChunk]:
    if _ACTIVE_INDEX is None:
        return []
    return _retrieve(query, _ACTIVE_INDEX, filters=filters, top_k=top_k, multi_query=True)


def rerank(query: str, retrieved: List[RetrievedChunk]) -> List[RetrievedChunk]:
    return _rerank(query, retrieved, top_k=TOP_K_RERANK)


def answer(
    query: str,
    contexts: List[RetrievedChunk],
    mode: str = "detailed",
    history: List[Dict[str, str]] | None = None,
    latency_ms: Dict[str, float] | None = None,
) -> AnswerResult:
    return _answer(query, contexts, mode=mode, history=history, latency_ms=latency_ms)


def summarize(chunks: List[DocumentChunk]) -> str:
    return _summarize([c.text for c in chunks])


def generate_eval_set(chunks: List[DocumentChunk], n: int = 5) -> List[str]:
    return _generate_eval_set([c.text for c in chunks], n=n)


def run_query(
    query: str,
    filters: str = "all",
    top_k: int = TOP_K_RETRIEVE,
    mode: str = "detailed",
    history: List[Dict[str, str]] | None = None,
) -> tuple[AnswerResult, List[RetrievedChunk], Dict[str, float]]:
    t0 = time.perf_counter()
    if _ACTIVE_INDEX is None:
        empty = AnswerResult(
            answer="No active index. Upload and process files first.",
            citations=[],
            latency_ms={},
            confidence=0.0,
            model_id="none",
        )
        return empty, [], {"total_query_ms": 0.0}

    retrieved, retrieve_latency = _retrieve(
        query,
        _ACTIVE_INDEX,
        filters=filters,
        top_k=top_k,
        multi_query=True,
        with_timings=True,
    )
    reranked, rerank_latency = _rerank(query, retrieved, top_k=TOP_K_RERANK, with_timings=True)

    latency = {}
    latency.update(retrieve_latency)
    latency.update(rerank_latency)
    ans = answer(query, reranked, mode=mode, history=history, latency_ms=latency)
    latency.update(ans.latency_ms)
    latency["total_query_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    metrics = retrieval_metrics(reranked)
    latency.update({f"metric_{k}": v for k, v in metrics.items()})
    return ans, reranked, latency

