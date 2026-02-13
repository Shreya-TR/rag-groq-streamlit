from typing import Dict, List

import numpy as np

from config import TOP_K_RETRIEVE
from rag.embeddings import embed_texts
from rag.types import DocumentChunk, RetrievedChunk, RetrievalIndex

try:
    import faiss
except ImportError:
    faiss = None


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return (v / norms).astype(np.float32)


def build_index(chunks: List[DocumentChunk]) -> RetrievalIndex:
    if not chunks:
        raise ValueError("No chunks to index.")

    texts = [c.text for c in chunks]
    emb = _normalize(embed_texts(texts))
    if faiss is not None:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        index_kind = "faiss_ip"
    else:
        index = None
        index_kind = "numpy_cosine"

    return RetrievalIndex(
        chunks=chunks,
        embeddings=emb,
        vector_index=index,
        embedding_model="jina-embeddings-v4_or_fallback",
        index_kind=index_kind,
    )


def _query_variants(query: str) -> List[str]:
    base = " ".join(query.strip().split())
    if not base:
        return []
    return [base, f"{base} definition", f"key points about {base}"]


def _search_once(query_vec: np.ndarray, retrieval_index: RetrievalIndex, top_k: int) -> List[tuple]:
    if retrieval_index.vector_index is not None:
        scores, idxs = retrieval_index.vector_index.search(query_vec, min(top_k, len(retrieval_index.chunks)))
        return list(zip(scores[0].tolist(), idxs[0].tolist()))

    sim = retrieval_index.embeddings @ query_vec[0]
    rank = np.argsort(-sim)[: min(top_k, len(retrieval_index.chunks))]
    return [(float(sim[i]), int(i)) for i in rank.tolist()]


def retrieve(
    query: str,
    retrieval_index: RetrievalIndex,
    filters: str = "all",
    top_k: int = TOP_K_RETRIEVE,
    multi_query: bool = True,
) -> List[RetrievedChunk]:
    variants = _query_variants(query) if multi_query else [query]
    best: Dict[int, RetrievedChunk] = {}

    for v in variants:
        q_emb = _normalize(embed_texts([v]))
        for score, idx in _search_once(q_emb, retrieval_index, top_k):
            if idx < 0 or idx >= len(retrieval_index.chunks):
                continue
            chunk = retrieval_index.chunks[idx]
            if filters != "all" and chunk.modality != filters:
                continue
            rc = RetrievedChunk(
                chunk_id=chunk.id,
                text=chunk.text,
                score=float(score),
                rerank_score=float(score),
                modality=chunk.modality,
                source_name=chunk.source_name,
                metadata=chunk.metadata,
            )
            prev = best.get(idx)
            if prev is None or rc.score > prev.score:
                best[idx] = rc

    ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)
    return ranked[: max(1, min(top_k, len(ranked)))]
