import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np

from rag.types import DocumentChunk, RetrievalIndex

try:
    import faiss
except ImportError:
    faiss = None


def _paths(store_dir: str):
    return {
        "meta": os.path.join(store_dir, "meta.json"),
        "chunks": os.path.join(store_dir, "chunks.json"),
        "embeddings": os.path.join(store_dir, "embeddings.npy"),
        "faiss": os.path.join(store_dir, "index.faiss"),
    }


def save_retrieval_index(retrieval_index: RetrievalIndex, store_dir: str) -> None:
    os.makedirs(store_dir, exist_ok=True)
    p = _paths(store_dir)

    meta = {
        "embedding_model": retrieval_index.embedding_model,
        "index_kind": retrieval_index.index_kind,
        "chunk_count": len(retrieval_index.chunks),
    }
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(p["chunks"], "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in retrieval_index.chunks], f, ensure_ascii=False, indent=2)

    np.save(p["embeddings"], retrieval_index.embeddings)

    if faiss is not None and retrieval_index.vector_index is not None and retrieval_index.index_kind.startswith("faiss"):
        faiss.write_index(retrieval_index.vector_index, p["faiss"])


def load_retrieval_index(store_dir: str) -> Optional[RetrievalIndex]:
    p = _paths(store_dir)
    if not (os.path.exists(p["meta"]) and os.path.exists(p["chunks"]) and os.path.exists(p["embeddings"])):
        return None

    with open(p["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(p["chunks"], "r", encoding="utf-8") as f:
        chunk_rows = json.load(f)

    chunks = [DocumentChunk(**row) for row in chunk_rows]
    embeddings = np.load(p["embeddings"]).astype(np.float32)
    index_kind = meta.get("index_kind", "numpy_cosine")

    vector_index = None
    if faiss is not None and index_kind.startswith("faiss") and os.path.exists(p["faiss"]):
        vector_index = faiss.read_index(p["faiss"])

    return RetrievalIndex(
        chunks=chunks,
        embeddings=embeddings,
        vector_index=vector_index,
        embedding_model=meta.get("embedding_model", "unknown"),
        index_kind=index_kind,
    )
