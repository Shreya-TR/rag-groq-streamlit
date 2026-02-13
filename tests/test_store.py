import os
import tempfile

import pytest

pytest.importorskip("numpy")

from rag.retriever import build_index
from rag.store import load_retrieval_index, save_retrieval_index
from rag.types import DocumentChunk


def test_save_and_load_retrieval_index():
    chunks = [
        DocumentChunk("c1", "alpha beta gamma", "text", "t1.txt"),
        DocumentChunk("c2", "delta epsilon zeta", "text", "t2.txt"),
    ]
    idx = build_index(chunks)
    with tempfile.TemporaryDirectory() as tmp:
        save_retrieval_index(idx, tmp)
        loaded = load_retrieval_index(tmp)
        assert loaded is not None
        assert len(loaded.chunks) == 2
        assert os.path.exists(os.path.join(tmp, "embeddings.npy"))
