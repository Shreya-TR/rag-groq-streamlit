import pytest

pytest.importorskip("numpy")

from rag.reranker import rerank
from rag.retriever import build_index, retrieve
from rag.types import DocumentChunk


def _chunks():
    return [
        DocumentChunk("c1", "python data science tutorial", "text", "a.txt"),
        DocumentChunk("c2", "sales chart trend increases yearly", "image", "img.png"),
        DocumentChunk("c3", "meeting transcript project timeline", "audio", "audio.wav"),
    ]


def test_retrieve_with_filter():
    idx = build_index(_chunks())
    out = retrieve("sales trend", idx, filters="image", top_k=3, multi_query=False)
    assert len(out) >= 1
    assert all(x.modality == "image" for x in out)


def test_rerank_produces_scores():
    idx = build_index(_chunks())
    retrieved = retrieve("project timeline", idx, filters="all", top_k=3, multi_query=False)
    out = rerank("project timeline", retrieved, top_k=2)
    assert len(out) >= 1
    assert all(hasattr(x, "rerank_score") for x in out)
