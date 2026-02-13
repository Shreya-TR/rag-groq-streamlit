from rag.eval import pass_fail, retrieval_metrics
from rag.types import RetrievedChunk


def test_pass_fail_threshold():
    assert pass_fail(0.8, threshold=0.5) == "PASS"
    assert pass_fail(0.2, threshold=0.5) == "FAIL"


def test_retrieval_metrics_shape():
    chunks = [
        RetrievedChunk("c1", "a", 0.8, 0.7, "text", "x"),
        RetrievedChunk("c2", "b", 0.6, 0.5, "text", "x"),
    ]
    m = retrieval_metrics(chunks)
    assert set(m.keys()) == {"avg_score", "score_spread", "coverage"}
    assert 0 <= m["coverage"] <= 1
