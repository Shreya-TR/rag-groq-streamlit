from rag.chunking import chunk_text


def test_chunking_overlap_produces_multiple_chunks():
    text = " ".join([f"w{i}" for i in range(200)])
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunking_empty_text():
    assert chunk_text("", chunk_size=50, overlap=10) == []
