from rag.ingest import ingest_files


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


def test_ingest_txt_file_creates_text_chunks():
    file = FakeUpload("notes.txt", b"hello world " * 200)
    chunks, latency, stats = ingest_files([file])
    assert len(chunks) > 0
    assert all(c.modality == "text" for c in chunks)
    assert stats["text"] > 0
    assert "ingest_ms" in latency
