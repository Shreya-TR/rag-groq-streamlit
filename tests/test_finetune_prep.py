import tempfile
from pathlib import Path

from finetune.prepare_sft_data import convert_csv_to_jsonl


def test_convert_csv_to_jsonl():
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "qa.csv"
        out_path = Path(td) / "train.jsonl"
        csv_path.write_text("question,answer\nWhat is RAG?,Retrieval Augmented Generation\n", encoding="utf-8")
        count = convert_csv_to_jsonl(str(csv_path), str(out_path))
        assert count == 1
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
