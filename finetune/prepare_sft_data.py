import argparse
import csv
import json
import os
from typing import Dict, Iterable


def _row_to_record(row: Dict[str, str]) -> Dict[str, str]:
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    return {
        "instruction": "Answer the question using clear and accurate explanation.",
        "input": q,
        "output": a,
    }


def convert_csv_to_jsonl(input_csv: str, output_jsonl: str) -> int:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    count = 0
    with open(input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        output_jsonl, "w", encoding="utf-8"
    ) as f_out:
        reader: Iterable[Dict[str, str]] = csv.DictReader(f_in)
        for row in reader:
            rec = _row_to_record(row)
            if not rec["input"] or not rec["output"]:
                continue
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert QA CSV to SFT JSONL.")
    parser.add_argument("--input_csv", required=True, help="CSV with question,answer columns.")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    count = convert_csv_to_jsonl(args.input_csv, args.output_jsonl)
    print(f"Wrote {count} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
