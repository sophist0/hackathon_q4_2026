"""Shuffle ground-truth prompts and number them in the new order.

Reads ``data/classified_muse_spark_prompts_4.csv``, shuffles the data rows,
prepends a ``Row`` column (1-based position in the shuffled file), and
writes ``data/shuffled_prompts_4.csv`` in the same pipe-delimited layout.
"""

import csv
import random
from pathlib import Path

SRC = Path("data/classified_muse_spark_prompts_4.csv")
DST = Path("data/shuffled_prompts_4.csv")


def main() -> None:
    with open(SRC, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]
    random.shuffle(rows)
    with open(DST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="|", fieldnames=["Row", *fieldnames])
        writer.writeheader()
        for number, row in enumerate(rows, start=1):
            writer.writerow({"Row": number, **row})
    print(f"wrote {len(rows)} rows to {DST}")


if __name__ == "__main__":
    main()
