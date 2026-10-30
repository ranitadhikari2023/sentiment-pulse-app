"""
Sentiment Pulse - CSV Batch Splitter Utility (split_csv.py)
Splits large CSV files (e.g. 1,000,000+ records) into RFC 4180 compliant batches
of at most 5,000 records each, preserving original headers and quoted fields.

Usage:
    python scripts/split_csv.py <input_file.csv> [--batch-size 5000] [--output-dir ./batches]
"""

import os
import sys
import csv
import argparse
from typing import Optional


def detect_encoding(file_path: str) -> str:
    """Detect encoding with UTF-8 BOM / Latin-1 fallbacks."""
    with open(file_path, "rb") as f:
        raw_prefix = f.read(4)
    if raw_prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                for _ in range(20):
                    f.readline()
            return enc
        except (UnicodeDecodeError, Exception):
            continue
    return "utf-8"


def detect_delimiter(file_path: str, encoding: str) -> str:
    """Detect comma, semicolon, tab, or pipe delimiter."""
    candidates = [",", ";", "\t", "|"]
    with open(file_path, "r", encoding=encoding, newline="") as f:
        sample = "".join([f.readline() for _ in range(10)])

    if not sample.strip():
        return ","

    counts = {delim: sample.count(delim) for delim in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def split_csv_file(
    input_path: str,
    batch_size: int = 5000,
    output_dir: Optional[str] = None
) -> int:
    """
    Splits input CSV into batches of at most batch_size records.
    Returns the total number of batches created.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    encoding = detect_encoding(input_path)
    delimiter = detect_delimiter(input_path, encoding)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_path) or ".", f"{base_name}_batches")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening: {input_path} (encoding: {encoding}, delimiter: {repr(delimiter)})")
    print(f"Batch limit: {batch_size:,} records per file")
    print(f"Output directory: {output_dir}")

    batch_idx = 1
    total_records = 0
    current_batch_rows = []

    with open(input_path, "r", encoding=encoding, newline="") as infile:
        reader = csv.reader(infile, delimiter=delimiter)

        # Read header
        header_row = None
        for row in reader:
            if row and any(cell.strip() for cell in row):
                header_row = row
                break

        if not header_row:
            print("Error: Input file contains no header or is empty.")
            return 0

        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue

            current_batch_rows.append(row)
            total_records += 1

            if len(current_batch_rows) >= batch_size:
                _write_batch(output_dir, base_name, batch_idx, header_row, current_batch_rows)
                print(f"  Batch {batch_idx:04d}: {len(current_batch_rows):,} records")
                batch_idx += 1
                current_batch_rows = []

        if current_batch_rows:
            _write_batch(output_dir, base_name, batch_idx, header_row, current_batch_rows)
            print(f"  Batch {batch_idx:04d}: {len(current_batch_rows):,} records")
            batch_count = batch_idx
        else:
            batch_count = batch_idx - 1

    print("\nSplit Complete:")
    print(f"  Total records: {total_records:,}")
    print(f"  Batches created: {batch_count}")
    print(f"  Directory: {output_dir}")
    return batch_count


def _write_batch(out_dir: str, base_name: str, batch_num: int, header: list, rows: list) -> str:
    filename = f"{base_name}_batch_{batch_num:04d}.csv"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large CSV into 5,000-record batches for Sentiment Pulse.")
    parser.add_argument("input_file", help="Path to the large CSV file")
    parser.add_argument("--batch-size", type=int, default=5000, help="Max records per batch (default: 5000)")
    parser.add_argument("--output-dir", help="Output directory for generated batches")

    args = parser.parse_args()
    try:
        split_csv_file(args.input_file, args.batch_size, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

