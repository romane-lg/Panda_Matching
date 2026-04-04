from __future__ import annotations

import argparse
from pathlib import Path

from panda_matching.ingest.io import load_records
from panda_matching.ingest.pipeline import run_import


def main() -> None:
    parser = argparse.ArgumentParser(description="Import records into Postgres")
    parser.add_argument("--source", required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    file_path = Path(args.file)
    records = load_records(file_path)
    run_id = run_import(source=args.source, records=records)
    print(f"Import run created: {run_id}")


if __name__ == "__main__":
    main()
