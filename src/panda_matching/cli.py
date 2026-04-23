"""Command-line interface for panda_matching."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from panda_matching.ingest.io import load_records
from panda_matching.ingest.pipeline import run_import


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panda-matching")
    sub = parser.add_subparsers(dest="command", required=False)

    import_parser = sub.add_parser("import", help="Import JSON/JSONL records into Postgres")
    import_parser.add_argument("--source", required=True, help="Data source name, e.g. pandas_api")
    import_parser.add_argument("--file", required=True, help="Path to .json/.jsonl/.ndjson file")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "import":
        path = Path(args.file)
        records = load_records(path)
        run_id = run_import(source=args.source, records=records)
        print(f"Import completed. run_id={run_id}")
        return

    print("panda_matching is set up. Use: panda-matching import --source ... --file ...")


if __name__ == "__main__":
    main()
