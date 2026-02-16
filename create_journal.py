"""
Create a dated paper-trading journal entry from a template.

Usage:
    python create_journal.py
    python create_journal.py --date 2026-02-12
    python create_journal.py --dir journals --force
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path


def build_template(entry_date: str) -> str:
    return f"""# Paper Trading Journal — {entry_date}

Date: {entry_date}

Market context (calm / trending / choppy):

Signal (Long / Short / Flat):

Entry price (paper):

Stop-loss:

Take-profit:

Position size rule used:

Outcome (Win/Loss, +R/-R):

Mistake made (if any):

One improvement for tomorrow:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a daily paper-trading journal file")
    parser.add_argument(
        "--date",
        help="Journal date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--dir",
        default="journals",
        help="Output directory for journal files (default: journals)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite file if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.date:
        try:
            journal_day = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("[ERROR] --date must be in YYYY-MM-DD format")
            return 1
    else:
        journal_day = date.today()

    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{journal_day.isoformat()}_journal.md"
    output_path = output_dir / file_name

    if output_path.exists() and not args.force:
        print(f"[INFO] Journal already exists: {output_path}")
        print("Use --force to overwrite.")
        return 0

    output_path.write_text(build_template(journal_day.isoformat()), encoding="utf-8")
    print(f"[OK] Created journal: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
