"""
Deduplicate and prune historical Polymarket snapshot rows.

Usage:
    python cleanup_polymarket_snapshots.py
    python cleanup_polymarket_snapshots.py --keep-days 90
"""

from __future__ import annotations

import argparse

from data_loader import cleanup_polymarket_snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Polymarket snapshot history file")
    parser.add_argument(
        "--keep-days",
        type=int,
        default=90,
        help="Keep only the most recent N days of snapshots (default: 90)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = cleanup_polymarket_snapshots(keep_days=args.keep_days)
    print("Polymarket snapshot cleanup complete")
    print(f"snapshot_file: {stats.get('snapshot_file')}")
    print(f"before_rows: {stats.get('before_rows')}")
    print(f"after_rows: {stats.get('after_rows')}")
    print(f"removed_rows: {stats.get('removed_rows')}")
    print(f"keep_days: {stats.get('keep_days')}")


if __name__ == "__main__":
    main()
