"""
Collect and append one Polymarket macro snapshot row.

Usage:
    python collect_polymarket_snapshot.py
    python collect_polymarket_snapshot.py --interval 1h
"""

from __future__ import annotations

import argparse

from data_loader import collect_polymarket_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one Polymarket snapshot row for FX features")
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Bucket interval for snapshot timestamp alignment (default: 1h)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snap = collect_polymarket_snapshot(interval=args.interval)
    print("Saved Polymarket snapshot")
    print(f"asof_timestamp: {snap.get('asof_timestamp')}")
    print(f"pm_sample_count: {snap.get('pm_sample_count')}")
    print("snapshot_file: data/polymarket_macro_snapshots.csv")


if __name__ == "__main__":
    main()
