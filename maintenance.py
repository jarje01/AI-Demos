"""
Unified maintenance utilities for Polymarket FX feature snapshots.

Usage:
    python maintenance.py --collect
    python maintenance.py --cleanup --keep-days 90
    python maintenance.py --collect --cleanup --interval 1h --keep-days 90
"""

from __future__ import annotations

import argparse
import logging

from data_loader import collect_polymarket_snapshot, cleanup_polymarket_snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maintenance tasks for Polymarket snapshot history")
    parser.add_argument("--collect", action="store_true", help="Collect one Polymarket snapshot row")
    parser.add_argument("--cleanup", action="store_true", help="Deduplicate/prune snapshot history")
    parser.add_argument("--interval", type=str, default="1h", help="Interval for collection timestamp alignment")
    parser.add_argument("--keep-days", type=int, default=90, help="Days of history to keep when cleaning")
    parser.add_argument("--quiet", action="store_true", help="Reduce console/log output for scheduled tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if not args.collect and not args.cleanup:
        if not args.quiet:
            print("No action selected. Use --collect and/or --cleanup.")
        return

    if args.collect:
        snap = collect_polymarket_snapshot(interval=args.interval)
        if not args.quiet:
            print("[collect] done")
            print(f"asof_timestamp: {snap.get('asof_timestamp')}")
            print(f"pm_sample_count: {snap.get('pm_sample_count')}")

    if args.cleanup:
        stats = cleanup_polymarket_snapshots(keep_days=args.keep_days)
        if not args.quiet:
            print("[cleanup] done")
            print(f"snapshot_file: {stats.get('snapshot_file')}")
            print(f"before_rows: {stats.get('before_rows')}")
            print(f"after_rows: {stats.get('after_rows')}")
            print(f"removed_rows: {stats.get('removed_rows')}")
            print(f"keep_days: {stats.get('keep_days')}")


if __name__ == "__main__":
    main()
