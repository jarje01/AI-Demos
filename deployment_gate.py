"""
Deployment Gate for FX walk-forward metrics.

Evaluates results/walk_forward_metrics.json against configurable thresholds and
returns a GO/NO-GO recommendation.

Usage:
    python deployment_gate.py
    python deployment_gate.py --metrics-path results/walk_forward_metrics.json
    python deployment_gate.py --min-confidence-note "paper-only"

Exit codes:
    0 -> hard gate passed (paper-trading GO)
    1 -> hard gate failed (NO-GO)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CriterionResult:
    name: str
    value: float | int | None
    target: str
    passed: bool
    hard_gate: bool


def _safe_ratio(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    if den == 0:
        return None
    return float(num) / float(den)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return f"{v:.4f}"
    return str(v)


def evaluate_gate(metrics: dict, args: argparse.Namespace) -> tuple[list[CriterionResult], dict[str, Any]]:
    total_folds = metrics.get("total_folds")
    valid_folds = metrics.get("valid_folds")
    valid_ratio = _safe_ratio(valid_folds, total_folds)

    total_trades = metrics.get("total_trades_all_folds")
    profitable_folds_pct = metrics.get("pct_profitable_folds")
    avg_sharpe = metrics.get("avg_sharpe")
    std_sharpe = metrics.get("std_sharpe")
    max_dd = metrics.get("combined_max_drawdown_pct")
    total_return = metrics.get("combined_total_return_pct")

    if max_dd is None or max_dd == 0:
        return_to_dd = None
    else:
        return_to_dd = total_return / abs(max_dd)

    criteria = [
        CriterionResult(
            name="valid_fold_ratio",
            value=valid_ratio,
            target=f">= {args.min_valid_fold_ratio}",
            passed=(valid_ratio is not None and valid_ratio >= args.min_valid_fold_ratio),
            hard_gate=True,
        ),
        CriterionResult(
            name="total_trades_all_folds",
            value=total_trades,
            target=f">= {args.min_total_trades}",
            passed=(total_trades is not None and total_trades >= args.min_total_trades),
            hard_gate=True,
        ),
        CriterionResult(
            name="pct_profitable_folds",
            value=profitable_folds_pct,
            target=f">= {args.min_profitable_folds_pct}",
            passed=(profitable_folds_pct is not None and profitable_folds_pct >= args.min_profitable_folds_pct),
            hard_gate=True,
        ),
        CriterionResult(
            name="avg_sharpe",
            value=avg_sharpe,
            target=f">= {args.min_avg_sharpe}",
            passed=(avg_sharpe is not None and avg_sharpe >= args.min_avg_sharpe),
            hard_gate=True,
        ),
        CriterionResult(
            name="combined_max_drawdown_pct",
            value=max_dd,
            target=f">= {args.max_allowed_drawdown_pct}",
            passed=(max_dd is not None and max_dd >= args.max_allowed_drawdown_pct),
            hard_gate=True,
        ),
        CriterionResult(
            name="return_to_drawdown",
            value=return_to_dd,
            target=f">= {args.min_return_to_drawdown}",
            passed=(return_to_dd is not None and return_to_dd >= args.min_return_to_drawdown),
            hard_gate=True,
        ),
        CriterionResult(
            name="std_sharpe_stability",
            value=std_sharpe,
            target=f"<= {args.max_std_sharpe}",
            passed=(std_sharpe is not None and std_sharpe <= args.max_std_sharpe),
            hard_gate=False,
        ),
    ]

    hard_failed = [c for c in criteria if c.hard_gate and not c.passed]
    soft_failed = [c for c in criteria if (not c.hard_gate) and not c.passed]

    summary = {
        "hard_pass": len(hard_failed) == 0,
        "soft_pass": len(soft_failed) == 0,
        "hard_failed": [c.name for c in hard_failed],
        "soft_failed": [c.name for c in soft_failed],
        "recommendation": (
            "GO_PAPER_AND_PREPARE_LIVE"
            if len(hard_failed) == 0 and len(soft_failed) == 0
            else "GO_PAPER_WITH_CAUTION"
            if len(hard_failed) == 0
            else "NO_GO"
        ),
        "source_metrics_path": str(args.metrics_path),
    }

    return criteria, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate walk-forward deployment gate")
    parser.add_argument(
        "--metrics-path",
        default="results/walk_forward_metrics.json",
        type=Path,
        help="Path to walk-forward metrics JSON",
    )

    parser.add_argument("--min-valid-fold-ratio", type=float, default=0.80)
    parser.add_argument("--min-total-trades", type=int, default=80)
    parser.add_argument("--min-profitable-folds-pct", type=float, default=0.55)
    parser.add_argument("--min-avg-sharpe", type=float, default=1.0)
    parser.add_argument("--max-allowed-drawdown-pct", type=float, default=-15.0)
    parser.add_argument("--min-return-to-drawdown", type=float, default=1.5)
    parser.add_argument("--max-std-sharpe", type=float, default=3.5)

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full gate result as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.metrics_path.exists():
        print(f"[ERROR] Metrics file not found: {args.metrics_path}")
        return 1

    with open(args.metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    criteria, summary = evaluate_gate(metrics, args)

    if args.json:
        payload = {
            "summary": summary,
            "criteria": [
                {
                    "name": c.name,
                    "value": c.value,
                    "target": c.target,
                    "passed": c.passed,
                    "hard_gate": c.hard_gate,
                }
                for c in criteria
            ],
            "metrics": metrics,
        }
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 68)
        print(" DEPLOYMENT GATE – WALK-FORWARD CHECK")
        print("=" * 68)
        for c in criteria:
            kind = "HARD" if c.hard_gate else "SOFT"
            status = "PASS" if c.passed else "FAIL"
            print(f"[{status}] ({kind}) {c.name:<24} value={_fmt(c.value):<10} target {c.target}")

        print("-" * 68)
        print(f"Recommendation: {summary['recommendation']}")
        if summary["hard_failed"]:
            print(f"Hard failures : {', '.join(summary['hard_failed'])}")
        if summary["soft_failed"]:
            print(f"Soft warnings : {', '.join(summary['soft_failed'])}")
        print("=" * 68)

    return 0 if summary["hard_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
