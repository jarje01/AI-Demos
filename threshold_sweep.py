"""
Threshold sweep utility for EUR/USD walk-forward tuning.

Sweeps selected ML and rules thresholds, runs walk-forward for each combination,
and ranks settings by return-to-drawdown.

Examples:
    python threshold_sweep.py
    python threshold_sweep.py --conf 0.33,0.35,0.37 --edge 0.001,0.003,0.006 --adx 16,18,20
    python threshold_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from itertools import product
from pathlib import Path

from config import MODEL_CONFIG, RULES_CONFIG, RISK_CONFIG, WALK_FORWARD_CONFIG, PATHS
from data_loader import download_data
from walk_forward import WalkForwardEngine


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep walk-forward thresholds and rank combinations")
    parser.add_argument("--conf", default="0.33,0.35,0.37", help="Comma-separated min_confidence values")
    parser.add_argument("--edge", default="0.001,0.003,0.006", help="Comma-separated min_edge_over_opposite values")
    parser.add_argument("--adx", default="16,18,20", help="Comma-separated ADX thresholds")
    parser.add_argument("--flat-edge", type=float, default=-1.0, help="min_edge_over_flat value")
    parser.add_argument("--min-valid-folds", type=int, default=2,
                        help="Minimum valid folds required for a combination to qualify in ranking")
    parser.add_argument("--quick", action="store_true", help="Use a very small sweep (fast sanity run)")
    parser.add_argument("--output", default="threshold_sweep.json", help="Output JSON filename under results/")
    return parser.parse_args()


def _score(result_metrics: dict) -> float | None:
    total_return = result_metrics.get("combined_total_return_pct")
    max_dd = result_metrics.get("combined_max_drawdown_pct")
    if total_return is None or max_dd is None or max_dd == 0:
        return None
    return total_return / abs(max_dd)


def run_sweep(args: argparse.Namespace) -> dict:
    if args.quick:
        conf_values = [0.34, 0.35]
        edge_values = [0.003, 0.006]
        adx_values = [18, 20]
    else:
        conf_values = _parse_float_list(args.conf)
        edge_values = _parse_float_list(args.edge)
        adx_values = _parse_int_list(args.adx)

    combos = list(product(conf_values, edge_values, adx_values))
    print(f"Running {len(combos)} combinations...")

    df = download_data()
    rows: list[dict] = []

    for i, (conf, edge, adx_thr) in enumerate(combos, start=1):
        print(f"[{i}/{len(combos)}] conf={conf}, edge={edge}, adx={adx_thr}")

        model_cfg = {
            **MODEL_CONFIG,
            "min_confidence": conf,
            "min_edge_over_opposite": edge,
            "min_edge_over_flat": args.flat_edge,
        }
        rules_cfg = {
            **RULES_CONFIG,
            "adx_trending_threshold": adx_thr,
        }

        engine = WalkForwardEngine(
            wf_cfg=WALK_FORWARD_CONFIG,
            risk_cfg=RISK_CONFIG,
            model_cfg=model_cfg,
            rules_cfg=rules_cfg,
        )

        try:
            result = engine.run(df)
            m = result.combined_metrics
            score = _score(m)

            rows.append(
                {
                    "min_confidence": conf,
                    "min_edge_over_opposite": edge,
                    "adx_trending_threshold": adx_thr,
                    "valid_folds": m.get("valid_folds"),
                    "total_trades_all_folds": m.get("total_trades_all_folds"),
                    "combined_total_return_pct": m.get("combined_total_return_pct"),
                    "combined_max_drawdown_pct": m.get("combined_max_drawdown_pct"),
                    "avg_sharpe": m.get("avg_sharpe"),
                    "return_to_drawdown": score,
                    "metrics": m,
                }
            )
        except Exception as exc:
            print(f"  -> failed: {exc}")
            rows.append(
                {
                    "min_confidence": conf,
                    "min_edge_over_opposite": edge,
                    "adx_trending_threshold": adx_thr,
                    "valid_folds": 0,
                    "total_trades_all_folds": 0,
                    "combined_total_return_pct": None,
                    "combined_max_drawdown_pct": None,
                    "avg_sharpe": None,
                    "return_to_drawdown": None,
                    "metrics": {"error": str(exc)},
                }
            )

    qualified = [r for r in rows if (r.get("valid_folds") or 0) >= args.min_valid_folds]

    ranked_qualified = sorted(
        qualified,
        key=lambda r: (-999 if r["return_to_drawdown"] is None else -r["return_to_drawdown"])
    )
    ranked_all = sorted(
        rows,
        key=lambda r: (-999 if r["return_to_drawdown"] is None else -r["return_to_drawdown"])
    )

    ranked = ranked_qualified if ranked_qualified else ranked_all
    best = ranked[0] if ranked else None

    return {
        "sweep_size": len(rows),
        "min_valid_folds": args.min_valid_folds,
        "qualified_count": len(qualified),
        "used_fallback_ranking": len(ranked_qualified) == 0,
        "ranked_qualified": ranked_qualified,
        "ranked_all": ranked_all,
        "ranked": ranked,
        "best": best,
    }


def main() -> int:
    args = parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r"`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
        category=UserWarning,
    )

    payload = run_sweep(args)

    out_dir = Path(PATHS["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    fd, tmp_path = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".tmp", dir=str(out_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if payload.get("best"):
        b = payload["best"]
        print("\nBest setting:")
        print(
            f"  conf={b['min_confidence']} edge={b['min_edge_over_opposite']} adx={b['adx_trending_threshold']} "
            f"ret={b['combined_total_return_pct']} dd={b['combined_max_drawdown_pct']} score={b['return_to_drawdown']}"
        )
    print(f"Saved sweep results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
