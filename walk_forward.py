"""
Walk-Forward Analysis – rolling train/test windows for realistic OOS evaluation.

The walk-forward framework repeatedly:
  1. Trains a fresh ML model on a fixed-length training window.
  2. Generates signals and runs the backtester on the subsequent test window.
  3. Steps both windows forward by step_size_bars.
  4. Aggregates all OOS equity curves and trade lists.

This avoids look-ahead bias and gives a realistic picture of live performance.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import WALK_FORWARD_CONFIG, RISK_CONFIG
from features import build_features
from model import FXModel
from signals import SignalGenerator
from backtest import Backtester, compute_metrics, PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict = field(default_factory=dict)
    model: FXModel = field(default=None, repr=False)
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = field(default=None, repr=False)


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    combined_equity: pd.Series = None
    combined_metrics: dict = field(default_factory=dict)
    feature_importance_avg: pd.Series = None


class WalkForwardEngine:
    """
    Executes a rolling walk-forward backtest over a feature DataFrame.
    """

    def __init__(
        self,
        wf_cfg: dict | None = None,
        risk_cfg: dict | None = None,
        model_cfg: dict | None = None,
        rules_cfg: dict | None = None,
    ):
        from config import MODEL_CONFIG, RULES_CONFIG
        self.wf_cfg = wf_cfg or WALK_FORWARD_CONFIG
        self.risk_cfg = risk_cfg or RISK_CONFIG
        self.model_cfg = model_cfg or MODEL_CONFIG
        self.rules_cfg = rules_cfg or RULES_CONFIG

    def run(self, df_raw: pd.DataFrame) -> WalkForwardResult:
        """
        Parameters
        ----------
        df_raw : Raw OHLCV DataFrame (full date range).

        Returns
        -------
        WalkForwardResult containing per-fold and combined results.
        """
        train_w = self.wf_cfg["train_window_bars"]
        test_w = self.wf_cfg["test_window_bars"]
        step = self.wf_cfg["step_size_bars"]
        min_trades = self.wf_cfg["min_trades_per_fold"]

        n = len(df_raw)

        # Auto-adjust fold geometry when configured windows do not fit the
        # available dataset (common when using shorter recent ranges).
        if (train_w + test_w) > n:
            train_w = max(300, int(n * 0.65))
            test_w = max(120, int(n * 0.25))
            if train_w + test_w > n:
                test_w = max(60, n - train_w)
            if train_w + test_w > n:
                train_w = max(200, n - test_w)
            step = max(30, int(test_w * 0.5))
            logger.warning(
                "Configured walk-forward windows exceed available data (%d bars). "
                "Auto-adjusted to train=%d, test=%d, step=%d.",
                n, train_w, test_w, step
            )

        folds: list[WalkForwardFold] = []
        feature_importances = []

        fold_id = 0
        start = 0

        logger.info(
            "Starting walk-forward: train=%d bars, test=%d bars, step=%d bars",
            train_w, test_w, step
        )

        while (start + train_w + test_w) <= n:
            train_slice = df_raw.iloc[start: start + train_w]
            test_slice = df_raw.iloc[start + train_w: start + train_w + test_w]

            fold_id += 1
            logger.info(
                "Fold %d  |  Train: %s → %s  |  Test: %s → %s",
                fold_id,
                train_slice.index[0].date(), train_slice.index[-1].date(),
                test_slice.index[0].date(), test_slice.index[-1].date(),
            )
            t0 = time.time()

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
            )

            try:
                # ── 1. Build features ──────────────────────────────────────
                train_feat = build_features(train_slice)

                # Build test features with warmup bars so long-lookback
                # indicators (e.g., SMA200) are available throughout OOS.
                warmup_bars = 250
                warmup_start = max(0, (start + train_w) - warmup_bars)
                test_with_warmup = df_raw.iloc[warmup_start: start + train_w + test_w]
                test_feat_full = build_features(test_with_warmup)
                test_feat = test_feat_full.loc[test_slice.index.intersection(test_feat_full.index)]

                if len(train_feat) < 50 or len(test_feat) < 10:
                    logger.warning("Fold %d skipped – insufficient data.", fold_id)
                    start += step
                    continue

                # ── 2. Train model ─────────────────────────────────────────
                model = FXModel(cfg=self.model_cfg)
                model.fit(train_feat)
                fold.model = model

                fi = model.feature_importance()
                if len(fi) > 0:
                    feature_importances.append(fi)

                # ── 3. Generate signals ────────────────────────────────────
                proba_df = model.get_signal_probabilities(test_feat)
                gen = SignalGenerator(cfg=self.rules_cfg, model_cfg=self.model_cfg)
                signals_df = gen.generate(test_feat, proba_df)

                # ── 4. Backtest ────────────────────────────────────────────
                bt = Backtester(cfg=self.risk_cfg)
                state = bt.run(test_feat, signals_df)

                fold.trades = state.closed_trades
                if len(state.equity_curve) > 0:
                    fold.equity_curve = pd.Series(
                        state.equity_curve, index=state.timestamps
                    )

                # ── 5. Metrics ─────────────────────────────────────────────
                if len(fold.trades) >= min_trades:
                    fold.metrics = compute_metrics(state, self.risk_cfg["initial_capital"])
                    # Remove bulky equity series from fold metrics to save memory
                    fold.metrics.pop("equity_curve", None)
                else:
                    fold.metrics = {
                        "warning": f"Only {len(fold.trades)} trades (min={min_trades})",
                        "total_trades": len(fold.trades),
                    }
                    logger.warning(
                        "Fold %d: only %d trades – metrics may be unreliable.",
                        fold_id, len(fold.trades)
                    )

            except Exception as exc:
                logger.error("Fold %d failed: %s", fold_id, exc, exc_info=True)
                fold.metrics = {"error": str(exc)}

            elapsed = time.time() - t0
            logger.info("Fold %d complete in %.1fs", fold_id, elapsed)
            folds.append(fold)
            start += step

        # ── Aggregate results ──────────────────────────────────────────────
        result = WalkForwardResult(folds=folds)
        result.combined_equity = self._combine_equity(folds)
        result.combined_metrics = self._combined_metrics(folds, result.combined_equity)

        if feature_importances:
            combined_fi = pd.concat(feature_importances, axis=1).mean(axis=1)
            result.feature_importance_avg = combined_fi.sort_values(ascending=False)

        return result

    # ─── Aggregation helpers ───────────────────────────────────────────────────

    def _combine_equity(self, folds: list[WalkForwardFold]) -> pd.Series | None:
        """Stitch together OOS equity curves by rebasing each fold."""
        curves = [f.equity_curve for f in folds if f.equity_curve is not None]
        if not curves:
            return None

        combined = []
        current_equity = self.risk_cfg["initial_capital"]

        for curve in curves:
            # Rebase: scale this fold's curve so it starts from current_equity
            fold_start = curve.iloc[0]
            scaled = curve / fold_start * current_equity
            combined.append(scaled)
            current_equity = scaled.iloc[-1]

        return pd.concat(combined).sort_index()

    def _combined_metrics(
        self, folds: list[WalkForwardFold], combined_equity: pd.Series | None
    ) -> dict:
        valid_folds = [f for f in folds if "win_rate" in f.metrics and "total_return_pct" in f.metrics]
        attempted_folds = [f for f in folds if "total_trades" in f.metrics]
        if not valid_folds:
            return {"error": "No valid folds."}

        all_trades_count = [f.metrics.get("total_trades", 0) for f in valid_folds]
        all_win_rates = [f.metrics["win_rate"] for f in valid_folds if "win_rate" in f.metrics]
        all_sharpes = [f.metrics["sharpe_ratio"] for f in valid_folds if "sharpe_ratio" in f.metrics]
        all_returns = [f.metrics["total_return_pct"] for f in valid_folds if "total_return_pct" in f.metrics]
        all_pf = [f.metrics["profit_factor"] for f in valid_folds if "profit_factor" in f.metrics and np.isfinite(f.metrics["profit_factor"])]
        all_gross_pnl = [f.metrics["gross_total_pnl"] for f in valid_folds if "gross_total_pnl" in f.metrics]
        all_net_pnl = [f.metrics["net_total_pnl"] for f in valid_folds if "net_total_pnl" in f.metrics]
        all_fees = [f.metrics["total_fees_paid"] for f in valid_folds if "total_fees_paid" in f.metrics]

        metrics: dict = {
            "total_folds": len(folds),
            "valid_folds": len(valid_folds),
            "attempted_folds": len(attempted_folds),
            "total_trades_all_folds": sum(all_trades_count),
            "avg_trades_per_fold": round(np.mean(all_trades_count), 1),
        }

        if all_win_rates:
            metrics["avg_win_rate"] = round(np.mean(all_win_rates), 4)
        if all_sharpes:
            metrics["avg_sharpe"] = round(np.mean(all_sharpes), 4)
            metrics["std_sharpe"] = round(np.std(all_sharpes), 4)
        if all_returns:
            metrics["avg_return_pct"] = round(np.mean(all_returns), 2)
            metrics["std_return_pct"] = round(np.std(all_returns), 2)
            metrics["pct_profitable_folds"] = round(
                sum(1 for r in all_returns if r > 0) / len(all_returns), 4
            )
        if all_pf:
            metrics["avg_profit_factor"] = round(np.mean(all_pf), 4)
        if all_gross_pnl:
            metrics["gross_total_pnl_all_folds"] = round(float(np.sum(all_gross_pnl)), 2)
        if all_net_pnl:
            metrics["net_total_pnl_all_folds"] = round(float(np.sum(all_net_pnl)), 2)
        if all_fees:
            metrics["total_fees_paid_all_folds"] = round(float(np.sum(all_fees)), 2)

        # Combined drawdown on stitched equity curve
        if combined_equity is not None and len(combined_equity) > 1:
            roll_max = combined_equity.cummax()
            dd = (combined_equity - roll_max) / roll_max
            metrics["combined_max_drawdown_pct"] = round(dd.min() * 100, 2)
            init = self.risk_cfg["initial_capital"]
            total_ret = (combined_equity.iloc[-1] - init) / init
            metrics["combined_total_return_pct"] = round(total_ret * 100, 2)

        return metrics


def print_wf_summary(result: WalkForwardResult):
    """Pretty-print walk-forward results."""
    print("\n" + "="*70)
    print("  WALK-FORWARD ANALYSIS RESULTS")
    print("="*70)

    m = result.combined_metrics
    print(f"\n  Folds Total/Valid      : {m.get('total_folds')} / {m.get('valid_folds')}")
    print(f"  Total Trades           : {m.get('total_trades_all_folds')}")
    print(f"  Avg Trades / Fold      : {m.get('avg_trades_per_fold')}")
    print(f"  Avg Win Rate           : {m.get('avg_win_rate', 0):.1%}")
    print(f"  Avg Sharpe             : {m.get('avg_sharpe', 0):.3f}  (±{m.get('std_sharpe', 0):.3f})")
    print(f"  Avg Return / Fold      : {m.get('avg_return_pct', 0):.2f}%  (±{m.get('std_return_pct', 0):.2f}%)")
    print(f"  % Profitable Folds     : {m.get('pct_profitable_folds', 0):.1%}")
    print(f"  Avg Profit Factor      : {m.get('avg_profit_factor', 0):.2f}")
    print(f"  Combined Total Return  : {m.get('combined_total_return_pct', 0):.2f}%")
    print(f"  Combined Max Drawdown  : {m.get('combined_max_drawdown_pct', 0):.2f}%")
    if "gross_total_pnl_all_folds" in m:
        print(f"  Gross P&L (all folds)  : ${m.get('gross_total_pnl_all_folds', 0):,.2f}")
    if "total_fees_paid_all_folds" in m:
        print(f"  Fees Paid (all folds)  : ${m.get('total_fees_paid_all_folds', 0):,.2f}")
    if "net_total_pnl_all_folds" in m:
        print(f"  Net P&L (all folds)    : ${m.get('net_total_pnl_all_folds', 0):,.2f}")

    print("\n  ── Per-Fold Summary ──")
    header = f"  {'Fold':>4}  {'Train Start':>12}  {'Test Start':>12}  {'Trades':>6}  {'Win%':>6}  {'Ret%':>7}  {'Sharpe':>7}"
    print(header)
    print("  " + "-"*68)

    for fold in result.folds:
        fm = fold.metrics
        trades = fm.get("total_trades", 0)
        win_r = fm.get("win_rate", float("nan"))
        ret = fm.get("total_return_pct", float("nan"))
        sharpe = fm.get("sharpe_ratio", float("nan"))
        err = fm.get("warning", fm.get("error", ""))
        note = f"  [{err[:30]}]" if err else ""

        print(
            f"  {fold.fold_id:>4}  {str(fold.train_start.date()):>12}  "
            f"{str(fold.test_start.date()):>12}  {trades:>6}  "
            f"{win_r:>5.1%}  {ret:>6.2f}%  {sharpe:>7.3f}{note}"
        )

    if result.feature_importance_avg is not None:
        print("\n  ── Top-10 Feature Importances (averaged) ──")
        for feat, imp in result.feature_importance_avg.head(10).items():
            print(f"  {feat:<35}: {imp:.4f}")

    print("="*70)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import download_data

    df = download_data()
    engine = WalkForwardEngine()
    result = engine.run(df)
    print_wf_summary(result)
