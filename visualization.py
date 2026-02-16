"""
Visualization – plots for equity curves, drawdown, signal overlays, and
walk-forward fold performance.

All plots are saved to PATHS['plots_dir'] and optionally shown interactively.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from config import PATHS

logger = logging.getLogger(__name__)


def _ensure_plots_dir():
    os.makedirs(PATHS["plots_dir"], exist_ok=True)


# ─── Equity Curve ─────────────────────────────────────────────────────────────

def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    filename: str = "equity_curve.png",
    initial_capital: float = 100_000.0,
    show: bool = False,
) -> str:
    """Plot equity curve with drawdown sub-plot."""
    _ensure_plots_dir()
    path = os.path.join(PATHS["plots_dir"], filename)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Equity
    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, color="#2196F3", linewidth=1.5, label="Portfolio Equity")
    ax1.axhline(initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.fill_between(equity.index, initial_capital, equity.values,
                     where=equity.values >= initial_capital, alpha=0.15, color="green")
    ax1.fill_between(equity.index, initial_capital, equity.values,
                     where=equity.values < initial_capital, alpha=0.15, color="red")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved equity curve to %s", path)
    return path


# ─── Price + Signals ──────────────────────────────────────────────────────────

def plot_signals_on_price(
    df_features: pd.DataFrame,
    signals_df: pd.DataFrame,
    filename: str = "price_signals.png",
    n_bars: int = 500,
    show: bool = False,
) -> str:
    """Overlay buy/sell signals on price chart with RSI subplot."""
    _ensure_plots_dir()
    path = os.path.join(PATHS["plots_dir"], filename)

    # Slice to last n_bars for readability
    df = df_features.tail(n_bars).copy()
    sig = signals_df.reindex(df.index)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("EUR/USD Price with ML+Rules Signals", fontsize=13, fontweight="bold")

    # Price
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#455A64", linewidth=1, label="Close")
    if "sma_50" in df.columns:
        ax1.plot(df.index, df["sma_50"], color="#FF9800", linewidth=1, alpha=0.7, label="SMA50")
    if "sma_200" in df.columns:
        ax1.plot(df.index, df["sma_200"], color="#9C27B0", linewidth=1, alpha=0.7, label="SMA200")

    longs = sig[sig["signal"] == 1].index
    shorts = sig[sig["signal"] == -1].index
    ax1.scatter(longs, df.loc[longs, "Close"], marker="^", color="#4CAF50", s=60, zorder=5, label="Long")
    ax1.scatter(shorts, df.loc[shorts, "Close"], marker="v", color="#F44336", s=60, zorder=5, label="Short")
    ax1.set_ylabel("Price (EUR/USD)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # RSI
    ax2 = axes[1]
    if "rsi" in df.columns:
        ax2.plot(df.index, df["rsi"], color="#2196F3", linewidth=1)
        ax2.axhline(70, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax2.axhline(30, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
        ax2.axhline(50, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax2.fill_between(df.index, 30, df["rsi"].clip(0, 30), alpha=0.2, color="green")
        ax2.fill_between(df.index, 70, df["rsi"].clip(70, 100), alpha=0.2, color="red")
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # MACD
    ax3 = axes[2]
    if "macd_hist" in df.columns:
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in df["macd_hist"]]
        ax3.bar(df.index, df["macd_hist"], color=colors, alpha=0.7, width=0.02)
        if "macd" in df.columns and "macd_signal" in df.columns:
            ax3.plot(df.index, df["macd"], color="#2196F3", linewidth=0.8)
            ax3.plot(df.index, df["macd_signal"], color="#FF9800", linewidth=0.8)
        ax3.axhline(0, color="gray", linewidth=0.5)
        ax3.set_ylabel("MACD")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved signal chart to %s", path)
    return path


# ─── Walk-Forward Results ─────────────────────────────────────────────────────

def plot_walk_forward_results(
    result,
    filename: str = "walk_forward.png",
    initial_capital: float = 100_000.0,
    show: bool = False,
) -> str:
    """Plot combined OOS equity + per-fold return/Sharpe bar charts."""
    _ensure_plots_dir()
    path = os.path.join(PATHS["plots_dir"], filename)

    folds = result.folds
    valid = [f for f in folds if "total_return_pct" in f.metrics]
    n_folds = len(valid)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("Walk-Forward Analysis – EUR/USD ML Trading System",
                 fontsize=14, fontweight="bold")

    # Combined equity
    ax1 = fig.add_subplot(gs[0, :])
    if result.combined_equity is not None:
        eq = result.combined_equity
        ax1.plot(eq.index, eq.values, color="#2196F3", linewidth=1.5)
        ax1.axhline(initial_capital, color="gray", linestyle="--", alpha=0.5)
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max * 100
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(eq.index, dd.values, 0, color="#F44336", alpha=0.3)
        ax1_twin.set_ylabel("Drawdown (%)", color="#F44336", fontsize=9)
        ax1_twin.tick_params(axis="y", labelcolor="#F44336")
    ax1.set_title("Combined OOS Equity Curve (rebased)", fontsize=11)
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)

    if n_folds == 0:
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    fold_ids = [f.fold_id for f in valid]
    returns = [f.metrics["total_return_pct"] for f in valid]
    sharpes = [f.metrics.get("sharpe_ratio", 0) for f in valid]
    win_rates = [f.metrics.get("win_rate", 0) * 100 for f in valid]
    n_trades = [f.metrics.get("total_trades", 0) for f in valid]

    # Per-fold returns
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in returns]
    bars = ax2.bar(fold_ids, returns, color=colors, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Per-Fold OOS Return (%)", fontsize=10)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Return (%)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Sharpe per fold
    ax3 = fig.add_subplot(gs[1, 1])
    sharpe_colors = ["#2196F3" if s > 0 else "#FF9800" for s in sharpes]
    ax3.bar(fold_ids, sharpes, color=sharpe_colors, alpha=0.8, edgecolor="white")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(1.0, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
    ax3.set_title("Per-Fold Sharpe Ratio", fontsize=10)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("Sharpe")
    ax3.grid(True, alpha=0.3, axis="y")

    # Win rate per fold
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(fold_ids, win_rates, color="#9C27B0", alpha=0.8, edgecolor="white")
    ax4.axhline(50, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
    ax4.set_title("Per-Fold Win Rate (%)", fontsize=10)
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Win Rate (%)")
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis="y")

    # Trades per fold
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(fold_ids, n_trades, color="#FF9800", alpha=0.8, edgecolor="white")
    ax5.set_title("Trades per Fold", fontsize=10)
    ax5.set_xlabel("Fold")
    ax5.set_ylabel("# Trades")
    ax5.grid(True, alpha=0.3, axis="y")

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved walk-forward chart to %s", path)
    return path


# ─── Feature Importance ───────────────────────────────────────────────────────

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    filename: str = "feature_importance.png",
    show: bool = False,
) -> str:
    _ensure_plots_dir()
    path = os.path.join(PATHS["plots_dir"], filename)

    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(top.index, top.values, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (averaged across WF folds)", fontsize=12)
    ax.set_xlabel("Mean Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance chart to %s", path)
    return path


# ─── Trade Distribution ───────────────────────────────────────────────────────

def plot_trade_distribution(
    trades: list,
    filename: str = "trade_distribution.png",
    show: bool = False,
) -> str:
    """Histogram of trade P&L + MAE/MFE scatter."""
    _ensure_plots_dir()
    path = os.path.join(PATHS["plots_dir"], filename)

    pnls = [t.pnl for t in trades if t.pnl is not None]
    if not pnls:
        logger.warning("No trades to plot distribution for.")
        return ""

    maes = [t.max_mae for t in trades if t.max_mae is not None]
    mfes = [t.max_mfe for t in trades if t.max_mfe is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Trade Analysis", fontsize=13, fontweight="bold")

    # P&L distribution
    ax1 = axes[0]
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
    ax1.hist(pnls, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=1)
    ax1.axvline(np.mean(pnls), color="orange", linewidth=1.5, linestyle="--", label=f"Mean: ${np.mean(pnls):.0f}")
    ax1.set_title("P&L Distribution")
    ax1.set_xlabel("P&L ($)")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # MAE vs MFE scatter
    ax2 = axes[1]
    if maes and mfes:
        win_colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax2.scatter(maes, mfes, c=win_colors, alpha=0.6, s=20, edgecolors="none")
        ax2.set_xlabel("Max Adverse Excursion (MAE)")
        ax2.set_ylabel("Max Favourable Excursion (MFE)")
        ax2.set_title("MAE vs MFE")
        legend_elements = [Patch(facecolor="#4CAF50", label="Winners"),
                           Patch(facecolor="#F44336", label="Losers")]
        ax2.legend(handles=legend_elements, fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved trade distribution chart to %s", path)
    return path
