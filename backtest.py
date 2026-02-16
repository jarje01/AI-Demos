"""
Backtesting Engine – event-driven position management with ATR-based
stops/targets and full performance analytics.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from config import RISK_CONFIG

logger = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: int               # +1 long / -1 short
    stop_loss: float
    take_profit: float
    size: float                  # units of base currency
    risk_amount: float           # $ amount risked
    entry_fee: float = 0.0

    exit_bar: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    gross_pnl: Optional[float] = None
    exit_fee: Optional[float] = None
    total_fees: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    max_mae: Optional[float] = None      # maximum adverse excursion
    max_mfe: Optional[float] = None      # maximum favourable excursion


@dataclass
class PortfolioState:
    equity: float
    cash: float
    open_trades: list[Trade] = field(default_factory=list)
    closed_trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[pd.Timestamp] = field(default_factory=list)


# ─── Backtester ───────────────────────────────────────────────────────────────

class Backtester:
    """
    Bar-by-bar simulator.

    Rules applied each bar:
      1. Check if existing open trades hit stop-loss or take-profit.
      2. If no open trade and a signal is present → open new trade.
      3. Record equity curve.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or RISK_CONFIG
        self.fee_rate = self.cfg.get("trading_fee_bps_per_side", 0.0) / 10_000

    def run(
        self,
        df_features: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> PortfolioState:
        """
        Parameters
        ----------
        df_features  : Full feature DataFrame (must include OHLCV + ATR).
        signals_df   : Output of SignalGenerator.generate().

        Returns
        -------
        PortfolioState with closed_trades and equity_curve populated.
        """
        init_cap = self.cfg["initial_capital"]
        state = PortfolioState(equity=init_cap, cash=init_cap)
        max_dd_threshold = self.cfg["max_drawdown_pct"] / 100
        peak_equity = init_cap
        halted = False

        bars = df_features.index
        n = len(bars)

        for i, timestamp in enumerate(bars):
            if halted:
                state.equity_curve.append(state.equity)
                state.timestamps.append(timestamp)
                continue

            row = df_features.loc[timestamp]
            signal_row = signals_df.loc[timestamp] if timestamp in signals_df.index else None

            open_price = row["Open"]
            high_price = row["High"]
            low_price = row["Low"]
            close_price = row["Close"]
            atr = row.get("atr", 0.001)

            # ── 1. Manage open trades ──────────────────────────────────────
            still_open = []
            for trade in state.open_trades:
                closed, trade = self._check_exit(
                    trade, i, timestamp, open_price, high_price, low_price, close_price
                )
                if closed:
                    state.cash += trade.pnl + trade.risk_amount  # return risk capital + P&L
                    state.closed_trades.append(trade)
                else:
                    still_open.append(trade)
            state.open_trades = still_open

            # ── 2. Open new trade ──────────────────────────────────────────
            if (
                signal_row is not None
                and int(signal_row["signal"]) != 0
                and len(state.open_trades) < self.cfg["max_open_trades"]
            ):
                direction = int(signal_row["signal"])
                trade = self._open_trade(
                    i, timestamp, close_price, direction, atr, state.cash
                )
                if trade is not None:
                    state.cash -= (trade.risk_amount + trade.entry_fee)
                    state.open_trades.append(trade)

            # ── 3. Equity ──────────────────────────────────────────────────
            unrealised = self._unrealised_pnl(state.open_trades, close_price)
            state.equity = state.cash + unrealised + sum(
                t.risk_amount for t in state.open_trades
            )

            peak_equity = max(peak_equity, state.equity)
            drawdown = (peak_equity - state.equity) / peak_equity
            if drawdown >= max_dd_threshold:
                logger.warning(
                    "Max drawdown %.1f%% hit at %s – trading halted.",
                    drawdown * 100, timestamp
                )
                halted = True

            state.equity_curve.append(state.equity)
            state.timestamps.append(timestamp)

        # ── Force-close any open trades at end ────────────────────────────
        if state.open_trades:
            last_ts = bars[-1]
            last_close = df_features.loc[last_ts, "Close"]
            for trade in state.open_trades:
                trade = self._force_close(trade, len(bars) - 1, last_ts, last_close)
                state.cash += trade.pnl + trade.risk_amount
                state.closed_trades.append(trade)
            state.open_trades = []

        logger.info(
            "Backtest complete: %d trades | Final equity: $%.2f",
            len(state.closed_trades), state.equity
        )
        return state

    # ─── Trade lifecycle ───────────────────────────────────────────────────────

    def _open_trade(
        self,
        bar: int,
        timestamp: pd.Timestamp,
        entry_price: float,
        direction: int,
        atr: float,
        available_cash: float,
    ) -> Optional[Trade]:
        atr_stop = atr * self.cfg["atr_stop_multiplier"]
        atr_tp = atr * self.cfg["take_profit_multiplier"]

        if direction == 1:
            stop_loss = entry_price - atr_stop
            take_profit = entry_price + atr_tp
        else:
            stop_loss = entry_price + atr_stop
            take_profit = entry_price - atr_tp

        risk_pct = self.cfg["risk_per_trade_pct"] / 100
        risk_amount = available_cash * risk_pct
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance < 1e-10:
            return None

        size = risk_amount / stop_distance
        entry_fee = entry_price * size * self.fee_rate

        return Trade(
            entry_bar=bar,
            entry_time=timestamp,
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            risk_amount=risk_amount,
            entry_fee=entry_fee,
            max_mae=0.0,
            max_mfe=0.0,
        )

    def _check_exit(
        self,
        trade: Trade,
        bar: int,
        timestamp: pd.Timestamp,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> tuple[bool, Trade]:
        """
        Check whether this bar triggers stop-loss or take-profit.
        Uses high/low to detect intra-bar hits (conservative: assume worst fill).
        """
        d = trade.direction

        # Update MAE / MFE
        if d == 1:
            adverse = (trade.entry_price - low_price)
            favourable = (high_price - trade.entry_price)
        else:
            adverse = (high_price - trade.entry_price)
            favourable = (trade.entry_price - low_price)

        trade.max_mae = max(trade.max_mae, adverse)
        trade.max_mfe = max(trade.max_mfe, favourable)

        hit_sl = (d == 1 and low_price <= trade.stop_loss) or \
                 (d == -1 and high_price >= trade.stop_loss)
        hit_tp = (d == 1 and high_price >= trade.take_profit) or \
                 (d == -1 and low_price <= trade.take_profit)

        if hit_sl and hit_tp:
            # Both hit same bar: assume TP first if in favourable direction
            hit_tp = True
            hit_sl = False

        if hit_sl:
            exit_price = trade.stop_loss
            reason = "stop_loss"
        elif hit_tp:
            exit_price = trade.take_profit
            reason = "take_profit"
        else:
            return False, trade

        trade.exit_bar = bar
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.gross_pnl = d * (exit_price - trade.entry_price) * trade.size
        trade.exit_fee = exit_price * trade.size * self.fee_rate
        trade.total_fees = trade.entry_fee + trade.exit_fee
        trade.pnl = trade.gross_pnl - trade.exit_fee
        trade.pnl_pct = d * (exit_price - trade.entry_price) / trade.entry_price
        return True, trade

    def _force_close(
        self, trade: Trade, bar: int, timestamp: pd.Timestamp, price: float
    ) -> Trade:
        trade.exit_bar = bar
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = "end_of_data"
        trade.gross_pnl = trade.direction * (price - trade.entry_price) * trade.size
        trade.exit_fee = price * trade.size * self.fee_rate
        trade.total_fees = trade.entry_fee + trade.exit_fee
        trade.pnl = trade.gross_pnl - trade.exit_fee
        trade.pnl_pct = trade.direction * (price - trade.entry_price) / trade.entry_price
        return trade

    def _unrealised_pnl(self, open_trades: list[Trade], current_price: float) -> float:
        total = 0.0
        for t in open_trades:
            total += t.direction * (current_price - t.entry_price) * t.size
        return total


# ─── Performance Metrics ──────────────────────────────────────────────────────

def compute_metrics(state: PortfolioState, initial_capital: float | None = None) -> dict:
    """Compute full suite of performance statistics from a PortfolioState."""
    init_cap = initial_capital or RISK_CONFIG["initial_capital"]
    trades = state.closed_trades

    if not trades:
        return {"error": "No trades executed."}

    eq = pd.Series(state.equity_curve, index=state.timestamps)
    pnls = [t.pnl for t in trades]
    gross_pnls = [t.gross_pnl for t in trades if t.gross_pnl is not None]
    total_fees = sum(t.total_fees for t in trades if t.total_fees is not None)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    total_return = (eq.iloc[-1] - init_cap) / init_cap
    n = len(trades)
    win_rate = len(winners) / n if n > 0 else 0

    avg_win = np.mean(winners) if winners else 0
    avg_loss = abs(np.mean(losers)) if losers else 0
    profit_factor = (sum(winners) / abs(sum(losers))) if losers else np.inf

    # Drawdown
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Sharpe (annualised, assuming hourly bars)
    returns = eq.pct_change().dropna()
    bars_per_year = 252 * 6.5   # ~1638 hourly bars in a trading year
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(bars_per_year)
    else:
        sortino = 0.0

    # Calmar
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0

    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Exit reason breakdown
    reasons = pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_return_pct": round(total_return * 100, 2),
        "gross_total_pnl": round(sum(gross_pnls), 2),
        "total_fees_paid": round(total_fees, 2),
        "net_total_pnl": round(sum(pnls), 2),
        "total_pnl": round(sum(pnls), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "final_equity": round(eq.iloc[-1], 2),
        "initial_capital": init_cap,
        "exit_reasons": reasons,
        "equity_curve": eq,
    }


def print_metrics(metrics: dict):
    """Pretty-print performance metrics to stdout."""
    if "error" in metrics:
        print(f"[ERROR] {metrics['error']}")
        return
    print("\n" + "="*60)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("="*60)
    print(f"  Trades          : {metrics['total_trades']}")
    print(f"  Win Rate        : {metrics['win_rate']:.1%}")
    print(f"  Profit Factor   : {metrics['profit_factor']:.2f}")
    print(f"  Total Return    : {metrics['total_return_pct']:.2f}%")
    print(f"  Gross P&L       : ${metrics['gross_total_pnl']:,.2f}")
    print(f"  Fees Paid       : ${metrics['total_fees_paid']:,.2f}")
    print(f"  Net P&L         : ${metrics['net_total_pnl']:,.2f}")
    print(f"  Avg Win         : ${metrics['avg_win']:,.2f}")
    print(f"  Avg Loss        : ${metrics['avg_loss']:,.2f}")
    print(f"  Expectancy      : ${metrics['expectancy_per_trade']:,.2f}")
    print(f"  Max Drawdown    : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio    : {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio   : {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio    : {metrics['calmar_ratio']:.3f}")
    print(f"  Final Equity    : ${metrics['final_equity']:,.2f}")
    print("\n  Exit Reasons:")
    for k, v in metrics["exit_reasons"].items():
        print(f"    {k:<20}: {v}")
    print("="*60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import download_data, split_data
    from features import build_features
    from model import FXModel
    from signals import SignalGenerator

    df = download_data()
    train_raw, test_raw = split_data(df)
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)

    mdl = FXModel()
    mdl.fit(train_feat)
    proba_df = mdl.get_signal_probabilities(test_feat)

    gen = SignalGenerator()
    sigs = gen.generate(test_feat, proba_df)

    bt = Backtester()
    state = bt.run(test_feat, sigs)
    metrics = compute_metrics(state)
    print_metrics(metrics)
