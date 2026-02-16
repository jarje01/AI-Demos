"""
Signal Generator – combines ML model probabilities with rules-based filters
to produce final trading signals.

Signal conventions:
  +1 = Long (buy EUR/USD)
  -1 = Short (sell EUR/USD)
   0 = Flat (no position)
"""

import logging
import numpy as np
import pandas as pd

from config import RULES_CONFIG

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Applies a layered rules-based filter stack on top of ML probabilities
    to produce final trading decisions.

    Layer order:
      1. ML Probability gate (must exceed threshold in a given direction)
      2. Trend filter          (price vs SMA200)
      3. Volatility filter     (ATR within acceptable range)
      4. RSI filter            (avoid extreme overbought/oversold entries)
      5. Regime filter         (ADX trending-market check)
      6. Trade cooldown        (minimum bars between signals)
    """

    def __init__(self, cfg: dict | None = None, model_cfg: dict | None = None):
        from config import MODEL_CONFIG
        self.cfg = cfg or RULES_CONFIG
        model_cfg = model_cfg or MODEL_CONFIG
        self.min_conf = model_cfg.get("min_confidence", model_cfg.get("signal_threshold", 0.55))
        self.min_edge_opposite = model_cfg.get("min_edge_over_opposite", 0.05)
        self.min_edge_flat = model_cfg.get("min_edge_over_flat", 0.0)

    # ─── Main entry point ─────────────────────────────────────────────────────

    def generate(
        self,
        df_features: pd.DataFrame,
        proba_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_features : Feature DataFrame (must contain OHLCV + indicators)
        proba_df    : DataFrame with columns [prob_short, prob_flat, prob_long]

        Returns
        -------
        DataFrame with columns:
            raw_signal     – ML-only signal before filters
            signal         – Final filtered signal (+1/0/-1)
            signal_reason  – Human-readable reason for each decision
        """
        df = df_features.copy()
        proba = proba_df.reindex(df.index).ffill()

        out = pd.DataFrame(index=df.index)
        out["prob_long"] = proba["prob_long"]
        out["prob_short"] = proba["prob_short"]
        out["prob_flat"] = proba["prob_flat"]

        # ── Step 1: ML raw signal ──────────────────────────────────────────
        raw = self._ml_raw_signal(proba)
        out["raw_signal"] = raw

        # ── Step 2: Apply filter stack ────────────────────────────────────
        signals = raw.copy()
        reasons = pd.Series("", index=df.index)

        # Trend filter
        if self.cfg["trend_filter_enabled"]:
            signals, reasons = self._apply_trend_filter(signals, reasons, df)

        # Volatility filter
        if self.cfg["volatility_filter_enabled"]:
            signals, reasons = self._apply_volatility_filter(signals, reasons, df)

        # RSI filter
        if self.cfg["rsi_filter_enabled"]:
            signals, reasons = self._apply_rsi_filter(signals, reasons, df)

        # Regime filter
        if self.cfg["regime_filter_enabled"]:
            signals, reasons = self._apply_regime_filter(signals, reasons, df)

        # Trade cooldown
        signals = self._apply_cooldown(signals)

        # Fill empty reasons for passing trades
        reasons[reasons == ""] = "PASS"

        out["signal"] = signals
        out["signal_reason"] = reasons

        self._log_summary(out)
        return out

    # ─── ML raw signal ────────────────────────────────────────────────────────

    def _ml_raw_signal(self, proba: pd.DataFrame) -> pd.Series:
        """
        Convert probabilities to directional signal via confidence+edge gates.

        Long when:
          prob_long >= min_conf
          prob_long - prob_short >= min_edge_over_opposite
          prob_long - prob_flat >= min_edge_over_flat

        Short mirrors the above conditions.
        """
        signal = pd.Series(0, index=proba.index, dtype=int)

        long_conf = proba["prob_long"] >= self.min_conf
        short_conf = proba["prob_short"] >= self.min_conf

        long_edge_opp = (proba["prob_long"] - proba["prob_short"]) >= self.min_edge_opposite
        short_edge_opp = (proba["prob_short"] - proba["prob_long"]) >= self.min_edge_opposite

        long_edge_flat = (proba["prob_long"] - proba["prob_flat"]) >= self.min_edge_flat
        short_edge_flat = (proba["prob_short"] - proba["prob_flat"]) >= self.min_edge_flat

        long_mask = long_conf & long_edge_opp & long_edge_flat
        short_mask = short_conf & short_edge_opp & short_edge_flat

        signal[long_mask] = 1
        signal[short_mask] = -1

        both = long_mask & short_mask
        if both.any():
            long_strength = proba.loc[both, "prob_long"] - proba.loc[both, "prob_short"]
            pick_long = long_strength >= 0
            signal.loc[both] = np.where(pick_long.values, 1, -1)

        return signal

    # ─── Individual filters ───────────────────────────────────────────────────

    def _apply_trend_filter(
        self, signals: pd.Series, reasons: pd.Series, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """
        Only allow long signals when price > SMA200 (bullish regime).
        Only allow short signals when price < SMA200 (bearish regime).
        """
        sma_col = f"sma_{self.cfg['trend_sma']}"
        if sma_col not in df.columns:
            logger.warning("Trend filter: column %s not found – skipping.", sma_col)
            return signals, reasons

        above_trend = df["Close"] > df[sma_col]
        below_trend = df["Close"] < df[sma_col]

        # Kill long signals in downtrend
        mask_blocked_long = (signals == 1) & ~above_trend
        signals[mask_blocked_long] = 0
        reasons[mask_blocked_long] = "BLOCKED:trend_bearish"

        # Kill short signals in uptrend
        mask_blocked_short = (signals == -1) & ~below_trend
        signals[mask_blocked_short] = 0
        reasons[mask_blocked_short] = "BLOCKED:trend_bullish"

        return signals, reasons

    def _apply_volatility_filter(
        self, signals: pd.Series, reasons: pd.Series, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Block trades when ATR is outside its acceptable range."""
        if "atr" not in df.columns or "atr_rolling_mean" not in df.columns:
            logger.warning("Volatility filter: ATR columns not found – skipping.")
            return signals, reasons

        atr_norm = df["atr"] / df["atr_rolling_mean"].replace(0, np.nan)
        too_low = atr_norm < self.cfg["atr_min_multiplier"]
        too_high = atr_norm > self.cfg["atr_max_multiplier"]

        mask_active = signals != 0
        mask_low = mask_active & too_low
        mask_high = mask_active & too_high

        signals[mask_low] = 0
        reasons[mask_low] = "BLOCKED:vol_too_low"

        signals[mask_high] = 0
        reasons[mask_high] = "BLOCKED:vol_too_high"

        return signals, reasons

    def _apply_rsi_filter(
        self, signals: pd.Series, reasons: pd.Series, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """
        Avoid buying into overbought conditions or selling into oversold.
        This is an anti-chasing filter.
        """
        if "rsi" not in df.columns:
            return signals, reasons

        overbought = df["rsi"] > self.cfg["rsi_overbought"]
        oversold = df["rsi"] < self.cfg["rsi_oversold"]

        mask_long_ob = (signals == 1) & overbought
        mask_short_os = (signals == -1) & oversold

        signals[mask_long_ob] = 0
        reasons[mask_long_ob] = "BLOCKED:rsi_overbought"

        signals[mask_short_os] = 0
        reasons[mask_short_os] = "BLOCKED:rsi_oversold"

        return signals, reasons

    def _apply_regime_filter(
        self, signals: pd.Series, reasons: pd.Series, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Only trade when ADX indicates a trending market."""
        if "adx" not in df.columns:
            return signals, reasons

        weak_trend = df["adx"] < self.cfg["adx_trending_threshold"]
        mask_blocked = (signals != 0) & weak_trend

        signals[mask_blocked] = 0
        reasons[mask_blocked] = "BLOCKED:weak_trend(ADX)"

        return signals, reasons

    def _apply_cooldown(self, signals: pd.Series) -> pd.Series:
        """
        Enforce a minimum number of bars between consecutive signals.
        Prevents rapid-fire signals during choppy periods.
        """
        cooldown = self.cfg["min_bars_between_trades"]
        result = signals.copy()
        last_trade_bar = -cooldown - 1

        for i, (idx, val) in enumerate(result.items()):
            if val != 0:
                if (i - last_trade_bar) < cooldown:
                    result[idx] = 0
                else:
                    last_trade_bar = i

        return result

    # ─── Diagnostics ──────────────────────────────────────────────────────────

    def _log_summary(self, out: pd.DataFrame):
        total = len(out)
        n_long = (out["signal"] == 1).sum()
        n_short = (out["signal"] == -1).sum()
        n_flat = (out["signal"] == 0).sum()
        n_raw_active = (out["raw_signal"] != 0).sum()
        n_filtered = n_raw_active - (n_long + n_short)
        logger.info(
            "Signals → Long: %d | Short: %d | Flat: %d | Filtered out: %d / %d",
            n_long, n_short, n_flat, n_filtered, total
        )

    def filter_breakdown(self, out: pd.DataFrame) -> pd.Series:
        """Return counts of each filter rejection reason."""
        blocked = out[out["signal"] == 0]["signal_reason"]
        return blocked.value_counts()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import download_data, split_data
    from features import build_features
    from model import FXModel

    df = download_data()
    train_raw, test_raw = split_data(df)
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)

    model = FXModel()
    model.fit(train_feat)
    proba_df = model.get_signal_probabilities(test_feat)

    gen = SignalGenerator()
    signals_df = gen.generate(test_feat, proba_df)

    print("\n=== Signal Breakdown ===")
    print(signals_df["signal"].value_counts())
    print("\n=== Filter Breakdown ===")
    print(gen.filter_breakdown(signals_df))
    print("\nSample signals:")
    active = signals_df[signals_df["signal"] != 0]
    print(active[["signal", "prob_long", "prob_short", "signal_reason"]].head(10))
