"""
Feature Engineering – computes technical indicators and ML-ready features.
"""

import logging
import numpy as np
import pandas as pd

from config import FEATURE_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)


# ─── Low-level indicator helpers ──────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast: int, slow: int, signal: int
          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger_bands(series: pd.Series, period: int, num_std: float
                     ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(series, period)
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int, d_period: int) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    typical = (high + low + close) / 3
    sma_tp = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical - sma_tp) / (0.015 * mad.replace(0, np.nan))

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    return obv

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average Directional Index (ADX) – measures trend strength."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr14 = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


# ─── Main feature builder ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, cfg: dict | None = None, include_target: bool = True) -> pd.DataFrame:
    """
    Compute all technical indicators and return a feature-enriched DataFrame.

    Parameters
    ----------
    df  : OHLCV DataFrame (columns: Open, High, Low, Close, Volume)
    cfg : Optional override for FEATURE_CONFIG

    Returns
    -------
    DataFrame with original OHLCV columns plus all feature columns.
    NaN rows at the start are dropped.
    """
    cfg = cfg or FEATURE_CONFIG
    f = df.copy()
    c = f["Close"]
    h = f["High"]
    lo = f["Low"]
    v = f["Volume"]

    # ── Moving averages ──
    for p in cfg["sma_periods"]:
        f[f"sma_{p}"] = _sma(c, p)
        f[f"price_vs_sma_{p}"] = (c - _sma(c, p)) / _sma(c, p)

    for p in cfg["ema_periods"]:
        f[f"ema_{p}"] = _ema(c, p)

    # MA crossover signals
    f["sma_10_20_cross"] = f["sma_10"] - f["sma_20"]
    f["sma_20_50_cross"] = f["sma_20"] - f["sma_50"]
    f["ema_12_26_cross"] = f["ema_12"] - f["ema_26"]

    # ── RSI ──
    f["rsi"] = _rsi(c, cfg["rsi_period"])
    f["rsi_overbought"] = (f["rsi"] > 70).astype(int)
    f["rsi_oversold"] = (f["rsi"] < 30).astype(int)

    # ── MACD ──
    macd_line, signal_line, histogram = _macd(c, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    f["macd"] = macd_line
    f["macd_signal"] = signal_line
    f["macd_hist"] = histogram
    f["macd_cross"] = (macd_line > signal_line).astype(int)

    # ── Bollinger Bands ──
    upper, mid, lower = _bollinger_bands(c, cfg["bb_period"], cfg["bb_std"])
    f["bb_upper"] = upper
    f["bb_mid"] = mid
    f["bb_lower"] = lower
    f["bb_width"] = (upper - lower) / mid
    f["bb_position"] = (c - lower) / (upper - lower).replace(0, np.nan)

    # ── ATR ──
    f["atr"] = _atr(h, lo, c, cfg["atr_period"])
    f["atr_pct"] = f["atr"] / c          # normalised ATR

    # ── Stochastic ──
    stoch_k, stoch_d = _stochastic(h, lo, c, cfg["stoch_k"], cfg["stoch_d"])
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_d
    f["stoch_cross"] = (stoch_k > stoch_d).astype(int)

    # ── CCI ──
    f["cci"] = _cci(h, lo, c, cfg["cci_period"])

    # ── Williams %R ──
    f["williams_r"] = _williams_r(h, lo, c, cfg["williams_r_period"])

    # ── OBV ──
    if cfg["obv_enabled"]:
        obv = _obv(c, v)
        f["obv"] = obv
        f["obv_ema"] = _ema(obv, 20)
        f["obv_trend"] = (f["obv"] > f["obv_ema"]).astype(int)

    # ── ADX ──
    f["adx"] = _adx(h, lo, c, 14)
    f["is_trending"] = (f["adx"] > 25).astype(int)

    # ── Lagged returns ──
    for lag in cfg["lagged_returns"]:
        f[f"return_lag_{lag}"] = c.pct_change(lag)

    # ── Price position (high/low range) ──
    for p in [5, 10, 20]:
        rolling_high = h.rolling(p).max()
        rolling_low = lo.rolling(p).min()
        rng = (rolling_high - rolling_low).replace(0, np.nan)
        f[f"price_pos_{p}"] = (c - rolling_low) / rng

    # ── Volatility regime ──
    f["atr_rolling_mean"] = f["atr"].rolling(50).mean()
    f["vol_regime"] = f["atr"] / f["atr_rolling_mean"].replace(0, np.nan)

    # ── Time features ──
    f["hour"] = f.index.hour
    f["day_of_week"] = f.index.dayofweek
    f["is_london_session"] = ((f["hour"] >= 8) & (f["hour"] < 17)).astype(int)
    f["is_ny_session"] = ((f["hour"] >= 13) & (f["hour"] < 22)).astype(int)
    f["is_overlap"] = ((f["hour"] >= 13) & (f["hour"] < 17)).astype(int)

    if include_target:
        # ── Forward return (prediction target) ──
        horizon = cfg["forward_return_horizon"]
        future_return = c.pct_change(horizon).shift(-horizon)

        label_min_return_bps = cfg.get("label_min_return_bps")
        if label_min_return_bps is None:
            round_trip_fee_bps = float(RISK_CONFIG.get("trading_fee_bps_per_side", 0.0)) * 2.0
            label_edge_bps = float(cfg.get("label_edge_bps", 0.0))
            label_min_return_bps = round_trip_fee_bps + label_edge_bps

        label_threshold = float(label_min_return_bps) / 10_000.0

        f["target_return"] = future_return
        f["target_class"] = np.where(
            future_return > label_threshold,
            1,
            np.where(future_return < -label_threshold, -1, 0),
        )

    before = len(f)
    f.dropna(inplace=True)
    logger.info("Built features: %d rows (dropped %d NaN rows)", len(f), before - len(f))
    return f


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes OHLCV, targets, derived raw)."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "target_return", "target_class",
        "bb_upper", "bb_mid", "bb_lower",   # raw band levels (prices)
        "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "macd", "macd_signal",
        "atr_rolling_mean",
        "obv", "obv_ema",
    }
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import download_data
    df = download_data()
    features = build_features(df)
    cols = get_feature_columns(features)
    print(f"Feature columns ({len(cols)}): {cols}")
    print(features[cols].tail(3))
