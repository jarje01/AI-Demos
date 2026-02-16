"""
Configuration settings for the FX Rules-Based ML Trading System
"""

# ─── Data Settings ────────────────────────────────────────────────────────────
DATA_CONFIG = {
    "provider": "oanda",            # 'yahoo' (default) or 'oanda'
    "symbol": "EURUSD=X",
    "interval": "1h",                  # intraday default (Yahoo supports recent history)
    "train_start": "2024-02-01",       # keep within ~2 years for 1h Yahoo availability
    "train_end": "2025-07-31",
    "test_start": "2025-08-01",
    "test_end": "auto",                # uses today's date when set to 'auto'

    # OANDA settings (used when provider == 'oanda')
    # API key is read from env var OANDA_API_KEY
    "oanda_environment": "practice",   # 'practice' or 'live'
    "oanda_instrument": "EUR_USD",
    "oanda_price": "M",                # M=mid, B=bid, A=ask
}

# ─── Feature Engineering ──────────────────────────────────────────────────────
FEATURE_CONFIG = {
    # Moving averages
    "sma_periods": [10, 20, 50, 200],
    "ema_periods": [12, 26],

    # Oscillators
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # Volatility
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,

    # Momentum
    "stoch_k": 14,
    "stoch_d": 3,
    "cci_period": 20,
    "williams_r_period": 14,

    # Volume / misc
    "obv_enabled": True,
    "lagged_returns": [1, 2, 3, 5, 10],
    "forward_return_horizon": 1,       # bars ahead to predict
    "label_edge_bps": 0.0,             # disable extra label edge for baseline recovery
    "label_min_return_bps": 1.0,       # approximate legacy 0.0001 threshold
}

# ─── Prediction Market Features (optional MVP) ───────────────────────────────
PREDICTION_MARKET_CONFIG = {
    "enabled": True,
    "provider": "polymarket",
    "refresh_snapshot_on_load": True,
    "snapshot_file": "data/polymarket_macro_snapshots.csv",
    "max_markets": 150,
    "min_liquidity": 1000.0,
    "min_volume": 1000.0,
    "merge_tolerance_hours": 168,  # one week
}

# ─── Model Settings ───────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "model_type": "random_forest",     # 'random_forest', 'gradient_boost', 'xgboost'
    "target_type": "classification",   # 'classification' (long/short/flat)
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 50,
    "random_state": 42,
    "n_jobs": -1,

    # Probability calibration
    "enable_calibration": False,
    "calibration_method": "sigmoid",  # 'sigmoid' (Platt) or 'isotonic'
    "calibration_fraction": 0.2,       # tail holdout of training set used for calibration
    "min_calibration_samples": 120,
    "calibration_cv_splits": 3,

    # XGBoost specific (used if model_type == 'xgboost')
    "xgb_learning_rate": 0.05,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8,

    # ML filter thresholds (used by SignalGenerator)
    # A trade is allowed only when directional probability has enough
    # absolute confidence and edge versus opposite/flat classes.
    "min_confidence": 0.335,
    "min_edge_over_opposite": 0.003,
    "min_edge_over_flat": -1.0,

    # Backward-compatible legacy threshold (fallback only)
    "signal_threshold": 0.55,
}

# ─── Rules-Based Filters ──────────────────────────────────────────────────────
RULES_CONFIG = {
    # Trend filter – only trade in direction of long-term trend
    "trend_filter_enabled": True,
    "trend_sma": 200,                  # price > SMA200 → bullish bias

    # Volatility filter – skip when ATR is too high or too low
    "volatility_filter_enabled": True,
    "atr_min_multiplier": 0.5,        # skip if ATR < 0.5 * rolling mean ATR
    "atr_max_multiplier": 2.5,        # skip if ATR > 2.5 * rolling mean ATR

    # RSI filter – avoid overbought/oversold extremes on entry
    "rsi_filter_enabled": True,
    "rsi_oversold": 30,
    "rsi_overbought": 70,

    # Regime filter (based on ADX)
    "regime_filter_enabled": True,
    "adx_period": 14,
    "adx_trending_threshold": 18,     # tight-sweep robustness candidate

    # Minimum time between trades (bars)
    "min_bars_between_trades": 2,
}

# ─── Strategy Profiles ───────────────────────────────────────────────────────
# Use with: python main.py --profile safe|aggressive
STRATEGY_PROFILES = {
    "safe": {
        "model": {
            "min_confidence": 0.335,
            "min_edge_over_opposite": 0.003,
        },
        "rules": {
            "adx_trending_threshold": 18,
        },
    },
    "balanced": {
        "model": {
            "min_confidence": 0.34,
            "min_edge_over_opposite": 0.008,
        },
        "rules": {
            "adx_trending_threshold": 16,
        },
    },
    "aggressive": {
        "model": {
            "min_confidence": 0.34,
            "min_edge_over_opposite": 0.003,
        },
        "rules": {
            "adx_trending_threshold": 20,
        },
    },
}

# ─── Risk Management ──────────────────────────────────────────────────────────
RISK_CONFIG = {
    "initial_capital": 100_000.0,
    "risk_per_trade_pct": 1.0,        # % of capital risked per trade
    "atr_stop_multiplier": 1.5,       # stop = entry ± ATR * multiplier
    "take_profit_multiplier": 2.0,    # TP at 2x ATR (risk:reward = 1:2)
    "trading_fee_bps_per_side": 1.5,  # 1.5 bps per entry and per exit
    "max_open_trades": 1,
    "max_drawdown_pct": 15.0,         # halt trading if DD exceeds this
    "trailing_stop_enabled": False,
}

# ─── Walk-Forward Settings ────────────────────────────────────────────────────
WALK_FORWARD_CONFIG = {
    "train_window_bars": 400,         # ~2-3 weeks on 1h bars (weekday FX hours)
    "test_window_bars": 120,          # ~1 week OOS on 1h bars
    "step_size_bars": 60,             # overlapping folds
    "min_trades_per_fold": 3,
    "retrain_frequency": "quarterly", # informational label
}

# ─── Paths ────────────────────────────────────────────────────────────────────
PATHS = {
    "data_dir": "data/",
    "models_dir": "models/",
    "results_dir": "results/",
    "plots_dir": "plots/",
}
