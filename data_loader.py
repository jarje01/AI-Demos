"""
Data Loader – downloads and caches EUR/USD historical FX data via yfinance.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

from config import DATA_CONFIG, PATHS, PREDICTION_MARKET_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

POLYMARKET_API_URL = "https://gamma-api.polymarket.com/markets"


def _load_local_env(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into process env (if unset)."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        logger.warning("Could not read %s: %s", env_path, exc)


def _resolve_end_date(end_value: str | None) -> str:
    """Resolve dynamic end-date config values into an ISO date string."""
    if end_value is None:
        return date.today().isoformat()
    if isinstance(end_value, str) and end_value.strip().lower() == "auto":
        return date.today().isoformat()
    return end_value


def _cache_path(symbol: str, interval: str) -> str:
    os.makedirs(PATHS["data_dir"], exist_ok=True)
    return os.path.join(PATHS["data_dir"], f"{symbol.replace('=','_')}_{interval}.parquet")


def _to_oanda_instrument(symbol: str) -> str:
    if "_" in symbol:
        return symbol
    core = symbol.replace("=X", "")
    if len(core) == 6:
        return f"{core[:3]}_{core[3:]}"
    return symbol


def _interval_to_oanda_granularity(interval: str) -> str:
    mapping = {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1d": "D",
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported OANDA interval: {interval}")
    return mapping[interval]


def _bars_per_day(interval: str) -> int:
    return {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }.get(interval, 24)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _effective_snapshot_timestamp(interval: str, now_utc: pd.Timestamp | None = None) -> pd.Timestamp:
    now_utc = now_utc or pd.Timestamp.now(tz="UTC")
    floor_map = {
        "1m": "min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "h",
        "4h": "4h",
        "1d": "D",
    }
    freq = floor_map.get(interval, "h")
    return now_utc.floor(freq).tz_convert(None)


def _extract_polymarket_yes_prices(market: dict) -> tuple[float | None, float | None, float | None]:
    outcomes = _parse_json_list(market.get("outcomes"))
    prices = _parse_json_list(market.get("outcomePrices"))
    if len(outcomes) < 2 or len(outcomes) != len(prices):
        return None, None, None

    yes_idx = None
    no_idx = None
    for idx, outcome in enumerate(outcomes):
        text = str(outcome).strip().lower()
        if text == "yes":
            yes_idx = idx
        elif text == "no":
            no_idx = idx

    if yes_idx is None or no_idx is None:
        return None, None, None

    yes_price = _safe_float(prices[yes_idx], default=-1.0)
    no_price = _safe_float(prices[no_idx], default=-1.0)
    if not (0.0 <= yes_price <= 1.0 and 0.0 <= no_price <= 1.0):
        return None, None, None

    best_bid = _safe_float(market.get("bestBid"), yes_price)
    best_ask = _safe_float(market.get("bestAsk"), yes_price)
    if not (0.0 <= best_bid <= 1.0 and 0.0 <= best_ask <= 1.0):
        best_bid, best_ask = yes_price, yes_price

    mid = (best_bid + best_ask) / 2.0 if best_ask >= best_bid else yes_price
    spread = max(0.0, best_ask - best_bid)
    return mid, spread, no_price


def _fetch_polymarket_active_markets(max_markets: int = 150, page_size: int = 200) -> list[dict]:
    rows: list[dict] = []
    offset = 0

    while len(rows) < max_markets:
        params = {
            "limit": min(page_size, max_markets - len(rows)),
            "offset": offset,
            "active": True,
            "closed": False,
            "archived": False,
        }
        resp = requests.get(POLYMARKET_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        offset += len(chunk)
        if len(chunk) < params["limit"]:
            break

    return rows


def _build_polymarket_snapshot(interval: str) -> dict:
    cfg = PREDICTION_MARKET_CONFIG
    max_markets = int(cfg.get("max_markets", 150))
    min_liquidity = float(cfg.get("min_liquidity", 1000.0))
    min_volume = float(cfg.get("min_volume", 1000.0))

    markets = _fetch_polymarket_active_markets(max_markets=max_markets)
    rows: list[dict] = []
    for market in markets:
        mid, spread, _ = _extract_polymarket_yes_prices(market)
        if mid is None:
            continue

        liquidity = _safe_float(market.get("liquidityNum", market.get("liquidity")), 0.0)
        volume = _safe_float(market.get("volumeNum", market.get("volume")), 0.0)
        if liquidity < min_liquidity or volume < min_volume:
            continue

        rows.append(
            {
                "yes_mid": mid,
                "spread": spread,
                "one_hour_change": _safe_float(market.get("oneHourPriceChange"), 0.0),
                "one_day_change": _safe_float(market.get("oneDayPriceChange"), 0.0),
                "liquidity": liquidity,
                "volume": volume,
                "is_active": bool(market.get("active", False)),
            }
        )

    asof_ts = _effective_snapshot_timestamp(interval)
    if not rows:
        return {
            "asof_timestamp": asof_ts,
            "pm_sample_count": 0,
            "pm_active_ratio": 0.0,
            "pm_yes_mid_mean": 0.5,
            "pm_yes_mid_std": 0.0,
            "pm_change_1h_mean": 0.0,
            "pm_change_24h_mean": 0.0,
            "pm_spread_mean_bps": 0.0,
            "pm_liquidity_median": 0.0,
            "pm_volume_median": 0.0,
        }

    snap = pd.DataFrame(rows)
    return {
        "asof_timestamp": asof_ts,
        "pm_sample_count": int(len(snap)),
        "pm_active_ratio": float(snap["is_active"].mean()),
        "pm_yes_mid_mean": float(snap["yes_mid"].mean()),
        "pm_yes_mid_std": float(snap["yes_mid"].std(ddof=0)),
        "pm_change_1h_mean": float(snap["one_hour_change"].mean()),
        "pm_change_24h_mean": float(snap["one_day_change"].mean()),
        "pm_spread_mean_bps": float(snap["spread"].mean() * 10_000.0),
        "pm_liquidity_median": float(snap["liquidity"].median()),
        "pm_volume_median": float(snap["volume"].median()),
    }


def _append_polymarket_snapshot(snapshot: dict) -> None:
    snapshot_file = str(PREDICTION_MARKET_CONFIG.get("snapshot_file", "data/polymarket_macro_snapshots.csv"))
    os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
    row = pd.DataFrame([snapshot])
    if os.path.exists(snapshot_file):
        row.to_csv(snapshot_file, mode="a", index=False, header=False)
    else:
        row.to_csv(snapshot_file, index=False)


def collect_polymarket_snapshot(interval: str | None = None) -> dict:
    """Fetch and persist one Polymarket macro snapshot row, returning the saved payload."""
    interval = interval or DATA_CONFIG.get("interval", "1h")
    snapshot = _build_polymarket_snapshot(interval=interval)
    _append_polymarket_snapshot(snapshot)
    logger.info("Polymarket snapshot updated (%s, sample=%s)", snapshot["asof_timestamp"], snapshot["pm_sample_count"])
    return snapshot


def _load_polymarket_snapshots() -> pd.DataFrame:
    snapshot_file = str(PREDICTION_MARKET_CONFIG.get("snapshot_file", "data/polymarket_macro_snapshots.csv"))
    if not os.path.exists(snapshot_file):
        return pd.DataFrame()

    df = pd.read_csv(snapshot_file)
    if df.empty or "asof_timestamp" not in df.columns:
        return pd.DataFrame()

    df["asof_timestamp"] = pd.to_datetime(df["asof_timestamp"], errors="coerce")
    df = df.dropna(subset=["asof_timestamp"]).sort_values("asof_timestamp")
    df = df.drop_duplicates(subset=["asof_timestamp"], keep="last")
    return df


def cleanup_polymarket_snapshots(keep_days: int = 90) -> dict:
    """Deduplicate and prune old Polymarket snapshots, returning cleanup stats."""
    snapshot_file = str(PREDICTION_MARKET_CONFIG.get("snapshot_file", "data/polymarket_macro_snapshots.csv"))
    if not os.path.exists(snapshot_file):
        return {
            "snapshot_file": snapshot_file,
            "before_rows": 0,
            "after_rows": 0,
            "removed_rows": 0,
            "keep_days": int(keep_days),
        }

    raw = pd.read_csv(snapshot_file)
    before = int(len(raw))
    if before == 0 or "asof_timestamp" not in raw.columns:
        return {
            "snapshot_file": snapshot_file,
            "before_rows": before,
            "after_rows": before,
            "removed_rows": 0,
            "keep_days": int(keep_days),
        }

    cleaned = raw.copy()
    cleaned["asof_timestamp"] = pd.to_datetime(cleaned["asof_timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["asof_timestamp"]).sort_values("asof_timestamp")
    cleaned = cleaned.drop_duplicates(subset=["asof_timestamp"], keep="last")

    keep_days = max(1, int(keep_days))
    cutoff = pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(days=keep_days)
    cleaned = cleaned[cleaned["asof_timestamp"] >= cutoff]

    cleaned.to_csv(snapshot_file, index=False)
    after = int(len(cleaned))
    removed = before - after

    logger.info(
        "Polymarket snapshot cleanup complete: before=%d after=%d removed=%d keep_days=%d",
        before,
        after,
        removed,
        keep_days,
    )
    return {
        "snapshot_file": snapshot_file,
        "before_rows": before,
        "after_rows": after,
        "removed_rows": removed,
        "keep_days": keep_days,
    }


def add_polymarket_features(df: pd.DataFrame, interval: str | None = None) -> pd.DataFrame:
    interval = interval or DATA_CONFIG.get("interval", "1h")
    cfg = PREDICTION_MARKET_CONFIG
    if not bool(cfg.get("enabled", False)):
        return df
    if str(cfg.get("provider", "polymarket")).lower() != "polymarket":
        return df

    if bool(cfg.get("refresh_snapshot_on_load", True)):
        try:
            collect_polymarket_snapshot(interval=interval)
        except Exception as exc:
            logger.warning("Polymarket snapshot refresh failed: %s", exc)

    snapshots = _load_polymarket_snapshots()
    if snapshots.empty:
        logger.info("No Polymarket snapshot history found; returning neutral PM features.")
        out = df.copy()
        out["pm_sample_count"] = 0.0
        out["pm_active_ratio"] = 0.0
        out["pm_yes_mid_mean"] = 0.5
        out["pm_yes_mid_std"] = 0.0
        out["pm_change_1h_mean"] = 0.0
        out["pm_change_24h_mean"] = 0.0
        out["pm_spread_mean_bps"] = 0.0
        out["pm_liquidity_median"] = 0.0
        out["pm_volume_median"] = 0.0
        out["pm_snapshot_age_hours"] = 999.0
        out["pm_data_available"] = 0
        return out

    left = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    left["timestamp"] = pd.to_datetime(left["timestamp"], errors="coerce")
    left = left.sort_values("timestamp")

    right = snapshots.sort_values("asof_timestamp")
    tolerance = pd.Timedelta(hours=float(cfg.get("merge_tolerance_hours", 168)))

    merged = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="asof_timestamp",
        direction="backward",
        tolerance=tolerance,
    )

    merged["pm_data_available"] = merged["asof_timestamp"].notna().astype(int)
    age_hours = (merged["timestamp"] - merged["asof_timestamp"]).dt.total_seconds() / 3600.0
    merged["pm_snapshot_age_hours"] = age_hours

    merged["pm_sample_count"] = merged["pm_sample_count"].fillna(0.0)
    merged["pm_active_ratio"] = merged["pm_active_ratio"].fillna(0.0)
    merged["pm_yes_mid_mean"] = merged["pm_yes_mid_mean"].fillna(0.5)
    merged["pm_yes_mid_std"] = merged["pm_yes_mid_std"].fillna(0.0)
    merged["pm_change_1h_mean"] = merged["pm_change_1h_mean"].fillna(0.0)
    merged["pm_change_24h_mean"] = merged["pm_change_24h_mean"].fillna(0.0)
    merged["pm_spread_mean_bps"] = merged["pm_spread_mean_bps"].fillna(0.0)
    merged["pm_liquidity_median"] = merged["pm_liquidity_median"].fillna(0.0)
    merged["pm_volume_median"] = merged["pm_volume_median"].fillna(0.0)
    merged["pm_snapshot_age_hours"] = merged["pm_snapshot_age_hours"].fillna(999.0)

    merged = merged.drop(columns=["asof_timestamp"])
    merged = merged.set_index("timestamp")
    merged.index.name = df.index.name
    return merged


def _download_oanda_data(
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    _load_local_env()
    token = os.getenv("OANDA_API_KEY")
    if not token:
        raise RuntimeError("OANDA_API_KEY env var is required when DATA_CONFIG['provider'] == 'oanda'.")

    env_name = str(DATA_CONFIG.get("oanda_environment", "practice")).lower()
    if env_name == "live":
        base_url = "https://api-fxtrade.oanda.com/v3"
    else:
        base_url = "https://api-fxpractice.oanda.com/v3"

    instrument = DATA_CONFIG.get("oanda_instrument") or _to_oanda_instrument(symbol)
    granularity = _interval_to_oanda_granularity(interval)
    price = str(DATA_CONFIG.get("oanda_price", "M")).upper()

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    if "T" not in str(end):
        end_dt = end_dt + pd.Timedelta(days=1)

    now_utc = pd.Timestamp.now(tz="UTC")
    if end_dt > now_utc:
        end_dt = now_utc

    max_candles = 4500
    days_per_chunk = max(1, int(max_candles / _bars_per_day(interval)))

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }

    rows: list[dict] = []
    cursor = start_dt
    endpoint = f"{base_url}/instruments/{instrument}/candles"

    while cursor < end_dt:
        chunk_end = min(cursor + pd.Timedelta(days=days_per_chunk), end_dt)
        params = {
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
            "price": price,
        }

        resp = requests.get(endpoint, headers=headers, params=params, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"OANDA API error {resp.status_code}: {resp.text[:400]}")

        payload = resp.json()
        candles = payload.get("candles", [])
        for candle in candles:
            if not candle.get("complete", True):
                continue
            mid = candle.get("mid")
            if not mid:
                continue
            rows.append(
                {
                    "timestamp": pd.to_datetime(candle["time"], utc=True),
                    "Open": float(mid["o"]),
                    "High": float(mid["h"]),
                    "Low": float(mid["l"]),
                    "Close": float(mid["c"]),
                    "Volume": float(candle.get("volume", 0.0)),
                }
            )

        cursor = chunk_end

    if not rows:
        raise RuntimeError("OANDA returned no candles for requested range.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index).tz_convert(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_latest_feed_quote(symbol: str | None = None, interval: str | None = None) -> dict | None:
    """Return latest feed quote metadata (bid/ask/spread) when available."""
    provider = str(DATA_CONFIG.get("provider", "yahoo")).lower()
    if provider != "oanda":
        return None

    _load_local_env()
    token = os.getenv("OANDA_API_KEY")
    if not token:
        return None

    symbol = symbol or DATA_CONFIG["symbol"]
    interval = interval or DATA_CONFIG["interval"]

    env_name = str(DATA_CONFIG.get("oanda_environment", "practice")).lower()
    if env_name == "live":
        base_url = "https://api-fxtrade.oanda.com/v3"
    else:
        base_url = "https://api-fxpractice.oanda.com/v3"

    instrument = DATA_CONFIG.get("oanda_instrument") or _to_oanda_instrument(symbol)
    granularity = _interval_to_oanda_granularity(interval)
    endpoint = f"{base_url}/instruments/{instrument}/candles"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Datetime-Format": "RFC3339",
    }
    params = {
        "count": 2,
        "granularity": granularity,
        "price": "BA",
    }

    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=20)
        if resp.status_code >= 400:
            logger.warning("OANDA quote request failed (%s): %s", resp.status_code, resp.text[:200])
            return None
        payload = resp.json()
        candles = payload.get("candles", [])
        if not candles:
            return None

        complete = [c for c in candles if c.get("complete", True)]
        candle = complete[-1] if complete else candles[-1]

        bid = candle.get("bid")
        ask = candle.get("ask")
        if not bid or not ask:
            return None

        bid_px = float(bid["c"])
        ask_px = float(ask["c"])
        spread = ask_px - bid_px

        return {
            "provider": "oanda",
            "quote_timestamp": str(pd.to_datetime(candle["time"], utc=True).tz_convert(None)),
            "sell_bid": bid_px,
            "buy_ask": ask_px,
            "spread": spread,
            "spread_pips": spread * 10_000.0,
        }
    except Exception as exc:
        logger.warning("Failed to fetch latest feed quote: %s", exc)
        return None


def download_data(
    symbol: str | None = None,
    interval: str | None = None,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for *symbol* at *interval* resolution.

    Falls back to synthetic data when yfinance is unavailable (for offline
    testing / CI environments).

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is a timezone-naive DatetimeIndex.
    """
    symbol = symbol or DATA_CONFIG["symbol"]
    interval = interval or DATA_CONFIG["interval"]
    provider = str(DATA_CONFIG.get("provider", "yahoo")).lower()
    start = start or DATA_CONFIG["train_start"]
    raw_end = end if end is not None else DATA_CONFIG["test_end"]
    end_is_dynamic = (raw_end is None) or (isinstance(raw_end, str) and raw_end.strip().lower() == "auto")
    end = _resolve_end_date(raw_end)

    # yfinance `end` is exclusive; for intraday + dynamic end we extend by 1 day
    # so today's bars are included in the response window.
    yf_end = end
    if end_is_dynamic and interval in {"1h", "30m", "15m", "5m", "1m"}:
        yf_end = (pd.to_datetime(end).date() + timedelta(days=1)).isoformat()

    # Yahoo intraday limits: clamp start into supported lookback window.
    # 1h bars are only available for roughly the last 730 days.
    if interval == "1h":
        end_dt = pd.to_datetime(end).date()
        min_start_dt = end_dt - timedelta(days=729)
        start_dt = pd.to_datetime(start).date()
        if start_dt < min_start_dt:
            logger.info(
                "Clamping start date for %s from %s to %s (Yahoo lookback limit).",
                interval,
                start_dt,
                min_start_dt,
            )
            start = min_start_dt.isoformat()

    cache = _cache_path(symbol, interval)

    if use_cache and os.path.exists(cache):
        logger.info("Loading cached data from %s", cache)
        df = pd.read_parquet(cache)
        if end_is_dynamic and not df.empty:
            latest_cached = pd.to_datetime(df.index).max().date()
            if latest_cached >= date.today():
                return df.loc[start:end]
            logger.info("Cache is stale (latest=%s, today=%s) – refreshing from source.", latest_cached, date.today())
        else:
            return df.loc[start:end]

    if provider == "oanda":
        logger.info("Downloading %s (%s) from OANDA: %s to %s …", symbol, interval, start, end)
        raw = _download_oanda_data(symbol=symbol, interval=interval, start=start, end=end)
    else:
        if yf is None:
            logger.warning("yfinance not installed – generating synthetic EUR/USD data.")
            return _synthetic_data(start, end, interval)

        logger.info("Downloading %s (%s) from %s to %s …", symbol, interval, start, yf_end)
        raw = yf.download(symbol, start=start, end=yf_end, interval=interval, progress=False, auto_adjust=True)

        if raw.empty:
            logger.warning("yfinance returned empty data – falling back to synthetic data.")
            return _synthetic_data(start, end, interval)

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    raw.to_parquet(cache)
    logger.info("Saved %d rows to %s", len(raw), cache)

    return raw.loc[start:end]


def _synthetic_data(start: str, end: str, interval: str) -> pd.DataFrame:
    """Generate realistic-looking synthetic EUR/USD OHLCV data."""
    freq_map = {"1h": "h", "1d": "D", "30m": "30min", "15m": "15min", "4h": "4h"}
    freq = freq_map.get(interval, "h")

    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)

    rng = np.random.default_rng(42)

    # Simulate a mean-reverting EUR/USD price path around 1.10
    returns = rng.normal(0, 0.0008, n)           # ~8 pip hourly vol
    log_price = np.cumsum(returns) + np.log(1.10)
    close = np.exp(log_price)

    spread = rng.uniform(0.00005, 0.0002, n)
    high = close + rng.uniform(0, 0.001, n)
    low = close - rng.uniform(0, 0.001, n)
    open_ = close + rng.normal(0, 0.0003, n)
    volume = rng.integers(500, 5000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)

    # Remove weekend rows (FX market closed Sat–Sun)
    df = df[df.index.dayofweek < 5].copy()
    logger.info("Generated %d bars of synthetic data (%s to %s).", len(df), start, end)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and out-of-sample test sets."""
    train_end = _resolve_end_date(DATA_CONFIG["train_end"])
    test_end = _resolve_end_date(DATA_CONFIG["test_end"])
    train = df.loc[DATA_CONFIG["train_start"]: train_end]
    test = df.loc[DATA_CONFIG["test_start"]: test_end]
    logger.info("Train: %d rows (%s – %s)", len(train),
                train.index[0].date(), train.index[-1].date())
    logger.info("Test:  %d rows (%s – %s)", len(test),
                test.index[0].date(), test.index[-1].date())
    return train, test


if __name__ == "__main__":
    df = download_data()
    print(df.tail())
    print(f"\nShape: {df.shape}")
