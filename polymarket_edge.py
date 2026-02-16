from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

API_URL = "https://gamma-api.polymarket.com/markets"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"


@dataclass
class EdgeConfig:
    fee_bps_round_trip: float = 100.0
    min_volume: float = 25000.0
    min_liquidity: float = 1500.0
    reversion_strength: float = 1.0
    max_kelly_fraction: float = 0.02
    max_fair_deviation_bps: float = 300.0
    max_edge_bps: float = 400.0
    max_bet_usd: float = 50.0


def _parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_yes_no_prices(market: dict) -> tuple[float | None, float | None]:
    outcomes = _parse_list(market.get("outcomes"))
    prices = _parse_list(market.get("outcomePrices"))
    if len(outcomes) != len(prices) or len(outcomes) < 2:
        return None, None

    yes_idx = None
    no_idx = None
    for idx, outcome in enumerate(outcomes):
        text = str(outcome).strip().lower()
        if text == "yes":
            yes_idx = idx
        elif text == "no":
            no_idx = idx

    if yes_idx is None or no_idx is None:
        return None, None

    yes_price = _to_float(prices[yes_idx], default=-1)
    no_price = _to_float(prices[no_idx], default=-1)
    if not (0 <= yes_price <= 1) or not (0 <= no_price <= 1):
        return None, None
    return yes_price, no_price


def _extract_yes_no_prices_kalshi(market: dict) -> tuple[float | None, float | None]:
    yes_ask = _to_float(market.get("yes_ask"), -1)
    no_ask = _to_float(market.get("no_ask"), -1)
    if yes_ask < 0 or no_ask < 0:
        return None, None
    yes_price = yes_ask / 100.0
    no_price = no_ask / 100.0
    if not (0 <= yes_price <= 1 and 0 <= no_price <= 1):
        return None, None
    return yes_price, no_price


def fetch_markets(max_markets: int = 400, page_size: int = 200) -> list[dict]:
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
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        offset += len(chunk)
        if len(chunk) < params["limit"]:
            break

    return rows


def fetch_markets_kalshi(max_markets: int = 400, page_size: int = 200) -> list[dict]:
    rows: list[dict] = []
    cursor: str | None = None

    while len(rows) < max_markets:
        params = {
            "limit": min(page_size, max_markets - len(rows)),
            "status": "open",
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(KALSHI_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        chunk = payload.get("markets", []) if isinstance(payload, dict) else []
        if not chunk:
            break

        rows.extend(chunk)
        cursor = payload.get("cursor") if isinstance(payload, dict) else None
        if not cursor or len(chunk) < params["limit"]:
            break

    return rows


def load_priors(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Priors file not found: {file_path}")

    df = pd.read_csv(file_path)
    required = {"market_id", "fair_yes_prob"}
    if not required.issubset(set(df.columns)):
        raise ValueError("Priors CSV must include columns: market_id,fair_yes_prob")

    priors: dict[str, float] = {}
    for _, row in df.iterrows():
        market_id = str(row["market_id"])
        prob = float(row["fair_yes_prob"])
        if 0 <= prob <= 1:
            priors[market_id] = prob
    return priors


def _kelly_fraction(prob_win: float, price: float) -> float:
    if not (0 < prob_win < 1 and 0 < price < 1):
        return 0.0
    b = (1 - price) / price
    q = 1 - prob_win
    raw = (b * prob_win - q) / b
    return max(0.0, raw)


def _parse_utc_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def score_market(
    market: dict,
    cfg: EdgeConfig,
    priors: dict[str, float],
    provider: Literal["polymarket", "kalshi"] = "polymarket",
) -> dict | None:
    now_utc = pd.Timestamp.now(tz="UTC")

    if provider == "kalshi":
        yes_price, no_price = _extract_yes_no_prices_kalshi(market)
    else:
        yes_price, no_price = _extract_yes_no_prices(market)
    if yes_price is None or no_price is None:
        return None

    if provider == "kalshi":
        market_id = str(market.get("ticker", ""))
        question = str(market.get("title", market.get("subtitle", "")))
        slug = market_id
        volume = _to_float(market.get("volume"), 0.0)
        liquidity = _to_float(market.get("liquidity"), 0.0)
        end_time_raw = market.get("close_time")
        status = str(market.get("status", "")).lower()
        is_active = status == "open"
        is_closed = status in {"closed", "settled", "finalized", "resolved"}
        accepting_orders = is_active
    else:
        market_id = str(market.get("id", ""))
        question = str(market.get("question", ""))
        slug = str(market.get("slug", ""))
        volume = _to_float(market.get("volumeNum", market.get("volume")), 0.0)
        liquidity = _to_float(market.get("liquidityNum", market.get("liquidity")), 0.0)
        end_time_raw = market.get("endDate")
        is_active = bool(market.get("active", False))
        is_closed = bool(market.get("closed", False))
        accepting_orders = bool(market.get("acceptingOrders", False))

    end_ts = _parse_utc_timestamp(end_time_raw)
    if end_ts is not None:
        days_to_end = (end_ts - now_utc).total_seconds() / 86400.0
        end_time_utc = end_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        days_to_end = None
        end_time_utc = ""

    if volume < cfg.min_volume or liquidity < cfg.min_liquidity:
        return None

    if provider == "kalshi":
        best_bid = _to_float(market.get("yes_bid"), yes_price * 100.0) / 100.0
        best_ask = _to_float(market.get("yes_ask"), yes_price * 100.0) / 100.0
    else:
        best_bid = _to_float(market.get("bestBid"), yes_price)
        best_ask = _to_float(market.get("bestAsk"), yes_price)

    if not (0 <= best_bid <= 1 and 0 <= best_ask <= 1):
        best_bid = yes_price
        best_ask = yes_price

    mid = (best_bid + best_ask) / 2 if best_ask >= best_bid else yes_price

    if provider == "kalshi":
        previous_price = _to_float(market.get("previous_price"), _to_float(market.get("last_price"), 0.0)) / 100.0
        last_price = _to_float(market.get("last_price"), previous_price * 100.0) / 100.0
        one_hour_change = last_price - previous_price
        one_day_change = one_hour_change
    else:
        one_hour_change = _to_float(market.get("oneHourPriceChange"), 0.0)
        one_day_change = _to_float(market.get("oneDayPriceChange"), 0.0)

    if market_id in priors:
        fair_yes = float(priors[market_id])
        method = "custom_prior"
    else:
        momentum = 0.6 * one_hour_change + 0.4 * one_day_change
        fair_yes = mid - cfg.reversion_strength * momentum
        fair_yes = max(0.01, min(0.99, fair_yes))

        liq_weight = min(1.0, liquidity / 10000.0)
        fair_yes = 0.5 + liq_weight * (fair_yes - 0.5)
        max_dev = max(0.0, cfg.max_fair_deviation_bps) / 10000.0
        lower = max(0.01, mid - max_dev)
        upper = min(0.99, mid + max_dev)
        fair_yes = max(lower, min(upper, fair_yes))
        method = "heuristic"

    fee = cfg.fee_bps_round_trip / 10000.0
    fair_minus_market_mid_bps = (fair_yes - mid) * 10000.0
    ev_yes = fair_yes - best_ask - fee
    ev_no = (1 - fair_yes) - no_price - fee

    side = "YES" if ev_yes >= ev_no else "NO"
    edge = ev_yes if side == "YES" else ev_no
    if edge <= 0:
        return None

    if side == "YES":
        kelly = _kelly_fraction(fair_yes, best_ask)
        entry_price = best_ask
        prob_win = fair_yes
    else:
        kelly = _kelly_fraction(1 - fair_yes, no_price)
        entry_price = no_price
        prob_win = 1 - fair_yes

    kelly = min(kelly, cfg.max_kelly_fraction)
    spread = max(0.0, best_ask - best_bid)

    cap_for_scaling = max(cfg.max_kelly_fraction, 0.001)
    recommended_bet_usd = min(max(0.0, cfg.max_bet_usd), max(0.0, (kelly / cap_for_scaling) * cfg.max_bet_usd))

    capped_edge = min(edge, max(0.0, cfg.max_edge_bps) / 10000.0)
    edge_bps = capped_edge * 10000
    spread_bps = spread * 10000

    edge_component = min(60.0, max(0.0, edge_bps / 40.0))
    liq_component = min(25.0, max(0.0, liquidity / 400.0))
    spread_penalty = min(35.0, max(0.0, spread_bps / 20.0))
    action_score = max(0.0, min(100.0, edge_component + liq_component - spread_penalty))

    if action_score >= 70:
        action_bucket = "GREEN"
    elif action_score >= 45:
        action_bucket = "YELLOW"
    else:
        action_bucket = "RED"

    return {
        "market_id": market_id,
        "question": question,
        "slug": slug,
        "side": side,
        "entry_price": round(entry_price, 4),
        "fair_yes_prob": round(fair_yes, 4),
        "market_yes_mid": round(mid, 4),
        "fair_minus_market_mid_bps": round(fair_minus_market_mid_bps, 1),
        "edge_bps": round(edge_bps, 1),
        "ev_per_1usd": round(capped_edge, 4),
        "prob_win": round(prob_win, 4),
        "suggested_kelly_fraction": round(kelly, 4),
        "recommended_bet_usd": round(recommended_bet_usd, 2),
        "spread": round(spread, 4),
        "volume": round(volume, 2),
        "liquidity": round(liquidity, 2),
        "one_hour_change": round(one_hour_change, 4),
        "one_day_change": round(one_day_change, 4),
        "method": method,
        "action_score": round(action_score, 1),
        "action_bucket": action_bucket,
        "provider": provider,
        "end_time_utc": end_time_utc,
        "days_to_end": round(days_to_end, 2) if days_to_end is not None else None,
        "is_active": is_active,
        "is_closed": is_closed,
        "accepting_orders": accepting_orders,
        "url": f"https://kalshi.com/markets/{slug}" if provider == "kalshi" else f"https://polymarket.com/event/{slug}",
    }


def scan_edges(
    max_markets: int,
    top_n: int,
    cfg: EdgeConfig,
    priors_path: str | None,
    output_path: str,
    provider: Literal["polymarket", "kalshi"] = "polymarket",
) -> pd.DataFrame:
    if provider == "kalshi":
        markets = fetch_markets_kalshi(max_markets=max_markets)
    else:
        markets = fetch_markets(max_markets=max_markets)
    priors = load_priors(priors_path)

    scored: list[dict] = []
    for market in markets:
        row = score_market(market, cfg=cfg, priors=priors, provider=provider)
        if row is not None:
            scored.append(row)

    df = pd.DataFrame(scored)
    if df.empty:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    df = df.sort_values(["action_score", "edge_bps", "liquidity"], ascending=False).head(top_n)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediction market edge scanner (Polymarket/Kalshi)")
    parser.add_argument("--provider", choices=["polymarket", "kalshi"], default="polymarket")
    parser.add_argument("--max-markets", type=int, default=400)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-volume", type=float, default=25000)
    parser.add_argument("--min-liquidity", type=float, default=1500)
    parser.add_argument("--fee-bps-round-trip", type=float, default=100.0)
    parser.add_argument("--reversion-strength", type=float, default=1.0)
    parser.add_argument("--max-kelly-fraction", type=float, default=0.02)
    parser.add_argument("--max-fair-deviation-bps", type=float, default=300.0)
    parser.add_argument("--max-edge-bps", type=float, default=400.0)
    parser.add_argument("--max-bet-usd", type=float, default=50.0)
    parser.add_argument("--priors-csv", type=str, default=None)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.strip() if isinstance(args.output, str) else ""
    if not output_path:
        output_path = f"results/{args.provider}_edge_candidates.csv"

    cfg = EdgeConfig(
        fee_bps_round_trip=args.fee_bps_round_trip,
        min_volume=args.min_volume,
        min_liquidity=args.min_liquidity,
        reversion_strength=args.reversion_strength,
        max_kelly_fraction=args.max_kelly_fraction,
        max_fair_deviation_bps=args.max_fair_deviation_bps,
        max_edge_bps=args.max_edge_bps,
        max_bet_usd=args.max_bet_usd,
    )

    df = scan_edges(
        max_markets=args.max_markets,
        top_n=args.top,
        cfg=cfg,
        priors_path=args.priors_csv,
        output_path=output_path,
        provider=args.provider,
    )

    if df.empty:
        print("No positive-edge candidates found under current filters.")
        print(f"Saved empty result file to {output_path}")
        return

    cols = [
        "action_bucket",
        "action_score",
        "side",
        "days_to_end",
        "is_active",
        "is_closed",
        "fair_minus_market_mid_bps",
        "edge_bps",
        "recommended_bet_usd",
        "entry_price",
        "fair_yes_prob",
        "market_yes_mid",
        "suggested_kelly_fraction",
        "spread",
        "liquidity",
        "question",
    ]
    print(df[cols].to_string(index=False))
    print(f"\nSaved {len(df)} candidates to {output_path}")


if __name__ == "__main__":
    main()
