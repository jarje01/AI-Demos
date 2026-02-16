from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st

from polymarket_edge import EdgeConfig, scan_edges


MAX_BET_USD = 50.0


def _default_results(provider: str) -> Path:
    return Path("results") / f"{provider}_edge_candidates.csv"


def _default_watchlist(provider: str) -> Path:
    return Path("results") / f"{provider}_watchlist_log.csv"


def _run_scan(
    provider: Literal["polymarket", "kalshi"],
    max_markets: int,
    top_n: int,
    min_volume: float,
    min_liquidity: float,
    fee_bps_round_trip: float,
    reversion_strength: float,
    max_kelly_fraction: float,
    max_fair_deviation_bps: float,
    max_edge_bps: float,
    priors_csv: str | None,
    output_path: str,
) -> pd.DataFrame:
    cfg = EdgeConfig(
        fee_bps_round_trip=fee_bps_round_trip,
        min_volume=min_volume,
        min_liquidity=min_liquidity,
        reversion_strength=reversion_strength,
        max_kelly_fraction=max_kelly_fraction,
        max_fair_deviation_bps=max_fair_deviation_bps,
        max_edge_bps=max_edge_bps,
    )

    priors = priors_csv.strip() if priors_csv else None
    if not priors:
        priors = None

    return scan_edges(
        max_markets=max_markets,
        top_n=top_n,
        cfg=cfg,
        priors_path=priors,
        output_path=output_path,
        provider=provider,
    )


def _load_existing(path: str) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


def _append_watchlist(df: pd.DataFrame, top_n: int, path: Path) -> tuple[bool, str]:
    try:
        snapshot = df.head(top_n).copy()
        snapshot.insert(0, "saved_at_utc", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"))
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(path, mode="a", header=not path.exists(), index=False)
        return True, f"Saved top {len(snapshot)} picks to {path}"
    except Exception as exc:
        return False, f"Failed to save watchlist: {exc}"


def main() -> None:
    st.set_page_config(page_title="Prediction Market Edge Scanner", layout="wide")
    st.title("Prediction Market Edge Scanner")
    st.caption("Research tool for ranking potential prediction-market mispricings.")

    with st.sidebar:
        st.header("Scanner Settings")
        provider = st.selectbox("Provider", ["polymarket", "kalshi"], index=0)
        if provider == "kalshi":
            if st.button("Use looser Kalshi preset"):
                st.session_state["pm_max_markets"] = 500
                st.session_state["pm_top_n"] = 40
                st.session_state["pm_min_volume"] = 0.0
                st.session_state["pm_min_liquidity"] = 0.0
                st.session_state["pm_fee_bps"] = 100.0
                st.session_state["pm_reversion_strength"] = 0.8
                st.session_state["pm_max_kelly_fraction"] = 0.02
                st.session_state["pm_max_fair_deviation_bps"] = 500.0
                st.session_state["pm_max_edge_bps"] = 600.0
            st.caption("Discovery mode: higher recall, lower precision.")

        max_markets = st.slider("Max markets", min_value=50, max_value=1000, value=300, step=50, key="pm_max_markets")
        top_n = st.slider("Top candidates", min_value=5, max_value=100, value=25, step=5, key="pm_top_n")

        min_volume = st.number_input("Min volume", min_value=0.0, value=25000.0, step=5000.0, key="pm_min_volume")
        min_liquidity = st.number_input("Min liquidity", min_value=0.0, value=1500.0, step=250.0, key="pm_min_liquidity")
        fee_bps_round_trip = st.number_input("Round-trip fee (bps)", min_value=0.0, value=100.0, step=5.0, key="pm_fee_bps")

        reversion_strength = st.slider("Reversion strength", min_value=0.0, max_value=2.0, value=1.0, step=0.1, key="pm_reversion_strength")
        max_kelly_fraction = st.slider("Max Kelly fraction", min_value=0.001, max_value=0.10, value=0.02, step=0.001, key="pm_max_kelly_fraction")
        max_fair_deviation_bps = st.number_input(
            "Max fair deviation (bps)",
            min_value=25.0,
            max_value=3000.0,
            value=300.0,
            step=25.0,
            key="pm_max_fair_deviation_bps",
            help="Caps heuristic fair value distance from market mid; lower is more conservative.",
        )
        max_edge_bps = st.number_input(
            "Max edge cap (bps)",
            min_value=25.0,
            max_value=3000.0,
            value=400.0,
            step=25.0,
            key="pm_max_edge_bps",
            help="Clips computed edge before scoring and ranking to reduce outlier dominance.",
        )

        priors_csv = st.text_input(
            "Priors CSV (optional)",
            value="",
            help="CSV with columns: market_id,fair_yes_prob",
        )
        output_path = st.text_input("Output CSV", value=str(_default_results(provider)))

        run_scan = st.button("Run scanner")

    if run_scan:
        with st.spinner("Scanning active markets..."):
            try:
                df = _run_scan(
                    provider=provider,
                    max_markets=max_markets,
                    top_n=top_n,
                    min_volume=min_volume,
                    min_liquidity=min_liquidity,
                    fee_bps_round_trip=fee_bps_round_trip,
                    reversion_strength=reversion_strength,
                    max_kelly_fraction=max_kelly_fraction,
                    max_fair_deviation_bps=max_fair_deviation_bps,
                    max_edge_bps=max_edge_bps,
                    priors_csv=priors_csv,
                    output_path=output_path,
                )
                st.session_state["pm_df"] = df
                st.success(f"Scan complete. Saved to {output_path}")
            except Exception as exc:
                st.error(f"Scan failed: {exc}")

    df = st.session_state.get("pm_df")
    if df is None:
        df = _load_existing(output_path)

    if df is None:
        st.info("Run the scanner to generate candidates.")
        st.stop()

    if df.empty:
        st.warning("No candidates passed current filters.")
        st.stop()

    if "recommended_bet_usd" in df.columns:
        st.caption(f"Recommended stake uses Kelly scaling with a hard cap of ${MAX_BET_USD:.0f} per bet.")
    elif "suggested_kelly_fraction" in df.columns:
        cap_for_scaling = max(max_kelly_fraction, 0.001)
        scaled = (df["suggested_kelly_fraction"].astype(float) / cap_for_scaling) * MAX_BET_USD
        df = df.copy()
        df["recommended_bet_usd"] = scaled.clip(lower=0.0, upper=MAX_BET_USD).round(2)
        st.caption(f"Recommended stake uses Kelly scaling with a hard cap of ${MAX_BET_USD:.0f} per bet.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", int(len(df)))
    c2.metric("Avg edge (bps)", f"{df['edge_bps'].mean():.1f}")
    c3.metric("Median liquidity", f"{df['liquidity'].median():,.0f}")
    c4.metric("Max edge (bps)", f"{df['edge_bps'].max():.1f}")

    if "action_bucket" in df.columns:
        b1, b2, b3 = st.columns(3)
        bucket_counts = df["action_bucket"].value_counts()
        b1.metric("GREEN", int(bucket_counts.get("GREEN", 0)))
        b2.metric("YELLOW", int(bucket_counts.get("YELLOW", 0)))
        b3.metric("RED", int(bucket_counts.get("RED", 0)))

    st.subheader("Top Opportunities")
    display_cols = [
        "recommended_bet_usd",
        "action_bucket",
        "action_score",
        "side",
        "days_to_end",
        "end_time_utc",
        "is_active",
        "is_closed",
        "accepting_orders",
        "fair_minus_market_mid_bps",
        "edge_bps",
        "entry_price",
        "fair_yes_prob",
        "market_yes_mid",
        "suggested_kelly_fraction",
        "spread",
        "volume",
        "liquidity",
        "method",
        "question",
        "url",
    ]
    present_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[present_cols], use_container_width=True)

    if "edge_bps" in df.columns:
        chart_df = df[["question", "edge_bps"]].head(20).set_index("question")
        st.bar_chart(chart_df)

    st.subheader("Daily Watchlist")
    watchlist_n = st.slider("Watchlist size", min_value=3, max_value=15, value=5, step=1)
    if st.button("Save daily watchlist"):
        ok, msg = _append_watchlist(df, top_n=watchlist_n, path=_default_watchlist(provider))
        if ok:
            st.success(msg)
        else:
            st.error(msg)


if __name__ == "__main__":
    main()
