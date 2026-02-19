from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "results" / "kalshi_only_bet_recommendations.csv"


def _load_recommendations(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    numeric_cols = [
        "kalshi_yes_price",
        "kalshi_no_price",
        "polymarket_yes_price",
        "yes_price_gap",
        "abs_price_gap",
        "option_similarity",
        "source_arb_potential_score",
        "kalshi_liquidity_score",
        "polymarket_liquidity_score",
        "days_to_resolution",
        "drift_score",
        "expected_value_kalshi_only",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "kalshi.question",
        "polymarket.question",
        "kalshi_option",
        "recommended_kalshi_bet",
        "confidence_tier",
        "risk_notes",
        "recommendation_reason",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def _empty_message(path: Path) -> None:
    st.warning("No recommendation rows loaded.")
    st.info(f"Expected CSV at: {path}")
    st.caption("Run the recommendation cell in the notebook to regenerate this file.")


def main() -> None:
    st.set_page_config(page_title="Kalshi Top Bets", layout="wide")
    st.title("Kalshi Top Bets")
    st.caption("Simple ranked list of recommended Kalshi bets from your latest scanner output.")

    with st.sidebar:
        st.header("Data")
        csv_path_input = st.text_input("Recommendations CSV", value=str(DEFAULT_CSV))
        csv_path = Path(csv_path_input)

        auto_refresh = st.toggle("Auto refresh", value=False)
        refresh_seconds = st.slider(
            "Refresh interval (sec)",
            min_value=15,
            max_value=300,
            value=60,
            step=15,
            disabled=not auto_refresh,
        )
        if st.button("Reload now"):
            st.rerun()

        if auto_refresh:
            if st_autorefresh is not None:
                st_autorefresh(interval=refresh_seconds * 1000, key="kalshi_bets_autorefresh")
            else:
                st.warning("Install streamlit-autorefresh to enable auto refresh.")

    df = _load_recommendations(csv_path)
    if df is None or df.empty:
        _empty_message(csv_path)
        st.stop()

    with st.sidebar:
        st.header("Filters")

        top_n = st.slider("Top rows", min_value=5, max_value=100, value=30, step=5)

        side_options = sorted(df["recommended_kalshi_bet"].dropna().unique().tolist()) if "recommended_kalshi_bet" in df.columns else []
        selected_sides = st.multiselect("Bet side", options=side_options, default=side_options)

        tier_options = sorted(df["confidence_tier"].dropna().unique().tolist()) if "confidence_tier" in df.columns else []
        selected_tiers = st.multiselect("Confidence", options=tier_options, default=tier_options)

        if "abs_price_gap" in df.columns and not df["abs_price_gap"].dropna().empty:
            min_gap = float(df["abs_price_gap"].min())
            max_gap = float(df["abs_price_gap"].max())
            min_abs_gap = st.slider(
                "Min abs price gap",
                min_value=min_gap,
                max_value=max_gap,
                value=min_gap,
                step=0.01,
            )
        else:
            min_abs_gap = 0.0

        query = st.text_input("Search text", value="").strip().lower()

    filtered = df.copy()

    if selected_sides and "recommended_kalshi_bet" in filtered.columns:
        filtered = filtered[filtered["recommended_kalshi_bet"].isin(selected_sides)]

    if selected_tiers and "confidence_tier" in filtered.columns:
        filtered = filtered[filtered["confidence_tier"].isin(selected_tiers)]

    if "abs_price_gap" in filtered.columns:
        filtered = filtered[filtered["abs_price_gap"] >= min_abs_gap]

    if query:
        haystack = pd.Series("", index=filtered.index, dtype="object")
        for col in ["kalshi.question", "polymarket.question", "kalshi_option", "risk_notes", "recommendation_reason"]:
            if col in filtered.columns:
                haystack = haystack + " " + filtered[col].astype(str).str.lower()
        filtered = filtered[haystack.str.contains(query, na=False)]

    sort_default = "expected_value_kalshi_only" if "expected_value_kalshi_only" in filtered.columns else "abs_price_gap"
    if sort_default not in filtered.columns:
        sort_default = filtered.columns[0]

    s1, s2 = st.columns([3, 1])
    sort_col = s1.selectbox("Sort by", options=list(filtered.columns), index=list(filtered.columns).index(sort_default))
    descending = s2.toggle("Descending", value=True)

    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=not descending, na_position="last")

    view = filtered.head(top_n).copy().reset_index(drop=True)
    view.insert(0, "rank", range(1, len(view) + 1))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total rows", int(len(df)))
    m2.metric("Filtered rows", int(len(filtered)))
    if "expected_value_kalshi_only" in filtered.columns and not filtered["expected_value_kalshi_only"].dropna().empty:
        m3.metric("Avg expected value", f"{filtered['expected_value_kalshi_only'].mean():.3f}")
    else:
        m3.metric("Avg expected value", "n/a")
    if "confidence_tier" in filtered.columns:
        high_n = int((filtered["confidence_tier"].str.lower() == "high").sum())
        m4.metric("High confidence", high_n)
    else:
        m4.metric("High confidence", 0)

    column_order = [
        "rank",
        "recommended_kalshi_bet",
        "confidence_tier",
        "kalshi.question",
        "kalshi_option",
        "kalshi_yes_price",
        "polymarket_yes_price",
        "abs_price_gap",
        "expected_value_kalshi_only",
        "drift_score",
        "days_to_resolution",
        "kalshi_liquidity_score",
        "polymarket_liquidity_score",
        "risk_notes",
        "recommendation_reason",
    ]
    present = [col for col in column_order if col in view.columns]
    remainder = [col for col in view.columns if col not in present]
    show_cols = present + remainder

    st.subheader(f"Top Bets ({len(view)} shown)")
    st.dataframe(
        view[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "kalshi_yes_price": st.column_config.NumberColumn("Kalshi YES", format="%.2f"),
            "polymarket_yes_price": st.column_config.NumberColumn("PM YES", format="%.2f"),
            "abs_price_gap": st.column_config.NumberColumn("Abs Gap", format="%.3f"),
            "expected_value_kalshi_only": st.column_config.NumberColumn("EV (Kalshi)", format="%.3f"),
            "drift_score": st.column_config.NumberColumn("Drift", format="%.3f"),
            "kalshi_liquidity_score": st.column_config.NumberColumn("Kalshi Liq", format="%.2f"),
            "polymarket_liquidity_score": st.column_config.NumberColumn("PM Liq", format="%.2f"),
        },
    )

    st.download_button(
        label="Download displayed rows",
        data=view[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="kalshi_top_bets_view.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
