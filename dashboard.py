from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

PROFILE_FILES = {
    "balanced": {
        "metrics": RESULTS_DIR / "walk_forward_metrics_balanced.json",
        "gate": RESULTS_DIR / "gate_balanced.json",
    },
    "aggressive": {
        "metrics": RESULTS_DIR / "walk_forward_metrics_aggressive.json",
        "gate": RESULTS_DIR / "gate_aggressive.json",
    },
    "safe": {
        "metrics": RESULTS_DIR / "walk_forward_metrics_safe.json",
        "gate": RESULTS_DIR / "gate_safe.json",
    },
    "quickwinner": {
        "metrics": RESULTS_DIR / "walk_forward_metrics_quickwinner.json",
        "gate": RESULTS_DIR / "gate_quickwinner.json",
    },
}


def _read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return raw.decode("latin-1")


def _load_json(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        text = _read_text_auto(path)
        return json.loads(text), None
    except Exception as exc:
        return None, f"Failed to parse JSON at {path}: {exc}"


def _load_signals(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df
    except Exception:
        return None


def _pct(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}%"


def _num(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _spread_quality(spread_pips: float | None) -> tuple[str, str]:
    if spread_pips is None:
        return "Unknown", "⚪"
    if spread_pips <= 1.0:
        return "Tight", "🟢"
    if spread_pips <= 2.0:
        return "Normal", "🟡"
    return "Wide", "🔴"


def _format_utc_and_local(ts_value: str | None) -> tuple[str, str]:
    if not ts_value:
        return "n/a", "n/a"
    try:
        ts_utc = pd.to_datetime(ts_value, utc=True)
        utc_str = ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        local_tz = datetime.now().astimezone().tzinfo
        local_str = ts_utc.tz_convert(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        return utc_str, local_str
    except Exception:
        return str(ts_value), "n/a"


def _format_central(ts_value: str | None) -> str:
    if not ts_value:
        return "n/a"
    try:
        ts_utc = pd.to_datetime(ts_value, utc=True)
        central = ts_utc.tz_convert(ZoneInfo("America/Chicago"))
        return central.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "n/a"


def _trade_readiness(live_signal: dict | None, gate: dict | None) -> tuple[str, str, str]:
    if not live_signal:
        return "WAIT", "🟡", "No live signal payload available"

    signal_label = str(live_signal.get("signal_label", "Flat"))
    actionable = signal_label in {"Long", "Short"}
    if not actionable:
        return "WAIT", "🟡", f"Signal is {signal_label}"

    spread_pips = live_signal.get("spread_pips")
    if spread_pips is None:
        return "WAIT", "🟡", "Spread unavailable"

    max_spread_pips = 2.0
    if float(spread_pips) > max_spread_pips:
        return "WAIT", "🔴", f"Spread too wide ({spread_pips:.2f} pips > {max_spread_pips:.2f})"

    recommendation = None
    if gate is not None:
        recommendation = gate.get("summary", {}).get("recommendation")

    if recommendation and str(recommendation).startswith("GO"):
        return "READY", "🟢", f"Actionable signal ({signal_label}), spread OK, gate={recommendation}"

    if recommendation:
        return "WAIT", "🟡", f"Actionable signal and spread OK, but gate={recommendation}"

    return "READY", "🟢", f"Actionable signal ({signal_label}) and spread OK"


def _recommended_order_type(live_signal: dict | None, readiness: str) -> tuple[str, str]:
    if not live_signal:
        return "No Trade", "No live signal data"

    signal_label = str(live_signal.get("signal_label", "Flat"))
    spread_pips = live_signal.get("spread_pips")

    if readiness != "READY" or signal_label not in {"Long", "Short"}:
        return "No Trade", "Wait for READY + Long/Short signal"

    if spread_pips is None:
        return "Limit", "Spread unknown: prefer controlled entry"

    spread_pips = float(spread_pips)
    if spread_pips <= 1.0:
        return "Market", "Tight spread: prioritize fill"
    if spread_pips <= 2.0:
        return "Limit", "Normal spread: control entry price"
    return "No Trade", "Spread too wide"


def _append_paper_trade_log(record: dict, path: Path) -> tuple[bool, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([record])
        write_header = not path.exists()
        df.to_csv(path, mode="a", header=write_header, index=False)
        return True, f"Logged snapshot to {path}"
    except Exception as exc:
        return False, f"Failed to log snapshot: {exc}"


def main() -> None:
    st.set_page_config(page_title="FX ML Signal Dashboard", layout="wide")
    st.title("FX ML Signal Dashboard")

    with st.sidebar:
        st.header("Inputs")
        profile = st.selectbox("Profile", ["balanced", "safe", "aggressive", "quickwinner"], index=0)
        lookback = st.slider("Recent rows", min_value=20, max_value=250, value=80, step=10)
        auto_refresh = st.toggle("Auto refresh", value=False)
        refresh_seconds = st.slider(
            "Refresh interval (sec)",
            min_value=15,
            max_value=300,
            value=60,
            step=15,
            disabled=not auto_refresh,
        )
        if st.button("Refresh now"):
            st.rerun()

        if st.button("Run live refresh now"):
            with st.spinner("Running live refresh…"):
                cmd = [
                    sys.executable,
                    str(BASE_DIR / "main.py"),
                    "--mode",
                    "live",
                    "--profile",
                    profile,
                    "--refresh-data",
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if proc.returncode == 0:
                    st.success("Live refresh completed")
                    st.rerun()
                else:
                    st.error("Live refresh failed")
                    if proc.stderr:
                        st.caption(proc.stderr[-1200:])

        if auto_refresh:
            if st_autorefresh is not None:
                st_autorefresh(interval=refresh_seconds * 1000, key="fx_dashboard_autorefresh")
            else:
                st.warning("Install 'streamlit-autorefresh' to enable auto refresh.")
        st.caption(f"Signals are read from {RESULTS_DIR / 'signals_latest.csv'}")

    files = PROFILE_FILES[profile]
    metrics, metrics_err = _load_json(files["metrics"])
    gate, gate_err = _load_json(files["gate"])
    live_signal, live_signal_err = _load_json(RESULTS_DIR / "live_signal_latest.json")
    signals = _load_signals(RESULTS_DIR / "signals_latest.csv")

    if signals is None or signals.empty:
        st.error("No signals file found or file could not be parsed: results/signals_latest.csv")
        st.stop()

    latest = signals.iloc[-1]
    latest_non_flat = signals[signals["signal_label"] != "Flat"]
    latest_trade_signal = latest_non_flat.iloc[-1] if not latest_non_flat.empty else None

    st.subheader("Current Signal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest label", str(latest.get("signal_label", "n/a")))
    c2.metric("Timestamp", latest["timestamp"].strftime("%Y-%m-%d"))
    c3.metric("Reason", str(latest.get("signal_reason", "n/a")))
    c4.metric("Raw signal", str(int(latest.get("raw_signal", 0))))

    c5, c6, c7 = st.columns(3)
    c5.metric("Prob Long", _num(latest.get("prob_long"), 3))
    c6.metric("Prob Short", _num(latest.get("prob_short"), 3))
    c7.metric("Prob Flat", _num(latest.get("prob_flat"), 3))

    if latest_trade_signal is not None:
        st.info(
            "Most recent non-flat signal: "
            f"{latest_trade_signal['signal_label']} on {latest_trade_signal['timestamp'].strftime('%Y-%m-%d')}"
        )

    st.subheader("Data Freshness")
    f1, f2 = st.columns(2)
    f1.metric("Backtest signal timestamp", latest["timestamp"].strftime("%Y-%m-%d"))
    if live_signal is not None:
        live_raw = live_signal.get("live_signal_timestamp")
        bar_raw = live_signal.get("market_bar_timestamp")

        try:
            live_dt = pd.to_datetime(live_raw, utc=True)
            bar_dt = pd.to_datetime(bar_raw, utc=True)
            same_ts = live_dt == bar_dt
        except Exception:
            same_ts = False

        if same_ts:
            central_bar = _format_central(bar_raw)
            bar_utc, _ = _format_utc_and_local(bar_raw)
            f2.metric("Latest market bar (Central)", central_bar)
            st.caption(f"Latest market bar (UTC): {bar_utc}")
        else:
            live_utc, _ = _format_utc_and_local(live_raw)
            bar_utc, _ = _format_utc_and_local(bar_raw)
            f2.metric("Live inference timestamp (UTC)", live_utc)
            st.caption(f"Latest market bar (UTC): {bar_utc}")
            st.caption(f"Live inference (Central): {_format_central(live_raw)}")
            st.caption(f"Latest market bar (Central): {_format_central(bar_raw)}")
    else:
        f2.metric("Live inference timestamp", "n/a")
        st.caption(live_signal_err or "Run a backtest to generate results/live_signal_latest.json")

    if live_signal is not None:
        st.subheader("Live Inference Signal")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Signal", str(live_signal.get("signal_label", "n/a")))
        l2.metric("Reason", str(live_signal.get("signal_reason", "n/a")))
        l3.metric("Prob Long", _num(live_signal.get("prob_long"), 3))
        l4.metric("Prob Short", _num(live_signal.get("prob_short"), 3))

        readiness, readiness_icon, readiness_reason = _trade_readiness(live_signal, gate)
        st.markdown(f"### Trade Readiness: {readiness_icon} {readiness}")
        st.caption(f"{readiness_reason}")

        order_type, order_reason = _recommended_order_type(live_signal, readiness)
        st.markdown("### Execution Playbook")
        st.markdown(f"**Recommended order type:** {order_type}")
        st.caption(order_reason)
        if (
            live_signal.get("long_entry") is not None
            and live_signal.get("long_stop") is not None
            and live_signal.get("long_take_profit") is not None
            and live_signal.get("short_entry") is not None
            and live_signal.get("short_stop") is not None
            and live_signal.get("short_take_profit") is not None
        ):
            st.caption(
                "Long setup: "
                f"Entry {_num(live_signal.get('long_entry'), 5)} | "
                f"SL {_num(live_signal.get('long_stop'), 5)} | "
                f"TP {_num(live_signal.get('long_take_profit'), 5)}"
            )
            st.caption(
                "Short setup: "
                f"Entry {_num(live_signal.get('short_entry'), 5)} | "
                f"SL {_num(live_signal.get('short_stop'), 5)} | "
                f"TP {_num(live_signal.get('short_take_profit'), 5)}"
            )
            st.caption(
                f"Risk template: SL = ATR×{_num(live_signal.get('atr_stop_multiplier'), 2)}, "
                f"TP = ATR×{_num(live_signal.get('take_profit_multiplier'), 2)}"
            )
        st.caption("Market: immediate fill at current price")
        st.caption("Limit: fill only at your specified better price")
        st.caption("Stop: breakout entry after trigger price is reached")
        st.caption("Use expiry for pending Limit/Stop orders so stale entries auto-cancel")

        if st.button("Log paper trade snapshot"):
            log_path = RESULTS_DIR / "paper_trade_log.csv"
            record = {
                "logged_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
                "profile": profile,
                "signal_label": live_signal.get("signal_label"),
                "signal_reason": live_signal.get("signal_reason"),
                "trade_readiness": readiness,
                "recommended_order_type": order_type,
                "recommended_order_reason": order_reason,
                "live_signal_timestamp": live_signal.get("live_signal_timestamp"),
                "market_bar_timestamp": live_signal.get("market_bar_timestamp"),
                "quote_timestamp": live_signal.get("quote_timestamp"),
                "prob_long": live_signal.get("prob_long"),
                "prob_short": live_signal.get("prob_short"),
                "prob_flat": live_signal.get("prob_flat"),
                "buy_ask": live_signal.get("buy_ask"),
                "sell_bid": live_signal.get("sell_bid"),
                "spread_pips": live_signal.get("spread_pips"),
                "long_entry": live_signal.get("long_entry"),
                "long_stop": live_signal.get("long_stop"),
                "long_take_profit": live_signal.get("long_take_profit"),
                "short_entry": live_signal.get("short_entry"),
                "short_stop": live_signal.get("short_stop"),
                "short_take_profit": live_signal.get("short_take_profit"),
                "gate_recommendation": (gate or {}).get("summary", {}).get("recommendation"),
            }
            ok, msg = _append_paper_trade_log(record, log_path)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        if live_signal.get("buy_ask") is not None and live_signal.get("sell_bid") is not None:
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Buy (Ask)", _num(live_signal.get("buy_ask"), 5))
            q2.metric("Sell (Bid)", _num(live_signal.get("sell_bid"), 5))
            q3.metric("Spread", _num(live_signal.get("spread"), 5))
            q4.metric("Spread (pips)", _num(live_signal.get("spread_pips"), 2))

            spread_pips = live_signal.get("spread_pips")
            quality, icon = _spread_quality(spread_pips)
            st.markdown(f"**Spread quality:** {icon} {quality}")
            quote_utc, quote_local = _format_utc_and_local(live_signal.get("quote_timestamp"))
            st.caption(
                f"Quote source: {live_signal.get('quote_provider', 'n/a')} | "
                f"Quote time (UTC): {quote_utc}"
            )
            st.caption(f"Quote time (Local): {quote_local}")

    st.subheader("Walk-Forward Snapshot")
    if metrics is None:
        st.warning(metrics_err or f"No metrics file found for profile '{profile}': {files['metrics']}")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        total_folds = metrics.get("total_folds")
        valid_folds = metrics.get("valid_folds")
        valid_ratio = (valid_folds / total_folds) if total_folds else None

        m1.metric("Combined Return", _pct(metrics.get("combined_total_return_pct")))
        m2.metric("Max Drawdown", _pct(metrics.get("combined_max_drawdown_pct")))
        m3.metric("Avg Sharpe", _num(metrics.get("avg_sharpe"), 4))
        m4.metric("Valid Folds", f"{valid_folds}/{total_folds}" if total_folds else "n/a")
        m5.metric("Valid Fold Ratio", _num(valid_ratio, 4))

    st.subheader("Gate Status")
    if gate is None:
        st.warning(gate_err or f"No gate file found for profile '{profile}': {files['gate']}")
    else:
        summary = gate.get("summary", {})
        st.write(f"Recommendation: {summary.get('recommendation', 'n/a')}")
        hard_failed = summary.get("hard_failed", [])
        soft_failed = summary.get("soft_failed", [])
        st.write(f"Hard failed: {', '.join(hard_failed) if hard_failed else 'none'}")
        st.write(f"Soft failed: {', '.join(soft_failed) if soft_failed else 'none'}")

    st.subheader("Recent Signal History")
    recent = signals.tail(lookback).copy()

    plot_cols = ["timestamp", "prob_long", "prob_short", "prob_flat"]
    st.line_chart(recent[plot_cols].set_index("timestamp"))

    signal_counts = recent["signal_label"].value_counts().rename_axis("signal").reset_index(name="count")
    st.bar_chart(signal_counts.set_index("signal"))

    table_cols = [
        "timestamp",
        "signal_label",
        "signal_reason",
        "prob_long",
        "prob_short",
        "prob_flat",
    ]
    st.dataframe(recent[table_cols].iloc[::-1], use_container_width=True)


if __name__ == "__main__":
    main()
