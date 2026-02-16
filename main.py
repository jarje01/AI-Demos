"""
Main Entry Point – orchestrates the full FX ML trading pipeline.

Usage:
    python main.py                   # Full pipeline (train + backtest + walk-forward)
    python main.py --mode train      # Train model only
    python main.py --mode backtest   # Backtest only (loads saved model)
    python main.py --mode walkforward # Walk-forward only
    python main.py --mode report     # Generate plots from saved results
    python main.py --mode journal    # Create dated paper-trading journal entry
    python main.py --refresh-data    # Force refresh and bypass cached OHLCV data
"""

import os
import sys
import json
import logging
import argparse
import pickle
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    RISK_CONFIG,
    WALK_FORWARD_CONFIG,
    PATHS,
    RULES_CONFIG,
    STRATEGY_PROFILES,
)
from data_loader import download_data, split_data, get_latest_feed_quote, add_polymarket_features
from features import build_features
from model import FXModel
from signals import SignalGenerator
from backtest import Backtester, compute_metrics, print_metrics
from walk_forward import WalkForwardEngine, print_wf_summary
from visualization import (
    plot_equity_curve, plot_signals_on_price, plot_walk_forward_results,
    plot_feature_importance, plot_trade_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PATHS.get("results_dir", "results/"), "pipeline.log"),
                            mode="w", delay=True),
    ]
)
logger = logging.getLogger("main")


# ─── Pipeline steps ────────────────────────────────────────────────────────────

def step_load_data(force_refresh: bool = False) -> pd.DataFrame:
    logger.info("━━━ Step 1: Load Data ━━━")
    df = download_data(use_cache=not force_refresh)
    df = add_polymarket_features(df, interval=DATA_CONFIG.get("interval"))
    if force_refresh:
        logger.info("Data refresh forced (--refresh-data).")
    logger.info("Loaded %d bars of %s (%s)", len(df), DATA_CONFIG["symbol"], DATA_CONFIG["interval"])
    return df


def step_train(train_raw: pd.DataFrame) -> FXModel:
    logger.info("━━━ Step 2: Train Model ━━━")
    train_feat = build_features(train_raw)
    model = FXModel()
    model.fit(train_feat)
    model.save()

    # Quick CV check
    cv_result = model.cross_validate(train_feat, cv=3)
    logger.info(
        "Cross-val accuracy: %.3f ± %.3f",
        cv_result["cv_accuracy_mean"], cv_result["cv_accuracy_std"]
    )

    fi = model.feature_importance()
    if len(fi) > 0:
        logger.info("Top-5 features: %s", fi.head(5).to_dict())

    return model


def step_backtest(
    model: FXModel,
    test_raw: pd.DataFrame,
) -> dict:
    logger.info("━━━ Step 3: Out-of-Sample Backtest ━━━")
    test_feat = build_features(test_raw)

    # Model evaluation
    eval_metrics = model.evaluate(test_feat)
    logger.info("Model accuracy (OOS): %.3f", eval_metrics["accuracy"])

    # Signal generation
    proba_df = model.get_signal_probabilities(test_feat)
    gen = SignalGenerator()
    signals_df = gen.generate(test_feat, proba_df)
    filter_bd = gen.filter_breakdown(signals_df)
    logger.info("Filter breakdown:\n%s", filter_bd.to_string())

    # Persist per-bar signals for paper-trading review
    signal_label_map = {1: "Long", -1: "Short", 0: "Flat"}
    signals_export = signals_df.copy()
    signals_export["signal_label"] = signals_export["signal"].map(signal_label_map).fillna("Flat")
    signals_export.index.name = "timestamp"
    signals_path = os.path.join(PATHS["results_dir"], "signals_latest.csv")
    signals_export.to_csv(signals_path)
    logger.info("Signals saved to %s", signals_path)

    step_live_signal(model, test_raw)

    # Backtest
    bt = Backtester()
    state = bt.run(test_feat, signals_df)
    metrics = compute_metrics(state)
    print_metrics(metrics)

    # Save results
    os.makedirs(PATHS["results_dir"], exist_ok=True)
    results_path = os.path.join(PATHS["results_dir"], "backtest_metrics.json")
    json_safe = {k: v for k, v in metrics.items() if k != "equity_curve"}
    with open(results_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    logger.info("Metrics saved to %s", results_path)

    # Charts
    eq = metrics.get("equity_curve")
    if eq is not None:
        plot_equity_curve(eq, title="OOS Backtest – EUR/USD", filename="oos_equity_curve.png")
        plot_signals_on_price(test_feat, signals_df, filename="oos_price_signals.png")

    if state.closed_trades:
        plot_trade_distribution(state.closed_trades, filename="trade_distribution.png")

    return metrics


def step_walk_forward(df: pd.DataFrame):
    logger.info("━━━ Step 4: Walk-Forward Analysis ━━━")
    engine = WalkForwardEngine()
    result = engine.run(df)
    print_wf_summary(result)

    # Persist walk-forward result
    os.makedirs(PATHS["results_dir"], exist_ok=True)
    wf_path = os.path.join(PATHS["results_dir"], "walk_forward_result.pkl")
    with open(wf_path, "wb") as f:
        pickle.dump(result, f)
    logger.info("Walk-forward result saved to %s", wf_path)

    # Charts
    plot_walk_forward_results(result, filename="walk_forward.png")
    if result.feature_importance_avg is not None:
        plot_feature_importance(result.feature_importance_avg, top_n=20,
                                filename="feature_importance.png")

    # Save combined metrics JSON
    wf_metrics_path = os.path.join(PATHS["results_dir"], "walk_forward_metrics.json")
    with open(wf_metrics_path, "w") as f:
        json.dump(result.combined_metrics, f, indent=2)

    return result


def step_journal(journal_date: str | None = None, journal_dir: str = "journals", force: bool = False):
    """Create a dated paper-trading journal markdown file."""
    if journal_date:
        try:
            parsed = datetime.strptime(journal_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("--journal-date must be in YYYY-MM-DD format") from exc
    else:
        parsed = date.today()

    output_dir = Path(journal_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{parsed.isoformat()}_journal.md"

    if output_path.exists() and not force:
        logger.info("Journal already exists at %s (use --journal-force to overwrite)", output_path)
        return output_path

    content = (
        f"# Paper Trading Journal — {parsed.isoformat()}\n\n"
        f"Date: {parsed.isoformat()}\n\n"
        "Market context (calm / trending / choppy):\n\n"
        "Signal (Long / Short / Flat):\n\n"
        "Entry price (paper):\n\n"
        "Stop-loss:\n\n"
        "Take-profit:\n\n"
        "Position size rule used:\n\n"
        "Outcome (Win/Loss, +R/-R):\n\n"
        "Mistake made (if any):\n\n"
        "One improvement for tomorrow:\n"
    )
    output_path.write_text(content, encoding="utf-8")
    logger.info("Journal created: %s", output_path)
    return output_path


def step_live_signal(model: FXModel, raw_df: pd.DataFrame) -> dict | None:
    """Generate live/latest-bar inference artifact (no forward-label shift)."""
    logger.info("━━━ Live Signal Inference (latest bar) ━━━")
    live_feat = build_features(raw_df, include_target=False)
    if live_feat.empty:
        logger.warning("Live inference skipped: no feature rows available.")
        return None

    gen = SignalGenerator()
    live_last = live_feat.tail(1)
    live_proba = model.get_signal_probabilities(live_last)
    live_signal_df = gen.generate(live_last, live_proba)
    live_row = live_signal_df.iloc[-1]
    feature_row = live_last.iloc[-1]
    live_label_map = {1: "Long", -1: "Short", 0: "Flat"}

    last_idx = pd.Timestamp(live_signal_df.index[-1])
    market_idx = pd.Timestamp(raw_df.index.max())
    ts_format = "%Y-%m-%d %H:%M:%S"
    live_payload = {
        "market_bar_timestamp": market_idx.strftime(ts_format),
        "live_signal_timestamp": last_idx.strftime(ts_format),
        "signal": int(live_row.get("signal", 0)),
        "signal_label": live_label_map.get(int(live_row.get("signal", 0)), "Flat"),
        "raw_signal": int(live_row.get("raw_signal", 0)),
        "signal_reason": str(live_row.get("signal_reason", "n/a")),
        "prob_long": float(live_row.get("prob_long", 0.0)),
        "prob_short": float(live_row.get("prob_short", 0.0)),
        "prob_flat": float(live_row.get("prob_flat", 0.0)),
        "close": float(feature_row.get("Close", 0.0)),
        "atr": float(feature_row.get("atr", 0.0)),
    }

    atr = float(feature_row.get("atr", 0.0))
    stop_mult = float(RISK_CONFIG.get("atr_stop_multiplier", 1.5))
    tp_mult = float(RISK_CONFIG.get("take_profit_multiplier", 2.0))
    stop_dist = atr * stop_mult
    tp_dist = atr * tp_mult
    live_payload.update(
        {
            "atr_stop_multiplier": stop_mult,
            "take_profit_multiplier": tp_mult,
            "stop_distance": stop_dist,
            "tp_distance": tp_dist,
            "stop_distance_pips": stop_dist * 10_000.0,
            "tp_distance_pips": tp_dist * 10_000.0,
        }
    )

    quote = get_latest_feed_quote()
    if quote:
        live_payload.update(
            {
                "quote_provider": quote.get("provider"),
                "quote_timestamp": quote.get("quote_timestamp"),
                "sell_bid": quote.get("sell_bid"),
                "buy_ask": quote.get("buy_ask"),
                "spread": quote.get("spread"),
                "spread_pips": quote.get("spread_pips"),
            }
        )

        buy_ask = quote.get("buy_ask")
        sell_bid = quote.get("sell_bid")
        if buy_ask is not None and sell_bid is not None:
            buy_ask = float(buy_ask)
            sell_bid = float(sell_bid)
            live_payload.update(
                {
                    "long_entry": buy_ask,
                    "long_stop": buy_ask - stop_dist,
                    "long_take_profit": buy_ask + tp_dist,
                    "short_entry": sell_bid,
                    "short_stop": sell_bid + stop_dist,
                    "short_take_profit": sell_bid - tp_dist,
                }
            )

    live_path = os.path.join(PATHS["results_dir"], "live_signal_latest.json")
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_payload, f, indent=2)
    logger.info("Live signal saved to %s", live_path)
    logger.info(
        "Live signal: %s at %s (reason=%s, p_long=%.3f, p_short=%.3f, p_flat=%.3f)",
        live_payload["signal_label"],
        live_payload["live_signal_timestamp"],
        live_payload["signal_reason"],
        live_payload["prob_long"],
        live_payload["prob_short"],
        live_payload["prob_flat"],
    )
    return live_payload


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="FX ML Trading System")
    parser.add_argument(
        "--mode",
        choices=["full", "train", "backtest", "walkforward", "report", "journal", "live"],
        default="full",
        help="Pipeline mode to run (default: full)"
    )
    parser.add_argument("--no-wf", action="store_true",
                        help="Skip walk-forward in full mode (faster)")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Force fresh download and bypass local OHLCV cache")
    parser.add_argument("--journal-date", type=str, default=None,
                        help="Journal date for --mode journal (YYYY-MM-DD, default: today)")
    parser.add_argument("--journal-dir", type=str, default="journals",
                        help="Journal directory for --mode journal (default: journals)")
    parser.add_argument("--journal-force", action="store_true",
                        help="Overwrite existing journal file in --mode journal")
    parser.add_argument(
        "--profile",
        choices=["safe", "balanced", "aggressive"],
        default="safe",
        help="Strategy profile preset for signal thresholds (default: safe)",
    )
    return parser.parse_args()


def apply_profile(profile_name: str):
    """Apply strategy profile overrides in-place to global model/rules config."""
    profile = STRATEGY_PROFILES.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown profile: {profile_name}")

    MODEL_CONFIG.update(profile.get("model", {}))
    RULES_CONFIG.update(profile.get("rules", {}))

    logger.info(
        "Applied profile '%s': min_confidence=%.3f, min_edge_over_opposite=%.3f, adx_trending_threshold=%s",
        profile_name,
        MODEL_CONFIG.get("min_confidence", 0.0),
        MODEL_CONFIG.get("min_edge_over_opposite", 0.0),
        RULES_CONFIG.get("adx_trending_threshold"),
    )


def main():
    args = parse_args()

    apply_profile(args.profile)

    # Ensure output directories exist
    for d in PATHS.values():
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  FX ML Trading System – EUR/USD")
    logger.info("  Mode: %s", args.mode)
    logger.info("  Profile: %s", args.profile)
    logger.info("=" * 60)

    if args.mode in ("full", "train", "backtest"):
        df = step_load_data(force_refresh=args.refresh_data)
        train_raw, test_raw = split_data(df)

    if args.mode == "train":
        step_train(train_raw)
        logger.info("Training complete. Model saved.")

    elif args.mode == "backtest":
        try:
            model = FXModel.load()
            logger.info("Loaded existing model.")
        except Exception:
            logger.info("No saved model found – training fresh model.")
            model = step_train(train_raw)
        step_backtest(model, test_raw)

    elif args.mode == "walkforward":
        df = step_load_data(force_refresh=args.refresh_data)
        step_walk_forward(df)

    elif args.mode == "report":
        wf_path = os.path.join(PATHS["results_dir"], "walk_forward_result.pkl")
        if os.path.exists(wf_path):
            with open(wf_path, "rb") as f:
                result = pickle.load(f)
            plot_walk_forward_results(result, filename="walk_forward.png")
            if result.feature_importance_avg is not None:
                plot_feature_importance(result.feature_importance_avg)
            print_wf_summary(result)
        else:
            logger.error("No walk-forward result found at %s – run with --mode walkforward first.", wf_path)

    elif args.mode == "journal":
        step_journal(journal_date=args.journal_date, journal_dir=args.journal_dir, force=args.journal_force)

    elif args.mode == "live":
        df = step_load_data(force_refresh=args.refresh_data)
        try:
            model = FXModel.load()
            logger.info("Loaded existing model.")
        except Exception:
            logger.info("No saved model found – training fresh model.")
            train_raw, _ = split_data(df)
            model = step_train(train_raw)
        step_live_signal(model, df)

    elif args.mode == "full":
        # Full pipeline
        model = step_train(train_raw)
        step_backtest(model, test_raw)

        if not args.no_wf:
            step_walk_forward(df)
        else:
            logger.info("Walk-forward skipped (--no-wf flag).")

    logger.info("Pipeline complete. Plots saved to: %s", PATHS["plots_dir"])
    logger.info("Results saved to: %s", PATHS["results_dir"])


if __name__ == "__main__":
    main()
