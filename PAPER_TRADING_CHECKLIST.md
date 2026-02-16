# EUR/USD ML System — Paper Trading Checklist (Beginner)

Use this checklist each session so you learn with a repeatable process.

---

## 1) One-time setup (first day)

- [ ] Install dependencies
  - `pip install -r requirements.txt`
- [ ] Configure OANDA key locally
  - Copy `.env.example` to `.env`
  - Set `OANDA_API_KEY=...` in `.env`
- [ ] Confirm data provider is set
  - In `config.py`, use `DATA_CONFIG["provider"] = "oanda"`
- [ ] Run baseline pipeline
  - `python main.py --mode full --no-wf`
  - `python main.py --mode walkforward --profile balanced --refresh-data`
- [ ] Run deployment gate
  - `python deployment_gate.py`
- [ ] Launch dashboard
  - `python -m streamlit run dashboard.py --server.port 8502`
- [ ] Confirm files exist:
  - `results/backtest_metrics.json`
  - `results/walk_forward_metrics.json`
  - `results/live_signal_latest.json`
  - `plots/walk_forward.png`

---

## 2) Daily session checklist (paper trading)

Before starting, create today's journal file:
- `python create_journal.py`

### A. Pre-session (5 minutes)

- [ ] Open dashboard and click `Run live refresh now`
- [ ] Confirm `Data Freshness` latest market bar time is current enough
- [ ] Check `Trade Readiness` and `Execution Playbook`
- [ ] Check gate recommendation in dashboard `Gate Status`
- [ ] If recommendation is `NO_GO`, skip new paper entries today and review settings.

### B. Signal workflow (bar close)

- [ ] Refresh live signal each cycle:
  - Dashboard button: `Run live refresh now`
  - CLI fallback: `python main.py --mode live --profile balanced --refresh-data`
- [ ] Only consider entries when `Trade Readiness = READY`
- [ ] Use dashboard execution template (`Long setup` / `Short setup` entry, SL, TP)
- [ ] Place paper trade only if spread is acceptable and risk rules are respected
- [ ] Click `Log paper trade snapshot` in dashboard
- [ ] Confirm a new row was appended to `results/paper_trade_log.csv`

### C. Risk guardrails (always)

- [ ] Never override stop-loss after entry
- [ ] No revenge trades
- [ ] No setting changes during an active week
- [ ] Keep one consistent position sizing rule
- [ ] Skip entries when spread quality is `Wide`

### D. End of session (5 minutes)

- [ ] Log outcome: win/loss, R-multiple, notes
- [ ] Capture one lesson learned
- [ ] Save updated metrics snapshot if you reran walk-forward/backtest
- [ ] Ensure today's snapshots are in `results/paper_trade_log.csv`

---

## 3) Weekly review checklist

- [ ] Rerun walk-forward:
  - `python main.py --mode walkforward --profile balanced --refresh-data`
- [ ] Run gate JSON for record:
  - `python deployment_gate.py --json`
- [ ] Check these core stats:
  - `combined_total_return_pct`
  - `combined_max_drawdown_pct`
  - `avg_sharpe`
  - `pct_profitable_folds`
- [ ] Compare against prior week (improving, flat, or degrading)
- [ ] If degraded for 2+ weeks, pause paper trading and investigate

---

## 4) Rookie-safe progression rules

- [ ] Minimum paper period before considering live: **6–8 weeks**
- [ ] Continue only if drawdown stays within your comfort threshold
- [ ] Continue only if process discipline is consistent (no rule breaking)
- [ ] Move to live only with small size first
- [ ] Do not promote to live if `Trade Readiness` is mostly `WAIT` due spread/gate

---

## 5) Simple daily journal template

Date:

Market context (calm / trending / choppy):

Signal (Long / Short / Flat):

Entry price (paper):

Stop-loss:

Take-profit:

Position size rule used:

Outcome (Win/Loss, +R/-R):

Mistake made (if any):

One improvement for tomorrow:
