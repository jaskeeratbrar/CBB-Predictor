# CBB Kalshi Trading Bot — Project Progress

## Status: Paper Trading Phase

Started: 2026-03-16
CBB Season Start: 2026-03-17

---

## Phase 1: Foundation (MVP) — COMPLETE

| Task | Status | Date |
|------|--------|------|
| Project scaffolding (.gitignore, requirements, config) | Done | 2026-03-16 |
| db.py — SQLite schema (8 tables) + 20 CRUD functions | Done | 2026-03-16 |
| data.py — ESPN free API pipeline (tested: 10 games fetched) | Done | 2026-03-16 |
| features.py — 6 features + Elo system (K=20, base 1500) | Done | 2026-03-16 |
| model.py — Calibrated LogReg (XGBoost ready for Phase 5) | Done | 2026-03-16 |
| main.py — CLI: init, predict, trade, status, review, backtest, run | Done | 2026-03-16 |

## Phase 2: Trading Engine — COMPLETE

| Task | Status | Date |
|------|--------|------|
| kalshi.py — RSA-PSS SHA256 auth, market discovery, order placement | Done | 2026-03-16 |
| trading.py — Fractional Kelly (10%), 7 risk rules, paper mode | Done | 2026-03-16 |
| Paper trading mode — simulated trades logged to DB | Done | 2026-03-16 |

## Phase 3: Automation + Review — COMPLETE

| Task | Status | Date |
|------|--------|------|
| scheduler.py — APScheduler (daily refresh, pre-game, settlement) | Done | 2026-03-16 |
| review.py — Daily MD logs + weekly reports + calibration analysis | Done | 2026-03-16 |

## Phase 4: Claude Code Integration — COMPLETE

| Task | Status | Date |
|------|--------|------|
| CLAUDE.md — Project context for agents | Done | 2026-03-16 |
| /predict skill | Done | 2026-03-16 |
| /trade skill (safety-gated, dry-run first) | Done | 2026-03-16 |
| /review skill | Done | 2026-03-16 |
| /backtest skill | Done | 2026-03-16 |
| Pre-trade warning hook + post-trade logging hook | Done | 2026-03-16 |
| data-fetcher agent (haiku) | Done | 2026-03-16 |
| trade-executor agent | Done | 2026-03-16 |

## Phase 5: Model Improvement — IN PROGRESS

| Task | Status | Date |
|------|--------|------|
| XGBoost ensemble (60/40 with LogReg) | Pending | — |
| Bayesian updating between weekly retrains | Pending | — |
| Feature expansion (tempo, def_eff, rest days, recent form, tournament flag) | Pending | — |
| Calibration tracking in weekly reports | Pending | — |
| Backtest harness for model validation | Pending | — |

---

## Infrastructure

| Item | Status | Notes |
|------|--------|-------|
| Git repo initialized | Done | 23 files, 4,878 lines |
| Linux portability | Done | pytz added, paths use pathlib |
| Kalshi API key configured | Done | Key ID + private key set |
| Paper mode | Active | config.json: paper_mode=true |
| Deploy to Linux machine | Pending | Clone repo + copy config + pip install |

---

## Risk Rules (hardcoded, never bypass)

1. Max 5% of bankroll per trade
2. 10% fractional Kelly
3. 15% daily loss limit
4. 3-loss cooldown (rest of day)
5. 3% minimum edge threshold
6. Max 5 open positions
7. Stop trading if bankroll < $2

---

## Key Decisions Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-03-16 | Start with LogReg only, not XGBoost | Faster to train, easier to debug, sufficient for MVP |
| 2026-03-16 | ESPN API as primary data source | Free, no auth, richest CBB data (schedules, scores, odds) |
| 2026-03-16 | 10% fractional Kelly | Extremely conservative for $15 bankroll — nearly all trades = 1 contract |
| 2026-03-16 | Paper mode first | Validate model accuracy before risking real money |
| 2026-03-16 | Pre-game only (no live betting) | Live betting needs real-time infra, too complex for v1 |
| 2026-03-16 | SQLite over Postgres | Zero setup, portable, sufficient for single-bot scale |

---

## Daily Performance Log

_Paper trades will be tracked here once the bot starts running._

| Date | Games | Trades | W-L | P&L | Bankroll | Notes |
|------|-------|--------|-----|-----|----------|-------|
| 2026-03-17 | — | — | — | — | $15.00 | First day, paper mode |

---

## Next Steps

1. Deploy to Linux machine
2. Run `python3 main.py init` to populate DB
3. Start paper trading: `python3 main.py run --paper`
4. After 1 week of paper trading with >55% accuracy: switch to live
5. Set `paper_mode: false` in config.json for real trading
