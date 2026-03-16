# CBB Kalshi Auto-Trading Bot

College basketball prediction and Kalshi trading bot. Binary outcome markets. Small bankroll ($10-20).

## Commands

```bash
python3 main.py init              # Initialize DB, fetch data, train model
python3 main.py predict [--date]  # Run predictions for today's games
python3 main.py trade [--dry-run] # Evaluate and execute trades
python3 main.py status            # Bankroll, positions, today's trades
python3 main.py review [--period] # Generate daily/weekly performance review
python3 main.py backtest --start YYYY-MM-DD --end YYYY-MM-DD
python3 main.py run               # Start scheduler (continuous mode)
```

## Architecture

Six core modules, one file each:
- `db.py` — SQLite schema + CRUD (8 tables: teams, games, team_stats, predictions, markets, trades, model_versions, daily_state)
- `data.py` — ESPN + NCAA free API pipeline (no auth needed)
- `features.py` — Feature engineering (6 features) + Elo system (K=20, base 1500)
- `model.py` — Calibrated LogReg ensemble (XGBoost added later). Bayesian updating between retrains.
- `kalshi.py` — Kalshi API client with RSA-PSS SHA256 auth
- `trading.py` — Kelly criterion sizing + 7 risk management rules

Orchestration:
- `scheduler.py` — APScheduler jobs (daily refresh, pre-game trades, settlement, weekly retrain)
- `review.py` — MD report generation (daily logs in logs/, weekly in reports/)
- `main.py` — CLI entry point

## Risk Rules (NEVER bypass these)

1. Max 5% of bankroll per trade
2. 10% fractional Kelly
3. 15% daily loss limit
4. 3-loss cooldown (rest of day)
5. 3% minimum edge threshold
6. Max 5 open positions
7. Stop trading if bankroll < $2

## Key Conventions

- All monetary values stored in **cents** (int, not float). $15 = 1500.
- SQLite database at `cbb.db` — never delete without backup
- Model artifacts in `models/` as pickle files
- Config at `config.json` (gitignored) — **never display API keys**
- Kalshi private key at `~/.kalshi/private_key.pem` — outside project
- Python 3.9+, no bare `pip` — use `python3 -m pip`
- Use `logging` module everywhere, not print (except CLI output)

## Data Sources

1. ESPN API (primary): `site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball` — free, no auth
2. NCAA API (secondary): 5 req/s rate limit, no auth
3. Historical: sportsreference package

## Workflow

Daily: 8am data refresh → 30min pre-game predictions → trade if edge > 3% → settle after games → daily log
Weekly: Monday 6am retrain model → Sunday 11:50pm weekly report
