"""CLI entry point for the CBB Kalshi Trading Bot.

Usage:
    python main.py init              Initialize DB, fetch data, train initial model
    python main.py predict [--date]  Run predictions for today's games
    python main.py trade [--dry-run] Evaluate and execute trades
    python main.py status            Show bankroll, positions, today's trades
    python main.py review [--period] Generate performance review (daily/weekly)
    python main.py backtest          Backtest model on historical data
    python main.py run               Start the scheduler for continuous operation
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import db
from data import DataPipeline
from features import FeatureEngine, update_elos_from_results, elo_win_probability
from model import PredictionModel

# Lazy imports for commands that need them
# from kalshi import KalshiClient
# from trading import TradingEngine
# from scheduler import BotScheduler
# from review import ReviewSystem

CONFIG_PATH = Path("config.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cbb-bot")


def load_config() -> dict:
    """Load and validate config.json."""
    if not CONFIG_PATH.exists():
        log.error("config.json not found. Copy config_example.json to config.json and fill in your API keys.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    required = ["kalshi", "bankroll", "risk", "data", "scheduler"]
    for key in required:
        if key not in config:
            log.error("Missing required config key: %s", key)
            sys.exit(1)
    return config


def cmd_init(args):
    """Initialize: create DB, fetch data, train initial model."""
    log.info("=== Initializing CBB Trading Bot ===")

    # 1. Initialize database
    db.init_db()
    log.info("Database initialized")

    # 2. Load config and fetch today's data
    config = load_config()
    pipeline = DataPipeline(config)

    log.info("Fetching today's games...")
    summary = pipeline.daily_refresh()
    log.info("Data refresh: %s", summary)

    # 3. Check if we have enough data to train
    with db.get_conn() as conn:
        game_count = conn.execute(
            "SELECT COUNT(*) FROM games WHERE status = 'post'"
        ).fetchone()[0]

    if game_count < 50:
        log.warning(
            "Only %d completed games in DB. Model needs more data for good predictions.",
            game_count,
        )
        log.info(
            "The model will use Elo-based predictions until enough data accumulates."
        )
        log.info("Initialization complete (no model trained yet — insufficient data).")
        return

    # 4. Build training data and train model
    log.info("Building training data from %d games...", game_count)
    engine = FeatureEngine()
    X_rows = []
    y_rows = []

    with db.get_conn() as conn:
        games = conn.execute(
            """SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score
               FROM games g WHERE g.status = 'post'"""
        ).fetchall()

    for game in games:
        home_stats = db.get_team_stats_latest(game["home_team_id"])
        away_stats = db.get_team_stats_latest(game["away_team_id"])
        if not home_stats or not away_stats:
            continue
        features = engine.build_features(
            dict(home_stats), dict(away_stats), {"home_advantage": 1, "spread": None}
        )
        X_rows.append([features[f] for f in engine.FEATURE_NAMES])
        y_rows.append(1 if game["home_score"] > game["away_score"] else 0)

    if len(X_rows) < 50:
        log.warning("Not enough feature data (%d rows). Skipping model training.", len(X_rows))
        return

    import numpy as np

    X = np.array(X_rows)
    y = np.array(y_rows)

    model = PredictionModel()
    metrics = model.train(X, y)
    model.save()
    log.info("Model trained: %s", metrics)
    log.info("=== Initialization complete ===")


def cmd_predict(args):
    """Run predictions for today's games."""
    config = load_config()
    db.init_db()

    target_date = args.date or date.today().isoformat()
    log.info("Predicting games for %s", target_date)

    # Refresh data
    pipeline = DataPipeline(config)
    pipeline.daily_refresh()

    # Get today's games
    games = db.get_games_by_date(target_date)
    if not games:
        log.info("No games found for %s", target_date)
        return

    # Try to load model, fall back to Elo-only
    engine = FeatureEngine()
    try:
        model = PredictionModel.load_latest()
        use_model = True
        log.info("Using model v%s", model.version)
    except FileNotFoundError:
        use_model = False
        log.info("No trained model found. Using Elo-based predictions.")

    print(f"\n{'='*80}")
    print(f"  CBB PREDICTIONS — {target_date}")
    print(f"{'='*80}")
    print(f"{'Game':<40} {'Home %':>8} {'Away %':>8} {'Spread':>8} {'Status':>10}")
    print(f"{'-'*80}")

    for game in games:
        home_stats = db.get_team_stats_latest(game["home_team_id"])
        away_stats = db.get_team_stats_latest(game["away_team_id"])

        if not home_stats or not away_stats:
            continue

        home_name = game.get("home_team_name", f"Team {game['home_team_id']}")
        away_name = game.get("away_team_name", f"Team {game['away_team_id']}")

        features = engine.build_features(
            dict(home_stats), dict(away_stats), {"home_advantage": 1, "spread": None}
        )

        if use_model:
            X = engine.features_to_array(features)
            p_home = float(model.predict(X)[0])
        else:
            home_elo = home_stats.get("elo", 1500.0) if home_stats else 1500.0
            away_elo = away_stats.get("elo", 1500.0) if away_stats else 1500.0
            p_home = elo_win_probability(home_elo, away_elo, home_court=True)

        p_away = 1.0 - p_home
        spread = features.get("spread", 0.0)

        # Save prediction
        db.save_prediction(
            game["id"],
            model.version if use_model else "elo_only",
            p_home,
            p_away,
            features,
        )

        matchup = f"{away_name} @ {home_name}"
        status = game.get("status", "scheduled")
        print(f"{matchup:<40} {p_home:>7.1%} {p_away:>7.1%} {spread:>+7.1f} {status:>10}")

    print(f"{'='*80}\n")


def cmd_trade(args):
    """Evaluate and execute trades."""
    from kalshi import KalshiClient
    from trading import TradingEngine

    config = load_config()
    db.init_db()

    # Initialize components
    kalshi = KalshiClient(
        config["kalshi"]["api_key_id"],
        config["kalshi"]["private_key_path"],
        config["kalshi"].get("base_url"),
    )
    engine = TradingEngine(config, kalshi, db)

    # Check risk limits first
    risk_check = engine.check_risk_limits()
    if not risk_check["can_trade"]:
        log.warning("Cannot trade: %s", ", ".join(risk_check["reasons"]))
        print("BLOCKED:", ", ".join(risk_check["reasons"]))
        return

    target_date = date.today().isoformat()
    games = db.get_games_by_date(target_date)

    if not games:
        log.info("No games today")
        return

    # Load predictions
    evaluations = []
    for game in games:
        with db.get_conn() as conn:
            pred = conn.execute(
                "SELECT * FROM predictions WHERE game_id = ? ORDER BY id DESC LIMIT 1",
                (game["id"],),
            ).fetchone()
        if not pred:
            continue

        # Find Kalshi market
        markets = kalshi.find_cbb_markets(target_date)
        if not markets:
            log.warning("No Kalshi CBB markets found for %s", target_date)
            continue

        # Try to match market to game (simplified — match by team name)
        home_name = game.get("home_name", "")
        away_name = game.get("away_name", "")
        matched_market = None
        for m in markets:
            title = m.get("title", "").lower()
            if home_name.lower() in title or away_name.lower() in title:
                matched_market = m
                break

        if not matched_market:
            continue

        evaluation = engine.evaluate_game(
            game["id"], pred["home_win_prob"], matched_market
        )
        if evaluation["should_trade"]:
            evaluations.append(evaluation)

    if not evaluations:
        print("No trades meet criteria today.")
        return

    # Display evaluations
    print(f"\n{'='*70}")
    print(f"  TRADE EVALUATIONS — {target_date}")
    print(f"{'='*70}")
    for ev in evaluations:
        print(
            f"  {ev.get('ticker', 'N/A')}: {ev['side'].upper()} "
            f"x{ev['contracts']} @ {ev['price']}¢  "
            f"edge={ev['edge']:.1%}  kelly={ev.get('kelly_frac', 0):.2%}"
        )
    print(f"{'='*70}")

    if args.dry_run:
        print("\n[DRY RUN] No trades placed.")
        return

    # Execute
    balance = kalshi.get_balance()
    print(f"\nBalance before: ${balance / 100:.2f}" if balance else "\nBalance: unknown")

    results = engine.execute_batch(evaluations)
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        print(f"  [{status}] {r.get('order_id', 'N/A')} — {r.get('error', '')}")

    balance = kalshi.get_balance()
    print(f"Balance after: ${balance / 100:.2f}" if balance else "")


def cmd_status(args):
    """Show current bot status."""
    config = load_config()
    db.init_db()

    today = date.today().isoformat()
    state = db.get_daily_state(today)
    bankroll = db.get_bankroll()

    print(f"\n{'='*50}")
    print(f"  CBB BOT STATUS — {today}")
    print(f"{'='*50}")
    print(f"  Bankroll:          ${bankroll / 100:.2f}" if bankroll else "  Bankroll:          N/A")
    print(f"  Today's P&L:       ${state['daily_pnl_cents'] / 100:+.2f}" if state else "  Today's P&L:       $0.00")
    print(f"  Trades today:      {state['trades_today'] if state else 0}")
    print(f"  Consec. losses:    {state['consecutive_losses'] if state else 0}")
    print(f"  Cooldown active:   {'YES' if state and state['is_cooldown'] else 'No'}")

    # Open positions
    open_trades = db.get_open_trades()
    if open_trades:
        print(f"\n  Open positions ({len(open_trades)}):")
        for t in open_trades:
            print(f"    {t.get('side', '?').upper()} x{t['contracts']} @ {t['price']}¢ (edge {t['edge']:.1%})")
    else:
        print(f"\n  No open positions.")

    print(f"{'='*50}\n")


def cmd_review(args):
    """Generate performance review."""
    from review import ReviewSystem

    db.init_db()
    review = ReviewSystem()

    if args.period == "weekly":
        path = review.generate_weekly_report()
        print(f"Weekly report: {path}")
    else:
        log_date = date.today() if not hasattr(args, "date") or not args.date else date.fromisoformat(args.date)
        path = review.generate_daily_log(log_date)
        print(f"Daily log: {path}")

    # Print summary
    summary = review.get_performance_summary(days=7 if args.period == "weekly" else 1)
    print(f"\nSummary ({args.period}):")
    print(f"  Trades: {summary['total_trades']}")
    print(f"  Record: {summary['wins']}-{summary['losses']}")
    if summary["total_trades"] > 0:
        print(f"  Win rate: {summary['win_rate']:.1%}")
        print(f"  P&L: ${summary['total_pnl_cents'] / 100:+.2f}")
        print(f"  ROI: {summary['roi']:.1%}")


def cmd_run(args):
    """Start the scheduler for continuous operation."""
    from scheduler import BotScheduler

    config = load_config()

    # --paper flag overrides config
    if getattr(args, "paper", False):
        config.setdefault("trading", {})["paper_mode"] = True

    paper = config.get("trading", {}).get("paper_mode", False)
    db.init_db()

    bot = BotScheduler(config)
    mode = "PAPER MODE" if paper else "LIVE MODE"
    log.info("Starting CBB Trading Bot scheduler [%s]...", mode)
    log.info("Press Ctrl+C to stop.")
    try:
        bot.run()
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")


def cmd_backtest(args):
    """Backtest model on historical data."""
    import numpy as np

    db.init_db()
    log.info("Backtesting from %s to %s", args.start, args.end)

    engine = FeatureEngine()
    try:
        model = PredictionModel.load_latest()
    except FileNotFoundError:
        log.error("No model found. Run 'python main.py init' first.")
        return

    # Get completed games in date range
    with db.get_conn() as conn:
        games = conn.execute(
            """SELECT g.id, g.date, g.home_team_id, g.away_team_id,
                      g.home_score, g.away_score
               FROM games g
               WHERE g.status = 'post' AND g.date BETWEEN ? AND ?
               ORDER BY g.date""",
            (args.start, args.end),
        ).fetchall()

    if not games:
        log.info("No completed games found in range")
        return

    correct = 0
    total = 0
    total_pnl = 0
    results = []

    for game in games:
        home_stats = db.get_team_stats_latest(game["home_team_id"])
        away_stats = db.get_team_stats_latest(game["away_team_id"])
        if not home_stats or not away_stats:
            continue

        features = engine.build_features(
            dict(home_stats), dict(away_stats), {"home_advantage": 1, "spread": None}
        )
        X = engine.features_to_array(features)
        p_home = float(model.predict(X)[0])

        actual_home_win = game["home_score"] > game["away_score"]
        predicted_home_win = p_home >= 0.5

        if predicted_home_win == actual_home_win:
            correct += 1
        total += 1

        # Simulate P&L with 55-cent average bet on predicted side
        # (simplified — real backtesting would need market prices)
        simulated_price = 55  # cents
        if predicted_home_win == actual_home_win:
            total_pnl += (100 - simulated_price)
        else:
            total_pnl -= simulated_price

    if total == 0:
        log.info("No games with sufficient data for backtesting")
        return

    accuracy = correct / total
    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS")
    print(f"  {args.start} to {args.end}")
    print(f"{'='*50}")
    print(f"  Games tested:  {total}")
    print(f"  Correct:       {correct}")
    print(f"  Accuracy:      {accuracy:.1%}")
    print(f"  Simulated P&L: ${total_pnl / 100:+.2f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="CBB Kalshi Trading Bot")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    subparsers.add_parser("init", help="Initialize DB, fetch data, train model")

    # predict
    p_predict = subparsers.add_parser("predict", help="Run predictions")
    p_predict.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")

    # trade
    p_trade = subparsers.add_parser("trade", help="Evaluate and execute trades")
    p_trade.add_argument("--dry-run", action="store_true", help="Preview without trading")

    # status
    subparsers.add_parser("status", help="Show bot status")

    # review
    p_review = subparsers.add_parser("review", help="Generate performance review")
    p_review.add_argument("--period", choices=["daily", "weekly"], default="daily")

    # backtest
    p_backtest = subparsers.add_parser("backtest", help="Backtest model")
    p_backtest.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_backtest.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")

    # run
    p_run = subparsers.add_parser("run", help="Start scheduler")
    p_run.add_argument("--paper", action="store_true", help="Paper trading mode (no real orders)")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "predict": cmd_predict,
        "trade": cmd_trade,
        "status": cmd_status,
        "review": cmd_review,
        "backtest": cmd_backtest,
        "run": cmd_run,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
