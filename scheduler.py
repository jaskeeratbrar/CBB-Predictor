"""
scheduler.py -- APScheduler-based job runner for the CBB trading bot.

Orchestrates the daily workflow:
    data refresh -> predictions -> trading -> settlement -> reporting

All jobs are wrapped in try/except so that a single failure never crashes
the scheduler.  Timing information is logged for every job invocation.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

import db
from data import DataPipeline
from features import FeatureEngine, update_elos_from_results
from kalshi import KalshiClient
from model import PredictionModel
from trading import TradingEngine
from review import ReviewSystem

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def _load_config() -> dict:
    """Read config.json from the project root."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


class BotScheduler:
    """APScheduler-based orchestrator for the CBB trading bot.

    Registers cron / interval / one-shot jobs that drive the full daily
    lifecycle: data ingestion, pre-game prediction, trade execution,
    settlement, and reporting.
    """

    def __init__(self, config: dict):
        self.config = config
        tz = config.get("scheduler", {}).get("timezone", "US/Eastern")
        self.scheduler = BlockingScheduler(timezone=tz)
        self._init_components()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Instantiate every subsystem the scheduler drives."""
        db.init_db()

        self.paper_mode = self.config.get("trading", {}).get("paper_mode", False)
        self.data = DataPipeline(self.config)
        self.features = FeatureEngine()

        try:
            self.model = PredictionModel.load_latest()
            log.info("Loaded model v%s", self.model.version)
        except FileNotFoundError:
            log.warning("No saved model found -- predictions will use Elo-only")
            self.model = PredictionModel()

        # In paper mode, Kalshi client is optional
        self.kalshi = None
        if not self.paper_mode:
            kalshi_cfg = self.config.get("kalshi", {})
            try:
                self.kalshi = KalshiClient(
                    api_key_id=kalshi_cfg["api_key_id"],
                    private_key_path=kalshi_cfg["private_key_path"],
                    base_url=kalshi_cfg.get("base_url"),
                )
            except Exception as exc:
                log.error("Failed to init KalshiClient: %s", exc)
        else:
            log.info("PAPER MODE — Kalshi client disabled, trades will be simulated")

        self.trading = TradingEngine(
            config=self.config,
            kalshi_client=self.kalshi,
        )

        self.review = ReviewSystem()

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def setup_jobs(self) -> None:
        """Register every scheduled job on the APScheduler instance.

        Jobs
        ----
        1. daily_data_refresh  -- Cron at 08:00 ET
        2. settle_and_update   -- Interval every 30 min, 12:00-01:00 ET
        3. daily_review        -- Cron at 23:59 ET
        4. weekly_retrain      -- Cron Monday 06:00 ET
        5. weekly_review       -- Cron Sunday 23:50 ET
        """
        sched_cfg = self.config.get("scheduler", {})
        refresh_hour = sched_cfg.get("daily_data_refresh_hour", 8)

        # 1 -- Morning data refresh
        self.scheduler.add_job(
            self.job_daily_refresh,
            trigger=CronTrigger(hour=refresh_hour, minute=0),
            id="daily_data_refresh",
            name="Daily data refresh",
            misfire_grace_time=3600,
            coalesce=True,
            replace_existing=True,
        )

        # 2 -- Settle & update results (every 30 min from noon to 1 AM)
        self.scheduler.add_job(
            self.job_settle,
            trigger=IntervalTrigger(minutes=30),
            id="settle_and_update",
            name="Settle trades & update results",
            misfire_grace_time=3600,
            coalesce=True,
            replace_existing=True,
        )

        # 3 -- End-of-day review
        self.scheduler.add_job(
            self.job_daily_review,
            trigger=CronTrigger(hour=23, minute=59),
            id="daily_review",
            name="Daily review log",
            misfire_grace_time=3600,
            coalesce=True,
            replace_existing=True,
        )

        # 4 -- Weekly retrain (Monday 6 AM)
        self.scheduler.add_job(
            self.job_weekly_retrain,
            trigger=CronTrigger(day_of_week="mon", hour=6, minute=0),
            id="weekly_retrain",
            name="Weekly model retrain",
            misfire_grace_time=3600,
            coalesce=True,
            replace_existing=True,
        )

        # 5 -- Weekly review (Sunday 11:50 PM)
        self.scheduler.add_job(
            self.job_weekly_review,
            trigger=CronTrigger(day_of_week="sun", hour=23, minute=50),
            id="weekly_review",
            name="Weekly summary report",
            misfire_grace_time=3600,
            coalesce=True,
            replace_existing=True,
        )

        log.info("Registered %d scheduled jobs", len(self.scheduler.get_jobs()))

    # ------------------------------------------------------------------
    # Dynamic pre-game scheduling
    # ------------------------------------------------------------------

    def schedule_pre_game_jobs(self, games: list[dict]) -> None:
        """Create a one-shot job for each game, firing N minutes before tip.

        Parameters
        ----------
        games : list[dict]
            Game dicts as returned by ``DataPipeline.fetch_todays_games()``.
            Each must contain ``espn_game_id`` and ``date`` (ISO-8601).
        """
        sched_cfg = self.config.get("scheduler", {})
        minutes_before = sched_cfg.get("prediction_minutes_before_tip", 30)
        scheduled = 0

        for game in games:
            espn_id = game.get("espn_game_id")
            game_dt_str = game.get("date")
            if not espn_id or not game_dt_str:
                continue

            # Parse tip-off time
            try:
                tip_off = datetime.fromisoformat(game_dt_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                log.warning("Cannot parse tip-off time for game %s: %s", espn_id, game_dt_str)
                continue

            run_at = tip_off - timedelta(minutes=minutes_before)

            # Skip if the run time is already in the past
            if run_at <= datetime.now(run_at.tzinfo):
                log.debug("Skipping pre-game job for %s — run time already passed", espn_id)
                continue

            job_id = f"pregame_{espn_id}"

            # Remove any existing job with the same ID to avoid duplicates
            existing = self.scheduler.get_job(job_id)
            if existing:
                log.debug("Pre-game job %s already scheduled, replacing", job_id)
                self.scheduler.remove_job(job_id)

            self.scheduler.add_job(
                self.job_pre_game,
                trigger=DateTrigger(run_date=run_at),
                args=[espn_id],
                id=job_id,
                name=f"Pre-game: {game.get('home', {}).get('name', '?')} vs "
                     f"{game.get('away', {}).get('name', '?')}",
                misfire_grace_time=3600,
                coalesce=True,
                replace_existing=True,
            )
            scheduled += 1

        log.info("Scheduled %d pre-game jobs (of %d games)", scheduled, len(games))

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    def job_daily_refresh(self) -> None:
        """Morning data refresh: pull today's schedule, refresh stats,
        and dynamically schedule pre-game prediction jobs."""
        start = time.monotonic()
        log.info("=== job_daily_refresh START ===")
        try:
            # Reload config in case bankroll or risk params changed
            self.config = _load_config()

            # Run the data pipeline
            summary = self.data.daily_refresh()
            log.info("Data refresh summary: %s", summary)

            # Fetch today's games and create pre-game jobs
            today = datetime.now().strftime("%Y%m%d")
            games = self.data.fetch_todays_games(today)
            self.schedule_pre_game_jobs(games)

        except Exception:
            log.exception("job_daily_refresh FAILED")
        finally:
            elapsed = time.monotonic() - start
            log.info("=== job_daily_refresh END (%.1fs) ===", elapsed)

    def job_pre_game(self, game_id: str) -> None:
        """Pre-game prediction and trade execution for a single game.

        Parameters
        ----------
        game_id : str
            ESPN game ID (``espn_game_id``).
        """
        start = time.monotonic()
        log.info("=== job_pre_game START [game %s] ===", game_id)
        try:
            # 1 -- Refresh data for this game's teams
            game_row = db.get_team_by_espn_id(game_id)  # look up DB row
            today = datetime.now().strftime("%Y-%m-%d")
            today_games = db.get_games_by_date(today)

            target_game = None
            for g in today_games:
                if str(g.get("espn_game_id")) == str(game_id):
                    target_game = g
                    break

            if not target_game:
                log.warning("Game %s not found in DB for today (%s)", game_id, today)
                return

            # Refresh stats for the two teams
            self.data.pre_game_refresh([target_game["id"]])

            # 2 -- Gather latest team stats
            home_stats = db.get_team_stats_latest(target_game["home_team_id"]) or {}
            away_stats = db.get_team_stats_latest(target_game["away_team_id"]) or {}

            # 3 -- Build features
            game_context = {
                "home_advantage": 1,
                "spread": target_game.get("spread"),
            }
            features = self.features.build_features(home_stats, away_stats, game_context)
            log.info("Features for game %s: %s", game_id, features)

            # 4 -- Predict
            X = self.features.features_to_array(features)
            home_win_prob = float(self.model.predict(X)[0])
            away_win_prob = 1.0 - home_win_prob
            log.info(
                "Prediction for game %s: P(home)=%.3f  P(away)=%.3f",
                game_id, home_win_prob, away_win_prob,
            )

            # Persist prediction
            pred_id = db.save_prediction(
                game_id=target_game["id"],
                model_version=self.model.version,
                home_prob=home_win_prob,
                away_prob=away_win_prob,
                features=features,
            )

            # 5 -- Find Kalshi market for this game
            today_iso = datetime.now().strftime("%Y-%m-%d")
            markets = self.kalshi.find_cbb_markets(date=today_iso)

            # Try to match by team names
            home_name = target_game.get("home_team_name", "").lower()
            away_name = target_game.get("away_team_name", "").lower()
            home_abbr = target_game.get("home_team_abbr", "").lower()
            away_abbr = target_game.get("away_team_abbr", "").lower()

            matched_market = None
            for mkt in markets:
                title_lower = mkt.get("title", "").lower()
                ticker_lower = mkt.get("ticker", "").lower()
                search_text = f"{title_lower} {ticker_lower}"
                # Match if both team abbreviations or names appear
                home_match = home_abbr in search_text or home_name in search_text
                away_match = away_abbr in search_text or away_name in search_text
                if home_match and away_match:
                    matched_market = mkt
                    break

            if not matched_market:
                log.info("No Kalshi market found for game %s (%s vs %s)", game_id, home_name, away_name)
                return

            log.info("Matched Kalshi market: %s", matched_market.get("ticker"))

            # Save market snapshot
            market_id = db.save_market(
                game_id=target_game["id"],
                kalshi_ticker=matched_market["ticker"],
                event_ticker=matched_market.get("event_ticker", ""),
                yes_price=matched_market.get("yes_price", 0),
                no_price=matched_market.get("no_price", 0),
                volume=matched_market.get("volume", 0),
            )

            # 6 -- Evaluate trade
            trade_decision = self.trading.evaluate_game(
                prediction={
                    "id": pred_id,
                    "home_win_prob": home_win_prob,
                    "away_win_prob": away_win_prob,
                },
                market=matched_market,
            )

            if not trade_decision:
                log.info("Trade evaluation returned no action for game %s", game_id)
                return

            # 7 -- Execute if passing all checks
            log.info("Trade decision for game %s: %s", game_id, trade_decision)
            self.trading.execute_trade(trade_decision)

        except Exception:
            log.exception("job_pre_game FAILED for game %s", game_id)
        finally:
            elapsed = time.monotonic() - start
            log.info("=== job_pre_game END [game %s] (%.1fs) ===", game_id, elapsed)

    def job_settle(self) -> None:
        """Check for completed games, update scores, and settle trades."""
        start = time.monotonic()
        log.info("=== job_settle START ===")
        try:
            # 1 -- Update results from ESPN
            result_summary = self.data.update_results()
            log.info("Results update: %s", result_summary)

            # 2 -- Settle open trades
            settlements = self.trading.settle_trades()
            log.info("Settlements: %s", settlements)

            # 3 -- Update Elo ratings for newly completed games
            completed = result_summary.get("results", [])
            if completed:
                elo_games = []
                for r in completed:
                    espn_id = r.get("espn_game_id")
                    # Look up DB game to get team IDs
                    today = datetime.now().strftime("%Y-%m-%d")
                    today_games = db.get_games_by_date(today)
                    for g in today_games:
                        if str(g.get("espn_game_id")) == str(espn_id):
                            elo_games.append({
                                "home_team_id": g["home_team_id"],
                                "away_team_id": g["away_team_id"],
                                "home_score": r["home_score"],
                                "away_score": r["away_score"],
                                "date": today,
                                "season": g.get("season", ""),
                            })
                            break

                if elo_games:
                    updated_elos = update_elos_from_results(elo_games)
                    log.info("Updated Elo for %d teams", len(updated_elos))

        except Exception:
            log.exception("job_settle FAILED")
        finally:
            elapsed = time.monotonic() - start
            log.info("=== job_settle END (%.1fs) ===", elapsed)

    def job_daily_review(self) -> None:
        """Generate end-of-day review log."""
        start = time.monotonic()
        log.info("=== job_daily_review START ===")
        try:
            self.review.generate_daily_log()
        except Exception:
            log.exception("job_daily_review FAILED")
        finally:
            elapsed = time.monotonic() - start
            log.info("=== job_daily_review END (%.1fs) ===", elapsed)

    def job_weekly_retrain(self) -> None:
        """Weekly model retraining on all historical data."""
        start = time.monotonic()
        log.info("=== job_weekly_retrain START ===")
        try:
            # 1 -- Pull all historical completed games from DB
            from datetime import date as date_type
            all_games: list[dict] = []
            # Scan back through recent seasons
            today = datetime.now().strftime("%Y-%m-%d")
            # Get all games with results
            with db.get_conn() as conn:
                rows = conn.execute(
                    """SELECT g.*,
                              ht.name AS home_team_name,
                              at.name AS away_team_name
                       FROM games g
                       JOIN teams ht ON g.home_team_id = ht.id
                       JOIN teams at ON g.away_team_id = at.id
                       WHERE g.status = 'post'
                         AND g.home_score IS NOT NULL
                         AND g.away_score IS NOT NULL
                       ORDER BY g.date"""
                ).fetchall()
                all_games = [dict(r) for r in rows]

            if not all_games:
                log.warning("No completed games found in DB -- skipping retrain")
                return

            log.info("Building features from %d historical games", len(all_games))

            # 2 -- Build features for every game
            training_data: list[dict] = []
            for game in all_games:
                home_stats = db.get_team_stats_latest(game["home_team_id"]) or {}
                away_stats = db.get_team_stats_latest(game["away_team_id"]) or {}
                game_context = {"home_advantage": 1, "spread": None}
                try:
                    feats = self.features.build_features(home_stats, away_stats, game_context)
                    label = 1 if game["home_score"] > game["away_score"] else 0
                    training_data.append({"features": feats, "label": label})
                except Exception:
                    log.debug("Skipping game %s during feature build", game.get("espn_game_id"))

            if len(training_data) < 50:
                log.warning("Only %d training samples -- skipping retrain (need >= 50)", len(training_data))
                return

            import numpy as np
            X = np.array(
                [[td["features"][f] for f in FeatureEngine.FEATURE_NAMES] for td in training_data],
                dtype=np.float64,
            )
            y = np.array([td["label"] for td in training_data], dtype=np.float64)

            # 3 -- Retrain (bumps version, saves model)
            metrics = self.model.retrain(X, y)
            log.info("Retrain complete. Metrics: %s", metrics)

            # 4 -- Reload model so future predictions use the new version
            self.model = PredictionModel.load_latest()
            log.info("Now using model v%s", self.model.version)

        except Exception:
            log.exception("job_weekly_retrain FAILED")
        finally:
            elapsed = time.monotonic() - start
            log.info("=== job_weekly_retrain END (%.1fs) ===", elapsed)

    def job_weekly_review(self) -> None:
        """Generate weekly summary report."""
        start = time.monotonic()
        log.info("=== job_weekly_review START ===")
        try:
            self.review.generate_weekly_report()
        except Exception:
            log.exception("job_weekly_review FAILED")
        finally:
            elapsed = time.monotonic() - start
            log.info("=== job_weekly_review END (%.1fs) ===", elapsed)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Set up all jobs and start the blocking scheduler."""
        self.setup_jobs()
        log.info("Scheduler starting with %d jobs", len(self.scheduler.get_jobs()))
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("Scheduler shutting down")

    def run_once(self, job_name: str) -> None:
        """Execute a single job immediately (for manual / CLI use).

        Parameters
        ----------
        job_name : str
            One of: ``'refresh'``, ``'settle'``, ``'review'``,
            ``'retrain'``, ``'weekly_review'``.

        Raises
        ------
        ValueError
            If *job_name* is not recognised.
        """
        dispatch = {
            "refresh": self.job_daily_refresh,
            "settle": self.job_settle,
            "review": self.job_daily_review,
            "retrain": self.job_weekly_retrain,
            "weekly_review": self.job_weekly_review,
        }
        func = dispatch.get(job_name)
        if func is None:
            raise ValueError(
                f"Unknown job '{job_name}'. Valid names: {', '.join(sorted(dispatch))}"
            )
        log.info("Manual run: %s", job_name)
        func()


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="CBB Trading Bot Scheduler")
    parser.add_argument(
        "--run-once",
        metavar="JOB",
        help="Run a single job immediately then exit. "
             "Choices: refresh, settle, review, retrain, weekly_review",
    )
    args = parser.parse_args()

    config = _load_config()
    bot = BotScheduler(config)

    if args.run_once:
        bot.run_once(args.run_once)
    else:
        bot.run()
