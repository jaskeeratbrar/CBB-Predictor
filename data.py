"""
data.py — Data pipeline for fetching college basketball data from ESPN's free API.

Provides the DataPipeline class which handles all external data fetching,
parsing, and persistence via the db module.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict

import requests

import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter.

    Ensures that consecutive calls are spaced apart so that no more than
    *max_per_second* requests are issued in any one-second window.
    """

    def __init__(self, max_per_second: int = 5):
        self.min_interval = 1.0 / max_per_second
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Block until enough time has elapsed since the previous call."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """Fetches college basketball data from ESPN and persists it via *db*."""

    def __init__(self, config: dict):
        data_cfg = config.get("data", {})
        self.espn_base = data_cfg.get(
            "espn_base",
            "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball",
        )
        rate = data_cfg.get("ncaa_rate_limit_per_sec", 5)
        self.limiter = RateLimiter(max_per_second=rate)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CBB-Predictor/1.0"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Issue a rate-limited GET and return the parsed JSON, or *None*
        on any failure."""
        self.limiter.wait()
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("GET %s failed: %s", url, exc)
            return None
        except ValueError as exc:
            logger.error("JSON decode error for %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_iso_date(iso_str: str) -> Optional[str]:
        """Return an ISO-8601 datetime string normalised to UTC, or *None*."""
        if not iso_str:
            return None
        try:
            # ESPN usually returns e.g. "2026-03-16T23:00Z"
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).isoformat()
        except (ValueError, TypeError) as exc:
            logger.warning("Could not parse date '%s': %s", iso_str, exc)
            return None

    @staticmethod
    def _map_status(status_obj: dict) -> str:
        """Map ESPN's status object to one of: scheduled, in, post."""
        try:
            type_obj = status_obj.get("type", {})
            if type_obj.get("completed", False):
                return "post"
            desc = type_obj.get("description", "").lower()
            if desc in ("in progress",):
                return "in"
            # ESPN uses "Scheduled" for games not yet started
            return "scheduled"
        except Exception:
            return "scheduled"

    # ------------------------------------------------------------------
    # ESPN API methods
    # ------------------------------------------------------------------

    def fetch_todays_games(self, date: Optional[str] = None) -> list[dict]:
        """Fetch the college basketball scoreboard for a given date.

        Parameters
        ----------
        date : str, optional
            Date in *YYYYMMDD* format.  Defaults to today (UTC).

        Returns
        -------
        list[dict]
            Each dict contains normalised game information.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y%m%d")
        # Accept YYYY-MM-DD or YYYYMMDD
        date = date.replace("-", "")

        url = f"{self.espn_base}/scoreboard"
        params = {"dates": date, "groups": "50", "limit": "365"}

        data = self._get(url, params=params)
        if data is None:
            return []

        games: list[dict] = []
        for event in data.get("events", []):
            try:
                game = self._parse_event(event)
                if game:
                    games.append(game)
            except Exception as exc:
                logger.error(
                    "Error parsing event %s: %s",
                    event.get("id", "?"),
                    exc,
                )
        logger.info("Fetched %d games for date %s", len(games), date)
        return games

    def _parse_event(self, event: dict) -> Optional[dict]:
        """Parse a single ESPN *event* object into a normalised game dict."""
        espn_game_id = event.get("id")
        game_date = self._parse_iso_date(event.get("date", ""))
        status = self._map_status(event.get("status", {}))

        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors", [])

        home: Optional[dict] = None
        away: Optional[dict] = None
        for comp in competitors:
            team_info = comp.get("team", {})
            entry = {
                "espn_team_id": team_info.get("id"),
                "name": team_info.get("displayName"),
                "abbreviation": team_info.get("abbreviation"),
                "score": comp.get("score"),
            }
            if comp.get("homeAway") == "home":
                home = entry
            else:
                away = entry

        if not home or not away:
            logger.warning(
                "Skipping event %s — could not identify home/away", espn_game_id
            )
            return None

        # Odds / spread (may not be present)
        odds_list = competition.get("odds", [])
        odds_info: dict = {}
        if odds_list:
            first = odds_list[0]
            odds_info = {
                "details": first.get("details"),
                "over_under": first.get("overUnder"),
                "spread": first.get("spread"),
            }

        return {
            "espn_game_id": espn_game_id,
            "date": game_date,
            "status": status,
            "home": home,
            "away": away,
            "odds": odds_info,
        }

    def fetch_team_roster_stats(self, espn_team_id: str) -> dict:
        """Fetch team-level aggregate statistics from ESPN.

        Parameters
        ----------
        espn_team_id : str
            ESPN numeric team identifier.

        Returns
        -------
        dict
            Normalised stats (points per game, rebounds, assists, etc.).
            Empty dict on failure.
        """
        url = f"{self.espn_base}/teams/{espn_team_id}/statistics"
        data = self._get(url)
        if data is None:
            return {}

        stats: dict = {}
        try:
            # ESPN nests stats under results[].stats[]
            results = data.get("results") or data.get("resultsSets") or []

            # The top-level "splits" approach (seen in the team stats endpoint)
            splits = data.get("splits", {})
            categories = splits.get("categories", [])
            if not categories and isinstance(results, list):
                # Try alternate structure
                for r in results:
                    categories.extend(r.get("categories", []))

            for category in categories:
                for stat in category.get("stats", []):
                    name = stat.get("abbreviation") or stat.get("name", "")
                    value = stat.get("value")
                    if name and value is not None:
                        stats[name.lower()] = value

            # Also try a flat stats list directly on the response
            if not stats:
                for stat in data.get("stats", []):
                    name = stat.get("abbreviation") or stat.get("name", "")
                    value = stat.get("value")
                    if name and value is not None:
                        stats[name.lower()] = value

        except Exception as exc:
            logger.error(
                "Error parsing stats for team %s: %s", espn_team_id, exc
            )

        logger.debug(
            "Fetched %d stat fields for team %s", len(stats), espn_team_id
        )
        return stats

    def fetch_standings(self) -> list[dict]:
        """Fetch current conference standings for Division I basketball.

        Returns
        -------
        list[dict]
            Each dict has team identification info plus win/loss records.
        """
        url = f"{self.espn_base}/standings"
        params = {"groups": "50"}
        data = self._get(url, params=params)
        if data is None:
            return []

        standings: list[dict] = []
        try:
            for child in data.get("children", []):
                conference = child.get("name", "Unknown")
                for entry in child.get("standings", {}).get("entries", []):
                    team_info = entry.get("team", {})
                    record_stats = {
                        s.get("abbreviation", s.get("name", "")).lower(): s.get("value")
                        for s in entry.get("stats", [])
                        if s.get("abbreviation") or s.get("name")
                    }
                    standings.append(
                        {
                            "espn_team_id": team_info.get("id"),
                            "name": team_info.get("displayName"),
                            "abbreviation": team_info.get("abbreviation"),
                            "conference": conference,
                            "wins": record_stats.get("wins") or record_stats.get("w"),
                            "losses": record_stats.get("losses") or record_stats.get("l"),
                            "conference_wins": record_stats.get("confwins") or record_stats.get("cw"),
                            "conference_losses": record_stats.get("conflosses") or record_stats.get("cl"),
                            "raw_stats": record_stats,
                        }
                    )
        except Exception as exc:
            logger.error("Error parsing standings: %s", exc)

        logger.info("Fetched standings for %d teams", len(standings))
        return standings

    # ------------------------------------------------------------------
    # Orchestration methods
    # ------------------------------------------------------------------

    def daily_refresh(self, target_date: str = None) -> dict:
        """Run the full daily data refresh cycle.

        Parameters
        ----------
        target_date : str, optional
            Date to fetch (YYYY-MM-DD or YYYYMMDD). Defaults to today.

        Returns
        -------
        dict
            ``{games_found, teams_updated, errors}``
        """
        errors: list[str] = []
        teams_updated = 0
        games_found = 0

        # 1 — Fetch games
        today = target_date or datetime.now(timezone.utc).strftime("%Y%m%d")
        games = self.fetch_todays_games(today)
        games_found = len(games)

        # 2/3 — Upsert teams & insert games
        seen_team_ids: set = set()
        current_season = datetime.now(timezone.utc).strftime("%Y")
        for game in games:
            home_db_id = None
            away_db_id = None
            for side in ("home", "away"):
                team = game[side]
                try:
                    tid = db.upsert_team(
                        name=team["name"],
                        espn_id=team["espn_team_id"],
                        abbreviation=team["abbreviation"],
                    )
                    if side == "home":
                        home_db_id = tid
                    else:
                        away_db_id = tid
                    seen_team_ids.add(team["espn_team_id"])
                    teams_updated += 1
                except Exception as exc:
                    msg = f"upsert_team failed for {team.get('name')}: {exc}"
                    logger.error(msg)
                    errors.append(msg)

            if home_db_id and away_db_id:
                try:
                    raw_date = game.get("date", today)
                    # Normalize to YYYY-MM-DD for consistent querying
                    game_date = raw_date[:10] if len(raw_date) >= 10 else raw_date
                    db.insert_game(
                        date=game_date,
                        home_id=home_db_id,
                        away_id=away_db_id,
                        espn_game_id=game["espn_game_id"],
                        season=current_season,
                    )
                except Exception as exc:
                    msg = f"insert_game failed for {game.get('espn_game_id')}: {exc}"
                    logger.error(msg)
                    errors.append(msg)

        # 4 — Standings
        standings = self.fetch_standings()
        for entry in standings:
            tid = entry.get("espn_team_id")
            if tid and tid not in seen_team_ids:
                try:
                    db.upsert_team(
                        espn_id=tid,
                        name=entry["name"],
                        abbreviation=entry.get("abbreviation"),
                    )
                    seen_team_ids.add(tid)
                    teams_updated += 1
                except Exception as exc:
                    msg = f"upsert_team (standings) failed for {entry.get('name')}: {exc}"
                    logger.error(msg)
                    errors.append(msg)

        # 5 — Team stats snapshots from standings data
        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for entry in standings:
            tid = entry.get("espn_team_id")
            if not tid:
                continue
            team_row = db.get_team_by_espn_id(tid)
            if not team_row:
                continue
            try:
                wins = int(entry.get("wins") or 0)
                losses = int(entry.get("losses") or 0)
                db.save_team_stats(
                    team_id=team_row["id"],
                    season=current_season,
                    date=today_iso,
                    wins=wins,
                    losses=losses,
                )
            except Exception as exc:
                msg = f"save_team_stats failed for team {tid}: {exc}"
                logger.error(msg)
                errors.append(msg)

        summary = {
            "games_found": games_found,
            "teams_updated": teams_updated,
            "errors": errors,
        }
        logger.info("daily_refresh complete: %s", summary)
        return summary

    def update_results(self) -> dict:
        """Check for completed games and update scores in the database.

        Returns
        -------
        dict
            ``{games_settled, results}``
        """
        results: list[dict] = []
        games_settled = 0

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        scheduled_games = db.get_games_by_date(today)

        if not scheduled_games:
            logger.info("update_results: no scheduled games found for %s", today)
            return {"games_settled": 0, "results": []}

        # Re-fetch the scoreboard to get live/final data
        today_fmt = datetime.now(timezone.utc).strftime("%Y%m%d")
        live_games = self.fetch_todays_games(today_fmt)
        live_lookup: dict[str, dict] = {
            g["espn_game_id"]: g for g in live_games
        }

        for db_game in scheduled_games:
            espn_id = str(db_game.get("espn_game_id", ""))
            if espn_id not in live_lookup:
                continue

            live = live_lookup[espn_id]
            if live["status"] != "post":
                continue

            home_score = live["home"].get("score")
            away_score = live["away"].get("score")
            try:
                home_score = int(home_score) if home_score is not None else None
                away_score = int(away_score) if away_score is not None else None
            except (ValueError, TypeError):
                logger.warning(
                    "Non-numeric score for game %s: home=%s away=%s",
                    espn_id, home_score, away_score,
                )
                continue

            try:
                db.update_game_result(
                    game_id=db_game["id"],
                    home_score=home_score,
                    away_score=away_score,
                    status="post",
                )
                games_settled += 1
                results.append(
                    {
                        "espn_game_id": espn_id,
                        "home_score": home_score,
                        "away_score": away_score,
                    }
                )
            except Exception as exc:
                logger.error("update_game_result failed for %s: %s", espn_id, exc)

        summary = {"games_settled": games_settled, "results": results}
        logger.info("update_results complete: %s", summary)
        return summary

    def pre_game_refresh(self, game_ids: list[int]) -> None:
        """Targeted stat refresh for specific games right before prediction.

        Fetches the latest team statistics for every team involved in the
        supplied *game_ids* and persists them via the database layer.

        Parameters
        ----------
        game_ids : list[int]
            Database primary-key IDs of games to refresh.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_games = db.get_games_by_date(today)
        if not all_games:
            logger.info("pre_game_refresh: no games returned for %s", today)
            return

        # Build a set of ESPN team IDs that participate in the requested games
        team_ids_to_refresh: set[str] = set()
        for g in all_games:
            if g.get("id") in game_ids:
                for key in ("home_team_espn_id", "away_team_espn_id"):
                    tid = g.get(key)
                    if tid:
                        team_ids_to_refresh.add(str(tid))

        if not team_ids_to_refresh:
            logger.info("pre_game_refresh: no matching teams for game_ids %s", game_ids)
            return

        for tid in team_ids_to_refresh:
            try:
                stats = self.fetch_team_roster_stats(tid)
                if stats:
                    db.save_team_stats(espn_team_id=tid, stats=stats)
                    logger.debug("Refreshed stats for team %s", tid)
            except Exception as exc:
                logger.error(
                    "pre_game_refresh: save_team_stats failed for team %s: %s",
                    tid,
                    exc,
                )

        logger.info(
            "pre_game_refresh complete: refreshed %d teams for %d games",
            len(team_ids_to_refresh),
            len(game_ids),
        )
