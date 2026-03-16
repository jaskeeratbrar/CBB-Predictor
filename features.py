"""Feature engineering for CBB game predictions.

Transforms raw team stats into model-ready feature vectors.
Manages Elo rating system with K=20, base 1500.
"""

import logging
import numpy as np
from typing import Optional

import db

log = logging.getLogger(__name__)

# Elo constants
ELO_BASE = 1500.0
ELO_K = 20
HOME_ADVANTAGE_ELO = 100  # ~64% expected win rate for home team


class FeatureEngine:
    """Builds feature vectors from raw team stats for game prediction."""

    FEATURE_NAMES = [
        "elo_diff",
        "win_pct_diff",
        "home_advantage",
        "off_eff_diff",
        "def_eff_diff",
        "spread",
    ]

    def build_features(
        self,
        home_stats: dict,
        away_stats: dict,
        game_context: dict,
    ) -> dict:
        """Build feature dict for a single game prediction.

        Args:
            home_stats: Latest team_stats row for home team.
            away_stats: Latest team_stats row for away team.
            game_context: Dict with keys: home_advantage (1/0/-1), spread (float or None).

        Returns:
            Dict of feature_name -> float.
        """
        home_elo = home_stats.get("elo", ELO_BASE)
        away_elo = away_stats.get("elo", ELO_BASE)

        home_wins = home_stats.get("wins", 0)
        home_losses = home_stats.get("losses", 0)
        away_wins = away_stats.get("wins", 0)
        away_losses = away_stats.get("losses", 0)

        home_total = home_wins + home_losses
        away_total = away_wins + away_losses
        home_win_pct = home_wins / home_total if home_total > 0 else 0.5
        away_win_pct = away_wins / away_total if away_total > 0 else 0.5

        home_off = home_stats.get("offensive_efficiency") or 100.0
        away_off = away_stats.get("offensive_efficiency") or 100.0
        home_def = home_stats.get("defensive_efficiency") or 100.0
        away_def = away_stats.get("defensive_efficiency") or 100.0

        spread = game_context.get("spread")
        if spread is None:
            # Estimate spread from Elo diff (~30 Elo points per point of spread)
            spread = (home_elo - away_elo + HOME_ADVANTAGE_ELO * game_context.get("home_advantage", 0)) / 30.0

        return {
            "elo_diff": home_elo - away_elo,
            "win_pct_diff": home_win_pct - away_win_pct,
            "home_advantage": float(game_context.get("home_advantage", 0)),
            "off_eff_diff": home_off - away_off,
            "def_eff_diff": home_def - away_def,  # positive = home has worse defense
            "spread": spread,
        }

    def build_batch(self, games: list[dict]) -> tuple:
        """Build feature matrix for multiple games.

        Args:
            games: List of dicts, each with home_stats, away_stats, game_context.

        Returns:
            (X as np.ndarray, feature_names as list[str])
        """
        rows = []
        for g in games:
            feats = self.build_features(
                g["home_stats"], g["away_stats"], g["game_context"]
            )
            rows.append([feats[f] for f in self.FEATURE_NAMES])
        X = np.array(rows, dtype=np.float64) if rows else np.empty((0, len(self.FEATURE_NAMES)))
        return X, self.FEATURE_NAMES

    def features_to_array(self, features: dict) -> np.ndarray:
        """Convert single feature dict to 1-row array for prediction."""
        return np.array([[features[f] for f in self.FEATURE_NAMES]], dtype=np.float64)


# --- Elo system ---

def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(
    winner_elo: float, loser_elo: float, k: int = ELO_K
) -> tuple:
    """Update Elo ratings after a game.

    Returns:
        (new_winner_elo, new_loser_elo)
    """
    expected_w = elo_expected(winner_elo, loser_elo)
    new_winner = winner_elo + k * (1.0 - expected_w)
    new_loser = loser_elo + k * (0.0 - (1.0 - expected_w))
    return round(new_winner, 1), round(new_loser, 1)


def update_elos_from_results(completed_games: list[dict]) -> dict:
    """Batch-update Elo ratings from completed game results.

    Args:
        completed_games: List of dicts with keys:
            home_team_id, away_team_id, home_score, away_score, date, season

    Returns:
        Dict of team_id -> new_elo
    """
    updated = {}
    for game in completed_games:
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        home_score = game["home_score"]
        away_score = game["away_score"]

        # Get current Elo from DB or cache
        if home_id in updated:
            home_elo = updated[home_id]
        else:
            stats = db.get_team_stats_latest(home_id)
            home_elo = stats["elo"] if stats else ELO_BASE

        if away_id in updated:
            away_elo = updated[away_id]
        else:
            stats = db.get_team_stats_latest(away_id)
            away_elo = stats["elo"] if stats else ELO_BASE

        if home_score > away_score:
            home_elo, away_elo = elo_update(home_elo, away_elo)
        elif away_score > home_score:
            away_elo, home_elo = elo_update(away_elo, home_elo)
        # Tie: no update (rare in CBB)

        updated[home_id] = home_elo
        updated[away_id] = away_elo

        # Persist to DB
        date = game.get("date", "")
        season = game.get("season", "")
        for tid, elo in [(home_id, home_elo), (away_id, away_elo)]:
            existing = db.get_team_stats_latest(tid)
            if existing:
                db.save_team_stats(
                    tid, season, date,
                    wins=existing.get("wins", 0),
                    losses=existing.get("losses", 0),
                    offensive_efficiency=existing.get("offensive_efficiency"),
                    defensive_efficiency=existing.get("defensive_efficiency"),
                    tempo=existing.get("tempo"),
                    sos=existing.get("sos"),
                    elo=elo,
                )

    log.info("Updated Elo for %d teams from %d games", len(updated), len(completed_games))
    return updated


def elo_win_probability(home_elo: float, away_elo: float, home_court: bool = True) -> float:
    """Quick Elo-based win probability for home team.

    Includes home-court advantage adjustment.
    """
    adj = HOME_ADVANTAGE_ELO if home_court else 0
    return elo_expected(home_elo + adj, away_elo)
