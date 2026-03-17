"""
SQLite database layer for the CBB Kalshi trading bot.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

DB_PATH = "cbb.db"

# ---------------------------------------------------------------------------
# Default bankroll (cents) used when creating a brand-new daily_state row and
# no prior row exists.  Matches config_example.json → bankroll.initial_cents.
# ---------------------------------------------------------------------------
_DEFAULT_INITIAL_BANKROLL = 1500


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Yield a connection with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not already exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS teams (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT UNIQUE,
                espn_id       TEXT,
                abbreviation  TEXT,
                conference    TEXT
            );

            CREATE TABLE IF NOT EXISTS games (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT,
                home_team_id  INTEGER,
                away_team_id  INTEGER,
                home_score    INTEGER,
                away_score    INTEGER,
                status        TEXT DEFAULT 'scheduled',
                espn_game_id  TEXT UNIQUE,
                season        TEXT
            );

            CREATE TABLE IF NOT EXISTS team_stats (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id               INTEGER,
                season                TEXT,
                date                  TEXT,
                wins                  INTEGER,
                losses                INTEGER,
                offensive_efficiency  REAL,
                defensive_efficiency  REAL,
                tempo                 REAL,
                sos                   REAL,
                elo                   REAL DEFAULT 1500.0,
                UNIQUE(team_id, date)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id        INTEGER,
                timestamp      TEXT,
                model_version  TEXT,
                home_win_prob  REAL,
                away_win_prob  REAL,
                features_json  TEXT
            );

            CREATE TABLE IF NOT EXISTS markets (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id             INTEGER,
                kalshi_ticker       TEXT,
                kalshi_event_ticker TEXT,
                yes_price           INTEGER,
                no_price            INTEGER,
                volume              INTEGER,
                fetched_at          TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id  INTEGER,
                market_id      INTEGER,
                timestamp      TEXT,
                side           TEXT,
                contracts      INTEGER,
                price          INTEGER,
                edge           REAL,
                kelly_size     REAL,
                order_id       TEXT,
                status         TEXT DEFAULT 'pending',
                pnl            INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                version     TEXT UNIQUE,
                trained_at  TEXT,
                accuracy    REAL,
                brier_score REAL,
                log_loss    REAL,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_state (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                date                TEXT UNIQUE,
                bankroll_cents      INTEGER,
                daily_pnl_cents     INTEGER DEFAULT 0,
                trades_today        INTEGER DEFAULT 0,
                consecutive_losses  INTEGER DEFAULT 0,
                is_cooldown         INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS slates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT,
                status          TEXT DEFAULT 'pending',
                initial_cents   INTEGER,
                current_cents   INTEGER,
                total_legs      INTEGER,
                legs_completed  INTEGER DEFAULT 0,
                created_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS slate_legs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slate_id    INTEGER REFERENCES slates(id),
                trade_id    INTEGER REFERENCES trades(id),
                leg_order   INTEGER,
                bet_cents   INTEGER,
                status      TEXT DEFAULT 'pending',
                UNIQUE(slate_id, leg_order)
            );
        """)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def upsert_team(
    name: str,
    espn_id: Optional[str] = None,
    abbreviation: Optional[str] = None,
    conference: Optional[str] = None,
) -> int:
    """Insert a team or update it if the name already exists, preserving the row id."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO teams (name, espn_id, abbreviation, conference)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   espn_id = COALESCE(excluded.espn_id, teams.espn_id),
                   abbreviation = COALESCE(excluded.abbreviation, teams.abbreviation),
                   conference = COALESCE(excluded.conference, teams.conference)""",
            (name, espn_id, abbreviation, conference),
        )
        row = conn.execute(
            "SELECT id FROM teams WHERE name = ?", (name,)
        ).fetchone()
        return row["id"]


def get_team_by_name(name: str) -> Optional[dict]:
    """Fuzzy-ish lookup using LIKE."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM teams WHERE name LIKE ?", (f"%{name}%",)
        ).fetchone()
        return dict(row) if row else None


def get_team_by_espn_id(espn_id: str) -> Optional[dict]:
    """Look up a team by its ESPN identifier."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM teams WHERE espn_id = ?", (espn_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

def insert_game(
    date: str,
    home_id: int,
    away_id: int,
    espn_game_id: str,
    season: str,
) -> int:
    """Insert a game (ignore if espn_game_id already exists) and return its id."""
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO games
               (date, home_team_id, away_team_id, espn_game_id, season)
               VALUES (?, ?, ?, ?, ?)""",
            (date, home_id, away_id, espn_game_id, season),
        )
        row = conn.execute(
            "SELECT id FROM games WHERE espn_game_id = ?", (espn_game_id,)
        ).fetchone()
        return row["id"]


def update_game_result(
    game_id: int,
    home_score: int,
    away_score: int,
    status: str,
) -> None:
    """Update the final score and status of a game."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE games SET home_score = ?, away_score = ?, status = ?
               WHERE id = ?""",
            (home_score, away_score, status, game_id),
        )


def get_games_by_date(date: str) -> list[dict]:
    """Return all games on a given date with home/away team info joined."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT g.*,
                      ht.name  AS home_team_name,
                      ht.abbreviation AS home_team_abbr,
                      at.name  AS away_team_name,
                      at.abbreviation AS away_team_abbr
               FROM games g
               JOIN teams ht ON g.home_team_id = ht.id
               JOIN teams at ON g.away_team_id = at.id
               WHERE g.date = ?
               ORDER BY g.id""",
            (date,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Team stats
# ---------------------------------------------------------------------------

def save_team_stats(
    team_id: int,
    season: str,
    date: str,
    **stats: object,
) -> None:
    """Insert or replace a team_stats row keyed on (team_id, date)."""
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO team_stats
               (team_id, season, date, wins, losses,
                offensive_efficiency, defensive_efficiency, tempo, sos, elo)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                team_id,
                season,
                date,
                stats.get("wins"),
                stats.get("losses"),
                stats.get("offensive_efficiency"),
                stats.get("defensive_efficiency"),
                stats.get("tempo"),
                stats.get("sos"),
                stats.get("elo", 1500.0),
            ),
        )


def get_team_stats_latest(team_id: int) -> Optional[dict]:
    """Return the most recent stats row for a team."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM team_stats
               WHERE team_id = ?
               ORDER BY date DESC LIMIT 1""",
            (team_id,),
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def save_prediction(
    game_id: int,
    model_version: str,
    home_prob: float,
    away_prob: float,
    features: dict,
) -> int:
    """Save a prediction row with JSON-serialized features and return its id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (game_id, timestamp, model_version, home_win_prob, away_win_prob, features_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                game_id,
                datetime.utcnow().isoformat(),
                model_version,
                home_prob,
                away_prob,
                json.dumps(features),
            ),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------

def save_market(
    game_id: int,
    kalshi_ticker: str,
    event_ticker: str,
    yes_price: int,
    no_price: int,
    volume: int,
) -> int:
    """Save a market snapshot and return its id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO markets
               (game_id, kalshi_ticker, kalshi_event_ticker,
                yes_price, no_price, volume, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id,
                kalshi_ticker,
                event_ticker,
                yes_price,
                no_price,
                volume,
                datetime.utcnow().isoformat(),
            ),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def save_trade(
    prediction_id: int,
    market_id: int,
    side: str,
    contracts: int,
    price: int,
    edge: float,
    kelly: float,
    order_id: str,
    status: str = "pending",
) -> int:
    """Record a new trade and return its id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (prediction_id, market_id, timestamp, side, contracts,
                price, edge, kelly_size, order_id, status, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                prediction_id,
                market_id,
                datetime.utcnow().isoformat(),
                side,
                contracts,
                price,
                edge,
                kelly,
                order_id,
                status,
            ),
        )
        return cur.lastrowid


def update_trade_result(trade_id: int, status: str, pnl: int) -> None:
    """Update a trade's status and realised PnL."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE trades SET status = ?, pnl = ? WHERE id = ?",
            (status, pnl, trade_id),
        )


def get_open_trades() -> list[dict]:
    """Return all trades with status='pending' or 'paper'."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status IN ('pending', 'paper') ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def get_trades_for_period(start_date: str, end_date: str) -> list[dict]:
    """Return trades whose timestamp falls within [start_date, end_date]."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM trades
               WHERE timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp""",
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Daily state / bankroll
# ---------------------------------------------------------------------------

def get_daily_state(date: str) -> dict:
    """Get or create the daily_state row for *date*.

    If the row does not exist yet, it is initialised with the bankroll carried
    forward from the most recent prior row, or the default initial bankroll if
    no prior row exists.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM daily_state WHERE date = ?", (date,)
        ).fetchone()

        if row:
            return dict(row)

        # Carry forward bankroll from the latest prior day
        prev = conn.execute(
            """SELECT bankroll_cents FROM daily_state
               WHERE date < ? ORDER BY date DESC LIMIT 1""",
            (date,),
        ).fetchone()
        bankroll = prev["bankroll_cents"] if prev else _DEFAULT_INITIAL_BANKROLL

        conn.execute(
            """INSERT INTO daily_state (date, bankroll_cents)
               VALUES (?, ?)""",
            (date, bankroll),
        )
        new_row = conn.execute(
            "SELECT * FROM daily_state WHERE date = ?", (date,)
        ).fetchone()
        return dict(new_row)


def update_daily_state(date: str, **kwargs: object) -> None:
    """Update arbitrary fields on the daily_state row for *date*."""
    if not kwargs:
        return

    # Ensure the row exists first
    get_daily_state(date)

    columns = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [date]
    with get_conn() as conn:
        conn.execute(
            f"UPDATE daily_state SET {columns} WHERE date = ?",
            values,
        )


def get_bankroll() -> int:
    """Return the current bankroll in cents from the most recent daily_state row."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT bankroll_cents FROM daily_state ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return row["bankroll_cents"] if row else _DEFAULT_INITIAL_BANKROLL


# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------

def save_model_version(
    version: str,
    accuracy: Optional[float] = None,
    brier: Optional[float] = None,
    log_loss: Optional[float] = None,
    notes: Optional[str] = None,
) -> None:
    """Record or update a model version row."""
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO model_versions
               (version, trained_at, accuracy, brier_score, log_loss, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                version,
                datetime.utcnow().isoformat(),
                accuracy,
                brier,
                log_loss,
                notes,
            ),
        )


# ---------------------------------------------------------------------------
# Slates (progressive rollover parlays)
# ---------------------------------------------------------------------------

def create_slate(date: str, initial_cents: int, total_legs: int) -> int:
    """Create a new slate and return its id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO slates (date, status, initial_cents, current_cents,
                                   total_legs, legs_completed, created_at)
               VALUES (?, 'pending', ?, ?, ?, 0, ?)""",
            (date, initial_cents, initial_cents, total_legs,
             datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def add_slate_leg(
    slate_id: int, trade_id: int, leg_order: int, bet_cents: int,
) -> int:
    """Add a leg to a slate and return its id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO slate_legs (slate_id, trade_id, leg_order, bet_cents, status)
               VALUES (?, ?, ?, ?, 'pending')""",
            (slate_id, trade_id, leg_order, bet_cents),
        )
        return cur.lastrowid


def update_slate_leg(leg_id: int, status: str) -> None:
    """Update a slate leg's status (won, lost, cancelled)."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE slate_legs SET status = ? WHERE id = ?",
            (status, leg_id),
        )


def update_slate(slate_id: int, **kwargs: object) -> None:
    """Update arbitrary fields on a slate row."""
    if not kwargs:
        return
    columns = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [slate_id]
    with get_conn() as conn:
        conn.execute(
            f"UPDATE slates SET {columns} WHERE id = ?", values,
        )


def get_active_slate(date: str) -> Optional[dict]:
    """Return the active or pending slate for a given date, if any."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM slates
               WHERE date = ? AND status IN ('pending', 'active')
               ORDER BY id DESC LIMIT 1""",
            (date,),
        ).fetchone()
        return dict(row) if row else None


def get_slate_for_date(date: str) -> Optional[dict]:
    """Return the most recent slate for a given date (any status)."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM slates WHERE date = ?
               ORDER BY id DESC LIMIT 1""",
            (date,),
        ).fetchone()
        return dict(row) if row else None


def get_slate_legs(slate_id: int) -> list:
    """Return all legs for a slate, ordered by leg_order."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT sl.*, t.side, t.price, t.edge, t.pnl,
                      t.order_id, t.status AS trade_status
               FROM slate_legs sl
               LEFT JOIN trades t ON sl.trade_id = t.id
               WHERE sl.slate_id = ?
               ORDER BY sl.leg_order""",
            (slate_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_slate_history(days: int = 7) -> list:
    """Return slates from the last N days."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM slates
               WHERE date >= date('now', ? || ' days')
               ORDER BY date DESC, id DESC""",
            (f"-{days}",),
        ).fetchall()
        return [dict(r) for r in rows]
