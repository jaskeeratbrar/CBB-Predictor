"""Trading engine with Kelly criterion sizing and risk management.

This is the brain of the CBB Kalshi auto-trading bot. It evaluates model
predictions against market prices, sizes bets using fractional Kelly,
enforces strict risk limits to protect a small bankroll ($10-20), and
handles trade execution and settlement.

All monetary values are stored and manipulated in CENTS (int, not float).
"""

import logging
from datetime import datetime, date
from typing import Optional, List, Dict

log = logging.getLogger(__name__)


class TradingEngine:
    """Evaluates edges, sizes bets, enforces risk limits, and executes trades."""

    def __init__(self, config: dict, kalshi_client=None, db_module=None):
        self.config = config
        self.kalshi = kalshi_client
        self.db = db_module or __import__("db")

        # Paper mode — full pipeline but no real orders
        self.paper_mode: bool = config.get("trading", {}).get("paper_mode", False)

        # Risk parameters — extracted once for readability
        risk = config.get("risk", {})
        self.max_bet_pct: float = risk.get("max_bet_pct", 0.05)
        self.kelly_fraction: float = risk.get("kelly_fraction", 0.10)
        self.daily_loss_limit_pct: float = risk.get("daily_loss_limit_pct", 0.15)
        self.consecutive_loss_cooldown: int = risk.get("consecutive_loss_cooldown", 3)
        self.min_edge_threshold: float = risk.get("min_edge_threshold", 0.03)
        self.max_open_positions: int = risk.get("max_open_positions", 5)
        self.min_bankroll_cents: int = risk.get("min_bankroll_cents", 200)

        mode_str = "PAPER" if self.paper_mode else "LIVE"
        log.info(
            "TradingEngine initialized [%s] — kelly_frac=%.2f, max_bet=%.1f%%, "
            "daily_loss_limit=%.1f%%, min_edge=%.1f%%, min_bankroll=%d cents",
            mode_str,
            self.kelly_fraction,
            self.max_bet_pct * 100,
            self.daily_loss_limit_pct * 100,
            self.min_edge_threshold * 100,
            self.min_bankroll_cents,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _today(self) -> str:
        """Return today's date as ISO string (YYYY-MM-DD)."""
        return date.today().isoformat()

    def _get_bankroll_cents(self) -> int:
        """Return current bankroll in cents from DB."""
        return self.db.get_bankroll()

    # ------------------------------------------------------------------
    # Kelly Criterion
    # ------------------------------------------------------------------

    def calculate_edge(self, p_model: float, market_price_cents: int, side: str) -> float:
        """Calculate edge for a specific side of a binary contract.

        Args:
            p_model: Model's estimated probability that the home team wins
                     (decimal, e.g. 0.65 for 65%).
            market_price_cents: Market price in cents for the YES contract
                                (1-99).
            side: 'yes' or 'no'.

        Returns:
            Edge as a decimal (e.g. 0.07 for 7% edge). Positive means the
            model thinks the market is mis-priced in our favour.
        """
        p_implied = market_price_cents / 100.0

        if side == "yes":
            edge = p_model - p_implied
        elif side == "no":
            # Betting NO means we think the event does NOT happen.
            # Our model prob for "no" = 1 - p_model.
            # Market implied prob for "no" = 1 - p_implied.
            # edge = (1 - p_model) - (1 - p_implied) = p_implied - p_model
            edge = p_implied - p_model
        else:
            log.error("Invalid side '%s' — must be 'yes' or 'no'", side)
            return 0.0

        return edge

    def fractional_kelly(self, edge: float, p_implied: float) -> float:
        """Calculate fractional Kelly bet size as a fraction of bankroll.

        Uses the standard Kelly formula for binary outcomes:
            full_kelly = edge / odds
        where odds = (1 / p_implied) - 1 for a binary contract.

        Then applies the fractional multiplier (default 10%) and caps at
        max_bet_pct (default 5%).

        Args:
            edge: Positive edge as decimal (e.g. 0.07).
            p_implied: Market-implied probability for the side we are
                       betting (decimal, e.g. 0.55).

        Returns:
            Fraction of bankroll to wager (e.g. 0.03 for 3%). Returns 0.0
            if edge is non-positive or inputs are invalid.
        """
        if edge <= 0:
            return 0.0

        # Guard against degenerate prices
        if p_implied <= 0.0 or p_implied >= 1.0:
            log.warning("Degenerate p_implied=%.4f — skipping Kelly calc", p_implied)
            return 0.0

        # Odds against: how much you win per unit risked
        odds = (1.0 / p_implied) - 1.0
        if odds <= 0:
            return 0.0

        # Full Kelly fraction
        full_kelly = edge / odds

        # Fractional Kelly (very conservative)
        frac_kelly = full_kelly * self.kelly_fraction

        # Cap at max bet percentage
        capped = min(frac_kelly, self.max_bet_pct)

        # Ensure non-negative
        return max(capped, 0.0)

    def size_bet(self, edge: float, p_model: float, market_price_cents: int, side: str) -> tuple:
        """Determine the number of contracts and price for a bet.

        Pipeline:
        1. Determine the implied probability for the side we are betting.
        2. Calculate fractional Kelly sizing.
        3. Convert to dollar amount based on current bankroll.
        4. Derive number of contracts from the per-contract price.
        5. Enforce minimum of 1 contract if there is positive edge.
        6. Cap total outlay at max_bet_pct of bankroll.

        Args:
            edge: Pre-computed edge for this side (decimal).
            p_model: Model probability of home win.
            market_price_cents: YES-side price in cents.
            side: 'yes' or 'no'.

        Returns:
            (num_contracts, price_cents) — both ints.
            Returns (0, 0) if the trade should not be placed.
        """
        if edge <= 0:
            return (0, 0)

        # Price the bettor actually pays per contract
        if side == "yes":
            price_cents = market_price_cents
        elif side == "no":
            price_cents = 100 - market_price_cents
        else:
            log.error("Invalid side '%s' in size_bet", side)
            return (0, 0)

        if price_cents <= 0 or price_cents >= 100:
            log.warning("Degenerate price_cents=%d — cannot size bet", price_cents)
            return (0, 0)

        # Implied probability for the side we are betting
        p_implied_for_side = price_cents / 100.0

        # Kelly fraction of bankroll
        kelly_frac = self.fractional_kelly(edge, p_implied_for_side)
        if kelly_frac <= 0:
            return (0, 0)

        bankroll = self._get_bankroll_cents()
        if bankroll <= 0:
            return (0, 0)

        # Dollar amount to risk (in cents)
        amount_cents = int(bankroll * kelly_frac)

        # Number of contracts (each costs price_cents)
        num_contracts = amount_cents // price_cents

        # Minimum 1 contract if we have positive edge and can afford it
        if num_contracts == 0 and price_cents <= bankroll:
            num_contracts = 1

        # Cap total outlay at max_bet_pct of bankroll
        max_outlay = int(bankroll * self.max_bet_pct)
        while num_contracts * price_cents > max_outlay and num_contracts > 1:
            num_contracts -= 1

        # Final sanity check — can we actually afford this?
        if num_contracts * price_cents > bankroll:
            num_contracts = bankroll // price_cents
            if num_contracts <= 0:
                log.info("Cannot afford even 1 contract at %d cents", price_cents)
                return (0, 0)

        log.info(
            "Sized bet: %d contracts @ %d cents (kelly_frac=%.4f, "
            "outlay=%d cents, bankroll=%d cents)",
            num_contracts, price_cents, kelly_frac,
            num_contracts * price_cents, bankroll,
        )
        return (num_contracts, price_cents)

    # ------------------------------------------------------------------
    # Risk Management
    # ------------------------------------------------------------------

    def check_risk_limits(self) -> dict:
        """Check ALL risk limits before placing a trade.

        Returns:
            {
                'can_trade': bool,
                'reasons': list[str]  — human-readable explanations for
                                        each failed check (empty if can_trade).
            }
        """
        reasons: list[str] = []
        today = self._today()

        # 1. Minimum bankroll
        bankroll = self._get_bankroll_cents()
        if bankroll < self.min_bankroll_cents:
            reasons.append(
                f"Bankroll {bankroll} cents is below minimum "
                f"{self.min_bankroll_cents} cents — trading suspended"
            )

        # 2. Daily loss limit
        state = self.db.get_daily_state(today)
        daily_pnl = state.get("daily_pnl_cents", 0)
        # daily_pnl is negative when losing
        loss_limit_cents = int(bankroll * self.daily_loss_limit_pct)
        if daily_pnl < 0 and abs(daily_pnl) >= loss_limit_cents:
            reasons.append(
                f"Daily loss {abs(daily_pnl)} cents has hit limit "
                f"{loss_limit_cents} cents (%.1f%% of bankroll)"
                % (self.daily_loss_limit_pct * 100)
            )

        # 3. Consecutive loss cooldown
        consec_losses = state.get("consecutive_losses", 0)
        is_cooldown = state.get("is_cooldown", 0)
        if is_cooldown:
            reasons.append(
                f"Cooldown active — {consec_losses} consecutive losses "
                f"(threshold: {self.consecutive_loss_cooldown})"
            )
        elif consec_losses >= self.consecutive_loss_cooldown:
            # Trigger cooldown
            reasons.append(
                f"{consec_losses} consecutive losses reached cooldown "
                f"threshold of {self.consecutive_loss_cooldown}"
            )
            self.db.update_daily_state(today, is_cooldown=1)

        # 4. Max open positions
        open_trades = self.db.get_open_trades()
        if len(open_trades) >= self.max_open_positions:
            reasons.append(
                f"{len(open_trades)} open positions — max is "
                f"{self.max_open_positions}"
            )

        can_trade = len(reasons) == 0

        if not can_trade:
            log.warning("Risk limits breached: %s", "; ".join(reasons))
        else:
            log.debug("All risk checks passed")

        return {"can_trade": can_trade, "reasons": reasons}

    def check_edge_threshold(self, edge: float) -> bool:
        """Return True if edge meets the minimum threshold for trading.

        Args:
            edge: Calculated edge as a decimal.

        Returns:
            True if edge >= min_edge_threshold, False otherwise.
        """
        meets = edge >= self.min_edge_threshold
        if not meets:
            log.debug(
                "Edge %.2f%% below threshold %.2f%% — skipping",
                edge * 100, self.min_edge_threshold * 100,
            )
        return meets

    # ------------------------------------------------------------------
    # Trade Evaluation
    # ------------------------------------------------------------------

    def find_best_side(self, p_home_win: float, market: dict) -> Optional[dict]:
        """Determine the most profitable side to bet given model vs market.

        Compares the edge on YES vs NO and returns whichever is larger
        (if any is positive).

        Args:
            p_home_win: Model's probability that the home team wins.
            market: Dict with at least 'yes_price' and 'no_price' (cents).

        Returns:
            Dict with keys {side, edge, price, p_model_for_side} or None
            if neither side has a positive edge.
        """
        yes_price = market.get("yes_price", 0)
        no_price = market.get("no_price", 0)

        # Sanity checks
        if not (1 <= yes_price <= 99) or not (1 <= no_price <= 99):
            log.warning(
                "Invalid market prices yes=%s no=%s — skipping",
                yes_price, no_price,
            )
            return None

        edge_yes = self.calculate_edge(p_home_win, yes_price, "yes")
        edge_no = self.calculate_edge(p_home_win, yes_price, "no")

        log.debug(
            "Edge analysis: YES=%.2f%%, NO=%.2f%% (model=%.2f%%, yes_price=%d)",
            edge_yes * 100, edge_no * 100, p_home_win * 100, yes_price,
        )

        # Pick the side with the largest positive edge
        if edge_yes > 0 and edge_yes >= edge_no:
            return {
                "side": "yes",
                "edge": edge_yes,
                "price": yes_price,
                "p_model_for_side": p_home_win,
            }
        elif edge_no > 0 and edge_no > edge_yes:
            return {
                "side": "no",
                "edge": edge_no,
                "price": no_price,
                "p_model_for_side": 1.0 - p_home_win,
            }

        log.debug("No positive edge on either side — no trade")
        return None

    def evaluate_game(self, game_id: int, p_home_win: float, market: dict) -> dict:
        """Full evaluation pipeline for a single game.

        Steps:
        1. Find the best side (YES/NO) with highest edge.
        2. Check whether edge meets the minimum threshold.
        3. Check all risk limits (bankroll, daily loss, cooldown, positions).
        4. Size the bet using fractional Kelly.

        Args:
            game_id: Internal DB game ID.
            p_home_win: Model probability the home team wins.
            market: Market dict with yes_price, no_price, kalshi_ticker, etc.

        Returns:
            Evaluation dict with keys:
                should_trade, side, contracts, price, edge, kelly_frac,
                reason, game_id, ticker
        """
        ticker = market.get("kalshi_ticker", "UNKNOWN")
        base = {
            "should_trade": False,
            "side": None,
            "contracts": 0,
            "price": 0,
            "edge": 0.0,
            "kelly_frac": 0.0,
            "reason": "",
            "game_id": game_id,
            "ticker": ticker,
        }

        log.info(
            "Evaluating game %d (ticker=%s): model P(home)=%.2f%%, "
            "market YES=%d NO=%d",
            game_id, ticker, p_home_win * 100,
            market.get("yes_price", 0), market.get("no_price", 0),
        )

        # 1. Find the best side
        best = self.find_best_side(p_home_win, market)
        if best is None:
            base["reason"] = "No positive edge on either side"
            log.info("SKIP game %d: %s", game_id, base["reason"])
            return base

        side = best["side"]
        edge = best["edge"]
        price = best["price"]

        # 2. Check edge threshold
        if not self.check_edge_threshold(edge):
            base["edge"] = edge
            base["side"] = side
            base["reason"] = (
                f"Edge {edge:.2%} below threshold {self.min_edge_threshold:.2%}"
            )
            log.info("SKIP game %d: %s", game_id, base["reason"])
            return base

        # 3. Check risk limits
        risk_check = self.check_risk_limits()
        if not risk_check["can_trade"]:
            base["edge"] = edge
            base["side"] = side
            base["reason"] = "Risk limits: " + "; ".join(risk_check["reasons"])
            log.info("SKIP game %d: %s", game_id, base["reason"])
            return base

        # 4. Size the bet
        yes_price = market.get("yes_price", 0)
        num_contracts, price_cents = self.size_bet(edge, p_home_win, yes_price, side)
        if num_contracts <= 0:
            base["edge"] = edge
            base["side"] = side
            base["reason"] = "Bet sizing returned 0 contracts"
            log.info("SKIP game %d: %s", game_id, base["reason"])
            return base

        # Compute the kelly fraction for logging
        p_implied_for_side = price_cents / 100.0
        kelly_frac = self.fractional_kelly(edge, p_implied_for_side)

        result = {
            "should_trade": True,
            "side": side,
            "contracts": num_contracts,
            "price": price_cents,
            "edge": edge,
            "kelly_frac": kelly_frac,
            "reason": (
                f"Edge {edge:.2%} on {side.upper()} @ {price_cents}c x "
                f"{num_contracts} contracts"
            ),
            "game_id": game_id,
            "ticker": ticker,
            "market": market,
            "p_home_win": p_home_win,
        }

        log.info(
            "TRADE game %d: %s %s @ %dc x %d (edge=%.2f%%, kelly=%.4f)",
            game_id, side.upper(), ticker, price_cents, num_contracts,
            edge * 100, kelly_frac,
        )
        return result

    # ------------------------------------------------------------------
    # Trade Execution
    # ------------------------------------------------------------------

    def execute_trade(self, evaluation: dict) -> dict:
        """Place an order on Kalshi and log it to the database.

        Args:
            evaluation: Output from evaluate_game() with should_trade=True.

        Returns:
            {success: bool, order_id: str|None, trade_id: int|None, error: str|None}
        """
        if not evaluation.get("should_trade"):
            return {
                "success": False,
                "order_id": None,
                "trade_id": None,
                "error": "Evaluation says should_trade=False",
            }

        ticker = evaluation["ticker"]
        side = evaluation["side"]
        contracts = evaluation["contracts"]
        price = evaluation["price"]
        edge = evaluation["edge"]
        kelly_frac = evaluation["kelly_frac"]
        game_id = evaluation["game_id"]

        log.info(
            "Executing trade: %s %s @ %dc x %d contracts",
            side.upper(), ticker, price, contracts,
        )

        # Paper mode: skip real order, generate paper order_id
        if self.paper_mode:
            order_id = f"paper_{int(datetime.now().timestamp())}"
            log.info(
                "[PAPER] Would place: %s %s @ %dc x %d (order_id=%s)",
                side.upper(), ticker, price, contracts, order_id,
            )
        else:
            # Place order on Kalshi
            try:
                order_result = self.kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    contracts=contracts,
                    price=price,
                )
            except Exception as exc:
                log.error("Kalshi order failed for %s: %s", ticker, exc)
                return {
                    "success": False,
                    "order_id": None,
                    "trade_id": None,
                    "error": f"Kalshi API error: {exc}",
                }

            order_id = order_result.get("order_id", "")
            if not order_id:
                log.error("No order_id returned from Kalshi for %s", ticker)
                return {
                    "success": False,
                    "order_id": None,
                    "trade_id": None,
                    "error": "No order_id in Kalshi response",
                }

        # Save market snapshot to DB
        market = evaluation.get("market", {})
        market_id = None
        try:
            market_id = self.db.save_market(
                game_id=game_id,
                kalshi_ticker=ticker,
                event_ticker=market.get("kalshi_event_ticker", ""),
                yes_price=market.get("yes_price", 0),
                no_price=market.get("no_price", 0),
                volume=market.get("volume", 0),
            )
        except Exception as exc:
            log.warning("Failed to save market snapshot: %s", exc)

        # Log trade to DB
        trade_status = "paper" if self.paper_mode else "pending"
        try:
            trade_id = self.db.save_trade(
                prediction_id=evaluation.get("prediction_id", 0),
                market_id=market_id or 0,
                side=side,
                contracts=contracts,
                price=price,
                edge=edge,
                kelly=kelly_frac,
                order_id=order_id,
                status=trade_status,
            )
        except Exception as exc:
            log.error("Failed to save trade to DB: %s", exc)
            trade_id = None

        # Update daily state
        today = self._today()
        try:
            state = self.db.get_daily_state(today)
            trades_today = state.get("trades_today", 0) + 1
            self.db.update_daily_state(today, trades_today=trades_today)
        except Exception as exc:
            log.warning("Failed to update daily_state trades_today: %s", exc)

        log.info(
            "Trade executed: order_id=%s, trade_id=%s, %s %s @ %dc x %d",
            order_id, trade_id, side.upper(), ticker, price, contracts,
        )

        return {
            "success": True,
            "order_id": order_id,
            "trade_id": trade_id,
            "error": None,
        }

    def execute_batch(self, evaluations: list[dict]) -> list[dict]:
        """Execute multiple trade evaluations, re-checking risk between each.

        Only evaluations with should_trade=True are attempted. Risk limits
        are re-verified before each execution because earlier trades in the
        batch may have changed the position count or bankroll exposure.

        Args:
            evaluations: List of evaluation dicts from evaluate_game().

        Returns:
            List of execution result dicts (one per attempted trade).
        """
        results: list[dict] = []
        tradeable = [e for e in evaluations if e.get("should_trade")]

        if not tradeable:
            log.info("No tradeable evaluations in batch")
            return results

        log.info("Executing batch of %d trades", len(tradeable))

        for i, evaluation in enumerate(tradeable):
            # Re-check risk limits before each trade
            risk_check = self.check_risk_limits()
            if not risk_check["can_trade"]:
                log.warning(
                    "Risk limits breached before trade %d/%d — stopping batch: %s",
                    i + 1, len(tradeable), "; ".join(risk_check["reasons"]),
                )
                # Mark remaining as skipped
                for remaining in tradeable[i:]:
                    results.append({
                        "success": False,
                        "order_id": None,
                        "trade_id": None,
                        "error": "Risk limits hit during batch: "
                                 + "; ".join(risk_check["reasons"]),
                    })
                break

            result = self.execute_trade(evaluation)
            results.append(result)

            if result["success"]:
                log.info(
                    "Batch trade %d/%d succeeded (order=%s)",
                    i + 1, len(tradeable), result["order_id"],
                )
            else:
                log.warning(
                    "Batch trade %d/%d failed: %s",
                    i + 1, len(tradeable), result.get("error"),
                )

        return results

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_trades(self) -> list[dict]:
        """Check completed games and calculate P&L for open trades.

        For each pending trade:
        1. Look up the associated game to see if it has completed.
        2. Determine the outcome — did our side win?
        3. Calculate P&L:
              win  = +(100 - price_paid) * contracts
              loss = -(price_paid) * contracts
        4. Update trade result in DB.
        5. Update daily_state (pnl, consecutive_losses).
        6. Update bankroll.

        Returns:
            List of {trade_id, result, pnl_cents} for each settled trade.
        """
        open_trades = self.db.get_open_trades()
        if not open_trades:
            log.debug("No open trades to settle")
            return []

        log.info("Checking %d open trades for settlement", len(open_trades))
        settlements: list[dict] = []
        today = self._today()

        for trade in open_trades:
            trade_id = trade["id"]
            market_id = trade.get("market_id")

            # Look up the game via the market row
            game = self._get_game_for_trade(trade)
            if game is None:
                log.debug(
                    "Trade %d: could not find associated game — skipping",
                    trade_id,
                )
                continue

            # Check if game is completed
            status = game.get("status", "")
            if status != "post":
                log.debug(
                    "Trade %d: game status='%s' (not post) — not yet settled",
                    trade_id, status,
                )
                continue

            # Determine outcome
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)
            home_won = home_score > away_score

            side = trade["side"]
            contracts = trade["contracts"]
            price_paid = trade["price"]

            # Did our side win?
            # YES bets win when home team wins; NO bets win when away team wins.
            if side == "yes":
                we_won = home_won
            else:
                we_won = not home_won

            # Calculate P&L
            if we_won:
                pnl_cents = (100 - price_paid) * contracts
                result_status = "won"
            else:
                pnl_cents = -(price_paid * contracts)
                result_status = "lost"

            log.info(
                "Trade %d settled: %s — %s @ %dc x %d = %+d cents",
                trade_id, result_status.upper(), side, price_paid,
                contracts, pnl_cents,
            )

            # Update trade in DB
            try:
                self.db.update_trade_result(trade_id, result_status, pnl_cents)
            except Exception as exc:
                log.error("Failed to update trade %d result: %s", trade_id, exc)
                continue

            # Update bankroll
            new_bankroll = self.update_bankroll_after_settlement(pnl_cents)

            # Update daily state
            try:
                state = self.db.get_daily_state(today)
                new_daily_pnl = state.get("daily_pnl_cents", 0) + pnl_cents

                if result_status == "lost":
                    new_consec = state.get("consecutive_losses", 0) + 1
                    is_cooldown = 1 if new_consec >= self.consecutive_loss_cooldown else 0
                    if is_cooldown:
                        log.warning(
                            "Cooldown triggered after %d consecutive losses",
                            new_consec,
                        )
                else:
                    # Win resets the consecutive loss counter
                    new_consec = 0
                    is_cooldown = 0

                self.db.update_daily_state(
                    today,
                    daily_pnl_cents=new_daily_pnl,
                    consecutive_losses=new_consec,
                    is_cooldown=is_cooldown,
                    bankroll_cents=new_bankroll,
                )
            except Exception as exc:
                log.warning("Failed to update daily_state after settlement: %s", exc)

            settlements.append({
                "trade_id": trade_id,
                "result": result_status,
                "pnl_cents": pnl_cents,
            })

        if settlements:
            total_pnl = sum(s["pnl_cents"] for s in settlements)
            wins = sum(1 for s in settlements if s["result"] == "won")
            losses = sum(1 for s in settlements if s["result"] == "lost")
            log.info(
                "Settlement complete: %d trades (%dW/%dL), total P&L = %+d cents",
                len(settlements), wins, losses, total_pnl,
            )

        return settlements

    def _get_game_for_trade(self, trade: dict) -> Optional[dict]:
        """Look up the game associated with a trade via its market row.

        Follows the chain: trade -> market_id -> markets.game_id -> games row.

        Returns:
            Game dict or None if not found.
        """
        market_id = trade.get("market_id")
        if not market_id:
            return None

        try:
            from db import get_conn
            with get_conn() as conn:
                # Get game_id from market
                market_row = conn.execute(
                    "SELECT game_id FROM markets WHERE id = ?", (market_id,)
                ).fetchone()
                if not market_row:
                    return None

                game_id = market_row["game_id"]
                game_row = conn.execute(
                    "SELECT * FROM games WHERE id = ?", (game_id,)
                ).fetchone()
                return dict(game_row) if game_row else None
        except Exception as exc:
            log.error("Error looking up game for trade %d: %s", trade["id"], exc)
            return None

    def update_bankroll_after_settlement(self, pnl_cents: int) -> int:
        """Update bankroll in DB after settlement.

        Args:
            pnl_cents: Profit/loss in cents (positive for wins, negative
                       for losses).

        Returns:
            New bankroll in cents.
        """
        current = self._get_bankroll_cents()
        new_bankroll = current + pnl_cents

        # Floor at zero — can't have negative bankroll
        if new_bankroll < 0:
            log.error(
                "Bankroll would go negative (%d cents) — flooring at 0",
                new_bankroll,
            )
            new_bankroll = 0

        today = self._today()
        try:
            self.db.update_daily_state(today, bankroll_cents=new_bankroll)
        except Exception as exc:
            log.error("Failed to persist new bankroll %d cents: %s", new_bankroll, exc)

        if new_bankroll < self.min_bankroll_cents:
            log.warning(
                "BANKROLL CRITICAL: %d cents (below minimum %d) — "
                "trading will be suspended",
                new_bankroll, self.min_bankroll_cents,
            )

        log.info(
            "Bankroll updated: %d -> %d cents (P&L: %+d)",
            current, new_bankroll, pnl_cents,
        )
        return new_bankroll
