"""Performance tracking and markdown report generation for the CBB Kalshi bot.

Generates daily performance logs (logs/YYYY-MM-DD.md) and weekly summary
reports (reports/week-YYYY-WW.md).  Provides summary statistics and model
calibration analysis.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import db

log = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
REPORTS_DIR = Path("reports")


class ReviewSystem:
    """Generate performance reports and compute summary statistics."""

    def __init__(self):
        LOGS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def format_cents(self, cents: int) -> str:
        """Format cents as a dollar string.

        Examples:
            1500  -> '$15.00'
            -75   -> '-$0.75'
            0     -> '$0.00'
        """
        if cents < 0:
            return f"-${abs(cents) / 100:.2f}"
        return f"${cents / 100:.2f}"

    def _pnl_str(self, cents: int) -> str:
        """Format P&L with explicit +/- sign."""
        if cents > 0:
            return f"+{self.format_cents(cents)}"
        return self.format_cents(cents)

    def _pct_str(self, value: float) -> str:
        """Format a ratio as a percentage string with one decimal."""
        return f"{value * 100:.1f}%"

    @staticmethod
    def _date_range_strings(d: date) -> tuple[str, str]:
        """Return ISO start/end timestamp strings covering a full calendar day."""
        start = datetime(d.year, d.month, d.day, 0, 0, 0).isoformat()
        end = datetime(d.year, d.month, d.day, 23, 59, 59).isoformat()
        return start, end

    def _get_model_version(self) -> str:
        """Return the current model version from the DB, or 'unknown'."""
        try:
            from db import get_conn
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT version FROM model_versions ORDER BY trained_at DESC LIMIT 1"
                ).fetchone()
                return row["version"] if row else "unknown"
        except Exception:
            return "unknown"

    def _trade_game_label(self, trade: dict) -> str:
        """Build a short game label for a trade row.

        Attempts to join through prediction -> game -> teams.  Falls back to
        the trade's market_id if the join fails.
        """
        try:
            from db import get_conn
            with get_conn() as conn:
                row = conn.execute(
                    """SELECT ht.abbreviation AS home, at.abbreviation AS away
                       FROM trades t
                       JOIN predictions p ON t.prediction_id = p.id
                       JOIN games g ON p.game_id = g.id
                       JOIN teams ht ON g.home_team_id = ht.id
                       JOIN teams at ON g.away_team_id = at.id
                       WHERE t.id = ?""",
                    (trade["id"],),
                ).fetchone()
                if row:
                    return f"{row['away']}@{row['home']}"
        except Exception:
            pass
        return f"market:{trade.get('market_id', '?')}"

    def _get_prediction_prob(self, trade: dict) -> Optional[float]:
        """Return the model probability associated with a trade's prediction."""
        try:
            from db import get_conn
            with get_conn() as conn:
                row = conn.execute(
                    """SELECT p.home_win_prob, p.away_win_prob, t.side
                       FROM trades t
                       JOIN predictions p ON t.prediction_id = p.id
                       WHERE t.id = ?""",
                    (trade["id"],),
                ).fetchone()
                if row:
                    side = trade.get("side", "yes")
                    # Convention: 'yes' side maps to home_win_prob
                    if side == "yes":
                        return row["home_win_prob"]
                    return row["away_win_prob"]
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Daily log
    # ------------------------------------------------------------------

    def generate_daily_log(self, log_date: date = None) -> str:
        """Generate a daily performance markdown file.

        Args:
            log_date: Date to generate the log for.  Defaults to today.

        Returns:
            Path string of the generated file.
        """
        log_date = log_date or date.today()
        date_str = log_date.isoformat()

        start_ts, end_ts = self._date_range_strings(log_date)
        trades = db.get_trades_for_period(start_ts, end_ts)

        state = db.get_daily_state(date_str)
        bankroll_end = state["bankroll_cents"]
        daily_pnl = state["daily_pnl_cents"]
        bankroll_start = bankroll_end - daily_pnl

        model_version = self._get_model_version()

        # Partition by result
        completed = [t for t in trades if t["status"] in ("won", "lost")]
        wins = [t for t in completed if t["status"] == "won"]
        losses = [t for t in completed if t["status"] == "lost"]

        win_count = len(wins)
        loss_count = len(losses)

        lines: list[str] = []
        lines.append(f"# Daily Log — {date_str}")
        lines.append("")

        # --- Summary ---
        lines.append("## Summary")
        lines.append(f"- **Model Version:** {model_version}")
        lines.append(f"- **Bankroll Start:** {self.format_cents(bankroll_start)}")
        lines.append(f"- **Bankroll End:** {self.format_cents(bankroll_end)}")
        pnl_pct = (daily_pnl / bankroll_start * 100) if bankroll_start else 0.0
        lines.append(f"- **Daily P&L:** {self._pnl_str(daily_pnl)} ({pnl_pct:+.1f}%)")
        lines.append(f"- **Record:** {win_count}-{loss_count}")
        if state.get("is_cooldown"):
            lines.append(f"- **Cooldown active** (consecutive losses: {state['consecutive_losses']})")
        lines.append("")

        # --- Trades table ---
        lines.append("## Trades")
        if not trades:
            lines.append("_No trades placed today._")
        else:
            lines.append("| # | Game | Side | Contracts | Price | Edge | Result | P&L |")
            lines.append("|---|------|------|-----------|-------|------|--------|-----|")
            for i, t in enumerate(trades, 1):
                game = self._trade_game_label(t)
                side = t.get("side", "—")
                contracts = t.get("contracts", 0)
                price = t.get("price", 0)
                edge = t.get("edge", 0.0)
                status = t.get("status", "pending")
                pnl = t.get("pnl", 0)
                lines.append(
                    f"| {i} | {game} | {side} | {contracts} | "
                    f"{self.format_cents(price)} | {edge:.1%} | {status} | "
                    f"{self._pnl_str(pnl)} |"
                )
        lines.append("")

        # --- Notes ---
        lines.append("## Notes")
        if completed:
            biggest_win_pnl = max((t["pnl"] for t in completed), default=0)
            biggest_loss_pnl = min((t["pnl"] for t in completed), default=0)
            lines.append(f"- **Biggest win:** {self._pnl_str(biggest_win_pnl)}")
            lines.append(f"- **Biggest loss:** {self._pnl_str(biggest_loss_pnl)}")

            avg_edge_winners = (
                sum(t["edge"] for t in wins) / len(wins) if wins else 0.0
            )
            avg_edge_losers = (
                sum(t["edge"] for t in losses) / len(losses) if losses else 0.0
            )
            lines.append(f"- **Avg edge on winners:** {avg_edge_winners:.1%}")
            lines.append(f"- **Avg edge on losers:** {avg_edge_losers:.1%}")
        else:
            lines.append("_No completed trades to analyse._")

        # Risk events
        risk_events: list[str] = []
        if state.get("is_cooldown"):
            risk_events.append(
                f"Cooldown triggered after {state['consecutive_losses']} consecutive losses"
            )
        if daily_pnl < 0 and bankroll_start > 0:
            loss_pct = abs(daily_pnl) / bankroll_start
            if loss_pct >= 0.15:
                risk_events.append(f"Daily loss limit reached ({loss_pct:.1%} of bankroll)")
        if risk_events:
            lines.append("")
            lines.append("### Risk Events")
            for evt in risk_events:
                lines.append(f"- {evt}")

        lines.append("")

        # Write file
        out_path = LOGS_DIR / f"{date_str}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Daily log written to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Weekly report
    # ------------------------------------------------------------------

    def generate_weekly_report(self, week_end: date = None) -> str:
        """Generate a weekly summary markdown report.

        Args:
            week_end: Last day (Sunday) of the reporting week.  Defaults to the
                      most recent Sunday (or today if today is Sunday).

        Returns:
            Path string of the generated file.
        """
        if week_end is None:
            today = date.today()
            # Roll back to the most recent Sunday
            days_since_sunday = (today.weekday() + 1) % 7
            week_end = today - timedelta(days=days_since_sunday)
            if week_end > today:
                week_end = today

        week_start = week_end - timedelta(days=6)

        start_ts = datetime(week_start.year, week_start.month, week_start.day, 0, 0, 0).isoformat()
        end_ts = datetime(week_end.year, week_end.month, week_end.day, 23, 59, 59).isoformat()

        trades = db.get_trades_for_period(start_ts, end_ts)
        completed = [t for t in trades if t["status"] in ("won", "lost")]
        wins = [t for t in completed if t["status"] == "won"]
        losses_list = [t for t in completed if t["status"] == "lost"]

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses_list)
        win_rate = win_count / len(completed) if completed else 0.0
        total_pnl = sum(t.get("pnl", 0) for t in completed)

        # Bankroll trajectory
        daily_values: list[tuple[str, int]] = []
        d = week_start
        while d <= week_end:
            state = db.get_daily_state(d.isoformat())
            daily_values.append((d.isoformat(), state["bankroll_cents"]))
            d += timedelta(days=1)

        bankroll_start = daily_values[0][1] if daily_values else 0
        bankroll_end = daily_values[-1][1] if daily_values else 0
        roi = total_pnl / bankroll_start if bankroll_start else 0.0

        iso_year, iso_week, _ = week_end.isocalendar()
        filename = f"week-{iso_year}-{iso_week:02d}.md"

        lines: list[str] = []
        lines.append(f"# Weekly Report — Week {iso_week:02d}, {iso_year}")
        lines.append(f"_{week_start.isoformat()} to {week_end.isoformat()}_")
        lines.append("")

        # --- Performance ---
        lines.append("## Performance")
        lines.append(f"- **Total trades:** {total_trades}")
        lines.append(f"- **Record:** {win_count}-{loss_count} ({self._pct_str(win_rate)})")
        lines.append(f"- **Total P&L:** {self._pnl_str(total_pnl)}")
        lines.append(f"- **ROI:** {self._pct_str(roi)}")
        lines.append(
            f"- **Bankroll:** {self.format_cents(bankroll_start)} "
            f"-> {self.format_cents(bankroll_end)}"
        )
        lines.append("")

        # --- Edge Analysis ---
        lines.append("## Edge Analysis")
        if completed:
            avg_edge_all = sum(t["edge"] for t in completed) / len(completed)
            avg_edge_win = sum(t["edge"] for t in wins) / len(wins) if wins else 0.0
            avg_edge_loss = (
                sum(t["edge"] for t in losses_list) / len(losses_list) if losses_list else 0.0
            )
            lines.append(f"- **Avg edge on placed trades:** {avg_edge_all:.1%}")
            lines.append(f"- **Avg edge on winners:** {avg_edge_win:.1%}")
            lines.append(f"- **Avg edge on losers:** {avg_edge_loss:.1%}")

            # Edge accuracy: do higher-edge bets win more often?
            if len(completed) >= 4:
                sorted_by_edge = sorted(completed, key=lambda t: t["edge"])
                mid = len(sorted_by_edge) // 2
                low_half = sorted_by_edge[:mid]
                high_half = sorted_by_edge[mid:]
                low_wr = sum(1 for t in low_half if t["status"] == "won") / len(low_half)
                high_wr = sum(1 for t in high_half if t["status"] == "won") / len(high_half)
                lines.append(
                    f"- **Edge accuracy:** lower-edge WR {self._pct_str(low_wr)} vs "
                    f"higher-edge WR {self._pct_str(high_wr)} "
                    f"({'aligned' if high_wr >= low_wr else 'INVERTED'})"
                )
            else:
                lines.append("- **Edge accuracy:** insufficient data (need 4+ completed trades)")
        else:
            lines.append("_No completed trades this week._")
        lines.append("")

        # --- Model Calibration ---
        lines.append("## Model Calibration")
        cal = self.get_model_calibration(days=7, ref_date=week_end)
        if cal["calibration_error"] is not None:
            lines.append(f"- **Calibration error (MAE):** {cal['calibration_error']:.4f}")
            lines.append("")
            lines.append("| Confidence Bin | Predicted | Actual | Samples |")
            lines.append("|----------------|-----------|--------|---------|")
            for i, (lo, hi) in enumerate(cal["bins"]):
                pred = cal["predicted"][i]
                act = cal["actual"][i]
                n = cal["n_samples"][i]
                pred_s = f"{pred:.1%}" if pred is not None else "—"
                act_s = f"{act:.1%}" if act is not None else "—"
                lines.append(f"| {lo:.0%}–{hi:.0%} | {pred_s} | {act_s} | {n} |")
        else:
            lines.append("_Insufficient prediction data for calibration._")
        lines.append("")

        # --- Risk Events ---
        lines.append("## Risk Events")
        cooldowns_triggered = 0
        daily_limits_hit = 0
        max_drawdown = 0
        running_pnl = 0
        peak_pnl = 0

        d = week_start
        while d <= week_end:
            state = db.get_daily_state(d.isoformat())
            if state.get("is_cooldown"):
                cooldowns_triggered += 1
            dpnl = state.get("daily_pnl_cents", 0)
            if dpnl < 0:
                # Check if daily loss limit was hit (>= 15% of starting bankroll)
                day_start_bankroll = state["bankroll_cents"] - dpnl
                if day_start_bankroll > 0 and abs(dpnl) / day_start_bankroll >= 0.15:
                    daily_limits_hit += 1
            running_pnl += dpnl
            peak_pnl = max(peak_pnl, running_pnl)
            drawdown = peak_pnl - running_pnl
            max_drawdown = max(max_drawdown, drawdown)
            d += timedelta(days=1)

        lines.append(f"- **Daily limits hit:** {daily_limits_hit} time(s)")
        lines.append(f"- **Cooldowns triggered:** {cooldowns_triggered} time(s)")
        lines.append(f"- **Max drawdown:** {self.format_cents(max_drawdown)}")
        lines.append("")

        # --- Bankroll Trajectory ---
        lines.append("## Bankroll Trajectory")
        lines.append("")
        lines.append("```")
        lines.append(self.ascii_bankroll_chart(daily_values))
        lines.append("```")
        lines.append("")

        # --- Recommendations ---
        lines.append("## Recommendations")
        recs = self._generate_recommendations(
            completed, wins, losses_list, win_rate, total_pnl,
            bankroll_start, cal, cooldowns_triggered, daily_limits_hit,
        )
        if recs:
            for r in recs:
                lines.append(f"- {r}")
        else:
            lines.append("_No actionable recommendations this week. Keep it up._")
        lines.append("")

        out_path = REPORTS_DIR / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Weekly report written to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_performance_summary(self, days: int = 7) -> dict:
        """Calculate summary stats for the last N days.

        Returns:
            Dict with keys: total_trades, wins, losses, win_rate,
            total_pnl_cents, roi, avg_edge, max_drawdown_cents,
            current_bankroll_cents.
        """
        end = date.today()
        start = end - timedelta(days=days - 1)

        start_ts = datetime(start.year, start.month, start.day, 0, 0, 0).isoformat()
        end_ts = datetime(end.year, end.month, end.day, 23, 59, 59).isoformat()

        trades = db.get_trades_for_period(start_ts, end_ts)
        completed = [t for t in trades if t["status"] in ("won", "lost")]

        wins = sum(1 for t in completed if t["status"] == "won")
        losses = sum(1 for t in completed if t["status"] == "lost")
        total_pnl = sum(t.get("pnl", 0) for t in completed)

        # Starting bankroll from first day in range
        start_state = db.get_daily_state(start.isoformat())
        starting_bankroll = start_state["bankroll_cents"]

        # Max drawdown: track peak-to-trough in cumulative P&L
        max_drawdown = 0
        running_pnl = 0
        peak_pnl = 0
        for t in completed:
            running_pnl += t.get("pnl", 0)
            peak_pnl = max(peak_pnl, running_pnl)
            drawdown = peak_pnl - running_pnl
            max_drawdown = max(max_drawdown, drawdown)

        avg_edge = (
            sum(t["edge"] for t in completed) / len(completed) if completed else 0.0
        )

        current_bankroll = db.get_bankroll()

        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(completed) if completed else 0.0,
            "total_pnl_cents": total_pnl,
            "roi": total_pnl / starting_bankroll if starting_bankroll else 0.0,
            "avg_edge": avg_edge,
            "max_drawdown_cents": max_drawdown,
            "current_bankroll_cents": current_bankroll,
        }

    # ------------------------------------------------------------------
    # Model calibration
    # ------------------------------------------------------------------

    def get_model_calibration(self, days: int = 30, ref_date: date = None) -> dict:
        """Check model probability calibration over the given window.

        Bins predictions into 5 buckets (0-20%, 20-40%, ..., 80-100%) and
        compares predicted win rate vs actual win rate per bin.

        Args:
            days: Number of days to look back.
            ref_date: End date of the window.  Defaults to today.

        Returns:
            Dict with keys: bins, predicted, actual, n_samples,
            calibration_error (mean absolute difference, or None if no data).
        """
        ref_date = ref_date or date.today()
        start = ref_date - timedelta(days=days - 1)

        start_ts = datetime(start.year, start.month, start.day, 0, 0, 0).isoformat()
        end_ts = datetime(ref_date.year, ref_date.month, ref_date.day, 23, 59, 59).isoformat()

        trades = db.get_trades_for_period(start_ts, end_ts)
        completed = [t for t in trades if t["status"] in ("won", "lost")]

        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        predicted: list[Optional[float]] = [None] * 5
        actual: list[Optional[float]] = [None] * 5
        n_samples: list[int] = [0] * 5

        # Collect (predicted_prob, outcome) pairs
        pairs: list[tuple[float, int]] = []
        for t in completed:
            prob = self._get_prediction_prob(t)
            if prob is not None:
                outcome = 1 if t["status"] == "won" else 0
                pairs.append((prob, outcome))

        if not pairs:
            return {
                "bins": bins,
                "predicted": predicted,
                "actual": actual,
                "n_samples": n_samples,
                "calibration_error": None,
            }

        # Assign to bins
        bin_preds: list[list[float]] = [[] for _ in range(5)]
        bin_outcomes: list[list[int]] = [[] for _ in range(5)]

        for prob, outcome in pairs:
            idx = min(int(prob / 0.2), 4)  # clamp 1.0 into the last bin
            bin_preds[idx].append(prob)
            bin_outcomes[idx].append(outcome)

        abs_errors: list[float] = []
        for i in range(5):
            n_samples[i] = len(bin_preds[i])
            if n_samples[i] > 0:
                predicted[i] = sum(bin_preds[i]) / n_samples[i]
                actual[i] = sum(bin_outcomes[i]) / n_samples[i]
                abs_errors.append(abs(predicted[i] - actual[i]))

        calibration_error = sum(abs_errors) / len(abs_errors) if abs_errors else None

        return {
            "bins": bins,
            "predicted": predicted,
            "actual": actual,
            "n_samples": n_samples,
            "calibration_error": calibration_error,
        }

    # ------------------------------------------------------------------
    # ASCII chart
    # ------------------------------------------------------------------

    def ascii_bankroll_chart(self, daily_values: list[tuple]) -> str:
        """Generate a simple ASCII bar chart of bankroll over time.

        Args:
            daily_values: List of (date_str, bankroll_cents) tuples.

        Returns:
            Multi-line string suitable for embedding in a markdown code block.
        """
        if not daily_values:
            return "(no data)"

        values = [v for _, v in daily_values]
        min_val = min(values)
        max_val = max(values)
        chart_width = 40

        lines: list[str] = []
        for date_str, cents in daily_values:
            # Short date label: MM-DD
            label = date_str[5:]  # "YYYY-MM-DD" -> "MM-DD"
            dollars = self.format_cents(cents)

            if max_val == min_val:
                bar_len = chart_width
            else:
                bar_len = max(1, int((cents - min_val) / (max_val - min_val) * chart_width))

            bar = "|" * bar_len
            lines.append(f"{label} {bar} {dollars}")

        # Scale legend
        lines.append("")
        lines.append(f"  min: {self.format_cents(min_val)}  max: {self.format_cents(max_val)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Automated recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        completed: list[dict],
        wins: list[dict],
        losses_list: list[dict],
        win_rate: float,
        total_pnl: int,
        bankroll_start: int,
        calibration: dict,
        cooldowns: int,
        limits_hit: int,
    ) -> list[str]:
        """Produce auto-generated suggestions based on weekly patterns."""
        recs: list[str] = []

        if not completed:
            recs.append(
                "No completed trades this week. Verify that the pipeline is "
                "running and markets are being matched."
            )
            return recs

        # 1. Win rate concerns
        if len(completed) >= 5 and win_rate < 0.40:
            recs.append(
                f"Win rate is low at {self._pct_str(win_rate)}. Consider raising "
                f"the minimum edge threshold to be more selective."
            )

        # 2. Edge analysis — losing on low-edge bets
        low_edge_trades = [t for t in completed if t["edge"] < 0.05]
        if low_edge_trades:
            low_edge_wr = (
                sum(1 for t in low_edge_trades if t["status"] == "won") / len(low_edge_trades)
            )
            low_edge_pnl = sum(t.get("pnl", 0) for t in low_edge_trades)
            if low_edge_pnl < 0:
                recs.append(
                    f"Trades with <5% edge went {self._pct_str(low_edge_wr)} "
                    f"win rate and net {self._pnl_str(low_edge_pnl)}. "
                    f"Raising the min_edge_threshold above 5% may improve profitability."
                )

        # 3. Edge inversion — higher edge bets losing more
        if len(completed) >= 6:
            sorted_by_edge = sorted(completed, key=lambda t: t["edge"])
            mid = len(sorted_by_edge) // 2
            low_half = sorted_by_edge[:mid]
            high_half = sorted_by_edge[mid:]
            low_wr = sum(1 for t in low_half if t["status"] == "won") / len(low_half)
            high_wr = sum(1 for t in high_half if t["status"] == "won") / len(high_half)
            if high_wr < low_wr - 0.10:
                recs.append(
                    f"Edge is inverted: higher-edge trades win at {self._pct_str(high_wr)} "
                    f"vs {self._pct_str(low_wr)} for lower-edge trades. The model's edge "
                    f"estimates may be miscalibrated — investigate feature drift or "
                    f"market-maker pricing changes."
                )

        # 4. Calibration drift
        if calibration.get("calibration_error") is not None:
            cal_err = calibration["calibration_error"]
            if cal_err > 0.15:
                recs.append(
                    f"Model calibration error is {cal_err:.2%}, which is high. "
                    f"Consider retraining or re-calibrating sooner than the weekly schedule."
                )
            # Check individual bins for systematic bias
            for i, (lo, hi) in enumerate(calibration["bins"]):
                pred = calibration["predicted"][i]
                act = calibration["actual"][i]
                n = calibration["n_samples"][i]
                if pred is not None and act is not None and n >= 3:
                    diff = act - pred
                    if diff > 0.15:
                        recs.append(
                            f"Model underpredicts in the {lo:.0%}-{hi:.0%} confidence bin "
                            f"(predicted {pred:.0%}, actual {act:.0%}, n={n}). "
                            f"These may be undervalued opportunities."
                        )
                    elif diff < -0.15:
                        recs.append(
                            f"Model overpredicts in the {lo:.0%}-{hi:.0%} confidence bin "
                            f"(predicted {pred:.0%}, actual {act:.0%}, n={n}). "
                            f"Consider reducing position sizes in this range."
                        )

        # 5. Risk management
        if cooldowns >= 3:
            recs.append(
                f"Cooldown was triggered {cooldowns} times this week. The "
                f"consecutive loss limit may be too tight, or the model is "
                f"mis-timing entries."
            )
        if limits_hit >= 2:
            recs.append(
                f"Daily loss limit was hit {limits_hit} times. Consider reducing "
                f"position sizes or tightening the Kelly fraction."
            )

        # 6. Overtrading
        avg_trades_per_day = len(completed) / 7
        if avg_trades_per_day > 8:
            recs.append(
                f"Averaging {avg_trades_per_day:.1f} trades/day — this is high. "
                f"Quality over quantity: consider raising edge thresholds to be "
                f"more selective."
            )

        # 7. Positive reinforcement
        if win_rate >= 0.55 and total_pnl > 0:
            recs.append(
                f"Solid week: {self._pct_str(win_rate)} win rate with "
                f"{self._pnl_str(total_pnl)} profit. Current strategy parameters "
                f"are performing well."
            )

        # 8. Sizing — check if winners are smaller than losers
        if wins and losses_list:
            avg_win_size = sum(t.get("pnl", 0) for t in wins) / len(wins)
            avg_loss_size = abs(sum(t.get("pnl", 0) for t in losses_list) / len(losses_list))
            if avg_loss_size > 0 and avg_win_size / avg_loss_size < 0.8:
                recs.append(
                    f"Average win ({self._pnl_str(int(avg_win_size))}) is smaller "
                    f"than average loss ({self.format_cents(int(avg_loss_size))}). "
                    f"The Kelly sizing may need adjustment, or the model is more "
                    f"confident on trades that end up losing."
                )

        return recs
