---
name: review
description: Generate and display performance reviews. Shows P&L, win rate, model accuracy, risk events, and recommendations.
user_invocable: true
---

# /review Skill

Generate performance reports and analyze bot performance.

## Steps

1. Determine period — check if user asked for "weekly" or a specific date, default to daily:
   ```bash
   python3 main.py review --period daily
   ```
   or
   ```bash
   python3 main.py review --period weekly
   ```

2. Read the generated report file:
   - Daily: `logs/YYYY-MM-DD.md`
   - Weekly: `reports/week-YYYY-WW.md`

3. Present key metrics to the user:
   - Total trades, win/loss record, win rate
   - P&L in dollars and as ROI percentage
   - Bankroll trajectory
   - Model accuracy and calibration
   - Risk events (cooldowns, limits hit)

4. If performance is declining, analyze patterns:
   - Is edge accuracy dropping? (predicting edge but losing)
   - Are losses concentrated in specific game types?
   - Is the model overconfident or underconfident?

5. Provide actionable recommendations based on the data.

## Notes
- Reports are auto-generated from the SQLite database
- If no trades exist for the period, note that and suggest checking predictions
- Compare current week to previous weeks if data exists
