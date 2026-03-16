---
name: backtest
description: Backtest the prediction model on historical data to validate accuracy and simulated ROI.
user_invocable: true
---

# /backtest Skill

Run backtests to validate model performance on historical data.

## Steps

1. Parse date range from user arguments. Default to current season if none given:
   ```bash
   python3 main.py backtest --start $START_DATE --end $END_DATE
   ```

2. Present the results:
   - Games tested
   - Prediction accuracy
   - Simulated P&L
   - Accuracy by confidence tier (high confidence vs low confidence)

3. Compare to baseline:
   - Random (50% accuracy)
   - Always pick home team (~60% CBB home win rate)
   - Elo-only predictions

4. If accuracy is below 55%, warn that the model may not be ready for real trading.

## Notes
- Backtesting requires completed games in the database
- If insufficient data, suggest running `python3 main.py init` to populate historical games
- Simulated P&L assumes 55-cent average market price (simplified)
