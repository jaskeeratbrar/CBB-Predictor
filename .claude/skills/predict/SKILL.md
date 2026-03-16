---
name: predict
description: Run CBB game predictions for today or a specific date. Shows win probabilities, spreads, and edge analysis for all games.
user_invocable: true
---

# /predict Skill

Run the CBB prediction pipeline and display results.

## Steps

1. First, ensure the database is initialized:
   ```bash
   python3 main.py init
   ```
   (Skip if already initialized — check if cbb.db exists)

2. Run predictions:
   ```bash
   python3 main.py predict $ARGUMENTS
   ```
   If the user provided a date, pass it as `--date YYYY-MM-DD`.

3. Read the output and present it clearly. If there are predictions with edge > 3%, highlight them as potential trades.

4. Show a summary:
   - Total games today
   - Games with positive edge (potential trades)
   - Model version being used
   - Any warnings (cooldown active, low bankroll, etc.)

5. If the user asks about a specific game, filter the output accordingly.

## Notes
- The prediction output shows: matchup, home win %, away win %, spread, status
- Predictions are saved to the database automatically
- If no model is trained, Elo-based predictions are used (less accurate but functional)
