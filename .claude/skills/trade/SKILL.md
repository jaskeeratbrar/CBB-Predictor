---
name: trade
description: Execute trades on Kalshi based on predictions. ALWAYS runs dry-run first and asks for confirmation before placing real money bets.
user_invocable: true
---

# /trade Skill

Execute the prediction-to-trade pipeline with safety checks.

## Steps

1. Check bot status first:
   ```bash
   python3 main.py status
   ```
   Report the current bankroll, open positions, and any risk warnings.

2. Run predictions if not already done:
   ```bash
   python3 main.py predict
   ```

3. **ALWAYS** run dry-run first:
   ```bash
   python3 main.py trade --dry-run
   ```
   Present the proposed trades clearly:
   - Market ticker
   - Side (yes/no)
   - Number of contracts
   - Price per contract
   - Edge percentage
   - Kelly fraction
   - Total risk in dollars

4. **ASK THE USER FOR CONFIRMATION** before proceeding. Show:
   - Total capital at risk
   - Current bankroll
   - Risk as percentage of bankroll

5. Only after explicit confirmation, execute:
   ```bash
   python3 main.py trade
   ```

6. Show results:
   ```bash
   python3 main.py status
   ```
   Report new bankroll and open positions.

## SAFETY RULES
- NEVER skip the dry-run step
- NEVER execute without user confirmation
- If risk limits are triggered, explain why and DO NOT override
- If bankroll is below $2, REFUSE to trade and warn the user
