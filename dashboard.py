"""Simple Flask dashboard for the CBB Kalshi trading bot.

Run: python3 dashboard.py
Open: http://localhost:5050
"""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import Flask, render_template_string

app = Flask(__name__)
DB_PATH = Path(__file__).resolve().parent / "cbb.db"

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CBB Predictor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { color: #58a6ff; margin-bottom: 4px; font-size: 1.8em; }
  .subtitle { color: #8b949e; margin-bottom: 24px; font-size: 0.9em; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card .label { color: #8b949e; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 1.6em; font-weight: 700; margin-top: 4px; }
  .green { color: #3fb950; }
  .red { color: #f85149; }
  .yellow { color: #d29922; }
  .blue { color: #58a6ff; }
  h2 { color: #f0f6fc; margin: 24px 0 12px; font-size: 1.2em; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
  th { text-align: left; color: #8b949e; font-size: 0.75em; text-transform: uppercase;
       letter-spacing: 0.5px; padding: 8px 12px; border-bottom: 1px solid #30363d; }
  td { padding: 10px 12px; border-bottom: 1px solid #21262d; font-size: 0.9em; }
  tr:hover { background: #161b22; }
  .pick-home { font-weight: 700; color: #58a6ff; }
  .prob-bar { display: inline-block; height: 6px; border-radius: 3px; background: #58a6ff; vertical-align: middle; margin-left: 6px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
  .badge-won { background: #238636; color: #fff; }
  .badge-lost { background: #da3633; color: #fff; }
  .badge-pending { background: #30363d; color: #8b949e; }
  .badge-paper { background: #1f2937; color: #d29922; border: 1px solid #d29922; }
  .badge-busted { background: #da3633; color: #fff; }
  .badge-active { background: #1f6feb; color: #fff; }
  .slate-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .slate-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .multiplier { font-size: 1.4em; font-weight: 700; }
  .footer { color: #484f58; font-size: 0.75em; margin-top: 32px; text-align: center; }
  .refresh { color: #484f58; font-size: 0.8em; }
</style>
</head>
<body>
<div class="container">
  <h1>CBB Predictor</h1>
  <p class="subtitle">{{ today }} &mdash; Paper Mode <span class="refresh">| Auto-refreshes every 60s</span></p>

  <div class="cards">
    <div class="card">
      <div class="label">Bankroll</div>
      <div class="value blue">${{ "%.2f"|format(bankroll / 100) }}</div>
    </div>
    <div class="card">
      <div class="label">Today's P&L</div>
      <div class="value {{ 'green' if daily_pnl >= 0 else 'red' }}">{{ "%+.2f"|format(daily_pnl / 100) }}</div>
    </div>
    <div class="card">
      <div class="label">Trades Today</div>
      <div class="value">{{ trades_today }}</div>
    </div>
    <div class="card">
      <div class="label">Record</div>
      <div class="value">{{ wins }}-{{ losses }}</div>
    </div>
  </div>

  <h2>Today's Picks</h2>
  {% if picks %}
  <table>
    <thead>
      <tr><th>Game</th><th>Home Win</th><th>Away Win</th><th>Spread</th><th>Status</th></tr>
    </thead>
    <tbody>
    {% for p in picks %}
      <tr>
        <td>
          {% if p.home_prob >= 0.5 %}<span class="pick-home">{{ p.home_name }}</span>{% else %}{{ p.home_name }}{% endif %}
          vs
          {% if p.home_prob < 0.5 %}<span class="pick-home">{{ p.away_name }}</span>{% else %}{{ p.away_name }}{% endif %}
        </td>
        <td>{{ "%.1f"|format(p.home_prob * 100) }}%<span class="prob-bar" style="width: {{ (p.home_prob * 80)|int }}px"></span></td>
        <td>{{ "%.1f"|format((1 - p.home_prob) * 100) }}%</td>
        <td>{{ "%+.1f"|format(p.spread) }}</td>
        <td>{{ p.status }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p style="color: #8b949e;">No predictions yet for today.</p>
  {% endif %}

  {% if slate %}
  <h2>Slate (Progressive Rollover)</h2>
  <div class="slate-box">
    <div class="slate-header">
      <div>
        <span class="badge badge-{{ slate.status }}">{{ slate.status|upper }}</span>
        &nbsp; {{ slate.legs_completed }}/{{ slate.total_legs }} legs
      </div>
      <div class="multiplier {{ 'green' if slate.current_cents > slate.initial_cents else 'red' if slate.current_cents < slate.initial_cents else '' }}">
        {{ "%.1f"|format(slate.current_cents / slate.initial_cents if slate.initial_cents else 0) }}x
      </div>
    </div>
    <div style="display: flex; gap: 24px; margin-bottom: 12px; font-size: 0.9em;">
      <div><span style="color: #8b949e;">Initial:</span> ${{ "%.2f"|format(slate.initial_cents / 100) }}</div>
      <div><span style="color: #8b949e;">Rolling:</span> ${{ "%.2f"|format(slate.current_cents / 100) }}</div>
      <div><span style="color: #8b949e;">P&L:</span>
        <span class="{{ 'green' if slate.current_cents > slate.initial_cents else 'red' }}">
          {{ "%+.2f"|format((slate.current_cents - slate.initial_cents) / 100) }}
        </span>
      </div>
    </div>
    {% if slate_legs %}
    <table>
      <thead><tr><th>Leg</th><th>Side</th><th>Price</th><th>Edge</th><th>Bet</th><th>Result</th></tr></thead>
      <tbody>
      {% for leg in slate_legs %}
        <tr>
          <td>{{ leg.leg_order }}</td>
          <td>{{ (leg.side or '?')|upper }}</td>
          <td>{{ leg.price or 0 }}¢</td>
          <td>{{ "%.1f"|format((leg.edge or 0) * 100) }}%</td>
          <td>${{ "%.2f"|format(leg.bet_cents / 100) }}</td>
          <td><span class="badge badge-{{ leg.status }}">{{ leg.status|upper }}</span></td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </div>
  {% endif %}

  {% if recent_trades %}
  <h2>Recent Trades</h2>
  <table>
    <thead><tr><th>Time</th><th>Side</th><th>Contracts</th><th>Price</th><th>Edge</th><th>P&L</th><th>Status</th></tr></thead>
    <tbody>
    {% for t in recent_trades %}
      <tr>
        <td>{{ t.timestamp[:16] }}</td>
        <td>{{ t.side|upper }}</td>
        <td>{{ t.contracts }}</td>
        <td>{{ t.price }}¢</td>
        <td>{{ "%.1f"|format(t.edge * 100) }}%</td>
        <td class="{{ 'green' if t.pnl > 0 else 'red' if t.pnl < 0 else '' }}">{{ "%+.2f"|format(t.pnl / 100) }}</td>
        <td><span class="badge badge-{{ t.status }}">{{ t.status|upper }}</span></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <div class="footer">CBB Predictor &mdash; Paper Mode &mdash; Kalshi Auto-Trading Bot</div>
</div>
<script>setTimeout(() => location.reload(), 60000);</script>
</body>
</html>
"""


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    conn = get_db()
    today = date.today().isoformat()

    # Bankroll & daily state
    state = conn.execute(
        "SELECT * FROM daily_state WHERE date = ?", (today,)
    ).fetchone()
    if not state:
        state = conn.execute(
            "SELECT * FROM daily_state ORDER BY date DESC LIMIT 1"
        ).fetchone()

    bankroll = state["bankroll_cents"] if state else 1500
    daily_pnl = state["daily_pnl_cents"] if state else 0
    trades_today = state["trades_today"] if state else 0

    # Today's trades for W-L record
    start_ts = f"{today}T00:00:00"
    end_ts = f"{today}T23:59:59"
    day_trades = conn.execute(
        "SELECT * FROM trades WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        (start_ts, end_ts),
    ).fetchall()

    wins = sum(1 for t in day_trades if t["status"] == "won")
    losses = sum(1 for t in day_trades if t["status"] == "lost")

    # Picks: games + predictions
    picks = []
    games = conn.execute(
        """SELECT g.*, ht.name AS home_name, at.name AS away_name
           FROM games g
           JOIN teams ht ON g.home_team_id = ht.id
           JOIN teams at ON g.away_team_id = at.id
           WHERE g.date = ? ORDER BY g.id""",
        (today,),
    ).fetchall()

    for g in games:
        pred = conn.execute(
            "SELECT * FROM predictions WHERE game_id = ? ORDER BY id DESC LIMIT 1",
            (g["id"],),
        ).fetchone()
        home_prob = pred["home_win_prob"] if pred else 0.5
        features = pred["features_json"] if pred else "{}"
        import json
        feats = json.loads(features) if features else {}
        picks.append({
            "home_name": g["home_name"],
            "away_name": g["away_name"],
            "home_prob": home_prob,
            "spread": feats.get("spread", 0) or 0,
            "status": g["status"],
        })

    # Slate
    slate_row = conn.execute(
        "SELECT * FROM slates WHERE date = ? ORDER BY id DESC LIMIT 1",
        (today,),
    ).fetchone()
    slate = dict(slate_row) if slate_row else None
    slate_legs = []
    if slate:
        legs = conn.execute(
            """SELECT sl.*, t.side, t.price, t.edge
               FROM slate_legs sl
               LEFT JOIN trades t ON sl.trade_id = t.id
               WHERE sl.slate_id = ? ORDER BY sl.leg_order""",
            (slate["id"],),
        ).fetchall()
        slate_legs = [dict(l) for l in legs]

    # Recent trades (last 20)
    recent = conn.execute(
        "SELECT * FROM trades ORDER BY id DESC LIMIT 20"
    ).fetchall()
    recent_trades = [dict(t) for t in recent]

    conn.close()

    return render_template_string(
        TEMPLATE,
        today=today,
        bankroll=bankroll,
        daily_pnl=daily_pnl,
        trades_today=trades_today,
        wins=wins,
        losses=losses,
        picks=picks,
        slate=slate,
        slate_legs=slate_legs,
        recent_trades=recent_trades,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
