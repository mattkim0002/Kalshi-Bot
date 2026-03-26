"""
Polymarket Bot Dashboard.

A simple web dashboard to monitor the trading bot.
Run alongside main.py:
    python3 dashboard.py

Then open http://localhost:8080 in your browser.
"""
import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

def load_json(path, default):
    try:
        f = Path(path)
        return json.loads(f.read_text()) if f.exists() else default
    except Exception:
        return default

def time_ago(ts):
    diff = time.time() - ts
    if diff < 60: return f"{int(diff)}s ago"
    if diff < 3600: return f"{int(diff/60)}m ago"
    if diff < 86400: return f"{int(diff/3600)}h ago"
    return f"{int(diff/86400)}d ago"

def pnl_color(val):
    if val > 0: return "#00c853"
    if val < 0: return "#ff1744"
    return "#aaa"

def build_html():
    positions = load_json("positions.json", {})
    trades = load_json("trades.json", [])

    total_exposure = sum(p.get("cost_basis", 0) for p in positions.values())
    total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions.values())

    # Daily P&L from today's closed trades
    today = time.time() - 86400
    daily_pnl = 0
    for t in trades:
        if t.get("action") == "SELL" and t.get("timestamp", 0) > today:
            entry = next((x for x in trades if x.get("market_id") == t.get("market_id") and x.get("action") == "BUY"), None)
            if entry:
                daily_pnl += t.get("amount_usd", 0) - entry.get("amount_usd", 0)

    # Recent trades (last 10, newest first)
    recent = sorted(trades, key=lambda x: x.get("timestamp", 0), reverse=True)[:10]

    # Build positions HTML
    pos_html = ""
    if positions:
        for mid, p in positions.items():
            pnl = p.get("unrealized_pnl", 0)
            pnl_pct = (pnl / p.get("cost_basis", 1)) * 100 if p.get("cost_basis") else 0
            color = pnl_color(pnl)
            direction = p.get("direction", "")
            dir_color = "#00c853" if direction == "YES" else "#ff6d00"
            entry = p.get("entry_price", 0)
            current = p.get("current_price", entry)
            hold = (time.time() - p.get("entry_time", time.time())) / 3600
            pos_html += f"""
            <div class="card position-card">
                <div class="pos-header">
                    <span class="pos-question">{p.get('question', mid)[:60]}</span>
                    <span class="badge" style="background:{dir_color}">{direction}</span>
                </div>
                <div class="pos-stats">
                    <div class="stat"><div class="stat-label">Invested</div><div class="stat-val">${p.get('cost_basis',0):.2f}</div></div>
                    <div class="stat"><div class="stat-label">Entry Price</div><div class="stat-val">{entry:.0%}</div></div>
                    <div class="stat"><div class="stat-label">Current Price</div><div class="stat-val">{current:.0%}</div></div>
                    <div class="stat"><div class="stat-label">P&L</div><div class="stat-val" style="color:{color}">{'+' if pnl>=0 else ''}{pnl:.2f} ({pnl_pct:+.1f}%)</div></div>
                    <div class="stat"><div class="stat-label">Held For</div><div class="stat-val">{hold:.1f}h</div></div>
                </div>
                <div class="pos-url"><a href="{p.get('url','#')}" target="_blank">View on Polymarket →</a></div>
            </div>"""
    else:
        pos_html = '<div class="empty">No open positions. Bot is scanning for opportunities.</div>'

    # Build trades HTML
    trades_html = ""
    for t in recent:
        action = t.get("action", "")
        color = "#00c853" if action == "BUY" else "#ff6d00"
        ts = t.get("timestamp", 0)
        trades_html += f"""
        <div class="trade-row">
            <span class="trade-badge" style="background:{color}">{action}</span>
            <span class="trade-q">{t.get('question','')[:45]}</span>
            <span class="trade-dir">{t.get('direction','')}</span>
            <span class="trade-amt">${t.get('amount_usd',0):.2f}</span>
            <span class="trade-time">{time_ago(ts)}</span>
        </div>"""
    if not trades_html:
        trades_html = '<div class="empty">No trades yet.</div>'

    total_color = pnl_color(total_pnl)
    daily_color = pnl_color(daily_pnl)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Polymarket Bot</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #f0f0f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  .live-dot {{ display: inline-block; width: 8px; height: 8px; background: #00c853; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 28px; }}
  .stat-card {{ background: #1a1a1a; border-radius: 12px; padding: 16px; }}
  .stat-card .label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .stat-card .value {{ font-size: 24px; font-weight: 700; }}
  .stat-card .sub {{ font-size: 11px; color: #555; margin-top: 4px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card {{ background: #1a1a1a; border-radius: 12px; padding: 16px; margin-bottom: 10px; }}
  .position-card {{ border-left: 3px solid #333; }}
  .pos-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; gap: 10px; }}
  .pos-question {{ font-size: 14px; font-weight: 500; flex: 1; }}
  .badge {{ font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 20px; color: #fff; white-space: nowrap; }}
  .pos-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 10px; margin-bottom: 10px; }}
  .stat {{ }}
  .stat-label {{ font-size: 10px; color: #555; margin-bottom: 2px; text-transform: uppercase; }}
  .stat-val {{ font-size: 14px; font-weight: 600; }}
  .pos-url a {{ font-size: 11px; color: #444; text-decoration: none; }}
  .pos-url a:hover {{ color: #888; }}
  .trade-row {{ display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #1f1f1f; font-size: 13px; flex-wrap: wrap; }}
  .trade-badge {{ font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 20px; color: #fff; white-space: nowrap; }}
  .trade-q {{ flex: 1; color: #ccc; min-width: 120px; }}
  .trade-dir {{ color: #666; font-size: 11px; }}
  .trade-amt {{ font-weight: 600; }}
  .trade-time {{ color: #444; font-size: 11px; white-space: nowrap; }}
  .empty {{ color: #444; font-size: 13px; padding: 20px 0; text-align: center; }}
  .section {{ margin-bottom: 28px; }}
  .refresh-note {{ text-align: center; color: #333; font-size: 11px; margin-top: 30px; }}
</style>
</head>
<body>
<h1><span class="live-dot"></span>Polymarket Bot</h1>
<p class="subtitle">Auto-refreshes every 30 seconds</p>

<div class="stats-grid">
  <div class="stat-card">
    <div class="label">Open Positions</div>
    <div class="value">{len(positions)}</div>
    <div class="sub">Active bets</div>
  </div>
  <div class="stat-card">
    <div class="label">Money in Bets</div>
    <div class="value">${total_exposure:.2f}</div>
    <div class="sub">Currently deployed</div>
  </div>
  <div class="stat-card">
    <div class="label">Unrealized P&L</div>
    <div class="value" style="color:{total_color}">{'+' if total_pnl>=0 else ''}{total_pnl:.2f}</div>
    <div class="sub">On open positions</div>
  </div>
  <div class="stat-card">
    <div class="label">Today's P&L</div>
    <div class="value" style="color:{daily_color}">{'+' if daily_pnl>=0 else ''}{daily_pnl:.2f}</div>
    <div class="sub">Realized today</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Trades</div>
    <div class="value">{len(trades)}</div>
    <div class="sub">All time</div>
  </div>
</div>

<div class="section">
  <h2>Open Positions</h2>
  {pos_html}
</div>

<div class="section">
  <h2>Recent Trades</h2>
  <div class="card">
    {trades_html}
  </div>
</div>

<p class="refresh-note">Last updated: {time.strftime('%I:%M:%S %p')}</p>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, format, *args):
        pass  # suppress access logs


if __name__ == "__main__":
    port = 8080
    print(f"Dashboard running at http://localhost:{port}")
    print("Open that URL in your browser. Press Ctrl+C to stop.")
    HTTPServer(("", port), Handler).serve_forever()
