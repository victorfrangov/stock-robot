"""Command line interface: `robot <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from robot.config import load_config


def _setup_logging(cfg, verbose: bool) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(RotatingFileHandler(cfg.path("logs", "robot.log"), maxBytes=5_000_000, backupCount=5))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt, handlers=handlers)
    for noisy in ("yfinance", "ib_async", "urllib3", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _pct(x) -> str:
    return f"{x * 100:6.2f}%" if isinstance(x, (int, float)) else str(x)


def cmd_data(cfg, a):
    from robot.data.update import update_all

    update_all(cfg, full=a.full, skip_fundamentals=a.no_fundamentals)


def cmd_backtest(cfg, a):
    from robot.backtest import run_backtest
    from robot.data.macro import load_macro
    from robot.data.prices import load_prices
    from robot.pipeline import get_panel

    if a.start:
        cfg["backtest"]["start"] = a.start
    rep = run_backtest(cfg, get_panel(cfg, a.rebuild), load_prices(cfg), load_macro(cfg), use_nn=not a.no_nn)
    rows = [("strategy (ensemble)", rep["score"])]
    rows += [(f"  {k.removeprefix('score_')} only", rep[k]) for k in rep if k.startswith("score_")]
    rows += [("SPY", rep["benchmark_spy"]), ("equal-weight S&P", rep["benchmark_equal_weight"])]
    print(f"\n{'':22s} {'CAGR':>8s} {'Vol':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'IC':>7s} {'Turn/yr':>8s}")
    for name, m in rows:
        print(f"{name:22s} {_pct(m['cagr']):>8s} {_pct(m['vol']):>8s} {m['sharpe']:7.2f} "
              f"{_pct(m['max_drawdown']):>8s} {m.get('ic_mean', float('nan')):7.4f} "
              f"{m.get('turnover_per_year', float('nan')):8.1f}")
    print(f"\nreport: {rep['out_dir']}")


def cmd_train(cfg, a):
    from robot.data.update import update_all
    from robot.pipeline import get_panel, train_production

    if not a.skip_data:
        update_all(cfg)
    print(f"model saved to {train_production(cfg, use_nn=not a.no_nn, panel=get_panel(cfg))}")


def cmd_signals(cfg, a):
    from robot.data.update import update_all
    from robot.pipeline import score_latest

    if a.refresh:
        update_all(cfg, current_only=True, skip_fundamentals=True)
    asof, scored = score_latest(cfg)
    print(f"Model ranking as of close {asof.date()} (top {a.n}):")
    print(scored.head(a.n)[["rank", "ticker", "score", "ret_21", "mom_12_1", "ey"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


def cmd_trade(cfg, a):
    from robot.live import run_trade

    s = run_trade(cfg, dry_run=a.dry_run, force_rebalance=a.force, refresh_data=not a.no_refresh)
    print(f"\nas of {s['asof']}  account {s.get('account')}  net liq ${s.get('net_liq', 0):,.0f}  "
          f"risk_off={s['risk_off']}  drawdown={s.get('drawdown_pct')}%")
    for r in s.get("risk", []):
        print(f"  RISK: {r}")
    if s.get("note"):
        print(s["note"])
    for o in s.get("orders", []):
        flag = f"  REJECTED: {o['reject']}" if o["reject"] else ""
        print(f"  {'BUY ' if o['qty'] > 0 else 'SELL'} {abs(o['qty']):6d} {o['ticker']:6s} "
              f"@ ~${o['price']:.2f}  ({o['current']:.0f} -> {o['target']:.0f}){flag}")
    if a.dry_run:
        print("\n(dry run - nothing was sent)")
    elif s.get("style"):
        print(f"\nsubmitted as {s['style']} orders")


def cmd_status(cfg, a):
    from robot.broker import Broker
    from robot.journal import Journal

    j = Journal(cfg)
    with Broker(cfg) as b:
        nl = b.net_liquidation()
        print(f"account {b.account}  net liquidation ${nl:,.2f}")
        pf = b.portfolio()
        total = sum(p.marketValue for p in pf)
        print(f"{len(pf)} positions, ${total:,.0f} invested ({total / nl:.0%})")
        for p in sorted(pf, key=lambda p: -p.marketValue):
            print(f"  {p.contract.symbol:6s} {p.position:7.0f} sh  ${p.marketValue:>10,.0f}  "
                  f"P/L ${p.unrealizedPNL:>9,.0f}")
        open_orders = [t for t in b.ib.openTrades() if t.order.orderRef == "stock-robot"]
        if open_orders:
            print(f"{len(open_orders)} robot orders pending:")
            for t in open_orders:
                print(f"  {t.order.action} {t.order.totalQuantity:.0f} {t.contract.symbol} ({t.order.tif}) "
                      f"{t.orderStatus.status}")
    eq = j.equity()
    if len(eq):
        print(f"\nequity history: start ${eq.iloc[0]:,.0f} -> now ${eq.iloc[-1]:,.0f} "
              f"({eq.iloc[-1] / eq.iloc[0] - 1:+.2%}), peak ${eq.max():,.0f}")
    print(f"last rebalance: {j.get('last_rebalance')}   halted: {j.get('halted')}")
    print("\nrecent runs:")
    print(j.recent("runs", 5).to_string(index=False))


def cmd_flatten(cfg, a):
    from robot.live import flatten

    if not a.yes:
        sys.exit("refusing without --yes (this sells every stock position in the paper account)")
    for f in flatten(cfg):
        print(f)


def cmd_resume(cfg, a):
    from robot.journal import Journal

    from datetime import datetime

    from robot.broker import ET

    j = Journal(cfg)
    j.set("halted", None)
    j.set("peak_reset", datetime.now(ET).date().isoformat())
    kill = cfg.root / cfg.risk.kill_switch_file
    if kill.exists():
        kill.unlink()
    print("halt cleared; trading resumes on the next run")


def cmd_kill(cfg, a):
    (cfg.root / cfg.risk.kill_switch_file).write_text("stop\n")
    print(f"kill switch created at {cfg.root / cfg.risk.kill_switch_file}; `robot resume` removes it")


def cmd_scheduler(cfg, a):
    from robot.scheduler import run_scheduler

    run_scheduler(cfg)


def cmd_install_service(cfg, a):
    from robot.config import PROJECT_ROOT

    label = "com.stockrobot.scheduler"
    exe = Path(sys.executable).parent / "robot"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    logs = cfg.path("logs", "launchd.log")
    plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key><array><string>{exe}</string><string>scheduler</string></array>
  <key>WorkingDirectory</key><string>{PROJECT_ROOT}</string>
  <key>EnvironmentVariables</key><dict><key>ROBOT_DATA_DIR</key><string>{cfg.root}</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{logs}</string>
  <key>StandardErrorPath</key><string>{logs}</string>
</dict></plist>
""")
    print(f"wrote {plist}\nstart it with:  launchctl load -w {plist}\nstop it with:   launchctl unload {plist}")


def cmd_inspect(cfg, a):
    from robot.pipeline import score_latest

    t = a.ticker.upper().replace(".", "-")
    asof, scored = score_latest(cfg)
    row = scored[scored["ticker"] == t]
    if row.empty:
        print(f"{t} is not in the current universe")
    else:
        r = row.iloc[0]
        print(f"{t}: rank {int(r['rank'])}/{len(scored)} as of {asof.date()}, score {r['score']:+.3f}")
        feats = r.drop(["date", "ticker", "rank", "score"]).dropna()
        print(feats.to_string(float_format=lambda v: f"{v:+.4f}"))
    if a.ib:
        try:
            from ib_fundamental import CompanyFinancials
        except ImportError:
            sys.exit("pip install -e ../ib_fundamental to enable --ib")
        from robot.broker import Broker, to_ib

        with Broker(cfg) as b:
            cf = CompanyFinancials(ib=b.ib, symbol=to_ib(t))
            for name in ("ratios", "analyst_forecast", "eps_ttm", "revenue_ttm"):
                try:
                    print(f"\n== IBKR {name}\n{getattr(cf, name)}")
                except Exception as e:  # fundamentals need an IBKR data subscription
                    print(f"\n== IBKR {name}: unavailable ({e})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="robot", description="stock-robot: daily swing-trading AI on IBKR paper")
    p.add_argument("-c", "--config", help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="download/refresh prices, macro and SEC fundamentals")
    s.add_argument("--full", action="store_true", help="re-download everything")
    s.add_argument("--no-fundamentals", action="store_true")
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("backtest", help="walk-forward backtest with realistic costs")
    s.add_argument("--no-nn", action="store_true", help="GBM only (much faster)")
    s.add_argument("--start", help="first out-of-sample date")
    s.add_argument("--rebuild", action="store_true", help="rebuild the feature panel")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("train", help="train the production model on all data")
    s.add_argument("--no-nn", action="store_true")
    s.add_argument("--skip-data", action="store_true", help="don't refresh data first")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("signals", help="print today's model ranking (no broker needed)")
    s.add_argument("-n", type=int, default=25)
    s.add_argument("--refresh", action="store_true", help="refresh prices first")
    s.set_defaults(fn=cmd_signals)

    s = sub.add_parser("trade", help="run the daily trading job against the paper account")
    s.add_argument("--dry-run", action="store_true", help="plan orders but don't send them")
    s.add_argument("--force", action="store_true", help="rebalance even if not due")
    s.add_argument("--no-refresh", action="store_true", help="skip the price refresh")
    s.set_defaults(fn=cmd_trade)

    sub.add_parser("status", help="account, positions, pending orders, history").set_defaults(fn=cmd_status)
    s = sub.add_parser("flatten", help="sell everything")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_flatten)
    sub.add_parser("kill", help="create the kill switch (stops all trading)").set_defaults(fn=cmd_kill)
    sub.add_parser("resume", help="clear halt state and kill switch").set_defaults(fn=cmd_resume)
    sub.add_parser("scheduler", help="run forever: daily trade + weekly retrain").set_defaults(fn=cmd_scheduler)
    sub.add_parser("install-service", help="install a macOS launchd agent running the scheduler") \
        .set_defaults(fn=cmd_install_service)
    s = sub.add_parser("inspect", help="why does the model like/dislike a ticker")
    s.add_argument("ticker")
    s.add_argument("--ib", action="store_true", help="also show IBKR fundamentals via ib_fundamental")
    s.set_defaults(fn=cmd_inspect)

    a = p.parse_args(argv)
    cfg = load_config(a.config)
    _setup_logging(cfg, a.verbose)
    a.fn(cfg, a)


if __name__ == "__main__":
    main()
