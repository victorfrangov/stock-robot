"""Run one model experiment on the DEV period and score it with the pre-declared metrics.

    python research/experiment.py NAME [--override cfg.yaml] [--no-nn] [--slots 2] [--notes "..."]

* Runs the walk-forward with `--end 2020-01-01` (the holdout stays sealed).
* Takes one of `--slots` machine-wide slots first (walk-forwards use ~3.5 GB each; the box has 16 GB).
* Writes <data>/reports/experiments/NAME.json and prints a one-line summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.metrics import evaluate, with_overrides  # noqa: E402
from robot.config import load_config  # noqa: E402

DEV_END = "2020-01-01"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Slot:
    """Directory-based semaphore that survives crashes (stale slots hold a dead PID)."""

    def __init__(self, root: Path, n: int):
        self.root, self.n, self.held = root, n, None
        root.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        while True:
            for i in range(self.n):
                d = self.root / f"slot-{i}"
                pidfile = d / "pid"
                if d.exists() and pidfile.exists():
                    try:
                        if not _alive(int(pidfile.read_text())):
                            pidfile.unlink()
                            d.rmdir()
                    except (ValueError, OSError):
                        pass
                try:
                    d.mkdir()
                except FileExistsError:
                    continue
                pidfile.write_text(str(os.getpid()))
                self.held = d
                return self
            time.sleep(15)

    def __exit__(self, *exc):
        if self.held:
            try:
                (self.held / "pid").unlink()
                self.held.rmdir()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--override", help="yaml with config overrides (same layout as config.yaml)")
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--notes", default="")
    ap.add_argument("--scores", help="skip the walk-forward and score this existing scores.parquet")
    a = ap.parse_args()

    base = load_config()
    overrides = yaml.safe_load(Path(a.override).read_text()) if a.override else {}
    cfg = with_overrides(base, overrides)
    out_dir = base.root / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.scores:
        scores_path, log_path = Path(a.scores), None
    else:
        cmd = [sys.executable, "-m", "robot.cli", *(["-c", a.override] if a.override else []), "backtest",
               "--end", DEV_END, *(["--no-nn"] if a.no_nn else [])]
        log_path = out_dir / f"{a.name}.log"
        with Slot(base.root / "locks", a.slots):
            t0 = time.time()
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parent.parent)
            elapsed = time.time() - t0
        text = log_path.read_text()
        m = re.search(r"report: (\S+)", text)
        if proc.returncode != 0 or not m:
            print(f"FAILED (exit {proc.returncode}); see {log_path}\n" + text[-3000:])
            sys.exit(1)
        scores_path = Path(m.group(1)) / "scores.parquet"
        print(f"walk-forward done in {elapsed / 60:.1f} min -> {scores_path}")

    scores = pd.read_parquet(scores_path)
    cols = [c for c in scores.columns if c.startswith("score")]
    results = {c: evaluate(cfg, scores, c) for c in cols}
    summary = {"name": a.name, "overrides": overrides, "no_nn": a.no_nn, "notes": a.notes,
               "scores": str(scores_path), "log": str(log_path) if log_path else None, "results": results}
    (out_dir / f"{a.name}.json").write_text(json.dumps(summary, indent=2, default=str))
    for c, r in results.items():
        print(f"{a.name:30s} {c:10s} top50_t={r['top50_t']:5.2f} top30_t={r['top30_t']:5.2f} ic={r['ic']:.4f} "
              f"| DEV excess={r['excess']:+.2%} IR={r['ir']:.2f} TE={r['te']:.1%} turn={r['turnover']:.0f}")


if __name__ == "__main__":
    main()
