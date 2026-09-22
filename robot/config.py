"""Configuration: defaults merged with config.yaml and environment overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "data_dir": "data",
    "data": {
        "start_date": "2003-01-01",
        "sec_user_agent": "stock-robot research bot",
        "fred_series": ["VIXCLS", "T10Y2Y", "DGS3MO", "DGS10"],
    },
    "label": {"horizon": 5},
    "model": {
        "ensemble": {"gbm": 0.7, "nn": 0.3},
        "train_sample_every": 2,
        # Raw rate levels are non-stationary: trees use them as a clock and memorise eras.
        "exclude_features": ["mkt_dgs10", "mkt_dgs3mo", "mkt_t10y2y", "mkt_vixcls"],
        "earnings_features": False,
        "industry_momentum": False,
        "gbm": {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 500,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
        },
        "nn": {
            "hidden_layers": [256, 128, 64],
            "dropout": 0.3,
            "epochs": 8,
            "batch_size": 4096,
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
    },
    "portfolio": {
        "mode": "topk",
        "score_halflife": 1,
        "trade_band": 0.0,
        "tilt_strength": 2.0,
        "tilt_exclude": 0.2,
        "tilt_cap_power": 1.0,
        "top_k": 50,
        "hold_buffer": 150,
        "rebalance_days": 5,
        "weighting": "cap",
        "max_weight": 0.10,
        "cash_buffer": 0.0,
        "regime_filter": False,
        "regime_exposure": 0.5,
        "min_price": 5.0,
    },
    "costs": {"commission_bps": 0.5, "slippage_bps": 5.0},
    "backtest": {"start": "2012-01-01", "retrain_months": 6, "min_train_years": 5},
    "broker": {
        "host": "127.0.0.1",
        "port": 4002,
        "client_id": 17,
        "paper_only": True,
        "manage_all_positions": False,
        "order_type": "auto",
    },
    "risk": {
        "max_order_value": 50000,
        "max_daily_loss_pct": 4.0,
        "max_drawdown_pct": 25.0,
        "kill_switch_file": "KILL",
    },
    "schedule": {"trade_time": "09:00", "retrain_weekday": 5, "retrain_time": "10:00"},
}


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @property
    def root(self) -> Path:
        path = Path(os.path.expanduser(self["data_dir"]))
        return path if path.is_absolute() else PROJECT_ROOT / path

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("ROBOT_CONFIG", PROJECT_ROOT / "config.yaml"))
    user = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = _merge(DEFAULTS, user or {})
    if os.environ.get("ROBOT_DATA_DIR"):
        cfg["data_dir"] = os.environ["ROBOT_DATA_DIR"]
    if os.environ.get("SEC_USER_AGENT"):
        cfg["data"]["sec_user_agent"] = os.environ["SEC_USER_AGENT"]
    if os.environ.get("IB_PORT"):
        cfg["broker"]["port"] = int(os.environ["IB_PORT"])
    return Config(cfg)
