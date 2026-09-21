"""Models that score stocks for the next `horizon` days.

- GBMModel: LightGBM regression on the cross-sectional rank of forward return.
- NNModel: the original stock-robot FeedForward net (LayerNorm + SiLU MLP),
  retrained on ranked features with an MSE-on-rank objective.
- Ensemble: blends the per-day ranks of each member's predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _prep(X: pd.DataFrame, mkt_mean: pd.Series | None = None, mkt_std: pd.Series | None = None) -> np.ndarray:
    """NN input: ranked features as-is (NaN -> 0 = median), market features z-scored."""
    X = X.copy()
    mkt = [c for c in X.columns if c.startswith("mkt_")]
    if mkt and mkt_mean is not None:
        X[mkt] = ((X[mkt] - mkt_mean[mkt]) / mkt_std[mkt]).clip(-5, 5)
    return np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0)


class GBMModel:
    name = "gbm"

    def __init__(self, params: dict | None = None):
        self.params = {"objective": "regression", "verbose": -1, "n_jobs": -1, **(params or {})}
        self.model: lgb.LGBMRegressor | None = None
        self.features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMModel":
        self.features = list(X.columns)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])

    def importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        self.model.booster_.save_model(str(path / "gbm.txt"))

    @classmethod
    def load(cls, path: Path, features: list[str]) -> "GBMModel":
        m = cls()
        m.features = features
        booster = lgb.Booster(model_file=str(path / "gbm.txt"))
        m.model = type("Wrapped", (), {"predict": lambda self, X: booster.predict(X), "booster_": booster})()
        return m


class NNModel:
    name = "nn"

    def __init__(self, params: dict | None = None):
        self.params = {"hidden_layers": [256, 128, 64], "dropout": 0.3, "epochs": 8, "batch_size": 4096,
                       "lr": 1e-3, "weight_decay": 1e-4, **(params or {})}
        self.net = None
        self.features: list[str] = []
        self.mkt_mean: pd.Series | None = None
        self.mkt_std: pd.Series | None = None

    @staticmethod
    def _device():
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build(self, n_in: int):
        import torch.nn as nn

        layers, prev = [], n_in
        for h in self.params["hidden_layers"]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(self.params["dropout"])]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y: np.ndarray, dates: np.ndarray | None = None) -> "NNModel":
        import torch

        torch.manual_seed(0)
        self.features = list(X.columns)
        mkt = [c for c in X.columns if c.startswith("mkt_")]
        self.mkt_mean = X[mkt].mean()
        self.mkt_std = X[mkt].std().replace(0, 1)
        Xn = _prep(X, self.mkt_mean, self.mkt_std)
        y = np.asarray(y, dtype=np.float32)

        # last 10% of dates is a validation set for early stopping
        if dates is not None:
            cut = np.quantile(dates.astype("datetime64[ns]").astype(np.int64), 0.9)
            val = dates.astype("datetime64[ns]").astype(np.int64) >= cut
        else:
            val = np.zeros(len(y), bool)
            val[int(len(y) * 0.9):] = True
        dev = self._device()
        net = self._build(Xn.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        Xt, yt = torch.from_numpy(Xn[~val]), torch.from_numpy(y[~val])
        Xv, yv = torch.from_numpy(Xn[val]).to(dev), torch.from_numpy(y[val]).to(dev)
        bs = self.params["batch_size"]
        best, best_state, patience = np.inf, None, 0
        for epoch in range(self.params["epochs"]):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), bs):
                idx = perm[i : i + bs]
                xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
                loss = torch.nn.functional.mse_loss(net(xb).squeeze(-1), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = float(torch.cat([
                    (net(Xv[i : i + 65536]).squeeze(-1) - yv[i : i + 65536]) ** 2 for i in range(0, len(Xv), 65536)
                ]).mean()) if len(Xv) else 0.0
            log.debug("nn epoch %d val_mse %.5f", epoch, vloss)
            if vloss < best - 1e-6:
                best, patience = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_state:
            net.load_state_dict(best_state)
        self.net = net.to("cpu").eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        Xn = torch.from_numpy(_prep(X[self.features], self.mkt_mean, self.mkt_std))
        with torch.no_grad():
            return torch.cat([self.net(Xn[i : i + 65536]).squeeze(-1) for i in range(0, len(Xn), 65536)]).numpy()

    def save(self, path: Path) -> None:
        import torch

        torch.save({"state": self.net.state_dict(), "params": self.params,
                    "mkt_mean": self.mkt_mean.to_dict(), "mkt_std": self.mkt_std.to_dict()}, path / "nn.pt")

    @classmethod
    def load(cls, path: Path, features: list[str]) -> "NNModel":
        import torch

        blob = torch.load(path / "nn.pt", map_location="cpu", weights_only=False)
        m = cls(blob["params"])
        m.features = features
        m.mkt_mean, m.mkt_std = pd.Series(blob["mkt_mean"]), pd.Series(blob["mkt_std"])
        m.net = m._build(len(features))
        m.net.load_state_dict(blob["state"])
        m.net.eval()
        return m


def _day_rank(scores: np.ndarray, dates: np.ndarray) -> np.ndarray:
    return pd.Series(scores).groupby(pd.Series(dates)).rank(pct=True).to_numpy() - 0.5


class Ensemble:
    def __init__(self, weights: dict[str, float], gbm_params: dict | None = None, nn_params: dict | None = None):
        self.weights = {k: v for k, v in weights.items() if v > 0}
        self.members: dict[str, GBMModel | NNModel] = {}
        if "gbm" in self.weights:
            self.members["gbm"] = GBMModel(gbm_params)
        if "nn" in self.weights:
            try:
                import torch  # noqa: F401

                self.members["nn"] = NNModel(nn_params)
            except ImportError:
                log.warning("torch not installed - ensemble runs GBM only")
                self.weights.pop("nn")
        self.features: list[str] = []
        self.meta: dict = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, dates: np.ndarray) -> "Ensemble":
        self.features = list(X.columns)
        for name, m in self.members.items():
            log.info("fitting %s on %d rows x %d features", name, *X.shape)
            if name == "nn":
                m.fit(X, y, dates)
            else:
                m.fit(X, y)
        return self

    def predict_members(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {name: m.predict(X[self.features]) for name, m in self.members.items()}

    def predict(self, X: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
        preds = self.predict_members(X)
        total = sum(self.weights[n] for n in preds)
        return sum(self.weights[n] * _day_rank(p, dates) for n, p in preds.items()) / total

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for m in self.members.values():
            m.save(path)
        self.meta = {**(meta or {}), "features": self.features, "weights": self.weights}
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "Ensemble":
        meta = json.loads((path / "meta.json").read_text())
        ens = cls.__new__(cls)
        ens.weights, ens.features, ens.meta, ens.members = meta["weights"], meta["features"], meta, {}
        if "gbm" in ens.weights:
            ens.members["gbm"] = GBMModel.load(path, ens.features)
        if "nn" in ens.weights and (path / "nn.pt").exists():
            ens.members["nn"] = NNModel.load(path, ens.features)
        ens.weights = {k: v for k, v in ens.weights.items() if k in ens.members}
        return ens
