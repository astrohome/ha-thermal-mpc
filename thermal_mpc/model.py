"""Grey-box multi-room RC thermal model.

Each room ``i`` is a lumped heat capacity connected to its neighbours and to
the outdoors through thermal conductances::

    dT_i/dt = sum_j g_ij (T_j - T_i) + g_io (T_out - T_i) + sum_u b_iu u + c_i

with every coefficient already divided by the room's heat capacity, so the
units are 1/h for conductances and K/h per unit of input for gains. ``c_i``
absorbs constant internal gains and sensor offsets.

The equation is linear in its parameters, so with a forward-Euler
discretisation each room is fitted independently by bounded least squares.
Bounds encode physics: conductances are non-negative, heating gains are
non-negative, cooling gains are non-positive. ``1 / g_io`` is the room's
time constant against the outdoors - a direct measure of thermal inertia.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .config import Config

HORIZON = pd.Timedelta(hours=6)
HORIZON_STRIDE = pd.Timedelta(hours=1)

BOUNDS = {
    "positive": (0.0, np.inf),
    "negative": (-np.inf, 0.0),
    "free": (-np.inf, np.inf),
}


@dataclass
class RoomParams:
    """Fitted coefficients for one room."""

    g_out: float
    g_rooms: dict[str, float]
    gains: dict[str, float]
    offset: float
    rmse_one_step: float
    n_samples: int

    @property
    def tau_out_h(self) -> float:
        """Time constant against outdoors in hours (inf if uncoupled)."""
        return 1.0 / self.g_out if self.g_out > 0 else float("inf")


@dataclass
class ThermalModel:
    """A fitted multi-room model."""

    step_h: float
    rooms: dict[str, RoomParams]
    inputs: list[str]

    def derivative(self, temps: np.ndarray, t_out: float, u: np.ndarray) -> np.ndarray:
        """Return dT/dt (K/h) for all rooms given the current state."""
        names = list(self.rooms)
        idx = {n: k for k, n in enumerate(names)}
        d = np.empty(len(names))
        for k, name in enumerate(names):
            p = self.rooms[name]
            val = p.g_out * (t_out - temps[k]) + p.offset
            for other, g in p.g_rooms.items():
                val += g * (temps[idx[other]] - temps[k])
            val += sum(p.gains[inp] * u[j] for j, inp in enumerate(self.inputs))
            d[k] = val
        return d

    def simulate(self, t0: np.ndarray, t_out: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Roll the model forward from ``t0``.

        ``t_out`` has shape (N,), ``u`` has shape (N, n_inputs). Returns the
        predicted temperatures, shape (N + 1, n_rooms), starting with ``t0``.
        """
        out = np.empty((len(t_out) + 1, len(t0)))
        out[0] = t0
        for k in range(len(t_out)):
            out[k + 1] = out[k] + self.step_h * self.derivative(out[k], t_out[k], u[k])
        return out

    def to_json(self, path: str | Path) -> None:
        """Save the model."""
        data = {
            "step_h": self.step_h,
            "inputs": self.inputs,
            "rooms": {k: asdict(v) for k, v in self.rooms.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> ThermalModel:
        """Load a saved model."""
        data = json.loads(Path(path).read_text())
        rooms = {k: RoomParams(**v) for k, v in data["rooms"].items()}
        return cls(step_h=data["step_h"], rooms=rooms, inputs=data["inputs"])


def _neighbours(config: Config) -> dict[str, list[str]]:
    nbrs: dict[str, list[str]] = {r: [] for r in config.rooms}
    for a, b in config.couplings:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def fit(df: pd.DataFrame, config: Config) -> ThermalModel:
    """Fit the model to a resampled dataset (see ``ha_history.build_dataset``)."""
    step_h = config.step / pd.Timedelta(hours=1)
    inputs = list(config.inputs)
    nbrs = _neighbours(config)
    # Forward difference: the change over the next step is explained by the
    # state now and the (time-averaged) inputs during that step.
    rooms = {}
    for room in config.rooms:
        col = f"T_{room}"
        t_i = df[col]
        dtdt = (t_i.shift(-1) - t_i) / step_h
        features = {"g_out": df["T_out"] - t_i}
        bounds = [BOUNDS["positive"]]
        for other in nbrs[room]:
            features[f"g_{other}"] = df[f"T_{other}"] - t_i
            bounds.append(BOUNDS["positive"])
        for inp in inputs:
            features[f"b_{inp}"] = df[inp]
            bounds.append(BOUNDS[config.inputs[inp].sign])
        features["offset"] = pd.Series(1.0, index=df.index)
        bounds.append(BOUNDS["free"])

        design = pd.DataFrame(features)
        mask = design.notna().all(axis=1) & dtdt.notna()
        n = int(mask.sum())
        if n < 10 * design.shape[1]:
            raise ValueError(
                f"room {room!r}: only {n} complete samples for "
                f"{design.shape[1]} parameters; collect more data"
            )
        A = design[mask].to_numpy()
        y = dtdt[mask].to_numpy()
        lo, hi = zip(*bounds, strict=True)
        res = lsq_linear(A, y, bounds=(lo, hi))
        coef = dict(zip(design.columns, res.x, strict=True))
        resid_k = (A @ res.x - y) * step_h  # one-step error in kelvin
        rooms[room] = RoomParams(
            g_out=float(coef["g_out"]),
            g_rooms={o: float(coef[f"g_{o}"]) for o in nbrs[room]},
            gains={i: float(coef[f"b_{i}"]) for i in inputs},
            offset=float(coef["offset"]),
            rmse_one_step=float(np.sqrt(np.mean(resid_k**2))),
            n_samples=n,
        )
    return ThermalModel(step_h=step_h, rooms=rooms, inputs=inputs)


def validate(
    model: ThermalModel,
    df: pd.DataFrame,
    horizon: pd.Timedelta = HORIZON,
    every: pd.Timedelta = HORIZON_STRIDE,
) -> pd.DataFrame:
    """Open-loop multi-step prediction error.

    From every ``every`` start point where all room temperatures are known,
    simulate ``horizon`` ahead using the *measured* outdoor temperature and
    inputs, and compare to measurements. Returns RMSE (K) per room at each
    horizon step. This is the number that matters for MPC; a good one-step
    fit can still drift badly over hours.
    """
    rooms = list(model.rooms)
    step = pd.Timedelta(hours=model.step_h)
    n_h = int(horizon / step)
    stride = max(1, int(every / step))
    temps = df[[f"T_{r}" for r in rooms]].to_numpy()
    t_out = df["T_out"].to_numpy()
    u = df[model.inputs].to_numpy() if model.inputs else np.zeros((len(df), 0))

    sq = np.zeros((n_h, len(rooms)))
    cnt = np.zeros((n_h, len(rooms)))
    for s in range(0, len(df) - n_h, stride):
        window_out = t_out[s : s + n_h]
        window_u = u[s : s + n_h]
        if np.isnan(temps[s]).any() or np.isnan(window_out).any():
            continue
        if np.isnan(window_u).any():
            continue
        pred = model.simulate(temps[s], window_out, window_u)[1:]
        err = pred - temps[s + 1 : s + 1 + n_h]
        ok = ~np.isnan(err)
        sq[ok] += err[ok] ** 2
        cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(sq / cnt)
    index = pd.timedelta_range(step, periods=n_h, freq=step, name="horizon")
    return pd.DataFrame(rmse, index=index, columns=rooms)
