"""Grey-box multi-room RC thermal model.

Each room ``i`` is a lumped heat capacity connected to its neighbours and to
the outdoors through thermal conductances::

    dT_i/dt = sum_j g_ij (T_j - T_i) + g_io (T_out - T_i) + sum_u b_iu u + c_i

Every coefficient is already divided by the room's heat capacity, so
conductances are in 1/h and gains in K/h per unit of input. ``1 / g_io`` is
the room's time constant against the outdoors, a direct measure of thermal
inertia. ``c_i`` absorbs constant internal gains and sensor offsets.

The equation is linear in its parameters, so with a forward-Euler
discretisation each room is fitted independently by sign-constrained least
squares. The signs encode physics: conductances and heating gains are
non-negative, cooling gains are non-positive.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .dataset import Dataset
from .identify import Layout, integral_fit, refine
from .lsq import FREE, NEGATIVE, POSITIVE, nnls, signed_lstsq

__all__ = [
    "FREE",
    "NEGATIVE",
    "POSITIVE",
    "ModelSpec",
    "NotEnoughDataError",
    "RoomParams",
    "ThermalModel",
    "fit",
    "nnls",
    "signed_lstsq",
    "validate",
]

MIN_INPUT_COVERAGE = 0.5
SAMPLES_PER_PARAM = 10


class NotEnoughDataError(ValueError):
    """Raised when the dataset cannot support a fit yet."""


@dataclass(frozen=True)
class ModelSpec:
    """Which dataset columns play which role."""

    rooms: list[str]
    outdoor: str
    inputs: dict[str, str]  # column -> sign
    couplings: list[tuple[str, str]] | None = None  # None = every pair

    def neighbours(self) -> dict[str, list[str]]:
        """Adjacency list of room couplings."""
        pairs = (
            self.couplings
            if self.couplings is not None
            else list(itertools.combinations(self.rooms, 2))
        )
        nbrs: dict[str, list[str]] = {r: [] for r in self.rooms}
        for a, b in pairs:
            nbrs[a].append(b)
            nbrs[b].append(a)
        return nbrs


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
    def tau_out_h(self) -> float | None:
        """Time constant against outdoors in hours (None if uncoupled)."""
        return 1.0 / self.g_out if self.g_out > 1e-9 else None


@dataclass
class ThermalModel:
    """A fitted multi-room model."""

    step_h: float
    outdoor: str
    rooms: dict[str, RoomParams]
    inputs: list[str]
    unused_inputs: list[str] = field(default_factory=list)

    def derivative(self, temps: np.ndarray, t_out: float, u: np.ndarray) -> np.ndarray:
        """Return dT/dt (K/h) for all rooms."""
        names = list(self.rooms)
        idx = {n: k for k, n in enumerate(names)}
        d = np.empty(len(names))
        for k, name in enumerate(names):
            p = self.rooms[name]
            val = p.g_out * (t_out - temps[k]) + p.offset
            for other, g in p.g_rooms.items():
                val += g * (temps[idx[other]] - temps[k])
            for j, inp in enumerate(self.inputs):
                val += p.gains.get(inp, 0.0) * u[j]
            d[k] = val
        return d

    def simulate(self, t0: np.ndarray, t_out: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Roll forward from ``t0``; returns shape (N + 1, n_rooms)."""
        out = np.empty((len(t_out) + 1, len(t0)))
        out[0] = t0
        for k in range(len(t_out)):
            out[k + 1] = out[k] + self.step_h * self.derivative(out[k], t_out[k], u[k])
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible data."""
        return {
            "step_h": self.step_h,
            "outdoor": self.outdoor,
            "inputs": self.inputs,
            "unused_inputs": self.unused_inputs,
            "rooms": {k: asdict(v) for k, v in self.rooms.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThermalModel:
        """Inverse of :meth:`to_dict`."""
        return cls(
            step_h=data["step_h"],
            outdoor=data["outdoor"],
            inputs=data["inputs"],
            unused_inputs=data.get("unused_inputs", []),
            rooms={k: RoomParams(**v) for k, v in data["rooms"].items()},
        )


def fit(
    ds: Dataset,
    spec: ModelSpec,
    rows: slice | None = None,
    window_h: float = 1.0,
    horizon_h: float | None = 6.0,
) -> ThermalModel:
    """Fit all rooms on ``ds`` (optionally only ``rows`` of it).

    Stage 1 regresses ``window_h``-hour temperature changes (see
    :mod:`.identify`); stage 2 refines on ``horizon_h``-hour open-loop
    prediction error. ``horizon_h=None`` skips stage 2.
    """
    step_h = ds.step / 3600.0
    sl = rows or slice(None)
    col = {k: v[sl] for k, v in ds.columns.items()}
    missing = [c for c in [*spec.rooms, spec.outdoor] if c not in col]
    if missing:
        raise NotEnoughDataError(f"no data yet for {', '.join(missing)}")

    # Inputs that were barely recorded would throw away most rows; leave them
    # out rather than starve the fit.
    inputs, unused = [], []
    for name in spec.inputs:
        cov = np.mean(~np.isnan(col[name])) if name in col else 0.0
        (inputs if cov >= MIN_INPUT_COVERAGE else unused).append(name)

    nbrs = spec.neighbours()
    layout = Layout(spec.rooms, nbrs, inputs, dict(spec.inputs))
    temps = np.column_stack([col[r] for r in spec.rooms])
    t_out = col[spec.outdoor]
    n = len(t_out)
    u = np.column_stack([col[i] for i in inputs]) if inputs else np.zeros((n, 0))

    window = max(1, round(window_h / step_h))
    theta, used = integral_fit(temps, t_out, u, layout, step_h, window)
    for (room, a, b), n_used in zip(layout.blocks(), used, strict=True):
        if n_used < SAMPLES_PER_PARAM * (b - a):
            raise NotEnoughDataError(
                f"{room}: {n_used} complete windows for {b - a} parameters"
            )
    if horizon_h:
        horizon = max(1, round(horizon_h / step_h))
        # About one segment per hour, capped so a long history stays fast.
        stride = max(round(1 / step_h), n // 1500)
        theta, _ = refine(theta, layout, temps, t_out, u, step_h, horizon, stride)

    # One-step residuals of the final model, for reference.
    A, g_out, G, c = layout.matrices(theta)
    pred = temps[:-1] + step_h * (
        temps[:-1] @ A.T + t_out[:-1, None] * g_out + u[:-1] @ G.T + c
    )
    resid = pred - temps[1:]
    rooms = {}
    for k, (room, a, _) in enumerate(layout.blocks()):
        ok = ~np.isnan(resid[:, k])
        nb = nbrs[room]
        b = a + 1 + len(nb)
        rooms[room] = RoomParams(
            g_out=float(theta[a]),
            g_rooms={o: float(theta[a + 1 + j]) for j, o in enumerate(nb)},
            gains={i: float(theta[b + j]) for j, i in enumerate(inputs)},
            offset=float(theta[b + len(inputs)]),
            rmse_one_step=float(np.sqrt(np.mean(resid[ok, k] ** 2)))
            if ok.any()
            else 0.0,
            n_samples=used[k],
        )
    return ThermalModel(
        step_h=step_h,
        outdoor=spec.outdoor,
        rooms=rooms,
        inputs=inputs,
        unused_inputs=unused,
    )


def validate(
    model: ThermalModel,
    ds: Dataset,
    rows: slice | None = None,
    horizon_h: float = 6.0,
    every_h: float = 1.0,
) -> dict[str, list[float | None]]:
    """Open-loop multi-step prediction RMSE (K) per room per horizon step.

    From every ``every_h`` start point where all rooms are known, simulate
    ``horizon_h`` ahead with the measured outdoor temperature and inputs and
    compare with measurements. This is the figure that matters for control:
    a good one-step fit can still drift badly over hours.
    """
    sl = rows or slice(None)
    names = list(model.rooms)
    temps = np.column_stack([ds.columns[r][sl] for r in names])
    t_out = ds.columns[model.outdoor][sl]
    n = len(t_out)
    u = (
        np.column_stack([ds.columns[i][sl] for i in model.inputs])
        if model.inputs
        else np.zeros((n, 0))
    )
    n_h = max(1, round(horizon_h / model.step_h))
    stride = max(1, round(every_h / model.step_h))
    sq = np.zeros((n_h, len(names)))
    cnt = np.zeros((n_h, len(names)))
    for s in range(0, n - n_h, stride):
        if np.isnan(temps[s]).any():
            continue
        w_out, w_u = t_out[s : s + n_h], u[s : s + n_h]
        if np.isnan(w_out).any() or np.isnan(w_u).any():
            continue
        err = model.simulate(temps[s], w_out, w_u)[1:] - temps[s + 1 : s + 1 + n_h]
        ok = ~np.isnan(err)
        sq[ok] += err[ok] ** 2
        cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(sq / cnt)
    return {
        room: [None if np.isnan(v) else float(v) for v in rmse[:, k]]
        for k, room in enumerate(names)
    }
