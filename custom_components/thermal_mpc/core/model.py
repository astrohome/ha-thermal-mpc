"""Grey-box multi-room RC thermal model.

Each room ``i`` is a lumped heat capacity connected to its neighbours and to
the outdoors through thermal conductances::

    dT_i/dt = sum_j g_ij (T_j - T_i) + g_io (T_out - T_i) + h_i (M_i - T_i)
              + sum_u b_iu u + c_i
    dM_i/dt = k_i (T_i - M_i)

``M_i`` is an optional hidden thermal-mass temperature (walls, floor,
furniture) that stores heat and gives it back slowly; see :mod:`.identify`.

Every coefficient is already divided by the room's heat capacity, so
conductances are in 1/h and gains in K/h per unit of input. ``1 / g_io`` is
the room's time constant against the outdoors, a direct measure of thermal
inertia. ``c_i`` absorbs constant internal gains and sensor offsets.

Signs encode physics: conductances and heating gains are non-negative,
cooling gains are non-positive.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .dataset import Dataset
from .identify import (
    MASS_TAU_GRID_H,
    Layout,
    Segments,
    integral_fit,
    observe_mass,
    open_loop_error,
    refine,
)
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
    mass_inputs: tuple[str, ...] = ()  # inputs that also heat mass (sun)

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
    mass_h: float = 0.0  # air <- mass coupling (1/h); 0 = no mass node
    mass_k: float = 0.0  # mass <- air rate (1/h)
    mass_gains: dict[str, float] = field(default_factory=dict)  # K/h per unit

    @property
    def tau_out_h(self) -> float | None:
        """Time constant against outdoors in hours (None if uncoupled)."""
        return 1.0 / self.g_out if self.g_out > 1e-9 else None

    @property
    def tau_mass_h(self) -> float | None:
        """How fast the thermal mass follows the air, in hours."""
        return 1.0 / self.mass_k if self.mass_h > 1e-9 and self.mass_k > 0 else None


@dataclass
class ThermalModel:
    """A fitted multi-room model."""

    step_h: float
    outdoor: str
    rooms: dict[str, RoomParams]
    inputs: list[str]
    unused_inputs: list[str] = field(default_factory=list)
    fusion: dict[str, Any] = field(default_factory=dict)  # sensor calibration
    # Full-duty level of scaled inputs (e.g. furnace kW for a gas meter).
    input_capacity: dict[str, float] = field(default_factory=dict)

    def derivative(
        self,
        temps: np.ndarray,
        t_out: float,
        u: np.ndarray,
        mass: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return air dT/dt (K/h) for all rooms (``mass`` defaults to ``temps``)."""
        names = list(self.rooms)
        idx = {n: k for k, n in enumerate(names)}
        mass = temps if mass is None else mass
        d = np.empty(len(names))
        for k, name in enumerate(names):
            p = self.rooms[name]
            val = p.g_out * (t_out - temps[k]) + p.offset
            val += p.mass_h * (mass[k] - temps[k])
            for other, g in p.g_rooms.items():
                val += g * (temps[idx[other]] - temps[k])
            for j, inp in enumerate(self.inputs):
                val += p.gains.get(inp, 0.0) * u[j]
            d[k] = val
        return d

    @property
    def mass_k(self) -> np.ndarray:
        """Mass rates per room (0 where there is no mass node)."""
        return np.array(
            [p.mass_k if p.mass_h > 1e-9 else 0.0 for p in self.rooms.values()]
        )

    @property
    def mass_inputs(self) -> list[str]:
        """Inputs that heat the thermal mass directly."""
        names = {q for p in self.rooms.values() for q in p.mass_gains}
        return [i for i in self.inputs if i in names]

    def _mass_gain_matrix(self) -> np.ndarray:
        return np.array(
            [
                [p.mass_gains.get(q, 0.0) for q in self.mass_inputs]
                for p in self.rooms.values()
            ]
        ).reshape(len(self.rooms), len(self.mass_inputs))

    def observe_mass(
        self, temps: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        """Hidden mass temperatures implied by measured signals (n, R).

        ``u`` holds the model inputs (columns in ``self.inputs`` order).
        """
        qi = [self.inputs.index(q) for q in self.mass_inputs]
        u_mass = u[:, qi] if u is not None and qi else None
        return observe_mass(
            temps, self.mass_k, self.step_h, u_mass, self._mass_gain_matrix()
        )

    def matrices(self) -> tuple[np.ndarray, ...]:
        """Return ``(A, g_out, G, c, h, k, S)`` as in :meth:`Layout.matrices`."""
        names = list(self.rooms)
        idx = {r: j for j, r in enumerate(names)}
        n = len(names)
        A = np.zeros((n, n))
        g_out, c, h = np.zeros(n), np.zeros(n), np.zeros(n)
        G = np.zeros((n, len(self.inputs)))
        for j, p in enumerate(self.rooms.values()):
            g_out[j] = p.g_out
            A[j, j] -= p.g_out + p.mass_h
            h[j] = p.mass_h
            for o, g in p.g_rooms.items():
                A[j, idx[o]] += g
                A[j, j] -= g
            G[j] = [p.gains.get(i, 0.0) for i in self.inputs]
            c[j] = p.offset
        return A, g_out, G, c, h, self.mass_k, self._mass_gain_matrix()

    def simulate_many(
        self,
        x0: np.ndarray,
        m0: np.ndarray,
        t_out: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Vectorised :meth:`simulate` for S scenarios.

        ``x0``/``m0`` (S, R); ``t_out`` (S, N); ``u`` (S, N, I). Returns air
        temperatures after each step, (S, N, R).
        """
        A, g_out, G, c, h, k, Sm = self.matrices()
        qi = [self.inputs.index(q) for q in self.mass_inputs]
        n_s, n = t_out.shape
        out = np.empty((n_s, n, x0.shape[1]))
        x, m = x0.astype(float), m0.astype(float)
        for t in range(n):
            us = u[:, t, :]
            dx = x @ A.T + t_out[:, t, None] * g_out + h * m + us @ G.T + c
            dm = k * (x - m) + (us[:, qi] @ Sm.T if qi else 0.0)
            x = x + self.step_h * dx
            m = m + self.step_h * dm
            out[:, t] = x
        return out

    def simulate(
        self,
        t0: np.ndarray,
        t_out: np.ndarray,
        u: np.ndarray,
        m0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Roll forward from air ``t0`` (and mass ``m0``, default ``t0``).

        Returns air temperatures, shape (N + 1, n_rooms).
        """
        k = self.mass_k
        S = self._mass_gain_matrix()
        qi = [self.inputs.index(q) for q in self.mass_inputs]
        out = np.empty((len(t_out) + 1, len(t0)))
        out[0] = t0
        m = np.array(t0 if m0 is None else m0, dtype=float)
        for s in range(len(t_out)):
            x = out[s]
            out[s + 1] = x + self.step_h * self.derivative(x, t_out[s], u[s], m)
            drive = S @ u[s][qi] if qi else 0.0
            m = m + self.step_h * (k * (x - m) + drive)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible data."""
        return {
            "step_h": self.step_h,
            "outdoor": self.outdoor,
            "inputs": self.inputs,
            "unused_inputs": self.unused_inputs,
            "fusion": self.fusion,
            "input_capacity": self.input_capacity,
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
            fusion=data.get("fusion", {}),
            input_capacity=data.get("input_capacity", {}),
            rooms={k: RoomParams(**v) for k, v in data["rooms"].items()},
        )


def fit(
    ds: Dataset,
    spec: ModelSpec,
    rows: slice | None = None,
    window_h: float = 1.0,
    horizon_h: float | None = 6.0,
    mass_grid_h: tuple[float | None, ...] = MASS_TAU_GRID_H,
) -> ThermalModel:
    """Fit all rooms on ``ds`` (optionally only ``rows`` of it).

    Stage 1 regresses ``window_h``-hour temperature changes (see
    :mod:`.identify`); stage 2 refines on ``horizon_h``-hour open-loop
    prediction error. ``horizon_h=None`` skips stage 2. ``mass_grid_h``
    lists the thermal-mass time constants to try (``None`` = no mass node);
    pass ``(None,)`` for a plain one-node-per-room model.
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
    horizon = max(1, round((horizon_h or 6.0) / step_h))
    # About one segment per hour, capped so a long history stays fast.
    stride = max(round(1 / step_h), n // 1500)
    seg = Segments.build(temps, t_out, u, horizon, stride)

    # Stage 1 for each candidate mass time constant (same for every room).
    # Refine the two that predict the training data best open loop, keep the
    # better one.
    candidates = []
    for tau in mass_grid_h:
        cand = Layout(
            spec.rooms,
            nbrs,
            inputs,
            dict(spec.inputs),
            [tau is not None] * len(spec.rooms),
            [q for q in spec.mass_inputs if q in inputs],
        )
        k = np.full(len(spec.rooms), 0.0 if tau is None else 1.0 / tau)
        theta, used = integral_fit(temps, t_out, u, cand, step_h, window, k)
        for (room, a, b), n_used in zip(cand.blocks(), used, strict=True):
            n_par = b - a - (0 if tau is not None else 1 + len(cand.mass_inputs))
            if n_used < SAMPLES_PER_PARAM * n_par:
                raise NotEnoughDataError(
                    f"{room}: {n_used} complete windows for {n_par} parameters"
                )
        err = (
            open_loop_error(theta, cand, temps, u, seg, step_h)
            if seg.starts.size
            else 0.0
        )
        candidates.append((err, theta, cand, used))
    candidates.sort(key=lambda c: c[0])
    if horizon_h:
        refined = []
        for _, theta, cand, used in candidates[:2]:
            theta, err = refine(theta, cand, temps, t_out, u, step_h, horizon, stride)
            refined.append((err if np.isfinite(err) else np.inf, theta, cand, used))
        candidates = sorted(refined, key=lambda c: c[0])
    _, theta, layout, used = candidates[0]

    # One-step residuals of the final model, for reference.
    A, g_out, G, c, h, kk, Sm = layout.matrices(theta)
    u_mass = u[:, layout.mass_input_idx] if layout.mass_inputs else None
    mass = observe_mass(temps, kk, step_h, u_mass, Sm)
    pred = temps[:-1] + step_h * (
        temps[:-1] @ A.T + t_out[:-1, None] * g_out + h * mass[:-1] + u[:-1] @ G.T + c
    )
    resid = pred - temps[1:]
    rooms = {}
    for k, (room, a, _) in enumerate(layout.blocks()):
        ok = ~np.isnan(resid[:, k])
        nb = nbrs[room]
        b = a + 1 + len(nb) + 1 + len(layout.mass_inputs)  # skip mass params
        rooms[room] = RoomParams(
            g_out=float(theta[a]),
            g_rooms={o: float(theta[a + 1 + j]) for j, o in enumerate(nb)},
            gains={i: float(theta[b + j]) for j, i in enumerate(inputs)},
            offset=float(theta[b + len(inputs)]),
            mass_h=float(h[k]),
            mass_k=float(kk[k]),
            mass_gains={q: float(Sm[k, j]) for j, q in enumerate(layout.mass_inputs)}
            if h[k] > 1e-9
            else {},
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
    a good one-step fit can still drift badly over hours. Hidden mass states
    are reconstructed from the *whole* dataset, so ``rows`` late in the data
    start from a settled estimate.
    """
    sl = rows or slice(None)
    names = list(model.rooms)
    all_temps = np.column_stack([ds.columns[r] for r in names])
    all_u = (
        np.column_stack([ds.columns[i] for i in model.inputs])
        if model.inputs
        else np.zeros((ds.rows, 0))
    )
    mass = model.observe_mass(all_temps, all_u)[sl]
    temps = all_temps[sl]
    u = all_u[sl]
    t_out = ds.columns[model.outdoor][sl]
    n = len(t_out)
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
        sim = model.simulate(temps[s], w_out, w_u, mass[s])
        err = sim[1:] - temps[s + 1 : s + 1 + n_h]
        ok = ~np.isnan(err)
        sq[ok] += err[ok] ** 2
        cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(sq / cnt)
    return {
        room: [None if np.isnan(v) else float(v) for v in rmse[:, k]]
        for k, room in enumerate(names)
    }
