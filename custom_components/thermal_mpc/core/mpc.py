"""Model-predictive planning of heating / cooling over the next day.

The model is linear, so every room's temperature over the horizon is the
free-running forecast plus a response matrix times the hourly duty
decisions::

    T = T_free + Phi @ x,     x = hourly heating / cooling duty in [0, 1]

The planner minimises, per hour of horizon::

    w_comfort * (K below / above each room's comfort band)^2
  + w_spread  * variance of temperature across rooms
  + w_energy  * duty * price

with a projected accelerated gradient method (FISTA). Cooling is priced
lower while the solar forecast covers the AC's draw. The first hour's duty
is the recommendation; the rest shows the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import ThermalModel

HEAT, COOL = "heating", "cooling"


@dataclass
class PlanSettings:
    """What "good" means for the planner."""

    target: float = 21.0  # °C, centre of the comfort band
    band: float = 0.5  # ± K without penalty
    comfort_weight: float = 1.0  # per K² per hour
    spread_weight: float = 0.5  # per K² of cross-room variance per hour
    energy_weight: float = 0.15  # per hour of full-duty heating
    cooling_price: float = 1.0  # relative to heating, before solar
    ac_kw: float = 3.0  # AC draw used to judge solar coverage
    allow_heat: bool = True
    allow_cool: bool = False
    horizon_h: int = 24
    block_h: float = 1.0  # decision granularity
    iterations: int = 400


@dataclass
class PlanInputs:
    """Everything known about the future, on the model's step grid."""

    start: float  # epoch s of the first step
    temps: np.ndarray  # (R,) current air temperatures
    mass: np.ndarray  # (R,) current mass temperatures
    t_out: np.ndarray  # (N,)
    u: np.ndarray  # (N, I) baseline inputs; heating/cooling columns ignored
    solar_kw: np.ndarray | None = None  # (N,) for pricing cooling
    sources: dict[str, str] = field(default_factory=dict)  # provenance


def plan(
    model: ThermalModel,
    inputs: PlanInputs,
    settings: PlanSettings,
    heat_col: str | None,
    cool_col: str | None,
) -> dict[str, Any]:
    """Optimise hourly duties; return the plan and predicted trajectories."""
    n = len(inputs.t_out)
    step_h = model.step_h
    per_block = max(1, round(settings.block_h / step_h))
    n_blocks = int(np.ceil(n / per_block))
    rooms = list(model.rooms)
    n_rooms = len(rooms)

    decisions: list[tuple[str, int]] = []
    if settings.allow_heat and heat_col in model.inputs:
        decisions += [(heat_col, b) for b in range(n_blocks)]
    if settings.allow_cool and cool_col in model.inputs:
        decisions += [(cool_col, b) for b in range(n_blocks)]

    base_u = inputs.u.copy()
    for col in (heat_col, cool_col):
        if col in model.inputs:
            base_u[:, model.inputs.index(col)] = 0.0

    # Scenario 0: free running; scenario d+1: unit duty in decision d.
    scen = np.repeat(base_u[None], len(decisions) + 1, axis=0)
    for d, (col, b) in enumerate(decisions):
        scen[d + 1, b * per_block : (b + 1) * per_block, model.inputs.index(col)] = 1.0
    x0 = np.repeat(inputs.temps[None], len(scen), axis=0)
    m0 = np.repeat(inputs.mass[None], len(scen), axis=0)
    t_out = np.repeat(inputs.t_out[None], len(scen), axis=0)
    sims = model.simulate_many(x0, m0, t_out, scen)  # (S, N, R)
    free = sims[0]
    phi = (sims[1:] - free[None]).reshape(len(decisions), -1).T  # (N*R, D)

    lo = settings.target - settings.band
    hi = settings.target + settings.band
    dt = step_h
    w_c, w_s = settings.comfort_weight, settings.spread_weight

    # Energy price per decision (per hour of full duty).
    cost = np.zeros(len(decisions))
    for d, (col, b) in enumerate(decisions):
        hours = min(per_block, n - b * per_block) * step_h
        price = 1.0
        if col == cool_col:
            price = settings.cooling_price
            if inputs.solar_kw is not None and settings.ac_kw > 0:
                sl = inputs.solar_kw[b * per_block : (b + 1) * per_block]
                covered = float(np.clip(np.mean(sl) / settings.ac_kw, 0.0, 1.0))
                price *= 1.0 - 0.8 * covered
        cost[d] = settings.energy_weight * price * hours

    def trajectory(x: np.ndarray) -> np.ndarray:
        return (free.ravel() + phi @ x).reshape(n, n_rooms) if x.size else free

    def gradient(x: np.ndarray) -> np.ndarray:
        t = trajectory(x)
        g_t = 2 * w_c * dt * (np.maximum(t - hi, 0) - np.maximum(lo - t, 0))
        g_t += 2 * w_s * dt * (t - t.mean(axis=1, keepdims=True)) / n_rooms
        return phi.T @ g_t.ravel() + cost

    x = np.zeros(len(decisions))
    if decisions:
        # Lipschitz bound of the smooth part -> step size.
        lip = 2 * dt * (w_c + w_s) * np.linalg.norm(phi, 2) ** 2 + 1e-9
        y, x_prev, tk = x.copy(), x.copy(), 1.0
        for _ in range(settings.iterations):
            x_new = np.clip(y - gradient(y) / lip, 0.0, 1.0)
            t_next = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            y = x_new + (tk - 1) / t_next * (x_new - x_prev)
            x_prev, x, tk = x_new, x_new, t_next

    planned = trajectory(x)
    duty = {HEAT: np.zeros(n_blocks), COOL: np.zeros(n_blocks)}
    for d, (col, b) in enumerate(decisions):
        duty[HEAT if col == heat_col else COOL][b] = x[d]

    def comfort_kh(t: np.ndarray) -> float:
        return float((np.maximum(lo - t, 0) + np.maximum(t - hi, 0)).sum() * dt)

    now_heat, now_cool = float(duty[HEAT][0]), float(duty[COOL][0])
    action = "idle"
    if now_heat >= 0.5 and now_heat >= now_cool:
        action = "heat"
    elif now_cool >= 0.5:
        action = "cool"
    times = [inputs.start + k * step_h * 3600 for k in range(n)]
    return {
        "action": action,
        "duty_now": {HEAT: now_heat, COOL: now_cool},
        "block_hours": settings.block_h,
        "duty": {k: [round(float(v), 3) for v in arr] for k, arr in duty.items()},
        "heating_hours": float(duty[HEAT].sum() * settings.block_h),
        "cooling_hours": float(duty[COOL].sum() * settings.block_h),
        "target": settings.target,
        "band": settings.band,
        "times": times,
        "t_out": [round(float(v), 2) for v in inputs.t_out],
        "planned": {
            r: [round(float(v), 3) for v in planned[:, k]] for k, r in enumerate(rooms)
        },
        "free": {
            r: [round(float(v), 3) for v in free[:, k]] for k, r in enumerate(rooms)
        },
        "discomfort_kh": {"planned": comfort_kh(planned), "free": comfort_kh(free)},
        "spread_k": float(np.mean(planned.std(axis=1))) if n_rooms > 1 else 0.0,
        "sources": inputs.sources,
    }
