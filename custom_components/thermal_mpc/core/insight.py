"""Explain a fitted model: per-room heat budgets and model-vs-measured replays."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import Dataset
from .model import ThermalModel

OUTDOOR = "outdoor"
OFFSET = "offset"


def budget(
    model: ThermalModel,
    temps: dict[str, float],
    t_out: float,
    inputs: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Split each room's dT/dt (K/h) into the paths that cause it.

    Returns ``{room: {"outdoor": x, "rooms": {other: x}, "inputs": {name: x},
    "offset": x, "net": x}}``. Positive terms warm the room, negative cool it.
    Terms whose driving value is unknown (NaN) come out as None.
    """
    out: dict[str, dict[str, Any]] = {}
    for room, p in model.rooms.items():
        t_i = temps.get(room, np.nan)
        terms_out = p.g_out * (t_out - t_i)
        rooms = {o: g * (temps.get(o, np.nan) - t_i) for o, g in p.g_rooms.items()}
        ins = {i: p.gains.get(i, 0.0) * inputs.get(i, np.nan) for i in model.inputs}
        values = [terms_out, *rooms.values(), *ins.values(), p.offset]
        known = [v for v in values if not np.isnan(v)]
        out[room] = {
            OUTDOOR: _num(terms_out),
            "rooms": {k: _num(v) for k, v in rooms.items()},
            "inputs": {k: _num(v) for k, v in ins.items()},
            OFFSET: _num(p.offset),
            "net": _num(sum(known)) if len(known) == len(values) else None,
        }
    return out


def mean_budget(
    model: ThermalModel, ds: Dataset, rows: slice
) -> dict[str, dict[str, Any]]:
    """Average of :func:`budget` over ``rows`` where every term is known."""
    names = list(model.rooms)
    cols = {k: ds.columns[k][rows] for k in [*names, model.outdoor, *model.inputs]}
    stacked = np.vstack(list(cols.values()))
    ok = ~np.isnan(stacked).any(axis=0)
    if not ok.any():
        return {}
    mean = {k: v[ok] for k, v in cols.items()}
    out: dict[str, dict[str, Any]] = {}
    for room, p in model.rooms.items():
        t_i = mean[room]
        terms_out = float(np.mean(p.g_out * (mean[model.outdoor] - t_i)))
        rooms = {o: float(np.mean(g * (mean[o] - t_i))) for o, g in p.g_rooms.items()}
        ins = {i: float(np.mean(p.gains.get(i, 0.0) * mean[i])) for i in model.inputs}
        net = terms_out + sum(rooms.values()) + sum(ins.values()) + p.offset
        out[room] = {
            OUTDOOR: terms_out,
            "rooms": rooms,
            "inputs": ins,
            OFFSET: p.offset,
            "net": net,
            "samples": int(ok.sum()),
        }
    return out


def replay(
    model: ThermalModel, ds: Dataset, hours: float = 48.0, segment_h: float = 6.0
) -> dict[str, Any]:
    """Model predictions over the last ``hours``, restarted every ``segment_h``.

    Each segment starts from the measured temperatures and then runs open
    loop with measured outdoor temperature and inputs, which is how the model
    will be used for control. Returns epoch-second timestamps, measured and
    predicted temperature per room (None where unavailable).
    """
    n = min(ds.rows, round(hours / model.step_h))
    if n == 0 or ds.start is None:
        return {"times": [], "measured": {}, "predicted": {}}
    sl = slice(ds.rows - n, ds.rows)
    names = list(model.rooms)
    temps = np.column_stack([ds.columns[r][sl] for r in names])
    t_out = ds.columns[model.outdoor][sl]
    u = (
        np.column_stack([ds.columns[i][sl] for i in model.inputs])
        if model.inputs
        else np.zeros((n, 0))
    )
    pred = np.full_like(temps, np.nan)
    seg = max(1, round(segment_h / model.step_h))
    for s in range(0, n, seg):
        e = min(n, s + seg)
        if np.isnan(temps[s]).any():
            continue
        # Stop a segment at the first unknown input rather than guess.
        bad = np.isnan(t_out[s:e]) | np.isnan(u[s:e]).any(axis=1)
        stop = s + (int(np.argmax(bad)) if bad.any() else e - s)
        if stop - s < 1:
            continue
        sim = model.simulate(temps[s], t_out[s:stop], u[s:stop])
        pred[s:stop] = sim[:-1]
    start = ds.start + (ds.rows - n) * ds.step
    return {
        "times": [start + k * ds.step for k in range(n)],
        "segment_hours": segment_h,
        "measured": {r: [_num(v) for v in temps[:, k]] for k, r in enumerate(names)},
        "predicted": {r: [_num(v) for v in pred[:, k]] for k, r in enumerate(names)},
    }


def _num(v: float) -> float | None:
    return None if v is None or np.isnan(v) else round(float(v), 4)
