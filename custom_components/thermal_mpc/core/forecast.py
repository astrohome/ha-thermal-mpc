"""Turn weather / solar forecasts into model inputs on the 5-minute grid."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np


def interpolate(
    points: Iterable[tuple[float, float]], times: np.ndarray
) -> np.ndarray | None:
    """Linearly interpolate (epoch s, value) points onto ``times``.

    Held constant beyond the ends. None when there are no usable points.
    """
    pts = sorted((t, v) for t, v in points if v is not None and np.isfinite(v))
    if not pts:
        return None
    t, v = np.array(pts).T
    return np.interp(times, t, v)


def hourly_energy_to_kw(
    wh_hours: dict[str, float], times: np.ndarray
) -> np.ndarray | None:
    """Forecast.Solar style ``{iso_hour: Wh}`` -> average kW on ``times``.

    Each value is the energy for the hour starting at its timestamp, so the
    average power over that hour is Wh / 1000.
    """
    if not wh_hours:
        return None
    starts, kw = [], []
    for iso, wh in wh_hours.items():
        try:
            starts.append(datetime.fromisoformat(iso).timestamp())
        except (TypeError, ValueError):
            continue
        kw.append(float(wh) / 1000.0)
    if not starts:
        return None
    order = np.argsort(starts)
    starts = np.array(starts)[order]
    kw = np.array(kw)[order]
    idx = np.searchsorted(starts, times, side="right") - 1
    out = np.where(idx >= 0, kw[np.clip(idx, 0, None)], 0.0)
    # Nothing known after the last hour: assume night-like zero production.
    out[times >= starts[-1] + 3600] = 0.0
    return out


def persistence(history: np.ndarray, steps_per_day: int, n: int) -> np.ndarray:
    """Repeat the last day of ``history`` (NaNs filled) for ``n`` steps."""
    day = np.asarray(history[-steps_per_day:], dtype=float)
    if day.size == 0 or np.isnan(day).all():
        return np.zeros(n)
    day = np.where(np.isnan(day), np.nanmean(day), day)
    reps = int(np.ceil(n / day.size))
    return np.tile(day, reps)[:n]
