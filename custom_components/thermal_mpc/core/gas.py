"""Gas meter as the heating input: units, smoothing and furnace capacity."""

from __future__ import annotations

import numpy as np

# kWh of heat per unit of the meter (natural gas at ~1037 BTU/ft³).
KWH_PER_UNIT: dict[str, float] = {
    "m³": 10.55,
    "ft³": 0.3039,
    "CCF": 30.39,
    "MCF": 303.9,
    "therm": 29.307,
    "Wh": 0.001,
    "kWh": 1.0,
    "MWh": 1000.0,
    "MJ": 1 / 3.6,
    "GJ": 277.8,
}
UNITS = list(KWH_PER_UNIT)
_ALIASES = {"m3": "m³", "ft3": "ft³", "cf": "ft³", "therms": "therm"}

SMOOTH_STEPS = 3  # centred moving average (15 min at 5-minute steps)
FULL_FIRE_DUTY = 0.8  # steps with at least this duty count as full fire
CAPACITY_PERCENTILE = 90
MIN_FULL_FIRE_H = 2.0


def normalise_unit(unit: str | None) -> str | None:
    """Map a unit string to a key of :data:`KWH_PER_UNIT` (None if unknown)."""
    if not unit:
        return None
    if unit in KWH_PER_UNIT:
        return unit
    low = unit.strip().lower()
    for known in KWH_PER_UNIT:
        if known.lower() == low:
            return known
    return _ALIASES.get(low)


def smooth(x: np.ndarray, width: int = SMOOTH_STEPS) -> np.ndarray:
    """Centred moving average ignoring NaN neighbours; NaN stays NaN.

    Counter reports are lumpy (a whole unit at a time), which looks like
    noise at 5-minute steps.
    """
    x = np.asarray(x, dtype=float)
    if width <= 1 or x.size == 0:
        return x.copy()
    half = width // 2
    valid = ~np.isnan(x)
    pad = np.concatenate([np.zeros(half), np.where(valid, x, 0.0), np.zeros(half)])
    cnt = np.concatenate([np.zeros(half), valid.astype(float), np.zeros(half)])
    kernel = np.ones(2 * half + 1)
    total = np.convolve(pad, kernel, mode="valid")
    n = np.convolve(cnt, kernel, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / n
    out[~valid] = np.nan
    return out


def capacity(gas_kw: np.ndarray, duty: np.ndarray, step_h: float) -> float | None:
    """Full-fire heat input (kW) of the furnace, or None if not seen enough.

    The high percentile of the (smoothed) gas rate over steps where the
    thermostat reports heating nearly all the time.
    """
    gas_kw = np.asarray(gas_kw, dtype=float)
    duty = np.asarray(duty, dtype=float)
    ok = ~np.isnan(gas_kw) & ~np.isnan(duty) & (duty >= FULL_FIRE_DUTY)
    if ok.sum() * step_h < MIN_FULL_FIRE_H:
        return None
    value = float(np.percentile(gas_kw[ok], CAPACITY_PERCENTILE))
    return value if value > 0 else None
