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
MIN_COVERAGE = 0.5  # share of usable rows with a gas rate to model gas
# Above this a 5-minute rate is not a house furnace (a counter misread).
MAX_RATE_KW = 150.0
# Full-fire heat input of a house furnace (about 10k to 200k BTU/h).
PLAUSIBLE_CAPACITY_KW = (3.0, 60.0)


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


def rescale(gas_kw: np.ndarray, ratio: float) -> np.ndarray:
    """Convert a stored gas rate to a new meter unit (``ratio`` = new/old kWh).

    The rate is proportional to the unit's kWh, so the conversion is exact.
    A meter that changed what it counts (raw ft³ under a "CCF" label, then
    true CCF) leaves the older part off by the ratio after the change: rates
    no house furnace reaches are dropped rather than kept mis-scaled.
    """
    out = np.asarray(gas_kw, dtype=float) * ratio
    with np.errstate(invalid="ignore"):
        out[out > MAX_RATE_KW] = np.nan
    return out


def coverage(gas_kw: np.ndarray, usable: np.ndarray) -> float:
    """Share of ``usable`` rows (temperatures known) that have a gas rate."""
    usable = np.asarray(usable, dtype=bool)
    if not usable.any():
        return 0.0
    return float((~np.isnan(np.asarray(gas_kw, dtype=float)[usable])).mean())


def capacity_warning(value: float, unit: str | None) -> str | None:
    """Explanation if ``value`` (kW) is not a plausible house furnace."""
    low, high = PLAUSIBLE_CAPACITY_KW
    if low <= value <= high:
        return None
    return (
        f"Furnace capacity {value:.2f} kW is outside {low:g}-{high:g} kW: "
        f"check the gas meter unit (now {unit or 'unknown'})"
    )
