"""Combine several temperature sensors in one room into one room temperature.

Each sensor ``s`` in a room is modelled as::

    x_s(t) = T_room(t) + bias_s + sun_s * S(t) + noise_s

where ``S`` is solar power smoothed over about an hour (a sensor in a sunny
window or plant pot warms up with a lag). Per sensor we learn the bias, the
sun exposure and the noise level by comparing it with the other sensors in
the same room (alternating least squares). Biases are relative (weighted
mean zero: the room level is what the average sensor says); sun exposure is
measured from the least exposed sensor, taken as shaded, since sunlight can
only warm a sensor.

The fused room temperature is the inverse-variance weighted mean of the
corrected readings that are available at each step, so a sensor dropping
out does not break the room series.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .dataset import Dataset

SUN_TAU_H = 1.0
MIN_OVERLAP_ROWS = 288  # one day of 5-minute steps with >= 2 sensors
MIN_SIGMA = 0.02  # K; floor so one very smooth sensor cannot take all weight
PASSES = 25


@dataclass
class SensorCal:
    """Learned calibration of one sensor."""

    bias: float = 0.0  # K, reads this much high
    sun: float = 0.0  # K per kW of (smoothed) solar power, relative
    sigma: float = 0.1  # K, noise vs. the room's other sensors

    @property
    def weight(self) -> float:
        """Inverse-variance weight."""
        return 1.0 / max(self.sigma, MIN_SIGMA) ** 2


Zones = dict[str, list[str]]  # zone column -> sensor columns
Fusion = dict[str, dict[str, SensorCal]]


def smooth_sun(solar: np.ndarray | None, step_h: float, n: int) -> np.ndarray:
    """Solar power low-passed with ``SUN_TAU_H`` (zeros when unknown)."""
    if solar is None:
        return np.zeros(n)
    a = min(1.0, step_h / SUN_TAU_H)
    x = np.where(np.isnan(solar), 0.0, solar)
    out = np.empty(n)
    f = 0.0
    for t in range(n):
        f += a * (x[t] - f)
        out[t] = f
    return out


def _corrected(x: np.ndarray, cal: SensorCal, sun: np.ndarray) -> np.ndarray:
    return x - cal.bias - cal.sun * sun


def _fuse(xs: np.ndarray, cals: list[SensorCal], sun: np.ndarray) -> np.ndarray:
    """Weighted mean of available corrected readings; NaN if none."""
    corr = np.column_stack([_corrected(xs[:, j], c, sun) for j, c in enumerate(cals)])
    w = np.array([c.weight for c in cals])
    ok = ~np.isnan(corr)
    num = np.where(ok, corr, 0.0) @ w
    den = ok @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def learn(
    ds: Dataset, zones: Zones, solar: str | None = None, rows: slice | None = None
) -> Fusion:
    """Learn per-sensor calibrations for every zone with >= 2 sensors."""
    sl = rows or slice(None)
    step_h = ds.step / 3600.0
    sol = ds.columns.get(solar) if solar else None
    sun_all = smooth_sun(sol, step_h, ds.rows)[sl]
    out: Fusion = {}
    for zone, sensors in zones.items():
        present = [s for s in sensors if s in ds.columns]
        cals = {s: SensorCal() for s in sensors}
        if len(present) < 2:
            out[zone] = cals
            continue
        xs = np.column_stack([ds.columns[s][sl] for s in present])
        ok = ~np.isnan(xs)
        if ((ok.sum(axis=1)) >= 2).sum() < MIN_OVERLAP_ROWS:
            out[zone] = cals
            continue
        use_sun = sol is not None and np.nanstd(sun_all) > 1e-6
        rows_ok = ok.sum(axis=1) >= 2
        # Alternating least squares: room temperature <- corrected sensors,
        # then each sensor's bias / sun term <- its residual to the room.
        for _ in range(PASSES):
            current = [cals[s] for s in present]
            room = _fuse(xs, current, sun_all)
            room[~rows_ok] = np.nan
            for j, s in enumerate(present):
                resid = xs[:, j] - room
                m = ~np.isnan(resid)
                if m.sum() < MIN_OVERLAP_ROWS:
                    continue
                if use_sun:
                    A = np.column_stack([np.ones(m.sum()), sun_all[m]])
                    coef, *_ = np.linalg.lstsq(A, resid[m], rcond=None)
                    bias, sun = float(coef[0]), float(coef[1])
                else:
                    bias, sun = float(np.mean(resid[m])), 0.0
                # The room estimate is already corrected, so this is the
                # sensor's total offset, not an increment.
                cals[s] = SensorCal(bias=bias, sun=sun, sigma=current[j].sigma)
            # Relative calibration: weighted-mean bias and sun exposure are 0.
            w = np.array([cals[s].weight for s in present])
            mb = float(np.average([cals[s].bias for s in present], weights=w))
            # Sun can only warm a sensor: treat the least exposed as shaded.
            ms = float(min(cals[s].sun for s in present))
            for s in present:
                c = cals[s]
                cals[s] = SensorCal(bias=c.bias - mb, sun=c.sun - ms, sigma=c.sigma)
            # Noise from leave-one-out residuals, so no sensor vouches for itself.
            current = [cals[s] for s in present]
            for j, s in enumerate(present):
                others = [k for k in range(len(present)) if k != j]
                ref = _fuse(xs[:, others], [current[k] for k in others], sun_all)
                resid = _corrected(xs[:, j], current[j], sun_all) - ref
                m = ~np.isnan(resid)
                if m.sum() >= MIN_OVERLAP_ROWS:
                    rest = resid[m] - np.mean(resid[m])
                    cals[s] = SensorCal(
                        bias=current[j].bias,
                        sun=current[j].sun,
                        sigma=float(np.std(rest)),
                    )
        out[zone] = cals
    return out


def fuse(
    ds: Dataset, zones: Zones, fusion: Fusion, solar: str | None = None
) -> dict[str, np.ndarray]:
    """Fused temperature series for every zone over the whole dataset."""
    step_h = ds.step / 3600.0
    sol = ds.columns.get(solar) if solar else None
    sun = smooth_sun(sol, step_h, ds.rows)
    out = {}
    for zone, sensors in zones.items():
        present = [s for s in sensors if s in ds.columns]
        if not present:
            out[zone] = np.full(ds.rows, np.nan)
            continue
        cals = [fusion.get(zone, {}).get(s, SensorCal()) for s in present]
        xs = np.column_stack([ds.columns[s] for s in present])
        out[zone] = _fuse(xs, cals, sun)
    return out


def fuse_now(
    values: dict[str, float], zones: Zones, fusion: Fusion, sun_now: float
) -> dict[str, float]:
    """Fused zone temperatures from single current readings."""
    out = {}
    sun = np.array([sun_now])
    for zone, sensors in zones.items():
        cals = [fusion.get(zone, {}).get(s, SensorCal()) for s in sensors]
        xs = np.array([[values.get(s, np.nan) for s in sensors]], dtype=float)
        out[zone] = float(_fuse(xs, cals, sun)[0])
    return out


def view(ds: Dataset, zones: Zones, fusion: Fusion, solar: str | None) -> Dataset:
    """Dataset with a fused column per zone added (raw columns kept)."""
    cols = dict(ds.columns)
    cols.update(fuse(ds, zones, fusion, solar))
    return Dataset(step=ds.step, start=ds.start, columns=cols)


def to_dict(fusion: Fusion) -> dict[str, Any]:
    """JSON-ready form."""
    return {z: {s: asdict(c) for s, c in cals.items()} for z, cals in fusion.items()}


def from_dict(data: dict[str, Any] | None) -> Fusion:
    """Inverse of :func:`to_dict`."""
    return {
        z: {s: SensorCal(**c) for s, c in cals.items()}
        for z, cals in (data or {}).items()
    }
