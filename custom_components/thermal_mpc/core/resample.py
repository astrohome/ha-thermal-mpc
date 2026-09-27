"""Turn irregular state changes into a regular time grid."""

from __future__ import annotations

import numpy as np


def time_weighted_mean(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    n_bins: int,
    step: float,
    min_coverage: float = 0.5,
) -> np.ndarray:
    """Average a zero-order-hold signal over bins ``[start + k*step, +step)``.

    ``times`` are epoch seconds of state changes and each value holds until the
    next change, which is how the recorder stores states. Averaging over time
    (rather than sampling) turns on/off signals into duty cycles. NaN values
    (unavailable/unknown) are excluded, and bins where less than
    ``min_coverage`` of the time has a valid value come out as NaN.
    """
    if n_bins <= 0:
        return np.empty(0)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    # Keep the last of duplicate timestamps.
    if len(times):
        keep = np.append(times[1:] != times[:-1], True)
        times, values = times[keep], values[keep]

    end = start + n_bins * step
    edges = start + step * np.arange(n_bins)
    inside = times[(times > start) & (times < end)]
    t = np.union1d(np.union1d(inside, edges), [end])
    # Value in force at each breakpoint: last change at or before it.
    pos = np.searchsorted(times, t, side="right") - 1
    held = np.full(len(t), np.nan)
    ok = pos >= 0
    held[ok] = values[pos[ok]]

    dur = np.diff(t)
    held = held[:-1]
    bin_idx = np.minimum(((t[:-1] - start) // step).astype(int), n_bins - 1)
    valid = ~np.isnan(held)
    weighted = np.bincount(
        bin_idx[valid], weights=held[valid] * dur[valid], minlength=n_bins
    )
    covered = np.bincount(bin_idx[valid], weights=dur[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = weighted / covered
    mean[covered < min_coverage * step] = np.nan
    return mean
