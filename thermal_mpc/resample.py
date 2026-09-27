"""Turn irregular HA state changes into a regular time grid."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_weighted_mean(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step: pd.Timedelta,
    min_coverage: float = 0.5,
) -> pd.Series:
    """Average a zero-order-hold signal over each ``[t, t + step)`` bin.

    HA records a row only when a value changes, and each value holds until the
    next row. Averaging over time (rather than sampling) turns on/off signals
    into duty cycles and keeps energy-like inputs unbiased. NaN values
    (unavailable/unknown) are excluded; bins where less than ``min_coverage``
    of the time has a valid value are NaN.
    """
    edges = pd.date_range(start, end, freq=step, inclusive="left").as_unit("ns")
    if len(edges) == 0:
        return pd.Series(dtype=float)
    # Integer arithmetic below needs one resolution (pandas 3 may infer "us").
    series = series.copy()
    series.index = pd.DatetimeIndex(series.index).as_unit("ns")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    # Breakpoints: every change inside the window plus every bin edge.
    points = series.index[(series.index > start) & (series.index < end)]
    t = points.union(edges).union(pd.DatetimeIndex([end]).as_unit("ns"))
    # Value in force at each breakpoint (last change at or before it).
    pos = series.index.searchsorted(t, side="right") - 1
    values = np.where(
        pos >= 0, series.to_numpy(dtype=float)[np.maximum(pos, 0)], np.nan
    )

    t_ns = t.asi8
    dur = np.diff(t_ns).astype(float)  # duration each value holds
    values = values[:-1]
    bin_idx = np.searchsorted(edges.asi8, t_ns[:-1], side="right") - 1

    valid = ~np.isnan(values)
    n = len(edges)
    weighted = np.bincount(
        bin_idx[valid], weights=values[valid] * dur[valid], minlength=n
    )
    covered = np.bincount(bin_idx[valid], weights=dur[valid], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = weighted / covered
    mean[covered < min_coverage * step.value] = np.nan
    return pd.Series(mean, index=edges)
