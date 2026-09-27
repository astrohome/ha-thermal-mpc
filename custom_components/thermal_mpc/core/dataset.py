"""A regular-grid, column-oriented training dataset that serialises to JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """Aligned columns sampled every ``step`` seconds from ``start``."""

    step: float
    start: float | None = None
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        """Number of time steps."""
        return len(next(iter(self.columns.values()))) if self.columns else 0

    @property
    def end(self) -> float | None:
        """Epoch seconds just after the last bin."""
        return None if self.start is None else self.start + self.rows * self.step

    def append(self, start: float, block: dict[str, np.ndarray]) -> None:
        """Append a block that begins exactly where the dataset ends.

        Columns absent from the block are padded with NaN; new columns are
        back-filled with NaN.
        """
        n_new = len(next(iter(block.values())))
        if self.start is None or self.rows == 0:
            self.start = start
            self.columns = {k: np.asarray(v, float) for k, v in block.items()}
            return
        if abs(start - self.end) > 1e-6:
            raise ValueError(f"block starts at {start}, dataset ends at {self.end}")
        n_old = self.rows
        for name in set(self.columns) | set(block):
            old = self.columns.get(name, np.full(n_old, np.nan))
            new = np.asarray(block.get(name, np.full(n_new, np.nan)), float)
            self.columns[name] = np.concatenate([old, new])

    def keep_columns(self, names: set[str]) -> None:
        """Drop columns that are no longer configured."""
        self.columns = {k: v for k, v in self.columns.items() if k in names}

    def trim(self, max_rows: int) -> None:
        """Keep only the most recent ``max_rows`` steps and drop empty leading rows."""
        if not self.columns:
            return
        drop = max(0, self.rows - max_rows)
        stacked = np.vstack(list(self.columns.values()))
        any_data = ~np.isnan(stacked).all(axis=0)
        drop = max(drop, int(np.argmax(any_data))) if any_data.any() else self.rows
        if drop:
            self.columns = {k: v[drop:] for k, v in self.columns.items()}
            self.start = None if self.rows == 0 else self.start + drop * self.step

    def coverage(self) -> dict[str, float]:
        """Fraction of non-NaN values per column."""
        return {
            k: float(np.mean(~np.isnan(v))) if len(v) else 0.0
            for k, v in self.columns.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise compactly (NaN as null, 3 decimals)."""
        return {
            "step": self.step,
            "start": self.start,
            "columns": {
                k: [None if np.isnan(x) else round(float(x), 3) for x in v]
                for k, v in self.columns.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        """Inverse of :meth:`to_dict`."""
        cols = {
            k: np.array([np.nan if x is None else x for x in v], dtype=float)
            for k, v in data.get("columns", {}).items()
        }
        return cls(step=data["step"], start=data.get("start"), columns=cols)
