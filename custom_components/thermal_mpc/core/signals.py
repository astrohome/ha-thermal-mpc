"""Mapping of Home Assistant states to numbers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

UNUSABLE = ("unavailable", "unknown", "", None)


@dataclass(frozen=True, slots=True)
class Signal:
    """How to turn one entity (state or attribute) into a numeric series."""

    entity_id: str
    attribute: str | None = None
    mapping: Mapping[str, float] | None = None
    default: float | None = None
    scale: float = 1.0
    counter: bool = False  # cumulative meter: resample as a rate per hour

    def value(self, state: str | None, attributes: Mapping[str, Any]) -> float:
        """Convert one recorded state to a number (NaN when unusable)."""
        if state in UNUSABLE:
            return float("nan")
        raw = attributes.get(self.attribute) if self.attribute else state
        if raw in UNUSABLE:
            return float("nan")
        if self.mapping is not None:
            mapped = self.mapping.get(str(raw), self.default)
            return float("nan") if mapped is None else float(mapped)
        try:
            return float(raw) * self.scale
        except (TypeError, ValueError):
            return float("nan")
