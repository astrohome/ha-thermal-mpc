"""Configuration loading."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SIGNS = ("positive", "negative", "free")


@dataclass(frozen=True)
class Signal:
    """How to turn one HA entity (state or attribute) into a numeric series."""

    entity: str
    attribute: str | None = None
    map: dict[str, float] | None = None
    default: float | None = None
    scale: float = 1.0
    sign: str = "free"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Signal:
        """Build a signal from its YAML mapping."""
        mapping = raw.get("map")
        if mapping is not None:
            # YAML 1.1 turns unquoted on/off into booleans; undo that.
            mapping = {
                (("on" if k else "off") if isinstance(k, bool) else str(k)): float(v)
                for k, v in mapping.items()
            }
        sign = raw.get("sign", "free")
        if sign not in SIGNS:
            raise ValueError(f"sign must be one of {SIGNS}, got {sign!r}")
        default = raw.get("default")
        return cls(
            entity=raw["entity"],
            attribute=raw.get("attribute"),
            map=mapping,
            default=None if default is None else float(default),
            scale=float(raw.get("scale", 1.0)),
            sign=sign,
        )

    def value(self, state: str, attributes: dict[str, Any]) -> float:
        """Convert one recorded HA state row to a number (NaN if unusable)."""
        if state in ("unavailable", "unknown"):
            return float("nan")
        raw = attributes.get(self.attribute) if self.attribute else state
        if raw is None:
            return float("nan")
        if self.map is not None:
            mapped = self.map.get(str(raw), self.default)
            return float("nan") if mapped is None else mapped
        try:
            return float(raw) * self.scale
        except (TypeError, ValueError):
            return float("nan")


@dataclass
class Config:
    """Top-level configuration."""

    rooms: dict[str, Signal]
    outdoor: Signal
    inputs: dict[str, Signal]
    step: pd.Timedelta
    couplings: list[tuple[str, str]]
    ha_url: str = ""
    ha_token_env: str = "HA_TOKEN"

    @property
    def signals(self) -> dict[str, Signal]:
        """All signals keyed by column name."""
        out = {f"T_{name}": sig for name, sig in self.rooms.items()}
        out["T_out"] = self.outdoor
        out.update(self.inputs)
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        """Build a config from a parsed YAML document."""
        rooms = {k: Signal.from_dict(v) for k, v in raw["rooms"].items()}
        if len(rooms) < 1:
            raise ValueError("at least one room is required")
        couplings_raw = raw.get("couplings")
        if couplings_raw is None:
            couplings = list(itertools.combinations(rooms, 2))
        else:
            couplings = []
            for a, b in couplings_raw:
                if a not in rooms or b not in rooms:
                    raise ValueError(f"unknown room in coupling [{a}, {b}]")
                couplings.append((a, b))
        ha = raw.get("home_assistant", {})
        return cls(
            rooms=rooms,
            outdoor=Signal.from_dict(raw["outdoor"]),
            inputs={k: Signal.from_dict(v) for k, v in raw.get("inputs", {}).items()},
            step=pd.Timedelta(raw.get("step", "5min")),
            couplings=couplings,
            ha_url=ha.get("url", "").rstrip("/"),
            ha_token_env=ha.get("token_env", "HA_TOKEN"),
        )

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a YAML config file."""
        with Path(path).open() as fh:
            return cls.from_dict(yaml.safe_load(fh))
