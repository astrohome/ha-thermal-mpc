"""Fetch recorder history from Home Assistant's REST API."""

from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
import requests

from .config import Config, Signal
from .resample import time_weighted_mean

CHUNK = pd.Timedelta(days=1)


class HistoryClient:
    """Minimal client for ``/api/history/period``."""

    def __init__(self, url: str, token: str, timeout: float = 60.0) -> None:
        """Create a client for the HA instance at ``url``."""
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.timeout = timeout

    def fetch(
        self, entity_ids: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> dict[str, list[dict]]:
        """Return all recorded rows per entity, including attribute-only updates.

        Requests are split into day-long chunks to keep responses small. The
        first row of each chunk is the state in force at the chunk start, so
        zero-order hold works across chunk boundaries.
        """
        ids = sorted(set(entity_ids))
        rows: dict[str, list[dict]] = {e: [] for e in ids}
        t = start
        while t < end:
            t_next = min(t + CHUNK, end)
            resp = self.session.get(
                f"{self.url}/api/history/period/{t.isoformat()}",
                params={
                    "end_time": t_next.isoformat(),
                    "filter_entity_id": ",".join(ids),
                    # Keep attribute changes (hvac_action, weather temperature).
                    "significant_changes_only": "0",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            for entity_rows in resp.json():
                if entity_rows:
                    rows[entity_rows[0]["entity_id"]].extend(entity_rows)
            t = t_next
        return rows


def to_series(rows: list[dict], signal: Signal) -> pd.Series:
    """Convert HA history rows for one entity into a numeric step series."""
    if not rows:
        return pd.Series(dtype=float)
    # last_updated also moves on attribute-only changes (e.g. hvac_action).
    stamps = [r.get("last_updated") or r["last_changed"] for r in rows]
    index = pd.to_datetime(stamps, utc=True, format="ISO8601")
    values = [signal.value(r["state"], r.get("attributes") or {}) for r in rows]
    return pd.Series(values, index=index, dtype=float)


def build_dataset(
    config: Config, rows: dict[str, list[dict]], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Resample every configured signal onto the model grid."""
    columns = {}
    for name, signal in config.signals.items():
        series = to_series(rows.get(signal.entity, []), signal)
        columns[name] = time_weighted_mean(series, start, end, config.step)
    df = pd.DataFrame(columns)
    df.index.name = "time"
    return df


def export(config: Config, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch history from HA and return the resampled dataset."""
    token = os.environ.get(config.ha_token_env)
    if not token:
        raise RuntimeError(f"set ${config.ha_token_env} to a long-lived HA token")
    client = HistoryClient(config.ha_url, token)
    entities = [s.entity for s in config.signals.values()]
    rows = client.fetch(entities, start, end)
    return build_dataset(config, rows, start, end)
