"""Collect training data from the recorder and (re)fit the thermal model."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
from homeassistant.components.recorder import get_instance, history
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    BACKFILL,
    CHUNK,
    COLLECT_INTERVAL,
    CONF_CLIMATE,
    CONF_FAN,
    CONF_OUTDOOR,
    CONF_ROOMS,
    CONF_SOLAR,
    CONF_VENTILATION,
    DOMAIN,
    FIT_INTERVAL,
    FIT_RETRY_INTERVAL,
    HOLDOUT_FRACTION,
    MIN_FIT_DAYS,
    RECORDER_LAG,
    RETENTION_DAYS,
    SAVE_DELAY_SECONDS,
    STEP_SECONDS,
    STORAGE_VERSION,
    VALIDATION_HORIZON_H,
)
from .core.dataset import Dataset
from .core.model import (
    FREE,
    NEGATIVE,
    POSITIVE,
    ModelSpec,
    NotEnoughDataError,
    ThermalModel,
    fit,
    validate,
)
from .core.resample import time_weighted_mean
from .core.signals import Signal

_LOGGER = logging.getLogger(__name__)

ThermalConfigEntry = ConfigEntry["ThermalCoordinator"]

ON = {"on": 1.0}
POWER_SCALE_TO_KW = {"W": 0.001, "kW": 1.0, "MW": 1000.0}


@dataclass
class Column:
    """A dataset column: where it comes from and what it means to the model."""

    signal: Signal
    label: str
    sign: str | None = None  # None for temperatures, sign constraint for inputs


@dataclass
class FitResult:
    """Outcome of the latest fit attempt."""

    model: ThermalModel | None = None
    validation: dict[str, list[float | None]] = field(default_factory=dict)
    last_fit: datetime | None = None
    last_attempt: datetime | None = None
    error: str | None = None
    failed: bool = False  # True for real errors, False for "not enough data yet"


class ThermalCoordinator(DataUpdateCoordinator[None]):
    """Keep a rolling 5-minute dataset and a fitted model for one house."""

    config_entry: ThermalConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ThermalConfigEntry) -> None:
        """Build column definitions from the entry's options."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=COLLECT_INTERVAL,
        )
        self.columns = self._build_columns(dict(entry.options))
        self.rooms = [k for k in self.columns if k.startswith("room:")]
        self.outdoor = next(k for k in self.columns if k.startswith("outdoor:"))
        self.spec = ModelSpec(
            rooms=self.rooms,
            outdoor=self.outdoor,
            inputs={k: c.sign for k, c in self.columns.items() if c.sign},
        )
        self.dataset = Dataset(step=float(STEP_SECONDS))
        self.result = FitResult()
        self._collect_lock = asyncio.Lock()
        # How far the recorder has been read, even if nothing usable was found.
        self._collected_until: float | None = None
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}"
        )

    def _build_columns(self, opts: dict[str, Any]) -> dict[str, Column]:
        cols: dict[str, Column] = {}
        for eid in opts[CONF_ROOMS]:
            cols[f"room:{eid}"] = Column(Signal(eid), self._name(eid))
        outdoor = opts[CONF_OUTDOOR]
        attr = "temperature" if outdoor.startswith("weather.") else None
        cols[f"outdoor:{outdoor}"] = Column(Signal(outdoor, attr), "Outdoor")
        climate = opts[CONF_CLIMATE]
        for action, sign in (("heating", POSITIVE), ("cooling", NEGATIVE)):
            sig = Signal(climate, "hvac_action", {action: 1.0}, default=0.0)
            cols[f"{action}:{climate}"] = Column(sig, action.capitalize(), sign)
        if solar := opts.get(CONF_SOLAR):
            unit = self._unit(solar)
            scale = POWER_SCALE_TO_KW.get(unit or "", 1.0)
            if unit not in POWER_SCALE_TO_KW:
                _LOGGER.warning("%s has unit %r; assuming kW", solar, unit)
            cols[f"solar_kw:{solar}"] = Column(
                Signal(solar, scale=scale), "Solar", POSITIVE
            )
        for key, label in ((CONF_FAN, "Fan"), (CONF_VENTILATION, "Ventilation")):
            if eid := opts.get(key):
                sig = Signal(eid, mapping=ON, default=0.0)
                cols[f"{key}:{eid}"] = Column(sig, label, FREE)
        return cols

    def _name(self, entity_id: str) -> str:
        state = self.hass.states.get(entity_id)
        if state and state.name:
            return state.name
        entry = er.async_get(self.hass).async_get(entity_id)
        if entry and (entry.name or entry.original_name):
            return entry.name or entry.original_name or entity_id
        return entity_id.partition(".")[2].replace("_", " ").title()

    def _unit(self, entity_id: str) -> str | None:
        entry = er.async_get(self.hass).async_get(entity_id)
        if entry and entry.unit_of_measurement:
            return entry.unit_of_measurement
        state = self.hass.states.get(entity_id)
        return state.attributes.get("unit_of_measurement") if state else None

    # ------------------------------------------------------------------ storage

    async def async_load(self) -> None:
        """Restore the dataset and model saved by a previous run."""
        data = await self._store.async_load()
        if not data:
            return
        if data.get("dataset"):
            self.dataset = Dataset.from_dict(data["dataset"])
            self.dataset.keep_columns(set(self.columns))
        if data.get("model"):
            model = ThermalModel.from_dict(data["model"])
            # A model for a different set of rooms is useless; refit instead.
            if set(model.rooms) == set(self.rooms) and model.outdoor == self.outdoor:
                self.result.model = model
                self.result.validation = data.get("validation", {})
                if last := data.get("last_fit"):
                    self.result.last_fit = dt_util.parse_datetime(last)

    def _data_to_save(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "model": self.result.model.to_dict() if self.result.model else None,
            "validation": self.result.validation,
            "last_fit": self.result.last_fit.isoformat()
            if self.result.last_fit
            else None,
        }

    def async_schedule_save(self) -> None:
        """Persist soon; Store flushes pending saves on shutdown."""
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY_SECONDS)

    # --------------------------------------------------------------- collection

    async def _async_update_data(self) -> None:
        await self._async_collect()
        if self._fit_due():
            await self.async_fit()
        self.async_schedule_save()

    async def async_retrain(self) -> None:
        """Collect the latest data and refit now."""
        await self._async_collect()
        await self.async_fit()
        self.async_schedule_save()
        self.async_update_listeners()

    async def _async_collect(self) -> None:
        """Pull everything the recorder has since the dataset's end."""
        async with self._collect_lock:
            await self._async_collect_locked()

    async def _async_collect_locked(self) -> None:
        step = float(STEP_SECONDS)
        now = dt_util.utcnow() - RECORDER_LAG
        end = math.floor(now.timestamp() / step) * step
        earliest = math.floor((end - BACKFILL.total_seconds()) / step) * step
        start = self.dataset.end or self._collected_until or earliest
        if start < earliest:
            start = earliest
        if self.dataset.end is not None and start > self.dataset.end:
            # Offline longer than the backfill window: pad the gap.
            gap = round((start - self.dataset.end) / step)
            self.dataset.append(
                self.dataset.end, {k: np.full(gap, np.nan) for k in self.columns}
            )

        recorder = get_instance(self.hass)
        chunk = CHUNK.total_seconds()
        t = start
        while t < end:
            t_next = min(t + chunk, end)
            block = await recorder.async_add_executor_job(
                partial(self._fetch_block, t, t_next)
            )
            self.dataset.append(t, block)
            t = t_next
        self.dataset.trim(int(RETENTION_DAYS * 86400 / step))
        self._collected_until = max(start, end)

    def _fetch_block(self, start: float, end: float) -> dict[str, np.ndarray]:
        """Read ``[start, end)`` from the recorder and resample (runs in executor)."""
        entity_ids = sorted({c.signal.entity_id for c in self.columns.values()})
        states = history.get_significant_states(
            self.hass,
            dt_util.utc_from_timestamp(start),
            dt_util.utc_from_timestamp(end),
            entity_ids,
            include_start_time_state=True,
            # Attribute-only changes matter (hvac_action, weather temperature).
            significant_changes_only=False,
            minimal_response=False,
            no_attributes=False,
        )
        n_bins = round((end - start) / STEP_SECONDS)
        block = {}
        for key, col in self.columns.items():
            raw = states.get(col.signal.entity_id, [])
            rows = [s for s in raw if isinstance(s, State)]
            times = np.array([s.last_updated.timestamp() for s in rows])
            values = np.array([col.signal.value(s.state, s.attributes) for s in rows])
            block[key] = time_weighted_mean(
                times, values, start, n_bins, float(STEP_SECONDS)
            )
        return block

    # ---------------------------------------------------------------------- fit

    @property
    def training_days(self) -> float:
        """Days of rows where every room and the outdoor temperature are known."""
        needed = [*self.rooms, self.outdoor]
        if not self.dataset.rows or any(k not in self.dataset.columns for k in needed):
            return 0.0
        stacked = np.vstack([self.dataset.columns[k] for k in needed])
        complete = int((~np.isnan(stacked)).all(axis=0).sum())
        return complete * self.dataset.step / 86400

    def _fit_due(self) -> bool:
        if self.training_days < MIN_FIT_DAYS:
            return False
        now = dt_util.utcnow()
        last = self.result.last_attempt
        if self.result.model is None:
            return last is None or now - last >= FIT_RETRY_INTERVAL
        last_fit = self.result.last_fit
        return last_fit is None or now - last_fit >= FIT_INTERVAL

    async def async_fit(self) -> None:
        """Fit on the dataset, validate on a hold-out tail, then refit on all."""
        self.result.last_attempt = dt_util.utcnow()
        try:
            model, validation = await self.hass.async_add_executor_job(
                _fit_and_validate, self.dataset, self.spec
            )
        except NotEnoughDataError as err:
            self.result.error = f"Not enough data: {err}"
            self.result.failed = False
            _LOGGER.info("Thermal model not fitted yet: %s", err)
            return
        except Exception as err:  # noqa: BLE001 - keep collecting on a bad fit
            self.result.error = f"Fit failed: {err}"
            self.result.failed = True
            _LOGGER.exception("Thermal model fit failed")
            return
        self.result.model = model
        self.result.validation = validation
        self.result.last_fit = self.result.last_attempt
        self.result.error = None
        self.result.failed = False

    def label(self, column: str) -> str:
        """Human-readable name of a dataset column."""
        col = self.columns.get(column)
        return col.label if col else column


def _fit_and_validate(
    ds: Dataset, spec: ModelSpec
) -> tuple[ThermalModel, dict[str, list[float | None]]]:
    split = int(ds.rows * (1 - HOLDOUT_FRACTION))
    holdout_model = fit(ds, spec, slice(0, split))
    validation = validate(
        holdout_model, ds, slice(split, None), horizon_h=VALIDATION_HORIZON_H
    )
    return fit(ds, spec), validation
