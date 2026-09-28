"""Collect training data from the recorder and (re)fit the thermal model."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
from homeassistant.components.recorder import get_instance, history
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
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
    CONF_GAS_METER,
    CONF_GAS_UNIT,
    CONF_GROUP_BY_AREA,
    CONF_OUTDOOR,
    CONF_ROOMS,
    CONF_SOLAR,
    CONF_VENTILATION,
    DOMAIN,
    FIT_INTERVAL,
    FIT_RETRY_INTERVAL,
    FIT_VERSION,
    GAS_UNIT_AUTO,
    HOLDOUT_FRACTION,
    MIN_FIT_DAYS,
    RECORDER_LAG,
    RETENTION_DAYS,
    SAVE_DELAY_SECONDS,
    STEP_SECONDS,
    STORAGE_VERSION,
    VALIDATION_HORIZON_H,
)
from .core import fusion as fusion_mod
from .core import gas as gas_mod
from .core.dataset import Dataset
from .core.insight import budget, mean_budget, observed_mass, replay
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
from .core.resample import counter_rate, time_weighted_mean
from .core.signals import Signal
from .planner import async_plan

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
        self.gas_unit: str | None = None  # unit of the gas meter
        self.gas_kwh_per_unit = 1.0
        self.columns = self._build_columns(dict(entry.options))
        self.heat_col = f"heating:{entry.options[CONF_CLIMATE]}"
        self.gas_col = next((k for k in self.columns if k.startswith("gas:")), None)
        # Raw sensor columns ("room:<entity>") are grouped into zones (rooms):
        # one per area, or one per sensor when grouping is off. The model
        # works on zones; their temperature fuses the sensors.
        self.sensors = [k for k in self.columns if k.startswith("room:")]
        self.zones, self.zone_labels = self._build_zones(
            entry.options.get(CONF_GROUP_BY_AREA, True)
        )
        self.rooms = list(self.zones)
        self.solar_col = next(
            (k for k in self.columns if k.startswith("solar_kw:")), None
        )
        self._view_cache: tuple[Any, Dataset] | None = None
        self.outdoor = next(k for k in self.columns if k.startswith("outdoor:"))
        inputs = {k: c.sign for k, c in self.columns.items() if c.sign}
        if self.gas_col:
            # Gas measures the heat burned; the duty would be collinear with it.
            # Duty is still collected (capacity, fallback, thermostat check).
            inputs.pop(self.heat_col, None)
        self.spec = ModelSpec(
            rooms=self.rooms,
            outdoor=self.outdoor,
            inputs=inputs,
            # Sun lands on floors and walls: let it heat the thermal mass too.
            mass_inputs=tuple(k for k in self.columns if k.startswith("solar_kw:")),
        )
        self.dataset = Dataset(step=float(STEP_SECONDS))
        self.result = FitResult()
        self.plan: dict[str, Any] | None = None
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
        if gas := opts.get(CONF_GAS_METER):
            self.gas_unit, self.gas_kwh_per_unit = self._gas_unit(
                gas, opts.get(CONF_GAS_UNIT)
            )
            sig = Signal(gas, scale=self.gas_kwh_per_unit, counter=True)
            cols[f"gas:{gas}"] = Column(sig, "Gas heat", POSITIVE)
        for key, label in ((CONF_FAN, "Fan"), (CONF_VENTILATION, "Ventilation")):
            if eid := opts.get(key):
                sig = Signal(eid, mapping=ON, default=0.0)
                cols[f"{key}:{eid}"] = Column(sig, label, FREE)
        return cols

    def _gas_unit(self, entity_id: str, override: str | None) -> tuple[str, float]:
        """Meter unit (option, else the entity's) and kWh of heat per unit."""
        if override and override != GAS_UNIT_AUTO:
            unit = gas_mod.normalise_unit(override)
        else:
            reported = self._unit(entity_id)
            unit = gas_mod.normalise_unit(reported)
            if unit is None:
                _LOGGER.warning(
                    "%s has unit %r; assuming kWh (set the gas unit option)",
                    entity_id,
                    reported,
                )
        unit = unit or "kWh"
        return unit, gas_mod.KWH_PER_UNIT[unit]

    def _area_id(self, entity_id: str) -> str | None:
        entry = er.async_get(self.hass).async_get(entity_id)
        if entry is None:
            return None
        if entry.area_id:
            return entry.area_id
        if entry.device_id and (
            device := dr.async_get(self.hass).async_get(entry.device_id)
        ):
            return device.area_id
        return None

    def _build_zones(
        self, by_area: bool
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        zones: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        areas = ar.async_get(self.hass)
        for col in self.sensors:
            eid = col.removeprefix("room:")
            area_id = self._area_id(eid) if by_area else None
            if area_id:
                key = f"zone:{area_id}"
                area = areas.async_get_area(area_id)
                labels[key] = area.name if area else area_id
            else:
                key = f"zone:{eid}"
                labels[key] = self.columns[col].label
            zones.setdefault(key, []).append(col)
        return zones, labels

    @property
    def fusion(self) -> fusion_mod.Fusion:
        """Current sensor calibration (from the fitted model, else neutral)."""
        model = self.result.model
        return fusion_mod.from_dict(model.fusion) if model else {}

    def view(self) -> Dataset:
        """Dataset with fused zone temperatures added (cached)."""
        key = (
            id(self.dataset),
            self.dataset.rows,
            self.dataset.start,
            id(self.result.model),
        )
        if self._view_cache is None or self._view_cache[0] != key:
            self._view_cache = (
                key,
                _view(
                    self.dataset,
                    self.zones,
                    self.fusion,
                    self.solar_col,
                    self.gas_col,
                ),
            )
        return self._view_cache[1]

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
            if (
                set(model.rooms) == set(self.rooms)
                and model.outdoor == self.outdoor
                and all(
                    set(model.fusion.get(z, {})) <= set(s)
                    for z, s in self.zones.items()
                )
            ):
                self.result.model = model
                self.result.validation = data.get("validation", {})
                # A model from older fitting code, or for other inputs (a gas
                # meter added or removed), is shown until the refit, which
                # happens at the first update (last_fit unset).
                if (
                    data.get("fit_version") == FIT_VERSION
                    and self._inputs_match(model)
                    and (last := data.get("last_fit"))
                ):
                    self.result.last_fit = dt_util.parse_datetime(last)

    def _inputs_match(self, model: ThermalModel) -> bool:
        """Whether ``model`` was fitted for the configured inputs."""
        fitted = set(model.inputs) | set(model.unused_inputs)
        fallback = _duty_spec(self.spec, self.gas_col, self.heat_col)
        return fitted in (set(self.spec.inputs), set(fallback.inputs))

    def _data_to_save(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "model": self.result.model.to_dict() if self.result.model else None,
            "validation": self.result.validation,
            "fit_version": FIT_VERSION,
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
        await self.async_update_plan()
        self.async_schedule_save()

    async def async_update_plan(self) -> None:
        """Re-run the shadow planner; failures only disable the plan."""
        try:
            self.plan = await async_plan(self)
        except Exception:  # noqa: BLE001 - planning must never stop collection
            _LOGGER.exception("Planning failed")
            self.plan = None

    async def async_retrain(self) -> None:
        """Collect the latest data, refit and re-plan now."""
        await self._async_collect()
        await self.async_fit()
        await self.async_update_plan()
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
            resample = counter_rate if col.signal.counter else time_weighted_mean
            block[key] = resample(times, values, start, n_bins, float(STEP_SECONDS))
        return block

    # ---------------------------------------------------------------------- fit

    @property
    def training_days(self) -> float:
        """Days of rows where every room and the outdoor temperature are known."""
        needed = [*self.rooms, self.outdoor]
        if not self.dataset.rows or self.outdoor not in self.dataset.columns:
            return 0.0
        ds = self.view()
        stacked = np.vstack([ds.columns[k] for k in needed])
        complete = int((~np.isnan(stacked)).all(axis=0).sum())
        return complete * ds.step / 86400

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
                _fit_and_validate,
                self.dataset,
                self.spec,
                self.zones,
                self.solar_col,
                self.gas_col,
                self.heat_col,
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
        """Human-readable name of a dataset column or zone."""
        if column in self.zone_labels:
            return self.zone_labels[column]
        col = self.columns.get(column)
        return col.label if col else column

    # ----------------------------------------------------------------- overview

    def live_values(self) -> dict[str, float]:
        """Read the current value of every column from the state machine.

        Falls back to the latest dataset value when an entity is unavailable.
        """
        out = {}
        view = self.view() if self.dataset.rows else None
        for key, col in self.columns.items():
            if col.signal.counter:
                # A meter's state is its running total; the rate is only
                # known from the collected (smoothed) data.
                out[key] = (
                    _last_known(view.columns[key])
                    if view is not None and key in view.columns
                    else float("nan")
                )
                continue
            state = self.hass.states.get(col.signal.entity_id)
            value = (
                col.signal.value(state.state, state.attributes)
                if state
                else float("nan")
            )
            # Room sensors stay NaN when offline: fusion uses the others.
            if (
                math.isnan(value)
                and key in self.dataset.columns
                and key not in self.sensors
            ):
                out[key] = _last_known(self.dataset.columns[key])
            else:
                out[key] = value
        # Zone (room) temperatures from the calibrated sensors.
        sun_now = 0.0
        if self.solar_col and self.solar_col in self.dataset.columns:
            sun_now = (
                float(
                    fusion_mod.smooth_sun(
                        self.dataset.columns[self.solar_col],
                        self.dataset.step / 3600,
                        self.dataset.rows,
                    )[-1]
                )
                if self.dataset.rows
                else 0.0
            )
        zones_now = fusion_mod.fuse_now(out, self.zones, self.fusion, sun_now)
        for zone, value in zones_now.items():
            if math.isnan(value) and view is not None:
                value = _last_known(view.columns[zone])
            out[zone] = value
        return out

    def sensor_info(self, zone: str, live: dict[str, float]) -> list[dict[str, Any]]:
        """Per-sensor calibration and share of the room reading."""
        cals = self.fusion.get(zone, {})
        weights = {
            s: cals.get(s, fusion_mod.SensorCal()).weight for s in self.zones[zone]
        }
        online = [s for s in self.zones[zone] if not math.isnan(live.get(s, math.nan))]
        total = sum(weights[s] for s in online) or 1.0
        out = []
        for s in self.zones[zone]:
            cal = cals.get(s)
            out.append(
                {
                    "entity_id": self.columns[s].signal.entity_id,
                    "name": self.columns[s].label,
                    "temperature": _json_num(live.get(s)),
                    "share": round(weights[s] / total, 3) if s in online else 0.0,
                    "bias_k": round(cal.bias, 3) if cal else None,
                    "sun_k_per_kw": round(cal.sun, 3) if cal else None,
                    "noise_k": round(cal.sigma, 3) if cal else None,
                }
            )
        return out

    async def async_overview(self) -> dict[str, Any]:
        """Everything the panel needs, as JSON-ready data."""
        live = self.live_values()
        model = self.result.model
        insight: dict[str, Any] = {}
        if model is not None:
            insight = await self.hass.async_add_executor_job(
                _insight, model, self.view(), live
            )
        per_step = STEP_SECONDS / 3600
        rooms = []
        for room in self.rooms:
            params = model.rooms.get(room) if model else None
            curve = self.result.validation.get(room, [])
            rooms.append(
                {
                    "id": room,
                    "name": self.label(room),
                    "sensors": self.sensor_info(room, live),
                    "temperature": _json_num(live.get(room)),
                    "tau_out_h": params.tau_out_h if params else None,
                    "tau_mass_h": params.tau_mass_h if params else None,
                    "mass_h": params.mass_h if params else None,
                    "one_step_rmse": params.rmse_one_step if params else None,
                    "coupling_h": {
                        # Slower than ~6 weeks is "not coupled" for display.
                        o: 1 / g
                        for o, g in params.g_rooms.items()
                        if g > 1e-3
                    }
                    if params
                    else {},
                    "gains": params.gains if params else {},
                    "validation": {
                        f"{h}h": curve[i]
                        for h in (1, 3, 6)
                        if (i := round(h / per_step) - 1) < len(curve)
                    },
                }
            )
        return {
            "entry_id": self.config_entry.entry_id,
            "title": self.config_entry.title,
            "status": "trained"
            if model
            else ("error" if self.result.failed else "collecting"),
            "message": self.result.error,
            "training_days": self.training_days,
            "min_fit_days": MIN_FIT_DAYS,
            "last_fit": self.result.last_fit.isoformat()
            if self.result.last_fit
            else None,
            "labels": {k: c.label for k, c in self.columns.items()},
            "outdoor": {
                "id": self.outdoor,
                "temperature": _json_num(live.get(self.outdoor)),
            },
            "inputs": {
                k: _json_num(live.get(k)) for k, c in self.columns.items() if c.sign
            },
            "unused_inputs": model.unused_inputs if model else [],
            "plan": self.plan,
            "rooms": rooms,
            **insight,
        }


def _view(
    raw: Dataset,
    zones: fusion_mod.Zones,
    fusion: fusion_mod.Fusion,
    solar: str | None,
    gas: str | None,
) -> Dataset:
    """Dataset with fused zone temperatures and a smoothed gas rate."""
    ds = fusion_mod.view(raw, zones, fusion, solar)
    if gas and gas in ds.columns:
        ds.columns[gas] = gas_mod.smooth(ds.columns[gas])
    return ds


def _duty_spec(spec: ModelSpec, gas: str | None, duty: str) -> ModelSpec:
    """``spec`` with the gas input replaced by the thermostat's heating duty."""
    if not gas or gas not in spec.inputs:
        return spec
    inputs = {duty if k == gas else k: s for k, s in spec.inputs.items()}
    return replace(spec, inputs=inputs)


def _fit_and_validate(
    raw: Dataset,
    spec: ModelSpec,
    zones: fusion_mod.Zones,
    solar: str | None,
    gas: str | None = None,
    duty: str | None = None,
) -> tuple[ThermalModel, dict[str, list[float | None]]]:
    # Calibrate sensors against each other, then fit on fused room temps.
    cal = fusion_mod.learn(raw, zones, solar)
    ds = _view(raw, zones, cal, solar, gas)
    capacity: dict[str, float] = {}
    if gas and duty and gas in spec.inputs:
        cap = gas_mod.capacity(
            ds.columns.get(gas, np.zeros(0)),
            ds.columns.get(duty, np.zeros(0)),
            ds.step / 3600,
        )
        if cap is None:
            # Full fire not seen yet, so a planned duty cannot be turned into
            # kW: model the heating duty instead until it has been.
            _LOGGER.info("Furnace capacity not known yet; modelling heating duty")
            spec = _duty_spec(spec, gas, duty)
        else:
            capacity[gas] = cap
    split = int(ds.rows * (1 - HOLDOUT_FRACTION))
    holdout_model = fit(ds, spec, slice(0, split))
    validation = validate(
        holdout_model, ds, slice(split, None), horizon_h=VALIDATION_HORIZON_H
    )
    model = fit(ds, spec)
    model.fusion = fusion_mod.to_dict(cal)
    model.input_capacity = {k: v for k, v in capacity.items() if k in model.inputs}
    return model, validation


def _insight(
    model: ThermalModel, ds: Dataset, live: dict[str, float]
) -> dict[str, Any]:
    temps = {r: live.get(r, float("nan")) for r in model.rooms}
    inputs = {i: live.get(i, float("nan")) for i in model.inputs}
    day = round(24 / model.step_h)
    mass_now = {}
    if ds.rows:
        last = observed_mass(model, ds)[-1]
        mass_now = {r: float(last[k]) for k, r in enumerate(model.rooms)}
    return {
        "budget_now": budget(
            model, temps, live.get(model.outdoor, np.nan), inputs, mass_now
        ),
        "budget_24h": mean_budget(model, ds, slice(max(0, ds.rows - day), ds.rows)),
        "replay": replay(model, ds),
    }


def _json_num(v: float | None) -> float | None:
    return None if v is None or math.isnan(v) else round(v, 3)


def _last_known(series: np.ndarray) -> float:
    known = series[~np.isnan(series)]
    return float(known[-1]) if known.size else float("nan")
