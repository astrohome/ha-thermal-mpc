"""Sensors exposing training progress and the fitted model."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import EntityCategory, UnitOfTemperature, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import STEP_SECONDS
from .coordinator import ThermalConfigEntry, ThermalCoordinator
from .entity import ThermalEntity

STATUS_OPTIONS = ["collecting", "trained", "error"]
REPORT_HORIZONS_H = (1, 3, 6)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ThermalConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add model sensors."""
    coordinator = entry.runtime_data
    entities: list[SensorEntity] = [
        StatusSensor(coordinator),
        TrainingDataSensor(coordinator),
        PredictionErrorSensor(coordinator),
        RecommendedActionSensor(coordinator),
        RecommendedSetpointSensor(coordinator),
        PlannedHeatingSensor(coordinator),
    ]
    entities += [TimeConstantSensor(coordinator, room) for room in coordinator.rooms]
    # Drop sensors for rooms removed in the options flow.
    registry = er.async_get(hass)
    current = {e.unique_id for e in entities}
    for reg in er.async_entries_for_config_entry(registry, entry.entry_id):
        if reg.domain == "sensor" and reg.unique_id not in current:
            registry.async_remove(reg.entity_id)
    async_add_entities(entities)


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


class StatusSensor(ThermalEntity, SensorEntity):
    """collecting / trained / error, with the reason."""

    _attr_translation_key = "status"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = STATUS_OPTIONS

    def __init__(self, coordinator: ThermalCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "status")

    @property
    def native_value(self) -> str:
        """Current state of the model."""
        result = self.coordinator.result
        if result.model is not None:
            return "trained"
        return "error" if result.failed else "collecting"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Timestamps and the last error."""
        result = self.coordinator.result
        model = result.model
        return {
            "last_fit": result.last_fit.isoformat() if result.last_fit else None,
            "last_attempt": result.last_attempt.isoformat()
            if result.last_attempt
            else None,
            "message": result.error,
            "unused_inputs": [self.coordinator.label(i) for i in model.unused_inputs]
            if model
            else [],
        }


class TrainingDataSensor(ThermalEntity, SensorEntity):
    """Days of complete training rows, with per-signal coverage."""

    _attr_translation_key = "training_data"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = UnitOfTime.DAYS
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: ThermalCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "training_data")

    @property
    def native_value(self) -> float:
        """Days where every room and outdoor temperature are known."""
        return round(self.coordinator.training_days, 2)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Span and coverage of each signal."""
        ds = self.coordinator.dataset
        return {
            "span_days": round(ds.rows * ds.step / 86400, 2),
            "coverage": {
                self.coordinator.label(k): round(v, 3) for k, v in ds.coverage().items()
            },
        }


class PredictionErrorSensor(ThermalEntity, SensorEntity):
    """Worst-room open-loop RMSE 6 h ahead on held-out data."""

    _attr_translation_key = "prediction_error"
    _attr_native_unit_of_measurement = "K"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: ThermalCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "prediction_error")

    @property
    def native_value(self) -> float | None:
        """Max over rooms of the error at the full validation horizon."""
        finals = [
            curve[-1]
            for curve in self.coordinator.result.validation.values()
            if curve and curve[-1] is not None
        ]
        return _r(max(finals)) if finals else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Per-room error at a few horizons."""
        per_step_h = STEP_SECONDS / 3600
        out: dict[str, Any] = {}
        for room, curve in self.coordinator.result.validation.items():
            name = self.coordinator.label(room)
            out[name] = {
                f"{h}h": _r(curve[idx])
                for h in REPORT_HORIZONS_H
                if (idx := round(h / per_step_h) - 1) < len(curve)
            }
        return out


class TimeConstantSensor(ThermalEntity, SensorEntity):
    """How many hours a room takes to lose ~63 % of its lead over outdoors."""

    _attr_translation_key = "time_constant"
    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: ThermalCoordinator, room: str) -> None:
        """Initialise for one room column."""
        super().__init__(coordinator, f"time_constant_{room.removeprefix('room:')}")
        self.room = room
        self._attr_translation_placeholders = {"room": coordinator.label(room)}

    @property
    def native_value(self) -> float | None:
        """Time constant against outdoors in hours."""
        model = self.coordinator.result.model
        if model is None or self.room not in model.rooms:
            return None
        return _r(model.rooms[self.room].tau_out_h, 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Gains (K/h per unit input) and couplings (h) to other rooms."""
        model = self.coordinator.result.model
        if model is None or self.room not in model.rooms:
            return {}
        p = model.rooms[self.room]
        label = self.coordinator.label
        return {
            "gains_k_per_h": {label(k): _r(v) for k, v in p.gains.items()},
            "coupling_hours": {
                label(k): _r(1 / v, 1) for k, v in p.g_rooms.items() if v > 1e-6
            },
            "thermal_mass_hours": _r(p.tau_mass_h, 1),
            "thermal_mass_coupling_per_h": _r(p.mass_h),
            "offset_k_per_h": _r(p.offset),
            "one_step_rmse_k": _r(p.rmse_one_step),
            "samples": p.n_samples,
        }


class RecommendedActionSensor(ThermalEntity, SensorEntity):
    """What the shadow planner would tell the thermostat to do right now."""

    _attr_translation_key = "recommended_action"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["heat", "cool", "idle", "off"]

    def __init__(self, coordinator: ThermalCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "recommended_action")

    @property
    def native_value(self) -> str | None:
        """Action for the current hour."""
        plan = self.coordinator.plan
        if plan is None:
            return None
        if plan.get("hvac_mode") == "off":
            return "off"
        return plan["action"]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Plan summary; compare with thermostat_action to judge the plan."""
        plan = self.coordinator.plan
        if plan is None:
            return {}
        thermostat = plan.get("thermostat_action")
        busy = {"heating": "heat", "cooling": "cool"}.get(thermostat or "", "idle")
        return {
            "duty_now": {k: _r(v, 2) for k, v in plan["duty_now"].items()},
            "thermostat_action": thermostat,
            "agrees_with_thermostat": busy == plan["action"],
            "heating_hours_24h": _r(plan["heating_hours"], 2),
            "cooling_hours_24h": _r(plan["cooling_hours"], 2),
            "target": plan["target"],
            "target_source": plan.get("target_source"),
            "band": plan["band"],
            "discomfort_kh_planned": _r(plan["discomfort_kh"]["planned"], 2),
            "discomfort_kh_free_running": _r(plan["discomfort_kh"]["free"], 2),
            "room_spread_k": _r(plan["spread_k"], 2),
            "forecast_sources": plan.get("sources", {}),
            "planned_at": plan.get("created"),
        }


class RecommendedSetpointSensor(ThermalEntity, SensorEntity):
    """Setpoint that would make the thermostat follow the plan (shadow)."""

    _attr_translation_key = "recommended_setpoint"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: ThermalCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "recommended_setpoint")

    @property
    def native_value(self) -> float | None:
        """Current thermostat reading nudged in the direction the plan wants."""
        plan = self.coordinator.plan
        return None if plan is None else plan.get("recommended_setpoint")


class PlannedHeatingSensor(ThermalEntity, SensorEntity):
    """Hours of full-duty heating the plan expects over the next 24 h."""

    _attr_translation_key = "planned_heating"
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: ThermalCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "planned_heating")

    @property
    def native_value(self) -> float | None:
        """Planned heating hours."""
        plan = self.coordinator.plan
        return None if plan is None else _r(plan["heating_hours"], 2)
