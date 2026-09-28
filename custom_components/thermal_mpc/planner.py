"""Shadow-mode planner: gather forecasts, run the MPC, derive a recommendation.

Nothing here writes to the thermostat. The result is published as sensors
and on the panel so the plan can be compared with what the thermostat does.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import (
    CONF_BAND,
    CONF_CLIMATE,
    CONF_ENERGY_WEIGHT,
    CONF_SOLAR_FORECAST,
    CONF_SPREAD_WEIGHT,
    CONF_TARGET,
    CONF_WEATHER_FORECAST,
    DEFAULT_BAND,
    DEFAULT_ENERGY_WEIGHT,
    DEFAULT_SPREAD_WEIGHT,
    DEFAULT_TARGET,
    PLAN_HORIZON_H,
    SETPOINT_NUDGE,
)
from .core.forecast import hourly_energy_to_kw, interpolate, persistence
from .core.insight import observed_mass
from .core.mpc import PlanInputs, PlanSettings, plan
from .solar_forecast import async_get_wh_hours

if TYPE_CHECKING:
    from .coordinator import ThermalCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_weather_forecast(
    hass: HomeAssistant, entity_id: str
) -> list[tuple[float, float]]:
    """Hourly (epoch s, °C) points from ``weather.get_forecasts``."""
    try:
        resp = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"type": "hourly"},
            target={"entity_id": entity_id},
            blocking=True,
            return_response=True,
        )
    except (HomeAssistantError, ValueError) as err:
        _LOGGER.debug("No hourly forecast from %s: %s", entity_id, err)
        return []
    points = []
    for item in (resp or {}).get(entity_id, {}).get("forecast", []):
        when = dt_util.parse_datetime(str(item.get("datetime")))
        temp = item.get("temperature")
        if when is not None and isinstance(temp, (int, float)):
            points.append((when.timestamp(), float(temp)))
    return points


def _settings(
    coordinator: ThermalCoordinator, climate_state: Any
) -> tuple[PlanSettings, str]:
    opts = coordinator.config_entry.options
    target = opts.get(CONF_TARGET)
    target_source = "option"
    if target is None and climate_state is not None:
        target = climate_state.attributes.get("temperature")
        target_source = "thermostat setpoint"
    if not isinstance(target, (int, float)):
        target, target_source = DEFAULT_TARGET, "default"
    mode = climate_state.state if climate_state is not None else "heat"
    return (
        PlanSettings(
            target=float(target),
            band=float(opts.get(CONF_BAND, DEFAULT_BAND)),
            energy_weight=float(opts.get(CONF_ENERGY_WEIGHT, DEFAULT_ENERGY_WEIGHT)),
            spread_weight=float(opts.get(CONF_SPREAD_WEIGHT, DEFAULT_SPREAD_WEIGHT)),
            allow_heat=mode in ("heat", "heat_cool", "auto"),
            allow_cool=mode in ("cool", "heat_cool", "auto"),
            horizon_h=PLAN_HORIZON_H,
        ),
        target_source,
    )


async def async_plan(coordinator: ThermalCoordinator) -> dict[str, Any] | None:
    """Build forecast inputs and run the planner (None without a model)."""
    model = coordinator.result.model
    ds = coordinator.view()
    if model is None or ds.rows == 0:
        return None
    hass = coordinator.hass
    opts = coordinator.config_entry.options
    step_s = ds.step
    n = round(PLAN_HORIZON_H * 3600 / step_s)
    start = math.floor(dt_util.utcnow().timestamp() / step_s) * step_s
    times = start + step_s * np.arange(n)
    per_day = round(86400 / step_s)
    live = coordinator.live_values()
    sources: dict[str, str] = {}

    # Outdoor temperature: forecast, anchored at the live value.
    outdoor_entity = coordinator.columns[coordinator.outdoor].signal.entity_id
    weather = opts.get(CONF_WEATHER_FORECAST) or (
        outdoor_entity if outdoor_entity.startswith("weather.") else None
    )
    points = await async_weather_forecast(hass, weather) if weather else []
    now_out = live.get(coordinator.outdoor, float("nan"))
    if points and np.isfinite(now_out):
        points = [(start, now_out), *[p for p in points if p[0] > start]]
    t_out = interpolate(points, times) if points else None
    if t_out is None:
        t_out = persistence(ds.columns[coordinator.outdoor], per_day, n)
        sources["outdoor"] = "yesterday repeated"
    else:
        sources["outdoor"] = weather or ""

    # Baseline inputs: solar from forecast; fan / HRV at their recent duty.
    u = np.zeros((n, len(model.inputs)))
    solar_kw = None
    for j, name in enumerate(model.inputs):
        history = ds.columns.get(name, np.zeros(0))
        if name.startswith("solar_kw:"):
            fc_entry = opts.get(CONF_SOLAR_FORECAST)
            wh = await async_get_wh_hours(hass, fc_entry) if fc_entry else {}
            series = hourly_energy_to_kw(wh, times) if wh else None
            if series is None:
                series = persistence(history, per_day, n)
                sources["solar"] = "yesterday repeated"
            else:
                sources["solar"] = "solar forecast"
            u[:, j] = series
            solar_kw = series
        elif name.startswith(("heating:", "cooling:")):
            continue  # decided by the planner
        else:
            recent = history[-per_day:]
            u[:, j] = float(np.nanmean(recent)) if np.isfinite(recent).any() else 0.0

    temps = np.array([live.get(r, np.nan) for r in model.rooms])
    last = np.array([ds.columns[r][-1] for r in model.rooms])
    temps = np.where(np.isfinite(temps), temps, last)
    if not np.isfinite(temps).all():
        return None
    mass = observed_mass(model, ds)[-1]

    climate_id = opts[CONF_CLIMATE]
    climate = hass.states.get(climate_id)
    settings, target_source = _settings(coordinator, climate)
    heat_col = next((c for c in model.inputs if c.startswith("heating:")), None)
    cool_col = next((c for c in model.inputs if c.startswith("cooling:")), None)
    inputs = PlanInputs(
        start=start,
        temps=temps,
        mass=np.where(np.isfinite(mass), mass, temps),
        t_out=t_out,
        u=u,
        solar_kw=solar_kw,
        sources=sources,
    )
    result = await hass.async_add_executor_job(
        plan, model, inputs, settings, heat_col, cool_col
    )
    result["target_source"] = target_source
    result["hvac_mode"] = climate.state if climate else None
    result["thermostat_action"] = (
        climate.attributes.get("hvac_action") if climate else None
    )
    result["recommended_setpoint"] = _setpoint(result, climate)
    result["created"] = dt_util.utcnow().isoformat()
    return result


def _setpoint(result: dict[str, Any], climate: Any) -> float | None:
    """Setpoint that would make the thermostat do what the plan wants now.

    The thermostat only runs when its own sensor is past the setpoint, so
    nudge the setpoint a little past its reading in the needed direction.
    """
    if climate is None:
        return None
    current = climate.attributes.get("current_temperature")
    if not isinstance(current, (int, float)):
        return None
    mode = climate.state
    action = result["action"]
    if action == "heat":
        value = current + SETPOINT_NUDGE
    elif action == "cool":
        value = current - SETPOINT_NUDGE
    elif mode == "cool":
        value = current + SETPOINT_NUDGE
    else:
        value = current - SETPOINT_NUDGE
    return round(value * 2) / 2
