"""Gas meter as the heating input."""

from datetime import timedelta

import numpy as np
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.thermal_mpc.const import (
    CONF_CLIMATE,
    CONF_GAS_METER,
    CONF_GAS_PRICE,
    CONF_GAS_UNIT,
    CONF_OUTDOOR,
    CONF_ROOMS,
    DOMAIN,
)

FT3_KWH = 0.3039
OPTIONS = {
    CONF_ROOMS: ["sensor.living", "sensor.bedroom"],
    CONF_OUTDOOR: "weather.home",
    CONF_CLIMATE: "climate.thermostat",
    CONF_GAS_METER: "sensor.gas",
    CONF_GAS_UNIT: "ft³",  # the meter claims CCF but counts ft³
}
HEAT = "heating:climate.thermostat"
GAS = "gas:sensor.gas"


async def _setup(hass: HomeAssistant, options=None) -> MockConfigEntry:
    hass.states.async_set(
        "sensor.gas", "260504", {"unit_of_measurement": "CCF", "device_class": "gas"}
    )
    entry = MockConfigEntry(
        domain=DOMAIN, title="Thermal model", options=options or OPTIONS, unique_id="x"
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    return entry


async def test_gas_meter_replaces_heating_duty(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    coordinator = entry.runtime_data
    assert GAS in coordinator.spec.inputs
    assert HEAT not in coordinator.spec.inputs
    assert HEAT in coordinator.columns  # still collected
    assert coordinator.gas_unit == "ft³"
    assert coordinator.gas_kwh_per_unit == pytest.approx(FT3_KWH)
    assert coordinator.label(GAS) == "Gas heat"
    gas = hass.states.get("sensor.thermal_model_planned_gas_24_h")
    assert gas is not None
    assert gas.attributes["unit_of_measurement"] == "ft³"


async def test_unit_from_the_meter(hass: HomeAssistant) -> None:
    entry = await _setup(hass, {**OPTIONS, CONF_GAS_UNIT: "auto"})
    assert entry.runtime_data.gas_unit == "CCF"
    assert entry.runtime_data.gas_kwh_per_unit == pytest.approx(30.39)


async def test_counter_backfill(hass: HomeAssistant, freezer) -> None:
    """The meter's running total becomes kW of heat per 5-minute step."""
    start = dt_util.utcnow().replace(minute=0, second=0, microsecond=0)
    freezer.move_to(start - timedelta(hours=3))
    hass.states.async_set("sensor.living", "20.0")
    hass.states.async_set("sensor.bedroom", "19.0")
    hass.states.async_set("weather.home", "cloudy", {"temperature": -5.0})
    hass.states.async_set("climate.thermostat", "heat", {"hvac_action": "idle"})
    hass.states.async_set("sensor.gas", "1000", {"unit_of_measurement": "CCF"})
    await hass.async_block_till_done(wait_background_tasks=True)
    freezer.move_to(start - timedelta(minutes=62))
    hass.states.async_set("sensor.gas", "1005", {"unit_of_measurement": "CCF"})
    freezer.move_to(start - timedelta(minutes=32))
    hass.states.async_set("sensor.gas", "1010", {"unit_of_measurement": "CCF"})
    await async_wait_recording_done(hass)

    freezer.move_to(start + timedelta(minutes=5))
    entry = await _setup(hass)
    ds = entry.runtime_data.dataset
    rate = ds.columns[GAS]
    step_h = ds.step / 3600
    assert np.nansum(rate) * step_h == pytest.approx(10 * FT3_KWH)
    assert np.nanmax(rate) == pytest.approx(5 * FT3_KWH / step_h)
    assert rate[-1] == 0.0
    assert np.count_nonzero(np.nan_to_num(rate)) == 2


def _week(gas_kw: float | None) -> dict[str, np.ndarray]:
    n = 7 * 288
    hours = np.arange(n) / 12
    t_out = -5 + 5 * np.sin(2 * np.pi * hours / 24)
    heat = (np.sin(2 * np.pi * hours / 3) > 0).astype(float)
    q = heat * (gas_kw or 20.0)
    temps = np.empty((n, 2))
    temps[0] = [20, 19]
    for k in range(n - 1):
        a, b = temps[k]
        temps[k + 1] = temps[k] + (1 / 12) * np.array(
            [
                (t_out[k] - a) / 30 + (b - a) / 4 + 0.1 * q[k],
                (t_out[k] - b) / 20 + (a - b) / 5 + 0.075 * q[k],
            ]
        )
    return {
        "room:sensor.living": temps[:, 0],
        "room:sensor.bedroom": temps[:, 1],
        "outdoor:weather.home": t_out,
        HEAT: heat,
        "cooling:climate.thermostat": np.zeros(n),
        GAS: q if gas_kw else np.full(n, np.nan),
    }


async def _fit_and_plan(hass: HomeAssistant, entry, columns) -> dict:
    coordinator = entry.runtime_data
    ds = coordinator.dataset
    ds.columns, ds.start = {}, None
    ds.append(0.0, columns)
    await coordinator.async_fit()
    hass.states.async_set(
        "climate.thermostat",
        "heat",
        {"hvac_action": "idle", "current_temperature": 19.0, "temperature": 23.0},
    )
    hass.states.async_set("sensor.living", "19.0")
    hass.states.async_set("sensor.bedroom", "18.5")
    await coordinator.async_update_plan()
    coordinator.async_update_listeners()
    await hass.async_block_till_done()
    return coordinator.plan


async def test_fit_and_plan_in_gas_mode(hass: HomeAssistant) -> None:
    entry = await _setup(hass, {**OPTIONS, CONF_GAS_PRICE: 0.02})
    coordinator = entry.runtime_data
    plan = await _fit_and_plan(hass, entry, _week(20.0))

    model = coordinator.result.model
    assert GAS in model.inputs
    assert HEAT not in model.inputs
    assert model.input_capacity[GAS] == pytest.approx(20.0, rel=0.1)
    assert model.rooms["zone:sensor.living"].gains[GAS] == pytest.approx(0.1, rel=0.1)

    assert plan["action"] == "heat"
    assert plan["heat_capacity_kw"] == model.input_capacity[GAS]
    assert plan["heating_kwh"] > 0
    assert plan["gas_unit"] == "ft³"
    assert plan["gas_volume"] == pytest.approx(plan["heating_kwh"] / FT3_KWH)
    assert plan["gas_cost"] == pytest.approx(plan["gas_volume"] * 0.02)

    gas = hass.states.get("sensor.thermal_model_planned_gas_24_h")
    assert float(gas.state) == pytest.approx(plan["gas_volume"], abs=0.01)
    assert gas.attributes["capacity_kw"] == pytest.approx(
        model.input_capacity[GAS], abs=0.01
    )
    action = hass.states.get("sensor.thermal_model_recommended_action")
    assert action.attributes["planned_gas"] == pytest.approx(
        plan["gas_volume"], abs=0.01
    )
    assert action.attributes["planned_cost"] is not None

    # The model (with its capacity) survives a reload without a refit.
    coordinator.async_schedule_save()
    await coordinator._store._async_handle_write_data()  # noqa: SLF001
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    reloaded = entry.runtime_data.result
    assert reloaded.model.input_capacity == model.input_capacity
    assert reloaded.last_fit is not None


async def test_falls_back_to_duty_without_capacity(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    plan = await _fit_and_plan(hass, entry, _week(None))
    model = entry.runtime_data.result.model
    assert HEAT in model.inputs
    assert GAS not in model.inputs
    assert model.input_capacity == {}
    assert plan["action"] == "heat"
    assert plan["heat_capacity_kw"] is None
    assert plan["gas_volume"] is None
    gas = hass.states.get("sensor.thermal_model_planned_gas_24_h")
    assert gas.state == "unknown"


async def test_options_flow_stores_gas(hass: HomeAssistant) -> None:
    base = {k: v for k, v in OPTIONS.items() if not k.startswith("gas")}
    entry = await _setup(hass, base)
    assert GAS not in entry.runtime_data.columns
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], OPTIONS
    )
    assert result["step_id"] == "planner"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"band": 0.5, "energy_weight": 0.15, "spread_weight": 0.5, "gas_price": 1.2},
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.options[CONF_GAS_METER] == "sensor.gas"
    assert entry.options[CONF_GAS_UNIT] == "ft³"
    assert entry.options[CONF_GAS_PRICE] == 1.2
    assert HEAT not in entry.runtime_data.spec.inputs
