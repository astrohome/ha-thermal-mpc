"""End-to-end tests against a real (in-memory) recorder."""

from datetime import timedelta

import numpy as np
import pytest
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.thermal_mpc.const import (
    CONF_CLIMATE,
    CONF_OUTDOOR,
    CONF_ROOMS,
    CONF_SOLAR,
    CONF_VENTILATION,
    DOMAIN,
)

OPTIONS = {
    CONF_ROOMS: ["sensor.living", "sensor.bedroom"],
    CONF_OUTDOOR: "weather.home",
    CONF_CLIMATE: "climate.thermostat",
    CONF_SOLAR: "sensor.pv",
    CONF_VENTILATION: "sensor.hrv",
}


async def test_config_flow(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM

    bad = {**OPTIONS, CONF_OUTDOOR: "sensor.living"}
    result = await hass.config_entries.flow.async_configure(result["flow_id"], bad)
    assert result["errors"] == {CONF_OUTDOOR: "outdoor_is_room"}

    result = await hass.config_entries.flow.async_configure(result["flow_id"], OPTIONS)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"] == OPTIONS
    await hass.async_block_till_done()


async def test_options_flow(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, title="Thermal model", options=OPTIONS, unique_id="x"
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    new = {**OPTIONS, CONF_ROOMS: ["sensor.living"]}
    result = await hass.config_entries.options.async_configure(result["flow_id"], new)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.options[CONF_ROOMS] == ["sensor.living"]
    assert hass.states.get("sensor.thermal_model_bedroom_time_constant") is None


async def test_backfill_from_recorder(hass: HomeAssistant, freezer) -> None:
    """History recorded before setup lands on the 5-minute grid."""
    start = dt_util.utcnow().replace(minute=0, second=0, microsecond=0)
    freezer.move_to(start - timedelta(hours=3))
    hass.states.async_set(
        "sensor.living",
        "20.0",
        {"unit_of_measurement": "°C", "friendly_name": "Living"},
    )
    hass.states.async_set("sensor.bedroom", "19.0", {"friendly_name": "Bedroom"})
    hass.states.async_set("weather.home", "cloudy", {"temperature": -5.0})
    hass.states.async_set("climate.thermostat", "heat", {"hvac_action": "idle"})
    hass.states.async_set("sensor.pv", "1500", {"unit_of_measurement": "W"})
    hass.states.async_set("sensor.hrv", "off")
    await hass.async_block_till_done()

    freezer.move_to(start - timedelta(minutes=60))
    # Attribute-only change: must still be picked up.
    hass.states.async_set("climate.thermostat", "heat", {"hvac_action": "heating"})
    hass.states.async_set("weather.home", "cloudy", {"temperature": -6.0})
    freezer.move_to(start - timedelta(minutes=30))
    hass.states.async_set("sensor.hrv", "unavailable")
    await async_wait_recording_done(hass)

    freezer.move_to(start + timedelta(minutes=5))
    entry = MockConfigEntry(
        domain=DOMAIN, title="Thermal model", options=OPTIONS, unique_id="x"
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)

    ds = entry.runtime_data.dataset
    assert ds.end == pytest.approx(start.timestamp())
    cols = ds.columns
    heat = cols["heating:climate.thermostat"]
    assert np.nanmax(heat[-12:]) == 1.0  # heating for the last hour
    assert heat[-13] == 0.0
    assert cols["outdoor:weather.home"][-1] == -6.0
    assert cols["solar_kw:sensor.pv"][-1] == 1.5
    assert np.isnan(cols["ventilation:sensor.hrv"][-1])
    assert cols["room:sensor.living"][-1] == 20.0

    status = hass.states.get("sensor.thermal_model_model_status")
    assert status.state == "collecting"
    training = hass.states.get("sensor.thermal_model_training_data")
    assert float(training.state) == pytest.approx(3 / 24, abs=0.01)


async def test_fit_updates_sensors(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN, title="Thermal model", options=OPTIONS, unique_id="x"
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coordinator = entry.runtime_data

    # Replace the (empty) dataset with a week of synthetic data.
    rng = np.random.default_rng(0)
    n = 7 * 288
    hours = np.arange(n) / 12
    t_out = -5 + 5 * np.sin(2 * np.pi * hours / 24)
    heat = (np.sin(2 * np.pi * hours / 3) > 0).astype(float)
    solar = np.clip(np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0, None) * 3
    vent = (rng.random(n) > 0.5).astype(float)
    temps = np.empty((n, 2))
    temps[0] = [20, 19]
    for k in range(n - 1):
        a, b = temps[k]
        temps[k + 1] = temps[k] + (1 / 12) * np.array(
            [
                (t_out[k] - a) / 30 + (b - a) / 4 + 2 * heat[k] + 0.5 * solar[k],
                (t_out[k] - b) / 20 + (a - b) / 5 + 1.5 * heat[k],
            ]
        )
    ds = coordinator.dataset
    ds.columns, ds.start = {}, None
    ds.append(
        0.0,
        {
            "room:sensor.living": temps[:, 0],
            "room:sensor.bedroom": temps[:, 1],
            "outdoor:weather.home": t_out,
            "heating:climate.thermostat": heat,
            "cooling:climate.thermostat": np.zeros(n),
            "solar_kw:sensor.pv": solar,
            "ventilation:sensor.hrv": vent,
        },
    )
    await coordinator.async_fit()
    coordinator.async_update_listeners()
    await hass.async_block_till_done()

    assert hass.states.get("sensor.thermal_model_model_status").state == "trained"
    tau = hass.states.get("sensor.thermal_model_living_time_constant")
    assert float(tau.state) == pytest.approx(30, rel=0.05)
    assert tau.attributes["gains_k_per_h"]["Heating"] == pytest.approx(2, rel=0.05)
    err = hass.states.get("sensor.thermal_model_prediction_error_6_h")
    assert float(err.state) < 0.05

    # Model survives a reload through .storage.
    coordinator.async_schedule_save()
    await coordinator._store._async_handle_write_data()  # noqa: SLF001
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get("sensor.thermal_model_model_status").state == "trained"
