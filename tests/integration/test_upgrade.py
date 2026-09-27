"""Options flow discovery of solar forecasts, and refits after code upgrades."""

from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.thermal_mpc import solar_forecast
from custom_components.thermal_mpc.const import (
    CONF_CLIMATE,
    CONF_OUTDOOR,
    CONF_ROOMS,
    CONF_SOLAR_FORECAST,
    DOMAIN,
    STORAGE_VERSION,
)

OPTIONS = {
    CONF_ROOMS: ["sensor.living", "sensor.bedroom"],
    CONF_OUTDOOR: "weather.home",
    CONF_CLIMATE: "climate.thermostat",
}


async def _fake_fn(hass, entry_id):
    return {"wh_hours": {"2026-01-01T12:00:00+00:00": 1500}}


async def test_solar_forecast_offered_without_energy_dashboard(
    hass: HomeAssistant,
) -> None:
    provider = MockConfigEntry(domain="forecast_solar", title="Roof")
    provider.add_to_hass(hass)
    entry = MockConfigEntry(
        domain=DOMAIN, title="Thermal model", options=OPTIONS, unique_id="x"
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)

    async def fn_for(hass, domain):
        return _fake_fn if domain == "forecast_solar" else None

    with patch.object(solar_forecast, "_forecast_fn", fn_for):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        keys = {str(k) for k in result["data_schema"].schema}
        assert CONF_SOLAR_FORECAST in keys
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {**OPTIONS, CONF_SOLAR_FORECAST: provider.entry_id}
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {"band": 0.5, "energy_weight": 0.15, "spread_weight": 0.5},
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert await solar_forecast.async_get_wh_hours(hass, provider.entry_id) == {
            "2026-01-01T12:00:00+00:00": 1500
        }
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.options[CONF_SOLAR_FORECAST] == provider.entry_id


async def test_model_from_older_code_is_refitted(hass: HomeAssistant, hass_storage):
    entry = MockConfigEntry(
        domain=DOMAIN, title="Thermal model", options=OPTIONS, unique_id="x"
    )
    rooms = {
        f"room:{e}": {
            "g_out": 0.0,
            "g_rooms": {},
            "gains": {},
            "offset": 0.0,
            "rmse_one_step": 0.0,
            "n_samples": 0,
        }
        for e in OPTIONS[CONF_ROOMS]
    }
    hass_storage[f"{DOMAIN}.{entry.entry_id}"] = {
        "version": STORAGE_VERSION,
        "key": f"{DOMAIN}.{entry.entry_id}",
        "data": {
            "dataset": None,
            "model": {
                "step_h": 5 / 60,
                "outdoor": "outdoor:weather.home",
                "inputs": [],
                "rooms": rooms,
            },
            "validation": {},
            "last_fit": "2026-09-27T06:37:36+00:00",  # no fit_version: old code
        },
    }
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coordinator = entry.runtime_data
    assert coordinator.result.model is not None  # shown until refit
    assert coordinator.result.last_fit is None  # so the next update refits
