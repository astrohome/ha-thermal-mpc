"""Several sensors in one area become one calibrated room."""

import numpy as np
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.thermal_mpc.const import (
    CONF_CLIMATE,
    CONF_OUTDOOR,
    CONF_ROOMS,
    DOMAIN,
)


async def test_sensors_in_one_area_are_fused(hass: HomeAssistant) -> None:
    area = ar.async_get(hass).async_create("Living Room")
    reg = er.async_get(hass)
    ids = []
    for uid in ("wall", "plant"):
        ent = reg.async_get_or_create("sensor", "test", uid, suggested_object_id=uid)
        reg.async_update_entity(ent.entity_id, area_id=area.id)
        ids.append(ent.entity_id)
    options = {
        CONF_ROOMS: [*ids, "sensor.bedroom"],
        CONF_OUTDOOR: "weather.home",
        CONF_CLIMATE: "climate.thermostat",
    }
    entry = MockConfigEntry(
        domain=DOMAIN, title="Thermal model", options=options, unique_id="x"
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    c = entry.runtime_data

    living = f"zone:{area.id}"
    assert c.zones == {
        living: [f"room:{i}" for i in ids],
        "zone:sensor.bedroom": ["room:sensor.bedroom"],
    }
    assert c.label(living) == "Living Room"

    # A week where the plant sensor reads 0.6 K high; fit learns that.
    rng = np.random.default_rng(0)
    n = 7 * 288
    h = np.arange(n) / 12
    t_out = -5 + 5 * np.sin(2 * np.pi * h / 24)
    heat = (np.sin(2 * np.pi * h / 3) > 0).astype(float)
    temps = np.empty((n, 2))
    temps[0] = [20, 19]
    for k in range(n - 1):
        a, b = temps[k]
        temps[k + 1] = temps[k] + (1 / 12) * np.array(
            [
                (t_out[k] - a) / 30 + (b - a) / 4 + 2 * heat[k],
                (t_out[k] - b) / 20 + (a - b) / 5 + 1.5 * heat[k],
            ]
        )
    ds = c.dataset
    ds.columns, ds.start = {}, None
    ds.append(
        0.0,
        {
            f"room:{ids[0]}": temps[:, 0] + rng.normal(0, 0.02, n),
            f"room:{ids[1]}": temps[:, 0] + 0.6 + rng.normal(0, 0.05, n),
            "room:sensor.bedroom": temps[:, 1],
            "outdoor:weather.home": t_out,
            "heating:climate.thermostat": heat,
            "cooling:climate.thermostat": np.zeros(n),
        },
    )
    await c.async_fit()
    cal = c.fusion[living]
    wall, plant = (cal[f"room:{i}"] for i in ids)
    assert abs((plant.bias - wall.bias) - 0.6) < 0.05
    assert wall.weight > plant.weight  # the quieter sensor counts more
    assert set(c.result.model.rooms) == {living, "zone:sensor.bedroom"}

    hass.states.async_set(ids[0], "21.0")
    hass.states.async_set(ids[1], "unavailable")
    live = c.live_values()
    assert abs(live[living] - (21.0 - wall.bias)) < 1e-6  # plant offline: wall only
    info = c.sensor_info(living, live)
    assert [s["share"] for s in info] == [1.0, 0.0]
