"""Thermal MPC: learn a multi-room thermal model of the house from the recorder.

Everything runs inside Home Assistant: history is read straight from the
recorder, the resampled training set and fitted model live in ``.storage``,
and results are exposed as sensors. No tokens or external services.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN, PLATFORMS, STORAGE_VERSION
from .coordinator import ThermalConfigEntry, ThermalCoordinator


async def async_setup_entry(hass: HomeAssistant, entry: ThermalConfigEntry) -> bool:
    """Set up from a config entry."""
    coordinator = ThermalCoordinator(hass, entry)
    await coordinator.async_load()
    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    # The first collection can backfill weeks of history; don't block startup.
    entry.async_create_background_task(
        hass, coordinator.async_refresh(), f"{DOMAIN} initial collection"
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ThermalConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: ThermalConfigEntry) -> None:
    """Delete the stored dataset when the entry is removed."""
    await Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}").async_remove()


async def _async_update_listener(
    hass: HomeAssistant, entry: ThermalConfigEntry
) -> None:
    """Reload when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
