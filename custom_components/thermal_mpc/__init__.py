"""Thermal MPC: learn a multi-room thermal model of the house from the recorder.

Everything runs inside Home Assistant: history is read straight from the
recorder, the resampled training set and fitted model live in ``.storage``,
and results are exposed as sensors. No tokens or external services.
"""

from __future__ import annotations

from pathlib import Path

from homeassistant.components import panel_custom
from homeassistant.components.http import StaticPathConfig
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import async_get_integration

from . import websocket
from .const import (
    DOMAIN,
    PANEL_COMPONENT,
    PANEL_URL_PATH,
    PLATFORMS,
    STATIC_URL,
    STORAGE_VERSION,
)
from .coordinator import ThermalConfigEntry, ThermalCoordinator

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

FRONTEND_DIR = Path(__file__).parent / "frontend"


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the websocket API and the sidebar panel."""
    websocket.async_register(hass)
    if hass.http is None or "frontend" not in hass.config.components:
        return True  # headless setups (tests): the API still works
    version = (await async_get_integration(hass, DOMAIN)).version
    await hass.http.async_register_static_paths(
        [StaticPathConfig(STATIC_URL, str(FRONTEND_DIR), cache_headers=False)]
    )
    await panel_custom.async_register_panel(
        hass,
        webcomponent_name=PANEL_COMPONENT,
        frontend_url_path=PANEL_URL_PATH,
        module_url=f"{STATIC_URL}/{PANEL_COMPONENT}.js?v={version}",
        sidebar_title="Thermal model",
        sidebar_icon="mdi:home-thermometer-outline",
        require_admin=False,
        config={},
    )
    return True


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
