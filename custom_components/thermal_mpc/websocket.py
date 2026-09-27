"""Websocket API used by the Thermal model panel."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import ThermalCoordinator


def _coordinators(hass: HomeAssistant) -> dict[str, ThermalCoordinator]:
    return {
        e.entry_id: e.runtime_data
        for e in hass.config_entries.async_entries(DOMAIN)
        if e.state is ConfigEntryState.LOADED
    }


@websocket_api.websocket_command({vol.Required("type"): f"{DOMAIN}/overview"})
@websocket_api.async_response
async def ws_overview(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict
) -> None:
    """Return model, live heat budgets and replays for every entry."""
    entries: list[dict[str, Any]] = [
        await c.async_overview() for c in _coordinators(hass).values()
    ]
    connection.send_result(msg["id"], {"entries": entries})


@websocket_api.websocket_command(
    {vol.Required("type"): f"{DOMAIN}/retrain", vol.Required("entry_id"): str}
)
@websocket_api.async_response
async def ws_retrain(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict
) -> None:
    """Collect and refit one entry now."""
    coordinator = _coordinators(hass).get(msg["entry_id"])
    if coordinator is None:
        connection.send_error(msg["id"], "not_found", "Unknown or unloaded entry")
        return
    await coordinator.async_retrain()
    connection.send_result(msg["id"], {"status": coordinator.result.error or "ok"})


def async_register(hass: HomeAssistant) -> None:
    """Register the commands."""
    websocket_api.async_register_command(hass, ws_overview)
    websocket_api.async_register_command(hass, ws_retrain)
