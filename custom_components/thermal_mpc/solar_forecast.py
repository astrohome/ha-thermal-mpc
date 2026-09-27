"""Find and read solar-production forecasts from other integrations.

Any integration that ships an ``energy`` platform with
``async_get_solar_forecast`` (Forecast.Solar, Solcast, Open-Meteo Solar
Forecast, ...) works. The platform is imported directly, so this does not
depend on the Energy dashboard being set up.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.loader import IntegrationNotFound, async_get_integration

_LOGGER = logging.getLogger(__name__)

ForecastFn = Callable[[HomeAssistant, str], Awaitable[dict[str, Any] | None]]


async def _forecast_fn(hass: HomeAssistant, domain: str) -> ForecastFn | None:
    try:
        integration = await async_get_integration(hass, domain)
        if not integration.platforms_exists(["energy"]):
            return None
        platform = await integration.async_get_platform("energy")
    except (IntegrationNotFound, ImportError):
        return None
    except Exception:  # noqa: BLE001 - a broken third-party platform is skipped
        _LOGGER.debug("Could not load energy platform of %s", domain, exc_info=True)
        return None
    return getattr(platform, "async_get_solar_forecast", None)


async def async_forecast_entries(hass: HomeAssistant) -> list[ConfigEntry]:
    """Config entries that can provide a solar-production forecast."""
    usable: dict[str, bool] = {}
    out = []
    for entry in hass.config_entries.async_entries():
        if entry.domain not in usable:
            usable[entry.domain] = await _forecast_fn(hass, entry.domain) is not None
        if usable[entry.domain]:
            out.append(entry)
    return out


async def async_get_wh_hours(hass: HomeAssistant, entry_id: str) -> dict[str, float]:
    """``{iso_hour: Wh}`` for one config entry, or {} if unavailable."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None:
        return {}
    fn = await _forecast_fn(hass, entry.domain)
    if fn is None:
        return {}
    try:
        data = await fn(hass, entry_id)
    except Exception:  # noqa: BLE001 - a missing forecast must not stop planning
        _LOGGER.debug("Solar forecast from %s failed", entry.title, exc_info=True)
        return {}
    return (data or {}).get("wh_hours", {})
