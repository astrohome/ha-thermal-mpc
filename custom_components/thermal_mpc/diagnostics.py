"""Diagnostics: the fitted model plus the full training set for offline analysis."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from .coordinator import ThermalConfigEntry


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ThermalConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator = entry.runtime_data
    result = coordinator.result
    return {
        "options": dict(entry.options),
        "columns": {k: c.label for k, c in coordinator.columns.items()},
        "training_days": coordinator.training_days,
        "last_fit": result.last_fit.isoformat() if result.last_fit else None,
        "error": result.error,
        "model": result.model.to_dict() if result.model else None,
        "validation_rmse_k": result.validation,
        "dataset": coordinator.dataset.to_dict(),
    }
