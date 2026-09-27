"""Base entity for Thermal MPC."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import ThermalCoordinator


class ThermalEntity(CoordinatorEntity[ThermalCoordinator]):
    """Groups all entities of one entry under a service device."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: ThermalCoordinator, key: str) -> None:
        """Initialise with a key unique within the entry."""
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            entry_type=DeviceEntryType.SERVICE,
        )
