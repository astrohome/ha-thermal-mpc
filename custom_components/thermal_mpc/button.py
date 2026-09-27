"""Retrain button."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import ThermalConfigEntry, ThermalCoordinator
from .entity import ThermalEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ThermalConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add the retrain button."""
    async_add_entities([RetrainButton(entry.runtime_data)])


class RetrainButton(ThermalEntity, ButtonEntity):
    """Fit the model now instead of waiting for the daily refit."""

    _attr_translation_key = "retrain"
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: ThermalCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "retrain")

    async def async_press(self) -> None:
        """Collect the latest data and refit."""
        await self.coordinator.async_retrain()
