"""Config flow for Thermal MPC: pick the entities that describe the house."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import EntitySelector, EntitySelectorConfig

from .const import (
    CONF_CLIMATE,
    CONF_FAN,
    CONF_OUTDOOR,
    CONF_ROOMS,
    CONF_SOLAR,
    CONF_VENTILATION,
    DOMAIN,
)

ON_OFF_DOMAINS = ["sensor", "binary_sensor", "fan", "switch", "select"]

SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ROOMS): EntitySelector(
            EntitySelectorConfig(
                domain="sensor", device_class="temperature", multiple=True
            )
        ),
        vol.Required(CONF_OUTDOOR): EntitySelector(
            EntitySelectorConfig(domain=["sensor", "weather"])
        ),
        vol.Required(CONF_CLIMATE): EntitySelector(
            EntitySelectorConfig(domain="climate")
        ),
        vol.Optional(CONF_SOLAR): EntitySelector(
            EntitySelectorConfig(domain="sensor", device_class="power")
        ),
        vol.Optional(CONF_FAN): EntitySelector(
            EntitySelectorConfig(domain=ON_OFF_DOMAINS)
        ),
        vol.Optional(CONF_VENTILATION): EntitySelector(
            EntitySelectorConfig(domain=ON_OFF_DOMAINS)
        ),
    }
)


def _validate(user_input: dict[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    if not user_input.get(CONF_ROOMS):
        errors[CONF_ROOMS] = "no_rooms"
    elif user_input[CONF_OUTDOOR] in user_input[CONF_ROOMS]:
        errors[CONF_OUTDOOR] = "outdoor_is_room"
    return errors


class ThermalMpcConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle initial setup."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for rooms, outdoor source, thermostat and optional inputs."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate(user_input)
            if not errors:
                await self.async_set_unique_id(user_input[CONF_CLIMATE])
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Thermal model", data={}, options=user_input
                )
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(SCHEMA, user_input),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return ThermalMpcOptionsFlow()


class ThermalMpcOptionsFlow(OptionsFlow):
    """Change the entities after setup (reloads the entry)."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show the same form, pre-filled."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate(user_input)
            if not errors:
                return self.async_create_entry(data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                SCHEMA, user_input or dict(self.config_entry.options)
            ),
            errors=errors,
        )
