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
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_BAND,
    CONF_CLIMATE,
    CONF_ENERGY_WEIGHT,
    CONF_FAN,
    CONF_GAS_METER,
    CONF_GAS_PRICE,
    CONF_GAS_UNIT,
    CONF_GROUP_BY_AREA,
    CONF_OUTDOOR,
    CONF_ROOMS,
    CONF_SOLAR,
    CONF_SOLAR_FORECAST,
    CONF_SPREAD_WEIGHT,
    CONF_TARGET,
    CONF_VENTILATION,
    CONF_WEATHER_FORECAST,
    DEFAULT_BAND,
    DEFAULT_ENERGY_WEIGHT,
    DEFAULT_SPREAD_WEIGHT,
    DOMAIN,
    GAS_UNIT_AUTO,
)
from .core.gas import UNITS as GAS_UNITS
from .solar_forecast import async_forecast_entries

ON_OFF_DOMAINS = ["sensor", "binary_sensor", "fan", "switch", "select"]
PLANNER_KEYS = (
    CONF_TARGET,
    CONF_BAND,
    CONF_ENERGY_WEIGHT,
    CONF_SPREAD_WEIGHT,
    CONF_GAS_PRICE,
)


async def _solar_forecast_options(hass: HomeAssistant) -> list[SelectOptionDict]:
    """Config entries of integrations that provide a solar forecast."""
    return [
        SelectOptionDict(
            value=e.entry_id, label=e.title or f"{e.domain} ({e.entry_id[-6:]})"
        )
        for e in await async_forecast_entries(hass)
    ]


async def _entities_schema(hass: HomeAssistant) -> vol.Schema:
    fields: dict[Any, Any] = {
        vol.Required(CONF_ROOMS): EntitySelector(
            EntitySelectorConfig(
                domain="sensor", device_class="temperature", multiple=True
            )
        ),
        vol.Optional(CONF_GROUP_BY_AREA, default=True): BooleanSelector(),
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
        vol.Optional(CONF_GAS_METER): EntitySelector(
            EntitySelectorConfig(domain="sensor", device_class=["gas", "energy"])
        ),
        vol.Optional(CONF_GAS_UNIT, default=GAS_UNIT_AUTO): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=GAS_UNIT_AUTO, label="From the meter"),
                    *(SelectOptionDict(value=u, label=u) for u in GAS_UNITS),
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(CONF_WEATHER_FORECAST): EntitySelector(
            EntitySelectorConfig(domain="weather")
        ),
    }
    if solar := await _solar_forecast_options(hass):
        fields[vol.Optional(CONF_SOLAR_FORECAST)] = SelectSelector(
            SelectSelectorConfig(options=solar, mode=SelectSelectorMode.DROPDOWN)
        )
    return vol.Schema(fields)


def _number(
    minimum: float, maximum: float, step: float | str, unit: str | None = None
) -> NumberSelector:
    config = NumberSelectorConfig(
        min=minimum, max=maximum, step=step, mode=NumberSelectorMode.BOX
    )
    if unit:
        config["unit_of_measurement"] = unit
    return NumberSelector(config)


PLANNER_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_TARGET): _number(10, 30, 0.5, "°C"),
        vol.Required(CONF_BAND, default=DEFAULT_BAND): _number(0.1, 3, 0.1, "K"),
        vol.Required(CONF_ENERGY_WEIGHT, default=DEFAULT_ENERGY_WEIGHT): _number(
            0, 5, 0.05
        ),
        vol.Required(CONF_SPREAD_WEIGHT, default=DEFAULT_SPREAD_WEIGHT): _number(
            0, 5, 0.05
        ),
        vol.Optional(CONF_GAS_PRICE): _number(0, 1000, "any"),
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
            data_schema=self.add_suggested_values_to_schema(
                await _entities_schema(self.hass), user_input
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return ThermalMpcOptionsFlow()


class ThermalMpcOptionsFlow(OptionsFlow):
    """Change entities, then planner settings (reloads the entry)."""

    def __init__(self) -> None:
        """Initialise."""
        self._entities: dict[str, Any] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Entities, pre-filled."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate(user_input)
            if not errors:
                self._entities = user_input
                return await self.async_step_planner()
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                await _entities_schema(self.hass),
                user_input or dict(self.config_entry.options),
            ),
            errors=errors,
        )

    async def async_step_planner(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Comfort target and trade-offs for the planner."""
        if user_input is not None:
            return self.async_create_entry(data={**self._entities, **user_input})
        current = {
            k: v for k, v in self.config_entry.options.items() if k in PLANNER_KEYS
        }
        return self.async_show_form(
            step_id="planner",
            data_schema=self.add_suggested_values_to_schema(PLANNER_SCHEMA, current),
        )
