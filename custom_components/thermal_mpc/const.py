"""Constants for the Thermal MPC integration."""

from __future__ import annotations

from datetime import timedelta
from typing import Final

from homeassistant.const import Platform

DOMAIN: Final = "thermal_mpc"
PLATFORMS: Final = [Platform.BUTTON, Platform.SENSOR]

CONF_ROOMS: Final = "rooms"
CONF_OUTDOOR: Final = "outdoor"
CONF_CLIMATE: Final = "climate"
CONF_SOLAR: Final = "solar"
CONF_FAN: Final = "fan"
CONF_VENTILATION: Final = "ventilation"
CONF_GROUP_BY_AREA: Final = "group_by_area"
CONF_WEATHER_FORECAST: Final = "weather_forecast"
CONF_SOLAR_FORECAST: Final = "solar_forecast"
CONF_GAS_METER: Final = "gas_meter"
CONF_GAS_UNIT: Final = "gas_unit"
CONF_GAS_PRICE: Final = "gas_price"
GAS_UNIT_AUTO: Final = "auto"
CONF_TARGET: Final = "target"
CONF_BAND: Final = "band"
CONF_ENERGY_WEIGHT: Final = "energy_weight"
CONF_SPREAD_WEIGHT: Final = "spread_weight"

DEFAULT_TARGET: Final = 21.0
DEFAULT_BAND: Final = 0.5
DEFAULT_ENERGY_WEIGHT: Final = 0.15
DEFAULT_SPREAD_WEIGHT: Final = 0.5
PLAN_HORIZON_H: Final = 24
SETPOINT_NUDGE: Final = 1.0  # K past the thermostat's reading

STEP_SECONDS: Final = 300
COLLECT_INTERVAL: Final = timedelta(minutes=15)
# Settle time so the recorder has committed the states we read.
RECORDER_LAG: Final = timedelta(minutes=2)
BACKFILL: Final = timedelta(days=60)
CHUNK: Final = timedelta(days=1)
RETENTION_DAYS: Final = 120
SAVE_DELAY_SECONDS: Final = 600

FIT_INTERVAL: Final = timedelta(hours=24)
FIT_RETRY_INTERVAL: Final = timedelta(hours=1)
MIN_FIT_DAYS: Final = 3.0
HOLDOUT_FRACTION: Final = 0.2
VALIDATION_HORIZON_H: Final = 6.0

STORAGE_VERSION: Final = 1
# Bump when fitting changes so stored models are refitted at the next update.
FIT_VERSION: Final = 5

PANEL_URL_PATH: Final = "thermal-model"
PANEL_COMPONENT: Final = "thermal-mpc-panel"
STATIC_URL: Final = "/thermal_mpc_static"
