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

PANEL_URL_PATH: Final = "thermal-model"
PANEL_COMPONENT: Final = "thermal-mpc-panel"
STATIC_URL: Final = "/thermal_mpc_static"
