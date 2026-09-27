# Home Assistant setup

## 1. Keep more history (urgent)

The recorder purges raw states after 10 days by default, and the model needs
weeks of 5-minute data. In `configuration.yaml`, merging with any existing
`recorder:` block:

```yaml
recorder:
  purge_keep_days: 60
```

The database grows by roughly tens of MB per week on a typical install. If
that is a concern, add `include:` or `exclude:` filters. In the meantime, run
`thermal-mpc export --days 10 --append` daily (cron, or an HA
`shell_command`) so nothing is lost: the CSV accumulates beyond the purge
window.

## 2. What the default config uses

| Column | Entity | Notes |
|---|---|---|
| `T_living_room` | `sensor.timmerflotte_temp_hmd_sensor_temperature` | |
| `T_bedroom` | `sensor.timmerflotte_temp_hmd_sensor_temperature_2` | |
| `T_thermostat` | `sensor.aprilaire_indoor_temperature_controlling_sensor` | 0.5 K steps |
| `T_primary_bedroom` | `sensor.kaa_temperature` | plant sensor |
| `T_office` | `sensor.office_plant_temperature` | plant sensor |
| `T_out` | `weather.forecast_home` → `temperature` | met.no, not measured |
| `heating` / `cooling` | `climate.aprilaire_thermostat` → `hvac_action` | duty cycle |
| `solar_kw` | `sensor.vue2_solar_power` | real-time PV as irradiance proxy |
| `fan` | `sensor.aprilaire_fan_status` | |
| `ventilation` | `sensor.aprilaire_ventilation_status` | HRV |

Also available for later use (energy cost term, heat-delivery calibration):
`sensor.vue2_furnance_power`, `sensor.vue2_ac_power`,
`sensor.meter_gas_meter_gas_consumption`, the Growatt per-string power
(useful if the strings face different directions), and
`sensor.power_production_now` (Forecast.Solar).

## 3. Gaps worth closing, in order of value

1. **A dedicated temperature sensor per thermal zone**, including the
   basement, office and kitchen. Place them at mid-wall height, away from
   sun, vents and electronics. Cheap Zigbee or Thread sensors are fine.
2. **An outdoor sensor** on the north side, in the shade. met.no is smoothed
   and hourly.
3. **CO₂ in the bedrooms**, so the HRV can respond to demand instead of a
   schedule.
4. **Window/door contacts**, at least for doors that are often open. Without
   them, opening a door looks like a sudden change in the house's physics.
5. *(Optional, hardware)* **zone dampers or smart vents**, the only way to
   get real per-room control.

## 4. Thermostat settings before closing the loop

* Put the thermostat on **permanent hold** with its own schedule disabled, so
  the program doesn't fight the controller.
* Note the minimum on/off cycle times and staging thresholds configured on
  the thermostat; the controller will respect them.
