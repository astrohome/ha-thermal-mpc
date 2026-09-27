# Home Assistant setup

## 1. Recorder retention

The integration backfills from the recorder on first start, so a longer
retention (already set to 60 days here) gives it more to learn from right
away:

```yaml
recorder:
  purge_keep_days: 60
```

After setup the integration keeps its own 5-minute copy in `.storage` for
120 days, so retention only matters for the initial backfill.

## 2. Suggested setup for this house

| Field | Entity | Notes |
|---|---|---|
| Rooms | `sensor.timmerflotte_temp_hmd_sensor_temperature` (living) | |
| | `sensor.timmerflotte_temp_hmd_sensor_temperature_2` (bedroom) | |
| | `sensor.aprilaire_indoor_temperature_controlling_sensor` | 0.5 K steps, coarse |
| | `sensor.kaa_temperature` (primary bedroom) | plant sensor |
| | `sensor.office_plant_temperature` (office) | plant sensor |
| Outdoor | `weather.forecast_home` | met.no, not measured |
| Thermostat | `climate.aprilaire_thermostat` | |
| Solar power | `sensor.vue2_solar_power` | real-time, W |
| Fan | `sensor.aprilaire_fan_status` | |
| Ventilation | `sensor.aprilaire_ventilation_status` | HRV |

Useful later for the energy-cost term: `sensor.vue2_furnance_power`,
`sensor.vue2_ac_power`, `sensor.meter_gas_meter_gas_consumption`, the Growatt
per-string power (useful if the strings face different directions), and
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
