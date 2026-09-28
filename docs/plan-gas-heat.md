# Plan: gas meter as the heating input

Status: **planned, not started.** Nothing below is implemented yet.

## Why

The heating input today is the thermostat's `hvac_action` duty cycle
(0–1 per 5-minute step). That assumes every running minute delivers the same
heat, which is wrong for a two-stage or modulating furnace. Gas volume
measures the heat actually burned:

* heating value of natural gas varies only a few percent, and furnace
  efficiency is roughly constant; the model's learned heating gain absorbs
  any constant factor;
* it lets the planner report and weigh energy in real units (ft³ / m³ / GJ)
  and cost, instead of "hours of heating".

## Facts about this house (checked live)

* `sensor.meter_gas_meter_gas_consumption`: `state_class: total`,
  `device_class: gas`, unit **"CCF"**, value ~260 504. As CCF that would be
  26 million ft³, which isn't plausible, so it is almost certainly a raw
  ft³ counter. The integration must allow a unit override.
* The meter only reports when the counter changes (no change for 2 h with
  the furnace idle). Expected resolution ~1 ft³ ≈ 0.30 kWh, i.e. about one
  count per minute at full fire: enough for 5-minute steps once smoothed.
* The water heater appears electric (`sensor.vue2_hot_water_tank_power`),
  so gas is mostly the furnace. Other gas appliances (stove, fireplace)
  would add noise.
* `sensor.get_energy_gas_consumption` (GJ, billing period) could later
  supply the price.

## Design

### 1. Collection (`core/resample.py`, `core/signals.py`, `coordinator.py`)

* `Signal` gains `counter: bool = False`.
* New `counter_rate(times, values, start, n_bins, step)`: hold the counter
  value at every bin edge (zero-order hold, as the recorder stores it),
  take differences per bin and divide by the step in hours, giving
  units/h. Negative differences (meter reset or replacement) and unknown
  edges give NaN.
* `_fetch_block` calls `counter_rate` instead of `time_weighted_mean` for
  counter signals.
* Scale to kW of heat with `Signal.scale` = kWh per unit (the rate is linear
  in the scale):

  | unit | kWh per unit |
  |---|---|
  | m³ | 10.55 |
  | ft³ | 0.3039 (1037 BTU/ft³) |
  | CCF | 30.39 |
  | MCF | 303.9 |
  | therm | 29.307 |
  | kWh / MWh | 1 / 1000 |
  | MJ / GJ | 1/3.6 / 277.8 |

  The unit comes from the entity registry or state attribute. An option
  `gas_unit` (auto / m³ / ft³ / CCF / therm / kWh / MJ / GJ) overrides it
  (needed here: set ft³).
* Dataset column `gas:<entity_id>`, label "Gas heat", sign `positive`, in kW.
  The raw per-step rate is stored; smoothing happens in the view (below), so
  it can be retuned without losing data.

### 2. Model inputs (`coordinator.py`)

* When a gas meter is configured, `ModelSpec.inputs` uses `gas:<id>` and
  **drops** `heating:<climate>` (both would be collinear). The heating duty
  column is still collected for the thermostat-agreement attribute and as a
  fallback.
* `coordinator.view()` replaces the gas column with a centred 3-step
  (15 min) moving average. Lumpy counter reports otherwise look like noise.
  Centred is fine for fitting; the planner never feeds measured gas forward.
* Bump `FIT_VERSION` so existing models refit with the new input.

### 3. Furnace capacity (`coordinator._fit_and_validate`, `core/model.py`)

* `ThermalModel` gains `input_capacity: dict[str, float]` (stored with the
  model).
* Capacity of the gas input = 90th percentile of smoothed gas kW over steps
  where the heating duty is ≥ 0.8. It needs at least ~2 h of such steps;
  otherwise it is unknown and the planner falls back to duty mode (below).

### 4. Planner (`core/mpc.py`, `planner.py`)

* `plan()` gets a per-decision input scale: a heating decision `x ∈ [0, 1]`
  drives the heat input at `x * capacity` kW instead of `x`. Outputs keep
  duty 0–1 plus `heat_capacity_kw`.
* `heat_col` = gas column if it is in `model.inputs` and has a capacity,
  else the heating-duty column (today's behaviour).
* Reported per plan: `heating_kwh = Σ duty · block_h · capacity`,
  `gas_volume = heating_kwh / kWh_per_unit` (in the meter's unit) and,
  if a price is set, `gas_cost`.
* New options (Planner page): `gas_price` per unit (currency from
  `hass.config.currency`). The energy weight keeps its meaning (per hour of
  full-fire heating), so tuning carries over.

### 5. Config / options (`config_flow.py`, strings)

* Entities page: optional `gas_meter` (EntitySelector, `sensor`,
  device_class `gas` or `energy`) and `gas_unit` (SelectSelector, default
  auto).
* Planner page: optional `gas_price`.

### 6. Sensors and panel

* New sensor **Planned gas (24 h)** in the meter's unit, with
  `heating_kwh`, `cost` and `capacity_kw` attributes.
* **Recommended action** attributes gain `planned_gas` and `planned_cost`.
* Panel plan tile "Planned heating": `4.5 h · 42 ft³ · $0.61`.
* The heat budget shows "Gas heat" instead of "Heating"; the gain is in K/h
  per kW.

## Tests

* `counter_rate`: steady counting gives a constant rate; a counter reset
  gives NaN; an unavailable gap gives NaN edges; reports at irregular times
  give the right average.
* Unit table and override (the CCF-labelled ft³ counter gives ~17 kW at
  60 ft³/h).
* Fit on a synthetic two-stage furnace: low stage 60 %, high stage 100 % of
  capacity; `hvac_action` shows only on/off. With gas as input, the 6 h
  prediction error must be clearly lower than with duty.
* Capacity estimate within 10 % of the true full-fire rate.
* Planner in gas mode: duty × capacity drives the model; gas volume and cost
  are reported; falls back to duty mode without a capacity.
* Integration: options flow stores `gas_meter` / `gas_unit` / `gas_price`;
  the spec drops the heating-duty input when gas is set.

## Risks / open questions

* Other gas loads (stove, fireplace, gas dryer) contaminate the signal. A
  later option could subtract a baseline, or only count gas while
  `hvac_action == heating`.
* Delivered heat lags gas burn by a few minutes (heat exchanger, ducts).
  The 15-minute smoothing covers most of it; a thermal-mass node already
  models the slower part.
* If the counter label is fixed upstream to real CCF, the unit override must
  be switched back, or auto-detection added: flag capacity > 60 kW as
  implausible for a house.
