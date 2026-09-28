# Thermal MPC for Home Assistant

A custom integration that learns a **multi-room thermal model of your house**
from data already in the recorder. It learns heat loss to outdoors, heat
exchange between rooms, thermal inertia, solar gain, and the effect of the
furnace/AC, fan and HRV. The goal (next milestone) is model-predictive control
that uses the weather and solar forecast to drive the thermostat, fan and HRV:
consistent temperatures across rooms at minimum energy.

Everything runs inside Home Assistant. It needs no tokens, add-ons or
external services.

## Install (HACS)

1. HACS → ⋮ → *Custom repositories* → add `https://github.com/astrohome/ha-thermal-mpc`,
   category **Integration**.
2. Install **Thermal MPC** and restart Home Assistant.
3. *Settings → Devices & services → Add integration → Thermal MPC* and pick:
   * **Room temperature sensors**: one per room, dedicated wall sensors work best
   * **Outdoor temperature**: a sensor, or a weather entity (its `temperature`
     attribute is used)
   * **Thermostat**: its `hvac_action` gives the heating/cooling duty cycle
   * *optional* **Solar power**: PV output as a proxy for sunlight (W or kW)
   * *optional* **Fan** / **Ventilation**: anything whose state is `on` while
     running, e.g. `sensor.aprilaire_fan_status`, `sensor.aprilaire_ventilation_status`

On first start it backfills up to 60 days from the recorder in the
background. After that it collects every 15 minutes and keeps its own
5-minute training set in `.storage` for 120 days, so training data outlives
the recorder's purge window.

### Several sensors per room

Add as many temperature sensors as you have. Sensors in the same Home
Assistant **area** are treated as one room (you can turn this off in the
options). For each sensor the integration learns, by comparing it with the
others in the room:

* **offset:** it reads consistently high or low,
* **sun exposure:** it warms up when the sun is out (a plant pot by a
  window); the least exposed sensor counts as shaded,
* **noise:** jumpy or coarse sensors (a thermostat's 0.5 °C steps) count
  less.

The room temperature is the calibrated, noise-weighted blend of whichever
sensors are online, so one sensor dropping out doesn't break the room. The
panel's room cards list each sensor's share, offset, sun exposure and noise.

## What you get

| Entity | Meaning |
|---|---|
| `sensor.thermal_model_model_status` | `collecting` → `trained` (or `error`, with `message` attribute) |
| `sensor.thermal_model_training_data` | days of complete rows; `coverage` attribute per signal |
| `sensor.thermal_model_prediction_error_6_h` | worst-room error 6 h ahead on held-out data; attributes per room at 1/3/6 h |
| `sensor.thermal_model_<room>_time_constant` | hours to lose ~63 % of the room's lead over outdoors (thermal inertia); attributes hold heating/cooling/solar/fan/HRV gains in K/h and room-to-room coupling |
| `button.thermal_model_retrain_model` | fit now instead of waiting for the daily refit (also on the panel) |

The first fit happens once there are 3 days of complete data, then daily.
Treat the model as usable for control when the 6 h prediction error is below
~0.3 K. **Download diagnostics** on the integration exports the model, its
validation and the full training set for offline analysis.

## Thermal model panel

The integration adds a **Thermal model** entry to the sidebar. It shows:

* **Where the heat goes.** Rooms and outdoors drawn as a flow diagram. Animated
  arrows point the way heat is moving, and their thickness and labels give how
  fast each path changes the room's temperature (K/h). You can view it right
  now or as a 24 h average. A colored stripe shows how far each room is from
  the house average.
* **Heat budget per room.** Every path (outdoor, each neighbouring room,
  heating, cooling, sun, fan, HRV, unexplained baseline) as a diverging bar:
  blue cools the room, red warms it, and the net change comes last.
* **Model check.** Measured temperature against the model over the last 48 h.
  The model is restarted from the measurement every 6 h and runs on its own
  in between, which is how a controller would use it.

## The model

For each room *i* (coefficients divided by the room's heat capacity):

```
dT_i/dt = Σ_j g_ij (T_j − T_i) + g_io (T_out − T_i) + h_i (M_i − T_i) + Σ_u b_iu · u + c_i
dM_i/dt = k_i (T_i − M_i) + s_i · solar
```

* `g_io`: conductance to outdoors. `1/g_io` is the time constant.
* `g_ij`: coupling between rooms (walls, open doors, stairwells).
* `M_i`: optional hidden **thermal mass** (walls, floor, furniture). It
  stores heat, including sun landing on the floor, and gives it back slowly.
  It isn't measured; it's reconstructed from the room's air temperature and
  solar power. `1/k_i` is how fast it follows the air.
* `u`: inputs time-averaged over each 5-minute step, i.e. heating/cooling
  duty cycle, PV power, fan and HRV duty.
* `c_i`: constant internal gains or sensor bias.

Fitting runs in two stages, in plain numpy:

1. **Integral regression.** Regress hour-long temperature changes on the
   summed regressors, with signs constrained (conductances ≥ 0, heating ≥ 0,
   cooling ≤ 0). Regressing noisy 5-minute differences instead biases slow
   rooms to "no heat loss at all".
   The thermal-mass time constant is picked from a grid (2–32 h, or none) by
   open-loop error. With it fixed, the mass is a filtered copy of measured
   signals, so the air equation stays linear.
2. **Multi-step refinement.** Projected Levenberg-Marquardt minimises the
   6 h open-loop prediction error over overlapping segments, the error that
   matters for planning.

The integration trains on 80 % of the data, reports the prediction error on
the last 20 %, then refits on everything. Physical parameters carry over
between seasons far better than a black-box model, and the daily refit on a
rolling window follows seasonal drift.

`thermal_mpc.export_dataset` writes the training set and model to
`thermal_mpc_dataset.yaml` in the config folder for offline analysis.

## Shadow-mode planner

Every 15 minutes the integration plans heating and cooling duty for each hour
of the next 24 h. It uses the model, the hourly weather forecast, and a solar
forecast from any provider the Energy dashboard can use (e.g. Forecast.Solar).
Without forecasts it repeats yesterday. The model is linear, so each room's
path is the free-running forecast plus a response matrix times the duties.
The planner minimises:

* time outside the comfort band (target ± band; the target defaults to the
  thermostat's setpoint),
* spread between rooms,
* energy, with cooling priced lower while the sun covers the AC.

Tune the trade-offs in the integration's options (Planner page).

**Nothing is sent to the thermostat.** The plan is published as:

| Entity | Meaning |
|---|---|
| `sensor.thermal_model_recommended_action` | heat / cool / idle for this hour, plus whether the thermostat is doing the same thing |
| `sensor.thermal_model_recommended_setpoint` | setpoint that would make the thermostat follow the plan now (its reading ±1 K) |
| `sensor.thermal_model_planned_heating_24_h` | hours of full-duty heating planned |

The panel's **Next 24 hours** card shows planned room temperatures against
the comfort band, the hourly duty, and the outdoor forecast.

## Roadmap

1. ✅ Data collection and thermal model with validation
2. ✅ Thermal mass, forecast-driven prediction
3. ✅ Shadow-mode MPC. Compare its recommendations with the thermostat for a
   week or two.
4. ⏳ Closed loop: drive the setpoint with small offsets (±1–1.5 K, never
   +30/+15), with a watchdog that restores a safe setpoint if the
   integration stops.
5. ⏳ Fan circulation and HRV as planned actions (fan mixing needs a
   state-dependent term, not a constant gain).

A single-zone forced-air system cannot heat one room without heating the
others. The controller can narrow the spread between rooms through timing and
fan circulation; removing it needs zone dampers or smart vents.

See [docs/ha-setup.md](docs/ha-setup.md) for sensor placement and gaps worth
closing.

## Releases

Every push to `main` that passes the **Validate** workflow is released
automatically, so HACS offers numbered versions. The **Release** workflow
bumps the patch version from the latest `vX.Y.Z` tag, or the minor or major
version if a commit has a trailer line `Bump: minor` or `Bump: major`. If
`manifest.json` declares a higher version, it uses that instead. It stamps
the version into the manifest inside `thermal_mpc.zip`, which HACS installs.

## Development

```bash
pip install pytest-homeassistant-custom-component ruff
pytest && ruff check . && ruff format --check .
```
