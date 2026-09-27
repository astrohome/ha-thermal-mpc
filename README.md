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
dT_i/dt = Σ_j g_ij (T_j − T_i) + g_io (T_out − T_i) + Σ_u b_iu · u + c_i
```

* `g_io`: conductance to outdoors. `1/g_io` is the time constant.
* `g_ij`: coupling between rooms (walls, open doors, stairwells).
* `u`: inputs time-averaged over each 5-minute step, i.e. heating/cooling
  duty cycle, PV power, fan and HRV duty.
* `c_i`: constant internal gains or sensor bias.

Fitting runs in two stages, in plain numpy:

1. **Integral regression.** Regress hour-long temperature changes on the
   summed regressors, with signs constrained (conductances ≥ 0, heating ≥ 0,
   cooling ≤ 0). Regressing noisy 5-minute differences instead biases slow
   rooms to "no heat loss at all".
2. **Multi-step refinement.** Projected Levenberg-Marquardt minimises the
   6 h open-loop prediction error over overlapping segments, the error that
   matters for planning.

The integration trains on 80 % of the data, reports the prediction error on
the last 20 %, then refits on everything. Physical parameters carry over
between seasons far better than a black-box model, and the daily refit on a
rolling window follows seasonal drift.

`thermal_mpc.export_dataset` writes the training set and model to
`thermal_mpc_dataset.yaml` in the config folder for offline analysis.

## Roadmap

1. ✅ Data collection and thermal model with validation
2. ⏳ Forecast-driven prediction sensors (next 12–24 h per room)
3. ⏳ Shadow-mode MPC: log what it *would* do with setpoint offset, fan and HRV
4. ⏳ Closed loop with watchdog. It uses small setpoint offsets (±1–1.5 K),
   never +30/+15, so a two-stage furnace stays on low stage and the house is
   safe if the controller stops.

A single-zone forced-air system cannot heat one room without heating the
others. The controller can narrow the spread between rooms through timing and
fan circulation; removing it needs zone dampers or smart vents.

See [docs/ha-setup.md](docs/ha-setup.md) for sensor placement and gaps worth
closing.

## Development

```bash
pip install pytest-homeassistant-custom-component ruff
pytest && ruff check . && ruff format --check .
```
