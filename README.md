# ha-thermal-mpc

Comfort-first, energy-aware heating, cooling and fresh-air control for Home
Assistant. A multi-room grey-box thermal model learns how the house gains and
loses heat (conductance between rooms and to outdoors, thermal inertia, solar
gain, furnace/AC effect). A model-predictive controller will then use the
weather and solar forecast to drive the AprilAire thermostat, fan and HRV.

Built for a single-zone forced-air house with an AprilAire 8800-series
thermostat (HVAC + HRV via `select.aprilaire_fresh_air_*`), Growatt/Emporia
Vue solar metering, and met.no + Forecast.Solar forecasts.

## Status

| Step | State |
|---|---|
| 1. Data collection: HA history → 5-min dataset | ✅ `thermal-mpc export` |
| 2. Thermal model fit + multi-step validation | ✅ `thermal-mpc fit` |
| 3. Shadow-mode MPC (log decisions, don't actuate) | ⏳ |
| 4. Closed loop with watchdog | ⏳ |

## Quick start

```bash
pip install -e '.[dev]'
cp config.example.yaml config.yaml          # entity IDs already filled in
export HA_TOKEN=...                         # HA profile → Security → long-lived token

thermal-mpc export --days 10 --append       # run daily; recorder only keeps 10 days
thermal-mpc fit                             # after ~2-3 weeks of data
```

`fit` trains on the first 80 % of the data and reports open-loop prediction
error on the last 20 %, e.g. how far off the model is 1 h and 6 h ahead when
run with no feedback. Aim for < 0.3 K at 6 h before trusting it for control.

See [docs/ha-setup.md](docs/ha-setup.md) for recorder retention and the
sensors worth adding.

## The model

For each room *i* (coefficients divided by the room's heat capacity):

```
dT_i/dt = Σ_j g_ij (T_j − T_i) + g_io (T_out − T_i) + Σ_u b_iu · u + c_i
```

* `g_io` — conductance to outdoors; `1/g_io` is the room's time constant
  (thermal inertia) in hours.
* `g_ij` — coupling between rooms (walls, open doors, stairwells).
* `u` — inputs, time-averaged per step: heating/cooling duty cycle, PV power
  (irradiance proxy), fan and HRV duty.
* `c_i` — constant internal gains / sensor bias.

It is linear in the parameters, so each room is fitted by bounded least
squares in milliseconds. The bounds enforce physics: conductances ≥ 0,
heating gain ≥ 0, cooling gain ≤ 0. Because the parameters are physical, a
model fitted in the autumn carries over to winter much better than a
black-box model would. Re-fitting weekly on a rolling window handles
seasonal drift such as sun angle and infiltration.

## Control plan (next)

Every 5–15 min the controller will solve a 12–24 h plan that minimises

```
comfort-band violation + λ₁·(room spread) + λ₂·(gas + electricity cost) + λ₃·(switching)
```

using met.no temperature and cloud cover and Forecast.Solar output, then
apply only the first step. The controls are:

* **Thermostat setpoint offset** — small (±1–1.5 K around the current
  reading), never +30/+15. That keeps a two-stage furnace on low stage and
  keeps the house safe if the controller dies. A watchdog restores a
  default setpoint if no command arrives for 20 min.
* **Fan** `auto`/`Circulate`/`on` — the only lever for evening out rooms
  in a single-zone system.
* **HRV** `select.aprilaire_fresh_air_mode` / `…_event` — ventilate when
  outdoor air is mildest (and, once a CO₂ sensor exists, when needed).

A single-zone system cannot heat one room without heating the others. The
controller reduces the spread between rooms through timing and mixing;
eliminating it needs zone dampers or smart vents.

## Development

```bash
pytest && ruff check . && ruff format --check .
```
