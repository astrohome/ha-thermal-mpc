import numpy as np
import pytest

from custom_components.thermal_mpc.core.forecast import (
    hourly_energy_to_kw,
    interpolate,
    persistence,
)
from custom_components.thermal_mpc.core.model import RoomParams, ThermalModel
from custom_components.thermal_mpc.core.mpc import PlanInputs, PlanSettings, plan

MODEL = ThermalModel(
    step_h=5 / 60,
    outdoor="out",
    inputs=["heat", "cool", "solar"],
    rooms={
        "south": RoomParams(
            1 / 30,
            {"north": 0.1},
            {"heat": 1.5, "cool": -2.0, "solar": 0.6},
            0.0,
            0,
            0,
            mass_h=0.5,
            mass_k=1 / 8,
            mass_gains={"solar": 0.3},
        ),
        "north": RoomParams(
            1 / 20,
            {"south": 0.1},
            {"heat": 1.2, "cool": -1.5, "solar": 0.05},
            0.0,
            0,
            0,
        ),
    },
)
N = 288
STEP_S = 300


def inputs(t_out=-5.0, solar_peak=0.0, temps=(20.0, 19.5)):
    h = np.arange(N) / 12
    solar = np.clip(solar_peak * np.sin(2 * np.pi * (h - 6) / 24), 0, None)
    u = np.zeros((N, 3))
    u[:, 2] = solar
    return PlanInputs(
        start=0.0,
        temps=np.array(temps),
        mass=np.array(temps),
        t_out=np.full(N, t_out),
        u=u,
        solar_kw=solar,
    )


def test_simulate_many_matches_simulate():
    inp = inputs(solar_peak=3.0)
    ref = MODEL.simulate(inp.temps, inp.t_out, inp.u, inp.mass)[1:]
    many = MODEL.simulate_many(
        inp.temps[None], inp.mass[None], inp.t_out[None], inp.u[None]
    )[0]
    assert many == pytest.approx(ref)


def test_cold_day_heats_and_beats_free_running():
    out = plan(MODEL, inputs(), PlanSettings(target=21, band=0.5), "heat", "cool")
    assert out["action"] == "heat"
    assert out["heating_hours"] > 4
    assert out["discomfort_kh"]["planned"] < 0.2 * out["discomfort_kh"]["free"]
    assert all(0 <= d <= 1 for d in out["duty"]["heating"])
    assert out["duty"]["cooling"] == [0.0] * 24  # cooling not allowed


def test_sun_reduces_planned_heating():
    s = PlanSettings(target=21, band=0.5)
    dull = plan(MODEL, inputs(t_out=5.0), s, "heat", "cool")
    sunny = plan(MODEL, inputs(t_out=5.0, solar_peak=4.0), s, "heat", "cool")
    assert sunny["heating_hours"] < dull["heating_hours"]


def test_warm_day_cools_when_allowed():
    s = PlanSettings(target=22, band=0.5, allow_heat=False, allow_cool=True)
    out = plan(MODEL, inputs(t_out=32.0, temps=(24.5, 24.0)), s, "heat", "cool")
    assert out["action"] == "cool"
    assert out["cooling_hours"] > 0


def test_energy_weight_trades_comfort():
    cheap = plan(MODEL, inputs(), PlanSettings(energy_weight=0.01), "heat", "cool")
    dear = plan(MODEL, inputs(), PlanSettings(energy_weight=5.0), "heat", "cool")
    assert dear["heating_hours"] < cheap["heating_hours"]


def test_gas_mode_scales_duty_by_furnace_capacity():
    """Gas input in kW: duty x capacity drives the model like duty did."""
    capacity = 15.0
    rooms = {
        name: RoomParams(
            p.g_out,
            p.g_rooms,
            {
                ("gas" if k == "heat" else k): v / capacity if k == "heat" else v
                for k, v in p.gains.items()
            },
            p.offset,
            0,
            0,
            mass_h=p.mass_h,
            mass_k=p.mass_k,
            mass_gains=p.mass_gains,
        )
        for name, p in MODEL.rooms.items()
    }
    gas_model = ThermalModel(
        step_h=MODEL.step_h,
        outdoor="out",
        inputs=["gas", "cool", "solar"],
        rooms=rooms,
        input_capacity={"gas": capacity},
    )
    s = PlanSettings(target=21, band=0.5)
    duty = plan(MODEL, inputs(), s, "heat", "cool")
    gas = plan(gas_model, inputs(), s, "gas", "cool", heat_capacity_kw=capacity)
    assert gas["action"] == "heat"
    assert gas["duty"]["heating"] == pytest.approx(duty["duty"]["heating"], abs=1e-3)
    assert gas["heat_capacity_kw"] == capacity
    assert gas["heating_kwh"] == pytest.approx(gas["heating_hours"] * capacity)
    assert duty["heating_kwh"] is None
    # Without the capacity a duty of 1 would mean 1 kW: far too weak to help.
    weak = plan(gas_model, inputs(), s, "gas", "cool")
    assert weak["discomfort_kh"]["planned"] > 2 * gas["discomfort_kh"]["planned"]


def test_forecast_helpers():
    times = np.array([0.0, 1800.0, 3600.0, 7200.0])
    assert interpolate([(0, 0.0), (3600, 10.0)], times).tolist() == [0, 5, 10, 10]
    assert interpolate([], times) is None
    kw = hourly_energy_to_kw(
        {"1970-01-01T00:00:00+00:00": 1500, "1970-01-01T01:00:00+00:00": 500},
        times,
    )
    assert kw.tolist() == [1.5, 1.5, 0.5, 0.0]
    assert persistence(np.array([1.0, np.nan, 3.0]), 3, 5).tolist() == [
        1,
        2,
        3,
        1,
        2,
    ]
