import numpy as np
import pandas as pd
import pytest

from thermal_mpc.config import Config
from thermal_mpc.model import RoomParams, ThermalModel, fit, validate

CONFIG = Config.from_dict(
    {
        "step": "5min",
        "rooms": {"south": {"entity": "sensor.s"}, "north": {"entity": "sensor.n"}},
        "outdoor": {"entity": "weather.home"},
        "inputs": {
            "heating": {"entity": "climate.t", "sign": "positive"},
            "solar_kw": {"entity": "sensor.pv", "sign": "positive"},
        },
    }
)

TRUE = ThermalModel(
    step_h=5 / 60,
    inputs=["heating", "solar_kw"],
    rooms={
        "south": RoomParams(
            g_out=1 / 30,
            g_rooms={"north": 1 / 4},
            gains={"heating": 2.0, "solar_kw": 0.8},
            offset=0.1,
            rmse_one_step=0,
            n_samples=0,
        ),
        "north": RoomParams(
            g_out=1 / 20,
            g_rooms={"south": 1 / 5},
            gains={"heating": 1.5, "solar_kw": 0.1},
            offset=0.1,
            rmse_one_step=0,
            n_samples=0,
        ),
    },
)


def synthetic(days: int = 21, noise: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 288
    hours = np.arange(n) * TRUE.step_h
    t_out = (
        -5
        + 8 * np.sin(2 * np.pi * (hours - 9) / 24)
        + rng.normal(0, 1, n).cumsum() * 0.05
    )
    solar = np.clip(3 * np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0, None)
    solar *= rng.uniform(0.3, 1.0, n // 288 + 1).repeat(288)[:n]  # cloudy days
    # Crude bang-bang thermostat on the north room so heating is excited.
    temps = np.empty((n + 1, 2))
    temps[0] = [20.0, 19.0]
    heat = np.zeros(n)
    for k in range(n):
        heat[k] = (
            1.0 if temps[k, 1] < 19.5 else (0.0 if temps[k, 1] > 20.5 else heat[k - 1])
        )
        u = np.array([heat[k], solar[k]])
        temps[k + 1] = temps[k] + TRUE.step_h * TRUE.derivative(temps[k], t_out[k], u)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC", name="time")
    return pd.DataFrame(
        {
            "T_south": temps[:-1, 0] + rng.normal(0, noise, n),
            "T_north": temps[:-1, 1] + rng.normal(0, noise, n),
            "T_out": t_out,
            "heating": heat,
            "solar_kw": solar,
        },
        index=idx,
    )


def test_recovers_parameters():
    model = fit(synthetic(noise=0.0), CONFIG)
    for room, true in TRUE.rooms.items():
        got = model.rooms[room]
        assert got.g_out == pytest.approx(true.g_out, rel=0.05)
        for other, g in true.g_rooms.items():
            assert got.g_rooms[other] == pytest.approx(g, rel=0.05)
        for inp, b in true.gains.items():
            assert got.gains[inp] == pytest.approx(b, rel=0.05, abs=0.02)


def test_multistep_prediction_with_noise():
    df = synthetic(noise=0.02)
    split = int(len(df) * 0.8)
    model = fit(df.iloc[:split], CONFIG)
    rmse = validate(model, df.iloc[split:], horizon=pd.Timedelta(hours=6))
    assert rmse.iloc[-1].max() < 0.3


def test_json_roundtrip(tmp_path):
    path = tmp_path / "m.json"
    TRUE.to_json(path)
    loaded = ThermalModel.from_json(path)
    assert loaded.rooms["south"].gains == TRUE.rooms["south"].gains


def test_too_little_data():
    with pytest.raises(ValueError, match="collect more data"):
        fit(synthetic(days=1).iloc[:50], CONFIG)
