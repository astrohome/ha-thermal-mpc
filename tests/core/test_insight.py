import numpy as np
import pytest

from custom_components.thermal_mpc.core.dataset import Dataset
from custom_components.thermal_mpc.core.insight import budget, mean_budget, replay
from custom_components.thermal_mpc.core.model import RoomParams, ThermalModel

MODEL = ThermalModel(
    step_h=5 / 60,
    outdoor="out",
    inputs=["heat"],
    rooms={
        "a": RoomParams(0.1, {"b": 0.5}, {"heat": 2.0}, 0.01, 0, 0),
        "b": RoomParams(0.2, {"a": 0.0}, {"heat": 1.0}, 0.0, 0, 0),
    },
)


def test_budget_terms_sum_to_derivative():
    temps = {"a": 21.0, "b": 19.0}
    out = budget(MODEL, temps, 5.0, {"heat": 0.5})
    a = out["a"]
    assert a["outdoor"] == pytest.approx(0.1 * (5 - 21))
    assert a["rooms"]["b"] == pytest.approx(0.5 * (19 - 21))
    assert a["inputs"]["heat"] == pytest.approx(1.0)
    expected = MODEL.derivative(np.array([21.0, 19.0]), 5.0, np.array([0.5]))
    assert a["net"] == pytest.approx(expected[0])
    assert out["b"]["net"] == pytest.approx(expected[1])


def test_budget_unknown_input_gives_none():
    out = budget(MODEL, {"a": 21.0, "b": 19.0}, 5.0, {"heat": float("nan")})
    assert out["a"]["inputs"]["heat"] is None
    assert out["a"]["net"] is None


def _dataset(n=600):
    rng = np.random.default_rng(0)
    heat = (rng.random(n) > 0.5).astype(float)
    t_out = np.full(n, 5.0)
    temps = np.empty((n, 2))
    temps[0] = [21, 19]
    for k in range(n - 1):
        temps[k + 1] = temps[k] + MODEL.step_h * MODEL.derivative(
            temps[k], t_out[k], np.array([heat[k]])
        )
    ds = Dataset(step=300.0)
    ds.append(0.0, {"a": temps[:, 0], "b": temps[:, 1], "out": t_out, "heat": heat})
    return ds, temps


def test_mean_budget_matches_mean_derivative():
    ds, temps = _dataset()
    out = mean_budget(MODEL, ds, slice(0, ds.rows))
    derivs = [
        MODEL.derivative(temps[k], 5.0, np.array([ds.columns["heat"][k]]))
        for k in range(ds.rows)
    ]
    assert out["a"]["net"] == pytest.approx(np.mean(derivs, axis=0)[0])
    assert out["a"]["samples"] == ds.rows


def test_replay_is_exact_for_the_true_model():
    ds, temps = _dataset()
    rep = replay(MODEL, ds, hours=24, segment_h=6)
    n = round(24 / MODEL.step_h)
    assert len(rep["times"]) == n
    pred = np.array(rep["predicted"]["a"], dtype=float)
    assert pred == pytest.approx(temps[-n:, 0], abs=1e-3)


def test_replay_stops_at_missing_input():
    ds, _ = _dataset()
    ds.columns["heat"][-10] = np.nan
    rep = replay(MODEL, ds, hours=6, segment_h=6)
    pred = rep["predicted"]["a"]
    assert pred[-11] is not None
    assert all(v is None for v in pred[-10:])
