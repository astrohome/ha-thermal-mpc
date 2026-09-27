import itertools

import numpy as np
import pytest

from custom_components.thermal_mpc.core.dataset import Dataset
from custom_components.thermal_mpc.core.model import (
    ModelSpec,
    NotEnoughDataError,
    RoomParams,
    ThermalModel,
    fit,
    nnls,
    signed_lstsq,
    validate,
)

SPEC = ModelSpec(
    rooms=["south", "north"],
    outdoor="out",
    inputs={"heating": "positive", "cooling": "negative", "solar": "positive"},
)

TRUE = ThermalModel(
    step_h=5 / 60,
    outdoor="out",
    inputs=["heating", "cooling", "solar"],
    rooms={
        "south": RoomParams(
            1 / 30,
            {"north": 1 / 4},
            {"heating": 2.0, "cooling": -1.5, "solar": 0.8},
            0.1,
            0,
            0,
        ),
        "north": RoomParams(
            1 / 20,
            {"south": 1 / 5},
            {"heating": 1.5, "cooling": -1.0, "solar": 0.1},
            0.1,
            0,
            0,
        ),
    },
)


def synthetic(days=21, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 288
    hours = np.arange(n) * TRUE.step_h
    t_out = 8 + 10 * np.sin(2 * np.pi * (hours - 9) / 24)
    t_out += rng.normal(0, 1, n).cumsum() * 0.05
    solar = np.clip(3 * np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0, None)
    solar *= rng.uniform(0.3, 1.0, n // 288 + 1).repeat(288)[:n]
    temps = np.empty((n + 1, 2))
    temps[0] = [20.0, 19.0]
    heat, cool = np.zeros(n), np.zeros(n)
    for k in range(n):
        t = temps[k, 1]
        prev_h = heat[k - 1] if k else 0.0
        prev_c = cool[k - 1] if k else 0.0
        heat[k] = 1.0 if t < 19.5 else (0.0 if t > 20.5 else prev_h)
        cool[k] = 1.0 if t > 23.5 else (0.0 if t < 22.5 else prev_c)
        u = np.array([heat[k], cool[k], solar[k]])
        temps[k + 1] = temps[k] + TRUE.step_h * TRUE.derivative(temps[k], t_out[k], u)
    ds = Dataset(step=300.0)
    ds.append(
        0.0,
        {
            "south": temps[:-1, 0] + rng.normal(0, noise, n),
            "north": temps[:-1, 1] + rng.normal(0, noise, n),
            "out": t_out,
            "heating": heat,
            "cooling": cool,
            "solar": solar,
        },
    )
    return ds


def brute_force_nnls(A, b):
    best, best_x = np.inf, None
    for support in itertools.product([False, True], repeat=A.shape[1]):
        s = np.array(support)
        x = np.zeros(A.shape[1])
        if s.any():
            x[s] = np.linalg.lstsq(A[:, s], b, rcond=None)[0]
        if (x < -1e-12).any():
            continue
        r = np.linalg.norm(A @ x - b)
        if r < best:
            best, best_x = r, x
    return best_x


@pytest.mark.parametrize("seed", range(20))
def test_nnls_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(30, 5))
    b = rng.normal(size=30)
    assert nnls(A, b) == pytest.approx(brute_force_nnls(A, b), abs=1e-8)


def test_signed_lstsq_respects_signs():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(200, 3))
    b = A @ np.array([-1.0, 2.0, 0.5]) + rng.normal(0, 0.01, 200)
    x = signed_lstsq(A, b, ["positive", "negative", "free"])
    assert x[0] == pytest.approx(0.0, abs=1e-12)  # true -1 clipped to 0
    assert x[1] == pytest.approx(0.0, abs=1e-12)  # true +2 clipped to 0
    unconstrained = signed_lstsq(A, b, ["free", "free", "free"])
    assert unconstrained == pytest.approx([-1.0, 2.0, 0.5], abs=0.01)


def test_recovers_parameters():
    model = fit(synthetic(), SPEC)
    for room, true in TRUE.rooms.items():
        got = model.rooms[room]
        assert got.g_out == pytest.approx(true.g_out, rel=0.05)
        for other, g in true.g_rooms.items():
            assert got.g_rooms[other] == pytest.approx(g, rel=0.05)
        for inp, b in true.gains.items():
            assert got.gains[inp] == pytest.approx(b, rel=0.05, abs=0.02)


def test_multistep_prediction_with_noise():
    ds = synthetic(noise=0.02)
    split = int(ds.rows * 0.8)
    model = fit(ds, SPEC, slice(0, split))
    rmse = validate(model, ds, slice(split, None), horizon_h=6)
    assert max(r[-1] for r in rmse.values()) < 0.3


def test_sparse_input_is_left_out():
    ds = synthetic(days=7)
    ds.columns["cooling"][:] = np.nan
    model = fit(ds, SPEC)
    assert model.unused_inputs == ["cooling"]
    assert "cooling" not in model.rooms["south"].gains


def test_roundtrip():
    back = ThermalModel.from_dict(TRUE.to_dict())
    assert back.rooms["south"].gains == TRUE.rooms["south"].gains


def test_not_enough_data():
    ds = synthetic(days=1)
    ds.trim(50)
    with pytest.raises(NotEnoughDataError):
        fit(ds, SPEC)
