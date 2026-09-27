import numpy as np
import pytest

from custom_components.thermal_mpc.core.dataset import Dataset
from custom_components.thermal_mpc.core.identify import Layout, simulate_batch
from custom_components.thermal_mpc.core.model import ModelSpec, fit, validate

SPEC = ModelSpec(
    rooms=["a", "b", "c"],
    outdoor="out",
    inputs={"heat": "positive", "solar": "positive", "vent": "free"},
)
TRUE_TAU = {"a": 30.0, "b": 60.0, "c": 25.0}


def noisy_house(seed: int, days: int = 14) -> Dataset:
    """3 rooms, 0.05 K sensor noise, outdoor seen as hourly steps with error."""
    rng = np.random.default_rng(seed)
    n = days * 288
    h = np.arange(n) / 12
    true_out = 8 + 7 * np.sin(2 * np.pi * (h - 9) / 24)
    cloud = rng.uniform(0.2, 1, days + 1).repeat(288)[:n]
    solar = np.clip(3 * np.sin(2 * np.pi * (h % 24 - 6) / 24), 0, None) * cloud
    g_out = np.array([1 / TRUE_TAU[r] for r in "abc"])
    C = np.zeros((3, 3))
    for a, b, g in [(0, 1, 0.12), (1, 2, 0.08)]:
        C[a, b] = C[b, a] = g
    hg, sg, vg = np.array([1.5, 1, 1.2]), np.array([0.5, 0.05, 0.3]), -0.15
    T = np.empty((n, 3))
    T[0] = [21, 20, 19.5]
    heat = np.zeros(n)
    vent = (rng.random(n) > 0.7).astype(float)
    for k in range(n - 1):
        prev = heat[k - 1] if k else 0.0
        heat[k] = 1.0 if T[k, 0] < 20.5 else (0.0 if T[k, 0] > 21.3 else prev)
        d = g_out * (true_out[k] - T[k]) + C @ T[k] - C.sum(1) * T[k]
        d += hg * heat[k] + sg * solar[k] + vg * vent[k]
        T[k + 1] = T[k] + d / 12
    err = np.repeat(rng.normal(0, 0.6, n // 12 + 1), 12)[:n]
    seen_out = np.repeat((true_out + err)[::12], 12)[:n]
    ds = Dataset(step=300.0)
    ds.append(
        0.0,
        {
            **{r: T[:, k] + rng.normal(0, 0.05, n) for k, r in enumerate("abc")},
            "out": seen_out,
            "heat": heat,
            "solar": solar,
            "vent": vent,
        },
    )
    return ds


@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_sensors_do_not_collapse_time_constants(seed):
    """Regression: one-step fits drove slow rooms' g_out to exactly zero."""
    ds = noisy_house(seed)
    split = int(ds.rows * 0.8)
    model = fit(ds, SPEC, slice(0, split))
    for room, tau in TRUE_TAU.items():
        assert model.rooms[room].tau_out_h == pytest.approx(tau, rel=0.15)
    rmse = validate(model, ds, slice(split, None))
    assert max(curve[-1] for curve in rmse.values()) < 0.15


def test_batch_simulation_matches_reference():
    ds = noisy_house(0, days=4)
    model = fit(ds, SPEC, horizon_h=None)
    layout = Layout(SPEC.rooms, SPEC.neighbours(), model.inputs, dict(SPEC.inputs))
    theta = []
    for r in SPEC.rooms:
        p = model.rooms[r]
        theta += [p.g_out, *[p.g_rooms[o] for o in SPEC.neighbours()[r]]]
        theta += [p.gains[i] for i in model.inputs] + [p.offset]
    x0 = np.array([[21.0, 20.0, 19.0]])
    t_out = ds.columns["out"][None, :50]
    u = np.stack([ds.columns[i][:50] for i in model.inputs], axis=-1)[None]
    batch = simulate_batch(np.array(theta), layout, x0, t_out, u, model.step_h)
    ref = model.simulate(x0[0], t_out[0], u[0])[1:]
    assert batch[0] == pytest.approx(ref)
