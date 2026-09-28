import numpy as np
import pytest

from custom_components.thermal_mpc.core import gas
from custom_components.thermal_mpc.core.dataset import Dataset
from custom_components.thermal_mpc.core.model import ModelSpec, fit, validate
from custom_components.thermal_mpc.core.resample import counter_rate
from custom_components.thermal_mpc.core.signals import Signal

STEP = 300.0
FULL_FIRE_KW = 20.0
FT3 = gas.KWH_PER_UNIT["ft³"]


def test_unit_table_and_aliases():
    assert gas.normalise_unit("ft³") == "ft³"
    assert gas.normalise_unit("ft3") == "ft³"
    assert gas.normalise_unit("ccf") == "CCF"
    assert gas.normalise_unit("m3") == "m³"
    assert gas.normalise_unit("gallons") is None
    assert gas.normalise_unit(None) is None
    assert gas.KWH_PER_UNIT["therm"] == pytest.approx(29.307)
    assert gas.KWH_PER_UNIT["GJ"] == pytest.approx(277.8)


def test_ccf_labelled_ft3_counter_with_override():
    """The meter says CCF but counts ft³; the override gives house-sized kW."""
    minutes = np.arange(0, 61)
    counter = 260_504 + minutes  # 60 ft³/h
    times = minutes * 60.0 - 30.0
    as_ft3 = Signal("sensor.gas", scale=FT3, counter=True)
    values = np.array([as_ft3.value(str(v), {}) for v in counter])
    kw = counter_rate(times, values, 0.0, 12, STEP)
    assert kw == pytest.approx(np.full(12, 60 * 0.3039))  # ~18 kW
    as_ccf = counter_rate(times, counter * gas.KWH_PER_UNIT["CCF"], 0.0, 12, STEP)
    assert as_ccf[0] > 1000  # 1.8 MW: clearly not a house


def test_smooth_is_centred_and_keeps_gaps():
    x = np.array([0.0, 3.0, 0.0, 0.0, np.nan, 6.0])
    out = gas.smooth(x)
    assert out[:4].tolist() == pytest.approx([1.5, 1.0, 1.0, 0.0])
    assert np.isnan(out[4])
    assert out[5] == 6.0


def two_stage_house(seed: int, days: int = 14) -> Dataset:
    """Two rooms heated by a two-stage furnace.

    Each heating cycle runs on the low stage (60 %) or the high stage
    (100 %) at random; ``hvac_action`` only shows on/off. The gas meter
    counts whole ft³ and reports only when the count changes.
    """
    rng = np.random.default_rng(seed)
    n = days * 288
    h = np.arange(n) / 12
    t_out = -2 + 6 * np.sin(2 * np.pi * (h - 9) / 24)
    temps = np.empty((n, 2))
    temps[0] = [20.5, 20.0]
    duty = np.zeros(n)
    heat_kw = np.zeros(n)
    stage = 1.0
    for k in range(n - 1):
        prev = duty[k - 1] if k else 0.0
        a, b = temps[k]
        duty[k] = 1.0 if a < 20.3 else (0.0 if a > 21.0 else prev)
        if duty[k] and not prev:
            stage = 0.6 if rng.random() < 0.6 else 1.0
        heat_kw[k] = duty[k] * stage * FULL_FIRE_KW
        temps[k + 1] = temps[k] + (1 / 12) * np.array(
            [
                (t_out[k] - a) / 30 + (b - a) / 5 + 0.12 * heat_kw[k],
                (t_out[k] - b) / 40 + (a - b) / 5 + 0.08 * heat_kw[k],
            ]
        )
    counter = np.floor(np.concatenate([[0.0], np.cumsum(heat_kw / 12)]) / FT3)
    reports = np.flatnonzero(np.diff(counter, prepend=-1))
    kw = counter_rate(reports * STEP, counter[reports] * FT3, 0.0, n, STEP)
    ds = Dataset(step=STEP)
    ds.append(
        0.0,
        {
            "a": temps[:, 0] + rng.normal(0, 0.03, n),
            "b": temps[:, 1] + rng.normal(0, 0.03, n),
            "out": t_out,
            "duty": duty,
            "gas": gas.smooth(kw),
        },
    )
    return ds


def test_capacity_estimate():
    ds = two_stage_house(0)
    cap = gas.capacity(ds.columns["gas"], ds.columns["duty"], STEP / 3600)
    assert cap == pytest.approx(FULL_FIRE_KW, rel=0.1)
    # Not enough full-fire time yet: unknown.
    short = slice(0, 20)  # 100 min
    assert gas.capacity(ds.columns["gas"][short], np.ones(20), STEP / 3600) is None


def test_gas_input_predicts_two_stage_furnace_better_than_duty():
    ds = two_stage_house(1)
    split = int(ds.rows * 0.8)
    errors = {}
    for heat in ("gas", "duty"):
        spec = ModelSpec(rooms=["a", "b"], outdoor="out", inputs={heat: "positive"})
        model = fit(ds, spec, slice(0, split), mass_grid_h=(None,))
        curves = validate(model, ds, slice(split, None))
        errors[heat] = max(c[-1] for c in curves.values())
    assert errors["gas"] < 0.6 * errors["duty"]
