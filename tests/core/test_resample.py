import numpy as np
import pytest

from custom_components.thermal_mpc.core.dataset import Dataset
from custom_components.thermal_mpc.core.resample import time_weighted_mean
from custom_components.thermal_mpc.core.signals import Signal

STEP = 300.0


def m(minutes: float) -> float:
    return minutes * 60.0


def twm(times, values, n_bins):
    return time_weighted_mean(np.array(times), np.array(values), 0.0, n_bins, STEP)


def test_duty_cycle():
    out = twm([m(-30), m(2), m(7)], [0.0, 1.0, 0.0], 3)
    assert out.tolist() == pytest.approx([0.6, 0.4, 0.0])


def test_value_before_window_is_held():
    assert twm([m(-100)], [21.0], 2).tolist() == [21.0, 21.0]


def test_unavailable_and_coverage():
    # Valid 1 min of bin 0; unavailable until minute 7, so bin 1 is 60 % valid.
    out = twm([m(-1), m(1), m(7)], [20.0, np.nan, 22.0], 2)
    assert np.isnan(out[0])
    assert out[1] == 22.0


def test_no_data_before_first_change():
    out = twm([m(6)], [19.0], 2)
    assert np.isnan(out[0])
    assert out[1] == 19.0


def test_unsorted_and_duplicate_times():
    out = twm([m(2), m(-5), m(2)], [5.0, 1.0, 3.0], 1)
    assert out[0] == pytest.approx((1.0 * 2 + 3.0 * 3) / 5)


def test_signal_mapping():
    sig = Signal(
        "climate.t", attribute="hvac_action", mapping={"heating": 1.0}, default=0.0
    )
    assert sig.value("heat", {"hvac_action": "heating"}) == 1.0
    assert sig.value("heat", {"hvac_action": "idle"}) == 0.0
    assert np.isnan(sig.value("unavailable", {}))
    assert Signal("sensor.pv", scale=0.001).value("1500", {}) == 1.5
    assert np.isnan(Signal("sensor.x").value("abc", {}))


def test_dataset_append_trim_roundtrip():
    ds = Dataset(step=STEP)
    ds.append(0.0, {"a": np.array([np.nan, np.nan, 1.0])})
    ds.append(3 * STEP, {"a": np.array([2.0]), "b": np.array([5.0])})
    assert ds.rows == 4
    assert np.isnan(ds.columns["b"][:3]).all()
    ds.trim(10)  # drops the two all-NaN leading rows
    assert ds.start == 2 * STEP
    assert ds.columns["a"].tolist() == [1.0, 2.0]
    back = Dataset.from_dict(ds.to_dict())
    assert back.start == ds.start
    assert np.isnan(back.columns["b"][0])
    with pytest.raises(ValueError):
        ds.append(0.0, {"a": np.array([1.0])})
