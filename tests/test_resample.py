import numpy as np
import pandas as pd
import pytest

from thermal_mpc.config import Config, Signal
from thermal_mpc.ha_history import build_dataset
from thermal_mpc.resample import time_weighted_mean

T0 = pd.Timestamp("2026-01-01T00:00:00Z")
STEP = pd.Timedelta("5min")


def ts(minutes: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes)


def test_duty_cycle():
    # Off before window, on at minute 2, off at minute 7.
    s = pd.Series([0.0, 1.0, 0.0], index=[ts(-30), ts(2), ts(7)])
    out = time_weighted_mean(s, T0, ts(15), STEP)
    assert out.tolist() == pytest.approx([0.6, 0.4, 0.0])


def test_value_before_window_is_held():
    s = pd.Series([21.0], index=[ts(-100)])
    out = time_weighted_mean(s, T0, ts(10), STEP)
    assert out.tolist() == [21.0, 21.0]


def test_unavailable_and_coverage():
    # Valid for 1 min of the first bin, then unavailable until minute 7.
    s = pd.Series([20.0, np.nan, 22.0], index=[ts(-1), ts(1), ts(7)])
    out = time_weighted_mean(s, T0, ts(10), STEP)
    assert np.isnan(out.iloc[0])  # 20 % coverage < 50 %
    assert out.iloc[1] == 22.0  # 60 % coverage, only valid value is 22


def test_no_data_before_first_row():
    s = pd.Series([19.0], index=[ts(6)])
    out = time_weighted_mean(s, T0, ts(10), STEP)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 19.0


def test_signal_mapping_handles_yaml_booleans():
    sig = Signal.from_dict({"entity": "sensor.x", "map": {True: 1.0}, "default": 0})
    assert sig.value("on", {}) == 1.0
    assert sig.value("off", {}) == 0.0
    assert np.isnan(sig.value("unavailable", {}))


def test_build_dataset_from_history_rows():
    config = Config.from_dict(
        {
            "rooms": {"a": {"entity": "sensor.a"}},
            "outdoor": {"entity": "weather.home", "attribute": "temperature"},
            "inputs": {
                "heating": {
                    "entity": "climate.t",
                    "attribute": "hvac_action",
                    "map": {"heating": 1},
                    "default": 0,
                }
            },
        }
    )

    def row(entity, minute, state, **attrs):
        return {
            "entity_id": entity,
            "state": state,
            "attributes": attrs,
            "last_updated": ts(minute).isoformat(),
        }

    rows = {
        "sensor.a": [row("sensor.a", -1, "20.5")],
        "weather.home": [row("weather.home", -1, "cloudy", temperature=-5)],
        "climate.t": [
            row("climate.t", -1, "heat", hvac_action="idle"),
            row("climate.t", 5, "heat", hvac_action="heating"),
        ],
    }
    df = build_dataset(config, rows, T0, ts(10))
    assert df["T_a"].tolist() == [20.5, 20.5]
    assert df["T_out"].tolist() == [-5.0, -5.0]
    assert df["heating"].tolist() == [0.0, 1.0]
