import numpy as np
import pytest

from custom_components.thermal_mpc.core import fusion
from custom_components.thermal_mpc.core.dataset import Dataset

ZONES = {"zone:living": ["room:a", "room:b", "room:c"], "zone:bed": ["room:d"]}


def room(days=6, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 288
    h = np.arange(n) / 12
    truth = 21 + np.sin(2 * np.pi * h / 24) + 0.3 * np.sin(2 * np.pi * h / 5)
    solar = np.clip(3 * np.sin(2 * np.pi * (h % 24 - 6) / 24), 0, None)
    sun = fusion.smooth_sun(solar, 1 / 12, n)
    ds = Dataset(step=300.0)
    ds.append(
        0.0,
        {
            "room:a": truth + rng.normal(0, 0.03, n),
            "room:b": truth + 0.5 + 0.4 * sun + rng.normal(0, 0.03, n),
            "room:c": np.round((truth + rng.normal(0, 0.03, n)) * 2) / 2,
            "room:d": truth - 1,
            "solar": solar,
        },
    )
    return ds, truth


def test_learns_relative_bias_sun_and_noise():
    ds, _ = room()
    cal = fusion.learn(ds, ZONES, "solar")["zone:living"]
    assert cal["room:b"].bias - cal["room:a"].bias == pytest.approx(0.5, abs=0.05)
    assert cal["room:b"].sun - cal["room:a"].sun == pytest.approx(0.4, abs=0.05)
    assert cal["room:c"].weight < cal["room:a"].weight / 10  # 0.5 K steps
    assert fusion.learn(ds, ZONES, "solar")["zone:bed"]["room:d"].bias == 0.0


def test_fused_series_tracks_truth_and_survives_dropout():
    ds, truth = room()
    cal = fusion.learn(ds, ZONES, "solar")
    ds.columns["room:a"][1000:1200] = np.nan  # best sensor offline for a while
    fused = fusion.fuse(ds, ZONES, cal, "solar")["zone:living"]
    err = fused - truth
    # A constant offset is expected (relative calibration); wiggle is not.
    assert np.std(err) < 0.03
    # No level jump when the best sensor drops out or comes back.
    for edge in (1000, 1200):
        jump = (fused[edge] - fused[edge - 1]) - (truth[edge] - truth[edge - 1])
        assert abs(jump) < 0.1


def test_without_solar_and_too_little_overlap():
    ds, _ = room(days=6)
    cal = fusion.learn(ds, ZONES)
    assert cal["zone:living"]["room:b"].sun == 0.0
    short = Dataset(step=300.0)
    short.append(0.0, {k: v[:100] for k, v in ds.columns.items()})
    cal = fusion.learn(short, ZONES, "solar")["zone:living"]
    assert all(c.bias == 0.0 for c in cal.values())


def test_fuse_now_and_roundtrip():
    ds, _ = room()
    cal = fusion.learn(ds, ZONES, "solar")
    back = fusion.from_dict(fusion.to_dict(cal))
    assert back["zone:living"]["room:b"].bias == cal["zone:living"]["room:b"].bias
    now = fusion.fuse_now(
        {"room:a": 21.0, "room:b": 21.5, "room:c": float("nan"), "room:d": 20.0},
        ZONES,
        cal,
        0.0,
    )
    assert now["zone:bed"] == 20.0
    assert now["zone:living"] == pytest.approx(
        21.0 - cal["zone:living"]["room:a"].bias, abs=0.05
    )
