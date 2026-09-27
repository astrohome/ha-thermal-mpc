import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from next_version import next_version  # noqa: E402


def test_first_release_uses_manifest():
    assert next_version([], "0.6.1", ["anything"]) == "0.6.1"


def test_patch_bump_by_default():
    assert next_version(["v0.6.1", "v0.5.0"], "0.6.1", ["fix"]) == "0.6.2"


def test_minor_and_major_markers():
    assert next_version(["v0.6.3"], "0.6.1", ["feat [minor]"]) == "0.7.0"
    assert next_version(["v0.6.3"], "0.6.1", ["x", "break [MAJOR]"]) == "1.0.0"


def test_manifest_can_force_a_higher_version():
    assert next_version(["v0.6.3"], "0.9.0", ["fix"]) == "0.9.0"


def test_ignores_non_version_tags_and_sorts_numerically():
    assert next_version(["v0.10.0", "v0.9.9", "latest"], "0.1.0", []) == "0.10.1"
