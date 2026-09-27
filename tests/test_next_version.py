import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from next_version import next_version  # noqa: E402


def test_first_release_uses_manifest():
    assert next_version([], "0.6.1", ["anything"]) == "0.6.1"


def test_patch_bump_by_default():
    assert next_version(["v0.6.1", "v0.5.0"], "0.6.1", ["fix"]) == "0.6.2"


def test_minor_and_major_trailers():
    assert next_version(["v0.6.3"], "0.6.1", ["Feature\n\nBump: minor"]) == "0.7.0"
    assert next_version(["v0.6.3"], "0.6.1", ["x", "Break\n\nbump: MAJOR\n"]) == "1.0.0"


def test_prose_mentioning_markers_does_not_bump():
    """Regression: a commit describing the rule released v1.0.0 by accident."""
    msg = "Put [major] in a message, or write Bump: major on its own line."
    assert next_version(["v1.0.0"], "0.6.1", [msg]) == "1.0.1"


def test_manifest_can_force_a_higher_version():
    assert next_version(["v0.6.3"], "0.9.0", ["fix"]) == "0.9.0"


def test_ignores_non_version_tags_and_sorts_numerically():
    assert next_version(["v0.10.0", "v0.9.9", "latest"], "0.1.0", []) == "0.10.1"
