"""Work out the next release version.

Rules:
* start from the newest ``vX.Y.Z`` tag (none yet: 0.0.0);
* bump patch, or minor/major if any commit since that tag has a line that is
  exactly ``Bump: minor`` / ``Bump: major`` (a trailer, so prose that merely
  mentions the markers does not count);
* if ``manifest.json`` already declares a higher version, use that instead
  (lets a human pick a version by editing the manifest).

Prints the version without the ``v``. Usage::

    python scripts/next_version.py <manifest.json>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
BUMP = re.compile(r"^\s*bump:\s*(major|minor)\s*$", re.IGNORECASE | re.MULTILINE)


def parse(version: str) -> tuple[int, int, int]:
    """Split ``X.Y.Z`` into integers."""
    major, minor, patch = (int(x) for x in version.split("."))
    return major, minor, patch


def next_version(tags: list[str], manifest_version: str, messages: list[str]) -> str:
    """Pure version rule, separated for testing."""
    versions = sorted(
        tuple(int(g) for g in m.groups()) for t in tags if (m := TAG.match(t))
    )
    major, minor, patch = versions[-1] if versions else (0, 0, 0)
    levels = {m.lower() for msg in messages for m in BUMP.findall(msg)}
    if "major" in levels:
        bumped = (major + 1, 0, 0)
    elif "minor" in levels:
        bumped = (major, minor + 1, 0)
    else:
        bumped = (major, minor, patch + 1)
    chosen = max(bumped, parse(manifest_version))
    return ".".join(str(x) for x in chosen)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def main() -> None:
    """Read tags, commit messages and the manifest from the checkout."""
    manifest = json.loads(Path(sys.argv[1]).read_text())
    tags = _git("tag", "--list", "v*").split()
    latest = max(
        (t for t in tags if TAG.match(t)),
        key=lambda t: tuple(int(g) for g in TAG.match(t).groups()),
        default=None,
    )
    log_range = f"{latest}..HEAD" if latest else "HEAD"
    messages = _git("log", "--format=%B%x00", log_range).split("\0")
    print(next_version(tags, manifest["version"], messages))


if __name__ == "__main__":
    main()
