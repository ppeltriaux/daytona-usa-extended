import shutil
from pathlib import Path

import pytest

from dtools import config


def pytest_runtest_setup(item):
    """A test marked `mame` starts MAME: skipped when mame is not on PATH. One marked `custom_mame` needs the custom
    build: skipped when dtools.config.custom_mame() finds none (tools/build-mame.sh)."""
    if item.get_closest_marker("mame") and shutil.which("mame") is None:
        pytest.skip("mame is not on PATH")
    if item.get_closest_marker("custom_mame") and config.custom_mame() is None:
        pytest.skip("the custom MAME is not built (tools/build-mame.sh)")


@pytest.fixture
def romset_zip() -> Path:
    d = config.rom_dir()
    if d is None or not (d / "daytona.zip").is_file():
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return d / "daytona.zip"


@pytest.fixture
def dc_path() -> Path:
    p = config.dc_path()
    if p is None or not p.is_file():
        pytest.skip("set DAYTONA_DC to the Dreamcast disc .cue or .7z")
    return p
