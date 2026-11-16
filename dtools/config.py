"""Locations of local game data, taken from environment variables.

Game data (ROM sets, disc images) is never part of the repository; every
consumer resolves it through these helpers so tests can skip cleanly when
the data is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def rom_dir() -> Path | None:
    """Directory holding daytona.zip and model1io.zip (DAYTONA_ROMDIR)."""
    return _env_path("DAYTONA_ROMDIR")


def dc_path() -> Path | None:
    """Dreamcast disc: 'Daytona USA (USA).cue' or the Redump .7z (DAYTONA_DC)."""
    return _env_path("DAYTONA_DC")


def out_dir() -> Path:
    """Scratch output directory (DAYTONA_OUT, default <repo>/out); created on demand."""
    d = _env_path("DAYTONA_OUT") or REPO_ROOT / "out"
    d.mkdir(parents=True, exist_ok=True)
    return d


CUSTOM_MAME_PATCH = Path("tools") / "mame-daytona-rom-space.patch"
"""The ROM-space patch, relative to the repo: tools/build-mame.sh applies it to build/mame."""
CUSTOM_MAME_STAMP = Path("build") / "mame" / ".daytona-rom-space"
"""Written by tools/build-mame.sh after a successful build: the SHA-256 (hex) of the patch the binary was built from."""


def custom_mame() -> Path | None:
    """The custom MAME build (larger `daytona` ROM regions, docs/spike/2026-11-sixteen.md "Room"), or None.

    DAYTONA_CUSTOM_MAME when set (None unless it is an executable file); else <repo>/build/mame/model2 when it is
    executable and the build stamp names the SHA-256 of the current patch (a build of another patch, or one not
    finished, is not it)."""
    import hashlib
    env = _env_path("DAYTONA_CUSTOM_MAME")
    if env is not None:
        return env if env.is_file() and os.access(env, os.X_OK) else None
    exe = REPO_ROOT / "build" / "mame" / "model2"
    patch, stamp = REPO_ROOT / CUSTOM_MAME_PATCH, REPO_ROOT / CUSTOM_MAME_STAMP
    if not (exe.is_file() and os.access(exe, os.X_OK) and patch.is_file() and stamp.is_file()):
        return None
    return exe if stamp.read_text(encoding="utf-8").strip() == hashlib.sha256(patch.read_bytes()).hexdigest() else None
