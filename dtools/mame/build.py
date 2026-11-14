"""Run a patched `daytona` set in MAME and compare snapshots.

`run_set` packages the Phase 3 pattern (write the set, copy `model1io.zip` beside it, run `autoplay.lua`) and
guards against MAME silently loading the unpatched parent set: MAME prints a `WRONG CHECKSUMS` block exactly when
the set it loaded differs from its known hashes. Lua errors inside tap callbacks are swallowed in MAME 0.289, so a
probe that stops logging is caught by the empty-log check.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from dtools.arcade.romset import CUSTOM_FILES, RomSet
from dtools.mame import runner
from dtools.mame.runner import MameError

LOG_ENVS = {"ground": "DAYTONA_GROUND_LOG", "rate": "DAYTONA_RATE_LOG", "geo": "DAYTONA_GEO_LOG",
            "progress": "DAYTONA_PROGRESS_LOG", "cells": "DAYTONA_CELLS_LOG"}


def run_set(rs: RomSet | None, workdir: Path, seconds: int, romset_zip: Path, env: dict[str, str],
            logs: Sequence[str] = (), mame: str | Path = "mame") -> dict[str, Path]:
    """Run `autoplay.lua` for `seconds` on `rs` (written to `workdir/roms/daytona.zip`, `model1io.zip` copied from
    beside `romset_zip`) or, with `rs` None, on the set in `romset_zip`'s directory, with the MAME binary `mame`
    (runner.run_mame; `config.custom_mame()` for the custom build). Returns the log paths named in `logs` ("ground",
    "rate", "geo", "progress", "cells") plus "snap" (the snapshot directory). MameError when `WRONG CHECKSUMS` is
    printed and `rs` is None, or is not printed and `rs` is a stock set, or when a log is missing or empty. A custom
    `rs` is proven loaded instead by MAME naming each of its CUSTOM_FILES "NO GOOD DUMP KNOWN" (the patch declares
    them NO_DUMP, so an unpatched custom set prints no WRONG CHECKSUMS): MameError naming the files it did not (stock
    MAME knows none of them)."""
    unknown = [n for n in logs if n not in LOG_ENVS]
    if unknown:
        raise MameError(f"unknown log name(s) {unknown}; known: {sorted(LOG_ENVS)}")
    workdir = Path(workdir)
    romset_zip = Path(romset_zip)
    if rs is None:
        rom_dir = romset_zip.parent
    else:
        rom_dir = workdir / "roms"
        rs.write_zip(rom_dir / "daytona.zip")
        shutil.copy(romset_zip.parent / "model1io.zip", rom_dir / "model1io.zip")
    paths = {n: workdir / f"{n}.txt" for n in logs}
    for p in paths.values():
        p.unlink(missing_ok=True)                    # a log left by an earlier run must not pass the check below
    full_env = {**env, **{LOG_ENVS[n]: str(p) for n, p in paths.items()}}
    proc = runner.run_mame(runner.LUA_DIR / "autoplay.lua", seconds, workdir / "run", rom_dir, env=full_env,
                           mame=mame)
    said = proc.stdout + proc.stderr
    seen = "WRONG CHECKSUMS" in said
    if seen and rs is None:
        raise MameError("MAME printed WRONG CHECKSUMS for the unpatched set")
    if rs is not None and rs.custom:
        unloaded = [n for n in CUSTOM_FILES if f"{n} NO GOOD DUMP KNOWN" not in said]
        if unloaded:
            raise MameError(f"MAME did not load the custom set's {', '.join(unloaded)} (is {mame} the custom "
                            f"build, tools/build-mame.sh?)")
    elif rs is not None and not seen:
        raise MameError("no WRONG CHECKSUMS block: MAME did not load the patched set")
    for n, p in paths.items():
        if not p.is_file() or p.stat().st_size == 0:
            raise MameError(f"empty log: {n}")
    return {**paths, "snap": workdir / "run" / "snap"}


def snapshot_match(a: Path, b: Path, box: tuple[int, int, int, int] | None = None, flip: str | None = None,
                   tol: int = 16, mask: np.ndarray | None = None) -> float:
    """Fraction of pixels in `box` (left, top, right, bottom; default the whole image) of `a` and `b` whose RGB
    channels all differ by at most `tol`, after flipping `b` inside the box (`"h"` left-right, `"v"` top-bottom).
    With `mask` (bool, the box's height x width, in `a`'s coordinates) only its pixels count. ValueError for a mask
    of another shape or without a pixel."""
    if flip not in (None, "h", "v"):
        raise ValueError(f"flip must be None, 'h' or 'v', not {flip!r}")
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
    if ia.shape != ib.shape:
        raise ValueError(f"image sizes differ: {ia.shape[1]}x{ia.shape[0]} and {ib.shape[1]}x{ib.shape[0]}")
    if box is not None:
        left, top, right, bottom = box
        ia, ib = ia[top:bottom, left:right], ib[top:bottom, left:right]
    if flip == "h":
        ib = ib[:, ::-1]
    elif flip == "v":
        ib = ib[::-1, :]
    if ia.size == 0:
        raise ValueError("empty box")
    ok = (np.abs(ia - ib) <= tol).all(axis=2)
    if mask is None:
        return float(ok.mean())
    mask = np.asarray(mask)
    if mask.shape != ok.shape or mask.dtype != bool or not mask.any():
        raise ValueError(f"mask of shape {mask.shape} ({mask.dtype}): expected a bool {ok.shape} with a pixel set")
    return float(ok[mask].mean())
