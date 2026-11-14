"""MAME tests on the custom build (tools/build-mame.sh with tools/mame-daytona-rom-space.patch): the larger `daytona`
ROM regions change nothing the original courses do. docs/spike/2026-11-sixteen.md, "Room"; Review Focus 2.

Marked `mame` (they start stock MAME too) and `custom_mame` (skipped when dtools.config.custom_mame() finds no build
of the current patch)."""

import hashlib
from pathlib import Path

import pytest

from dtools import config
from dtools.arcade import geo
from dtools.arcade.romset import CUSTOM_FREE, RomSet

SECONDS = 45                # boot, attract, both select screens and each course's race past its start (frame 2588)
GEO_FROM = 600              # every display list from the coin (frame 600) on: attract's end, the selects, the race
SNAPS = "650,760,960,1350,1700,2000,2300,2550"   # attract, Circuit / Mission Select, fly-by, race (autoplay frames)
TEX_GAPS = tuple((lo // 2, (lo + n) // 2) for lo, n in CUSTOM_FREE["textures"])   # word indices: the filled gaps


def _custom() -> Path:
    exe = config.custom_mame()
    if exe is None:
        pytest.skip("the custom MAME is not built (tools/build-mame.sh)")
    return exe


def _lists(path: Path):
    """({frame: SHA-1 of the walked display list}, every object / direct-data command's (oba, tpa, tha))."""
    hashes, refs = {}, set()
    for frame, start, _, words in geo.parse_log(path):
        cmds = list(geo.walk(words, start // 4))
        hashes[frame] = hashlib.sha1(repr([(c.kind, c.args) for c in cmds]).encode()).hexdigest()
        for c in cmds:
            if c.kind in (0x01, 0x11, 0x02, 0x12):
                refs.add((c.args.get("oba"), c.args["tpa"], c.args["tha"]))
    return hashes, refs


def _snaps(snap_dir: Path) -> list[bytes]:
    """The PNG snapshots' pixels, in the order MAME wrote them."""
    import numpy as np
    from PIL import Image
    return [np.asarray(Image.open(p).convert("RGB")).tobytes() for p in sorted(Path(snap_dir).rglob("*.png"))]


@pytest.mark.mame
@pytest.mark.custom_mame
def test_custom_build_runs_originals(tmp_path, romset_zip):
    """Beginner, Expert and Advanced (DAYTONA_COURSE 0, 2, 1), SECONDS each, three ways: stock MAME on the stock set
    (the reference), the custom build on the stock set (its five custom files absent: the new areas stay zero), and
    the custom build on the custom set (RomSet.as_custom: every new area 0xff). Against the reference, call for call:
    the same geometry display list on every frame from GEO_FROM (every command, jumps followed), the same render-rate
    log on every frame, and pixel-identical snapshots at SNAPS and the run's end (the rasterizer reads polygon and
    texture ROM, which the lists only point at). On the custom set no object command names a polygon-ROM index at or
    past 0x400000 (the stock mask) and no texture point / header address falls in the textures' new areas, so the
    larger polygon mask and the filled gaps are never read by the originals."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    custom = _custom()
    rs = RomSet.from_zip(romset_zip).as_custom()
    env = {"DAYTONA_GEO_FROM": str(GEO_FROM), "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": SNAPS}
    for course in (0, 2, 1):
        got = {}
        for name, s, exe in (("stock", None, "mame"), ("custom-build", None, str(custom)),
                             ("custom-set", rs, str(custom))):
            out = mame_build.run_set(s, tmp_path / f"{course}-{name}", SECONDS, romset_zip,
                                     {"DAYTONA_COURSE": str(course), **env}, logs=("geo", "rate"), mame=exe)
            lists, refs = _lists(out["geo"])
            out["geo"].unlink()                                 # about 600 MB each
            got[name] = lists, runner.parse_rate(out["rate"]), _snaps(out["snap"]), refs
        lists, rate, snaps, _ = got["stock"]
        assert set(lists) == set(range(GEO_FROM, max(lists) + 1)) and max(lists) >= SECONDS * 57, course
        assert len(rate) > SECONDS * 57 and len(snaps) == len(SNAPS.split(",")) + 1, course
        assert len(set(lists.values())) > 1000, course                  # the lists change: a race was drawn
        for name in ("custom-build", "custom-set"):
            c_lists, c_rate, c_snaps, _ = got[name]
            assert c_lists == lists, (course, name, min(f for f in lists if c_lists.get(f) != lists[f]))
            assert c_rate == rate, (course, name)
            assert c_snaps == snaps, (course, name, [k for k, (a, b) in enumerate(zip(c_snaps, snaps)) if a != b])
        refs = got["custom-set"][3]
        assert refs
        for oba, tpa, tha in refs:
            if oba is not None and oba & 0x01800000 == 0x00800000:
                assert oba & 0x7FFFFF < 0x400000, (course, hex(oba))
            for a in (tpa, tha):
                if not a & 0x800000:
                    assert not any(lo <= a & 0x7FFFFF < hi for lo, hi in TEX_GAPS), (course, hex(a))
