import numpy as np
import pytest
from PIL import Image

from dtools.arcade.romset import RomSet
from dtools.mame import build, runner
from dtools.mame.runner import MameError


def _img(path, arr):
    Image.fromarray(arr.astype(np.uint8), "RGB").save(path)
    return path


def test_snapshot_match_synthetic(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (24, 40, 3))
    a = _img(tmp_path / "a.png", arr)
    flipped = _img(tmp_path / "f.png", arr[:, ::-1])
    vflipped = _img(tmp_path / "v.png", arr[::-1])
    black = _img(tmp_path / "k.png", np.zeros((24, 40, 3)))
    assert build.snapshot_match(a, a) == 1.0
    assert build.snapshot_match(a, flipped, flip="h") == 1.0
    assert build.snapshot_match(a, vflipped, flip="v") == 1.0
    assert build.snapshot_match(a, black) < 0.1
    assert build.snapshot_match(a, flipped) < 0.5
    box = (4, 2, 20, 10)                                    # the flip happens inside the box
    sub = arr.copy()
    sub[2:10, 4:20] = arr[2:10, 4:20][:, ::-1]
    assert build.snapshot_match(a, _img(tmp_path / "s.png", sub), box=box, flip="h") == 1.0
    near = _img(tmp_path / "n.png", np.clip(arr + 16, 0, 255))
    assert build.snapshot_match(a, near, tol=16) == 1.0 or build.snapshot_match(a, near, tol=40) == 1.0


def test_snapshot_match_mask(tmp_path):
    """With `mask` (the box's shape, in `a`'s coordinates) only the masked pixels count: half the image differs, the
    other half (masked) matches; ValueError for a mask of another shape or with no pixel."""
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, (24, 40, 3))
    other = arr.copy()
    other[:, 20:] = 255 - other[:, 20:]
    a, b = _img(tmp_path / "a.png", arr), _img(tmp_path / "b.png", other)
    left = np.zeros((24, 40), bool)
    left[:, :20] = True
    assert build.snapshot_match(a, b, mask=left) == 1.0 and build.snapshot_match(a, b, mask=~left) < 0.2
    box = (0, 0, 40, 12)
    assert build.snapshot_match(a, b, box, mask=left[:12]) == 1.0
    flipped = _img(tmp_path / "f.png", other[:, ::-1])                  # the mask is in a's coordinates
    assert build.snapshot_match(a, flipped, flip="h", mask=left) == 1.0
    for bad in (left[:12], np.zeros((24, 40), bool)):
        with pytest.raises(ValueError):
            build.snapshot_match(a, b, mask=bad)


def test_log_names_map_to_env(tmp_path):
    assert build.LOG_ENVS == {"ground": "DAYTONA_GROUND_LOG", "rate": "DAYTONA_RATE_LOG",
                              "geo": "DAYTONA_GEO_LOG", "progress": "DAYTONA_PROGRESS_LOG",
                              "cells": "DAYTONA_CELLS_LOG", "pit": "DAYTONA_PIT_LOG"}
    with pytest.raises(MameError, match="unknown log"):
        build.run_set(None, tmp_path, 1, tmp_path / "daytona.zip", {}, logs=("nope",))


def _fake_run(monkeypatch, tmp_path, stdout, write=None):
    seen = {}

    def fake(script, seconds, workdir, rom_dir, env=None, **kw):
        workdir.mkdir(parents=True, exist_ok=True)
        seen.update(env=env, rom_dir=rom_dir, script=script)
        for k, v in (write or {}).items():
            open(env[k], "w").write(v)

        class P:
            pass
        p = P()
        p.stdout, p.stderr = stdout, ""
        return p
    monkeypatch.setattr(runner, "run_mame", fake)
    return seen


def test_run_set_empty_log_and_env(tmp_path, monkeypatch):
    zip_ = tmp_path / "src" / "daytona.zip"
    seen = _fake_run(monkeypatch, tmp_path, "")
    with pytest.raises(MameError, match="empty log: rate"):
        build.run_set(None, tmp_path / "w", 1, zip_, {"A": "1"}, logs=("rate",))
    assert seen["env"]["DAYTONA_RATE_LOG"] == str(tmp_path / "w" / "rate.txt") and seen["env"]["A"] == "1"
    (tmp_path / "w").mkdir(exist_ok=True)
    (tmp_path / "w" / "rate.txt").write_text("stale\n")                # a reused workdir: the run writes nothing
    with pytest.raises(MameError, match="empty log: rate"):
        build.run_set(None, tmp_path / "w", 1, zip_, {}, logs=("rate",))
    _fake_run(monkeypatch, tmp_path, "", write={"DAYTONA_RATE_LOG": "R 1\n", "DAYTONA_PROGRESS_LOG": "x"})
    out = build.run_set(None, tmp_path / "w", 1, zip_, {}, logs=("rate", "progress"))
    assert set(out) == {"rate", "progress", "snap"}
    _fake_run(monkeypatch, tmp_path, "WRONG CHECKSUMS:\n")
    with pytest.raises(MameError, match="WRONG CHECKSUMS"):
        build.run_set(None, tmp_path / "w", 1, zip_, {})


@pytest.mark.mame
def test_run_set_checks_the_load(tmp_path, romset_zip):
    """Presence of MAME's WRONG CHECKSUMS block is pinned both ways: the original set runs clean (rs=None); a
    patched set run as rs=None raises; an unchanged write_zip rewrite passed as rs raises (no block)."""
    import shutil
    rs = RomSet.from_zip(romset_zip)
    patched = rs.with_patch("textures", 0xBFFFF0, bytes(b ^ 0xFF for b in rs.textures[0xBFFFF0:0xC00000]))
    build.run_set(None, tmp_path / "orig", 3, romset_zip, {"DAYTONA_COURSE": "0"})
    # a patched set sitting in the rom dir, run without rs: the block is printed, so the call raises
    pd = tmp_path / "pdir"
    patched.write_zip(pd / "daytona.zip")
    shutil.copy(romset_zip.parent / "model1io.zip", pd / "model1io.zip")
    with pytest.raises(MameError, match="WRONG CHECKSUMS"):
        build.run_set(None, tmp_path / "p1", 3, pd / "daytona.zip", {"DAYTONA_COURSE": "0"})
    # a patched set as rs: loads with the block
    build.run_set(patched, tmp_path / "p2", 3, romset_zip, {"DAYTONA_COURSE": "0"})
    # the unchanged rewrite prints no block (found on MAME 0.289): rs set but nothing patched raises
    with pytest.raises(MameError, match="did not load the patched set"):
        build.run_set(rs, tmp_path / "u", 3, romset_zip, {"DAYTONA_COURSE": "0"})


def test_run_set_custom_set_and_custom_mame(tmp_path, monkeypatch):
    """run_set passes `mame` to run_mame. A custom set (RomSet.custom) is proven loaded by MAME naming each of its
    five files "NO GOOD DUMP KNOWN" (the patch declares them NO_DUMP: loaded unchecked, so an unpatched custom set
    prints no WRONG CHECKSUMS); without that (stock MAME, or one file missing) MameError."""
    from dtools.arcade.romset import CUSTOM_FILES
    zip_ = tmp_path / "src" / "daytona.zip"
    zip_.parent.mkdir()
    (zip_.parent / "model1io.zip").write_bytes(b"")
    names = "".join(f"{n} NO GOOD DUMP KNOWN\n" for n in CUSTOM_FILES)
    small = {n: b"\0" * 8 for n in __import__("dtools.arcade.romset", fromlist=["CRCS"]).CRCS}
    c = RomSet.from_files(small, verify=False).as_custom()
    calls = []

    def fake_factory(stdout):
        def fake(script, seconds, workdir, rom_dir, env=None, **kw):
            calls.append(kw.get("mame"))
            workdir.mkdir(parents=True, exist_ok=True)

            class P:
                pass
            p = P()
            p.stdout, p.stderr = stdout, ""
            return p
        return fake

    monkeypatch.setattr(runner, "run_mame", fake_factory(names))
    build.run_set(c, tmp_path / "a", 1, zip_, {}, mame="/x/model2")
    assert calls == ["/x/model2"]
    assert (tmp_path / "a" / "roms" / "daytona.zip").is_file()
    monkeypatch.setattr(runner, "run_mame", fake_factory(names.replace("dx-tex2.bin NO", "dx-tex2.bin NOT FOUND (NO")))
    with pytest.raises(MameError, match="dx-tex2.bin"):
        build.run_set(c, tmp_path / "b", 1, zip_, {}, mame="/x/model2")
    monkeypatch.setattr(runner, "run_mame", fake_factory("WRONG CHECKSUMS:\n"))
    with pytest.raises(MameError, match="custom"):
        build.run_set(c, tmp_path / "c", 1, zip_, {})
    assert calls[-1] == "mame"                                           # the default
    monkeypatch.setattr(runner, "run_mame", fake_factory("dx-data.bin NOT FOUND (NO GOOD DUMP KNOWN)\n"))
    build.run_set(None, tmp_path / "d", 1, zip_, {}, mame="/x/model2")      # the stock set on the custom build
