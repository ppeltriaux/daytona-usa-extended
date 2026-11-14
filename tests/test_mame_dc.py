"""The Dreamcast in MAME's `dc` driver: the BIOS set, the disc image, the run, and the boot to the course select."""

import hashlib
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest

from dtools import config
from dtools.dc.disc import DcDisc
from dtools.dc.errors import DcError
from dtools.mame import dc, runner
from dtools.mame.runner import LUA_DIR

BOOT_SIZE, FLASH_SIZE = 2 * 1024 * 1024, 128 * 1024


def _bios_zip(path, boot: bytes, flash: bytes, names=("dc_boot.bin", "dc_flash.bin")):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in zip(names, (boot, flash)):
            z.writestr(name, data)
    return path


def test_bios_set_checks_crc(tmp_path):
    src = _bios_zip(tmp_path / "bios.zip", b"\x01" * BOOT_SIZE, b"\x02" * FLASH_SIZE)
    with pytest.raises(DcError, match=r"(?s)dc_boot\.bin.*89f2b1a1"):
        dc.bios_set(src, tmp_path / "out")
    assert not (tmp_path / "out" / "dc.zip").exists()


def test_bios_set_names(tmp_path, monkeypatch):
    boot, flash = b"\x01" * BOOT_SIZE, b"\x02" * FLASH_SIZE
    monkeypatch.setattr(dc, "BOOT_CRC", zlib.crc32(boot))       # a synthetic boot ROM stands in for 101d
    for src in (_bios_zip(tmp_path / "bios.zip", boot, flash), tmp_path / "dir"):
        if src.suffix != ".zip":
            src.mkdir()
            (src / "dc_boot.bin").write_bytes(boot)
            (src / "dc_flash.bin").write_bytes(flash)
        out = dc.bios_set(src, tmp_path / f"out-{src.stem}")
        assert out == tmp_path / f"out-{src.stem}" / "dc.zip"
        with zipfile.ZipFile(out) as z:
            assert sorted(z.namelist()) == ["dcus_ntsc.bin", "mpr-21931.ic501"]
            assert z.read("mpr-21931.ic501") == boot and z.read("dcus_ntsc.bin") == flash


def test_bios_set_names_a_missing_file(tmp_path):
    src = _bios_zip(tmp_path / "bios.zip", b"\x01" * BOOT_SIZE, b"", names=("dc_boot.bin", "other.bin"))
    with pytest.raises(DcError, match="dc_flash.bin"):
        dc.bios_set(src, tmp_path / "out")
    with pytest.raises(DcError, match="not a zip or a directory"):
        dc.bios_set(tmp_path / "absent.zip", tmp_path / "out")


def test_bios_set_filesystem_failures_are_dc_errors(tmp_path, monkeypatch):
    """An out_dir that cannot be made, or a dc.zip that cannot be replaced, is a DcError naming the path; no
    dc.zip.partial is left behind."""
    boot, flash = b"\x01" * BOOT_SIZE, b"\x02" * FLASH_SIZE
    monkeypatch.setattr(dc, "BOOT_CRC", zlib.crc32(boot))
    src = _bios_zip(tmp_path / "bios.zip", boot, flash)
    (tmp_path / "file").write_bytes(b"")
    with pytest.raises(DcError, match="file"):
        dc.bios_set(src, tmp_path / "file")                        # a file in the way of out_dir
    out = tmp_path / "out"
    (out / "dc.zip" / "x").mkdir(parents=True)                     # a non-empty directory in the way of dc.zip
    with pytest.raises(DcError, match="dc.zip"):
        dc.bios_set(src, out)
    assert sorted(p.name for p in out.iterdir()) == ["dc.zip"]


CUE = """REM SINGLE-DENSITY AREA
FILE "d (Track 1).bin" BINARY
  TRACK 01 MODE1/2352
    INDEX 01 00:00:00
FILE "d (Track 2).bin" BINARY
  TRACK 02 AUDIO
    INDEX 00 00:00:00
    INDEX 01 00:02:00
REM HIGH-DENSITY AREA
FILE "d (Track 3).bin" BINARY
  TRACK 03 MODE1/2352
    INDEX 01 00:00:00
"""


class _Disc:
    """What disc_image reads of a DcDisc: the cue and the archive it came from (None: opened from a cue)."""
    def __init__(self, cue, archive=None):
        self.cue, self.archive = cue, archive


def test_disc_image_references_the_tracks(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "d.cue").write_text(CUE)
    for t in (1, 2, 3):
        (src / f"d (Track {t}).bin").write_bytes(bytes([t]) * 2352)
    image = dc.disc_image(_Disc(src / "d.cue"), tmp_path / "img")
    assert image == tmp_path / "img" / "d.cue" and image.read_text() == CUE
    for t in (1, 2, 3):
        link = tmp_path / "img" / f"d (Track {t}).bin"
        assert link.is_symlink() and link.resolve() == (src / f"d (Track {t}).bin").resolve()
    assert dc.disc_image(_Disc(src / "d.cue"), tmp_path / "img") == image       # a second call reuses the links


def test_disc_image_names_a_missing_track(tmp_path):
    (tmp_path / "d.cue").write_text(CUE)
    (tmp_path / "d (Track 3).bin").write_bytes(b"\x00" * 2352)
    with pytest.raises(DcError, match=r"d \(Track 1\)\.bin"):
        dc.disc_image(_Disc(tmp_path / "d.cue"), tmp_path / "img")


def test_disc_image_refuses_a_track_outside_the_cue_directory(tmp_path):
    """The cue is read from a disc dump, not written by us: a FILE name with a directory part would put a link (or an
    extracted file) outside out_dir."""
    (tmp_path / "d.cue").write_text(CUE.replace('"d (Track 2).bin"', '"../d (Track 2).bin"'))
    for t in (1, 2, 3):
        (tmp_path / f"d (Track {t}).bin").write_bytes(b"\x00" * 2352)
    with pytest.raises(DcError, match=r"\.\./d \(Track 2\)\.bin"):
        dc.disc_image(_Disc(tmp_path / "d.cue"), tmp_path / "img")


def test_disc_image_in_the_cue_directory_keeps_the_cue(tmp_path):
    (tmp_path / "d.cue").write_text(CUE)
    for t in (1, 2, 3):
        (tmp_path / f"d (Track {t}).bin").write_bytes(b"\x00" * 2352)
    assert dc.disc_image(_Disc(tmp_path / "d.cue"), tmp_path) == tmp_path / "d.cue"
    assert (tmp_path / "d.cue").read_text() == CUE


def _cue_and_tracks(d):
    d.mkdir(exist_ok=True)
    (d / "d.cue").write_text(CUE)
    for t in (1, 2, 3):
        (d / f"d (Track {t}).bin").write_bytes(b"\x00" * 2352)
    return d / "d.cue"


def test_disc_image_filesystem_failures_are_dc_errors(tmp_path):
    cue = _cue_and_tracks(tmp_path / "src")
    (tmp_path / "file").write_bytes(b"")
    with pytest.raises(DcError, match="file"):
        dc.disc_image(_Disc(cue), tmp_path / "file")               # out_dir is a file
    img = tmp_path / "img"
    (img / "d (Track 1).bin" / "x").mkdir(parents=True)            # a non-empty directory where a link goes
    with pytest.raises(DcError, match=r"d \(Track 1\)\.bin"):
        dc.disc_image(_Disc(cue), img)
    img2 = tmp_path / "img2"
    (img2 / "d.cue" / "x").mkdir(parents=True)                     # a directory where the cue copy goes
    with pytest.raises(DcError, match=r"d\.cue"):
        dc.disc_image(_Disc(cue), img2)


def test_extract_member_replace_failure_is_a_dc_error(tmp_path, monkeypatch):
    """The extracted track cannot be moved into place (a non-empty directory has its name): DcError, and the
    .partial directory is removed."""
    def fake_tar(*args):
        if args[0] == "-tf":
            return subprocess.CompletedProcess(args, 0, "t.bin\n", "")
        (Path(args[3]) / "t.bin").write_bytes(b"\x07")             # -xf archive -C <partial> member
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(dc.dcdisc, "_tar", fake_tar)
    (tmp_path / "t.bin" / "x").mkdir(parents=True)
    with pytest.raises(DcError, match="t.bin"):
        dc._extract_member(tmp_path / "a.7z", "t.bin", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.bin"]


def test_disc_image_extracts_the_small_tracks_from_the_archive(tmp_path, monkeypatch):
    """Opened from a .7z, the disc has only the cue and track 3 extracted; the image takes tracks 1 and 2 from the
    archive into out_dir (they are small) and links track 3."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "d.cue").write_text(CUE)
    (src / "d (Track 3).bin").write_bytes(b"\x03" * 2352)
    archive = tmp_path / "d.7z"
    archive.write_bytes(b"")
    calls = []

    def fake_extract(arch, member, dest):
        calls.append((arch, member))
        (dest / member).write_bytes(b"\x07" * 2352)

    monkeypatch.setattr(dc, "_extract_member", fake_extract)
    image = dc.disc_image(_Disc(src / "d.cue", archive), tmp_path / "img")
    assert calls == [(archive, "d (Track 1).bin"), (archive, "d (Track 2).bin")]
    assert not (image.parent / "d (Track 1).bin").is_symlink()
    assert (image.parent / "d (Track 3).bin").is_symlink()
    calls.clear()
    dc.disc_image(_Disc(src / "d.cue", archive), tmp_path / "img")
    assert calls == []                                              # extracted once


def test_run_dc_runs_the_dc_system_with_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "running_mame", lambda: [])
    monkeypatch.chdir(tmp_path)
    args = tmp_path / "args"
    mame = tmp_path / "fake-mame"
    mame.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {args}\necho 'Average speed: 100%'\n")
    mame.chmod(0o755)
    dc.run_dc(Path("x.lua"), 3, Path("run"), Path("roms"), Path("d.cue"), {}, mame=str(mame))
    argv = args.read_text().splitlines()
    assert argv[0] == "dc"                                          # MAME runs in the workdir: paths made absolute
    assert argv[argv.index("-cdrom") + 1] == str((tmp_path / "d.cue").resolve())
    assert argv[argv.index("-rompath") + 1] == str((tmp_path / "roms").resolve())
    assert argv[argv.index("-autoboot_script") + 1] == str((tmp_path / "x.lua").resolve())
    assert argv[argv.index("-snapshot_directory") + 1] == str((tmp_path / "run" / "snap").resolve())
    assert argv[argv.index("-str") + 1] == "3"


def test_run_dc_fails_on_a_lua_error(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "running_mame", lambda: [])
    mame = tmp_path / "fake-mame"
    mame.write_text("#!/bin/sh\necho '[LUA ERROR] boom'\necho 'Average speed: 100%'\n")
    mame.chmod(0o755)
    with pytest.raises(runner.MameError, match="boom"):
        dc.run_dc(tmp_path / "x.lua", 1, tmp_path / "run", tmp_path, tmp_path / "d.cue", {}, mame=str(mame))


# Measured 2026-11-08 (MAME 0.289, BIOS 101d, fresh NVRAM; docs/spike/2026-11-dc-decode.md, "MAME `dc`"): with
# dc_select.lua's presses the course select ("COURSE SETTING", Three Seven Speedway) is on screen by frame 3690.
COURSE_SELECT_FRAME = 3690
# The select's "COURSE SETTING" title (the 702 x 576 snapshot's box below, RGB bytes): SHA-256 measured on frame 3690
# of five separate fresh-NVRAM runs and on all eight entries of a walk through the select; other screens (memory
# card, the Single Race setup) hash differently. The 3D course preview below it varies between runs, the title does
# not. A hash, not the image, is kept.
TITLE_BOX = (95, 78, 470, 102)
TITLE_SHA256 = "3e66119f17846338db483ac175504034dddd462fc8af81d5b2245cb7a0dc49bd"


@pytest.fixture
def dc_bios() -> dict:
    d = config.rom_dir()
    if d is None or not (d / "dc_bios.zip").is_file():
        pytest.skip("set DAYTONA_ROMDIR to a directory containing dc_bios.zip")
    return d / "dc_bios.zip"


@pytest.mark.mame
def test_dc_boots(tmp_path, dc_bios, dc_path):
    from PIL import Image

    roms = dc.bios_set(dc_bios, tmp_path / "roms").parent
    with DcDisc.open(dc_path) as disc:
        image = dc.disc_image(disc, tmp_path / "img")
    proc = dc.run_dc(LUA_DIR / "dc_select.lua", 78, tmp_path / "run", roms, image,
                     {"DC_SNAPS": str(COURSE_SELECT_FRAME)})
    assert f"SNAP {COURSE_SELECT_FRAME}" in proc.stdout
    snap = Image.open(tmp_path / "run" / "snap" / "dc" / "0000.png").convert("RGB")
    assert snap.size == (702, 576)
    assert snap.getbbox() is not None                               # not a black frame
    assert len(snap.getcolors(1 << 24)) > 100                      # a drawn screen, not one flat colour
    assert hashlib.sha256(snap.crop(TITLE_BOX).tobytes()).hexdigest() == TITLE_SHA256   # the course select's title
