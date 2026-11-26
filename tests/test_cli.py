"""The `dtools` command (dtools/cli.py) and the packaging half of dtools/release.py (Phase 12 Task 5, M7):
`check_inputs` (Review Focus 1-3: the ROM set, the disc, the output directory), `write_release` and its report, the
printed MAME command (Review Focus 4), and `dtools build` on the player's own inputs. Offline tests run on synthetic
inputs in tmp_path (a ROM set of tiny files with romset.CRCS and ROM_SIZES patched to match, the synthetic Dreamcast
image of tests/test_disc.py with DISC_TRACK3 patched to match); `test_cli_build` reads DAYTONA_ROMDIR and DAYTONA_DC
and skips without them."""

import hashlib
import os
import random
import re
import shutil
import stat
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from dtools import cli, config
from dtools import release as R
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import COPRO_PAIR, CRCS, REGIONS, RomSet
from dtools.dc.disc import DcDisc
from dtools.dc.errors import DcError
from dtools.dc.iso import IsoError
from tests.dc_images import SECTOR
from tests.test_disc import TAR, TRACK3, _TAR, _archive_with_disc, make_disc

RAW = 2352


def _fake_files() -> dict[str, bytes]:
    """The 36 stock names, 16 bytes each, every file different but the TGP data pair, zero (no game data; its two
    2 MB halves must read the same for split_files)."""
    return {n: bytes(16) if n in COPRO_PAIR else bytes([k + 1]) * 16 for k, n in enumerate(CRCS)}


def _match(monkeypatch, files: dict[str, bytes]) -> None:
    """romset.CRCS and release.ROM_SIZES patched to `files`' own CRC32s and sizes: the fake set is "MAME's"."""
    for n, blob in files.items():
        monkeypatch.setitem(CRCS, n, zlib.crc32(blob))
        monkeypatch.setitem(R.ROM_SIZES, n, len(blob))


def _zip(path: Path, files: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for n, blob in files.items():
            z.writestr(n, blob)
    return path


def _disc(directory: Path, monkeypatch, **ip) -> Path:
    """A synthetic US disc (tests/test_disc.py make_disc: track 3 an ISO with every course file) and DISC_TRACK3
    patched to its track 3's size and CRC32; the cue's path."""
    directory.mkdir(parents=True, exist_ok=True)
    cue = make_disc(directory, **ip)
    blob = (directory / TRACK3).read_bytes()
    monkeypatch.setattr(R, "DISC_TRACK3", (len(blob), zlib.crc32(blob)))
    return cue


def _inputs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """(roms, cue, out), each good: the check passes."""
    files = _fake_files()
    _match(monkeypatch, files)
    roms = tmp_path / "roms"
    _zip(roms / "daytona.zip", files)
    _zip(roms / "model1io.zip", {"io.bin": b"io"})
    return roms, _disc(tmp_path / "disc", monkeypatch), tmp_path / "out"


def _problems(report: R.Report) -> list[str]:
    return [ln for ln in report.lines if ln.startswith("problem: ")]


def test_check_rom_set(tmp_path, monkeypatch):
    """Review Focus 1. A good set passes (its 36 files verified by size and CRC32, model1io.zip present); each of no
    daytona.zip, a member missing, a member of the wrong size (both sizes named), a member of the right size and the
    wrong CRC (both CRCs named), a damaged member (its stored CRC right, its bytes not), a file that is not a zip, a set
    this tool built (dx-*.bin) and a ROM directory that does not exist is a problem line naming the file and `ok` false;
    model1io.zip missing is a `warning:` line and `ok` stays true. ROM_SIZES names the 36 files of romset.CRCS, and the
    loader's interleaved pairs split back to exactly those sizes."""
    assert set(R.ROM_SIZES) == set(CRCS)
    blank = RomSet.from_files({n: bytes(R.ROM_SIZES[n]) for n in CRCS}, verify=False)
    assert {n: len(b) for n, b in blank.split_files().items()} == R.ROM_SIZES
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    rep = R.check_inputs(roms, cue, out)
    assert rep.ok and not _problems(rep), rep.lines
    assert any("36 files" in ln and "verified" in ln for ln in rep.lines)
    assert any("model1io.zip: present" in ln for ln in rep.lines)
    files = _fake_files()
    z = roms / "daytona.zip"

    def bad(blobs=None, raw=None) -> list[str]:
        if raw is not None:
            z.write_bytes(raw)
        else:
            _zip(z, blobs)
        r = R.check_inputs(roms, cue, out)
        assert not r.ok
        return _problems(r)

    name = "mpr-16528.10"
    (p,) = bad({n: b for n, b in files.items() if n != name})
    assert name in p and "missing" in p
    (p,) = bad({**files, name: b"\x00" * 17})
    assert name in p and "17 bytes" in p and "expected 16" in p
    other = b"\x55" * 16
    (p,) = bad({**files, name: other})
    assert name in p and f"{zlib.crc32(other):08x}" in p and f"{zlib.crc32(files[name]):08x}" in p
    _zip(z, files)                                    # stored CRC right, the bytes damaged on disk
    blob = bytearray(z.read_bytes())
    at = blob.find(files[name])
    blob[at] ^= 0xFF
    (p,) = bad(raw=bytes(blob))
    assert "daytona.zip" in p
    (p,) = bad(raw=b"not a zip at all")
    assert "daytona.zip" in p and "not a zip" in p
    (p,) = bad({**files, "dx-data.bin": b"\xff"})
    assert "dx-data.bin" in p and "built" in p
    z.unlink()
    (p,) = _problems(R.check_inputs(roms, cue, out))
    assert "daytona.zip" in p
    r = R.check_inputs(tmp_path / "nowhere", cue, out)
    assert not r.ok and any("nowhere" in ln for ln in _problems(r))
    _zip(z, files)
    (roms / "model1io.zip").unlink()
    r = R.check_inputs(roms, cue, out)
    assert r.ok and not _problems(r), r.lines
    assert any(ln.startswith("warning: ") and "model1io.zip" in ln for ln in r.lines)


def test_check_disc(tmp_path, monkeypatch):
    """Review Focus 2. With DAYTONA_DC set, the real disc (its extraction's cue) passes: track 3 is DISC_TRACK3 in size
    and CRC32 and DcDisc.verify finds nothing. A synthetic disc passes from its .cue (spaces in the path accepted) and
    from a .7z (extracted under <out>/work, the archive untouched). Refused, each with the problem named: track 3 not
    beside the cue; track 3 of the wrong size (both sizes named) or the wrong CRC32 (both named); another product; a
    disc that does not exist; a `.gdi` or `.iso`, naming the two accepted shapes (the Redump .7z and the .cue)."""
    real = config.dc_path()
    if real is not None and real.is_file():
        with DcDisc.open(real) as d:                    # the extraction already under config.out_dir()
            rep = R.check_inputs(tmp_path / "roms-none", d.cue, tmp_path / "real-out")
        lines = [ln for ln in rep.lines if "roms-none" not in ln]
        assert not [ln for ln in lines if ln.startswith("problem: ")], lines
        assert any(f"{R.DISC_TRACK3[0]} bytes" in ln and f"{R.DISC_TRACK3[1]:08x}" in ln for ln in lines)
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    assert R.check_inputs(roms, cue, out).ok
    spaced = _disc(tmp_path / "my discs" / "Daytona USA (USA)", monkeypatch)
    rep = R.check_inputs(roms, spaced, out)
    assert rep.ok and not _problems(rep), rep.lines
    if "bsdtar" in _TAR:
        archive = _archive_with_disc(tmp_path / "seven")
        before = archive.read_bytes()
        monkeypatch.setattr(R, "DISC_TRACK3", (len((tmp_path / "seven" / "src" / TRACK3).read_bytes()),
                                               zlib.crc32((tmp_path / "seven" / "src" / TRACK3).read_bytes())))
        rep = R.check_inputs(roms, archive, out)
        assert rep.ok and not _problems(rep), rep.lines
        assert (out / "work" / "dc" / "Game" / TRACK3).is_file() and archive.read_bytes() == before
        _disc(tmp_path / "disc", monkeypatch)
    size, crc = R.DISC_TRACK3
    monkeypatch.setattr(R, "DISC_TRACK3", (size + 2352, crc))
    p = _problems(R.check_inputs(roms, cue, out))
    assert any(TRACK3 in ln and f"{size} bytes" in ln and f"expected {size + 2352}" in ln for ln in p), p
    monkeypatch.setattr(R, "DISC_TRACK3", (size, crc ^ 1))
    p = _problems(R.check_inputs(roms, cue, out))
    assert any(TRACK3 in ln and f"{crc:08x}" in ln and f"{crc ^ 1:08x}" in ln for ln in p), p
    other = _disc(tmp_path / "jp", monkeypatch, product="HDR-0000")
    p = _problems(R.check_inputs(roms, other, out))
    assert any("HDR-0000" in ln for ln in p), p
    (cue.parent / TRACK3).unlink()
    p = _problems(R.check_inputs(roms, cue, out))
    assert any(TRACK3 in ln and "beside" in ln for ln in p), p
    p = _problems(R.check_inputs(roms, tmp_path / "absent.cue", out))
    assert any("absent.cue" in ln for ln in p), p
    for suffix in (".gdi", ".iso", ".chd"):
        f = tmp_path / f"Daytona{suffix}"
        f.write_bytes(b"x")
        p = _problems(R.check_inputs(roms, f, out))
        assert len(p) == 1 and suffix in p[0] and ".7z" in p[0] and ".cue" in p[0], p


def test_out_dir(tmp_path, monkeypatch):
    """Review Focus 3. `--out` that is the ROM directory by another spelling (a trailing slash, relative, through a
    symlink, through `..`), a file, or a directory that cannot be written is refused; the ROM set is then never written.
    `write_release` over an earlier build replaces daytona.zip only once the new one is complete: one that fails
    part-way leaves the earlier daytona.zip as it was and no temporary file; one that succeeds leaves only daytona.zip,
    model1io.zip and build-report.txt, the zip's members `member_sha1`."""
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    before = (roms / "daytona.zip").read_bytes()
    link = tmp_path / "link"
    link.symlink_to(roms, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    for spelling in (str(roms) + "/", "roms", "./roms/", Path("disc") / ".." / "roms", link, str(link) + "/"):
        rep = R.check_inputs(roms, cue, spelling)
        assert not rep.ok and any("ROM directory" in ln for ln in _problems(rep)), (spelling, rep.lines)
    assert (roms / "daytona.zip").read_bytes() == before
    afile = tmp_path / "afile"
    afile.write_text("x")
    assert any("afile" in ln for ln in _problems(R.check_inputs(roms, cue, afile)))
    if hasattr(os, "geteuid") and os.geteuid() != 0:       # POSIX only: a read-only folder, and root ignores it
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        try:
            p = _problems(R.check_inputs(roms, cue, ro / "out"))
            assert any("cannot be written" in ln for ln in p), p
        finally:
            ro.chmod(0o700)
    rs = RomSet.from_files(_fake_files(), verify=False)
    out.mkdir(exist_ok=True)
    (out / "daytona.zip").write_bytes(b"the earlier build")
    real = zipfile.ZipFile.writestr
    calls = []

    def failing(self, *a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise OSError(28, "No space left on device")
        return real(self, *a, **k)
    monkeypatch.setattr(zipfile.ZipFile, "writestr", failing)
    with pytest.raises(R.ReleaseError, match="No space left"):
        R.write_release(rs, out, roms)
    assert (out / "daytona.zip").read_bytes() == b"the earlier build"
    assert sorted(p.name for p in out.iterdir()) == ["daytona.zip"]
    monkeypatch.setattr(zipfile.ZipFile, "writestr", real)
    path = R.write_release(rs, out, roms)
    assert path == (out / "daytona.zip").resolve()
    assert sorted(p.name for p in out.iterdir()) == ["build-report.txt", "daytona.zip", "model1io.zip"]
    with zipfile.ZipFile(path) as z:
        got = {n: hashlib.sha1(z.read(n)).hexdigest() for n in z.namelist()}
    assert got == R.member_sha1(rs) and len(got) == 36
    assert (out / "model1io.zip").read_bytes() == (roms / "model1io.zip").read_bytes()
    with pytest.raises(R.ReleaseError):
        R.write_release("not a set", out, roms)
    with pytest.raises(R.ReleaseError):
        R.member_sha1(None)


def _fake_cli(monkeypatch, rs: RomSet, seen: dict, built=None):
    """cli's inputs replaced: check_inputs passes, the set and the disc are fakes, release_set records DAYTONA_OUT."""
    monkeypatch.setattr(R, "check_inputs", lambda roms, disc, out: R.Report(True, [f"checked {roms}"]))
    monkeypatch.setattr(cli, "RomSet", SimpleNamespace(from_zip=lambda p: rs))

    class Disc:
        @staticmethod
        def open(path, out=None):
            seen["disc_out"] = out
            return Disc()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    def release_set(s, d, car=None):
        seen["env"], seen["car"] = os.environ.get("DAYTONA_OUT"), car
        if built is not None:
            raise built
        return s
    monkeypatch.setattr(cli, "DcDisc", Disc)
    monkeypatch.setattr(R, "release_set", release_set)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_posix_command_through_sh():
    for odd in ("/tmp/with space", '/tmp/a"b', "/tmp/$HOME", "/tmp/`id`", "/tmp/back\\slash", "/tmp/it's",
                "/tmp/bang!", "/tmp/new\nline"):
        cmd = R.mame_command(Path(odd), "Linux")  # what /bin/sh makes of it: the six words, the paths verbatim
        got = subprocess.run(["/bin/sh", "-c", "printf '%s\\0' " + cmd], capture_output=True, text=True, check=True)
        assert got.stdout.split("\0")[:-1] == ["build/mame/model2", "daytona", "-rompath", odd, "-nvram_directory",
                                                f"{odd}/nvram"], (odd, cmd)


def test_windows_command():
    assert R.mame_command(Path(r"C:\games\daytona out"), "Windows") == (
        r'.\mame.exe daytona -rompath "C:\games\daytona out" -nvram_directory "C:\games\daytona out\nvram"')


def test_windows_command_odd_paths():
    assert R.mame_command(Path(r"C:\Users\Zoë Smith\out"), "Windows") == (
        r'.\mame.exe daytona -rompath "C:\Users\Zoë Smith\out" -nvram_directory "C:\Users\Zoë Smith\out\nvram"')
    for ch in '"%$`':
        with pytest.raises(R.ReleaseError, match=re.escape(ch)):
            R.mame_command(Path(rf"C:\a{ch}b"), "Windows")


def test_windows_out_drive_root_or_trailing_backslash():
    for bad in ("C:\\", "D:/"):                  # a drive root (on Windows, Path keeps its backslash)
        with pytest.raises(R.ReleaseError, match="backslash"):
            R.mame_command(Path(bad), "Windows")
    assert "backslash" in R._windows_out_problem("C:\\games\\")      # as typed, before Path normalises it
    assert R._windows_out_problem("C:\\games\\out") == ""
    for ok in (".", "..", "out", r"..\out", r"C:\games\out"):
        assert R._windows_out_problem(ok) == "", ok
    for bad in ("C:\\", "C:", r"\\srv\share", "\\\\srv\\share\\"):
        assert R._windows_out_problem(bad), bad


def test_windows_notes():
    note = R.custom_note("Windows")
    assert R.WINDOWS_MAME_ZIP == "daytona-usa-extended-mame-0.289-win64.zip"
    assert R.WINDOWS_MAME_ZIP in note and "tools/build-mame.sh" not in note
    assert R.WINDOWS_MAME_ZIP in R.run_from("Windows")
    assert R.custom_note("Darwin") == R.CUSTOM_NOTE
    assert R.run_from("Linux") == "run it from the repository root, with the custom MAME:"


def test_build_report_and_command(tmp_path, monkeypatch, capsys):
    """Review Focus 4. The printed command and build-report.txt name `build/mame/model2`, `-rompath "<out>"` and
    `-nvram_directory "<out>/nvram"` (out absolute), say "stock MAME 0.289 will not run" the set and name
    tools/build-mame.sh; the report holds the commit line, the Python and MAME versions tested, the input checks, every
    member's sha1 sorted and the region sha1s. `dtools build` runs with DAYTONA_OUT at <out>/work (the re-mesh cache and
    the disc's extraction) and restores it after; a DecodeError in the build exits 1 printing one line (no traceback);
    a failed check exits 1 and builds nothing; `check` exits 0 on good inputs; `--out` is required."""
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    rs = RomSet.from_files(_fake_files(), verify=False)
    R.write_release(rs, out, roms, checks=["roms ok", "disc ok"])
    text = (out / "build-report.txt").read_text()
    o = out.resolve()
    for want in (R.mame_command(o), R.custom_note(), R.run_from(), "stock MAME 0.289 will not run", "commit: ",
                 "Python 3.15.0", "MAME 0.289", "roms ok", "disc ok"):
        assert want in text, want
    sha = R.member_sha1(rs)
    lines = text.splitlines()
    member = [ln.split() for ln in lines if ln.startswith("  ") and len(ln.split()) == 2 and ln.split()[0] in sha]
    assert [m[0] for m in member] == sorted(sha) and all(sha[n] == h for n, h in member)
    for r in REGIONS:
        assert f"{r} {hashlib.sha1(getattr(rs, r)).hexdigest()}" in text
    if sys.platform != "win32":                  # the exact POSIX texts (the Windows ones: test_windows_*)
        assert R.mame_command(o) == f'build/mame/model2 daytona -rompath "{o}" -nvram_directory "{o}/nvram"'
        assert "tools/build-mame.sh" in text and 'build/mame/model2 daytona' in text

    seen: dict = {}
    _fake_cli(monkeypatch, rs, seen)
    env = os.environ.get("DAYTONA_OUT")
    capsys.readouterr()
    assert cli.main(["build", "--roms", str(roms), "--disc", str(cue), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert R.mame_command(o) in printed and "stock MAME 0.289 will not run" in printed
    assert R.run_from() in printed and R.custom_note() in printed
    if sys.platform != "win32":
        assert "tools/build-mame.sh" in printed
    assert seen["env"] == str(o / "work") and seen["disc_out"] == o / "work"
    assert os.environ.get("DAYTONA_OUT") == env

    _fake_cli(monkeypatch, rs, seen, built=R.ReleaseError("no room for course 5\nsecond line"))
    capsys.readouterr()
    assert cli.main(["build", "--roms", str(roms), "--disc", str(cue), "--out", str(out)]) == 1
    cap = capsys.readouterr()
    assert cap.err.splitlines() == ["build failed: no room for course 5 second line"]
    assert "Traceback" not in cap.out + cap.err and os.environ.get("DAYTONA_OUT") == env
    assert issubclass(R.ReleaseError, DecodeError)

    monkeypatch.setattr(R, "check_inputs", lambda roms, disc, out: R.Report(False, ["problem: bad"]))
    seen.clear()
    assert cli.main(["build", "--roms", str(roms), "--disc", str(cue), "--out", str(out)]) == 1
    assert "env" not in seen and "not ready" in capsys.readouterr().out
    monkeypatch.undo()
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    assert cli.main(["check", "--roms", str(roms), "--disc", str(cue), "--out", str(out)]) == 0
    assert "ok: ready to build" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli.main(["check", "--roms", str(roms), "--disc", str(cue)])
    assert e.value.code == 2


def test_cli_build(romset_zip, dc_path, tmp_path):
    """`dtools build` on the player's own set and disc (the .7z: extracted under <out>/work) into tmp_path: each
    member of the written daytona.zip has the sha1 `member_sha1(release_set(...))` gives in this process, model1io.zip
    is copied byte for byte, build-report.txt is written (every member sha1 in it), out holds nothing else but the work
    folder, and the ROM directory and the disc are not written (names, sizes and times unchanged). The re-mesh cache is
    seeded from config.out_dir()'s (the cold build is Step 5's pin, RELEASE_SHA1)."""
    from tests.test_release import _release_cars as _release
    roms = romset_zip.parent
    out = tmp_path / "Daytona USA Extended"
    warm = config.out_dir() / "cache" / "remesh"
    if warm.is_dir():
        shutil.copytree(warm, out / "work" / "cache" / "remesh")

    def stamp(d: Path) -> dict:
        return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir()}
    before, disc_before = stamp(roms), (dc_path.stat().st_size, dc_path.stat().st_mtime_ns)
    try:
        assert cli.main(["build", "--roms", str(roms), "--disc", str(dc_path), "--out", str(out)]) == 0
        assert stamp(roms) == before and (dc_path.stat().st_size, dc_path.stat().st_mtime_ns) == disc_before
        assert sorted(p.name for p in out.iterdir()) == ["build-report.txt", "daytona.zip", "model1io.zip", "work"]
        with DcDisc.open(dc_path) as disc:
            want = R.member_sha1(_release(romset_zip, disc))
        with zipfile.ZipFile(out / "daytona.zip") as z:
            got = {n: hashlib.sha1(z.read(n)).hexdigest() for n in z.namelist()}
        assert got == want and len(got) == 41
        assert (out / "model1io.zip").read_bytes() == (roms / "model1io.zip").read_bytes()
        report = (out / "build-report.txt").read_text()
        assert all(f"{n} {h}" in report for n, h in want.items())
    finally:
        shutil.rmtree(out / "work", ignore_errors=True)


def _damage(blob: bytes, seed: int) -> bytes:
    """`blob` with 8 bytes overwritten at random (Random(seed)) in the user data of sectors 16-40 (the volume
    descriptor and the directories): the right size, a damaged file system."""
    rng = random.Random(seed)
    b = bytearray(blob)
    for _ in range(8):
        sector, at = rng.randrange(16, 41), rng.randrange(SECTOR)
        b[sector * RAW + 16 + at] = rng.randrange(256)
    return bytes(b)


DAMAGE = {52: struct.error, 314: IndexError}
"""Seeds of `_damage` and what DcDisc.verify raises on the synthetic disc so damaged (asserted in the test)."""


def _check_line(roms, disc, out, capsys) -> str:
    """`dtools check` on the inputs: exit 1, no traceback, one problem line naming the disc's track 3 or cue; that
    line."""
    capsys.readouterr()
    assert cli.main(["check", "--roms", str(roms), "--disc", str(disc), "--out", str(out)]) == 1
    cap = capsys.readouterr()
    assert "Traceback" not in cap.out + cap.err and not cap.err, cap
    (p,) = [ln for ln in cap.out.splitlines() if ln.startswith("problem: ")]
    return p


def test_check_damaged_disc(tmp_path, monkeypatch, capsys):
    """Review Focus 2, a cut-short or damaged track 3 (Task 5 review, Important 1): never a traceback from `check` or
    `build`, one problem line, exit 1. Track 3 really cut short (20 sectors: the volume descriptor there, the
    directories not; 10: not even the descriptor) is named by its size; damaged at the right size (DAMAGE's seeds: the
    directory bytes DcDisc.verify trips on with struct.error and IndexError) by its CRC32, and DcDisc.verify is not run
    once the size or the CRC32 has failed. Should the size and the CRC32 pass (DISC_TRACK3 patched to the damaged
    file's own), the file system that cannot be read (IsoError, IndexError, struct.error) is one problem line. The
    .7z, which is opened as it is extracted, cut short the same: one line. `build` refuses with the same line."""
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    track = cue.parent / TRACK3
    good = track.read_bytes()
    size, crc = R.DISC_TRACK3

    def verify_not_run(self):
        raise AssertionError("DcDisc.verify ran after the size or CRC32 check failed")
    for sectors in (20, 10):
        track.write_bytes(good[:sectors * RAW])
        with monkeypatch.context() as m:
            m.setattr(DcDisc, "verify", verify_not_run)
            rep = R.check_inputs(roms, cue, out)
        assert not rep.ok
        (p,) = _problems(rep)
        assert TRACK3 in p and f"{sectors * RAW} bytes" in p and f"expected {size}" in p, p
        assert TRACK3 in _check_line(roms, cue, out, capsys)
        capsys.readouterr()
        assert cli.main(["build", "--roms", str(roms), "--disc", str(cue), "--out", str(out)]) == 1
        cap = capsys.readouterr()
        assert "Traceback" not in cap.out + cap.err and "not ready" in cap.out
    for seed, error in DAMAGE.items():
        blob = _damage(good, seed)
        track.write_bytes(blob)
        with DcDisc.open(cue) as d, pytest.raises(error):     # what the review reproduced: the path is real
            d.verify()
        with monkeypatch.context() as m:
            m.setattr(DcDisc, "verify", verify_not_run)
            (p,) = _problems(R.check_inputs(roms, cue, out))
        assert TRACK3 in p and f"{zlib.crc32(blob):08x}" in p and f"expected {crc:08x}" in p, p
        assert TRACK3 in _check_line(roms, cue, out, capsys)
    cut = good[:20 * RAW]
    for blob, error in [(cut, IsoError)] + [(_damage(good, s), e) for s, e in DAMAGE.items()]:
        track.write_bytes(blob)
        monkeypatch.setattr(R, "DISC_TRACK3", (len(blob), zlib.crc32(blob)))
        rep = R.check_inputs(roms, cue, out)
        assert not rep.ok
        (p,) = _problems(rep)
        assert str(cue) in p and "file system cannot be read" in p and error.__name__ in p, p
        assert "file system cannot be read" in _check_line(roms, cue, out, capsys)
    monkeypatch.setattr(R, "DISC_TRACK3", (size, crc))
    track.write_bytes(good)
    assert R.check_inputs(roms, cue, out).ok
    if "bsdtar" in _TAR:
        src = tmp_path / "seven" / "src"
        src.mkdir(parents=True)
        make_disc(src)
        for sectors in (20, 10):
            (src / TRACK3).write_bytes(good[:sectors * RAW])
            archive = tmp_path / "seven" / f"Cut{sectors}.7z"
            subprocess.run([TAR, "-cf", str(archive), "--format", "7zip", "-C", str(src), "Game.cue", TRACK3],
                           check=True)
            rep = R.check_inputs(roms, archive, out)
            assert not rep.ok
            (p,) = _problems(rep)
            assert f"Cut{sectors}" in p or TRACK3 in p, p
            assert "\n" not in p
            _check_line(roms, archive, out, capsys)


def test_cli_check_errors(tmp_path, monkeypatch, capsys):
    """`check_inputs` raising an error of the inputs (a DecodeError, an OSError) is one line on stderr and exit 1 from
    `check` and from `build`; any other exception is a defect and keeps its traceback (not swallowed)."""
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    for exc in (DcError("the disc\nis odd"), OSError(5, "Input/output error"), R.ReleaseError("bad path")):
        def boom(roms, disc, out, exc=exc):
            raise exc
        monkeypatch.setattr(R, "check_inputs", boom)
        for command in ("check", "build"):
            capsys.readouterr()
            assert cli.main([command, "--roms", str(roms), "--disc", str(cue), "--out", str(out)]) == 1
            cap = capsys.readouterr()
            assert len(cap.err.splitlines()) == 1 and cap.err.startswith("check failed: "), cap.err
            assert "Traceback" not in cap.out + cap.err
    assert "the disc is odd" in cli._one_line(DcError("the disc\nis odd"))

    def bug(roms, disc, out):
        raise TypeError("a defect")
    monkeypatch.setattr(R, "check_inputs", bug)
    with pytest.raises(TypeError, match="a defect"):
        cli.main(["check", "--roms", str(roms), "--disc", str(cue), "--out", str(out)])


def test_failed_check_leaves_no_out(tmp_path, monkeypatch):
    """A failed check removes the `--out` it created itself (and every directory it made on the way, a .7z's
    extraction included); an `--out` that was there before stays, and a check that passes leaves `--out` for the
    build."""
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    (cue.parent / TRACK3).unlink()
    fresh = tmp_path / "new" / "deeper" / "out"
    assert not R.check_inputs(roms, cue, fresh).ok
    assert not (tmp_path / "new").exists()
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    assert not R.check_inputs(roms, cue, out).ok
    assert (out / "keep.txt").read_text() == "mine"
    _disc(tmp_path / "disc", monkeypatch)
    assert R.check_inputs(roms, cue, fresh).ok and fresh.is_dir()
    shutil.rmtree(tmp_path / "new")
    if "bsdtar" in _TAR:
        archive = _archive_with_disc(tmp_path / "seven")
        t3 = (tmp_path / "seven" / "src" / TRACK3).read_bytes()
        monkeypatch.setattr(R, "DISC_TRACK3", (len(t3), zlib.crc32(t3)))
        (roms / "daytona.zip").write_bytes(b"not a zip")       # the disc extracted and checked, the ROMs not
        assert not R.check_inputs(roms, archive, fresh).ok
        assert not (tmp_path / "new").exists()


def _earlier_build(out: Path) -> None:
    out.mkdir(exist_ok=True)
    (out / "daytona.zip").write_bytes(b"the earlier build")
    (out / "build-report.txt").write_text("the earlier report\n")


def test_written_files(tmp_path, monkeypatch):
    """daytona.zip, model1io.zip and build-report.txt get the mode a plain open gives (0666 less the umask: 0644
    under 022, 0600 under 077), not mkstemp's 0600. The report never describes another zip than the one beside it: a
    failure while the report is written leaves the earlier zip and the earlier report as they were, a failure while
    the files are renamed into place leaves no earlier report beside a new zip; no temporary file stays."""
    roms, cue, out = _inputs(tmp_path, monkeypatch)
    rs = RomSet.from_files(_fake_files(), verify=False)
    names = ["build-report.txt", "daytona.zip", "model1io.zip"]
    for umask, mode in ((0o022, 0o644), (0o077, 0o600), (0o027, 0o640)):
        old = os.umask(umask)
        try:
            R.write_release(rs, out, roms)
        finally:
            os.umask(old)
        if sys.platform != "win32":                                      # POSIX file modes and umask
            assert {n: stat.S_IMODE((out / n).stat().st_mode) for n in names} == dict.fromkeys(names, mode), umask
    shutil.rmtree(out)

    _earlier_build(out)
    real_write = Path.write_text

    def failing_write(self, *a, **k):
        if self.name.startswith(".build-report.txt."):
            raise OSError(28, "No space left on device")
        return real_write(self, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", failing_write)
        with pytest.raises(R.ReleaseError, match="No space left"):
            R.write_release(rs, out, roms)
    assert (out / "daytona.zip").read_bytes() == b"the earlier build"
    assert (out / "build-report.txt").read_text() == "the earlier report\n"
    assert sorted(p.name for p in out.iterdir()) == ["build-report.txt", "daytona.zip"]

    real_replace = os.replace

    def failing_replace(src, dst, *a, **k):
        if Path(dst).name == "build-report.txt":
            raise OSError(5, "Input/output error")
        return real_replace(src, dst, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(R.os, "replace", failing_replace)
        with pytest.raises(R.ReleaseError, match="Input/output"):
            R.write_release(rs, out, roms)
    left = sorted(p.name for p in out.iterdir())
    assert not [n for n in left if n.endswith(".partial")], left
    if (out / "daytona.zip").read_bytes() != b"the earlier build":         # a new zip: the old report is gone
        assert not (out / "build-report.txt").exists()
    else:
        assert (out / "build-report.txt").read_text() == "the earlier report\n"
    R.write_release(rs, out, roms)
    assert "the earlier" not in (out / "build-report.txt").read_text()
