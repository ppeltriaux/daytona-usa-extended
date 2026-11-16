"""Tests for the Dreamcast disc wrapper, using synthetic GD-ROM track images."""

import subprocess
import sys
import re
from pathlib import Path

import pytest

from dtools.dc import disc as disc_module
from dtools.dc.disc import (
    COURSE_COUNT,
    GD_BASE,
    PRODUCT,
    VARIANTS,
    DcDisc,
    DiscError,
    course_files,
)
from dtools.dc.errors import DcError
from dtools.dc.iso import IsoError
from tests.dc_images import SECTOR, make_image

TRACK3 = "Game (Track 3).bin"

TAR = disc_module._tar_program()
"""The program dtools reads archives with (`bsdtar` when on PATH, else `tar`); the tests write their .7z with it."""
_TAR = subprocess.run([TAR, "--version"], capture_output=True, text=True).stdout
needs_bsdtar = pytest.mark.skipif("bsdtar" not in _TAR, reason="the .7z tests need bsdtar (--format 7zip)")


def ip_header(product: str = PRODUCT, title: str = "DAYTONAUSA", magic: bytes = b"SEGA SEGAKATANA ") -> bytes:
    ip = bytearray(b" " * SECTOR)
    ip[0x00:0x10] = magic
    ip[0x40:0x40 + 10] = product.ljust(10).encode("ascii")
    ip[0x4A:0x50] = b"V1.004"
    ip[0x60:0x70] = b"1ST_READ.BIN    "
    ip[0x80:0x80 + len(title)] = title.encode("ascii")
    return bytes(ip)


def all_course_files() -> dict[str, bytes]:
    files = {}
    for n in range(COURSE_COUNT):
        for v in VARIANTS:
            for name in course_files(n, v).values():
                files[name] = f"{name}".encode()
    return files


def make_cue(directory: Path, track3: str = TRACK3, with_track3: bool = True) -> Path:
    text = 'FILE "Game (Track 1).bin" BINARY\n  TRACK 01 MODE1/2352\n    INDEX 01 00:00:00\n'
    if with_track3:
        text += f'FILE "{track3}" BINARY\n  TRACK 03 MODE1/2352\n    INDEX 01 00:00:00\n'
    cue = directory / "Game.cue"
    cue.write_text(text)
    return cue


def make_disc(directory: Path, files=None, **ip_kwargs) -> Path:
    files = all_course_files() if files is None else files
    (directory / TRACK3).write_bytes(make_image(files, base=GD_BASE, ip=ip_header(**ip_kwargs)))
    return make_cue(directory)


def test_cue_opens_track_3_named_before_it(tmp_path):
    cue = make_disc(tmp_path)
    disc = DcDisc.open(cue)
    assert disc.product == "MK-51037"
    assert disc.title == "DAYTONAUSA"
    assert disc.iso.lba_base == GD_BASE
    assert disc.read("CS1_POL.BIN") == b"CS1_POL.BIN"
    assert disc.course_file(3, "ghost", "MR") == b"GHOST_27.BIN"
    assert disc.verify() == []


def test_wrong_product_is_reported(tmp_path):
    cue = make_disc(tmp_path, product="MK-81213")
    assert "product MK-81213, expected MK-51037" in DcDisc.open(cue).verify()


def test_non_dreamcast_header_is_reported(tmp_path):
    cue = make_disc(tmp_path, magic=b"SEGA SEGASATURN ")
    problems = DcDisc.open(cue).verify()
    assert len(problems) >= 1
    assert problems[0].startswith("not a Dreamcast disc (header b'SEGA SEGASATURN ')")
    assert not any(p.startswith("product") for p in problems)


def test_missing_course_file_is_named(tmp_path):
    files = all_course_files()
    del files["CS3MR_POL.BIN"]
    cue = make_disc(tmp_path, files=files)
    assert DcDisc.open(cue).verify() == ["CS3MR_POL.BIN"]


def test_cue_without_track_3_raises(tmp_path):
    with pytest.raises(DiscError, match="TRACK 03"):
        DcDisc.open(make_cue(tmp_path, with_track3=False))


def test_cue_with_a_track_line_before_any_file_line_says_so(tmp_path):
    cue = tmp_path / "Game.cue"
    cue.write_text("  TRACK 03 MODE1/2352\n    INDEX 01 00:00:00\n")
    with pytest.raises(DiscError, match=r"TRACK 03 .*no FILE line before it"):
        DcDisc.open(cue)


def test_missing_bin_beside_cue_raises(tmp_path):
    with pytest.raises(DiscError, match="Track 3"):
        DcDisc.open(make_cue(tmp_path))


@needs_bsdtar
def test_7z_is_extracted_once_into_out(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    make_disc(src)
    (src / "Game (Track 1).bin").write_bytes(b"not extracted")
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    archive = archive_dir / "Game.7z"
    subprocess.run(
        [TAR, "-cf", str(archive), "--format", "7zip", "-C", str(src), "Game.cue", TRACK3, "Game (Track 1).bin"],
        check=True,
    )
    out = tmp_path / "out"

    extractions = _count_extractions(monkeypatch)

    first = DcDisc.open(archive, out=out)
    second = DcDisc.open(archive, out=out)

    assert first.product == second.product == PRODUCT
    assert len(extractions()) == 2                                       # the cue, then track 3; once
    assert sorted(p.name for p in (out / "dc" / "Game").iterdir()) == sorted(["Game.cue", TRACK3])
    assert sorted(p.name for p in archive_dir.iterdir()) == ["Game.7z"]
    _assert_extracted(src, out / "dc" / "Game")


def _archive_with_disc(tmp_path, prefix=""):
    src = tmp_path / "src"
    (src / prefix).mkdir(parents=True)
    make_disc(src / prefix)
    archive = tmp_path / "Game.7z"
    members = [f"{prefix}Game.cue", f"{prefix}{TRACK3}"]
    subprocess.run([TAR, "-cf", str(archive), "--format", "7zip", "-C", str(src), *members], check=True)
    return archive


def _assert_extracted(src, dest, prefix=""):
    """The extracted cue and track are byte for byte the ones that went into the archive."""
    for name in ("Game.cue", TRACK3):
        assert (dest / prefix / name).read_bytes() == (src / prefix / name).read_bytes(), name


def _count_extractions(monkeypatch):
    """Record each member extracted, at the reader seam: tar or py7zr, whichever reads the archive on this system."""
    members = []
    for cls in (disc_module._TarArchive, disc_module._SevenZipArchive):
        real = cls.extract

        def recording(self, member, dest, _real=real):
            members.append(member)
            return _real(self, member, dest)

        monkeypatch.setattr(cls, "extract", recording)
    return lambda: list(members)


@needs_bsdtar
def test_leftover_partial_extraction_is_not_reused(tmp_path, monkeypatch):
    archive = _archive_with_disc(tmp_path)
    out = tmp_path / "out"
    partial = out / "dc" / "Game.partial"
    partial.mkdir(parents=True)
    (partial / TRACK3).write_bytes(b"truncated")
    (partial / "Game.cue").write_text("junk")
    extractions = _count_extractions(monkeypatch)
    disc = DcDisc.open(archive, out=out)
    assert disc.product == PRODUCT
    assert len(extractions()) == 2                                             # the cue, then track 3
    assert not partial.exists()
    _assert_extracted(tmp_path / "src", out / "dc" / "Game")


@needs_bsdtar
def test_nested_archive_layout_is_extracted_once(tmp_path, monkeypatch):
    archive = _archive_with_disc(tmp_path, prefix="sub/")
    out = tmp_path / "out"
    extractions = _count_extractions(monkeypatch)
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert len(extractions()) == 2                                             # the cue, then track 3; once
    _assert_extracted(tmp_path / "src", out / "dc" / "Game", "sub/")


SATURN_CUE = (
    'FILE "Sat (Track 01).bin" BINARY\n  TRACK 01 MODE1/2352\n    INDEX 01 00:00:00\n'
    'FILE "Sat (Track 02).bin" BINARY\n  TRACK 02 AUDIO\n    INDEX 01 00:02:00\n'
    'FILE "Sat (Track 03).bin" BINARY\n  TRACK 03 AUDIO\n    INDEX 01 00:02:00\n'
)
SATURN_TRACKS = ["Sat (Track 01).bin", "Sat (Track 02).bin", "Sat (Track 03).bin"]


def raw_sector(user: bytes) -> bytes:
    """One MODE1/2352 frame: 16 bytes of sync and header, 2048 user bytes, 288 bytes of EDC/ECC."""
    return b"\x00" + b"\xff" * 10 + b"\x00" + b"\x00\x02\x00\x01" + user.ljust(SECTOR, b" ") + b"\x00" * 288


def saturn_ip(product: str = "MK-81200", magic: bytes = b"SEGA SEGASATURN ") -> bytes:
    ip = bytearray(b" " * SECTOR)
    ip[0x00:0x10] = magic
    ip[0x10:0x20] = b"SEGA ENTERPRISES"
    ip[0x20:0x2A] = product.ljust(10).encode("ascii")
    ip[0x2A:0x30] = b"V1.000"
    return bytes(ip)


def make_saturn_disc(directory: Path, ip: bytes | None = None) -> Path:
    """A Redump-shaped Saturn dump: data track 1 with the Saturn IP, audio tracks 2 and 3."""
    (directory / SATURN_TRACKS[0]).write_bytes(raw_sector(saturn_ip() if ip is None else ip) * 2)
    for name in SATURN_TRACKS[1:]:
        (directory / name).write_bytes(b"\x00" * 2352 * 4)
    cue = directory / "Sat.cue"
    cue.write_text(SATURN_CUE)
    return cue


def test_saturn_cue_is_refused_with_its_product(tmp_path):
    cue = make_saturn_disc(tmp_path)
    with pytest.raises(DiscError, match=r"not a Dreamcast disc: SEGA SEGASATURN, product MK-81200") as info:
        DcDisc.open(cue)
    assert "AUDIO" in str(info.value)


def test_unknown_header_with_an_audio_track_3_is_named(tmp_path):
    cue = make_saturn_disc(tmp_path, ip=saturn_ip(magic=b"SONY PLAYSTATION"))
    with pytest.raises(DiscError, match=r"not a Dreamcast disc \(header b'SONY PLAYSTATION'\)"):
        DcDisc.open(cue)


@needs_bsdtar
def test_saturn_7z_is_refused_after_extracting_the_cue_alone(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    make_saturn_disc(src)
    archive = tmp_path / "Sat.7z"
    subprocess.run([TAR, "-cf", str(archive), "--format", "7zip", "-C", str(src), "Sat.cue", *SATURN_TRACKS],
                   check=True)
    out = tmp_path / "out"
    extractions = _count_extractions(monkeypatch)
    with pytest.raises(DiscError, match=r"SEGA SEGASATURN, product MK-81200"):
        DcDisc.open(archive, out=out)
    extracted = extractions()
    assert extracted == ["Sat.cue"]
    assert not (out / "dc" / "Sat").exists()
    assert not (out / "dc" / "Sat.partial").exists()


def test_nonexistent_cue_raises_disc_error(tmp_path):
    with pytest.raises(DiscError, match="cue"):
        DcDisc.open(tmp_path / "absent.cue")


@needs_bsdtar
def test_missing_tar_raises_disc_error(tmp_path, monkeypatch):
    archive = _archive_with_disc(tmp_path)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(disc_module, "_tar_program", lambda: "dtools-no-such-tar")   # Windows finds System32 even without PATH
    monkeypatch.setitem(sys.modules, "py7zr", None)                      # no py7zr fallback either
    with pytest.raises(DiscError, match="tar"):
        DcDisc.open(archive, out=tmp_path / "out")


def test_course_files_variant_numbering():
    assert course_files(0, "R")["ghost"] == "GHOST_8.BIN"     # 1ST_READ.BIN: 8 = REV, 16 = MIR, 24 = MIR-REV
    assert course_files(0, "M")["ghost"] == "GHOST_16.BIN"
    assert course_files(1, "MR")["ghost"] == "GHOST_25.BIN"
    assert course_files(0)["data"] == "CS_DATA0.BIN"
    assert course_files(2, "M")["data"] == "CSDAT2M.BIN"
    assert course_files(5)["bg_name"] == "NAME_BG05.BIN"


def test_course_files_refuses():
    for n in (8, -1, 1.5, "1", True, None):
        with pytest.raises(DcError, match="course"):
            course_files(n)
    for v in ("X", "r", None, 0, ("R",)):
        with pytest.raises(DcError, match="variant"):
            course_files(0, v)
    with pytest.raises(DcError, match="nope"):
        course_files(0, "", "nope")
    assert course_files(3, "MR", "model") == "CS3MR_POL.BIN"
    assert course_files(3, "MR", kind="ghost") == "GHOST_27.BIN"
    assert set(course_files(7, "M")) == {"model", "textures", "data", "model_v", "textures_v", "ghost", "bg_lg",
                                         "bg_chp", "bg_name"}


def test_course_files_accepts_numpy_integers():
    """n follows tablebase.is_int, as the encoders and screens do: numpy integers yes, bool and numpy bool no."""
    import numpy as np
    assert course_files(np.int64(3), "R", "data") == "CSDAT3R.BIN"
    assert course_files(np.uint8(7))["bg_name"] == "NAME_BG07.BIN"
    for n in (np.bool_(True), np.bool_(False), False, np.float64(1.0), np.int64(8), np.int8(-1)):
        with pytest.raises(DcError, match="course"):
            course_files(n)


def test_disc_course_file_refuses_a_bad_kind(tmp_path):
    disc = DcDisc.open(make_disc(tmp_path))
    with pytest.raises(DcError, match="nope"):
        disc.course_file(1, "nope")
    with pytest.raises(DcError, match="course"):
        disc.course_file(9, "model")


def test_dc_errors_are_decode_errors():
    from dtools.arcade.errors import DecodeError
    assert issubclass(DcError, DecodeError)
    assert issubclass(DiscError, DcError)
    assert issubclass(IsoError, DcError)


@needs_bsdtar
@pytest.mark.skipif(sys.platform == "win32", reason="names with * or ? cannot exist on a Windows filesystem")
def test_archive_member_names_are_not_patterns(tmp_path):
    """A member name holding glob characters reaches tar literally: only that member is read."""
    src = tmp_path / "src"
    src.mkdir()
    names = ["Gam*.cue", "Gamx.cue", "Gam[e].cue", TRACK3]
    for name in names:
        (src / name).write_bytes(b"x")
    (src / "Gam*.cue").write_text(f'FILE "{TRACK3}" BINARY\n  TRACK 03 MODE1/2352\n')
    archive = tmp_path / "g.7z"
    subprocess.run([TAR, "-cf", str(archive), "--format", "7zip", "-C", str(src), *names], check=True)
    dest = tmp_path / "dest"
    dest.mkdir()
    disc_module._tar("-xf", str(archive), "-C", str(dest), disc_module._literal("Gam*.cue"))
    assert [p.name for p in dest.iterdir()] == ["Gam*.cue"]
    assert disc_module._archive_head(archive, "Gam[e].cue") == b"x"
    assert disc_module._archive_head(archive, "Gam*.cue").startswith(b"FILE")


@needs_bsdtar
def test_archive_member_names_with_brackets_are_not_patterns(tmp_path):
    """`[` is legal in Windows file names: a member `Gam[e].cue` reaches tar literally, `Game.cue` is not matched."""
    src = tmp_path / "src"
    src.mkdir()
    names = ["Gam[e].cue", "Game.cue"]
    (src / "Gam[e].cue").write_bytes(b"bracket")
    (src / "Game.cue").write_bytes(b"plain")
    archive = tmp_path / "g.7z"
    subprocess.run([TAR, "-cf", str(archive), "--format", "7zip", "-C", str(src), *names], check=True)
    dest = tmp_path / "dest"
    dest.mkdir()
    disc_module._tar("-xf", str(archive), "-C", str(dest), disc_module._literal("Gam[e].cue"))
    assert [p.name for p in dest.iterdir()] == ["Gam[e].cue"]
    assert disc_module._archive_head(archive, "Gam[e].cue") == b"bracket"


@needs_bsdtar
def test_extraction_over_an_invalid_dest_keeps_unrelated_files(tmp_path):
    archive = _archive_with_disc(tmp_path)
    out = tmp_path / "out"
    dest = out / "dc" / "Game"
    dest.mkdir(parents=True)
    (dest / "notes.txt").write_text("mine")
    (dest / "Game.cue").write_text("junk")
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert (dest / "notes.txt").read_text() == "mine"


@needs_bsdtar
def test_truncated_track_3_is_not_reused(tmp_path):
    archive = _archive_with_disc(tmp_path)
    out = tmp_path / "out"
    assert DcDisc.open(archive, out=out).product == PRODUCT
    track = out / "dc" / "Game" / TRACK3
    track.write_bytes(track.read_bytes()[:5000])               # a track cut short by an old interrupted run
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert track.stat().st_size > 5000


@needs_bsdtar
def test_unreadable_listing_means_no_reuse(tmp_path, monkeypatch):
    """A `tar -tvf` line the size parser cannot read (owner name with spaces) never lets a track be reused."""
    archive = _archive_with_disc(tmp_path)
    out = tmp_path / "out"
    DcDisc.open(archive, out=out)
    track = out / "dc" / "Game" / TRACK3
    track.write_bytes(track.read_bytes()[:5000])
    real_tar = disc_module._tar

    def odd_listing(*args):
        done = real_tar(*args)
        if args[0] == "-tvf":
            done.stdout = done.stdout.replace(" 0 ", " 0 some owner ", 1)
        return done

    monkeypatch.setattr(disc_module, "_tar", odd_listing)
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert track.stat().st_size == (tmp_path / "src" / TRACK3).stat().st_size


def test_member_sizes_skips_unreadable_lines():
    ok = "-rw-r--r--  0 0      0      1234  8 Nov 10:52 Game (Track 3).bin"
    assert disc_module._member_sizes(ok + "\n-rw-r--r--  0 a b c 12  8 Nov 10:52 x") == {"Game (Track 3).bin": 1234}


def test_cue_naming_an_absolute_track_under_dest_is_a_disc_error(tmp_path):
    dest = tmp_path / "dc" / "Game"
    dest.mkdir(parents=True)
    other = tmp_path / "elsewhere.bin"
    other.write_bytes(b"x")
    (dest / "Game.cue").write_text(f'FILE "{other}" BINARY\n  TRACK 03 MODE1/2352\n')
    with pytest.raises(DiscError, match="not inside"):
        disc_module._find_extracted(dest, {})


def test_disc_is_a_context_manager_and_closes_the_track(tmp_path):
    with DcDisc.open(make_disc(tmp_path)) as disc:
        f = disc.iso._f
        assert not f.closed
    assert f.closed
    disc.close()


def test_open_closes_the_track_when_the_ip_sector_is_missing(tmp_path, monkeypatch):
    (tmp_path / TRACK3).write_bytes(make_image({"A.BIN": b"x"}, base=GD_BASE)[:2352 * 17])   # ends after the PVD
    cue = make_cue(tmp_path)
    opened = []
    real_init = disc_module.Iso2352.__init__

    def spy(self, *a, **k):
        real_init(self, *a, **k)
        opened.append(self)

    monkeypatch.setattr(disc_module.Iso2352, "__init__", spy)
    monkeypatch.setattr(DcDisc, "__init__", lambda self, cue, iso: iso.sector(GD_BASE + 99999))
    with pytest.raises(DiscError, match="past end"):
        DcDisc.open(cue)
    assert opened and opened[0]._f.closed


def test_real_disc(dc_path):
    disc = DcDisc.open(dc_path)
    assert disc.verify() == []
    assert disc.title.startswith("DAYTONAUSA")
    assert len(disc.course_file(1, "model")) == 2334752
    assert disc.course_file(0, "model")[:8] == bytes.fromhex("0002000017110020")


def test_tar_program_prefers_bsdtar_when_on_path(monkeypatch):
    monkeypatch.setattr(disc_module, "WINDOWS_TAR", Path("no-such-dir") / "tar.exe")      # the PATH rule is the test
    monkeypatch.setattr(disc_module.shutil, "which", lambda name: "/usr/bin/bsdtar" if name == "bsdtar" else None)
    assert disc_module._tar_program() == "bsdtar"


def test_tar_program_falls_back_to_tar(monkeypatch):
    monkeypatch.setattr(disc_module, "WINDOWS_TAR", Path("no-such-dir") / "tar.exe")
    monkeypatch.setattr(disc_module.shutil, "which", lambda name: None)
    assert disc_module._tar_program() == "tar"


def test_archive_no_tar_can_read_names_bsdtar(tmp_path, monkeypatch):
    """GNU tar cannot read .7z: the DiscError names the remedy (bsdtar, libarchive-tools, or the .cue)."""
    archive = tmp_path / "Game.7z"
    archive.write_bytes(b"7z\xbc\xaf\x27\x1c" + bytes(64))
    monkeypatch.setattr(disc_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(disc_module, "_tar", lambda *a: subprocess.CompletedProcess(
        a, 2, "", "tar: This does not look like a tar archive"))
    with pytest.raises(DiscError, match=r"bsdtar.*libarchive-tools.*\.cue"):
        DcDisc.open(archive, out=tmp_path / "out")


def test_archive_bsdtar_failure_names_no_remedy(tmp_path, monkeypatch):
    """When bsdtar (which reads .7z) is the program and still cannot list the archive, its own message is the error:
    the install-bsdtar remedy is only for plain `tar`."""
    archive = tmp_path / "Game.7z"
    archive.write_bytes(b"7z\xbc\xaf\x27\x1c" + bytes(64))
    monkeypatch.setattr(disc_module, "WINDOWS_TAR", Path("no-such-dir") / "tar.exe")      # the PATH rule is the test
    monkeypatch.setattr(disc_module.shutil, "which", lambda name: "/usr/bin/bsdtar" if name == "bsdtar" else None)
    monkeypatch.setattr(disc_module, "_tar", lambda *a: subprocess.CompletedProcess(
        a, 1, "", "bsdtar: Truncated 7-Zip file data"))
    with pytest.raises(DiscError, match="Truncated 7-Zip file data") as err:
        DcDisc.open(archive, out=tmp_path / "out")
    assert "libarchive-tools" not in str(err.value)


def _py7zr_archive(tmp_path, prefix=""):
    """The disc archive written with py7zr (no bsdtar needed); the source files are under tmp_path / "src"."""
    py7zr = pytest.importorskip("py7zr")
    src = tmp_path / "src"
    (src / prefix).mkdir(parents=True)
    make_disc(src / prefix)
    archive = tmp_path / "Game.7z"
    with py7zr.SevenZipFile(archive, "w") as z:
        for name in (f"{prefix}Game.cue", f"{prefix}{TRACK3}"):
            z.write(src / name, name)
    return archive


def test_py7zr_when_tar_cannot_list(tmp_path, monkeypatch):
    archive = _py7zr_archive(tmp_path)
    monkeypatch.setattr(disc_module._TarArchive, "lists", staticmethod(lambda archive: False))
    out = tmp_path / "out"
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert sorted(p.name for p in (out / "dc" / "Game").iterdir()) == sorted(["Game.cue", TRACK3])
    _assert_extracted(tmp_path / "src", out / "dc" / "Game")


@needs_bsdtar
def test_py7zr_extracts_the_same_bytes(tmp_path):
    archive = _py7zr_archive(tmp_path)
    got = {}
    for kind in (disc_module._TarArchive, disc_module._SevenZipArchive):
        dest = tmp_path / kind.__name__
        dest.mkdir()
        reader = kind(archive)
        assert sorted(reader.members()) == sorted(["Game.cue", TRACK3])
        assert reader.sizes()[TRACK3] == (tmp_path / "src" / TRACK3).stat().st_size
        for m in ("Game.cue", TRACK3):
            reader.extract(m, dest)
        got[kind] = {m: (dest / m).read_bytes() for m in ("Game.cue", TRACK3)}
        assert reader.head(TRACK3, 64) == (tmp_path / "src" / TRACK3).read_bytes()[:64]
    assert got[disc_module._TarArchive] == got[disc_module._SevenZipArchive]


def test_py7zr_partial_not_reused(tmp_path, monkeypatch):
    archive = _py7zr_archive(tmp_path)
    monkeypatch.setattr(disc_module._TarArchive, "lists", staticmethod(lambda archive: False))
    out = tmp_path / "out"
    partial = out / "dc" / "Game.partial"
    partial.mkdir(parents=True)
    (partial / TRACK3).write_bytes(b"truncated")
    (partial / "Game.cue").write_text("junk")
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert not partial.exists()
    _assert_extracted(tmp_path / "src", out / "dc" / "Game")


def test_windows_prefers_system32_tar(monkeypatch, tmp_path):
    fake = tmp_path / "tar.exe"
    fake.write_bytes(b"")
    monkeypatch.setattr(disc_module, "WINDOWS_TAR", fake)
    monkeypatch.setattr(disc_module.sys, "platform", "win32")
    monkeypatch.setattr(disc_module.shutil, "which", lambda name: None if name == "bsdtar" else r"C:\Git\usr\bin\tar.exe")
    assert disc_module._tar_program() == str(fake)
    monkeypatch.setattr(disc_module, "WINDOWS_TAR", tmp_path / "absent.exe")
    assert disc_module._tar_program() == "tar"


def test_tar_remedy_names_windows():
    assert "Windows" in disc_module._TAR_REMEDY and "py7zr" in disc_module._TAR_REMEDY


def test_remedy_without_py7zr_does_not_say_pip_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "py7zr", None)
    text = disc_module._remedy_text()
    assert "pip install" not in text and "System32" in text and ".cue" in text


def test_tar_lists_but_cannot_extract_falls_back(tmp_path, monkeypatch):
    archive = _py7zr_archive(tmp_path)

    def broken(self, member, dest):
        raise DiscError("tar: cannot decode LZMA2")

    monkeypatch.setattr(disc_module._TarArchive, "lists", staticmethod(lambda archive: True))
    monkeypatch.setattr(disc_module._TarArchive, "extract", broken)
    out = tmp_path / "out"
    assert DcDisc.open(archive, out=out).product == PRODUCT
    assert not (out / "dc" / "Game.partial").exists()
    _assert_extracted(tmp_path / "src", out / "dc" / "Game")


def test_tar_and_py7zr_both_failing_names_both(tmp_path, monkeypatch):
    archive = _py7zr_archive(tmp_path)

    def broken(self, member, dest):
        raise DiscError("boom")

    monkeypatch.setattr(disc_module._TarArchive, "lists", staticmethod(lambda archive: True))
    monkeypatch.setattr(disc_module._TarArchive, "extract", broken)
    monkeypatch.setattr(disc_module._SevenZipArchive, "extract", broken)
    with pytest.raises(DiscError, match="boom.*py7zr: boom"):
        DcDisc.open(archive, out=tmp_path / "out")


def test_py7zr_list_errors_are_disc_errors(tmp_path, monkeypatch):
    py7zr = pytest.importorskip("py7zr")
    archive = _py7zr_archive(tmp_path)

    def broken(self):
        raise RuntimeError("bad header")

    monkeypatch.setattr(py7zr.SevenZipFile, "list", broken)
    with pytest.raises(DiscError, match="bad header"):
        disc_module._SevenZipArchive(archive).members()


def test_py7zr_member_in_a_folder(tmp_path, monkeypatch):
    archive = _py7zr_archive(tmp_path, prefix="Game/")
    monkeypatch.setattr(disc_module._TarArchive, "lists", staticmethod(lambda archive: False))
    reader = disc_module._SevenZipArchive(archive)
    assert reader.head(f"Game/{TRACK3}", 16) == (tmp_path / "src" / "Game" / TRACK3).read_bytes()[:16]
    out = tmp_path / "out"
    assert DcDisc.open(archive, out=out).product == PRODUCT
    _assert_extracted(tmp_path / "src", out / "dc" / "Game", "Game/")
