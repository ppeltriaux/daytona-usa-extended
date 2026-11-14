"""The Dreamcast Daytona USA disc: track 3 of the Redump dump, and its course files."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable

from dtools import config
from dtools.arcade.tablebase import is_int
from dtools.dc.errors import DcError
from dtools.dc.iso import Iso2352, IsoError

PRODUCT = "MK-51037"
GD_BASE = 45000
COURSE_COUNT = 8
VARIANTS = ("", "R", "M", "MR")             # 1ST_READ.BIN: entry 8 = REV, 16 = MIR, 24 = MIR-REV
DREAMCAST_MAGIC = b"SEGA SEGAKATANA "

DATA_MODE = "MODE1/2352"            # track 3's mode on the Dreamcast dump: the high-density data track
SATURN_MAGIC = b"SEGA SEGASATURN "
RAW_SECTOR = 2352
_USER_OFFSET = {"MODE1/2352": 16, "MODE2/2352": 24, "MODE1/2048": 0, "MODE2/2336": 8}   # first user byte
_FILE_RE = re.compile(r'FILE\s+"(.+)"\s+\w+', re.IGNORECASE)
_TRACK_RE = re.compile(r"TRACK\s+(\d+)\s+(\S+)", re.IGNORECASE)


class DiscError(DcError):
    pass


def course_files(n: int, v: str = "", kind: str | None = None) -> dict[str, str] | str:
    """File names of course n (0-7), variant v (one of VARIANTS), by kind; the one name when `kind` is given.
    n is a Python or numpy integer, never a bool (`tablebase.is_int`). DcError for an n outside 0-7 (or not an
    integer), a v not in VARIANTS or a kind that is not one of the nine."""
    if not is_int(n) or not 0 <= n < COURSE_COUNT:
        raise DcError(f"course {n!r}: the Dreamcast courses are 0-{COURSE_COUNT - 1}")
    n = int(n)
    if not isinstance(v, str) or v not in VARIANTS:
        raise DcError(f"course {n}: variant {v!r} is not one of {VARIANTS}")
    files = {
        "model": f"CS{n}{v}_POL.BIN",
        "textures": f"CS{n}{v}_TEX.BIN",
        "data": f"CS_DATA{n}.BIN" if v == "" else f"CSDAT{n}{v}.BIN",
        "model_v": f"VCS{n}{v}_POL.BIN",
        "textures_v": f"VCS{n}{v}_TEX.BIN",
        "ghost": f"GHOST_{8 * VARIANTS.index(v) + n}.BIN",
        "bg_lg": f"LG_BG{n}.BIN",
        "bg_chp": f"CHP_BG{n}.BIN",
        "bg_name": f"NAME_BG{n:02d}.BIN",   # two digits on the disc
    }
    if kind is None:
        return files
    if not isinstance(kind, str) or kind not in files:
        raise DcError(f"course {n}: no file kind {kind!r} (one of {', '.join(files)})")
    return files[kind]


def _cue_tracks(cue: Path) -> dict[int, tuple[str, str]]:
    """Track number -> (the .bin named by the FILE line before it, its mode: MODE1/2352, AUDIO, ...)."""
    try:
        text = Path(cue).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DiscError(f"{cue}: cannot read the cue sheet: {exc}") from exc
    tracks: dict[int, tuple[str, str]] = {}
    last_file = None
    for line in text.splitlines():
        line = line.strip()
        if m := _FILE_RE.match(line):
            last_file = m.group(1)
        elif m := _TRACK_RE.match(line):
            if last_file is None:
                if int(m.group(1)) == 3:
                    raise DiscError(f"{cue}: TRACK 03 has no FILE line before it")
                continue
            tracks.setdefault(int(m.group(1)), (last_file, m.group(2).upper()))
    return tracks


def _track3_name(cue: Path) -> str:
    """The .bin named by the FILE line directly before 'TRACK 03'."""
    track = _cue_tracks(cue).get(3)
    if track is None:
        raise DiscError(f"{cue}: no TRACK 03 (a FILE line followed by TRACK 03)")
    return track[0]


def _check_track3(cue: Path, head: Callable[[str], bytes]) -> str:
    """Track 3's file name when the cue's track 3 is a MODE1/2352 data track. Otherwise DiscError naming the disc
    from track 1's IP header (`head(name)` returns the first raw sector of the track file `name`): a Saturn dump
    (data track 1, audio tracks after it) is refused as `not a Dreamcast disc: SEGA SEGASATURN, product MK-81200`."""
    tracks = _cue_tracks(cue)
    if 3 not in tracks:
        raise DiscError(f"{cue}: no TRACK 03 (a FILE line followed by TRACK 03)")
    name, mode = tracks[3]
    if mode == DATA_MODE:
        return name
    why = f"{Path(cue).name}: track 3 is {mode}, not {DATA_MODE}"
    if 1 not in tracks or tracks[1][1] not in _USER_OFFSET:
        raise DiscError(f"not a Dreamcast GD-ROM dump ({why}; no data track 1 to read the header from)")
    t1_name, t1_mode = tracks[1]
    off = _USER_OFFSET[t1_mode]
    ip = head(t1_name)[off:off + 0x100]
    magic = ip[0x00:0x10]
    if magic == SATURN_MAGIC:
        product = ip[0x20:0x2A].decode("latin-1").strip()
        raise DiscError(f"not a Dreamcast disc: {magic.decode('latin-1').strip()}, product {product} ({why})")
    if magic == DREAMCAST_MAGIC:
        product = ip[0x40:0x4A].decode("latin-1").strip()
        raise DiscError(f"a Dreamcast disc (product {product}) but not a GD-ROM dump with the game on track 3 "
                        f"({why})")
    raise DiscError(f"not a Dreamcast disc (header {magic!r}) ({why})")


def track3_path(cue: Path) -> Path:
    """The track 3 file beside the `cue` (its name checked as `_check_track3` does, track 1's header read from the
    file beside the cue when needed); DiscError as `_check_track3` raises. The file's existence is not checked."""
    return cue.parent / _check_track3(cue, lambda name: _file_head(cue.parent / name))


def _file_head(path: Path) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(RAW_SECTOR)
    except OSError as exc:
        raise DiscError(f"cannot read the header of {path}: {exc}") from exc


def _literal(member: str) -> str:
    """`member` escaped so that tar matches it literally, not as a pattern (`*`, `?`, `[` and `\\`)."""
    return re.sub(r"([\\*?\[])", r"\\\1", member)


_TAR_REMEDY = ("only libarchive's bsdtar reads .7z archives: install it (Debian/Ubuntu: libarchive-tools; macOS has "
               "it as tar) or pass the disc's .cue instead")


def _tar_program() -> str:
    """`bsdtar` when it is on PATH (GNU tar, Linux's `tar`, cannot read .7z), else `tar` (bsdtar on macOS)."""
    return "bsdtar" if shutil.which("bsdtar") else "tar"


def _remedy(program: str) -> str:
    """"; " + _TAR_REMEDY when `program` is plain `tar` (maybe GNU tar, which cannot read .7z), else nothing (bsdtar
    reads .7z: its own message says what is wrong)."""
    return f"; {_TAR_REMEDY}" if program == "tar" else ""


def _tar(*args: str) -> subprocess.CompletedProcess:
    program = _tar_program()
    try:
        return subprocess.run([program, *args], capture_output=True, text=True)
    except OSError as exc:
        raise DiscError(f"cannot run {program}: {exc}{_remedy(program)}") from exc


def _archive_head(archive: Path, member: str) -> bytes:
    """The first raw sector of `member`, streamed from the archive (nothing is written to disk)."""
    program = _tar_program()
    try:
        proc = subprocess.Popen([program, "-xOf", str(archive), _literal(member)],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise DiscError(f"cannot run {program}: {exc}{_remedy(program)}") from exc
    try:
        return proc.stdout.read(RAW_SECTOR)
    finally:
        proc.kill()
        proc.communicate()


def _member_sizes(listing: str) -> dict[str, int]:
    """Member path -> size from `tar -tvf` output (perms, links, owner, group, size, three date fields, name)."""
    sizes = {}
    for line in listing.splitlines():
        parts = line.split(None, 8)
        if len(parts) == 9 and parts[4].isdigit():
            sizes[parts[8]] = int(parts[4])
    return sizes


def _find_extracted(dest: Path, sizes: dict[str, int]) -> Path | None:
    """A cue under dest (at any depth) whose track 3 file is present and exactly the size the archive lists for it
    (a track cut short by an interrupted extraction, or one whose size the listing does not give, is not reused).
    DiscError for a cue naming a track outside dest or a track that cannot be statted."""
    if not dest.is_dir():
        return None
    for cue in sorted(dest.rglob("*.cue")):
        name = _safe_track3(cue)
        track = cue.parent / name
        if not name or not track.is_file():
            continue
        try:
            want = sizes.get(track.relative_to(dest).as_posix())
            have = track.stat().st_size
        except ValueError as exc:
            raise DiscError(f"{cue}: track 3 {name!r} is not inside {dest}") from exc
        except OSError as exc:
            raise DiscError(f"{track}: cannot stat the extracted track: {exc}") from exc
        if want is not None and have == want:
            return cue
    return None


def _extract_7z(archive: Path, dest: Path) -> Path:
    """Extract the cue and track 3 from a Redump .7z into dest; return the cue path.

    Members are extracted at their archive paths (flat in Redump archives). The cue is extracted first, alone, and
    checked (`_check_track3`, reading track 1's header from the archive when needed) before any track is extracted,
    so a Saturn archive is refused having written only its cue, which is then removed. The extraction goes into a
    sibling '<dest>.partial' directory; on success it is renamed to dest, or, when dest exists but holds no usable
    extraction, its two files replace the stale ones in dest (nothing else in dest is touched). An interrupted
    extraction is never reused: the reuse check compares the track's size with the archive's.
    """
    listing = _tar("-tvf", str(archive))
    if listing.returncode != 0:
        raise DiscError(f"{archive}: cannot list archive: {listing.stderr.strip()}{_remedy(_tar_program())}")
    sizes = _member_sizes(listing.stdout)
    found = _find_extracted(dest, sizes)
    if found is not None:
        return found
    members = [m for m in _tar("-tf", str(archive)).stdout.splitlines() if m and not m.endswith("/")]
    cues = [m for m in members if m.lower().endswith(".cue")]
    if not cues:
        raise DiscError(f"{archive}: archive has no .cue (members: {members[:5]})")
    cue_member = cues[0]
    folder = str(PurePosixPath(cue_member).parent)

    def member(name: str) -> str:
        m = name if folder == "." else f"{folder}/{name}"
        if m not in members:
            raise DiscError(f"{archive}: the cue names {name!r}, which the archive does not hold")
        return m

    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        done = _tar("-xf", str(archive), "-C", str(partial), _literal(cue_member))
        if done.returncode != 0:
            raise DiscError(f"{archive}: extraction of the cue failed: {done.stderr.strip()}")
        track3 = member(_check_track3(partial / cue_member, lambda name: _archive_head(archive, member(name))))
        done = _tar("-xf", str(archive), "-C", str(partial), _literal(track3))
        if done.returncode != 0:
            raise DiscError(f"{archive}: extraction failed: {done.stderr.strip()}")
        if dest.exists():
            for m in (cue_member, track3):
                target = dest / m
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(partial / m, target)
            shutil.rmtree(partial)
        else:
            partial.rename(dest)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return dest / cue_member


def _safe_track3(cue: Path) -> str:
    try:
        return _track3_name(cue)
    except DiscError:
        return ""


class DcDisc:
    def __init__(self, cue: Path, iso: Iso2352):
        self.cue = cue
        self.iso = iso
        self.archive: Path | None = None    # the .7z it was extracted from (open() sets it); None for a cue
        ip = iso.sector(GD_BASE)
        self._magic = ip[0x00:0x10]
        self.product = ip[0x40:0x4A].decode("latin-1").strip()
        self.title = ip[0x80:0x100].decode("latin-1").strip()

    @classmethod
    def open(cls, path: Path, out: Path | None = None) -> DcDisc:
        path = Path(path)
        if path.suffix.lower() == ".7z":
            base = Path(out) if out is not None else config.out_dir()
            cue = _extract_7z(path, base / "dc" / path.stem)
        else:
            cue = path
        track3 = track3_path(cue)
        if not track3.is_file():
            raise DiscError(f"Track 3 not found beside the cue: {track3}")
        try:
            iso = Iso2352(track3, lba_base=GD_BASE)
        except IsoError as exc:
            raise DiscError(str(exc)) from exc
        try:
            disc = cls(cue, iso)
        except IsoError as exc:
            iso.close()
            raise DiscError(str(exc)) from exc
        except BaseException:
            iso.close()
            raise
        disc.archive = path if path.suffix.lower() == ".7z" else None
        return disc

    def close(self) -> None:
        """Close the track file (harmless a second time)."""
        self.iso.close()

    def __enter__(self) -> DcDisc:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def verify(self) -> list[str]:
        """Problems found; an empty list means the disc is the expected one and complete."""
        if self._magic != DREAMCAST_MAGIC:
            return [f"not a Dreamcast disc (header {self._magic!r})"]
        problems = []
        if self.product != PRODUCT:
            problems.append(f"product {self.product}, expected {PRODUCT}")
        present = {e.path for e in self.iso.entries()}
        for n in range(COURSE_COUNT):
            for v in VARIANTS:
                for name in course_files(n, v).values():
                    if name not in present:
                        problems.append(name)
        return problems

    def read(self, name: str) -> bytes:
        return self.iso.read(name)

    def course_file(self, n: int, kind: str, v: str = "") -> bytes:
        return self.read(course_files(n, v, kind))
