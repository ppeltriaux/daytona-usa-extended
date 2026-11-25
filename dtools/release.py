"""The release set, Daytona USA Extended (Phase 12, M7; spec docs/superpowers/specs/2026-11-13-release-design.md,
"What gets built" item 2): one chain from the player's own MAME 0.289 `daytona` set and US Dreamcast disc.

`release_set` is Phase 11's shipped chain (tests/test_courses_mame.py `m5b_set`; the M5b playtest set,
tools/desert_city_playtest.py `m5b_set`, adds the start words) with the release row on the select:

1. `course.grow_ids`, `course.guard_new_ids`, `course.widen_ids` (sixteen course ids, the new ids' guards, the wide
   tables);
2. `build.converted_set(..., n, slot=n, scenery="model")` for each of RELEASE_CONVERTED, in that order (each step
   places at the first free bytes, so the order fixes the bytes);
   With `direction` (v0.2), each with `road_copy=True` (build.ROAD_COPY: the road copied across cell edges, so the
   game's culling no longer drops the road under the camera);
2b. (M9 stage 3, `direction=True`, the default) the eight courses in reverse after them, at ids 8-15:
   `build.converted_set(..., n, slot=n + 8, variant="R", scenery="keep")` for the arcade's three (their own
   scenery), then `scenery="model"` for 3-7;
3. `course.grow_rankings` (a ranking table per id);
4. `select.patch_select(..., select.ENTRIES, faceted_maps=direction)` (v0.2's faceted maps with the screen only;
   without it v0.1.2's painted ones): the release row, every course under its own name (the originals' over
   their name box, select.WINDOW_NAME);
5. `build.race_times` on each of RELEASE_TIME_IDS (v0.1.2: the converted courses' own time tables, build.RACE_TIMES;
   with `direction`, M9 stage 4, REVERSE_TIME_IDS too: the reverses' own;
   the originals keep the arcade's), then `build.long_time` on each of `long_ids` (none by default; test-only, as
   `times=False`: the gates' sets);
5b. `build.pit_stalls` on each of RELEASE_PIT_IDS (and with `direction`, M9 stage 4, `pit_stalls(..., n, "R")` for
   the reverses at ids 8-15) (v0.1.2: the converted courses' own pit stalls, the Dreamcast's,
   at the end of build.ID_DATA_FREE; before, they stopped on Beginner's and the stop ended at once); test-only
   `pits=False` leaves them out (v0.1.1's chain); with `direction`, then `course.patch_pit_heading` (the pit stop's
   angle differences wrapped to 16 bits, the trigger, the stall steering and the waypoint-passed test: Dinosaur
   Canyon's and Desert City's reverse boxes face -32768);
6. `title.extended_title`: "Extended" in the attract logo (title.PICKED, Pascal's pick), in the free custom data run
   title.TITLE_RUN (outside build.ID_DATA_FREE, so the views below land where they land without it) and two
   repointed words;
7. `build.viewed_set` (its tables are not owned spans: nothing allocates after it), with `direction` the reverse
   ids' too (build.REVERSE_VIEW);
8. with `direction`, `dirscreen.patch_direction`: the Course Direction screen after Circuit Select (NORMAL or
   REVERSE: course n or n + 8), in its own code and data areas.

The packaging (Phase 12 Task 5, "What gets built" item 4; the `dtools` command, dtools/cli.py): `check_inputs` verifies
the player's inputs (MAME 0.289's `daytona.zip` by size and CRC32, ROM_SIZES and romset.CRCS; track 3 of the US
Dreamcast disc, DISC_TRACK3, and DcDisc.verify; an output directory that is not the ROM directory under any spelling and
can be written); `write_release` writes `<out>/daytona.zip` (complete before it replaces an earlier one), copies
`model1io.zip` and writes `<out>/build-report.txt`; `member_sha1` is the byte-for-byte measure of a build (the zip's
own bytes vary with its timestamps).

Errors are ReleaseError (a DecodeError); the steps' own errors pass through."""

from __future__ import annotations

import hashlib
import os
import platform
import secrets
import shlex
import shutil
import struct
import subprocess
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import golden
from dtools.arcade import select
from dtools.arcade import title as _title
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import CRCS, CUSTOM_FILES, REGIONS, RomSet
from dtools.arcade.tablebase import is_int
from dtools.convert import build
from dtools.dc.disc import PRODUCT, DcDisc, DiscError, track3_path


class ReleaseError(DecodeError):
    pass


RELEASE_CONVERTED = (3, 4, 7, 6, 5)
"""The Dreamcast courses the release set converts, in order, each at its own course id: Desert City, National Park
Speedway, Mermaid Lake, Rin Rin Rink, Circuit Pixie (Phase 11's order, tests/test_courses_mame.py M5B_CONVERTED)."""
RELEASE_TIME_IDS = (3, 4, 5, 6, 7)
"""The course ids whose time tables `release_set` rewrites with build.RACE_TIMES (v0.1.2): the converted courses."""
REVERSE_TIME_IDS = tuple(range(8, 16))
"""M9 stage 4: the reverses' own time tables (build.RACE_TIMES[8..15]), written with `direction`."""
RELEASE_LONG_IDS = (3, 4, 5, 6, 7)
"""The course ids whose start time v0.1-v0.1.1 raised to build.LONG_TIME (the M5b playtest set's, tools/
desert_city_playtest.py M5B_LONG_TIME_IDS): `release_set(rs, disc, RELEASE_LONG_IDS, times=False, pits=False)` is
v0.1.1's release chain."""
REVERSE_IDS = 8
"""M9: each course n's reverse is at id n + REVERSE_IDS (the eight in reverse at ids 8-15)."""
ARCADE_COURSES = 3
"""Ids 0-2, the arcade's own courses: in reverse they keep their scenery (stage 2's "keep"); 3-7 re-mesh theirs."""
RELEASE_PIT_IDS = (3, 4, 5, 6, 7)
"""The course ids `release_set` gives the Dreamcast's pit stalls (build.pit_stalls, v0.1.2): the converted courses."""


def _long_ids(long_ids) -> tuple[int, ...]:
    if not isinstance(long_ids, (tuple, list)):
        raise ReleaseError(f"long_ids {long_ids!r:.60} is not a tuple of course ids")
    for k in long_ids:
        if not is_int(k) or k not in RELEASE_CONVERTED:
            raise ReleaseError(f"long_ids: {k!r:.60} is not a converted course id ({min(RELEASE_CONVERTED)}-"
                               f"{max(RELEASE_CONVERTED)}; the originals keep the arcade's times)")
    return tuple(int(k) for k in long_ids)


def release_set(rs: RomSet, disc: DcDisc, long_ids: tuple[int, ...] = (),
                *, title: bool = True, times: bool = True, pits: bool = True, direction: bool = True,
                far_view: bool | None = None) -> RomSet:
    """The release set from the stock `rs` and the US Dreamcast `disc` (the module docstring's chain): ids 3-7
    converted (RELEASE_CONVERTED), the rankings, the select with the release row (select.ENTRIES), the converted
    courses' own time tables (build.race_times on RELEASE_TIME_IDS), the start time raised on `long_ids` (none by
    default; e.g. RELEASE_LONG_IDS), the converted courses' own pit stalls (build.pit_stalls on RELEASE_PIT_IDS), the
    Extended title (title.extended_title), then the views. Test-only: `pits=False` leaves the pit stalls out (with
    `times=False` and RELEASE_LONG_IDS, v0.1.1's chain); `direction=False` leaves out the reverse courses, their
    views and the Course Direction screen (v0.1.2's chain; the gates of the courses race it: the screen moves the race
    52 frames and changes nothing else, tests/test_dirscreen_mame.py test_race_after_screen_as_stock);
    `far_view=False` (default: as `direction`) leaves out v0.3's far view and depth test: the converted courses
    thinned as v0.2's and without model.DEPTH_TEST,
    build.viewed_set's views, the polygon cap 5,000 (v0.2's chain, RELEASE_SHA1_V02); the far view needs `direction`;
    `times=False` leaves the race times out (the converted courses keep Beginner's words, as v0.1-v0.1.1: the release
    gates' time-over frames are measured on those), `title=False` the title step (the release gates compare the
    titled set's ground log with the untitled one's, tests/test_courses_mame.py `_gate`). ReleaseError for `rs` not a
    RomSet, `disc` not a DcDisc, a long id that is not one of 3-7 (bool refused), or `title`, `times` or `pits` not a
    bool (all keyword-only), before any step runs."""
    if not isinstance(rs, RomSet):
        raise ReleaseError(f"{type(rs).__name__} is not a RomSet")
    if not isinstance(disc, DcDisc):
        raise ReleaseError(f"{type(disc).__name__} is not a DcDisc")
    long_ids = _long_ids(long_ids)
    far_view = direction if far_view is None else far_view
    for name, v in (("title", title), ("times", times), ("pits", pits), ("direction", direction),
                    ("far_view", far_view)):
        if not isinstance(v, bool):
            raise ReleaseError(f"{name} {v!r:.60} is not a bool")
    if far_view and not direction:
        raise ReleaseError("far_view needs direction (its views cover the reverse ids)")
    out = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    for n in RELEASE_CONVERTED:
        out = build.converted_set(out, disc, n, slot=n, scenery="model", **({"road_copy": True} if direction else {}),
                                  **({"thin": False, "depth": True} if far_view else {}))
    for n in range(REVERSE_IDS) if direction else ():
        out = build.converted_set(out, disc, n, slot=n + REVERSE_IDS, variant="R",
                                  scenery="keep" if n < ARCADE_COURSES else "model", road_copy=True,
                                  **({"thin": False, "depth": True} if far_view else {}))
    out = select.patch_select(C.grow_rankings(out), select.ENTRIES, faceted_maps=direction)
    for k in (RELEASE_TIME_IDS + (REVERSE_TIME_IDS if direction else ())) if times else ():
        out = build.race_times(out, golden.decode_course(out, k))
    for k in long_ids:
        out = build.long_time(out, golden.decode_course(out, k))
    for k in RELEASE_PIT_IDS if pits else ():
        out = build.pit_stalls(out, disc, k)
    if pits and direction:                      # M9 stage 4: the reverses' own stalls, the pit stop's angles
        for n in range(REVERSE_IDS):
            out = build.pit_stalls(out, disc, n, "R")
        out = C.patch_pit_heading(out)
    out = _title.extended_title(out) if title else out
    if not direction:
        return build.viewed_set(out)
    from dtools.arcade import dirscreen
    return dirscreen.patch_direction(build.far_viewed_set(out) if far_view
                                     else build.viewed_set(out, build.REVERSE_VIEW))


# ---- packaging (Task 5): the inputs checked, the set written ----

ROM_ZIP = "daytona.zip"
IO_ZIP = "model1io.zip"
"""The two zips MAME loads for `daytona` from the ROM directory: the set (read and checked) and the I/O board's (MAME
needs it to run the set; the build only copies it)."""
WORK = "work"
"""Under --out: the build's work files (the .7z's extraction `dc/`, the re-mesh cache `cache/remesh`); the CLI points
DAYTONA_OUT there for its process."""
REPORT = "build-report.txt"

ROM_SIZES: dict[str, int] = {
    **dict.fromkeys(("epr-16722a.12", "epr-16723a.13"), 0x20000),
    **dict.fromkeys(("mpr-16528.10", "mpr-16529.11", "mpr-16808.8", "mpr-16809.9"), 0x200000),
    **dict.fromkeys(("epr-16724a.6", "epr-16725a.7"), 0x80000),
    **dict.fromkeys(("mpr-16537.ic28", "mpr-16536.ic29"), 0x200000),
    **dict.fromkeys(("mpr-16523.ic16", "mpr-16518.ic20", "mpr-16524.ic17", "mpr-16519.ic21", "mpr-16525.ic18",
                     "mpr-16520.ic22", "mpr-16772.ic19", "mpr-16771.ic23"), 0x200000),
    **dict.fromkeys(("mpr-16522.25", "mpr-16521.24", "mpr-16770.27", "mpr-16769.26"), 0x200000),
    **dict.fromkeys(("epr-16726.bin", "epr-16720.7", "epr-16721.8"), 0x20000),
    **dict.fromkeys(("mpr-16491.32", "mpr-16492.33", "mpr-16493.4", "mpr-16494.5"), 0x200000),
    "epr-16488a.ic12": 0x10000,
    **dict.fromkeys(("opr-14742a.45", "opr-14743a.46", "opr-14744.58", "opr-14745.59", "opr-14746.62",
                     "opr-14747.63"), 0x20000),
}
"""Size in bytes of each of the 36 files of MAME 0.289's `daytona` (model2.cpp, ROM_START( daytona ) and
MODEL2_CPU_BOARD; the names of romset.CRCS): checked with the CRC32s before the set is read."""
DISC_TRACK3 = (1185760800, 0xAA3F9CDE)
"""(size in bytes, CRC32) of the file the cue names as track 3 on the Redump dump of the US Dreamcast *Daytona USA*
(MK-51037): the high-density data track the converter reads (measured on the dump here while planning)."""
DISC_SHAPES = "the Redump .7z of the US disc, or its .cue with the track .bin files beside it"
TESTED_PYTHON = "3.15.0"
TESTED_MAME = "MAME 0.289 with tools/mame-daytona-rom-space.patch (the custom build of tools/build-mame.sh)"
CUSTOM_NOTE = ("stock MAME 0.289 will not run this set: its ROM regions are grown past the stock ones. Build the custom "
               "MAME once with tools/build-mame.sh (MAME 0.289 plus tools/mame-daytona-rom-space.patch) and run the "
               "command from the repository root. Keep -nvram_directory: MAME's default nvram/daytona belongs to the "
               "stock game, whose best times this set's first boot would reset.")
WINDOWS_MAME_ZIP = "daytona-usa-extended-mame-0.289-win64.zip"
"""The prebuilt patched MAME for Windows (mame.exe and what it needs), the zip the Windows texts name."""
_WINDOWS_NOTE = ("stock MAME 0.289 will not run this set: its ROM regions are grown past the stock ones. Unzip "
                 f"{WINDOWS_MAME_ZIP} (the custom MAME 0.289 with tools/mame-daytona-rom-space.patch, prebuilt) and run "
                 "the command from the folder you unzipped it into. Keep -nvram_directory: MAME's default nvram/daytona "
                 "belongs to the stock game, whose best times this set's first boot would reset.")
_CHUNK = 16 << 20
_UNREADABLE = (DecodeError, OSError, IndexError, struct.error, RecursionError)
"""What reading a damaged disc image raises (iso.py parses the bytes as they are: IsoError for an extent past the end,
IndexError and struct.error for a cut directory record, RecursionError for a directory that holds itself): caught only
around the disc's opening and DcDisc.verify, where these come from the input, not from a defect."""


@dataclass
class Report:
    """What `check_inputs` found: `ok` when nothing stops a build; `lines` the problems ("problem: ..."), then what was
    verified and the warnings ("warning: ...")."""
    ok: bool
    lines: list[str]


def _path(v, what: str) -> Path:
    if not isinstance(v, (str, os.PathLike)):
        raise ReleaseError(f"{what} {v!r:.60} is not a path")
    return Path(v)


def _check_roms(roms: Path) -> tuple[list[str], list[str]]:
    """(problems, notes) of the ROM directory: daytona.zip of MAME 0.289's 36 files by name, size and CRC32 (the zip's
    directory), then read whole (RomSet.from_zip: a damaged member fails its CRC there); model1io.zip a warning."""
    if not roms.is_dir():
        return [f"ROM directory {roms} does not exist (it must hold {ROM_ZIP} and {IO_ZIP} of MAME 0.289)"], []
    problems: list[str] = []
    notes: list[str] = []
    z = roms / ROM_ZIP
    if not z.is_file():
        problems.append(f"{roms} holds no {ROM_ZIP} (MAME 0.289's `daytona` set)")
    else:
        try:
            with zipfile.ZipFile(z) as f:
                infos = {i.filename: i for i in f.infolist() if "/" not in i.filename}   # merged sets: clones below
        except zipfile.BadZipFile:
            problems.append(f"{z} is not a zip file (MAME 0.289's `daytona` set is a zip of 36 ROM files)")
        except OSError as e:
            problems.append(f"{z} cannot be read: {e}")
        else:
            bad = []
            for n, crc in CRCS.items():
                i = infos.get(n)
                if i is None:
                    bad.append(f"{z}: {n} is missing")
                elif i.file_size != ROM_SIZES[n]:
                    bad.append(f"{z}: {n} is {i.file_size} bytes, expected {ROM_SIZES[n]} (not MAME 0.289's file)")
                elif i.CRC != crc:
                    bad.append(f"{z}: {n} has CRC32 {i.CRC:08x}, expected {crc:08x} (not MAME 0.289's file)")
            built = sorted(n for n in CUSTOM_FILES if n in infos)
            if built:
                bad.append(f"{z} holds {', '.join(built)}: a set this tool built, not MAME's stock `daytona` "
                           f"(give the set MAME 0.289 verifies)")
            if not bad:
                try:
                    RomSet.from_zip(z)
                except (zipfile.BadZipFile, DecodeError, OSError, RuntimeError, NotImplementedError, EOFError,
                        zlib.error) as e:
                    bad.append(f"{z}: {' '.join(str(e).split())}")
                else:
                    notes.append(f"{z}: the {len(CRCS)} files of MAME 0.289's `daytona` verified (sizes and CRC32s)")
            problems += bad
    io = roms / IO_ZIP
    if io.is_file():
        notes.append(f"{io}: present (copied beside the built set)")
    else:
        notes.append(f"warning: {io} is missing: the build does not need it, MAME needs it beside {ROM_ZIP} to run "
                     f"the set")
    return problems, notes


def _why(e: BaseException, named: bool = False) -> str:
    """An exception as one line: its message with the newlines folded; the type named first when `named`, or when
    it is not one of this tool's own (a DecodeError) or has no message."""
    text = " ".join(str(e).split())
    if text and isinstance(e, DecodeError) and not named:
        return text
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def _crc32_file(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        while block := f.read(_CHUNK):
            crc = zlib.crc32(block, crc)
    return crc


def _check_disc(disc: Path, work: Path | None) -> tuple[list[str], list[str]]:
    """(problems, notes) of the disc: a .7z (its cue and track 3 extracted under `work`, once) or a .cue; the file the
    cue names as track 3 beside it, DISC_TRACK3 in size and CRC32; then, only when both match, DcDisc.verify (the
    product, every course file). Any other shape is refused naming the two accepted ones. Every problem is one line:
    a disc image that cannot be read (_UNREADABLE) is named, never raised. `work` None: the output directory is
    unusable, a .7z is not extracted."""
    if not disc.is_file():
        return [f"disc {disc} does not exist (give {DISC_SHAPES})"], []
    suffix = disc.suffix.lower()
    if suffix not in (".7z", ".cue"):
        return [f"{disc}: a {suffix or 'file without an extension'} is not read: give {DISC_SHAPES}"], []
    if suffix == ".7z" and work is None:
        return [f"{disc}: not checked: its track 3 is extracted under --out, which cannot be used (see above)"], []
    problems: list[str] = []
    notes: list[str] = []
    try:
        if suffix == ".7z":
            with DcDisc.open(disc, out=work) as d:
                cue = d.cue
            notes.append(f"{disc}: the cue and track 3 extracted to {cue.parent}")
        else:
            cue = disc
        track = track3_path(cue)
    except (DiscError, OSError) as e:
        return [f"{disc}: {_why(e)}"], notes
    except _UNREADABLE as e:            # the .7z's track 3 is opened as it is extracted: cut short or damaged
        return [f"{disc}: track 3 cannot be read ({_why(e, named=True)}): a damaged or cut-short dump"], notes
    if not track.is_file():
        return [f"{cue.name} names {track.name} as track 3, which is not beside it in {cue.parent} (give "
                f"{DISC_SHAPES})"], notes
    size, crc = DISC_TRACK3
    try:
        have = track.stat().st_size
        if have != size:
            problems.append(f"{track.name}: {have} bytes, expected {size} (track 3 of the US disc's Redump dump): cut "
                            f"short, or another release of the disc")
        elif (got := _crc32_file(track)) != crc:
            problems.append(f"{track.name}: CRC32 {got:08x}, expected {crc:08x} (track 3 of the US disc's Redump "
                            f"dump): damaged, or another release of the disc")
        else:
            notes.append(f"{track.name}: track 3 verified ({size} bytes, CRC32 {crc:08x})")
    except OSError as e:
        return [f"{track}: cannot be read: {_why(e)}"], notes
    if problems:                        # track 3 is not the dump's: its file system is not read (it may be cut short)
        return problems, notes
    try:
        with DcDisc.open(cue) as d:
            missing = d.verify()
    except DiscError as e:
        problems.append(f"{cue}: {_why(e)}")
    except _UNREADABLE as e:
        problems.append(f"{cue}: track 3's file system cannot be read ({_why(e, named=True)}): a damaged dump")
    else:
        if missing:
            more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
            problems.append(f"{cue}: not the US Daytona USA disc ({PRODUCT}): {'; '.join(missing[:6])}{more}")
        else:
            notes.append(f"{cue}: product {PRODUCT}, every course file present")
    return problems, notes


def _check_out(out: Path, roms: Path) -> tuple[list[str], list[str]]:
    """(problems, notes) of the output directory: not the ROM directory under any spelling (resolved: a trailing
    slash, relative, `..`, a symlink; or the same file), not a file, created and written (a probe file)."""
    try:
        same = out.resolve() == roms.resolve() or (out.exists() and roms.exists() and os.path.samefile(out, roms))
        same = same or ((out / ROM_ZIP).exists() and (roms / ROM_ZIP).exists()
                        and os.path.samefile(out / ROM_ZIP, roms / ROM_ZIP))
    except OSError as e:
        return [f"output directory {out} cannot be examined: {e}"], []
    if same:
        return [f"output directory {out} is the ROM directory {roms}: the input set is never overwritten (choose "
                f"another --out)"], []
    if out.exists() and not out.is_dir():
        return [f"output directory {out} is a file, not a directory"], []
    if platform.system() == "Windows" and _windows_out_problem(str(out)):
        return [_windows_out_problem(str(out))], []
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".dtools-write-test"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as e:
        return [f"output directory {out} cannot be written: {e}"], []
    return [], [f"output directory {out.resolve()}"]


def _first_missing(path: Path) -> Path | None:
    """The outermost directory of `path` (made absolute) that does not exist yet: what `mkdir(parents=True)` would
    create first. None when `path` exists."""
    p, top = path.absolute(), None
    while not os.path.lexists(p):
        p, top = p.parent, p
    return top


def check_inputs(roms: Path, disc: Path, out: Path) -> Report:
    """What a build needs: the ROM directory with MAME 0.289's `daytona.zip` (each of its 36 files by size and CRC32;
    `model1io.zip` beside it, a warning when absent), the US Dreamcast disc (DISC_SHAPES; track 3 DISC_TRACK3 in size
    and CRC32, then DcDisc.verify) and an output directory that is not the ROM directory and can be written. The check
    creates `out` (a .7z is extracted under <out>/WORK, for the build to reuse); a check that fails, or raises, removes
    the directories it created (an `out` that was there before stays). The inputs are never written. ReleaseError for
    an argument that is not a path."""
    roms, disc, out = _path(roms, "roms"), _path(disc, "disc"), _path(out, "out")
    created = _first_missing(out)
    report = None
    try:
        rp, rn = _check_roms(roms)
        op, on = _check_out(out, roms)
        dp, dn = _check_disc(disc, None if op else out / WORK)
        problems = rp + dp + op
        report = Report(not problems, [f"problem: {p}" for p in problems] + rn + dn + on)
    finally:
        if created is not None and (report is None or not report.ok) and created.is_dir() \
                and not created.is_symlink():
            shutil.rmtree(created, ignore_errors=True)
    return report


def member_sha1(rs: RomSet) -> dict[str, str]:
    """sha1 (hex) of each file `rs.split_files()` writes into daytona.zip (the 36 stock names, a custom set's five
    dx-*.bin), sorted by name: the byte-for-byte measure of a build (the zip's own bytes vary with its timestamps).
    ReleaseError for `rs` not a RomSet."""
    if not isinstance(rs, RomSet):
        raise ReleaseError(f"{type(rs).__name__} is not a RomSet")
    return {n: hashlib.sha1(b).hexdigest() for n, b in sorted(rs.split_files().items())}


def region_sha1(rs: RomSet) -> dict[str, str]:
    """sha1 (hex) of each of the set's five images (romset.REGIONS), the measure RELEASE_SHA1 pins."""
    if not isinstance(rs, RomSet):
        raise ReleaseError(f"{type(rs).__name__} is not a RomSet")
    return {r: hashlib.sha1(getattr(rs, r)).hexdigest() for r in REGIONS}


def _shell_path(p: Path) -> str:
    """`p` for a POSIX shell: in double quotes (the README's form) when nothing in it is special there, else
    shlex.quote's single quotes (a `"`, `$`, backtick, backslash, `!` or newline would break the double quotes)."""
    text = str(p)
    return shlex.quote(text) if any(c in text for c in '"$`\\!\n') else f'"{text}"'


def _windows_out_problem(text: str) -> str:
    """Why `text` cannot be the Windows output folder in the run command ("" when it can): a drive root or a path
    ending in a backslash, whose last `\\` would escape the closing double quote in cmd."""
    win = PureWindowsPath(text)
    is_root = bool(win.root) and win.parent == win or (bool(win.drive) and not win.root and len(win.parts) == 1)
    if text.endswith("\\") or is_root:
        return (f"the output folder {text} is a drive root or ends with a backslash, which would escape the closing "
                "quote in the run command: use a named folder without a trailing backslash")
    return ""


def mame_command(out: Path, system: str | None = None) -> str:
    """The command that runs the set in `out`: off Windows the custom MAME (build/mame/model2, from the repository
    root) with the set's own NVRAM directory, the paths quoted for a POSIX shell (_shell_path); on Windows (`system`
    "Windows", default platform.system()) `mame.exe` with Windows path text in double quotes. ReleaseError for a
    Windows path holding `"`, `%`, `$` or a backtick (cmd and PowerShell would expand them: rename the folder)."""
    out = Path(out)
    if (platform.system() if system is None else system) != "Windows":
        return f"build/mame/model2 daytona -rompath {_shell_path(out)} -nvram_directory {_shell_path(out / 'nvram')}"
    text = str(out)
    for c in '"%$`':
        if c in text:
            raise ReleaseError(f"the output folder {text} holds {c!r}, which cmd or PowerShell would expand in the "
                               "run command: rename the folder")
    win = PureWindowsPath(text)
    if _windows_out_problem(text):
        raise ReleaseError(_windows_out_problem(text))
    return f'.\\mame.exe daytona -rompath "{win}" -nvram_directory "{win / "nvram"}"'


def custom_note(system: str | None = None) -> str:
    """The custom-build note: CUSTOM_NOTE off Windows, the prebuilt zip's note on Windows."""
    return _WINDOWS_NOTE if (platform.system() if system is None else system) == "Windows" else CUSTOM_NOTE


def run_from(system: str | None = None) -> str:
    """The line before the run command: where to run it from."""
    if (platform.system() if system is None else system) == "Windows":
        return f"run it from the folder you unzipped {WINDOWS_MAME_ZIP} into:"
    return "run it from the repository root, with the custom MAME:"


def _commit() -> str:
    """The repository's commit (with "+ uncommitted changes" when the tracked files differ), or "unknown" outside a
    git checkout (an exported copy)."""
    root = config.REPO_ROOT
    if not (root / ".git").exists():
        return "unknown"
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
    except OSError:
        return "unknown"
    if head.returncode != 0:
        return "unknown"
    return head.stdout.strip() + (" + uncommitted changes" if dirty.returncode == 0 and dirty.stdout.strip() else "")


def _stage(out: Path, name: str, staged: dict[str, Path], write) -> None:
    """`write(tmp)` into a new temporary file in `out` (recorded in `staged` before it is written, so the caller
    removes it whatever happens), created as a plain open creates a file: mode 0666 less the umask (mkstemp's would
    be 0600)."""
    while True:
        tmp = out / f".{name}.{secrets.token_hex(6)}.partial"
        try:
            os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
            break
        except FileExistsError:
            continue
    staged[name] = tmp
    write(tmp)


def write_release(rs: RomSet, out: Path, rom_dir: Path, *, checks: Sequence[str] = ()) -> Path:
    """Write the built set `rs` into `out`: `daytona.zip`, `model1io.zip` copied from `rom_dir` when there, and
    `build-report.txt`, each written whole under a temporary name first and only then renamed into place (the earlier
    report removed just before: an earlier build's files are replaced only by complete ones, and the report never
    sits beside a zip it does not describe), each of mode 0666 less the umask. The report holds the commit or
    "unknown", the Python and MAME versions tested, the input checks `checks`, `member_sha1` sorted, the region sha1s,
    the command `mame_command` and the custom-build note CUSTOM_NOTE. The path of daytona.zip, absolute.
    ReleaseError for `rs` not a RomSet or a file that cannot be written; split_files' RomSetError passes through."""
    if not isinstance(rs, RomSet):
        raise ReleaseError(f"{type(rs).__name__} is not a RomSet")
    out, rom_dir = _path(out, "out").resolve(), _path(rom_dir, "rom_dir")
    members = member_sha1(rs)
    io = rom_dir / IO_ZIP
    staged: dict[str, Path] = {}
    try:
        out.mkdir(parents=True, exist_ok=True)
        _stage(out, ROM_ZIP, staged, rs.write_zip)
        if io.is_file():
            _stage(out, IO_ZIP, staged, lambda tmp: shutil.copyfile(io, tmp))
        lines = ["Daytona USA Extended: build report", "",
                 f"commit: {_commit()}",
                 f"python: {platform.python_version()} (tested with Python {TESTED_PYTHON})",
                 f"mame: tested with {TESTED_MAME}", "",
                 "inputs checked:", *(f"  {c}" for c in checks), "",
                 f"written: {out / ROM_ZIP} ({len(members)} files)",
                 f"  {IO_ZIP}: " + ("copied" if io.is_file() else f"not copied, absent from {rom_dir}: MAME needs it "
                                                                   f"beside {ROM_ZIP}"), "",
                 "member sha1 (each file daytona.zip holds; the zip's own bytes vary with its timestamps):",
                 *(f"  {n} {h}" for n, h in members.items()), "",
                 "region sha1:", *(f"  {r} {h}" for r, h in region_sha1(rs).items()), "",
                 run_from(), f"  {mame_command(out)}", "", f"note: {custom_note()}"]
        _stage(out, REPORT, staged, lambda tmp: tmp.write_text("\n".join(lines) + "\n", encoding="utf-8"))
        # all three complete: the earlier report goes first, so it never sits beside a zip it does not describe
        (out / REPORT).unlink(missing_ok=True)
        for name in (ROM_ZIP, IO_ZIP, REPORT):
            if name in staged:
                os.replace(staged[name], out / name)
                del staged[name]
    except OSError as e:
        raise ReleaseError(f"cannot write the set into {out}: {e}") from None
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
    return out / ROM_ZIP
