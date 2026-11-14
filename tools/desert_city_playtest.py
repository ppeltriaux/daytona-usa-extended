"""Write the Desert City (Dreamcast n 3) playtest set: `out/play/desert-city-m3b/daytona.zip`, with `model1io.zip`
copied beside it from DAYTONA_ROMDIR.

The set is `build.converted_set(rs, disc, 3, scenery="model")` (the course's converted scenery; `--scenery empty`
gives Phase 6's M3a set, into desert-city-m3a) with the first word (the start time) of every time table set to
`build.LONG_TIME` (12000 ticks), as the long-time MAME runs, so a human can drive a lap: on the converted set the
start time is Beginner's 3200 ticks (50 s to the first checkpoint). M6 sets the real times. Writes game data under
`out/` only (git-ignored); nothing here is kept in the repository.

`--sixteen` (Phase 8, M4a) writes `out/play/sixteen-m4a/` instead: the grown set (course.grow_ids) with the new
ids' guards (course.guard_new_ids), Desert City converted into its own course id 3 (`converted_set(..., slot=3,
scenery="model")`), then the sixteen-entry select (select.patch_select, Desert City live; since Phase 9's final review
the conversion comes first, as patch_select refuses a live id 3 without its course: the same bytes as Phase 8's
order, which converted after the select), Beginner, Expert and Advanced as they are, the same long start time on id
3's own time tables, and a README.txt with the command. It is a custom
set: it needs the custom MAME build (tools/build-mame.sh; `dtools.config.custom_mame()`). Since Phase 9 Task 5 it
carries the new select (idle wheel, generated panels; `select.patch_select` with `select.M4A_ENTRIES`).

`--mirrors` (Phase 9, M4b) writes `out/play/mirrors-m4b/`: Phase 9's final chain (`mirror_set`), image for image
Phase 9's final set (tests/test_mirrors_mame.py `m4b_set`; its `final_set` is Phase 10's since Task 3, with National
Park Speedway at id 4): grow_ids, guard_new_ids, widen_ids, Desert City
converted at id 3 (`converted_set(..., 3, slot=3, scenery="model")`), the mirrors of Beginner, Expert, Advanced and
Desert City at ids 8-11 (`mirror.mirrored_entry`, in that order: each takes the first free bytes), a ranking table
per course id (`course.grow_rankings`), the select with Phase 9's row `select.M4B_ENTRIES` (8-11 live, National Park
Speedway's entry 4 a placeholder, as it was: the set holds no id 4; Phase 10 Task 3 froze the row so the set keeps its
bytes); then (`long_times`) the start
time of Desert City's own time tables (id 3) and its mirror's (id 11, the same tables: a mirror shares its original's)
raised to `build.LONG_TIME` (12,000 ticks), as on the M4a set, so a lap can be driven (M6 sets the real times). The
originals and their mirrors keep the stock times. So the set is `m4b_set` image for image except the start words
`long_time` writes (tests/test_playtest.py). The set's new
SRAM signature resets backup SRAM on its first boot, so it must run with its own NVRAM directory (`MIRRORS_COMMAND`),
never MAME's shared `nvram/` (the stock game's records would be wiped, and the stock game would reset these).

`--m5a` (Phase 10, M5a) writes `out/play/m5a/`: Phase 10's final chain (`m5a_set`), image for image the tests'
`final_set` (tests/test_mirrors_mame.py) but for the start words: grow_ids, guard_new_ids, widen_ids, Desert City,
National Park Speedway and Mermaid Lake converted at their own ids 3, 4 and 7 (`converted_set(..., n, slot=n,
scenery="model")`, in that order; id 4's re-mesh is read from the on-disk cache `out/cache/remesh` after one cold run
of about 250 s), the mirrors of 0-4 and 7 at ids 8-12 and 15 (`mirror.mirrored_entry`, in that order), a ranking table
per course id, the select with Phase 10's frozen row `select.M5A_ENTRIES` (0-4, 7 and 8-11 live; 5, 6
and 12-15 locked, so mirrors 12 and 15 are on the set but no entry reaches them: mirror mode leaves the product,
Pascal 2026-11-11); then `long_times(..., M5A_LONG_TIME_IDS)`: `build.long_time` on the converted courses and their
mirrors (3, 4, 7, 11, 12, 15; a mirror shares its original's tables). The originals and their mirrors keep the stock
times. Its own NVRAM directory as the M4b set (`M5A_COMMAND`). The row is frozen (Phase 11 Task 1, as Phase 10
froze M4B_ENTRIES): Phase 11's new `select.ENTRIES` moves no byte of this set.

`--m5b` (Phase 11, M5b) writes `out/play/m5b/`: the shipped chain (`m5b_set`), image for image the tests' `m5b_set`
(tests/test_courses_mame.py) but for the start words: grow_ids, guard_new_ids, widen_ids, Desert City, National Park
Speedway, Mermaid Lake, Rin Rin Rink and Circuit Pixie converted at their own ids 3, 4, 7, 6 and 5 (in that order; the
re-meshes of 4, 5 and 6 are read from `out/cache/remesh` after one cold run), a ranking table per course id, the select
with the row of eight (`select.M5B_ENTRIES`, all eight live; no mirror is built), then `long_times(..., M5B_LONG_TIME_IDS)`
(`build.long_time` on ids 3-7), then `build.viewed_set` last (Circuit Pixie's and Rin Rin Rink's views: its tables are
not in golden.owned_spans, so nothing may allocate after it; `long_time` writes only time-table words, so the views
land on the same bytes: tests/test_playtest.py). Beginner, Advanced and Expert keep the stock times. Its own NVRAM
directory, `out/play/m5b/nvram` (`M5B_COMMAND`).

Usage (from the repository root):
    DAYTONA_ROMDIR=$PWD/ROMS DAYTONA_DC="$PWD/ROMS/Daytona USA (USA)-Dreamcast.7z" \\
        .venv/bin/python tools/desert_city_playtest.py [out_dir] [--scenery empty|model] [--sixteen | --mirrors |
        --m5a | --m5b]
Then: mame daytona -rompath "$PWD/out/play/desert-city-m3b"
  or (--sixteen): build/mame/model2 daytona -rompath "$PWD/out/play/sixteen-m4a"
  or (--mirrors): build/mame/model2 daytona -rompath "$PWD/out/play/mirrors-m4b" \\
                      -nvram_directory "$PWD/out/play/mirrors-m4b/nvram"
  or (--m5a): build/mame/model2 daytona -rompath "$PWD/out/play/m5a" -nvram_directory "$PWD/out/play/m5a/nvram"
  or (--m5b): build/mame/model2 daytona -rompath "$PWD/out/play/m5b" -nvram_directory "$PWD/out/play/m5b/nvram"
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dtools import config  # noqa: E402
from dtools.arcade import course as C  # noqa: E402
from dtools.arcade import golden as G  # noqa: E402
from dtools.arcade import mirror as M  # noqa: E402
from dtools.arcade import select as S  # noqa: E402
from dtools.arcade.romset import RomSet  # noqa: E402
from dtools.convert import build  # noqa: E402
from dtools.dc import pol  # noqa: E402
from dtools.dc.data import decode_data  # noqa: E402
from dtools.dc.disc import DcDisc, course_files  # noqa: E402
from dtools.dc.exe import EXE_NAME  # noqa: E402


def main(out_dir: Path, scenery: str = "model") -> Path:
    rom_dir, dc = config.rom_dir(), config.dc_path()
    if rom_dir is None or dc is None:
        sys.exit("set DAYTONA_ROMDIR and DAYTONA_DC")
    rs = RomSet.from_zip(rom_dir / "daytona.zip")
    disc = DcDisc.open(dc)
    try:
        name = course_files(3, kind="data")
        # converted_set and converted_course take the same scenery data; convert the model once
        data = scenery_data = None
        if scenery == "model":
            data = decode_data(disc.read(name), name)
            mn = course_files(3, kind="model")
            scenery_data = build.model_scenery(rs, data, pol.decode_pol(disc.read(mn), mn),
                                               disc.read(course_files(3, kind="textures")), build.THIN.get(3))
        c = build.converted_course(rs, data or decode_data(disc.read(name), name), disc.read(EXE_NAME), 3,
                                   scenery=scenery, scenery_data=scenery_data)
        s = build.long_time(build.converted_set(rs, disc, 3, scenery=scenery, scenery_data=scenery_data), c)
    finally:
        disc.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = s.write_zip(out_dir / "daytona.zip")
    shutil.copy(rom_dir / "model1io.zip", out_dir / "model1io.zip")
    return path


README = """Daytona USA (Model 2) with sixteen course entries (Phase 8, M4a)

Circuit Select scrolls through sixteen entries with the wheel: Beginner, Advanced, Expert as on the
arcade, then DESERT CITY (the Dreamcast's course, its own course id 3), then placeholders (greyed:
they cannot be chosen). Desert City starts standing, like Expert and Advanced; its start time is
{ticks} ticks, so a lap can be driven (M6 sets the real times). Its race rankings and best laps
are not kept (M4b); Beginner's, Expert's and Advanced's are.

This is a custom set (dx-*.bin): it needs the custom MAME build. From the repository root:

    build/mame/model2 daytona -rompath "{path}"

MAME shows a ROM warning once per launch (the dx-*.bin files are NO_DUMP): press a key.
"""


def sixteen(out_dir: Path) -> Path:
    """The M4a playtest set (module docstring, `--sixteen`) and its README.txt in `out_dir`."""
    rom_dir, dc = config.rom_dir(), config.dc_path()
    if rom_dir is None or dc is None:
        sys.exit("set DAYTONA_ROMDIR and DAYTONA_DC")
    base = C.guard_new_ids(C.grow_ids(RomSet.from_zip(rom_dir / "daytona.zip")))
    disc = DcDisc.open(dc)
    try:
        name = course_files(3, kind="data")
        data = decode_data(disc.read(name), name)
        mn = course_files(3, kind="model")
        sd = build.model_scenery(base, data, pol.decode_pol(disc.read(mn), mn),
                                 disc.read(course_files(3, kind="textures")), build.THIN.get(3), slot=3)
        c = build.converted_course(base, data, disc.read(EXE_NAME), 3, slot=3, scenery="model", scenery_data=sd)
        s = build.converted_set(base, disc, 3, slot=3, scenery="model", scenery_data=sd)
        s = build.long_time(S.patch_select(s, S.M4A_ENTRIES), c)
    finally:
        disc.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = s.write_zip(out_dir / "daytona.zip")
    shutil.copy(rom_dir / "model1io.zip", out_dir / "model1io.zip")
    (out_dir / "README.txt").write_text(README.format(ticks=build.LONG_TIME, path=out_dir.resolve()))
    return path


MIRRORS_COMMAND = 'build/mame/model2 daytona -rompath "{path}" -nvram_directory "{path}/nvram"'
"""The M4b set's command, from the repository root (so MAME reads cfg/daytona.cfg, the controls mapped there). Its
own NVRAM directory (MAME 0.289's `-nvram_directory`; the default is `nvram/` in the working directory, the stock
game's): the set's signature `course.SIGNATURE_RANKED` resets an older backup SRAM on boot."""

MIRRORS_README = """Daytona USA (Model 2) with the mirrors live (Phase 9, M4b)

Circuit Select scrolls through sixteen entries with the wheel: Beginner, Advanced, Expert, DESERT CITY,
four placeholders (greyed: they cannot be chosen), then the mirrors: BEGINNER, ADVANCED, EXPERT and
DESERT CITY mirrored (live, MIRROR on their panels), and four greyed mirror placeholders. An untouched
wheel races Beginner when the select timer ends. Every course keeps its own ranking table.

Desert City and its mirror start with {ticks} ticks, so a lap can be driven (M6 sets the real
times); Beginner, Advanced, Expert and their mirrors keep the arcade's times.

This is a custom set (dx-*.bin): it needs the custom MAME build. Run it from the repository root (your
controls in cfg/daytona.cfg are read from there) with its own NVRAM directory:

    {command}

Keep the -nvram_directory: the set resets backup SRAM on its first boot (new ranking layout), and MAME's
shared nvram/daytona belongs to the stock game. Without it the stock game's records are wiped, and the
stock game then resets this set's.

MAME shows a ROM warning once per launch (the dx-*.bin files are NO_DUMP): press a key.
"""


M4B_MIRRORED = (0, 1, 2, 3)
"""The originals Phase 9's set mirrors (ids 8-11), frozen: mirror.MIRROR_ORIGINALS also holds 4 and 7 since Phase 10
Task 5, which this set does not convert."""


def mirror_set(rs: RomSet, disc: DcDisc) -> RomSet:
    """The M4b set from the stock `rs` (module docstring, `--mirrors`): the chain in its order, the order the MAME
    tests' `m4b_set` uses (each step places at the first free bytes, so the order fixes the bytes), with Phase 9's row
    (select.M4B_ENTRIES)."""
    out = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    out = build.converted_set(out, disc, 3, slot=3, scenery="model")
    for o in M4B_MIRRORED:
        out = M.mirrored_entry(out, o)
    return S.patch_select(C.grow_rankings(out), S.M4B_ENTRIES)


LONG_TIME_IDS = (3, 11)
"""The course ids whose time tables the playtest set's start time is raised on: Desert City and its mirror (which
shares Desert City's tables). The originals and their mirrors keep the stock times."""


def long_times(s: RomSet, ids: tuple[int, ...] = LONG_TIME_IDS) -> RomSet:
    """`s` with `build.long_time` (`build.LONG_TIME`) on the time tables of each of `ids` (LONG_TIME_IDS, the M4b
    set's)."""
    for k in ids:
        s = build.long_time(s, G.decode_course(s, k))
    return s


def mirrors(out_dir: Path) -> Path:
    """The M4b playtest set (`long_times(mirror_set(...))`), `model1io.zip` and a README.txt with `MIRRORS_COMMAND`
    in `out_dir`."""
    rom_dir, dc = config.rom_dir(), config.dc_path()
    if rom_dir is None or dc is None:
        sys.exit("set DAYTONA_ROMDIR and DAYTONA_DC")
    disc = DcDisc.open(dc)
    try:
        s = long_times(mirror_set(RomSet.from_zip(rom_dir / "daytona.zip"), disc))
    finally:
        disc.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = s.write_zip(out_dir / "daytona.zip")
    shutil.copy(rom_dir / "model1io.zip", out_dir / "model1io.zip")
    command = MIRRORS_COMMAND.format(path=out_dir.resolve())
    (out_dir / "README.txt").write_text(MIRRORS_README.format(command=command, ticks=build.LONG_TIME))
    return path


M5A_CONVERTED = (3, 4, 7)
"""The courses the M5a set converts, in order, each at its own id: Desert City, National Park Speedway, Mermaid Lake
(tests/test_mirrors_mame.py CONVERTED)."""
M5A_MIRRORED = (0, 1, 2, 3, 4, 7)
"""The originals the M5a set mirrors, in order (ids o + 8: 8-12 and 15; tests/test_mirrors_mame.py MIRRORED)."""
M5A_LONG_TIME_IDS = (3, 4, 7, 11, 12, 15)
"""The M5a set's course ids whose start time is raised: the converted courses and their mirrors (each mirror shares
its original's time tables, so 11, 12 and 15 add no word)."""
M5A_COMMAND = MIRRORS_COMMAND
"""The M5a set's command (the M4b set's form, with the M5a set's path): from the repository root, its own NVRAM
directory."""

M5A_README = """Daytona USA (Model 2) with National Park Speedway and Mermaid Lake (Phase 10, M5a)

Circuit Select scrolls through sixteen entries with the wheel: Beginner, Advanced, Expert, DESERT CITY,
NATIONAL PARK SPEEDWAY, two greyed placeholders (Circuit Pixie and Rin Rin Rink, next phase), MERMAID LAKE,
then the mirrors of Beginner, Advanced, Expert and Desert City (live, MIRROR on their panels) and four
greyed mirror placeholders. Mirror mode leaves the product (Pascal, 2026-11-11): the next phase shrinks the
row to the eight courses. The mirrors of National Park Speedway and Mermaid Lake are built into the set but
no entry reaches them. An untouched wheel races Beginner when the select timer ends.

Desert City, National Park Speedway, Mermaid Lake and their mirrors start with {ticks} ticks, so a lap can
be driven (M6 sets the real times); Beginner, Advanced, Expert and their mirrors keep the arcade's times.

Worth watching for on Mermaid Lake: the whole course is raised 41 units so the arcade's ground answers the
right surface in the tunnel (segments 316-345, under segment 81). A car that hits an invisible wall, jumps
or sinks, or a camera that misbehaves on high ground, is the risk to report.

This is a custom set (dx-*.bin): it needs the custom MAME build. Run it from the repository root (your
controls in cfg/daytona.cfg are read from there) with its own NVRAM directory:

    {command}

Keep the -nvram_directory: the set resets backup SRAM on its first boot (new ranking layout), and MAME's
shared nvram/daytona belongs to the stock game.

MAME shows a ROM warning once per launch (the dx-*.bin files are NO_DUMP): press a key.
"""


def m5a_set(rs: RomSet, disc: DcDisc) -> RomSet:
    """The M5a set from the stock `rs` before its start words (module docstring, `--m5a`): the chain in the order
    tests/test_mirrors_mame.py `final_set` uses (each step places at the first free bytes, so the order fixes the
    bytes), with Phase 10's frozen row (select.M5A_ENTRIES)."""
    out = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    for n in M5A_CONVERTED:
        out = build.converted_set(out, disc, n, slot=n, scenery="model")
    for o in M5A_MIRRORED:
        out = M.mirrored_entry(out, o)
    return S.patch_select(C.grow_rankings(out), S.M5A_ENTRIES)


def m5a(out_dir: Path) -> Path:
    """The M5a playtest set (`long_times(m5a_set(...), M5A_LONG_TIME_IDS)`), `model1io.zip` and a README.txt with
    `M5A_COMMAND` in `out_dir`."""
    rom_dir, dc = config.rom_dir(), config.dc_path()
    if rom_dir is None or dc is None:
        sys.exit("set DAYTONA_ROMDIR and DAYTONA_DC")
    disc = DcDisc.open(dc)
    try:
        s = long_times(m5a_set(RomSet.from_zip(rom_dir / "daytona.zip"), disc), M5A_LONG_TIME_IDS)
    finally:
        disc.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = s.write_zip(out_dir / "daytona.zip")
    shutil.copy(rom_dir / "model1io.zip", out_dir / "model1io.zip")
    command = M5A_COMMAND.format(path=out_dir.resolve())
    (out_dir / "README.txt").write_text(M5A_README.format(command=command, ticks=build.LONG_TIME))
    return path


M5B_CONVERTED = (3, 4, 7, 6, 5)
"""The courses the M5b set converts, in order, each at its own id: Desert City, National Park Speedway, Mermaid Lake,
Rin Rin Rink, Circuit Pixie (tests/test_courses_mame.py M5B_CONVERTED)."""
M5B_LONG_TIME_IDS = (3, 4, 5, 6, 7)
"""The M5b set's course ids whose start time is raised: the five converted courses (no mirror is built)."""
M5B_COMMAND = MIRRORS_COMMAND
"""The M5b set's command (the M4b set's form, with the M5b set's path): from the repository root, so MAME reads the
controls mapped in cfg/daytona.cfg (a joystick's), with its own NVRAM directory."""

M5B_README = """Daytona USA (Model 2) with all eight courses (Phase 11, M5b)

Circuit Select scrolls through eight entries with the wheel: Beginner, Advanced, Expert, DESERT CITY,
NATIONAL PARK SPEEDWAY, CIRCUIT PIXIE, RIN RIN RINK and MERMAID LAKE, all live. No mirror is on the
row (mirror mode left the product; reverse comes in M9). An untouched wheel races Beginner when the
select timer ends.

The five Dreamcast courses start with {ticks} ticks, so a lap can be driven (M6 sets the real times);
Beginner, Advanced and Expert keep the arcade's times.

Worth watching for:
- Circuit Pixie and Rin Rin Rink draw a shorter view than the other courses (the row the car is in,
  the next row and the two far corners of the stock 5 x 5 block): far scenery pops in later there.
- Mermaid Lake is raised 41 units for its tunnel (segments 316-345): an invisible wall, a jump or a
  sink, or a camera fault on high ground is the risk to report.

This is a custom set (dx-*.bin): it needs the custom MAME build. Run it from the repository root (your
controls, a joystick's included, are read from cfg/daytona.cfg there) with its own NVRAM directory:

    {command}

Keep the -nvram_directory: the set resets backup SRAM on its first boot (new ranking layout), and MAME's
shared nvram/daytona belongs to the stock game.

MAME shows a ROM warning once per launch (the dx-*.bin files are NO_DUMP): press a key.
"""


def m5b_set(rs: RomSet, disc: DcDisc, long_ids: tuple[int, ...] = M5B_LONG_TIME_IDS) -> RomSet:
    """The M5b set from the stock `rs` (module docstring, `--m5b`): the chain in the order tests/test_courses_mame.py
    `m5b_set` uses (each step places at the first free bytes, so the order fixes the bytes), with the row of eight
    (select.M5B_ENTRIES), the start time raised on `long_ids` (M5B_LONG_TIME_IDS; `()` gives the tests' `m5b_set` itself)
    before `build.viewed_set`, the last step."""
    out = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    for n in M5B_CONVERTED:
        out = build.converted_set(out, disc, n, slot=n, scenery="model")
    out = S.patch_select(C.grow_rankings(out), S.M5B_ENTRIES)
    return build.viewed_set(long_times(out, long_ids))


def m5b(out_dir: Path) -> Path:
    """The M5b playtest set (`m5b_set(...)`), `model1io.zip` and a README.txt with `M5B_COMMAND` in `out_dir`."""
    rom_dir, dc = config.rom_dir(), config.dc_path()
    if rom_dir is None or dc is None:
        sys.exit("set DAYTONA_ROMDIR and DAYTONA_DC")
    disc = DcDisc.open(dc)
    try:
        s = m5b_set(RomSet.from_zip(rom_dir / "daytona.zip"), disc)
    finally:
        disc.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = s.write_zip(out_dir / "daytona.zip")
    shutil.copy(rom_dir / "model1io.zip", out_dir / "model1io.zip")
    command = M5B_COMMAND.format(path=out_dir.resolve())
    (out_dir / "README.txt").write_text(M5B_README.format(command=command, ticks=build.LONG_TIME))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", nargs="?", type=Path)
    ap.add_argument("--scenery", choices=("empty", "model"), default="model")
    ap.add_argument("--sixteen", action="store_true", help="the M4a set: Desert City at its own id 3 (custom MAME)")
    ap.add_argument("--mirrors", action="store_true", help="the M4b set: the mirrors live as ids 8-11 (custom MAME)")
    ap.add_argument("--m5a", action="store_true",
                    help="the M5a set: National Park Speedway (4) and Mermaid Lake (7) live (custom MAME)")
    ap.add_argument("--m5b", action="store_true",
                    help="the M5b set: the row of eight, all five Dreamcast courses live (custom MAME)")
    a = ap.parse_args()
    if a.sixteen + a.mirrors + a.m5a + a.m5b > 1:
        ap.error("--sixteen, --mirrors, --m5a and --m5b are four sets: pick one")
    if a.m5b:
        out = a.out_dir or REPO / "out" / "play" / "m5b"
        print(m5b(out), f"(the row of eight; ids 3-7: start time {build.LONG_TIME} ticks)")
        print("Run from the repository root, with the set's own NVRAM directory (never MAME's shared nvram/):")
        print("    " + M5B_COMMAND.format(path=out.resolve()))
        sys.exit(0)
    if a.m5a:
        out = a.out_dir or REPO / "out" / "play" / "m5a"
        print(m5a(out), f"(ids 3, 4, 7 and their mirrors 11, 12, 15: start time {build.LONG_TIME} ticks)")
        print("Run from the repository root, with the set's own NVRAM directory (never MAME's shared nvram/):")
        print("    " + M5A_COMMAND.format(path=out.resolve()))
        sys.exit(0)
    if a.mirrors:
        out = a.out_dir or REPO / "out" / "play" / "mirrors-m4b"
        print(mirrors(out), f"(the mirrors live as ids 8-11; Desert City and its mirror: start time {build.LONG_TIME}"
              " ticks)")
        print("Run from the repository root, with the set's own NVRAM directory (never MAME's shared nvram/):")
        print("    " + MIRRORS_COMMAND.format(path=out.resolve()))
        sys.exit(0)
    if a.sixteen:
        out = a.out_dir or REPO / "out" / "play" / "sixteen-m4a"
        print(sixteen(out), f"(Desert City at id 3, start time {build.LONG_TIME} ticks)")
        sys.exit(0)
    out = a.out_dir or REPO / "out" / "play" / ("desert-city-m3b" if a.scenery == "model" else "desert-city-m3a")
    print(main(out, a.scenery), f"(scenery {a.scenery}, start time {build.LONG_TIME} ticks)")
