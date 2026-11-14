"""MAME tests of Phase 10 (M5a): courses 4-7 converted at their own ids (plan docs/superpowers/plans/
2026-11-10-phase10-national-park-mermaid-lake.md; discovery log docs/spike/2026-11-courses-4-7.md), and of Phase 11
(M5b): the row of eight on the shipped chain without mirrors (`m5b_set`; plan docs/superpowers/plans/
2026-11-12-phase11-circuit-pixie-rin-rin-rink.md).

Marked `mame` and `custom_mame` (every set here is a custom one: skipped when dtools.config.custom_mame() finds no
build of the current patch)."""

import shutil
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade import select as S
from dtools.arcade.romset import RomSet
from dtools.convert import build
from dtools.mame.road import road_file
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

SNAP_DIR = config.out_dir() / "phase10" / "snaps"

# --- Task 3: National Park Speedway (Dreamcast n 4) at course id 4 --------------------------------------------------

PARK_ENTRY = 4                  # select.ENTRIES[4]: National Park Speedway, course id 4
THIN_MARGIN = 300
"""Set before thinning (Review Focus 4, Phase 9 Task 4's ruling; not revisited after measuring, the controller's
ruling): build.QUAD[4], REMESH[4] and THIN[4] (the controller's levers, in order, then the Phase 7 recipe) were chosen
so the replayed lap (the unthinned run's visible cells and cars' share, on the reduced cells) draws at most
BAR - THIN_MARGIN (3,951) polygons
per list with the frame's polygon count at most course.OBC - THIN_MARGIN (4,700), and the gate holds the measured lap
to the same. 300 is Phase 9's largest mirror-against-original gap (298: Advanced's mirror; Desert City's own +231),
so the course's mirror (id 12) inherits room under the bar, and covers a lap driven otherwise than the autoplay's
(the cars' share moves with them)."""
PARK_GO = 1651
"""The race's first frame (the clock's first tick): Expert's standing countdown, as Desert City's (ID3_GO)."""
PARK_REPEAT = frozenset({PARK_GO + 1})
"""The frame after the green light draws a list equal to the one before (a standing start: nothing moved), as on
Desert City and stock Expert."""
PARK_TIME_OVER = 6892
"""The clock out on the set without the long time: Beginner's start time (3200 ticks) and the extensions of the two
checkpoints the player reaches first (PARK_RISES; Desert City's player reaches none)."""
PARK_RISES = 2
PARK_SECONDS = 172              # the long-time run, to frame 9894: past the player's lap and every AI car's second lap
PARK_LAP_END = 8539             # the player (wheel centred) reaches the lap line, lap 1
PARK_LONG_LAPS = 2              # laps each AI car completes in the long-time run
PARK_MAX, PARK_MAX_AT, PARK_COUNT_MAX = 3754, 6950, 4255
"""The drawn-polygon maximum per list over the player's lap (GO to PARK_LAP_END, the long-time set) and its frame
(segment 337), and the largest polygon count (the cells log) over it. Task 3 fix round 1: 3,909 and 4,423 (the
replay's prediction exactly); since Task 4's fix round 1 (build.NAMED_PROPS[4]: the big wheel's frame, object 3,
dropped) 25 fewer drawn at the same frame and 28 fewer counted, measured; the ground calls (CALLS_PARK_*), the
time-over and the lap are unchanged. v0.1.1 (build.shadow_volumes takes kind-3 volumes: CS4's 23 in ID colours
dropped): 3,884 and 4,395 before, 130 fewer drawn at the same frame and 140 fewer counted, measured; the rest
unchanged."""
CALLS_PARK_TO_TIME_OVER = {0: 24588, 1: 4266, 2: 1570, 3: 1421, 4: 486, 5: 199, 6: 90, 8: 218, 9: 502, 14: 24,
                           16: 23, 20: 206, 22: 141, 24: 173, 25: 21, 26: 248, 28: 80, 29: 1, 30: 134, 32: 219,
                           33: 2, 34: 751, 35: 751, 36: 764, 37: 192, 38: 285, 39: 714}
CALLS_PARK_LONG_TIME = {0: 33047, 1: 4268, 2: 2070, 3: 1419, 4: 535, 5: 1632, 6: 655, 7: 1477, 8: 216, 9: 502,
                        10: 1284, 11: 819, 12: 832, 13: 1015, 14: 389, 15: 116, 16: 490, 17: 566, 18: 1625, 20: 206,
                        21: 1636, 22: 141, 23: 637, 24: 173, 25: 621, 26: 248, 27: 464, 28: 80, 29: 296, 30: 134,
                        31: 1221, 32: 219, 33: 529, 34: 1368, 35: 1458, 36: 944, 37: 229, 38: 285, 39: 714}
"""The ground calls per car (test_convert_mame._ground_hits) from PARK_GO to the time-over / over the long-time run,
exact, from the runs (Desert City's convention, CALLS_ID3_*)."""
PARK_SNAPS = {"select": 750, "start": PARK_GO + 30, "mid-lap": 5600}
"""The snapshots: the select with entry 4 under the cursor, the start (GO + 30), mid-lap."""


def park_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """The final set (tests/test_mirrors_mame.py `final_set`: grow_ids, guard_new_ids, widen_ids, Desert City at id
    3, National Park Speedway at id 4 (build.REMESH[4], QUAD[4], NAMED_PROPS[4]), since Task 4 Mermaid Lake at id 7,
    the mirrors of 0-3 at 8-11, a ranking table per id, the select with M5A_ENTRIES, entries 4 and 7 live)."""
    from tests.test_mirrors_mame import final_set
    return final_set(rs, disc)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_national_park_id4(tmp_path, romset_zip, disc):  # noqa: F811
    """National Park Speedway (n 4) at its own course id 4 on the final set (`park_set`), entered from the select
    (DAYTONA_ENTRY 4), custom MAME:

    - the long-time set (build.long_time on id 4's own time tables), PARK_SECONDS: test_desert_city_id3's race checks
      (test_convert_mame._drive_checks: a standing start, the clock from PARK_GO, every car's progress and laps,
      every ground call a polygon, the checkpoints 125, 257 and the lap line, a new display list every frame but
      PARK_REPEAT) and the frame-rate bar over the player's lap PARK_GO..PARK_LAP_END (every one of the 426
      segments, every frame counted): every list draws at most BAR polygons and the frame's polygon count stays below
      course.OBC (no cap hit), no visible cell's scenery is left out and none is drawn outside the visible cells; the
      thinning's margin, set before thinning (THIN_MARGIN), holds on the run: at most BAR - THIN_MARGIN drawn and
      OBC - THIN_MARGIN counted; the maximum (PARK_MAX at PARK_MAX_AT) and the largest count frozen; snapshots
      (PARK_SNAPS) to out/phase10/snaps/task3; backup SRAM after the run holds the fresh reset's words for every
      course id (test_convert_mame._check_fresh_sram: id 4 writes none);
    - the set without the long time: id 4 to its time-over (the checks to PARK_TIME_OVER), the game over, then
      Beginner (entry 0) as the session's second game (DAYTONA_SECOND at test_mirrors_mame.SESSION_COIN2), against
      Beginner then Beginner on the same set: Phase 9's relaxed check (Review Focus 1 there; a second game follows
      the session's history, so not call for call): the TGP ground slot of each course load (the attract's 0, id 4
      for its game and the attract after it, then Beginner's 0, whose list block and pool are Beginner's), the
      second game draws exactly the objects, texture points and headers the reference's draws, texture RAM (both
      sheets) and luma RAM after its load are the reference's, and backup SRAM differs only in the bookkeeping block
      (test_mirrors_mame.SRAM_BOOKKEEPING: id 4's game is not counted)."""
    assert build.QUAD.get(4) == (10.0, 0.125) and build.REMESH.get(4) == 0.1 and 4 not in build.THIN
    assert build.NAMED_PROPS.get(4) == (3,)
    _gate(tmp_path, romset_zip, park_set(RomSet.from_zip(romset_zip), disc), park_gate())


# --- Task 4: Mermaid Lake (Dreamcast n 7) at course id 7 ----------------------------------------------------------

LAKE_ENTRY = 7                  # select.ENTRIES[7]: Mermaid Lake, course id 7
LAKE_GO = 1651
"""The race's first frame: Expert's standing countdown, as National Park Speedway's."""
LAKE_REPEAT = frozenset({LAKE_GO + 1})
LAKE_AUTOPILOT = 1
"""The long-time run's player is driven (autoplay.lua's DAYTONA_AUTOPILOT, the road line's centre 6 segments ahead)
for its first lap, then lets go of the gas (it stops at segment 60), so the run's checks see one lap, as on the other
courses; its lap is the bar's (discovery log, "Mermaid Lake", "The gate's driver"). With the paddle centred the player
finishes its lap only at 8440, at half the pace off the walls, and reaches lap 2's first checkpoint (segment 120)
about frame 9480, before the run's least end (GO + 8000); that lap's bar was measured too: 3,837 drawn at most (1715),
the count 4,537, nothing over BAR - THIN_MARGIN."""
LAKE_TIME_OVER = 6051
"""The clock out on the set without the long time (the paddle centred, as Task 3's): Beginner's start time and the
extensions of the checkpoints the player reaches first (LAKE_RISES: checkpoint 120; the lifted tunnel lets the centred
paddle through, the player at segment 358 at the time-over)."""
LAKE_RISES = 1
LAKE_SECONDS = 175              # to frame 10066: past GO + 8000; every AI car on lap 2 by 9323, none on 3 before 10439
LAKE_LAP_END = 4955             # the driven player reaches the lap line, lap 1 (then lets go: it stops at segment 60)
LAKE_LONG_LAPS = 2
LAKE_MAX, LAKE_MAX_AT, LAKE_COUNT_MAX = 3897, 3053, 4537
"""The drawn-polygon maximum per list over the player's lap and its frame, and the largest polygon count: unreduced
(QUAD, REMESH and THIN name no 7), within BAR - THIN_MARGIN and course.OBC - THIN_MARGIN: 3,897 at 3053 (segment 162),
354 under the bar; the count's largest the bunched field after the green light."""
CALLS_LAKE_TO_TIME_OVER = {0: 22118, 1: 3118, 2: 486, 3: 1392, 4: 330, 6: 72, 8: 567}
CALLS_LAKE_LONG_TIME = {0: 36881, 1: 5619, 2: 2884, 3: 2984, 4: 1079, 5: 2504, 6: 2571, 7: 2207, 8: 1815, 9: 1513,
                        10: 1407, 11: 348, 12: 1126, 13: 161, 14: 253, 15: 98, 16: 11, 17: 234, 18: 2305, 19: 1048,
                        20: 10, 21: 545, 22: 516, 23: 225, 24: 250, 25: 243, 26: 120, 27: 280, 28: 139, 29: 137,
                        30: 150, 31: 268, 32: 44, 33: 172, 34: 161, 35: 299, 36: 321, 37: 283, 38: 245, 39: 334}
"""The ground calls per car from LAKE_GO to the time-over / over the long-time run, exact, from the runs."""
LAKE_SNAPS = {"select": 750, "start": LAKE_GO + 30, "mid-lap": 3300}
"""The snapshots: the select with entry 7 under the cursor, the start (GO + 30), mid-lap (segment 204)."""


def lake_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """The final set (tests/test_mirrors_mame.py `final_set`): Desert City at id 3, National Park Speedway at id 4,
    Mermaid Lake at id 7 (build.LIFT[7], NAMED_PROPS[7]), the mirrors of 0-3, a ranking table per id, the select with
    M5A_ENTRIES (entries 4 and 7 live)."""
    from tests.test_mirrors_mame import final_set
    return final_set(rs, disc)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_mermaid_lake_id7(tmp_path, romset_zip, disc):  # noqa: F811
    """Mermaid Lake (n 7) at its own course id 7 on the final set (`lake_set`), entered from the select
    (DAYTONA_ENTRY 7), custom MAME: test_national_park_id4's checks (`_gate`) with LAKE_* in place of PARK_*:

    - the long-time set, LAKE_SECONDS, the player driven for its lap (LAKE_AUTOPILOT): the race checks and the
      frame-rate bar over the player's lap LAKE_GO..LAKE_LAP_END (every one of the 500 segments, every frame
      counted): at most BAR - THIN_MARGIN drawn and OBC - THIN_MARGIN counted, no cap hit, no visible cell left out,
      none drawn outside; the maximum frozen (unreduced: Phase 7's estimate holds); snapshots (LAKE_SNAPS) to
      out/phase10/snaps/task4 (mermaid-lake-*.png); backup SRAM fresh after the run;
    - the set without the long time (the paddle centred): id 7 to its time-over, then Beginner as the session's
      second game, against Beginner then Beginner (Phase 9's relaxed check);
    - the course is lifted (build.LIFT[7]: every ground call in the tunnel, segments 316-345, answers the road's own
      polygon: the race checks' "every ground call a polygon" and the cars' progress through it)."""
    assert build.LIFT.get(7) == 41.0 and build.NAMED_PROPS.get(7) == (2,)
    assert 7 not in build.QUAD and 7 not in build.REMESH and 7 not in build.THIN
    _gate(tmp_path, romset_zip, lake_set(RomSet.from_zip(romset_zip), disc), lake_gate())


class Gate(NamedTuple):
    """One course's gate (`_gate`): its id and select entry, the snapshots' file stem and folder (under SNAP_DIR, or
    an absolute path), the frozen numbers of its runs, the autopilot (the laps it drives the long-time run, 0 for
    the centred paddle), and the TGP slots the attract loads after the course's own attract before the session's
    second coin (a course whose game ends early enough sees the attract move on: Desert City's, Expert's slot 2)."""
    course: int
    entry: int
    name: str
    label: str
    dest: str
    seconds: int
    go: int
    repeat: frozenset
    lap_end: int
    long_laps: int | tuple             # laps each AI car completes (a tuple: any of them)
    maximum: tuple
    calls_long_time: dict
    calls_to_time_over: dict
    time_over: int
    rises: int
    snaps: dict
    autopilot: int = 0
    attract: tuple = ()


def park_gate(dest: str = "task3") -> Gate:
    """National Park Speedway's gate (test_national_park_id4), its snapshots to `dest`."""
    return Gate(course=4, entry=PARK_ENTRY, name="national-park", label="National Park Speedway", dest=dest,
                seconds=PARK_SECONDS, go=PARK_GO, repeat=PARK_REPEAT, lap_end=PARK_LAP_END, long_laps=PARK_LONG_LAPS,
                maximum=(PARK_MAX, PARK_MAX_AT, PARK_COUNT_MAX), calls_long_time=CALLS_PARK_LONG_TIME,
                calls_to_time_over=CALLS_PARK_TO_TIME_OVER, time_over=PARK_TIME_OVER, rises=PARK_RISES,
                snaps=PARK_SNAPS)


def lake_gate(dest: str = "task4") -> Gate:
    """Mermaid Lake's gate (test_mermaid_lake_id7), its snapshots to `dest`."""
    return Gate(course=7, entry=LAKE_ENTRY, name="mermaid-lake", label="Mermaid Lake", dest=dest,
                seconds=LAKE_SECONDS, go=LAKE_GO, repeat=LAKE_REPEAT, lap_end=LAKE_LAP_END, long_laps=LAKE_LONG_LAPS,
                maximum=(LAKE_MAX, LAKE_MAX_AT, LAKE_COUNT_MAX), calls_long_time=CALLS_LAKE_LONG_TIME,
                calls_to_time_over=CALLS_LAKE_TO_TIME_OVER, time_over=LAKE_TIME_OVER, rises=LAKE_RISES,
                snaps=LAKE_SNAPS, autopilot=LAKE_AUTOPILOT)


def _gate(tmp_path, romset_zip, full: RomSet, g: Gate, untitled: RomSet | None = None) -> dict:
    """test_national_park_id4's checks (its docstring) for the course `g` on the set `full`, whose select row (8 or
    16 entries: autoplay.lua's DAYTONA_ROW 8 for a row of 8) names the course at entry `g.entry`, live. Returns the
    long-time run's cells log (frame -> (count, visible cells in draw order)) and the lap's frames, for checks of
    the caller's own (Rin Rin Rink's view).

    With `untitled` (the release set's gates, tests/test_release_mame.py: `full` carries the Extended title and
    `untitled` is the same chain without it), the time-over session's ground calls per car need only the same sum
    and each car within 1 of the frozen count (_drive_checks' `split`), and the session is run on `untitled` too: its
    ground log must equal `full`'s call for call and value for value (every call's arguments and result, in order,
    and the TGP slot loads), only the frames the calls are logged in may differ. Why (Phase 12 Task 4, discovery log
    docs/spike/2026-11-release.md, "The title built"): the title's longer boot upload shifts the i960 a fraction of a
    frame against MAME's frame_done, so a call or a progress sample may land on the next frame, and
    `_ground_hits`' nearest-segment assignment then gives one call to another car (National Park Speedway: cars 1 and
    4); the game's work is the same, as the equal ground logs show. Every other number stays exact."""
    from PIL import Image
    from dtools.arcade import tgp
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_convert_mame import (BAR, ID3_HUD, _cars, _check_fresh_sram, _drive_checks, _drive_checks_calls,
                                         _geo_per_frame, _new_lists)
    from tests.test_custom_mame import _custom, _lists
    from tests.test_mirrors_mame import SESSION_COIN2, SESSION_FROM, SESSION_SECONDS, SESSION_TEX_AT, SRAM_BOOKKEEPING
    exe = str(_custom())
    c = G.decode_course(full, g.course)
    row = S.select_row(full)
    assert row.records[0][g.entry].course == g.course and row.live[g.entry]
    row_env = {"DAYTONA_ROW": "8"} if len(row.live) == 8 else {}    # a row of 16: autoplay.lua's default
    assert c.rolling_start[1] is False and c.actor_table == build.EMPTY_ACTORS
    owner = {p.oba: cell for cell, lst in enumerate(c.cells) for p in c.object_lists[lst]}
    occupied = set(owner.values())
    dest = SNAP_DIR / g.dest
    dest.mkdir(parents=True, exist_ok=True)

    # the long-time run: the race and the bars
    long_set = build.long_time(full, c)
    driver = {}
    if g.autopilot:
        (tmp_path / "long_time").mkdir(parents=True)
        driver = {"DAYTONA_AUTOPILOT": str(road_file(c, tmp_path / "long_time" / "road.txt")),
                  "DAYTONA_AUTOPILOT_LAPS": str(g.autopilot)}
    out = mame_build.run_set(long_set, tmp_path / "long_time", g.seconds, romset_zip,
                             {"DAYTONA_ENTRY": str(g.entry), "DAYTONA_GEO_FROM": str(g.go),
                              "DAYTONA_GEO_EVERY": "1", "DAYTONA_CELLS_FROM": str(g.go),
                              "DAYTONA_SNAPS": ",".join(map(str, g.snaps.values())), **driver, **row_env},
                             logs=("ground", "rate", "progress", "geo", "cells"), mame=exe)
    per = _geo_per_frame(out["geo"], long_set)                  # first: it removes the log, whatever fails later
    samples, _ = runner.parse_ground(out["ground"])
    runner.require_calls(samples)
    rows, timer = runner.parse_progress(out["progress"])
    rate = runner.parse_rate(out["rate"])
    run = (samples, rate, _cars(rows), timer)
    calls = _drive_checks_calls(c, run, g.go)
    shots = sorted(out["snap"].rglob("*.png"))
    for p, (label, frame) in zip(shots, g.snaps.items()):      # before the frozen numbers: a failing run leaves them
        shutil.copy(p, dest / f"{g.name}-{label}-{frame}.png")
        if label == "mid-lap":
            Image.open(p).crop(ID3_HUD).save(dest / f"{g.name}-hud-map-{frame}.png")
    cells = runner.parse_cells(out["cells"])
    player = {r.frame: (r.lap, r.segment) for r in run[2][0]}
    lap = range(g.go, g.lap_end + 1)
    left_out, strays = [], []
    for f in lap:
        drawn = {owner[a] for a in per[f][1] if a in owner}
        visible = {x for x in cells[f][1] if x in occupied}
        left_out += [f] if visible - drawn else []
        strays += [f] if drawn - set(cells[f][1]) else []
    top = max(lap, key=lambda f: (per[f][0], -f))
    busiest = max(lap, key=lambda f: (cells[f][0], -f))
    report = dict(calls=calls, laps={car: (r[0].lap, r[-1].lap) for car, r in run[2].items()},
                  lap_end=min((f for f, (n, _) in player.items() if f > g.go and n == 1), default=None),
                  max=(per[top][0], top, player[top][1]), count_max=(cells[busiest][0], busiest),
                  over=sum(per[f][0] > BAR for f in lap), cap_hits=sum(cells[f][0] >= C.OBC for f in lap),
                  left_out=len(left_out), strays=len(strays), margin=BAR - per[top][0])
    print(f"\nlong-time run: {report}")
    _drive_checks("long_time", c, run, calls_long_time=g.calls_long_time, go=g.go, repeat=g.repeat,
                  long_laps=g.long_laps)
    assert player[g.go][0] == 0 and player[g.lap_end] == (1, 0) and player[g.lap_end - 1][0] == 0
    assert {player[f][1] for f in lap} == set(range(len(c.road)))
    assert set(lap) - g.repeat <= _new_lists(rate)
    assert set(lap) <= set(per) and set(lap) <= set(cells)
    assert not left_out and not strays and report["over"] == report["cap_hits"] == 0
    assert per[top][0] <= BAR - THIN_MARGIN and cells[busiest][0] <= C.OBC - THIN_MARGIN, report
    assert (per[top][0], top, cells[busiest][0]) == g.maximum
    _check_fresh_sram((tmp_path / "long_time" / "run" / "nvram" / "daytona" / "backup1").read_bytes(), full)

    # the course to its time-over, then Beginner, in one session; against Beginner then Beginner
    env = {"DAYTONA_SECOND": f"0:{SESSION_COIN2}", "DAYTONA_GEO_FROM": str(SESSION_FROM), "DAYTONA_GEO_EVERY": "1",
           "DAYTONA_TEXDUMP_AT": str(SESSION_TEX_AT), "DAYTONA_SNAPS": str(SESSION_TEX_AT + 600), **row_env}
    sessions = {}
    for first in (g.entry, 0):
        wd = tmp_path / f"session-{first}"
        (wd / "tex").mkdir(parents=True)
        o = mame_build.run_set(full, wd, SESSION_SECONDS, romset_zip,
                               {"DAYTONA_ENTRY": str(first), **env, "DAYTONA_TEXDUMP_DIR": str((wd / "tex").resolve())},
                               logs=("geo", "rate", "progress", "ground"), mame=exe)
        lists, refs = _lists(o["geo"])
        o["geo"].unlink()
        samples, _ = runner.parse_ground(o["ground"])
        rows, s_timer = runner.parse_progress(o["progress"])
        sessions[first] = dict(lists=lists, refs=refs, tex=wd / "tex", timer=s_timer, out=o,
                               run=(samples, runner.parse_rate(o["rate"]), _cars(rows), s_timer),
                               slots=runner.parse_ground_slots(o["ground"]),
                               sram=(wd / "run" / "nvram" / "daytona" / "backup1").read_bytes())
    a, b = sessions[g.entry], sessions[0]
    calls = _drive_checks_calls(c, a["run"], g.go, g.time_over)
    print(f"to the time-over: ground calls per car {calls}")
    _drive_checks("converted", c, a["run"], calls_to_time_over=g.calls_to_time_over, go=g.go,
                  time_over=g.time_over, repeat=g.repeat, rises_to_time_over=g.rises, split=1 if untitled else 0)
    if untitled is not None:
        wd = tmp_path / f"untitled-{g.entry}"
        o = mame_build.run_set(untitled, wd, SESSION_SECONDS, romset_zip, {"DAYTONA_ENTRY": str(g.entry), **env},
                               logs=("ground",), mame=exe)
        mine = runner.parse_ground(a["out"]["ground"])[0]
        theirs = runner.parse_ground(o["ground"])[0]

        def values(samples):
            return [(s.dest, s.y, s.id, s.x, s.z, s.height, s.poly) for s in samples]
        assert len(mine) == len(theirs) > 10000 and values(mine) == values(theirs)
        assert [x for _, x in a["slots"]] == [x for _, x in runner.parse_ground_slots(o["ground"])]
        moved = sum(m.frame != t.frame for m, t in zip(mine, theirs))
        print(f"the untitled set's session: {len(theirs)} ground calls equal value for value, {moved} logged on "
              f"another frame")
    sa, sb = ([x for _, x in r["slots"]] for r in (a, b))
    own = len(sa) - 1 - len(g.attract)          # the course's game and its attract, then g.attract's loads
    assert len(sa) == len(sb) + len(g.attract) >= 3 and sa[0] == sb[0] == 0 and set(sa[1:own]) == {g.course} and \
        sa[own:-1] == list(g.attract) and set(sb[1:-1]) == {0}, (a["slots"], b["slots"])
    assert sa[-1] == sb[-1] == 0 and a["slots"][-1][0] > SESSION_COIN2 and b["slots"][-1][0] > SESSION_COIN2

    def bases(slot: int) -> tuple[int, int]:
        return tuple(struct.unpack_from("<I", full.copro_data, 4 * (w + slot))[0] for w in (tgp.SLOT_LIST,
                                                                                             tgp.SLOT_POOL))
    assert bases(sa[-1]) == bases(0) != bases(g.course)
    assert min(f for f, v in b["timer"].items() if v == 0) < SESSION_COIN2       # the first game is over
    assert len(b["lists"]) > 2000 and a["refs"] == b["refs"]
    for kind in ("tex0", "tex1", "luma"):
        got, want = (r["tex"] / f"{kind}.{SESSION_TEX_AT}.bin" for r in (a, b))
        assert got.read_bytes() == want.read_bytes(), kind
    differ = np.flatnonzero(np.frombuffer(a["sram"], np.uint8) != np.frombuffer(b["sram"], np.uint8))
    assert len(differ) and all(SRAM_BOOKKEEPING[0] <= x < SRAM_BOOKKEEPING[1] for x in differ), differ
    print(f"Beginner after {g.label}: {sum(a['lists'].get(f) != v for f, v in b['lists'].items())} of "
          f"{len(b['lists'])} lists differ from Beginner after Beginner; drawn refs equal; SRAM differs at "
          f"{[hex(0x1d00000 + int(x)) for x in differ]}")
    shot = sorted(a["out"]["snap"].rglob("*.png"))
    if shot:
        shutil.copy(shot[0], dest / f"beginner-after-{g.name}-{SESSION_TEX_AT + 600}.png")
    return dict(cells=cells, lap=lap)


# --- Task 5: the mirrors of National Park Speedway (id 12) and Mermaid Lake (id 15), the select -------------------

MIRROR_MARGIN = 298
"""How much worse than its original's maximum a mirror's drawn-polygon maximum per list may be over its lap, set
before measuring (recorded here before the first MAME run of test_mirrors_12_15): Phase 9's rule
(tests/test_mirrors_mame.py DESERT_MARGIN), Phase 9 Task 3's largest mirror-against-original gap, 298 (Advanced's
mirror, 3,633 against 3,931). The binding bar stays the hard one: at most BAR drawn per list and no cap hit. The
mirrors inherit their originals' reductions (build.QUAD[4], REMESH[4], NAMED_PROPS[4] and [7]); none of their own."""
MIRROR_ROW = tuple(e._replace(live=True, label="") if e.course in (12, 15) else e for e in S.M5A_ENTRIES)
"""A test-only row: Phase 10's row (select.M5A_ENTRIES) with entries 12 and 15 live (entry 15 without COMING SOON, as
a live entry with no origin line: its band draws MIRROR alone). Not shipped: no shipped row reaches 12 and 15 (Pascal,
2026-11-11: mirror mode leaves the product; since Phase 11 the shipped row is the 8 courses, the mirrors off the
select and off the shipped chain; reverse, with a NORMAL / REVERSE screen after the select, is M9); this row only lets
the gate enter the mirrors from the select."""
MIRROR_GATES = {12: dict(original=4, entry=12, original_entry=PARK_ENTRY, name="mirror-12"),
                15: dict(original=7, entry=15, original_entry=LAKE_ENTRY, name="mirror-15")}
"""Per mirror id: its original, its select entry (MIRROR_ROW) and its original's, the snapshots' stem."""
MIRROR_PARTS = {12: 1907, 15: 1787}
"""Per mirror, the frame of the first ground call without a mirrored partner in its original's run
(test_mirror._parting, from test_mirrors_mame.RACE_LOAD; measured, zero margin: MAME is deterministic): 256 and 136
frames after GO (Desert City's mirror 329, Advanced's and Expert's 600 and 21)."""
MIRROR_OPPONENTS_TO = {12: 1662, 15: 1655}
"""Per mirror, the last frame through which every opponent stays within test_mirror.SYMMETRY_TOL (XZ) of its
original's car mirrored, from frame 1330 (measured: car 3 parts first at 1663 on 12, car 5 at 1656 on 15; Desert
City's mirror 1655)."""
MIRROR_LAP_END = {12: 8323, 15: 4935}
"""Per mirror, the frame its player reaches the lap line, lap 1 (measured; the originals' PARK_LAP_END 8539 with the
centred paddle, LAKE_LAP_END 4955 driven)."""
MIRROR_MAX = {12: (3883, 6872, 4427), 15: (3837, 1715, 4537)}
"""Per mirror, (the drawn-polygon maximum per list over its player's lap, its frame, the largest polygon count over
it) (measured, frozen): 12 draws 3,883 at 6872 (segment 338), +129 on National Park Speedway's 3,754 (inside
MIRROR_MARGIN; 368 under BAR), the count 4,427 (573 under course.OBC; v0.1.1's kind-3 volumes: 4,013 and 4,567
before, 130 and 140 fewer, as the original); 15 draws 3,837 at 1715, 60 fewer than Mermaid
Lake's 3,897, the count 4,537 at 1715, the field bunched after the green light (Mermaid Lake's own largest count, the
same frame)."""
MIRROR_CAMERA = {12: (0.0036, 5, 0.8717, 0.8745), 15: (0.3402, 956, 0.8686, 1.6557)}
"""Per mirror, the scripted camera against its original's mirrored over CAMERA_FROM..PAIRED_TO (the fly-by), measured
and frozen (4 decimals): (the largest x or y gap, the largest angle gap in s16 units: pitch, roll negated, yaw
0x8000 - yaw; the least and the largest z offset from the plane's mirror). Phase 9 measured x, y within 0.01, the angles
within 2 and z 0.86-0.871 on 8-11 (test_mirrors_mame.CAMERA_Z_OFFSET, the game's constant not found). On 12 x and y
hold, z sits 0.8717-0.8745 (0.004 past Phase 9's top) and the pitch up to 5 off (yaw 3, roll 0), on every frame. On 15
the first frames depart (1332-1384: x, y up to 0.34, the yaw up to 956, about 5 degrees, z up to 1.66) and settle into
Phase 9's bounds by 1385 to PAIRED_TO (MIRROR_CAMERA_SETTLED). Not traced (the time box: the camera offset was Phase
9's Not found too); the fly-by shows each course mirrored (snapshots)."""
MIRROR_CAMERA_SETTLED = {12: None, 15: 1385}
"""Per mirror, the first frame from which to PAIRED_TO the camera holds Phase 9's rule (CAMERA_Z_OFFSET, CAMERA_ANGLE,
x and y within SYMMETRY_TOL): None on 12, whose z offset and pitch never do (MIRROR_CAMERA)."""


def mirror_row_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """The final chain (tests/test_mirrors_mame.py `final_set`: the mirrors of 0-4 and 7, the rankings) with the select
    patched with MIRROR_ROW (12 and 15 live): the test hook that reaches the mirrors from the select (mirror mode
    is dropped, Pascal 2026-11-11: no shipped row reaches 12 and 15). Every table but the select's is final_set's."""
    from tests.test_mirrors_mame import _final_chain
    return S.patch_select(_final_chain(rs, disc), MIRROR_ROW)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_mirrors_12_15(tmp_path, romset_zip, disc):  # noqa: F811
    """Phase 10 Task 5: the mirrors of National Park Speedway (id 12) and Mermaid Lake (id 15), each entered from the
    select on `mirror_row_set` (MIRROR_ROW: entries 12 and 15 live for the test), against its original entered from
    the select on the same set, with the long time (build.long_time on the original's time tables, which the mirror
    shares) and the original's driver (the centred paddle on 4 and 12; the autopilot for one lap on 7 and 15,
    LAKE_AUTOPILOT, each on its own course's road line), custom MAME (Phase 9 Task 4's checks,
    test_mirror_desert_city):

    - the original on this set holds its gate's numbers (PARK_* / LAKE_*: the maximum, its frame and the count over
      the same lap, the lap's end): the test row moves nothing;
    - its original's start: the countdown timer equal to the original's on every frame until either player's first
      checkpoint raises it, GO equal (1651, standing), the player still (XZ) to GO, rolling_start False;
    - the pairing: every ground call of the race (from RACE_LOAD) pairs with the original's mirrored up to
      MIRROR_PARTS (the attract's calls equal); every opponent within SYMMETRY_TOL of its mirror through
      MIRROR_OPPONENTS_TO; the scripted camera from CAMERA_FROM to PAIRED_TO against the original's mirrored, frozen
      as measured (MIRROR_CAMERA; Phase 9's rule from MIRROR_CAMERA_SETTLED: Mermaid Lake's mirror's first frames
      and National Park Speedway's z offset lie past Phase 9's measured bounds, findings, not traced);
    - the bar over the mirror's lap (GO to MIRROR_LAP_END, every segment, every frame): a new display list on every
      frame but GO + 1, at most BAR drawn per list, the polygon count below course.OBC (no cap hit), the maximum no
      worse than its original's by more than MIRROR_MARGIN (set before measuring), no scenery left out or drawn
      outside the visible cells; the numbers printed and frozen (MIRROR_MAX); backup SRAM fresh after the run;
    - snapshots to out/phase10/snaps/task5: mirror-<id>-{select,start,mid-lap,hud-map}-<frame>.png and the original's
      at the same frames (mirror-<id>-original-*.png)."""
    import math
    from PIL import Image
    from dtools.arcade import mirror as M
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_convert_mame import BAR, ID3_HUD, _check_fresh_sram, _new_lists
    from tests.test_custom_mame import _custom
    from tests.test_mirror import SYMMETRY_TOL, _calls, _parting
    from tests.test_mirrors_mame import (BARS_FROM, CAMERA_ANGLE, CAMERA_FROM, CAMERA_Z_OFFSET, PAIRED_TO, RACE_LOAD,
                                         _geo_counts, _go)
    assert MIRROR_MARGIN == 298 and BAR - PARK_MAX >= MIRROR_MARGIN and BAR - LAKE_MAX >= MIRROR_MARGIN
    exe = str(_custom())
    full = mirror_row_set(RomSet.from_zip(romset_zip), disc)
    row = S.select_row(full)
    assert row.live[12] and row.live[15] and not S.M5A_ENTRIES[12].live and not S.M5A_ENTRIES[15].live
    dest = SNAP_DIR / "task5"
    dest.mkdir(parents=True, exist_ok=True)
    gates = {4: dict(seconds=PARK_SECONDS, go=PARK_GO, repeat=PARK_REPEAT, lap_end=PARK_LAP_END, autopilot=0,
                     maximum=(PARK_MAX, PARK_MAX_AT, PARK_COUNT_MAX), mid=PARK_SNAPS["mid-lap"]),
             7: dict(seconds=LAKE_SECONDS, go=LAKE_GO, repeat=LAKE_REPEAT, lap_end=LAKE_LAP_END,
                     autopilot=LAKE_AUTOPILOT, maximum=(LAKE_MAX, LAKE_MAX_AT, LAKE_COUNT_MAX),
                     mid=LAKE_SNAPS["mid-lap"])}
    failures = []
    for k, mg in MIRROR_GATES.items():
        o, g = mg["original"], gates[mg["original"]]
        co, ck = G.decode_course(full, o), G.decode_course(full, k)
        assert ck.rolling_start[1] is False and ck.time_tables == co.time_tables and ck.actor_table == \
            build.EMPTY_ACTORS
        long_set = build.long_time(full, co)
        shots = {"select": 750, "start": g["go"] + 30, "mid-lap": g["mid"]}
        runs = {}
        for which, e, c in (("original", mg["original_entry"], co), ("mirror", mg["entry"], ck)):
            wd = tmp_path / f"{k}-{which}"
            driver = {}
            if g["autopilot"]:
                wd.mkdir(parents=True)
                driver = {"DAYTONA_AUTOPILOT": str(road_file(c, wd / "road.txt")),
                          "DAYTONA_AUTOPILOT_LAPS": str(g["autopilot"])}
            run = mame_build.run_set(long_set, wd, g["seconds"], romset_zip,
                                     {"DAYTONA_ENTRY": str(e), "DAYTONA_GEO_FROM": str(BARS_FROM),
                                      "DAYTONA_GEO_EVERY": "1", "DAYTONA_CELLS_FROM": str(BARS_FROM),
                                      "DAYTONA_SNAPS": ",".join(map(str, shots.values())), **driver},
                                     logs=("geo", "rate", "ground", "progress", "cells"), mame=exe)
            _, per = _geo_counts(run["geo"], long_set, BARS_FROM)
            runs[which] = dict(run=run, per=per, rate=runner.parse_rate(run["rate"]), course=c,
                               cells=runner.parse_cells(run["cells"]), progress=runner.parse_progress(run["progress"]),
                               shots=sorted(run["snap"].rglob("*.png")),
                               sram=(wd / "run" / "nvram" / "daytona" / "backup1").read_bytes())
            stem = mg["name"] + ("-original" if which == "original" else "")
            for p, (label, frame) in zip(runs[which]["shots"], shots.items()):     # before any assert
                shutil.copy(p, dest / f"{stem}-{label}-{frame}.png")
                if label == "mid-lap":
                    Image.open(p).crop(ID3_HUD).save(dest / f"{stem}-hud-map-{frame}.png")

        def lap(r: dict, go: int) -> dict:
            """The bars over the player's lap 1 (test_mirror_desert_city's measures)."""
            c = r["course"]
            rows, _ = r["progress"]
            player = {x.frame: (x.lap, x.segment) for x in rows if x.car == 0}
            ends = [f for f, (n, _) in player.items() if f > go and n == 1]
            end = min(ends) if ends else max(player)
            frames = range(go, end + 1)
            owner = {p.oba: cell for cell, lst in enumerate(c.cells) for p in c.object_lists[lst]}
            occupied = set(owner.values())
            per, cells = r["per"], r["cells"]
            assert set(frames) <= set(per) and set(frames) <= set(cells)
            left_out = [f for f in frames if {x for x in cells[f][1] if x in occupied}
                        - {owner[a] for a in per[f][1] if a in owner}]
            strays = [f for f in frames if {owner[a] for a in per[f][1] if a in owner} - set(cells[f][1])]
            top = max(frames, key=lambda f: (per[f][0], -f))
            busiest = max(frames, key=lambda f: (cells[f][0], -f))
            return dict(end=end, lapped=bool(ends), segments=len({player[f][1] for f in frames}), road=len(c.road),
                        max=(per[top][0], top, player[top][1]), count_max=(cells[busiest][0], busiest),
                        over=sum(per[f][0] > BAR for f in frames),
                        cap_hits=sum(cells[f][0] >= C.OBC for f in frames), left_out=left_out, strays=len(strays),
                        missed=sorted(set(frames) - _new_lists(r["rate"])))
        a, m = runs["original"], runs["mirror"]
        o_lap = lap(a, g["go"])
        # the start
        (_, o_timer), (_, m_timer) = a["progress"], m["progress"]
        rise = min(f for t in (o_timer, m_timer) for f in t if f - 1 in t and t[f] > t[f - 1])
        go = _go(m_timer)
        player = runner.parse_car_positions(m["run"]["progress"])
        (x0, _, z0), (x1, _, z1) = player[BARS_FROM][0], player[go - 1][0]
        still = math.hypot(x1 - x0, z1 - z0)
        # the pairing
        ca, cb = _calls(a["run"]["ground"]), _calls(m["run"]["ground"])
        attract = {f: v for f, v in ca.items() if f < RACE_LOAD} == {f: v for f, v in cb.items() if f < RACE_LOAD}
        ca, cb = ({f: v for f, v in x.items() if RACE_LOAD <= f} for x in (ca, cb))
        first = _parting(ca, cb, max(ca))
        pa = runner.parse_car_positions(a["run"]["progress"])

        def near(car: int, f: int) -> bool:
            x, _, z = M.mirror_world(pa[f][car])
            return math.hypot(x - player[f][car][0], z - player[f][car][2]) <= SYMMETRY_TOL
        cars = [car for car in pa[1330] if car]
        opp = min((next((f for f in range(1330, max(pa) + 1) if not near(car, f)), max(pa) + 1), car)
                  for car in cars)
        # the camera
        o_cam, m_cam = (runner.parse_camera(r["run"]["progress"]) for r in (a, m))
        offsets, cam_xy, angles, rule = [], 0.0, 0, {}
        for f in range(CAMERA_FROM, PAIRED_TO + 1):
            (p, pitch, yaw, roll), (q, q_pitch, q_yaw, q_roll) = o_cam[f], m_cam[f]
            xy = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
            turn = max(abs(pitch - q_pitch), abs(roll + q_roll), abs(((yaw + q_yaw) & 0xFFFF) - 0x8000))
            cam_xy, angles = max(cam_xy, xy), max(angles, turn)
            offsets.append((-1 - p[2]) - q[2])
            rule[f] = xy <= SYMMETRY_TOL and turn <= CAMERA_ANGLE and \
                CAMERA_Z_OFFSET[0] <= offsets[-1] <= CAMERA_Z_OFFSET[1]
        settled = next((f for f in rule if all(rule[x] for x in rule if x >= f)), None)
        m_lap = lap(m, go)
        report = dict(go=go, clock_equal_to=rise - 1, still=still, parts=first and first.frame, opponents_to=opp,
                      camera_xy=cam_xy, camera_angles=angles, camera_z=(min(offsets), max(offsets)),
                      camera_settled=settled,
                      original_lap=o_lap, lap=m_lap, margin=m_lap["max"][0] - o_lap["max"][0])
        print(f"\nentry {mg['entry']} (the mirror of {o}, id {k}): {report}")
        try:
            assert (o_lap["max"][0], o_lap["max"][1], o_lap["count_max"][0]) == g["maximum"], o_lap
            assert o_lap["end"] == g["lap_end"], o_lap
            assert rise > g["go"] + 1000
            assert {f: v for f, v in m_timer.items() if f < rise} == {f: v for f, v in o_timer.items() if f < rise}
            assert go == _go(o_timer) == g["go"] and still < 1e-3, (go, still)
            assert attract
            assert first is not None and first.frame == MIRROR_PARTS[k], first
            assert opp[0] - 1 == MIRROR_OPPONENTS_TO[k], opp
            assert (round(cam_xy, 4), angles, round(min(offsets), 4), round(max(offsets), 4)) == MIRROR_CAMERA[k]
            assert settled == MIRROR_CAMERA_SETTLED[k], settled
            # the bar
            assert m_lap["lapped"] and m_lap["segments"] == m_lap["road"] and m_lap["end"] == MIRROR_LAP_END[k], m_lap
            assert set(m_lap["missed"]) <= g["repeat"] and set(m_lap["missed"]) == set(o_lap["missed"]), m_lap
            assert m_lap["over"] == m_lap["cap_hits"] == m_lap["strays"] == 0 and m_lap["left_out"] == [], m_lap
            assert m_lap["max"][0] <= BAR and m_lap["count_max"][0] < C.OBC
            assert m_lap["max"][0] <= o_lap["max"][0] + MIRROR_MARGIN, (m_lap["max"], o_lap["max"])
            assert (m_lap["max"][0], m_lap["max"][1], m_lap["count_max"][0]) == MIRROR_MAX[k], m_lap
            _check_fresh_sram(m["sram"], full)
        except AssertionError as err:
            failures.append((k, repr(err)[:400]))
    assert not failures, failures


SELECT_M5A_LOCKED = {640 + 30 * 5 + 5: 5, 640 + 30 * 6 + 5: 6, 1005: 12, 1035: 13, 1065: 14, 1130: 15}
"""Gas steps on each locked entry of the shipped row (5, 6, 12-15; entry k held from 640 + 30 k for 30 frames, 15 to
the timeout: tests/test_mirrors_mame.py PANEL_LOCKED_STEPS for 12-15): each step and the 20 frames after it lie
inside its entry's hold."""
SELECT_M5A_RACES = (4, 7)
"""The entries past the originals the shipped row makes live this phase: each races its course from the select."""
SELECT_M5A_TO = 2500        # the live entries' runs: the select log to here (the race is loaded by then)
SELECT_M5A_SECONDS = 45


@pytest.mark.mame
@pytest.mark.custom_mame
def test_select_m5a(tmp_path, romset_zip, disc):  # noqa: F811
    """Review Focus 5 (Phase 10 Task 5): Phase 10's frozen row (select.M5A_ENTRIES: 0-4, 7 and 8-11 live; 5, 6 and
    12-15 locked, the mirrors 12 and 15 on the set but not on the row) on the final set (`final_set`), custom MAME:

    - the sweep (tests/test_mirrors_mame.py test_select_panels' measures): the wheel steps through the sixteen entries
      (PANEL_HOLD frames each from PANEL_SWEEP_FROM); at each the game holds the entry, its course id and first_of; a
      snapshot per entry to out/phase10/snaps/task5/select-m5a-entryNN-<name>.png; gas steps on each locked entry
      (SELECT_M5A_LOCKED: 5, 6, 12, 13, 14, 15) refuse the start (state 1, the entry held, the timer runs on); the
      timer's end on entry 15 races entry 0 and the vote stores 0;
    - the upload unchanged: the generated chars in char RAM are select.graphics' for M5A_ENTRIES, 127 chars in four runs
      (the same as Task 4's set: test_convert_build.test_final_set_with_mirrors_12_15);
    - the untouched wheel (IDLE_WHEEL, no step): the cursor stays on entry 0 through the select and the timeout races
      Beginner (course id 0, the vote 0);
    - entries 4 and 7 race (SELECT_M5A_RACES): DAYTONA_ENTRY starts each (state 2 at the step on that entry, its course
      id), the vote stores its id and the race is that course (the log's course id at SELECT_M5A_TO); a snapshot of
      each race."""
    from dtools.arcade import tiles as TL
    from tests.test_custom_mame import _custom
    from tests.test_mirrors_mame import (IDLE_SNAPS, IDLE_TO, IDLE_WHEEL, PANEL_HOLD, PANEL_SECONDS, PANEL_SWEEP_FROM,
                                         SELECT_OPEN, _wide_votes, final_set)
    from tests.test_sixteen_mame import _select_run
    exe = str(_custom())
    full = final_set(RomSet.from_zip(romset_zip), disc)
    row = S.select_row(full)
    assert row.live == tuple(e.live for e in S.M5A_ENTRIES)
    assert [k for k in range(16) if not row.live[k]] == sorted(SELECT_M5A_LOCKED.values()) == [5, 6, 12, 13, 14, 15]
    assert G.own_tables(full, 12) and G.own_tables(full, 15)        # on the set, not on the row
    g = S.graphics(full, S.M5A_ENTRIES)
    assert g.uploaded() == 127 and len(g.runs) == 4
    dest = SNAP_DIR / "task5"
    dest.mkdir(parents=True, exist_ok=True)
    frames = [PANEL_SWEEP_FROM + PANEL_HOLD * k for k in range(16)]
    at = [f + PANEL_HOLD - 4 for f in frames]
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{S.paddle_for_entry(k)}" for k, f in enumerate(frames)),
           "DAYTONA_SNAPS": ",".join(map(str, at)), "DAYTONA_STEPS": ",".join(map(str, SELECT_M5A_LOCKED)),
           "DAYTONA_GAS_FROM": "2000", "DAYTONA_SELECT_FROM": "600", "DAYTONA_SELECT_TO": str(IDLE_TO),
           "DAYTONA_SELECT_CHARS": "700"}
    log, _, snap = _select_run(full, tmp_path / "sweep", romset_zip, PANEL_SECONDS, env, exe)
    votes = _wide_votes(tmp_path / "sweep" / "trace.txt")
    shots = sorted(Path(snap).rglob("*.png"))
    for k, p in enumerate(shots[:16]):
        e = S.M5A_ENTRIES[k]
        shutil.copy(p, dest / f"select-m5a-entry{k:02d}-{e.name.lower().replace(' ', '-')}"
                              f"{'-mirror' if e.mirror else ''}{'' if e.live else '-locked'}.png")
    assert len(shots) >= 16
    for k, f in enumerate(at):
        assert log["F"][f][:5] == (1, k, k, S.M5A_ENTRIES[k].course, S.first_of(k)), (k, log["F"][f])
    for step, k in SELECT_M5A_LOCKED.items():
        after = [log["F"][f] for f in range(step, step + 20)]
        assert all(v[0] == 1 and v[1] == k for v in after) and after[-1][5] < after[0][5], (k, after[0], after[-1])
    go = min(f for f, v in log["F"].items() if f > max(SELECT_M5A_LOCKED) and v[0] == 2)
    assert log["F"][go - 1][1] == 15 and log["F"][go][1:4] == (0, 0, 0), (log["F"][go - 1], log["F"][go])
    assert votes and set(votes) == {0}, votes
    assert log["F"][IDLE_TO][3] == 0
    for first, n in g.runs:                                # the upload: the generated chars, as before
        a = 32 * (first - S.CHAR_HOLE)
        assert log["C"][a:a + 32 * n] == TL.encode_chars(g.chars[first - S.CHAR_HOLE:first - S.CHAR_HOLE + n])

    # the untouched wheel: entry 0 throughout, the timeout races Beginner
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{p}" for f, p in IDLE_WHEEL.items()), "DAYTONA_STEPS": "none",
           "DAYTONA_GAS_FROM": "2000", "DAYTONA_SELECT_FROM": "600",
           "DAYTONA_SELECT_TO": str(IDLE_TO), "DAYTONA_SNAPS": ",".join(map(str, IDLE_SNAPS.values()))}
    log, _, snap = _select_run(full, tmp_path / "idle", romset_zip, PANEL_SECONDS, env, exe)
    votes = _wide_votes(tmp_path / "idle" / "trace.txt")
    for p, (label, frame) in zip(sorted(Path(snap).rglob("*.png")), IDLE_SNAPS.items()):
        shutil.copy(p, dest / f"select-m5a-idle-{label}-{frame}.png")
    go = min(f for f, v in log["F"].items() if f > SELECT_OPEN and v[0] == 2)
    state1 = {f: v for f, v in log["F"].items() if SELECT_OPEN <= f < go and v[0] == 1}
    assert len(state1) > 900 and all(v[1:4] == (0, 0, 0) for v in state1.values())
    assert log["F"][go][1:4] == (0, 0, 0) and votes and set(votes) == {0}, votes
    assert log["F"][IDLE_TO][3] == 0

    # entries 4 and 7 race from the select
    for e in SELECT_M5A_RACES:
        wd = tmp_path / f"entry{e}"
        log, _, snap = _select_run(full, wd, romset_zip, SELECT_M5A_SECONDS,
                                   {"DAYTONA_ENTRY": str(e), "DAYTONA_SELECT_FROM": "600",
                                    "DAYTONA_SELECT_TO": str(SELECT_M5A_TO), "DAYTONA_SNAPS": "1681"}, exe)
        votes = _wide_votes(wd / "trace.txt")
        for p in sorted(Path(snap).rglob("*.png"))[:1]:
            shutil.copy(p, dest / f"select-m5a-entry{e:02d}-race-1681.png")
        step = min(f for f, v in log["F"].items() if v[0] == 2)
        k = S.M5A_ENTRIES[e].course
        assert log["F"][step][1:4] == (e, e, k), (e, log["F"][step])
        assert votes and set(votes) == {k}, (e, votes)
        assert log["F"][SELECT_M5A_TO][3] == k, (e, log["F"][SELECT_M5A_TO])


M3_ENTRIES = (9, 10)
"""The Task 3 review's M3 (parked to Task 5): Advanced's and Expert's mirrors (entries 9 and 10, ids 10 and 9) raced on
the final set (they were last raced on Phase 9's mirror set; entry 8 races in test_rankings_per_entry and 11 in
test_mirror_desert_city, on the final chain)."""
M3_MAX = {9: (3633, 1755, 4151), 10: (4134, 1717, 4746)}
"""Per entry, Phase 9 Task 3's measured (max drawn per list, its frame, the largest polygon count) over the run
(docs/spike/2026-11-mirrors-live.md, "The MAME gate: test_mirrors_of_originals")."""


@pytest.mark.mame
@pytest.mark.custom_mame
def test_mirrors_9_10_on_final_set(tmp_path, romset_zip, disc):  # noqa: F811
    """The Task 3 review's M3: entries 9 and 10 (the mirrors of Advanced and Expert, ids 10 and 9) from the select on
    this task's final set (`final_set`: ids 3, 4 and 7 converted, the mirrors of 0-4 and 7, the rankings), against the
    same entries on Phase 9's mirror set (tests/test_mirrors_mame.py `mirrors`: the widened chain with the mirrors of
    0-2 alone), MIRROR_SECONDS each: the same drawn polygons per list on every frame from BARS_FROM (the objects'
    addresses moved with the chain, their content not: test_convert_build.test_final_set_with_mirrors_12_15), the same
    polygon count and visible cells per frame, the same ground calls (every frame, attract included), the same clock,
    the same render-rate writes and video mode per frame, pixel-identical race snapshots (MIRROR_SNAPS); and so on
    the final set the bar holds as Phase 9 measured it: M3_MAX, no cap hit, no stray, nothing left out but
    test_mirrors_mame.LOAD_LEFT_OUT, no list missed but GO + 1."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_convert_mame import BAR, _new_lists
    from tests.test_custom_mame import _custom, _snaps
    from tests.test_mirror import _calls
    from tests.test_mirrors_mame import (BARS_FROM, LOAD_LEFT_OUT, MIRROR_GO, MIRROR_SECONDS, MIRROR_SNAPS, _geo_counts,
                                         _go, final_set, mirrors)
    exe = str(_custom())
    stock = RomSet.from_zip(romset_zip)
    sets = {"phase9": mirrors(stock), "final": final_set(stock, disc)}
    dest = SNAP_DIR / "task5"
    dest.mkdir(parents=True, exist_ok=True)
    for e in M3_ENTRIES:
        k = S.M5A_ENTRIES[e].course
        shots = {"fly-by": 1500, "start": MIRROR_GO[e] + MIRROR_SNAPS["start"], "mid-lap": MIRROR_SNAPS["mid-lap"]}
        got = {}
        for name, s in sets.items():
            assert S.select_row(s).live[e] and S.select_row(s).records[0][e].course == k
            run = mame_build.run_set(s, tmp_path / f"{e}-{name}", MIRROR_SECONDS, romset_zip,
                                     {"DAYTONA_ENTRY": str(e), "DAYTONA_GEO_FROM": str(BARS_FROM),
                                      "DAYTONA_GEO_EVERY": "1", "DAYTONA_CELLS_FROM": str(BARS_FROM),
                                      "DAYTONA_SNAPS": ",".join(map(str, shots.values()))},
                                     logs=("geo", "rate", "ground", "progress", "cells"), mame=exe)
            _, per = _geo_counts(run["geo"], s, BARS_FROM)
            got[name] = dict(per=per, cells=runner.parse_cells(run["cells"]), calls=_calls(run["ground"]),
                             timer=runner.parse_progress(run["progress"])[1], rate=runner.parse_rate(run["rate"]),
                             snaps=_snaps(run["snap"]), shots=sorted(run["snap"].rglob("*.png")))
            if name == "final":
                for p, (label, frame) in zip(got[name]["shots"], shots.items()):
                    shutil.copy(p, dest / f"m3-entry{e:02d}-id{k}-{label}-{frame}.png")
        a, b = got["phase9"], got["final"]
        assert {f: n for f, (n, _) in b["per"].items()} == {f: n for f, (n, _) in a["per"].items()}, e
        assert b["cells"] == a["cells"] and b["calls"] == a["calls"] and b["timer"] == a["timer"], e
        # the rate log's frames, write counts and video mode (its fourth field hashes the display list's words, which
        # name the objects' moved addresses)
        assert [r[:3] for r in b["rate"]] == [r[:3] for r in a["rate"]], e
        assert b["snaps"][:len(shots)] == a["snaps"][:len(shots)], e
        c = G.decode_course(sets["final"], k)
        per, cells, timer = b["per"], b["cells"], b["timer"]
        go = _go(timer)
        race = range(BARS_FROM, max(timer) + 1)
        owner = {p.oba: cell for cell, lst in enumerate(c.cells) for p in c.object_lists[lst]}
        occupied = set(owner.values())
        left_out = {f for f in race if {x for x in cells[f][1] if x in occupied} - {owner[x] for x in per[f][1]
                                                                                   if x in owner}}
        strays = [f for f in race if {owner[x] for x in per[f][1] if x in owner} - set(cells[f][1])]
        top = max(race, key=lambda f: (per[f][0], -f))
        busiest = max(race, key=lambda f: (cells[f][0], -f))
        missed = set(range(go, max(timer) + 1)) - _new_lists(b["rate"])
        report = dict(go=go, max=(per[top][0], top), count_max=(cells[busiest][0], busiest), left_out=sorted(left_out),
                      strays=len(strays), missed=sorted(missed), cap_hits=sum(cells[f][0] >= C.OBC for f in race))
        print(f"\nentry {e} (id {k}) on the final set: {report}")
        assert go == MIRROR_GO[e] and (per[top][0], top, cells[busiest][0]) == M3_MAX[e], report
        assert per[top][0] <= BAR and report["cap_hits"] == 0 and not strays, report
        assert left_out == LOAD_LEFT_OUT[e] and missed <= {go + 1}, report


# --- Phase 11 Task 2 (M5b): the row of eight; the shipped chain builds no mirrors ------------------------------------

M5B_CONVERTED = (3, 4, 7, 6, 5)
"""The courses the shipped chain converts, in order, each at its own id (build.converted_set(..., n, slot=n,
scenery="model")); Task 3 added 6 (Rin Rin Rink, shifted by build.SHIFT[6]), Task 4 added 5 (Circuit Pixie)."""


def m5b_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """The shipped set of Phase 11 (the controller's ruling, plan "the shipped chain builds no mirrors"): grow_ids,
    guard_new_ids, widen_ids, Desert City, National Park Speedway, Mermaid Lake, (Task 3) Rin Rin Rink and (Task 4)
    Circuit Pixie converted at ids 3, 4, 7, 6 and 5 (M5B_CONVERTED; tests/test_mirrors_mame.py `_converted`), a ranking
    table per id (course.grow_rankings), the select with the row of eight (select.M5B_ENTRIES, frozen; all live since Task 4), then
    the per-course views (build.viewed_set: Circuit Pixie's and Rin Rin Rink's, build.VIEW[5] and VIEW[6], both
    build.VIEW_12; every other id reads the shared tables). No mirror step: ids 8-15 hold no course of their own and no
    entry reaches them (they stay free for reverse, M9)."""
    from tests.test_mirrors_mame import _converted
    return build.viewed_set(S.patch_select(C.grow_rankings(_converted(rs, disc, M5B_CONVERTED)), S.M5B_ENTRIES))


P11_SNAP_DIR = config.out_dir() / "phase11" / "snaps" / "task2"
ROW8 = {"DAYTONA_ROW": "8"}         # autoplay.lua: DAYTONA_ENTRY picks an entry of the row of eight


@pytest.mark.mame
@pytest.mark.custom_mame
def test_originals_on_m5b_set(tmp_path, romset_zip, disc):  # noqa: F811
    """The Global Constraint on the shipped set (`m5b_set`: no mirror, the row of eight): entries 0-2 (Beginner,
    Advanced, Expert) from its select race as the stock set's panels 0-2 on stock MAME, as on Phase 10's final set
    (tests/test_mirrors_mame.py test_originals_on_final_set's measures, `originals_as_stock`): the same geometry
    display list on every frame from RACE_FROM, the same render-rate writes and video mode, pixel-identical race
    snapshots. Phase 10's final set races them so too, so the shipped set races them call for call as it does."""
    from tests.test_mirrors_mame import originals_as_stock
    originals_as_stock(tmp_path, romset_zip, m5b_set(RomSet.from_zip(romset_zip), disc), ROW8)


DESERT_ATTRACT = (2,)
"""The session's TGP slot loads after Desert City's game (measured): its time-over at 4850 ends the game early, so its
attract (slot 3, at 5936) gives way to the next attract race, Expert's slot 2 at 9103, before the second coin
(test_mirrors_mame.SESSION_COIN2, 9300); National Park Speedway's and Mermaid Lake's games end too late for that."""


@pytest.mark.mame
@pytest.mark.custom_mame
def test_desert_city_id3_m5b(tmp_path, romset_zip, disc):  # noqa: F811
    """Desert City's Phase 8 gate on the shipped set (`m5b_set`, entry 3 of the row of eight), through `_gate`
    (test_national_park_id4's checks) with Desert City's frozen numbers (tests/test_convert_mame.py ID3_*: the long-time
    run's ground calls per car CALLS_ID3_LONG_TIME, the maximum 3,871 drawn at 6287 and the count 4,568 over the lap
    ID3_GO..ID3_LAP_END, a new list every frame but ID3_REPEAT, fresh SRAM; to the time-over CALLS_ID3_TO_TIME_OVER at
    ID3_TIME_OVER with no rise; then Beginner as the session's second game against Beginner then Beginner). Snapshots
    to out/phase11/snaps/task2/gates."""
    _gate(tmp_path, romset_zip, m5b_set(RomSet.from_zip(romset_zip), disc), desert_gate(str(P11_SNAP_DIR / "gates")))


def desert_gate(dest: str) -> Gate:
    """Desert City's gate on the row of eight (test_desert_city_id3_m5b: tests/test_convert_mame.py ID3_*), its
    snapshots to `dest`."""
    from tests.test_convert_mame import (CALLS_ID3_LONG_TIME, CALLS_ID3_TO_TIME_OVER, ID3_COUNT_MAX, ID3_GO,
                                         ID3_LAP_END, ID3_MAX, ID3_MAX_AT, ID3_REPEAT, ID3_SECONDS, ID3_TIME_OVER,
                                         LONG_LAPS)
    return Gate(course=3, entry=3, name="desert-city", label="Desert City", dest=dest,
                seconds=ID3_SECONDS, go=ID3_GO, repeat=ID3_REPEAT, lap_end=ID3_LAP_END, long_laps=LONG_LAPS,
                maximum=(ID3_MAX, ID3_MAX_AT, ID3_COUNT_MAX), calls_long_time=CALLS_ID3_LONG_TIME,
                calls_to_time_over=CALLS_ID3_TO_TIME_OVER, time_over=ID3_TIME_OVER, rises=0,
                snaps={"select": 750, "start": ID3_GO + 30, "mid-lap": 5600}, attract=DESERT_ATTRACT)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_national_park_id4_m5b(tmp_path, romset_zip, disc):  # noqa: F811
    """test_national_park_id4's gate (PARK_*, every number as on Phase 10's final set) on the shipped set (`m5b_set`,
    entry 4 of the row of eight). Snapshots to out/phase11/snaps/task2/gates."""
    _gate(tmp_path, romset_zip, m5b_set(RomSet.from_zip(romset_zip), disc), park_gate(str(P11_SNAP_DIR / "gates")))


@pytest.mark.mame
@pytest.mark.custom_mame
def test_mermaid_lake_id7_m5b(tmp_path, romset_zip, disc):  # noqa: F811
    """test_mermaid_lake_id7's gate (LAKE_*, the autopilot's lap) on the shipped set (`m5b_set`, entry 7 of the row of
    eight). Snapshots to out/phase11/snaps/task2/gates."""
    _gate(tmp_path, romset_zip, m5b_set(RomSet.from_zip(romset_zip), disc), lake_gate(str(P11_SNAP_DIR / "gates")))


# --- Phase 11 Task 3: Rin Rin Rink (Dreamcast n 6) at course id 6, its own view ----------------------------------

RINK_ENTRY = 6                  # select.ENTRIES[6]: Rin Rin Rink, course id 6
RINK_GO = 1651
"""The race's first frame: Expert's standing countdown, as the other converted courses'."""
RINK_REPEAT = frozenset({RINK_GO + 1})
RINK_SECONDS = 169              # to about frame 9720: past GO + 8000, every AI car on lap 2, none on lap 3 (9831)
RINK_LAP_END = 7611             # the player (wheel centred) reaches the lap line, lap 1
RINK_LONG_LAPS = 2
RINK_MAX, RINK_MAX_AT, RINK_COUNT_MAX = 3788, 2067, 4666
"""The drawn-polygon maximum per list over the player's lap and its frame (segment 23), and the largest polygon
count: with QUAD[6], REMESH[6], NAMED_PROPS[6], THIN[6] and the course's own view (build.VIEW[6]), within
BAR - THIN_MARGIN and course.OBC - THIN_MARGIN (set before measuring). v0.1.1 (build.shadow_volumes takes
kind-3 volumes: CS6's 149:0 and 83:1-4 dropped): 3,816 and 4,694 before, 28 fewer drawn at the same frame and 28 fewer
counted, measured; the rest unchanged."""
CALLS_RINK_TO_TIME_OVER = {0: 16247, 1: 2784, 2: 715, 3: 477}
CALLS_RINK_LONG_TIME = {0: 35863, 1: 2950, 2: 1664, 3: 789, 4: 971, 5: 465, 6: 722, 7: 939, 8: 150, 10: 869, 11: 2266,
                        12: 868, 13: 401, 14: 1655, 15: 462, 16: 1698, 17: 482, 18: 1096, 19: 833, 20: 145, 21: 496,
                        22: 116, 23: 765, 24: 4, 25: 478, 26: 282, 27: 22, 28: 189, 29: 106, 30: 296, 31: 174, 32: 83,
                        33: 176, 34: 142, 35: 95, 36: 261, 37: 273, 38: 108, 39: 29}
RINK_TIME_OVER = 4850
"""The clock out on the set without the long time (the paddle centred): Beginner's start time, no checkpoint reached
(the one checkpoint is at segment 263; RINK_RISES 0), as Desert City's; its attract then gives way to Expert's
(DESERT_ATTRACT)."""
RINK_RISES = 0
RINK_SNAPS = dict(sorted({"select": 750, "start": RINK_GO + 30, "mid-lap": 4500, "heaviest": RINK_MAX_AT}.items(),
                         key=lambda kv: kv[1]))
"""The snapshots: the select with entry 6 under the cursor, the start, mid-lap, the heaviest frame, in frame order
(`_gate` names the captures in the dict's order)."""


@pytest.mark.mame
@pytest.mark.custom_mame
def test_rin_rin_rink_id6(tmp_path, romset_zip, disc):  # noqa: F811
    """Rin Rin Rink (n 6) at its own course id 6 on the shipped set (`m5b_set`, entry 6 of the row of eight), entered
    from the select, custom MAME: test_national_park_id4's checks (`_gate`) with RINK_* in place of PARK_*:

    - the long-time set, RINK_SECONDS, the paddle centred: the race checks and the frame-rate bar over the player's lap
      RINK_GO..RINK_LAP_END (every one of the 500 segments, every frame counted): at most BAR - THIN_MARGIN drawn and
      OBC - THIN_MARGIN counted, no cap hit, no visible cell left out, none drawn outside; the maximum frozen;
      snapshots (RINK_SNAPS) to out/phase11/snaps/task3; backup SRAM fresh after the run;
    - the set without the long time: id 6 to its time-over, then Beginner as the session's second game, against
      Beginner then Beginner (Phase 9's relaxed check);
    - the course is shifted (build.SHIFT[6]: every ground call a polygon on the shifted ground) and reads its own view
      (build.VIEW[6], course.view_ids): on every frame of the lap the cells the game opens (its own cells tap; the
      first is the reference cell, the block's centre) lie in VIEW[6]'s cells round it for one heading quadrant (fix
      round 1, M4; the stock view's run opens cells outside them on every frame)."""
    assert build.SHIFT.get(6) == -64.0 and build.QUAD.get(6) == (10.0, 0.125) and build.REMESH.get(6) == 0.1
    assert build.THIN.get(6) == {136: 0.5} and 6 in build.VIEW and build.NAMED_PROPS.get(6)[:3] == (2, 3, 4)
    full = m5b_set(RomSet.from_zip(romset_zip), disc)
    assert C.viewed(full) and C.view_tables(full, 6) != C.view_tables(full, 0)
    seen = _gate(tmp_path, romset_zip, full, rink_gate(str(config.out_dir() / "phase11" / "snaps" / "task3")))
    # the opened-cells check takes the first cell in draw order (cells_log.lua's order) as the reference cell
    outside = outside_view(seen, build.VIEW[6])
    assert not outside, outside[:8]


def rink_gate(dest: str) -> Gate:
    """Rin Rin Rink's gate (test_rin_rin_rink_id6: RINK_*), its snapshots to `dest`."""
    return Gate(course=6, entry=RINK_ENTRY, name="rin-rin-rink", label="Rin Rin Rink", dest=dest,
                seconds=RINK_SECONDS, go=RINK_GO, repeat=RINK_REPEAT, lap_end=RINK_LAP_END, long_laps=RINK_LONG_LAPS,
                maximum=(RINK_MAX, RINK_MAX_AT, RINK_COUNT_MAX), calls_long_time=CALLS_RINK_LONG_TIME,
                calls_to_time_over=CALLS_RINK_TO_TIME_OVER, time_over=RINK_TIME_OVER, rises=RINK_RISES,
                snaps=RINK_SNAPS, attract=DESERT_ATTRACT)


def outside_view(seen: dict, view) -> list[int]:
    """The frames of `seen["lap"]` (`_gate`'s result) on which the cells the game opens (its cells tap: the first is the
    reference cell, the block's centre) do not all lie in the view `view`'s cells ((d, c) as build.VIEW names them)
    round the reference cell for one heading quadrant (build.view_tables' rotation)."""
    turn = (lambda d, c: (d, c), lambda d, c: (c, -d), lambda d, c: (-d, c), lambda d, c: (c, d))
    keep = [{turn[q](d, c) for d, c in view} for q in range(4)]
    return [f for f in seen["lap"] if seen["cells"][f][1] and not any(
        {((x >> 4) - (seen["cells"][f][1][0] >> 4), (x & 15) - (seen["cells"][f][1][0] & 15))
         for x in seen["cells"][f][1]} <= k for k in keep)]


# --- Phase 11 Task 4: Circuit Pixie (Dreamcast n 5) at course id 5, the 12-cell view ---------------------------------

PIXIE_ENTRY = 5                 # select.ENTRIES[5]: Circuit Pixie, course id 5
PIXIE_GO = 1651
"""The race's first frame: Expert's standing countdown, as the other converted courses'."""
PIXIE_REPEAT = frozenset({PIXIE_GO + 1})
PIXIE_SECONDS = 171             # to about frame 9836: past GO + 8000
PIXIE_LAP_END = 4627            # the player (wheel centred) reaches the lap line, lap 1
PIXIE_LONG_LAPS = (4, 5, 6)
"""The laps each AI car completes over the long-time run: on the short oval (129 segments, an AI lap of 1,500-1,740
frames) the field spreads over more than a lap before GO + 8000, so no frame past it has every car on one lap
(measured: 9651 has cars on laps 4 and 5, the first on 6 at 9783); any of these three (`_drive_checks` with a tuple)."""
PIXIE_MAX, PIXIE_MAX_AT, PIXIE_COUNT_MAX = 3723, 1805, 4521
"""The drawn-polygon maximum per list over the player's lap and its frame, and the largest polygon count: with
NAMED_PROPS[5], QUAD[5], REMESH[5] and the course's own view (build.VIEW[5], VIEW_12), within BAR - THIN_MARGIN and
course.OBC - THIN_MARGIN (set before measuring)."""
CALLS_PIXIE_TO_TIME_OVER = {0: 11665, 1: 974, 2: 288, 3: 457, 4: 16, 5: 485, 6: 728, 7: 88, 10: 586, 11: 1133, 12: 1198,
                            13: 727, 14: 548, 15: 455, 16: 448, 17: 369, 18: 895, 19: 762, 20: 263, 21: 304, 22: 427,
                            23: 675, 24: 149, 25: 1507, 26: 1655, 27: 138, 28: 549, 29: 293, 30: 168, 31: 268, 32: 602,
                            33: 668, 34: 702, 35: 1202, 36: 225, 37: 218, 38: 4497, 39: 2544}
CALLS_PIXIE_LONG_TIME = {0: 21226, 1: 1295, 2: 982, 3: 1150, 4: 809, 5: 896, 6: 1466, 7: 857, 8: 1237, 9: 235, 10: 711,
                         11: 1862, 12: 2363, 13: 1230, 14: 1068, 15: 2364, 16: 685, 17: 1202, 18: 1413, 19: 1845,
                         20: 573, 21: 407, 22: 847, 23: 867, 24: 713, 25: 2790, 26: 2458, 27: 404, 28: 844, 29: 552,
                         30: 171, 31: 761, 32: 900, 33: 1716, 34: 1391, 35: 3277, 36: 857, 37: 557, 38: 5768, 39: 3275}
PIXIE_TIME_OVER = 6051
"""The clock out on the set without the long time (the paddle centred): Beginner's start time, extended once when the
player reaches the lap line (frame 4627; the lap line is the course's one checkpoint: PIXIE_RISES 1); the game ends
late enough that its own attract (slot 5, at 7137) still runs at the second coin (no other slot loads)."""
PIXIE_RISES = 1
PIXIE_SNAPS = dict(sorted({"select": 750, "start": PIXIE_GO + 30, "heaviest": PIXIE_MAX_AT,
                           "mid-lap": 3200}.items(), key=lambda kv: kv[1]))
"""The snapshots: the select with entry 5 under the cursor (all eight entries live), the start, the heaviest frame
(the grid, the field bunched), mid-lap, in frame order (`_gate` names the captures in the dict's order)."""


@pytest.mark.mame
@pytest.mark.custom_mame
def test_circuit_pixie_id5(tmp_path, romset_zip, disc):  # noqa: F811
    """Circuit Pixie (n 5) at its own course id 5 on the shipped set (`m5b_set`, entry 5 of the row of eight, all eight
    live), entered from the select, custom MAME: test_national_park_id4's checks (`_gate`) with PIXIE_* in place of
    PARK_*:

    - the long-time set, PIXIE_SECONDS, the paddle centred: the race checks and the frame-rate bar over the player's
      lap PIXIE_GO..PIXIE_LAP_END (every one of the 130 segments, every frame counted): at most BAR - THIN_MARGIN drawn
      and OBC - THIN_MARGIN counted, no cap hit, no visible cell left out, none drawn outside; the maximum frozen;
      snapshots (PIXIE_SNAPS) to out/phase11/snaps/task4; backup SRAM fresh after the run;
    - the set without the long time: id 5 to its time-over, then Beginner as the session's second game, against
      Beginner then Beginner (Phase 9's relaxed check);
    - the course reads its own view (build.VIEW[5], Pascal's ruling 2026-11-12: option A, Rin Rin Rink's 12-cell shape,
      all scenery kept): on every frame of the lap the cells the game opens lie in VIEW[5]'s cells round the reference
      cell for one heading quadrant (`outside_view`)."""
    assert build.QUAD.get(5) == (10.0, 0.125) and build.REMESH.get(5) == 0.1 and 5 not in build.THIN
    assert build.VIEW.get(5) == build.VIEW_12 and build.NAMED_PROPS.get(5)[:1] == (3,)
    assert 117 not in build.NAMED_PROPS[5]                  # Pascal's ruling: all scenery kept, the pit row's too
    full = m5b_set(RomSet.from_zip(romset_zip), disc)
    assert C.viewed(full) and C.view_tables(full, 5) != C.view_tables(full, 0)
    seen = _gate(tmp_path, romset_zip, full, pixie_gate(str(config.out_dir() / "phase11" / "snaps" / "task4")))
    outside = outside_view(seen, build.VIEW[5])
    assert not outside, outside[:8]


def pixie_gate(dest: str) -> Gate:
    """Circuit Pixie's gate (test_circuit_pixie_id5: PIXIE_*), its snapshots to `dest`."""
    return Gate(course=5, entry=PIXIE_ENTRY, name="circuit-pixie", label="Circuit Pixie", dest=dest,
                seconds=PIXIE_SECONDS, go=PIXIE_GO, repeat=PIXIE_REPEAT, lap_end=PIXIE_LAP_END,
                long_laps=PIXIE_LONG_LAPS, maximum=(PIXIE_MAX, PIXIE_MAX_AT, PIXIE_COUNT_MAX),
                calls_long_time=CALLS_PIXIE_LONG_TIME, calls_to_time_over=CALLS_PIXIE_TO_TIME_OVER,
                time_over=PIXIE_TIME_OVER, rises=PIXIE_RISES, snaps=PIXIE_SNAPS)


M5B_SWEEP_UP = [640 + 30 * k for k in range(8)]
"""The sweep to the right: entry k held from 640 + 30 k (select.paddle_for_entry(k, 8)), 30 frames each."""
M5B_STOP_RIGHT = 880                # paddle 255 (the wheel hard right) for 30 frames: entry 7 still
M5B_SWEEP_DOWN = [910 + 30 * j for j in range(8)]
"""The sweep back: entry 7 - j held from 910 + 30 j."""
M5B_STOP_LEFT = 1150                # paddle 0 for 30 frames: entry 0 still
M5B_TIMEOUT_ENTRY, M5B_TIMEOUT_FROM = 5, 1180
"""The wheel then holds entry 5 (Circuit Pixie) to the select's timeout (about 1580). Since Task 4 every entry is live,
so the timeout races the entry held (Circuit Pixie, the vote 5); while entry 5 was locked (Tasks 2 and 3) the
timeout on it raced Beginner (Review Focus 1: a locked entry's timeout gives way to entry 0, still checked on the frozen
row of 16, test_select_m5a)."""
M5B_RACES = tuple(range(8))
"""The live entries of the row of eight (all eight since Task 4): DAYTONA_ENTRY starts each, its course."""
M5B_RACE_TO, M5B_RACE_SECONDS = 2500, 45
M5B_LAP_SECONDS = 205
"""Desert City's one-lap race (test_mirrors_mame.one_lap of id 3, on build.long_time, so the clock does not run out
before the player's lap), booted on the SRAM the shipped set's first boot left: the goal at frame 10805 (the race
time 9,155; from a fresh NVRAM the same race reaches it at 10181, ID3_LAP_END, 8,531: a different SRAM, a different
race), the record 280 frames later (0x227408 reads the goal flag at 11085; on Beginner the record follows the goal by
20 frames), measured in probe runs (discovery log, "The row of eight")."""
M5B_DESERT_RECORD = 9155
"""The race time (0x5010cc, ticks) the one-lap Desert City race records: measured by test_select_m5b, frozen."""


BLINK_FRAMES = tuple(M5B_SWEEP_UP[0] + d for d in range(22, 30))
"""Eight consecutive frames of the hold on entry 0 (before the next hold at 670): the cursor's blink (the blink map
when 0x540058 & 7 > 4, docs/formats/model2-course-tables.md "Select screen") shows in some and not in others."""


def _row_checks(tmp_path, romset_zip, full: RomSet, entries, dest: Path, prefix: str = "select") -> None:
    """test_select_m5b's checks of a row of 8 (`entries`, all live) on `full`, custom MAME, autoplay.lua with
    DAYTONA_ROW 8 (its rankings excepted):

    - the sweep (select_log.lua): the wheel steps through the eight entries to the right (M5B_SWEEP_UP), is held hard
      right (M5B_STOP_RIGHT: entry 7, nothing past it), steps back through the eight to the left (M5B_SWEEP_DOWN) and
      is held hard left (M5B_STOP_LEFT: entry 0); at every hold the game holds the entry (0x501468, 0x5e(g13)), its
      course id (0x501460) and first_of(k, 8) (0x501b24); at each hold the three positions show, in tile layer 0, one
      of their entries' record maps (the original's own for an original, any mode, variant, normal or blink) and, in
      layer 1, their window maps (select.graphics), and the window mask covers each position's window rows only
      (tests/test_sixteen_mame.py `_mask_rows`: an original's name rows under WINDOW_NAME, never its label rows); the
      generated chars in char RAM are select.graphics' for the row; on entry 0 the cursor's blink (BLINK_FRAMES)
      shows both phases, layer 1 unchanged; a snapshot per entry to `dest`/<prefix>-entryNN-<name>.png, entry 0's
      blink phases to <prefix>-entry00-<name>-blink-off.png / -blink-on.png, and of both stops;
    - the timeout on entry M5B_TIMEOUT_ENTRY races its course (the vote that id); the untouched wheel (IDLE_WHEEL,
      inside the centre band, no step): the cursor on entry 0 through the select, the timeout races entry 0's course;
    - each entry starts its course from the select (state 2 at the step on that entry, its course id, the vote that
      id, the race that course at M5B_RACE_TO); a snapshot of each race."""
    from dtools.arcade import tiles as TL
    from tests.test_custom_mame import _custom
    from tests.test_mirrors_mame import IDLE_SNAPS, IDLE_TO, IDLE_WHEEL, PANEL_SECONDS, SELECT_OPEN, _wide_votes
    from tests.test_sixteen_mame import _mask_rows, _select_run
    exe = str(_custom())
    row = S.select_row(full)
    assert len(row.live) == len(entries) == 8 and row.live == tuple(e.live for e in entries) and all(row.live)
    assert [r.course for r in row.records[0]] == [e.course for e in entries]
    g = S.graphics(full, entries)
    tiles = {S.slot_of(k, 8): r.tile for k, r in enumerate(row.records[0])}
    dest.mkdir(parents=True, exist_ok=True)

    def slug(k: int) -> str:
        return f"{prefix}-entry{k:02d}-{entries[k].name.lower().replace(' ', '-')}"

    # the sweep each way, the stops, the timeout on entry M5B_TIMEOUT_ENTRY
    holds = ([(f, k, S.paddle_for_entry(k, 8)) for k, f in enumerate(M5B_SWEEP_UP)] + [(M5B_STOP_RIGHT, 7, 255)] +
             [(f, 7 - j, S.paddle_for_entry(7 - j, 8)) for j, f in enumerate(M5B_SWEEP_DOWN)] +
             [(M5B_STOP_LEFT, 0, 0), (M5B_TIMEOUT_FROM, M5B_TIMEOUT_ENTRY, S.paddle_for_entry(M5B_TIMEOUT_ENTRY, 8))])
    at = [f + 26 for f, _, _ in holds]
    snaps = sorted(set(at) | set(BLINK_FRAMES))
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{p}" for f, _, p in holds),
           "DAYTONA_SNAPS": ",".join(map(str, snaps)), "DAYTONA_STEPS": "none", "DAYTONA_GAS_FROM": "2000", "DAYTONA_SELECT_TILES": ",".join(map(str, snaps)),
           "DAYTONA_SELECT_FROM": "600", "DAYTONA_SELECT_TO": str(IDLE_TO), "DAYTONA_SELECT_CHARS": "700", **ROW8}
    log, _, snap = _select_run(full, tmp_path / "sweep", romset_zip, PANEL_SECONDS, env, exe)
    votes = _wide_votes(tmp_path / "sweep" / "trace.txt")
    pngs = sorted(Path(snap).rglob("*.png"))
    # MAME also writes one snapshot when the run ends (autoplay.lua), sorted last: exactly one beyond the frames asked
    assert len(pngs) == len(snaps) + 1, (len(pngs), len(snaps))   # any other count would misalign the pairing
    shots = dict(zip(snaps, pngs))
    for k in range(8):
        shutil.copy(shots[at[k]], dest / f"{slug(k)}.png")
    for label, i in (("stop-right", 8), ("stop-left", 17), ("timeout-hold", 18)):
        shutil.copy(shots[at[i]], dest / f"{prefix}-{label}-{at[i]}.png")
    seen = {f: log["F"][f][:5] for f in at}
    print(f"\nsweep: {[(f, k, seen[a]) for (f, k, _), a in zip(holds, at)]}")
    blank = (S.LAYER1_BLANK,) * (C.PANEL_ROWS * C.PANEL_COLS)

    def shown(f: int, s: int, layer: int) -> tuple[int, ...]:
        x, y = tiles[s]
        return tuple(log["T"][(f, layer, y + r)][x + c] for r in range(C.PANEL_ROWS) for c in range(C.PANEL_COLS))
    for (f, k, _), a in zip(holds, at):
        assert seen[a] == (1, k, k, entries[k].course, S.first_of(k, 8)), (f, k, seen[a])
        first = S.first_of(k, 8)
        for s in range(3):
            e = first + s
            assert shown(a, s, 0) in {TL.tile_map(full, m).tiles() for recs in row.records for m in recs[e].maps}, \
                (k, s)
            assert shown(a, s, 1) == (g.name_maps[e] or blank), (k, s)
        name, label = _mask_rows(g.windows, first, tiles)
        assert log["M"][a] == [tuple(name)] * 40 + [tuple(label)] * 24, (k, log["M"][a][0], log["M"][a][-1])
    # the cursor's blink on entry 0 (position 0): both phases, the window unchanged
    normal = {TL.tile_map(full, m).tiles() for recs in row.records for m in recs[0].maps[0::2]}
    blinked = {TL.tile_map(full, m).tiles() for recs in row.records for m in recs[0].maps[1::2]}
    phase = {f: shown(f, 0, 0) in blinked for f in BLINK_FRAMES}
    assert all(shown(f, 0, 0) in normal | blinked for f in BLINK_FRAMES)
    assert set(phase.values()) == {False, True}, phase
    assert all(shown(f, 0, 1) == (g.name_maps[0] or blank) for f in BLINK_FRAMES)
    for on in (False, True):
        f = min(f for f in BLINK_FRAMES if phase[f] == on)
        shutil.copy(shots[f], dest / f"{slug(0)}-blink-{'on' if on else 'off'}.png")
    entries_seen = [log["F"][f][1] for f in range(M5B_SWEEP_UP[0] + 26, M5B_STOP_LEFT + 30)]
    assert set(entries_seen) == set(range(8)) and max(entries_seen) == 7          # nothing past the row
    assert all(log["F"][f][3] < 8 for f in log["F"] if SELECT_OPEN <= f < M5B_TIMEOUT_FROM + 300)
    go = min(f for f, v in log["F"].items() if f > M5B_TIMEOUT_FROM and v[0] == 2)
    kt = entries[M5B_TIMEOUT_ENTRY].course
    assert log["F"][go - 1][1] == M5B_TIMEOUT_ENTRY and log["F"][go][1:4] == (M5B_TIMEOUT_ENTRY, M5B_TIMEOUT_ENTRY,
                                                                              kt), (log["F"][go - 1], log["F"][go])
    assert votes and set(votes) == {kt}, votes
    assert log["F"][IDLE_TO][3] == kt                                      # the race is the held entry's course
    for first, n in g.runs:                                # the upload: the row's generated chars
        a = 32 * (first - S.CHAR_HOLE)
        assert log["C"][a:a + 32 * n] == TL.encode_chars(g.chars[first - S.CHAR_HOLE:first - S.CHAR_HOLE + n])

    # the untouched wheel: entry 0 throughout, the timeout races its course
    k0 = entries[0].course
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{p}" for f, p in IDLE_WHEEL.items()), "DAYTONA_STEPS": "none",
           "DAYTONA_GAS_FROM": "2000", "DAYTONA_SELECT_FROM": "600", "DAYTONA_SELECT_TO": str(IDLE_TO),
           "DAYTONA_SNAPS": ",".join(map(str, IDLE_SNAPS.values())), **ROW8}
    log, _, snap = _select_run(full, tmp_path / "idle", romset_zip, PANEL_SECONDS, env, exe)
    votes = _wide_votes(tmp_path / "idle" / "trace.txt")
    for p, (label, frame) in zip(sorted(Path(snap).rglob("*.png")), IDLE_SNAPS.items()):
        shutil.copy(p, dest / f"{prefix}-idle-{label}-{frame}.png")
    go = min(f for f, v in log["F"].items() if f > SELECT_OPEN and v[0] == 2)
    state1 = {f: v for f, v in log["F"].items() if SELECT_OPEN <= f < go and v[0] == 1}
    assert len(state1) > 900 and all(v[1:4] == (0, 0, k0) for v in state1.values())
    assert log["F"][go][1:4] == (0, 0, k0) and votes and set(votes) == {k0}, votes
    assert log["F"][IDLE_TO][3] == k0

    # each entry races its course from the select
    for e in M5B_RACES:
        wd = tmp_path / f"entry{e}"
        log, _, snap = _select_run(full, wd, romset_zip, M5B_RACE_SECONDS,
                                   {"DAYTONA_ENTRY": str(e), "DAYTONA_SELECT_FROM": "600",
                                    "DAYTONA_SELECT_TO": str(M5B_RACE_TO), "DAYTONA_SNAPS": "1681", **ROW8}, exe)
        votes = _wide_votes(wd / "trace.txt")
        for p in sorted(Path(snap).rglob("*.png"))[:1]:
            shutil.copy(p, dest / f"{prefix}-entry{e:02d}-race-1681.png")
        step = min(f for f, v in log["F"].items() if v[0] == 2)
        k = entries[e].course
        print(f"entry {e}: step at {step} {log['F'][step]}, votes {sorted(set(votes))}, "
              f"course at {M5B_RACE_TO} {log['F'][M5B_RACE_TO][3]}")
        assert log["F"][step][1:4] == (e, e, k), (e, log["F"][step])
        assert votes and set(votes) == {k}, (e, votes)
        assert log["F"][M5B_RACE_TO][3] == k, (e, log["F"][M5B_RACE_TO])


@pytest.mark.mame
@pytest.mark.custom_mame
def test_select_m5b(tmp_path, romset_zip, disc):  # noqa: F811
    """Review Focus 1-3 of Phase 11 on the shipped set (`m5b_set`: the row of eight, select.M5B_ENTRIES: all eight live
    (6 since Task 3, 5 since Task 4); no mirror on the set), custom MAME, autoplay.lua with DAYTONA_ROW 8:

    - Review Focus 1 and 2 (`_row_checks`, shared with Phase 12's release row since Phase 12 Task 2): the sweep each
      way and the stops, the tiles and the window mask at every hold, the generated chars, the cursor's blink, the
      timeout on entry 5 (Circuit Pixie, the vote 5), the untouched wheel (the timeout races Beginner), each entry's
      start; snapshots to out/phase11/snaps/task2/select-m5b-*.png;
    - Review Focus 3: the rankings' signature reset wipes a stock NVRAM once: a stock-booted SRAM (stock MAME: the
      stock signature and tables) boots the shipped set to course.SIGNATURE_RANKED and sixteen fresh tables (no
      newest entry); a one-lap Desert City race (entry 3) from that SRAM, every SRAM write traced, writes after the boot
      only ranking table 3 and its newest-entry word (RANKING_WRITER_IPS), no per-id counter, and the record on top;
      a boot on the SRAM it left keeps the record (no second reset)."""
    from tests.test_mirrors_mame import (BOOT_SECONDS, _at, _recorded, _word, check_own_ranking_writes, one_lap,
                                         ranking_run, ranking_tables, traced_ranking_run)
    full = m5b_set(RomSet.from_zip(romset_zip), disc)
    assert [k for k in range(3, C.COURSE_IDS) if G.own_tables(full, k)] == [3, 4, 5, 6, 7]  # no mirror
    assert M5B_RACES == tuple(range(8))
    _row_checks(tmp_path, romset_zip, full, S.M5B_ENTRIES, P11_SNAP_DIR, "select-m5b")

    # Review Focus 3: the signature reset wipes a stock NVRAM once; a lap of Desert City writes table 3 only
    lay = C.rankings_layout(full)
    fresh = [C.fresh_ranking(full, k) for k in range(C.COURSE_IDS)]
    stock_sram = ranking_run(tmp_path, romset_zip, "stock-boot", None, {}, BOOT_SECONDS, mame="mame")
    assert _at(stock_sram, 0x01D00200, 24) == C.SIGNATURE_STOCK
    booted = ranking_run(tmp_path, romset_zip, "m5b-boot", full, ROW8, BOOT_SECONDS, seed="stock-boot")
    assert _at(booted, 0x01D00200, 24) == C.SIGNATURE_RANKED and ranking_tables(booted, lay) == fresh
    assert {_word(booted, a) for a in lay.last_entry} == {0xFFFFFFFF}
    lap_set = one_lap(build.long_time(full, G.decode_course(full, 3)), (3,))
    lapped, log = traced_ranking_run(tmp_path, romset_zip, "desert-lap", lap_set, {"DAYTONA_ENTRY": "3", **ROW8},
                                     M5B_LAP_SECONDS, seed="m5b-boot")
    t = ranking_tables(lapped, lay)
    print(f"Desert City's one-lap record: {struct.unpack_from('<6I', t[3], 0)}; tables changed "
          f"{[k for k in range(C.COURSE_IDS) if t[k] != fresh[k]]}")
    assert [k for k in range(C.COURSE_IDS) if t[k] != fresh[k]] == [3]
    assert _recorded(t[3], fresh[3], M5B_DESERT_RECORD)
    assert [_word(lapped, a) for a in lay.last_entry] == [lay.tables[3] if k == 3 else 0xFFFFFFFF
                                                          for k in range(C.COURSE_IDS)]
    check_own_ranking_writes(log, lay, 3)         # the system area, table 3 and its newest-entry word only
    again = ranking_run(tmp_path, romset_zip, "m5b-reboot", full, ROW8, BOOT_SECONDS, seed="desert-lap")
    assert _at(again, 0x01D00200, 24) == C.SIGNATURE_RANKED and ranking_tables(again, lay) == t    # no second reset
