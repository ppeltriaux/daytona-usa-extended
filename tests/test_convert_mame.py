"""MAME tests of the converted set (dtools/convert/build.py): the Dreamcast's courses raced in the arcade.

The known answer first: the Dreamcast's Beginner (n 0), converted into Beginner's slot with Beginner's own scenery
(`scenery="keep"`), against the original set. docs/spike/2026-11-convert.md, "Beginner from the Dreamcast". Then
Desert City (n 3) in Beginner's slot with no scenery: "Desert City in MAME". Then the model scenery (Phase 7): the
object-library bound patch draws the originals unchanged, and one quad of real scenery is drawn where it is placed:
"Object library"."""

import collections
import dataclasses
import hashlib
import math
import shutil
import struct

import pytest

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade.romset import RomSet
from dtools.arcade.tablebase import DATA_BASE
from dtools.convert import build
from dtools.dc.data import decode_data
from dtools.dc.disc import course_files
from dtools.dc.exe import EXE_NAME
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

GO = 1890                   # Beginner's race starts (green light) at frame 1890 of the autoplay
GO_20 = GO + 1150           # GO + 20 s at 57.52 Hz: the span the converted race is held to
SECONDS = 60                # to frame 3451, past GO_20 (45 s ends at frame 2588)
# Frozen from the discovery runs (docs/spike/2026-11-convert.md, "Beginner from the Dreamcast"): the converted set's
# ground calls equal the original's call for call to frame 1879 and its render-rate log frame for frame to 1457;
# the first difference is an AI car's position (1e-4 in the logged x at 1880) from the AI speed, which the Dreamcast
# stores within 5.73e-6 of the arcade's but not f32-equal (no rule gives the arcade's from it). With the arcade's
# speeds put back the converted set races identically for the whole run.
IDENTICAL_TO = 1879
RATE_IDENTICAL_TO = 1457
# At GO_20 every car is on its original lap and within SEGMENT_SPREAD road segments of its original place. 2 is the
# exact measured maximum (car 0, the player: segment 135 against 133), not a tolerance.
SEGMENT_SPREAD = 2


def _speeds_of(lines, ref):
    """`lines` with each AI point's speed taken from `ref` (the same shape); every other field kept."""
    return tuple(tuple(p._replace(speed=q.speed) for p, q in zip(lp, lq)) for lp, lq in zip(lines, ref))


def _run(rs, path, romset_zip):
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    out = mame_build.run_set(rs, path, SECONDS, romset_zip, {"DAYTONA_COURSE": "0"},
                             logs=("ground", "rate", "progress"))
    samples, _ = runner.parse_ground(out["ground"])
    runner.require_calls(samples)
    rows, timer = runner.parse_progress(out["progress"])
    return samples, runner.parse_rate(out["rate"]), rows, timer


def _upto(seq, frame: int, at=lambda s: s.frame):
    return [s for s in seq if at(s) <= frame]


def _at(rows, frame: int) -> dict[int, tuple[int, int]]:
    """{car: (lap, segment)} at `frame` from parse_progress rows."""
    return {r.car: (r.lap, r.segment) for r in rows if r.frame == frame}


@pytest.mark.mame
def test_dc_beginner_drives_identically(tmp_path, romset_zip, disc):  # noqa: F811
    """Three 60 s autoplay runs on Beginner (DAYTONA_COURSE 0): the original set; the converted set
    (`converted_set(rs, disc, 0)`: Beginner's race tables and ground rebuilt from the Dreamcast and written into the
    free runs, the scenery kept); and that same converted Course with the arcade's AI speeds put back, its only
    field that is not the arcade's value (the Dreamcast's speeds are within 5.73e-6 of them).

    - the converted run's ground calls equal the original's call for call up to IDENTICAL_TO and its render-rate log
      frame for frame up to RATE_IDENTICAL_TO, and part the frame after (the AI speed); at GO + 20 s every car is
      on its original lap and within SEGMENT_SPREAD road segments of its original place;
    - with the arcade's speeds the converted run equals the original for the whole run: ground calls, rate and
      progress logs, past GO + 20 s. So the Dreamcast ground (its moved and turned off-road cells included), the
      step table, road line, line table and checkpoint table in the free runs and their repointed words race as
      the arcade's own;
    - MAME printed WRONG CHECKSUMS for both converted runs (run_set checks it): the converted sets were loaded."""
    rs = RomSet.from_zip(romset_zip)
    name = course_files(0, kind="data")
    c = build.converted_course(rs, decode_data(disc.read(name), name), disc.read(EXE_NAME), 0)
    converted = build.converted_set(rs, disc, 0)
    assert build.encode(rs, c).split_files() == converted.split_files()
    ref = G.decode_course(rs, 0)
    lines = _speeds_of(c.ai_lines, ref.ai_lines)
    assert lines == ref.ai_lines and lines != c.ai_lines        # the speed is the one AI field that differs
    arcade_speed = build.encode(rs, dataclasses.replace(c, ai_lines=lines))

    runs = {k: _run(s, tmp_path / k, romset_zip)
            for k, s in (("original", None), ("converted", converted), ("arcade_speed", arcade_speed))}
    samples, rate, rows, timer = runs["original"]
    assert max(s.frame for s in samples) > GO_20 and len(samples) > 15000         # the race ran past GO + 20 s
    assert max(r[0] for r in rate) > GO_20

    got = runs["arcade_speed"]
    assert len(got[0]) == len(samples)
    assert got[0] == samples, next(i for i, (a, b) in enumerate(zip(got[0], samples)) if a != b)
    assert got[1] == rate
    assert got[2] == rows and got[3] == timer

    g, r, p, _ = runs["converted"]
    assert _upto(g, IDENTICAL_TO) == _upto(samples, IDENTICAL_TO)
    assert _upto(g, IDENTICAL_TO + 1) != _upto(samples, IDENTICAL_TO + 1)          # pinned: they part at 1880
    assert _upto(r, RATE_IDENTICAL_TO, lambda t: t[0]) == _upto(rate, RATE_IDENTICAL_TO, lambda t: t[0])
    assert _upto(r, RATE_IDENTICAL_TO + 1, lambda t: t[0]) != _upto(rate, RATE_IDENTICAL_TO + 1, lambda t: t[0])
    want, have = _at(rows, GO_20), _at(p, GO_20)
    cars = ref.car_count[1]                                                          # 40 on Beginner
    assert set(want) == set(have) == set(range(cars)), (cars, sorted(want), sorted(have))
    assert {k: v[0] for k, v in have.items()} == {k: v[0] for k, v in want.items()}  # every car on its lap
    segments = len(ref.road)                     # a car's segment runs 0-195 on Beginner, then the lap counts
    spread = {k: abs((have[k][0] - want[k][0]) * segments + have[k][1] - want[k][1]) for k in want}
    assert max(spread.values()) <= SEGMENT_SPREAD, {k: (want[k], have[k]) for k in want}


# ---- Desert City (n 3) ----

# Frozen from the discovery runs (docs/spike/2026-11-convert.md, "Desert City in MAME"). With the standing start
# (build.ROLLING_STARTS[3]) the race clock starts at frame 1331: Beginner's slot holds the pre-race state (game state
# 0x501090 = 0x100000) one frame, where Expert holds it to its green light (1652), and only Beginner's rolling-start
# countdown held the clock to 1893; the per-id length is Not found.
N3_GO = 1331
N3_SECONDS = 180            # to frame 10354
# The autoplay holds the wheel centred, so the player's car drives into the walls: on the converted set it is on
# segment 134 when the clock (Beginner's start time, 3200 ticks) runs out at N3_TIME_OVER, 2 segments before the first
# checkpoint (136), and the race ends (RESULTS, GAME OVER). The long-time run raises every time table's start word
# to LONG_TIME (12000 ticks, past the run's end) so the race runs the whole 180 s: the AI cars lap, the player
# reaches the checkpoints. Only the start words differ from the converted set.
N3_TIME_OVER = 4530         # the clock reads 0
LONG_TIME = 12000
RISE_AFTER = 1              # frames from the player's reaching a checkpoint segment to the clock's rise (measured)
LONG_LAPS = 2               # laps every AI car completes in the long-time run (measured: all 39, first 4053-5736)
# The ground calls per car from N3_GO (`_ground_hits`: the call is the car's whose segment centre is nearest), exact,
# from the discovery runs: the ground is queried for the cars near the player, so to the time-over 8 of the 40 cars
# (20,709 calls) and over the long-time run all 40 (66,908 calls), every one returning a polygon. Pinned so a car
# with a handful of calls cannot pass the 99.9 % gate trivially.
CALLS_TO_TIME_OVER = {0: 16084, 1: 2329, 2: 742, 3: 679, 4: 311, 5: 228, 6: 155, 8: 181}
CALLS_LONG_TIME = {0: 36005, 1: 4091, 2: 1309, 3: 1652, 4: 310, 5: 1958, 6: 221, 7: 186, 8: 238, 9: 39, 10: 1982,
                   11: 365, 12: 981, 13: 789, 14: 1341, 15: 459, 16: 692, 17: 186, 18: 66, 19: 178, 20: 303, 21: 1077,
                   22: 1279, 23: 922, 24: 1074, 25: 157, 26: 99, 27: 578, 28: 373, 29: 138, 30: 90, 31: 206, 32: 182,
                   33: 1533, 34: 665, 35: 1131, 36: 20, 37: 2149, 38: 577, 39: 1307}


def _long_time(s: RomSet, c) -> RomSet:
    """build.long_time with LONG_TIME (also the playtest set's, tools/desert_city_playtest.py)."""
    return build.long_time(s, c, LONG_TIME)


def _cars(rows) -> dict[int, list]:
    out = collections.defaultdict(list)
    for r in rows:
        out[r.car].append(r)
    return out


def _new_lists(rate) -> set[int]:
    """Frames whose display list is new: one write to the read start and a list head unlike the previous frame's."""
    return {f for (f, w, _, h), (_, _, _, h0) in zip(rate[1:], rate) if w >= 1 and h != h0}


def _ground_hits(samples, cars, road, upto: int, go: int = N3_GO) -> tuple[collections.Counter, collections.Counter]:
    """(calls, hits) per car from `go` (N3_GO) to `upto`: a call is the car's whose segment centre is nearest (x, z); a
    hit returns a polygon. The caller checks that every car has a progress row on every frame of the window (a frame
    without one would be skipped here)."""
    at = {(r.frame, r.car): r.segment for rows in cars.values() for r in rows}
    calls, hits = collections.Counter(), collections.Counter()
    for s in samples:
        if not go <= s.frame <= upto or any((s.frame, k) not in at for k in cars):
            continue
        car = min(cars, key=lambda k: math.hypot(road[at[s.frame, k]].centre[0] - s.x,
                                                 road[at[s.frame, k]].centre[2] - s.z))
        calls[car] += 1
        hits[car] += s.height is not None
    return calls, hits


@pytest.mark.mame
def test_desert_city_drives(tmp_path, romset_zip, disc):  # noqa: F811
    """Desert City (n 3) converted into Beginner's slot with no scenery (`converted_set(rs, disc, 3,
    scenery="empty")`), raced N3_SECONDS on Beginner's panel (DAYTONA_COURSE 0), and the same set with every time
    table's start word raised to LONG_TIME (the race then runs the whole run; the playtest set). MAME printed WRONG
    CHECKSUMS for both (run_set checks it).

    - the race is reached: a standing start, the clock counts down one tick a frame from N3_GO (its start word the
      frame before); on the converted set it reads 0 at N3_TIME_OVER, with no rise (the player, wheel centred, is
      in the walls short of segment 136);
    - every AI car's segment steps by 0 or 1 a frame (the road is continuous), its lap counts only at the wrap
      from 464 to 0; in the long-time run every AI car passes every one of Desert City's segments 0-464 and completes
      exactly LONG_LAPS laps;
    - every car has a progress row on every frame of the race; its ground calls are exactly the measured counts
      (CALLS_TO_TIME_OVER, CALLS_LONG_TIME) and at least 99.9 % of them (the player's included: it stays on the
      ground) return a polygon;
    - the clock rises at the player's checkpoint passes, segments 136, 263 and the lap line (0), each RISE_AFTER
      frame after the player reaches the segment, by the checkpoint's extension for the first lap less
      the frame's tick, and at no other frame;
    - the rate log has a new display list on every frame of the race (no scenery is drawn: the polygon bar is
      Phase 7's)."""
    rs = RomSet.from_zip(romset_zip)
    name = course_files(3, kind="data")
    c = build.converted_course(rs, decode_data(disc.read(name), name), disc.read(EXE_NAME), 3, scenery="empty")
    converted = build.converted_set(rs, disc, 3, scenery="empty")
    assert build.encode(rs, c, build.RACE_FIELDS + build.SCENERY_FIELDS).split_files() == converted.split_files()
    assert c.rolling_start == (C.ROLLING_STARTS, False)
    assert [cp.segment for cp in c.checkpoints] == [136, 263, 0] and len(c.road) - 1 == 464

    for k, s in (("converted", converted), ("long_time", _long_time(converted, c))):
        _drive_checks(k, c, _n3_run(s, tmp_path / k, N3_SECONDS, romset_zip)[0])


def _n3_run(s: RomSet, wd, seconds: int, romset_zip, env=None, logs=("ground", "rate", "progress")):
    """One autoplay run of `s` on Beginner's panel with the ground, rate and progress logs (and `logs`):
    ((samples, rate, cars, timer), run_set's paths)."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    out = mame_build.run_set(s, wd, seconds, romset_zip, {"DAYTONA_COURSE": "0", **(env or {})}, logs=logs)
    samples, _ = runner.parse_ground(out["ground"])
    runner.require_calls(samples)
    rows, timer = runner.parse_progress(out["progress"])
    return (samples, runner.parse_rate(out["rate"]), _cars(rows), timer), out


def _drive_checks(k: str, c, run, calls_to_time_over=None, calls_long_time=None, go: int = N3_GO,
                  time_over: int = N3_TIME_OVER, repeat: frozenset = frozenset(),
                  long_laps: int | tuple = LONG_LAPS,
                  rises_to_time_over: int = 0, split: int = 0) -> None:
    """test_desert_city_drives' checks on one run of n 3's converted Course `c`: `k` "converted" (to the time-over)
    or "long_time" (the start words raised to LONG_TIME, the whole run). The ground calls per car are
    CALLS_TO_TIME_OVER / CALLS_LONG_TIME, or `calls_to_time_over` / `calls_long_time` for a run of another set; the
    race clock starts at `go` (N3_GO) and runs out at `time_over` (N3_TIME_OVER) on the converted set (Desert City
    at its own id 3 has the standing start's countdown: test_desert_city_id3). Frames in `repeat` may draw a list
    equal to the frame before's (it is written, but nothing moved: the frame after a standing start's green light,
    as on stock Expert). Over the long-time run every AI car completes `long_laps` laps (LONG_LAPS, Desert City's), or a
    number of laps in `long_laps` when it is a tuple (a short course whose field spreads over more than a lap before
    GO + 8000: Circuit Pixie, Phase 11 Task 4). To the time-over the clock rises `rises_to_time_over` times (0: the
    player reaches no checkpoint, Desert City; else at the first checkpoints, each RISE_AFTER frames after the player
    reaches it, by its extension). Over the long-time run the clock rises at every checkpoint the player reaches, lap
    after lap (lap 1's on most courses; on Circuit Pixie's short oval the lap line twice), each by its lap's
    extension. With `split` (the release set's gates, Phase 12 Task 4) the "converted" calls need only the same cars,
    the same sum and each car within `split` of its count: tests/test_courses_mame.py `_gate` says why."""
    road, n = c.road, len(c.road) - 1                       # segments 0..464; the lap line is the wrap to 0
    start = c.time_tables[(C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)][2][0]
    marks = [cp.segment for cp in c.checkpoints]
    samples, rate, cars, timer = run
    first = start if k == "converted" else LONG_TIME
    end = time_over if k == "converted" else max(timer)
    assert sorted(cars) == list(range(c.car_count[1])), k                  # 40 cars
    assert timer[go - 1] == first and timer[go] == first - 1, (k, timer[go - 1], timer[go])
    rises = [f for f in range(go, end + 1) if timer[f] > timer[f - 1]]
    assert all(timer[f] == timer[f - 1] - 1 for f in range(go, end + 1) if f not in rises), k
    for car, rows in cars.items():
        race = [r for r in rows if go - 1 <= r.frame <= end]
        for a, b in zip(race, race[1:]):
            if car:
                assert (b.segment - a.segment) % n in (0, 1), (k, a, b)
            if b.lap != a.lap:
                assert (a.segment, b.segment, b.lap) == (n, 0, a.lap + 1), (k, a, b)
    frames = collections.Counter(r.frame for rows in cars.values() for r in rows if go <= r.frame <= end)
    assert set(frames) == set(range(go, end + 1)) and set(frames.values()) == {len(cars)}, k
    calls, hits = _ground_hits(samples, cars, road, end, go)
    want = calls_to_time_over or CALLS_TO_TIME_OVER if k == "converted" else calls_long_time or CALLS_LONG_TIME
    if split and k == "converted":
        assert set(calls) == set(want) and sum(calls.values()) == sum(want.values()) and \
            all(abs(calls[car] - want[car]) <= split for car in want), (k, dict(calls))
    else:
        assert dict(calls) == want, (k, dict(calls))
    assert all(hits[car] >= 0.999 * calls[car] for car in calls), (k, {car: (hits[car], calls[car])
                                                                     for car in calls})
    assert set(range(go, end + 1)) - repeat <= _new_lists(rate), k
    assert all(w >= 1 for f, w, _, _ in rate if f in repeat), k
    if k == "converted":
        assert timer[time_over] == 0 and timer[time_over - 1] == 1 and len(rises) == rises_to_time_over, rises
        player = [r for r in cars[0] if go <= r.frame <= end]
        for f, cp in zip(rises, c.checkpoints):
            r = next(r.frame for r in player if r.segment == cp.segment and (cp.segment or r.lap == 1))
            assert f - r == RISE_AFTER and timer[f] - timer[f - 1] == cp.extensions[0] - 1, (f, r, cp)
        return
    assert end >= go + 8000                                                 # the race ran the whole run
    for car, rows in cars.items():
        if car:
            race = [r for r in rows if r.frame >= go]
            assert {r.segment for r in race} == set(range(n + 1)), car
            laps = long_laps if isinstance(long_laps, tuple) else (long_laps,)
            assert race[-1].lap - race[0].lap in laps, (car, race[0], race[-1])
    player = [r for r in cars[0] if r.frame >= go]
    reached = []                    # (frame, lap, checkpoint): every checkpoint the player reaches, lap after lap
    for lap in range(len(c.checkpoints[0].extensions)):
        for cp in c.checkpoints:
            f = next((r.frame for r in player if r.segment == cp.segment and r.lap == lap + (cp.segment == 0)), None)
            if f is None:
                break
            reached.append((f, lap, cp))
        else:
            continue
        break
    assert len(reached) >= len(marks) and len(rises) == len(reached), (rises, reached)    # every one of lap 1's
    for f, (r, lap, cp) in zip(rises, reached):     # a lap's extension is its time table word (Checkpoint.extensions)
        assert f - r == RISE_AFTER, (f, r, cp.segment)
        assert timer[f] - timer[f - 1] == cp.extensions[lap] - 1, (f, lap, cp)


# ---- the model scenery (Phase 7, Task 1): docs/spike/2026-11-convert.md, "Object library" ----

BOUND_SECONDS = 60          # to frame 3452: boot, attract, both select screens and each course's race past its start
BOUND_GEO_FROM, BOUND_GEO_EVERY = 600, 2
QUAD_SNAP = 1900            # the chase camera behind the player (segment 37), the quad 3 segments ahead (z 268.5): the
#                             player passes it at about 1925; the camera 45 units short of it (the discovery runs)
QUAD_SECONDS = 82           # to frame 4716: past N3_TIME_OVER, for Phase 6's checks of the converted run
QUAD_MARGIN = 3             # px: the projected quad's edge, either way (MAME rounds its spans)
QUAD_PIXELS = 2700          # the projected quad's area at QUAD_SNAP is at least this (measured 2,750: about 50 x 54)
# The quad set's ground calls per car to the time-over (`_ground_hits`), frozen from the discovery runs. The same 20,709
# calls as the M3a set's (CALLS_TO_TIME_OVER), in the same order with the same arguments and results; drawing the quad
# (on screen 1850-1945) costs the i960 time, so one call of frame 1945 is made at 1946 and AI car 3 is a segment
# ahead at frame 2041 only, which moves 3 calls between the cars near the player. The player's race is identical.
CALLS_QUAD_TO_TIME_OVER = {0: 16084, 1: 2330, 2: 742, 3: 676, 4: 313, 5: 228, 6: 155, 8: 181}


def _list_hashes(path) -> dict[int, str]:
    """{frame: SHA-1 of the walked geometry list (every command's kind and arguments, jumps followed)} of a
    geo_log.lua log."""
    from dtools.arcade import geo
    return {frame: hashlib.sha1(repr([(c.kind, c.args) for c in geo.walk(words, start // 4)]).encode()).hexdigest()
            for frame, start, _, words in geo.parse_log(path)}


@pytest.mark.mame
def test_library_bound_patch_draws_originals(tmp_path, romset_zip):
    """The object-library bound patch alone (course.LIBRARY_BOUND_PATCH: the lower bound of the draw routines 0x17acc
    and 0x17cf8 moved down to DATA_FREE's start; what converted_set adds with scenery="model") changes nothing the
    originals draw. Beginner, Expert and Advanced (DAYTONA_COURSE 0, 2, 1), BOUND_SECONDS each on the original set and
    on the set with only the patch: the same geometry display list (every command of the walked list, jumps followed)
    on every second frame from BOUND_GEO_FROM (attract, the select screens, the race), and the same render-rate log on
    every frame. Every list pointer of the original set lies inside OBJECT_LIBRARY, so the wider bound admits no list
    the original skips. MAME printed WRONG CHECKSUMS for the patched sets (run_set checks it)."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    rs = RomSet.from_zip(romset_zip)
    patched = C.patch_library_bound(rs)
    assert C.library_bounds(patched) == C.PATCHED_LIBRARY
    env = {"DAYTONA_GEO_FROM": str(BOUND_GEO_FROM), "DAYTONA_GEO_EVERY": str(BOUND_GEO_EVERY)}
    for course in (0, 1, 2):
        got = {}
        for name, s in (("original", None), ("patched", patched)):
            out = mame_build.run_set(s, tmp_path / f"{course}-{name}", BOUND_SECONDS, romset_zip,
                                     {"DAYTONA_COURSE": str(course), **env}, logs=("geo", "rate"))
            got[name] = _list_hashes(out["geo"]), runner.parse_rate(out["rate"])
            out["geo"].unlink()                         # about 430 MB each
        (lists, rate), (p_lists, p_rate) = got["original"], got["patched"]
        frames = set(range(BOUND_GEO_FROM, max(lists) + 1, BOUND_GEO_EVERY))
        assert set(lists) == frames and max(lists) >= BOUND_SECONDS * 57 and len(rate) > BOUND_SECONDS * 57, course
        assert p_lists == lists, (course, min(f for f in lists if p_lists.get(f) != lists[f]))
        assert p_rate == rate, course
        assert len(set(lists.values())) > 1000, course                  # the lists change: a race was drawn


def _project(cmds, oba: int, points):
    """The object command for `oba` in a walked list, and `points` (world) on the screen as the rasterizer puts them
    (model2_v.cpp): under the list's last matrix (0x0b; translate 0x0c rewrites its last three words) and focus
    (0x09), p' = (x m0 + y m3 + z m6 + m9, x m1 + y m4 + z m7 + m10, x m2 + y m5 + z m8 + m11) times (fx, fy, 1); on
    screen (centre x - window x0 + x' / z', window y1 - centre y - y' / z'), the window (0x03) and its centre
    (`coords[2 + sel]`, sel = the object opcode's bits 29-30) as the window command sends them (12-bit signed; the
    window spans the 496 x 384 screen: renderer x offset -x0, y offset y1 - 384). (None, []) when the list does not
    draw `oba`."""
    def s12(v):
        return v - 0x1000 if v & 0x800 else v

    mat = focus = window = None
    for c in cmds:
        if c.kind in (0x0B, 0x1B):
            mat = list(c.args["mat"])
        elif c.kind in (0x0C, 0x1C):
            mat[9:12] = list(c.args["translate"])
        elif c.kind in (0x09, 0x19):
            focus = c.args["focus"]
        elif c.kind in (0x03, 0x13):
            window = [(s12(x), s12(y)) for x, y in c.args["coords"]]
        elif c.kind in (0x01, 0x11) and c.args["oba"] == oba:
            cx, cy = window[2 + ((c.opcode >> 29) & 3)]
            out = []
            for x, y, z in points:
                px = x * mat[0] + y * mat[3] + z * mat[6] + mat[9]
                py = x * mat[1] + y * mat[4] + z * mat[7] + mat[10]
                pz = x * mat[2] + y * mat[5] + z * mat[8] + mat[11]
                out.append((cx - window[0][0] + focus[0] * px / pz, window[1][1] - cy - focus[1] * py / pz))
            return c, out
    return None, []


@pytest.mark.mame
def test_one_quad_drawn(tmp_path, romset_zip, disc):  # noqa: F811
    """One quad of real scenery (tests/test_convert_build.py, `desert_quad`): QUAD_SIZE x QUAD_SIZE (10 units), flat
    colour (palette entry MAGENTA, (31, 0, 31)), standing across Desert City's main straight at segment QUAD_SEGMENT
    with its bottom edge QUAD_LIFT (2 units) above the road (upright: the chase camera rides about 2 units above the
    road, so a quad lying flat at that height would be edge-on), written by converted_set(rs, disc, 3,
    scenery="model"): its list in DATA_FREE (drawn through the bound patch), its object in the polygon run, its header
    and texture points in the textures run. Raced QUAD_SECONDS, with the M3a set (scenery="empty") run to the same
    frame:

    - the geometry list at QUAD_SNAP draws the quad (an object command for its oba, tpa and tha, obc OBC) and the
      object it names decodes, from the set's polygon ROM, to the quad's points;
    - projected with that list's matrix, focus and window (`_project`), the quad covers at least QUAD_PIXELS of the
      snapshot; inside it (QUAD_MARGIN px in from its edge) the pixels are magenta, and the M3a snapshot has none
      there; outside it (QUAD_MARGIN px out) the two snapshots are identical;
    - the converted race still passes Phase 6's checks (test_desert_city_drives, "converted": the clock, the cars'
      progress, every ground call a polygon, a new display list every frame): on the M3a run exactly as frozen, on
      the quad run with its own split of the calls per car (CALLS_QUAD_TO_TIME_OVER); and the player's race is the
      M3a run's: the same clock, render-rate log and player's progress on every frame, the same ground calls in the
      same order with the same results, each made in the same frame or the next.

    The two snapshots are copied to out/phase7/snaps/task1/."""
    from PIL import Image, ImageDraw
    import numpy as np
    from dtools.arcade import geo
    from dtools.arcade import polyrom as P
    from dtools.convert import tables
    from dtools.mame import build as mame_build
    from tests.test_convert_build import (HEADER_AT, MAGENTA, OBJECT_AT, POINTS_AT, QUAD_LIFT, QUAD_SEGMENT,
                                          QUAD_SIZE, desert_quad)
    rs = RomSet.from_zip(romset_zip)
    name = course_files(3, kind="data")
    d = decode_data(disc.read(name), name)
    sd = desert_quad(tables.road_line(d))
    c = build.converted_course(rs, d, disc.read(EXE_NAME), 3, scenery="model", scenery_data=sd)
    quad_set = build.converted_set(rs, disc, 3, scenery="model", scenery_data=sd)
    assert build.encode(rs, c, build.RACE_FIELDS + build.MODEL_FIELDS).split_files() == \
        quad_set.split_files()
    obj = sd.objects[OBJECT_AT]
    corners = [obj.seed0, obj.seed1, obj.records[0].p1, obj.records[0].p0]          # round the quad
    x, y, z = c.road[QUAD_SEGMENT].centre
    assert (min(p[1] for p in corners), max(p[1] for p in corners)) == (pytest.approx(y + QUAD_LIFT),
                                                                         pytest.approx(y + QUAD_LIFT + QUAD_SIZE))
    assert max(p[0] for p in corners) - min(p[0] for p in corners) == QUAD_SIZE and {p[2] for p in corners} == {z}
    assert sd.headers[HEADER_AT].colour == MAGENTA and not sd.headers[HEADER_AT].textured

    snap_env = {"DAYTONA_SNAPS": str(QUAD_SNAP)}
    m3a = build.converted_course(rs, d, disc.read(EXE_NAME), 3, scenery="empty")
    empty_run, empty = _n3_run(build.converted_set(rs, disc, 3, scenery="empty"), tmp_path / "empty", QUAD_SECONDS,
                               romset_zip, snap_env)
    run, out = _n3_run(quad_set, tmp_path / "quad", QUAD_SECONDS, romset_zip,
                       {**snap_env, "DAYTONA_GEO_FROM": str(QUAD_SNAP), "DAYTONA_GEO_EVERY": "100000"},
                       logs=("ground", "rate", "progress", "geo"))
    shots = {k: sorted(p["snap"].rglob("*.png"))[0] for k, p in (("empty", empty), ("quad", out))}
    keep = config.out_dir() / "phase7" / "snaps" / "task1"
    keep.mkdir(parents=True, exist_ok=True)
    for k, p in shots.items():
        shutil.copy(p, keep / f"{k}_{QUAD_SNAP}.png")

    (frame, start, _, words), = list(geo.parse_log(out["geo"]))
    assert frame == QUAD_SNAP
    cmd, screen = _project(geo.walk(words, start // 4), 0x800000 | OBJECT_AT, corners)
    assert cmd is not None, "the list does not draw the quad"
    assert (cmd.args["tpa"], cmd.args["tha"], cmd.args["obc"]) == (POINTS_AT, HEADER_AT, C.OBC)
    drawn = P.decode_object(quad_set.polygons, OBJECT_AT, cmd.args["obc"], G.GEOMETRY_MODE)
    assert drawn == obj and set(P.polygons(drawn)[0].points) == set(corners)

    a = np.asarray(Image.open(shots["quad"]).convert("RGB"), dtype=np.int16)
    e = np.asarray(Image.open(shots["empty"]).convert("RGB"), dtype=np.int16)
    assert a.shape == e.shape

    def mask(grow: int) -> np.ndarray:
        img = Image.new("1", (a.shape[1], a.shape[0]), 0)
        ImageDraw.Draw(img).polygon(screen, fill=1)
        m = np.asarray(img, dtype=bool)
        for _ in range(abs(grow)):                                  # grow or shrink by one pixel (4-neighbour)
            n = m.copy()
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (n | np.roll(m, (dy, dx), (0, 1))) if grow > 0 else (n & np.roll(m, (dy, dx), (0, 1)))
            m = n
        return m

    inner, outer = mask(-QUAD_MARGIN), mask(QUAD_MARGIN)
    magenta = (a[..., 0] >= 200) & (a[..., 1] <= 40) & (a[..., 2] >= 200)
    assert mask(0).sum() >= QUAD_PIXELS and inner.sum() > 0.6 * mask(0).sum(), (mask(0).sum(), inner.sum())
    assert magenta[inner].all(), magenta[inner].mean()
    assert not ((e[..., 0] >= 200) & (e[..., 1] <= 40) & (e[..., 2] >= 200))[outer].any()
    differ = (np.abs(a - e) > 16).any(axis=2)
    assert not differ[~outer].any(), np.argwhere(differ & ~outer)[:8].tolist()

    _drive_checks("converted", m3a, empty_run)
    _drive_checks("converted", c, run, CALLS_QUAD_TO_TIME_OVER)
    (samples, rate, cars, timer), (e_samples, e_rate, e_cars, e_timer) = run, empty_run
    assert timer == e_timer and rate == e_rate and cars[0] == e_cars[0]
    assert len(samples) == len(e_samples)
    for got, want in zip(samples, e_samples):
        assert dataclasses.replace(got, frame=0, n=0) == dataclasses.replace(want, frame=0, n=0)
        assert got.frame - want.frame in (0, 1)


# ---- Desert City seen (Phase 7, Task 6): docs/spike/2026-11-convert.md, "Desert City seen" ----

SEEN_SECONDS = 180          # the long-time run to frame 10354: the player's lap ends at SEEN_LAP_END
SEEN_LAP_END = 9880         # the player (wheel centred, from segment 460) reaches the lap line, lap 1 (measured)
FIRST_CORNER, CHECKPOINT = 90, 136          # segments: the hairpin's entry (85-115) and the first checkpoint
FIRST_CORNER_AT, CHECKPOINT_AT = 2701, 4743  # the frames the player first reaches them (measured; the M3a run's too)
# The places snapshot (frame: name): GO, GO + 5 s, + 10 s, + 30 s at 57.52 Hz; FIRST_CORNER_AT and CHECKPOINT_AT.
SEEN_PLACES = {N3_GO: "go", N3_GO + 288: "go_plus_5s", N3_GO + 575: "go_plus_10s", FIRST_CORNER_AT: "first_corner",
               N3_GO + 1726: "go_plus_30s", CHECKPOINT_AT: "checkpoint_136"}
SEEN_SNAPS = "task7"        # out/phase7/snaps/<this>/: Task 6's are in task6 (before thinning)
SEEN_DIFFER = 0.1           # at least this fraction of a snapshot differs from the M3a one there (> 16 on a channel;
#                             measured 12.2 % at GO + 5 s, the fly-by beside the car, to 91.2 % at GO)
SEEN_COLOURS = 50           # ... and the differing pixels hold at least this many distinct colours, not one flat
#                             colour (measured 79 at the first corner, sand and rock, to 390 at the checkpoint)
BAR, BEGINNER_MAX = 4251, 3533              # drawn polygons per list: the frame-rate bar; Beginner's own maximum
# Frozen from the runs: the drawn polygons per display list (geo.count_polygons) on every frame from N3_GO to
# SEEN_LAP_END (8,550 lists): the maximum, its frame, and the frames over BAR. Task 6 (before thinning): 4,779 at
# 5222 (segment 170, Beginner's fly-by camera beside the car; the frame cap also left out 3 cells there), 917 frames
# over. Task 7 (the props dropped, build.THIN): docs/spike/2026-11-convert.md, "Thinning": 4,078 at 8752. Task 8 fix
# (the HUD outline merged from 465 quads to build.OUTLINE_QUADS, 64: 401 fewer on every race list): 3,677 at 8752.
SEEN_MAX, SEEN_MAX_AT, SEEN_OVER_BAR = 3677, 8752, 0
# The ground calls per car (`_ground_hits`) of the scenery set's runs, frozen from the runs: drawing the scenery costs
# the i960 time, which moves a call between AI cars near the player (the player's own race is the M3a run's, frame for
# frame). Task 6 (unthinned): to the time-over the M3a split exactly; over the long-time run one call from car 22 to
# car 5. Task 7 (thinned): to the time-over one call from car 4 to car 1.
CALLS_SEEN_TO_TIME_OVER = {**CALLS_TO_TIME_OVER, 1: 2330, 4: 310}
CALLS_SEEN_LONG_TIME = {**CALLS_LONG_TIME, 5: 1959, 22: 1278}


def _geo_per_frame(path, rs: RomSet) -> dict[int, tuple[int, frozenset]]:
    """{frame: (drawn polygons, the object addresses the list draws)} of every list of a geo_log.lua log, in order
    (polygon RAM carried from list to list). The log is removed afterwards, whatever happens (about 2.4 GB)."""
    from dtools.arcade import geo
    ram = geo.PolyRam()
    out = {}
    try:
        for frame, start, _, words in geo.parse_log(path):
            cmds = geo.walk(words, start // 4)
            out[frame] = (geo.count_polygons(cmds, rs, ram),
                          frozenset(c.args["oba"] for c in cmds if c.kind in (0x01, 0x11)))
    finally:
        path.unlink(missing_ok=True)
    return out


@pytest.mark.mame
def test_desert_city_seen(tmp_path, romset_zip, disc):  # noqa: F811
    """Desert City (n 3) with its own scenery (`converted_set(rs, disc, 3, scenery="model")`) in Beginner's slot,
    raced on Beginner's panel: the converted set to past the time-over, the same set with the start words at LONG_TIME
    (`_long_time`) the whole SEEN_SECONDS with the geometry list logged on every frame from N3_GO, and the M3a set
    (scenery="empty") with the same start words to the last place, snapshots at SEEN_PLACES on both long-time runs:

    - Phase 6's race checks hold on both scenery runs (`_drive_checks`: the clock, the cars' progress and laps, every
      ground call a polygon, the checkpoints, a new display list every frame), with their own frozen split of the
      ground calls per car (CALLS_SEEN_TO_TIME_OVER, CALLS_SEEN_LONG_TIME); the player's progress and the clock are
      the M3a run's on every frame both log;
    - Review Focus 5: the drawn count is taken on every frame from N3_GO to SEEN_LAP_END, a full lap of the player's
      camera (the player passes every segment from 460 round to the lap line); its maximum (SEEN_MAX at SEEN_MAX_AT)
      and the frames over BAR (SEEN_OVER_BAR) frozen; Beginner's 3,533 printed beside it;
    - at each place the player is where SEEN_PLACES says, and the snapshot differs from the M3a one on at least
      SEEN_DIFFER of its pixels (scenery is drawn), which hold at least SEEN_COLOURS distinct colours (not one flat
      colour).

    The snapshots are copied to out/phase7/snaps/SEEN_SNAPS/ (<place>_arcade_<frame>.png and
    <place>_m3a_<frame>.png)."""
    from PIL import Image
    import numpy as np
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    rs = RomSet.from_zip(romset_zip)
    name = course_files(3, kind="data")
    c = build.converted_course(rs, decode_data(disc.read(name), name), disc.read(EXE_NAME), 3, scenery="empty")
    seen = build.converted_set(rs, disc, 3, scenery="model")
    assert G.decode_course(seen, 0).time_tables == c.time_tables      # the race tables keep M3a's places
    snaps = {"DAYTONA_SNAPS": ",".join(str(f) for f in SEEN_PLACES)}
    last = max(SEEN_PLACES)

    _drive_checks("converted", c, _n3_run(seen, tmp_path / "converted", QUAD_SECONDS, romset_zip)[0],
                  CALLS_SEEN_TO_TIME_OVER)
    run, out = _n3_run(_long_time(seen, c), tmp_path / "long_time", SEEN_SECONDS, romset_zip,
                       {**snaps, "DAYTONA_GEO_FROM": str(N3_GO), "DAYTONA_GEO_EVERY": "1"},
                       logs=("ground", "rate", "progress", "geo"))
    drawn = _geo_per_frame(out["geo"], seen)                    # first: it removes the log, whatever fails later
    _drive_checks("long_time", c, run, calls_long_time=CALLS_SEEN_LONG_TIME)
    e_out = mame_build.run_set(_long_time(build.converted_set(rs, disc, 3, scenery="empty"), c), tmp_path / "m3a",
                               last // 57 + 3, romset_zip, {"DAYTONA_COURSE": "0", **snaps}, logs=("progress",))
    e_rows, e_timer = runner.parse_progress(e_out["progress"])
    _, _, cars, timer = run
    e_player = [r for r in e_rows if r.car == 0]
    assert e_player == [r for r in cars[0] if r.frame <= e_player[-1].frame] and e_player[-1].frame > last
    assert all(timer[f] == t for f, t in e_timer.items())

    # Review Focus 5: every frame from GO over the player's full lap
    player = {r.frame: (r.lap, r.segment) for r in cars[0]}
    lap = [player[f] for f in range(N3_GO, SEEN_LAP_END + 1)]
    assert lap[0] == (0, 460) and lap[-1] == (1, 0) and lap[-2][0] == 0
    assert {s for _, s in lap} == set(range(len(c.road)))
    assert set(range(N3_GO, SEEN_LAP_END + 1)) <= set(drawn)
    window = {f: drawn[f][0] for f in range(N3_GO, SEEN_LAP_END + 1)}
    top = max(window, key=lambda f: (window[f], -f))
    over = sum(n > BAR for n in window.values())
    print(f"\ndrawn per list, GO to the lap's end ({len(window)} frames): max {window[top]} at {top} (segment "
          f"{player[top][1]}), {over} frames over the bar {BAR}; Beginner's own maximum {BEGINNER_MAX}")
    assert (window[top], top, over) == (SEEN_MAX, SEEN_MAX_AT, SEEN_OVER_BAR)

    # the places: the scenery is drawn there
    keep = config.out_dir() / "phase7" / "snaps" / SEEN_SNAPS
    keep.mkdir(parents=True, exist_ok=True)
    shots = {}
    for k, p in (("arcade", out), ("m3a", e_out)):
        every = sorted(p["snap"].rglob("*.png"))
        assert len(every) > len(SEEN_PLACES), (k, len(every))     # the places, then the one MAME writes at the end
        shots[k] = every[:len(SEEN_PLACES)]
    first = {}
    for f, (lp, s) in sorted(player.items()):
        first.setdefault((lp, s), f)
    assert (first[(0, FIRST_CORNER)], first[(0, CHECKPOINT)]) == (FIRST_CORNER_AT, CHECKPOINT_AT)
    for (frame, place), a_path, e_path in zip(sorted(SEEN_PLACES.items()), shots["arcade"], shots["m3a"]):
        for k, p in (("arcade", a_path), ("m3a", e_path)):
            shutil.copy(p, keep / f"{place}_{k}_{frame}.png")
        a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.int16)
        e = np.asarray(Image.open(e_path).convert("RGB"), dtype=np.int16)
        # the whole frame is compared: the player's race, the cars and the HUD are the M3a run's at the same frame,
        # so the pixels that differ are the scenery region (what the M3a set leaves as backdrop and bare ground)
        differ = (np.abs(a - e) > 16).any(axis=2)
        colours = len(np.unique(a[differ], axis=0))
        print(f"{place} ({frame}, segment {player[frame][1]}): {differ.mean():.1%} differs, {colours} colours")
        assert differ.mean() >= SEEN_DIFFER and colours >= SEEN_COLOURS, (place, differ.mean(), colours)


# ---- the frame rate (Phase 7, Task 7): docs/spike/2026-11-convert.md, "Thinning" ----

# Frozen from the runs (Task 7): the drawn polygons per list over the lap (the maximum and its frame), and the frame's
# polygon count (cells_log.lua, 0x5010e8) at its largest over the lap. Task 6 before thinning (this test on its set):
# 4,779 at 5222, 917 frames over BAR, the count 5,577 at 5216, the cap leaving scenery out on 285 frames. Task 7:
# 4,078 at 8752, the count 4,724 at 6061. Task 8 fix (the HUD outline merged to 64 quads, its budget word 64 against
# 465): 3,677 at 8752, the count 4,323 at 6061 (both 401 lower; 332 frames over BEGINNER_MAX against 1,110).
RATE_MAX, RATE_MAX_AT = 3677, 8752
RATE_COUNT_MAX = 4323


@pytest.mark.mame
def test_desert_city_frame_rate(tmp_path, romset_zip, disc):  # noqa: F811
    """Review Focus 5 on Desert City with its own thinned scenery (`converted_set(rs, disc, 3, scenery="model")`:
    the props dropped, build.THIN), raced on Beginner's panel with the start words at LONG_TIME for SEEN_SECONDS, the
    geometry list (geo_log.lua) and the visible cells and polygon count (cells_log.lua) logged on every frame from
    N3_GO:

    - the player's camera covers a full lap, N3_GO to SEEN_LAP_END (every segment), and every frame of it has a new
      display list (renderrate.lua);
    - every list of that lap draws at most BAR polygons (geo.count_polygons, cars and HUD included); the maximum
      (RATE_MAX at RATE_MAX_AT) frozen, Beginner's own maximum BEGINNER_MAX printed beside it;
    - the frame cap (course.OBC, 5,000, against the summed budget words) leaves no scenery out: on every frame each
      visible cell that holds objects has its objects in that frame's list (and every scenery object the list draws
      is a visible cell's, which ties the two logs' frames); the largest polygon count (RATE_COUNT_MAX) frozen."""
    from dtools.mame import runner
    rs = RomSet.from_zip(romset_zip)
    name = course_files(3, kind="data")
    c = build.converted_course(rs, decode_data(disc.read(name), name), disc.read(EXE_NAME), 3, scenery="empty")
    seen = build.converted_set(rs, disc, 3, scenery="model")
    course = G.decode_course(seen, 0)
    owner = {p.oba: cell for cell, k in enumerate(course.cells) for p in course.object_lists[k]}
    occupied = set(owner.values())
    run, out = _n3_run(_long_time(seen, c), tmp_path / "long_time", SEEN_SECONDS, romset_zip,
                       {"DAYTONA_GEO_FROM": str(N3_GO), "DAYTONA_GEO_EVERY": "1", "DAYTONA_CELLS_FROM": str(N3_GO)},
                       logs=("ground", "rate", "progress", "geo", "cells"))
    per = _geo_per_frame(out["geo"], seen)                      # first: it removes the log, whatever fails later
    _, rate, cars, _ = run
    lap = range(N3_GO, SEEN_LAP_END + 1)
    player = {r.frame: (r.lap, r.segment) for r in cars[0]}
    assert player[N3_GO] == (0, 460) and player[SEEN_LAP_END] == (1, 0) and player[SEEN_LAP_END - 1][0] == 0
    assert {player[f][1] for f in lap} == set(range(len(c.road)))
    assert set(lap) <= _new_lists(rate)
    cells = runner.parse_cells(out["cells"])
    assert set(lap) <= set(per) and set(lap) <= set(cells)
    left_out, strays = {}, {}
    for f in lap:
        drawn = {owner[a] for a in per[f][1] if a in owner}
        visible = [k for k in cells[f][1] if k in occupied]
        if set(visible) - drawn:
            left_out[f] = sorted(set(visible) - drawn)
        if drawn - set(visible):
            strays[f] = sorted(drawn - set(visible))
    counts = {f: per[f][0] for f in lap}
    top = max(lap, key=lambda f: (counts[f], -f))
    over = [f for f in lap if counts[f] > BAR]
    busiest = max(lap, key=lambda f: (cells[f][0], -f))
    print(f"\ndrawn per list, GO to the lap's end ({len(lap)} frames): max {counts[top]} at {top} (segment "
          f"{player[top][1]}), {len(over)} over the bar {BAR}; Beginner's own maximum {BEGINNER_MAX}; frames over it "
          f"{sum(n > BEGINNER_MAX for n in counts.values())}; the polygon count's largest {cells[busiest][0]} at "
          f"{busiest}; frames the cap left scenery out {len(left_out)} {dict(list(left_out.items())[:5])}; "
          f"scenery drawn outside the visible list {len(strays)} {dict(list(strays.items())[:5])}")
    assert not strays
    assert not over and not left_out
    assert (counts[top], top, cells[busiest][0]) == (RATE_MAX, RATE_MAX_AT, RATE_COUNT_MAX)


# ---- what scenery wakes up (Phase 7, Task 8): docs/spike/2026-11-convert.md, "Confirmed with scenery" ----

READS_FROM = N3_GO - 3      # 1328 (memtrace's frame count): the race's first frame of culling and of the camera script
READS_TO = SEEN_LAP_END + 20  # the trace window's end (exclusive): past the player's lap
READS_TAPS = {"DAYTONA_TRACE_READ": "286f7e0-286fa5f,2806b00-280efff",
              "DAYTONA_TRACE_WRITE": "5010e8-5010eb!,5014a8-5014ab!,5016c0-5016df!"}
VIEW_FIRST_READ = 0x17754   # ip after `ld 0x0(r8)` at 0x17750: the view test's first point of the quadrant's table
VIEW_SECOND_READ = 0x17828  # ip after `ld 0x50(r8)` at 0x17824: the second group's first point, read on every path.
#                             The trace logs 256-byte pages: quadrants 2 and 3 share their first point's page
#                             (0x286f900), so the pair of pages (first point, +0x50) names the table exactly
CAMERA_JUMP_READ = 0x6618   # ip after `ld 0x8(g0),g0` at 0x6614: the player follows the script's jump record
COUNT_RESET = 0x1848        # ips after the stores to the frame's polygon count 0x5010e8: the reset (0x1840),
COUNT_LIST = 0x17C0C        # 0x17acc's add (an actor's, a car's or the HUD's list; 0x17c04)
COUNT_CELL = 0x17D58        # and 0x17cf8's (a grid cell's list, or a car part's; 0x17d50)
HUD_FROM = N3_GO - 1        # 1330: the race HUD's first frame (its outline list is read from here; 1328 has none)
CAMERA_PERIOD = 3600        # slot 0's script loops after 3600 ticks, one a frame: the fly-by again at 4929 and 8529
# Frozen from the runs (trace sessions 1-2): the frame's polygon count before the first scenery cell (the HUD outline,
# the HUD's other objects and the cars), its largest over the lap and the frame (the frame memtrace logs the visible
# list in); the room the cap leaves at the lap's largest count (RATE_COUNT_MAX: here at 6060, the cells log's 6061).
# Task 8 (the 465-quad outline): 2,295 at 7290, margin 276; the fix (64 quads): both 401 better.
OTHER_MAX, OTHER_MAX_AT = 1894, 7290
COUNT_MARGIN = 677


@pytest.mark.mame
def test_desert_city_scenery_reads(tmp_path, romset_zip, disc):  # noqa: F811
    """What drawing Desert City's scenery wakes up (`converted_set(rs, disc, 3, scenery="model")` with the start words
    at LONG_TIME, Beginner's panel), traced over the player's lap (memtrace.lua, frames 1100 to READS_TO):

    - the view test reads its tables on every frame that culls (from READS_FROM; every frame from N3_GO): the table
      of the heading's quadrant (`(heading + 0x2000) >> 14 & 3`, the heading 0x5014a8 as written), whatever the
      cell, told apart by the pages of two of its points (VIEW_SECOND_READ: the four tables' pairs differ); the
      reference cell (the visible list's first, 0x5016c1) is a cell of the road line on every frame and every road
      cell is one, all four quadrants are used, and every visible cell lies in the 5 x 5 block round the reference
      cell;
    - the camera script read is slot 0's (`course.camera_script`, car-relative keyframes) and no other: every page
      read in 0x2806b00-0x280efff holds its keyframes or its jump record, from READS_FROM on, and it loops every
      CAMERA_PERIOD frames;
    - the frame's polygon count (0x5010e8) starts every frame from HUD_FROM (the race HUD) with the HUD outline's
      budget word (the outline's record count, OUTLINE_QUADS: 465 before the fix), before any scenery cell is
      tested against the cap (OBC); the trailing run of 0x17cf8's adds is the visible occupied cells' (their budgets
      sum to it); the largest count is RATE_COUNT_MAX (the cells log's), COUNT_MARGIN below the cap, and the count
      before the scenery peaks at OTHER_MAX."""
    from dtools.arcade import trace
    from dtools.mame import build as mame_build
    rs = RomSet.from_zip(romset_zip)
    name = course_files(3, kind="data")
    c = build.converted_course(rs, decode_data(disc.read(name), name), disc.read(EXE_NAME), 3, scenery="empty")
    seen = build.converted_set(rs, disc, 3, scenery="model")
    d = G.decode_course(seen, 0)
    budget = {cell: sum(p.raw[0] for p in d.object_lists[k]) for cell, k in enumerate(d.cells)}
    (outline,) = d.hud_map.entries
    tables = [struct.unpack("<I", seen.read_i960(C.VIEW_TESTS + 4 * q, 4))[0] for q in range(4)]
    script = C.camera_script(seen, struct.unpack("<I", seen.read_i960(C.CAMERA_SCRIPTS, 4))[0])
    jump = max(k.addr for k in script.keyframes) + C.KEYFRAME_SIZE     # the jump record after the last one
    script_pages = set(range(script.start & ~0xFF, jump + C.KEYFRAME_SIZE, 0x100))
    road = {C.cell_of(p.centre[0], p.centre[2]) for p in c.road}

    log = tmp_path / "trace.txt"
    mame_build.run_set(_long_time(seen, c), tmp_path / "run", SEEN_SECONDS, romset_zip,
                       {"DAYTONA_COURSE": "0", "DAYTONA_TRACE_LOG": str(log), "DAYTONA_TRACE_FROM": "1100",
                        "DAYTONA_TRACE_TO": str(READS_TO), **READS_TAPS})
    events = trace.parse_trace(log)
    view, camera, jumps = collections.defaultdict(set), collections.defaultdict(set), []
    heading, frames, cycles, cur = None, {}, [], None
    for e in events:
        if e.kind == "R" and 0x1770C <= e.ip < 0x17918:
            view[e.frame].add((e.ip, e.addr))
        elif e.kind == "R" and 0x2806B00 <= e.addr <= 0x280EFFF:
            camera[e.frame].add(e.addr)
            if e.ip == CAMERA_JUMP_READ:
                jumps.append(e.frame)
        elif e.kind != "V":
            continue
        elif e.addr == 0x5014A8:
            heading = e.value & 0xFFFF
        elif e.addr == 0x5010E8 and e.ip == COUNT_RESET:
            cur = {"adds": [], "cells": {}}
            cycles.append(cur)
        elif cur is None:
            raise AssertionError(f"a write to {e.addr:#x} at frame {e.frame} before the first reset of the count")
        elif e.addr == 0x5010E8:
            cur["adds"].append((e.ip, e.value))
        elif 0x5016C1 <= e.addr <= 0x5016DF:
            cur["cells"][e.addr - 0x5016C1] = e.value
        elif e.addr == 0x5016C0:                                  # the count, stored after the cells (0x17500)
            cells = [cur["cells"][i] for i in range(e.value)]
            frames[e.frame] = (cells, heading, cur)

    # the view test
    race = sorted(frames)                     # every frame culls from N3_GO; at the start 1329 has none (1330 two)
    assert race[0] == READS_FROM and set(range(N3_GO, READS_TO)) <= set(race) and sorted(view) == race
    quadrants, offsets = collections.Counter(), set(C.culling(seen).offsets)
    pages = [({t & ~0xFF}, {(t + 0x50) & ~0xFF}) for t in tables]         # (first point, second group) per quadrant
    assert len({(min(a), min(b)) for a, b in pages}) == 4
    for f in race:
        cells, h, _ = frames[f]
        assert h is not None, f"frame {f}: a visible list before the first heading write"
        q = (h + 0x2000) >> 14 & 3
        quadrants[q] += 1
        assert ({a for ip, a in view[f] if ip == VIEW_FIRST_READ},
                {a for ip, a in view[f] if ip == VIEW_SECOND_READ}) == pages[q], (f, q)
        r = cells[0]
        assert r in road, (f, hex(r))
        assert all(k - r in offsets and abs((k & 15) - (r & 15)) <= 2 for k in cells), (f, cells)
    refs = {frames[f][0][0] for f in race}
    assert refs == road and set(quadrants) == {0, 1, 2, 3}

    # the camera script
    assert min(camera) == READS_FROM and set().union(*camera.values()) <= script_pages
    assert jumps == [READS_FROM + CAMERA_PERIOD + 1, READS_FROM + 2 * CAMERA_PERIOD + 1]

    # the frame's polygon count
    other, final = {}, {}
    for f in race:
        cells, _, cyc = frames[f]
        adds = [(ip, v - p) for (ip, v), p in zip(cyc["adds"], [0] + [v for _, v in cyc["adds"]])]
        assert f < HUD_FROM or adds[0] == (COUNT_LIST, outline.raw[0]), (f, adds[:3])
        k = len(adds)
        while k and adds[k - 1][0] == COUNT_CELL:
            k -= 1
        assert sum(n for _, n in adds[k:]) == sum(budget[cell] for cell in cells), f
        other[f], final[f] = sum(n for _, n in adds[:k]), sum(n for _, n in adds)
    busiest = max(race, key=lambda f: (final[f], -f))
    top = max(race, key=lambda f: (other[f], -f))
    print(f"\nview test read on {len(view)} frames, quadrants {dict(sorted(quadrants.items()))}, reference cells "
          f"{len(refs)} (the road's); camera pages {sorted(hex(p) for p in set().union(*camera.values()))}, jumps "
          f"{jumps}; the count: the HUD outline's {outline.raw[0]} first on every frame, largest {final[busiest]} at "
          f"{busiest} (before the scenery {other[busiest]}), margin {C.OBC - final[busiest]}; before the scenery at "
          f"most {other[top]} at {top}")
    assert outline.raw[0] == len(d.hud_map.objects[C.check_rom_object(outline.oba)].records) == build.OUTLINE_QUADS
    assert final[busiest] == RATE_COUNT_MAX and C.OBC - final[busiest] == COUNT_MARGIN
    assert (other[top], top) == (OTHER_MAX, OTHER_MAX_AT)


# ---- Desert City at its own course id 3 (Phase 8, Task 5): docs/spike/2026-11-sixteen.md, "Desert City at id 3" ----

# Frozen from the runs (custom build, the select's entry 3, coin at 600). At id 3 Desert City takes Expert's path at
# 0x3eac (state 20): the standing start's 320-frame countdown, the lights, the clock from ID3_GO, as Expert's green
# light (1652); in Beginner's slot it started at 1331 with no countdown.
ID3_GO = 1651
ID3_REPEAT = frozenset({ID3_GO + 1})
"""The frame after the green light writes a display list equal to the one before (nothing has moved yet): stock MAME
on Expert does the same (its clock starts at 1653 and frame 1654 repeats), Beginner's slot never had it (no
countdown)."""
ID3_TIME_OVER = 4850        # Beginner's start time (3200 ticks) run out on the set without the long time
ID3_SECONDS = 185           # the long-time run, to frame 10641: past the player's lap
ID3_LAP_END = 10181         # the player (wheel centred, from segment 460) reaches the lap line, lap 1
ID3_SNAPS = {750: "select-desert-city", 1600: "grid", 5600: "mid-race"}
ID3_HUD = (372, 166, 476, 292)  # (left, top, right, bottom) px of the race HUD's course map on a snapshot
ID3_SNAP_DIR = config.out_dir() / "phase8" / "snaps" / "task5"
CALLS_ID3_TO_TIME_OVER = {0: 16100, 1: 2370, 2: 808, 3: 1096, 4: 382, 5: 200, 6: 56, 8: 171, 9: 29}
"""The ground calls per car (`_ground_hits`) from ID3_GO to the time-over / over the long-time run, exact, from the
runs (the standing start's countdown moves the field: Phase 6's split does not carry over)."""
CALLS_ID3_LONG_TIME = {0: 35981, 1: 3621, 2: 2156, 3: 1198, 4: 1442, 5: 479, 6: 648, 7: 408, 8: 960, 9: 801, 10: 214,
                       11: 303, 12: 853, 13: 314, 14: 1154, 15: 414, 16: 2691, 17: 254, 18: 294, 19: 64, 20: 330,
                       21: 1049, 22: 338, 23: 398, 24: 359, 25: 168, 26: 745, 27: 390, 28: 1505, 29: 1984, 30: 908,
                       31: 300, 32: 996, 33: 223, 34: 212, 35: 842, 36: 1376, 37: 834, 38: 664, 39: 601}
# The bars over the player's lap (Phase 7: 3,677 at 8752, the count 4,323): the race starts 320 frames later and the
# AI cars meet the player elsewhere, so the busiest frames move; the count's largest is now the bunched field just
# after the green light (1715), 432 under the cap.
ID3_MAX, ID3_MAX_AT, ID3_COUNT_MAX = 3871, 6287, 4568
COIN2 = 7000                # the second game's coin (Review Focus 4): the first game is over and the attract runs
ID3_TEX_AT = ID3_GO + 50    # texture RAM in Desert City's race (its own sheet-0 block uploaded)
BEGINNER_FROM = COIN2 + 550     # the second game's race (as RACE_FROM, 1150, after the coin at 600)
BEGINNER_RATE_FROM = COIN2 + 800
BEGINNER_SNAPS = (COIN2 + 800, COIN2 + 1400, COIN2 + 1900, COIN2 + 2400)
BEGINNER_TEX_AT = COIN2 + 800   # texture RAM in the second game's race (Beginner's course loaded at about COIN2 + 730)
BEGINNER_SECONDS = (COIN2 + 2700) // 57 + 1


def _id3_run(s: RomSet, wd, seconds: int, romset_zip, env: dict, logs, exe):
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    out = mame_build.run_set(s, wd, seconds, romset_zip, {"DAYTONA_ENTRY": "3", **env}, logs=logs, mame=exe)
    samples, _ = runner.parse_ground(out["ground"])
    runner.require_calls(samples)
    rows, timer = runner.parse_progress(out["progress"])
    return (samples, runner.parse_rate(out["rate"]), _cars(rows), timer), out


@pytest.mark.mame
@pytest.mark.custom_mame
def test_desert_city_id3(tmp_path, romset_zip, disc):  # noqa: F811
    """Desert City at its own course id 3 (`converted_set(base, disc, 3, slot=3, scenery="model")` on the grown set
    with the new ids' guards, test_convert_build._guarded, then the sixteen-entry select with M4A_ENTRIES: converted
    before the select since Phase 9's final review, Phase 8's order's bytes), entered from the select (DAYTONA_ENTRY
    3), on the custom build:

    - the long-time set (`_long_time`), ID3_SECONDS: Phase 6's race checks (`_drive_checks`: a standing start, the
      clock from ID3_GO, every car's progress and laps, every ground call a polygon, the checkpoints, a new display
      list every frame) and Phase 7's bars over the player's lap ID3_GO..ID3_LAP_END (every segment): every list draws
      at most BAR polygons, the cap leaves no scenery out and draws none outside the visible cells; the maximum
      (ID3_MAX at ID3_MAX_AT) and the largest polygon count (ID3_COUNT_MAX) frozen; snapshots of the select with
      Desert City under the cursor, the grid and mid-race, and the HUD map cut from mid-race, to ID3_SNAP_DIR; the
      player completes lap 1 at ID3_LAP_END, and backup SRAM after the run is the fresh reset's everywhere a course
      id names (`_check_fresh_sram`). Not a discriminating check by itself: a control run with id 3's SRAM entries
      pointed back at Beginner's words left them fresh too (a lap in the race writes none of them; the ranking and
      best-lap writers run at a finished race or in the time attack; discovery log), so the sink is proven offline
      (test_guard_new_ids) and this pins that the run-time state stays fresh;
    - Review Focus 4, one session: the set without the long time races id 3 to its time-over (Phase 6's checks, the
      clock 0 at ID3_TIME_OVER), the game ends, and the second game (DAYTONA_SECOND, coin at COIN2) races entry 0,
      Beginner. A second game's race depends on the session's history (the frame counter 0x5010b0, the AI tables
      0x501858, the last start time 0x5013c4 differ after a different first game: discovery log), so the reference
      is twofold: with the same first game (Expert) the converted set's second game equals stock MAME's on the stock
      set call for call (the display list on every frame from BEGINNER_FROM, the render-rate log from
      BEGINNER_RATE_FROM, the snapshots); after Desert City, Beginner draws exactly the stock run's objects, texture
      points and headers, and texture RAM (both sheets) and luma RAM after its load are the stock run's, where
      Desert City's race had its own sheet 0. Backup SRAM after the session: no counter written (Desert City's game
      ended; Expert's game in the other session counts), every ranking table as the Expert session's, Beginner's the
      fresh reset's."""
    from PIL import Image
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    import tests.test_convert_build as tcb               # tcb._guarded: test_widened_desert_city swaps the chain
    from dtools.arcade import select as S
    from tests.test_custom_mame import _custom, _lists, _snaps
    exe = str(_custom())
    base = tcb._guarded(RomSet.from_zip(romset_zip))
    name, mn, tn = (course_files(3, kind=k) for k in ("data", "model", "textures"))
    from dtools.dc import pol
    d = decode_data(disc.read(name), name)
    sd = build.model_scenery(base, d, pol.decode_pol(disc.read(mn), mn), disc.read(tn), build.THIN[3], slot=3)
    c = build.converted_course(base, d, disc.read(EXE_NAME), 3, slot=3, scenery="model", scenery_data=sd)
    seen = S.patch_select(build.converted_set(base, disc, 3, slot=3, scenery="model", scenery_data=sd), S.M4A_ENTRIES)
    course = G.decode_course(seen, 3)
    assert course.rolling_start[1] is False and course.sheet_sources[0] != G.decode_course(seen, 0).sheet_sources[0]
    owner = {p.oba: cell for cell, k in enumerate(course.cells) for p in course.object_lists[k]}
    occupied = set(owner.values())

    # the long-time run: the race and the bars
    snaps = ",".join(map(str, ID3_SNAPS))
    run, out = _id3_run(_long_time(seen, c), tmp_path / "long_time", ID3_SECONDS, romset_zip,
                        {"DAYTONA_GEO_FROM": str(ID3_GO), "DAYTONA_GEO_EVERY": "1", "DAYTONA_CELLS_FROM": str(ID3_GO),
                         "DAYTONA_SNAPS": snaps}, ("ground", "rate", "progress", "geo", "cells"), exe)
    per = _geo_per_frame(out["geo"], seen)                      # first: it removes the log, whatever fails later
    _, rate, cars, timer = run
    player = {r.frame: (r.lap, r.segment) for r in cars[0]}
    lap = range(ID3_GO, ID3_LAP_END + 1)
    calls = _drive_checks_calls(c, run, ID3_GO)
    print(f"\nlong-time run: ground calls per car {calls}")
    _drive_checks("long_time", c, run, calls_long_time=CALLS_ID3_LONG_TIME, go=ID3_GO,
                  time_over=ID3_TIME_OVER, repeat=ID3_REPEAT)
    assert player[ID3_GO] == (0, 460) and player[ID3_LAP_END] == (1, 0) and player[ID3_LAP_END - 1][0] == 0
    assert {player[f][1] for f in lap} == set(range(len(c.road)))
    assert set(lap) - ID3_REPEAT <= _new_lists(rate)
    cells = runner.parse_cells(out["cells"])
    assert set(lap) <= set(per) and set(lap) <= set(cells)
    left_out, strays = {}, {}
    for f in lap:
        drawn = {owner[a] for a in per[f][1] if a in owner}
        visible = [k for k in cells[f][1] if k in occupied]
        if set(visible) - drawn:
            left_out[f] = sorted(set(visible) - drawn)
        if drawn - set(visible):
            strays[f] = sorted(drawn - set(visible))
    counts = {f: per[f][0] for f in lap}
    top = max(lap, key=lambda f: (counts[f], -f))
    over = [f for f in lap if counts[f] > BAR]
    busiest = max(lap, key=lambda f: (cells[f][0], -f))
    print(f"drawn per list, GO to the lap's end ({len(lap)} frames): max {counts[top]} at {top} (segment "
          f"{player[top][1]}), {len(over)} over the bar {BAR}; Phase 7's {RATE_MAX} at {RATE_MAX_AT}; the polygon "
          f"count's largest {cells[busiest][0]} at {busiest} (Phase 7's {RATE_COUNT_MAX}); frames the cap left "
          f"scenery out {len(left_out)}; scenery drawn outside the visible list {len(strays)}")
    assert not strays and not over and not left_out
    assert (counts[top], top, cells[busiest][0]) == (ID3_MAX, ID3_MAX_AT, ID3_COUNT_MAX)
    ID3_SNAP_DIR.mkdir(parents=True, exist_ok=True)
    shots = sorted(out["snap"].rglob("*.png"))
    assert len(shots) > len(ID3_SNAPS)
    for p, label in zip(shots, ID3_SNAPS.values()):
        shutil.copy(p, ID3_SNAP_DIR / f"{label}.png")
    Image.open(shots[list(ID3_SNAPS.values()).index("mid-race")]).crop(ID3_HUD).save(ID3_SNAP_DIR / "hud-map.png")
    # the sink at run time: the player completed lap 1 at ID3_LAP_END, and backup SRAM holds the fresh reset's
    # newest-entry words, time-attack and best-lap words, every ranking table and no counter (see the docstring)
    _check_fresh_sram((tmp_path / "long_time" / "run" / "nvram" / "daytona" / "backup1").read_bytes(), base)

    # Review Focus 4: id 3, then Beginner, in one session
    env = {"DAYTONA_GEO_FROM": str(BEGINNER_FROM), "DAYTONA_GEO_EVERY": "1",
           "DAYTONA_SNAPS": ",".join(map(str, BEGINNER_SNAPS)), "DAYTONA_TEXDUMP_AT": f"{ID3_TEX_AT},{BEGINNER_TEX_AT}"}
    runs = {}
    for k, s, binary, sel in (("id3_then_beginner", seen, exe, {"DAYTONA_ENTRY": "3", "DAYTONA_SECOND": f"0:{COIN2}"}),
                              ("expert_then_beginner", seen, exe, {"DAYTONA_ENTRY": "2",
                                                                   "DAYTONA_SECOND": f"0:{COIN2}"}),
                              ("stock", None, "mame", {"DAYTONA_COURSE": "2", "DAYTONA_SECOND": f"c0:{COIN2}"})):
        wd = tmp_path / k
        (wd / "tex").mkdir(parents=True)
        logs = ("ground", "rate", "progress", "geo") if k == "id3_then_beginner" else ("rate", "geo")
        o = mame_build.run_set(s, wd, BEGINNER_SECONDS, romset_zip,
                               {**sel, **env, "DAYTONA_TEXDUMP_DIR": str((wd / "tex").resolve())}, logs=logs,
                               mame=binary)
        lists, refs = _lists(o["geo"])
        o["geo"].unlink()
        rate = runner.parse_rate(o["rate"])
        runs[k] = dict(out=o, lists=lists, refs=refs, rate=[r for r in rate if r[0] >= BEGINNER_RATE_FROM],
                       snaps=_snaps(o["snap"]), tex=wd / "tex", sram=(wd / "run" / "nvram" / "daytona" / "backup1"))
        if k == "id3_then_beginner":
            samples, _ = runner.parse_ground(o["ground"])
            rows, b_timer = runner.parse_progress(o["progress"])
            both = (samples, rate, _cars(rows), b_timer)
    calls = _drive_checks_calls(c, both, ID3_GO, ID3_TIME_OVER)
    print(f"to the time-over: ground calls per car {calls}")
    _drive_checks("converted", c, both, calls_to_time_over=CALLS_ID3_TO_TIME_OVER, go=ID3_GO,
                  time_over=ID3_TIME_OVER, repeat=ID3_REPEAT)
    a, b, ref = runs["id3_then_beginner"], runs["expert_then_beginner"], runs["stock"]
    # the converted set races Beginner after a game call for call as the stock set does (the same first game)
    assert set(ref["lists"]) == set(range(BEGINNER_FROM, max(ref["lists"]) + 1))
    assert len(set(ref["lists"].values())) > 1000
    assert b["lists"] == ref["lists"], min(f for f in ref["lists"] if b["lists"].get(f) != ref["lists"][f])
    assert b["rate"] == ref["rate"] and len(ref["rate"]) > 1000
    assert len(b["snaps"]) == len(ref["snaps"]) == len(BEGINNER_SNAPS) + 1 and b["snaps"] == ref["snaps"]
    # after Desert City, Beginner's course is Beginner's: what the lists draw (objects, texture points and headers)
    # and texture RAM (both sheets: the texel block upload) and luma RAM after its load are the stock run's; the race
    # itself differs only where the session's history moves the AI (docs/spike/2026-11-sixteen.md)
    print(f"Beginner after Desert City: {sum(a['lists'].get(f) != v for f, v in ref['lists'].items())} of "
          f"{len(ref['lists'])} lists differ from the stock run's (after Expert); drawn refs "
          f"{len(a['refs'])} / {len(ref['refs'])}, extra {len(a['refs'] - ref['refs'])}, missing "
          f"{len(ref['refs'] - a['refs'])}")
    assert a["refs"] == ref["refs"]
    for kind in ("tex0", "tex1", "luma"):
        got, want = (r["tex"] / f"{kind}.{BEGINNER_TEX_AT}.bin" for r in (a, ref))
        assert got.read_bytes() == want.read_bytes(), kind
    first = (a["tex"] / f"tex0.{ID3_TEX_AT}.bin").read_bytes()
    assert first != (a["tex"] / f"tex0.{BEGINNER_TEX_AT}.bin").read_bytes()       # Desert City's own block, then
    assert first != (ref["tex"] / f"tex0.{ID3_TEX_AT}.bin").read_bytes()          # Beginner's again
    # SRAM: Desert City's game counted nothing (the counters' guard) and wrote no ranking; Expert's game did count
    sram, sram_b = a["sram"].read_bytes(), b["sram"].read_bytes()
    assert struct.unpack_from("<12I", sram, 0x140) == (0,) * 12
    assert struct.unpack_from("<12I", sram_b, 0x140)[1] == 1
    fresh = C.sram_sink(base)
    for t in C.rankings_layout().tables:
        assert sram[t - C.SRAM[0]:t - C.SRAM[0] + 24 * 24] == sram_b[t - C.SRAM[0]:t - C.SRAM[0] + 24 * 24]
    assert sram[0x360:0x360 + 576] == fresh[:576]
    shutil.copy(sorted(a["out"]["snap"].rglob("*.png"))[1], ID3_SNAP_DIR / "beginner-after-desert-city.png")


def _check_fresh_sram(sram: bytes, rs: RomSet) -> None:
    """Backup SRAM `sram` (a session's backup1) holds what the reset leaves for every SRAM word a course id names: the
    newest-entry words 0xffffffff (course.rankings_layout(rs): 0x1d00a20-0x1d00a2f on a stock-ranked set, the sixteen
    of a ranked one, Phase 9), 0x1d00b60-0x1d00ba3 `course.fresh_settings`, every ranking table of the layout
    `course.fresh_ranking`, the twelve counters 0x1d00140-0x1d0016f zero."""
    at = C.SRAM[0]
    lay = C.rankings_layout(rs)
    if lay.tables == C.rankings_layout().tables:
        assert sram[0xA20:0xA30] == b"\xff" * 16
    else:
        assert all(sram[a - at:a - at + 4] == b"\xff" * 4 for a in lay.last_entry)
    assert sram[0xB60:0xBA4] == C.fresh_settings(rs)
    for k, t in enumerate(lay.tables):
        assert sram[t - at:t - at + 24 * 24] == C.fresh_ranking(rs, k), k
    assert struct.unpack_from("<12I", sram, 0x140) == (0,) * 12


def _drive_checks_calls(c, run, go: int, upto: int | None = None) -> dict[int, int]:
    """The ground calls per car of a run from `go` to `upto` (default the timer's last frame), for freezing."""
    samples, _, cars, timer = run
    calls, _ = _ground_hits(samples, cars, c.road, upto or max(timer), go)
    return dict(calls)
