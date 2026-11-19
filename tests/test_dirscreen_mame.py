"""MAME tests of the Course Direction screen (M9 stage 3; dtools/arcade/dirscreen.py; plan
docs/superpowers/plans/2026-11-18-reverse-stage3-direction.md): the choice, the timer and the course id, the race
after it, and each course in reverse raced from it, on the release set (release.release_set: ids 8-15, the row of
eight, the screen). Marked `mame` and `custom_mame`."""

import pytest

from dtools.arcade import course as C
from dtools.arcade import dirscreen as D
from dtools.arcade import select as S
from dtools.arcade.romset import RomSet
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

SCREEN_FIRST = 928                  # the screen's first running frame (state 18 logged; its set-up, state 17, at 927,
                                    # where Mission Select began without it)
CONFIRMED = 948                     # the gas at 945 (autoplay's DIR_STEP) takes the choice here: state 19
MISSION = 981                       # Mission Select's choice (state 9) after the 32-frame hold and its layers (its set-up,
                                    # state 8, runs within the frame the hold ends: not logged); 54 frames later
TIMEOUT = SCREEN_FIRST - 1 + D.SCREEN_FRAMES   # idle: the timer (640, set at the set-up frame) at 0: state 19
_SETS: dict = {}


def _set(romset_zip, disc):  # noqa: F811
    key = str(romset_zip)
    if key not in _SETS:
        from dtools.release import release_set
        _SETS[key] = release_set(RomSet.from_zip(romset_zip), disc)
    return _SETS[key]


def _log(path):
    """{frame: (state, choice, armed, timer, course id, wheel, linked, votes, transmission)} of dir_log.lua."""
    out = {}
    for line in open(path):
        p = line.split()
        if p and p[0] == "D":
            out[int(p[1])] = tuple(int(v) for v in p[2:11])
    return out


def _run(tmp_path, romset_zip, rs, entry: int, mode: str, **env):
    from dtools import config
    from dtools.mame import build as mb
    log = tmp_path / "dir.txt"
    mb.run_set(rs, tmp_path, 50, romset_zip, {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": str(entry), "DAYTONA_DIRECTION": mode,
                                              "DAYTONA_DIR_LOG": str(log), "DAYTONA_DIR_FROM": "900",
                                              "DAYTONA_DIR_TO": "2800", **env}, mame=config.custom_mame())
    return _log(log)


def _states(log):
    """[(first frame, state)] of each run of one state."""
    out, last = [], None
    for f in sorted(log):
        if log[f][0] != last:
            out.append((f, log[f][0]))
            last = log[f][0]
    return out


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("entry", (0, 3, 7))
def test_direction_choice(tmp_path, romset_zip, disc, entry):  # noqa: F811
    """For entries 0, 3 and 7 (Three Seven Speedway, Desert City, Mermaid Lake), four runs of the screen:

    - `normal` (the paddle centred, the gas at 945): armed at its second frame, NORMAL, the hold from CONFIRMED,
      Mission Select from MISSION, the course id n through the race load (states 14-16);
    - `reverse` (centred, then right from 935, the gas): REVERSE, the id n + 8 from MISSION through the race load,
      widen_ids' shadows refreshed (the race loads n + 8's course: the id stays n + 8 to the log's end);
    - `held` (the paddle right from Mission Select's paddle frame, 800, never centred on the screen): never armed,
      NORMAL, id n (the D6 rule);
    - `idle` (no gas): NORMAL at the timer's end (TIMEOUT), id n.
    Mission Select after the screen: its own states 8-11 in order, and with DAYTONA_GEAR=manual on the `reverse` run
    (the paddle to Manual after the screen's gas) its transmission byte 1."""
    rs = _set(romset_zip, disc)
    n = S.ENTRIES[entry].course
    seen = {}
    for mode, extra in (("normal", {}), ("reverse", {"DAYTONA_GEAR": "manual"}), ("held", {}), ("idle", {})):
        wd = tmp_path / mode
        wd.mkdir()
        seen[mode] = _run(wd, romset_zip, rs, entry, mode, **extra)
    for mode in ("normal", "reverse", "held"):
        log, states = seen[mode], _states(seen[mode])
        assert (SCREEN_FIRST, D.STATE_RUN) in states and (CONFIRMED, D.STATE_HOLD) in states, (mode, states[:8])
        assert (MISSION, D.MISSION_SETUP + 1) in states, (mode, states[:10])
        after = [s for _, s in states if _ >= MISSION]
        assert after[:5] == [9, 10, 11, 12, 13] and 16 in after, (mode, after[:10])
        want_choice, want_id = {"normal": (0, n), "reverse": (1, n + D.REVERSE_OFFSET), "held": (0, n)}[mode]
        assert log[CONFIRMED][1] == want_choice, (mode, log[CONFIRMED])
        assert {log[f][4] for f in log if f < MISSION - 1} == {n}, mode
        assert {log[f][4] for f in log if f >= MISSION} == {want_id}, mode
        assert log[CONFIRMED][2] == (0 if mode == "held" else 1), mode
    assert seen["reverse"][max(seen["reverse"])][8] == 1                # manual chosen on Mission Select after it
    idle = _states(seen["idle"])
    assert (SCREEN_FIRST, D.STATE_RUN) in idle and (TIMEOUT, D.STATE_HOLD) in idle, idle[:6]
    assert seen["idle"][TIMEOUT][1] == 0 and {v[4] for v in seen["idle"].values()} == {n}


RACE_SHIFT = 52
"""Frames the race begins later after the screen (NORMAL) than in the stock game: autoplay.lua's LATE."""


def _cars(path) -> dict:
    """{frame: sorted S, C and T lines (frame dropped)} of a progress_log.lua log."""
    out: dict = {}
    for line in open(path):
        p = line.split()
        if p and p[0] in ("S", "C", "T"):
            out.setdefault(int(p[1]), []).append(" ".join([p[0]] + p[2:]))
    return {f: sorted(v) for f, v in out.items()}


@pytest.mark.mame
@pytest.mark.custom_mame
def test_race_after_screen_as_stock(tmp_path, romset_zip, disc):  # noqa: F811
    """Plan Task 3, step 1: the screen changes nothing but the race's start. National Park Speedway (entry 4) raced
    from the screen with NORMAL equals, every car's segment, lap and position and the race timer on every frame, the
    release set without the screen raced with its Mission Select step RACE_SHIFT frames later (DAYTONA_DIRECTION=late):
    the AI cars depend on the race's start frame (a frame either way diverges), so only an exact shift compares."""
    from dtools import config
    from dtools.mame import build as mb
    from dtools.release import release_set
    base = release_set(RomSet.from_zip(romset_zip), disc, direction=False)
    rs = D.patch_direction(base)                        # the same set but for the screen
    logs = {}
    for name, s, mode in (("screen", rs, "normal"), ("stock", base, "late")):
        wd = tmp_path / name
        wd.mkdir()
        out = mb.run_set(s, wd, 60, romset_zip, {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": "4", "DAYTONA_DIRECTION": mode,
                                                 "DAYTONA_SNAPS": "0"}, logs=("progress",), mame=config.custom_mame())
        logs[name] = _cars(out["progress"])
    common = sorted(set(logs["screen"]) & set(logs["stock"]))
    assert len(common) > 2000 and any(len([x for x in logs["stock"][f] if x[0] == "S"]) == 40 for f in common)
    diff = [f for f in common if logs["screen"][f] != logs["stock"][f]]
    assert not diff, (len(diff), diff[:3])


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("entry", range(8))
def test_reverse_from_screen(tmp_path, romset_zip, disc, entry):  # noqa: F811
    """Each course in reverse, raced from the screen on the release set (REVERSE: the wheel centred, then right, the
    gas): the course id n + 8 from Mission Select on (the race loads it) and the player car, driven by the autopilot
    along id n + 8's own road line (dtools.mame.road.road_file, as the reverse gates drive), runs course n backwards:
    matched to the nearest of course n's road points every 10 frames, it steps back along them at least four times as
    often as forward, over at least 30 steps (gas alone spun the car round on Sea-Side Street Galaxy)."""
    import numpy as np
    from dtools import config
    from dtools.arcade import golden
    from dtools.mame import build as mb
    from dtools.mame.road import road_file
    rs = _set(romset_zip, disc)
    n = S.ENTRIES[entry].course
    log = tmp_path / "dir.txt"
    road = road_file(golden.decode_course(rs, n + D.REVERSE_OFFSET), tmp_path / "road.txt")
    out = mb.run_set(rs, tmp_path, 60, romset_zip, {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": str(entry),
                                                    "DAYTONA_DIRECTION": "reverse", "DAYTONA_DIR_LOG": str(log),
                                                    "DAYTONA_DIR_FROM": "900", "DAYTONA_DIR_TO": "2800",
                                                    "DAYTONA_SNAPS": "0", "DAYTONA_AUTOPILOT": str(road),
                                                    "DAYTONA_AUTOPILOT_LAPS": "1"},
                     logs=("progress",), mame=config.custom_mame())
    ids = _log(log)
    assert {v[4] for f, v in ids.items() if f >= MISSION} == {n + D.REVERSE_OFFSET}, entry
    pos = []
    for line in open(out["progress"]):
        p = line.split()
        if p and p[0] == "C" and p[2] == "0":
            pos.append((float(p[3]), float(p[5])))
    pts = np.array([q.centre for q in C.road_line(rs, C.course_table(rs)[n])])[:, [0, 2]]
    size = len(pts)
    idx = [int(np.argmin(((pts - xz) ** 2).sum(1))) for xz in pos[::10]]
    steps = [((b - a + size // 2) % size) - size // 2 for a, b in zip(idx, idx[1:])]
    back, fwd = sum(s < 0 for s in steps), sum(s > 0 for s in steps)
    assert back >= 30 and back >= 4 * fwd, (entry, back, fwd)


MISSION_LATE = 927                  # Mission Select's choice (state 9) in the stock game: its set-up at 926
MISSION_AFTER = 20                  # the frames into Mission Select's choice compared (before any step)
SCREEN_CHARS = (0x3D00, 0x4000)     # the screen's chars (dirgfx: banner, panel, the two sides): left in char RAM
SCREEN_PALETTES = (0x7A, 0x80)      # their palette blocks (16 entries each)
TL_ROWS, VISIBLE_ROWS = 64, 48      # a tile layer's rows (and columns); the rows on the 384-line screen


def _dump(rs, wd, romset_zip, env, frame):
    """(tile RAM layers 0-2, char RAM, palette RAM) as u16 arrays at `frame` (title_log.lua's dumps)."""
    import numpy as np
    from dtools import config
    from dtools.mame import build as mb
    wd.mkdir()
    dumps = wd / "dumps"
    dumps.mkdir()
    mb.run_set(rs, wd, 22, romset_zip, {"DAYTONA_ROW": "8", "DAYTONA_SNAPS": "0", "DAYTONA_TITLE_LOG": str(wd / "t.txt"),
                                        "DAYTONA_TITLE_DIR": str(dumps), "DAYTONA_TITLE_DUMPS": str(frame), **env},
               mame=config.custom_mame())
    read = [np.fromfile(dumps / f"{frame}.{k}", "<u2") for k in ("tile", "char", "pal")]
    return read[0][:3 * 0x1000], read[1], read[2]


@pytest.mark.mame
@pytest.mark.custom_mame
def test_mission_select_after_screen(tmp_path, romset_zip, disc):  # noqa: F811
    """Review (stage 3, Important 2): Mission Select after the screen (NORMAL, Desert City) is the stock game's, frame
    for frame from its choice state: MISSION_AFTER frames into it, tile RAM layers 0-2 (its background, banner,
    boxes, panel and timer), palette RAM and char RAM equal the release set's without the screen (autoplay `late`), but
    the screen's own chars and palette blocks, left behind and named by no tile on the 48 visible rows. (Chars
    0x3000-0x3dff are the course backdrop's, uploaded again by every course load: Mission Select's layer 2 keeps the
    last backdrop's tiles on rows 49-54, below the screen, in the stock game too.)"""
    import numpy as np
    from dtools.release import release_set
    k = MISSION_AFTER
    tile, char, pal = _dump(_set(romset_zip, disc), tmp_path / "screen", romset_zip,
                            {"DAYTONA_ENTRY": "3", "DAYTONA_DIRECTION": "normal"}, MISSION + k)
    plain = release_set(RomSet.from_zip(romset_zip), disc, direction=False)
    s_tile, s_char, s_pal = _dump(plain, tmp_path / "stock", romset_zip,
                                  {"DAYTONA_ENTRY": "3", "DAYTONA_DIRECTION": "late"}, MISSION_LATE + k)
    lo, hi = SCREEN_CHARS
    keep = np.ones(len(char), bool)
    keep[16 * lo:16 * hi] = False
    pkeep = np.ones(len(pal), bool)
    pkeep[16 * SCREEN_PALETTES[0]:16 * SCREEN_PALETTES[1]] = False
    visible = tile.reshape(3, TL_ROWS, TL_ROWS)[:, :VISIBLE_ROWS]
    assert not any(lo <= (int(t) & 0x3FFF) < hi for t in visible.ravel()), "a visible tile names the screen's chars"
    diff = np.flatnonzero(tile != s_tile)
    assert not diff.size, [(int(i) // 0x1000, (int(i) % 0x1000) // 64, int(i) % 64) for i in diff[:6]]
    assert np.array_equal(pal[pkeep], s_pal[pkeep]) and np.array_equal(char[keep], s_char[keep])


@pytest.mark.mame
@pytest.mark.custom_mame
def test_direction_guards(tmp_path, romset_zip, disc):  # noqa: F811
    """Review (stage 3, Important 1): state 7's two exits that MAME's default TWIN cabinet never takes alone, by the
    bytes forced (dir_log.lua DAYTONA_DIR_POKE; the vote count is the link's cabinets with a vote, 0x3470-0x352c: 1
    alone): another cabinet's vote (count 2 through the wipe) skips the screen (Mission Select at its stock frames,
    926 and 927, course id n); the single-cabinet dispatcher (link flag 0 from Circuit Select) runs the screen from
    its copied table and REVERSE gives n + 8."""
    (tmp_path / "votes").mkdir()
    votes = _run(tmp_path / "votes", romset_zip, _set(romset_zip, disc), 3, "reverse",
                 DAYTONA_DIR_POKE="501b03:2:900:930")
    states = _states(votes)
    assert all(s not in (D.STATE_SETUP, D.STATE_RUN, D.STATE_HOLD) for _, s in states), states[:8]
    assert (926, D.MISSION_SETUP) in states and (927, D.MISSION_SETUP + 1) in states, states[:8]
    assert {v[4] for v in votes.values()} == {3}
    (tmp_path / "single").mkdir()
    single = _run(tmp_path / "single", romset_zip, _set(romset_zip, disc), 3, "reverse",
                  DAYTONA_DIR_POKE="5fe588:0:600:1100")
    states = _states(single)
    assert {v[6] for f, v in single.items() if f <= 1100} == {0}
    assert any(s == D.STATE_RUN for _, s in states) and any(s == D.STATE_HOLD for _, s in states), states[:10]
    assert {v[4] for f, v in single.items() if f >= MISSION} == {3 + D.REVERSE_OFFSET}


REVERSE_RANK_SECONDS = 150          # GO (1703) and Desert City reverse's lap (64 s), the goal, the record


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("course", (8, 11))
def test_reverse_best_time(tmp_path, romset_zip, disc, course):  # noqa: F811
    """M9 stage 4, Task 4 (spec 5.4): a reverse race's best time ranks into its own table. On the release set with
    the race at one lap (tests/test_mirrors_mame.py `one_lap`), Three Seven Speedway reverse (8) and Desert City
    reverse (11), each entered from the screen with REVERSE and driven by the autopilot on its own road line to the
    goal, from a fresh NVRAM with every SRAM write traced: after the boot only id `course`'s ranking table and its
    newest-entry word are written, by the race's end and the name entry after it (`check_own_ranking_writes`); the
    record is on top of that table, the rest of it fresh (`_recorded`; its name the name entry's 0, nothing typed, or
    the race end's blank); the normal course's table (id `course` - 8) and every other id's stay fresh;
    the SRAM signature is the ranked set's (no signature change: v0.1.x players keep their times)."""
    import struct
    from dtools.arcade import golden
    from dtools.mame.road import road_file
    from tests.test_mirrors_mame import (_at, _recorded, check_own_ranking_writes, one_lap, ranking_tables,
                                         traced_ranking_run)
    rel = _set(romset_zip, disc)
    test = one_lap(rel, (course,))
    lay = C.rankings_layout(rel)
    fresh = [C.fresh_ranking(rel, k) for k in range(C.COURSE_IDS)]
    entry = next(k for k, e in enumerate(S.ENTRIES) if e.course == course - D.REVERSE_OFFSET)
    road = road_file(golden.decode_course(test, course), tmp_path / "road.txt")
    sram, log = traced_ranking_run(tmp_path, romset_zip, "reverse", test, {
        "DAYTONA_ROW": "8", "DAYTONA_ENTRY": str(entry), "DAYTONA_DIRECTION": "reverse",
        "DAYTONA_AUTOPILOT": str(road), "DAYTONA_AUTOPILOT_LAPS": "99"}, REVERSE_RANK_SECONDS)
    check_own_ranking_writes(log, lay, course)
    tables = ranking_tables(sram, lay)
    time = struct.unpack_from("<I", tables[course], 4)[0]
    assert 0 < time < struct.unpack_from("<I", fresh[course], 4)[0], time
    record = bytearray(tables[course])
    if struct.unpack_from("<I", record, 20)[0] == 0:     # the name entry ran (0x228490), nothing typed: its 0
        struct.pack_into("<I", record, 20, 0x202020)     # in place of the race end's blank name
    assert _recorded(bytes(record), fresh[course], time), course
    assert all(tables[k] == fresh[k] for k in range(C.COURSE_IDS) if k != course)
    assert _at(sram, 0x01D00200, 24) == C.SIGNATURE_RANKED
