"""MAME tests of Phase 12 (M7): the release set (dtools/release.py `release_set`; plan docs/superpowers/plans/
2026-11-13-phase12-names-title-packaging.md; discovery log docs/spike/2026-11-release.md).

Marked `mame` and `custom_mame` (the release set is a custom one: skipped when dtools.config.custom_mame() finds no
build of the current patch)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import select as S
from dtools.arcade import tiles as TL
from dtools.arcade.romset import RomSet
from dtools.release import release_set
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_courses_mame import M5B_SWEEP_UP, ROW8, _row_checks

SNAP_DIR = config.out_dir() / "phase12" / "snaps" / "task7"
"""Task 2's snapshots are out/phase12/snaps/task2/ (before Task 6 gave entries 3-7 their own panels), Task 6's
out/phase12/snaps/task6/ (before Task 7 painted their maps)."""
MAP_BOXES = ((36, 167, 148, 279), (192, 167, 304, 279), (348, 167, 460, 279))
"""Each panel position's course map in a snapshot (x0, y0, x1, y1, end exclusive; 496 x 384): the 112 x 112 px where
the release set's snapshot of entry 4's hold differs from the set without the own panels' maps (measured, Task 6)."""
MODE_SWEEP_TO = 1000            # select_log's last frame in a mode sweep (the sweep's holds end by 902)

_SETS: dict = {}


def _release(romset_zip, disc) -> RomSet:  # noqa: F811
    """release_set of the player's set and disc, built once per session (each MAME test reads it, none writes it)."""
    key = str(romset_zip)
    if key not in _SETS:
        _SETS[key] = release_set(RomSet.from_zip(romset_zip), disc)
    return _SETS[key]


def _m5b(romset_zip, disc) -> RomSet:  # noqa: F811
    """The M5b playtest set (tools/desert_city_playtest.py `m5b_set`), built once per session."""
    from tests.test_playtest import _tool
    key = ("m5b", str(romset_zip))
    if key not in _SETS:
        _SETS[key] = _tool().m5b_set(RomSet.from_zip(romset_zip), disc)
    return _SETS[key]


def _mode_sweep(rs: RomSet, workdir: Path, romset_zip, mode: int):
    """The sweep of test_select_m5b (M5B_SWEEP_UP: the wheel on each entry of the row of eight for 30 frames) with the
    game mode held at `mode` (select_log.lua's DAYTONA_SELECT_MODE), the tiles logged and a snapshot at each hold
    (the hold's frame + 26): (the parsed select log, the holds' frames, {frame: snapshot path})."""
    from tests.test_custom_mame import _custom
    from tests.test_sixteen_mame import _select_run
    at = [f + 26 for f in M5B_SWEEP_UP]
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{S.paddle_for_entry(k, 8)}" for k, f in enumerate(M5B_SWEEP_UP)),
           "DAYTONA_SNAPS": ",".join(map(str, at)), "DAYTONA_SELECT_TILES": ",".join(map(str, at)),
           "DAYTONA_STEPS": "none", "DAYTONA_GAS_FROM": "2000", "DAYTONA_SELECT_FROM": "600",
           "DAYTONA_SELECT_TO": str(MODE_SWEEP_TO), "DAYTONA_SELECT_MODE": str(mode), **ROW8}
    log, _, snap = _select_run(rs, workdir, romset_zip, 20, env, str(_custom()))
    shots = sorted(Path(snap).rglob("*.png"))
    assert len(shots) >= len(at)
    return log, at, dict(zip(at, shots))


def _box(path: Path, s: int) -> np.ndarray:
    from PIL import Image
    x0, y0, x1, y1 = MAP_BOXES[s]
    return np.asarray(Image.open(path).convert("RGB"))[y0:y1, x0:x1].astype(int)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_select_release(tmp_path, romset_zip, disc):  # noqa: F811
    """Phase 12 Tasks 2 and 6 on the release set (the release row, select.ENTRIES: every course under its own name, the
    originals' names on a layer-1 window over their name box, WINDOW_NAME; entries 3-7 their own panels and maps),
    custom MAME, autoplay.lua with DAYTONA_ROW 8: test_select_m5b's sweep, stops, idle wheel and per-entry starts
    (`_row_checks`): at every hold the entries' tiles, the window mask's name rows over an original's position and
    its label rows clear there (the band Sega's, blinking with the cursor: entry 0 in both blink phases), no window
    over an own panel, the generated chars in char RAM select.graphics' for the release row; snapshots per entry to
    out/phase12/snaps/task6/select-entryNN-<name>.png. Then Task 6 per game mode (`_mode_sweep`, the mode held at 0, 1,
    2): at each hold the three positions' layer-0 tiles are their entries' maps for that mode (an own panel's band
    its laps in that mode, an original's Sega's), layer 1 their windows; each own panel's course map is on screen
    (its MAP_BOXES box differs from the M5b playtest set's at the same hold, where those entries have no map, in
    every pixel row) and an original's map is Sega's (the box equal); snapshots per entry and mode to
    out/phase12/snaps/task7/select-entryNN-<name>-modeM.png. The select's frame counters (0x5010b0, 0x540058) and
    its state, entry, course and first position equal the M5b playtest set's on every frame from 600 to
    MODE_SWEEP_TO (mode 0), the frames with the own panels' maps on screen (the holds of entries 3-7) among them: the
    maps' polygons (Task 7's paintings, selmap.PAINT_MAX_DRAWN each) cost no frame."""
    full = _release(romset_zip, disc)
    assert [S.window_of(e) for e in S.ENTRIES[:3]] == [S.WINDOW_NAME] * 3
    assert [S.window_of(e) for e in S.ENTRIES[3:]] == [S.WINDOW_NONE] * 5
    _row_checks(tmp_path, romset_zip, full, S.ENTRIES, SNAP_DIR)
    row, g = S.select_row(full), S.graphics(full, S.ENTRIES)
    tiles = {S.slot_of(k, 8): r.tile for k, r in enumerate(row.records[0])}     # position -> its tile (x, y)
    blank = (S.LAYER1_BLANK,) * (C.PANEL_ROWS * C.PANEL_COLS)
    ref_log, ref_at, ref_shots = _mode_sweep(_m5b(romset_zip, disc), tmp_path / "m5b-mode0", romset_zip, 0)
    for mode in range(C.MODES):
        log, at, shots = _mode_sweep(full, tmp_path / f"mode{mode}", romset_zip, mode)
        assert at == ref_at
        if mode == 0:
            frames = sorted(ref_log["F"])
            assert frames == sorted(log["F"]) and frames[0] == 600 and frames[-1] == MODE_SWEEP_TO
            moved = [f for f in frames if log["F"][f][:5] != ref_log["F"][f][:5] or
                     log["F"][f][6:8] != ref_log["F"][f][6:8]]
            assert not moved, (moved[:4], [(log["F"][f], ref_log["F"][f]) for f in moved[:2]])
            assert set(ref_at[3:]) <= set(frames)                        # the own maps on screen, compared too
            print(f"\nframe counters at {MODE_SWEEP_TO}: {log['F'][MODE_SWEEP_TO][6:8]} (the M5b set's: "
                  f"{ref_log['F'][MODE_SWEEP_TO][6:8]})")
        for k, a in enumerate(at):
            first = S.first_of(k, 8)
            assert log["F"][a][:5] == (1, k, k, S.ENTRIES[k].course, first), (mode, k, log["F"][a])
            name = S.ENTRIES[k].name.lower().replace(" ", "-")
            shutil.copy(shots[a], SNAP_DIR / f"select-entry{k:02d}-{name}-mode{mode}.png")
            for s in range(3):
                e = first + s
                x, y = tiles[s]
                for layer, want in ((0, {TL.tile_map(full, m).tiles() for m in row.records[mode][e].maps}),
                                    (1, {g.name_maps[e] or blank})):
                    got = tuple(log["T"][(a, layer, y + r)][x + c] for r in range(C.PANEL_ROWS)
                                for c in range(C.PANEL_COLS))
                    assert got in want, (mode, k, s, layer)
                if S.ENTRIES[e].own_panel:
                    assert TL.tile_map(full, row.records[mode][e].maps[0]).tiles() == g.panels[e][mode][0]
                if mode == 0:
                    mine, ref = _box(shots[a], s), _box(ref_shots[a], s)
                    if S.ENTRIES[e].own_panel:
                        assert (np.abs(mine - ref).sum(-1) > 0).any(1).all(), (k, s)
                    else:
                        assert (mine == ref).all(), (k, s)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_originals_on_release_set(tmp_path, romset_zip, disc):  # noqa: F811
    """The Global Constraint on the release set: entries 0-2 from its select race as the stock set's panels 0-2 on
    stock MAME, call for call (tests/test_mirrors_mame.py `originals_as_stock`: the same geometry display list on every
    frame from RACE_FROM, the same render-rate writes and video mode, pixel-identical race snapshots), so the names'
    window and its draw branch add nothing to the race. M9 stage 3: through the Course Direction screen (NORMAL),
    against stock with its Mission Select step 52 frames later (autoplay DAYTONA_DIRECTION=late)."""
    from tests.test_mirrors_mame import originals_as_stock
    originals_as_stock(tmp_path, romset_zip, _release(romset_zip, disc), {**ROW8, "DAYTONA_DIRECTION": "normal"},
                       {"DAYTONA_DIRECTION": "late"})


# --- Phase 12 Task 4: the Extended title --------------------------------------------------------------------------

TITLE_SNAP_DIR = config.out_dir() / "phase12" / "snaps" / "task4"
"""The title's snapshots: attract-logo-<frame>.png (the release set; the reference title snapshot for Phase 13) and
attract-logo-<frame>-m5b.png (the M5b playtest set, the logo without the word); the gates' under gates/."""
ATTRACT_DUMPS = (3450, 6617)
"""A frame of each logo of the attract loop (title.LOGO_FRAMES: 3265-3668 and 6432-6835 from boot, no coin), the wipe
done: tile, char and palette RAM dumped, a snapshot."""
ATTRACT_TO = 6900
"""The attract loop's frames compared: boot, the loop (the demo race, Beginner's ranking, the V.R. demo, the logo)
twice, past the second logo's end."""
ATTRACT_COIN = ATTRACT_TO + 1
"""Then a coin: Circuit Select opens 15 frames later (autoplay.lua's DAYTONA_COIN_AT)."""
ATTRACT_SECONDS = (ATTRACT_COIN + 120) // 57 + 1
SELECT_STATE = 1                # 0x5043e2 on Circuit Select (select_log.lua's F lines)


def _attract(rs: RomSet, workdir: Path, romset_zip) -> dict:
    """One attract run of `rs` (custom MAME, autoplay.lua with no coin until ATTRACT_COIN, title_log.lua): {"lists":
    {frame: display list hash} from frame 1, "rate": render-rate rows, "counters": {frame: (state, 0x5010b0,
    0x540058)}, "dumps": {frame: (tile, char, palette RAM as u16 arrays)}, "snaps": {frame: snapshot path}}."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_custom_mame import _custom, _lists
    log, dumps = (workdir / "title.txt").resolve(), (workdir / "dumps").resolve()
    dumps.mkdir(parents=True)
    out = mame_build.run_set(rs, workdir, ATTRACT_SECONDS, romset_zip,
                             {"DAYTONA_COIN_AT": str(ATTRACT_COIN), "DAYTONA_GEO_FROM": "1", "DAYTONA_GEO_EVERY": "1",
                              "DAYTONA_SNAPS": ",".join(map(str, ATTRACT_DUMPS)), "DAYTONA_TITLE_LOG": str(log),
                              "DAYTONA_TITLE_DIR": str(dumps), "DAYTONA_TITLE_DUMPS": ",".join(map(str, ATTRACT_DUMPS)),
                              **ROW8}, logs=("geo", "rate"), mame=str(_custom()))
    lists, _ = _lists(out["geo"])
    out["geo"].unlink()
    counters = {}
    for line in log.read_text().splitlines():
        f = line.split()
        if f[0] == "C":
            counters[int(f[1])] = tuple(int(x, 16) for x in f[2:5])
    shots = sorted(Path(out["snap"]).rglob("*.png"))
    assert len(shots) >= len(ATTRACT_DUMPS)
    return dict(lists=lists, rate=runner.parse_rate(out["rate"]), counters=counters,
                dumps={f: tuple(np.fromfile(dumps / f"{f}.{k}", "<u2") for k in ("tile", "char", "pal"))
                       for f in ATTRACT_DUMPS},
                snaps=dict(zip(ATTRACT_DUMPS, shots)))


@pytest.mark.mame
@pytest.mark.custom_mame
def test_attract_says_extended(tmp_path, romset_zip, disc):  # noqa: F811
    """Task 4 on the release set (custom MAME, no coin until ATTRACT_COIN), against the M5b playtest set run the same
    way (tools/desert_city_playtest.py `m5b_set`: the chain without the names and the title):
    - at each logo frame (ATTRACT_DUMPS) tile RAM's layer 0 holds the title's map (title.MAP_WORD's copy, the word's
      cells `0x4000 | char`) on every cell the HUD does not write, layer 2 is the M5b set's, char RAM holds every char
      `title.logo` decodes (the word's in title.ROOM among them) and palette RAM every logo palette (pens 1-15; the
      word's 0x80 + block, title_palette's), and outside the word's chars and palette char and palette RAM are the M5b
      set's; the snapshot is `title.render` of the set pixel for pixel above the HUD (tile rows < 38) and differs from
      the M5b set's only inside the picked variant's box (snapshots to TITLE_SNAP_DIR, the reference title snapshot
      for Phase 13);
    - over the attract loop from boot to ATTRACT_TO (it never shows Circuit Select) every frame's geometry display
      list and render-rate row equal the M5b set's;
    - the state byte and the frame counters 0x5010b0 / 0x540058 equal the M5b set's on every frame of the run, through
      the coin and Circuit Select's opening (the select-open frame, state SELECT_STATE, the same): the boot's longer
      upload costs no frame."""
    from PIL import Image
    from dtools.arcade import title as TT
    rel, ref = _release(romset_zip, disc), _m5b(romset_zip, disc)
    TITLE_SNAP_DIR.mkdir(parents=True, exist_ok=True)
    got = {name: _attract(s, tmp_path / name, romset_zip) for name, s in (("release", rel), ("m5b", ref))}
    a, b = got["release"], got["m5b"]
    # the frames: lists, render rate, counters
    frames = range(1, ATTRACT_TO + 1)
    assert set(frames) <= set(b["lists"]) and set(frames) <= set(a["lists"])
    moved = [f for f in frames if a["lists"][f] != b["lists"][f]]
    assert not moved, moved[:8]
    ra, rb = ([r for r in x["rate"] if r[0] <= ATTRACT_TO] for x in (a, b))
    assert len(ra) == len(rb) == ATTRACT_TO and ra == rb
    assert a["counters"] == b["counters"] and max(a["counters"]) > ATTRACT_COIN + 60
    opened = min(f for f, c in a["counters"].items() if f > ATTRACT_COIN and c[0] == SELECT_STATE)
    assert opened - ATTRACT_COIN < 30 and all(c[0] != SELECT_STATE for f, c in a["counters"].items()
                                              if f <= ATTRACT_TO)
    print(f"\nCircuit Select opens at {opened}: counters {a['counters'][opened]} (the M5b set's "
          f"{b['counters'][opened]}); {len(set(b['lists'].values()))} distinct display lists over the attract loop")
    # the logo frames: RAM and snapshots
    lg, lg0 = TT.logo(rel), TT.logo(ref)
    g0 = lg.layers[0]
    title_map = np.array(g0.map.tiles()).reshape(g0.map.rows, g0.map.cols)
    word = (title_map & 0x4000) != 0
    nums = sorted({int(t) & TL.CHAR_MASK for t in title_map[word]})
    colours = sorted({0x80 + (c >> 7) for c in nums})
    assert word.any() and not (set(lg0.chars) & set(nums)) and set(nums) <= set(lg.chars)
    hud = np.zeros((TL.TILE_LAYER, TL.TILE_LAYER), bool)
    hud[TT.HUD_ROWS[0]:TT.HUD_ROWS[1], TT.HUD_COLS[0]:TT.HUD_COLS[1]] = True
    x0, y0, x1, y1 = TT.VARIANTS[TT.PICKED].box
    for f in ATTRACT_DUMPS:
        (tile, char, pal), (tile0, char0, pal0) = a["dumps"][f], b["dumps"][f]
        l0 = tile[:0x1000].reshape(64, 64)
        mine = l0[g0.row:g0.row + g0.map.rows, g0.col:g0.col + g0.map.cols]
        keep = ~hud[g0.row:g0.row + g0.map.rows, g0.col:g0.col + g0.map.cols]
        assert np.array_equal(mine[keep], title_map[keep]), f
        assert np.array_equal(tile[0x2000:0x3000], tile0[0x2000:0x3000]), f
        differ = np.flatnonzero(tile[:0x1000] != tile0[:0x1000])
        cells = {(g0.row + r) * 64 + g0.col + c for r, c in zip(*np.nonzero(word))}
        assert set(differ.tolist()) == cells, f
        chars = TL.decode_chars(char.astype("<u2").tobytes())
        for c, px in lg.chars.items():
            assert np.array_equal(chars[c], px), (f, hex(c))
        for colour, words in lg.palette.items():
            assert tuple(int(w) for w in pal[16 * colour + 1:16 * colour + 16]) == words[1:], (f, hex(colour))
        char_differ = {int(i) // 16 for i in np.flatnonzero(char != char0)}
        assert char_differ and char_differ <= set(nums), f
        pal_differ = {int(i) >> 4 for i in np.flatnonzero(pal != pal0)}
        assert pal_differ == set(colours), f
        shot, shot0 = (np.asarray(Image.open(x["snaps"][f]).convert("RGB")) for x in (a, b))
        top = 8 * TT.HUD_ROWS[0]
        assert np.array_equal(shot[:top], TT.render(rel, lg)[:top]), f
        assert np.array_equal(shot0[:top], TT.render(ref, lg0)[:top]), f
        ys, xs = np.nonzero((shot != shot0).any(-1))
        assert len(ys) > 1000 and ys.min() >= y0 and ys.max() < y1 and xs.min() >= x0 and xs.max() < x1, f
        shutil.copy(a["snaps"][f], TITLE_SNAP_DIR / f"attract-logo-{f}.png")
        shutil.copy(b["snaps"][f], TITLE_SNAP_DIR / f"attract-logo-{f}-m5b.png")


_SHORT: dict = {}


def _release_short(romset_zip, disc, title: bool = True) -> RomSet:  # noqa: F811
    """release_set(..., long_ids=(), times=False, direction=False) of the player's set and disc (the gates' set: the
    Phase 11 gates' frozen frames; the Course Direction screen only moves the race, test_race_after_screen_as_stock),
    with or without the title (release_set's test-only `title`), built once per session."""
    key = (str(romset_zip), title)
    if key not in _SHORT:
        _SHORT[key] = release_set(RomSet.from_zip(romset_zip), disc, long_ids=(), title=title, times=False,
                                  direction=False)
    return _SHORT[key]


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("course", (3, 4, 5, 6, 7))
def test_release_gates(tmp_path, romset_zip, disc, course):  # noqa: F811
    """Each converted course's Phase 11 gate (tests/test_courses_mame.py `_gate`, test_national_park_id4's checks:
    the long-time race and its frozen numbers, the time-over session, then Beginner) with every number unchanged, on
    the release set without its race times (`release_set(rs, disc, long_ids=(), times=False)`: the converted courses
    keep Beginner's words, on which the time-over frames were measured; `_gate` raises the start itself for its
    long-time run; tests/test_release.py test_release_race_times proves the time tables are the only difference from
    the release set, and test_race_times_driven races the release set's own times),
    entry `course` of the release row; Circuit Pixie (5) and Rin Rin Rink (6) also read their own views on every frame
    of the lap (`outside_view`). Snapshots to out/phase12/snaps/task4/gates/.

    One relaxation (the controller's ruling, Phase 12 Task 4 fix round 1): in the time-over session the per-car split
    of the ground calls may move by 1 per car, the sum exact, and the session's ground log must equal the untitled
    release set's (`release_set(..., title=False)`) call for call and value for value (`_gate`'s `untitled`): the
    title's boot upload shifts the i960 a fraction of a frame against MAME's frame_done, which moves one call between
    cars 1 and 4 on National Park Speedway with the game's work unchanged. Phase 11's gates on m5b_set stay exact."""
    from dtools.arcade import title as TT
    from dtools.convert import build
    from tests.test_courses_mame import (_gate, desert_gate, lake_gate, outside_view, park_gate, pixie_gate,
                                         rink_gate)
    gates = {3: desert_gate, 4: park_gate, 5: pixie_gate, 6: rink_gate, 7: lake_gate}
    full = _release_short(romset_zip, disc)
    untitled = _release_short(romset_zip, disc, title=False)
    assert TT.titled(full) and not TT.titled(untitled)
    assert S.ENTRIES[course].course == course and S.select_row(full).live[course]
    g = gates[course](str(TITLE_SNAP_DIR / "gates"))
    assert g.course == g.entry == course
    seen = _gate(tmp_path, romset_zip, full, g, untitled=untitled)
    if course in (5, 6):
        outside = outside_view(seen, build.VIEW[course])
        assert not outside, outside[:8]


RACE_TIMES_MARGIN = {3: 4260, 4: 4170, 5: 1452, 6: 3782, 7: 4056,
                     8: 1848, 9: 204, 10: 1727, 11: 5036, 12: 4108, 13: 769, 14: 3600, 15: 4038}
"""Ticks left on the clock when the autopilot's player car crosses the lap line at the end of lap 1 (before the lap
line's word is added), per converted course, on the release set (test_race_times_driven, measured 15 Nov 2026 with GO
at frame 1651; re-measured 18 Nov 2026 through the Course Direction screen, GO at 1703 on all five: 4257, 3977, 3708
and 4157 then. The release set without the screen with its Mission Select step 52 frames later gives these same
margins and checkpoint frames: the AI field, which depends on the race's start frame, meets the autopilot elsewhere;
the time tables are unchanged). Ids 8-15 (M9 stage 4) measured 18 Nov 2026 from the screen with REVERSE, GO at 1703
(Three Seven Speedway in reverse 1937, Sea-Side Street Galaxy in reverse 1704). The autopilot laps about 25 % slower than the AI field's median (Desert City's lap 1: 4,497
frames against the field's 3,569), so it is no reference for a whole race: docs/spike/2026-11-race-times.md."""


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("course", (3, 4, 5, 6, 7) + tuple(range(8, 16)))
def test_race_times_driven(tmp_path, romset_zip, disc, course):  # noqa: F811
    """v0.1.2 (docs/superpowers/specs/2026-11-15-race-times-design.md, 7.5): on the release set, mode 0 difficulty 1,
    the autopilot drives entry `course` (the player car on the course's road line, dtools.mame.road.road_file) past the
    lap line of lap 1 with time left: the clock reads word 0 of build.RACE_TIMES at GO and counts down one tick a frame;
    at every checkpoint the player reaches on lap 1, then the lap line, it rises RISE_AFTER frames later by that pass's
    word (less the frame's tick); it never reaches 0 before the lap line. The ticks left at the lap line are printed and
    must equal RACE_TIMES_MARGIN[course] once recorded. M9 stage 4: ids 8-15 too, each entered from the Course
    Direction screen with REVERSE (the entry of course n - 8), the autopilot on id n's own road line."""
    import math
    from dtools.arcade import golden as G
    from dtools.convert import build
    from dtools.mame import build as mame_build, runner
    from dtools.mame.road import road_file
    from tests.test_convert_mame import RISE_AFTER
    from tests.test_custom_mame import _custom
    exe = _custom()
    rel = _release(romset_zip, disc)
    entry = next(k for k, e in enumerate(S.ENTRIES) if e.course == course % 8)
    c = G.decode_course(rel, course)
    words = c.time_tables[(C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)][2]
    assert words == build.race_table(*build.RACE_TIMES[course][(C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)], len(words))
    passes = len(c.checkpoints)                         # the checkpoints, then the lap line
    seconds = math.ceil(50 + sum(words[:passes + 1]) / 57.524)
    road = road_file(c, tmp_path / "road.txt")
    env = {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": str(entry), "DAYTONA_AUTOPILOT": str(road),
           "DAYTONA_AUTOPILOT_LAPS": "99", "DAYTONA_SNAPS": "0",
           "DAYTONA_DIRECTION": "reverse" if course >= 8 else "normal"}
    out = mame_build.run_set(rel, tmp_path, seconds, romset_zip, env, logs=("progress",), mame=exe)
    rows, timer = runner.parse_progress(out["progress"])
    go = next(f for f in sorted(timer) if f - 1 in timer and timer[f] < timer[f - 1])
    assert timer[go - 1] == words[0] and timer[go] == words[0] - 1, (timer[go - 1], timer[go])
    player = [r for r in rows if r.car == 0 and r.frame >= go]
    reached = []
    for cp in c.checkpoints:                            # lap 1's checkpoints, then the lap line (lap 1 -> 2)
        f = next((r.frame for r in player if r.segment == cp.segment and r.lap == (cp.segment == 0)), None)
        assert f is not None, (course, cp.segment, "the player never reached it")
        reached.append((f, cp))
    end = reached[-1][0] + RISE_AFTER
    assert all(timer[f] > 0 for f in range(go, end + 1)), (course, "time over before the lap line")
    rises = [f for f in range(go, end + 1) if timer[f] > timer[f - 1]]
    assert all(timer[f] == timer[f - 1] - 1 for f in range(go, end + 1) if f not in rises)
    assert len(rises) == len(reached), (rises, reached)
    for j, (f, (r, cp)) in enumerate(zip(rises, reached)):
        assert f - r == RISE_AFTER, (f, r, cp.segment)
        assert timer[f] - timer[f - 1] == words[1 + j] - 1, (course, j, timer[f] - timer[f - 1], words[1 + j])
    margin = timer[end - 1]
    print(f"\ncourse {course}: GO {go}, passes at {[f for f, _ in reached]}, ticks left at the lap line {margin} "
          f"({margin / 64:.1f} s on the HUD)")
    if course in RACE_TIMES_MARGIN:
        assert margin == RACE_TIMES_MARGIN[course], margin


# ---- the pit stops (v0.1.2) ----

PIT_SECONDS = 170
"""Emulated seconds per pit run: GO (frame 1703 through the Course Direction screen; 1651 before), the autopilot's first lap to the box (Desert City's lap 1 alone is
4,497 frames), then the stop (1,249-1,351 frames) and the 600 frames of driving after it the test reads (140 s ended
inside Desert City's stop)."""
PIT_AHEAD = {8: 10}
"""The autopilot's look-ahead in segments (DAYTONA_AUTOPILOT_AHEAD; 6 otherwise): Three Seven Speedway in reverse enters
its pit lane on a curve that ends at the box, and at 6 (or 2, 3) the car runs wide of the box polygon (z -72..-52) on
every lap, at z -70..-80; at 10 it drives onto it (the stop 1,346 frames)."""
PIT_SECONDS_LONG = {9: 205}
"""Longer pit runs: Sea-Side Street Galaxy in reverse reaches its box at frame 9334 (the box late in a 991-segment lap), so the
stop and the 600 frames after it end near frame 11,300 (170 s ended inside the stop at 9780)."""
STALL_REACH = 2.0
"""World units: how far the stopped car may stand from its stall's last waypoint with a speed (stall 0's: the
waypoint before the speed-0 one, which the car aims at while it brakes and never reaches)."""


def _pit_rows(path: Path) -> list[dict]:
    """pit_log.lua's P lines as dicts (frame, handler, seg, flags, timer, pit, x, z)."""
    out = []
    for line in Path(path).read_text().splitlines():
        w = line.split()
        if w and w[0] == "P":
            out.append({"frame": int(w[1]), "handler": int(w[2], 16), "seg": int(w[3]), "flags": int(w[4], 16),
                        "x": float(w[15]), "z": float(w[17]), "timer": int(w[18]), "pit": int(w[19])})
    return out


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("course", (0, 3, 4, 5, 6, 7) + tuple(range(8, 16)))
def test_pit_stop_driven(tmp_path, romset_zip, disc, course):  # noqa: F811
    """v0.1.2: on the release set the autopilot drives entry `course` into the pit lane (dtools.mame.road.pit_road_file:
    the pit polygons' centres on the lane's segments) and the pit probe (pit_log.lua) logs the player car every frame.
    The first stop: the trigger 0xdf9c starts it on the box polygon (handler 0xe288, the step's STEP_PIT_BOX flag, on
    the pit's box segment); the stop 0xe384 then runs without a break until it ends with the stop counted out: the pit
    word 0x501a8c set to 640, held while the car drives to its stall, counted down one a frame to 1 (0xe598), the
    race timer held the whole time (it rises only by a time word when the car crosses the lap line or a checkpoint in
    the stop: the lap line lies in the pit lane on National Park Speedway, Circuit Pixie and Mermaid Lake), the car
    standing within STALL_REACH of stall 0's last waypoint with a speed
    (course.pit_stalls: a single cabinet is car ident 0); after it the car drives on (back to 0xdc64, at least 10
    segments in 600 frames). Beginner (entry 0) is the control: the arcade's own stalls. Before v0.1.2 every converted
    course read Beginner's stalls and the stop ended 7 frames after the box (spike/pit_drive.py; National Park Speedway:
    0xe384 at 5000-5006, the pit word never counted). M9 stage 4: ids 8-15 too, each entered from the Course
    Direction screen with REVERSE (entry n - 8), on its own stalls (build.pit_stalls(..., "R"))."""
    from dtools.arcade import golden as G, tgp
    from dtools.mame import build as mame_build
    from dtools.mame.road import pit_road_file
    from tests.test_custom_mame import _custom
    exe = _custom()
    rel = _release(romset_zip, disc)
    c = G.decode_course(rel, course)
    g = c.ground_data or tgp.decode_ground(rel, course)
    stall = C.pit_stalls(rel, course)[0]
    last = max(j for j, w in enumerate(stall) if w.speed > 0)
    road = pit_road_file(c, g, tmp_path / "pit_road.txt")
    entry = next(k for k, e in enumerate(S.ENTRIES) if e.course == course % 8)
    env = {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": str(entry), "DAYTONA_AUTOPILOT": str(road),
           "DAYTONA_AUTOPILOT_LAPS": "99", "DAYTONA_SNAPS": "0",
           "DAYTONA_DIRECTION": "reverse" if course >= 8 else "normal",
           **({"DAYTONA_AUTOPILOT_AHEAD": str(PIT_AHEAD[course])} if course in PIT_AHEAD else {})}
    out = mame_build.run_set(rel, tmp_path, PIT_SECONDS_LONG.get(course, PIT_SECONDS), romset_zip, env, logs=("pit",),
                             mame=exe)
    rows = _pit_rows(out["pit"])
    at = {r["frame"]: r for r in rows}
    start = next((r for r in rows if r["handler"] == 0xE288), None)
    assert start is not None, (course, "no pit stop started")
    assert start["flags"] & C.STEP_PIT_BOX and start["seg"] == c.pit.box_segment, (course, start)
    f0 = start["frame"]
    stop = []
    f = f0 + 1
    while at[f]["handler"] == 0xE384:
        stop.append(at[f])
        f += 1
    after = [r for r in rows if r["frame"] >= f]
    assert stop and stop[0]["pit"] == 640, (course, f0, stop[:2])
    counting = [r for r in stop if r["pit"] < 640]
    assert [r["pit"] for r in counting] == list(range(639, 639 - len(counting), -1)), (course, len(stop))
    assert counting and counting[-1]["pit"] in (1, 0), (course, len(stop), counting[-1:] if counting else None)
    assert after[0]["handler"] == 0xDC64 and after[0]["pit"] == 1, (course, after[0])
    words = set(c.time_tables[(C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)][2])
    steps = [b["timer"] - a["timer"] for a, b in zip(stop, stop[1:])]
    assert all(d == 0 or d in words for d in steps), (course, "the race timer ran during the stop",
                                                      sorted(set(steps)))
    still = {(round(r["x"], 3), round(r["z"], 3)) for r in counting}
    assert len(still) == 1, (course, "the car moved while the stop counted", sorted(still)[:4])
    x, z = still.pop()
    gap = float(np.hypot(x - stall[last].x, z - stall[last].z))
    assert gap <= STALL_REACH, (course, (x, z), stall[last], gap)
    later = [r for r in after if r["frame"] <= f + 600]
    moved = (later[-1]["seg"] - after[0]["seg"]) % len(c.road)
    assert all(r["handler"] == 0xDC64 for r in later) and moved >= 10, (course, moved)
    print(f"\ncourse {course}: box at frame {f0}, the stop {len(stop)} frames ({len(counting)} standing at "
          f"({x:.3f}, {z:.3f}), {gap:.2f} from stall 0's waypoint {last}), timer held at {stop[0]['timer']}")
