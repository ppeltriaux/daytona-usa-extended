"""MAME tests of the grown set (course.grow_ids: course ids 0-15, every per-id array moved to a sixteen-entry table
in the custom data area). Review Focus 1 and 2 of the Phase 8 plan; docs/spike/2026-11-sixteen.md, "Ids".

Marked `mame` and `custom_mame` (skipped when dtools.config.custom_mame() finds no build of the current patch)."""

import struct
from pathlib import Path

import pytest

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import trace
from dtools.arcade.romset import RomSet
from tests.test_custom_mame import GEO_FROM, SECONDS, SNAPS, _lists, _snaps

TRACE_SECONDS = 180         # boot, attract, the selects, the race, its time-out, game over and the ranking screen (~9824)
OLD_RECORD = (0x02600080, 0x0260009F)   # the original id-3 stub record (Beginner's line table follows at 0x026000a0)
SELECT_ID_AT = 0x00238634   # mode table 0, panel 0's course id byte: 0x226c74 turns the panel into the id
COURSE_ID = 0x00501460
VOTE_IP = 0x32E4            # the trace ip of the course vote's store (0x32dc stob g2,0x501460)


def _custom() -> str:
    exe = config.custom_mame()
    if exe is None:
        pytest.skip("the custom MAME is not built (tools/build-mame.sh)")
    return str(exe)


def old_ranges() -> list[tuple[int, int]]:
    """Inclusive i960 ranges, 4-aligned, over every original per-id array grow_ids moves (GROWN_ARRAYS,
    IdArray.old_span) and the stub record, merged where they touch (one tap per range: memtrace.lua), less exactly the
    bytes OLD_ARRAY_READERS name (4-aligned words, so a read of them does not fire the tap and any other read of the
    array still does). WIDE_ARRAYS are left out: a set that is not widened reads them where they were."""
    spans = sorted([(a.old_span[0] & ~3, (a.old_span[0] + a.old_span[1] - 1) | 3) for a in C.GROWN_ARRAYS.values()]
                   + [OLD_RECORD])
    out: list[tuple[int, int]] = []
    for lo, hi in spans:
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    for _, at, n, _ in C.OLD_ARRAY_READERS.values():
        assert at % 4 == 0 and n % 4 == 0
        cut = []
        for lo, hi in out:
            if lo <= at and at + n - 1 <= hi:
                cut += [r for r in ((lo, at - 1), (at + n, hi)) if r[0] <= r[1]]
            else:
                cut.append((lo, hi))
        out = cut
    return out


def _new_range() -> tuple[int, int]:
    end = max(a.new + C.COURSE_IDS * a.stride for a in C.GROWN_ARRAYS.values())
    return C.ID_HOME, (end - 1) | 3


READ_IPS = {"TIME_TABLES[0]": 0x45F0, "LAP_COUNTS[0]": 0x45F4, "CAR_COUNTS[0]": 0x45F8, "CAMERA_SCRIPTS": 0x641C,
            "HUD_MAP_PLACES": 0x12EFC, "TUNNEL_LIGHTS": 0x6D2C, "SELECT_PALETTES": 0x226C38,
            "RANKING_SCREEN_2273E8": 0x227264}
"""Trace ips of the reads of arrays whose site is not the displacement of the reading load: the per-id levels the
course load reaches through a pointer word (0x45ec-0x45f4 `ld (gN)[r10*4]`) and the arrays an `lda` addresses for a
later `ld` (0x6418, 0x12ef8, 0x6d28, 0x226c34, 0x227260)."""


def _arrays_read(reads) -> set[str]:
    """The GROWN_ARRAYS the trace shows read in the new home, from its (ip, page) pairs: an 8-byte MEMB load whose
    displacement is a site logs ip = site + 4 (the address after the instruction); a READ_IPS reader must read the
    array's page."""
    out = set()
    for n, a in C.GROWN_ARRAYS.items():
        for ip, page in reads:
            if any(s + 4 == ip for s in a.sites) or (READ_IPS.get(n) == ip and page <= a.new < page + 0x100):
                out.add(n)
    return out


def _trace(rs: RomSet, workdir: Path, romset_zip: Path, course: int, exe: str, label: str):
    from dtools.mame import build as mame_build
    log = (workdir / f"trace-{label}.txt").resolve()
    spec = ",".join(f"{lo:x}-{hi:x}" for lo, hi in old_ranges() + [_new_range()])
    mame_build.run_set(rs, workdir / f"trace-{label}", TRACE_SECONDS, romset_zip,
                       {"DAYTONA_COURSE": str(course), "DAYTONA_TRACE_LOG": str(log), "DAYTONA_TRACE_READ": spec,
                        "DAYTONA_TRACE_WRITE": f"{COURSE_ID:x}-{COURSE_ID + 3:x}!"}, mame=exe)
    return trace.parse_trace(log)


def _check_trace(events, what: str, want: int) -> set[str]:
    """The course vote stores id `want` (the race runs as that id); no read of an original array (old_ranges leaves
    out only OLD_ARRAY_READERS' bytes), some read of the new home; the arrays read there."""
    votes = [e.value for e in events if e.kind == "V" and e.addr == COURSE_ID and e.ip == VOTE_IP]
    assert votes and set(votes) == {want}, (what, votes)
    events = [e for e in events if e.kind == "R"]
    lo, hi = _new_range()
    old = [e for e in events if not lo <= e.addr <= hi]
    assert not old, (what, sorted({(hex(e.ip), hex(e.addr)) for e in old})[:20])
    reads = {(e.ip, e.addr) for e in events if lo <= e.addr <= hi}
    assert reads, what
    read = _arrays_read(reads)
    print(f"\n{what}: {len(read)} of {len(C.GROWN_ARRAYS)} arrays read in the new home: {sorted(read)}")
    return read


@pytest.mark.mame
@pytest.mark.custom_mame
def test_grown_originals(tmp_path, romset_zip):
    """Review Focus 1 and 2. Beginner, Expert and Advanced (DAYTONA_COURSE 0, 2, 1) on the grown set (custom build)
    race call for call as on the stock set (stock MAME): the same geometry display list on every frame from GEO_FROM,
    the same render-rate log, pixel-identical snapshots (SECONDS each, as test_custom_mame). Then a memtrace over every
    original per-id array (less the OLD_ARRAY_READERS bytes), the stub record and the new home, TRACE_SECONDS on each
    course and on id 3 (panel 0's course byte set to 3, so Beginner's panel picks id 3, Beginner's values), with the
    writes to the id byte: the course vote (0x32ac, ID_PATCHES) stores the picked id, 3 included; no read of an
    original array, and reads in the new home (the positive control: the custom data area is only reachable through
    the patched readers); the course-load arrays among them on every run."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    exe = _custom()
    stock = RomSet.from_zip(romset_zip)
    grown = C.grow_ids(stock)
    env = {"DAYTONA_GEO_FROM": str(GEO_FROM), "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": SNAPS}
    for course in (0, 2, 1):
        got = {}
        for name, s, binary in (("stock", None, "mame"), ("grown", grown, exe)):
            out = mame_build.run_set(s, tmp_path / f"{course}-{name}", SECONDS, romset_zip,
                                     {"DAYTONA_COURSE": str(course), **env}, logs=("geo", "rate"), mame=binary)
            lists, _ = _lists(out["geo"])
            out["geo"].unlink()                                 # about 600 MB each
            got[name] = lists, runner.parse_rate(out["rate"]), _snaps(out["snap"])
        lists, rate, snaps = got["stock"]
        assert set(lists) == set(range(GEO_FROM, max(lists) + 1)) and max(lists) >= SECONDS * 57, course
        assert len(rate) > SECONDS * 57 and len(snaps) == len(SNAPS.split(",")) + 1, course
        assert len(set(lists.values())) > 1000, course
        g_lists, g_rate, g_snaps = got["grown"]
        assert g_lists == lists, (course, min(f for f in lists if g_lists.get(f) != lists[f]))
        assert g_rate == rate, course
        assert g_snaps == snaps, (course, [k for k, (a, b) in enumerate(zip(g_snaps, snaps)) if a != b])
    load = {"COURSE_POINTERS", "COURSE_SHEETS", "ACTOR_TABLES", "GRID_TABLES", "SEGMENT_FLAGS", "CHECKPOINT_TABLES",
            "GRID_BASES", "ROLLING_STARTS", "TIME_TABLES[0]", "LAP_COUNTS[0]", "CAR_COUNTS[0]", "HUD_MAP_LISTS",
            "HUD_MAP_PLACES", "HUD_MAP_ZOOMS", "CAMERA_SCRIPTS"}
    id3 = grown.with_patch("program", SELECT_ID_AT - 0x200000, b"\x03")
    for course, s, what, want in ((0, grown, "Beginner", 0), (2, grown, "Expert", 1), (1, grown, "Advanced", 2),
                                  (0, id3, "id3", 3)):
        read = _check_trace(_trace(s, tmp_path, romset_zip, course, exe, what), what, want)
        assert load <= read, (what, sorted(load - read))


# --- the sixteen-entry select (Task 4) -----------------------------------------------------------------------------

SWEEP_FROM, SWEEP_HOLD = 640, 30        # the select opens at 615 (coin at 600); one entry every 30 frames
LOCKED_STEP = 1130                      # a gas step on entry 15 (locked) after the sweep
SELECT_SECONDS = 180                    # the whole run: boot, attract, the selects, the race, game over (RAM proof)
RACE_FROM = 1150                        # after Mission Select (its 3D maps end at state 12, ~1092): the race
RACE_SNAPS = "1400,2000,2500"           # the race (and MAME's last snapshot)
RATE_HASH_FROM = 1400
"""renderrate.lua hashes 256 words from the display list's start whatever the list's length, so while the race intro's
lists are short the hash also covers words the select's lists left in the buffer (Expert from the select: up to frame
1329). The writes and the video mode are compared from RACE_FROM, the whole line from here."""
SNAP_DIR = Path(__file__).resolve().parent.parent / "out" / "phase8" / "snaps" / "task4"
VOTE_IPS = {0x3058, 0x3078, 0x3080, 0x3088, 0x32C4, 0x3508, 0x3510, 0x3520, 0x3528, 0x3538}
"""Trace ips of the course vote's accesses (0x3048-0x3080 the clear and priorities, 0x32ac the vote, 0x3470 the
count): the instructions at ip - 4 or ip - 8."""
BOOT_CLEAR_IP = 0x1798                  # 0x1794 stq: the boot clears 0x500000-0x5fefff (frame 0)
STEP_SHOWN, WIPE_AT = 850, 900          # a step at 760: state 3 (names up) at 850; state 5 (the wipe) 893-912


def _mask_rows(windows, first: int, tiles) -> tuple[list[int], list[int]]:
    """The window mask words (four u16) of the name rows (80-119) and of the label rows (120-143) when entries
    first..first+2 are shown (select.graphics' windows): each position's interior where its window covers those
    rows (Phase 9 Task 5: a generated panel all eight, a mirror of an original the label rows; Phase 12 Task 2: an
    original under WINDOW_NAME the name rows, never the label rows)."""
    from dtools.arcade import select as S
    name, label = [0, 0, 0, 0], [0, 0, 0, 0]
    for s in range(3):
        w = windows[first + s]
        bits = struct.unpack("<4H", S.slot_mask(tiles[s][0]))
        if w in (S.WINDOW_ALL, S.WINDOW_NAME):
            name = [a | b for a, b in zip(name, bits)]
        if w in (S.WINDOW_ALL, S.WINDOW_LABEL):
            label = [a | b for a, b in zip(label, bits)]
    return name, label


def parse_select_log(path: Path) -> dict:
    """select_log.lua's lines: F per frame (state, entry, cursor, course id, first, timer, the frame counters
    0x5010b0 and 0x540058: index or slice them, the line may grow), T tiles per (frame,
    layer, row), C chars, P palettes, A watched accesses ((kind, ip, address, mask) -> (first frame, count))."""
    out = {"F": {}, "T": {}, "M": {}, "C": None, "P": {}, "A": {}, "end": None}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if p[0] == "F":
            out["F"][int(p[1])] = tuple(int(v, 16) for v in p[2:])
        elif p[0] == "T":
            out["T"][(int(p[1]), int(p[2]), int(p[3]))] = tuple(int(p[4][i:i + 4], 16) for i in range(0, 256, 4))
        elif p[0] == "M":
            words = [int(p[2][i:i + 4], 16) for i in range(0, len(p[2]), 4)]
            out["M"][int(p[1])] = [tuple(words[4 * r:4 * r + 4]) for r in range(len(words) // 4)]
        elif p[0] == "C":
            out["C"] = bytes.fromhex(p[2])
        elif p[0] == "P":
            out["P"][int(p[2], 16)] = tuple(int(p[3][i:i + 4], 16) for i in range(0, len(p[3]), 4))
        elif p[0] == "A":
            out["A"][(p[1], int(p[2], 16), int(p[3], 16), int(p[4], 16))] = (int(p[5]), int(p[6]))
        elif p[0] == "END":
            out["end"] = int(p[1])
    assert out["end"], f"{path}: no END line"
    return out


def _votes(trace_log: Path) -> list[int]:
    return [e.value for e in trace.parse_trace(trace_log) if e.kind == "V" and e.addr == COURSE_ID and e.ip == VOTE_IP]


def _select_run(rs, workdir: Path, romset_zip: Path, seconds: int, env: dict, exe: str):
    from dtools.mame import build as mame_build
    log, tlog = (workdir / "select.txt").resolve(), (workdir / "trace.txt").resolve()
    out = mame_build.run_set(rs, workdir, seconds, romset_zip,
                             {"DAYTONA_SELECT_LOG": str(log), "DAYTONA_TRACE_LOG": str(tlog),
                              "DAYTONA_TRACE_WRITE": f"{COURSE_ID:x}-{COURSE_ID + 3:x}!", **env}, mame=exe)
    return parse_select_log(log), _votes(tlog), out["snap"]


@pytest.mark.mame
@pytest.mark.custom_mame
def test_select_scrolls(tmp_path, romset_zip, monkeypatch):
    """Review Focus 3 and the vote move. On the grown set with the select (select.patch_select, M4A_ENTRIES: Phase 8's
    row, the mirrors locked), the wheel
    steps through the sixteen entries (one every SWEEP_HOLD frames from SWEEP_FROM, select.paddle_for_entry): at each
    the game holds entry k (0x501468, 0x5043de), its course id (0x501460) and first_of(k) (0x501b24), and tile layers
    0 and 1 hold, at the three positions, the panel maps and name maps of entries first..first+2 (the tiles the
    snapshot shows; the names' chars and palettes are in char and palette RAM). Snapshots per entry go to SNAP_DIR.
    A gas step on entry 15 (locked) changes nothing (state 1, the timer runs on); the timer's end moves the choice to
    entry 0 (Beginner: Phase 9 Task 5's idle-wheel ruling; Phase 8 moved it to the nearest live entry on the left,
    Desert City) and the vote stores id 0, never a locked id. The run lasts
    SELECT_SECONDS (boot to game over) with every access to select.SELECT_RAM watched: only the boot's clear, the
    vote's instructions and the select code touch it, and only the vote's bytes and FIRST_RAM. Then a set whose entry
    15 is live (id 15): DAYTONA_ENTRY 15 starts it (state 2 at the step) and the vote stores 15 (ids above 3 survive
    the moved vote)."""
    import shutil
    import numpy as np
    from PIL import Image
    from dtools.arcade import select as S
    from dtools.arcade import tiles as TL
    exe = _custom()
    grown = C.guard_new_ids(C.grow_ids(RomSet.from_zip(romset_zip)))     # live new ids need the guards
    from tests.test_select import unconverted_select
    rs = unconverted_select(grown)      # Phase 8's row (the mirrors locked: the set has none; id 3 not converted)
    row, gr = S.select_row(rs), S.graphics(grown, S.M4A_ENTRIES)
    lay = S.layout_of(rs)
    tiles = [C.decode_select_panel(grown.read_i960(C.SELECT_TABLES[0] + 44 * s, 44), s).tile for s in range(3)]
    frames = [SWEEP_FROM + SWEEP_HOLD * k for k in range(16)]
    at = [f + SWEEP_HOLD - 4 for f in frames]
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{S.paddle_for_entry(k)}" for k, f in enumerate(frames)),
           "DAYTONA_SNAPS": ",".join(map(str, at)), "DAYTONA_STEPS": str(LOCKED_STEP), "DAYTONA_GAS_FROM": "2000",
           "DAYTONA_SELECT_FROM": "600", "DAYTONA_SELECT_TO": "2400", "DAYTONA_SELECT_TILES": ",".join(map(str, at)),
           "DAYTONA_SELECT_CHARS": "700",
           "DAYTONA_SELECT_WATCH": f"{S.SELECT_RAM[0]:x}-{S.SELECT_RAM[1] - 1:x}"}
    log, votes, snap = _select_run(rs, tmp_path / "sweep", romset_zip, SELECT_SECONDS, env, exe)
    assert log["end"] >= SELECT_SECONDS * 57
    # the row: per entry, the RAM and the tiles at the three positions
    blank = (S.LAYER1_BLANK,) * 180
    for k, f in enumerate(at):
        first = S.first_of(k)
        assert log["F"][f][:5] == (1, k, k, S.M4A_ENTRIES[k].course, first), (k, log["F"][f])
        for s in range(3):
            e = first + s
            x, y = tiles[s]
            maps = {m for recs in row.records for m in recs[e].maps}       # the game's mode and variant
            for layer, want in ((0, [TL.tile_map(rs, m).tiles() for m in maps]), (1, [gr.name_maps[e] or blank])):
                got = tuple(log["T"][(f, layer, y + r)][x + c] for r in range(10) for c in range(18))
                assert got in [tuple(w) for w in want], (k, s, layer)
        # the window mask (what makes layer 1 visible): the rows each position's window covers
        name, label = _mask_rows(gr.windows, first, tiles)
        assert log["M"][f] == [tuple(name)] * 40 + [tuple(label)] * 24, (k, log["M"][f][0], log["M"][f][-1])
    assert any(log["M"][f][0] != (0, 0, 0, 0) for f in at) and log["M"][at[0]][0] == (0, 0, 0, 0)
    for first, n in gr.runs:                           # the uploaded chars (the rest of a block is never written)
        at = 32 * (first - S.CHAR_HOLE)
        assert log["C"][at:at + 32 * n] == TL.encode_chars(gr.chars[first - S.CHAR_HOLE:first - S.CHAR_HOLE + n])
    assert log["P"][0x760] == sum(gr.palettes, ()) and log["P"][0xF60] == gr.grey * S.BLOCKS
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    shots = sorted(Path(snap).rglob("*.png"))
    assert len(shots) >= 16
    for k, p in enumerate(shots[:16]):
        name = S.M4A_ENTRIES[k].name.lower().replace(" ", "-") + ("-mirror" if S.M4A_ENTRIES[k].mirror else "")
        shutil.copy(p, SNAP_DIR / f"entry{k:02d}-{name}.png")
    # the locked step and the timer's fallback
    after = [log["F"][f] for f in range(LOCKED_STEP, LOCKED_STEP + 20)]
    assert all(v[0] == 1 and v[1] == 15 for v in after) and after[-1][5] < after[0][5]
    first2 = min(f for f, v in log["F"].items() if f > LOCKED_STEP and v[0] == 2)
    assert log["F"][first2][1:4] == (0, 0, 0), log["F"][first2]
    assert votes and set(votes) == {0}, votes
    # the RAM proof: who touches SELECT_RAM, where
    code_end = S.CODE_AT + len(S.code(lay).bytes())
    vote_bytes = range(S.VOTE_RAM, S.VOTE_RAM + 1 + 2 * 16)
    for (kind, ip, addr, mask), (first_frame, n) in log["A"].items():
        if ip == BOOT_CLEAR_IP:
            assert kind == "W" and first_frame == 0
            continue
        assert ip in VOTE_IPS or S.CODE_AT < ip <= code_end, (kind, hex(ip), hex(addr))
        lanes = [addr + i for i in range(4) if mask >> (8 * i) & 0xFF]
        assert all(b in vote_bytes or b in (S.FIRST_RAM, S.FOLLOW_RAM) for b in lanes), (kind, hex(ip), hex(addr),
                                                                                          hex(mask))
    assert any(ip in VOTE_IPS for (_, ip, _, _) in log["A"]) and any(ip > S.CODE_AT for (_, ip, _, _) in log["A"])
    (SNAP_DIR / "select_ram_watch.txt").write_text(
        "".join(f"{k} {hex(ip)} {hex(a)} {m:#010x} first {ff} count {n}\n"
                for (k, ip, a, m), (ff, n) in sorted(log["A"].items(), key=lambda kv: (kv[1][0], kv[0][1]))))
    # a live entry past the originals: id 15 through the moved vote
    live15 = S.M4A_ENTRIES[:15] + (S.M4A_ENTRIES[15]._replace(live=True),)
    # Phase 9 Task 3's patch_select refuses a live id 4-15 the set holds no course at (select._unfilled); this run is
    # about the vote's path for an id above 3 only (it races Beginner's copy at id 15), so the refusal is bypassed
    monkeypatch.setattr(S, "_unfilled", lambda rs, entries: [])
    rs15 = S.patch_select(grown, live15)
    log15, votes15, snap15 = _select_run(rs15, tmp_path / "live15", romset_zip, SECONDS,
                                         {"DAYTONA_ENTRY": "15", "DAYTONA_SELECT_FROM": "600",
                                          "DAYTONA_SELECT_TO": "2500", "DAYTONA_SNAPS": f"{WIPE_AT},2500",
                                          "DAYTONA_SELECT_TILES": f"{STEP_SHOWN},{WIPE_AT}"}, exe)
    step = min(f for f, v in log15["F"].items() if v[0] == 2)
    assert 760 <= step <= 775 and log15["F"][step][1:4] == (15, 15, 15), log15["F"][step]
    assert votes15 and set(votes15) == {15}, votes15
    assert log15["F"][2500][3] == 15
    # the names were up after the step, and leave before the wipe (exit, state 4): mask clear, layer 1 blank
    assert log15["F"][STEP_SHOWN][0] == 3 and log15["M"][STEP_SHOWN][0] != (0, 0, 0, 0)
    assert log15["F"][WIPE_AT][0] == 5, log15["F"][WIPE_AT]
    assert log15["M"][WIPE_AT] == [(0, 0, 0, 0)] * 64
    assert all(t == S.LAYER1_BLANK for r in range(9, 19) for t in log15["T"][(WIPE_AT, 1, r)])
    wipe, race = sorted(Path(snap15).rglob("*.png"))[:2]
    shutil.copy(wipe, SNAP_DIR / "entry15-wipe.png")
    shutil.copy(race, SNAP_DIR / "entry15-live-test-race.png")
    assert np.asarray(Image.open(race)).std() > 10


@pytest.mark.mame
@pytest.mark.custom_mame
def test_originals_from_select(tmp_path, romset_zip):
    """Entries 0-2 (Beginner, Advanced, Expert: DAYTONA_ENTRY 0, 1, 2) on the patched set (custom build) race as the
    stock set's panels 0-2 (DAYTONA_COURSE 0, 1, 2, stock MAME): the same geometry display list on every frame from
    RACE_FROM (the selects draw differently: Expert's row shows Advanced, Expert and Desert City), the same render-rate
    writes and video mode from RACE_FROM and the same lines from RATE_HASH_FROM, the snapshots at RACE_SNAPS and the last one pixel-identical. Then Expert from the select
    under a memtrace over every original per-id array (Review Focus 1 on the final set): the vote stores 1, no
    original array is read."""
    from dtools.arcade import select as S
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    exe = _custom()
    from tests.test_select import unconverted_select
    rs = unconverted_select(C.guard_new_ids(C.grow_ids(RomSet.from_zip(romset_zip))))     # id 3 not converted
    env = {"DAYTONA_GEO_FROM": str(RACE_FROM), "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": RACE_SNAPS}
    for k in range(3):
        got = {}
        for name, s, binary, sel in (("stock", None, "mame", {"DAYTONA_COURSE": str(k)}),
                                     ("select", rs, exe, {"DAYTONA_ENTRY": str(k)})):
            out = mame_build.run_set(s, tmp_path / f"{k}-{name}", SECONDS, romset_zip, {**sel, **env},
                                     logs=("geo", "rate"), mame=binary)
            lists, _ = _lists(out["geo"])
            out["geo"].unlink()
            rate = [r for r in runner.parse_rate(out["rate"]) if r[0] >= RACE_FROM]
            got[name] = lists, rate, _snaps(out["snap"])
        lists, rate, snaps = got["stock"]
        assert set(lists) == set(range(RACE_FROM, max(lists) + 1)) and max(lists) >= SECONDS * 57, k
        assert len(set(lists.values())) > 1000 and len(rate) > (SECONDS * 57 - RACE_FROM), k
        s_lists, s_rate, s_snaps = got["select"]
        assert s_lists == lists, (k, min(f for f in lists if s_lists.get(f) != lists[f]))
        assert [r[:3] for r in s_rate] == [r[:3] for r in rate], k
        assert [r for r in s_rate if r[0] >= RATE_HASH_FROM] == [r for r in rate if r[0] >= RATE_HASH_FROM], k
        assert len(s_snaps) == len(snaps) == len(RACE_SNAPS.split(",")) + 1 and s_snaps == snaps, k
    log = (tmp_path / "trace-expert.txt").resolve()
    spec = ",".join(f"{lo:x}-{hi:x}" for lo, hi in old_ranges() + [_new_range()])
    mame_build.run_set(rs, tmp_path / "trace-expert", TRACE_SECONDS, romset_zip,
                       {"DAYTONA_ENTRY": "2", "DAYTONA_TRACE_LOG": str(log), "DAYTONA_TRACE_READ": spec,
                        "DAYTONA_TRACE_WRITE": f"{COURSE_ID:x}-{COURSE_ID + 3:x}!"}, mame=exe)
    _check_trace(trace.parse_trace(log), "Expert from the select", 1)
