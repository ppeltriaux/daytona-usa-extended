"""MAME tests of Phase 9 (M4b): the mirrors live as course ids 8-11 (plan docs/superpowers/plans/
2026-11-09-phase9-mirrors-live.md; discovery log docs/spike/2026-11-mirrors-live.md).

Marked `mame` and `custom_mame` (every set here is a custom one: skipped when dtools.config.custom_mame() finds no
build of the current patch)."""

import collections
import shutil
import struct
from pathlib import Path

import pytest

from dtools.arcade import course as C
from dtools.arcade import select as S
from dtools.arcade import tgp
from dtools.arcade import trace
from dtools.arcade.romset import RomSet
from dtools.arcade.tablebase import DATA_BASE
from dtools.convert import build
from tests.test_custom_mame import GEO_FROM, SECONDS, SNAPS, _lists, _snaps
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_sixteen_mame import _custom

SNAP_DIR = Path(__file__).resolve().parent.parent / "out" / "phase9" / "snaps"
UPLOAD_COUNT_IP = 0x1DD88       # 0x1dd84 ld (g6),g5: the char list's copy reads a block's count
UPLOAD_READ_IP = 0x1DDAC        # 0x1dda8 ldos (g7),g4: ... and its chars, a halfword at a time
UPLOAD_STORE_IP = 0x1DDBC       # 0x1ddb8 stos g4,(g6): into char RAM
UPLOADS = 2                     # Beginner's backdrop is uploaded twice in the run: the attract race, the race
TRACE_SECONDS = 60              # past the race's load (frame 1330)


def grown(rs: RomSet) -> RomSet:
    """Phase 8's chain before Desert City and the select: grow_ids, guard_new_ids."""
    return C.guard_new_ids(C.grow_ids(rs))


def sixteen(rs: RomSet) -> RomSet:
    """Phase 8's set without Desert City: `grown`, the sixteen-entry select with Phase 8's row (select.M4A_ENTRIES), id
    3 not converted (test_select.unconverted_select: patch_select refuses a live unconverted id 3 since Phase 9's final
    review; these runs race the originals only, and the set keeps the bytes it had)."""
    from tests.test_select import unconverted_select
    return unconverted_select(grown(rs))


def moved_backdrop(rs: RomSet, course: int = 0, run: tuple[int, int] = build.ID_DATA_FREE) -> RomSet:
    """`rs` with `course`'s backdrop char blocks (its upload set's char list, course.Backdrop) copied, unchanged, to
    the first free bytes of the data run `run` (build._first_fit: free and owned by nothing; tgp.write_free) one after
    another, and the char list's source words pointed at the copies. The original blocks stay where they are."""
    b = C.backdrop(rs, C.course_table(rs)[course])
    need = sum(4 + 32 * u.count for u in b.uploads.chars)
    at = build._first_fit(rs, "data", run, need, "the moved backdrop")
    out = rs
    for u in b.uploads.chars:
        blob = rs.data[u.src - DATA_BASE:u.src - DATA_BASE + 4 + 32 * u.count]
        out = tgp.write_free(out, "data", at, blob)
        at += len(blob)
    at -= need
    for i, u in enumerate(b.uploads.chars):
        out = out.with_patch("data", b.uploads.char_list - DATA_BASE + 8 * i, struct.pack("<I", DATA_BASE + at))
        at += 4 + 32 * u.count
    return out


@pytest.mark.mame
@pytest.mark.custom_mame
def test_backdrop_in_custom_area(tmp_path, romset_zip):
    """Task 1 (discovery log, "Backdrops"): the race-time char uploader (0x1dd70, the course's upload set's char list)
    reads char blocks in the custom data area (i960 0x03000000+). Beginner with its backdrop's three char blocks
    copied to build.ID_DATA_FREE and the char list pointed there (`moved_backdrop`), entered from the select
    (DAYTONA_ENTRY 0), races call for call as Phase 8's set: the same geometry display list on every frame from
    GEO_FROM, the same render-rate log, pixel-identical snapshots (the race's at 2000, 2300, 2550 show the backdrop;
    copied to out/phase9/snaps/task1). Then a read trace over the original blocks and the copies: the uploader reads
    every char of the copies (UPLOADS times: the attract race and the race) and nothing of the originals."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    exe = _custom()
    base = sixteen(RomSet.from_zip(romset_zip))
    moved = moved_backdrop(base)
    old_b, new_b = (C.backdrop(s, C.course_table(s)[0]) for s in (base, moved))
    assert all(u.src >= 0x03000000 for u in new_b.uploads.chars)
    assert [(u.dst, u.count) for u in new_b.uploads.chars] == [(u.dst, u.count) for u in old_b.uploads.chars]
    assert [u.data(moved) for u in new_b.uploads.chars] == [u.data(base) for u in old_b.uploads.chars]
    env = {"DAYTONA_ENTRY": "0", "DAYTONA_GEO_FROM": str(GEO_FROM), "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": SNAPS}
    got = {}
    for name, s in (("sixteen", base), ("moved", moved)):
        out = mame_build.run_set(s, tmp_path / name, SECONDS, romset_zip, env, logs=("geo", "rate"), mame=exe)
        lists, _ = _lists(out["geo"])
        out["geo"].unlink()                                 # about 600 MB each
        got[name] = lists, runner.parse_rate(out["rate"]), _snaps(out["snap"]), sorted(out["snap"].rglob("*.png"))
    lists, rate, snaps, shots = got["sixteen"]
    assert set(lists) == set(range(GEO_FROM, max(lists) + 1)) and max(lists) >= SECONDS * 57
    assert len(set(lists.values())) > 1000 and len(rate) > SECONDS * 57
    assert len(snaps) == len(SNAPS.split(",")) + 1
    m_lists, m_rate, m_snaps, m_shots = got["moved"]
    assert m_lists == lists, min(f for f in lists if m_lists.get(f) != lists[f])
    assert m_rate == rate
    assert m_snaps == snaps, [k for k, (a, b) in enumerate(zip(m_snaps, snaps)) if a != b]
    dest = SNAP_DIR / "task1"
    dest.mkdir(parents=True, exist_ok=True)
    for frame, p in zip(SNAPS.split(",")[-3:], m_shots[-4:-1]):
        shutil.copy(p, dest / f"beginner-backdrop-moved-{frame}.png")
    # the trace: the copies read by the uploader, the originals not at all
    old = [(u.src, u.src + 4 + 32 * u.count - 1) for u in old_b.uploads.chars]
    new = [(u.src, u.src + 4 + 32 * u.count - 1) for u in new_b.uploads.chars]
    log = (tmp_path / "trace.txt").resolve()
    mame_build.run_set(moved, tmp_path / "trace", TRACE_SECONDS, romset_zip,
                       {"DAYTONA_ENTRY": "0", "DAYTONA_TRACE_LOG": str(log),
                        "DAYTONA_TRACE_READ": ",".join(f"{lo:x}-{hi:x}" for lo, hi in old + new),
                        "DAYTONA_TRACE_WRITE": "10e0000-10fffff"}, mame=exe)
    events = trace.parse_trace(log)
    reads = collections.Counter()
    for e in events:
        if e.kind == "R":
            where = "new" if any(lo <= e.addr <= hi for lo, hi in new) else "old"
            reads[(where, e.ip)] += e.count
    chars = sum(u.count for u in new_b.uploads.chars)
    assert reads == {("new", UPLOAD_COUNT_IP): UPLOADS * len(new), ("new", UPLOAD_READ_IP): UPLOADS * 16 * chars}, \
        {(w, hex(ip)): n for (w, ip), n in reads.items()}
    stores = {e.ip for e in events if e.kind == "W"}
    assert stores == {UPLOAD_STORE_IP}


# --- Task 2: a mirror tests as its original (course.widen_ids) -----------------------------------------------------

WIDE_TRACE_SECONDS = 60     # Beginner from the select, past the race's load (frame 1330) and its first lap's start


def widened_chain(rs: RomSet) -> RomSet:
    """This phase's chain before Desert City and the select: grow_ids, guard_new_ids, widen_ids."""
    return C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))


def widened(rs: RomSet) -> RomSet:
    """This phase's chain to the select without mirrors or Desert City: `widened_chain`, the sixteen-entry select with
    Phase 8's row (select.M4A_ENTRIES: the set has no mirror, so patch_select refuses ENTRIES' live 8-10), id 3 not
    converted (test_select.unconverted_select, as `sixteen`)."""
    from tests.test_select import unconverted_select
    return unconverted_select(widened_chain(rs))


BILLBOARD_READ_IPS = {0x221E9C, 0x221EA0}  # 0x221e98 ld 0x0(r8),r5; 0x221e9c ld 0x8(r8),r6: the point's x and z
BOOT_CLEAR_IP = 0x1798                     # 0x1794 stq: the boot clears 0x500000-0x5fefff


def _update_ips() -> range:
    """Trace ips (the address after the instruction) of course.wide_code's `update`, up to its `ret`."""
    from dtools.arcade import disasm
    lines = disasm.listing_bytes(C.wide_code().bytes(), C.WIDE_CODE)
    end = next(ln.addr for ln in lines if ln.text == "ret")
    return range(C.WIDE_CODE + 4, end + 1)


def _shadow_read_ips() -> dict[str, set[int]]:
    """Trace ips of the widened sites' reads of their shadows, by kind: an 8-byte load logs its address + 8; the
    billboard's point is read by its handler's own loads through r8 (BILLBOARD_READ_IPS)."""
    out = {"and7": {s.load + 8 for s in C.LITERAL_ID_SITES.values() if s.rule == "and7"},
           "tune": {a + 8 for a in C.TUNE_SITES}, "visit": {0x17420},
           "lanes": {a + 8 for a in C.LANE_READERS}, "billboard": set(BILLBOARD_READ_IPS)}
    return out


@pytest.mark.mame
@pytest.mark.custom_mame
def test_widened_originals(tmp_path, romset_zip, disc):  # noqa: F811
    """Review Focus 2: widening changes neither the originals nor Desert City. Phase 8's playtest set (`grown`, Desert
    City converted to id 3: build.converted_set(..., 3, slot=3, scenery="model"), then the select with M4A_ENTRIES;
    since Phase 9's final review converted before the select: the same bytes as Phase 8's order) and the same built on
    the widened chain (`widened_chain`) differ only in the program's widened sites' displacements, the writers and
    WIDE_HOME, and decode ids 0-3 alike but for id_arrays' two WIDE_ARRAYS. Entries 0-3 (Beginner, Advanced, Expert,
    Desert City: ids 0, 2, 1, 3) entered from the select (DAYTONA_ENTRY) race call for call on both, SECONDS each on
    the custom build: the same geometry display list on every frame from GEO_FROM (the coin: both selects and the
    race), the same render-rate log (from frame 0: boot and attract keep their timing), pixel-identical snapshots.
    Then Beginner from the select on the widened set under a trace over WIDE_HOME's tables, the shadows (WIDE_RAM)
    and the two shared tables they replace (the positive control): every kind of widened site reads its shadow, only
    the update reads the per-id tables and writes the shadows, and nothing reads the old GRID_LANES or
    BILLBOARD_REF."""
    import numpy as np
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_golden import _differ
    from dtools.arcade import golden as G
    exe = _custom()
    stock = RomSet.from_zip(romset_zip)
    sets = {name: S.patch_select(build.converted_set(chain(stock), disc, 3, slot=3, scenery="model"), S.M4A_ENTRIES)
            for name, chain in (("phase8", grown), ("widened", widened_chain))}
    a, b = sets["phase8"], sets["widened"]
    changed = np.flatnonzero(np.frombuffer(a.program, np.uint8) != np.frombuffer(b.program, np.uint8))
    allowed = {C.rom_word_at(at)[1] + 4 + k for at in C.WIDE_SHADOWS for k in range(4)}
    allowed |= {C.rom_word_at(at)[1] + k for at in C.ID_WRITERS for k in range(4)}
    allowed |= set(range(C.WIDE_HOME, C.WIDE_HOME + C.WIDE_SIZE))
    assert len(changed) and set(changed.tolist()) <= allowed
    assert (a.data, a.copro_data, a.polygons, a.textures) == (b.data, b.copro_data, b.polygons, b.textures)
    for k in range(4):
        x, y = G.decode_course(a, k), G.decode_course(b, k)
        assert _differ(x, y) == {"id_arrays"}, k
        assert {n for n in x.id_arrays if x.id_arrays[n] != y.id_arrays[n]} == C.WIDE_ARRAYS, k
    env = {"DAYTONA_GEO_FROM": str(GEO_FROM), "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": SNAPS}
    for entry in range(4):
        got = {}
        for name, s in sets.items():
            out = mame_build.run_set(s, tmp_path / f"{entry}-{name}", SECONDS, romset_zip,
                                     {"DAYTONA_ENTRY": str(entry), **env}, logs=("geo", "rate"), mame=exe)
            lists, _ = _lists(out["geo"])
            out["geo"].unlink()                                 # about 600 MB each
            got[name] = lists, runner.parse_rate(out["rate"]), _snaps(out["snap"])
        lists, rate, snaps = got["phase8"]
        assert set(lists) == set(range(GEO_FROM, max(lists) + 1)) and max(lists) >= SECONDS * 57, entry
        assert len(set(lists.values())) > 1000 and len(rate) > SECONDS * 57, entry
        assert len(snaps) == len(SNAPS.split(",")) + 1, entry
        w_lists, w_rate, w_snaps = got["widened"]
        assert w_lists == lists, (entry, min(f for f in lists if w_lists.get(f) != lists[f]))
        assert w_rate == rate, entry
        assert w_snaps == snaps, (entry, [k for k, (p, q) in enumerate(zip(w_snaps, snaps)) if p != q])
    # the positive control: the sites read their shadows, only the update reads the per-id tables and writes the
    # shadows (after the boot's clear), and nothing reads the shared tables any more
    log = (tmp_path / "trace.txt").resolve()
    shadows = (C.WIDE_RAM, C.WIDE_RAM + C.WIDE_RAM_SIZE - 1)
    spans = [(C.WIDE_HOME, C.WIDE_CODE - 1), shadows, (C.GRID_LANES, C.GRID_LANES + 3),
             (C.BILLBOARD_REF, C.BILLBOARD_REF + 11)]
    mame_build.run_set(b, tmp_path / "trace", WIDE_TRACE_SECONDS, romset_zip,
                       {"DAYTONA_ENTRY": "0", "DAYTONA_TRACE_LOG": str(log),
                        "DAYTONA_TRACE_READ": ",".join(f"{lo:x}-{hi:x}" for lo, hi in spans),
                        "DAYTONA_TRACE_WRITE": f"{shadows[0]:x}-{shadows[1]:x}"}, mame=exe)
    reads, writes = collections.Counter(), collections.Counter()
    for e in trace.parse_trace(log):
        if e.kind == "R":
            reads[(e.ip, e.addr)] += e.count
        elif e.kind == "W":
            writes[(e.ip, e.addr)] += e.count
    update = _update_ips()
    kinds = _shadow_read_ips()
    by_kind = {k: sum(n for (ip, _), n in reads.items() if ip in v) for k, v in kinds.items()}
    tables = sum(n for (ip, page), n in reads.items() if page < C.WIDE_CODE)
    print(f"\nwidened set, Beginner from the select: shadow reads per kind {by_kind}, per-id table reads by the "
          f"update {tables}; reads {dict(reads)}; writes {dict(writes)}")
    for (ip, page), n in reads.items():
        assert (ip in update) if page < C.WIDE_CODE else (page == C.WIDE_RAM & ~0xFF and
                                                          ip in set().union(*kinds.values())), (hex(ip), hex(page))
    assert all(v for v in by_kind.values()) and tables, by_kind
    assert {ip for ip, _ in writes} <= set(update) | {BOOT_CLEAR_IP} and set(update) & {ip for ip, _ in writes}


@pytest.mark.mame
@pytest.mark.custom_mame
def test_widened_desert_city(tmp_path, romset_zip, disc, monkeypatch):  # noqa: F811
    """Review Focus 2 past 45 s: Phase 8's `test_desert_city_id3` (the long-time run's frozen ground calls per car
    and bars, max 3,871 drawn at 6287, 0 over 4,251, 0 cap hits, the lap at 10181, fresh SRAM; the time-over and the
    second game, Beginner after Desert City) on Desert City converted onto the widened chain (`widened_chain`), its
    snapshots to out/phase9/snaps/task2-id3. Every number holds: the shadows cost the race not a cycle (a first design
    with trampolines moved car 0's ground calls by one: discovery log, "Literal ids")."""
    import tests.test_convert_build as tcb
    import tests.test_convert_mame as tcm
    monkeypatch.setattr(tcb, "_guarded", widened_chain)
    monkeypatch.setattr(tcm, "ID3_SNAP_DIR", SNAP_DIR / "task2-id3")
    tcm.test_desert_city_id3(tmp_path, romset_zip, disc)


# --- Task 3: the mirrors of the originals live (mirror.mirrored_entry) ---------------------------------------------

MIRROR_ENTRIES = {0: 8, 1: 9, 2: 10}
"""Entry of an original -> entry of its mirror (select.ENTRIES): Beginner (id 0) -> 8 (id 8), Advanced (id 2) -> 9
(id 10), Expert (id 1) -> 10 (id 9)."""
RACE_LOAD = 1300            # the race's set-up is at 1328-1330: the ground calls before it are the attract's (Beginner)
PAIRED_TO = 1440            # Phase 4's SYMMETRIC_TO: every ground call of the race pairs to here on all three
GROUND_PARTS = {8: 1441, 9: 2251, 10: 1674}
"""Per mirror entry, the frame of the first ground call without a mirrored partner in its original's run (measured,
zero margin as Phase 4's floors: MAME is deterministic). Beginner's mirror parts exactly where Phase 4's in-place mirror
did (SYMMETRIC_TO + 1: the player's rolling-start drive rounds one way); Advanced's after its GO, Expert's at 1674."""
OPPONENTS_TO = {9: 1655, 10: 1658}
"""Per mirror entry of Advanced and Expert: every opponent stays within SYMMETRY_TOL (XZ) of its original's car
mirrored on every frame from 1330 through this frame (measured: the first opponent to part, car 6 on Advanced, car 3
on Expert). Beginner's mirror holds Phase 4's SYMMETRIC_CARS exactly (and GRID_EXCEPTION is off from the set-up)."""
LOAD_LEFT_OUT = {8: set(), 9: set(), 10: {1332}}
"""Per mirror entry, the frames whose list leaves a visible cell's scenery out (test_desert_city_id3's measure), all at
the race's load and none at the cap (the frame's polygon count there 335 of course.OBC's 5,000): Expert's list at
1332 (after 1331's empty one) draws 292 polygons and leaves nine visible cells out, as its original's does at 1332
with the same counts, the cells mirrored (rows 7-9 of columns 9-11 against rows 6-8): measured with the cells log on
both (discovery log, "Mirrors of the originals")."""
MIRROR_SECONDS = 90         # the mirrors' runs: the bars over the fly-by and the first 55 s or so of the race
BARS_FROM = 1330            # the race's first list (the fly-by)
MIRROR_GO = {8: 1893, 9: 1651, 10: 1653}
"""Per mirror entry, the race's first frame: the timer's first tick (`_go`), measured, and equal to the original's
(the green light is a few frames before: 1890 on Beginner's rolling start, 1645 after Expert's and Advanced's
countdown, autoplay.lua; Desert City's standing start ticks at 1651 too, test_convert_mame.ID3_GO; Expert's at
1653)."""
MIRROR_SNAPS = {"fly-by": 1500, "start": 30, "mid-lap": 4000}
"""The mirrors' snapshots: the fly-by (frame 1500), the start (GO + 30) and mid-lap (frame 4000)."""
CAMERA_FROM = 1332
"""The camera check's first frame: 1331 is the race's load (the empty display list), where Expert's mirror logs a
camera half-placed (its y 79.7 against 70.0, one frame); from 1332 every frame to PAIRED_TO is compared."""
CAMERA_Z_OFFSET = (0.86, 0.871)
"""The mirror's scripted camera (progress log `K`: 0x501720 and the angles 0x501738-0x50173c) against the original's
from CAMERA_FROM to PAIRED_TO: x and y equal (within SYMMETRY_TOL), pitch equal, yaw 0x8000 - yaw and roll negated
(within CAMERA_ANGLE), but z is not the original's mirrored in the plane: it is -1 - z less 0.861-0.870 (measured;
Expert's 0.870 on every frame, Beginner's 0.870 falling to 0.861 as the car turns), a constant the game adds to the
placed camera that the keyframes' mirror does not cancel (cause not found within the time box: discovery log, "Mirrors
of the originals")."""
CAMERA_ANGLE = 2            # s16 units: the angles derive from f32 through the TGP's atan2 (0x0a), rounded
SESSION_COIN2 = 9300
"""The second game's coin of the session runs: Beginner's (and its mirror's) first game times out at 7134 (the timer
reaches 0; the field drives on behind the game-over), the attract is back by about 9000."""
SESSION_FROM = SESSION_COIN2 + 550
SESSION_TEX_AT = SESSION_COIN2 + 800
SESSION_SECONDS = (SESSION_COIN2 + 2700) // 57 + 1
SRAM_BOOKKEEPING = (0x100, 0x200)
"""Backup SRAM offsets of the bookkeeping block 0x1d00100-0x1d0017f (its checksum word and the play counters
0x1d00140-0x1d0016f, course.COUNTER_GUARD) and its backup copy 0x1d00180-0x1d001ff: where a session whose first game
is the mirror (not counted: ids 3-15 are guarded) may differ from one whose first game is Beginner."""


def mirrors(rs: RomSet) -> RomSet:
    """This task's set: grow_ids, guard_new_ids, widen_ids, the mirrors of Beginner, Expert and Advanced at ids 8, 9,
    10 (mirror.mirrored_entry), then the sixteen-entry select with Task 3's row (test_mirror.originals_row: 8-10
    live, patch_select refuses them before the mirrors exist; 11 locked, the set has no Desert City). The same bytes
    as the select patched first (test_mirror.test_mirrors_then_select)."""
    from dtools.arcade import mirror as M
    from tests.test_mirror import originals_row
    from tests.test_select import unconverted_select
    out = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    for o in (0, 1, 2):
        out = M.mirrored_entry(out, o)
    return unconverted_select(out, originals_row())         # id 3 not converted: its check lifted


def _geo_counts(path, rs: RomSet, start: int = 0):
    """({frame: SHA-1 of the walked list}, {frame: (drawn polygons, the object addresses drawn)} for frames from
    `start`) of a geo_log.lua log, removed afterwards (polygon RAM carried through every list)."""
    import hashlib
    from dtools.arcade import geo
    ram, hashes, counts = geo.PolyRam(), {}, {}
    try:
        for frame, at, _, words in geo.parse_log(path):
            cmds = list(geo.walk(words, at // 4))
            hashes[frame] = hashlib.sha1(repr([(c.kind, c.args) for c in cmds]).encode()).hexdigest()
            n = geo.count_polygons(cmds, rs, ram)
            if frame >= start:
                counts[frame] = (n, frozenset(c.args["oba"] for c in cmds if c.kind in (0x01, 0x11)))
    finally:
        path.unlink(missing_ok=True)
    return hashes, counts


def _go(timer: dict[int, int]) -> int:
    """The race's first frame: the first frame whose timer is below the first logged value (the countdown's end)."""
    first = timer[min(timer)]
    return min(f for f in sorted(timer) if timer[f] < first)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_mirrors_of_originals(tmp_path, romset_zip):
    """Task 3 (discovery log, "Mirrors of the originals"). On `mirrors` (the widened chain with the mirrors of
    Beginner, Expert and Advanced at ids 8-10, entries 8-10 live), custom MAME:

    - the Global Constraint: entries 0-2 (Beginner, Advanced, Expert) race call for call as on the widened set
      without mirrors (Phase 8's row, select.M4A_ENTRIES: Task 2's set), SECONDS each: the same geometry display list
      on every frame from GEO_FROM, the same render-rate log, pixel-identical snapshots;
    - each mirror from the select (entries 8-10, MIRROR_SECONDS): Review Focus 3, the start as its original's (the
      countdown timer equal to the original's on every frame; Beginner's mirror rolling, the player moving before GO;
      Expert's and Advanced's standing, the player still to GO); every ground call of the race (from RACE_LOAD) pairs
      with the original's mirrored (test_mirror._parting: dest equal, x and the heights within SYMMETRY_TOL, z
      mirrored) through PAIRED_TO and to its GROUND_PARTS frame, the attract's calls (Beginner in both) equal; the
      opponents within SYMMETRY_TOL of their mirrors (Beginner's mirror: Phase 4's SYMMETRIC_CARS, GRID_EXCEPTION off;
      the others: every opponent through OPPONENTS_TO); the scripted camera mirrored from CAMERA_FROM to PAIRED_TO
      (CAMERA_Z_OFFSET: x, y and the angles as the rule says, z a constant 0.86-0.87 off the plane's mirror); the frame-rate bar from BARS_FROM to the end (a new display
      list on every frame of the race but where the original has none, GO + 1 after a standing start as on Desert
      City, test_convert_mame.ID3_REPEAT; at most BAR drawn polygons in every list; no cap hit: the frame's polygon
      count below course.OBC on every frame; no scenery drawn outside the visible cells, and none left out but at the
      race's load, LOAD_LEFT_OUT: test_desert_city_id3's measures); the numbers printed; snapshots (MIRROR_SNAPS) of the mirror and the original to out/phase9/snaps/task3;
    - Review Focus 1, one session each: the mirror of Beginner then Beginner (DAYTONA_SECOND at SESSION_COIN2) against
      Beginner then Beginner (the TGP ground slot each course load sends: the mirror's 8 for its game and the attract
      after it, then Beginner's 0, whose list block and pool are Beginner's, as the reference's). A second game's race follows the session's history (Phase 8: the frame counter, the AI
      tables, the last start time), and a mirrored first race leaves another history, so the two second games are not
      call for call (measured: the field differs from the race's first frame); what the mirror could leak is checked
      instead, as Phase 8 did after Desert City: the second game draws exactly the objects, texture points and
      headers the reference's draws, texture RAM (both sheets) and luma RAM after its load are the reference's, and
      backup SRAM differs only in the bookkeeping block (SRAM_BOOKKEEPING: the mirror's game is not counted)."""
    import math
    import numpy as np
    from dtools.arcade import golden as G
    from dtools.arcade import mirror as M
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_convert_mame import BAR, _new_lists
    from tests.test_mirror import GRID_EXCEPTION, SYMMETRIC_CARS, SYMMETRY_TOL, _calls, _parting
    exe = _custom()
    stock = RomSet.from_zip(romset_zip)
    out = mirrors(stock)
    ref = widened(stock)
    dest = SNAP_DIR / "task3"
    dest.mkdir(parents=True, exist_ok=True)
    names = {e: S.M5A_ENTRIES[e].name.lower() + ("-mirror" if S.M5A_ENTRIES[e].mirror else "") for e in range(11)}

    # the originals: call for call against the set without mirrors, and the runs the mirrors pair with
    env = {"DAYTONA_GEO_FROM": str(GEO_FROM), "DAYTONA_GEO_EVERY": "1"}
    originals = {}
    for e in MIRROR_ENTRIES:
        got = {}
        for name, s in (("ref", ref), ("mirrors", out)):
            o = mame_build.run_set(s, tmp_path / f"{e}-{name}", SECONDS, romset_zip,
                                   {"DAYTONA_ENTRY": str(e), "DAYTONA_SNAPS": SNAPS, **env},
                                   logs=("geo", "rate", "ground", "progress"), mame=exe)
            hashes, counts = _geo_counts(o["geo"], s, BARS_FROM)
            got[name] = hashes, runner.parse_rate(o["rate"]), _snaps(o["snap"]), o, counts
        (lists, rate, snaps, _, _), (m_lists, m_rate, m_snaps, run, counts) = got["ref"], got["mirrors"]
        assert set(lists) == set(range(GEO_FROM, max(lists) + 1)) and len(set(lists.values())) > 1000, e
        assert m_lists == lists, (e, min(f for f in lists if m_lists.get(f) != lists[f]))
        assert m_rate == rate and len(rate) > SECONDS * 57, e
        assert m_snaps == snaps and len(snaps) == len(SNAPS.split(",")) + 1, e
        originals[e] = run, counts
    # the mirrors
    report = {}
    for e, me in MIRROR_ENTRIES.items():
        o_run, o_counts = originals[e]
        k = S.M5A_ENTRIES[me].course
        shots = {"fly-by": 1500, "start": MIRROR_GO[me] + MIRROR_SNAPS["start"], "mid-lap": MIRROR_SNAPS["mid-lap"]}
        snaps = ",".join(map(str, shots.values()))
        m_run = mame_build.run_set(out, tmp_path / f"{me}-mirror", MIRROR_SECONDS, romset_zip,
                                   {"DAYTONA_ENTRY": str(me), "DAYTONA_GEO_FROM": str(BARS_FROM),
                                    "DAYTONA_GEO_EVERY": "1", "DAYTONA_CELLS_FROM": str(BARS_FROM),
                                    "DAYTONA_SNAPS": snaps}, logs=("geo", "rate", "ground", "progress", "cells"),
                                   mame=exe)
        _, per = _geo_counts(m_run["geo"], out, BARS_FROM)
        c = G.decode_course(out, k)
        # Review Focus 3: the start
        (_, o_timer), (_, m_timer) = (runner.parse_progress(r["progress"]) for r in (o_run, m_run))
        assert {f: v for f, v in m_timer.items() if f in o_timer} == o_timer, e
        go = _go(m_timer)
        assert go == _go(o_timer) == MIRROR_GO[me], (e, go)
        player = runner.parse_car_positions(m_run["progress"])
        (x0, _, z0), (x1, _, z1) = player[BARS_FROM][0], player[go - 1][0]
        moved = math.hypot(x1 - x0, z1 - z0)            # XZ: a standing car settles on the ground (y) at first
        rolling = c.rolling_start[1]
        assert rolling == G.decode_course(out, S.M5A_ENTRIES[e].course).rolling_start[1] == (k == 8), e
        assert (moved > 10.0) if rolling else (moved < 1e-3), (e, moved)
        # the pairing
        a, b = _calls(o_run["ground"]), _calls(m_run["ground"])
        assert {f: v for f, v in a.items() if f < RACE_LOAD} == {f: v for f, v in b.items() if f < RACE_LOAD}
        a, b = ({f: v for f, v in x.items() if RACE_LOAD <= f} for x in (a, b))
        assert sum(len(v) for f, v in a.items() if f <= PAIRED_TO) > 500, e
        assert _parting(a, b, GROUND_PARTS[me] - 1) is None, e
        first = _parting(a, b, GROUND_PARTS[me])
        assert first is not None and first.frame == GROUND_PARTS[me], (e, first)
        pa = runner.parse_car_positions(o_run["progress"])

        def near(car: int, f: int) -> bool:
            x, _, z = M.mirror_world(pa[f][car])
            return math.hypot(x - player[f][car][0], z - player[f][car][2]) <= SYMMETRY_TOL
        floors = SYMMETRIC_CARS if k == 8 else {car: OPPONENTS_TO[me] for car in pa[1330] if car}
        for car, until in floors.items():
            assert all(near(car, f) for f in range(1330, until + 1)), (e, car)
        if k == 8:
            assert not near(GRID_EXCEPTION, 1330)
        # the camera script's play, mirrored (mirror.mirror_camera_record)
        o_cam, m_cam = (runner.parse_camera(r["progress"]) for r in (o_run, m_run))
        offsets = []
        for f in range(CAMERA_FROM, PAIRED_TO + 1):
            (p, pitch, yaw, roll), (q, q_pitch, q_yaw, q_roll) = o_cam[f], m_cam[f]
            assert max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= SYMMETRY_TOL, (e, f, p, q)
            assert abs(pitch - q_pitch) <= CAMERA_ANGLE and abs(roll + q_roll) <= CAMERA_ANGLE, (e, f)
            assert abs(((yaw + q_yaw) & 0xFFFF) - 0x8000) <= CAMERA_ANGLE, (e, f, yaw, q_yaw)
            offsets.append((-1 - p[2]) - q[2])
        assert CAMERA_Z_OFFSET[0] <= min(offsets) and max(offsets) <= CAMERA_Z_OFFSET[1], (e, min(offsets),
                                                                                          max(offsets))
        # the frame-rate bar and the cap
        cells = runner.parse_cells(m_run["cells"])
        owner = {p.oba: cell for cell, lst in enumerate(c.cells) for p in c.object_lists[lst]}
        occupied = set(owner.values())
        race = range(BARS_FROM, max(m_timer) + 1)
        assert set(race) <= set(per) and set(race) <= set(cells), e
        left_out = [f for f in race if {x for x in cells[f][1] if x in occupied} - {owner[x] for x in per[f][1]
                                                                                   if x in owner}]
        strays = [f for f in race if {owner[x] for x in per[f][1] if x in owner} - set(cells[f][1])]
        top = max(race, key=lambda f: (per[f][0], -f))
        busiest = max(race, key=lambda f: (cells[f][0], -f))
        rate = runner.parse_rate(m_run["rate"])
        missed = set(range(go, max(m_timer) + 1)) - _new_lists(rate)
        o_missed = set(range(go, max(o_timer) + 1)) - _new_lists(runner.parse_rate(o_run["rate"]))
        o_top = max((f for f in o_counts if f >= BARS_FROM), key=lambda f: (o_counts[f][0], -f))
        report[me] = dict(go=go, rolling=rolling, parts=first.frame, camera_z=(min(offsets), max(offsets)),
                          max=(per[top][0], top),
                          original_max=(o_counts[o_top][0], o_top), count_max=(cells[busiest][0], busiest),
                          over=sum(per[f][0] > BAR for f in race), cap_hits=sum(cells[f][0] >= C.OBC for f in race),
                          left_out=left_out, strays=len(strays), missed_lists=sorted(missed))
        print(f"\nentry {me} ({names[me]}, id {k}): {report[me]}")
        assert set(left_out) == LOAD_LEFT_OUT[me] and all(cells[f][0] < 500 for f in left_out), (e, report[me])
        assert cells[busiest][0] < C.OBC and not strays and per[top][0] <= BAR, (e, report[me])
        assert missed == o_missed and missed <= {go + 1}, (e, missed, o_missed)    # a standing start repeats GO + 1
        for p, (label, frame) in zip(sorted(m_run["snap"].rglob("*.png")), shots.items()):
            shutil.copy(p, dest / f"{names[me]}-{label}-{frame}.png")
        for p, frame in zip(sorted(o_run["snap"].rglob("*.png")), SNAPS.split(",")):
            if int(frame) in (1700, 2300):
                shutil.copy(p, dest / f"{names[e]}-{frame}.png")

    # Review Focus 1: Beginner after its mirror
    env = {"DAYTONA_SECOND": f"0:{SESSION_COIN2}", "DAYTONA_GEO_FROM": str(SESSION_FROM), "DAYTONA_GEO_EVERY": "1",
           "DAYTONA_TEXDUMP_AT": str(SESSION_TEX_AT)}
    sessions = {}
    for first in (8, 0):
        wd = tmp_path / f"session-{first}"
        (wd / "tex").mkdir(parents=True)
        o = mame_build.run_set(out, wd, SESSION_SECONDS, romset_zip,
                               {"DAYTONA_ENTRY": str(first), **env, "DAYTONA_TEXDUMP_DIR": str((wd / "tex").resolve())},
                               logs=("geo", "progress", "ground"), mame=exe)
        lists, refs = _lists(o["geo"])
        o["geo"].unlink()
        _, timer = runner.parse_progress(o["progress"])
        sessions[first] = dict(lists=lists, refs=refs, tex=wd / "tex", timer=timer,
                               slots=runner.parse_ground_slots(o["ground"]),
                               sram=(wd / "run" / "nvram" / "daytona" / "backup1").read_bytes())
    a, b = sessions[8], sessions[0]
    # the TGP's ground slot per course load (function 0x0d): the attract's 0, the first game's own id (the attract
    # after the game replays it: measured, its load at 8220), the second game's 0; and the list block and pool the
    # second game's slot names are Beginner's (no slot leaks)
    sa, sb = ([x for _, x in r["slots"]] for r in (a, b))
    assert len(sa) == len(sb) >= 3 and sa[0] == sb[0] == 0 and set(sa[1:-1]) == {8} and set(sb[1:-1]) == {0}, \
        (a["slots"], b["slots"])
    assert sa[-1] == sb[-1] == 0 and [f for f, _ in a["slots"]] == [f for f, _ in b["slots"]]
    assert a["slots"][-1][0] > SESSION_COIN2 and b["slots"][-1][0] > SESSION_COIN2

    def bases(slot: int) -> tuple[int, int]:
        return tuple(struct.unpack_from("<I", out.copro_data, 4 * (w + slot))[0] for w in (tgp.SLOT_LIST,
                                                                                            tgp.SLOT_POOL))
    assert bases(sa[-1]) == bases(sb[-1]) == bases(0) != bases(8)
    assert min(f for f, v in b["timer"].items() if v == 0) < SESSION_COIN2       # the first game is over
    assert len(b["lists"]) > 2000 and a["refs"] == b["refs"]
    for kind in ("tex0", "tex1", "luma"):
        got, want = (r["tex"] / f"{kind}.{SESSION_TEX_AT}.bin" for r in (a, b))
        assert got.read_bytes() == want.read_bytes(), kind
    differ = np.flatnonzero(np.frombuffer(a["sram"], np.uint8) != np.frombuffer(b["sram"], np.uint8))
    assert len(differ) and all(SRAM_BOOKKEEPING[0] <= x < SRAM_BOOKKEEPING[1] for x in differ), differ
    print(f"Beginner after its mirror: {sum(a['lists'].get(f) != v for f, v in b['lists'].items())} of "
          f"{len(b['lists'])} lists differ from Beginner after Beginner; drawn refs equal; SRAM differs at "
          f"{[hex(0x1d00000 + int(x)) for x in differ]}")


# --- Task 4: Desert City mirrored (mirror.mirrored_entry of id 3, course id 11) ---------------------------------------

DESERT_ENTRY, DESERT_MIRROR_ENTRY = 3, 11       # select.ENTRIES: Desert City (id 3), its mirror (id 11)
DESERT_PARTS = 1980
"""The frame of the first ground call of Desert City's mirror without a mirrored partner in Desert City's run
(test_mirror._parting, from RACE_LOAD; measured, zero margin: AI car 5's call at (-321.6, -153.7), 329 frames after
GO, as Advanced's and Expert's mirrors part after theirs, GROUND_PARTS)."""
DESERT_OPPONENTS_TO = 1655
"""Every opponent of the mirror within SYMMETRY_TOL (XZ) of Desert City's car mirrored on every frame from 1330
through this frame (measured: car 1 parts first, at 1656, 5 frames after GO; Advanced's mirror holds to 1655 too,
OPPONENTS_TO)."""
DESERT_LEFT_OUT: frozenset = frozenset()
"""The frames of the mirror's lap whose list leaves a visible cell's scenery out (test_desert_city_id3's measure):
none, as on Desert City."""
DESERT_MIRROR_LAP_END = 10244
"""The mirror's player (wheel centred, from segment 460) reaches the lap line, lap 1 (Desert City's at
test_convert_mame.ID3_LAP_END, 10181; measured: the two players part with the field at 1980)."""
DESERT_MIRROR_MAX, DESERT_MIRROR_MAX_AT, DESERT_MIRROR_COUNT_MAX = 4102, 6319, 4826
"""The mirror's drawn-polygon maximum per list over its player's lap (GO to DESERT_MIRROR_LAP_END, the long-time set)
and its frame, and the largest polygon count (the cells log) over it (measured, as test_convert_mame.ID3_MAX)."""
DESERT_MARGIN = 298
"""How much worse than Desert City's maximum (test_convert_mame.ID3_MAX, 3,871 at 6287 on the same window) the
mirror's may be, set before measuring (the controller's ruling, fix round 1): Task 3's largest mirror-against-original
gap, 298 (Advanced's mirror, 3,633 against 3,931). The binding bar stays the hard limit, BAR drawn per list and no cap
hit. Measured: +231 (DESERT_MIRROR_MAX 4,102, 149 under BAR). Its cause is not traced: the scenery is Desert City's
polygons mirrored, and after the parting (DESERT_PARTS) the field is elsewhere; the busiest frame, 6319, is also the
mirror's largest polygon count (4,826, 174 under course.OBC)."""
ENTRY4_PANEL = (330, 74, 470, 150)
"""(left, top, right, bottom) px of entry 4's panel on the select's snapshot with entry 3 under the cursor (its pixels
differ from Phase 8's at x 336-463, y 80-143 once entry 4 is live, Phase 10 Task 3)."""
DESERT_SNAPS = {"select": 750, "fly-by": 1500, "start": 1681, "mid-lap": 5600}
"""The snapshots of Desert City and its mirror: the select with the entry under the cursor (750), the fly-by (1500),
the start (GO + 30) and mid-lap (5600, Phase 8's mid-race frame of Desert City; the HUD map cut from it)."""


CONVERTED = (3, 4, 7)
"""The courses the chain converts, in order, each at its own id (build.converted_set(..., n, slot=n,
scenery="model")): Desert City (Phase 8), since Phase 10 Task 3 National Park Speedway and since Task 4 Mermaid Lake,
whose entries 4 and 7 are live in select.ENTRIES (patch_select refuses a live id without its course)."""
MIRRORED = (0, 1, 2, 3, 4, 7)
"""The originals the chain mirrors, in order (mirror.mirrored_entry, ids o + 8): Beginner, Expert, Advanced, Desert City
(Phase 9) and, since Phase 10 Task 5, National Park Speedway (id 12) and Mermaid Lake (id 15), after the mirrors of
0-3, so the tables of 8-11 stay where they were. Entries 12 and 15 stay locked in select.ENTRIES (Pascal, 2026-11-11:
mirror mode leaves the product; Phase 11 shrinks the row to the 8 courses): the mirrors are on the set, not on the
row."""
TASK3_CONVERTED = (3, 4)
TASK3_ENTRIES = (
    S.Entry(0, "BEGINNER", "", False, True, "BEGINNER"),
    S.Entry(2, "ADVANCED", "", False, True, "ADVANCED"),
    S.Entry(1, "EXPERT", "", False, True, "EXPERT"),
    S.Entry(3, "DESERT CITY", "", False, True, "CIRCUIT EDITION"),
    S.Entry(4, "NATIONAL PARK SPEEDWAY", "", False, True, "CIRCUIT EDITION"),
    S.Entry(5, "CIRCUIT PIXIE", "COMING SOON", False, False),
    S.Entry(6, "RIN RIN RINK", "COMING SOON", False, False),
    S.Entry(7, "MERMAID LAKE", "COMING SOON", False, False),
    S.Entry(8, "BEGINNER", "", True, True, "BEGINNER"),
    S.Entry(10, "ADVANCED", "", True, True, "ADVANCED"),
    S.Entry(9, "EXPERT", "", True, True, "EXPERT"),
    S.Entry(11, "DESERT CITY", "", True, True, "CIRCUIT EDITION"),
    S.Entry(12, "NATIONAL PARK SPEEDWAY", "", True, False, "CIRCUIT EDITION"),
    S.Entry(13, "CIRCUIT PIXIE", "COMING SOON", True, False),
    S.Entry(14, "RIN RIN RINK", "COMING SOON", True, False),
    S.Entry(15, "MERMAID LAKE", "COMING SOON", True, False),
)
"""The chain and the row of Phase 10 Task 3's final set (`task3_set`, FINAL_SHA1): Mermaid Lake (entry 7) a
placeholder. A literal since Task 5 (the Task 4 review's M5): the live row (select.ENTRIES) can change without moving
it."""
TASK4_ENTRIES = TASK3_ENTRIES[:7] + (S.Entry(7, "MERMAID LAKE", "", False, True),) + TASK3_ENTRIES[8:]
"""The row of Phase 10 Task 4's final set (`task4_set`, FINAL_SHA1_T4): Task 3's with Mermaid Lake (entry 7) live,
entries 12 and 15 locked; built from TASK3_ENTRIES only, never from the live row."""


def _converted(rs: RomSet, disc, courses=CONVERTED) -> RomSet:  # noqa: F811
    """grow_ids, guard_new_ids, widen_ids, then each of `courses` (CONVERTED) converted at its own id."""
    out = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    for n in courses:
        out = build.converted_set(out, disc, n, slot=n, scenery="model")
    return out


def desert_mirrors(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """This task's set, the chain's order (test_convert_build.test_desert_city_before_select): grow_ids,
    guard_new_ids, widen_ids, Desert City converted at id 3 (build.converted_set(..., 3, slot=3, scenery="model"))
    and, since Phase 10 Task 3, National Park Speedway at id 4 and (Task 4) Mermaid Lake at id 7 (`_converted`: the
    mirrors' tables now land after theirs), the mirrors of Beginner, Expert, Advanced and Desert City at ids 8-11
    and (Phase 10 Task 5) of National Park Speedway and Mermaid Lake at 12 and 15 (mirror.mirrored_entry of MIRRORED),
    the select with M5A_ENTRIES (4, 7 and 8-11 live; 12 and 15 locked)."""
    from dtools.arcade import mirror as M
    out = _converted(rs, disc)
    for o in MIRRORED:
        out = M.mirrored_entry(out, o)
    return S.patch_select(out, S.M5A_ENTRIES)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_mirror_desert_city(tmp_path, romset_zip, disc):  # noqa: F811
    """Task 4 (discovery log, "Desert City mirrored"). On `desert_mirrors` with the start words raised (build.long_time
    on Desert City's time tables, which its mirror shares: the long-time set of test_desert_city_id3), custom MAME:

    - the Global Constraint: entries 0-2 (Beginner, Advanced, Expert; the set without long time) race call for call
      as on Task 3's reference set (the widened chain with Phase 8's row, no mirrors), SECONDS each: the geometry
      list on every frame from GEO_FROM, the rate log, the snapshots (SNAPS). Entry 3, Desert City, races call for
      call as on Phase 8's row (the widened chain, Desert City converted, then the select with M4A_ENTRIES: Phase
      8's order's bytes, test_convert_build.test_desert_city_before_select; long time too), SECONDS: the same
      geometry list on every frame from GEO_FROM, the same rate log, pixel-identical snapshots; and over its
      long-time lap holds Phase 8's numbers (test_convert_mame.ID3_MAX at ID3_MAX_AT, ID3_COUNT_MAX; lap 1 at
      ID3_LAP_END);
    - entry 11 from the select, Desert City's mirror (ID3_SECONDS): Review Focus 3, Desert City's standing start (the
      countdown timer equal to Desert City's on every frame to the first checkpoint, GO at ID3_GO, the player still
      to GO, rolling_start False); every ground call of the race pairs with Desert City's mirrored up to DESERT_PARTS
      (test_mirror._parting; the attract's calls equal); every opponent within SYMMETRY_TOL of its mirror through
      DESERT_OPPONENTS_TO; the scripted camera mirrored (Task 3's rule, CAMERA_Z_OFFSET) from CAMERA_FROM to
      PAIRED_TO; the frame-rate bar over the mirror's lap (GO to DESERT_MIRROR_LAP_END, every segment): a new display
      list on every frame but GO + 1 (ID3_REPEAT), at most BAR drawn polygons per list and the polygon count below
      course.OBC on every frame (the binding bar: no cap hit), the maximum no worse than Desert City's ID3_MAX by
      more than DESERT_MARGIN (set before measuring), no scenery drawn outside the visible cells and none left out
      (DESERT_LEFT_OUT); the numbers printed and frozen;
    - snapshots (DESERT_SNAPS) of the mirror and of Desert City, and the HUD map cut from mid-lap, to
      out/phase9/snaps/task4."""
    import math
    from PIL import Image
    from dtools.arcade import golden as G
    from dtools.arcade import mirror as M
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_convert_mame import (BAR, ID3_COUNT_MAX, ID3_GO, ID3_HUD, ID3_LAP_END, ID3_MAX, ID3_MAX_AT,
                                         ID3_REPEAT, ID3_SECONDS, _new_lists)
    from tests.test_mirror import SYMMETRY_TOL, _calls, _parting
    exe = _custom()
    stock = RomSet.from_zip(romset_zip)
    full = desert_mirrors(stock, disc)
    phase8 = S.patch_select(build.converted_set(widened_chain(stock), disc, 3, slot=3, scenery="model"),
                            S.M4A_ENTRIES)
    c3 = G.decode_course(full, 3)
    long_set, long_p8 = build.long_time(full, c3), build.long_time(phase8, c3)
    k = S.M5A_ENTRIES[DESERT_MIRROR_ENTRY].course
    mc = G.decode_course(full, k)
    assert k == 11 and mc.rolling_start[1] is False and mc.time_tables == c3.time_tables
    dest = SNAP_DIR / "task4"
    dest.mkdir(parents=True, exist_ok=True)
    snaps = ",".join(map(str, DESERT_SNAPS.values()))

    # the Global Constraint: the originals against Task 3's reference set (the widened chain with Phase 8's row, no
    # mirrors), call for call, as test_mirrors_of_originals did
    base = widened(stock)
    env = {"DAYTONA_GEO_FROM": str(GEO_FROM), "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": SNAPS}
    for e in MIRROR_ENTRIES:
        got = {}
        for name, s in (("ref", base), ("desert", full)):
            o = mame_build.run_set(s, tmp_path / f"{e}-{name}", SECONDS, romset_zip, {"DAYTONA_ENTRY": str(e), **env},
                                   logs=("geo", "rate"), mame=exe)
            lists, _ = _lists(o["geo"])
            o["geo"].unlink()
            got[name] = lists, runner.parse_rate(o["rate"]), _snaps(o["snap"])
        (lists, rate, shots), (d_lists, d_rate, d_shots) = got["ref"], got["desert"]
        assert set(lists) == set(range(GEO_FROM, max(lists) + 1)) and len(set(lists.values())) > 1000, e
        assert d_lists == lists, (e, min(f for f in lists if d_lists.get(f) != lists[f]))
        assert d_rate == rate and len(rate) > SECONDS * 57, e
        assert d_shots == shots and len(shots) == len(SNAPS.split(",")) + 1, e
    print(f"\nentries {sorted(MIRROR_ENTRIES)} on this set: call for call as on Task 3's reference set")

    # Desert City against Phase 8's row (Phase 8's order's bytes), then its long-time lap on this set
    early = ",".join(str(f) for f in DESERT_SNAPS.values() if f < SECONDS * 57)
    ref = mame_build.run_set(long_p8, tmp_path / "phase8", SECONDS, romset_zip,
                             {"DAYTONA_ENTRY": str(DESERT_ENTRY), "DAYTONA_GEO_FROM": str(GEO_FROM),
                              "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": early}, logs=("geo", "rate"), mame=exe)
    ref_lists, _ = _geo_counts(ref["geo"], long_p8)
    ref_rate, ref_snaps = runner.parse_rate(ref["rate"]), _snaps(ref["snap"])
    runs = {}
    for e in (DESERT_ENTRY, DESERT_MIRROR_ENTRY):
        run = mame_build.run_set(long_set, tmp_path / f"entry-{e}", ID3_SECONDS, romset_zip,
                                 {"DAYTONA_ENTRY": str(e), "DAYTONA_GEO_FROM": str(GEO_FROM), "DAYTONA_GEO_EVERY": "1",
                                  "DAYTONA_CELLS_FROM": str(BARS_FROM), "DAYTONA_SNAPS": snaps},
                                 logs=("geo", "rate", "ground", "progress", "cells"), mame=exe)
        hashes, per = _geo_counts(run["geo"], long_set, BARS_FROM)
        runs[e] = dict(run=run, hashes=hashes, per=per, rate=runner.parse_rate(run["rate"]),
                       cells=runner.parse_cells(run["cells"]), progress=runner.parse_progress(run["progress"]),
                       snaps=_snaps(run["snap"]), shots=sorted(run["snap"].rglob("*.png")))
    d3 = runs[DESERT_ENTRY]
    assert set(ref_lists) == set(range(GEO_FROM, max(ref_lists) + 1)) and len(set(ref_lists.values())) > 1000
    last = max(ref_lists)                                       # the shorter run's last frame may be cut short
    assert {f: d3["hashes"][f] for f in ref_lists if f < last} == {f: v for f, v in ref_lists.items() if f < last}
    assert len(ref_rate) > SECONDS * 57
    assert [x for x in d3["rate"] if x[0] < ref_rate[-1][0]] == [x for x in ref_rate if x[0] < ref_rate[-1][0]]
    assert len(ref_snaps) == early.count(",") + 2 and d3["snaps"][1:len(ref_snaps) - 1] == ref_snaps[1:-1]
    # the select (750): entry 4's panel is live on this set (Phase 10 Task 3: NATIONAL PARK SPEEDWAY, CIRCUIT EDITION,
    # against Phase 8's COMING SOON); every pixel outside that panel is Phase 8's
    import numpy as np
    outside = np.ones((384, 496), dtype=bool)
    outside[ENTRY4_PANEL[1]:ENTRY4_PANEL[3], ENTRY4_PANEL[0]:ENTRY4_PANEL[2]] = False
    assert mame_build.snapshot_match(d3["shots"][0], sorted(ref["snap"].rglob("*.png"))[0], tol=0, mask=outside) == 1.0

    def lap(e: int, go: int) -> dict:
        """The bars over the player's lap 1 of entry `e`'s run (test_desert_city_id3's measures)."""
        r = runs[e]
        c = G.decode_course(full, S.M5A_ENTRIES[e].course)
        rows, _ = r["progress"]
        player = {x.frame: (x.lap, x.segment) for x in rows if x.car == 0}
        ends = [f for f, (n, _) in player.items() if f > go and n == 1]
        end = min(ends) if ends else max(player)                # the run's end when the lap is not completed
        frames = range(go, end + 1)
        owner = {p.oba: cell for cell, lst in enumerate(c.cells) for p in c.object_lists[lst]}
        occupied = set(owner.values())
        per, cells = r["per"], r["cells"]
        assert set(frames) <= set(per) and set(frames) <= set(cells), e
        left_out = [f for f in frames if {x for x in cells[f][1] if x in occupied}
                    - {owner[a] for a in per[f][1] if a in owner}]
        strays = [f for f in frames if {owner[a] for a in per[f][1] if a in owner} - set(cells[f][1])]
        top = max(frames, key=lambda f: (per[f][0], -f))
        busiest = max(frames, key=lambda f: (cells[f][0], -f))
        return dict(end=end, lapped=bool(ends), segments=len({player[f][1] for f in frames}), road=len(c.road),
                    max=(per[top][0], top), count_max=(cells[busiest][0], busiest),
                    over=sum(per[f][0] > BAR for f in frames), cap_hits=sum(cells[f][0] >= C.OBC for f in frames),
                    left_out=left_out, strays=len(strays),
                    missed=sorted(set(frames) - _new_lists(r["rate"])))
    d3_lap = lap(DESERT_ENTRY, ID3_GO)
    print(f"\nentry 3 (Desert City, id 3) on this set: {d3_lap}")
    assert (d3_lap["max"], d3_lap["count_max"][0], d3_lap["end"]) == ((ID3_MAX, ID3_MAX_AT), ID3_COUNT_MAX,
                                                                      ID3_LAP_END)

    # the mirror: the start
    m = runs[DESERT_MIRROR_ENTRY]
    (_, o_timer), (_, m_timer) = d3["progress"], m["progress"]
    # the clock: equal on every frame (the countdown, the race) until either player's first checkpoint raises it
    rise = min(f for t in (o_timer, m_timer) for f in t if f - 1 in t and t[f] > t[f - 1])
    assert rise > ID3_GO + 1000
    assert {f: v for f, v in m_timer.items() if f < rise} == {f: v for f, v in o_timer.items() if f < rise}
    go = _go(m_timer)
    assert go == _go(o_timer) == ID3_GO, go
    player = runner.parse_car_positions(m["run"]["progress"])
    (x0, _, z0), (x1, _, z1) = player[BARS_FROM][0], player[go - 1][0]
    assert math.hypot(x1 - x0, z1 - z0) < 1e-3                  # standing: XZ still to GO
    # the pairing with Desert City
    a, b = _calls(d3["run"]["ground"]), _calls(m["run"]["ground"])
    assert {f: v for f, v in a.items() if f < RACE_LOAD} == {f: v for f, v in b.items() if f < RACE_LOAD}
    a, b = ({f: v for f, v in x.items() if RACE_LOAD <= f} for x in (a, b))
    first = _parting(a, b, max(a))
    pa = runner.parse_car_positions(d3["run"]["progress"])

    def near(car: int, f: int) -> bool:
        x, _, z = M.mirror_world(pa[f][car])
        return math.hypot(x - player[f][car][0], z - player[f][car][2]) <= SYMMETRY_TOL
    cars = [car for car in pa[1330] if car]
    opp = min((next((f for f in range(1330, max(pa) + 1) if not near(car, f)), max(pa) + 1), car) for car in cars)
    # the camera
    o_cam, m_cam = (runner.parse_camera(r["run"]["progress"]) for r in (d3, m))
    offsets, cam_xy = [], 0.0
    for f in range(CAMERA_FROM, PAIRED_TO + 1):
        (p, pitch, yaw, roll), (q, q_pitch, q_yaw, q_roll) = o_cam[f], m_cam[f]
        cam_xy = max(cam_xy, abs(p[0] - q[0]), abs(p[1] - q[1]))
        assert abs(pitch - q_pitch) <= CAMERA_ANGLE and abs(roll + q_roll) <= CAMERA_ANGLE, f
        assert abs(((yaw + q_yaw) & 0xFFFF) - 0x8000) <= CAMERA_ANGLE, (f, yaw, q_yaw)
        offsets.append((-1 - p[2]) - q[2])
    m_lap = lap(DESERT_MIRROR_ENTRY, go)
    report = dict(go=go, clock_equal_to=rise - 1, parts=first and first.frame, opponents_to=opp, camera_xy=cam_xy,
                  camera_z=(min(offsets), max(offsets)), lap=m_lap, margin=m_lap["max"][0] - ID3_MAX)
    print(f"entry 11 (Desert City's mirror, id 11): {report}")
    # snapshots before the frozen numbers, so a failing run still leaves them to look at
    for name, r in (("desert-city", d3), ("desert-city-mirror", m)):
        for p, (label, frame) in zip(r["shots"], DESERT_SNAPS.items()):
            shutil.copy(p, dest / f"{name}-{label}-{frame}.png")
            if label == "mid-lap":
                Image.open(p).crop(ID3_HUD).save(dest / f"{name}-hud-map-{frame}.png")
    assert first is not None and first.frame == DESERT_PARTS, first
    assert opp[0] - 1 == DESERT_OPPONENTS_TO, opp
    assert cam_xy <= SYMMETRY_TOL and CAMERA_Z_OFFSET[0] <= min(offsets) and max(offsets) <= CAMERA_Z_OFFSET[1]
    # the frame-rate bar
    assert m_lap["lapped"] and m_lap["segments"] == m_lap["road"] and m_lap["end"] == DESERT_MIRROR_LAP_END, m_lap
    assert set(m_lap["missed"]) <= ID3_REPEAT and set(m_lap["missed"]) == set(d3_lap["missed"]), m_lap["missed"]
    assert m_lap["over"] == m_lap["cap_hits"] == m_lap["strays"] == 0 and set(m_lap["left_out"]) == DESERT_LEFT_OUT
    assert (m_lap["max"], m_lap["count_max"][0]) == ((DESERT_MIRROR_MAX, DESERT_MIRROR_MAX_AT),
                                                     DESERT_MIRROR_COUNT_MAX)
    assert m_lap["max"][0] <= ID3_MAX + DESERT_MARGIN and m_lap["count_max"][0] < C.OBC


# --- Task 5: the select's generated panels, the MIRROR marking, the idle wheel (select.patch_select) ----------------

PANEL_SWEEP_FROM, PANEL_HOLD = 640, 30      # as test_sixteen_mame.test_select_scrolls: one entry every 30 frames
PANEL_LOCKED_STEPS = {1005: 12, 1035: 13, 1065: 14, 1130: 15}
"""Gas steps on each locked mirror placeholder, 12-15 (entry k held from 640 + 30 k for 30 frames; 15 to the timeout):
each step and the 20 frames after it lie inside its entry's hold."""
IDLE_WHEEL = {640: 128, 900: 120, 1200: 136}
"""The untouched run's paddle: centred, then off-centre to the left and to the right, all inside the centre band
(select.IDLE_BAND: wheel values 0, -5, 5)."""
SELECT_OPEN = 616                           # the select's first frame (state 0 at 616-617, state 1 from 618)
IDLE_SNAPS = {"select": 900, "race": 2300}  # the untouched run: the select (entry 0 under the cursor), the race
IDLE_TO = 4500                              # select_log's last frame: the race (the course id is the vote's)
PANEL_SECONDS = 90                          # each run: the select's timeout (~1580), Mission Select's, the vote, the race


def _wide_votes(trace_log: Path) -> list[int]:
    """The course ids the vote stores (memtrace V lines at the course id): on a widened set (course.widen_ids) the
    vote's `stob g2,0x501460` (0x32dc) runs in its trampoline `writer_32dc`, so its trace ip is the trampoline's + 8,
    not test_sixteen_mame.VOTE_IP."""
    ip = C.wide_code().labels["writer_32dc"] + 8
    return [e.value for e in trace.parse_trace(trace_log) if e.kind == "V" and e.addr == C.ID_BYTE and e.ip == ip]


@pytest.mark.mame
@pytest.mark.custom_mame
def test_select_panels(tmp_path, romset_zip, disc):  # noqa: F811
    """Task 5 (discovery log, "Panels"; Review Focus 5) on `desert_mirrors`, the phase's full row (select.ENTRIES:
    0-3 and 8-11 live), custom MAME:

    - the sweep: the wheel steps through the sixteen entries in order (PANEL_HOLD frames each from
      PANEL_SWEEP_FROM, select.paddle_for_entry); before it, the wheel untouched, the cursor is on entry 0 from the
      select's first frame. At each entry the game holds the entry, its course id and first_of; tile layers 0 and 1
      hold the records' panel maps and the window maps of the three positions shown, and the window mask covers the
      generated entries' interiors and the mirrors' label rows (`_mask_rows`); the generated chars and the twenty
      palettes are in char and palette RAM. One snapshot per entry, to out/phase9/snaps/task5/entryNN-<name>.png.
      Gas steps on each of entries 12-15 (locked placeholders) change nothing (state 1, the timer runs); the timer's
      end on entry 15 races entry 0 (Beginner, id 0: the ruling), and the vote stores 0.
    - the untouched wheel (IDLE_WHEEL: centred, then resting off-centre either side inside the centre band; DAYTONA_STEPS
      none; the gas from 2000, after the select's timer, ~1578, so no pedal edge picks the entry): the cursor stays on
      entry 0 through the
      select, the timeout races entry 0, the vote stores 0 and the race is course id 0 (snapshots IDLE_SNAPS)."""
    from dtools.arcade import tiles as TL
    from tests.test_sixteen_mame import _mask_rows, _select_run
    exe = _custom()
    full = desert_mirrors(RomSet.from_zip(romset_zip), disc)
    row, lay = S.select_row(full), S.layout_of(full)
    g = S.graphics(full, S.M5A_ENTRIES)
    assert row.live == tuple(e.live for e in S.M5A_ENTRIES) and row.live[8:12] == (True,) * 4
    tiles = [C.decode_select_panel(full.read_i960(C.SELECT_TABLES[0] + 44 * s, 44), s).tile for s in range(3)]
    frames = [PANEL_SWEEP_FROM + PANEL_HOLD * k for k in range(16)]
    at = [f + PANEL_HOLD - 4 for f in frames]
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{S.paddle_for_entry(k)}" for k, f in enumerate(frames)),
           "DAYTONA_SNAPS": ",".join(map(str, at)), "DAYTONA_STEPS": ",".join(map(str, PANEL_LOCKED_STEPS)),
           "DAYTONA_GAS_FROM": "2000", "DAYTONA_SELECT_FROM": "600", "DAYTONA_SELECT_TO": str(IDLE_TO),
           "DAYTONA_SELECT_TILES": ",".join(map(str, at)), "DAYTONA_SELECT_CHARS": "700"}
    log, _, snap = _select_run(full, tmp_path / "sweep", romset_zip, PANEL_SECONDS, env, exe)
    votes = _wide_votes(tmp_path / "sweep" / "trace.txt")
    dest = SNAP_DIR / "task5"
    dest.mkdir(parents=True, exist_ok=True)
    shots = sorted(Path(snap).rglob("*.png"))
    assert len(shots) >= 16
    for k, p in enumerate(shots[:16]):
        e = S.M5A_ENTRIES[k]
        shutil.copy(p, dest / f"entry{k:02d}-{e.name.lower().replace(' ', '-')}{'-mirror' if e.mirror else ''}.png")
    # the untouched wheel before the sweep: entry 0 from the select's first frame
    pre = {f: v for f, v in log["F"].items() if SELECT_OPEN <= f < PANEL_SWEEP_FROM + 2}
    assert pre and all(v[0] in (0, 1) and v[1:4] == (0, 0, 0) for v in pre.values()), pre
    assert pre[SELECT_OPEN][0] == 0 and any(v[0] == 1 for v in pre.values())
    blank = (S.LAYER1_BLANK,) * 180
    for k, f in enumerate(at):
        first = S.first_of(k)
        assert log["F"][f][:5] == (1, k, k, S.M5A_ENTRIES[k].course, first), (k, log["F"][f])
        for s in range(3):
            e = first + s
            x, y = tiles[s]
            maps = {m for recs in row.records for m in recs[e].maps}
            for layer, want in ((0, [TL.tile_map(full, m).tiles() for m in maps]), (1, [g.name_maps[e] or blank])):
                got = tuple(log["T"][(f, layer, y + r)][x + c] for r in range(10) for c in range(18))
                assert got in [tuple(w) for w in want], (k, s, layer)
        name, label = _mask_rows(g.windows, first, tiles)
        assert log["M"][f] == [tuple(name)] * 40 + [tuple(label)] * 24, (k, log["M"][f][0], log["M"][f][-1])
    for first, n in g.runs:                            # the uploaded chars (the rest of a block is never written)
        at = 32 * (first - S.CHAR_HOLE)
        assert log["C"][at:at + 32 * n] == TL.encode_chars(g.chars[first - S.CHAR_HOLE:first - S.CHAR_HOLE + n])
    assert log["P"][0x760] == sum(g.palettes, ()) and log["P"][0xF60] == g.grey * 10
    # the locked steps and the timer's end
    for step, k in PANEL_LOCKED_STEPS.items():
        after = [log["F"][f] for f in range(step, step + 20)]
        assert all(v[0] == 1 and v[1] == k for v in after) and after[-1][5] < after[0][5], (k, after[0], after[-1])
    go = min(f for f, v in log["F"].items() if f > max(PANEL_LOCKED_STEPS) and v[0] == 2)
    assert log["F"][go - 1][1] == 15 and log["F"][go][1:4] == (0, 0, 0), (log["F"][go - 1], log["F"][go])
    assert votes and set(votes) == {0}, votes
    assert log["F"][IDLE_TO][3] == 0

    # the untouched wheel: entry 0 throughout, the timeout races Beginner
    assert all(abs(S.wheel_value(p)) <= S.IDLE_BAND for p in IDLE_WHEEL.values())
    env = {"DAYTONA_WHEEL_SCRIPT": ",".join(f"{f}:{p}" for f, p in IDLE_WHEEL.items()), "DAYTONA_STEPS": "none",
           "DAYTONA_GAS_FROM": "2000", "DAYTONA_SELECT_FROM": "600",
           "DAYTONA_SELECT_TO": str(IDLE_TO), "DAYTONA_SNAPS": ",".join(map(str, IDLE_SNAPS.values()))}
    log, _, snap = _select_run(full, tmp_path / "idle", romset_zip, PANEL_SECONDS, env, exe)
    votes = _wide_votes(tmp_path / "idle" / "trace.txt")
    go = min(f for f, v in log["F"].items() if f > SELECT_OPEN and v[0] == 2)
    state1 = {f: v for f, v in log["F"].items() if SELECT_OPEN <= f < go and v[0] == 1}
    assert len(state1) > 900 and all(v[1:4] == (0, 0, 0) for v in state1.values())
    assert max(IDLE_WHEEL) + 30 < go                       # every resting position held well inside the select
    assert log["F"][go][1:4] == (0, 0, 0) and log["F"][go - 1][5] <= 1, (log["F"][go - 1], log["F"][go])
    assert votes and set(votes) == {0}, votes
    assert log["F"][IDLE_TO][3] == 0
    for p, (label, frame) in zip(sorted(Path(snap).rglob("*.png")), IDLE_SNAPS.items()):
        shutil.copy(p, dest / f"idle-{label}-{frame}.png")
    print(f"\nsnapshots: {sorted(x.name for x in dest.iterdir())}")


def final_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """The final set: `desert_mirrors`' chain with course.grow_rankings (Task 6) after the mirrors and before the
    select: grow_ids, guard_new_ids, widen_ids, Desert City converted at id 3 and (since Phase 10 Task 3) National Park
    Speedway at id 4 and (since Task 4) Mermaid Lake at id 7 (`_converted`), the mirrors of 0-3 at 8-11 and (since
    Task 5) of 4 and 7 at 12 and 15 (MIRRORED), a ranking table per id, the select with M5A_ENTRIES (12 and 15 locked:
    mirror mode is dropped, Pascal 2026-11-11; `mirror_row_set` makes them live for the tests). Phase 9's own final
    set is `m4b_set`, Task 3's `task3_set`, Task 4's `task4_set`."""
    return S.patch_select(_final_chain(rs, disc), S.M5A_ENTRIES)


def _final_chain(rs: RomSet, disc, mirrored=MIRRORED) -> RomSet:  # noqa: F811
    """`final_set` before the select: `_converted`, mirrored_entry of each of `mirrored`, course.grow_rankings."""
    from dtools.arcade import mirror as M
    out = _converted(rs, disc)
    for o in mirrored:
        out = M.mirrored_entry(out, o)
    return C.grow_rankings(out)


def task5_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """Phase 10 Task 5's final set with its row frozen: `final_set`'s chain (the mirrors of 0-4 and 7) with the select
    patched with TASK4_ENTRIES (this task changes no entry), not the live row: the byte check
    (tests/test_convert_build_courses.py test_final_set_with_mirrors_12_15) stays valid when Phase 11 replaces
    ENTRIES. The same bytes as `final_set` (M5A_ENTRIES == TASK4_ENTRIES:
    tests/test_select.py test_frozen_rows_unchanged)."""
    return S.patch_select(_final_chain(rs, disc), TASK4_ENTRIES)


def task4_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """Phase 10 Task 4's final set (tests/test_convert_build_courses.py FINAL_SHA1_T4, c7a4855's
    `final_set`): `final_set`'s chain with the mirrors of 0-3 only (no id 12 or 15), the select with TASK4_ENTRIES."""
    return S.patch_select(_final_chain(rs, disc, MIRRORED[:4]), TASK4_ENTRIES)


def task3_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """Phase 10 Task 3's final set's chain (tests/test_convert_build_courses.py FINAL_SHA1; since Task 4's fix round 1
    id 4 without NAMED_PROPS[4], the big wheel's frame):
    `final_set`'s chain with Desert City and National Park Speedway converted (TASK3_CONVERTED), the mirrors of 0-3, a
    ranking table per id, the select with TASK3_ENTRIES (entry 7 a placeholder; the set holds no id 7)."""
    from dtools.arcade import mirror as M
    out = _converted(rs, disc, TASK3_CONVERTED)
    for o in (0, 1, 2, 3):
        out = M.mirrored_entry(out, o)
    return S.patch_select(C.grow_rankings(out), TASK3_ENTRIES)


def m4b_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """Phase 9's final set, frozen (the M4b playtest set's, tools/desert_city_playtest.py --mirrors): grow_ids,
    guard_new_ids, widen_ids, Desert City converted at id 3, the mirrors of 0-3 at 8-11, a ranking table per id, the
    select with Phase 9's row (select.M4B_ENTRIES: entry 4 a placeholder; the set holds no id 4)."""
    from dtools.arcade import mirror as M
    out = build.converted_set(C.widen_ids(C.guard_new_ids(C.grow_ids(rs))), disc, 3, slot=3, scenery="model")
    for o in (0, 1, 2, 3):
        out = M.mirrored_entry(out, o)
    return S.patch_select(C.grow_rankings(out), S.M4B_ENTRIES)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_originals_on_final_set(tmp_path, romset_zip, disc):  # noqa: F811
    """The Global Constraint on the phase's final set (`final_set`: the mirrors, the rankings, the new select;
    Task 6 reran it with the rankings): entries 0-2 (Beginner,
    Advanced, Expert) race as the stock set's panels 0-2 on stock MAME (test_sixteen_mame.test_originals_from_select's
    measures): the same geometry display list on every frame from RACE_FROM, the same render-rate writes and video
    mode from RACE_FROM and lines from RATE_HASH_FROM, pixel-identical race snapshots (RACE_SNAPS and the last)."""
    originals_as_stock(tmp_path, romset_zip, final_set(RomSet.from_zip(romset_zip), disc))


def originals_as_stock(tmp_path, romset_zip, full: RomSet, row: dict | None = None,
                       stock_env: dict | None = None) -> None:
    """test_originals_on_final_set's checks on the set `full`: entries 0-2 from its select (autoplay.lua's
    DAYTONA_ENTRY, with `row`'s env: {"DAYTONA_ROW": "8"} for a row of 8) against the stock set's panels 0-2 on stock
    MAME (with `stock_env`'s env: a set with the Course Direction screen races 52 frames later, stock with
    DAYTONA_DIRECTION=late)."""
    from dtools.mame import build as mame_build
    from dtools.mame import runner
    from tests.test_sixteen_mame import RACE_FROM, RACE_SNAPS, RATE_HASH_FROM
    exe = _custom()
    env = {"DAYTONA_GEO_FROM": str(RACE_FROM), "DAYTONA_GEO_EVERY": "1", "DAYTONA_SNAPS": RACE_SNAPS}
    for k in range(3):
        got = {}
        for name, s, binary, sel in (("stock", None, "mame", {"DAYTONA_COURSE": str(k), **(stock_env or {})}),
                                     ("final", full, exe, {"DAYTONA_ENTRY": str(k), **(row or {})})):
            out = mame_build.run_set(s, tmp_path / f"{k}-{name}", SECONDS, romset_zip, {**sel, **env},
                                     logs=("geo", "rate"), mame=binary)
            lists, _ = _lists(out["geo"])
            out["geo"].unlink()
            rate = [r for r in runner.parse_rate(out["rate"]) if r[0] >= RACE_FROM]
            got[name] = lists, rate, _snaps(out["snap"])
        (lists, rate, snaps), (f_lists, f_rate, f_snaps) = got["stock"], got["final"]
        assert set(lists) == set(range(RACE_FROM, max(lists) + 1)) and len(set(lists.values())) > 1000, k
        assert f_lists == lists, (k, min(f for f in lists if f_lists.get(f) != lists[f]))
        assert [r[:3] for r in f_rate] == [r[:3] for r in rate], k
        assert [r for r in f_rate if r[0] >= RATE_HASH_FROM] == [r for r in rate if r[0] >= RATE_HASH_FROM], k
        assert len(f_snaps) == len(snaps) == len(RACE_SNAPS.split(",")) + 1 and f_snaps == snaps, k


# --- Phase 10 Task 1: the shared empty actor table (build.EMPTY_ACTORS; docs/spike/2026-11-courses-4-7.md) ----------

SPAWN_READ_IP = 0x220050    # 0x22004c ld (r13),r10: the spawner (0x22002c) loads one actor-table word (ip after it)
ACTOR_POINTER = 0x501444    # ACTOR_TABLES[id], stored by the load (0x46bc) and read by the spawner (0x220040)
SHARED_SECONDS = 60         # past the race's load (frame 1328) and GO


@pytest.mark.mame
@pytest.mark.custom_mame
def test_shared_actor_table(tmp_path, romset_zip, disc):  # noqa: F811
    """Review Focus 3 (Phase 10 Task 1): the shared table is only read, and not past its end. On `final_set`, whose
    Desert City (id 3) and its mirror (id 11) both name build.EMPTY_ACTORS, entries 3 and 11 from the select, each
    under a trace over the table (program window and its program-ROM alias), the 1,024 bytes after it (mirror 8's
    table and lists) and the pointer word ACTOR_POINTER: the race's load stores EMPTY_ACTORS there, the spawner reads
    the table once, on the next frame, at SPAWN_READ_IP only, as many reads in each 256-byte page as the table has
    words there (256 in all) and nothing past them; nothing writes the table, and nothing reads its alias."""
    from dtools.mame import build as mame_build
    exe = _custom()
    full = final_set(RomSet.from_zip(romset_zip), disc)
    t, n = build.EMPTY_ACTORS, 4 * C.GRID_CELLS
    alias = t - C.PROGRAM_WINDOW[0] + 0x20000
    reads = [(t, t + 2 * n - 1), (alias, alias + n - 1), (ACTOR_POINTER, ACTOR_POINTER + 3)]
    writes = [(t, t + n - 1), (alias, alias + n - 1), (ACTOR_POINTER, ACTOR_POINTER + 3)]
    for e in (DESERT_ENTRY, DESERT_MIRROR_ENTRY):
        assert _actor_table_of(full, S.M5A_ENTRIES[e].course) == t, e
        log = (tmp_path / f"trace-{e}.txt").resolve()
        mame_build.run_set(full, tmp_path / f"run-{e}", SHARED_SECONDS, romset_zip,
                           {"DAYTONA_ENTRY": str(e), "DAYTONA_TRACE_LOG": str(log),
                            "DAYTONA_TRACE_READ": ",".join(f"{a:x}-{b:x}" for a, b in reads),
                            "DAYTONA_TRACE_WRITE": ",".join(f"{a:x}-{b:x}!" for a, b in writes)}, mame=exe)
        events = trace.parse_trace(log)
        # R / W events carry their 256-byte page (memtrace aggregates per page), V events the exact address
        pages = {p: (min(p + 0x100, t + n) - max(p, t)) // 4 for p in range(t & ~0xFF, t + n, 0x100)}
        beyond = range((t + n + 0xFF) & ~0xFF, t + 2 * n, 0x100)      # pages wholly past the table
        stores = [x for x in events if x.kind == "V" and x.addr == ACTOR_POINTER]
        table = [x for x in events if x.kind == "R" and x.addr in pages]
        past = [x for x in events if x.kind == "R" and x.addr in beyond]
        pointer, lo = ACTOR_POINTER & ~0xFF, alias & ~0xFF
        other = [x for x in events if (x.kind == "W" and x.addr != pointer)
                 or (x.kind == "V" and x.addr != ACTOR_POINTER) or (x.kind == "R" and lo <= x.addr < alias + n)]
        per_page = collections.Counter()
        for x in table:
            per_page[x.addr] += x.count
        print(f"\nentry {e}: pointer stores {[(x.frame, hex(x.value)) for x in stores]}, table reads "
              f"{sum(per_page.values())} at {sorted({hex(x.ip) for x in table})} on frames "
              f"{sorted({x.frame for x in table})}, per page {dict((hex(k), v) for k, v in sorted(per_page.items()))}, "
              f"past it {len(past)}, writes or alias reads {len(other)}")
        assert stores and stores[-1].value == t, e                 # the race's load (the attract's is Beginner's)
        assert {x.ip for x in table} == {SPAWN_READ_IP} and {x.frame for x in table} == {stores[-1].frame + 1}, e
        # each page read as many times as it holds table words (256 in all): the last page's bytes past the table
        # share it, so a read past the end would show as one read too many there
        assert per_page == pages and sum(pages.values()) == C.GRID_CELLS, (e, per_page)
        assert past == [] and other == [], (e, past[:4], other[:4])


def _actor_table_of(rs: RomSet, k: int) -> int:
    """ACTOR_TABLES[k] of the grown set `rs`."""
    return struct.unpack("<I", rs.read_i960(C.id_entry(C.id_tables(rs), "ACTOR_TABLES", k), 4))[0]


# --- Task 6: rankings per entry (course.grow_rankings) --------------------------------------------------------------

RANK_SECONDS = 80           # a one-lap race (one_lap): the goal and its record at 3829-3849, then the name entry
BOOT_SECONDS = 15           # past the boot's signature check and reset (frame 26)
BOOT_END = 100              # the frames of the boot: the reset's writes (fresh NVRAM: all sixteen tables) end at 26
RECORDS = {"stock": 1683, "mirror": 1657, "second": 1677}
"""The race times (0x5010cc, frames) the one-lap races record (measured, discovery log "Rankings"): Beginner from a
fresh NVRAM (the stock set, Phase 8's, the final set: 1,683, Phase 1's figure), Beginner's mirror (entry 8, fresh) and
Beginner booted on an SRAM that passes the signature check (no reset at boot: the stock set and the final set alike,
6 frames quicker)."""
RANKING_WRITER_IPS = frozenset({0x227458, 0x2274A0, 0x2274C0, 0x2274CC, 0x2274D8, 0x227534, 0x227538, 0x227544,
                                0x227548, 0x227554, 0x227558, 0x227564, 0x227568, 0x227574, 0x227578, 0x227590,
                                0x228494})
"""Trace ips of the race's end writing a record (0x227408: the new last entry 0x227454-0x2274d4, the bubble's swaps
0x227530-0x227574, the newest-entry word 0x22758c); and (M9 stage 4, a run long enough to reach the name entry:
tests/test_dirscreen_mame.py test_reverse_best_time) the name entry's initials 0x5013a0 into the record's +0x14
(0x228490)."""
SYSTEM_WRITER_IPS = frozenset({0x1DB74, 0x228F9C, 0x228FA4, 0x22925C, 0x229274, 0x22928C, 0x22929C, 0x2292A4,
                               0x2292AC, 0x2292BC, 0x2292C4, 0x2292CC, 0x22958C, 0x229648, 0x2297F0, 0x229800,
                               0x229810, 0x229820, 0x2298C4, 0x2298D4, 0x2298E4, 0x2298F4,
                               0x228FDC, 0x228FF8, 0x229008, 0x229010})
"""Trace ips of the writes into the system area 0x1d00000-0x1d001ff after the boot of a mirror's game (measured, entry
8, discovery log "Rankings"): the coin counter (0x1db74), the race-end totals 0x1d00138 / 0x1d0013c (0x228f98,
0x228fa0), the checksum and the backup copy 0x1d00180-0x1d001ff (0x229258-0x2292c8), the other counters 0x1d00110,
0x1d00118, 0x1d00128-0x1d0012e; and (M9 stage 4, a run long enough to reach the game's end:
tests/test_dirscreen_mame.py test_reverse_best_time) the game's totals at 0x228fb0: the play time 0x1d0010c, the
longest and shortest 0x1d00130 / 0x1d00134, then 0x1d00138 / 0x1d0013c (ips 0x228fdc-0x229010, each the instruction
after its store)."""
COUNTERS = (0x01D00140, 0x01D00170)
"""The per-id bookkeeping counters (four u32[3] arrays, course.COUNTER_GUARD): a mirror's game writes none of them."""


def one_lap(rs: RomSet, ids) -> RomSet:
    """`rs` with the lap count of course ids `ids` (LAP_COUNTS, mode 0) set to 1, so autoplay's race reaches its goal
    and the record is written within RANK_SECONDS (Phase 1's make_laps1: docs/spike/2026-11-arcade-course-tables.md,
    "Rankings"). A test variant: the race itself is the set's."""
    out = rs
    for k in ids:
        at = struct.unpack("<I", rs.read_i960(C.LAP_COUNTS, 4))[0] + 4 * k
        out = out.with_patch(*C.rom_word_at(at), C.encode_laps(1))
    return out


def _sram(workdir: Path) -> bytes:
    """The backup SRAM (MAME's `backup1`, little-endian bytes of 0x01d00000-0x01d03fff) a run_set run left."""
    return (workdir / "run" / "nvram" / "daytona" / "backup1").read_bytes()


def _seed(workdir: Path, src: Path) -> None:
    """Start the run in `workdir` on the NVRAM `src`'s run left (its whole nvram directory: backup SRAM, I/O board)."""
    shutil.copytree(src / "run" / "nvram", workdir / "run" / "nvram")


def _at(sram: bytes, addr: int, n: int) -> bytes:
    return sram[addr - C.SRAM[0]:addr - C.SRAM[0] + n]


def _word(sram: bytes, addr: int) -> int:
    return struct.unpack("<I", _at(sram, addr, 4))[0]


def _recorded(table: bytes, fresh: bytes, time: int) -> bool:
    """`table` is `fresh` with one record of `time` on top: entry 0 (rank 1, time, view 2, flags 0, the time shown,
    no name), every other entry the fresh entry above it under its own rank (0x227508 swaps fields 0x04-0x14)."""
    rows = [struct.unpack_from("<6I", table, 24 * i) for i in range(C.RANKING_ENTRIES)]
    old = [struct.unpack_from("<6I", fresh, 24 * i) for i in range(C.RANKING_ENTRIES)]
    return rows[0] == (1, time, 2, 0, time, 0x202020) and all(
        rows[i] == (i + 1,) + old[i - 1][1:] for i in range(1, C.RANKING_ENTRIES))


def ranking_tables(sram: bytes, lay) -> list[bytes]:
    """The sixteen ranking tables of the layout `lay` (course.rankings_layout) in backup SRAM `sram`."""
    return [_at(sram, t, C.RANKING_TABLE_SIZE) for t in lay.tables]


def ranking_run(tmp_path: Path, romset_zip, name: str, s, sel: dict, seconds: int = RANK_SECONDS, seed=None,
                mame=None, extra=None) -> bytes:
    """The backup SRAM a run_set run of the set `s` (None: the stock set) in tmp_path / name left: autoplay env `sel`
    and `extra`, on the custom build unless `mame` names another; started on the NVRAM the run `seed` (a sibling
    name) left, else a fresh one."""
    from dtools.mame import build as mame_build
    if seed is not None:
        _seed(tmp_path / name, tmp_path / seed)
    mame_build.run_set(s, tmp_path / name, seconds, romset_zip, {**sel, **(extra or {})}, mame=mame or _custom())
    return _sram(tmp_path / name)


def traced_ranking_run(tmp_path: Path, romset_zip, name: str, s, sel: dict, seconds: int = RANK_SECONDS,
                       seed=None) -> tuple[bytes, Path]:
    """ranking_run with every write into backup SRAM traced: (the SRAM, the trace log)."""
    log = (tmp_path / f"{name}.trace").resolve()
    return ranking_run(tmp_path, romset_zip, name, s, sel, seconds, seed, extra={
        "DAYTONA_TRACE_LOG": str(log), "DAYTONA_TRACE_WRITE": f"{C.SRAM[0]:x}-{C.SRAM[1] - 1:x}!"}), log


def check_own_ranking_writes(log: Path, lay, k: int) -> list:
    """Review Focus 1 of Task 6 (and Phase 11's Review Focus 3): after the boot (BOOT_END) a race on course id `k`
    writes the system area only from SYSTEM_WRITER_IPS and none of the per-id counters (COUNTER_GUARD), and past the
    system area only id k's ranking table and newest-entry word, from RANKING_WRITER_IPS. Returns the trace's write
    values (for the caller's own boot checks)."""
    events = [e for e in trace.parse_trace(log) if e.kind == "V"]
    system = [e for e in events if e.frame > BOOT_END and e.addr < 0x01D00200]
    assert system and {e.ip for e in system} <= SYSTEM_WRITER_IPS, sorted({hex(e.ip) for e in system})
    assert not [e for e in system if COUNTERS[0] <= e.addr < COUNTERS[1]]     # COUNTER_GUARD: id k is not counted
    late = [e for e in events if e.frame > BOOT_END and e.addr >= 0x01D00200]
    own = range(lay.tables[k], lay.tables[k] + C.RANKING_TABLE_SIZE)
    assert late and {e.ip for e in late} <= RANKING_WRITER_IPS, sorted({hex(e.ip) for e in late})
    assert all(e.addr in own or e.addr == lay.last_entry[k] for e in late), \
        sorted({hex(e.addr) for e in late if e.addr not in own})
    return events


@pytest.mark.mame
@pytest.mark.custom_mame
def test_rankings_per_entry(tmp_path, romset_zip, disc):  # noqa: F811
    """Task 6 (discovery log, "Rankings"), on `final_set` and its one-lap variant (`one_lap` of ids 0 and 8):

    - Review Focus 4: an SRAM written by the stock set (stock MAME) and one by Phase 8's set (`sixteen`), each with a
      one-lap Beginner record in the stock table (0x1d00360) and the stock signature, boot the final set to the new
      signature and sixteen fresh tables (course.fresh_ranking: ids 0-2 the stock defaults) with no newest entry; the
      old tables are left as they were (no longer read) and nothing past the new words is written;
    - Review Focus 1: entry 8 (Beginner's mirror) raced to its goal from a fresh NVRAM writes id 8's table and its
      newest-entry word only (a write trace of all SRAM: after the boot every write past the system area is
      RANKING_WRITER_IPS' into those, every write inside it SYSTEM_WRITER_IPS', none into the per-id counters), the
      record on top; id 0's table stays fresh;
    - then Beginner (entry 0), booted on that SRAM: its record lands in id 0's table, equal to the one the stock set
      records booted on its own SRAM (RECORDS["second"]), id 8's record is kept, the other fourteen stay fresh; and
      Beginner from a fresh NVRAM records the stock set's table byte for byte (RECORDS["stock"])."""
    stock = RomSet.from_zip(romset_zip)
    full = final_set(stock, disc)
    test = one_lap(full, (0, 8))
    lay = C.rankings_layout(full)
    fresh = [C.fresh_ranking(full, k) for k in range(C.COURSE_IDS)]
    assert fresh[:3] == [C.fresh_ranking(stock, k) for k in range(3)]

    def tables(sram: bytes) -> list[bytes]:
        return ranking_tables(sram, lay)

    def run(name, s, sel, seconds=RANK_SECONDS, seed=None, mame=None):
        return ranking_run(tmp_path, romset_zip, name, s, sel, seconds, seed, mame)

    stock_t0 = (0x01D00360, C.RANKING_TABLE_SIZE)
    # the old SRAMs: one-lap Beginner records in the stock table, the stock signature
    olds = {"stock": run("stock", one_lap(stock, (0,)), {"DAYTONA_COURSE": "0"}, mame="mame"),
            "phase8": run("phase8", one_lap(sixteen(stock), (0,)), {"DAYTONA_ENTRY": "0"})}
    for name, sram in olds.items():
        assert _at(sram, 0x01D00200, 24) == C.SIGNATURE_STOCK, name
        assert _recorded(_at(sram, *stock_t0), C.fresh_ranking(stock, 0), RECORDS["stock"]), name
        assert set(_at(sram, C.RANKING_PAD, C.SRAM[1] - C.RANKING_PAD)) == {0xFF}, name    # never written
    # Review Focus 4: the final set boots each to sixteen fresh tables
    for name, old in olds.items():
        sram = run(f"boot-{name}", full, {}, BOOT_SECONDS, seed=name)
        assert _at(sram, 0x01D00200, 24) == C.SIGNATURE_RANKED, name
        assert tables(sram) == fresh, name
        assert {_word(sram, a) for a in lay.last_entry} == {0xFFFFFFFF}, name
        assert _at(sram, 0x01D00220, 0x01D00C28 - 0x01D00220) == _at(old, 0x01D00220, 0x01D00C28 - 0x01D00220), name
        assert set(_at(sram, lay.free[0], C.SRAM[1] - lay.free[0])) == {0xFF}, name
    # Review Focus 1: a record on entry 8 lands in id 8's table only
    sram8, log = traced_ranking_run(tmp_path, romset_zip, "entry8", test, {"DAYTONA_ENTRY": "8"})
    t8 = tables(sram8)
    assert [k for k in range(C.COURSE_IDS) if t8[k] != fresh[k]] == [8]
    assert _recorded(t8[8], fresh[8], RECORDS["mirror"])
    assert [_word(sram8, a) for a in lay.last_entry] == [lay.tables[8] if k == 8 else 0xFFFFFFFF
                                                         for k in range(C.COURSE_IDS)]
    events = check_own_ranking_writes(log, lay, 8)      # the system area, id 8's table and newest-entry word only
    boot = {(a - lay.tables[0]) // C.RANKING_TABLE_SIZE for e in events if e.frame <= BOOT_END
            for a in (e.addr,) if lay.tables[0] <= a < C.RANKING_LAST_SRAM}
    assert boot == set(range(C.COURSE_IDS))            # fresh NVRAM: the reset writes all sixteen at boot
    # then Beginner on that SRAM, as the stock set on its own
    sram0 = run("entry0", test, {"DAYTONA_ENTRY": "0"}, seed="entry8")
    t0 = tables(sram0)
    assert [k for k in range(C.COURSE_IDS) if t0[k] != fresh[k]] == [0, 8]
    assert t0[8] == t8[8] and _recorded(t0[0], fresh[0], RECORDS["second"])
    assert [_word(sram0, a) for a in lay.last_entry] == [
        lay.tables[k] if k in (0, 8) else 0xFFFFFFFF for k in range(C.COURSE_IDS)]
    again = run("stock-again", one_lap(stock, (0,)), {"DAYTONA_COURSE": "0"}, seed="stock", mame="mame")
    assert _at(again, stock_t0[0], 24) == t0[0][:24]
    sram_f = run("entry0-fresh", test, {"DAYTONA_ENTRY": "0"})
    assert tables(sram_f)[0] == _at(olds["stock"], *stock_t0)
