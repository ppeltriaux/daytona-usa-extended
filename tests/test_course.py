import re
import struct

import numpy as np
import pytest

from dtools.arcade.course import (ACTOR_TABLES, COURSE_COUNT, COURSE_ENTRY_SIZE, COURSE_NAMES, COURSE_POINTERS,
                                  COURSE_TABLE, GRID_CELLS, GRID_TABLES, LIBRARY_BOUND_AT, OBC, OBJECT_LIBRARY,
                                  Actor, CourseEntry,
                                  CourseError, Placement, actors, cell_box, cell_of, check_rom_object, course_table,
                                  encode_actor, encode_course_entry, encode_placement, scenery,
                                  unmodelled_actors)
from dtools.arcade.romset import RomSet

DATA = 0x02000000
FREE_RUN = (0x65D120, 0x65D120 + 667360)       # the data image's free run (docs/formats/model2-rom-images.md)


def _rs(program: bytes | bytearray, data: bytes | bytearray) -> RomSet:
    return RomSet(program=bytes(program), data=bytes(data), copro_data=b"", polygons=b"", textures=b"", extra={})


def _words(course: int) -> list[int]:
    """Eight distinct record words for a synthetic course: six data-ROM pointers, then two unknown words."""
    base = DATA + 0x100000 * (course + 1)
    return [base + 0x10 * k for k in range(6)] + [0xA0000000 + course, 0xB0000000 + course]


def _synthetic(records=None, pointers=None) -> RomSet:
    program, data = bytearray(256 * 1024), bytearray(16 << 20)
    for i in range(COURSE_COUNT):
        ptr = (pointers or {}).get(i, COURSE_TABLE + i * COURSE_ENTRY_SIZE)
        struct.pack_into("<I", program, COURSE_POINTERS + 4 * i, ptr)
        words = (records or {}).get(i, _words(i))
        struct.pack_into("<8I", data, COURSE_TABLE - DATA + i * COURSE_ENTRY_SIZE, *words)
    return _rs(program, data)


def test_course_table_synthetic():
    entries = course_table(_synthetic())
    assert [e.course for e in entries] == [0, 1, 2]                 # id order: Beginner, Expert, Advanced
    assert [e.name for e in entries] == ["Beginner", "Expert", "Advanced"]
    e = entries[1]
    assert (e.upload_list, e.step_table, e.block_ref_a, e.block_ref_b, e.line_table, e.record_table) == \
        (0x200000, 0x200010, 0x200020, 0x200030, 0x200040, 0x200050)  # data-ROM offsets, not i960 addresses
    assert e.raw == (0xA0000001, 0xB0000001)
    assert e.blobs() == {"upload_list": 0x200000, "step_table": 0x200010, "block_ref_a": 0x200020,
                         "block_ref_b": 0x200030, "line_table": 0x200040, "record_table": 0x200050}


def test_constants():
    assert COURSE_TABLE == 0x02600020 and COURSE_ENTRY_SIZE == 0x20 and COURSE_POINTERS == 0x4770
    assert COURSE_COUNT == 3 and COURSE_NAMES == ("Beginner", "Expert", "Advanced")


def test_pointer_outside_data_rom_raises():
    words = _words(2)
    words[4] = 0x00238634                                           # line_table pointing into the program window
    with pytest.raises(CourseError, match=r"0x2600070.*0x238634"):  # the field's address and its value
        course_table(_synthetic(records={2: words}))


def test_pointer_past_data_rom_raises():
    words = _words(0)
    words[0] = 0x03000000
    with pytest.raises(CourseError, match=r"0x2600020.*0x3000000"):
        course_table(_synthetic(records={0: words}))


def test_record_pointer_not_at_table_raises():
    # the game's per-id pointer array (program ROM) must point at the records this decoder reads
    with pytest.raises(CourseError, match=r"0x4774.*0x2600060"):
        course_table(_synthetic(pointers={1: COURSE_TABLE + 2 * COURSE_ENTRY_SIZE}))


def test_course_entry_round_trip():
    rs = _synthetic()
    for e in course_table(rs):
        off = COURSE_TABLE - DATA + e.course * COURSE_ENTRY_SIZE
        assert encode_course_entry(e) == rs.data[off:off + COURSE_ENTRY_SIZE]


@pytest.mark.parametrize("field, value", [("line_table", -4), ("upload_list", 0x2000000), ("record_table", 1.5)])
def test_encode_refuses_offset_outside_data_rom(field, value):
    e = course_table(_synthetic())[0]
    with pytest.raises(CourseError, match=field):
        encode_course_entry(CourseEntry(**{**e.__dict__, field: value}))


@pytest.mark.parametrize("raw", [(1,), (1, 2, 3), (0, 1 << 32), (-1, 0)])
def test_encode_refuses_bad_raw(raw):
    e = course_table(_synthetic())[0]
    with pytest.raises(CourseError, match="raw"):
        encode_course_entry(CourseEntry(**{**e.__dict__, "raw": raw}))


def test_name_of_unknown_id():
    e = CourseEntry(3, 0, 0, 0, 0, 0, 0, (0, 0))
    assert e.name == "course 3"


def test_course_table_real(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    entries = course_table(rs)
    assert [e.course for e in entries] == [0, 1, 2]
    for e in entries:
        for name, off in e.blobs().items():
            assert 0 <= off < len(rs.data), (e.name, name)
            assert not FREE_RUN[0] <= off < FREE_RUN[1], (e.name, name)
        off = COURSE_TABLE - DATA + e.course * COURSE_ENTRY_SIZE
        assert encode_course_entry(e) == rs.data[off:off + COURSE_ENTRY_SIZE]
    beginner, expert, advanced = entries
    assert beginner.step_table == 0x000000 and expert.step_table == 0x010A24 and advanced.step_table == 0x043200
    assert (beginner.upload_list, expert.upload_list, advanced.upload_list) == (0x074240, 0x074268, 0x081180)
    assert (beginner.line_table, expert.line_table, advanced.line_table) == (0x6000A0, 0x60B360, 0x641DC0)
    # the step table and the record table start with the same count (195, 990, 486)
    for e in entries:
        n = struct.unpack_from("<I", rs.data, e.step_table)[0]
        assert struct.unpack_from("<I", rs.data, e.record_table)[0] == n
    assert [struct.unpack_from("<I", rs.data, e.step_table)[0] for e in entries] == [195, 990, 486]


# --- scenery: the cell grid, the actors, the object entries ------------------------------------------------------

LIB = OBJECT_LIBRARY[0] - DATA                 # data-ROM offset of the object library (0x83fbb4)
WINDOW = 0x00200000                            # program window address - program offset
STATIC_PROP, SWITCH, BILLBOARD, OTHER = 0x220118, 0x220A54, 0x221E40, 0x2201BC
LDA_G0 = 0x8C803000                            # `lda <abs32>,g0`, the instruction word before each switch list


def _entry(course: int) -> CourseEntry:
    return CourseEntry(course, 0, 0, 0, 0, 0, 0, (0, 0))


def _scenery_rs(course: int = 1, **broken) -> RomSet:
    """A synthetic set for `course`: a 256-cell grid whose cell k holds one object (cell 5 two), and one actor slot with
    a static prop, a switch, a view-angle billboard and an actor whose handler is not modelled."""
    program, data = bytearray(256 * 1024), bytearray(16 << 20)
    lists = LIB + 0x40                          # object lists, 0-terminated, inside the library
    cells = []
    for k in range(GRID_CELLS):
        cells.append(DATA + lists)
        for j in range(2 if k == 5 else 1):
            struct.pack_into("<4I", data, lists, 0x800000 | (0x1000 + 0x20 * k + j), 0x50000 + 8 * k, 4 * k + j, k % 3)
            lists += 16
        lists += 4                              # the terminating oba 0
    extra = []                                  # lists for the actors, after the cells'
    for k in range(3 + 64):
        extra.append(DATA + lists)
        struct.pack_into("<4I", data, lists, 0x900000 | (0x100 * k), 0x60000 + k, 0x7000 + k, 0)
        lists += 20
    grid = 0x900000
    struct.pack_into(f"<{GRID_CELLS}I", data, grid, *cells)
    if "cell" in broken:
        struct.pack_into("<I", data, grid + 4 * 7, broken["cell"])
    if "ram_object" in broken:
        struct.pack_into("<I", data, cells[9] - DATA, broken["ram_object"])
    struct.pack_into("<I", data, GRID_TABLES - DATA + 4 * course, broken.get("grid", DATA + grid))
    # the switch's three lists, held by `lda` instructions in the handler's code
    for at, ptr in zip((0x220ABC, 0x220AC8, 0x220AD4), extra[:3]):
        struct.pack_into("<2I", program, at - WINDOW, broken.get("lda", LDA_G0), ptr)
    billboard_array = 0x235000
    struct.pack_into("<64I", program, billboard_array - WINDOW, *extra[3:])
    table, slot_list = 0x234000, 0x234800
    struct.pack_into("<I", program, ACTOR_TABLES - WINDOW + 4 * course, table)
    struct.pack_into("<I", program, table - WINDOW + 4 * 3, slot_list)
    recs = [((-64.0, 11.5, -64.0), -5463, 0, STATIC_PROP, extra[3]),
            ((162.25, 39.5, -224.0), 0, 0, BILLBOARD, billboard_array),
            ((-100.0, 0.0, -90.0), 12, 2, SWITCH, extra[1]),
            ((48.0, 3.0, -42.0), 0, 7, OTHER, extra[4])]
    struct.pack_into("<I", program, slot_list - WINDOW, len(recs))
    for i, (pos, ang, ident, handler, obj) in enumerate(recs):
        struct.pack_into("<3fhH2I", program, slot_list - WINDOW + 4 + 24 * i, *pos, ang, ident, handler, obj)
    for at in LIBRARY_BOUND_AT:         # the draw routines' bound checks (`lda lo,r3` / `lda hi,r4`): library_bounds
        struct.pack_into("<4I", program, at - 4, 0x8C183000, OBJECT_LIBRARY[0], 0x8C203000, OBJECT_LIBRARY[1])
    return _rs(program, data)


def test_constants_scenery():
    assert GRID_TABLES == 0x02805298 and ACTOR_TABLES == 0x00233BFC and GRID_CELLS == 256 and OBC == 5000
    assert OBJECT_LIBRARY == (0x0283FBB4, 0x0284F4EC)


def test_cell_of_and_box():
    # 0x172a0: col = (round(x) + 1024) >> 7 & 15, row = (round(z) + 1024) >> 7 & 15, cell = row << 4 | col
    assert cell_of(319.68, 27.25) == 138                          # the Beginner camera at frame 2430: cell 138
    assert cell_of(-1024.0, -1024.0) == 0 and cell_of(1023.4, 1023.4) == 255
    assert cell_of(-896.4, -1024.0) == 1 and cell_of(-896.6, -1024.0) == 0        # cvtri rounds to nearest
    assert cell_box(138) == (256.0, 0.0, 384.0, 128.0)            # x0, z0, x1, z1
    with pytest.raises(CourseError):
        cell_box(256)


def test_check_rom_object_rejects_ram_and_out_of_range():
    assert check_rom_object(0x80094C) == 0x94C
    assert check_rom_object(0xBFFFFF) == 0x3FFFFF
    for oba, why in ((0x01000004, "RAM 1"), (0x01800004, "RAM 1"), (0x00000004, "RAM 0"), (0x0000010A, "RAM 0"),
                     (0x00C00000, "0x400000"), (0x00FFFFFF, "0x400000"), (0x02800000, "bits 25"), (-1, "u32"),
                     (1 << 32, "u32")):
        with pytest.raises(CourseError, match=why):
            check_rom_object(oba)


def test_placement_round_trip_synthetic():
    rs = _scenery_rs()
    places = scenery(rs, _entry(1), allow_unmodelled=True)
    grid = [p for p in places if p.actor is None]
    assert len(grid) == GRID_CELLS + 1                            # cell 5 holds two objects
    assert [p.cell for p in grid[:7]] == [0, 1, 2, 3, 4, 5, 5]
    p = grid[3]
    assert (p.oba, p.obc, p.tpa, p.tha, p.raw, p.budget) == (0x801060, OBC, 0x50018, 12, (0,), 0)
    assert p.in_world and p.entry == LIB + 0x40 + 3 * 20
    for p in places:
        assert encode_placement(p) == rs.data[p.entry:p.entry + 16]
    acts = actors(rs, _entry(1))
    assert [(a.slot, a.handler, a.ident) for a in acts] == [(3, STATIC_PROP, 0), (3, BILLBOARD, 0), (3, SWITCH, 2),
                                                             (3, OTHER, 7)]
    assert acts[0].position == (-64.0, 11.5, -64.0) and acts[0].angle == -5463 and acts[0].addr == 0x234804
    for a in acts:
        assert encode_actor(a) == rs.read_i960(a.addr, 24)
    by = {}
    for p in places:
        if p.actor is not None:
            by.setdefault(p.actor.handler, []).append(p)
    assert set(by) == {STATIC_PROP, BILLBOARD, SWITCH}           # OTHER: handler not modelled, no placement
    assert [p.in_world for p in by[STATIC_PROP]] == [False]       # drawn under V * T(position) * R(angle)
    assert len(by[BILLBOARD]) == 64 and all(p.in_world for p in by[BILLBOARD])
    assert [p.oba for p in by[SWITCH]] == [0x900000, 0x900100, 0x900200] and all(p.in_world for p in by[SWITCH])
    assert all(p.cell == cell_of(p.actor.position[0], p.actor.position[2]) for p in by[BILLBOARD])
    assert by[STATIC_PROP][0].cell == cell_of(-64.0, -64.0)


def test_scenery_structures_round_trip_synthetic():
    """The structures around the entries (Task 9): the cell grid, object lists with their terminators, the actor table,
    an actor list with its count, the billboard array and the switch instructions encode back to their bytes."""
    from dtools.arcade.course import (actor_table, billboard_lists, cell_grid, encode_actor_list, encode_actor_table,
                                      encode_billboard_lists, encode_cell_grid, encode_object_list,
                                      encode_switch_lists, object_list, switch_lists)
    rs = _scenery_rs()
    grid, cells = cell_grid(rs, _entry(1))
    assert grid == DATA + 0x900000 and len(cells) == GRID_CELLS
    assert encode_cell_grid(cells) == rs.read_i960(grid, 4 * GRID_CELLS)
    for c in (0, 5, 255):
        places = object_list(rs, cells[c])
        assert len(places) == (2 if c == 5 else 1) and all(p.cell == -1 for p in places)
        blob = encode_object_list(places)
        assert len(blob) == 16 * len(places) + 4 and blob == rs.read_i960(cells[c], len(blob))
    table, slots = actor_table(rs, _entry(1))
    assert table == 0x234000 and slots[3] == 0x234800 and sum(1 for x in slots if x) == 1
    assert encode_actor_table(slots) == rs.read_i960(table, 4 * GRID_CELLS)
    acts = actors(rs, _entry(1))
    assert encode_actor_list(acts) == rs.read_i960(slots[3], 4 + 24 * len(acts))
    bb = billboard_lists(rs, 0x235000)
    assert encode_billboard_lists(bb) == rs.read_i960(0x235000, 4 * 64)
    sw = switch_lists(rs)
    assert {a: rs.read_i960(a, 8) for a in (0x220ABC, 0x220AC8, 0x220AD4)} == encode_switch_lists(sw)
    with pytest.raises(CourseError, match="0x220abc"):
        switch_lists(_scenery_rs(lda=0x8C883000))


def test_scenery_refuses_unmodelled_actors_unless_allowed():
    rs = _scenery_rs()
    with pytest.raises(NotImplementedError, match=r"1 actor.*0x2201bc.*Scenery placement / Not found / not modelled"):
        scenery(rs, _entry(1))
    places = scenery(rs, _entry(1), allow_unmodelled=True)
    assert len(places) == GRID_CELLS + 1 + 1 + 64 + 3              # grid, static prop, billboard, switch
    skipped = unmodelled_actors(rs, _entry(1))
    assert [(a.handler, a.ident, a.addr) for a in skipped] == [(OTHER, 7, 0x234804 + 3 * 24)]
    assert skipped == [a for a in actors(rs, _entry(1)) if a.handler == OTHER]


@pytest.mark.parametrize("broken, match", [
    ({"grid": 0x00233C0C}, r"0x280529c.*0x233c0c"),             # course 1's grid pointer, into the program window
    ({"cell": DATA + 0x100}, r"cell 7 .*0x290001c.*0x2000100"),  # a cell list outside the object library
    ({"ram_object": 0x0000004}, r"RAM 0"),                       # a ROM object is expected
    ({"lda": 0x8C883000}, r"0x220abc"),                          # the switch handler is not the expected code
])
def test_scenery_refuses_unexpected_tables(broken, match):
    with pytest.raises(CourseError, match=match):
        scenery(_scenery_rs(**broken), _entry(1), allow_unmodelled=True)


def test_encode_placement_refuses_what_does_not_fit():
    p = scenery(_scenery_rs(), _entry(1), allow_unmodelled=True)[0]
    good = p.__dict__
    for field, value, match in (("obc", 4999, "obc"), ("oba", 0x4, "RAM 0"), ("oba", 0x1000004, "RAM 1"),
                                ("tpa", 1 << 32, "tpa"), ("tha", -1, "tha"), ("raw", (1, 2), "raw"),
                                ("raw", (1 << 32,), "raw")):
        with pytest.raises(CourseError, match=match):
            encode_placement(Placement(**{**good, field: value}))


def test_encode_actor_refuses_what_does_not_fit():
    a = actors(_scenery_rs(), _entry(1))[0]
    for field, value, match in (("angle", 1 << 15, "angle"), ("ident", -1, "ident"), ("handler", 1 << 32, "handler"),
                                ("position", (1e40, 0.0, 0.0), "position"), ("position", (0.0, 1.0), "position")):
        with pytest.raises(CourseError, match=match):
            encode_actor(Actor(**{**a.__dict__, field: value}))


# The 103 objects Phase 1 saw drawn under the view matrix on Beginner (out/spike/arcade/objects.csv, under_view
# "always"): 22 tiles and 42 one-record stubs from the cell grid, two one-polygon objects of a switch actor and 37 of
# the 64 view-angle billboard variants.
PHASE1_VIEW_OBJECTS = frozenset({
    0x80094C, 0x80096D, 0x80098E, 0x8009AF, 0x800B1A, 0x800B3B, 0x800B5C, 0x800B7D, 0x800B9E, 0x800BBF, 0x800CC7,
    0x800CE8, 0x800D09, 0x800D2A, 0x800D4B, 0x801648, 0x802C6F, 0x80458E, 0x804855, 0x8048FD, 0x804912, 0x804927,
    0x804A5E, 0x805AE5, 0x806F36, 0x807333, 0x808004, 0x80926B, 0x809280, 0x809313, 0x809328, 0x80933D, 0x80AA7C,
    0x80B685, 0x80BBA4, 0x80C02D, 0x80CA24, 0x80E389, 0x80E39E, 0x80E431, 0x80E446, 0x80E45B, 0x80E9CA, 0x80F80D,
    0x810B96, 0x81160F, 0x81231C, 0x8125A7, 0x8125BC, 0x812664, 0x812679, 0x81268E, 0x8126A3, 0x8126B8, 0x8126CD,
    0x8126E2, 0x8126F7, 0x81270C, 0x8127B4, 0x8127C9, 0x8127DE, 0x8127F3, 0x812808, 0x81281D, 0x8D9E75, 0x8D9EB4,
    0x92538B, 0x9256D5, 0x925A1F, 0x925F0E, 0x9260B3, 0x926258, 0x9263FD, 0x926747, 0x926A91, 0x926C36, 0x927125,
    0x9277B9, 0x92795E, 0x927CA8, 0x927FF2, 0x92833C, 0x9284E1, 0x92882B, 0x9289D0, 0x928D1A, 0x929209, 0x9293AE,
    0x929553, 0x929A42, 0x929BE7, 0x929F31, 0x92A0D6, 0x92A27B, 0x92A5C5, 0x92A76A, 0x92A90F, 0x92AAB4, 0x92AC59,
    0x92ADFE, 0x92B2ED, 0x92B637, 0x92B981,
})


def test_beginner_scenery_covers_phase1_objects(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    places = scenery(rs, course_table(rs)[0], allow_unmodelled=True)
    seen: dict[int, set] = {}
    for p in places:
        seen.setdefault(p.oba, set()).add((p.obc, p.tpa, p.tha))
    assert PHASE1_VIEW_OBJECTS <= set(seen), sorted(hex(o) for o in PHASE1_VIEW_OBJECTS - set(seen))
    for oba in PHASE1_VIEW_OBJECTS:                               # one obc/tpa/tha per object, as in Phase 1
        assert len(seen[oba]) == 1 and next(iter(seen[oba]))[0] == 5000, hex(oba)
    # spot values from objects.csv: a tile, a stub, a one-polygon object, a billboard variant
    assert seen[0x800D4B] == {(5000, 0x50338, 0x19C)} and seen[0x80094C] == {(5000, 0x50240, 0x120)}
    assert seen[0x8D9EB4] == {(5000, 0x19DED2, 0x124914)} and seen[0x92538B] == {(5000, 0x1C2134, 0x1352CC)}
    assert all(p.in_world for p in places if p.oba in PHASE1_VIEW_OBJECTS)
    grid = [p for p in places if p.actor is None]
    assert [p.cell for p in grid] == list(range(GRID_CELLS))      # one object per cell on the real set
    assert sum(1 for p in grid if p.budget) == 22                 # the 22 world tiles; the other cells hold stubs
    assert len(PHASE1_VIEW_OBJECTS & {p.oba for p in grid}) == 64
    assert {p.cell for p in grid if p.oba == 0x808004} == {122}   # drawn while the camera is in cells 122-140


def test_scenery_real_all_courses(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    tiles, unmodelled = [], []
    for e in course_table(rs):
        with pytest.raises(NotImplementedError, match="Scenery placement / Not found / not modelled"):
            scenery(rs, e)                                        # every course has actors of unmodelled handlers
        places = scenery(rs, e, allow_unmodelled=True)
        skipped = unmodelled_actors(rs, e)
        assert all(a.handler not in (STATIC_PROP, SWITCH, BILLBOARD) for a in skipped)
        modelled = {p.actor for p in places if p.actor is not None}
        assert not modelled & set(skipped)
        assert len(modelled) + len(skipped) == len(actors(rs, e)), e.name
        unmodelled.append(len(skipped))
        grid = [p for p in places if p.actor is None]
        assert [p.cell for p in grid] == list(range(GRID_CELLS)), e.name
        tiles.append(sum(1 for p in grid if p.budget))
        for p in places:
            assert encode_placement(p) == rs.data[p.entry:p.entry + 16], (e.name, hex(p.oba))
        for a in actors(rs, e):
            assert encode_actor(a) == rs.read_i960(a.addr, 24), (e.name, hex(a.addr))
    assert tiles == [22, 95, 58]                                  # Beginner, Expert, Advanced
    assert unmodelled == [31, 63, 32]                             # actors scenery() skips


# --- road line and AI lines ----------------------------------------------------------------------------------------
import math

from dtools.arcade.course import (AI_LINE_COUNT, AI_LINE_WRAP, AI_POINT_SIZE, LANES, ROAD_POINT_SIZE, AiPoint, Lane,
                                  RoadPoint, ai_lines, ai_path, centreline, encode_ai_line, encode_ai_point,
                                  encode_line_table, encode_road_line, encode_road_point, road_line)
from dtools.arcade import polyrom

ROAD_AT, LINES_AT = 0x100000, 0x200000          # data-ROM offsets of the synthetic record table and line table


def _f32(*v: float) -> tuple[float, ...]:
    """Values rounded to single precision, so that they survive a decode."""
    return struct.unpack(f"<{len(v)}f", struct.pack(f"<{len(v)}f", *v))


def _road_points(n: int) -> list[RoadPoint]:
    """n + 1 cross-sections around a circle of radius 100, lanes 1-6 three units apart, the rest zero."""
    out = []
    for i in range(n + 1):
        a = 2 * math.pi * i / (n + 1)
        lanes = []
        for k in range(LANES):
            if 1 <= k <= 6:
                r = 100 + 3 * (k - 3.5)
                lanes.append(Lane(_f32(r * math.cos(a), 0.5 * k, r * math.sin(a)), -3 + k, -0x1556 + i, 7 - k, 0))
            else:
                lanes.append(Lane((0.0, 0.0, 0.0), 0, 0, 0, 0))
        out.append(RoadPoint(1, 6, 0x1111 * (i % 3 - 1), tuple(lanes)))
    return out


def _ai_line(n: int, k: int) -> list[AiPoint]:
    return [AiPoint(_f32(100.0 * math.cos(i / (n + 1) * 6.28) + k, 1.0, 100.0 * math.sin(i / (n + 1) * 6.28)),
                    1.25 + k / 8, -2, 0x2AAB - 3 * i, 4, -12 + k, 0x1F6 + 16 * k, 2 * i + k) for i in range(n + 1)]


def _lines_rs(n: int = 9, road=None, lines=None, entry_patch=None):
    """A synthetic set whose course 1 has a record table at ROAD_AT and a line table at LINES_AT."""
    data = bytearray(16 << 20)
    road_blob = encode_road_line(road if road is not None else _road_points(n))
    data[ROAD_AT:ROAD_AT + len(road_blob)] = road_blob
    lines = lines if lines is not None else [_ai_line(n, k) for k in range(AI_LINE_COUNT)]
    blob = encode_line_table(LINES_AT, lines)
    data[LINES_AT:LINES_AT + len(blob)] = blob
    for at, b in (entry_patch or {}).items():
        data[at:at + len(b)] = b
    entry = CourseEntry(1, 0, 0, 0, 0, LINES_AT, ROAD_AT, (0, 0))
    return _rs(bytearray(256 * 1024), data), entry


def test_road_line_round_trip_synthetic():
    rs, entry = _lines_rs()
    pts = road_line(rs, entry)
    assert pts == _road_points(9)
    assert struct.unpack_from("<2I", rs.data, ROAD_AT) == (9, LANES)        # n, then the lanes per record
    for i, p in enumerate(pts):
        at = ROAD_AT + 8 + ROAD_POINT_SIZE * i
        assert encode_road_point(p) == rs.data[at:at + ROAD_POINT_SIZE]
    assert ROAD_POINT_SIZE == 164 and LANES == 8
    p = pts[0]
    assert p.used_lanes == p.lanes[1:7]
    assert p.heading == -0x1556 and p.curve == -0x1111
    cx, cy, cz = centreline(pts)[0]                     # midpoint of lanes 1 and 6
    assert (round(cx, 4), round(cy, 4), round(cz, 4)) == (100.0, 1.75, 0.0)
    assert round(p.width, 4) == round(math.hypot(15, 2.5), 4)   # lane 1 to lane 6, 3D (y climbs 0.5 per lane)
    assert len(centreline(pts)) == 10


def test_ai_lines_round_trip_synthetic():
    rs, entry = _lines_rs()
    got = ai_lines(rs, entry)
    assert got == [_ai_line(9, k) for k in range(AI_LINE_COUNT)]
    first = LINES_AT + 8 * AI_LINE_COUNT                 # the lists follow the eight (pointer, count) pairs
    stride = (10 + AI_LINE_WRAP) * AI_POINT_SIZE          # n + 1 points, then the first AI_LINE_WRAP again
    for k in range(AI_LINE_COUNT):
        ptr, count = struct.unpack_from("<2I", rs.data, LINES_AT + 8 * k)
        assert (ptr, count) == (DATA + first + k * stride, 10)
        at = first + k * stride
        assert encode_ai_point(got[k][0]) == rs.data[at:at + AI_POINT_SIZE]
        assert encode_ai_line(got[k]) == rs.data[at:at + stride]
        assert rs.data[at + 10 * AI_POINT_SIZE:at + stride] == rs.data[at:at + AI_LINE_WRAP * AI_POINT_SIZE]
    assert AI_POINT_SIZE == 28 and AI_LINE_WRAP == 8 and AI_LINE_COUNT == 8
    assert ai_path(got[0])[1] == got[0][1].position


@pytest.mark.parametrize("patch, match", [
    ({ROAD_AT + 4: struct.pack("<I", 6)}, r"0x2100004.*6 lanes"),                    # lanes per record not 8
    ({ROAD_AT: struct.pack("<I", 0x100000)}, r"0x2100000"),                           # n runs past data ROM
    ({LINES_AT + 8 * 3 + 4: struct.pack("<I", 7)}, r"0x220001c.*7.*10"),              # a line count other than n + 1
    ({LINES_AT + 8 * 2: struct.pack("<I", 0x238634)}, r"0x2200010.*0x238634"),       # a list outside data ROM
    ({LINES_AT + 0x40 + 10 * 28: b"\x01"}, r"line 0.*wrap"),                           # the repeat differs from the head
    ({ROAD_AT + 8 + 164 * 2: bytes([5, 2])}, r"record 2.*lanes 5-2"),                 # lane_lo > lane_hi
])
def test_lines_refuse_unexpected_tables(patch, match):
    rs, entry = _lines_rs(entry_patch=patch)
    with pytest.raises(CourseError, match=match):
        road_line(rs, entry)
        ai_lines(rs, entry)


def test_line_pointers_outside_data_rom_raise():
    rs, _ = _lines_rs()
    with pytest.raises(CourseError, match="record table"):
        road_line(rs, CourseEntry(1, 0, 0, 0, 0, LINES_AT, 0xFFFFFC, (0, 0)))
    with pytest.raises(CourseError, match="line table"):
        ai_lines(rs, CourseEntry(1, 0, 0, 0, 0, 0xFFFFF0, ROAD_AT, (0, 0)))


def test_encode_road_point_refuses_what_does_not_fit():
    good = _road_points(1)[0]
    lane = good.lanes[1]
    bad = [
        RoadPoint(3, 2, 0, good.lanes),                                   # lane_lo > lane_hi
        RoadPoint(1, 8, 0, good.lanes),                                   # lane index past 7
        RoadPoint(1, 6, 0x8000, good.lanes),                              # curve past s16
        RoadPoint(1, 6, 0, good.lanes[:7]),                               # seven lanes
        RoadPoint(1, 6, 0, (Lane((1e40, 0.0, 0.0), 0, 0, 0, 0),) + good.lanes[1:]),     # past f32
        RoadPoint(1, 6, 0, (Lane((math.nan, 0.0, 0.0), 0, 0, 0, 0),) + good.lanes[1:]),  # not finite
        RoadPoint(1, 6, 0, (Lane((0.0, 0.0), 0, 0, 0, 0),) + good.lanes[1:]),            # two coordinates
        RoadPoint(1, 6, 0, (lane._replace(heading=-0x8001),) + good.lanes[1:]),          # heading past s16
        RoadPoint(1, 6, 0, (lane._replace(pad=0x10000),) + good.lanes[1:]),              # pad past u16
    ]
    for p in bad:
        with pytest.raises(CourseError):
            encode_road_point(p)
    with pytest.raises(CourseError, match="at least one"):
        encode_road_line([])


def test_encode_ai_point_refuses_what_does_not_fit():
    good = _ai_line(1, 0)[0]
    for change in ({"position": (0.0, 1e39, 0.0)}, {"position": (0.0, math.inf, 0.0)}, {"speed": 1e39},
                   {"speed": math.nan}, {"pitch": 0x8000}, {"heading": -0x8001}, {"roll": 1 << 16},
                   {"steer": 0x8000}, {"lane": 0x8000}, {"lane": -0x8001}, {"step": 0x10000}, {"step": -1}):
        with pytest.raises(CourseError):
            encode_ai_point(good._replace(**change))
    with pytest.raises(CourseError, match="8 lines"):
        encode_line_table(LINES_AT, [_ai_line(3, 0)] * 7)
    with pytest.raises(CourseError, match="same number of points"):
        encode_line_table(LINES_AT, [_ai_line(3, 0)] * 7 + [_ai_line(4, 0)])
    with pytest.raises(CourseError, match="at least 8"):
        encode_ai_line(_ai_line(5, 0)[:7])


def _tile_vertices(rs: RomSet, entry: CourseEntry) -> list[tuple[float, float]]:
    """XZ of every point of the course's world tiles (the grid objects with a budget), decoded from the polygon ROM."""
    out = []
    for p in scenery(rs, entry, allow_unmodelled=True):
        if p.actor is None and p.budget:
            obj = polyrom.decode_object(rs.polygons, check_rom_object(p.oba), mode=1)
            for poly in polyrom.polygons(obj):
                out += [(q[0], q[2]) for q in poly.points]
    return out


def _closed_length(path) -> float:
    return sum(math.dist((path[i][0], path[i][2]), (path[i - 1][0], path[i - 1][2])) for i in range(len(path)))


def test_beginner_road_line_on_world_tiles(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    e = course_table(rs)[0]
    cl = centreline(road_line(rs, e))
    assert len(cl) == 196
    verts = _tile_vertices(rs, e)
    far = []
    for x, _, z in cl:
        d = min(math.dist((x, z), v) for v in verts)
        if d > 15:
            far.append((round(x, 1), round(z, 1), round(d, 1)))
    assert not far
    assert 1000 < _closed_length(cl) < 6000


def test_road_and_ai_lines_real_all_courses(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    counts, laps, ai_laps = [], [], []
    for e in course_table(rs):
        road = road_line(rs, e)
        n_plus_1 = len(road)
        blob = rs.data[e.record_table:e.record_table + 8 + ROAD_POINT_SIZE * n_plus_1]
        assert encode_road_line(road) == blob, e.name
        lines = ai_lines(rs, e)
        assert len(lines) == AI_LINE_COUNT and all(len(l) == n_plus_1 for l in lines), e.name
        size = 8 * AI_LINE_COUNT + AI_LINE_COUNT * (n_plus_1 + AI_LINE_WRAP) * AI_POINT_SIZE
        assert encode_line_table(e.line_table, lines) == rs.data[e.line_table:e.line_table + size], e.name
        steps = struct.unpack_from("<I", rs.data, e.step_table + 4)[0]
        assert all(0 <= p.step < steps for l in lines for p in l), e.name    # each point's step-table index
        assert all(0 <= p.lane <= 0x700 for l in lines for p in l), e.name    # lateral lane coordinate, 8.8
        cl = centreline(road)
        for l in lines:                                                       # every AI line runs along the road
            assert max(math.dist((a.position[0], a.position[2]), (c[0], c[2])) for a, c in zip(l, cl)) < 25, e.name
        counts.append(n_plus_1)
        laps.append(round(_closed_length(cl)))
        ai_laps.append(round(_closed_length(ai_path(lines[0]))))
    assert counts == [196, 991, 487]                                          # Beginner, Expert, Advanced
    assert laps == [1555, 7670, 3783]
    assert ai_laps == [1563, 7548, 3798]


# --- checkpoints, time limit, start grid, pit -----------------------------------------------------------------------
from dtools.arcade.course import (CAR_COUNTS, CHECKPOINT_TABLES, GRID_BASES, GRID_LANES, LAP_COUNTS, MAX_CHECKPOINTS,
                                  PIT_BOX_FLAGS, PIT_LANE_FLAGS, ROLLING_STARTS, SPAWN_RECORD_SIZE, SPAWN_RECORDS,
                                  SPAWN_TABLES, STEP_PIT_LANE, STEP_ROAD, TICKS_PER_SECOND, TIME_TABLES, Checkpoint, GridSlot, Pit,
                                  Step, car_count, checkpoints, encode_checkpoint_table, encode_grid_base, encode_pit,
                                  encode_spawn_record, encode_step_table, encode_time_table, grid_base, lane_position,
                                  laps, pit, start_grid, step_table, time_table)

STEP_AT, CP_AT, TT_AT, PTR_AT = 0x300000, 0x310000, 0x320000, 0x330000   # synthetic blobs (data-ROM offsets)
WINDOW = 0x200000                                                        # program window i960 address - program offset


def _put(buf: bytearray, off: int, fmt: str, *v) -> None:
    struct.pack_into(fmt, buf, off, *v)


def _steps() -> list[Step]:
    """Ten road polygons (segments 0-9, flag road), then a pit lane on segments 8, 9, 0, 1 with its box on 9."""
    return ([Step(i, STEP_ROAD) for i in range(10)]
            + [Step(8, PIT_LANE_FLAGS), Step(9, PIT_BOX_FLAGS), Step(0, PIT_LANE_FLAGS), Step(1, PIT_LANE_FLAGS)])


def _slot(i: int) -> GridSlot:
    return GridSlot(i, tuple(OBJECT_LIBRARY[0] + 0x400 + 0x40 * i + 20 * k for k in range(8)), i % 8, 2 * ((i + 1) // 2),
                    _f32(280.0 + i)[0], i % 3, 0, 0, (0.0, 0.0, 0.0))


def _race_rs(rolling: bool = False, patch=None, base: int = 7):
    """A synthetic set for course 1: the road line of _lines_rs (n = 9), a step table, a checkpoint table with two
    checkpoints (segments 3 and 6), 2 laps and a 7-word time table at mode 0 / difficulty 1, 5 cars from spawn table
    1 with grid base 7, and the lane pair (0x180, 0x580)."""
    rs, _ = _lines_rs()
    program, data = bytearray(rs.program), bytearray(rs.data)
    blob = encode_step_table(9, _steps())
    data[STEP_AT:STEP_AT + len(blob)] = blob
    cps = [Checkpoint(0, 3, 0, False, (0.0, 0.0, 0.0), ()), Checkpoint(1, 6, 0x0101, False, (0.0, 0.0, 0.0), ())]
    blob = encode_checkpoint_table(cps)
    data[CP_AT:CP_AT + len(blob)] = blob
    _put(data, CHECKPOINT_TABLES - DATA + 4, "<I", DATA + CP_AT)
    words = [3200, 1201, 841, 1081, 721, 1081, 721]
    data[TT_AT:TT_AT + 28] = encode_time_table(words)
    # TIME_TABLES[mode 0] -> per-id array at PTR_AT; [id 1] -> per-difficulty array at PTR_AT + 0x40; [1] -> TT_AT
    _put(data, TIME_TABLES - DATA, "<I", DATA + PTR_AT)
    _put(data, PTR_AT + 4, "<I", DATA + PTR_AT + 0x40)
    _put(data, PTR_AT + 0x40 + 4, "<I", DATA + TT_AT)
    _put(data, LAP_COUNTS - DATA, "<I", DATA + PTR_AT + 0x80)            # LAP_COUNTS[mode 0] -> u32[4] by id
    _put(data, PTR_AT + 0x80 + 4, "<I", 2)
    _put(data, CAR_COUNTS - DATA, "<I", DATA + PTR_AT + 0xc0)            # CAR_COUNTS[0] -> u32[4] by id
    _put(data, PTR_AT + 0xc0 + 4, "<I", 5)
    _put(data, ROLLING_STARTS - DATA + 2, "<h", 0 if rolling else -1)
    for i in range(SPAWN_RECORDS):
        rec = encode_spawn_record(_slot(i))
        at = SPAWN_TABLES[1] - DATA + SPAWN_RECORD_SIZE * i
        data[at:at + SPAWN_RECORD_SIZE] = rec
    _put(program, GRID_BASES - WINDOW + 2, "<h", base)
    _put(program, GRID_LANES - WINDOW, "<2H", 0x180, 0x580)
    for at, b in (patch or {}).items():
        data[at:at + len(b)] = b
    entry = CourseEntry(1, 0, STEP_AT, 0, 0, LINES_AT, ROAD_AT, (0, 0))
    return _rs(program, data), entry


def test_race_constants():
    assert CHECKPOINT_TABLES == 0x02823C10 and TIME_TABLES == 0x02805228 and LAP_COUNTS == 0x02805234
    assert CAR_COUNTS == 0x02805240 and ROLLING_STARTS == 0x028050E8 and SPAWN_TABLES == (0x0281E3D4, 0x0281EC14)
    assert SPAWN_RECORD_SIZE == 48 and SPAWN_RECORDS == 44 and GRID_BASES == 0x00230E48 and GRID_LANES == 0x00231B30
    assert TICKS_PER_SECOND == 64 and MAX_CHECKPOINTS == 30
    assert SPAWN_TABLES[0] + SPAWN_RECORD_SIZE * SPAWN_RECORDS == SPAWN_TABLES[1]


def test_checkpoints_round_trip_synthetic():
    rs, entry = _race_rs()
    assert laps(rs, entry) == 2
    words = time_table(rs, entry)
    assert words == [3200, 1201, 841, 1081, 721, 1081, 721]              # 1 + laps * (checkpoints + lap line)
    cps = checkpoints(rs, entry)
    cl = centreline(road_line(rs, entry))
    assert [(c.index, c.segment, c.raw, c.lap_line) for c in cps] == [(0, 3, 0, False), (1, 6, 0x0101, False),
                                                                      (2, 0, 0, True)]
    assert [c.position for c in cps] == [cl[3], cl[6], cl[0]]
    # pass k = lap * (count + 1) + index + 1 indexes the time table
    assert [c.extensions for c in cps] == [(1201, 721), (841, 1081), (1081, 721)]
    blob = encode_checkpoint_table(cps)                                   # the lap line is implicit, not stored
    assert blob == rs.data[CP_AT:CP_AT + 12] and struct.unpack_from("<I", blob)[0] == 2
    assert encode_time_table(words) == rs.data[TT_AT:TT_AT + 28]


def test_checkpoints_refuse_unexpected_tables():
    with pytest.raises(CourseError, match=r"0x2823c14.*0x238634"):      # per-id pointer outside data ROM
        checkpoints(*_race_rs(patch={CHECKPOINT_TABLES - DATA + 4: struct.pack("<I", 0x238634)}))
    with pytest.raises(CourseError, match=r"0x2310000.*31 checkpoints"):  # count past the u32 bit set
        checkpoints(*_race_rs(patch={CP_AT: struct.pack("<I", 31)}))
    with pytest.raises(CourseError, match=r"segment 12.*n 9"):            # a checkpoint past the road line
        checkpoints(*_race_rs(patch={CP_AT + 4: struct.pack("<H", 12)}))
    with pytest.raises(CourseError, match=r"laps 0"):
        laps(*_race_rs(patch={PTR_AT + 0x80 + 4: struct.pack("<I", 0)}))
    rs, entry = _race_rs()
    with pytest.raises(CourseError, match="mode 3"):
        time_table(rs, entry, mode=3)
    with pytest.raises(CourseError, match="difficulty 4"):
        time_table(rs, entry, difficulty=4)


def test_checkpoint_encoders_refuse_what_does_not_fit():
    cp = Checkpoint(0, 3, 0, False, (0.0, 0.0, 0.0), ())
    for bad in (cp._replace(segment=0x10000), cp._replace(segment=-1), cp._replace(raw=0x10000)):
        with pytest.raises(CourseError):
            encode_checkpoint_table([bad])
    with pytest.raises(CourseError, match="31 checkpoints"):
        encode_checkpoint_table([cp] * 31)
    for bad in ([], [3200, -1], [3200, 1 << 32], [3200, 1.5]):
        with pytest.raises(CourseError):
            encode_time_table(bad)


def test_start_grid_synthetic_standing():
    rs, entry = _race_rs()
    assert car_count(rs, entry) == 5 and grid_base(rs, entry) == 7
    road = road_line(rs, entry)
    slots = start_grid(rs, entry)
    assert [s.car for s in slots] == [0, 1, 2, 3, 4]
    assert [s.segment for s in slots] == [7, 0, 0, 2, 2]                 # (start + base) mod n: 0, 2, 2, 4, 4 + 7
    assert [s.lane for s in slots] == [0x580, 0x180, 0x580, 0x180, 0x580]  # even cars 0x231b30[1], odd [0]
    for s in slots:
        assert s.position == lane_position(road[s.segment], s.lane)
        assert encode_spawn_record(s) == rs.read_i960(SPAWN_TABLES[1] + SPAWN_RECORD_SIZE * s.car, SPAWN_RECORD_SIZE)
        assert s[:6] == _slot(s.car)[:6]
    assert encode_grid_base(7) == struct.pack("<h", 7)


def test_start_grid_synthetic_rolling():
    rs, entry = _race_rs(rolling=True)
    # rolling start: the player and every car at segment >= n >> 1 (4) take 0x231b30[car & 1], the others keep
    # 0x231b30[(car & 1) ^ 1]; with base 7 the cars are at segments 7, 0, 0, 2, 2: only the player swaps
    assert [s.lane for s in start_grid(rs, entry)] == [0x180, 0x180, 0x580, 0x180, 0x580]
    rs, entry = _race_rs(rolling=True, base=3)                            # segments 3, 5, 5, 7, 7: all swap
    assert [s.lane for s in start_grid(rs, entry)] == [0x180, 0x580, 0x180, 0x580, 0x180]
    rs, entry = _race_rs(rolling=False, base=3)
    assert [s.lane for s in start_grid(rs, entry)] == [0x580, 0x180, 0x580, 0x180, 0x580]


def test_lane_position_blends_two_lanes():
    p = _road_points(1)[0]
    a, b = p.lanes[1].position, p.lanes[2].position
    w = _f32(0x80 / 255)[0]                                   # data 0x02871c80[frac]: frac / 255 as f32
    got = lane_position(p, 0x180)
    assert all(abs(g - (x + (y - x) * w)) < 1e-9 for g, x, y in zip(got, a, b))
    assert lane_position(p, 0x000) == p.lanes[1].position     # lanes clamped to lane_lo..lane_hi (1-6)
    assert lane_position(p, 0x700) == p.lanes[6].position
    assert lane_position(p, 0x300) == p.lanes[3].position


def test_start_grid_refuses_unexpected_tables():
    with pytest.raises(CourseError, match=r"car count 45"):
        start_grid(*_race_rs(patch={PTR_AT + 0xc0 + 4: struct.pack("<I", 45)}))
    with pytest.raises(CourseError, match=r"line 9"):
        start_grid(*_race_rs(patch={SPAWN_TABLES[1] - DATA + SPAWN_RECORD_SIZE * 2 + 0x20: struct.pack("<I", 9)}))
    rs, entry = _race_rs()
    with pytest.raises(CourseError, match="spawn table 2"):
        start_grid(rs, entry, table=2)


def test_spawn_encoders_refuse_what_does_not_fit():
    good = _slot(3)
    for change in ({"objects": good.objects[:7]}, {"objects": (1 << 32,) + good.objects[1:]}, {"line": 8},
                   {"line": -1}, {"start": 0x10000}, {"speed": math.nan}, {"speed": 1e39}, {"kind": 0x100}):
        with pytest.raises(CourseError):
            encode_spawn_record(good._replace(**change))
    for bad in (-1, 0x8000, 1.0):
        with pytest.raises(CourseError):
            encode_grid_base(bad)


def test_pit_round_trip_synthetic():
    rs, entry = _race_rs()
    steps = step_table(rs, entry)
    assert steps == _steps()
    assert encode_step_table(9, steps) == rs.data[STEP_AT:STEP_AT + 8 + 4 * len(steps)]
    p = pit(rs, entry)
    assert p == Pit(entry_segment=8, exit_segment=1, box_segment=9, lane_polygons=(10, 12, 13), box_polygon=11)
    assert encode_pit(9, steps, p) == steps
    bare = [s if s.flags == STEP_ROAD else Step(s.segment, 0) for s in steps]
    assert encode_pit(9, bare, p) == steps


def test_no_pit_and_pit_refusals():
    rs, entry = _race_rs(patch={STEP_AT + 4: struct.pack("<I", 10)})     # only the ten road polygons
    assert pit(rs, entry) is None
    steps = _steps()
    two_runs = steps[:10] + [Step(8, PIT_LANE_FLAGS), Step(9, PIT_BOX_FLAGS), Step(4, PIT_LANE_FLAGS)]
    blob = encode_step_table(9, two_runs)
    rs, entry = _race_rs(patch={STEP_AT: blob})
    with pytest.raises(CourseError, match="one run"):
        pit(rs, entry)
    p = Pit(8, 1, 9, (10, 12, 13), 11)
    with pytest.raises(CourseError, match="polygon 14"):
        encode_pit(9, steps, p._replace(lane_polygons=(10, 12, 14)))
    with pytest.raises(CourseError, match="road polygon 2"):
        encode_pit(9, steps, p._replace(lane_polygons=(10, 12, 2)))
    with pytest.raises(CourseError, match="box"):
        encode_pit(9, steps, p._replace(box_segment=8))
    with pytest.raises(CourseError, match="entry"):
        encode_pit(9, steps, p._replace(entry_segment=9))
    with pytest.raises(CourseError):
        encode_step_table(9, [Step(10, 1)])                               # segment past n
    with pytest.raises(CourseError):
        encode_step_table(9, [Step(1, 0x10000)])                          # flags past u16


def test_encode_pit_writes_only_the_pit_bits():
    """Phase 4 residuals: other flag bits (Advanced's 0x100) survive, a lone STEP_PIT_LANE off the pit is refused,
    and a step segment past n is refused."""
    steps = _steps()
    p = Pit(8, 1, 9, (10, 12, 13), 11)
    keep = list(steps)
    keep[3] = Step(3, STEP_ROAD | 0x100)
    keep[10] = Step(8, PIT_LANE_FLAGS | 0x100)
    assert encode_pit(9, keep, p) == keep                                  # 0x100 kept on a road and a pit polygon
    bare = list(keep)
    for i in (10, 11, 12, 13):
        bare[i] = Step(bare[i].segment, 0x100 if i == 10 else 0)
    assert encode_pit(9, bare, p) == keep
    for flags in (STEP_ROAD | STEP_PIT_LANE, STEP_PIT_LANE):
        lone = list(steps)
        lone[4] = Step(4, flags)                                           # a lone 0x40 off the pit
        with pytest.raises(CourseError, match="already pit"):
            encode_pit(9, lone, p)
    with pytest.raises(CourseError, match="segment 10"):
        encode_pit(9, steps[:13] + [Step(10, 0)], p)                       # a step segment past n


def test_pit_checks_the_step_table_against_the_road_line():
    blob = encode_step_table(12, _steps()[:10] + [Step(8, PIT_LANE_FLAGS), Step(9, PIT_BOX_FLAGS)])
    rs, entry = _race_rs(patch={STEP_AT: blob})                            # step n = 12, road line n = 9
    with pytest.raises(CourseError, match="road line"):
        pit(rs, entry)


# real set ------------------------------------------------------------------------------------------------------------

def test_checkpoints_real_all_courses(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    segs, firsts, initial, lap_counts = [], [], [], []
    for e in course_table(rs):
        cps = checkpoints(rs, e)
        assert 1 <= len(cps) <= 8, e.name
        assert cps[-1].lap_line and cps[-1].segment == 0 and not any(c.lap_line for c in cps[:-1])
        cl = centreline(road_line(rs, e))
        for c in cps:                                                     # every checkpoint on the centreline
            assert min(math.dist((c.position[0], c.position[2]), (q[0], q[2])) for q in cl) < 15, e.name
        ptr = struct.unpack_from("<I", rs.data, CHECKPOINT_TABLES - DATA + 4 * e.course)[0] - DATA
        size = 4 + 4 * (len(cps) - 1)
        assert encode_checkpoint_table(cps) == rs.data[ptr:ptr + size], e.name
        words = time_table(rs, e)
        n_laps = laps(rs, e)
        assert len(words) == 1 + n_laps * len(cps)
        t = struct.unpack_from("<I", rs.data, TIME_TABLES - DATA)[0]
        t = struct.unpack_from("<I", rs.data, t - DATA + 4 * e.course)[0]
        t = struct.unpack_from("<I", rs.data, t - DATA + 4 * 1)[0] - DATA
        assert encode_time_table(words) == rs.data[t:t + 4 * len(words)], e.name
        segs.append([c.segment for c in cps])
        firsts.append(cps[0].extensions[0])
        initial.append(words[0])
        lap_counts.append(n_laps)
    assert segs == [[0], [150, 257, 404, 539, 730, 0], [142, 275, 0]]      # Beginner, Expert, Advanced
    assert lap_counts == [8, 2, 4]
    assert initial == [3200, 3840, 3840]                                  # the timer's first value in the runs
    assert firsts == [1201, 560, 1083]                                    # the first jump of the timer in the runs
    assert checkpoints(rs, course_table(rs)[2])[1].extensions[0] == 723    # Advanced, second jump


def test_start_grid_real_all_courses(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    counts, players = [], []
    for e in course_table(rs):
        road = road_line(rs, e)
        n = len(road) - 1
        slots = start_grid(rs, e)
        counts.append(len(slots))
        assert {s.lane for s in slots} == {0x180, 0x580}, e.name           # two wide: lanes 1.5 and 5.5
        ahead = [(s.segment - slots[0].segment) % n for s in slots]
        assert ahead == sorted(ahead) and ahead[0] == 0, e.name            # the player at the back, the field ahead
        assert all(s.line < AI_LINE_COUNT for s in slots)
        for a, b in zip(slots[1::2], slots[2::2]):                         # rows of two side by side
            if a.start == b.start:
                assert a.lane != b.lane, (e.name, a.car)
                assert math.dist(a.position, b.position) == pytest.approx(12.0, abs=0.5)
        for s in slots:
            c = road[s.segment].centre
            assert math.dist((s.position[0], s.position[2]), (c[0], c[2])) < 15, (e.name, s.car)
            at = SPAWN_TABLES[1] + SPAWN_RECORD_SIZE * s.car
            assert encode_spawn_record(s) == rs.read_i960(at, SPAWN_RECORD_SIZE), (e.name, s.car)
        assert encode_grid_base(grid_base(rs, e)) == rs.read_i960(GRID_BASES + 2 * e.course, 2)
        players.append((slots[0].segment, n - slots[0].segment))
    assert counts == [40, 30, 20]
    # standing starts (Expert, Advanced): the player 6 / 4 segments behind the lap line; Beginner rolls from 100
    assert players == [(100, 95), (984, 6), (482, 4)]


def test_pit_real_all_courses(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    got = []
    for e in course_table(rs):
        steps = step_table(rs, e)
        n = struct.unpack_from("<I", rs.data, e.step_table)[0]
        assert encode_step_table(n, steps) == rs.data[e.step_table:e.step_table + 8 + 4 * len(steps)], e.name
        p = pit(rs, e)
        assert p is not None, e.name
        assert encode_pit(n, steps, p) == steps
        got.append((p.entry_segment, p.exit_segment, p.box_segment, len(p.lane_polygons)))
    assert got == [(175, 75, 5, 96), (117, 189, 136, 72), (465, 50, 479, 96)]


# --- review fix 1: inverses for laps, car count, rolling start; table pointers; spawn object bounds ----------------
from dtools.arcade.course import (MAX_LAPS, encode_car_count, encode_laps, encode_rolling_start, encode_table_pointer,
                                  race_value_addresses, rolling_start)


def test_race_values_round_trip_synthetic():
    for rolling in (False, True):
        rs, entry = _race_rs(rolling=rolling)
        at = race_value_addresses(rs, entry)
        assert rolling_start(rs, entry) is rolling
        assert encode_rolling_start(rolling) == rs.read_i960(at["rolling_start"], 2)
        assert encode_laps(laps(rs, entry)) == rs.read_i960(at["laps"], 4)
        assert encode_car_count(car_count(rs, entry)) == rs.read_i960(at["car_count"], 4)
        assert encode_grid_base(grid_base(rs, entry)) == rs.read_i960(at["grid_base"], 2)
        assert at["checkpoint_table"] == DATA + CP_AT and at["time_table"] == DATA + TT_AT
        assert encode_table_pointer(at["checkpoint_table"]) == rs.read_i960(at["checkpoint_pointer"], 4)
        assert encode_table_pointer(at["time_table"]) == rs.read_i960(at["time_pointer"], 4)
    assert at["checkpoint_pointer"] == CHECKPOINT_TABLES + 4 and at["rolling_start"] == ROLLING_STARTS + 2
    assert at["grid_base"] == GRID_BASES + 2
    assert MAX_LAPS == 127


def test_race_value_encoders_refuse_what_does_not_fit():
    for bad in (0, MAX_LAPS + 1, -1, 2.0):
        with pytest.raises(CourseError):
            encode_laps(bad)
    for bad in (0, SPAWN_RECORDS + 1, -1, "40"):
        with pytest.raises(CourseError):
            encode_car_count(bad)
    for bad in (0, 1, -1, None):
        with pytest.raises(CourseError):
            encode_rolling_start(bad)
    for bad in (0x00500000, 0x00238634, DATA - 4, DATA + DATA_SIZE_TEST, 1 << 32):
        with pytest.raises(CourseError):
            encode_table_pointer(bad)
    assert encode_rolling_start(True) == struct.pack("<h", 0) and encode_rolling_start(False) == struct.pack("<h", -1)


DATA_SIZE_TEST = 0x2000000      # the i960's data window (tablebase.DATA_WINDOW): a custom set's image


def test_race_value_decoders_refuse_unexpected_values():
    with pytest.raises(CourseError, match=r"laps 128"):
        laps(*_race_rs(patch={PTR_AT + 0x80 + 4: struct.pack("<I", 128)}))
    with pytest.raises(CourseError, match=r"car count 0"):
        car_count(*_race_rs(patch={PTR_AT + 0xc0 + 4: struct.pack("<I", 0)}))
    with pytest.raises(CourseError, match=r"0x28050ea.*2"):       # neither 0 (rolling) nor -1 (standing)
        rolling_start(*_race_rs(patch={ROLLING_STARTS - DATA + 2: struct.pack("<h", 2)}))


def test_spawn_objects_bound_to_the_object_library():
    lo, hi = OBJECT_LIBRARY
    good = _slot(3)
    for bad in (0x00500000, DATA, lo - 4, hi, 0x0284F4F0):
        with pytest.raises(CourseError, match="object library"):
            encode_spawn_record(good._replace(objects=(bad,) + good.objects[1:]))
    assert encode_spawn_record(good._replace(objects=(lo,) + good.objects[1:]))
    rec = SPAWN_TABLES[1] - DATA + SPAWN_RECORD_SIZE * 2 + 4 * 5
    with pytest.raises(CourseError, match=r"spawn record 2.*0x500000.*object library"):
        start_grid(*_race_rs(patch={rec: struct.pack("<I", 0x00500000)}))


def test_race_values_real_all_courses(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    rolling, values = [], []
    lo, hi = OBJECT_LIBRARY
    for e in course_table(rs):
        at = race_value_addresses(rs, e)
        assert encode_laps(laps(rs, e)) == rs.read_i960(at["laps"], 4), e.name
        assert encode_car_count(car_count(rs, e)) == rs.read_i960(at["car_count"], 4), e.name
        assert encode_rolling_start(rolling_start(rs, e)) == rs.read_i960(at["rolling_start"], 2), e.name
        assert encode_grid_base(grid_base(rs, e)) == rs.read_i960(at["grid_base"], 2), e.name
        for k in ("checkpoint", "time"):
            assert encode_table_pointer(at[f"{k}_table"]) == rs.read_i960(at[f"{k}_pointer"], 4), e.name
        n_cp = len(checkpoints(rs, e)) - 1
        assert encode_checkpoint_table(checkpoints(rs, e)) == rs.read_i960(at["checkpoint_table"], 4 + 4 * n_cp)
        words = time_table(rs, e)
        assert encode_time_table(words) == rs.read_i960(at["time_table"], 4 * len(words))
        for table in (0, 1):
            for s in start_grid(rs, e, table=table):
                assert all(lo <= o < hi for o in s.objects), (e.name, table, s.car)
        rolling.append(rolling_start(rs, e))
        values.append((laps(rs, e), car_count(rs, e)))
    assert rolling == [True, False, False]
    assert values == [(8, 40), (2, 30), (4, 20)]


# --- select screen, backdrop, rankings (Task 8b) -----------------------------------------------------------------

from dtools.arcade.course import (PANEL_COLS, PANEL_ROWS, RANKING_LAST, RANKING_TABLES, SELECT_MAP_OBJECTS,
                                  SELECT_PANEL_OF_COURSE, SELECT_PANELS, SELECT_RECORD_SIZE, SELECT_SCREEN,
                                  SELECT_TABLES, SELECT_VARIANTS, SRAM, WHEEL_HI_AT, WHEEL_LO_AT, MapObject,
                                  SelectPanel, WheelZones, backdrop, decode_select_panel, encode_map_object,
                                  encode_panel_of_course, encode_ranking_pointers, encode_select_panel,
                                  encode_wheel_zones, ranking_pointers, rankings_layout, select_assets, wheel_zones)
from dtools.arcade.tiles import (BACKDROP_SEGMENTS, CHAR_RAM, PaletteUpload, TileMap, encode_backdrop_segment,
                                 encode_char_block, encode_char_list, encode_palette_list, encode_tile_map,
                                 encode_upload_set, tile_map)
from tests.test_tiles import _lists_rs

import numpy as np

WHEEL_WORDS = {WHEEL_LO_AT: 0x3B00E014, WHEEL_HI_AT: 0x592EDE02}   # cmpibge 0,r3,.. / shlo 2,27,r5 (the real code)


def test_backdrop_synthetic():
    rs, *_ = _lists_rs()
    e = CourseEntry(1, 0x5000, 0, 0, 0, 0, 0, (0, 0))               # upload_list -> the set with 8 segments
    b = backdrop(rs, e)
    assert b.course == 1 and b.uploads.addr == DATA + 0x5000
    assert len(b.segments) == BACKDROP_SEGMENTS and b.segments[0].rows == 2 and b.segments[0].cols == 32
    assert b.top_tile == 0x3000 and b.bottom_tile == 0x3020           # segment 0, column 0, first and last row
    assert b.panorama().shape == (2, 32 * BACKDROP_SEGMENTS)
    assert encode_backdrop_segment(b.segments[3]) == rs.read_i960(b.uploads.segments[3], 12 + 2 * 64)
    with pytest.raises(CourseError, match="no backdrop"):
        backdrop(rs, CourseEntry(1, 0x5100, 0, 0, 0, 0, 0, (0, 0)))  # an upload set without segments


def test_wheel_zones_round_trip_and_refusals():
    program = bytearray(256 * 1024)
    for at, w in WHEEL_WORDS.items():
        struct.pack_into("<I", program, at - WINDOW, w)
    rs = _rs(program, bytearray(16 << 20))
    z = wheel_zones(rs)
    assert z == WheelZones(0, 108)
    assert [z.panel(w) for w in (-50, 0, 1, 108, 109)] == [0, 0, 1, 1, 2]
    enc = encode_wheel_zones(z)
    assert enc == {at: struct.pack("<I", w) for at, w in WHEEL_WORDS.items()}
    assert wheel_zones(_rs(_patched(program, encode_wheel_zones(WheelZones(5, 31 << 3))), rs.data)) == \
        WheelZones(5, 248)
    with pytest.raises(CourseError, match="hi"):
        encode_wheel_zones(WheelZones(0, 33))                           # 33 is not n << s with n, s in 0..31
    with pytest.raises(CourseError, match="lo"):
        encode_wheel_zones(WheelZones(32, 108))
    with pytest.raises(CourseError, match="lo"):
        encode_wheel_zones(WheelZones(100, 96))
    # `shlo` keeps 32 bits and the compare after it is signed: the encoder refuses a hi past 0x7fffffff, the decoder
    # wraps what such a word gives
    with pytest.raises(CourseError, match="0x80000000"):
        encode_wheel_zones(WheelZones(0, 16 << 31))
    with pytest.raises(CourseError, match="0x80000000"):
        encode_wheel_zones(WheelZones(0, 1 << 31))
    top = WheelZones(0, 31 << 26)                                       # 0x7c000000, the largest that fits
    assert wheel_zones(_rs(_patched(program, encode_wheel_zones(top)), rs.data)) == top
    raw = (WHEEL_WORDS[WHEEL_HI_AT] & ~((0x1F << 14) | 0x1F))
    for n, s, hi in ((16, 31, 0), (1, 31, -(1 << 31)), (3, 30, -(1 << 30))):
        patched = _patched(program, {WHEEL_HI_AT: struct.pack("<I", raw | n << 14 | s)})
        assert wheel_zones(_rs(patched, rs.data)).hi == hi
    struct.pack_into("<I", program, WHEEL_LO_AT - WINDOW, 0x3A00E014)  # another opcode
    with pytest.raises(CourseError, match=hex(WHEEL_LO_AT)):
        wheel_zones(_rs(program, rs.data))


def _patched(buf: bytearray, patches: dict[int, bytes]) -> bytearray:
    out = bytearray(buf)
    for at, b in patches.items():
        out[at - WINDOW:at - WINDOW + len(b)] = b
    return out


def test_select_panel_record_round_trip_and_refusals():
    p = SelectPanel(1, 2, (0xF7, 0x18C), (22, 9), tuple(DATA + 0x100 * k for k in range(6)))
    b = encode_select_panel(p)
    assert len(b) == SELECT_RECORD_SIZE
    assert b[8] == 2                                                   # 0x226c74 reads the id as the record's byte 8
    assert decode_select_panel(b, 1) == p
    for bad, match in [(p._replace(course=256), "course"), (p._replace(tile=(47, 9)), "tile"),
                       (p._replace(tile=(0, 55)), "tile"), (p._replace(cursor=(0x1000, 0)), "cursor"),
                       (p._replace(maps=p.maps[:5]), "6"), (p._replace(maps=(0x238634,) * 6), "outside data ROM")]:
        with pytest.raises(CourseError, match=match):
            encode_select_panel(bad)
    o = MapObject(2, (5.0, -1.0, 9.0), 0x028448F8, (0x028448F8, 0, 0, 0))
    ob = encode_map_object(o)
    assert len(ob) == 32 and struct.unpack_from("<3fI", ob) == (5.0, -1.0, 9.0, 0x028448F8)
    with pytest.raises(CourseError, match="object library"):
        encode_map_object(o._replace(objects=DATA))


# a synthetic Circuit Select: three tables (one per mode) of 4 records, two upload sets, panel maps

SEL_CHARS = 0x3480                                                     # first panel char (char RAM 0x10e9000)


def _panel_map(course: int, mode: int, variant: int, blink: int) -> TileMap:
    """A 10 x 18 panel: frame chars shared by all, name chars by (course, variant, blink), label chars by (course,
    mode)."""
    tiles = []
    for r in range(PANEL_ROWS):
        for c in range(PANEL_COLS):
            if r in (0, PANEL_ROWS - 1) or c in (0, PANEL_COLS - 1):
                ch = SEL_CHARS
            elif r < 6:
                ch = SEL_CHARS + 1 + course * 6 + variant * 2 + blink
            else:
                ch = SEL_CHARS + 0x20 + course * 3 + mode
            tiles.append(ch)
    return TileMap(-0x8000, True, PANEL_ROWS, PANEL_COLS, tuple(tiles))   # tiles = entry | 0x8000 (priority)


def _select_rs():
    program, data = bytearray(256 * 1024), bytearray(16 << 20)
    for at, w in WHEEL_WORDS.items():
        struct.pack_into("<I", program, at - WINDOW, w)
    ids = (0, 2, 1, 3)                                                   # panel -> course id, the real order
    struct.pack_into("<4I", program, SELECT_PANEL_OF_COURSE - WINDOW, 0, 2, 1, 3)
    off = 0x10000
    for mode, table in enumerate(SELECT_TABLES):
        for i in range(4):
            maps = []
            for variant in range(SELECT_VARIANTS):
                for blink in (0, 1):
                    m = _panel_map(ids[i] if i < 3 else 2, mode, variant, blink)
                    b = encode_tile_map(m)
                    data[off:off + len(b)] = b
                    maps.append(DATA + off)
                    off += 0x200
            p = SelectPanel(i, ids[i], (0x5F + 152 * min(i, 2), 0x18C), (3 + 19 * min(i, 2), 9), tuple(maps))
            program[table - WINDOW + SELECT_RECORD_SIZE * i:table - WINDOW + SELECT_RECORD_SIZE * (i + 1)] = \
                encode_select_panel(p)
    for i in range(SELECT_PANELS):
        o = MapObject(i, (-5.0 + 5 * i, -1.0, 9.0), OBJECT_LIBRARY[0] + 0x10 * i, (0, 0, 0, 0))
        program[SELECT_MAP_OBJECTS - WINDOW + 32 * i:SELECT_MAP_OBJECTS - WINDOW + 32 * (i + 1)] = encode_map_object(o)
    # chars: the boot set uploads 0x3480.. with zeros, the select set overrides them (the last upload wins)
    blk0 = encode_char_block(b"\0" * 32 * 0x40)
    blk1 = encode_char_block(bytes(k & 0xFF for k in range(32 * 0x40)))
    data[0x200000:0x200000 + len(blk0)] = blk0
    data[0x210000:0x210000 + len(blk1)] = blk1
    for k, (blk, at) in enumerate(((0x200000, 0x220000), (0x210000, 0x230000))):
        cl = encode_char_list([(DATA + blk, 0x010E9000)])
        data[at:at + len(cl)] = cl
        pl = encode_palette_list([PaletteUpload(0x01800D20, (k, k))])
        data[at + 0x100:at + 0x100 + len(pl)] = pl
        us = encode_upload_set(DATA + at, DATA + at + 0x100)
        data[at + 0x200:at + 0x200 + len(us)] = us
    empty = 0x240000                                                    # an upload set with two empty lists
    data[empty + 0x10:empty + 0x18] = encode_upload_set(DATA + empty, DATA + empty)
    for slot, at in zip(SELECT_SCREEN, (0x220200, 0x230200)):
        struct.pack_into("<I", data, slot - DATA, DATA + at)
    for slot in (0x02600000, 0x02700000):                                # the boot sets
        struct.pack_into("<I", data, slot - DATA, DATA + empty + 0x10)
    return _rs(program, data)


def test_select_assets_synthetic():
    rs = _select_rs()
    a = select_assets(rs)
    assert len(a.panels) == len(SELECT_TABLES) and all(len(t) == SELECT_PANELS for t in a.panels)
    assert [p.course for p in a.panels[0]] == [0, 2, 1]                  # panel order: Beginner, Advanced, Expert
    assert a.wheel == WheelZones(0, 108) and a.panel_of_course == (0, 2, 1)
    assert [c.course for c in a.courses] == [0, 1, 2] and [c.panel for c in a.courses] == [0, 2, 1]
    beg, exp, adv = a.courses
    assert adv.tile == (22, 9) and exp.tile == (41, 9)
    assert len(beg.maps) == 3 and all(len(v) == SELECT_VARIANTS for v in beg.maps)
    assert beg.maps[1][1][0].tiles()[0] == SEL_CHARS | 0x8000
    # name: rows 1-5 inside the frame, by variant and blink; label: rows 6-8 inside the frame, by mode
    assert beg.name == tuple(SEL_CHARS + 1 + k for k in range(6))
    assert adv.name == tuple(SEL_CHARS + 1 + 2 * 6 + k for k in range(6))
    assert exp.label == tuple(SEL_CHARS + 0x20 + 3 + m for m in range(3))
    assert beg.map_object.objects == OBJECT_LIBRARY[0] and adv.map_object.position == (0.0, -1.0, 9.0)
    # where each name char's 32 bytes come from on the Circuit Select screen: the select set overrides the boot set
    assert a.char_source[SEL_CHARS + 1] == DATA + 0x210000 + 4 + 32
    assert len(a.uploads) == 4 and a.uploads[-1].addr == DATA + 0x230200
    for mode, t in zip(range(3), SELECT_TABLES):
        for p in a.panels[mode]:
            at = t + SELECT_RECORD_SIZE * p.panel
            assert encode_select_panel(p) == rs.read_i960(at, SELECT_RECORD_SIZE)
            for addr in p.maps:
                m = tile_map(rs, addr)
                assert encode_tile_map(m) == rs.read_i960(addr, 12 + 2 * m.rows * m.cols)


def test_panel_of_course_round_trip_and_refusals():
    rs = _select_rs()
    a = select_assets(rs)
    assert encode_panel_of_course(a.panel_of_course, a.panels[0]) == rs.read_i960(SELECT_PANEL_OF_COURSE, 12)
    for bad, match in [((0, 2), "2 entries"), ((0, 2, 3), "course 2.* 3"), ((0, -1, 1), "course 1.* -1"),
                       ((0, 1, 2), "course 1.* 1, but panel 1 shows 2"), ((0, 2, 1.0), "course 2")]:
        with pytest.raises(CourseError, match=match):
            encode_panel_of_course(bad, a.panels[0])


def test_select_assets_refuses_unexpected_tables():
    rs = _select_rs()
    program = bytearray(rs.program)
    program[SELECT_TABLES[0] - WINDOW + 8] = 7                          # panel 0 -> course id 7
    with pytest.raises(CourseError, match=r"0x238634.*7"):
        select_assets(_rs(program, rs.data))
    program = bytearray(rs.program)
    struct.pack_into("<I", program, SELECT_PANEL_OF_COURSE - WINDOW + 4, 0)   # id 1 -> panel 0: not the inverse
    with pytest.raises(CourseError, match=hex(SELECT_PANEL_OF_COURSE + 4)):
        select_assets(_rs(program, rs.data))
    data = bytearray(rs.data)                                            # a panel map that is not 10 x 18
    addr = struct.unpack_from("<I", rs.program, SELECT_TABLES[1] - WINDOW + 20)[0]
    struct.pack_into("<I", data, addr - DATA + 4, 9)
    with pytest.raises(CourseError, match=r"10 x 18"):
        select_assets(_rs(rs.program, data))
    data = bytearray(rs.data)                                            # the select set uploads no char at all
    struct.pack_into("<I", data, 0x230000, 0)
    struct.pack_into("<I", data, 0x220000, 0)
    with pytest.raises(CourseError, match=r"char 0x3480"):
        select_assets(_rs(rs.program, data))


def test_rankings_layout():
    lay = rankings_layout()
    assert lay.base == SRAM[0] == 0x01D00000 and lay.size == 0x4000
    assert lay.tables == (0x01D00360, 0x01D005A0, 0x01D007E0)           # id order: Beginner, Expert, Advanced
    assert lay.entries == 24 and lay.entry_size == 24
    assert lay.entry(2, 0) == 0x01D007E0 and lay.entry(0, 23) == 0x01D00360 + 23 * 24
    assert lay.field(1, 2, "time") == 0x01D005A0 + 2 * 24 + 4
    assert [f[0] for f in lay.fields] == ["rank", "time", "view", "flags", "shown_time", "name"]
    assert lay.last_entry == (0x01D00A20, 0x01D00A24, 0x01D00A2C)
    assert lay.free[0] >= max(hi for _, _, hi in lay.used) and lay.free[1] == SRAM[1]
    # the three tables are inside SRAM and do not overlap anything else the game keeps there
    spans = sorted([(t, t + lay.entries * lay.entry_size) for t in lay.tables] + [(lo, hi) for _, lo, hi in lay.used])
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0
    with pytest.raises(CourseError, match="rank"):
        lay.entry(0, 24)
    with pytest.raises(CourseError, match="field"):
        lay.field(0, 0, "score")


def test_ranking_pointers_round_trip_and_refusals():
    program = bytearray(256 * 1024)
    struct.pack_into("<4I", program, RANKING_TABLES - WINDOW, 0x01D00360, 0x01D005A0, 0x01D007E0, 0x01D007E0)
    rs = _rs(program, bytearray(16 << 20))
    assert ranking_pointers(rs) == (0x01D00360, 0x01D005A0, 0x01D007E0)
    assert encode_ranking_pointers((0x01D00360, 0x01D005A0, 0x01D007E0)) == bytes(program[RANKING_TABLES - WINDOW:
                                                                                         RANKING_TABLES - WINDOW + 12])
    for bad in [(0x01D00360,), (0x01D00361, 0x01D005A0, 0x01D007E0), (0x01D03F00, 0x01D005A0, 0x01D007E0),
                (0x01CFFF00, 0x01D005A0, 0x01D007E0)]:
        with pytest.raises(CourseError):
            encode_ranking_pointers(bad)
    struct.pack_into("<I", program, RANKING_TABLES - WINDOW + 4, 0x02000000)
    with pytest.raises(CourseError, match=hex(RANKING_TABLES + 4)):
        ranking_pointers(_rs(program, rs.data))


def test_select_assets_real(romset_zip):
    """Integration: three panels decoded, in panel order Beginner, Advanced, Expert."""
    rs = RomSet.from_zip(romset_zip)
    a = select_assets(rs)
    assert [p.course for p in a.panels[0]] == [0, 2, 1]
    assert [[p.tile for p in t] for t in a.panels] == [[(3, 9), (22, 9), (41, 9)]] * 3
    assert a.wheel == WheelZones(0, 108) and a.panel_of_course == (0, 2, 1)
    assert [c.panel for c in a.courses] == [0, 2, 1]
    for c in a.courses:
        for mode in range(3):
            for variant in range(SELECT_VARIANTS):
                for m in c.maps[mode][variant]:
                    assert (m.rows, m.cols, m.base) == (PANEL_ROWS, PANEL_COLS, -0x8000)
        assert c.name and c.label and all(ch in a.char_source for ch in c.name + c.label)
        assert OBJECT_LIBRARY[0] <= c.map_object.objects < OBJECT_LIBRARY[1]
    # the Japanese name chars (variant 1, as MAME shows them) of the three panels are all different
    names = [set(c.name) for c in a.courses]
    assert not (names[0] & names[1]) and not (names[1] & names[2]) and not (names[0] & names[2])
    # golden: every record, map, list and char block encodes back to the ROM bytes
    for mode, t in enumerate(SELECT_TABLES):
        for p in a.panels[mode]:
            assert encode_select_panel(p) == rs.read_i960(t + SELECT_RECORD_SIZE * p.panel, SELECT_RECORD_SIZE)
            for addr in p.maps:
                m = tile_map(rs, addr)
                assert encode_tile_map(m) == rs.read_i960(addr, 12 + 2 * m.rows * m.cols)
    for u in a.uploads:
        assert encode_upload_set(u.char_list, u.palette_list) == rs.read_i960(u.addr, 8)
        assert encode_char_list([(c.src, c.dst) for c in u.chars]) == rs.read_i960(u.char_list, 8 * len(u.chars) + 4)
        for c in u.chars:
            assert encode_char_block(c.data(rs)) == rs.read_i960(c.src, 4 + 32 * c.count)
        pl = encode_palette_list(u.palette)
        assert pl == rs.read_i960(u.palette_list, len(pl))
    for i, c in enumerate(a.courses):
        o = c.map_object
        assert encode_map_object(o) == rs.read_i960(SELECT_MAP_OBJECTS + 32 * o.panel, 32)
    assert {at: rs.read_i960(at, 4) for at in (WHEEL_LO_AT, WHEEL_HI_AT)} == encode_wheel_zones(a.wheel)
    assert encode_panel_of_course(a.panel_of_course, a.panels[0]) == rs.read_i960(SELECT_PANEL_OF_COURSE, 12)


def test_backdrop_real(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    rows = []
    for e in course_table(rs):
        b = backdrop(rs, e)
        assert b.uploads.addr == DATA + e.upload_list and len(b.segments) == BACKDROP_SEGMENTS
        assert all(s.cols == 32 and s.flags == 0x10000 for s in b.segments)
        assert len({s.rows for s in b.segments}) == 1
        rows.append(b.segments[0].rows)
        assert all(0x010E0000 <= c.dst < CHAR_RAM[1] for c in b.uploads.chars)     # chars 0x3000+
        assert all(p.dst >= 0x01800C00 for p in b.uploads.palette)                   # palettes 0x60+
        for addr, s in zip(b.uploads.segments, b.segments):
            assert encode_backdrop_segment(s) == rs.read_i960(addr, 12 + 2 * s.rows * s.cols)
        assert encode_upload_set(b.uploads.char_list, b.uploads.palette_list, b.uploads.segments) == \
            rs.read_i960(b.uploads.addr, 40)
        # every tile names a char the course's own list uploads
        uploaded = {c.first_char + k for c in b.uploads.chars for k in range(c.count)}
        assert {t & 0x3FFF for s in b.segments for t in s.tiles} <= uploaded
    assert rows == [49, 54, 43]


def test_rankings_real(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    lay = rankings_layout()
    assert ranking_pointers(rs) == lay.tables
    assert struct.unpack("<3I", rs.read_i960(RANKING_LAST, 12)) == lay.last_entry
    # the boot-time defaults (0x2270d0): (defaults, table) pairs, in id order
    assert struct.unpack("<6I", rs.read_i960(0x002390CC, 24))[1::2] == lay.tables
    assert rs.read_i960(lay.signature_source, lay.signature[1]).startswith(b"S-RAM CHECK OK")


# --- tables not decoded: stubs naming their discovery-log entry, and the panel order --------------------------------

def test_tables_not_decoded_raise_naming_their_log_entry():
    """Each table M1 did not decode has a decoder that raises NotImplementedError naming where its state is
    written down (the discovery log's entry, or the plan that owns it)."""
    import dtools.arcade.course as C
    e = CourseEntry(0, 0, 0, 0, 0, 0, 0, (0, 0))
    cases = [
        (lambda: C.time_corrections(None, e), r"time corrections.*Checkpoints, start grid, pit / Not found"),
    ]
    for call, match in cases:
        with pytest.raises(NotImplementedError, match=match):
            call()
    # the ground (grid and attribute block) is decoded since M1b, by tgp.decode_ground, the HUD map since M1c
    # (hud_map, below) and the per-segment flags since M3a (segment_flags): no stub is left for them
    assert not hasattr(C, "ground_grid") and not hasattr(C, "ground_attributes")


def test_panel_courses_is_the_select_order():
    """PANEL_COURSES maps the Circuit Select panel (autoplay.lua's DAYTONA_COURSE) to the course id, and is its own
    inverse: it equals the game's panel-of-course array (synthetic set: the real values)."""
    import dtools.arcade.course as C
    a = select_assets(_select_rs())
    assert [p.course for p in a.panels[0]] == list(C.PANEL_COURSES) == [0, 2, 1]
    assert a.panel_of_course == C.PANEL_COURSES
    assert [C.PANEL_COURSES[C.PANEL_COURSES[k]] for k in range(3)] == [0, 1, 2]


def test_panel_courses_on_the_real_set(romset_zip):
    import dtools.arcade.course as C
    a = select_assets(RomSet.from_zip(romset_zip))
    assert [p.course for p in a.panels[0]] == list(C.PANEL_COURSES) and a.panel_of_course == C.PANEL_COURSES


def test_constants_single_home():
    """DATA_BASE, DATA_SIZE and the program window are defined once, in tablebase; the other names are the same
    objects."""
    from dtools.arcade import course as C, romset, tablebase, texture
    assert (tablebase.DATA_BASE, tablebase.DATA_SIZE) == (0x02000000, 0x01000000)
    assert tablebase.PROGRAM_WINDOW == (0x00220000, 0x00240000)
    assert C._WINDOW is tablebase.PROGRAM_WINDOW is romset.I960_WINDOW
    assert texture.DATA_BASE is tablebase.DATA_BASE and C.DATA_BASE is tablebase.DATA_BASE
    assert romset.DATA_SIZE is tablebase.DATA_SIZE and C.DATA_SIZE is tablebase.DATA_SIZE


# --- the race HUD's course map (docs/spike/2026-11-mirror.md, "HUD map") ----------------------------------------------

HUD_REAL = {    # course id: outline list, its one entry's oba, the outline's records, angle, zooms
    0: (0x028437B4, 0x8D5DA4, 25, 0x4000, (7e-8, 5e-8, 3.5e-8, 0.0)),
    1: (0x028458D4, 0x94DD59, 145, -20480, (2.7e-8, 2.4e-8, 1.7e-8, 0.0)),
    2: (0x02843890, 0x8D61BF, 106, 0, (4e-8, 3e-8, 2e-8, 0.0)),
}


def test_hud_map_real(romset_zip):
    """The HUD map of each course, as the discovery found it: the outline list HUD_MAP_LISTS[id] with one entry, a
    polygon object at y 0 in world coordinates (its x and z extents hold the road line's), the place record's angle
    and pad, the four zooms."""
    import dtools.arcade.course as C
    rs = RomSet.from_zip(romset_zip)
    for e in course_table(rs):
        m = C.hud_map(rs, e)
        lst, oba, records, angle, zooms = HUD_REAL[e.course]
        assert m.course == e.course and m.outline_list == lst
        assert [p.oba for p in m.entries] == [oba] and set(m.objects) == {check_rom_object(oba)}
        obj = m.objects[check_rom_object(oba)]
        points = [obj.seed0, obj.seed1] + [q for r in obj.records for q in (r.p0, r.p1) if q is not None]
        assert len(obj.records) == records and all(y == 0.0 for _, y, _ in points)
        road = [p.centre for p in C.road_line(rs, e)]          # world coordinates: the outline bounds the road
        for k in (0, 2):
            assert min(q[k] for q in points) <= min(q[k] for q in road) <= max(q[k] for q in road) \
                <= max(q[k] for q in points)
        assert m.angle == angle and m.pad == 0
        assert all(abs(a - b) <= 1e-6 * abs(b) for a, b in zip(m.zooms, zooms))
        assert len(m.position) == 3 and 0 < m.position[2] < 1e-3          # in front of the HUD's camera


def test_hud_map_round_trip(romset_zip):
    """encode_hud_map(hud_map(rs, e)) re-emits the ROM bytes on all three courses: the list pointer, the list, the
    place record and the zooms (i960 addresses), and the outline objects re-encode to their polygon-ROM words."""
    import dtools.arcade.course as C
    from dtools.arcade import polyrom as P
    rs = RomSet.from_zip(romset_zip)
    for e in course_table(rs):
        m = C.hud_map(rs, e)
        out = C.encode_hud_map(m)
        assert [a for a, _ in out] == [C.HUD_MAP_LISTS + 4 * e.course, m.outline_list,
                                       C.HUD_MAP_PLACES + C.HUD_MAP_PLACE_SIZE * e.course,
                                       C.HUD_MAP_ZOOMS + 4 * C.HUD_MAP_ZOOM_LEVELS * e.course]
        for addr, blob in out:
            assert rs.read_i960(addr, len(blob)) == blob, hex(addr)
        for index, obj in m.objects.items():
            words = P.encode_object(obj)
            assert rs.polygons[4 * index:4 * (index + len(words))] == struct.pack(f"<{len(words)}I", *words)


def _hud(**kw):
    import dtools.arcade.course as C
    from dtools.arcade import polyrom as P
    obj = P.Object((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [P.Record(1, (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0))],
                   20)
    h = dict(course=0, outline_list=OBJECT_LIBRARY[0], entries=(Placement(0x800004, OBC, 0, 0, (0,), -1, 0, True),),
             objects={4: obj}, position=(1.0, 2.0, 3.0), angle=0x4000, pad=0, zooms=(7e-8, 5e-8, 3.5e-8, 0.0))
    h.update(kw)
    return C.HudMap(**h)


def test_encode_hud_map_synthetic():
    import dtools.arcade.course as C
    out = C.encode_hud_map(_hud(course=1))
    assert out == [(C.HUD_MAP_LISTS + 4, struct.pack("<I", OBJECT_LIBRARY[0])),
                   (OBJECT_LIBRARY[0], struct.pack("<4I", 0x800004, 0, 0, 0) + bytes(4)),
                   (C.HUD_MAP_PLACES + 16, struct.pack("<3fhH", 1.0, 2.0, 3.0, 0x4000, 0)),
                   (C.HUD_MAP_ZOOMS + 16, struct.pack("<4f", 7e-8, 5e-8, 3.5e-8, 0.0))]


@pytest.mark.parametrize("kw, match", [
    (dict(course=3), "course"),
    (dict(outline_list=0x02000000), "object library"),
    (dict(entries=()), "no entry"),
    (dict(objects={}), "object 0x4"),
    (dict(objects={4: None}), "polyrom.Object"),
    (dict(position=(1.0, 2.0)), "position"),
    (dict(position=(float("nan"), 0.0, 0.0)), "position"),
    (dict(angle=0x8000), "angle"),
    (dict(angle=True), "angle"),
    (dict(pad=-1), "pad"),
    (dict(zooms=(1.0, 2.0, 3.0)), "zooms"),
    (dict(zooms=None), "zooms"),
])
def test_encode_hud_map_refuses(kw, match):
    import dtools.arcade.course as C
    with pytest.raises(CourseError, match=match):
        C.encode_hud_map(_hud(**kw))
    with pytest.raises(CourseError):
        C.encode_hud_map("map")


def test_hud_map_refuses_a_list_outside_the_library():
    """The decoder checks the per-id list pointer against the object library (the draw 0x17acc skips such a list)."""
    import dtools.arcade.course as C
    rs = _synthetic()                       # HUD_MAP_LISTS[id] is 0 there
    with pytest.raises(CourseError, match="HUD map"):
        C.hud_map(rs, course_table(rs)[0])


# --- per-segment flags (Phase 6, Task 6) ------------------------------------------------------------------------------

FLAGS_AT = 0x300000                                # data-ROM offset of the synthetic flag table


def _flags_rs(flags, pointer=DATA + FLAGS_AT, n: int = 9):
    """_lines_rs's course 1 (n + 1 road points) with SEGMENT_FLAGS[1] = `pointer` and `flags` (u16) at FLAGS_AT."""
    import dtools.arcade.course as C
    blob = struct.pack(f"<{len(flags)}H", *flags)
    return _lines_rs(n, entry_patch={C.SEGMENT_FLAGS - DATA + 4: struct.pack("<I", pointer), FLAGS_AT: blob})


def test_segment_flags_synthetic():
    """segment_flags reads SEGMENT_FLAGS[id] and one u16 per road segment (n + 1) there; encode_segment_flags writes
    them back. CourseError for an entry that is not a CourseEntry, a pointer outside data ROM, a table running past
    it, values that are not u16 integers (bool refused), an empty or too long table."""
    import dtools.arcade.course as C
    flags = [0, 0x400, 0x500, 0x400, 0, 0, 0x100, 0, 0, 0xFFFF]
    rs, e = _flags_rs(flags)
    assert C.segment_flags(rs, e) == tuple(flags)
    assert C.encode_segment_flags(C.segment_flags(rs, e)) == rs.data[FLAGS_AT:FLAGS_AT + 20]
    assert C.encode_segment_flags(np.array(flags, dtype=np.uint16)) == rs.data[FLAGS_AT:FLAGS_AT + 20]
    for bad, match in ((None, "CourseEntry"), ("x", "CourseEntry")):
        with pytest.raises(CourseError, match=match):
            C.segment_flags(rs, bad)
    with pytest.raises(CourseError, match="RomSet"):
        C.segment_flags(None, e)
    with pytest.raises(CourseError, match=r"per-segment flags.*outside data ROM"):
        C.segment_flags(*_flags_rs(flags, pointer=0x00500000))
    with pytest.raises(CourseError, match=r"per-segment flags.*outside data ROM"):
        C.segment_flags(*_flags_rs(flags, pointer=DATA + (16 << 20) - 4))
    for bad in ([], [True], [-1], [0x10000], [1.0], None, "ab", [0] * 0x10000):
        with pytest.raises(CourseError):
            C.encode_segment_flags(bad)


def test_segment_flags_round_trip(romset_zip):
    """ids 0-2: segment_flags decodes one u16 per road segment (n + 1: 196 / 991 / 487) from SEGMENT_FLAGS[id]
    (0x028058cc / 0x02805a54 / 0x02806212, back to back) and encode_segment_flags gives the ROM's bytes back. The
    values (docs/spike/2026-11-convert.md, "Per-segment flags"): Beginner's and Advanced's are all 0; Expert's are
    SEGMENT_AI_FREE (0x400) on segments 190-240 and with it SEGMENT_TUNNEL (0x100) on 208-218, its tunnel (the
    zone table's range 209-217). The id-3 stub repeats Beginner's pointer."""
    import dtools.arcade.course as C
    rs = RomSet.from_zip(romset_zip)
    want = {0: (0x028058CC, 196), 1: (0x02805A54, 991), 2: (0x02806212, 487)}
    for e in course_table(rs):
        at, count = want[e.course]
        assert struct.unpack("<I", rs.read_i960(C.SEGMENT_FLAGS + 4 * e.course, 4))[0] == at
        flags = C.segment_flags(rs, e)
        assert len(flags) == count == len(C.road_line(rs, e)), e.name
        assert C.encode_segment_flags(flags) == rs.read_i960(at, 2 * count), e.name
        set_ = {s: f for s, f in enumerate(flags) if f}
        if e.course == 1:
            assert set_ == {s: 0x400 | (0x100 if 208 <= s <= 218 else 0) for s in range(190, 241)}
        else:
            assert set_ == {}, e.name
    assert C.SEGMENT_TUNNEL == 0x100 and C.SEGMENT_AI_FREE == 0x400
    assert struct.unpack("<I", rs.read_i960(C.SEGMENT_FLAGS + 12, 4))[0] == want[0][0]
    assert want[0][0] + 2 * 196 == want[1][0] and want[1][0] + 2 * 991 == want[2][0]


# ---- culling and camera scripts (Phase 7 Task 8: docs/spike/2026-11-convert.md, "Confirmed with scenery") ----

def _culling_rs(rs: RomSet, **broken) -> RomSet:
    """`rs` with one culling table broken: `pointer` (quadrant 1's table pointer), `mask` (a mask bit past the block),
    `nan` (a point offset), `order` (a repeated block index), `offsets` (one block offset)."""
    from dtools.arcade import course as C
    q1 = struct.unpack("<I", rs.read_i960(C.VIEW_TESTS + 4, 4))[0]
    patch = {"pointer": ("program", C.VIEW_TESTS + 4, struct.pack("<I", 0x01000000)),
             "mask": ("data", q1 - DATA + 0x5C + 8, struct.pack("<I", 1 << 25)),
             "nan": ("data", q1 - DATA + 0x0C, struct.pack("<f", float("nan"))),
             "order": ("program", C.VISIT_ORDER + 3, bytes([12])),
             "offsets": ("program", C.VISIT_OFFSETS + 24, bytes([33]))}
    (k,) = broken
    region, off, blob = patch[k]
    return rs.with_patch(region, off, blob)


def test_culling_real(romset_zip):
    """The view test's four quadrant tables, the visit orders and the block offsets (`0x172e8`, `0x1770c`) are
    geometry in cell units round the reference cell, shared by all courses: no cell index, no world extent, nothing
    sized for one course (Phase 7 Task 8, read on every race frame of Desert City). Each table: 12 points at whole
    cells (multiples of 128) from the reference cell's low corner, within -256..384, each opening block cells
    (5 x 5, bit i = (i // 5 - 2, i % 5 - 2)); the visit orders start at the reference cell (block index 12)."""
    from dtools.arcade import course as C
    rs = RomSet.from_zip(romset_zip)
    c = C.culling(rs)
    assert [struct.unpack("<I", rs.read_i960(C.VIEW_TESTS + 4 * q, 4))[0] for q in range(4)] == \
        [0x0286F7E0, 0x0286F880, 0x0286F920, 0x0286F9C0]
    assert len(c.view_tests) == 4 and all(len(t) == 12 for t in c.view_tests)
    for t in c.view_tests:
        for p in t:
            assert p.dx % 128 == 0 and p.dz % 128 == 0 and -256 <= p.dx <= 384 and -256 <= p.dz <= 384
            assert 0 < p.mask < 1 << C.BLOCK_CELLS
    assert c.view_tests[0][0] == C.ViewPoint(-128.0, 128.0, 0x739C00)      # quadrant 0's first point (frozen)
    assert c.order[0] == c.order_advanced[0] == 12
    assert c.offsets == tuple(16 * dr + dc for dr in range(-2, 3) for dc in range(-2, 3))


@pytest.mark.parametrize("broken, match", [("pointer", "view-test table 1.*outside data ROM"),
                                           ("mask", "point 0x5c.*25-cell mask"), ("nan", "point 0xc.*not a finite"),
                                           ("order", "not a permutation"), ("offsets", "not 16 \\* dr \\+ dc")])
def test_culling_refuses(romset_zip, broken, match):
    from dtools.arcade import course as C
    rs = _culling_rs(RomSet.from_zip(romset_zip), **{broken: True})
    with pytest.raises(CourseError, match=match):
        C.culling(rs)


def test_culling_refuses_not_a_romset():
    from dtools.arcade import course as C
    with pytest.raises(CourseError, match="is not a RomSet"):
        C.culling(b"daytona")


def test_camera_scripts_real(romset_zip):
    """The camera scripts 0x63d8 plays (Phase 7 Task 8): slot 0's (CAMERA_SCRIPTS[0], Beginner's, and its variants
    CAMERA_SCRIPTS_SLOT0) place every keyframe's position and target in car 0's frame (the player's) and loop on
    themselves after 3600 ticks, so a course converted into slot 0 is framed round the player's car wherever it is
    (the fly-by, frames 1328-1738 of the race and again every 3600). Expert's and Advanced's run on into a shared loop
    (0x2808360) whose first keyframes are absolute world positions (Expert's first keyframe too): a course converted
    into those slots would be filmed from their worlds' places (the named risk)."""
    from dtools.arcade import course as C
    rs = RomSet.from_zip(romset_zip)
    table = struct.unpack("<4I", rs.read_i960(C.CAMERA_SCRIPTS, 16))
    variants = struct.unpack("<4I", rs.read_i960(C.CAMERA_SCRIPTS_SLOT0, 16))
    assert table == (0x02806B78, 0x02807274, 0x028071A8, 0)
    assert set(variants) == {0x0280701C, 0x02806E90}
    for start in (table[0], *set(variants)):
        s = C.camera_script(rs, start)
        assert s.loop == start and len(s.keyframes) == 6 and s.keyframes[-1].time == 3600
        assert all(k.position_car == k.target_car == 0 and k.skip == 0 for k in s.keyframes)
        assert s.keyframes[0].position == (-4.0, 14.0, 10.0)                # the fly-by's first point, over the car
    for start in table[1:3]:
        s = C.camera_script(rs, start)
        assert s.loop == 0x02808360
        absolute = [k for k in s.keyframes if C.CAMERA_ABSOLUTE in (k.position_car, k.target_car)]
        assert [k.addr for k in absolute if k.addr >= s.loop] == [0x02808360, 0x028083A0]
    assert C.camera_script(rs, table[1]).keyframes[0].position == (400.0, 80.0, -70.0)


def test_camera_script_refuses(romset_zip):
    from dtools.arcade import course as C
    rs = RomSet.from_zip(romset_zip)
    for bad, match in ((True, "is not an address"), (0x6C98, "outside data ROM"), (0x02806B7A, "not 4-aligned"),
                       (DATA + (16 << 20) - 32, "outside data ROM")):
        with pytest.raises(CourseError, match=match):
            C.camera_script(rs, bad)
    with pytest.raises(CourseError, match="is not a RomSet"):
        C.camera_script(None, 0x02806B78)
    jump = rs.with_patch("data", 0x806B78 + 64 * 6 + 8, struct.pack("<I", 0x03000000))
    with pytest.raises(CourseError, match="keyframe.*outside data ROM"):
        C.camera_script(jump, 0x02806B78)
    odd = rs.with_patch("data", 0x806B78 + 64 * 6 + 8, struct.pack("<I", 0x02806B7A))
    with pytest.raises(CourseError, match="jump at 0x2806cf8 goes to 0x2806b7a"):
        C.camera_script(odd, 0x02806B78)
    nan = rs.with_patch("data", 0x806B78 + 64 + 0x1C, struct.pack("<f", float("nan")))
    with pytest.raises(CourseError, match="keyframe at 0x2806bb8 has a point that is not finite"):
        C.camera_script(nan, 0x02806B78)
    zeros = rs.with_patch("data", FREE_RUN[0], bytes(64 * 300))       # all-zero keyframes: never a jump
    with pytest.raises(CourseError, match="no loop within 256"):
        C.camera_script(zeros, DATA + FREE_RUN[0])


# --- course ids 0-15 (Phase 8, Task 3): grow_ids ---------------------------------------------------------------------

def _value_refs(rs: RomSet, value: int) -> set[int]:
    """i960 addresses of every program-ROM word (any byte offset) and data-ROM word (4-aligned, first 16 MB) equal to
    `value`; the program ROM's second half is named by its window address (0x220000+)."""
    from dtools.arcade import disasm
    out = {a if a < 0x20000 else a + 0x200000 for a in disasm.find_refs(rs, value, ("program",))}
    blob = rs.data[:16 << 20]
    i = blob.find(value.to_bytes(4, "little"))
    while i != -1:
        if i % 4 == 0:
            out.add(DATA + i)
        i = blob.find(value.to_bytes(4, "little"), i + 1)
    return out


def _mirror(addr: int) -> int:
    """The data address in the first 1 MB copy (0x02800000-) of an address in a ROM_COPY mirror (0x02900000-)."""
    return 0x02800000 + (addr - 0x02800000) % 0x100000 if 0x02800000 <= addr < 0x03000000 else addr


def test_id_arrays_layout():
    """COURSE_IDS sixteen-entry tables, one per GROWN_ARRAYS array, and the records of ids 3-15, all inside grow_ids'
    home (ID_HOME up to SRAM_SINK, the custom data area's first bytes: reserved, romset.RESERVED, and so not in
    CUSTOM_FREE) without overlapping; the WIDE_ARRAYS' and widen_ids' other tables inside WIDE_HOME (reserved too),
    before its code; record_address keeps ids 0-2 at COURSE_TABLE."""
    from dtools.arcade import course as C
    from dtools.arcade.romset import CUSTOM_AREAS, CUSTOM_FREE, RESERVED
    assert C.COURSE_IDS == 16 and C.ID_HOME == DATA + CUSTOM_AREAS["data"][0][0]
    lo, n = C.ID_HOME - DATA, C.SRAM_SINK - C.ID_HOME
    assert (lo, n) in [(o, m) for o, m, _ in RESERVED["data"]]
    assert all(o + m <= lo or lo + n <= o for o, m in CUSTOM_FREE["data"])
    spans = sorted([(C.ID_RECORDS, C.COURSE_IDS * COURSE_ENTRY_SIZE)]
                   + [(a.new, C.COURSE_IDS * a.stride) for a in C.GROWN_ARRAYS.values()])
    for (a0, n0), (a1, _) in zip(spans, spans[1:]):
        assert a0 + n0 <= a1, hex(a1)
    assert all(DATA + lo <= a and a + m <= DATA + lo + n and a % 4 == 0 for a, m in spans)
    for name, a in C.ID_ARRAYS.items():
        assert a.sites and 1 <= a.keep <= a.entries <= 4 and a.stride in (1, 2, 4, 8, 12, 16, 20), name
        assert a.old_span == (a.old, a.stride * a.entries)
    assert set(C.GROWN_ARRAYS) == set(C.ID_ARRAYS) - C.WIDE_ARRAYS and C.WIDE_ARRAYS == {"GRID_LANES", "BILLBOARD_REF"}
    wide = sorted([(C.ID_ARRAYS[n].new, C.COURSE_IDS * C.ID_ARRAYS[n].stride) for n in C.WIDE_ARRAYS]
                  + [(C.TUNES, C.COURSE_IDS), (C.VISIT_IDS, C.COURSE_IDS)])
    for (a0, n0), (a1, _) in zip(wide, wide[1:]):
        assert a0 + n0 <= a1, hex(a1)
    assert C.WIDE_HOME <= wide[0][0] and wide[-1][0] + wide[-1][1] <= C.WIDE_CODE < C.WIDE_HOME + C.WIDE_SIZE
    assert (C.WIDE_HOME, C.WIDE_SIZE) in [(o, m) for o, m, _ in RESERVED["program"]]
    assert all(a.keep == a.entries == 1 for n, a in C.ID_ARRAYS.items() if n in C.WIDE_ARRAYS)
    assert [C.record_address(i) for i in (0, 2, 3, 15)] == [COURSE_TABLE, COURSE_TABLE + 0x40, C.ID_RECORDS + 0x60,
                                                            C.ID_RECORDS + 0x1E0]
    for bad in (-1, 16, True, 1.0, None):
        with pytest.raises(CourseError):
            C.record_address(bad)


def test_id_arrays_readers_found(romset_zip):
    """Review Focus 1, the static half: on the original set every site holds its array's address, and every ROM word
    that holds it is a site (program ROM at any offset; data ROM 4-aligned, the ROM_COPY mirrors 0x02900000-
    0x02ffffff counting as their first copy, which with_patch writes with it), so no reader that loads an array's
    address whole is left unpatched. ID_PATCHES' words are the original instructions."""
    from dtools.arcade import course as C
    from dtools.arcade import disasm
    rs = RomSet.from_zip(romset_zip)
    for name, a in C.ID_ARRAYS.items():
        assert [struct.unpack("<I", rs.read_i960(s, 4))[0] for s in a.sites] == [a.old] * len(a.sites), name
        assert {_mirror(r) for r in _value_refs(rs, a.old)} == set(a.sites), name
    for addr, (old, _, what) in C.ID_PATCHES.items():
        assert struct.unpack("<I", rs.read_i960(addr, 4))[0] == old, hex(addr)
        want = {0x32BC: "mov     3,g9", 0x32D0: "subi    g9,3,g2"}.get(addr, "and     3,")
        assert disasm.listing(rs, addr, 1)[0].text.startswith(want), hex(addr)


def test_grow_ids_readers(romset_zip):
    """grow_ids: every site of every GROWN_ARRAYS array points at its new table (id_table follows it; WIDE_ARRAYS stay
    at their original addresses, every site as it was), each table holds
    the original entries 0..keep-1 and entry 0's bytes for the rest (COURSE_POINTERS: ids 3-15 point at their
    records, copies of Beginner's), the ID_PATCHES instructions are rewritten, TGP slot 3 has slot 0's words, and no
    other byte of the set changes (the original arrays stay as they are)."""
    import dataclasses
    from dtools.arcade import course as C
    from dtools.arcade import disasm
    stock = RomSet.from_zip(romset_zip)
    g = C.grow_ids(stock)
    assert g.custom and C.grown(g) and not C.grown(stock) and C.id_count(g) == 16 and C.id_count(stock) == 3
    expected: dict[str, set[int]] = {"program": set(), "data": set(), "copro_data": set()}

    def mark(addr: int, n: int) -> None:
        region, off = C._rom_word_at(addr)
        copies = [off]
        if region == "data" and 0x800000 <= off < 0x1000000:           # with_patch writes every ROM_COPY mirror
            copies = [0x800000 + k * 0x100000 + off % 0x100000 for k in range(8)]
        for o in copies:
            expected[region].update(range(o, o + n))

    for name in C.WIDE_ARRAYS:
        assert C.id_table(g, name) == C.ID_ARRAYS[name].old
    for name, a in C.GROWN_ARRAYS.items():
        assert C.id_table(g, name) == a.new and C.id_table(stock, name) == a.old, name
        assert [struct.unpack("<I", g.read_i960(s, 4))[0] for s in a.sites] == [a.new] * len(a.sites), name
        assert g.read_i960(a.old, a.stride * a.entries) == stock.read_i960(a.old, a.stride * a.entries), name
        table = g.read_i960(a.new, C.COURSE_IDS * a.stride)
        assert table[:a.stride * a.keep] == stock.read_i960(a.old, a.stride * a.keep), name
        if name == "COURSE_POINTERS":
            assert struct.unpack("<16I", table)[3:] == tuple(C.record_address(i) for i in range(3, 16))
        else:
            assert table[a.stride * a.keep:] == table[:a.stride] * (C.COURSE_IDS - a.keep), name
        mark(a.new, len(table))
        for s in a.sites:
            mark(s, 4)
    for addr, (_, new, _) in C.ID_PATCHES.items():
        assert struct.unpack("<I", g.read_i960(addr, 4))[0] == new
        mark(addr, 4)
    assert disasm.listing(g, 0xD474, 1)[0].text == "and     15,r3,r3"
    assert disasm.listing(g, 0xB820, 1)[0].text == "cmpobl  2,g9,0xb848"
    assert [ln.text for ln in disasm.listing(g, 0x32BC, 6)][0::5] == ["mov     4,g9", "subi    g9,4,g2"]
    beginner = stock.read_i960(COURSE_TABLE, COURSE_ENTRY_SIZE)
    for i in range(3, 16):
        assert g.read_i960(C.record_address(i), COURSE_ENTRY_SIZE) == beginner
        mark(C.record_address(i), COURSE_ENTRY_SIZE)
    words = lambda rs, half: struct.unpack_from("<32I", rs.copro_data, half + 0x40)
    for half in (0, len(stock.copro_data) // 2):                     # the region's two 2 MB copies
        before, after = words(stock, half), words(g, half)
        assert (after[3], after[19]) == (before[0], before[16]) == (0x30, 0x19C90)
        assert after[:3] + after[4:19] + after[20:] == before[:3] + before[4:19] + before[20:]
        expected["copro_data"].update(range(half + 0x4C, half + 0x50))
        expected["copro_data"].update(range(half + 0x8C, half + 0x90))
    custom = stock.as_custom()
    for region in ("program", "data", "copro_data"):
        a, b = (np.frombuffer(getattr(x, region), np.uint8) for x in (custom, g))
        changed = set(np.flatnonzero(a != b).tolist())
        assert changed <= expected[region], (region, sorted(changed - expected[region])[:8])
    assert (g.polygons, g.textures) == (custom.polygons, custom.textures)
    entries = course_table(g)
    assert len(entries) == 16 and entries[:3] == course_table(stock)
    assert all(dataclasses.replace(e, course=0) == entries[0] for e in entries[3:])
    assert [e.name for e in entries[2:5]] == ["Advanced", "course 3", "course 4"]


def test_grow_ids_refuses(romset_zip):
    """grow_ids refuses what is not a RomSet, a set already grown, a set whose reader does not hold the original
    address (naming the array and the word), a changed ID_PATCHES instruction and a new home that is not free."""
    from dtools.arcade import course as C
    stock = RomSet.from_zip(romset_zip)
    with pytest.raises(CourseError, match="is not a RomSet"):
        C.grow_ids(None)
    g = C.grow_ids(stock)
    with pytest.raises(CourseError, match="already grown"):
        C.grow_ids(g)
    moved = stock.with_patch("program", 0x1448, struct.pack("<I", 0x15B0))
    with pytest.raises(CourseError, match=r"COURSE_SHEETS: the word at 0x1448 is 0x15b0"):
        C.grow_ids(moved)
    changed = stock.with_patch("program", 0xD474, struct.pack("<I", 0x5818C887))
    with pytest.raises(CourseError, match=r"ID_PATCHES 0xd474"):
        C.grow_ids(changed)
    used = stock.as_custom().with_patch("data", C.ID_ARRAYS["GRID_TABLES"].new - DATA, b"\0")
    with pytest.raises(CourseError, match=r"GRID_TABLES: the new home 0x3.* is not free"):
        C.grow_ids(used)


def test_id_table_synthetic():
    """id_table on a set without the reader code (a synthetic program): the original address, so the decoders read
    such sets as before; unknown names and non-sets are refused."""
    from dtools.arcade import course as C
    rs = _synthetic()
    assert C.id_table(rs, "COURSE_POINTERS") == COURSE_POINTERS and not C.grown(rs) and C.id_count(rs) == 3
    assert C.id_tables(rs) == {n: a.old for n, a in C.ID_ARRAYS.items()}
    assert C.array_at(None, "GRID_TABLES") == GRID_TABLES and C.array_at({"GRID_TABLES": 5}, "GRID_TABLES") == 5
    assert C.ids_of({n: a.new for n, a in C.ID_ARRAYS.items()}) == 16 and C.ids_of({}) == 3
    for bad in ("NOPE", 3, None):
        with pytest.raises(CourseError, match="not a per-id array"):
            C.id_table(rs, bad)
    with pytest.raises(CourseError, match="is not a RomSet"):
        C.id_table(b"", "GRID_TABLES")
    with pytest.raises(CourseError, match="not a per-id array"):
        C.array_at(None, "NOPE")


def test_grow_ids_refuses_a_moved_tgp_slot(romset_zip):
    """grow_ids gives TGP slot 3 slot 0's words only while it still holds the stub's (0x18000, 0x465a0)."""
    from dtools.arcade import course as C
    stock = RomSet.from_zip(romset_zip)
    moved = stock.with_patch("copro_data", 4 * 0x13, struct.pack("<I", 0x30))
    with pytest.raises(CourseError, match=r"TGP slot 3: copro_data word 0x13 is 0x30, not the stub's 0x18000"):
        C.grow_ids(moved)


# --- the new ids' guards (Phase 8, Task 5) -----------------------------------------------------------------------

def test_guard_new_ids(romset_zip):
    """course.guard_new_ids on the grown set: the two `cmpobe 3` skips become `cmpobe 31` (unidasm), the counter
    routine's two blocks start `cmpobl 2,g6,0x229094` and otherwise take the same values (`mov g6,g5` / `mov g6,g4`
    for the no-op `and 0xffff`); the sink holds what a freshly reset SRAM holds for Beginner (the reset 0x2270d0
    replayed: rank, time (rank - 1) * 0x140 + 0x2800, view 0xffffffff, flags, shown time, name; the newest-entry word
    0xffffffff; the time-attack words 0x500 and best laps 0x2800); entries 3-15 of every SRAM_SINK_ARRAYS array name
    the sink, entries 0-2 are as before; nothing else changes. Refused: not a RomSet, a stock set, a guarded set, a
    moved word, a moved SRAM entry, a used sink."""
    from dtools.arcade import course as C
    from dtools.arcade import disasm
    stock = RomSet.from_zip(romset_zip)
    g = C.grow_ids(stock)
    out = C.guard_new_ids(g)
    assert C.guarded(out) and not C.guarded(g) and not C.guarded(stock)
    texts = {}
    for at, n in ((0x102E4, 4), (0x10B54, 4), (0x229020, 12), (0x229040, 12)):
        region, off = C.rom_word_at(at)
        texts[at] = [ln.text for ln in disasm.listing_bytes(out.program[off:off + n], at)]
    assert texts[0x102E4] == ["cmpobe  31,r3,0x10370"] and texts[0x10B54] == ["cmpobe  31,r3,0x10f2c"]
    assert texts[0x229020] == ["cmpobl  2,g6,0x229094", "mov     g6,g5", "mov     g6,g5"]
    assert texts[0x229040] == ["cmpobl  2,g6,0x229094", "mov     g6,g4", "mov     g6,g4"]
    sink = out.read_i960(C.SRAM_SINK, C.SRAM_SINK_SIZE)
    names = (0x47414E, 0x534F47, 0x534159)
    for k in range(C.RANKING_ENTRIES):
        rank, time, view, flags, shown, name = struct.unpack_from("<6I", sink, 24 * k)
        assert (rank, time, view, shown) == (k + 1, 0x2800 + 0x140 * k, 0xFFFFFFFF, time), k
        if k < 3:
            assert name == names[k]
    assert struct.unpack_from("<5I", sink, C.SINK_LAST) == (0xFFFFFFFF, 0x500, 0x2800, 0x500, 0x2800)
    assert sink[C.SINK_LAST + 20:] == bytes(C.SRAM_SINK_SIZE - C.SINK_LAST - 20)
    for name, words in C.SRAM_SINK_ARRAYS.items():
        a = C.ID_ARRAYS[name]
        assert out.read_i960(a.new, 3 * a.stride) == g.read_i960(a.new, 3 * a.stride), name
        for i in range(3, 16):
            assert struct.unpack(f"<{len(words)}I", out.read_i960(a.new + a.stride * i, a.stride)) == \
                tuple(C.SRAM_SINK + w for w in words), (name, i)
    # nothing else: program only the four sites, data only the sink and the arrays' entries 3-15
    prog = np.flatnonzero(np.frombuffer(out.program, np.uint8) != np.frombuffer(g.program, np.uint8))
    sites = {C.rom_word_at(a)[1] + k for a in C.ID3_LITERALS for k in range(4)} | \
        {C.rom_word_at(a)[1] + k for a in C.COUNTER_GUARD for k in range(12)}
    assert set(prog.tolist()) <= sites
    data = np.flatnonzero(np.frombuffer(out.data, np.uint8) != np.frombuffer(g.data, np.uint8))
    allowed = set(range(C.SRAM_SINK - DATA, C.SRAM_SINK - DATA + C.SRAM_SINK_SIZE))
    for name in C.SRAM_SINK_ARRAYS:
        a = C.ID_ARRAYS[name]
        allowed |= set(range(a.new - DATA + 3 * a.stride, a.new - DATA + 16 * a.stride))
    assert set(data.tolist()) <= allowed
    assert (out.copro_data, out.polygons, out.textures) == (g.copro_data, g.polygons, g.textures)
    # refusals
    for bad, match in ((None, "is not a RomSet"), (stock, "grown set"), (out, "already")):
        with pytest.raises(CourseError, match=match):
            C.guard_new_ids(bad)
    moved = g.with_patch("program", 0x102E4, struct.pack("<I", 0x3218E090))
    with pytest.raises(CourseError, match=r"ID3_LITERALS 0x102e4"):
        C.guard_new_ids(moved)
    moved = g.with_patch("program", 0x29024, struct.pack("<I", 0xFFFE))
    with pytest.raises(CourseError, match=r"COUNTER_GUARD 0x229020"):
        C.guard_new_ids(moved)
    a = C.ID_ARRAYS["BEST_LAPS_2805100"]
    moved = g.with_patch("data", a.new - DATA + 4 * 5, struct.pack("<I", 0x1D00C40))
    with pytest.raises(CourseError, match=r"BEST_LAPS_2805100\[5\]"):
        C.guard_new_ids(moved)
    used = g.with_patch("data", C.SRAM_SINK - DATA + 0x100, b"\0")
    with pytest.raises(CourseError, match="sink"):
        C.guard_new_ids(used)
    with pytest.raises(CourseError):
        C.sram_sink(None)
    with pytest.raises(CourseError):
        C.guarded(b"")


def test_rom_object_index():
    """The encoders' polygon bound (course.rom_object_index): the 32 MB decoding limit (POLY_WINDOW words), where
    check_rom_object without a set keeps the original's 16 MB."""
    from dtools.arcade import course as C
    assert C.rom_object_index(0xC00000) == 0x400000 and C.rom_object_index(0xFFFFFF) == 0x7FFFFF
    with pytest.raises(CourseError, match="16 MB"):
        check_rom_object(0xC00000)
    for bad in (0x1000000, 0x400000, True, -1, 0x2800000, 1.0, None):
        with pytest.raises(CourseError):
            C.rom_object_index(bad)


def test_fix1_sram_sink_checked_and_reserved(romset_zip):
    """Fix round 1. The sink's times are computed as the reset computes them and refused outside u32 (a rank byte 0
    with a time base below 0x140 would be negative, a base near 2**32 past it) with CourseError, through
    guard_new_ids too; fresh_ranking / fresh_settings replay the reset for ids 0-2 (the sink is Beginner's). The sink
    is registered in romset.RESERVED: tgp.write_free refuses it, the converter's free-space search skips it, and
    CUSTOM_FREE leaves it out."""
    from dtools.arcade import course as C
    from dtools.arcade import romset, tgp
    from dtools.convert import ConvertError, build
    stock = RomSet.from_zip(romset_zip)
    g = C.grow_ids(stock)
    first = C.fresh_ranking(g, 0)
    assert C.sram_sink(g)[:576] == first and struct.unpack_from("<6I", first, 0)[:2] == (1, 0x2800)
    assert [struct.unpack_from("<I", C.fresh_ranking(g, c), 4)[0] for c in range(3)] == [0x2800, 0x3C00, 0x3C00]
    assert struct.unpack("<17I", C.fresh_settings(g))[:8] == (0x500, 0xF00, 0x1E00, 0xFFFFFFFF, 0x2800, 0x3C00,
                                                             0x3C00, 0xFFFFFFFF)
    low = g.with_patches({("program", 0x39122): b"\0", ("program", 0x3909C): b"\0\0\0\0"})   # the reviewer's repro
    high = g.with_patch("program", 0x3909C, struct.pack("<I", 0xFFFFFFF0))
    for bad in (low, high):
        with pytest.raises(CourseError, match="outside u32"):
            C.sram_sink(bad)
        with pytest.raises(CourseError, match="outside u32"):
            C.guard_new_ids(bad)
    for bad in ((None, 0), (g, 3), (g, True), (g, -1)):
        with pytest.raises(CourseError):
            C.fresh_ranking(*bad)
    with pytest.raises(CourseError):
        C.fresh_settings(None)
    off, size = C.SRAM_SINK - DATA, C.SRAM_SINK_SIZE
    assert (off, size) in [(lo, n) for lo, n, _ in romset.RESERVED["data"]]
    assert romset.reserved("data", off + size - 4, 4) and romset.reserved("data", off + size, 4) is None
    assert all(lo + n <= off or off + size <= lo for lo, n in romset.CUSTOM_FREE["data"])
    with pytest.raises(tgp.TgpError, match="reserved"):
        tgp.write_free(g, "data", off + 0x100, b"\0" * 4)       # still 0xff on the grown set: refused by name
    with pytest.raises(tgp.TgpError, match="reserved for grow_ids"):     # final review fix 1: record slots 0-2,
        tgp.write_free(g, "data", C.ID_HOME - DATA, b"\0" * 4)            # unused and 0xff, are grow_ids' too
    usable = build._usable(g, "data", (), (off - 0x100, size + 0x200))
    assert not usable[0x100:0x100 + size].any() and usable[0x100 + size:].all()      # before it: grow_ids' tables
    with pytest.raises(ConvertError):
        build._check_span(g, "data", off, 8, "probe", (), (off - 0x100, size + 0x200))


# --- a mirror tests as its original (Phase 9, Task 2): widen_ids ----------------------------------------------------

_ID_SEEDS = (0x00501460, 0x00501468, 0x00501A98, 0x00501B40, 0x00501B41, 0x00501B42)
"""The course id, the select's entry, the music id, and on a widened set the id's shadows course.SH_PATH, SH_TUNE,
SH_VISIT (test_widen_idempotent_and_refuses ties them to course)."""
_ARITH = ("mov", "and", "andnot", "or", "subi", "subo", "addi", "addo", "shlo", "shli", "shro", "shri", "remi",
          "remo", "mulo", "muli", "divo", "divi", "notbit", "clrbit", "setbit")
_CONDITIONAL = ("be", "bne", "bl", "ble", "bg", "bge", "bo", "bno")


_ID_STORES = {
    0x213C: "stib    r4,0x501460",          # the attract's next course (id rem 4) back to the id: ID_WRITERS
    0x340C: "st      r3,0x501a98",          # the music id (followed: a seed)
    0x3BD0: "st      r3,0x501a98",          # the music id (followed: a seed)
    0x3818: "stob    r3,0x540c17",          # link RAM: linked mode only (decision 20)
    0x46DC: "st      r10,(g11)[g12]",       # the TGP FIFO: function 0x0d, slot = id (sixteen slots already)
    0x5720: "stob    r3,0x501460",          # linked mode's copy back to the id: ID_WRITERS
    0x1AFC8: "stob    r3,0x17(r14)",        # a link packet's course byte (linked mode)
    0x1B14C: "stob    r9,0x31(r10)",        # the music id into a link packet (linked mode)
    0x1B29C: "stob    r3,0x17(r15)",        # a link packet's course byte (linked mode)
}
"""The stores of the course id (or of a value the sweep derives from it) on the guarded set: its copies in memory,
which the sweep does not follow but through the music id. Each is accounted for; a new one fails
test_literal_sites_cover, to be followed or explained."""


def _program_lines(rs: RomSet) -> dict[int, str]:
    """i960 address -> unidasm's text for the whole program ROM: 0-0x1ffff, and the second half at its window
    0x220000-0x23ffff (where it runs)."""
    from dtools.arcade import disasm
    out = {}
    for base, lo, hi in ((0, 0, 0x20000), (0x220000, 0x20000, 0x40000)):
        out.update({ln.addr: ln.text for ln in disasm.listing_bytes(rs.program[lo:hi], base)})
    return out


def _operands(text: str) -> tuple[str, list[str]]:
    op, _, rest = text.partition(" ")
    return op, [o.strip() for o in re.split(r",(?![^\[]*\])", rest.strip())] if rest.strip() else []


def _id_compares(rs: RomSet) -> dict[int, str]:
    """address -> text of every compare of the course id (or a value derived from it) against a literal: `_id_sweep`'s
    first half."""
    return _id_sweep(rs)[0]


def _id_sweep(rs: RomSet) -> tuple[dict[int, str], dict[int, str]]:
    """The sweep (discovery log, "Literal ids"): every load of the course id, the select's entry, the music id or the
    id's shadows (_ID_SEEDS, by absolute address) followed forward through its register: into the registers
    arithmetic derives from it (mov, and, shifts, adds ...), along the fall-through and every branch target, into a
    called routine for the global registers that hold it (locals stay with the caller), until each register is
    overwritten, the routine returns or 80 instructions. A load whose address is computed from it ends the trail
    (table data, not the id). Returns (address -> text of every compare of such a register against a literal,
    address -> text of every store of such a register: the id's copies in memory, which the sweep does not follow
    and _ID_STORES lists)."""
    lines = _program_lines(rs)
    order = sorted(lines)
    nxt = dict(zip(order, order[1:]))
    out: dict[int, str] = {}
    stores: dict[int, str] = {}
    seeds = []
    for a, text in lines.items():
        op, o = _operands(text)
        if op.startswith("ld") and o and re.fullmatch(r"0x[0-9a-f]+", o[0]) and int(o[0], 16) in _ID_SEEDS:
            seeds.append((a, o[-1]))
    for seed, reg in seeds:
        work, seen = [(nxt[seed], frozenset([reg]), 0)], set()
        while work:
            a, regs, depth = work.pop()
            if a not in lines or not regs or depth > 80 or (a, regs) in seen:
                continue
            seen.add((a, regs))
            op, o = _operands(lines[a])
            used = any(re.search(rf"\b{r}\b", x) for x in o for r in regs)
            if used and op.startswith(("cmp", "test")) and any(re.fullmatch(r"\d+", x) for x in o[:2]):
                out[a] = lines[a]
            if op.startswith("st") and o and o[0] in regs:
                stores[a] = lines[a]
            live = set(regs)
            if o and re.fullmatch(r"[rg]\d+", o[-1]) and not op.startswith(("st", "cmp", "chk", "b", "call", "test")):
                if op in _ARITH and any(x in regs for x in o[:-1]):
                    live.add(o[-1])
                else:
                    live.discard(o[-1])
            target = re.search(r"0x([0-9a-f]+)$", lines[a])
            to = int(target.group(1), 16) if target else None
            if op == "ret":
                continue
            if op in ("call", "bal"):
                g = frozenset(r for r in live if r.startswith("g"))
                if g and to is not None:
                    work.append((to, g, depth + 1))
                live = {r for r in live if r.startswith("r")}
            if op == "b":
                if to is not None:
                    work.append((to, frozenset(live), depth + 1))
                continue
            if to is not None and (op in _CONDITIONAL or (op.startswith("cmp") and len(o) == 3)):
                work.append((to, frozenset(live), depth + 1))
            if a in nxt:
                work.append((nxt[a], frozenset(live), depth + 1))
    return out, stores


@pytest.fixture(scope="module")
def _wide_sets():
    from dtools import config
    from dtools.arcade import course as C
    d = config.rom_dir()
    if d is None or not (d / "daytona.zip").is_file():
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    stock = RomSet.from_zip(d / "daytona.zip")
    g = C.guard_new_ids(C.grow_ids(stock))
    return stock, g, C.widen_ids(g)


def test_literal_sites_cover(_wide_sets):
    """Review Focus 2's static half. The sweep (`_id_compares`) over the guarded set finds exactly the literal tests
    of LITERAL_ID_SITES and LITERAL_ID_KEPT, with their text, and over the widened set the same tests, unchanged. Every
    widened site keeps its instruction word (not a cycle more: MAME's i960 charges by opcode) and its displacement
    names its shadow: an "and7" load reads SH_PATH, the music id's two loads SH_TUNE ("table": TUNES), the culling
    walk's SH_VISIT ("table": VISIT_IDS), GRID_LANES' readers SH_LANES and BILLBOARD_REF's SH_REF. The update decodes
    to its reads of the id and the per-id tables and its writes of the five shadows, and every ID_WRITERS instruction
    branches to a trampoline that runs it, calls the update and branches back."""
    from dtools.arcade import course as C
    from dtools.arcade import disasm
    _, g, w = _wide_sets
    sweep, stores = _id_sweep(g)
    want = {a: s.insn for a, s in C.LITERAL_ID_SITES.items()} | {a: t for a, (t, _) in C.LITERAL_ID_KEPT.items()}
    assert sweep == want, (sorted(set(sweep) - set(want)), sorted(set(want) - set(sweep)))
    assert stores == _ID_STORES, {hex(a): t for a, t in stores.items() if _ID_STORES.get(a) != t}
    w_sweep, w_stores = _id_sweep(w)
    assert w_sweep == want
    labels = C.wide_code().labels                           # on the widened set two writers run in their trampolines,
    moved = {labels[f"writer_{a:x}"]: t for a, t in _ID_STORES.items() if a in C.ID_WRITERS}     # and the update
    assert w_stores == {a: t for a, t in _ID_STORES.items() if a not in C.ID_WRITERS} | moved | {   # stores id & 7
        labels["update"] + 16: f"stob    r4,{C.SH_PATH:#x}"}
    assert {s.rule for s in C.LITERAL_ID_SITES.values()} == {"and7", "table"}
    assert _ID_SEEDS[3:] == (C.SH_PATH, C.SH_TUNE, C.SH_VISIT)
    text = lambda rs, a, n=1: [ln.text for ln in disasm.listing(rs, a, n)]
    for at, s in C.LITERAL_ID_SITES.items():
        assert text(g, at) == text(w, at) == [s.insn], hex(at)
        assert text(g, s.load) == [f"{C._LOAD_OPS[s.load]:8s}0x501460,{s.reg}"], hex(at)
        shadow = {"and7": C.SH_PATH}.get(s.rule, C.SH_VISIT if s.load == 0x17418 else C.SH_TUNE)
        assert text(w, s.load) == [f"{C._LOAD_OPS[s.load]:8s}{shadow:#x},{s.reg}"], hex(at)
    for at in C.TUNE_SITES:
        assert text(w, at) == [f"ldob    {C.SH_TUNE:#x},r3"]
    for at, x in C.LANE_READERS.items():
        assert text(g, at) == [f"ldos    0x231b30[{x}*2],{x}"]
        assert text(w, at) == [f"ldos    {C.SH_LANES:#x}[{x}*2],{x}"]
    assert text(g, 0x221E88) == ["lda     0x235bb4,r8"] and text(w, 0x221E88) == [f"lda     {C.SH_REF:#x},r8"]
    for at in C.WIDE_SHADOWS:                               # the instruction word stays: the same cycles
        assert w.read_i960(at, 4) == g.read_i960(at, 4) == C.WIDE_ORIGINALS[at][:4], hex(at)
    blob = w.read_i960(C.WIDE_CODE, len(C.wide_code().bytes()))
    lines = disasm.listing_bytes(blob, C.WIDE_CODE)
    assert sum(len(ln.bytes) for ln in lines) == len(blob) and not any("?" in ln.text for ln in lines)
    texts = [ln.text for ln in lines]
    end = texts.index("ret")
    assert texts[:3] == ["ldob    0x501460,r3", "and     15,r3,r3", "and     7,r3,r4"]
    assert [t.split(",")[-1] for t in texts[:end] if t.startswith("st")] == [
        f"{a:#x}" for a in (C.SH_PATH, C.SH_TUNE, C.SH_VISIT, C.SH_LANES, C.SH_REF, C.SH_REF + 4, C.SH_REF + 8)]
    assert {f"{a:#x}" for a in (C.TUNES, C.VISIT_IDS, C.WIDE_LANES, C.WIDE_BILLBOARD)} <= {
        t.split()[1].split("(")[0].split("[")[0].rstrip(",").split(",")[0] for t in texts[:end] if t.startswith("ld")}
    for at, (n, _) in C.ID_WRITERS.items():
        head = text(w, at)[0]
        t = int(head.split()[1], 16)
        assert head.startswith("b       ") and t > C.WIDE_CODE + 4 * end, hex(at)
        assert text(w, t, 3) == [text(g, at)[0], f"call    {C.WIDE_CODE:#010x}", f"b       {at + n:#010x}"], hex(at)
        assert g.read_i960(at, n) == C.WRITER_ORIGINALS[at]


def test_widen_idempotent_and_refuses(_wide_sets):
    """widen_ids: twice is once (the widened set comes back as it is; the guards commute with it); only the sites'
    displacement words, the writers' instructions and WIDE_HOME change; WIDE_HOME holds the tables (GRID_LANES and
    BILLBOARD_REF per id, every entry the shared value; TUNES TUNE_DEFAULTS; VISIT_IDS id & 7) and the code; the
    WIDE_ARRAYS' sites name their shadows and id_table gives the per-id tables; WIDE_HOME and WIDE_RAM are reserved
    (romset.RESERVED, RESERVED_RAM: owned, so no allocator takes them). Refused: not a RomSet, a set that is not grown,
    a site or writer that is not the original instruction, a changed culling test, a WIDE_HOME that is not free."""
    from dtools.arcade import course as C
    from dtools.arcade import golden as G
    from dtools.arcade import romset
    stock, g, w = _wide_sets
    assert C.widen_ids(w) is w and C.widened(w) and not C.widened(g) and not C.widened(stock)
    assert C.guard_new_ids(C.widen_ids(C.grow_ids(stock))).program == w.program     # the guards commute
    changed = np.flatnonzero(np.frombuffer(w.program, np.uint8) != np.frombuffer(g.program, np.uint8))
    allowed = {C.rom_word_at(a)[1] + 4 + k for a in C.WIDE_SHADOWS for k in range(4)}
    allowed |= {C.rom_word_at(a)[1] + k for a in C.ID_WRITERS for k in range(4)}
    allowed |= set(range(C.WIDE_HOME, C.WIDE_HOME + C.WIDE_SIZE))
    assert set(changed.tolist()) <= allowed and set(changed.tolist()) & set(range(C.WIDE_CODE, C.WIDE_CODE + 4))
    assert (w.data, w.copro_data, w.polygons, w.textures) == (g.data, g.copro_data, g.polygons, g.textures)
    assert w.read_i960(C.WIDE_LANES, 64) == g.read_i960(C.GRID_LANES, 4) * 16
    assert w.read_i960(C.WIDE_BILLBOARD, 192) == g.read_i960(C.BILLBOARD_REF, 12) * 16
    assert C.tunes(w) == C.TUNE_DEFAULTS == (0, 1, 2, 3, 0, 0, 0, 0) * 2
    assert C.visit_ids(w) == C.VISIT_DEFAULTS == tuple(range(8)) * 2
    for name in C.WIDE_ARRAYS:
        a = C.ID_ARRAYS[name]
        assert C.id_table(w, name) == a.new and C.id_table(g, name) == C.id_table(stock, name) == a.old
        shadow = {"GRID_LANES": C.SH_LANES, "BILLBOARD_REF": C.SH_REF}[name]
        assert all(struct.unpack("<I", w.read_i960(s, 4))[0] == shadow for s in a.sites)
        assert [C.id_entry(C.id_tables(w), name, k) for k in (0, 9)] == [a.new, a.new + 9 * a.stride]
        assert [C.id_entry(C.id_tables(g), name, k) for k in (0, 9)] == [a.old, a.old]
    assert romset.reserved("program", C.WIDE_HOME, 4).startswith("widen_ids")
    assert romset.reserved("program", C.WIDE_HOME - 4, 4).startswith("the sixteen-entry select")
    assert (C.WIDE_RAM, C.WIDE_RAM_SIZE) in [(a, n) for a, n, _ in romset.RESERVED_RAM]
    assert C.SH_REF + 12 == C.WIDE_RAM + C.WIDE_RAM_SIZE and C.SH_LANES % 4 == C.SH_REF % 4 == 0
    spans = [(o, n) for r, o, n, _ in G.owned_spans(w) if r == "program"]
    assert any(o <= C.WIDE_HOME and C.WIDE_HOME + C.WIDE_SIZE <= o + n for o, n in spans)
    for bad, match in ((None, "is not a RomSet"), (stock, "grown set"), (b"", "is not a RomSet")):
        with pytest.raises(CourseError, match=match):
            C.widen_ids(bad)
    lda = struct.pack("<I", 0x8C183000)                                    # lda <abs32>,r3
    moved = g.with_patch("program", 0x3EAC, lda)
    with pytest.raises(CourseError, match=r"widen_ids 0x3eac: the instruction is 0030188c"):
        C.widen_ids(moved)
    moved = g.with_patch("program", 0x24CE8, lda)                         # a window site, by its program offset
    with pytest.raises(CourseError, match=r"widen_ids 0x224ce8"):
        C.widen_ids(moved)
    moved = g.with_patch("program", 0x32DC, lda)                          # a writer
    with pytest.raises(CourseError, match=r"widen_ids 0x32dc"):
        C.widen_ids(moved)
    moved = g.with_patch("program", 0x17428, struct.pack("<I", 0x0C001C35))
    with pytest.raises(CourseError, match=r"0x17420"):
        C.widen_ids(moved)
    used = g.with_patch("program", C.WIDE_HOME + C.WIDE_SIZE - 1, b"\0")     # its last byte (RANK_HOME follows)
    with pytest.raises(CourseError, match=r"WIDE_HOME 0x1f800 \(\+0x600\) is not free.*0x1fdff"):
        C.widen_ids(used)
    for f in (C.widened, C.tunes, C.visit_ids):
        with pytest.raises(CourseError):
            f(None)
    for f in (C.tunes, C.visit_ids):
        with pytest.raises(CourseError, match="widened set"):
            f(g)
    for args in (({}, "GRID_LANES", 16), ({}, "GRID_LANES", True), ({}, "NOPE", 0), (None, "BILLBOARD_REF", 1.0)):
        with pytest.raises(CourseError):
            C.id_entry(*args)


SELECT_ID_STORES = {"wheel_store": "the select's wheel (select.code's label)",
                    "timeout_store": "the select's timeout (select.code's label)",
                    0x225FB8: "the original wheel's store, after select.HOOKS' branch at 0x225f8c: not reached"}
"""The select's stores to the course id: select.code's labels (resolved on the patched set's layout) and the original
wheel's dead store, by address."""


def test_id_writers(_wide_sets):
    """Every program store to the course id (an `st*` whose address is 0x501460) on the full chain (grow_ids,
    guard_new_ids, widen_ids, patch_select) is an ID_WRITERS trampoline's (the store, then `call update`), 0x2124 (the
    attract's first store, followed on the same straight run by 0x213c's, which refreshes the shadows) or one of the
    select's (SELECT_ID_STORES: no widened site reads the id before the vote's refresh). A writer added by a later
    task fails here: it must refresh the shadows (ID_WRITERS) or be explained. Stores through a computed address are
    out of a static scan's reach (the traced session found none: discovery log, "Literal ids")."""
    from dtools.arcade import course as C
    from dtools.arcade import select as S
    _, _, w = _wide_sets
    from tests.test_select import unconverted_select
    full = unconverted_select(w)        # no mirror, no Desert City on this set: Phase 8's row
    lines = _program_lines(full)
    stores = {a for a, t in lines.items() if t.startswith("st") and t.endswith(",0x501460")}
    labels = C.wide_code().labels
    tramps = {labels[f"writer_{a:x}"] for a, (n, _) in C.ID_WRITERS.items() if n == 8}
    select_labels = S.code(S.layout_of(full)).labels
    selects = {select_labels[k] if isinstance(k, str) else k for k in SELECT_ID_STORES}
    assert stores == tramps | {0x2124} | selects, sorted(hex(a) for a in stores)
    for t in tramps:
        assert lines[t + 8] == f"call    {C.WIDE_CODE:#010x}", hex(t)
    assert [lines[a] for a in (0x212C, 0x2130, 0x2138)] == ["mov     4,r3", "ldib    0x501460,r4", "remi    r3,r4,r4"]
    assert lines[0x213C].startswith("b       ") and int(lines[0x213C].split()[1], 16) == labels["writer_213c"]


# --- rankings per entry (M4b, Task 6) -------------------------------------------------------------------------------

from tests.test_convert_tables import disc  # noqa: E402, F401  (disc: the module-scoped disc fixture)


def _ranking_words(rs: RomSet, name: str) -> tuple[int, ...]:
    from dtools.arcade import course as C
    return struct.unpack("<16I", rs.read_i960(C.id_table(rs, name), 4 * C.COURSE_IDS))


def test_grow_rankings_layout(_wide_sets):
    """course.grow_rankings on the widened chain: sixteen ranking tables of 576 bytes from RANKING_SRAM (0x1d00c30,
    past the time-attack records' padding word 0x1d00c28 that 0x230b9c names) and sixteen newest-entry words after
    them, none overlapping another or an area the game uses, all inside the stock layout's free area; RANKING_TABLES
    and RANKING_LAST name them (no id the sink any more; the time-attack and best-lap words of ids 3-15 keep it); the
    reset 0x2270d0 loops over sixteen ids (unidasm) through RANK_DEFAULTS and RANK_BASES (mode * 16 + id) in
    RANK_HOME, reserved and owned; the signature at 0x5b78 changed; a fresh table of ids 0-2 is the stock one, of a
    mirror 8-10 its original's, of the others Beginner's. Nothing else changes. Refused: not a RomSet, a stock set, a
    set without the guards, a ranked set, a moved word, a used RANK_HOME, a moved entry."""
    from dtools.arcade import course as C
    from dtools.arcade import disasm
    from dtools.arcade import golden as G
    from dtools.arcade import romset
    stock, g, w = _wide_sets
    r = C.grow_rankings(w)
    assert C.ranked(r) and not C.ranked(w) and not C.ranked(stock)
    lay, old = C.rankings_layout(r), C.rankings_layout()
    size = C.RANKING_ENTRIES * C.RANKING_ENTRY_SIZE
    assert size == C.RANKING_TABLE_SIZE == 576 and C.RANKING_SRAM == 0x01D00C30
    assert lay.tables == tuple(C.RANKING_SRAM + size * k for k in range(16))
    assert lay.last_entry == tuple(C.RANKING_LAST_SRAM + 4 * k for k in range(16))
    assert C.RANKING_LAST_SRAM == C.RANKING_SRAM + 16 * size == 0x01D03030
    spans = sorted([(t, t + size) for t in lay.tables] + [(p, p + 4) for p in lay.last_entry])
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))                      # no overlap
    assert all(old.free[0] <= lo and hi <= old.free[1] for lo, hi in spans)         # inside the free area
    used = [(lo, hi) for _, lo, hi in old.used] + [(C.RANKING_PAD, C.RANKING_PAD + 4)]
    assert not any(lo < h and l < hi for lo, hi in spans for l, h in used)
    assert lay.free == (C.RANKING_LAST_SRAM + 64, C.SRAM[1])
    assert (lay.signature, lay.signature_source) == (old.signature, old.signature_source)
    # the arrays: every id its own table and word; the sink keeps the time-attack and best-lap words
    assert _ranking_words(r, "RANKING_TABLES") == lay.tables
    assert _ranking_words(r, "RANKING_LAST") == lay.last_entry
    for name, words in C.SRAM_SINK_ARRAYS.items():
        if name in ("RANKING_TABLES", "RANKING_LAST"):
            continue
        a = C.ID_ARRAYS[name]
        assert r.read_i960(a.new, 16 * a.stride) == w.read_i960(a.new, 16 * a.stride), name
    # the reset loop and its time base: same instructions, sixteen ids, the new tables
    def text(at, n):
        region, off = C.rom_word_at(at)
        return [ln.text for ln in disasm.listing_bytes(r.program[off:off + n], at)]
    assert text(0x2270D4, 12) == ["mov     16,r15", f"lda     {C.RANK_DEFAULTS:#x},r14"]
    assert text(0x227100, 4) == ["subo    r15,16,g1"]
    assert text(0x227148, 16) == ["shlo    4,r5,r5", "addo    g1,r5,r5", f"ld      {C.RANK_BASES:#x}[r5*4],r4"]
    assert r.read_i960(C.SIGNATURE_AT, 24) == C.SIGNATURE_RANKED != C.SIGNATURE_STOCK == \
        stock.read_i960(C.SIGNATURE_AT, 24)
    # defaults: (defaults list, table) per id, time bases per mode and id; the fresh tables they give
    for k in range(16):
        src = C.RANK_SOURCES[k]
        assert src == (k - 8 if 8 <= k <= 10 else k if k < 3 else 0)
        dlist, table = struct.unpack("<2I", r.read_i960(C.RANK_DEFAULTS + 8 * k, 8))
        assert (dlist, table) == (_ranking_defaults(stock)[src], lay.tables[k])
        for m in range(C.MODES):
            assert struct.unpack("<I", r.read_i960(C.RANK_BASES + 4 * (16 * m + k), 4))[0] == \
                struct.unpack("<I", stock.read_i960(C.RANKING_TIME_BASES + 4 * (4 * m + src), 4))[0], (k, m)
        assert C.fresh_ranking(r, k) == C.fresh_ranking(stock, src), k
    assert [C.fresh_ranking(r, k) for k in range(3)] == [C.fresh_ranking(stock, k) for k in range(3)]
    # ownership: RANK_HOME is reserved (owned whatever its bytes), apart from widen_ids' space
    assert romset.reserved("program", C.RANK_HOME, 4).startswith("grow_rankings")
    assert (C.RANK_HOME, C.RANK_SIZE) in [(o, n) for o, n, _ in romset.RESERVED["program"]]
    assert C.WIDE_HOME + C.WIDE_SIZE <= C.RANK_HOME and C.RANK_BASES + 4 * 16 * C.MODES <= C.RANK_HOME + C.RANK_SIZE
    assert C.WIDE_CODE + len(C.wide_code().bytes()) <= C.WIDE_HOME + C.WIDE_SIZE
    spans = [(o, n) for reg, o, n, _ in G.owned_spans(r) if reg == "program"]
    assert any(o <= C.RANK_HOME and C.RANK_HOME + C.RANK_SIZE <= o + n for o, n in spans)
    # nothing else changed
    prog = np.flatnonzero(np.frombuffer(r.program, np.uint8) != np.frombuffer(w.program, np.uint8))
    allowed = {C.rom_word_at(a)[1] + k for a in C.RANKING_PATCHES for k in range(4)}
    allowed |= set(range(C.SIGNATURE_AT, C.SIGNATURE_AT + 24)) | set(range(C.RANK_HOME, C.RANK_HOME + C.RANK_SIZE))
    assert len(prog) and set(prog.tolist()) <= allowed
    data = np.flatnonzero(np.frombuffer(r.data, np.uint8) != np.frombuffer(w.data, np.uint8))
    allowed = set()
    for name in ("RANKING_TABLES", "RANKING_LAST"):
        a = C.ID_ARRAYS[name]
        allowed |= set(range(a.new - DATA, a.new - DATA + 16 * a.stride))
    assert set(data.tolist()) <= allowed
    assert (r.copro_data, r.polygons, r.textures) == (w.copro_data, w.polygons, w.textures)
    # refusals
    for bad, match in ((None, "is not a RomSet"), (stock, "grown set"), (C.grow_ids(stock), "guards"),
                       (r, "ranked already")):
        with pytest.raises(CourseError, match=match):
            C.grow_rankings(bad)
    moved = w.with_patch(*C.rom_word_at(0x2270D4), struct.pack("<I", 0x5C781E04))
    with pytest.raises(CourseError, match="0x2270d4"):
        C.grow_rankings(moved)
    with pytest.raises(CourseError, match="signature"):
        C.grow_rankings(w.with_patch("program", C.SIGNATURE_AT, b"X"))
    with pytest.raises(CourseError, match="RANK_HOME"):
        C.grow_rankings(w.with_patch("program", C.RANK_HOME + 0x1F0, b"\x00"))
    a = C.ID_ARRAYS["RANKING_TABLES"]
    with pytest.raises(CourseError, match=r"RANKING_TABLES\[5\]"):
        C.grow_rankings(w.with_patch(*C.rom_word_at(a.new + 4 * 5), struct.pack("<I", 0x01D00360)))
    for f in (C.ranked, C.rankings_layout):
        with pytest.raises(CourseError):
            f(b"")
    with pytest.raises(CourseError, match="course id"):
        C.fresh_ranking(r, 16)
    with pytest.raises(CourseError, match="course id"):
        C.fresh_ranking(w, 3)


def _ranking_defaults(rs: RomSet) -> tuple[int, ...]:
    """The stock RANKING_DEFAULTS' defaults lists of ids 0-2."""
    from dtools.arcade import course as C
    return tuple(struct.unpack("<I", rs.read_i960(C.RANKING_DEFAULTS + 8 * k, 4))[0] for k in range(3))


def test_grow_rankings_keeps_courses(romset_zip, disc):  # noqa: F811
    """The chain with grow_rankings placed after the mirrors and before the select (grow_ids, guard_new_ids,
    widen_ids, Desert City converted at id 3, the mirrors of 0-3 at 8-11, grow_rankings, patch_select): ids 0-3 and
    8-11 decode as without it but for `rankings` (ids 0-2's table pointers, shared: now RANKING_SRAM's first three),
    re-emit losslessly, and the set is the same whichever of grow_rankings and patch_select comes first."""
    from dtools.arcade import course as C
    from dtools.arcade import golden as G
    from dtools.arcade import mirror as M
    from dtools.arcade import select as S
    from dtools.convert import build
    from tests.test_golden import _differ
    stock = RomSet.from_zip(romset_zip)
    before = build.converted_set(C.widen_ids(C.guard_new_ids(C.grow_ids(stock))), disc, 3, slot=3, scenery="model")
    for o in (0, 1, 2, 3):
        before = M.mirrored_entry(before, o)
    after = C.grow_rankings(before)
    for k in (0, 1, 2, 3, 8, 9, 10, 11):
        x, y = G.decode_course(before, k), G.decode_course(after, k)
        assert _differ(x, y) == {"rankings"}, k
        assert y.rankings == tuple(C.RANKING_SRAM + C.RANKING_TABLE_SIZE * i for i in range(3)), k
        assert G.verify_lossless(after, k) == [], k
    a = S.patch_select(after, S.M4B_ENTRIES)                       # Phase 9's row: the set holds no id 4
    b = C.grow_rankings(S.patch_select(before, S.M4B_ENTRIES))
    assert (a.program, a.data, a.copro_data, a.polygons, a.textures) == \
        (b.program, b.data, b.copro_data, b.polygons, b.textures)


# ---- the pit stalls (v0.1.2) ----

def test_pit_stalls_real(romset_zip):
    """CAR_TABLES_230E50 is the pit stop's waypoints (the stop 0xe384 steers the car to them, 0xe744): per course id a
    u32[8], one stall per car ident (car +0x4 & 7, a linked cabinet's), each PIT_WAYPOINTS records of x, z, speed f32
    and a 0 word. Beginner's stall 0 (a single cabinet's): in at 150 along z -62, over to -59 at 60, the stop at
    242.36 (speed 0), then the lane's end; the stalls about 21 units apart. Id 3 (the stub) repeats Advanced's. On a
    grown set every id reads the grown table, ids 3-15 Beginner's stalls."""
    from dtools.arcade import course as C
    rs = RomSet.from_zip(romset_zip)
    b = C.pit_stalls(rs, 0)
    assert len(b) == C.PIT_STALLS == 8 and {len(s) for s in b} == {C.PIT_WAYPOINTS} == {7}
    assert [(round(w.x, 2), round(w.z, 2), w.speed) for w in b[0]] == [
        (92.36, -62.0, 150.0), (162.36, -62.0, 100.0), (182.36, -62.0, 60.0), (202.36, -59.0, 60.0),
        (212.36, -59.0, 20.0), (242.36, -59.0, 0.0), (300.0, -59.0, 0.0)]
    assert [round(s[0].x - b[k + 1][0].x, 1) for k, s in enumerate(b[:-1])] == [20.7, 20.7, 21.5, 19.9, 22.4, 19.2, 21.2]
    assert C.pit_stalls(rs, 3) == C.pit_stalls(rs, 2) != b
    assert C.pit_stalls(rs, 1)[0][0].x == pytest.approx(-480.61)
    g = C.grow_ids(rs)
    assert all(C.pit_stalls(g, k) == C.pit_stalls(rs, k) for k in range(3))
    assert all(C.pit_stalls(g, k) == b for k in range(3, C.COURSE_IDS))


def test_pit_stalls_round_trip():
    """encode_pit_stalls at `addr`: the u32[8] stall pointers, then the stalls back to back (16 bytes a waypoint: x, z,
    speed f32, 0); pit_stalls reads them back through the id's entry. Refused: a wrong count, a value not finite or
    not single precision, an address not 4-aligned."""
    from dtools.arcade import course as C
    stalls = tuple(tuple(C.PitWaypoint(float(k), -float(j), 150.0 - 25 * j) for j in range(7)) for k in range(8))
    addr = DATA + 0x10000
    blob = C.encode_pit_stalls(stalls, addr)
    assert len(blob) == C.PIT_STALLS_SIZE == 32 + 8 * 7 * 16
    assert struct.unpack_from("<8I", blob) == tuple(addr + 32 + 112 * k for k in range(8))
    assert struct.unpack_from("<3fI", blob, 32 + 112 + 16) == (1.0, -1.0, 125.0, 0)
    program, data = bytearray(0x40000), bytearray(0x20000)
    data[0x10000:0x10000 + len(blob)] = blob
    table = C.ID_ARRAYS["CAR_TABLES_230E50"].old
    struct.pack_into("<I", program, table - 0x00220000 + 0x20000 + 4 * 2, addr)
    assert C.pit_stalls(_rs(program, data), 2) == stalls
    with pytest.raises(C.CourseError, match="8 stalls"):
        C.encode_pit_stalls(stalls[:7], addr)
    with pytest.raises(C.CourseError, match="7 waypoints"):
        C.encode_pit_stalls((stalls[0][:6],) + stalls[1:], addr)
    bad = ((C.PitWaypoint(float("inf"), 0.0, 0.0),) + stalls[0][1:],) + stalls[1:]
    with pytest.raises(C.CourseError, match="finite"):
        C.encode_pit_stalls(bad, addr)
    bad = ((C.PitWaypoint(0.1, 0.0, 0.0),) + stalls[0][1:],) + stalls[1:]
    with pytest.raises(C.CourseError, match="single precision"):
        C.encode_pit_stalls(bad, addr)
    with pytest.raises(C.CourseError, match="aligned"):
        C.encode_pit_stalls(stalls, addr + 2)
