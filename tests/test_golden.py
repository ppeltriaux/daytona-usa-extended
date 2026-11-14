"""The golden test of milestone M1 (dtools/arcade/golden.py): every course table decoded into one Course and
re-emitted byte for byte, plus the encoders' guards (each refuses a value one past its field)."""

import dataclasses
import struct

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade import polyrom as P
from dtools.arcade import texture as T
from dtools.arcade import tgp
from dtools.arcade import tiles as TL
from dtools.arcade.romset import RomSet
from dtools.arcade.tablebase import CourseError

DATA = 0x02000000
LIB = C.OBJECT_LIBRARY[0]


# --- encoder guards: one value past the field's width, per encoder -------------------------------------------------

def _entry(**kw):
    e = dict(course=0, upload_list=0x74240, step_table=0, block_ref_a=0x7E4, block_ref_b=0x7E8, line_table=0x6000A0,
             record_table=0x7EC, raw=(0, 0))
    e.update(kw)
    return C.CourseEntry(**e)


def _place(**kw):
    p = dict(oba=0x800004, obc=C.OBC, tpa=0, tha=0, raw=(0,), cell=0, entry=0, in_world=True)
    p.update(kw)
    return C.Placement(**p)


def _actor(**kw):
    a = dict(slot=0, addr=0x234010, position=(0.0, 0.0, 0.0), angle=0, ident=0, handler=C.STATIC_PROP, obj=LIB)
    a.update(kw)
    return C.Actor(**a)


_LANE = C.Lane((0.0, 0.0, 0.0), 0, 0, 0, 0)


def _road(**kw):
    p = dict(lane_lo=1, lane_hi=6, curve=0, lanes=(_LANE,) * C.LANES)
    p.update(kw)
    return C.RoadPoint(**p)


_AI = C.AiPoint((0.0, 0.0, 0.0), 1.0, 0, 0, 0, 0, 0, 0)


def _slot(**kw):
    s = dict(car=0, objects=(LIB,) * 8, line=0, start=0, speed=1.0, kind=0, segment=0, lane=0, position=(0, 0, 0))
    s.update(kw)
    return C.GridSlot(**s)


def _cp(**kw):
    c = dict(index=0, segment=0, raw=0, lap_line=False, position=(0, 0, 0), extensions=())
    c.update(kw)
    return C.Checkpoint(**c)


def _panel(**kw):
    p = dict(panel=0, course=0, cursor=(95, 396), tile=(3, 9), maps=(DATA,) * 6)
    p.update(kw)
    return C.SelectPanel(**p)


def _hdr(**kw):
    h = dict(tha=0x100, step=0, width=64, height=32, tex_x=64, tex_y=96, sheet=0, textured=True, translucent=False,
             checker=False, wrap_x=False, wrap_y=False, mirror_x=False, mirror_y=False, utex=False, utex_min_lod=0,
             utex_x=0, utex_y=0, luma_base=3, colour=0x229, raw=(0, 1, 0))
    h.update(kw)
    return T.TexHeader(**h)


_PANELS = (_panel(panel=0, course=0), _panel(panel=1, course=2), _panel(panel=2, course=1))
_STEPS = [C.Step(0, C.STEP_ROAD), C.Step(1, C.STEP_ROAD), C.Step(1, 0)]
_PIT = C.Pit(1, 1, 1, (), 2)
_OBJ = P.Object((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), [P.Record(1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))],
                20)


def _hud(**kw):
    h = dict(course=0, outline_list=LIB, entries=(_place(),), objects={4: _OBJ}, position=(0.0, 0.0, 9e-5), angle=0,
             pad=0, zooms=(5e-8,) * C.HUD_MAP_ZOOM_LEVELS)
    h.update(kw)
    return C.HudMap(**h)

# (encoder, call with every field fitting, call with one field one past its width, the error class)
GUARDS = [
    ("encode_course_entry", lambda: C.encode_course_entry(_entry()),
     lambda: C.encode_course_entry(_entry(step_table=C.DATA_WINDOW)), CourseError),
    ("encode_placement", lambda: C.encode_placement(_place()),
     lambda: C.encode_placement(_place(tpa=1 << 32)), CourseError),
    ("encode_actor", lambda: C.encode_actor(_actor()), lambda: C.encode_actor(_actor(angle=0x8000)), CourseError),
    ("encode_actor ident", lambda: C.encode_actor(_actor(ident=0xFFFF)),
     lambda: C.encode_actor(_actor(ident=0x10000)), CourseError),
    ("encode_road_point", lambda: C.encode_road_point(_road(curve=0x7FFF)),
     lambda: C.encode_road_point(_road(curve=0x8000)), CourseError),
    ("encode_road_point lanes", lambda: C.encode_road_point(_road(lane_hi=7)),
     lambda: C.encode_road_point(_road(lane_hi=8)), CourseError),
    ("encode_road_line", lambda: C.encode_road_line([_road()] * 0xFFFF),
     lambda: C.encode_road_line([_road()] * 0x10000), CourseError),
    ("encode_ai_point", lambda: C.encode_ai_point(_AI._replace(step=0xFFFF)),
     lambda: C.encode_ai_point(_AI._replace(step=0x10000)), CourseError),
    ("encode_ai_line", lambda: C.encode_ai_line([_AI] * 0xFFFF),
     lambda: C.encode_ai_line([_AI] * 0x10000), CourseError),
    ("encode_line_table", lambda: C.encode_line_table(0x600000, [[_AI] * 8] * 8),
     lambda: C.encode_line_table(C.DATA_WINDOW - 8 * 8 - 8 * 16 * 28 + 1, [[_AI] * 8] * 8), CourseError),
    ("encode_checkpoint_table", lambda: C.encode_checkpoint_table([_cp(segment=0xFFFF)]),
     lambda: C.encode_checkpoint_table([_cp(segment=0x10000)]), CourseError),
    ("encode_checkpoint_table count", lambda: C.encode_checkpoint_table([_cp()] * C.MAX_CHECKPOINTS),
     lambda: C.encode_checkpoint_table([_cp()] * (C.MAX_CHECKPOINTS + 1)), CourseError),
    ("encode_time_table", lambda: C.encode_time_table([0xFFFFFFFF]),
     lambda: C.encode_time_table([1 << 32]), CourseError),
    ("encode_spawn_record", lambda: C.encode_spawn_record(_slot(start=0xFFFF, kind=0xFF)),
     lambda: C.encode_spawn_record(_slot(start=0x10000)), CourseError),
    ("encode_spawn_record kind", lambda: C.encode_spawn_record(_slot(kind=0xFF)),
     lambda: C.encode_spawn_record(_slot(kind=0x100)), CourseError),
    ("encode_laps", lambda: C.encode_laps(C.MAX_LAPS), lambda: C.encode_laps(C.MAX_LAPS + 1), CourseError),
    ("encode_car_count", lambda: C.encode_car_count(C.SPAWN_RECORDS),
     lambda: C.encode_car_count(C.SPAWN_RECORDS + 1), CourseError),
    ("encode_table_pointer", lambda: C.encode_table_pointer(DATA + C.DATA_WINDOW - 4),
     lambda: C.encode_table_pointer(DATA + C.DATA_WINDOW), CourseError),
    ("encode_grid_base", lambda: C.encode_grid_base(0x7FFF), lambda: C.encode_grid_base(0x8000), CourseError),
    ("encode_segment_flags", lambda: C.encode_segment_flags([0xFFFF]), lambda: C.encode_segment_flags([0x10000]),
     CourseError),
    ("encode_segment_flags count", lambda: C.encode_segment_flags([0] * 0xFFFF),
     lambda: C.encode_segment_flags([0] * 0x10000), CourseError),
    ("encode_step_table", lambda: C.encode_step_table(1, [C.Step(1, 0xFFFF)]),
     lambda: C.encode_step_table(1, [C.Step(1, 0x10000)]), CourseError),
    ("encode_step_table segment", lambda: C.encode_step_table(1, [C.Step(1, 1)]),
     lambda: C.encode_step_table(1, [C.Step(2, 1)]), CourseError),
    ("encode_pit", lambda: C.encode_pit(1, _STEPS, _PIT),
     lambda: C.encode_pit(1, _STEPS, _PIT._replace(box_polygon=len(_STEPS))), CourseError),
    ("encode_rolling_start", lambda: C.encode_rolling_start(True), lambda: C.encode_rolling_start(1), CourseError),
    ("encode_wheel_zones", lambda: C.encode_wheel_zones(C.WheelZones(31, 0x7C000000)),
     lambda: C.encode_wheel_zones(C.WheelZones(32, 108)), CourseError),
    ("encode_wheel_zones hi", lambda: C.encode_wheel_zones(C.WheelZones(0, 0x7C000000)),
     lambda: C.encode_wheel_zones(C.WheelZones(0, 0x80000000)), CourseError),
    ("encode_select_panel", lambda: C.encode_select_panel(_panel(cursor=(0xFFF, 0))),
     lambda: C.encode_select_panel(_panel(cursor=(0x1000, 0))), CourseError),
    ("encode_select_panel course", lambda: C.encode_select_panel(_panel(course=0xFF)),
     lambda: C.encode_select_panel(_panel(course=0x100)), CourseError),
    ("encode_map_object", lambda: C.encode_map_object(C.MapObject(0, (0, 0, 0), LIB, (0xFFFFFFFF, 0, 0, 0))),
     lambda: C.encode_map_object(C.MapObject(0, (0, 0, 0), LIB, (1 << 32, 0, 0, 0))), CourseError),
    ("encode_hud_map", lambda: C.encode_hud_map(_hud(angle=0x7FFF, pad=0xFFFF)),
     lambda: C.encode_hud_map(_hud(pad=0x10000)), CourseError),
    ("encode_hud_map angle", lambda: C.encode_hud_map(_hud(angle=-0x8000)),
     lambda: C.encode_hud_map(_hud(angle=-0x8001)), CourseError),
    ("encode_panel_of_course", lambda: C.encode_panel_of_course((0, 2, 1), _PANELS),
     lambda: C.encode_panel_of_course((0, 3, 1), _PANELS), CourseError),
    ("encode_ranking_pointers", lambda: C.encode_ranking_pointers((C.SRAM[1] - 576,) * 3),
     lambda: C.encode_ranking_pointers((C.SRAM[1] - 572,) * 3), CourseError),
    ("encode_cell_grid", lambda: C.encode_cell_grid([C.OBJECT_LIBRARY[1] - 4] * C.GRID_CELLS),
     lambda: C.encode_cell_grid([C.OBJECT_LIBRARY[1]] * C.GRID_CELLS), CourseError),
    ("encode_cell_grid count", lambda: C.encode_cell_grid([LIB] * C.GRID_CELLS),
     lambda: C.encode_cell_grid([LIB] * (C.GRID_CELLS + 1)), CourseError),
    ("encode_object_list", lambda: C.encode_object_list([_place(tha=0xFFFFFFFF)]),
     lambda: C.encode_object_list([_place(tha=1 << 32)]), CourseError),
    ("encode_actor_table", lambda: C.encode_actor_table([0x0023FFF8] * C.GRID_CELLS),
     lambda: C.encode_actor_table([0x0023FFFC] * C.GRID_CELLS), CourseError),
    ("encode_actor_list", lambda: C.encode_actor_list([_actor(ident=0xFFFF)]),
     lambda: C.encode_actor_list([_actor(ident=0x10000)]), CourseError),
    ("encode_billboard_lists", lambda: C.encode_billboard_lists([LIB] * C.BILLBOARD_VARIANTS),
     lambda: C.encode_billboard_lists([LIB] * (C.BILLBOARD_VARIANTS + 1)), CourseError),
    ("encode_billboard_ref", lambda: C.encode_billboard_ref((3.4e38, 0.0, -380.0)),
     lambda: C.encode_billboard_ref((3.5e38, 0.0, -380.0)), CourseError),
    ("encode_switch_lists", lambda: C.encode_switch_lists([C.OBJECT_LIBRARY[1] - 4] * 3),
     lambda: C.encode_switch_lists([C.OBJECT_LIBRARY[1]] * 3), CourseError),
    ("encode_grid_lanes", lambda: C.encode_grid_lanes((0xFFFF, 0)), lambda: C.encode_grid_lanes((0x10000, 0)),
     CourseError),
    ("encode_tile_map", lambda: TL.encode_tile_map(TL.TileMap(0, False, 1, 1, (0x7F,))),
     lambda: TL.encode_tile_map(TL.TileMap(0, False, 1, 1, (0x80,))), CourseError),
    ("encode_tile_map base", lambda: TL.encode_tile_map(TL.TileMap(0x7FFF, True, 1, 1, (0,))),
     lambda: TL.encode_tile_map(TL.TileMap(0x8000, True, 1, 1, (0,))), CourseError),
    ("encode_chars", lambda: TL.encode_chars(np.full((1, 8, 8), 15)),
     lambda: TL.encode_chars(np.full((1, 8, 8), 16)), CourseError),
    ("encode_char_block", lambda: TL.encode_char_block(b"\0" * TL.CHAR_SIZE * TL.CHAR_COUNT),
     lambda: TL.encode_char_block(b"\0" * TL.CHAR_SIZE * (TL.CHAR_COUNT + 1)), CourseError),
    ("encode_char_list", lambda: TL.encode_char_list([(DATA, TL.CHAR_RAM[1] - TL.CHAR_SIZE)]),
     lambda: TL.encode_char_list([(DATA, TL.CHAR_RAM[1])]), CourseError),
    ("encode_palette_list", lambda: TL.encode_palette_list([TL.PaletteUpload(0x01800000, (0xFFFF, 0))]),
     lambda: TL.encode_palette_list([TL.PaletteUpload(0x01800000, (0x10000, 0))]), CourseError),
    ("encode_upload_set", lambda: TL.encode_upload_set(DATA + C.DATA_WINDOW - 4, DATA),
     lambda: TL.encode_upload_set(DATA + C.DATA_WINDOW - 3, DATA), CourseError),
    ("encode_backdrop_segment", lambda: TL.encode_backdrop_segment(TL.BackdropSegment(0xFFFFFFFF, 1, 32, (0,) * 32)),
     lambda: TL.encode_backdrop_segment(TL.BackdropSegment(1 << 32, 1, 32, (0,) * 32)), CourseError),
    ("encode_tex_header", lambda: T.encode_tex_header(_hdr(colour=0x3FF)),
     lambda: T.encode_tex_header(_hdr(colour=0x400)), T.TextureError),
    ("encode_texels", lambda: T.encode_texels(_hdr(), np.full((32, 64), 15)),
     lambda: T.encode_texels(_hdr(), np.full((32, 64), 16)), T.TextureError),
    ("encode_mip_bytes", lambda: T.encode_mip_bytes(_hdr(), 1, np.full((16, 32), 15), 0x200000),
     lambda: T.encode_mip_bytes(_hdr(), 1, np.full((16, 32), 16), 0x200000), T.TextureError),
    ("encode_texture_points", lambda: T.encode_texture_points((((0xFFFF, 0),) * 4,)),
     lambda: T.encode_texture_points((((0x10000, 0),) * 4,)), T.TextureError),
    ("encode_sheet_source", lambda: T.encode_sheet_source(C.DATA_WINDOW - T.SHEET_BLOCK),
     lambda: T.encode_sheet_source(C.DATA_WINDOW - T.SHEET_BLOCK + 1), T.TextureError),
    ("encode_object", lambda: P.encode_object(_OBJ), lambda: P.encode_object(
        P.Object(_OBJ.seed0, _OBJ.seed1, [P.Record(1 << 32, (0.0,) * 3, (0.0,) * 3, (0.0,) * 3)], 20)), P.PolyError),
]


@pytest.mark.parametrize("name, fits, past, error", GUARDS, ids=[g[0] for g in GUARDS])
def test_encode_rejects_oversized_fields(name, fits, past, error):
    fits()                                       # the largest value that fits is accepted ...
    with pytest.raises(error):                   # ... one more is refused
        past()


def test_every_encoder_has_a_guard_case():
    """No encoder of the course modules is left out of GUARDS."""
    names = {g[0].split()[0] for g in GUARDS}
    for mod in (C, TL, T, P):
        for n in dir(mod):
            if n.startswith("encode_") and callable(getattr(mod, n)):
                assert n in names, f"{mod.__name__}.{n} has no guard case"


# --- Course, encode_course, mismatches: synthetic --------------------------------------------------------------------

def _synthetic():
    """A set with course 1's record table (road line) and line table, and its course record and pointer."""
    from tests.test_course import LINES_AT, ROAD_AT, _lines_rs
    rs, _ = _lines_rs()
    entry = C.CourseEntry(1, 0, 0, 0, 0, LINES_AT, ROAD_AT, (0, 0))
    program, data = bytearray(rs.program), bytearray(rs.data)
    struct.pack_into("<I", program, C.COURSE_POINTERS + 4, C.COURSE_TABLE + C.COURSE_ENTRY_SIZE)
    rec = C.encode_course_entry(entry)
    at = C.COURSE_TABLE - DATA + C.COURSE_ENTRY_SIZE
    data[at:at + len(rec)] = rec
    rs = RomSet(program=bytes(program), data=bytes(data), copro_data=b"", polygons=b"", textures=b"", extra={})
    return rs, G.Course(course=1, entry=entry, record_pointer=C.COURSE_TABLE + C.COURSE_ENTRY_SIZE,
                        road=tuple(C.road_line(rs, entry)), ai_lines=tuple(map(tuple, C.ai_lines(rs, entry))))


def test_encode_course_synthetic():
    rs, course = _synthetic()
    out = G._encode(course)          # a partial Course: the tables that are set (encode_course refuses it)
    assert set(out) == {("program", C.COURSE_POINTERS + 4), ("data", C.COURSE_TABLE - DATA + C.COURSE_ENTRY_SIZE),
                        ("data", course.entry.record_table), ("data", course.entry.line_table)}
    assert G.mismatches(rs, course) == []
    names = {n for _, _, _, n in G.tables(course)}
    assert names == {"course pointer", "course record", "road line", "line table"}


def test_mismatches_name_the_table_and_offset():
    rs, course = _synthetic()
    road = list(course.road)
    road[3] = dataclasses.replace(road[3], curve=road[3].curve + 1)
    bad = dataclasses.replace(course, road=tuple(road))
    got = G.mismatches(rs, bad)
    first = course.entry.record_table + 8 + C.ROAD_POINT_SIZE * 3 + 2      # the curve field of record 3
    assert len(got) == 1 and "road line" in got[0] and f"data+{course.entry.record_table:#x}" in got[0]
    assert f"first at data+{first:#x}" in got[0]


def test_encode_course_refuses_conflicting_entries():
    rs, course = _synthetic()
    clash = dataclasses.replace(course, entry=dataclasses.replace(course.entry, record_table=course.entry.line_table))
    with pytest.raises(CourseError, match="line table.*road line|road line.*line table"):
        G._encode(clash)


def test_coverage_union():
    cov = [("data", 0, 10, "a"), ("data", 5, 10, "b"), ("data", 20, 4, "c"), ("program", 0, 4, "d")]
    assert G.covered_bytes(cov) == {"data": 19, "program": 4}


# --- the golden test: the three real courses ---------------------------------------------------------------------

GROUND_TABLES = frozenset({"ground grid", "ground attributes", "ground slot: list block address",
                           "ground slot: polygon pool address", "ground lists", "ground polygons"})


def _covers(rs, course: int):
    """verify_lossless, coverage and excluded_spans of `course` on `rs`: lossless, every always-covered table and the
    ground (tgp.ground_spans, exactly) covered, and what is not covered excluded by name, coverage not reaching into
    it where its extent is known. Returns the coverage."""
    assert G.verify_lossless(rs, course) == []
    cov = G.coverage(rs, course)
    names = {n for _, _, _, n in cov}
    assert names >= G.ALWAYS_COVERED, sorted(G.ALWAYS_COVERED - names)
    assert names >= GROUND_TABLES
    assert sorted(t for t in cov if t[3] in GROUND_TABLES) == sorted(tgp.ground_spans(rs, course))
    excluded = G.excluded_spans(rs, course)
    assert {n for *_, n in excluded} == set(G.EXCLUDED) - G.NO_EXTENT
    assert not names & set(G.EXCLUDED)
    clash = G.overlaps(cov, excluded)
    assert not clash, clash[:5]
    return cov


def _golden(romset_zip, course: int):
    rs = RomSet.from_zip(romset_zip)
    cov = _covers(rs, course)
    total = G.covered_bytes(cov)
    by_table: dict[str, int] = {}
    for region, off, n, name in cov:
        by_table[name] = by_table.get(name, 0) + n
    print(f"\n{C.COURSE_NAMES[course]}: covered bytes per region {total}")
    for name in sorted(by_table):
        print(f"  {name:28s} {by_table[name]:>10,}")
    return total


def test_beginner_round_trip(romset_zip):
    """Golden (M1): every table of Beginner (id 0) decodes into one Course and re-emits byte for byte, structures
    included (cell grid and list terminators, actor tables, lists and counts, line-table pairs, upload sets and their
    lists, panel records and maps, pointer words). Not covered (golden.EXCLUDED, each excluded by name):

    The ground is covered (M1b): the grid (*block_ref_a, 32 KB), the attribute block (*block_ref_b, padding
    included) and, in copro_data, the slot's two words, the list block and the polygon pool (tgp.ground_spans).

    - the handler data (`obj`) of actors whose handlers are not modelled ("Scenery placement / Not found / not
      modelled"; their 24-byte records are covered);
    - the select screen's 3D map objects in polygon ROM (geometry mode on Circuit Select not traced; the map object
      records and their object lists are covered);
    - the time corrections (0x280587c, 0x280583c), Advanced's progress mask and visit orders, the view-test tables
      (0x286f7e0-): not decoded (the per-segment flags are covered since Phase 6: test_golden_covers_segment_flags);
    - the upper levels of TIME_TABLES, LAP_COUNTS and CAR_COUNTS and CAR_COUNTS entry 1: fixed pointer arrays,
      not re-emitted (docs/formats/model2-course-tables.md, "Writing these values");
    - spawn records past the course's car count; the other course ids (the id-3 stub: record, select records, array
      entries); the per-id arrays no decoder reads (those outside golden.DECODED_ARRAYS);
    - the luma tables and polygon palette (colour_tables decodes them, no encoder; shared by all courses);
    - the rankings' SRAM contents, RANKING_LAST, RANKING_DEFAULTS and the default time bases (a layout only).
    """
    total = _golden(romset_zip, 0)
    assert total == G_EXPECTED[0]


def test_expert_round_trip(romset_zip):
    """As test_beginner_round_trip, for Expert (id 1); the same exclusions."""
    assert _golden(romset_zip, 1) == G_EXPECTED[1]


def test_advanced_round_trip(romset_zip):
    """As test_beginner_round_trip, for Advanced (id 2); the same exclusions."""
    assert _golden(romset_zip, 2) == G_EXPECTED[2]


G_EXPECTED = {           # bytes covered per region (the union of `coverage`), pinned from the first green run
    # Phase 2 (ground excluded): data 1,812,904 / 2,270,206 / 1,986,616, no copro_data. The ground adds the grid
    # (32,768) and the attribute block (1,184 / 7,240 / 2,988) to data, and copro_data: the two slot words, the list
    # block (18,856 / 88,296 / 41,160) and the pool (75,776 / 463,296 / 191,104).
    # Phase 4 (M1c) adds the HUD map: in data its list pointer, list, place record and zooms (4 + 20 + 16 + 16 = 56),
    # in polygons the outline object (260 / 1,460 / 1,070 words), in textures the outline's headers (24 / 133 / 98),
    # and Beginner's billboard reference point (BILLBOARD_REF, 12 program bytes).
    # Phase 6 (M3a) adds the per-segment flags: the pointer word and the table, one u16 per road segment (4 + 2 x
    # 196 / 991 / 487).
    # Phase 7 (M3b) adds the texture points: 4 bytes per vertex of every record of every placement decoded (the HUD
    # outline's included), 451,572 / 690,192 / 440,908 bytes, whose union is 451,068 / 689,804 / 440,468 (the car
    # models' placements read overlapping runs, word indices 0x16a276-0x16a278 and 0x5a9032-0x5a904e, which
    # Beginner's and Advanced's HUD outlines also overlap).
    0: {"copro_data": 8 + 18856 + 75776, "data": 1812904 + 32768 + 1184 + 56 + 4 + 392,
        "polygons": 1168720 + 4 * 260, "program": 2934 + 12, "textures": 193616 + 8 * 24 + 451068},
    1: {"copro_data": 8 + 88296 + 463296, "data": 2270206 + 32768 + 7240 + 56 + 4 + 1982,
        "polygons": 1762200 + 4 * 1460, "program": 3346, "textures": 297224 + 8 * 133 + 689804},
    2: {"copro_data": 8 + 41160 + 191104, "data": 1986616 + 32768 + 2988 + 56 + 4 + 974,
        "polygons": 1181800 + 4 * 1070, "program": 2586, "textures": 205424 + 8 * 98 + 440468},
}


def test_verify_lossless_sees_every_kind_of_table(romset_zip):
    """verify_lossless is not vacuous on the real set: one changed value in each kind of table of a decoded Course
    (entries, structures, geometry, textures, uploads) is reported, by that table's name."""
    rs = RomSet.from_zip(romset_zip)
    c = G.decode_course(rs, 0)
    rep = dataclasses.replace
    obj_at, obj = next(iter(c.objects.items()))
    hdr_at, hdr = next(iter(c.headers.items()))
    tx = c.texels[0]
    idx = tx.indices.copy()
    idx[0, 0] ^= 1
    mip = tx.mips[0].copy()
    mip[0, 0] ^= 1
    blk_at, blk = next(iter(c.char_blocks.items()))
    px = blk.copy()
    px[0, 0, 0] ^= 1
    map_at, m = next(iter(c.tile_maps.items()))
    lst_at, lst = next(iter(c.object_lists.items()))
    act_at, acts = next(iter(c.actor_lists.items()))
    key, (ptr, table, words) = next(iter(c.time_tables.items()))
    cells = list(c.cells)
    cells[0], cells[1] = cells[1], cells[0]
    slots = list(c.actor_slots)
    slots[slots.index(next(x for x in slots if x))] = 0
    g = c.ground_data
    lk, lv = next((k, v) for k, v in g.lists.items() if len(v) > 1 and v != v[::-1])
    assert g.grid[0] == 0
    pt_at, pts = next(iter(c.points.items()))
    cases = {
        "texture points": rep(c, points={**c.points, pt_at: (tuple((v ^ 8, u) for v, u in pts[0]),) + pts[1:]}),
        "cell grid": rep(c, cells=tuple(cells)),
        "actor table": rep(c, actor_slots=tuple(slots)),
        "actor list": rep(c, actor_lists={**c.actor_lists, act_at: acts[:-1]}),
        "object list": rep(c, object_lists={**c.object_lists, lst_at: (rep(lst[0], raw=(7,)),) + lst[1:]}),
        "polygon object": rep(c, objects={**c.objects, obj_at: rep(obj, records=obj.records[:-1],
                                                                     word_count=obj.word_count - 10)}),
        "texture header": rep(c, headers={**c.headers, hdr_at: rep(hdr, colour=hdr.colour ^ 1)}),
        "texels": rep(c, texels=(tx._replace(indices=idx),) + c.texels[1:]),
        "texels (mips)": rep(c, texels=(tx._replace(mips=(mip,) + tx.mips[1:]),) + c.texels[1:]),
        "char block": rep(c, char_blocks={**c.char_blocks, blk_at: px}),
        "tile map": rep(c, tile_maps={**c.tile_maps, map_at: m._replace(base=m.base + 1)}),
        "segment flags": rep(c, segment_flags=(c.segment_flags[0], (1,) + c.segment_flags[1][1:])),
        "segment flags pointer": rep(c, segment_flags=(c.segment_flags[0] + 2, c.segment_flags[1][:-1] + (0,))),
        "step table": rep(c, steps=(c.steps[0]._replace(flags=c.steps[0].flags ^ 0x100),) + c.steps[1:]),
        "line table": rep(c, ai_lines=((c.ai_lines[0][0]._replace(steer=1234),) + c.ai_lines[0][1:],)
                          + c.ai_lines[1:]),
        "spawn record": rep(c, spawn=((c.spawn[0][0]._replace(kind=9),) + c.spawn[0][1:],) + c.spawn[1:]),
        "time table": rep(c, time_tables={**c.time_tables, key: (ptr, table, (words[0] + 1,) + words[1:])}),
        "backdrop segment": rep(c, backdrop=c.backdrop._replace(segments=(c.backdrop.segments[0]._replace(
            flags=0),) + c.backdrop.segments[1:])),
        "ground grid": rep(c, ground_data=rep(g, grid=(lk,) + g.grid[1:])),       # cell 0 is off the ground
        "ground attributes": rep(c, ground_data=rep(g, attributes=g.attributes[:-1] + bytes([g.attributes[-1] ^ 4]))),
        "ground lists": rep(c, ground_data=rep(g, lists={**g.lists, lk: lv[::-1]})),
        "ground polygons": rep(c, ground_data=rep(g, polys=(rep(g.polys[0], direction=g.polys[0].direction + 1),)
                                                   + g.polys[1:])),
    }
    for name, bad in cases.items():
        got = G.mismatches(rs, bad)
        assert got and all(f": {name} at " in line for line in got), (name, got)
    assert G.mismatches(rs, c) == []


def test_overlaps():
    a = [("data", 0, 4, "a"), ("data", 10, 2, "b"), ("program", 0, 4, "c")]
    b = [("data", 3, 1, "x"), ("data", 4, 6, "y"), ("data", 11, 50, "z"), ("polygons", 0, 9, "w")]
    assert sorted(G.overlaps(a, b)) == [(a[0], b[0]), (a[1], b[2])]


# --- check_course: the rules between tables, and the views ---------------------------------------------------------

_POOL_AT = 0x02802F74           # data ROM: the spawner's actor count (course.ACTOR_POOL)


def _race():
    """A synthetic Course (id 1) with every race table set and consistent: 10 road points, 8 AI lines, 30 steps with
    a pit, one checkpoint, a 2-lap time table, two cars in both spawn tables, three actors."""
    _, base = _synthetic()
    road = list(base.road)
    n = len(road) - 1
    steps = [C.Step(i % (n + 1), C.STEP_ROAD) for i in range(26)] + \
        [C.Step(6, C.PIT_LANE_FLAGS), C.Step(7, C.PIT_BOX_FLAGS), C.Step(8, C.PIT_LANE_FLAGS), C.Step(5, 0)]
    pit = C._pit_of(steps, n, "pit")
    rows, laps = [(4, 0)], 2
    words = tuple(range(100, 100 + 1 + laps * (len(rows) + 1)))
    lanes = (0x180, 0x580)
    slots = tuple(C.grid_slot(car, (LIB,) * 8, car, 4 * car, 1.0, 0, road, 3, lanes, False) for car in range(2))
    return dataclasses.replace(
        base, step_n=n, steps=tuple(steps), pit=pit, checkpoint_table=DATA + 0x823C20,
        checkpoints=tuple(C.checkpoint_views(rows, road, words, laps)),
        time_tables={(C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY): (DATA + 0x805300, DATA + 0x820B50, words)},
        laps={C.DEFAULT_MODE: (DATA + 0x805400, laps)}, car_count=(DATA + 0x805500, 2),
        rolling_start=(C.ROLLING_STARTS + 2, False), grid_base=(C.GRID_BASES + 2, 3), grid_lanes=lanes,
        segment_flags=(DATA + 0x805A54, (0,) * 5 + (C.SEGMENT_AI_FREE,) * 5), spawn=(slots, slots),
        actor_table=0x00233C0C, actor_slots=(0x00234100,) + (0,) * (C.GRID_CELLS - 1),
        actor_lists={0x00234100: (_actor(),) * 3})


def test_check_course_accepts_a_consistent_course():
    c = _race()
    G._check_rules(c)               # _race() is partial: the rules apply to the tables that are set
    assert ("data", c.entry.step_table) in G._encode(c)


def _cp_edit(c, k, **kw):
    cps = list(c.checkpoints)
    cps[k] = cps[k]._replace(**kw)
    return dataclasses.replace(c, checkpoints=tuple(cps))


def _slot_edit(c, car, **kw):
    t = list(c.spawn[1])
    t[car] = t[car]._replace(**kw)
    return dataclasses.replace(c, spawn=(c.spawn[0], tuple(t)))


def _time_words(c, words):
    key, (a, t, _) = next(iter(c.time_tables.items()))
    return dataclasses.replace(c, time_tables={key: (a, t, tuple(words))})


# (case, mutation of _race(), the rule the error must name)
RULES = [
    ("AI line one point short", lambda c: dataclasses.replace(c, ai_lines=tuple(l[:-1] for l in c.ai_lines)),
     r"AI lines.*9 points.*10"),
    ("road line cut", lambda c: dataclasses.replace(c, road=c.road[:5]), r"AI lines.*road line 5"),
    ("step_n is not the road's n", lambda c: dataclasses.replace(c, step_n=8, steps=tuple(
        s._replace(segment=min(s.segment, 8)) for s in c.steps), pit=None), r"step table.*step_n 8.*n 9"),
    ("AI step past the steps", lambda c: dataclasses.replace(c, steps=c.steps[:20], pit=None),
     r"AI lines.*step 2\d.*20 steps"),
    ("checkpoint past the road", lambda c: _cp_edit(c, 0, segment=10), r"checkpoints.*segment 10.*n 9"),
    ("time table too short", lambda c: _time_words(c, c.time_tables[(0, 1)][2][:4]),
     r"time table.*4 words.*5.*laps 2.*checkpoints 1"),
    ("car count past the spawn records", lambda c: dataclasses.replace(c, car_count=(c.car_count[0], 3)),
     r"car count 3.*spawn table 0.*\[2\]"),
    ("spawn line past the AI lines", lambda c: _slot_edit(c, 1, line=8), r"spawn.*line 8"),
    ("spawn car twice", lambda c: dataclasses.replace(c, spawn=(c.spawn[0][:1] * 2, c.spawn[1])),
     r"spawn table 0.*car 0 twice|twice"),
    ("actor pool", lambda c: dataclasses.replace(c, actor_lists={0x00234100: (_actor(),) * 66}),
     r"actors.*66.*65"),
    ("segment flags one short", lambda c: dataclasses.replace(c, segment_flags=(DATA + 0x805A54, (0,) * 9)),
     r"segment flags.*9 values.*road line 10"),
    ("segment flags not a pair", lambda c: dataclasses.replace(c, segment_flags=(0,) * 10),
     r"segment flags.*not \(table address, flags\)"),
    ("segment flag not an integer", lambda c: dataclasses.replace(c, segment_flags=(DATA, (0,) * 9 + (True,))),
     r"segment flags.*not \(table address, flags\)"),
    ("record pointer moved", lambda c: dataclasses.replace(c, record_pointer=0x0265D120),
     r"record pointer.*0x265d120.*0x2600040"),
    ("pit edited", lambda c: dataclasses.replace(c, pit=c.pit._replace(box_segment=6)), r"pit.*encode_pit"),
    ("extension edited", lambda c: _cp_edit(c, 0, extensions=(1, 2)),
     r"checkpoint 0: extensions \(1, 2\).*\(101, 103\)"),
    ("phantom edited", lambda c: _cp_edit(c, 1, extensions=(102, 0)), r"checkpoint 1: extensions.*\(102, 104\)"),
    ("checkpoint position edited", lambda c: _cp_edit(c, 0, position=(0.0, 0.0, 0.0)), r"checkpoint 0: position"),
    ("lap line not last", lambda c: dataclasses.replace(c, checkpoints=c.checkpoints[::-1]), r"lap line"),
    ("lap line segment", lambda c: _cp_edit(c, 1, segment=3), r"lap line.*segment 3"),
    ("grid segment edited", lambda c: _slot_edit(c, 1, segment=0), r"spawn table 1 car 1: segment 0.*7"),
    ("grid lane edited", lambda c: _slot_edit(c, 0, lane=0x180), r"spawn table 1 car 0: lane 384 differs from 1408"),
    ("grid position edited", lambda c: _slot_edit(c, 0, position=(0, 0, 0)), r"spawn table 1 car 0: position"),
]


@pytest.mark.parametrize("case, mutate, match", RULES, ids=[r[0] for r in RULES])
def test_check_course_refuses_tables_that_disagree(case, mutate, match):
    bad = mutate(_race())
    with pytest.raises(CourseError, match=match):
        G._check_rules(bad)


# --- check_course: the ground (golden.Course.ground_data, tgp.Ground) ---------------------------------------------

_GRID_AT, _ATTR_AT, _REF_AT = 0x300000, 0x308000, 0x400000      # synthetic data-ROM offsets
_ROAD_POLYS = 30                                                 # _race()'s steps


def _sq(k: int) -> tgp.GroundPoly:
    """Square k (8 units) of a row along x, in the originals' winding (negative signed XZ area)."""
    x = 8.0 * k
    return tgp.GroundPoly(((x, 0.0, 0.0), (x, 0.0, 8.0), (x + 8, 0.0, 8.0), (x + 8, 0.0, 0.0)))


def _grounded():
    """_race() with a ground: 31 polygons, the first 30 (one per step) road (kind 1, one with a high nibble set), the
    last kind 2, all in one list at grid value 1 named by cell (64, 64); pointer words at _REF_AT, matching."""
    c = _race()
    grid = [0] * tgp.GRID_CELLS
    grid[64 * 128 + 64] = 1
    n = _ROAD_POLYS + 1
    g = tgp.Ground(course=1, grid=tuple(grid), attributes=bytes([0x21] + [1] * (_ROAD_POLYS - 1) + [2]),
                   polys=tuple(_sq(k) for k in range(n)), lists={1: tuple(range(n))}, base=1, raw=(),
                   grid_at=_GRID_AT, attributes_at=_ATTR_AT, list_base=0x30, pool_base=0x1000)
    entry = dataclasses.replace(c.entry, block_ref_a=_REF_AT, block_ref_b=_REF_AT + 4)
    return dataclasses.replace(c, entry=entry, ground=(DATA + _GRID_AT, DATA + _ATTR_AT), ground_data=g)


def test_check_course_accepts_a_consistent_ground():
    c = _grounded()
    assert len(c.steps) == _ROAD_POLYS
    G._check_rules(c)
    out = G._encode(c)
    assert out[("data", _ATTR_AT)] == c.ground_data.attributes + b"\xcd" * 1      # 31 bytes, padded to 32
    assert out[("data", _REF_AT)] == struct.pack("<I", DATA + _GRID_AT)
    assert out[("copro_data", 4 * (tgp.SLOT_LIST + 1))] == struct.pack("<I", 0x30)
    names = {n for *_, n in G.tables(c)}
    assert names >= GROUND_TABLES | {"ground grid pointer", "ground attribute pointer"}
    # a ground without the step table: the road rule is not applied
    G._check_rules(dataclasses.replace(c, step_n=None, steps=(), pit=None, ai_lines=()))


def _gd(c, **kw):
    return dataclasses.replace(c, ground_data=dataclasses.replace(c.ground_data, **kw))


def _cell_value(c, cell, v):
    grid = list(c.ground_data.grid)
    grid[cell] = v
    return _gd(c, grid=tuple(grid))


def _attr(c, k, v):
    a = bytearray(c.ground_data.attributes)
    a[k] = v
    return _gd(c, attributes=bytes(a))


# (case, mutation of _grounded(), the rule the error must name)
GROUND_RULES = [
    ("attribute byte missing", lambda c: _gd(c, attributes=c.ground_data.attributes[:-1]),
     r"ground: ground of course 1: attributes .* are not 31 bytes"),
    ("attribute padding stored", lambda c: _gd(c, attributes=c.ground_data.attributes + b"\xcd"),
     r"ground: ground of course 1: attributes .* are not 31 bytes"),
    ("attributes not bytes", lambda c: _gd(c, attributes=[1] * 31), r"ground: .*attributes \[1, 1.* are not 31 bytes"),
    ("polys not a sequence", lambda c: _gd(c, polys=None), r"ground: .*polys None is not a sequence"),
    ("road polygon not road", lambda c: _attr(c, 3, 2), r"ground: polygon 3 has kind 2.*first 30.*road"),
    ("non-road polygon is road", lambda c: _attr(c, 30, 0x11), r"ground: polygon 30 has kind 1.*first 30"),
    ("fewer polygons than steps", lambda c: _gd(c, polys=c.ground_data.polys[:20], attributes=b"\x01" * 20,
                                                lists={1: tuple(range(20))}),
     r"ground: 20 polygons.*30 steps"),
    ("grid value without a list", lambda c: _cell_value(c, 5, 7), r"ground: .*cell 5 .*grid value 0x7 has no list"),
    ("grid value not an int", lambda c: _cell_value(c, 5, "1"), r"ground: .*cell 5 .*grid value '1' is not a u16"),
    ("grid not 128 x 128", lambda c: _gd(c, grid=c.ground_data.grid[:-1]), r"ground: .*grid: 16383 items, expected 16384"),
    ("list names a polygon past the pool", lambda c: _gd(c, lists={1: tuple(range(32))}),
     r"ground: .*list 0x1 entry 31 polygon 31 is outside 0..30"),
    ("list entry not an int", lambda c: _gd(c, lists={1: (0, 1.0)}), r"ground: .*list 0x1 entry 1 polygon 1\.0 is outside"),
    ("lists not a dict", lambda c: _gd(c, lists=[(0,)]), r"ground: .*lists .* not a dict"),
    ("pointer word moved", lambda c: dataclasses.replace(c, ground=(c.ground[0], c.ground[1] + 4)),
     r"ground pointer.*attribute.*0x2308004.*0x2308000.*view"),
    ("grid moved, pointer not", lambda c: _gd(c, grid_at=_GRID_AT + 0x10000),
     r"ground pointer.*grid.*0x2300000.*0x2310000"),
    ("grid_at not an int", lambda c: _gd(c, grid_at="0x300000"), r"ground: .*grid_at '0x300000'.*outside data ROM"),
    ("another course's ground", lambda c: _gd(c, course=0), r"ground: ground_data.course 0 is not the Course's id 1"),
    ("course id an array", lambda c: _gd(c, course=np.array([1, 1])), r"ground: ground_data.course array"),
    ("slot not the course id", lambda c: _gd(c, base=5), r"ground: ground_data.base \(the slot\) 5 is not the "
                                                         r"Course's id 1"),
    ("slot not an int", lambda c: _gd(c, base="1"), r"ground: ground_data.base \(the slot\) '1' is not"),
    ("pointer word an array", lambda c: dataclasses.replace(c, ground=(np.array([1, 2]), c.ground[1])),
     r"ground pointer: the grid pointer word array"),
    ("pointer words not a pair", lambda c: dataclasses.replace(c, ground=(c.ground[0],)),
     r"ground pointer: ground .* is not the two pointer words"),
    ("not a Ground", lambda c: dataclasses.replace(c, ground_data=(1, 2)), r"ground: ground_data \(1, 2\) is not"),
]


def test_check_course_ground_one_message():
    """An attribute block one byte short is one violation, reported once and by the ground's own checks
    (tgp.ground_tables), not by a second copy of them in the rules."""
    bad = _gd(_grounded(), attributes=_grounded().ground_data.attributes[:-1])
    with pytest.raises(CourseError) as e:
        G._check_rules(bad)
    msg = str(e.value)
    assert msg.count("attributes") == 1 and "\n" not in msg and "are not 31 bytes" in msg


@pytest.mark.parametrize("case, mutate, match", GROUND_RULES, ids=[r[0] for r in GROUND_RULES])
def test_check_course_ground_rules(case, mutate, match):
    """One violation per ground rule: CourseError naming it (never a TypeError or another exception from arithmetic
    on a malformed field). `_grounded()` is partial, so the rules run directly; the real set is below."""
    bad = mutate(_grounded())
    with pytest.raises(CourseError, match=match):
        G._check_rules(bad)


def test_check_course_on_the_real_set(romset_zip):
    """Every real course passes check_course; the reviewer's mutations of Beginner (each accepted by encode_course
    before, then out-of-bounds reads in the game or silently dropped edits) are refused, naming the rule."""
    rs = RomSet.from_zip(romset_zip)
    assert C.ACTOR_POOL == struct.unpack_from("<I", rs.data, _POOL_AT - DATA)[0] == 65
    courses = [G.decode_course(rs, i) for i in range(C.COURSE_COUNT)]
    for c in courses:
        G.check_course(c)
    c = courses[0]
    rep = dataclasses.replace
    key, (pa, ta, w) = next(iter(c.time_tables.items()))
    cps = list(c.checkpoints)
    cps[0] = cps[0]._replace(extensions=tuple(x + 999 for x in cps[0].extensions))
    grid = list(c.spawn[1])
    grid[5] = grid[5]._replace(segment=grid[5].segment + 1)
    g = c.ground_data
    assert g == tgp.decode_ground(rs, 0) and c.ground == (DATA + g.grid_at, DATA + g.attributes_at)
    cases = {
        "AI lines": rep(c, ai_lines=tuple(l[:-1] for l in c.ai_lines)),
        "AI lines.*road line 100": rep(c, road=c.road[:100]),
        "time table": rep(c, time_tables={**c.time_tables, key: (pa, ta, w[:1])}),
        "checkpoint 0: extensions": rep(c, checkpoints=tuple(cps)),
        "record pointer": rep(c, record_pointer=0x0265D120),
        "step table.*step_n 190": rep(c, step_n=c.step_n - 5, steps=tuple(
            s._replace(segment=min(s.segment, c.step_n - 5)) for s in c.steps)),
        "car count 41": rep(c, car_count=(c.car_count[0], 41)),
        "pit": rep(c, pit=c.pit._replace(entry_segment=c.pit.entry_segment + 1)),
        "spawn table 1 car 5: segment": rep(c, spawn=(c.spawn[0], tuple(grid))),
        "ground: polygon 0 has kind 2": rep(c, ground_data=rep(g, attributes=b"\x02" + g.attributes[1:])),
        "ground: polygon 503 has kind 1": rep(c, ground_data=rep(g, attributes=g.attributes[:503] + b"\x01"
                                                                 + g.attributes[504:])),
        "ground: ground of course 0: attributes .* are not 1184 bytes": rep(c, ground_data=rep(g, attributes=g.attributes + b"\xcd")),
        "ground pointer": rep(c, ground=(c.ground[0], c.ground[1] + 4)),
    }
    for match, bad in cases.items():
        with pytest.raises(CourseError, match=match):
            G.encode_course(bad, rs)


def test_golden_on_a_relocated_ground(romset_zip):
    """Beginner's ground moved to the free spans (tgp.relocate_ground, with and without erasing the original): the set
    still decodes, re-emits losslessly on all three courses and gives coverage and excluded_spans, the ground's
    extents coming from tgp.ground_spans (at the new addresses), not from the next course's step table."""
    rs = RomSet.from_zip(romset_zip)
    for erase in (False, True):
        moved = tgp.relocate_ground(rs, 0, erase_original=erase)
        g = G.decode_course(moved, 0).ground_data
        assert (g.grid_at, g.attributes_at) == (tgp.DATA_FREE[0], tgp.DATA_FREE[0] + 2 * tgp.GRID_CELLS)
        assert g.list_base == tgp.COPRO_FREE[0] // 4
        for course in range(C.COURSE_COUNT):
            cov = _covers(moved, course)
            if course == 0:
                assert ("data", tgp.DATA_FREE[0], 2 * tgp.GRID_CELLS, "ground grid") in cov
                assert ("copro_data", tgp.COPRO_FREE[0], 4 * len(g.raw), "ground lists") in cov


# --- car models and scenery ------------------------------------------------------------------------------------------

def test_car_objects_synthetic():
    """car_objects: the objects of the lists the spawn records name; scenery_objects: the others."""
    car, prop = 0x800010, 0x800100
    c = dataclasses.replace(
        _race(), object_lists={LIB: (_place(oba=car),), LIB + 20: (_place(oba=prop),)},
        objects={0x10: _OBJ, 0x100: _OBJ})
    assert c.car_objects == frozenset({0x10})
    assert set(G.scenery_objects(c)) == {0x100}
    assert dataclasses.replace(c, spawn=()).car_objects == frozenset()
    with pytest.raises(CourseError, match="spawn is unset.*car models"):      # the cars would be returned as scenery
        G.scenery_objects(dataclasses.replace(c, spawn=()))


def test_car_objects_are_the_shared_spawn_models(romset_zip):
    """Real set: a course's car models are exactly the objects its spawn records reach, never also scenery; every
    polygon object Beginner shares with Expert or Advanced is a car model (the spawn tables are shared, per car
    index: Expert's 30 cars are Beginner's first 30), so a mirror that transforms scenery_objects leaves the other
    courses' cars alone."""
    rs = RomSet.from_zip(romset_zip)
    cs = [G.decode_course(rs, i) for i in range(C.COURSE_COUNT)]
    for c in cs:
        reached = {C.check_rom_object(p.oba) for t in c.spawn for s in t for lst in s.objects
                   for p in c.object_lists[lst]}
        assert c.car_objects == reached
        scenery = G.scenery_objects(c)
        assert not set(scenery) & c.car_objects and set(scenery) | c.car_objects == set(c.objects)
    b, e, a = cs
    shared = set(b.objects) & (set(e.objects) | set(a.objects))
    assert shared == e.car_objects and len(shared) == 173
    assert a.car_objects <= e.car_objects <= b.car_objects
    assert not set(G.scenery_objects(b)) & (set(e.objects) | set(a.objects))


def test_excluded_names_the_reads_no_table_explains():
    """golden.EXCLUDED names, without an extent (NO_EXTENT), the course-dependent reads not tied to any table (format
    doc, "Unknowns"), so "M1 green" is not read as covering them. The HUD map left that list in Phase 4 (covered:
    test_hud_map_covered), and with it the read at 0x12efc (its place record)."""
    reads = "course-dependent reads not tied to a table"
    assert reads in G.EXCLUDED and reads in G.NO_EXTENT <= set(G.EXCLUDED)
    assert not any("HUD" in name for name in G.EXCLUDED) and "0x12efc" not in G.EXCLUDED[reads]
    for addr in ("0x65d4", "0x6a34", "0xf548", "0x1149c", "0x221b80", "0x221c34", "0x23895c"):
        assert addr in G.EXCLUDED[reads], addr
    from pathlib import Path
    doc = (Path(__file__).parents[1] / "docs/formats/model2-course-tables.md").read_text()
    section = doc.split("### Not covered by M1's golden test")[1].split("\n### ")[0]
    assert "| HUD map" not in section and "| course-dependent reads not tied to a table |" in section
    row = next(ln for ln in section.splitlines() if ln.startswith("| course-dependent reads"))
    assert "`0x12efc`" not in row


def test_hud_map_covered(romset_zip):
    """The golden test covers the HUD map (M1c): decode_course holds course.hud_map's value, and coverage has its
    four i960 tables ("HUD map": the list pointer, the list, the place record, the zooms), its outline object
    ("HUD map outline", the whole object in polygon ROM) and the outline's texture headers."""
    rs = RomSet.from_zip(romset_zip)
    for k in range(C.COURSE_COUNT):
        c = G.decode_course(rs, k)
        h = C.hud_map(rs, c.entry)
        assert c.hud_map == h
        cov = G.coverage(rs, k)
        hud = sorted((r, o, n) for r, o, n, name in cov if name == "HUD map")
        assert hud == sorted((*G._where(a, len(b), "HUD map"), len(b)) for a, b in C.encode_hud_map(h))
        outline = [(r, o, n) for r, o, n, name in cov if name == "HUD map outline"]
        assert outline == [("polygons", 4 * i, 4 * obj.word_count) for i, obj in h.objects.items()]
        for p in h.entries:
            obj = h.objects[C.check_rom_object(p.oba)]
            for hd in T.object_headers(rs, p.tha, obj):
                assert c.headers[hd.tha & T.TEX_ADDR_MASK].words == hd.words


def test_golden_covers_texture_points(romset_zip):
    """The golden test covers the texture points (Phase 7): decode_course holds, per textures word index, the points
    (texture.texture_points) of every placement it decodes an object for (the drawn lists and the HUD outline), keyed
    by its tpa; coverage has them ("texture points", 2 words per vertex of every record, move records included) at
    each tpa; verify_lossless re-emits them on ids 0-2 (it is in verify_lossless, which now includes them); EXCLUDED
    no longer lists them, and no placement's points are left uncovered."""
    rs = RomSet.from_zip(romset_zip)
    assert not any("texture points" in name for name in G.EXCLUDED) and "texture points" in G.ALWAYS_COVERED
    for k in range(C.COURSE_COUNT):
        c = G.decode_course(rs, k)
        places = [(p, c.objects[C.check_rom_object(p.oba)]) for lst in c.object_lists.values() for p in lst
                  if C.check_rom_object(p.oba) in c.objects]
        places += [(p, c.hud_map.objects[C.check_rom_object(p.oba)]) for p in c.hud_map.entries]
        assert set(c.points) == {p.tpa & T.TEX_ADDR_MASK for p, _ in places}
        for p, obj in places:
            assert c.points[p.tpa & T.TEX_ADDR_MASK] == T.texture_points(rs, p.tpa, obj)
        cov = [(r, o, n) for r, o, n, name in G.coverage(rs, k) if name == "texture points"]
        assert cov == sorted(("textures", 2 * (p.tpa & T.TEX_ADDR_MASK),
                              4 * sum(4 if r.is_quad else 3 for r in obj.records)) for p, obj in places)
        assert G.verify_lossless(rs, k) == []
        assert not [t for t in G.excluded_spans(rs, k) if t[0] == "textures"]


def test_golden_covers_segment_flags(romset_zip):
    """The golden test covers the per-segment flags (Phase 6): decode_course holds (SEGMENT_FLAGS[id]'s table,
    course.segment_flags), coverage has the pointer word ("segment flags pointer") and the table ("segment flags",
    one u16 per road segment), verify_lossless re-emits both on ids 0-2, and EXCLUDED no longer lists them (only the
    id-3 stub's pointer, with the stub's other array entries)."""
    rs = RomSet.from_zip(romset_zip)
    for k in range(C.COURSE_COUNT):
        c = G.decode_course(rs, k)
        at = struct.unpack("<I", rs.read_i960(C.SEGMENT_FLAGS + 4 * k, 4))[0]
        assert c.segment_flags == (at, C.segment_flags(rs, c.entry))
        assert len(c.segment_flags[1]) == len(c.road)
        cov = G.coverage(rs, k)
        assert [(r, o, n) for r, o, n, name in cov if name == "segment flags pointer"] == \
            [("data", C.SEGMENT_FLAGS - DATA + 4 * k, 4)]
        assert [(r, o, n) for r, o, n, name in cov if name == "segment flags"] == [("data", at - DATA, 2 * len(c.road))]
        assert G.verify_lossless(rs, k) == []
        stub = [(r, o, n) for r, o, n, name in G.excluded_spans(rs, k) if o == C.SEGMENT_FLAGS - DATA + 12]
        assert stub == [("data", C.SEGMENT_FLAGS - DATA + 12, 4)]
    assert "per-segment flags" not in G.EXCLUDED
    assert {"segment flags pointer", "segment flags"} <= G.ALWAYS_COVERED


# --- partial Courses: unset fields are filled from the set (golden.complete) or refused ------------------------------

def test_table_fields_are_derived():
    names = {f.name for f in dataclasses.fields(G.Course)}
    assert set(G.TABLE_FIELDS) == names - {"course", "entry", "record_pointer", "id_arrays", "pit", "panels", "ground"}
    assert {"time_tables", "checkpoints", "ground_data", "spawn"} <= set(G.TABLE_FIELDS)


def test_check_course_partial_refused():
    """A Course with only `checkpoints` set cannot be checked without a set: the unset fields are named."""
    c = G.Course(course=1, entry=_entry(course=1), checkpoints=_race().checkpoints)
    for call in (lambda: G.check_course(c), lambda: G.encode_course(c)):
        with pytest.raises(CourseError, match=r"unset fields.*time_tables") as e:
            call()
        assert "laps" in str(e.value) and "road" in str(e.value)
        # billboards and switch_lists are empty on real courses (Expert, Advanced): never "unset"
        assert "billboards" not in str(e.value) and "switch_lists" not in str(e.value)


def test_encode_course_partial_filled(romset_zip):
    """Beginner with the car count lowered and nothing else set: the rest comes from the set, as if decoded."""
    rs = RomSet.from_zip(romset_zip)
    full = G.decode_course(rs, 0)
    count = (full.car_count[0], full.car_count[1] - 1)
    partial = G.Course(course=0, entry=full.entry, car_count=count)
    want = dataclasses.replace(full, car_count=count)
    filled = G.complete(partial, rs)
    assert filled.car_count == count and filled.road == full.road and filled.ground_data == full.ground_data
    assert G.encode_course(partial, rs) == G.encode_course(want, rs)
    assert G.encode_course(want, rs) == G.encode_course(want)       # complete: `rs` unused
    with pytest.raises(CourseError, match="unset"):
        G.encode_course(partial)                                    # no set: partial refused


def test_partial_laps_are_judged_against_the_sets_tables(romset_zip):
    """The Phase 2 residual: laps changed alone used to encode, leaving the ROM's time tables and checkpoint
    extensions to be mis-read. Beginner's 9-word time table is exactly 8 laps (1 + 8 x 1 words)."""
    rs = RomSet.from_zip(romset_zip)
    full = G.decode_course(rs, 0)
    higher = {m: (a, n + 1) for m, (a, n) in full.laps.items()}
    with pytest.raises(CourseError, match=r"time table.*10 words|time table.*9 words, needs 10"):
        G.encode_course(G.Course(course=0, entry=full.entry, laps=higher), rs)
    lower = {m: (a, n - 1) for m, (a, n) in full.laps.items()}
    with pytest.raises(CourseError, match=r"checkpoint 0: extensions"):       # the view still has the old laps
        G.encode_course(G.Course(course=0, entry=full.entry, laps=lower), rs)


def test_check_course_view_with_unset_source(romset_zip):
    """A pit edit with step_n (and the steps) unset: they are filled from the set, so the view is checked."""
    rs = RomSet.from_zip(romset_zip)
    full = G.decode_course(rs, 0)
    bad = G.Course(course=0, entry=full.entry, pit=full.pit._replace(box_segment=full.pit.box_segment + 1))
    with pytest.raises(CourseError, match=r"pit.*encode_pit"):
        G.check_course(bad, rs)
    G.check_course(dataclasses.replace(bad, pit=full.pit), rs)
    # a time-table-free edit of the checkpoints is checked against the set's time tables
    cps = list(full.checkpoints)
    cps[-1] = cps[-1]._replace(extensions=(7, 7))
    with pytest.raises(CourseError, match=r"extensions"):
        G.check_course(G.Course(course=0, entry=full.entry, checkpoints=tuple(cps)), rs)


def test_actorless_course_is_complete(romset_zip):
    """A course whose actor table names no list (every slot 0: the converter's empty scenery) has no actor lists, no
    unmodelled actors, no billboard, reference point or switch: those fields are empty for real, neither unset
    (`unset_fields`) nor refilled by `complete`. With any slot naming a list they are unset as before."""
    rs = RomSet.from_zip(romset_zip)
    full = G.decode_course(rs, 0)
    bare = dataclasses.replace(full, actor_slots=(0,) * C.GRID_CELLS, actor_lists={}, unmodelled=(), billboards={},
                               billboard_ref=None, switch_lists=())
    assert G.unset_fields(bare) == ()
    assert G.complete(bare, rs) is bare
    G.check_course(bare)
    patches = G.encode_course(bare)
    assert not any(name in ("actor list", "billboard lists", "billboard reference", "switch lists")
                   for _, _, _, name in G.tables(bare))
    assert patches == G.encode_course(bare, rs)
    for f in ("actor_lists", "unmodelled"):
        named = dataclasses.replace(bare, actor_slots=full.actor_slots)
        assert f in G.unset_fields(named)
        assert getattr(G.complete(named, rs), f) == getattr(full, f)
    for slots in ((), (0,) * 3, (False,) * C.GRID_CELLS, (0.0,) * C.GRID_CELLS, None):      # not an empty table
        assert "actor_lists" in G.unset_fields(dataclasses.replace(bare, actor_slots=slots))


# --- course ids 0-15 (Phase 8, Task 3): the grown set ----------------------------------------------------------------

GROWN_DIFFS = {"id_arrays", "laps", "car_count", "rolling_start", "grid_base"}
"""The Course fields of ids 0-2 that differ between the original and the grown set (course.grow_ids): where the
per-id arrays are (id_arrays) and the address halves of the per-id values read in place (laps: LAP_COUNTS[mode]'s
per-id level; car_count: CAR_COUNTS[0]'s; rolling_start: ROLLING_STARTS; grid_base: GRID_BASES). Every value is the
same; the time tables' addresses are not among them (TIME_TABLES' per-id level moved, the per-difficulty words it
points at did not)."""
ID_FIELDS = {"course", "entry", "record_pointer", "backdrop", "ground_data", "hud_map", "laps", "car_count",
             "rolling_start", "grid_base"}
"""The fields of a grown set's ids 3-15 that differ from Beginner's (their tables are Beginner's): the id itself (in
course, entry, backdrop, ground_data, hud_map), the record's address and the per-id value addresses."""


def _same(a, b) -> bool:
    """Deep equality of Course field values, numpy arrays compared by value."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _differ(a: G.Course, b: G.Course) -> set[str]:
    return {f.name for f in dataclasses.fields(G.Course) if not _same(getattr(a, f.name), getattr(b, f.name))}


def _addresses_only(a, b) -> bool:
    """(address, value) pairs, or a dict of them, that differ in the addresses only."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_addresses_only(a[k], b[k]) for k in a)
    return a[1] == b[1] and a[0] != b[0]


def test_grow_ids_round_trip(romset_zip):
    """Ids 0-2 decode the same before and after course.grow_ids, but for GROWN_DIFFS (id_arrays names the new
    tables; laps, car_count, rolling_start and grid_base differ in their addresses, not their values), and re-emit
    losslessly on the grown set, every table covered and nothing excluded reached (_covers). Ids 3-15 decode on the
    grown set as Beginner's tables under their own id (ID_FIELDS) and re-emit losslessly; id 3 is covered like the
    originals. The original set still has ids 0-2 only."""
    stock = RomSet.from_zip(romset_zip)
    g = C.grow_ids(stock)
    beginner = G.decode_course(g, 0)
    for i in range(C.COURSE_IDS):
        c = G.decode_course(g, i)
        assert G.verify_lossless(g, i) == [], i
        if i < C.COURSE_COUNT:
            before = G.decode_course(stock, i)
            assert _differ(before, c) == GROWN_DIFFS, i
            assert c.id_arrays == {n: a.old if n in C.WIDE_ARRAYS else a.new for n, a in C.ID_ARRAYS.items()}
            assert before.id_arrays == {n: a.old for n, a in C.ID_ARRAYS.items()}
            assert all(_addresses_only(getattr(before, f), getattr(c, f)) for f in GROWN_DIFFS - {"id_arrays"}), i
        else:
            assert _differ(beginner, c) == ID_FIELDS, i
            assert c.entry == dataclasses.replace(beginner.entry, course=i)
            assert c.record_pointer == C.record_address(i)
    for i in (0, 1, 2, 3):
        _covers(g, i)
    for rs, bad in ((stock, 3), (g, 16), (g, True)):
        with pytest.raises(CourseError, match="is not a course id"):
            G.decode_course(rs, bad)


def test_grown_course_ids_refused_on_original_arrays(romset_zip):
    """A Course of id 3-15 encodes only against grown arrays (Course.id_arrays at their new addresses): with the
    original arrays its per-id words would land past their four entries."""
    g = C.grow_ids(RomSet.from_zip(romset_zip))
    c = G.decode_course(g, 5)
    G.check_course(c)
    with pytest.raises(CourseError, match="course id: 5 is not 0..2"):
        G.check_course(dataclasses.replace(c, id_arrays={}))
    with pytest.raises(CourseError, match="not per-id arrays"):
        G.check_course(dataclasses.replace(c, id_arrays={**c.id_arrays, "NOPE": 1}))


def test_stale_id_arrays_refused(romset_zip):
    """A Course decoded from the original set carries the original arrays' addresses (id_arrays); encoded against a
    grown set its per-id words would land where the grown readers no longer read: refused (complete, so
    check_course and encode_course too). An empty id_arrays takes the set's."""
    stock = RomSet.from_zip(romset_zip)
    g = C.grow_ids(stock)
    c = G.decode_course(stock, 0)
    for f in (lambda: G.encode_course(c, g), lambda: G.check_course(c, g), lambda: G.complete(c, g)):
        with pytest.raises(CourseError, match="id_arrays differ from the set's per-id arrays"):
            f()
    with pytest.raises(CourseError, match=r"id arrays: rolling_start's address 0x28050e8 is not 0x3"):
        G.encode_course(dataclasses.replace(c, id_arrays={}), g)        # the value addresses are the old ones too
    fresh = dataclasses.replace(c, id_arrays={}, laps={}, car_count=None, rolling_start=None, grid_base=None)
    filled = G.complete(fresh, g)
    assert filled.id_arrays == C.id_tables(g)
    assert G.encode_course(filled, g) == G.encode_course(G.decode_course(g, 0))
    with pytest.raises(CourseError, match="id_arrays differ"):
        G.encode_course(G.decode_course(g, 1), stock)


def test_id_arrays_junk_refused(romset_zip):
    """Junk per-id array addresses or names raise CourseError, never a raw exception, and a bool is not an address."""
    rs = RomSet.from_zip(romset_zip)
    c = G.decode_course(rs, 0)
    for arrays in ({"GRID_TABLES": "x"}, {"GRID_TABLES": True}, {"GRID_TABLES": 1.0}, {"GRID_TABLES": -4},
                   {"GRID_TABLES": 0x00300000}, {"GRID_TABLES": DATA + C.DATA_WINDOW - 4}):
        with pytest.raises(CourseError, match="GRID_TABLES: address"):
            G.tables(dataclasses.replace(c, id_arrays={**c.id_arrays, **arrays}))
        with pytest.raises(CourseError):
            G.encode_course(dataclasses.replace(c, id_arrays={**c.id_arrays, **arrays}))
    for bad in ({"HUD_MAP_LISTS": "x"}, {"HUD_MAP_LISTS": True}, ["HUD_MAP_LISTS"], "x"):
        with pytest.raises(CourseError):
            C.encode_hud_map(c.hud_map, bad)
    for name in (["GRID_TABLES"], None, 3, "NOPE"):
        with pytest.raises(CourseError, match="not a per-id array"):
            C.array_at(None, name)
    assert C.array_at({"GRID_TABLES": DATA + 0x1000000}, "GRID_TABLES") == DATA + 0x1000000
    with pytest.raises(CourseError, match="is not a RomSet"):
        C.check_rom_object(0x00800000, b"")
    with pytest.raises(CourseError, match="is not a RomSet"):
        C.data_size(b"")
    with pytest.raises(T.TextureError, match="is not a RomSet"):
        T.sheet_source(b"", 1)


def test_original_set_refuses_writes_past_16mb(romset_zip):
    """End to end: a table the encoders accept at a custom-only address (data offset 0x1000000, i960 0x03000000) is
    refused by an original set when written (RomSet.with_patches: RomSetError, a DecodeError) and accepted by the
    custom one."""
    from dtools.arcade.errors import DecodeError
    from dtools.arcade.romset import RomSetError
    stock = RomSet.from_zip(romset_zip)
    c = G.decode_course(stock, 0)
    moved = dataclasses.replace(c, checkpoint_table=C.ID_HOME)
    entries = G.encode_course(moved, stock)
    assert ("data", C.ID_HOME - DATA) in entries
    with pytest.raises(RomSetError, match="past|outside|region") as e:
        stock.with_patches(entries)
    assert isinstance(e.value, DecodeError)
    custom = stock.as_custom().with_patches(entries)
    assert C.course_table(custom)[0] == c.entry


def test_lanes_billboard_per_id(romset_zip):
    """Phase 9, Task 2 (course.widen_ids): GRID_LANES and BILLBOARD_REF per id. Ids 0-3 and the mirrors' ids 8-10
    decode on the widened set as on the guarded one but for id_arrays, which differs exactly in the two WIDE_ARRAYS
    (their new tables), and re-emit losslessly, ids 0-2 covered as before (_covers, no excluded byte reached). Id 9's
    own entries (WIDE_LANES + 4 * 9, WIDE_BILLBOARD + 12 * 9) given swapped lanes and a mirrored point: id 9 decodes
    them (its spawn lanes follow), every other id keeps the shared values, id 9's Course writes exactly those entries
    (tables) and ids 0 and 9 re-emit losslessly. On a set that is not widened the two stay one shared entry
    (id_entry), so Phase 8's sets encode as they did; no grown id has tables of its own by them (_own_tables)."""
    g = C.guard_new_ids(C.grow_ids(RomSet.from_zip(romset_zip)))
    w = C.widen_ids(g)
    for k in (0, 1, 2, 3, 8, 9, 10):
        a, b = G.decode_course(g, k), G.decode_course(w, k)
        assert _differ(a, b) == {"id_arrays"}, k
        assert {n for n in a.id_arrays if a.id_arrays[n] != b.id_arrays[n]} == C.WIDE_ARRAYS, k
        assert all(b.id_arrays[n] == C.ID_ARRAYS[n].new and a.id_arrays[n] == C.ID_ARRAYS[n].old
                   for n in C.WIDE_ARRAYS)
        assert G.verify_lossless(w, k) == [], k
    for k in range(3):
        _covers(w, k)
    lanes_at = C.rom_word_at(C.WIDE_LANES + 4 * 9)
    ref_at = C.rom_word_at(C.WIDE_BILLBOARD + 12 * 9)
    out = w.with_patches({lanes_at: struct.pack("<2H", 0x57F, 0x17F), ref_at: struct.pack("<3f", 210.0, 0.0, 379.0)})
    assert C.grid_lanes(out, 9) == (0x57F, 0x17F) and C.billboard_ref(out, 9) == (210.0, 0.0, 379.0)
    for k in set(range(16)) - {9}:
        assert C.grid_lanes(out, k) == (0x180, 0x580) and C.billboard_ref(out, k) == (210.0, 0.0, -380.0), k
    got, before = G.decode_course(out, 9), G.decode_course(w, 9)
    assert got.grid_lanes == (0x57F, 0x17F) and got.billboard_ref == (210.0, 0.0, 379.0)
    assert [s.lane for s in got.spawn[1]] == [{0x180: 0x57F, 0x580: 0x17F}[s.lane] for s in before.spawn[1]]
    assert {(r, o) for r, o, _, n in G.tables(got) if n in ("grid lanes", "billboard reference")} == {lanes_at, ref_at}
    for k in (0, 9):
        assert G.verify_lossless(out, k) == [], k
    # not widened: one shared entry, where Phase 8 wrote it
    c = G.decode_course(g, 1)
    assert dict((a[:2], a[2]) for a in G.tables(c) if a[3] == "grid lanes") == {
        C.rom_word_at(C.GRID_LANES): struct.pack("<2H", 0x180, 0x580)}
    assert [k for k in range(3, 16) if G._own_tables(g, C.id_tables(g), k)] == \
        [k for k in range(3, 16) if G._own_tables(w, C.id_tables(w), k)] == []
