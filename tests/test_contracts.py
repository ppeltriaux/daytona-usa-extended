"""The contracts every arcade decoder / encoder shares (dtools/arcade): one error base, `DecodeError`, that every
module's error derives from, and one number rule: an integer field takes Python and numpy integers but not bool,
a float field takes finite Python and numpy numbers (not bool, NaN or infinity) that fit single precision."""

from dataclasses import replace

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import geo
from dtools.arcade import mirror
from dtools.arcade import polyrom as P
from dtools.arcade import romset
from dtools.arcade import texture as T
from dtools.arcade import tgp
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError as ErrorsDecodeError
from dtools.arcade.tablebase import CourseError, DecodeError
from dtools.arcade import golden as G
from tests.test_golden import (DATA, LIB, _AI, _LANE, _OBJ, _actor, _entry, _hdr, _hud, _panel, _place, _race,
                               _road, _slot)
from tests.test_tgp import _flat, _ground, _sq

NAN, INF = float("nan"), float("inf")


def test_one_error_base():
    assert DecodeError is ErrorsDecodeError
    for cls in (CourseError, T.TextureError, P.PolyError, geo.GeoError, romset.RomSetError, tgp.TgpError,
                mirror.MirrorError):
        assert issubclass(cls, DecodeError), cls


def test_dc_errors_share_the_decode_error_base():
    from dtools.dc.disc import DiscError
    from dtools.dc.errors import DcError
    from dtools.dc.iso import IsoError
    assert issubclass(DcError, DecodeError)
    assert issubclass(DiscError, DcError) and issubclass(IsoError, DcError)


# (name, the call with Python numbers, the same call with numpy scalars); both must give the same bytes
NUMPY = [
    ("encode_course_entry", lambda: C.encode_course_entry(_entry(step_table=0x10, raw=(1, 2))),
     lambda: C.encode_course_entry(_entry(step_table=np.int64(0x10), raw=(np.uint32(1), np.int32(2))))),
    ("encode_placement", lambda: C.encode_placement(_place(tpa=5, raw=(3,))),
     lambda: C.encode_placement(_place(oba=np.uint32(0x800004), tpa=np.uint32(5), raw=(np.int64(3),)))),
    ("encode_actor", lambda: C.encode_actor(_actor(position=(1.5, 0, -2.0), angle=5, ident=7)),
     lambda: C.encode_actor(_actor(position=(np.float32(1.5), np.int16(0), np.float64(-2.0)), angle=np.int16(5),
                                   ident=np.uint16(7), handler=np.uint32(C.STATIC_PROP)))),
    ("encode_road_point", lambda: C.encode_road_point(_road(curve=-3, lanes=(_LANE._replace(heading=3),) * C.LANES)),
     lambda: C.encode_road_point(_road(lane_lo=np.uint8(1), curve=np.int64(-3), lanes=(_LANE._replace(
         position=(np.float32(0), 0.0, 0.0), heading=np.int32(3)),) * C.LANES))),
    ("encode_ai_point", lambda: C.encode_ai_point(_AI._replace(lane=0x300, speed=0.5)),
     lambda: C.encode_ai_point(_AI._replace(lane=np.int64(0x300), speed=np.float32(0.5), step=np.uint16(0)))),
    ("encode_line_table", lambda: C.encode_line_table(0x600000, [[_AI] * 8] * 8),
     lambda: C.encode_line_table(np.int64(0x600000), [[_AI] * 8] * 8)),
    ("encode_time_table", lambda: C.encode_time_table([5, 6]), lambda: C.encode_time_table([np.uint32(5), np.int8(6)])),
    ("encode_step_table", lambda: C.encode_step_table(10, [C.Step(3, 1)]),
     lambda: C.encode_step_table(np.int64(10), [C.Step(np.uint16(3), np.uint16(1))])),
    ("encode_spawn_record", lambda: C.encode_spawn_record(_slot(line=2, speed=0.75)),
     lambda: C.encode_spawn_record(_slot(objects=(np.uint32(LIB),) * 8, line=np.int8(2), start=np.uint16(0),
                                         speed=np.float32(0.75), kind=np.uint8(0)))),
    ("encode_map_object", lambda: C.encode_map_object(C.MapObject(0, (1.0, 2.0, 3.0), LIB, (0, 0, 0, 0))),
     lambda: C.encode_map_object(C.MapObject(0, (np.float32(1), np.float64(2), 3), np.uint32(LIB),
                                             (np.uint32(0),) * 4))),
    ("encode_hud_map", lambda: C.encode_hud_map(_hud(position=(1.0, 0.0, 9e-5), angle=-5, pad=0)),
     lambda: C.encode_hud_map(_hud(course=np.int64(0), outline_list=np.uint32(LIB), position=(
         np.float32(1), np.int8(0), np.float64(9e-5)), angle=np.int16(-5), pad=np.uint16(0),
         zooms=np.array([5e-8] * C.HUD_MAP_ZOOM_LEVELS, np.float32)))),
    ("encode_select_panel", lambda: C.encode_select_panel(_panel()),
     lambda: C.encode_select_panel(_panel(course=np.uint8(0), cursor=(np.int32(95), np.int64(396)),
                                          tile=(np.int16(3), np.int16(9)), maps=(np.uint32(DATA),) * 6))),
    ("encode_cell_grid", lambda: C.encode_cell_grid([LIB] * C.GRID_CELLS),
     lambda: C.encode_cell_grid([np.uint32(LIB)] * C.GRID_CELLS)),
    ("encode_laps", lambda: C.encode_laps(8), lambda: C.encode_laps(np.int32(8))),
    ("encode_car_count", lambda: C.encode_car_count(40), lambda: C.encode_car_count(np.int64(40))),
    ("encode_grid_base", lambda: C.encode_grid_base(100), lambda: C.encode_grid_base(np.int16(100))),
    ("encode_grid_lanes", lambda: C.encode_grid_lanes((0x180, 0x580)),
     lambda: C.encode_grid_lanes((np.uint16(0x180), np.uint16(0x580)))),
    ("encode_table_pointer", lambda: C.encode_table_pointer(DATA), lambda: C.encode_table_pointer(np.uint32(DATA))),
    ("encode_tile_map", lambda: TL.encode_tile_map(TL.TileMap(0, True, 1, 2, (1, 2))),
     lambda: TL.encode_tile_map(TL.TileMap(np.int16(0), True, np.uint32(1), np.uint32(2), np.array([1, 2], np.int16)))),
    ("encode_palette_list", lambda: TL.encode_palette_list([TL.PaletteUpload(0x01800000, (1, 2))]),
     lambda: TL.encode_palette_list([TL.PaletteUpload(np.uint32(0x01800000), np.array([1, 2], np.uint16))])),
    ("encode_char_list", lambda: TL.encode_char_list([(DATA, TL.CHAR_RAM[0])]),
     lambda: TL.encode_char_list([(np.uint32(DATA), np.uint32(TL.CHAR_RAM[0]))])),
    ("encode_tex_header", lambda: T.encode_tex_header(_hdr()),
     lambda: T.encode_tex_header(_hdr(width=np.int64(64), luma_base=np.uint8(3)))),
    ("encode_object", lambda: P.encode_object(_OBJ),
     lambda: P.encode_object(P.Object((np.float32(0), 0.0, 0.0), _OBJ.seed1, _OBJ.records, np.int64(20)))),
    ("encode_ground", lambda: tgp.encode_ground(_ground()),
     lambda: tgp.encode_ground(_ground(grid=np.array(_ground().grid, np.uint16), base=np.int8(0),
                                       lists={np.int64(1): np.array([0, 1]), 4: (np.uint32(2), 3, 4)},
                                       list_base=np.int64(0x30), polys=(tgp.GroundPoly(
                                           tuple(np.array(c, np.float32) for c in _sq(0, 0).corners),
                                           centre=_sq(0, 0).centre),) + _ground().polys[1:]))),
]


@pytest.mark.parametrize("name, plain, numpy_", NUMPY, ids=[n[0] for n in NUMPY])
def test_numpy_scalars_are_numbers(name, plain, numpy_):
    assert numpy_() == plain()


# A bool where a number goes: refused (True is not the integer 1 nor the float 1.0 to an encoder)
BOOLS = [
    ("encode_course_entry", lambda: C.encode_course_entry(_entry(step_table=True))),
    ("encode_course_entry raw", lambda: C.encode_course_entry(_entry(raw=(True, 0)))),
    ("encode_placement", lambda: C.encode_placement(_place(tpa=True))),
    ("encode_actor angle", lambda: C.encode_actor(_actor(angle=True))),
    ("encode_actor ident", lambda: C.encode_actor(_actor(ident=False))),
    ("encode_actor position", lambda: C.encode_actor(_actor(position=(True, 0.0, 0.0)))),
    ("encode_road_point", lambda: C.encode_road_point(_road(curve=True))),
    ("encode_road_point position", lambda: C.encode_road_point(_road(lanes=(_LANE._replace(
        position=(0.0, False, 0.0)),) * C.LANES))),
    ("encode_ai_point", lambda: C.encode_ai_point(_AI._replace(lane=True))),
    ("encode_ai_point speed", lambda: C.encode_ai_point(_AI._replace(speed=True))),
    ("encode_time_table", lambda: C.encode_time_table([True])),
    ("encode_step_table", lambda: C.encode_step_table(10, [C.Step(True, 1)])),
    ("encode_spawn_record", lambda: C.encode_spawn_record(_slot(line=True))),
    ("encode_spawn_record speed", lambda: C.encode_spawn_record(_slot(speed=True))),
    ("encode_laps", lambda: C.encode_laps(True)),
    ("encode_car_count", lambda: C.encode_car_count(True)),
    ("encode_grid_base", lambda: C.encode_grid_base(False)),
    ("encode_select_panel", lambda: C.encode_select_panel(_panel(course=True))),
    ("encode_map_object", lambda: C.encode_map_object(C.MapObject(0, (0.0, 0.0, 0.0), LIB, (True, 0, 0, 0)))),
    ("encode_hud_map angle", lambda: C.encode_hud_map(_hud(angle=True))),
    ("encode_hud_map course", lambda: C.encode_hud_map(_hud(course=False))),
    ("encode_hud_map zoom", lambda: C.encode_hud_map(_hud(zooms=(True,) * C.HUD_MAP_ZOOM_LEVELS))),
    ("encode_tile_map", lambda: TL.encode_tile_map(TL.TileMap(0, True, 1, 1, (True,)))),
    ("encode_palette_list", lambda: TL.encode_palette_list([TL.PaletteUpload(0x01800000, (True, 0))])),
    ("encode_tex_header", lambda: T.encode_tex_header(_hdr(luma_base=True))),
    ("encode_sheet_source", lambda: T.encode_sheet_source(False)),
    ("encode_object", lambda: P.encode_object(P.Object((True, 0.0, 0.0), _OBJ.seed1, _OBJ.records, 20))),
    ("encode_object attr", lambda: P.encode_object(P.Object(_OBJ.seed0, _OBJ.seed1, [P.Record(
        True, (0.0,) * 3, (0.0,) * 3, None)], 20))),
    ("encode_ground direction", lambda: tgp.encode_ground(_ground(polys=(tgp.GroundPoly(_sq(0, 0).corners,
                                                                                         direction=True),)
                                                                         + _ground().polys[1:]))),
    ("encode_ground corner", lambda: tgp.encode_ground(_ground(polys=(tgp.GroundPoly(
        ((True, 0.0, 0.0),) + _sq(0, 0).corners[1:]),) + _ground().polys[1:]))),
    ("encode_ground list", lambda: tgp.encode_ground(_ground(lists={1: (False, 1), 4: (2, 3, 4)}))),
]


@pytest.mark.parametrize("name, call", BOOLS, ids=[b[0] for b in BOOLS])
def test_bool_is_not_a_number(name, call):
    with pytest.raises(DecodeError):
        call()


def _float_cases(v):
    return [
        ("encode_actor", lambda: C.encode_actor(_actor(position=(v, 0.0, 0.0)))),
        ("encode_road_point", lambda: C.encode_road_point(_road(lanes=(_LANE._replace(position=(0.0, 0.0, v)),)
                                                                * C.LANES))),
        ("encode_ai_point position", lambda: C.encode_ai_point(_AI._replace(position=(0.0, v, 0.0)))),
        ("encode_ai_point speed", lambda: C.encode_ai_point(_AI._replace(speed=v))),
        ("encode_spawn_record", lambda: C.encode_spawn_record(_slot(speed=v))),
        ("encode_map_object", lambda: C.encode_map_object(C.MapObject(0, (v, 0.0, 0.0), LIB, (0, 0, 0, 0)))),
        ("encode_hud_map position", lambda: C.encode_hud_map(_hud(position=(0.0, v, 1.0)))),
        ("encode_hud_map zoom", lambda: C.encode_hud_map(_hud(zooms=(v,) * C.HUD_MAP_ZOOM_LEVELS))),
        ("encode_billboard_ref", lambda: C.encode_billboard_ref((0.0, 0.0, v))),
        ("encode_object seed", lambda: P.encode_object(P.Object((v, 0.0, 0.0), _OBJ.seed1, _OBJ.records, 20))),
        ("encode_object point", lambda: P.encode_object(P.Object(_OBJ.seed0, _OBJ.seed1, [P.Record(
            1, (0.0,) * 3, (0.0, 0.0, v), (0.0,) * 3)], 20))),
        ("encode_object normal", lambda: P.encode_object(P.Object(_OBJ.seed0, _OBJ.seed1, [P.Record(
            1, (v, 0.0, 0.0), (0.0,) * 3, (0.0,) * 3)], 20))),
        ("encode_ground corner", lambda: tgp.encode_ground(_ground(polys=(tgp.GroundPoly(
            ((0.0, v, 0.0),) + _sq(0, 0).corners[1:]),) + _ground().polys[1:]))),
        ("encode_ground centre", lambda: tgp.encode_ground(_ground(polys=(tgp.GroundPoly(
            _sq(0, 0).corners, centre=(0.0, 0.0, v)),) + _ground().polys[1:]))),
    ]


FLOATS = [(f"{name} {v!r}", call) for v in (NAN, INF, -INF, np.float32("nan"), np.float64("-inf"), 1e39, 10 ** 400)
          for name, call in _float_cases(v)]


@pytest.mark.parametrize("name, call", FLOATS, ids=[f[0] for f in FLOATS])
def test_floats_must_be_finite_single_precision(name, call):
    with pytest.raises(DecodeError):
        call()


def test_is_number_refuses_integers_past_the_float_range():
    """The shared number rule (tablebase.is_number, changed in Phase 5): an integer past the float range is not a
    number (math.isfinite would raise OverflowError on it), and an arcade encoder's float field refuses it with its
    own error, not an OverflowError."""
    from dtools.arcade.tablebase import is_number
    assert is_number(10 ** 400) is False and is_number(-10 ** 400) is False
    assert is_number(10 ** 38) is True and is_number(np.int64(2 ** 62)) is True
    with pytest.raises(CourseError, match="speed"):
        C.encode_ai_point(_AI._replace(speed=10 ** 400))
    with pytest.raises(CourseError, match="position"):
        C.encode_ai_point(_AI._replace(position=(0.0, -10 ** 400, 0.0)))


# Input shapes the encoders used to pass to struct.pack or index unchecked (TypeError, IndexError, struct.error)
SHAPES = [
    ("select panel cursor of 3", lambda: C.encode_select_panel(_panel(cursor=(1, 2, 3)))),
    ("select panel tile of 1", lambda: C.encode_select_panel(_panel(tile=(1,)))),
    ("select panel cursor None", lambda: C.encode_select_panel(_panel(cursor=None))),
    ("select panel maps None", lambda: C.encode_select_panel(_panel(maps=None))),
    ("map object list None", lambda: C.encode_map_object(C.MapObject(0, (0.0, 0.0, 0.0), None, (0, 0, 0, 0)))),
    ("map object raw of 2", lambda: C.encode_map_object(C.MapObject(0, (0.0, 0.0, 0.0), LIB, (0, 0)))),
    ("map object position of 2", lambda: C.encode_map_object(C.MapObject(0, (0.0, 0.0), LIB, (0, 0, 0, 0)))),
    ("actor position None", lambda: C.encode_actor(_actor(position=None))),
    ("HUD map position None", lambda: C.encode_hud_map(_hud(position=None))),
    ("HUD map entries None", lambda: C.encode_hud_map(_hud(entries=None))),
    ("HUD map entry None", lambda: C.encode_hud_map(_hud(entries=(None,)))),
    ("HUD map objects None", lambda: C.encode_hud_map(_hud(objects=None))),
    ("HUD map object key str", lambda: C.encode_hud_map(_hud(objects={"4": _OBJ}))),
    ("HUD map zooms of 3", lambda: C.encode_hud_map(_hud(zooms=(0.0,) * 3))),
    ("mirror HUD map not a HudMap", lambda: mirror.mirror_hud_map("map")),
    ("mirror HUD map objects None", lambda: mirror.mirror_hud_map(_hud(objects=None))),
    ("mirror HUD map object None", lambda: mirror.mirror_hud_map(_hud(objects={4: None}))),
    ("mirror backdrop not a Backdrop", lambda: mirror.mirror_backdrop("backdrop", {})),
    ("mirror backdrop segments None", lambda: mirror.mirror_backdrop(C.Backdrop(0, None, None), {})),
    ("mirror backdrop tiles None", lambda: mirror.mirror_backdrop(_backdrop(tiles=None), {})),
    ("mirror backdrop tile str", lambda: mirror.mirror_backdrop(_backdrop(tiles=("a",) * 32), {})),
    ("mirror backdrop tile past u16", lambda: mirror.mirror_backdrop(_backdrop(tiles=(0x10000,) * 32), {})),
    ("mirror backdrop tile float", lambda: mirror.mirror_backdrop(_backdrop(tiles=(0x3000 + 0.5,) * 32), {})),
    ("mirror backdrop chars None", lambda: mirror.mirror_backdrop(_backdrop(), None)),
    ("mirror backdrop char missing", lambda: mirror.mirror_backdrop(_backdrop(), {})),
    ("mirror backdrop char of 7 x 8", lambda: mirror.mirror_backdrop(_backdrop(),
                                                                     {0x3000: np.zeros((7, 8), np.uint8)})),
    ("mirror backdrop char pixel 16", lambda: mirror.mirror_backdrop(_backdrop(), {0x3000: np.full((8, 8), 16)})),
    ("mirror backdrop free_char str", lambda: mirror.mirror_backdrop(_backdrop(), {0x3000: np.zeros((8, 8), np.uint8)},
                                                                     "0x3000")),
    ("backdrop chars blocks None", lambda: _backdrop().chars(None)),
    ("backdrop chars block missing", lambda: _backdrop().chars({})),
    ("backdrop chars block short", lambda: _backdrop().chars({0x2100000: np.zeros((0, 8, 8), np.uint8)})),
    ("flip chars of 8", lambda: C.flip_chars(np.zeros(8))),
    ("billboard reference None", lambda: C.encode_billboard_ref(None)),
    ("billboard reference of 2", lambda: C.encode_billboard_ref((0.0, 0.0))),
    ("mirror billboard lists of 63", lambda: mirror.mirror_billboard_lists((LIB,) * 63)),
    ("mirror billboard lists None", lambda: mirror.mirror_billboard_lists(None)),
    ("spawn objects None", lambda: C.encode_spawn_record(_slot(objects=None))),
    ("course entry raw None", lambda: C.encode_course_entry(_entry(raw=None))),
    ("placement raw None", lambda: C.encode_placement(_place(raw=None))),
    ("grid lanes None", lambda: C.encode_grid_lanes(None)),
    ("object list None", lambda: C.encode_object_list(None)),
    ("object list of None", lambda: C.encode_object_list([None])),
    ("checkpoint table of None", lambda: C.encode_checkpoint_table([None])),
    ("char list pair of 1", lambda: TL.encode_char_list([(DATA,)])),
    ("tile map entries None", lambda: TL.encode_tile_map(TL.TileMap(0, True, 1, 1, None))),
    ("backdrop tiles None", lambda: TL.encode_backdrop_segment(TL.BackdropSegment(0, 1, 32, None))),
    ("palette words None", lambda: TL.encode_palette_list([TL.PaletteUpload(0x01800000, None)])),
    ("time table None", lambda: C.encode_time_table(None)),
    ("cell grid None", lambda: C.encode_cell_grid(None)),
    ("object point None", lambda: P.encode_object(P.Object(None, _OBJ.seed1, _OBJ.records, 20))),
    ("road line None", lambda: C.encode_road_line(None)),
    ("ground polys None", lambda: tgp.encode_ground(_ground(polys=None))),
    ("ground corners of 3", lambda: tgp.encode_ground(_ground(polys=(tgp.GroundPoly(_sq(0, 0).corners[:3]),)
                                                                   + _ground().polys[1:]))),
    ("ground list None", lambda: tgp.encode_ground(_ground(lists={1: None, 4: (2, 3, 4)}))),
    ("ground grid None", lambda: tgp.encode_ground(_ground(grid=None))),
    # the query model reads a Ground too (cell (64, 64) holds x, z = 4)
    ("query not a Ground", lambda: tgp.query("x", 4.0, 1.0, 4.0)),
    ("query list index past the polygons", lambda: tgp.query(replace(_flat(_sq(0, 0)), lists={1: (3,)}), 4.0, 1.0,
                                                            4.0)),
    ("query list index not an int", lambda: tgp.query(replace(_flat(_sq(0, 0)), lists={1: ("0",)}), 4.0, 1.0, 4.0)),
    ("query list None", lambda: tgp.query(replace(_flat(_sq(0, 0)), lists={1: None}), 4.0, 1.0, 4.0)),
    ("query lists None", lambda: tgp.query(replace(_flat(_sq(0, 0)), lists=None), 4.0, 1.0, 4.0)),
    ("query grid value with no list", lambda: tgp.query(replace(_flat(_sq(0, 0)), lists={}), 4.0, 1.0, 4.0)),
    ("query grid value not a u16", lambda: tgp.query(replace(_flat(_sq(0, 0)), grid=([0],) * tgp.GRID_CELLS),
                                                     4.0, 1.0, 4.0)),
    ("query ragged grid", lambda: tgp.query(replace(_flat(_sq(0, 0)), grid=(0,) * (tgp.cell(4.0, 4.0)) + ((0,),) + (0,) * (
        tgp.GRID_CELLS - tgp.cell(4.0, 4.0) - 1)), 4.0, 1.0, 4.0)),
    ("query short grid", lambda: tgp.query(replace(_flat(_sq(0, 0)), grid=(0,) * 10), 4.0, 1.0, 4.0)),
    ("query grid None", lambda: tgp.query(replace(_flat(_sq(0, 0)), grid=None), 4.0, 1.0, 4.0)),
    ("query polygon not a GroundPoly", lambda: tgp.query(replace(_flat(_sq(0, 0)), polys=(object(),)), 4.0, 1.0,
                                                         4.0)),
    ("query polys None", lambda: tgp.query(replace(_flat(_sq(0, 0)), polys=None), 4.0, 1.0, 4.0)),
    ("query corners of 3", lambda: tgp.query(_flat(tgp.GroundPoly(_sq(0, 0).corners[:3])), 4.0, 1.0, 4.0)),
    ("query corner of 2", lambda: tgp.query(_flat(tgp.GroundPoly(((0.0, 0.0),) + _sq(0, 0).corners[1:])), 4.0, 1.0,
                                            4.0)),
    ("query corner not a number", lambda: tgp.query(_flat(tgp.GroundPoly((("a", 0.0, 0.0),) + _sq(0, 0).corners[1:])),
                                                    4.0, 1.0, 4.0)),
    ("query corner NaN", lambda: tgp.query(_flat(tgp.GroundPoly(((NAN, 0.0, 0.0),) + _sq(0, 0).corners[1:])), 4.0,
                                           1.0, 4.0)),
    ("surface not a Ground", lambda: tgp.surface("x", 0)),
    ("surface attributes None", lambda: tgp.surface(replace(_ground(), attributes=None), 0)),
]


def _backdrop(tiles=(0x3000,) * 32):
    """A one-row backdrop of char 0x3000, uploaded from a block at 0x2100000."""
    up = TL.UploadSet(0x2074240, 0x2074300, 0x2074400, (TL.CharUpload(0x2100000, 0x10E0000, 1),), (), (DATA,) * 8)
    return C.Backdrop(0, up, (TL.BackdropSegment(0x10000, 1, 32, tiles),) * 8)


@pytest.mark.parametrize("name, call", SHAPES, ids=[s[0] for s in SHAPES])
def test_shapes_are_checked(name, call):
    with pytest.raises(DecodeError):
        call()


# check_course's arithmetic on malformed race fields: each mutation of a Course is a CourseError from the type
# checks, never a TypeError (the rules on the partial Course of test_golden._race(), then check_course on Beginner)
def _slot0(c, **kw):
    return replace(c, spawn=((c.spawn[0][0]._replace(**kw),) + c.spawn[0][1:], c.spawn[1]))


def _road0(c, **kw):
    return replace(c, road=(replace(c.road[0], **kw),) + tuple(c.road[1:]))


RULE_SHAPES = [
    ("car count float", lambda c: replace(c, car_count=(DATA, 2.5))),
    ("car count None", lambda c: replace(c, car_count=(DATA, None))),
    ("car count not a pair", lambda c: replace(c, car_count=5)),
    ("laps None", lambda c: replace(c, laps={0: (DATA, None)})),
    ("laps not a pair", lambda c: replace(c, laps={0: 8})),
    ("time table words 5", lambda c: replace(c, time_tables={(0, 0): (DATA, DATA, 5)})),
    ("time table not a triple", lambda c: replace(c, time_tables={(0, 0): 5})),
    ("grid base None", lambda c: replace(c, grid_base=(DATA, None))),
    ("grid base not a pair", lambda c: replace(c, grid_base=3)),
    ("grid lanes None entry", lambda c: replace(c, grid_lanes=(None, 1))),
    ("rolling start not a pair", lambda c: replace(c, rolling_start=True)),
    ("spawn start str", lambda c: _slot0(c, start="x")),
    ("spawn start None", lambda c: _slot0(c, start=None)),
    ("spawn objects None", lambda c: _slot0(c, objects=None)),
    ("spawn slot None", lambda c: replace(c, spawn=((None,), ()))),
    ("checkpoint None", lambda c: replace(c, checkpoints=(None,))),
    ("step None", lambda c: replace(c, steps=(None,))),
    ("actor lists not a dict", lambda c: replace(c, actor_lists=(1, 2))),
    ("actor list None", lambda c: replace(c, actor_lists={0x00234100: None})),
    ("actor table slot str", lambda c: replace(c, actor_slots=("x",) + c.actor_slots[1:])),
    ("HUD map of another course", lambda c: replace(c, hud_map=_hud(course=(c.course + 1) % C.COURSE_COUNT))),
    ("HUD map not a HudMap", lambda c: replace(c, hud_map=5)),
    ("road lane_hi 9", lambda c: _road0(c, lane_hi=9)),
    ("road lane_lo -1", lambda c: _road0(c, lane_lo=-1)),
    ("road lane_lo str", lambda c: _road0(c, lane_lo="0")),
    ("road lane_lo past lane_hi", lambda c: _road0(c, lane_lo=c.road[0].lane_hi + 1)),
    ("road 7 lanes", lambda c: _road0(c, lanes=c.road[0].lanes[:7])),
    ("road lanes None", lambda c: _road0(c, lanes=None)),
    ("road lane None", lambda c: _road0(c, lanes=(None,) * C.LANES)),
    ("road lane position None", lambda c: _road0(c, lanes=tuple(v._replace(position=None) for v in c.road[0].lanes))),
    ("road lane position str", lambda c: _road0(c, lanes=tuple(v._replace(position=("a", 0.0, 0.0))
                                                               for v in c.road[0].lanes))),
    ("road point None", lambda c: replace(c, road=(None,) + tuple(c.road[1:]))),
]


@pytest.mark.parametrize("name, mutate", RULE_SHAPES, ids=[s[0] for s in RULE_SHAPES])
def test_check_course_shapes_are_checked(name, mutate):
    with pytest.raises(CourseError):
        G._check_rules(mutate(_race()))


def test_check_course_shapes_are_checked_with_the_set(romset_zip):
    """The same values in a complete Course (the real Beginner): check_course and encode_course raise CourseError."""
    rs = romset.RomSet.from_zip(romset_zip)
    full = G.decode_course(rs, 0)
    for name, mutate in RULE_SHAPES:
        bad = mutate(full)
        for call in (G.check_course, G.encode_course):
            with pytest.raises(CourseError):
                call(bad, rs)
