"""The mirror (M1c, dtools/arcade/mirror.py): a negation of world z on one decoded course, proven offline (Task 4 of
Phase 4) and driven in MAME (Task 5, the `mame` tests at the end). Rules: docs/formats/model2-course-tables.md, "What
the mirror (M1c) must transform"; measured here: docs/spike/2026-11-mirror.md, "Offline", "Symmetry", "Rules
checked"."""

import collections
import dataclasses
import functools
import math
import struct

import numpy as np
import pytest

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import geo
from dtools.arcade import golden as G
from dtools.arcade import mirror as M
from dtools.arcade import polyrom as P
from dtools.arcade import tgp
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet
from dtools.arcade.tablebase import DATA_BASE
from dtools.mame import build, runner
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_golden import _race
from tests.test_tgp import _cell, _ground, _sq


@functools.lru_cache(maxsize=1)
def _beginner():
    d = config.rom_dir()
    if d is None or not (d / "daytona.zip").is_file():
        return None
    rs = RomSet.from_zip(d / "daytona.zip")
    c = G.decode_course(rs, 0)
    return rs, c, M.mirror_course(c)


@functools.lru_cache(maxsize=1)
def _built():
    rs, _, _ = _beginner()
    return M.mirrored_set(rs)


@pytest.fixture
def beginner():
    got = _beginner()
    if got is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return got


def _eq(a, b) -> bool:
    """Deep equality that also compares numpy arrays (Course.char_blocks, Course.texels)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return (isinstance(b, (tuple, list)) and len(a) == len(b) and type(a) is type(b)
                and all(_eq(x, y) for x, y in zip(a, b)))
    return a == b


def _close(a, b, tol: float = 1e-4) -> bool:
    """_eq with floats within `tol` (world z mirrored twice comes back to within an f32 ulp: mirror_world)."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _eq(a, b)
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(_close(getattr(a, f.name), getattr(b, f.name), tol)
                                          for f in dataclasses.fields(a) if f.compare)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (tuple, list)):
        return (isinstance(b, (tuple, list)) and len(a) == len(b) and type(a) is type(b)
                and all(_close(x, y, tol) for x, y in zip(a, b)))
    return a == b


def _texels(c: G.Course) -> dict:
    """Course.texels by geometry: decode_course keeps the first header of each geometry it meets and in that order,
    so a set whose cells were permuted decodes the same texels in another order, under another header of the same
    geometry (golden.Texels)."""
    return {G._texel_key(t.header): (t.indices, t.mips) for t in c.texels}


def _differ(a: G.Course, b: G.Course, skip=()) -> list[str]:
    out = [f.name for f in dataclasses.fields(G.Course)
           if f.name not in skip + ("texels",) and not _eq(getattr(a, f.name), getattr(b, f.name))]
    if "texels" not in skip and not (len(a.texels) == len(b.texels) and _eq(_texels(a), _texels(b))):
        out.append("texels")
    return out


# --- the rules, no data --------------------------------------------------------------------------------------------

def test_mirror_error_is_a_decode_error():
    assert issubclass(M.MirrorError, DecodeError)


def test_mirror_point():
    assert M.mirror_point((1.5, 2.0, -3.25)) == (1.5, 2.0, 3.25)
    assert M.mirror_point(M.mirror_point((1.5, 2.0, -3.25))) == (1.5, 2.0, -3.25)
    for bad in ((1.0, 2.0), "abc", (1.0, 2.0, float("nan")), (1.0, 2.0, True), None):
        for rule in (M.mirror_point, M.mirror_world):
            with pytest.raises(M.MirrorError):
                rule(bad)
    with pytest.raises(M.MirrorError, match="outside f32"):
        M.mirror_world((0, 0, 1e39))                          # finite, but -1 - z does not fit an f32


def test_mirror_world_rounds_into_the_mirrored_cell():
    """mirror_world: (x, y, -1 - z) in f32, the mirror in PLANE z = -0.5; mirrored twice z comes back within an ulp
    (exactly where -1 - z is exact). Every z but an exact half (k + 0.5, which both sides round to even) lands in the
    mirrored row of the ground grid (`tgp.cell`, 128 rows) and of the scenery grid (0x172a0: (round(z) + 1024) >> 7,
    16 rows), cell edges included; about z = 0 (`mirror_point`) a z rounding onto an edge lands one row off."""
    assert M.mirror_world((1.5, 2.0, -3.25)) == (1.5, 2.0, 2.25)
    assert M.mirror_world((0.0, 0.0, -0.5)) == (0.0, 0.0, -0.5)          # the plane maps to itself
    assert M.mirror_world(M.mirror_world((7.0, 1.0, 150.49411010742188))) == (7.0, 1.0, 150.49411010742188)
    tiny = struct.unpack("<f", struct.pack("<f", 0.1))[0]
    back = M.mirror_world(M.mirror_world((0.0, 0.0, tiny)))[2]
    assert back != tiny and abs(back - tiny) < 1e-7                      # rounded: within an ulp of -1 - z
    scenery_row = lambda z: (round(z) + 1024) >> 7                     # noqa: E731
    plain_off = 0
    halves = [16 * k - 0.5 for k in range(-60, 60)]
    assert any(tgp.cell(0.0, M.mirror_world((0.0, 0.0, z))[2]) >> 7 != 127 - (tgp.cell(0.0, z) >> 7) for z in halves)
    for z in [k / 4 for k in range(-4000, 4000) if k % 4 != 2] + [16 * k + d for k in range(-60, 60)
                                                                  for d in (-0.49, -0.3, 0.3, 0.49)]:
        w = M.mirror_world((0.0, 0.0, z))[2]
        assert tgp.cell(0.0, w) >> 7 == 127 - (tgp.cell(0.0, z) >> 7), z
        assert scenery_row(w) == 15 - scenery_row(z), z
        plain_off += tgp.cell(0.0, -z) >> 7 != 127 - (tgp.cell(0.0, z) >> 7)
    assert plain_off > 0


@pytest.mark.parametrize("rule, values", [
    (M.mirror_heading, (-0x8000, -17750, -5462, 0, 0x1fff, 0x2000, 10922, 0x7fff)),
    (M.mirror_direction, (-0x8000, -32515, -1, 0, 0x3fff, 0x4000, 0x7fff)),
    (M.mirror_angle, (-0x8000, -29128, -1, 0, 1, 26031, 0x7fff)),
    (M.mirror_signed, (-0x8000, -5461, 0, 8738, 0x7fff)),
    (M.mirror_lane, (0, 0x17f, 0x180, 0x1ff, 0x200, 0x380, 0x57f, 0x580, 0x6ff, 0x700)),
])
def test_mirror_rules_are_involutions(rule, values):
    """Each integer rule maps an s16 to an s16 and undoes itself."""
    for v in values:
        w = rule(v)
        assert -0x8000 <= w <= 0x7FFF and rule(w) == v, (rule.__name__, v, w)
    with pytest.raises(M.MirrorError):
        rule(0x8000)
    with pytest.raises(M.MirrorError):
        rule(1.0)


def test_mirror_rules_values():
    """The format doc's values: heading h -> 0x2000 - h (mod 0x4000: the yaw 4h becomes 0x8000 - 4h), direction
    d -> 0x8000 - d (mod 0x10000), angle a -> -a, lane slot k -> 7 - k; lane l = k.f (8.8, blended by f / 255,
    `course.lane_position`) -> 0x6ff - l, i.e. (6 - k).(255 - f) (MAME, Phase 4 Task 5: 0x700 - l put every car
    1/255 lane off its mirrored place, and read its pitch and roll from the wrong lane slot)."""
    for h in (-5462, 0, 10922):
        assert (M.mirror_heading(h) - (0x2000 - h)) % 0x4000 == 0
        assert (4 * M.mirror_heading(h)) % 0x10000 == (0x8000 - 4 * h) % 0x10000
    assert M.mirror_heading(-5462) == 0x2000 + 5462          # continuous: no wrap where the s16 holds it
    assert M.mirror_direction(-21845) == struct.unpack("<h", struct.pack("<H", (0x8000 + 21845) & 0xFFFF))[0]
    assert M.mirror_angle(-0x8000) == -0x8000                 # a half turn is its own mirror
    assert (M.mirror_lane(0x180), M.mirror_lane(0x580)) == (0x57F, 0x17F)
    assert (M.mirror_lane(0x100), M.mirror_lane(0x400), M.mirror_lane(0x1FF), M.mirror_lane(0)) == (0x5FF, 0x2FF,
                                                                                                    0x500, 0x6FF)
    assert [M.mirror_lane_slot(k) for k in range(C.LANES)] == list(range(C.LANES - 1, -1, -1))


def _attitude(p: C.RoadPoint, lane: int) -> C.Lane:
    """The lane point whose pitch, heading and roll 0x140f0 hands a car at `lane` (0x14334-0x14340): slot
    (lane >> 8) + 1 clamped to lane_lo..lane_hi, not blended."""
    return p.lanes[min(max((lane >> 8) + 1, p.lane_lo), p.lane_hi)]


def test_mirror_lane_places_the_mirrored_point(beginner):
    """The game's placement 0x140f0 on the mirrored road at the mirrored lane is the mirror of the original's: the
    point (`course.lane_position`: lanes k and k + 1, clamped, blended by f / 255) for every lane value 0..0x7ff on
    every road point, every AI point's lane and every grid lane; the pitch, heading and roll (`_attitude`, slot
    k + 1) wherever the original's lane is at or above lane_lo. Below it (an AI lane under 1.0 on Beginner) the
    original reads slot lane_lo and the mirror slot 7 - lane_lo, which holds lane_lo + 1's (no slot can serve both:
    `mirror_road_point`)."""
    _, c, m = beginner
    n = len(c.road)                         # one road point per segment, as the AI lines
    pairs = [(seg, lane) for seg in range(n) for lane in range(0, 0x800, 7)]
    pairs += [(seg % n, p.lane) for line in c.ai_lines for seg, p in enumerate(line)]
    pairs += [(s.segment, s.lane) for s in c.spawn[1]]
    below = 0
    for seg, lane in pairs:
        p, q, mlane = c.road[seg], m.road[seg], M.mirror_lane(lane)
        want = M.mirror_world(C.lane_position(p, lane))
        got = C.lane_position(q, mlane)
        assert max(abs(a - b) for a, b in zip(want, got)) < 1e-4, (seg, hex(lane), want, got)
        a, b = _attitude(p, lane), _attitude(q, mlane)
        if lane >> 8 < p.lane_lo:
            below += 1
            a = p.lanes[p.lane_lo + 1]
        assert (b.pitch, b.heading, b.roll) == (a.pitch, M.mirror_heading(a.heading), -a.roll), (seg, hex(lane))
    assert below


def test_mirror_ground_synthetic():
    """Every polygon keeps the originals' winding (negative XZ area: ground_tables accepts it), the centre and the
    direction are mirrored, the grid rows move r -> 127 - r and the lists are kept."""
    g = _ground(polys=(_sq(0, 2, s=6), _sq(8, 2, s=6, d=-5), _sq(16, 2, s=6, y=2.5, d=0x4000),
                       _sq(16, 9, s=6, d=-0x8000), _sq(24, 2, s=6)))        # no polygon within 1 of a row edge
    mg = M.mirror_ground(g)
    tgp.ground_tables(mg)
    for p, q in zip(g.polys, mg.polys):
        assert q.corners == tuple(M.mirror_world(p.corners[k]) for k in M.QUAD_ORDER)
        assert q.centre == M.mirror_world(p.centre) and q.direction == M.mirror_direction(p.direction)
    for cell, v in enumerate(g.grid):
        assert mg.grid[(127 - (cell >> 7)) * 128 + (cell & 127)] == v
    assert mg.lists == g.lists and M.mirror_ground(mg) == g


def test_mirror_ground_triangle_keeps_its_degenerate_corners():
    """A triangle (corner 3 = corner 2, as on every original) mirrors with QUAD_ORDER to corners 1 = 2, where the
    TGP's alpha / beta solve (corners 0-2) has no extent; TRIANGLE_ORDER keeps the repeat at corners 2-3 and corner 2
    as the solve's apex, so a point inside answers the triangle's height."""
    a, b, c = (0.0, 1.0, 0.0), (0.0, 3.0, 8.0), (8.0, 5.0, 8.0)
    tri = tgp.GroundPoly((a, b, c, c), centre=(8 / 3, 3.0, 16 / 3), direction=0)
    g = dataclasses.replace(_ground(), polys=(tri,) + _ground().polys[1:])
    mg = M.mirror_ground(g)
    assert mg.polys[0].corners == tuple(M.mirror_world(tri.corners[k]) for k in M.TRIANGLE_ORDER)
    assert mg.polys[0].corners[2] == mg.polys[0].corners[3]
    for x, z in ((2.0, 5.0), (1.0, 7.0), (6.0, 7.5)):
        want = tgp.query(g, x, 3.0, z)
        assert want is not None and want[1] == 0
        got = tgp.query(mg, x, 3.0, -1.0 - z)
        assert got[1] == 0 and abs(got[0] - want[0]) < 1e-4
    assert M.mirror_ground(M.mirror_ground(mg)).polys == mg.polys


def test_mirror_ground_edge_rows_permuted():
    """Review Focus 4, synthetic: a polygon of row 65 (z 16-24) whose low edge is the row's edge z = 16, listed only
    in row 65. A point at z 16.3 rounds to 16, row 65, and hits it. Mirrored in PLANE (z -17.3 rounds to -17) the
    point is in row 62 = 127 - 65, which holds row 65's list: the rows permuted, the lists kept, every point near the
    edge answers the same. About z = 0 the point (z -16.3, rounding to -16) would be in row 63 and miss."""
    grid = [0] * tgp.GRID_CELLS
    grid[_cell(64, 64)] = 1                     # row 64 (z 0-16): polygon 0
    grid[_cell(65, 64)] = 3                     # row 65 (z 16-32): polygon 1 only
    g = _ground(grid=tuple(grid), polys=(_sq(0, 0, s=15.0), _sq(0, 16, y=2.0)), attributes=bytes([1, 1]),
                lists={1: (0,), 3: (1,)})
    assert tgp.query(g, 4.0, 2.0, 16.3)[1] == 1
    mg = M.mirror_ground(g)
    assert mg.lists == g.lists and mg.raw == g.raw
    assert mg.grid[_cell(63, 64)] == 1 and mg.grid[_cell(62, 64)] == 3
    assert tgp.query(mg, 4.0, 2.0, -16.3) is None                 # the plain negation's point: one row off
    for x, z in ((4.0, 8.0), (4.0, 20.0), (4.0, 15.4), (4.0, 15.51), (4.0, 16.0), (4.0, 16.3), (4.0, 16.49)):
        assert tgp.query(mg, x, 2.0, M.mirror_world((x, 2.0, z))[2]) == tgp.query(g, x, 2.0, z), (x, z)
    assert M.mirror_ground(mg) == g                               # an involution (the corners are exact here)


def test_mirror_course_refuses_non_courses():
    for bad in (None, "course", 3):
        with pytest.raises(M.MirrorError, match="not a golden.Course"):
            M.mirror_course(bad)
    with pytest.raises(M.MirrorError):
        M.mirror_ground("ground")


def test_mirror_partial_refused_synthetic():
    """Review Focus 1: a Course with table fields unset is refused, naming them (encode_course(c, rs) would fill
    them from the ROM, unmirrored)."""
    c = _race()                                   # race tables only: no ground, scenery, objects, ...
    with pytest.raises(M.MirrorError, match=r"unset.*ground_data") as e:
        M.mirror_course(c)
    for name in ("objects", "cells", "backdrop"):
        assert name in str(e.value)
    assert "billboards" not in str(e.value) and "switch_lists" not in str(e.value)


# --- the real set: Beginner ----------------------------------------------------------------------------------------

def test_mirror_partial_refused(beginner):
    """Review Focus 1 on the real set: Beginner with any one table field (or filled view) back at its default is
    refused naming it. The optional fields: billboards and billboard_ref (empty for real on Expert and Advanced) are
    refused too while the course has a view-angle billboard (the mirror reorders its variants and mirrors its
    reference point, Task 7); switch_lists, which no rule changes, is not, and an empty one encodes as the mirror of
    the complete Course (complete fills the ROM's value, which is the mirrored value)."""
    rs, c, m = beginner
    for name in G.TABLE_FIELDS + G.FILLED_VIEWS:
        if name in G.OPTIONAL_FIELDS or name == "id_arrays":     # empty = the original arrays (golden.unset_fields)
            continue
        default = next(f for f in dataclasses.fields(G.Course) if f.name == name)
        value = default.default_factory() if default.default_factory is not dataclasses.MISSING else default.default
        with pytest.raises(M.MirrorError, match=rf"unset.*\b{name}\b"):
            M.mirror_course(dataclasses.replace(c, **{name: value}))
    for name, value in (("billboards", {}), ("billboard_ref", None)):
        with pytest.raises(M.MirrorError, match=rf"unset.*\b{name}\b"):
            M.mirror_course(dataclasses.replace(c, **{name: value}))
    empty = M.mirror_course(dataclasses.replace(c, switch_lists=()))
    assert empty.switch_lists == ()
    assert G.encode_course(empty, rs) == G.encode_course(m, rs)


def test_mirror_malformed_refused(beginner):
    """Malformed tables raise MirrorError (a DecodeError), never TypeError or ValueError."""
    _, c, _ = beginner
    p = c.road[0]
    bad = {
        "road point": dataclasses.replace(c, road=(dataclasses.replace(p, lanes=p.lanes[:3]),) + c.road[1:]),
        "lane heading": dataclasses.replace(c, road=(dataclasses.replace(p, lanes=(p.lanes[0]._replace(
            heading=0x8000),) + p.lanes[1:]),) + c.road[1:]),
        "actor": dataclasses.replace(c, actor_lists={**c.actor_lists, next(iter(c.actor_lists)): ("junk",)}),
        "object": dataclasses.replace(c, objects={**c.objects, next(iter(G.scenery_objects(c))): "junk"}),
        "ground": dataclasses.replace(c, ground_data=dataclasses.replace(c.ground_data, polys="junk")),
        "actor lists not a dict": dataclasses.replace(c, actor_lists=(1, 2)),
        "no default time table": dataclasses.replace(c, time_tables={
            k: v for k, v in c.time_tables.items() if k != (C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)}),
        "no default laps": dataclasses.replace(c, laps={k: v for k, v in c.laps.items() if k != C.DEFAULT_MODE}),
    }
    for what, course in bad.items():
        with pytest.raises(M.MirrorError):
            M.mirror_course(course)


def test_mirror_involution(beginner):
    """mirror_course undoes itself on Beginner, up to two things: world z mirrored twice comes back within an f32 ulp
    where -1 - z rounds (`mirror_world`; every value within 1e-4 here), and the road's slot lane_lo comes back with
    lane_lo + 1's pitch, heading and roll (`mirror_road_point`: no mirrored slot holds lane_lo's). The ground's grid,
    lists, list block and attributes come back exactly (the rows are permuted, nothing is rebuilt), and a third
    mirror is the first again within the ulps."""
    rs, c, m = beginner
    mm = M.mirror_course(m)
    assert [f.name for f in dataclasses.fields(G.Course) if f.name not in ("road", "texels", "ground_data", "backdrop",
                                                                            "char_blocks")
            and not _close(getattr(mm, f.name), getattr(c, f.name))] == []
    # the backdrop: its segments and char RAM come back; its chars sit in new blocks in DATA_FREE (_backdrop)
    assert mm.backdrop.segments == c.backdrop.segments
    assert mm.backdrop.uploads._replace(chars=()) == c.backdrop.uploads._replace(chars=())
    assert [(u.dst, u.count) for u in mm.backdrop.uploads.chars] == [(u.dst, u.count) for u in c.backdrop.uploads.chars]
    assert _eq(mm.backdrop.chars(mm.char_blocks), c.backdrop.chars(c.char_blocks))
    backdrop_blocks = {u.src for u in c.backdrop.uploads.chars}
    assert _eq({k: v for k, v in mm.char_blocks.items() if k not in {u.src for u in mm.backdrop.uploads.chars}},
               {k: v for k, v in c.char_blocks.items() if k not in backdrop_blocks})
    lost = 0                    # the road: lane_lo's pitch and roll come back as lane_lo + 1's (mirror_road_point)
    for p, q in zip(c.road, mm.road):
        for k in range(C.LANES):
            a = p.lanes[p.lane_lo + 1] if k == p.lane_lo else p.lanes[k]
            assert _close(q.lanes[k], p.lanes[k]._replace(pitch=a.pitch, heading=a.heading, roll=a.roll))
        lost += any((u.pitch, u.heading, u.roll) != (v.pitch, v.heading, v.roll) for u, v in zip(p.lanes, q.lanes))
        assert dataclasses.replace(q, lanes=p.lanes) == p
    assert lost == M_ROAD_LOST
    g, gg = c.ground_data, mm.ground_data
    assert (gg.grid, gg.lists, gg.raw, gg.attributes) == (g.grid, g.lists, g.raw, g.attributes)
    assert all(_close(dataclasses.replace(p, raw=()), dataclasses.replace(q, raw=())) for p, q in zip(g.polys, gg.polys))
    zs = [(a[2], b[2]) for p, q in zip(g.polys, gg.polys) for a, b in zip(p.corners, q.corners)]
    assert sum(a != b for a, b in zs) == M_GROUND_ULPS and max(abs(a - b) for a, b in zs) < 1e-4
    m3 = M.mirror_course(mm)
    assert _close(dataclasses.replace(m3, ground_data=None), dataclasses.replace(m, ground_data=None))   # same blocks
    assert all(_close(dataclasses.replace(p, raw=()), dataclasses.replace(q, raw=()))
               for p, q in zip(m3.ground_data.polys, m.ground_data.polys))
    a, b = G._encode(mm), G._encode(c)
    names = {(r, o): n for r, o, _, n in G.tables(c)}
    differ = {names.get(k, "?") for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert not differ & {"ground grid", "ground lists", "ground attributes", "billboard lists",
                         "billboard reference", "backdrop segment"}, differ


M_GROUND_ULPS = 74        # Beginner's ground corner z values (of 4,736) mirrored twice that are not bit-exact
M_ROAD_LOST = 90          # Beginner's road points whose lane_lo attitude differs from lane_lo + 1's (of 196)


def test_mirrored_set_round_trip(beginner):
    """The built set decodes to exactly mirror_course(decode_course(rs, 0)) and re-emits it byte for byte (every
    table of the mirrored Course, as verify_lossless compares them)."""
    rs, c, m = beginner
    built = _built()
    d = G.decode_course(built, 0)
    assert _differ(d, m) == []
    assert G.mismatches(built, m) == []
    assert G.verify_lossless(built, 0) == []
    assert built.copro_data[:len(built.copro_data) // 2] == built.copro_data[len(built.copro_data) // 2:]


def test_mirror_leaves_other_courses(beginner):
    """Review Focus 2: Expert's and Advanced's tables re-emit from the built set except GRID_LANES (shared by every
    course; swapped in this build), and no car model (shared by every course) changes a byte."""
    rs, c, m = beginner
    built = _built()
    cars = set(c.car_objects)
    for k in (1, 2):
        other = G.decode_course(rs, k)
        lines = G.mismatches(built, other)
        assert len(lines) == 1 and "grid lanes at program+0x31b30" in lines[0], lines     # GRID_LANES, window
        cars |= other.car_objects
    objects = {**c.objects, **{k: o for i in (1, 2) for k, o in G.decode_course(rs, i).objects.items()}}
    for key in cars:
        span = slice(4 * key, 4 * (key + objects[key].word_count))
        assert built.polygons[span] == rs.polygons[span], key
    assert C.grid_lanes(built) == (0x57F, 0x17F) and C.grid_lanes(rs) == (0x180, 0x580)


def test_mirror_once(beginner):
    """Review Focus 3: every scenery object is mirrored exactly once (a second negation would restore its bytes):
    the objects whose bytes changed in the built set are exactly the scenery objects that are not their own mirror
    (none is: every object has a z). A Course naming one object from two cell lists mirrors it once."""
    rs, c, m = beginner
    built = _built()
    scenery = G.scenery_objects(c)
    local = M._local_objects(c, c.name)
    symmetric = {k for k, o in scenery.items()
                 if P.encode_object(M.mirror_object(o, k not in local)) == P.encode_object(o)}
    changed = {k for k, o in c.objects.items()
               if built.polygons[4 * k:4 * (k + o.word_count)] != rs.polygons[4 * k:4 * (k + o.word_count)]}
    assert changed == set(scenery) - symmetric
    assert len(changed) == len(scenery) == 325 and not symmetric
    for k in scenery:
        assert m.objects[k] == M.mirror_object(c.objects[k], k not in local)
        assert [r.attr for r in m.objects[k].records] == [r.attr for r in c.objects[k].records]   # bit 17 kept
    for k in c.car_objects:
        assert m.objects[k] is c.objects[k]

    # one object reached from two cell lists
    lists = sorted({p for p in c.cells if c.object_lists[p]})
    a = lists[0]
    shared = next(p for p in c.object_lists[a] if C.check_rom_object(p.oba) in scenery)
    key = C.check_rom_object(shared.oba)
    b = next(p for p in lists if all(C.check_rom_object(q.oba) != key for q in c.object_lists[p]))
    twice = dataclasses.replace(c, object_lists={**c.object_lists, b: c.object_lists[b] + (shared,)})
    mt = M.mirror_course(twice)
    assert mt.objects[key] == M.mirror_object(c.objects[key], key not in local) != c.objects[key]


def test_mirror_actor_objects_local(beginner):
    """Beginner's two STATIC_PROP actors (0x220118: drawn under V * T(position) * Ry(angle)) draw objects 875514 and
    875859 from their lists: those points are object-local, so z is plainly negated (the actor's position carries
    the plane); the other 323 scenery objects are drawn in world coordinates and take -1 - z. No object is drawn
    both ways, and a Course naming one both ways is refused."""
    _, c, m = beginner
    props = sorted((a.position, a.angle, a.obj) for v in c.actor_lists.values() for a in v
                   if a.handler == C.STATIC_PROP)
    assert [(p, a) for p, a, _ in props] == [((-119.0, 2.5, -101.0), -29128), ((314.0, 8.0, 99.80000305175781), 0)]
    local = M._local_objects(c, c.name)
    assert local == {875514, 875859} == {C.check_rom_object(q.oba) for _, _, o in props for q in c.object_lists[o]}
    for k in local:
        o, mo = c.objects[k], m.objects[k]
        assert mo == M.mirror_object(o, world=False) != M.mirror_object(o)
        assert mo.seed0 == M.mirror_point(o.seed0) and all(
            q.p0 == M.mirror_point(r.p0) for r, q in zip(o.records, mo.records))
    for key, v in c.actor_lists.items():
        for a, b in zip(v, m.actor_lists[key], strict=True):
            assert b.position == M.mirror_world(a.position) and b.angle == M.mirror_angle(a.angle)
    world_list = next(p for p in c.cells if p and c.object_lists[p])
    shared = c.object_lists[world_list][0]
    prop_list = props[0][2]
    both = dataclasses.replace(c, object_lists={**c.object_lists, prop_list: c.object_lists[prop_list] + (shared,)})
    with pytest.raises(M.MirrorError, match="both in world coordinates and under an actor"):
        M.mirror_course(both)


def test_mirror_road_side(beginner):
    """The mirrored road line: lane k's position is the original's lane 7 - k mirrored (`mirror_world`), lane_lo /
    lane_hi follow; its pitch, heading and roll are lane 8 - k's (lane_hi's on the new lane_lo: `mirror_road_point`); the
    heading still points along the road (its (-sin, cos) of 4h against the next point) and the used lanes still grow
    to the same side of travel."""
    _, c, m = beginner

    def travel(p):                          # the direction of travel: (-sin, cos) of the yaw 4 * heading, in (x, z)
        a = (4 * p.heading & 0xFFFF) / 0x10000 * 2 * math.pi
        return -math.sin(a), math.cos(a)

    def checks(road, i):
        p, nxt = road[i], road[(i + 1) % len(road)]
        d = travel(p)
        ahead = (nxt.centre[0] - p.centre[0], nxt.centre[2] - p.centre[2])
        lo, hi = p.lanes[p.lane_lo].position, p.lanes[p.lane_hi].position
        return d[0] * ahead[0] + d[1] * ahead[1], d[0] * (hi[2] - lo[2]) - d[1] * (hi[0] - lo[0])

    for i, (p, q) in enumerate(zip(c.road, m.road)):
        assert (q.lane_lo, q.lane_hi) == (7 - p.lane_hi, 7 - p.lane_lo)
        assert q.curve == -p.curve
        for k in range(C.LANES):
            assert q.lanes[k].position == M.mirror_world(p.lanes[7 - k].position), (i, k)
            a = p.lanes[min(8 - k, p.lane_hi) if q.lane_lo <= k <= q.lane_hi else 7 - k]
            assert q.lanes[k].heading == M.mirror_heading(a.heading), (i, k)
            assert (q.lanes[k].roll, q.lanes[k].pitch, q.lanes[k].pad) == (-a.roll, a.pitch, p.lanes[7 - k].pad)
        (along, side), (m_along, m_side) = checks(c.road, i), checks(m.road, i)
        assert along > 0 and m_along > 0, i            # the heading points along the road, mirrored too
        assert side * m_side > 0, i                     # lane_lo -> lane_hi lies to the same side of travel
    for line, mine in zip(c.ai_lines, m.ai_lines):
        for p, q in zip(line, mine):
            assert q.position == M.mirror_world(p.position) and q.lane == M.mirror_lane(p.lane)
            assert (q.roll, q.steer, q.pitch, q.speed, q.step) == (-p.roll, -p.steer, p.pitch, p.speed, p.step)
    assert m.grid_lanes == (0x57F, 0x17F)
    G.check_course(m)                       # the views (grid slots, checkpoints) were recomputed from the sources


def _edge_distance(g: tgp.Ground, poly: int, x: float, z: float) -> float:
    """XZ distance from (x, z) to the nearest edge of polygon `poly` of `g` (original coordinates)."""
    cs = g.polys[poly].corners
    best = math.inf
    for k in range(4):
        (ax, _, az), (bx, _, bz) = cs[k], cs[(k + 1) % 4]
        dx, dz = bx - ax, bz - az
        n = dx * dx + dz * dz
        t = 0.0 if n == 0 else min(1.0, max(0.0, ((x - ax) * dx + (z - az) * dz) / n))
        best = min(best, math.hypot(x - ax - t * dx, z - az - t * dz))
    return best


EDGE_EPS = 0.01


def test_mirror_ground_queries(beginner, capsys):
    """Review Focus 4: the mirrored ground answers tgp.query at the mirrored point (x, y, -1 - z: `mirror_world`) as
    the original at (x, y, z) (the same polygon, the height within 1e-4) for every used lane point of the road line,
    every polygon centre, a 4-unit lattice offset by 0.37 over the ground's extent and every lattice point on a
    16-unit edge, the cell edges included: mirrored in PLANE the game's rounding puts a point in exactly the mirrored
    row, so the permuted rows answer it.

    The exceptions are points within EDGE_EPS of a polygon's edge (most of them road lane points, which are road
    polygon corners): a reversed winding changes the f32 rounding of the TGP's edge test (a point on a shared edge
    picks the other polygon; heights equal within 1e-4) and moves the plane test's boundary from edge 1-2 to edge
    2-3 (a point on that edge of a non-planar quad takes the other triangle's plane, as the original does on edge
    1-2). Each such point is checked to be within EDGE_EPS of a polygon either answer names, and every one is a hit
    on both sides (no point falls through)."""
    rs, c, m = beginner
    g, mg = c.ground_data, m.ground_data
    inv = tgp.inv_table(rs)
    pts = [("road", lane.position) for p in c.road for lane in p.used_lanes]
    pts += [("centre", p.centre) for p in g.polys]
    xs = [q[0] for p in g.polys for q in p.corners]
    zs = [q[2] for p in g.polys for q in p.corners]
    x0, x1, z0, z1 = math.floor(min(xs) / 16) * 16, max(xs), math.floor(min(zs) / 16) * 16, max(zs)
    pts += [("lattice", (x + 0.37, 0.0, z + 0.37)) for x in np.arange(x0, x1, 4.0) for z in np.arange(z0, z1, 4.0)]
    pts += [("edge", (x, 0.0, z)) for x in np.arange(x0, x1, 4.0) for z in np.arange(z0, z1, 16.0)]
    pts += [("edge", (x, 0.0, z)) for x in np.arange(x0, x1, 16.0) for z in np.arange(z0, z1, 4.0)]
    counts, off = {}, {}
    on_row_edge = 0
    for kind, (x, y, z) in pts:
        x, y, z = float(x), float(y), float(z)
        want, got = tgp.query(g, x, y, z, inv), tgp.query(mg, x, y, M.mirror_world((x, y, z))[2], inv)
        counts[kind] = counts.get(kind, 0) + 1
        on_row_edge += (round(z) + 1024) % 16 == 0 and want is not None
        if want == got or (want and got and want[1] == got[1] and abs(want[0] - got[0]) <= 1e-4):
            continue
        assert want is not None and got is not None, (kind, x, y, z, want, got)
        near = min(_edge_distance(g, i, x, z) for i in {want[1], got[1]})
        assert near < EDGE_EPS, (kind, x, y, z, want, got, near)
        off[kind] = off.get(kind, 0) + 1
    with capsys.disabled():
        print(f"\nmirror ground queries: {counts} points ({on_row_edge} hits on a row edge); "
              f"on a polygon edge, answered differently: {off}")
    assert off == {"road": 129, "edge": 15}       # measured (docs/spike/2026-11-mirror.md); a corner-order or plane
    #                                                  change moves it


def test_mirror_ground_lossless(beginner):
    """mirror_ground's output passes ground_tables and fits in place: every block (grid, attributes, slot words,
    pool, list block) keeps its place and size, and the list block is the original's word for word."""
    rs, c, m = beginner
    g, mg = c.ground_data, m.ground_data
    before = {n: (r, o, len(b)) for r, o, b, n in tgp.ground_tables(g)}
    after = {n: (r, o, len(b)) for r, o, b, n in tgp.ground_tables(mg)}
    assert after == before
    assert mg.raw == g.raw and mg.lists == g.lists
    assert len(mg.polys) == len(g.polys) and mg.attributes == g.attributes


def test_mirror_hud_map(beginner):
    """The HUD map (docs/spike/2026-11-mirror.md, "HUD map"): the outline is drawn in world coordinates, in the frame
    the cars' markers are placed in by their world (x, z), so its points are mirrored as world points (z -> -1 - z,
    the axis that carries world z: horizontal on screen) and its normals as directions; the list, the place record
    and the zooms are unchanged. The built set decodes to the mirrored map; Expert's and Advanced's are untouched.
    The outline is reached from no other table (mirrored once)."""
    rs, c, m = beginner
    h, mh = c.hud_map, m.hud_map
    assert (mh.course, mh.outline_list, mh.entries, mh.position, mh.angle, mh.pad, mh.zooms) == \
        (h.course, h.outline_list, h.entries, h.position, h.angle, h.pad, h.zooms)
    assert set(mh.objects) == set(h.objects) and not set(h.objects) & set(c.objects)
    for k, o in h.objects.items():
        mo = mh.objects[k]
        assert mo == M.mirror_object(o, True) != o
        for a, b in zip([o.seed0, o.seed1] + [q for r in o.records for q in (r.p0, r.p1)],
                        [mo.seed0, mo.seed1] + [q for r in mo.records for q in (r.p0, r.p1)]):
            assert (a is None) == (b is None)
            if a is not None:
                assert b[:2] == a[:2] and b[2] == np.float32(-1.0 - a[2])
        assert [r.normal for r in mo.records] == [None if r.normal is None else M.mirror_point(r.normal)
                                                  for r in o.records]
        assert [r.attr for r in mo.records] == [r.attr for r in o.records]
        zs = [q[2] for r in o.records for q in (r.p0, r.p1) if q is not None]
        mzs = [q[2] for r in mo.records for q in (r.p0, r.p1) if q is not None]
        assert math.isclose(min(mzs), -1 - max(zs), abs_tol=1e-4) and math.isclose(max(mzs), -1 - min(zs), abs_tol=1e-4)
    built = _built()
    assert C.hud_map(built, c.entry) == mh
    for k in (1, 2):
        e = C.course_table(rs)[k]
        assert C.hud_map(built, e) == C.hud_map(rs, e)


def test_mirror_billboards(beginner):
    """Beginner's view-angle billboard (actor at (162.3, 39.5, -224.1), 64 distinct variant lists at 0x235bc0): the
    variants in reverse order (variant k -> 63 - k) and its reference point BILLBOARD_REF mirrored as a world point
    ((210, 0, -380) -> (210, 0, 379)): the update 0x221e60 shows variant ((-a) >> 5) & 63 of the camera's angle a from
    that point, which the mirror negates (docs/spike/2026-11-mirror.md, "Billboards"). Both rules undo themselves;
    the built set holds them; Expert and Advanced have no billboard (and no reference point in their Course)."""
    rs, c, m = beginner
    assert list(c.billboards) == [0x235BC0] and len(set(c.billboards[0x235BC0])) == C.BILLBOARD_VARIANTS
    assert [a.obj for v in c.actor_lists.values() for a in v if a.handler == C.VIEW_BILLBOARD] == [0x235BC0]
    lists = c.billboards[0x235BC0]
    assert m.billboards == {0x235BC0: tuple(lists[63 - k] for k in range(64))}
    assert M.mirror_billboard_lists(M.mirror_billboard_lists(lists)) == lists
    assert c.billboard_ref == (210.0, 0.0, -380.0) and m.billboard_ref == (210.0, 0.0, 379.0)
    built = _built()
    assert C.billboard_lists(built, 0x235BC0) == m.billboards[0x235BC0] and C.billboard_ref(built) == m.billboard_ref
    for k in (1, 2):
        other = G.decode_course(rs, k)
        assert other.billboards == {} and other.billboard_ref is None
        assert not [a for v in other.actor_lists.values() for a in v if a.handler == C.VIEW_BILLBOARD]
    with pytest.raises(M.MirrorError):
        M.mirror_billboard_lists(lists[:63])


# --- the backdrop (Phase 4 Task 7) ------------------------------------------------------------------------------

FLIPPED_CHARS = 3312
"""Beginner's panorama: 3,327 chars, 11 symmetric (their own flip), 4 whose flip is another char of the same palette
block (the tiles swap to it), the other 3,312 flipped into new chars (docs/spike/2026-11-mirror.md, "Backdrop")."""


def _synthetic_backdrop():
    """A two-row backdrop over four chars of palette block 0x60 (0x3000-0x3003) and one of 0x61 (0x3080): 0x3000 is
    symmetric, 0x3002 is 0x3001 flipped, 0x3003 is asymmetric, 0x3080 is 0x3003 flipped but in another palette."""
    from dtools.arcade import tiles as TL
    a = np.arange(64, dtype=np.uint8).reshape(8, 8) % 16
    chars = {0x3000: np.full((8, 8), 5, np.uint8), 0x3001: a, 0x3002: a[:, ::-1].copy(),
             0x3003: (a * 3 % 16).astype(np.uint8), 0x3080: (a * 3 % 16).astype(np.uint8)[:, ::-1].copy()}
    pattern = [0x3000, 0x3001, 0x3002, 0x3003, 0x3080, 0x3003]
    tiles = np.array([[pattern[(q + r) % 6] for q in range(256)] for r in range(2)], np.uint16)
    segs = tuple(TL.BackdropSegment(0x10000, 2, 32, tuple(int(t) for t in tiles[:, 32 * k:32 * k + 32].ravel()))
                 for k in range(8))
    up = TL.UploadSet(0x2074240, 0x2074300, 0x2074400, (TL.CharUpload(0x2100000, 0x10E0000, 4),
                                                        TL.CharUpload(0x2100100, 0x10E1000, 1)), (), tuple(range(8)))
    return C.Backdrop(0, up, segs), chars


def test_mirror_backdrop_synthetic():
    """The rule on a synthetic backdrop: column q of the mirror is column (MIRROR_COLUMN - q) & 255 of the original,
    each tile's char flipped left to right in the same palette block (the tile's palette follows from its char
    index); a symmetric char is kept, a char whose flip is another char of its block takes that char, every other
    flip is a new char at a free index of the block; the rule undoes itself; MirrorError when a block has no room."""
    b, chars = _synthetic_backdrop()
    mb, new = M.mirror_backdrop(b, chars)
    assert mb.uploads == b.uploads and [s.rows for s in mb.segments] == [2] * 8
    pan, mpan = b.panorama(), mb.panorama()
    ram = {**chars, **new}
    for r in range(2):
        for q in range(256):
            t, mt = int(pan[r, (M.MIRROR_COLUMN - q) & 255]), int(mpan[r, q])
            assert mt >> 7 == t >> 7 and np.array_equal(ram[mt], chars[t][:, ::-1]), (r, q)
    assert sorted(new) == [0x3003, 0x3080]           # 0x3003 -> flip(0x3003), 0x3080 -> flip(0x3080), in place
    assert {int(v) for v in mpan.ravel()} == set(chars)
    assert int(mpan[0, M.MIRROR_COLUMN & 255]) == 0x3000          # the symmetric char kept
    mmb, new2 = M.mirror_backdrop(mb, ram)
    assert mmb == b and all(np.array_equal({**ram, **new2}[k], v) for k, v in chars.items())
    with pytest.raises(M.MirrorError, match="no free char"):
        M.mirror_backdrop(b, chars, free_char=0x3080)
    with pytest.raises(M.MirrorError):
        M.mirror_backdrop(b, {k: v for k, v in chars.items() if k != 0x3001})       # a tile's char missing
    for bad in (None, "backdrop"):
        with pytest.raises(M.MirrorError):
            M.mirror_backdrop(bad, chars)


def test_mirror_backdrop_involution(beginner):
    """Beginner's backdrop mirrored twice is the original: the segments, the upload set and every char of its char
    RAM; once, column q shows the original's column (MIRROR_COLUMN - q) & 255 flipped, in the same palette, and the
    top and bottom fill tiles stay (Beginner's are uniform)."""
    rs, c, m = beginner
    b = c.backdrop
    chars = b.chars(c.char_blocks)
    mb, new = M.mirror_backdrop(b, chars)
    ram = {**chars, **new}
    pan, mpan = b.panorama(), mb.panorama()
    for q in range(256):
        for r in range(pan.shape[0]):
            t, mt = int(pan[r, (M.MIRROR_COLUMN - q) & 255]), int(mpan[r, q])
            assert mt >> 7 == t >> 7 and np.array_equal(ram[mt & 0x3FFF], chars[t & 0x3FFF][:, ::-1])
    assert (mb.top_tile, mb.bottom_tile) == (b.top_tile, b.bottom_tile)
    mmb, new2 = M.mirror_backdrop(mb, ram)
    assert mmb == b
    back = {**ram, **new2}
    assert back.keys() == chars.keys() and all(np.array_equal(back[k], v) for k, v in chars.items())


def test_flipped_chars_dedup(beginner):
    """The new chars are the distinct flips (per palette block) of the panorama's chars that are neither symmetric
    nor the flip of another char of their block: FLIPPED_CHARS on Beginner, at most 4,096 (chars 0x3000-0x3fff), each
    in the 128-char block of the char it flips, all inside the backdrop's upload ranges."""
    rs, c, m = beginner
    b = c.backdrop
    chars = b.chars(c.char_blocks)
    _, new = M.mirror_backdrop(b, chars)
    used = {int(v) & 0x3FFF for v in np.unique(b.panorama())}
    images = {(k >> 7, chars[k].tobytes()) for k in chars}
    want = {(k >> 7, np.ascontiguousarray(chars[k][:, ::-1]).tobytes()) for k in used}
    want -= images                                   # symmetric, or the flip of another char of the block
    assert len(new) == len(want) == FLIPPED_CHARS <= 4096
    assert {(k >> 7, v.tobytes()) for k, v in new.items()} == want
    assert all(0x3000 <= k <= 0x3FFF for k in new) and set(new) <= set(chars)
    assert sum(np.array_equal(chars[k], chars[k][:, ::-1]) for k in used) == 11


def test_mirror_backdrop_in_course(beginner):
    """mirror_course writes the mirrored backdrop's chars as new char blocks in the data ROM's free run (DATA_FREE),
    one block per upload of the original (same destination and count), and the char list points at them; the
    original blocks are left in the ROM, untouched; the built set decodes to the mirrored char RAM and segments;
    nothing else is written in the free run."""
    rs, c, m = beginner
    b, mb = c.backdrop, m.backdrop
    chars = b.chars(c.char_blocks)
    want_b, new = M.mirror_backdrop(b, chars)
    assert mb.segments == want_b.segments and mb.uploads.addr == b.uploads.addr
    assert mb.uploads.char_list == b.uploads.char_list and mb.uploads.palette == b.uploads.palette
    lo = tgp.DATA_FREE[0] + 0x02000000
    at = lo
    for u, mu in zip(b.uploads.chars, mb.uploads.chars, strict=True):
        assert (mu.src, mu.dst, mu.count) == (at, u.dst, u.count)
        at += 4 + 32 * u.count
    assert at - lo == BACKDROP_FREE_BYTES <= tgp.DATA_FREE[1]
    ram = {**chars, **new}
    got = mb.chars(m.char_blocks)
    assert got.keys() == ram.keys() and all(np.array_equal(got[k], v) for k, v in ram.items())
    assert not {u.src for u in b.uploads.chars} & set(m.char_blocks)
    names = {n for r, o, blob, n in G.tables(m) if r == "data" and o < tgp.DATA_FREE[0] + tgp.DATA_FREE[1]
             and o + len(blob) > tgp.DATA_FREE[0]}
    assert names == {"char block"}
    built = _built()
    d = G.decode_course(built, 0)
    assert d.backdrop == mb and d.backdrop.chars(d.char_blocks).keys() == ram.keys()
    for u in b.uploads.chars:                        # the original blocks stay in the ROM
        assert built.read_i960(u.src, 4 + 32 * u.count) == rs.read_i960(u.src, 4 + 32 * u.count)


BACKDROP_FREE_BYTES = 106476
"""Data-ROM bytes the mirrored Beginner's char blocks take from DATA_FREE: 3,327 chars of 32 bytes in three blocks
(the original's 1,771, 646 and 910 chars) with their 4-byte counts."""


def _char_srcs(c: G.Course) -> list[int]:
    return [u.src for u in c.backdrop.uploads.chars]


def test_mirrored_sets_compose(beginner):
    """Two mirrored_set calls on one set (Beginner, then Advanced) keep each other's char blocks: the second course's
    blocks go to the free run's first free bytes after the first's, and both courses decode (Important 1 of the final
    review: both were written at DATA_FREE's start, and Beginner's char list then read Advanced's chars)."""
    rs, c, m = beginner
    first = _built()
    both = M.mirrored_set(first, 2)
    d0, d2 = G.decode_course(both, 0), G.decode_course(both, 2)
    want = m.backdrop.chars(m.char_blocks)
    got = d0.backdrop.chars(d0.char_blocks)
    assert got.keys() == want.keys() and all(np.array_equal(got[k], want[k]) for k in want)
    lo = tgp.DATA_FREE[0] + 0x02000000
    assert _char_srcs(d0)[0] == lo
    assert _char_srcs(d2)[0] == lo + BACKDROP_FREE_BYTES
    m2 = M.mirror_course(G.decode_course(rs, 2), data_at=tgp.DATA_FREE[0] + BACKDROP_FREE_BYTES)
    got2, want2 = d2.backdrop.chars(d2.char_blocks), m2.backdrop.chars(m2.char_blocks)
    assert got2.keys() == want2.keys() and all(np.array_equal(got2[k], want2[k]) for k in want2)


def test_mirrored_set_twice_reuses_its_blocks(beginner):
    """Mirroring the mirrored Beginner again writes its new char blocks over the first mirror's (bytes that belong to
    the course's own current blocks count as free), and the result decodes to the original's char RAM."""
    rs, c, _ = beginner
    twice = M.mirrored_set(_built(), 0)
    d = G.decode_course(twice, 0)
    assert _char_srcs(d)[0] == tgp.DATA_FREE[0] + 0x02000000
    want, got = c.backdrop.chars(c.char_blocks), d.backdrop.chars(d.char_blocks)
    assert got.keys() == want.keys() and all(np.array_equal(got[k], want[k]) for k in want)


def test_mirrored_set_after_relocate_ground(beginner):
    """A ground relocated to DATA_FREE's start (relocate_ground's default data_at) is not overwritten: the char
    blocks go after it, and the moved ground decodes as before."""
    rs, _, _ = beginner
    moved = tgp.relocate_ground(rs, 2, erase_original=True)
    built = M.mirrored_set(moved, 0)
    assert tgp.decode_ground(built, 2) == tgp.decode_ground(moved, 2)
    g = tgp.decode_ground(moved, 2)
    assert _char_srcs(G.decode_course(built, 0))[0] - 0x02000000 >= g.attributes_at + len(g.attributes)


def test_mirrored_set_refuses_a_used_target(beginner):
    """An explicit data_at whose span holds bytes that are neither free (0xff) nor the course's own char blocks is
    refused, naming the first such byte; so is one outside the free run, and a free run too small for the blocks."""
    rs, _, _ = beginner
    with pytest.raises(M.MirrorError, match="not free"):
        M.mirrored_set(_built(), 2, data_at=tgp.DATA_FREE[0])
    with pytest.raises(M.MirrorError, match="free run"):
        M.mirrored_set(rs, 0, data_at=tgp.DATA_FREE[0] - 4)
    with pytest.raises(M.MirrorError, match="free run"):
        M.mirrored_set(rs, 0, data_at=tgp.DATA_FREE[0] + tgp.DATA_FREE[1] - 4)


def test_mirror_backdrop_overflow(beginner, monkeypatch):
    """A free run too small for the mirrored char blocks is refused, stating the bytes needed and available."""
    _, c, _ = beginner
    monkeypatch.setattr(tgp, "DATA_FREE", (tgp.DATA_FREE[0], 1000))
    with pytest.raises(M.MirrorError, match=f"need {BACKDROP_FREE_BYTES} bytes.*1000"):
        M.mirror_course(c)


def test_mirror_backdrop_new_char_outside_uploads(beginner, monkeypatch):
    """A new char that no upload of the backdrop covers cannot be written (the char list keeps its uploads): refused."""
    _, c, _ = beginner
    real = M.mirror_backdrop

    def stray(b, chars, free_char=0x3000):
        mb, new = real(b, chars, free_char)
        return mb, {**new, 0x3FFF: np.zeros((8, 8), np.uint8)}
    covered = {k for u in c.backdrop.uploads.chars for k in range(u.first_char, u.first_char + u.count)}
    assert 0x3FFF not in covered
    monkeypatch.setattr(M, "mirror_backdrop", stray)
    with pytest.raises(M.MirrorError, match="outside the upload ranges"):
        M.mirror_course(c)


# --- the backdrop in the custom data area (Phase 9 Task 1) ----------------------------------------------------------

STOCK_MIRRORED_SHA1 = "21d133ff854aa1edd66b9473e602f0eb703749ce"
"""SHA-1 of the five images of `mirrored_set(rs, 0)` on the stock set, as Phase 8 built it (before the custom run):
the stock placement is unchanged."""


@functools.lru_cache(maxsize=1)
def _desert3():
    """(grow_ids and guard_new_ids, that with Desert City converted into its own id 3 and the select of Phase 8's row:
    patch_select(converted_set(base, disc, 3, slot=3, scenery="model"), M4A_ENTRIES), Phase 8's playtest set without
    its long start time; converted before the select, as patch_select needs since Phase 9's final review)
    Its id 3 owns tables in the custom data area (its texel block holds 0xff runs) and object lists in DATA_FREE, and
    its ground pool in the custom copro_data area holds 0xffffffff words. None without DAYTONA_ROMDIR / DAYTONA_DC."""
    from dtools.arcade import select as S
    from dtools.convert import build as B
    from dtools.dc.disc import DcDisc
    d, dc = config.rom_dir(), config.dc_path()
    if d is None or not (d / "daytona.zip").is_file() or dc is None or not dc.is_file():
        return None
    base = C.guard_new_ids(C.grow_ids(RomSet.from_zip(d / "daytona.zip")))
    disc = DcDisc.open(dc)
    try:
        return base, S.patch_select(B.converted_set(base, disc, 3, slot=3, scenery="model"), S.M4A_ENTRIES)
    finally:
        disc.close()


@pytest.fixture
def desert3():
    got = _desert3()
    if got is None:
        pytest.skip("set DAYTONA_ROMDIR and DAYTONA_DC")
    return got


def _spans_overlap(a: tuple[int, int], spans) -> bool:
    return any(o < a[0] + a[1] and a[0] < o + n for o, n in spans)


def test_backdrop_custom_run(beginner, desert3):
    """mirrored_set(..., run=...) on a custom set puts the mirrored backdrop's char blocks in the given custom-data run
    (build.ID_DATA_FREE), at its first bytes free (0xff) and owned by no table (golden.owned_spans: id 3's tables and
    1 MB texel block lie at the run's start), the char list pointing there; the stock DATA_FREE is untouched and the
    set decodes to the mirror placed there. `_char_run` over the run is False on every owned byte. On the stock set
    the default placement is unchanged (STOCK_MIRRORED_SHA1, and run=tgp.DATA_FREE is the default); a custom run on a
    stock set, or a run outside DATA_FREE and the custom data area, or of the wrong shape, is refused."""
    import hashlib
    from dtools.arcade import romset
    from dtools.convert import build as B
    rs, _, _ = beginner
    h = hashlib.sha1()
    for r in romset.REGIONS:
        h.update(getattr(_built(), r))
    assert h.hexdigest() == STOCK_MIRRORED_SHA1
    assert M.mirrored_set(rs, 0, run=tgp.DATA_FREE).data == _built().data
    _, out = desert3
    run = B.ID_DATA_FREE
    lo, n = run
    owned = [(o, k) for r, o, k, _ in G.owned_spans(out) if r == "data"]
    c = G.decode_course(out, 2)
    m = M.mirrored_set(out, 2, run=run)
    d = G.decode_course(m, 2)
    srcs = [(u.src - DATA_BASE, 4 + 32 * u.count) for u in d.backdrop.uploads.chars]
    assert all(lo <= o and o + k <= lo + n for o, k in srcs), [hex(o) for o, _ in srcs]
    assert not any(_spans_overlap(s, owned) for s in srcs)
    first = srcs[0][0]
    usable = B._usable(out, "data", (), run)
    assert first == lo + int(B._fits(usable, sum(k for _, k in srcs))[0])
    want = M.mirror_course(c, first, [C.backdrop(out, e) for e in C.course_table(out) if e.course != 2], run=run)
    assert _eq(d.backdrop.chars(d.char_blocks), want.backdrop.chars(want.char_blocks))
    assert _differ(d, want) == []
    flo, fn = tgp.DATA_FREE
    assert m.data[flo:flo + fn] == out.data[flo:flo + fn]
    # the run's usable bytes: never an owned one
    others = [C.backdrop(out, e) for e in C.course_table(out) if e.course != 2]
    ok = M._char_run(out, c, others, run)
    paint = np.zeros(n, bool)
    for o, k in owned:
        if o < lo + n and o + k > lo:
            paint[max(o, lo) - lo:min(o + k, lo + n) - lo] = True
    assert paint.any() and not (ok & paint).any()
    # refused
    for bad in ((rs, run), (out, (0, 16)), (out, (lo, B.ID_DATA_FREE[1] + 0x1000000)), (out, (lo,)),
                (out, (True, 16)), (out, "run"), (out, (lo + 2, 1 << 20))):
        with pytest.raises(M.MirrorError):
            M.mirrored_set(bad[0], 2, run=bad[1])


def test_char_run_cached(beginner, monkeypatch):
    """Two mirrored_set calls on one set decode its owned space (golden.owned_spans) once: `_char_run` reads the
    per-set cache the converter's `_usable` reads (golden.owned), not a decode of its own."""
    rs, _, _ = beginner
    fresh = dataclasses.replace(rs)                 # a new set object: nothing cached for it
    calls = []
    real = G.owned_spans
    monkeypatch.setattr(G, "owned_spans", lambda s: calls.append(s) or real(s))
    a = M.mirrored_set(fresh, 0)
    b = M.mirrored_set(fresh, 2)
    assert len(calls) == 1 and calls[0] is fresh
    assert a.data == _built().data and G.decode_course(b, 2).road != G.decode_course(rs, 2).road


# --- Phase 9 Task 3: the mirrors of the originals as course ids of their own (mirrored_entry) -------------------------

def originals_row():
    """Task 3's row: Phase 9's row (select.M4B_ENTRIES: ENTRIES as Phase 9 left it, National Park Speedway's entry 4
    a placeholder) with Desert City's mirror (entry 11, live since Task 4) locked, for a set holding the mirrors of the
    originals alone (no Desert City: patch_select refuses a live id 11 without its course). Entry 3 stays live, so on
    such a set the select goes through test_select.unconverted_select (id 3 not converted)."""
    from dtools.arcade import select as S
    return tuple(e._replace(live=False) if e.course == 3 + M.MIRROR_ID_OFFSET else e for e in S.M4B_ENTRIES)


@functools.lru_cache(maxsize=1)
def _entries():
    """(the widened set: grow_ids, guard_new_ids, widen_ids; then with mirrored_entry of 0, of 0 and 1, of 0, 1 and
    2; then that with the sixteen-entry select, `originals_row`: the MAME tests' set, test_mirrors_mame.mirrors).
    None without DAYTONA_ROMDIR."""
    from dtools.arcade import select as S
    got = _beginner()
    if got is None:
        return None
    sets = [C.widen_ids(C.guard_new_ids(C.grow_ids(got[0])))]
    for o in (0, 1, 2):
        sets.append(M.mirrored_entry(sets[-1], o))
    from tests.test_select import unconverted_select       # no Desert City: id 3's check lifted (live in the row)
    return tuple(sets) + (unconverted_select(sets[-1], originals_row()),)


@pytest.fixture
def entries():
    got = _entries()
    if got is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return got


def _resolved(c: G.Course, lst: int) -> tuple:
    """Object list `lst` of `c` with its addresses resolved: per entry the object itself, tpa, tha, budget."""
    return tuple((c.objects[C.rom_object_index(p.oba)], p.tpa, p.tha, p.raw) for p in c.object_lists[lst])


def _content(c: G.Course, switch: bool = True) -> dict:
    """What a course draws and races, free of where its tables are: the tables the mirror writes for its id resolved
    through their pointers (cells to their lists' objects, actors to their lists or billboard variants, the HUD
    outline to its objects, the backdrop to its char RAM, the ground to its polygons and lists), every other field as
    it is. Without `switch` the SWITCH actors are left out (the mirror keeps the original's)."""
    def actor(a: C.Actor):
        obj = (_resolved(c, a.obj) if a.handler == C.STATIC_PROP else
               tuple(_resolved(c, x) for x in c.billboards[a.obj]) if a.handler == C.VIEW_BILLBOARD else a.obj)
        return a.slot, a.position, a.angle, a.ident, a.handler, obj
    g = c.ground_data
    return dict(
        road=c.road, ai_lines=c.ai_lines, steps=(c.step_n, c.steps), checkpoints=c.checkpoints,
        times={k: v[2] for k, v in c.time_tables.items()}, laps={m: v[1] for m, v in c.laps.items()},
        car_count=c.car_count[1], rolling=c.rolling_start[1], grid_base=c.grid_base[1], lanes=c.grid_lanes,
        spawn=c.spawn, flags=c.segment_flags[1], ref=c.billboard_ref,
        ground=(g.grid, g.attributes, g.polys, g.lists), backdrop=(c.backdrop.segments,
                                                                     c.backdrop.chars(c.char_blocks)),
        cells=tuple(_resolved(c, x) for x in c.cells),
        actors=tuple(actor(a) for v in c.actor_lists.values() for a in v if switch or a.handler != C.SWITCH),
        hud=(c.hud_map.position, c.hud_map.angle, c.hud_map.pad, c.hud_map.zooms,
             tuple((c.hud_map.objects[C.rom_object_index(p.oba)], p.tpa, p.tha, p.raw) for p in c.hud_map.entries)),
        headers=c.headers, points=c.points, sheets=c.sheet_sources, texels=_texels(c))


def _free(rs: RomSet, region: str) -> list[tuple[int, int]]:
    """The free runs mirrored_entry writes into, per region (build: the list home DATA_FREE, ID_DATA_FREE, the custom
    copro and polygon areas, PROGRAM_FREE)."""
    from dtools.convert import build as B
    return {"data": [tgp.DATA_FREE, B.ID_DATA_FREE], "copro_data": [B.NEW_ID_RUNS["copro_data"]],
            "polygons": [B.NEW_ID_RUNS["polygons"]], "program": [B.PROGRAM_FREE]}.get(region, [])


def test_mirrored_entry_round_trip(entries):
    """mirrored_entry(rs, o) for o 0, 1, 2 in turn: course id o + 8 decodes to mirror_course(decode_course(rs, o))
    with every table resolved through its pointers (`_content`: the same objects, lists, actors, ground, backdrop,
    race tables) but the switch (its three lists are program code every course shares: the mirror keeps the original's
    switch actor and objects, a known loss), and re-emits losslessly; ids 0-3 decode as before and every byte of their
    tables (golden.coverage) is unchanged; every byte the call changes lies in a free run where the input held 0xff,
    or in the mirror id's own words (record, per-id entries, TGP slot, TUNES / VISIT_IDS byte), or is the object
    library's bound patch."""
    from dtools.convert import build as B
    for o in (0, 1, 2):
        rs, out = entries[o], entries[o + 1]
        k = o + M.MIRROR_ID_OFFSET
        orig = G.decode_course(rs, o)
        got = G.decode_course(out, k)
        want = M.mirror_course(orig)
        a, b = _content(got, switch=False), _content(want, switch=False)
        assert [f for f in a if not _eq(a[f], b[f])] == [], o
        assert G.verify_lossless(out, k) == [], o
        assert got.course == got.entry.course == k and got.record_pointer == C.record_address(k)
        assert got.sheet_sources == orig.sheet_sources and got.backdrop.uploads.palette_list == \
            orig.backdrop.uploads.palette_list
        if orig.switch_lists:                               # the switch: the original's, unmirrored
            sw = [x for v in got.actor_lists.values() for x in v if x.handler == C.SWITCH]
            osw = [x for v in orig.actor_lists.values() for x in v if x.handler == C.SWITCH]
            assert [(x.position, x.angle, x.obj) for x in sw] == [(x.position, x.angle, x.obj) for x in osw]
            assert got.switch_lists == orig.switch_lists
            assert all(_eq(_resolved(got, x), _resolved(orig, x)) for x in got.switch_lists)
        for i in range(4):
            assert _differ(G.decode_course(out, i), G.decode_course(rs, i)) == [], (o, i)
            for region, off, n, _ in G.coverage(rs, i):
                assert getattr(out, region)[off:off + n] == getattr(rs, region)[off:off + n], (o, i, region, off)
        from dtools.arcade.romset import COPRO_SIZE
        own = M._own_words(rs, k)
        own += [(r, lo + COPRO_SIZE // 2, n) for r, lo, n in own if r == "copro_data"]  # its mirrored 2 MB half
        bound = [("program", at, 4) for at in C.LIBRARY_BOUND_AT]
        for region in ("program", "data", "copro_data", "polygons", "textures"):
            x, y = (np.frombuffer(getattr(s, region), np.uint8) for s in (rs, out))
            changed = np.flatnonzero(x != y)
            inside = np.zeros(len(changed), bool)
            for lo, n in _free(rs, region):
                inside |= (changed >= lo) & (changed < lo + n) & (x[changed] == 0xFF)
            for r, lo, n in own + bound:
                if r == region:
                    inside |= (changed >= lo) & (changed < lo + n)
            assert inside.all(), (o, region, [hex(v) for v in changed[~inside][:4]])


def test_mirrored_entry_arrays(entries):
    """Every per-id array entry of a mirror (course.ID_ARRAYS) equals its original's, except the stated ones: its own
    tables (mirror.OWN_ARRAYS: record, cell grid, actor table, HUD outline list), the mirrored ones
    (MIRRORED_ARRAYS: grid lanes and the billboard point mirrored; the camera script mirrored to a new script; the
    tunnel light positions and progress masks where the original has them) and the unshared ones (UNSHARED_ARRAYS:
    the SRAM words stay the guard's sink, the select's panel of course stays the select's). TUNES is the
    original's, VISIT_IDS mirror.VISIT_OF_MIRROR (none for Advanced's), TGP slot k its own ground. Each mirrored
    table's bytes are checked against its rule, and golden.owned_spans names it (a later writer cannot overwrite it)."""
    from dtools.arcade import select as S
    out = entries[4]
    arrays = C.id_tables(out)
    from tests.test_select import unconverted_select
    stock = unconverted_select(entries[0])          # no mirror, no Desert City: the sink's entries, the select's array
    for o in (0, 1, 2):
        k = o + M.MIRROR_ID_OFFSET
        differ = set()
        for name, a in C.ID_ARRAYS.items():
            if out.read_i960(C.id_entry(arrays, name, k), a.stride) != out.read_i960(C.id_entry(arrays, name, o),
                                                                                     a.stride):
                differ.add(name)
        allowed = M.OWN_ARRAYS | M.MIRRORED_ARRAYS | M.UNSHARED_ARRAYS
        assert differ <= allowed, (o, differ - allowed)
        assert M.OWN_ARRAYS - {"COURSE_POINTERS"} <= differ and {"GRID_LANES", "CAMERA_SCRIPTS"} <= differ, o
        assert ("BILLBOARD_REF" in differ) == (o == 0)                  # only Beginner has a billboard
        assert ("PROGRESS_MASKS" in differ) == (o == 2)                 # only Advanced has records
        assert ("TUNNEL_LIGHTS" in differ) == (o != 1)                  # Expert's record has no zone
        for name in M.UNSHARED_ARRAYS:                                  # the sink / the select's, as before
            a = C.ID_ARRAYS[name]
            assert out.read_i960(C.id_entry(arrays, name, k), a.stride) == \
                stock.read_i960(C.id_entry(arrays, name, k), a.stride), (o, name)
        got, orig = G.decode_course(out, k), G.decode_course(out, o)
        assert got.grid_lanes == tuple(M.mirror_lane(v) for v in orig.grid_lanes)
        assert C.tunes(out)[k] == o and C.visit_ids(out)[k] == M.VISIT_OF_MIRROR[o]
        # the camera script
        word = lambda n, i: struct.unpack("<I", out.read_i960(C.id_entry(arrays, n, i), 4))[0]  # noqa: E731
        script, oscript = (C.camera_script(out, word("CAMERA_SCRIPTS", i)) for i in (k, o))
        assert len(script.keyframes) == len(oscript.keyframes)
        for x, y in zip(script.keyframes, oscript.keyframes):
            assert (x.time, x.position_car, x.target_car, x.fov, x.rest, x.roll) == \
                (y.time, y.position_car, y.target_car, y.fov, y.rest, -y.roll)
            for p, q, car in ((x.position, y.position, y.position_car), (x.target, y.target, y.target_car)):
                assert _close(p, M.mirror_world(q) if car == C.CAMERA_ABSOLUTE else (-q[0], q[1], q[2]))
        # the tunnel light positions
        count = out.read_i960(C.id_entry(arrays, "TUNNEL_BYTES", k), 1)[0]
        rec, orec = (struct.unpack("<5I", out.read_i960(C.id_entry(arrays, "TUNNEL_LIGHTS", i), 20)) for i in (k, o))
        assert rec[:4] == orec[:4]
        if count:
            mask = struct.unpack("<I", out.read_i960(rec[3], 4))[0]
            for z in range(count):
                p, q = (struct.unpack("<3f", out.read_i960(r[4] + 12 * z, 12)) for r in (rec, orec))
                assert p == (M.mirror_world(q) if mask >> z & 1 else q), (o, z)
        # the progress masks
        masks, omasks = (M.mask_records(out, word("PROGRESS_MASKS", i)) for i in (k, o))
        assert len(masks) == len(omasks) and len(masks) == (4 if o == 2 else 0)
        for x, y in zip(masks, omasks):
            assert x[:4] == y[:4] and struct.unpack("<16H", x[4:]) == struct.unpack("<16H", y[4:])[::-1]
        # owned: every table of the mirror's own extras, exactly the bytes written (the masks' end record too)
        spans = {w: (r, o_, n) for r, o_, n, w in G.owned_spans(out) if w.endswith(f"of course {k}")}
        if masks:
            r, at, n = spans[f"progress masks of course {k}"]
            assert (r, at, n) == ("data", word("PROGRESS_MASKS", k) - DATA_BASE, len(M.mirror_masks(omasks)))
        names = set(spans)
        assert {f"camera script of course {k}", f"progress masks of course {k}"} <= names
        assert (f"tunnel light positions of course {k}" in names) == bool(count)


def test_mirrors_then_select(entries, monkeypatch):
    """The mirrors written before the select (this chain: patch_select refuses M4B_ENTRIES' live 8-10 on a set without
    them) give the same bytes as the select patched first, the order the MAME runs of test_mirrors_of_originals used
    (with patch_select's check of the live ids switched off, as it was then): neither writes where the other reads
    (the select's space is RESERVED, the mirrors' per-id entries are ids 8-10's, SELECT_PANEL_OF_COURSE the
    select's alone)."""
    from dtools.arcade import romset
    from dtools.arcade import select as S
    monkeypatch.setattr(S, "_unfilled", lambda rs, e: [])
    first = S.patch_select(entries[0], originals_row())
    for o in (0, 1, 2):
        first = M.mirrored_entry(first, o)
    assert [r for r in romset.REGIONS if getattr(first, r) != getattr(entries[4], r)] == []


def test_mirrored_entry_refuses(entries, beginner):
    """mirrored_entry refuses (CourseError) an original outside MIRROR_ORIGINALS (0-4 and 7 since Phase 10 Task 5; a
    bool too), a second mirror of the same original, a set missing a layer (not grown, grown but not guarded, guarded
    but not widened) and an original 3, 4 or 7 on a set whose id is Beginner's copy; a RomSet is required."""
    from dtools.arcade import select as S
    rs, _, _ = beginner
    w, one = entries[0], entries[1]
    assert M.MIRROR_ORIGINALS == (0, 1, 2, 3, 4, 7)
    for bad in (5, 6, -1, 8, 12, True, "0", 0.0, None):
        with pytest.raises(C.CourseError):
            M.mirrored_entry(w, bad)
    with pytest.raises(C.CourseError, match="filled already"):
        M.mirrored_entry(one, 0)
    grown = C.grow_ids(rs)
    for s, layer in ((rs, "grown"), (grown, "guarded"), (C.guard_new_ids(grown), "widened")):
        with pytest.raises(C.CourseError, match=layer):
            M.mirrored_entry(s, 0)
    for o in (3, 4, 7):
        with pytest.raises(C.CourseError, match="no course of its own"):
            M.mirrored_entry(w, o)
    with pytest.raises(C.CourseError):
        M.mirrored_entry("set", 0)
    assert issubclass(C.CourseError, DecodeError) and issubclass(M.MirrorError, DecodeError)
    from tests.test_select import unconverted_select
    patched = unconverted_select(w)                                  # Phase 8's row (no Desert City on this set)
    assert not S.patched(w) and S.patched(patched)
    assert M.mirrored_entry(patched, 1).data != one.data             # the select is not a layer: before or after


def test_mirror_camera_record_and_masks():
    """The extras' rules on synthetic records: a car-relative keyframe's points flip x, an absolute one's mirror in
    PLANE, the roll negates, the rest stays, a jump is kept; twice is the identity (but the absolute point's ulp).
    Progress-mask records reverse their rows and end with (MASK_END, MASK_END); malformed input is refused."""
    raw = struct.pack("<iIi3fi3f2f4I", 150, 0, 0, -4.0, 14.0, 10.0, -1, 3.0, 1.0, 5.0, -10.0, 83.0, 1, 2, 3, 4)
    m = M.mirror_camera_record(raw)
    got = struct.unpack("<iIi3fi3f2f4I", m)
    assert got[:3] == (150, 0, 0) and got[3:6] == (4.0, 14.0, 10.0) and got[6] == -1
    assert got[7:10] == M.mirror_world((3.0, 1.0, 5.0)) and got[10] == 10.0 and got[11] == 83.0 and got[12:] == (1, 2,
                                                                                                                3, 4)
    assert M.mirror_camera_record(M.mirror_camera_record(raw)) == raw
    jump = struct.pack("<iIi", -1, 0, 0x02806B78) + bytes(52)
    assert M.mirror_camera_record(jump) == jump
    for bad in (raw[:60], "x" * 64, struct.pack("<iIi3f", 0, 0, 0, float("nan"), 0.0, 0.0) + bytes(40)):
        with pytest.raises(M.MirrorError):
            M.mirror_camera_record(bad)
    rec = struct.pack("<2H16H", 190, 277, *range(16))
    out = M.mirror_masks([rec])
    assert out[:36] == struct.pack("<2H16H", 190, 277, *range(15, -1, -1))
    assert out[36:] == struct.pack("<2H", M.MASK_END, M.MASK_END)          # the end record: what 0x17688 reads
    for bad in ([rec[:30]], "x", [None]):
        with pytest.raises(M.MirrorError):
            M.mirror_masks(bad)


def test_mirror_camera_script_layout(beginner):
    """mirror_camera_script lays the play out from the given address: Beginner's six keyframes then a jump back to the
    first (its own script ends in a jump to its start); Expert's keyframes, its jump to the shared loop and the loop,
    the loop's closing jump pointing into the copy; course.camera_script walks each copy to the mirrored keyframes."""
    rs, _, _ = beginner
    for start, n in ((0x02806B78, 6), (0x02807274, 9)):
        at = 0x02700000
        blob = M.mirror_camera_script(rs, start, at)
        s = rs.with_patch("data", at - DATA_BASE, blob)
        got, want = C.camera_script(s, at), C.camera_script(rs, start)
        assert len(got.keyframes) == len(want.keyframes) == n
        assert at <= got.loop < at + len(blob)
        assert all(x.addr - at < len(blob) for x in got.keyframes)
        assert [x.time for x in got.keyframes] == [x.time for x in want.keyframes]


# --- Phase 10 Task 5: the mirrors of National Park Speedway (id 12) and Mermaid Lake (id 15) -------------------------

ROUNDING_NOTCH = ((-113.08726501464844, 0.0, 127.02533721923828), (-113.1014633178711, 0.0, 127.0267562866211),
                  (-121.08726501464844, 0.0, 127.8253402709961), (-119.08726501464844, 0.0, 130.02532958984375))
"""A synthetic convex polygon (the arcade's winding) whose corner 1 lies on the line through 0 and 2 within f32: its
mirror's -1 - z rounds corner 1's z (127 -> -128, one bit lost) to the notch side, past tgp.MIN_AREA."""


def test_mirror_ground_poly_nudges_a_rounding_notch():
    """Phase 10 Task 5: a polygon whose mirror is not convex only by the rounding of -1 - z (a corner on its
    neighbours' line within f32 that the mirror's rounding puts on the notch side; Mermaid Lake's cell 1123, repaired
    "ulps" by the converter, Phase 10 Task 2) gets the converter's own "ulps" repair (convert.ground: the corner moved
    outward by the fewest f32 ulps, at most NUDGE_ULPS on each axis) instead of a refusal; a polygon whose mirror is
    convex is mirrored exactly as before (no corner moves); a wider notch is still refused."""
    from dtools.convert import ground as GR
    p = tgp.GroundPoly(ROUNDING_NOTCH, (1,), None, 0)
    tgp._encode_poly(p, 0)                                          # the original is convex
    plain = tuple(M.mirror_world(ROUNDING_NOTCH[k]) for k in M.QUAD_ORDER)
    assert max(GR._turns(plain)) > tgp.MIN_AREA                     # the plain mirror has the notch
    q = M.mirror_ground_poly(p)
    tgp._encode_poly(q, 0)                                          # the mirror is convex
    assert len(q.raw) == tgp.POLY_WORDS
    moved = [k for k in range(4) if q.corners[k] != plain[k]]
    assert len(moved) == 1
    k = moved[0]
    for axis in (0, 2):
        a, b = np.float32(plain[k][axis]), np.float32(q.corners[k][axis])
        steps = 0
        while a != b and steps <= GR.NUDGE_ULPS:
            a = np.nextafter(a, b)
            steps += 1
        assert a == b and steps <= GR.NUDGE_ULPS, (axis, steps)
    assert q.corners[k][1] == plain[k][1]
    # a convex mirror: unchanged (the rule every earlier mirror was built with)
    sq = _sq(0, 2, s=6)
    assert M.mirror_ground_poly(sq).corners == tuple(M.mirror_world(sq.corners[k]) for k in M.QUAD_ORDER)
    # a wider notch, correctly wound after the mirror (the arcade's winding in, QUAD_ORDER out): refused. Corner 1
    # 0.01 onto the notch side is a notch the converter's "line" repair would close (inside MAX_MOVE): the mirror
    # takes only "ulps" (the `how` check); 0.2 is past MAX_MOVE (the converter refuses it: ConvertError's path)
    for dz, match in ((0.01, r"wider than the mirror's rounding \(line\)"), (0.2, "past MAX_MOVE")):
        c = ROUNDING_NOTCH[:1] + ((ROUNDING_NOTCH[1][0], 0.0, ROUNDING_NOTCH[1][2] + dz),) + ROUNDING_NOTCH[2:]
        mirrored = tuple(M.mirror_world(c[k]) for k in M.QUAD_ORDER)
        assert GR._area(mirrored) <= -tgp.MIN_AREA and max(GR._turns(mirrored)) > tgp.MIN_AREA, dz
        if dz < GR.MAX_MOVE:
            assert GR._convex(mirrored, "x")[1] == "line"           # what the converter would do with it
        with pytest.raises(M.MirrorError, match=match):
            M.mirror_ground_poly(tgp.GroundPoly(c, (1,), None, 0))


M5_MIRRORS = {4: 12, 7: 15}
"""Phase 10 Task 5: the originals mirrored this task and their mirrors' ids (mirror.MIRROR_ID_OFFSET)."""


@functools.lru_cache(maxsize=1)
def _m5():
    """(the final chain to the mirrors of 0-3: grow_ids, guard_new_ids, widen_ids, Desert City, National Park Speedway
    and Mermaid Lake converted at ids 3, 4 and 7 (tests/test_mirrors_mame.py `_converted`), mirrored_entry of 0-3;
    then that with mirrored_entry of 4 (id 12); then with mirrored_entry of 7 (id 15)). None without DAYTONA_ROMDIR /
    DAYTONA_DC."""
    from dtools.dc.disc import DcDisc
    from tests.test_mirrors_mame import _converted
    d, dc = config.rom_dir(), config.dc_path()
    if d is None or not (d / "daytona.zip").is_file() or dc is None or not dc.is_file():
        return None
    disc = DcDisc.open(dc)
    try:
        out = _converted(RomSet.from_zip(d / "daytona.zip"), disc)
    finally:
        disc.close()
    for o in (0, 1, 2, 3):
        out = M.mirrored_entry(out, o)
    m12 = M.mirrored_entry(out, 4)
    return out, m12, M.mirrored_entry(m12, 7)


@pytest.fixture
def m5():
    got = _m5()
    if got is None:
        pytest.skip("set DAYTONA_ROMDIR and DAYTONA_DC")
    return got


def test_mirrored_4_and_7_round_trip(m5):
    """mirror.mirrored_entry(rs, 4) and then (rs, 7) on the final chain (Phase 9 Task 3's and Task 4's checks): course
    id o + 8 decodes to mirror_course(decode_course(rs, o)) with every table resolved through its pointers (`_content`)
    and re-emits losslessly; the original's race behaviour is its own, shared (the standing start, its grid base,
    checkpoints, time tables, laps, car count: the same words), its tune, VISIT_IDS mirror.VISIT_OF_MIRROR[o] (the
    original's own value: neither is Advanced's 2); its texel blocks (COURSE_SHEETS, in the custom data area) are
    shared, not copied, as are its texture headers and points, and the call adds far less than a block to the data
    ROM; every per-id entry of the mirror is the original's but the stated ones (mirror.OWN_ARRAYS, MIRRORED_ARRAYS,
    UNSHARED_ARRAYS), ACTOR_TABLES the original's too (no actor: the shared EMPTY_ACTORS, no program byte taken);
    its camera script its own, every keyframe the original's mirrored (mirror_camera_record's rule); Mermaid Lake's
    mirror is built from the lifted id 7 (build.LIFT[7]: the mirror keeps every height, so its ground
    stays above 0); every other filled id (0-4, 7, 8-11, and 12 for the second) decodes as before and every byte of its
    tables is unchanged; every byte the call changes lies in a free run where the input held 0xff, in the mirror id's
    own words or the bound patch."""
    from dtools.arcade import texture as T
    from dtools.arcade.romset import COPRO_SIZE
    from dtools.convert import build as B
    before, m12, m15 = m5
    for o, rs, out, others in ((4, before, m12, (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)),
                               (7, m12, m15, (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12))):
        k = M5_MIRRORS[o]
        assert k == o + M.MIRROR_ID_OFFSET
        orig, got = G.decode_course(rs, o), G.decode_course(out, k)
        a, b = _content(got), _content(M.mirror_course(orig))
        assert [f for f in a if not _eq(a[f], b[f])] == [], o
        assert G.verify_lossless(out, k) == [], o
        assert got.course == got.entry.course == k and got.record_pointer == C.record_address(k)
        # the original's behaviour, shared
        assert got.rolling_start[1] is False and got.grid_base[1] == orig.grid_base[1] == B.GRID_BASES[o], o
        assert [cp.segment for cp in got.checkpoints] == [cp.segment for cp in orig.checkpoints], o
        assert got.time_tables == orig.time_tables and got.car_count[1] == orig.car_count[1], o
        assert {m: v[1] for m, v in got.laps.items()} == {m: v[1] for m, v in orig.laps.items()}, o
        assert C.tunes(out)[k] == C.tunes(out)[o] and C.visit_ids(out)[k] == M.VISIT_OF_MIRROR[o] == o
        # the texel blocks, headers and points: shared
        assert got.sheet_sources == orig.sheet_sources and got.headers == orig.headers and got.points == orig.points
        block = orig.sheet_sources[0]                   # the course's own 1 MB block (sheet 1 is the common block)
        assert B.ID_DATA_FREE[0] <= block and out.data[block:block + T.SHEET_BLOCK] == \
            rs.data[block:block + T.SHEET_BLOCK], (o, hex(block))
        assert [T.sheet_source(out, s, k) for s in range(2)] == [T.sheet_source(out, s, o) for s in range(2)]
        added = int((np.frombuffer(out.data, np.uint8) != np.frombuffer(rs.data, np.uint8)).sum())
        print(f"\nthe mirror of {o} (id {k}) adds {added} bytes to the data ROM")
        assert added < T.SHEET_BLOCK // 2, o
        # the per-id arrays
        arrays = C.id_tables(out)
        differ = {n for n, x in C.ID_ARRAYS.items()
                  if out.read_i960(C.id_entry(arrays, n, k), x.stride) != out.read_i960(C.id_entry(arrays, n, o),
                                                                                        x.stride)}
        assert differ <= M.OWN_ARRAYS | M.MIRRORED_ARRAYS | M.UNSHARED_ARRAYS, (o, differ)
        assert {"GRID_TABLES", "HUD_MAP_LISTS", "GRID_LANES", "CAMERA_SCRIPTS"} <= differ, o
        assert not {"ACTOR_TABLES", "COURSE_SHEETS", "BILLBOARD_REF", "PROGRESS_MASKS"} & differ, o
        assert got.actor_table == orig.actor_table == B.EMPTY_ACTORS and B.program_room(out) == B.program_room(rs)
        # the camera script: the mirror's own copy, each keyframe the original's mirrored (mirror_camera_record's rule:
        # a car-relative point (-x, y, z), an absolute one mirror_world, the roll negated), addresses aside
        word = lambda n, i: struct.unpack("<I", out.read_i960(C.id_entry(arrays, n, i), 4))[0]  # noqa: E731
        script, oscript = (C.camera_script(out, word("CAMERA_SCRIPTS", i)) for i in (k, o))
        assert word("CAMERA_SCRIPTS", k) != word("CAMERA_SCRIPTS", o) and len(script.keyframes) == len(
            oscript.keyframes) > 0, o
        for x, y in zip(script.keyframes, oscript.keyframes):
            assert (x.time, x.position_car, x.target_car, x.fov, x.rest, x.roll) == \
                (y.time, y.position_car, y.target_car, y.fov, y.rest, -y.roll), o
            for pt, q, car in ((x.position, y.position, y.position_car), (x.target, y.target, y.target_car)):
                assert _close(pt, M.mirror_world(q) if car == C.CAMERA_ABSOLUTE else (-q[0], q[1], q[2])), o
        # the lift: the mirror keeps every height (Mermaid Lake's ground above 0, as its original's)
        ys = [c[1] for p in got.ground_data.polys for c in p.corners]
        oys = [c[1] for p in orig.ground_data.polys for c in p.corners]
        assert sorted(ys) == sorted(oys) and (min(ys) > 0) == (o in B.LIFT), o
        # every other id unchanged
        for i in others:
            assert _differ(G.decode_course(out, i), G.decode_course(rs, i)) == [], (o, i)
            for region, off, n, _ in G.coverage(rs, i):
                assert getattr(out, region)[off:off + n] == getattr(rs, region)[off:off + n], (o, i, region, off)
        own = M._own_words(rs, k)
        own += [(r, lo + COPRO_SIZE // 2, n) for r, lo, n in own if r == "copro_data"]   # its mirrored 2 MB half
        bound = [("program", at, 4) for at in C.LIBRARY_BOUND_AT]
        for region in ("program", "data", "copro_data", "polygons", "textures"):
            x, y = (np.frombuffer(getattr(s, region), np.uint8) for s in (rs, out))
            changed = np.flatnonzero(x != y)
            inside = np.zeros(len(changed), bool)
            for lo, n in _free(rs, region):
                inside |= (changed >= lo) & (changed < lo + n) & (x[changed] == 0xFF)
            for r, lo, n in own + bound:
                if r == region:
                    inside |= (changed >= lo) & (changed < lo + n)
            assert inside.all(), (o, region, [hex(v) for v in changed[~inside][:4]])


def test_mirrored_15_nudges_one_polygon(m5):
    """Mermaid Lake's mirror (id 15): its ground is the lifted id 7's mirrored polygon for polygon (mirror_ground_poly)
    and every polygon but one is the exact mirror (mirror_world of each corner, QUAD_ORDER / TRIANGLE_ORDER); the one
    is cell 1123 (convert.ground.REPAIRED[7]: "ulps"), whose mirror the rounding of -1 - z leaves not convex, nudged
    back by the converter's "ulps" rule (test_mirror_ground_poly_nudges_a_rounding_notch). National Park Speedway's
    mirror (id 12) needs none."""
    from dtools.convert import ground as GR
    before, m12, m15 = m5
    for o, rs, out, want in ((4, before, m12, []), (7, m12, m15, [1123])):
        orig, got = G.decode_course(rs, o), G.decode_course(out, M5_MIRRORS[o])
        moved = []
        for i, (p, q) in enumerate(zip(orig.ground_data.polys, got.ground_data.polys)):
            order = M.TRIANGLE_ORDER if p.corners[2] == p.corners[3] else M.QUAD_ORDER
            if q.corners != tuple(M.mirror_world(p.corners[j]) for j in order):
                moved.append(i)
        assert moved == want, (o, moved)
        assert all(GR.REPAIRED[o].get(i) == "ulps" for i in moved)


M4_ROAD_WITHIN = 2554
M4_ROAD_OUTSIDE = ((139, 6), (144, 1))
"""Course 4's road slots past Phase 5's 0.004 (measured): 2 of 2,556, (road point, slot), the largest 0.0159 (Phase 5:
course 4's road "within 0.016"); both beside the AI points below."""
M4_AI_MAX = 0.773
M4_AI_OUTSIDE = tuple((k, i) for k in range(8) for i in range(140, 147))
"""Course 4's AI points past the point bound (measured): exactly Phase 5's 56, points 140-146 of all eight lines (seven
segments across the whole road), up to 0.773 off by the check's measure (the largest of |dx|, |dy|, |dz|; Phase 5
gave "up to 0.81" by its own): the Dreamcast's `M` file moved that stretch; every other point is exact."""
M4_ROAD_HEADINGS = (1212, 4)
M4_AI_HEADINGS = (1664, 179)
"""(within Phase 5's 2, the largest) of course 4's road and AI headings (measured): the road as on course 3 (3 or 4
off: Phase 5's rule and the arcade's 0x2000 - h one unit apart in either frame add); the AI up to 179 (Phase 5:
course 4's line headings up to 181 off, recomputed)."""
M7_OUTSIDE = {"road headings", "AI headings"}
"""Course 7 (lifted, both files): only the headings past Phase 5's 2 (measured: road 1,470 of 3,000 within, AI 1,919
of 4,000, both at most 4, course 3's case); every position, corner and centre within its bound, the grid equal."""


def test_mirror_check_4_and_7(disc, m5):  # noqa: F811
    """build.mirror_check(disc, 4) and (disc, 7) (Phase 9 Task 4's check): the arcade's mirror of the converted course
    (the mirror's rules on the converter's tables: what ids 12 and 15 of the set hold) against the Dreamcast's own
    mirror, CSDATnM.BIN converted the same way, z moved by -1. The bounds (build.MIRROR_TOLERANCES) are Phase 5's a
    priori numbers and are not changed; what lies past them is pinned here as measured (`outside`), not widened.

    Course 7 is lifted on the set (build.LIFT[7], 41 units), so the check lifts both files the same (like with like:
    the plain one as converted_set does, the `M` one by the same lift; the mirror keeps every height), and the
    arcade side is then id 15 as the set holds it.

    Course 4 is Phase 5's exception (docs/spike/2026-11-dc-decode.md, "Mirror files"): its `M` file is not an exact
    mirror. 56 of its AI line points (Phase 5: "56 points up to 0.81 off"; the brief's "56 road points" are these AI
    points) were moved by the Dreamcast's tools, and its road slots are within 0.016 (the other courses' 0.003-0.004),
    its headings up to 181 off. For the check that means the `M` file is no reference at those points: the arcade's
    mirror is exact by construction (mirror_world of the plain point), so the gap there measures the Dreamcast's
    edit, not the arcade's mirror. They are pinned as findings, point by point."""
    from tests.test_convert_tables import _data
    from dtools.convert import build as B
    tol = B.MIRROR_TOLERANCES
    assert tol == {"road": 0.004, "point": 2.0 ** -14, "centre": 0.35, "heading": 2, "lane": 134}
    _, m12, m15 = m5
    found = {}
    for n, out in ((4, m12), (7, m15)):
        r = B.mirror_check(disc, n)
        found[n] = r
        print(f"\nmirror_check(disc, {n}): {r}")
        assert r["lift"] == B.LIFT.get(n, 0.0) and r["plane"] == -1.0, n
        # the arcade side is the mirror the set holds
        d = _data(disc, n)
        if n in B.LIFT:
            d = B.lifted_data(d, B.LIFT[n])
        road, lines, g = B._mirror_tables(d)
        got = G.decode_course(out, M5_MIRRORS[n])
        assert got.road == road and got.ai_lines == lines, n
        assert (got.ground_data.polys, got.ground_data.grid, got.ground_data.lists) == (g.polys, g.grid, g.lists), n
    r4, r7 = found[4], found[7]
    # course 4: within the bounds
    assert r4["road"]["lane_ranges"] == (426, 426)
    assert r4["ai_lines"]["order"] == (7, 0, 1, 2, 3, 4, 5, 6)          # Phase 5: rotated on courses 4-7
    assert r4["ai_lines"]["steps"] == r4["ai_lines"]["lanes_within"] == r4["ai_lines"]["count"] == 3408
    g = r4["ground"]
    assert g["count"] == g["corners_within"] == g["centres_within"] == g["directions_equal"] == 2169
    assert r4["grid"]["equal"] == 16384 and r4["steps"] == (485, 485) and r4["attributes"] == (2169, 2169)
    assert r4["checkpoints"] == {"arcade": (125, 257), "dreamcast": (125, 257)}
    # course 4: past the bounds (findings, measured)
    assert set(r4["outside"]) == {"road positions", "road headings", "AI positions", "AI headings"}
    assert (r4["road"]["count"], r4["road"]["within"]) == (2556, M4_ROAD_WITHIN) and r4["road"]["max"] < 0.016
    assert r4["road"]["outside"] == M4_ROAD_OUTSIDE
    assert (r4["ai_lines"]["within"], round(r4["ai_lines"]["max"], 3)) == (3352, M4_AI_MAX)
    assert r4["ai_lines"]["outside"] == M4_AI_OUTSIDE and len(M4_AI_OUTSIDE) == 56
    assert (r4["road"]["headings_within"], r4["road"]["headings_max"]) == M4_ROAD_HEADINGS
    assert (r4["ai_lines"]["headings_within"], r4["ai_lines"]["headings_max"]) == M4_AI_HEADINGS
    # course 7 (lifted, both files)
    assert set(r7["outside"]) == M7_OUTSIDE, r7["outside"]
    assert r7["checkpoints"] == {"arcade": (120, 363), "dreamcast": (120, 363)}
    assert r7["road"]["count"] == r7["road"]["within"] == 3000 and r7["road"]["outside"] == ()
    assert r7["ai_lines"]["count"] == r7["ai_lines"]["within"] == 4000 and r7["ai_lines"]["outside"] == ()
    assert r7["ai_lines"]["order"] == (7, 0, 1, 2, 3, 4, 5, 6) and r7["ai_lines"]["lanes_max"] <= tol["lane"]
    g = r7["ground"]
    assert g["count"] == g["corners_within"] == g["centres_within"] == g["directions_equal"] == 2718
    assert r7["grid"]["equal"] == 16384 and r7["steps"] == (596, 596) and r7["attributes"] == (2718, 2718)
    assert (r7["road"]["headings_within"], r7["road"]["headings_max"]) == (1470, 4)
    assert (r7["ai_lines"]["headings_within"], r7["ai_lines"]["headings_max"]) == (1919, 4)


# --- in MAME: the mirrored Beginner in the Beginner slot (Phase 4 Task 5) ---------------------------------------------

GO = 1890                   # Beginner's race start in the autoplay (autoplay.lua)
NEAR_TO = GO + 1150         # 20 s of racing: through here every car stays within NEAR_D of its mirror
SYMMETRIC_TO = 1440
"""The last frame up to which every ground call of the mirrored run agrees with the original's (`_parting`), measured
(docs/spike/2026-11-mirror.md, "Symmetry"): at frame 1441, 449 frames before GO, the player's rolling-start drive
(i960 0x11808-0x1184c: the yaw steered towards TGP function 0x0f's angle to the lane point ahead) leaves 0x4000 by
one unit on the original and not on the mirror, and its wheel calls part by 0.01: the game's own one-way rounding,
which no table can fix (the controller's ruling: exact symmetry to here, near symmetry to NEAR_TO)."""
NEAR_D = 26.0
"""Through NEAR_TO every car's mirrored position (x, -1 - z) is within NEAR_D (XZ distance) of the mirror run's car at
the same frame. Measured maximum 25.54 (car 15 at frame 2987: the mirror's car 15 drives the same line about 20
frames behind, a steady offset from 2500 on; car 18 is 8.1 off from frame 1330: GRID_EXCEPTION). After NEAR_TO
the field diverges (up to 121 by frame 3700, passes and contacts), docs/spike/2026-11-mirror.md, "Symmetry"."""
SYMMETRY_TOL = 0.01
RACE_SECONDS = 90           # the drives: GO + 3,287 frames
GEO_SECONDS = 44            # the polygon runs: frames 1890-2490 logged (one list per frame from GEO_FROM)
GEO_WINDOW = (GO, GO + 600)
HUD_FRAME = GO + 288         # GO + 5 s at 57.52 Hz: the snapshot test_mirrored_hud_map_on_screen compares
HUD_BOX = (385, 176, 466, 284)
"""The HUD map's box on the 496 x 384 snapshot (left, top, right, bottom exclusive): the outline is at x 401-455,
y 177-276 on the original, 395-449 mirrored. Its columns pair as c and 850 - c under the flip, about screen x 425, where
world z = -0.5 (the mirror plane) is drawn: docs/spike/2026-11-mirror.md, "HUD map"."""
HUD_MATCH = 0.99
"""snapshot_match over HUD_BOX of the outline's pixels (pure black), original against the mirror flipped left to
right: measured 0.9957 (the prototype run; the markers, which may differ by a few pixels, hide parts of the outline).
Unflipped, the same pair matches 0.790, and the unmirrored outline (Task 5's set) flipped 0.792."""
BACKDROP_BAND = (0, 0, 496, 100)
"""The band above the horizon on the 496 x 384 snapshot at HUD_FRAME (left, top, right, bottom exclusive): the
panorama's sky starts at screen y 25 (layer-2 row 6 under the vertical scroll 0x17) and its horizon (the grass's
top) is near y 100; polygons (trees, the wall) and the HUD cover most of it."""
HUD_BOXES = ((28, 0, 114, 82), (346, 6, 474, 82), (148, 0, 348, 94), (380, 74, 472, 290))
"""The race HUD on tile layer 0 (in front, not mirrored): LAP and the lap time, POSITION, the speedometer, the map's
box. A snapshot flipped puts the mirror's HUD at x 495 - x, so the comparison leaves out both."""
BACKDROP_MATCH = 0.9
"""snapshot_match over BACKDROP_BAND, flipped, of the original's sky pixels (blue-dominant, `_sky`) outside HUD_BOXES
and their mirror images (3,270 pixels): measured 0.9199 (the prototype run; best with the flipped mirror shifted 1 px,
0.940 over the band to y 140: the mirror's camera yaw is 31 units, one pixel, short of 0x8000 - yaw at this frame). The
same pixels match 0.564 against Task 6's set (backdrop unmirrored: clouds where the original shows sky) and 0.092
unflipped: docs/spike/2026-11-mirror.md, "Backdrop"."""
POLYGON_BAR = 3533          # Phase 1: Beginner's most drawn polygons in one list (every list counted)
POLYGON_GATE = 4251         # the M1c gate: the originals' most (Advanced), which the game draws at 60 Hz


def _rom_zip():
    d = config.rom_dir()
    if d is None or not (d / "daytona.zip").is_file():
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return d / "daytona.zip"


@pytest.fixture(scope="module")
def drives(tmp_path_factory):
    """The original and the mirrored set, RACE_SECONDS each on Beginner with the ground, progress and rate logs and one
    snapshot at HUD_FRAME (one pair for every drive test of the module)."""
    zip_ = _rom_zip()
    built = M.mirrored_set(RomSet.from_zip(zip_))
    wd = tmp_path_factory.mktemp("drives")
    logs = ("ground", "progress", "rate")
    env = {"DAYTONA_COURSE": "0", "DAYTONA_SNAPS": str(HUD_FRAME)}        # the HUD map test's snapshot
    return (build.run_set(None, wd / "original", RACE_SECONDS, zip_, env, logs),
            build.run_set(built, wd / "mirror", RACE_SECONDS, zip_, env, logs))


def _calls(path) -> dict[int, list[runner.GroundSample]]:
    out = collections.defaultdict(list)
    for s in runner.parse_ground(path)[0]:
        out[s.frame].append(s)
    return out


def _partner(s: runner.GroundSample, t: runner.GroundSample, tol: float) -> bool:
    """t is s mirrored: dest equal, x within tol, z mirrored (`mirror_world`) within tol, both without ground or
    heights within tol."""
    if (s.dest != t.dest or abs(s.x - t.x) > tol or abs(M.mirror_world((s.x, s.y, s.z))[2] - t.z) > tol
            or (s.height is None) != (t.height is None)):
        return False
    return s.height is None or abs(s.height - t.height) <= tol


def _parting(a: dict, b: dict, until: int, tol: float = SYMMETRY_TOL) -> runner.GroundSample | None:
    """The first call (of either run, by frame) up to frame `until` that has no mirrored partner in the other run, or
    None. Within a frame the calls pair one to one in any order: the car models are not mirrored, so a car's left
    wheel query on the mirror is the mirror of the original's right wheel query (calls 1 and 2, 3 and 4 swap). A
    call may pair with one a frame later or earlier (where the frame boundary falls in the game's loop); one left
    over after that parts."""
    left_a, left_b, parted = [], [], []
    for f in range(min(min(a, default=until), min(b, default=until)), until + 2):
        pool_a, pool_b = left_a + a.get(f, []), left_b + b.get(f, [])
        used = [False] * len(pool_b)
        left_a = []
        for s in pool_a:
            i = next((i for i, t in enumerate(pool_b) if not used[i] and _partner(s, t, tol)), None)
            if i is None:
                left_a.append(s)
            else:
                used[i] = True
        left_b = [t for i, t in enumerate(pool_b) if not used[i]]
        parted += [s for s in left_a + left_b if s.frame < f]
        left_a, left_b = [s for s in left_a if s.frame == f], [t for t in left_b if t.frame == f]
    parted = sorted((s for s in parted if s.frame <= until), key=lambda s: (s.frame, s.n))
    return parted[0] if parted else None


@pytest.mark.mame
def test_mirrored_beginner_symmetric(drives):
    """Every ground call of the mirrored run up to SYMMETRIC_TO is the original's mirrored (x equal, z negated, dest
    equal, heights equal; within SYMMETRY_TOL) and the two part right after it, as measured: 636 calls, the course
    set-up's at frame 501 and the player's wheels and centre every frame from 1328 (the grid and the rolling
    start)."""
    original, mirror = (_calls(r["ground"]) for r in drives)
    assert min(original) < 600 and max(original) > SYMMETRIC_TO + 1000
    assert sum(len(v) for f, v in original.items() if f <= SYMMETRIC_TO) > 600
    assert _parting(original, mirror, SYMMETRIC_TO) is None
    first = _parting(original, mirror, SYMMETRIC_TO + 1)
    assert first is not None and first.frame == SYMMETRIC_TO + 1, first


GRID_EXCEPTION = 18
"""The one car not on its mirrored place at the race set-up. Its standing place (segment 144, lane 0x580; mirrored
0x17f) lies exactly on the edge road polygons 375 (segment 145) and 378 (segment 146) share; the original's TGP
answers 378, the mirror's 375 (the reversed winding's f32 edge test: "Offline", points on a shared edge), and
0x10f50-0x10f80 takes the car's segment from the step table of the polygon under it, so after the rolling start's
step it is on segment 147 in the original and 146 in the mirror, 8 units behind (docs/spike/2026-11-mirror.md,
"Symmetry", trace of car 18's +0x12c at frame 1328)."""


@pytest.mark.mame
def test_mirrored_beginner_grid(drives):
    """Frame 1330 (the race set-up done, the field not yet moved): every car but GRID_EXCEPTION is on the original's
    segment and at the original's position mirrored in PLANE (within 0.001), and GRID_EXCEPTION is one segment
    behind its original."""
    (o_rows, _), (m_rows, _) = (runner.parse_progress(r["progress"]) for r in drives)
    o_seg = {r.car: r.segment for r in o_rows if r.frame == 1330}
    m_seg = {r.car: r.segment for r in m_rows if r.frame == 1330}
    o_pos, m_pos = (runner.parse_car_positions(r["progress"])[1330] for r in drives)
    assert sorted(o_seg) == sorted(m_seg) == sorted(o_pos) == sorted(m_pos) == list(range(40))
    off = []
    for car in o_seg:
        x, y, z = M.mirror_world(o_pos[car])
        exact = o_seg[car] == m_seg[car] and max(abs(x - m_pos[car][0]), abs(y - m_pos[car][1]),
                                                   abs(z - m_pos[car][2])) <= 0.001
        if not exact:
            off.append(car)
    assert off == [GRID_EXCEPTION]
    assert o_seg[GRID_EXCEPTION] == m_seg[GRID_EXCEPTION] + 1 == 147


def test_expert_and_advanced_mirrors_build_lossless(beginner):
    """Phase 4 residual: mirrored_set(rs, 1) and (rs, 2) build, decode back to the mirror, and re-emit losslessly
    (checked ad hoc in the fix wave; test_with_patches_course_is_identity covers the cause)."""
    rs, _, _ = beginner
    for course in (1, 2):
        built = M.mirrored_set(rs, course)
        assert G.verify_lossless(built, course) == [], course
        assert G.decode_course(built, course).road != G.decode_course(rs, course).road, course


def test_own_blocks_refuses_a_block_two_backdrops_name(beginner):
    """Phase 4 residual: a free-run char block that another course's backdrop also uploads is not the course's own to
    reuse or overwrite."""
    rs, _, _ = beginner
    c = G.decode_course(_built(), 0)
    own = M._own_blocks(c)
    assert own and all(tgp.DATA_FREE[0] <= o < tgp.DATA_FREE[0] + tgp.DATA_FREE[1] for o in own)
    other = G.decode_course(rs, 1).backdrop
    assert M._own_blocks(c, [other]) == own                      # a backdrop with other blocks changes nothing
    with pytest.raises(M.MirrorError, match="also uploaded by another"):
        M._own_blocks(c, [c.backdrop])
    with pytest.raises(M.MirrorError, match="also uploaded by another"):
        M.mirror_course(c, others=[c.backdrop])
    with pytest.raises(M.MirrorError):
        M._own_blocks(c, ["junk"])
    # mirrored_set looks at the other courses' backdrops: all three mirrored in turn still build
    both = M.mirrored_set(M.mirrored_set(_built(), 1), 2)
    assert [G.verify_lossless(both, k) for k in (0, 1, 2)] == [[], [], []]


SYMMETRIC_CARS = {14: 2267, 17: 2269, 19: 2268, 20: 2268, 21: 2313, 22: 2268, 23: 2258, 24: 2258, 25: 2271,
                  26: 2312, 29: 2346, 31: 2339, 32: 2268, 33: 2338, 34: 2267, 35: 2340, 36: 2267}
"""Per opponent, the frame through which it stays within SYMMETRY_TOL (XZ) of the original's car mirrored in PLANE on
every frame from 1330 (the progress log's positions, same frame), measured on the `drives` pair (final review,
Important 4): the 17 opponents that drive their AI lines symmetrically for 368-456 frames after GO, through the
curves, so a rule the AI lines apply only after GO (lane, heading, roll, steer: "Rules checked") is gated, which
the exact gate (to SYMMETRIC_TO, 450 frames before GO) and NEAR_D (26 units) cannot see. A deliberately broken
mirror parts them early (docs/spike/2026-11-mirror.md, "Symmetry"). The other cars part earlier: the player at 1441
(SYMMETRIC_TO), cars 1-3 and 9-12 at 1893-1987, the rest between 1332 and 1866, car 18 at the set-up
(GRID_EXCEPTION).

Zero margin is deliberate (Phase 4 residual, closed in Phase 8): MAME is deterministic and the pair is driven by
a scripted autoplay from a fixed seed, so a given ROM set gives the same frame on every run; the evidence is two
independent runs of the pair, which agreed on every floor. A margin would only hide a real change (the floor is a
gate on the mirror's behaviour, and a deliberately broken mirror parts the cars early). If a floor fails, re-measure
and read the `-rf` output before loosening it."""


@pytest.mark.mame
def test_mirrored_beginner_cars_symmetric(drives):
    """Every car of SYMMETRIC_CARS is the original's mirrored (XZ within SYMMETRY_TOL) on every frame from 1330
    through its frame."""
    original, mirror = (runner.parse_car_positions(r["progress"]) for r in drives)
    for car, until in SYMMETRIC_CARS.items():
        for f in range(1330, until + 1):
            x, _, z = M.mirror_world(original[f][car])
            d = math.hypot(x - mirror[f][car][0], z - mirror[f][car][2])
            assert d <= SYMMETRY_TOL, (car, f, d)
    assert min(SYMMETRIC_CARS.values()) > GO + 360


@pytest.mark.mame
def test_mirrored_beginner_near_symmetric(drives):
    """Through NEAR_TO (GO + 20 s) every car of the mirrored run is within NEAR_D of the original's car mirrored in
    PLANE, frame by frame (the progress log's positions), and every car is logged on every frame from 1330."""
    original, mirror = (runner.parse_car_positions(r["progress"]) for r in drives)
    worst = 0.0
    for f in range(1330, NEAR_TO + 1):
        a, b = original[f], mirror[f]
        assert sorted(a) == sorted(b) and len(a) > 10, f
        for car, p in a.items():
            x, _, z = M.mirror_world(p)
            worst = max(worst, math.hypot(x - b[car][0], z - b[car][2]))
    assert worst <= NEAR_D, worst


def _progress(path):
    rows, timer = runner.parse_progress(path)
    cars = collections.defaultdict(list)
    for r in rows:
        cars[r.car].append(r)
    return cars, timer


def _rises(timer: dict[int, int]) -> list[tuple[int, int]]:
    """(frame, amount) of every frame where the race timer rises: a checkpoint's time extension."""
    return [(f, timer[f] - timer[f - 1]) for f in sorted(timer) if f - 1 in timer and timer[f] > timer[f - 1]]


def _new_lists(path) -> set[int]:
    """Frames whose display list is new: one write to the read start and a list head unlike the previous frame's."""
    rate = runner.parse_rate(path)
    return {f for (f, w, _, h), (_, _, _, h0) in zip(rate[1:], rate) if w >= 1 and h != h0}


@pytest.mark.mame
def test_mirrored_beginner_drives(drives):
    """The mirrored Beginner drives like the original over the whole run: the race is reached (every car's segment
    and lap from 1330 to the end, the timer counting down from GO); at least 99.9 % of every car's ground calls (a
    call is the car's whose segment centre is nearest) return a polygon; every car advances as on the original
    (segment steps of 0 or 1 per frame, laps counted only at the wrap to segment 0, as many laps, the final progress
    within 5 %, the player's within 2 %); the timer's extensions equal the original's in number and amount, each
    within 60 frames; the rate log has a new display list on every frame the original has one."""
    c = G.decode_course(RomSet.from_zip(_rom_zip()), 0)
    n = len(c.road)                         # segments 0..n - 1: the lap line is the wrap from the last to 0
    (o_cars, o_timer), (m_cars, m_timer) = (_progress(r["progress"]) for r in drives)
    end = max(r.frame for r in o_cars[0])
    assert end > NEAR_TO + 1000 and sorted(o_cars) == sorted(m_cars) == list(range(c.car_count[1]))
    for timer in (o_timer, m_timer):
        assert timer[GO + 600] < timer[GO] and min(timer) == 1330 and max(timer) == end
    for z_of, run, cars in ((lambda p: p[2], drives[0], o_cars), (lambda p: M.mirror_world(p)[2], drives[1], m_cars)):
        at = {(r.frame, r.car): r.segment for rows in cars.values() for r in rows}
        calls, hits = collections.Counter(), collections.Counter()
        for s in runner.parse_ground(run["ground"])[0]:
            if (s.frame, 0) not in at:                  # before the log's first frame (1330) or after its last
                continue
            car = min(cars, key=lambda k: math.hypot(c.road[at[s.frame, k]].centre[0] - s.x,
                                                     z_of(c.road[at[s.frame, k]].centre) - s.z))
            calls[car] += 1
            hits[car] += s.height is not None
        assert 0 in calls and len(calls) > 10
        assert all(hits[k] >= 0.999 * calls[k] for k in calls), {k: (hits[k], calls[k]) for k in calls}

    def advance(rows):
        laps = 0
        for a, b in zip(rows, rows[1:]):
            assert (b.segment - a.segment) % n in (0, 1), (a, b)
            if b.lap != a.lap:
                assert (a.segment, b.segment, b.lap) == (n - 1, 0, a.lap + 1), (a, b)
                laps += 1
        return laps, rows[-1].lap * n + rows[-1].segment
    for car in o_cars:
        (o_laps, o_end), (m_laps, m_end) = advance(o_cars[car]), advance(m_cars[car])
        assert m_laps == o_laps and abs(m_end - o_end) <= (0.02 if car == 0 else 0.05) * o_end, (car, o_end, m_end)
    o_rises, m_rises = _rises(o_timer), _rises(m_timer)
    assert o_rises and len(m_rises) == len(o_rises)
    assert all(b == a and abs(g - f) <= 60 for (f, a), (g, b) in zip(o_rises, m_rises)), (o_rises, m_rises)
    o_new, m_new = (_new_lists(r["rate"]) for r in drives)
    race = set(range(GO, end + 1))
    assert race <= o_new and o_new & race <= m_new


@pytest.fixture(scope="module")
def polygon_counts(tmp_path_factory):
    """count_polygons per logged list, GEO_SECONDS of the original and the mirrored set with DAYTONA_GEO_EVERY 1 from
    GO: {frame: drawn polygons} for each."""
    zip_ = _rom_zip()
    rs = RomSet.from_zip(zip_)
    built = M.mirrored_set(rs)
    wd = tmp_path_factory.mktemp("polygons")
    env = {"DAYTONA_COURSE": "0", "DAYTONA_GEO_FROM": str(GO), "DAYTONA_GEO_EVERY": "1"}
    out = []
    for name, s in (("original", None), ("mirror", built)):
        log = build.run_set(s, wd / name, GEO_SECONDS, zip_, env, ("geo",))["geo"]
        ram, counts = geo.PolyRam(), {}
        for frame, start, _, words in geo.parse_log(log):
            counts[frame] = geo.count_polygons(geo.walk(words, start // 4), s or rs, ram)
        log.unlink()                                     # 170 MB each
        out.append(counts)
    return out


@pytest.mark.mame
def test_mirrored_beginner_polygon_lists(polygon_counts):
    """A list on every frame of GEO_WINDOW on both sets, the original at most POLYGON_BAR per list (the procedure
    reproduces Phase 1's bar), and the mirror within 1 % of the original at the same frame on every frame of the
    window up to SYMMETRIC_TO (none while SYMMETRIC_TO is before GO)."""
    original, mirror = polygon_counts
    window = range(GEO_WINDOW[0], GEO_WINDOW[1] + 1)
    assert set(window) <= set(original) and set(window) <= set(mirror)
    assert max(original[f] for f in window) <= POLYGON_BAR
    symmetric = [f for f in window if f <= SYMMETRIC_TO]
    assert symmetric or SYMMETRIC_TO < GO
    assert all(abs(mirror[f] - original[f]) <= 0.01 * original[f] for f in symmetric)


@pytest.mark.mame
def test_mirrored_beginner_polygons(polygon_counts):
    """The M1c frame-rate gate on the mirror (the controller's ruling): at most POLYGON_GATE drawn polygons in every
    list of GEO_WINDOW (the new display list on every frame is test_mirrored_beginner_drives'). The mirror draws more
    than the original's bar POLYGON_BAR at a few frames (cells the original culls): docs/spike/2026-11-mirror.md,
    "Rules checked" (culling)."""
    _, mirror = polygon_counts
    assert max(mirror[f] for f in range(GEO_WINDOW[0], GEO_WINDOW[1] + 1)) <= POLYGON_GATE


def _outline(src, dst):
    """The HUD map outline's pixels of snapshot `src` as an image at `dst`: pure black (the outline's flat colour) as
    black, everything else white. The 3D scene behind the map is mirrored about the screen's centre, not about the
    map's axis, so the raw snapshots do not match there (0.20 over HUD_BOX)."""
    from PIL import Image
    a = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(np.where(a.max(axis=2) == 0, 0, 255).astype(np.uint8)).convert("RGB").save(dst)
    return dst


@pytest.mark.mame
def test_mirrored_hud_map_on_screen(drives, tmp_path):
    """At GO + 5 s the mirrored run's HUD map is the original's flipped left to right (world z is the screen's
    horizontal axis): the outline's pixels over HUD_BOX match at least HUD_MATCH, and clearly less unflipped (the
    comparison sees the flip)."""
    o, m = (_outline(r["snap"] / "daytona" / "0000.png", tmp_path / f"{k}.png") for k, r in zip("om", drives))
    flipped = build.snapshot_match(o, m, HUD_BOX, "h")
    assert flipped >= HUD_MATCH, flipped
    assert build.snapshot_match(o, m, HUD_BOX) < 0.9


def _sky(path) -> np.ndarray:
    """The original's backdrop sky over BACKDROP_BAND: blue-dominant pixels (the HUD, trees, wall and cars are not)
    outside HUD_BOXES and their mirror images."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hud = np.zeros(a.shape[:2], bool)
    for left, top, right, bottom in HUD_BOXES:
        hud[top:bottom, left:right] = True
    left, top, right, bottom = BACKDROP_BAND
    return ((b >= 160) & (b - r >= 60) & (b - g >= 30) & ~hud & ~hud[:, ::-1])[top:bottom, left:right]


@pytest.mark.mame
def test_mirrored_backdrop_on_screen(drives):
    """At GO + 5 s the mirrored run's horizon backdrop is the original's flipped left to right: over the band above the
    horizon, the original's sky pixels (outside the HUD) match the mirror's flipped at least BACKDROP_MATCH, and far
    less unflipped (the comparison sees the flip). The column offset MIRROR_COLUMN puts the flipped panorama where the
    mirrored camera yaw scrolls it."""
    o, m = (r["snap"] / "daytona" / "0000.png" for r in drives)
    sky = _sky(o)
    assert sky.sum() > 2000
    flipped = build.snapshot_match(o, m, BACKDROP_BAND, "h", mask=sky)
    assert flipped >= BACKDROP_MATCH, flipped
    assert build.snapshot_match(o, m, BACKDROP_BAND, mask=sky) < 0.5
