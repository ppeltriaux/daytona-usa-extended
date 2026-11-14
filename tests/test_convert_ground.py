"""Tests for the TGP ground converted from the Dreamcast course data (dtools/convert/ground.py).

The known answer: the Dreamcast's courses n 0-2 are the arcade's Beginner, Expert and Advanced, so their converted
grounds must equal the arcade's own (Review Focus 2: a corner order or sign slip passes every round trip, not this),
but for the off-road cells Phase 5 pinned (tests/test_dc_data.py, OFF_ROAD_DIFFERENCES). Desert City (n 3) must give
a ground the TGP can answer from at every point a car is led to (Review Focus 3)."""

import dataclasses
import functools
import math
import random
import struct

import numpy as np
import pytest

from dtools.arcade import tgp
from dtools.convert import ConvertError, ground, tables
from dtools.dc import data
from dtools.dc.data import Cell, CourseData
from tests.test_convert_tables import _data, disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_dc_data import OFF_ROAD_DIFFERENCES, make_data

# Addresses a test gives a converted ground so that tgp.encode_ground takes it (convert_ground leaves them 0 for
# Task 3's layout; ground_tables refuses a list block or pool inside the header): the grid and attribute block at
# data 0 and 0x8000, the list block at Beginner's word 0x30, the pool at word 0x8000 (16-word aligned, so no
# polygon crosses a 64K-word window; past the end of Expert's 22,074-word block from 0x30).
PLACED = dict(grid_at=0x0, attributes_at=0x8000, list_base=0x30, pool_base=0x8000)


def placed(g: tgp.Ground, like: tgp.Ground | None = None) -> tgp.Ground:
    """`g` with addresses: `like`'s four (the decoded original's), else PLACED."""
    if like is None:
        return dataclasses.replace(g, **PLACED)
    return dataclasses.replace(g, grid_at=like.grid_at, attributes_at=like.attributes_at, list_base=like.list_base,
                               pool_base=like.pool_base)


def area(p: tgp.GroundPoly) -> float:
    xz = [(c[0], c[2]) for c in p.corners]
    return sum(xz[k][0] * xz[(k + 1) % 4][1] - xz[(k + 1) % 4][0] * xz[k][1] for k in range(4)) / 2


def with_cell(d: CourseData, k: int, **kw) -> CourseData:
    return dataclasses.replace(d, cells=d.cells[:k] + (dataclasses.replace(d.cells[k], **kw),) + d.cells[k + 1:])


def square(x0: float, z0: float, x1: float, z1: float) -> tuple:
    """Four Dreamcast corners in make_data's order (x right, the Dreamcast's z up: positive area there)."""
    return ((x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1))


# ---- synthetic ----

def test_ground_synthetic():
    """make_data's five unit squares: corners and centre z-negated in the same order (the signed XZ area turns
    negative), direction - 0x8000 (s16); attributes copied; each polygon listed, in index order, in every grid cell
    from tgp.cell of its bounding box's low corner to that of its high corner; the lists laid out in cell order from
    word 1 (count, indices), the grid holding each list's word offset; base the course; addresses 0."""
    d = make_data()
    g = ground.convert_ground(d)
    assert isinstance(g, tgp.Ground) and len(g.polys) == 5
    for c, (cell, p) in enumerate(zip(d.cells, g.polys, strict=True)):
        assert p.corners == ((0.0, 0.0, -c), (1.0, 0.0, -c), (1.0, 0.0, -c - 1.0), (0.0, 0.0, -c - 1.0))
        assert p.centre == (0.5, 0.0, -c - 0.5)
        assert p.direction == 0x2AAB - 0x8000
        assert area(p) == -1.0
        assert struct.unpack(f"<{tgp.POLY_WORDS}I", tgp.encode_ground(placed(g))[("copro_data", 4 * 0x8000)]
                             [64 * c:64 * c + 64]) == p.raw
    assert math.copysign(1.0, g.polys[0].corners[0][2]) == 1.0          # 0.0 - 0.0: the arcade's +0.0, not -0.0
    assert g.attributes == bytes([1, 1, 1, 5, 5]) == tables.attributes(d)
    # x 0..1 is column 64; polygon 0 (z -1..0) reaches rows 63 and 64, polygons 1-4 (z down to -5) row 63 only
    lo, hi = 63 * 128 + 64, 64 * 128 + 64
    assert {k: v for k, v in enumerate(g.grid) if v} == {lo: 1, hi: 7}
    assert g.lists == {1: (0, 1, 2, 3, 4), 7: (0,)}
    assert g.raw == (0, 5, 0, 1, 2, 3, 4, 1, 0)
    assert (g.course, g.base, g.grid_at, g.attributes_at, g.list_base, g.pool_base) == (0, 0, 0, 0, 0, 0)
    # the list block encode_ground writes is `raw`
    blob = tgp.encode_ground(placed(g))[("copro_data", 4 * PLACED["list_base"])]
    assert struct.unpack(f"<{len(g.raw)}I", blob) == g.raw
    # the TGP finds every polygon from a point inside it
    for c in range(5):
        h, poly = tgp.query(g, 0.5, 0.0, -c - 0.5)
        assert poly == c and h == 0.0
    assert tgp.query(g, 0.5, 0.0, 0.5) is None
    assert ground.convert_ground(d, 2).base == 2 == ground.convert_ground(d, np.int64(2)).course


def test_ground_world_edges_and_types():
    """A polygon reaching the world's edges (x and z -1024 and 1023, WORLD inclusive) converts, listed in every cell;
    numpy numbers are taken; every value is a Python float or int."""
    assert ground.WORLD == (-1024, 1023)
    d = with_cell(make_data(), 4, corners=square(-1024.0, -1023.0, 1023.0, 1024.0))
    g = ground.convert_ground(d)
    assert [c[2] for c in g.polys[4].corners] == [1023.0, 1023.0, -1024.0, -1024.0]
    assert all(4 in g.lists[v] for v in g.grid)
    tgp.encode_ground(dataclasses.replace(g, grid_at=0x10000, attributes_at=0x0, list_base=0x30,
                                          pool_base=0x30000))
    d = with_cell(make_data(), 1, corners=tuple((np.float32(x), np.float64(y), np.int64(z))
                                                for x, y, z in d.cells[1].corners),
                  centre=(np.float32(0.5), 0, np.float64(1.5)), direction=np.uint16(0x8000))
    g = ground.convert_ground(d)
    p = g.polys[1]
    assert all(type(v) is float for c in p.corners + (p.centre,) for v in c) and type(p.direction) is int
    assert p.direction == 0 and p.centre == (0.5, 0.0, -1.5)


def test_ground_nudges_a_collinear_corner():
    """A corner on the line through its neighbours within f32 rounding but just on the notch side (the encoder's
    convexity test, MIN_AREA, refuses it; two of Desert City's off-road cells): moved outward by the fewest ulps
    that make the polygon convex (ground.repairs: "ulps"). A notch wider than MAX_MOVE is refused."""
    eps = 2.0 ** -20                                    # one ulp at 10
    # Dreamcast z up: a triangle (0, 0), (20, 0), (20, 20) with its 0-2 edge's midpoint (10, 10) as corner 3,
    # pushed a hair inside (the triangle lies below the line z = x)
    corners = ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 0.0, 20.0), (10.0, 0.0, 10.0 - eps))
    with pytest.raises(tgp.TgpError, match="convex"):          # plainly z-negated, the encoder refuses it
        tgp.encode_ground(_raw_ground(corners))
    d = with_cell(make_data(), 4, corners=corners, centre=(12.5, 0.0, 7.5))
    g = ground.convert_ground(d)
    p = g.polys[4]
    assert p.corners[:3] == ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 0.0, -20.0))
    moved = p.corners[3]
    assert moved != (10.0, 0.0, -(10.0 - eps)) and math.dist(moved, (10.0, 0.0, -(10.0 - eps))) <= 2 * eps
    assert ground.repairs(d) == {4: "ulps"} and ground.repairs(make_data()) == {}
    tgp.encode_ground(placed(g))
    for notch in (10.0 - 0.5, 10.0 - 0.1):             # moves of 0.35 and 0.071 units: past MAX_MOVE
        bad = with_cell(make_data(), 4, corners=corners[:3] + ((10.0, 0.0, notch),))
        with pytest.raises(ConvertError, match="cell 4.*convex.*MAX_MOVE"):
            ground.convert_ground(bad)
        with pytest.raises(ConvertError, match="cell 4.*convex"):
            ground.repairs(bad)


def _sloped(e: float, slope: float, lift: float = 0.0, rot: int = 0) -> tuple:
    """Dreamcast corners of the triangle (0, 0), (20, 0), (20, 20) on the plane y = slope * x, with a fourth corner
    `e` inside its 0-2 edge's midpoint (z 10 - e) and `lift` above that plane: a notch at corner 3, `rot` places
    it (the list rotated left by `rot`: the notch at corner (3 - rot) % 4)."""
    c = ((0.0, 0.0, 0.0), (20.0, 20.0 * slope, 0.0), (20.0, 20.0 * slope, 20.0), (10.0, 10.0 * slope + lift, 10.0 - e))
    return c[rot:] + c[:rot]


def _tgp_plane(corners, j: int):
    """The function h(x, z) of the TGP's triangle (the 0-2 split) that holds the cell's points when corner j is its
    notch (j 1: corners 0, 2, 3; j 3: 0, 1, 2), in float64."""
    a, b, c = (np.array(corners[i], float) for i in ((0, 2, 3) if j == 1 else (0, 1, 2)))
    n = np.cross(b - a, c - a)
    return lambda x, z: a[1] - (n[0] * (x - a[0]) + n[2] * (z - a[2])) / n[1]


def test_ground_moves_a_notch_corner_onto_the_line():
    """A notch wider than f32 rounding (courses 4-7, one cell each): its corner moves outward onto the line through
    its neighbours, by at most MAX_MOVE, then the ulp nudge closes it (ground.repairs: "line"). Off the 0-2 diagonal
    (corner 1 or 3) it takes the height of the TGP's plane that holds the cell, so the TGP answers that plane on the
    whole cell; on it (corner 0 or 2) the mean of the two planes there. Refused when the height the move can change
    (half the two planes' disagreement at the new place, plus the steepest plane's slope times the move: the
    polygon then reaches past the Dreamcast's cell by that much; plus the f32 error of the TGP's plane, F32_PLANE)
    passes HEIGHT_TOLERANCE."""
    for rot, j in ((0, 3), (2, 1)):
        corners = _sloped(0.01, 0.1, lift=0.3, rot=rot)
        d = with_cell(make_data(), 4, corners=corners, centre=(15.0, 1.5, 5.0))
        assert ground.repairs(d) == {4: "line"}
        g = ground.convert_ground(d)
        p, plain = g.polys[4], tuple((x, y, 0.0 - z) for x, y, z in corners)
        assert [k for k in range(4) if p.corners[k] != plain[k]] == [j]
        assert math.dist(p.corners[j][::2], (9.995, -9.995)) < 1e-4          # the projection onto x = -z
        h = _tgp_plane(plain, j)
        assert abs(p.corners[j][1] - h(p.corners[j][0], p.corners[j][2])) < 1e-5
        assert max(ground._turns(p.corners)) <= tgp.MIN_AREA and area(p) < 0
        g = placed(g)
        tgp.encode_ground(g)
        for x, z in ((15.0, -5.0), (19.0, -10.0), (10.0, -9.99), (5.0, -4.99), (19.9, -19.8)):
            got = tgp.query(g, x, h(x, z), z)
            assert got is not None and got[1] == 4 and abs(got[0] - h(x, z)) < 1e-5, (rot, x, z, got)
    # the notch at corner 2 (on the diagonal), the cell planar: the mean is that plane, nothing moves in height
    d = with_cell(make_data(), 4, corners=_sloped(0.01, 0.1, rot=1), centre=(15.0, 1.5, 5.0))
    assert ground.repairs(d) == {4: "line"}
    p = ground.convert_ground(d).polys[4]
    assert abs(p.corners[2][1] - 0.1 * p.corners[2][0]) < 1e-5
    # refused: slope 0.2 times the move 0.035 is 0.0071 > HEIGHT_TOLERANCE; on the diagonal, slope 0.15 with the
    # corner lifted 1 unit (its triangles' planes steeper, and disagreeing at the new place): 0.0064 (each with the
    # f32 term, about 1e-4)
    for corners in (_sloped(0.05, 0.2), _sloped(0.05, 0.15, lift=1.0, rot=1)):
        bad = with_cell(make_data(), 4, corners=corners)
        with pytest.raises(ConvertError, match="cell 4.*convex.*HEIGHT_TOLERANCE"):
            ground.convert_ground(bad)
    assert ground.MAX_MOVE == 0.0625 and ground.HEIGHT_TOLERANCE == 0.005


def _raw_ground(dc_corners) -> tgp.Ground:
    """make_data's ground with polygon 4 the plain z-negation of `dc_corners` (no nudge), placed."""
    g = placed(ground.convert_ground(make_data()))
    p = tgp.GroundPoly(tuple((x, y, 0.0 - z) for x, y, z in dc_corners))
    return dataclasses.replace(g, polys=g.polys[:4] + (p,))


def test_ground_refuses():
    """A corner outside WORLD, a non-finite corner, a grid value naming no list (and every other value the arcade's
    ground cannot hold, or a CourseData the race tables refuse): ConvertError."""
    d = make_data()
    c4 = d.cells[4].corners

    def corner(j, v):
        return c4[:j] + (v,) + c4[j + 1:]

    grid = list(d.grid)
    cases = [
        (dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=corner(1, (1023.5, 0.0, 5.0))),)),
         "cell 4.*corner 1.*outside"),
        (dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=corner(0, (-1024.5, 0.0, 4.0))),)),
         "cell 4.*corner 0.*outside"),
        (dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=corner(2, (1.0, 0.0, -1023.5))),)),
         "cell 4.*corner 2.*outside"),          # Dreamcast z -1023.5 is the arcade's 1023.5
        (dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=corner(3, (0.0, 0.0, 1024.5))),)),
         "cell 4.*corner 3.*outside"),
    ]
    for bad in (float("nan"), float("inf"), -float("inf"), 1e39, 10**400, True, np.bool_(False), "1", None):
        cases.append((dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=corner(2, (1.0, bad, 5.0))),)),
                      "cell 4.*corner 2"))
        cases.append((dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], centre=(0.5, 0.0, bad)),)),
                      "cell 4.*centre"))
    for bad in (-1, 0x10000, 0x1FFFF, True, 1.5, None):
        cases.append((dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], direction=bad),)),
                      "cell 4.*direction"))
    for v in (3, 0x7000, -1, True, 1.0, None):
        g2 = list(grid)
        g2[5 * 128 + 7] = v
        cases.append((dict(grid=tuple(g2)), r"grid square 647 \(row 5, column 7\)"))
    cases += [
        (dict(cells=d.cells[:4] + (None,)), "cell 4.*Cell"),
        (dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=c4[:3]),)), "cell 4.*corners"),
        (dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=tuple(reversed(c4))),)),
         "cell 4.*area"),                               # the other winding
        (dict(cells=d.cells[:4] + (dataclasses.replace(d.cells[4], corners=square(0.0, 4.0, 0.0, 5.0)),)),
         "cell 4.*area"),                               # no XZ extent
        (dict(grid=d.grid[:-1]), "grid"),
        (dict(grid=None), "grid"),
        (dict(lists=d.lists[:2] + ((2, 5),)), "list 2.*cell 5"),
        (dict(lists=d.lists[:2] + ((2, -1),)), "list 2"),
        (dict(lists=d.lists[:2] + (None,)), "list 2"),
        (dict(lists=None), "lists"),
        (dict(lists=d.lists[:2]), r"grid square .*list 2"),
        (dict(attributes=d.attributes[:4]), "attributes"),      # the race tables' check (tables._check)
        (dict(steps=()), "steps"),
        (dict(cells=None), "cells"),
    ]
    for kw, match in cases:
        with pytest.raises(ConvertError, match=match):
            ground.convert_ground(dataclasses.replace(d, **kw))
    for course in (-1, 16, True, np.bool_(False), 1.0, "0", None):     # ids 0-15 (a grown set's)
        with pytest.raises(ConvertError, match="course"):
            ground.convert_ground(d, course)
    for bad in (None, "CS_DATA3.BIN", dataclasses.astuple(d)):
        with pytest.raises(ConvertError, match="CourseData"):
            ground.convert_ground(bad)
    # a list block whose grid values would pass the u16 a grid square holds: four polygons over the whole world,
    # so every one of the 16,384 squares has a list of at least 4 (5 words): 81,920 words
    world = Cell(square(-1024.0, -1023.0, 1023.0, 1024.0), (0.0, 0.0, 0.0), 0)
    many = dataclasses.replace(d, cells=d.cells + (world,) * 4, attributes=d.attributes + bytes(4))
    with pytest.raises(ConvertError, match="u16"):
        ground.convert_ground(many)


def test_ground_never_raises_other_errors():
    """Fuzz: make_data with cells, grid or lists fields swapped for junk converts or is refused with ConvertError,
    never another error; whatever converts, tgp.encode_ground takes (placed)."""
    d = make_data()
    rng = random.Random(23)
    junk = (None, True, np.bool_(False), -1, 0x10000, 0x8000, 10**400, float("nan"), float("inf"), 1e39, -2000.0,
            1100.0, "x", 1.5, (), (1,), (1.0, 2.0, 3.0), np.int64(3), np.float32(2.0), 0, 1, 2)
    converted = 0
    for _ in range(2000):
        cells = list(d.cells)
        grid = list(d.grid)
        lists = [list(v) for v in d.lists]
        top = {}
        for _ in range(rng.randint(1, 3)):
            where = rng.random()
            v = rng.choice(junk)
            if where < 0.5:
                k = rng.randrange(5)
                f = rng.choice(("corners", "centre", "direction", "corner", "xyz"))
                base = cells[k].corners                 # an earlier swap may have left junk: start again from d's
                if not (isinstance(base, tuple) and len(base) == 4 and all(isinstance(c, tuple) for c in base)):
                    base = d.cells[k].corners
                if f == "corner":
                    cs = list(base)
                    cs[rng.randrange(4)] = v
                    cells[k] = dataclasses.replace(cells[k], corners=tuple(cs))
                elif f == "xyz":
                    cs = [list(c) for c in base]
                    cs[rng.randrange(4)][rng.randrange(3)] = v
                    cells[k] = dataclasses.replace(cells[k], corners=tuple(map(tuple, cs)))
                else:
                    cells[k] = dataclasses.replace(cells[k], **{f: v})
            elif where < 0.7:
                grid[rng.choice((0, 64 * 128 + 64, 64 * 128 + 65, rng.randrange(16384)))] = v
            elif where < 0.9:
                k = rng.randrange(3)
                if isinstance(lists[k], list) and lists[k] and rng.random() < 0.5:
                    lists[k][rng.randrange(len(lists[k]))] = v
                else:
                    lists[k] = v
            else:
                top[rng.choice(("cells", "grid", "lists", "attributes", "steps"))] = v
        fields = dict(cells=tuple(cells), grid=tuple(grid),
                      lists=tuple(tuple(v) if isinstance(v, list) else v for v in lists))
        bad = dataclasses.replace(d, **(fields | top))
        try:
            g = ground.convert_ground(bad)
        except ConvertError:
            continue
        converted += 1
        tgp.encode_ground(placed(g))
    assert converted > 50                   # measured 88: legal junk (a numpy number, 1100.0 as a y, 0-2 as an index)


# ---- the disc against the arcade ----

@functools.lru_cache(maxsize=1)
def _romset(path):
    from dtools.arcade.romset import RomSet
    return RomSet.from_zip(path)


def _same_ground(n: int, d: CourseData, rs) -> tgp.Ground:
    """Review Focus 2: n's converted ground against the arcade's id n: the polygon count; the road polygons' corners,
    centre and direction exactly, as values and as the f32 words encode_ground writes; the off-road polygons that
    differ (corners or centre moved, direction turned) exactly OFF_ROAD_DIFFERENCES' counts, and nothing else differs
    (the pool's records differ only there); the grid, the lists and the list block equal; the attributes equal."""
    road, moved, turned = OFF_ROAD_DIFFERENCES[n]
    want = tgp.decode_ground(rs, n)
    g = ground.convert_ground(d, n)
    assert len(data.road_cells(d)) == road
    assert len(g.polys) == len(want.polys) == len(d.cells)
    for k in range(road):
        p, q = g.polys[k], want.polys[k]
        assert (p.corners, p.centre, p.direction) == (q.corners, q.centre, q.direction), (n, k)

    def place(p, q):
        return p.corners != q.corners or p.centre != q.centre

    assert sum(place(p, q) for p, q in zip(g.polys, want.polys, strict=True)) == moved
    assert sum(p.direction != q.direction for p, q in zip(g.polys, want.polys, strict=True)) == turned
    differ = {k for k, (p, q) in enumerate(zip(g.polys, want.polys)) if place(p, q) or p.direction != q.direction}
    assert min(differ, default=road) >= road
    # the pool's bytes: the road's records byte for byte (no zero of the other sign), the rest only where they differ
    mine, theirs = tgp.encode_ground(placed(g, want)), tgp.encode_ground(want)
    pool = ("copro_data", 4 * want.pool_base)
    assert {k for k in range(len(g.polys)) if mine[pool][64 * k:64 * k + 64] != theirs[pool][64 * k:64 * k + 64]} \
        == differ
    assert all(p.raw == struct.unpack("<16I", mine[pool][64 * k:64 * k + 64]) for k, p in enumerate(g.polys))
    # the grid, lists and block: the arcade's own, rebuilt from the polygons
    assert g.grid == want.grid
    assert g.lists == want.lists
    assert g.raw == want.raw
    assert g.attributes == want.attributes == tables.attributes(d)
    assert (g.course, g.base) == (n, n)
    assert mine == {k: v for k, v in theirs.items() if k != pool} | {pool: mine[pool]}
    return g


def test_ground_beginner(disc, romset_zip):
    """n 0 (Three Seven Speedway) converts to Beginner's ground (id 0): 1,184 polygons, the 503 road ones exact, 2
    off-road cells moved and 8 turned (as Phase 5 pinned), grid, lists and attributes equal."""
    g = _same_ground(0, _data(disc, 0), _romset(romset_zip))
    assert (len(g.polys), len(g.lists), len(g.raw)) == (1184, 626, 4714)


def test_ground_expert_advanced(disc, romset_zip):
    """n 1 / 2 convert to Expert's / Advanced's grounds (ids 1 / 2): 43 / 3 off-road cells moved, none turned."""
    for n, counts in ((1, (7239, 2315, 22074)), (2, (2986, 1288, 10290))):
        g = _same_ground(n, _data(disc, n), _romset(romset_zip))
        assert (len(g.polys), len(g.lists), len(g.raw)) == counts


def test_ground_dc_grid_differs(disc, romset_zip):
    """The Dreamcast's grid (H2, rows flipped for the negated z) and lists (H7) give Beginner's list as a sequence
    on 15,815 of the 16,384 squares, as a set on 43 more, and another set on 526: H7 lists a cell by a looser rule
    (its bounding box within about 1.4 units of the square, docs/spike/2026-11-dc-decode.md). So the converter
    rebuilds the grid by the arcade's rule, which test_ground_beginner shows is exact."""
    d = _data(disc, 0)
    want = tgp.decode_ground(_romset(romset_zip), 0)
    seq = same = 0
    for k in range(tgp.GRID_CELLS):
        a = want.lists.get(want.grid[k], ())
        b = d.lists[d.grid[(127 - (k >> 7)) * 128 + (k & 127)]]
        seq += tuple(a) == tuple(b)
        same += tuple(a) != tuple(b) and sorted(a) == sorted(b)
    assert (seq, same, tgp.GRID_CELLS - seq - same) == (15815, 43, 526)


def _lane_points(road):
    return [ln.position for p in road for ln in p.used_lanes]


def test_ground_queries_beginner(disc, romset_zip):
    """Every road lane point and every AI point of id 0: tgp.query (with the TGP's 1/x table) on the converted
    ground gives the same polygon and height as on the decoded original. No point is over a pinned off-road cell
    (one whose list names a moved polygon), so none is excused."""
    from dtools.arcade import course as C
    rs = _romset(romset_zip)
    want = tgp.decode_ground(rs, 0)
    g = ground.convert_ground(_data(disc, 0), 0)
    inv = tgp.inv_table(rs)
    entry = C.course_table(rs)[0]
    points = _lane_points(C.road_line(rs, entry)) + [p.position for line in C.ai_lines(rs, entry) for p in line]
    assert len(points) > 2500
    moved = {k for k, (p, q) in enumerate(zip(g.polys, want.polys)) if p.corners != q.corners}
    excused = 0
    for x, y, z in points:
        if moved & set(want.lists.get(want.grid[tgp.cell(x, z)], ())):
            excused += 1
            continue
        a, b = tgp.query(g, x, y, z, inv), tgp.query(want, x, y, z, inv)
        assert a is not None and a == b, (x, y, z, a, b)
    assert excused == 0


def test_ground_desert_city(disc):
    """Review Focus 3, n 3 (Desert City): it converts; encode_ground takes it; every polygon's signed XZ area is
    negative and every corner inside WORLD; every AI point and every road lane point queries a polygon. Two
    off-road cells (1689, 1915) have a corner collinear within f32 rounding on the notch side; each is moved one ulp
    outward (convert_ground's nudge), every other corner is the Dreamcast's z-negated."""
    d = _data(disc, 3)
    g = ground.convert_ground(d)
    assert (len(g.polys), len(d.steps), len(g.lists), len(g.raw)) == (3073, 527, 993, 8739)
    assert all(area(p) < 0 for p in g.polys)
    lo, hi = ground.WORLD
    assert all(lo <= c[0] <= hi and lo <= c[2] <= hi for p in g.polys for c in p.corners)
    blocks = tgp.encode_ground(placed(g))
    assert len(blocks[("copro_data", 4 * PLACED["pool_base"])]) == 64 * 3073
    nudged = {}
    for k, (cell, p) in enumerate(zip(d.cells, g.polys, strict=True)):
        plain = tuple((x, y, 0.0 - z) for x, y, z in cell.corners)
        if p.corners != plain:
            nudged[k] = max(math.dist(a, b) for a, b in zip(p.corners, plain))
        assert p.centre == (cell.centre[0], cell.centre[1], 0.0 - cell.centre[2])
        assert p.direction == (cell.direction - 0x8000 + 0x8000) % 0x10000 - 0x8000
    assert sorted(nudged) == [1689, 1915] and max(nudged.values()) < 4e-6
    assert all(d.attributes[k] & 15 != 1 for k in nudged)          # off the road
    points = _lane_points(tables.road_line(d)) + [p.position for line in tables.ai_lines(d) for p in line]
    assert len(points) > 3700
    misses = [pt for pt in points if tgp.query(g, *pt) is None]
    assert misses == []


# ---- courses 4-7: the ground repair (Phase 10 Task 2) ----

# What the TGP's convexity check refuses per course before any repair, the count and its first cell (Phase 7):
# every such cell is repaired, by the ulp nudge or (one cell a course) the move onto the line
REFUSED = {4: (11, 2082), 5: (19, 410), 6: (10, 2256), 7: (13, 2334)}


def _notch(corners) -> int | None:
    """The corner where the plain z-negated `corners` turn the wrong way (tgp's test), or None."""
    t = ground._turns(corners)
    k = max(range(4), key=lambda i: t[i])
    return None if t[k] <= tgp.MIN_AREA else (k + 1) % 4


def test_ground_converts_4_to_7(disc):
    """n 4-7 convert: every polygon convex (no turn past MIN_AREA) with negative XZ area, tgp.encode_ground takes
    the ground (placed), one polygon per cell. The cells repaired are exactly the ones the TGP's check refuses
    (REFUSED's counts, ground.REPAIRED pinned): the move onto the line only on REFUSED's first cell, the rest the
    ulp nudge; every repaired polygon differs from the plain z-negated cell in its notch corner only, by at most
    MAX_MOVE ("line") or NUDGE_ULPS ulps a side ("ulps"); every other polygon is the plain cell."""
    for n, (count, first) in REFUSED.items():
        d = _data(disc, n)
        g = ground.convert_ground(d, n)
        assert len(g.polys) == len(d.cells) and (g.course, g.base) == (n, n)
        tgp.encode_ground(placed(g))
        assert all(max(ground._turns(p.corners)) <= tgp.MIN_AREA and area(p) <= -tgp.MIN_AREA for p in g.polys)
        refused = {}
        for k, (cell, p) in enumerate(zip(d.cells, g.polys, strict=True)):
            plain = tuple((x, y, 0.0 - z) for x, y, z in cell.corners)
            j = _notch(plain)
            if j is None:
                assert p.corners == plain, (n, k)
                continue
            refused[k] = j
            assert [i for i in range(4) if p.corners[i] != plain[i]] == [j], (n, k)
            moved = math.dist(p.corners[j][::2], plain[j][::2])
            if k == first:
                assert moved <= ground.MAX_MOVE
            else:
                ulp = float(np.spacing(np.float32(max(abs(v) for c in plain for v in c[::2]))))
                assert p.corners[j][1] == plain[j][1] and moved <= ground.NUDGE_ULPS * ulp * math.sqrt(2)
        assert len(refused) == count and min(k for k, how in ground.REPAIRED[n].items() if how == "line") == first
        assert ground.repairs(d) == ground.REPAIRED[n] == {k: "line" if k == first else "ulps" for k in refused}
        print(f"\nn {n}: {len(g.polys)} polygons, {len(refused)} repaired, notch corners "
              f"{sorted(set(refused.values()))}, the line's corner {refused[first]}")


# n 0-3's converted grounds (sha256 of encode_ground's blocks, placed at PLACED, in key order), before and after
# the repair: Review Focus 1, Desert City byte-identical and the originals untouched
GROUND_SHA256 = {
    0: "bc8e0a7f13508078ddbdf966146583df434b7c336a7866cba6c3db54b0c4938e",
    1: "1f178d230b2f384fc9c2d2ccb87715c53ac4eafc154a49bbe55ce5e58117aa92",
    2: "b876406d80dc3c938a4fee891771463ba9c54c12fe6cb94585d5eddc556f71ae",
    3: "4570492fc03a09d177b88a79867ce5050bd3048786b388d12af4baa5fad77b1b",
}


def test_desert_city_ground_unchanged(disc):
    """Review Focus 1: the repair changes no cell that converted before it. n 3's (Desert City) converted ground and
    n 0-2's are the bytes they were before Phase 10 Task 2 (GROUND_SHA256, recorded on the code before it); n 3's
    only repairs are its two ulp nudges, n 0-2 have none."""
    import hashlib
    for n, want in GROUND_SHA256.items():
        d = _data(disc, n)
        blocks = tgp.encode_ground(placed(ground.convert_ground(d, n)))
        assert hashlib.sha256(b"".join(b for _, b in sorted(blocks.items()))).hexdigest() == want, n
        assert ground.repairs(d) == ground.REPAIRED.get(n, {})
    assert ground.REPAIRED[3] == {1689: "ulps", 1915: "ulps"} and all(n not in ground.REPAIRED for n in (0, 1, 2))


EXACT = 1e-6        # a point this close to a cell's XZ outline is on it (float64 rounding of the test's own sums)
F32_EDGE = 6 * 2.0 ** -24   # the TGP's edge test: six f32 roundings, each at most this relative (section 7's order)


def _arcade_height(plain, x: float, z: float) -> float:
    """The height the TGP's rule gives a Dreamcast cell at (x, z), float64 (docs/formats/model2-tgp-ground.md section
    7, which n 0-2's cells reproduce): (x, z) - corner 2 = alpha (corner 0 - corner 2) + beta (corner 1 - corner 2),
    both >= 0 the plane of corners 0-1-2, else 0-2-3. Inside a cell that is its triangle of the 0-2 split; just
    outside (where the TGP's f32 edge test admits a point) the plane the TGP runs on with. A cell with its notch at
    corner 1 or 3 lies wholly in one triangle (0-2-3 or 0-1-2): that plane (the rule's other one is a sliver). This
    is what the conversion gives every convex cell, so the repair's tolerance is measured against it."""
    j = _notch(plain)
    if j in (1, 3):
        tri = (0, 2, 3) if j == 1 else (0, 1, 2)
    else:
        (x0, _, z0), (x1, _, z1), (x2, _, z2) = plain[0], plain[1], plain[2]
        p, q, u, v, s, t = x0 - x2, z0 - z2, x1 - x2, z1 - z2, x - x2, z - z2
        det = p * v - q * u
        alpha, beta = (s * v - t * u) / det, (p * t - q * s) / det
        tri = (0, 1, 2) if alpha >= 0 and beta >= 0 else (0, 2, 3)
    return _tri_height(plain, tri, x, z)


def _tri_height(plain, tri, x: float, z: float) -> float:
    a, b, c = (np.array(plain[i], float) for i in tri)
    nrm = np.cross(b - a, c - a)
    return float(a[1] - (nrm[0] * (x - a[0]) + nrm[2] * (z - a[2])) / nrm[1])


def _split13_height(plain, x: float, z: float) -> float:
    """A cell's height split along its 1-3 diagonal instead (triangles 0-1-3 and 1-2-3): the repair not chosen."""
    (_, _, _), (x1, _, z1), _, (x3, _, z3) = plain
    side = lambda px, pz: (x3 - x1) * (pz - z1) - (z3 - z1) * (px - x1)      # noqa: E731
    return _tri_height(plain, (0, 1, 3) if side(x, z) * side(plain[0][0], plain[0][2]) >= 0 else (1, 2, 3), x, z)


def _dc_routine(dc_corners, x: float, z: float) -> float:
    """The Dreamcast's own height of a cell at arcade (x, z), float64: 1ST_READ.BIN's routine at 0x8c034ee8
    (docs/spike/2026-11-courses-4-7.md, "Ground repair", disassembly), in the Dreamcast's frame (z = -z). A = the
    point's projection on edge 0-1 (parameter t), B = where the line from A through the point meets edge 3-2
    (parameter s; 0 when parallel), u = |A - point| / |A - B| (XZ distances, so unsigned); height = yA + u (yB - yA)
    with yA = y0 + t (y1 - y0), yB = y3 + s (y2 - y3). On a planar convex cell that is the plane; on a non-planar one
    a ruled surface, which neither diagonal split gives."""
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = dc_corners
    z = -z
    dx, dz = x1 - x0, z1 - z0
    t = ((x - x0) * dx + (z - z0) * dz) / (dx * dx + dz * dz)
    ax, az = x0 + t * dx, z0 + t * dz
    ex, ez = x2 - x3, z2 - z3
    den = (x - ax) * ez - ex * (z - az)
    s = 0.0 if den == 0 else ((z - az) * (x3 - x0) + (z - az) * (x0 - ax) - (z0 - az) * (x - ax)
                               - (z3 - z0) * (x - ax)) / den
    bx, bz = x3 + s * ex, z3 + s * ez
    u = math.hypot(ax - x, az - z) / math.hypot(ax - bx, az - bz)
    ya, yb = y0 + t * (y1 - y0), y3 + s * (y2 - y3)
    return ya + u * (yb - ya)


def _edge_slack(plain, k: int, x: float, z: float) -> float:
    """How far outside edge k the TGP's f32 edge test can admit (x, z), a priori: its six terms (section 7: z X_k,
    x Z_k, z X_k+1, x Z_k+1, Z_k+1 X_k, X_k+1 Z_k) each rounded by at most F32_EDGE of their magnitude, over the
    edge's length (the test's d is the edge length times the distance)."""
    (ax, _, az), (bx, _, bz) = plain[k], plain[(k + 1) % 4]
    terms = abs(z * ax) + abs(x * az) + abs(z * bx) + abs(x * bz) + abs(bz * ax) + abs(bx * az)
    return F32_EDGE * terms / max(math.hypot(bx - ax, bz - az), 1e-30)


def _near(plain, x: float, z: float, tgp_edge: bool = False) -> bool:
    """(x, z) inside the cell's XZ outline (even-odd) or within EXACT of an edge; with `tgp_edge` within each edge's
    _edge_slack instead (where the TGP's f32 edge test may still hit the polygon)."""
    inside = False
    for k in range(4):
        (ax, _, az), (bx, _, bz) = plain[k], plain[(k + 1) % 4]
        if (az > z) != (bz > z) and x < ax + (z - az) * (bx - ax) / (bz - az):
            inside = not inside
        ex, ez = bx - ax, bz - az
        t = min(1.0, max(0.0, ((x - ax) * ex + (z - az) * ez) / (ex * ex + ez * ez or 1.0)))
        if math.hypot(x - ax - t * ex, z - az - t * ez) <= (_edge_slack(plain, k, x, z) if tgp_edge else EXACT):
            return True
    return inside


class _DcGround:
    """The Dreamcast's cells (z negated, unrepaired) with a 16-unit square index, answering as the TGP chooses."""

    def __init__(self, d: CourseData):
        self.cells = d.cells
        self.plain = [tuple((float(x), float(y), 0.0 - float(z)) for x, y, z in c.corners) for c in d.cells]
        self.squares: dict[tuple[int, int], list[int]] = {}
        for k, p in enumerate(self.plain):
            xs, zs = [c[0] for c in p], [c[2] for c in p]
            for i in range(math.floor((min(xs) - EXACT) / 16), math.floor((max(xs) + EXACT) / 16) + 1):
                for j in range(math.floor((min(zs) - EXACT) / 16), math.floor((max(zs) + EXACT) / 16) + 1):
                    self.squares.setdefault((i, j), []).append(k)

    def answer(self, x: float, y: float, z: float, layer: bool = False) -> tuple[float, int] | None:
        """(height, cell) of the cells (x, z) is on (exactly: _near) with the smallest | |height| - y | (the TGP's
        choice; on ground below 0 it favours the smallest |height|), or with `layer` the smallest | height - y | (the
        ground nearest y); heights by _arcade_height; None when no cell is on (x, z)."""
        best = None
        for k in self.squares.get((math.floor(x / 16), math.floor(z / 16)), ()):
            if _near(self.plain[k], x, z):
                h = _arcade_height(self.plain[k], x, z)
                miss = abs(h - y) if layer else abs(abs(h) - y)
                if best is None or miss < best[0]:
                    best = (miss, h, k)
        return None if best is None else (best[1], best[2])


def _lattice(corners, steps: int = 8):
    """The steps x steps bilinear lattice of a quad's corners, (x, z) at parameters (i + 0.5) / steps."""
    out = []
    for i in range(steps):
        for j in range(steps):
            u, v = (i + 0.5) / steps, (j + 0.5) / steps
            w = ((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v)
            out.append((sum(wk * c[0] for wk, c in zip(w, corners)), sum(wk * c[2] for wk, c in zip(w, corners))))
    return out


# The one point where the TGP's answer is not the exact hit set's: n 6, the lattice of (ulp-nudged) cell 1062 at its
# corner 0, where the TGP's f32 edge test also hits the unrepaired sliver cell 2226 (0.007 units outside it) and,
# the ground below 0, the arcade's | |height| - y | prefers 2226's plane 0-2-3 there, 0.0206 units higher. The
# arcade's own rule on cells the repair did not touch: (n, kind, answering cell, x, z, error).
EXCEPTION = (6, "cell 1062", 2226, -523.64, 103.91, 0.0206)


def test_repaired_ground_heights(disc, romset_zip):
    """Review Focus 2, n 4-7: tgp.query (the TGP's 1/x table) on the repaired ground answers within
    ground.HEIGHT_TOLERANCE (0.005 units, stated before measuring: docs/spike/2026-11-courses-4-7.md, "Ground
    repair") of the TGP's rule on the unrepaired Dreamcast cells (_arcade_height: what the conversion gives every
    convex cell, so the repair's own error), at every road lane point and every AI point (y their own) and on an
    8 x 8 bilinear lattice over each repaired cell and over its repaired polygon (y the height there of the cell
    nearest the repaired cell's plane: a car standing on that ground). The reference answer: of the cells the point
    is exactly on, the one the TGP's choice would pick (_DcGround). Every point is answered; every answer is its own
    polygon's cell's height within the tolerance, that cell within its f32 edge reach of the point (_edge_slack); and
    every answer is the reference's within the tolerance but EXCEPTION, named: its answering cell unrepaired, the
    error 0.0206, the only one. The Dreamcast's own routine is test_repaired_ground_against_the_dreamcast's."""
    inv = tgp.inv_table(_romset(romset_zip))
    exceptions = []
    for n in REFUSED:
        d = _data(disc, n)
        g, dc = ground.convert_ground(d, n), _DcGround(d)
        points = [(x, y, z, "road") for x, y, z in _lane_points(tables.road_line(d))]
        points += [(*p.position, "ai") for line in tables.ai_lines(d) for p in line]
        for k in ground.REPAIRED[n]:
            for x, z in _lattice(dc.plain[k]) + _lattice(g.polys[k].corners):
                want = dc.answer(x, _arcade_height(dc.plain[k], x, z), z, layer=True)
                assert want is not None, (n, k, x, z)
                points.append((x, want[0], z, f"cell {k}"))
        worst, line = {}, REFUSED[n][1]
        for x, y, z, kind in points:
            got, want = tgp.query(g, x, y, z, inv), dc.answer(x, y, z)
            assert got is not None and want is not None, (n, kind, x, y, z, got, want)
            own = dc.plain[got[1]]
            assert _near(own, x, z, tgp_edge=True), (n, kind, x, y, z, got)
            assert abs(got[0] - _arcade_height(own, x, z)) <= ground.HEIGHT_TOLERANCE, (n, kind, x, y, z, got)
            err = abs(got[0] - want[0])
            if err > ground.HEIGHT_TOLERANCE:
                exceptions.append((n, kind, got[1], x, z, err))
                continue
            key = kind if kind in ("road", "ai") else "line" if kind == f"cell {line}" else "lattice"
            if err > worst.get(key, (-1.0,))[0]:
                worst[key] = (err, kind, x, z)
        print(f"\nn {n}: {len(points)} points; worst " +
              ", ".join(f"{key} {v[0]:.3g} ({v[1]} at {v[2]:.2f}, {v[3]:.2f})" for key, v in sorted(worst.items())))
    print(f"exceptions: {exceptions}")
    n, kind, cell, x, z, err = EXCEPTION
    assert len(exceptions) == 1 and exceptions[0][:3] == (n, kind, cell), exceptions
    assert cell not in ground.REPAIRED[n] and math.dist(exceptions[0][3:5], (x, z)) < 0.01
    assert abs(exceptions[0][5] - err) < 0.0005


def test_repaired_ground_against_the_dreamcast(disc, romset_zip):
    """The Dreamcast's own height routine (_dc_routine, 1ST_READ.BIN 0x8c034ee8) blends a non-planar cell as a ruled
    surface, which no diagonal split gives, so the conversion's TGP rule (_arcade_height) differs from it by the
    cell's non-planarity wherever a cell is not planar, repaired or not: on the road itself by up to 0.12 / 0.063 /
    0.23 / 0.13 units on n 4-7 (Desert City, which races: 0.24). The repair must add no more than
    HEIGHT_TOLERANCE to that: on the 8 x 8 lattice over each repaired cell, the repaired polygon's TGP height is within
    the tolerance plus the unrepaired cell's own gap of the routine. Named, as the one repaired cell whose whole
    distance from the routine passes the tolerance (the absolute gap, not what the repair adds: that stays within the
    tolerance on 2256 too, by the assert above on every lattice point): n 6 cell 2256 ("line", its notch corner 3
    0.011 off the plane of 0-1-2), 0.0114 from the routine; split along 1-3 instead it would be 0.0065, past it too
    (no split the TGP can hold gives the routine's surface); the other "line" cells' whole distances within the
    tolerance either way."""
    recip = tgp._recip_fn(tgp.inv_table(_romset(romset_zip)))
    road_gap = {}
    for n in REFUSED:
        d = _data(disc, n)
        g, dc = ground.convert_ground(d, n), _DcGround(d)
        for k, how in ground.REPAIRED[n].items():
            gap = split = 0.0
            for x, z in _lattice(dc.plain[k]):
                h = float(tgp._height(g.polys[k].corners, tgp._F(x), tgp._F(z), recip))
                want = _dc_routine(d.cells[k].corners, x, z)
                own = abs(_arcade_height(dc.plain[k], x, z) - want)
                assert abs(h - want) <= own + ground.HEIGHT_TOLERANCE, (n, k, x, z, h, want, own)
                gap, split = max(gap, abs(h - want)), max(split, abs(_split13_height(dc.plain[k], x, z) - want))
            if how == "line":
                print(f"\nn {n} cell {k}: TGP to the routine {gap:.3g}, a 1-3 split {split:.3g}")
                if (n, k) == (6, 2256):
                    assert 0.011 < gap < 0.012 and 0.006 < split < 0.007
                else:
                    assert gap <= ground.HEIGHT_TOLERANCE and split <= ground.HEIGHT_TOLERANCE
        steps = len(d.steps)
        for x, y, z in _lane_points(tables.road_line(d)):
            want = dc.answer(x, y, z)
            if want is not None and want[1] < steps:
                road_gap[n] = max(road_gap.get(n, 0.0), abs(want[0] - _dc_routine(d.cells[want[1]].corners, x, z)))
    print(f"the conversion's rule to the routine on the road: {road_gap}")
    assert {n: round(v, 2) for n, v in road_gap.items()} == {4: 0.12, 5: 0.06, 6: 0.23, 7: 0.13}
