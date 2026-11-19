"""The arcade's TGP ground (`tgp.Ground`) from a Dreamcast course: its cells (H6) become the polygons, the grid and
the lists are rebuilt by the arcade's rule, the attribute bytes are the race tables' (`tables.attributes`).

The rules (docs/spike/2026-11-convert.md, "Ground"; on n 0-2 the result equals the arcade's ids 0-2, grid, lists
and list block included, but for the off-road cells Phase 5 pinned: tests/test_dc_data.py, OFF_ROAD_DIFFERENCES):

- polygons: the cells in the Dreamcast's order (the road first, one per step, in step order). Each corner (x, y, z)
  becomes (x, y, 0.0 - z), in the same corner order: the Dreamcast's cells have positive signed XZ area in its own
  z (all eight courses), so negating z gives the arcade's negative area (the TGP's edge test is one-sided) with no
  reordering. `0.0 - z` keeps a zero +0.0, as the arcade's are (n 2's road has such z: byte for byte only so). The
  centre likewise; the direction word (a u16 on the disc) - 0x8000, as s16 (DIRECTION_OFFSET). A file whose every
  cell is wound the other way (the R files of n 0-2: each cell the plain one's corners mirrored) is re-wound whole,
  each cell's corners reversed (`rewound`), which gives the plain cell turned by two, as n 3-7's R files hold it;
  a file with both windings is refused. A reverse file's cell neither repair below makes convex takes the plain
  course's same-index cell (`plain=`), its corners in the R cell's order, when each is within PLAIN_SNAP of one of
  the R cell's (4R's off-road cell 1765, two corners 0.1 off in x and z); its centre and direction stay the R file's.
- a corner that lies on the line through its two neighbours within f32 rounding (at most NUDGE_ULPS ulps from it)
  but on the notch side is moved outward by the fewest f32 ulps (at most NUDGE_ULPS on each axis) that make the
  polygon convex: tgp's encoder refuses the notch, which is narrower than an ulp (n 3 has two such cells, off the
  road, moved 1.9e-6 units). The cell is repaired "ulps" (`repairs`, `REPAIRED`).
- a wider notch (courses 4-7, one cell each: 0.000115 to 0.0489 units; docs/spike/2026-11-courses-4-7.md, "Ground
  repair") has its corner moved outward onto the line through its neighbours (the projection, at most MAX_MOVE
  away), then nudged as above: "line". Its height is that of the TGP's plane there: the TGP splits a polygon along
  its 0-2 diagonal, so a notch at corner 1 or 3 (the diagonal outside the cell) puts the whole cell in the plane of
  the other three corners, and the corner, put on that plane, leaves every height of the cell as it was; a notch at
  corner 0 or 2 (on the diagonal) takes the mean of the two triangles' planes there, which moves a height by at most
  half their difference. The polygon then also covers the notch (the neighbours' ground already), where its plane
  runs on past the cell by the move: the steepest plane's slope times the move. Refused (ConvertError) when the
  move passes MAX_MOVE or the height bound (the two terms, plus the f32 error of the TGP's plane evaluation at
  |x|, |z| up to 1024 for the cell's slope and heights: F32_PLANE) passes HEIGHT_TOLERANCE.
- grid and lists: polygon i is listed in every square from `tgp.cell` of its corners' lowest x and z to `tgp.cell`
  of their highest (its XZ bounding box, in the squares the i960's rounding reads), each list in polygon order; the
  non-empty squares' lists follow each other in square order (row-major) from word 1 of the list block (count,
  then the indices), and a square's grid value is its list's word offset. That is the arcade's rule: it gives ids
  0-2's grid, lists and list block exactly. The Dreamcast's own grid and lists (H2, H7) are built for its negated z
  by a looser rule (526 of n 0's 16,384 squares hold another set than the arcade's), so they are checked (a grid
  value naming no list, a list naming no cell) but not copied.
- attributes: `tables.attributes(d)`, one byte per polygon; `base` = `course`; `raw` the list block's words; each
  polygon's `raw` its 16 words as `tgp.encode_ground` writes them, so a set written with it decodes to it.
- the four addresses (`grid_at`, `attributes_at`, `list_base`, `pool_base`) are 0: Task 3's layout places them
  (`tgp.encode_ground` refuses a ground until then).

ConvertError for a `d` the race tables refuse (`tables._check`), a cell the arcade's ground cannot hold (not a Cell;
corners that are not four finite single-precision triples or that lie outside WORLD; a centre that is not; a
direction word past u16; zero XZ area, or a winding unlike cell 0's; a notch the two repairs cannot close within
MAX_MOVE and HEIGHT_TOLERANCE), a malformed
H2 / H7, a list block too long for the grid's u16 values, or a course that is not an arcade course id."""

from __future__ import annotations

import itertools
import math
import struct

import numpy as np

from dtools.arcade import tgp
from dtools.arcade.course import COURSE_IDS
from dtools.arcade.tablebase import _f32, _int, _seq, is_int
from dtools.convert import ConvertError
from dtools.convert import tables
from dtools.dc.data import Cell, CourseData

WORLD = (-1024, 1023)               # every converted corner's x and z (inclusive): tgp.cell does not wrap inside it
DIRECTION_OFFSET = 0x8000           # arcade direction word = Dreamcast's - DIRECTION_OFFSET (s16)
NUDGE_ULPS = 4                      # a corner this many f32 ulps from its neighbours' line is collinear (see above)
MAX_MOVE = 0.0625                   # units: the farthest a notch corner moves onto its neighbours' line ("line")
HEIGHT_TOLERANCE = 0.005            # units: the most a repair may change a height the TGP answers (stated a priori:
                                    # docs/spike/2026-11-courses-4-7.md, "Ground repair")
PLAIN_SNAP = 0.25                   # units: the farthest a cell's corner may be from the plain course's corner it
                                    # borrows (`plain=`: 4R's cell 1765 is 0.1 off in x and z at two corners, 0.141;
                                    # its corners are over a unit apart)
F32_PLANE = 6 * 2.0 ** -24          # the TGP's plane height: six f32 roundings of terms up to COORD_LIMIT x slope
_U16_MAX = 0xFFFF


def _xyz(v, name: str, what: str) -> tuple[float, float, float]:
    """(x, y, 0.0 - z) of three finite single-precision numbers, each rounded to f32 (as the pool holds them)."""
    x, y, z = struct.unpack("<3f", _f32(v, 3, name, what, ConvertError))
    return x, y, 0.0 - z


def _turns(corners) -> list[float]:
    """turns[k]: the cross product of the edges into and out of corner k + 1 in XZ, as tgp's encoder computes it
    (> tgp.MIN_AREA is a notch for the arcade's negative winding)."""
    xz = [(c[0], c[2]) for c in corners]
    out = []
    for k in range(4):
        (ax, az), (bx, bz), (cx, cz) = xz[k], xz[(k + 1) % 4], xz[(k + 2) % 4]
        out.append((bx - ax) * (cz - bz) - (bz - az) * (cx - bx))
    return out


def _area(corners) -> float:
    xz = [(c[0], c[2]) for c in corners]
    return sum(xz[k][0] * xz[(k + 1) % 4][1] - xz[(k + 1) % 4][0] * xz[k][1] for k in range(4)) / 2


def _inside(c) -> bool:
    lo, hi = WORLD
    return lo <= c[0] <= hi and lo <= c[2] <= hi


def _step(v: float, k: int) -> float:
    """`v` moved k f32 ulps (k < 0: down)."""
    f = np.float32(v)
    toward = np.float32(np.inf if k > 0 else -np.inf)
    for _ in range(abs(k)):
        f = np.nextafter(f, toward)
    return float(f)


def _nudge(corners: tuple, j: int, what: str) -> tuple:
    """`corners` with corner j moved by the fewest f32 ulps (at most NUDGE_ULPS on each axis) that make the polygon
    convex; ConvertError when none does."""
    b = corners[j]
    best = None
    for dx, dz in itertools.product(range(-NUDGE_ULPS, NUDGE_ULPS + 1), repeat=2):
        moved = (_step(b[0], dx), b[1], _step(b[2], dz))
        trial = corners[:j] + (moved,) + corners[j + 1:]
        if _inside(moved) and max(_turns(trial)) <= tgp.MIN_AREA and _area(trial) <= -tgp.MIN_AREA:
            dist = math.hypot(moved[0] - b[0], moved[2] - b[2])
            if best is None or dist < best[0]:
                best = (dist, trial)
    if best is None:
        raise ConvertError(f"{what}: not convex at corner {j}, and no move of up to {NUDGE_ULPS} ulps makes it so")
    return best[1]


def _plane(corners: tuple, tri: tuple[int, int, int]):
    """(h(x, z), slope) of the plane through three of `corners` (float64), or None when it has no XZ extent."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = (corners[i] for i in tri)
    nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
    ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
    nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if not ny:
        return None
    return (lambda x, z: ay - (nx * (x - ax) + nz * (z - az)) / ny), math.hypot(nx, nz) / abs(ny)


def _onto_line(corners: tuple, j: int, ulp: float, what: str) -> tuple:
    """`corners` with notch corner j moved onto the line through its neighbours, at the height of the TGP's plane
    there (module docstring, "line"); the move counted with the ulp nudge that may follow (NUDGE_ULPS of `ulp` on
    each axis), the bound with the f32 error of the TGP's evaluation. ConvertError past MAX_MOVE or
    HEIGHT_TOLERANCE."""
    a, b, c = corners[j - 1], corners[j], corners[(j + 1) % 4]
    ex, ez = c[0] - a[0], c[2] - a[2]
    t = ((b[0] - a[0]) * ex + (b[2] - a[2]) * ez) / (ex * ex + ez * ez)
    x, z = a[0] + t * ex, a[2] + t * ez
    move = math.hypot(x - b[0], z - b[2]) + math.sqrt(2) * NUDGE_ULPS * ulp
    if move > MAX_MOVE:
        raise ConvertError(f"{what}: not convex at corner {j}: it lies {move:.3g} units inside its neighbours' line, "
                           f"past MAX_MOVE ({MAX_MOVE}): the TGP's edge test would miss the notch")
    tris = ((0, 2, 3),) if j == 1 else ((0, 1, 2),) if j == 3 else ((0, 1, 2), (0, 2, 3))
    planes = [_plane(corners, tri) for tri in tris]
    if None in planes:                  # not reached: a notch keeps both triangles' XZ extent (area checked)
        raise ConvertError(f"{what}: not convex at corner {j}, and a plane of the cell has no XZ extent")
    heights = [h(x, z) for h, _ in planes]
    y = sum(heights) / len(heights)
    slope = max(s for _, s in planes)
    f32 = F32_PLANE * (2 * tgp.COORD_LIMIT * slope + max(abs(v[1]) for v in corners))
    bound = (max(heights) - min(heights)) / 2 + slope * move + f32
    if not bound <= HEIGHT_TOLERANCE:
        raise ConvertError(f"{what}: not convex at corner {j} ({move:.3g} units inside its neighbours' line): moving "
                           f"it onto the line could change a height by {bound:.3g} units, past HEIGHT_TOLERANCE "
                           f"({HEIGHT_TOLERANCE})")
    moved = struct.unpack("<3f", struct.pack("<3f", x, y, z))
    return corners[:j] + (moved,) + corners[j + 1:]


def _convex(corners: tuple, what: str) -> tuple[tuple, str | None]:
    """(`corners`, None) for a convex polygon (negative area); else the repaired corners and how ("ulps": a
    collinear notch corner moved outward by the fewest ulps; "line": a wider notch's corner moved onto its
    neighbours' line, then nudged: module docstring); ConvertError for a notch neither closes."""
    turns = _turns(corners)
    k = max(range(4), key=lambda j: turns[j])
    if turns[k] <= tgp.MIN_AREA:
        return corners, None
    j = (k + 1) % 4
    a, b, c = corners[j - 1], corners[j], corners[(j + 1) % 4]
    base = math.hypot(c[0] - a[0], c[2] - a[2])
    ulp = float(np.spacing(np.float32(max(abs(v) for v in (a[0], a[2], b[0], b[2], c[0], c[2])))))
    off = turns[k] / base if base else math.inf
    if off <= NUDGE_ULPS * ulp:
        return _nudge(corners, j, what), "ulps"
    if not base:                        # not reached: a zero-length base has no turn (the area check before)
        raise ConvertError(f"{what}: not convex at corner {j}: its neighbours coincide")
    trial = _onto_line(corners, j, ulp, what)
    if max(_turns(trial)) <= tgp.MIN_AREA and _area(trial) <= -tgp.MIN_AREA and _inside(trial[j]):
        return trial, "line"
    return _nudge(trial, j, what), "line"


def _corners(cell, k: int) -> tuple[tuple, str]:
    """(the four corners of cell k, z negated, as `_xyz`; the cell's name for errors); ConvertError for a cell that
    is not a Cell or whose corners are not four finite triples inside WORLD."""
    what = f"course data: cell {k}"
    if not isinstance(cell, Cell):
        raise ConvertError(f"{what}: {type(cell).__name__} is not a Cell")
    corners = tuple(_xyz(c, f"corner {j}", what)
                    for j, c in enumerate(_seq(cell.corners, 4, "corners", what, ConvertError)))
    for j, c in enumerate(corners):
        if not _inside(c):
            raise ConvertError(f"{what}: corner {j} at x {c[0]}, z {c[2]} (z negated) is outside the world "
                               f"{WORLD[0]}..{WORLD[1]}")
    return corners, what


def _flip(cells) -> bool:
    """True when cell 0 is wound the other way (XZ area at least MIN_AREA once z is negated): the whole file is then
    re-wound (`rewound`). ConvertError as `_corners` for a malformed cell 0 (the error _poly would raise first)."""
    return len(cells) > 0 and _area(_corners(cells[0], 0)[0]) >= tgp.MIN_AREA


def _wound(corners, what: str, flip: bool) -> tuple:
    """`corners` in the arcade's winding: reversed when the file is re-wound (`flip`); ConvertError when the cell's
    own winding is not the file's (or its area is about zero)."""
    a = _area(corners)
    if flip:
        if not a >= tgp.MIN_AREA:
            raise ConvertError(f"{what}: XZ area {a} once z is negated is negative, unlike cell 0's: a file wound the "
                               f"other way (every cell, as the Dreamcast's R files of the arcade's three) is re-wound "
                               f"whole, a mixed one refused")
        return corners[::-1]
    if not a <= -tgp.MIN_AREA:
        raise ConvertError(f"{what}: XZ area {a} once z is negated is not negative: the TGP's one-sided edge test "
                           f"needs the arcade's winding (the Dreamcast's cells have positive area in its z)")
    return corners


def rewound(d: CourseData) -> bool:
    """Whether `d`'s cells are all wound the other way (positive XZ area once z is negated: the Dreamcast's R files
    of the arcade's three, n 0-2), so that convert_ground takes every cell's corners in reverse order; False for the
    arcade's winding. ConvertError for a file with both windings (naming its first cell unlike cell 0), and as
    convert_ground for a malformed cell or race table."""
    tables.attributes(d)                            # validates d's race tables (tables._check), cells a sequence
    flip = _flip(d.cells)
    for k, c in enumerate(d.cells):
        corners, what = _corners(c, k)
        _wound(corners, what, flip)
    return flip


def _from_plain(corners, plain_cell, k: int, plain_flip: bool) -> tuple | None:
    """The plain course's cell k's corners (in the arcade's winding) taken in the order of `corners`, each the one
    within PLAIN_SNAP of the corner it replaces; None unless all four match one each and keep the arcade's winding."""
    pc, what = _corners(plain_cell, k)
    pc = _wound(pc, f"plain {what}", plain_flip)
    out = []
    for c in corners:
        near = [q for q in pc if math.dist(q, c) <= PLAIN_SNAP and q not in out]
        if len(near) != 1:
            return None
        out.append(near[0])
    out = tuple(out)
    return out if _area(out) <= -tgp.MIN_AREA else None


def _poly(cell, k: int, flip: bool = False, plain_cell=None,
          plain_flip: bool = False) -> tuple[tgp.GroundPoly, str | None]:
    """Polygon k of a cell and how it was repaired (`_convex`); its corners reversed first when the file is re-wound
    (`flip`, `rewound`). A cell neither repair makes convex takes `plain_cell`'s corners when they are the same cell
    (`_from_plain`; how "plain"), else the refusal stands."""
    corners, what = _corners(cell, k)
    corners = _wound(corners, what, flip)
    try:
        corners, how = _convex(corners, what)
    except ConvertError as e:
        if isinstance(plain_cell, _Unmatched):
            raise ConvertError(f"{e}; {plain_cell.message}") from e
        swap = None if plain_cell is None else _from_plain(corners, plain_cell, k, plain_flip)
        if swap is None:
            raise
        corners, how = _convex(swap, what)[0], "plain"
    centre = _xyz(cell.centre, "centre", what)
    d = _int(cell.direction, "direction", 0, _U16_MAX, what, ConvertError)
    p = tgp.GroundPoly(corners, (), centre, d - DIRECTION_OFFSET)          # a u16 less 0x8000: an s16
    try:
        words = struct.unpack(f"<{tgp.POLY_WORDS}I", tgp._encode_poly(p, k))
    except tgp.TgpError as e:                   # not reached: the checks above are the encoder's (WORLD within it)
        raise ConvertError(f"{what}: {e}") from e
    return tgp.GroundPoly(corners, words, centre, p.direction), how


def _check_dc_grid(d: CourseData, ncell: int) -> None:
    """ConvertError unless H7 is lists of cell indices and H2 GRID_CELLS indices into them."""
    who = "course data"
    lists = _seq(d.lists, None, "lists", who, ConvertError)
    for j, entries in enumerate(lists):
        for e in _seq(entries, None, f"list {j}", who, ConvertError):
            _int(e, "cell", 0, ncell - 1, f"{who}: list {j} (of {ncell} cells)", ConvertError)
    for k, v in enumerate(_seq(d.grid, tgp.GRID_CELLS, "grid", who, ConvertError)):
        _int(v, "list", 0, len(lists) - 1, f"{who}: grid square {k} (row {k >> 7}, column {k & 127}) "
                                           f"(of {len(lists)} lists)", ConvertError)


def _grid(polys) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]], tuple[int, ...]]:
    """(grid, lists, list block words) by the arcade's rule (module docstring)."""
    squares: dict[int, list[int]] = {}
    for i, p in enumerate(polys):
        xs, zs = [c[0] for c in p.corners], [c[2] for c in p.corners]
        lo, hi = tgp.cell(min(xs), min(zs)), tgp.cell(max(xs), max(zs))
        for row in range(lo >> 7, (hi >> 7) + 1):
            for col in range(lo & 127, (hi & 127) + 1):
                squares.setdefault(row * 128 + col, []).append(i)
    grid = [0] * tgp.GRID_CELLS
    lists: dict[int, tuple[int, ...]] = {}
    block = [0]
    for sq in sorted(squares):
        v = len(block)
        if v > _U16_MAX:
            raise ConvertError(f"course data: the list of grid square {sq} (row {sq >> 7}, column {sq & 127}) starts "
                               f"at word {v:#x} of the list block, past the u16 a grid value holds")
        grid[sq] = v
        lists[v] = tuple(squares[sq])
        block += [len(lists[v]), *lists[v]]
    return tuple(grid), lists, tuple(block)


class _Unmatched:
    """What a cell may borrow when `plain` holds another number of cells (5R has one more than 5): nothing; a cell
    that needs to borrow is refused naming both counts."""
    def __init__(self, plain_count: int, count: int):
        self.message = (f"ground: the plain course data has {plain_count} cells, this one {count}: a cell can borrow "
                        f"only from the same course's plain file, cell for cell")


def _lenders(d: CourseData, plain) -> list:
    """Per cell of `d`, what it may borrow (`_poly`'s plain_cell): None without `plain`, the plain cell of the same
    index, or `_Unmatched` when the cell counts differ; and whether the plain file is re-wound. ConvertError for a
    `plain` that is not CourseData (d's cells already a sequence)."""
    if plain is None:
        return [None] * len(d.cells), False
    if not isinstance(plain, CourseData):
        raise ConvertError(f"ground: plain {type(plain).__name__} is not CourseData")
    cells = tuple(_seq(plain.cells, None, "cells", "ground: plain course data", ConvertError))
    if len(cells) != len(d.cells):
        return [_Unmatched(len(cells), len(d.cells))] * len(d.cells), False
    return list(cells), _flip(cells)


def convert_ground(d: CourseData, course: int = 0, plain: CourseData | None = None) -> tgp.Ground:
    """The arcade's ground of `d` for course id `course` (0-15: a grown set's ids; `base` too): the polygons from the
    cells, the grid, lists and list block rebuilt by the arcade's rule, the attribute bytes copied; addresses 0 (module
    docstring). `plain`, the plain course's data for a reverse file, lends a cell neither repair makes convex its
    same-index cell when every corner is within PLAIN_SNAP of one of it (4R's off-road cell 1765); otherwise unused.
    ConvertError for what the module docstring lists, and for a cell that must borrow from a `plain` with
    another number of cells."""
    if not is_int(course) or not 0 <= course < COURSE_IDS:
        raise ConvertError(f"ground: course {course!r} is not an arcade course id 0..{COURSE_IDS - 1}")
    course = int(course)
    attributes = tables.attributes(d)               # validates d's race tables (tables._check), cells a sequence
    flip = _flip(d.cells)
    lenders, pflip = _lenders(d, plain)
    polys = tuple(_poly(c, k, flip, lenders[k], pflip)[0] for k, c in enumerate(d.cells))
    _check_dc_grid(d, len(polys))
    grid, lists, block = _grid(polys)
    return tgp.Ground(course, grid, attributes, polys, lists, course, block, 0, 0, 0, 0)


def repairs(d: CourseData, plain: CourseData | None = None) -> dict[int, str]:
    """{cell index: how} for every cell of `d` convert_ground repairs to make it convex: "ulps" (a collinear corner
    nudged), "line" (a wider notch's corner moved onto its neighbours' line) or "plain" (the plain course's cell
    borrowed, `plain` as convert_ground's); module docstring. ConvertError as convert_ground for a `d` it refuses."""
    tables.attributes(d)                            # validates d's race tables (tables._check), cells a sequence
    flip = _flip(d.cells)
    lenders, pflip = _lenders(d, plain)
    out = {}
    for k, c in enumerate(d.cells):
        how = _poly(c, k, flip, lenders[k], pflip)[1]
        if how is not None:
            out[k] = how
    return out


# What `repairs` gives on each Dreamcast course n (= the arcade id it converts to) that has a repair: n 0-2 none;
# n 3 two ulp nudges (Phase 6); n 4-7 every cell the TGP's convexity check refuses (11 / 19 / 10 / 13), one moved
# onto the line each (docs/spike/2026-11-courses-4-7.md, "Ground repair"). tests/test_convert_ground.py pins it.
REPAIRED: dict[int, dict[int, str]] = {
    3: {1689: "ulps", 1915: "ulps"},
    4: {**dict.fromkeys((792, 1215, 1428, 1474, 1479, 2038, 2085, 2151, 2167, 2168), "ulps"), 2082: "line"},
    5: {**dict.fromkeys((130, 214, 215, 216, 218, 222, 225, 418, 559, 561, 563, 566, 568, 725, 773, 775, 818, 857),
                        "ulps"), 410: "line"},
    6: {**dict.fromkeys((594, 619, 621, 643, 864, 1062, 1134, 1427, 1487), "ulps"), 2256: "line"},
    7: {**dict.fromkeys((603, 822, 1123, 1313, 1448, 1782, 1883, 2263, 2311, 2356, 2671, 2696), "ulps"),
        2334: "line"},
}

# What `repairs(R, plain=P)` gives on each Dreamcast R file (course n's reverse, arcade id n + 8) that has a repair:
# n 0-2 none (re-wound, `rewound`); n 3-7 the plain file's repairs at the cells the R file keeps in place, 4R's cell
# 1765 borrowed from the plain file ("plain"), 5R (reordered, one cell more) its own indices. Measured 17 Nov 2026,
# pinned by tests/test_convert_ground.py.
REPAIRED_R: dict[int, dict[int, str]] = {
    3: {1689: "ulps", 1915: "ulps"},
    4: {**dict.fromkeys((1215, 1218, 1428, 1474, 1479, 2038, 2085, 2151, 2167, 2168), "ulps"), 1765: "plain",
        2082: "line"},
    5: dict.fromkeys((130, 214, 215, 216, 220, 221, 222, 238, 419, 560, 562, 564, 567, 569, 726, 775, 778, 821, 860),
                     "ulps"),
    6: {**dict.fromkeys((594, 619, 621, 643, 864, 1062, 1134, 1427, 1487), "ulps"), 2256: "line"},
    7: {**dict.fromkeys((603, 822, 1123, 1313, 1448, 1782, 1883, 2263, 2311, 2356, 2671, 2696), "ulps"),
        2334: "line"},
}
