"""The course maps below Circuit Select's panels for the courses without an original panel (Phase 12 Task 6, M7:
Pascal's request, "the track representation below their respective names"; painted in Task 7, "can we get some
backgrounds on the tracks?"): select.patch_select gives an own_panel entry's MAPS records an object list naming
`course_map`'s object. Discovery log: docs/spike/2026-11-release.md, "Maps and laps" and "Painted maps"; layout:
docs/formats/model2-course-tables.md, "The release row: own panels and maps".

The originals' maps are pictures: each a textured quad (x and z -2..2, y 0, normal (0, -1, 0)) drawn one to four
times with 128 x 128 four-bit textures in texture sheet 1, the common sheet the boot uploads, which has no free 128 x
128 room (only eight 64 x 64 tiles of one value, likely the transparent parts of textures in use); sheet 0 holds the
last course loaded, any of the eight, and the five maps' room is free in none of them for certain (Task 7). So a
converted course's map is drawn, not textured: flat (untextured) polygons on the same quad, built at build time from
the set itself (the player's set with the disc's courses converted into it), in the originals' style:

- the ground: the quad itself, in the colour of the course's uncovered surroundings (`paint`: OUTSIDE);
- the painting: the course's own scenery seen from straight above (`scenery_picture`), in a reduced palette of the
  originals' map colours (FAMILIES: grass, olive, sand, brown, water, white, greys, each in up to three shades),
  cleaned to a grid of up to one cell per texel (`paint`) and cut into strips of flat quads, one y-monotone piece of a
  colour each (`paint_pieces`), as many as PAINT_MAX_DRAWN allows (`map_paint`, PAINT_LEVELS);
- a dark frame two texels wide round it (the originals' textures' border);
- the road: its centre line (course.RoadPoint.centre) as a strip ROAD_TEXELS wide in the road grey over a strip
  EDGE_TEXELS wider in white (its edges), the loop merged where it is straightest (build.outline_points, the HUD
  outline's rule) to MAP_POINTS points (measured on the originals' textures, the road texels against the road's
  points); a white stripe across it at the lap line (road point 0), as Advanced's map draws its own;
- fitted as the originals are: its longer axis (x or z) vertical on screen, spanning FIT_TEXELS of 128 and centred;
  a road long in x as Beginner's map turns it, one long in z as Advanced's (`fit`).

Every polygon after the first takes the first one's depth (z-sort "old value"), so the map is one bucket of the
rasterizer's z-sort, where a later polygon draws over an earlier one (model2_v.cpp; seen in MAME): ground, painting,
frame, edges, road, stripe, in that order. Each part's colour is a flat header (build.FLAT_HEADER) in the palette
entry nearest the part's colour (textures.flat_entry: never one the game rewrites); the headers follow one another,
the last polygon of each part stepping to the next (`parts`).

Errors are SelectError (a DecodeError), from select."""

from __future__ import annotations

import math
import struct
from typing import NamedTuple

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import polyrom as P
from dtools.arcade import texture as T
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet
from dtools.arcade.select import SelectError
from dtools.arcade.tablebase import is_int, is_number

QUAD = 2.0
"""The map quad's half side: x and z -2..2 (the originals' map objects)."""
TEXELS = 128
"""The originals' map textures' side: a texel is 2 * QUAD / TEXELS of the quad."""
FIT_TEXELS = 104
"""The road centre line's longer extent in texels (Beginner's map: 105.6, Advanced's 104.2; Expert's runs off its
texture): 12 texels of margin either side, the road's edges inside them."""
ROAD_TEXELS = 6
"""The road's width in texels (the dark grey strip of Beginner's and Advanced's maps)."""
EDGE_TEXELS = 2
"""The white edge on each side of the road, in texels."""
FRAME_TEXELS = 2
"""The dark border round the originals' map textures, in texels."""
MAP_POINTS = 48
"""The most road points a map's strips keep (build.outline_points: the loop merged where it is straightest): each
strip is one quad per kept point. With them a map is 8 + 2 * MAP_POINTS polygons; the select's frame counters keep
step with the M5b set's with three such maps on screen (tests/test_release_mame.py)."""
ROAD_RGB = (68, 68, 68)
"""The road and frame grey: Beginner's map's road texels (index 1 of its base layer, colour 44), decoded."""
EDGE_RGB = (255, 255, 255)
"""The road's edges: the originals' white."""
START_TEXELS = 2
"""The start stripe's length along the road, in texels: Advanced's map draws its lap line as a white stripe two texels
wide across the road (Beginner's draws none; Expert's road runs off its texture); the painted maps draw it at the lap
line (road point 0)."""
RENDER = 256
"""The scenery picture's pixels across the quad (`scenery_picture`): two per texel of the originals' 128."""
PAINT_CELLS = 128
"""The painting's finest grid (`paint`): one cell per texel of the originals' maps, their level of detail."""
PAINT_LEVELS = (128, 64, 32)
"""The grids `map_paint` tries, finest first, until a map fits PAINT_MAX_DRAWN."""
PAINT_MAX_DRAWN = 600
"""The most drawn polygons a map may have (`drawn_polygons`), the measured budget (docs/spike/2026-11-release.md,
"Painted maps"; MAME's geometrizer takes no emulated time, so the frame counters alone cannot bound it): three maps on
screen and Sega's 12 polygons stay under the 2,804 an attract frame of the same run draws (a race draws up to 3,488,
the game stops at 5,000). The release set's maps are 251-443; the select's frame counters keep step with the M5b
playtest set's (tests/test_release_mame.py)."""
PAINT_TOLERANCE = 0.75
"""How far (cells) a strip's slanted side may leave a dropped row's run end (`_edges`)."""
PAINT_PEAK = 235
"""The scenery picture's 90th-percentile brightest channel once scaled (`paint`): its colours are textures' means,
darker than the originals' map colours."""
SHADE_MIN = 50
"""The fewest picture pixels a family needs before its pixels are split into dark, mid and light shades."""
FAMILIES = {
    "grass": ((0, 98, 54), (0, 152, 61), (0, 192, 65)),
    "olive": ((82, 108, 69), (101, 138, 81), (128, 186, 98)),
    "sand": ((222, 157, 112), (255, 214, 145), (255, 242, 162)),
    "brown": ((88, 73, 62), (132, 101, 78), (177, 129, 96)),
    "water": ((0, 158, 170), (0, 205, 222), (0, 255, 255)),
    "white": ((248, 248, 248),) * 3,
    "grey": ((170, 170, 170),) * 3,
    "grey_dark": ((118, 118, 118),) * 3,
}
"""The painting's palette, (dark, mid, light) per family: colours of the originals' map textures, decoded through the
colour tables (Beginner's and Expert's greens, Advanced's olive, sand and brown, Expert's water, their whites and
greys; a family with one shade repeats it)."""
NATURAL = ("grass", "olive", "sand", "brown")
"""The families a course's base (its uncovered surroundings) is chosen from; the first when none is present."""
OUTSIDE = {"grass": (0, 192, 65), "olive": (128, 186, 98), "sand": (255, 242, 162), "brown": (255, 228, 154)}
"""The uncovered surroundings' colour per base family: its light shade; brown's the sand Advanced's map lays its
brown relief on."""
QUAD_ATTR = 0x79081601
"""The originals' map quads' attribute (link 2, z-sort min, header step 1, single-sided), from which every polygon's
is made (`_attr`)."""
MOVE_ATTR = 0x00040001
"""A move record: the originals' (0x40401: link 0, never drawn) with z-sort "old value", so it keeps the map's depth."""
NORMAL = (0.0, -1.0, 0.0)
END = (0, 0x40401, 0, 0)
"""The end record's words as the originals' map objects store them."""
Z_MIN, Z_OLD = 1, 0             # attribute bits 10-11: the polygon's depth is its nearest corner's / the previous one's


def _attr(z: int, step: int) -> int:
    """QUAD_ATTR with z-sort mode `z` (Z_MIN, Z_OLD) and header step `step` (0 or 1)."""
    return QUAD_ATTR & ~(3 << 10) & ~(0x1F << 12) | z << 10 | step << 12


def _f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _point(x: float, z: float) -> tuple[float, float, float]:
    return (_f32(x), 0.0, _f32(z))


def _frame(road):
    """(the road's kept centres (x, z), the map frame's transform of a world (x, z) to the quad's): `fit`'s rule."""
    from dtools.convert import build as B
    try:
        cs = B.road_centres(road, "select map")
        kept, _ = B.outline_points(cs, MAP_POINTS)
    except DecodeError as e:
        raise SelectError(f"select map: {e}") from e
    pts = np.array([cs[k] for k in kept], float)
    lo, hi = pts.min(0), pts.max(0)
    ext, mid = hi - lo, (hi + lo) / 2
    if not ext.max() > 0:
        raise SelectError("select map: the road's centres are one point")
    s = FIT_TEXELS * 2 * QUAD / TEXELS / ext.max()
    if ext[0] >= ext[1]:
        return pts, lambda x, z: (-(z - mid[1]) * s, -(x - mid[0]) * s)
    return pts, lambda x, z: ((x - mid[0]) * s, -(z - mid[1]) * s)


def fit(road) -> list[tuple[float, float]]:
    """The road's kept centres (`build.outline_points`, at most MAP_POINTS) in the map quad's frame (x, z): its
    longer axis (of the kept centres' extent; x on a tie) scaled to FIT_TEXELS and centred, turned as the originals'
    maps turn their courses (measured: Beginner's road, long in x, has quad x = -z and quad z = -x; Advanced's, long in
    z, quad x = x and quad z = -z; both show the longer axis upright). SelectError for a road the HUD outline's
    `build.road_centres` refuses or whose kept centres are one point."""
    pts, to_quad = _frame(road)
    return [to_quad(x, z) for x, z in pts]


def _strip(cs, width: float) -> list[tuple[tuple, tuple]]:
    """Per point of the closed loop `cs` its edge square to the line there (the direction from the point before to
    the one after), the left end first (as the quad's: a strip drawn on the quad's side), `width` across."""
    out, n = [], len(cs)
    for k in range(n):
        x, z = cs[k]
        tx, tz = cs[(k + 1) % n][0] - cs[k - 1][0], cs[(k + 1) % n][1] - cs[k - 1][1]
        length = math.hypot(tx, tz)
        if not length:
            raise SelectError(f"select map: road point {k}'s neighbours are at one place")
        nx, nz = -tz / length * width / 2, tx / length * width / 2
        out.append((_point(x + nx, z + nz), _point(x - nx, z - nz)))
    return out


def _quad_point(y: float, u: float, cells: int) -> tuple[float, float, float]:
    """The quad's point at painting cell row `y`, column `u` of a `cells`-wide grid (row v down the map texture, column
    u along it: `_texel_of`'s axes)."""
    return _point(QUAD - u * 2 * QUAD / cells, y * 2 * QUAD / cells - QUAD)


def _texel_of(qx: float, qz: float, size: int) -> tuple[float, float]:
    """(u, v) of quad point (qx, qz) on a `size`-texel square laid out as the originals' map textures: their texture
    points put (2, -2) at (0, 0) and (-2, 2) at (size, size)."""
    return (QUAD - qx) * size / (2 * QUAD), (qz + QUAD) * size / (2 * QUAD)


def _check_points(points) -> None:
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise SelectError(f"select map: {points!r:.60} is not a loop of at least three points")
    for k, p in enumerate(points):
        if (not isinstance(p, (list, tuple)) or len(p) != 2 or not all(is_number(v) and abs(v) < QUAD for v in p)):
            raise SelectError(f"select map: point {k} {p!r:.60} is not two finite numbers inside the quad "
                              f"(|v| < {QUAD})")


def _check_pieces(pieces) -> None:
    """`pieces`: per painted colour (drawing order) a non-empty sequence of strips, each at least two edges, an edge
    two quad points (x, z) inside the quad (|v| <= QUAD)."""
    if not isinstance(pieces, (list, tuple)):
        raise SelectError(f"select map: pieces {pieces!r:.60} are not a sequence of colours' strips")
    for c, strips in enumerate(pieces):
        if not isinstance(strips, (list, tuple)) or not strips:
            raise SelectError(f"select map: colour {c}'s strips {strips!r:.60} are not a non-empty sequence")
        for k, st in enumerate(strips):
            if not isinstance(st, (list, tuple)) or len(st) < 2:
                raise SelectError(f"select map: colour {c} strip {k} {st!r:.60} is not at least two edges")
            for e in st:
                if (not isinstance(e, (list, tuple)) or len(e) != 2 or not all(
                        isinstance(q, (list, tuple)) and len(q) == 2 and all(is_number(v) and abs(v) <= QUAD
                                                                             for v in q) for q in e)):
                    raise SelectError(f"select map: colour {c} strip {k} edge {e!r:.60} is not two points (x, z) "
                                      f"inside the quad")


def map_object(points, pieces=()) -> P.Object:
    """The map's flat polygons for road centres `points` already in the quad's frame (`fit`) and the painting's
    strips `pieces` (`paint_pieces`, in the quad's frame): seeds the quad's first edge; the ground quad (Z_MIN, the
    map's depth; header step 1); per painted colour its strips, each a move to its first edge then one quad per further
    edge (the colour's last quad stepping to the next header); a move and the frame (a strip of four quads round the
    quad's border, FRAME_TEXELS wide), a move and the edge strip (ROAD_TEXELS + 2 * EDGE_TEXELS wide), a move and the
    road strip (ROAD_TEXELS wide); a move and the start stripe (`_stripe`: across the road at road point 0, the lap
    line); every polygon after the ground Z_OLD; the end record END. So one header per part (`parts`). SelectError for
    fewer than three points, a point that is not two finite numbers inside the quad, two neighbours of a point at one
    place, or pieces that are not strips of points inside the quad."""
    _check_points(points)
    _check_pieces(pieces)
    texel = 2 * QUAD / TEXELS
    inner = QUAD - FRAME_TEXELS * texel
    corners = [(-1, 1), (1, 1), (1, -1), (-1, -1)]       # the frame's loop, its outer side on the strip's left
    frame = [(_point(QUAD * a, QUAD * b), _point(inner * a, inner * b)) for a, b in corners]
    recs = [P.Record(_attr(Z_MIN, 1), NORMAL, _point(QUAD, QUAD), _point(QUAD, -QUAD))]
    for strips in pieces:
        for k, st in enumerate(strips):
            edges = [(_point(*a), _point(*b)) for a, b in st]
            recs.append(P.Record(MOVE_ATTR, NORMAL, *edges[0]))
            for i, (p0, p1) in enumerate(edges[1:], 1):
                last = k == len(strips) - 1 and i == len(edges) - 1
                recs.append(P.Record(_attr(Z_OLD, int(last)), NORMAL, p0, p1))
    for part, edges in (("frame", frame), ("edge", _strip(points, (ROAD_TEXELS + 2 * EDGE_TEXELS) * texel)),
                        ("road", _strip(points, ROAD_TEXELS * texel))):
        recs.append(P.Record(MOVE_ATTR, NORMAL, *edges[0]))
        loop = edges[1:] + edges[:1]
        for i, (p0, p1) in enumerate(loop):
            recs.append(P.Record(_attr(Z_OLD, int(i == len(loop) - 1)), NORMAL, p0, p1))
    first, second = _stripe(points)
    recs.append(P.Record(MOVE_ATTR, NORMAL, *first))
    recs.append(P.Record(_attr(Z_OLD, 0), NORMAL, *second))
    return P.Object(_point(-QUAD, QUAD), _point(-QUAD, -QUAD), recs, 6 + 10 * len(recs) + len(END), END)


def _stripe(points) -> tuple[tuple, tuple]:
    """The start stripe's two edges: across the road strip (ROAD_TEXELS) at road point 0 (the lap line, segment 0),
    START_TEXELS long along the line there (from the point before to the one after), each edge as `_strip`'s (the
    left end first)."""
    texel = 2 * QUAD / TEXELS
    (x, z), (ax, az), (bx, bz) = points[0], points[-1], points[1]
    length = math.hypot(bx - ax, bz - az)
    if not length:
        raise SelectError("select map: road point 0's neighbours are at one place")
    tx, tz = (bx - ax) / length, (bz - az) / length              # along the road
    along, across = START_TEXELS * texel / 2, ROAD_TEXELS * texel / 2
    nx, nz = -tz * across, tx * across                            # to its left, as _strip's
    return tuple((_point(x + d * tx * along + nx, z + d * tz * along + nz),
                  _point(x + d * tx * along - nx, z + d * tz * along - nz)) for d in (-1, 1))


# --- the painting: the course's own scenery seen from above -----------------------------------------------------

def _lab(rgb) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB colours 0-255 (last axis r, g, b)."""
    c = np.asarray(rgb, float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _colour_of(rs, h, course, tables, flat, cache) -> np.ndarray | None:
    key = h.words if h.textured else ("flat", h.colour)
    if key not in cache:
        if h.textured:
            px = T.decode_texels(rs, h, course, tables)
            ink = px[..., 3] > 0
            cache[key] = px[ink][:, :3].mean(0) if ink.any() else None
        else:
            cache[key] = np.asarray(flat[h.colour], float) if h.colour < len(flat) else None
    return cache[key]


def scenery_picture(rs: RomSet, course: int, size: int = RENDER) -> tuple[np.ndarray, np.ndarray]:
    """(colours (size, size, 3) floats 0-255, covered (size, size) bools): course `course`'s own scenery seen from
    straight above in the map frame (`fit`'s transform), laid out as the originals' map textures (`_texel_of`): every
    drawn polygon of the cell grid's placements (course.scenery; world coordinates; actor-placed objects are left out:
    the converter empties the actor table) filled with its colour (a textured polygon its texture's mean opaque texel
    colour, texture.decode_texels; a flat one its palette entry's flat colour, textures.flat_colours), the highest
    surface (least y: the geometry's y runs down) at each pixel centre winning. SelectError for a course whose road or
    scenery does not decode."""
    from dtools.convert import textures as TX
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    img, top = np.zeros((size, size, 3)), np.full((size, size), np.inf)
    try:
        table = C.course_table(rs)
        if not is_int(course) or not 0 <= course < len(table):
            raise SelectError(f"course id {course!r:.20} is not one of the set's")
        _, to_quad = _frame(C.road_line(rs, table[course]))
        tables = T.colour_tables(rs)
        flat = TX.flat_colours(tables)
        objects, cache = {}, {}
        for p in C.scenery(rs, table[course], allow_unmodelled=True):
            if not p.in_world:
                continue
            i = C.check_rom_object(p.oba, rs)
            obj = objects.get(i) or P.decode_object(rs.polygons, i, p.obc, 1)
            objects[i] = obj
            for poly, h in zip(P.polygons(obj), T.object_headers(rs, p.tha, obj)):
                if not poly.drawn:
                    continue
                colour = _colour_of(rs, h, course, tables, flat, cache)
                if colour is None:
                    continue
                q = [(*_texel_of(*to_quad(x, z), size), y) for x, y, z in poly.points]
                for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])) if len(q) == 4 else ((q[0], q[1], q[2]),):
                    _fill(tri, colour, img, top, xx, yy)
    except DecodeError as e:
        if isinstance(e, SelectError):
            raise
        raise SelectError(f"course {course}'s scenery picture: {e}") from e
    return img, np.isfinite(top)


def _fill(tri, colour, img, top, xx, yy) -> None:
    """Fill triangle `tri` ((u, v, y) per corner, pixels) with `colour` where its y at a pixel centre is below `top`."""
    (u0, v0, y0), (u1, v1, y1), (u2, v2, y2) = tri
    n = img.shape[0]
    lo_u, hi_u = max(0, int(math.floor(min(u0, u1, u2)))), min(n, int(math.ceil(max(u0, u1, u2))))
    lo_v, hi_v = max(0, int(math.floor(min(v0, v1, v2)))), min(n, int(math.ceil(max(v0, v1, v2))))
    d = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    if lo_u >= hi_u or lo_v >= hi_v or abs(d) < 1e-12:
        return
    X, Y = xx[lo_v:hi_v, lo_u:hi_u], yy[lo_v:hi_v, lo_u:hi_u]
    a = ((v1 - v2) * (X - u2) + (u2 - u1) * (Y - v2)) / d
    b = ((v2 - v0) * (X - u2) + (u0 - u2) * (Y - v2)) / d
    g = 1 - a - b
    y = a * y0 + b * y1 + g * y2
    t = top[lo_v:hi_v, lo_u:hi_u]
    win = (a >= 0) & (b >= 0) & (g >= 0) & (y < t)
    t[win] = y[win]
    img[lo_v:hi_v, lo_u:hi_u][win] = colour


def _road_mask(points, size: int, half: float) -> np.ndarray:
    """Pixels (of a `size`-texel square) whose centre is within `half` pixels of the road's kept line `points` (quad
    frame, a closed loop)."""
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    pts = [_texel_of(x, z, size) for x, z in points]
    d = np.full((size, size), np.inf)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy or 1), 0, 1)
        d = np.minimum(d, np.hypot(xx - x0 - t * dx, yy - y0 - t * dy))
    return d <= half


def _nearest_fill(labels: np.ndarray) -> np.ndarray:
    """`labels` with every negative cell given its nearest labelled neighbour's (4-connected growth, right, left,
    down, up in turn each step)."""
    out = labels.copy()
    if (out < 0).all():
        return out
    while (out < 0).any():
        for axis, step in ((1, 1), (1, -1), (0, 1), (0, -1)):
            sh = np.roll(out, step, axis)
            edge = (slice(None), 0 if step == 1 else -1) if axis else (0 if step == 1 else -1, slice(None))
            sh[edge] = -1
            m = (out < 0) & (sh >= 0)
            out[m] = sh[m]
    return out


def _majority(labels: np.ndarray, cells: int, count: int) -> np.ndarray:
    """`labels` (square, a multiple of `cells` across) reduced to `cells` x `cells` by the most frequent label of each
    block (ties to the lowest), labels 0..count-1."""
    f = labels.shape[0] // cells
    b = labels.reshape(cells, f, cells, f).transpose(0, 2, 1, 3).reshape(cells, cells, f * f)
    return np.stack([(b == c).sum(-1) for c in range(count)], -1).argmax(-1)


def _mode3(labels: np.ndarray, count: int) -> np.ndarray:
    """One pass of a 3 x 3 mode filter: a cell keeps its label when at least 3 of its 9 (edges repeated) share it,
    else takes the most frequent (ties to the lowest)."""
    n = labels.shape[0]
    p = np.pad(labels, 1, mode="edge")
    cnt = np.zeros((n, n, count), int)
    for dy in range(3):
        for dx in range(3):
            s = p[dy:dy + n, dx:dx + n]
            cnt += s[..., None] == np.arange(count)
    own = np.take_along_axis(cnt, labels[..., None], -1)[..., 0]
    return np.where(own >= 3, labels, cnt.argmax(-1))


class Painting(NamedTuple):
    """A course's painting: `cells` x `cells` labels (row v, column u, as `_texel_of`), each an index of `colours`
    (RGB); `order` the labels in drawing order, the first the ground (the whole quad); `road` the cells under the road
    strips (drawn over everything, so any label may fill them)."""
    cells: int
    labels: np.ndarray
    colours: tuple[tuple[int, int, int], ...]
    order: tuple[int, ...]
    road: np.ndarray


def paint(rs: RomSet, course: int, cells: int = PAINT_CELLS) -> Painting:
    """Course `course`'s painting on a `cells`-wide grid: its `scenery_picture` (RENDER pixels) in a reduced palette
    read like Sega's maps. Each covered pixel takes the FAMILIES family of the nearest of their shades (CIE Lab, the
    picture first scaled so its 90th-percentile brightest channel is PAINT_PEAK); within a family of SHADE_MIN pixels
    or more, a pixel more than half a standard deviation brighter than the family's median luminance takes its light
    shade, darker its dark shade, else the mid one. The base family is the NATURAL family covering most pixels (grass
    when none does); everything the scenery does not cover takes OUTSIDE[base], and the base family's light shade its
    mid one (so the uncovered surroundings read apart from the course's own ground). The road strips' cells (within
    ROAD_TEXELS / 2 + EDGE_TEXELS texels of the kept road line) take their neighbours' labels; then the grid is reduced
    to `cells` by block majority and cleaned by two 3 x 3 mode passes. `order`: OUTSIDE[base] first, then the other
    labels by their area outside the road (largest first, ties to the lower label). SelectError as scenery_picture, or
    for `cells` not a divisor of RENDER."""
    if not is_int(cells) or cells < 1 or RENDER % cells:
        raise SelectError(f"select map: cells {cells!r:.20} is not a divisor of {RENDER}")
    img, covered = scenery_picture(rs, course)
    names = list(FAMILIES)
    shades = np.array([c for n in names for c in FAMILIES[n]], float)
    count = 3 * len(names) + 1
    outside = count - 1
    points = fit(_road(rs, course))
    road = _road_mask(points, RENDER, (ROAD_TEXELS / 2 + EDGE_TEXELS) * RENDER / TEXELS)
    family = np.full(covered.shape, -1)
    shade = np.ones(covered.shape, int)
    seen = covered & ~road
    if covered.any():
        scale = PAINT_PEAK / max(float(np.percentile(img[covered].max(-1), 90)), 1.0)
        lab = _lab(np.clip(img * scale, 0, 255))
        ref = _lab(shades)
        nearest = np.argmin(((lab[..., None, :] - ref) ** 2).sum(-1), -1)
        family[covered] = nearest[covered] // 3
        lum = img @ np.array([0.299, 0.587, 0.114])
        for f in range(len(names)):
            m = seen & (family == f)
            if m.sum() < SHADE_MIN:
                continue
            med, sd = float(np.median(lum[m])), float(lum[m].std())
            shade[m & (lum > med + sd / 2)] = 2
            shade[m & (lum < med - sd / 2)] = 0
    area = np.bincount(family[seen].ravel(), minlength=len(names)) if seen.any() else np.zeros(len(names), int)
    base = max((names.index(n) for n in NATURAL), key=lambda f: (area[f], -f))
    if not area[base]:
        base = names.index(NATURAL[0])
    shade[(family == base) & (shade == 2)] = 1
    labels = np.where(family >= 0, family * 3 + shade, outside)
    labels[road] = -1
    labels = _nearest_fill(labels)
    labels[labels < 0] = outside
    grid = _majority(labels, cells, count)
    grid = _mode3(_mode3(grid, count), count)
    road_cells = _road_mask(points, cells, (ROAD_TEXELS / 2 + EDGE_TEXELS) * cells / TEXELS - 0.5)
    sizes = np.bincount(grid[~road_cells].ravel(), minlength=count)
    order = (outside,) + tuple(int(c) for c in np.argsort(-sizes, kind="stable") if sizes[c] and c != outside)
    colours = tuple(tuple(int(v) for v in c) for c in shades) + (OUTSIDE[names[base]],)
    return Painting(cells, grid, colours, order, road_cells)


def _spans(row: np.ndarray) -> list[tuple[int, int]]:
    """The runs of True in `row`: (first, end) pairs."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _chains(mask: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """`mask` cut into y-monotone pieces: each a list of (row, first, end) runs on consecutive rows, a run continuing
    the piece of the run above when each is the other's only overlapping run."""
    rows = [_spans(r) for r in mask]
    out, open_ = [], {}
    for r, sp in enumerate(rows):
        prev = rows[r - 1] if r else []
        new = {}
        for j, (a, b) in enumerate(sp):
            over = [i for i, (c, d) in enumerate(prev) if c < b and a < d]
            if len(over) == 1 and over[0] in open_:
                c, d = prev[over[0]]
                if sum(1 for e, f in sp if e < d and c < f) == 1:
                    piece = open_.pop(over[0])
                    piece.append((r, a, b))
                    new[j] = piece
                    continue
            piece = [(r, a, b)]
            out.append(piece)
            new[j] = piece
        open_ = new
    return out


def _edges(piece, tol: float) -> list[tuple[float, float, float]]:
    """A monotone piece's strip edges (row y, left, right): its first run's top, each run's middle row, its last run's
    bottom, then the fewest of them such that every dropped edge's ends lie within `tol` cells of the line between the
    kept ones round it (kept greedily from the top)."""
    e = ([(float(piece[0][0]), piece[0][1], piece[0][2])] + [(r + 0.5, a, b) for r, a, b in piece]
         + [(float(piece[-1][0] + 1), piece[-1][1], piece[-1][2])])
    keep, i = [0], 0
    while i < len(e) - 1:
        j = i + 1
        while j + 1 < len(e):
            y0, a0, b0 = e[i]
            y1, a1, b1 = e[j + 1]
            if any(abs(a0 + (a1 - a0) * (y - y0) / (y1 - y0) - a) > tol
                   or abs(b0 + (b1 - b0) * (y - y0) / (y1 - y0) - b) > tol for y, a, b in e[i + 1:j + 1]):
                break
            j += 1
        keep.append(j)
        i = j
    return [e[k] for k in keep]


def paint_pieces(painting: Painting, tol: float = PAINT_TOLERANCE) -> tuple[tuple[int, tuple], ...]:
    """The painting as strips: per label of `order` after the first (the ground quad's), its mask (the cells of that
    label or of any label drawn after it, and the road's: every later colour and the road draw over them) cut into
    y-monotone pieces (`_chains`) that hold at least one cell of the label, each piece's edges (`_edges`, `tol` cells)
    turned to quad points: (label, (strip, ...)) per label with pieces, a strip a tuple of edges ((x, z), (x, z)), its
    right end first (so every quad winds as the ground quad)."""
    n = painting.cells
    rank = np.full(len(painting.colours), len(painting.order))
    rank[list(painting.order)] = np.arange(len(painting.order))
    ranks = rank[painting.labels]
    out = []
    for i, label in enumerate(painting.order[1:], 1):
        own = painting.labels == label
        mask = (ranks >= i) | painting.road
        strips = []
        for piece in _chains(mask):
            if not any(own[r, a:b].any() for r, a, b in piece):
                continue
            strips.append(tuple((_quad_point(y, b, n)[::2], _quad_point(y, a, n)[::2]) for y, a, b in
                                _edges(piece, tol)))
        if strips:
            out.append((int(label), tuple(strips)))
    return tuple(out)


def map_paint(rs: RomSet, course: int) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, tuple], ...]]:
    """(the painting's ground colour then each painted label's colour as RGB, the strips per label: `paint_pieces`)
    for course `course`: at the finest of PAINT_LEVELS whose map stays within PAINT_MAX_DRAWN drawn polygons
    (`drawn_polygons`). SelectError as `paint`, or when even the coarsest level does not fit."""
    road = _road(rs, course)
    points = len(fit(road))
    for cells in PAINT_LEVELS:
        painting = paint(rs, course, cells)
        pieces = paint_pieces(painting)
        if drawn_polygons(points, pieces) <= PAINT_MAX_DRAWN:
            colours = (painting.colours[painting.order[0]],) + tuple(painting.colours[c] for c, _ in pieces)
            return colours, pieces
    raise SelectError(f"course {course}'s map: more than {PAINT_MAX_DRAWN} drawn polygons at {PAINT_LEVELS[-1]} "
                      f"cells")


def _road(rs: RomSet, course: int) -> list:
    """Course `course`'s road line. SelectError for a course id that is not one of the set's or a road that does not
    decode."""
    try:
        table = C.course_table(rs)
        if not is_int(course) or not 0 <= course < len(table):
            raise SelectError(f"course id {course!r:.20} is not one of the set's")
        return C.road_line(rs, table[course])
    except DecodeError as e:
        if isinstance(e, SelectError):
            raise
        raise SelectError(f"course {course}'s road: {e}") from e


def drawn_polygons(points: int, pieces) -> int:
    """The drawn polygons of a map of `points` road points and painted `pieces`: the ground, a quad per further edge
    of every strip, the frame's four, the edge and road strips' one per point each, the start stripe."""
    return 1 + sum(len(st) - 1 for _, strips in pieces for st in strips) + 4 + 2 * points + 1


def map_colours(rs: RomSet, course: int) -> tuple[int, ...]:
    """The palette entries of `parts` (the painting's ground, each painted colour, the frame, the edge, the road, the
    start stripe): each the entry nearest its colour (`_entries`). SelectError as map_paint."""
    return _entries(rs, map_paint(rs, course)[0])


def _entries(rs: RomSet, colours) -> tuple[int, ...]:
    """The palette entries nearest `colours` (textures.flat_entry: never one the game rewrites), then the frame's and
    road's (ROAD_RGB) and the edge's and stripe's (EDGE_RGB) in parts' order."""
    from dtools.convert import textures as TX
    try:
        tables = T.colour_tables(rs)
        road, edge = TX.flat_entry(ROAD_RGB, tables), TX.flat_entry(EDGE_RGB, tables)
        return tuple(TX.flat_entry(c, tables) for c in colours) + (road, edge, road, edge)
    except DecodeError as e:
        raise SelectError(f"the map colours: {e}") from e


def parts(pieces) -> tuple[str, ...]:
    """The map's parts in drawing order, one header each: the ground, a "paint <label>" per painted label of
    `pieces`, the frame, the edge, the road, the start stripe."""
    return ("ground",) + tuple(f"paint {c}" for c, _ in pieces) + ("frame", "edge", "road", "start")


def map_headers(colours) -> bytes:
    """The flat headers (build.FLAT_HEADER) of `colours` (the parts' palette entries), back to back, 8 bytes each."""
    from dtools.convert import build as B
    return b"".join(T.encode_tex_header(T.TexHeader(colour=c, **B.FLAT_HEADER)) for c in colours)


def course_map(rs: RomSet, course: int) -> tuple[P.Object, bytes]:
    """(the map object, its headers) of course `course` of `rs`: `map_object(fit(its road line), its painting's
    strips)` (`map_paint`) and the flat headers of its parts' palette entries (`map_headers`, `map_colours`).
    SelectError for a course whose road or scenery does not decode."""
    road = _road(rs, course)
    colours, pieces = map_paint(rs, course)
    return map_object(fit(road), tuple(strips for _, strips in pieces)), map_headers(_entries(rs, colours))
