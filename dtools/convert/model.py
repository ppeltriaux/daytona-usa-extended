"""The Dreamcast course model (`dtools.dc.pol`) as Model 2 polygon objects (`dtools.arcade.polyrom`), grouped by
scenery cell: `convert_model`.

The rules (docs/spike/2026-11-convert.md, "Model"; measured on Beginner's own scenery objects and the format doc,
docs/formats/model2-polygon-rom.md):

- positions: (x, y, z) -> (x, y, 0.0 - z), single precision as stored (the Phase 5 transform). Every vertex must lie in
  WORLD (x and the arcade's z), where the cell rule does not wrap.
- triangles: `pol.triangles` order. A triangle whose corners have zero cross product once transformed (two equal
  corners, or three collinear: `degenerate`) is dropped: it covers no pixel. A triangle list whose entry count is
  not a multiple of 3 is refused (`pol.triangles` would drop the remainder silently).
- the side a face is seen from: the Dreamcast culls by the strip's flag bits 0-1 (PowerVR culling mode; Phase 5).
  Mode 2 faces run clockwise about their stored normals in the Dreamcast's frame, mode 3 anticlockwise, and both are
  seen from the normal's side (CS3's non-degenerate triangles: 16,895 of 16,905 and 5,087 of 5,099; CS0's 6,949 of
  6,951 and all 1,570). After the z negation (a reflection) mode 2's corners run anticlockwise about the seen side
  and are kept, mode 3's are reversed. Modes 0 and 1 do not cull: the face is double-sided (attribute bit 17) and
  seen from the side its stored normals point to (their sum, z negated; a raw-normal mesh's first normal word packs
  the normal as signed bytes x << 16 | y << 8 | z, Phase 5).
- winding and normal (Beginner's 7,669 drawn polygons, every one): the rasterizer's corners (`polyrom.polygons`) run
  anticlockwise about the side the face is seen from (right-handed, the Newell normal points to the viewer), and the
  stored normal is the face's geometric normal pointing the other way: -Newell / |Newell| (within 0.03 degrees on
  all of Beginner's; a road's is (0, -1, 0)). The geometry engine lights by it and culls a back face (dot < 0) unless
  bit 17 is set. So every record's normal is its face's, whatever the Dreamcast's vertex normals were: the raw-normal
  meshes (whose normal words 1-2 hold a colour, Phase 5) are lit like the rest, by their face normal and TEX_PARAM;
  their colour is not carried.
- quads: two consecutive triangles of one strip become one quad when they share an edge (wound the same way), lie in
  one plane (normals within QUAD_ANGLE degrees), make a convex quad and, textured, carry the same (u, v) on the shared
  corners and an affine mapping over the quad (the fourth corner within QUAD_UV texel of the plane's prediction):
  then the quad draws what the two triangles drew. Others stay triangles. The three tests decide in exact rational
  arithmetic on the corners' and (u, v)s' values (`_merge`; Phase 13 Task 2 round 2: a least-squares solve's
  rounding differs between LAPACK builds, and a fourth corner exactly on the shear bound, National Park Speedway's
  cell 0x76, merged on macOS arm64 and not on Linux x86-64), so every machine merges the same triangles; a corner on
  a bound is within it.
- texture points past their range (Phase 11 Task 3; docs/spike/2026-11-courses-4-7.md, "Rin Rin Rink"): with the
  converted textures' sizes (`convert_model`'s `points`), a textured polygon whose texture points on an axis would
  not fit the u16 a point word holds (texture.POINT_MAX in 1/8 texel, after textures.polygon_points' shift by whole
  repeats) is cut along that axis at every multiple of `band(size)` repeats inside it: lines of constant u (or v),
  straight since the mapping is affine, so every piece is a convex slice within one band and fits (`band` is even,
  so a mirrored axis's two-repeat period keeps it). A cut point lies on an original edge, computed once from the
  edge's ends in the order of their texture coordinate (so two pieces sharing the cut get the same point, and so do
  two polygons sharing the edge when both are cut; a neighbour that fits is not cut and keeps the whole edge, a
  T-junction where a hairline crack can show), its coordinate on the cut axis exactly the cut's, the rest
  interpolated along the edge (the picture is unchanged: u and v are continuous across the cut). A slice of more than four corners is fanned
  into quads and a last triangle; a piece draws for the triangles its polygon drew (its `sources`). A polygon that
  fits is untouched.
- cells: a polygon goes to the cell of its corners' centroid (`cell_of`): Beginner's polygons cross cell edges
  freely (555 of 5,091 have a corner in another cell) and 5,062 have their centroid in their object's cell (the 29
  others at most 2.3 units out).
- long polygons cut on the cell grid (`cell_cut`, per course: build.CELL_CUT; v0.1.2): the game draws a cell's
  objects only when its view test opens that cell, so a polygon reaching far past the cell it is filed under leaves a
  hole where it lies in an open cell next to a closed one (Mermaid Lake's start straight, single road strips about
  232 units long filed one cell ahead of the grid: the backdrop showed through during the start's fly-by,
  docs/follow-ups/v0.1.2.md). A polygon reaching more than `cell_cut` units past its own cell (`overhang`) is cut along
  every cell edge it crosses (x or z = CELL * j - 1024.5, where `cell_of`'s rounding changes cell), as the texture
  point cut does: each cut point interpolated along its edge from the edge's ends in coordinate order (neighbours cut
  too agree), its coordinate on the cut axis exactly the edge's, (u, v) interpolated; slices fanned into quads and a
  last triangle; each piece filed under its own centroid's cell. A neighbour that is not cut keeps the whole edge (a
  T-junction).
- the road copied across cell edges (`road_copy`, per course: build.ROAD_COPY; v0.2, Pascal's report on Rin Rin
  Rink): the game's view test opens the cell of a point just ahead of the camera and the neighbours in a 90-degree
  wedge ahead of it, so where the road crosses a cell edge (at about 45 degrees, the wedge's quadrant boundary) it can
  drop the row of cells the camera, trailing the car, still sees: the road there vanished for a few frames and the
  backdrop or the ground under the road showed (measured with dtools/mame/lua/holes_log.lua). A near-flat polygon by
  the road is filed in each neighbouring cell it comes near as well as its own, so it draws whichever is open; only
  by the road and only near the edge, since a copy counts against the frame's object budget (OBC) when both cells
  are open.
- records (Beginner's attribute bits): a drawn polygon is a triangle (attr & 3 == 2) or a quad (1); link (bits 8-9,
  copied into bits 23-24 as on every Beginner record), z-sort 2 (max z), texture-header step 1 (one header per drawn
  polygon), bit 17 double-sided, texture parameter TEX_PARAM, distance coefficient `lod_coef` of the polygon's
  texel density (0 untextured). A polygon is joined to the one before it when they share an edge the link modes
  reach (triangles 1 and 3, quads 1, 2 and 3); a polygon that is not is preceded by a move record, MOVE_ATTR (a
  link-0 quad, never drawn: it only sets the strip's two points; normal zero, step 0, as Beginner's 0x00040401). A
  polygon with no successor ends with link 2. Each chain starts on the edge that reaches an unvisited neighbour.
- objects: per cell, in the Dreamcast's order, joined greedily (the lowest-numbered unvisited polygon that the last
  one reaches, else a move to the lowest-numbered unvisited). An object holds at most MAX_RECORDS records (the
  object count the game sends, course.OBC); a cell with more is split, each object starting from its seeds. The end
  record END. Geometry mode 1's layout (normals present), as every course object.

Before conversion a model may be thinned (`thin`, Phase 7 Task 7: per object a kept fraction of its triangles by
half-edge collapse, the road never touched; docs/spike/2026-11-convert.md, "Thinning") and, first, re-meshed
(`remesh`, Phase 10 Task 3: collapses across meshes within a deviation bound, the road included); a course may convert
with its own quad tolerances (`convert_model`'s `quad`; docs/spike/2026-11-courses-4-7.md, "National Park
Speedway").

Every public function validates its inputs and raises ConvertError (a DcError from `pol.triangles` is re-raised as
one)."""

from __future__ import annotations

import dataclasses
import heapq
import math
import struct
from collections import defaultdict
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import polyrom as P
from dtools.arcade import texture as T
from dtools.arcade.errors import DecodeError
from dtools.arcade.tablebase import _f32, _seq, is_int, is_number
from dtools.convert import ConvertError
from dtools.dc import pol

CELL = 128
"""A scenery cell's edge, world units (course.CELL_UNITS): 16 x 16 cells from GRID_ORIGIN -1024."""
WORLD = (-1024, 1023)
"""Every converted vertex's x and z (inclusive): `cell_of` does not wrap inside it (ground.WORLD)."""
CELL_EDGES = tuple(CELL * j - 1024.5 for j in range(17))
"""The x (and z) where `cell_of` changes cell: it rounds to nearest, so the edge between columns j - 1 and j lies at
CELL * j - 1024 - 0.5."""
MAX_RECORDS = C.OBC
"""Records per object at most: the object count (obc) the draw routines send with every object, 5,000 (course.OBC);
the parser stops after it. Each object's list entry also adds its record count (its `budget`, entry word 3, equal to
the record count on every Beginner object) to the frame's polygon budget, which is OBC too."""

ZSORT = 2                       # attr bits 10-11: max z, as 3,878 of Beginner's 4,248 textured quads
STEP = 1                        # attr bits 12-16: every drawn polygon steps tha (one header per drawn polygon)
TEX_PARAM = 2                   # attr bits 18-22: Beginner's commonest (2,782 of 5,091 drawn; 0 and 3 not decoded)
DOUBLE_SIDED = 1 << 17
MOVE_ATTR = 0x00040401
"""A move record's attribute: a quad, link 0 (always culled), z-sort 1, step 0, texture parameter 1, coefficient 0:
Beginner's 1,730 link-0 records' commonest."""
END = (0, 0x00040401, 0, 0)
"""The end record's words (attribute 0, then three words read as a normal): build.OUTLINE_END, the real outlines'."""
DEAD_LINK = 2                   # a polygon with no successor: link 2 (Beginner's commonest; 35 of its triangles)
QUAD_ANGLE = 0.5                # degrees between two triangles' normals for one quad
QUAD_UV = 1 / 8                 # texels: the fourth corner's (u, v) off the affine prediction (a texture point's unit)
LOD_STEPS = 3                   # distance coefficient steps per doubling of the texel density

_RAW_MASK, _RAW_BITS = pol.RAW_NORMAL, pol.RAW_NORMAL_BITS
_EXITS = {3: ((1, (2, 1)), (3, (0, 2))),                        # link, (A, B) = (P1(n-1), P0(n-1)) as corner indices
          4: ((2, (3, 2)), (1, (2, 1)), (3, (0, 3)))}


class Piece(NamedTuple):
    """One polygon object of a cell, with per record (in record order, move records included):

    - textures: the Dreamcast texture index of the mesh it comes from (-1 untextured; -1 for a move record);
    - uvs: the Dreamcast (u, v) of its corners, in the rasterizer's corner order (`polyrom.polygons`: P1(n-1),
      P0(n-1), P0(n)[, P1(n)]), 3 or 4 pairs; None for a move record. The texture points (`tpa`, 2 words per corner,
      v then u, in this order) are read for every record, a move record's 4 corners included (Beginner: each of its
      cell placements' tpa but the last is one object's all-record points before the next);
    - sources: the indices (into `pol.triangles(m)`) of the triangles it draws: 1 for a triangle, 2 for a quad, ()
      for a move record; the pieces of a cut polygon (texture points past their range) each carry the polygon's."""
    obj: P.Object
    textures: tuple[int, ...]
    uvs: tuple[tuple[tuple[float, float], ...] | None, ...]
    sources: tuple[tuple[int, ...], ...]


class _Poly(NamedTuple):
    corners: tuple                  # arcade positions, anticlockwise about the seen side
    uvs: tuple                      # Dreamcast (u, v) per corner
    texture: int
    double: bool
    coef: int
    sources: tuple[int, ...]


def cell_of(x, z) -> int:
    """The scenery cell of arcade world point (x, z), as the game's 0x172a0 computes it (course.cell_of): round to
    nearest, + 1024, >> 7 on each axis, row (from z) << 4 | column (from x). ConvertError for a value that is not a
    finite number (bool refused) or outside WORLD, where the rule would wrap."""
    for name, v in (("x", x), ("z", z)):
        if not is_number(v) or not WORLD[0] <= v <= WORLD[1]:
            raise ConvertError(f"cell of ({x!r:.30}, {z!r:.30}): {name} is not a finite number inside "
                               f"{WORLD[0]}..{WORLD[1]}")
    return C.cell_of(float(x), float(z))


def lod_coef(density) -> int:
    """The distance coefficient index (attr bits 27-31) of a polygon showing `density` texels per world unit:
    floor(LOD_STEPS * log2(density)), clamped to 0-31. Beginner's rule: it gives 91.7 % of its 4,607 textured drawn
    polygons' coefficient exactly and 98.4 % within one (docs/spike/2026-11-convert.md, "Model"). ConvertError for a
    density that is not a positive finite number."""
    if not is_number(density) or density <= 0:
        raise ConvertError(f"texel density {density!r:.30} is not a positive finite number")
    return max(0, min(31, math.floor(LOD_STEPS * math.log2(float(density)))))


# ---- validation ----

def _raw_normal(words, what: str) -> tuple[float, float, float]:
    """A raw-normal vertex's normal: its first word's signed bytes (x << 16 | y << 8 | z)."""
    if (not isinstance(words, (tuple, list)) or len(words) != 3
            or not all(is_int(w) and 0 <= w <= 0xFFFFFFFF for w in words)):
        raise ConvertError(f"{what}: raw normal {words!r:.60} is not three u32 words")
    w = int(words[0])
    return tuple(float(((w >> s) & 0xFF) - (((w >> s) & 0x80) << 1)) for s in (16, 8, 0))


def _walk(m) -> list[tuple]:
    """Per triangle of `pol.triangles(m)`, in its order: (strip key, flags, texture, raw normals). Validates the parts
    convert_model reads; ConvertError naming the object, mesh, strip and entry."""
    if not isinstance(m, pol.Model):
        raise ConvertError(f"{type(m).__name__} is not a pol.Model")
    textures = _seq(m.textures, None, "textures", "model", ConvertError)
    for i, t in enumerate(textures):
        if not isinstance(t, pol.TexEntry) or not all(is_int(s) and 0 < s <= 0xFFFF for s in (t.width, t.height)):
            raise ConvertError(f"model: texture {i} {t!r:.60} is not a pol.TexEntry with a positive width and height")
    out = []
    for oi, ob in enumerate(_seq(m.objects, None, "objects", "model", ConvertError)):
        if not isinstance(ob, pol.Object):
            raise ConvertError(f"model: object {oi}: {type(ob).__name__} is not a pol.Object")
        for mi, me in enumerate(_seq(ob.meshes, None, "meshes", f"object {oi}", ConvertError)):
            what = f"object {oi} mesh {mi}"
            if not isinstance(me, pol.Mesh):
                raise ConvertError(f"{what}: {type(me).__name__} is not a pol.Mesh")
            params = _seq(me.params, 4, "params", what, ConvertError)
            if not all(is_int(p) and 0 <= p <= 0xFFFFFFFF for p in params):
                raise ConvertError(f"{what}: params {params!r:.60} are not four u32 words")
            tex = me.texture
            if not is_int(tex) or not (tex == pol.NO_TEXTURE or 0 <= tex < len(textures)):
                raise ConvertError(f"{what}: texture index {tex!r:.30} is neither -1 nor inside the "
                                   f"{len(textures)}-entry texture table")
            raw = (params[0] & _RAW_MASK) == _RAW_BITS
            for si, st in enumerate(_seq(me.strips, None, "strips", what, ConvertError)):
                swhat = f"{what} strip {si}"
                if not isinstance(st, pol.Strip):
                    raise ConvertError(f"{swhat}: {type(st).__name__} is not a pol.Strip")
                if not is_int(st.flags) or not 0 <= st.flags <= 0xFFFFFFFF:
                    raise ConvertError(f"{swhat}: flags {st.flags!r:.30} are not a u32")
                entries = _seq(st.entries, None, "entries", swhat, ConvertError)
                for ei, e in enumerate(entries):
                    ewhat = f"{swhat} entry {ei}"
                    if isinstance(e, pol.Reuse):
                        if not is_int(e.raw) or not 0 <= e.raw <= 0xFFFFFFFF or not pol._is_ref(int(e.raw)):
                            raise ConvertError(f"{ewhat}: re-use word {e.raw!r:.30} is not a u32 0x5ff.....")
                        continue
                    if not isinstance(e, pol.Vertex):
                        raise ConvertError(f"{ewhat}: {type(e).__name__} is neither a pol.Vertex nor a pol.Reuse")
                    x, _, z = struct.unpack("<3f", _f32((e.x, e.y, e.z), 3, "vertex position", ewhat, ConvertError))
                    if not (WORLD[0] <= x <= WORLD[1] and WORLD[0] <= -z <= WORLD[1]):
                        raise ConvertError(f"{ewhat}: vertex ({e.x}, {e.y}, {e.z}) is outside the world: its x and "
                                           f"negated z must lie in {WORLD[0]}..{WORLD[1]}")
                    _f32((e.u, e.v), 2, "u, v", ewhat, ConvertError)
                    if raw:
                        _raw_normal(e.normal, ewhat)
                    else:
                        _f32(e.normal, 3, "normal", ewhat, ConvertError)
                if st.flags & pol.LIST_FLAG and len(entries) % 3:
                    raise ConvertError(f"{swhat}: a triangle list holds {len(entries)} entries, not a multiple of 3")
                n = len(entries) // 3 if st.flags & pol.LIST_FLAG else max(len(entries) - 2, 0)
                out += [((oi, mi, si), int(st.flags), int(tex), raw)] * n
    return out


# ---- polygons ----

def _pos(v: pol.Vertex) -> tuple[float, float, float]:
    x, y, z = struct.unpack("<3f", struct.pack("<3f", v.x, v.y, v.z))
    return x, y, 0.0 - z


def degenerate(corners) -> bool:
    """True for a triangle that covers no area: the cross product (b - a) x (c - a) of its three corners (arcade
    positions, as `convert_model` rounds them) is exactly zero in double precision (two equal corners, or three
    collinear). `convert_model` drops these. ConvertError for corners that are not three finite xyz triples."""
    if not isinstance(corners, (tuple, list)) or len(corners) != 3:
        raise ConvertError(f"triangle corners {corners!r:.60} are not three points")
    p = [np.frombuffer(_f32(c, 3, "corner", "triangle", ConvertError), "<f4").astype(float) for c in corners]
    return not np.cross(p[1] - p[0], p[2] - p[0]).any()


def _newell(points) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)


def _seen_normal(verts, raw: bool) -> np.ndarray:
    """The side the stored normals point to, in the arcade's frame (their sum, z negated)."""
    n = sum(np.array(_raw_normal(v.normal, "vertex") if raw else tuple(map(float, v.normal))) for v in verts)
    return np.array((n[0], n[1], -n[2]))


def _texel_area(uvs, size) -> float:
    w, h = size
    (ua, va), rest = uvs[0], uvs[1:]
    return sum(abs((u1 - ua) * w * (v2 - va) * h - (v1 - va) * h * (u2 - ua) * w) / 2
               for (u1, v1), (u2, v2) in zip(rest, rest[1:]))


def _coef(corners, uvs, size) -> int:
    if size is None:
        return 0
    p = np.asarray(corners, dtype=float)
    world = sum(np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])) for k in range(1, len(p) - 1)) / 2
    texel = _texel_area(uvs, size)
    if not texel > 0:
        return 0                                    # every corner on one texel: no density
    density = math.sqrt(texel / world)
    return lod_coef(density) if math.isfinite(density) else 31


def _triangle(tri, info, sizes, index) -> _Poly | None:
    """The oriented arcade triangle of one Dreamcast triangle, or None for a degenerate one."""
    _, flags, tex, raw = info
    verts = tri[:3]
    corners = [_pos(v) for v in verts]
    if degenerate(corners):
        return None
    face = _newell(corners)
    mode = flags & 3
    if mode == 3:
        keep = False
    elif mode == 2:
        keep = True
    else:
        keep = bool(face @ _seen_normal(verts, raw) >= 0)
    order = (0, 1, 2) if keep else (0, 2, 1)
    corners = tuple(corners[k] for k in order)
    uvs = tuple((float(verts[k].u), float(verts[k].v)) for k in order)
    size = sizes[tex] if tex >= 0 else None
    return _Poly(corners, uvs, tex, mode < 2, _coef(corners, uvs, size), (index,))


def _sub(a, b) -> tuple:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _xprod(a, b) -> tuple:
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _exact_newell(points) -> tuple:
    """`_newell` of points whose coordinates are Fractions, exactly (the sum of each edge's cross product)."""
    n = (0, 0, 0)
    for i, a in enumerate(points):
        c = _xprod(a, points[(i + 1) % len(points)])
        n = (n[0] + c[0], n[1] + c[1], n[2] + c[2])
    return n


def _merge(a: _Poly, b: _Poly, sizes, quad: tuple = (QUAD_ANGLE, QUAD_UV, False)) -> _Poly | None:
    """The quad of two triangles (QUAD rules in the module docstring; `quad` the tolerances: degrees, then texels, or
    with its third item True a fraction of the texture's smaller side), or None. Every test is exact (Fractions of the
    corners' and (u, v)s' float values): the fold (the normals' dot product against cos(angle) times their lengths,
    compared squared), the strict convexity (each turn against 1e-9 of |n|^2) and the shear (the fourth corner's
    offset from the least-squares affine prediction, from the 2 x 2 normal equations, against the bound, squared)."""
    angle, uv_tol, relative = quad
    if a.texture != b.texture or a.double != b.double:
        return None
    for k in range(3):
        p, q = a.corners[(k + 1) % 3], a.corners[(k + 2) % 3]
        if (q, p) in ((b.corners[j], b.corners[(j + 1) % 3]) for j in range(3)):
            break
    else:
        return None
    x = a.corners[k]
    jq = b.corners.index(q)
    d = b.corners[(jq + 2) % 3]                     # b = (q, p, d) rotated
    corners = (x, p, d, q)
    if len(set(corners)) != 4:
        return None
    exact = {v: tuple(map(Fraction, v)) for v in corners}
    na, nb = _exact_newell([exact[v] for v in a.corners]), _exact_newell([exact[v] for v in b.corners])
    dot, cos = _dot(na, nb), Fraction(math.cos(math.radians(angle)))
    if dot < 0 or dot * dot < cos * cos * _dot(na, na) * _dot(nb, nb):
        return None                                 # folded past `angle` (cos > 0: angle < 90 degrees)
    c = [exact[v] for v in corners]
    n = _exact_newell(c)
    bound = Fraction(1e-9) * _dot(n, n)
    for i in range(4):
        if not _dot(_xprod(_sub(c[(i + 1) % 4], c[i]), _sub(c[(i + 2) % 4], c[(i + 1) % 4])), n) > bound:
            return None                             # not strictly convex
    uv_a = dict(zip(a.corners, a.uvs))
    uv_b = dict(zip(b.corners, b.uvs))
    if a.texture >= 0:
        if uv_a[p] != uv_b[p] or uv_a[q] != uv_b[q]:
            return None
        w, h = sizes[a.texture]
        e1, e2, r = _sub(c[1], c[0]), _sub(c[3], c[0]), _sub(c[2], c[0])      # x->p, x->q, x->d
        g11, g12, g22, b1, b2 = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2), _dot(e1, r), _dot(e2, r)
        det = g11 * g22 - g12 * g12                 # > 0: strictly convex, so x->p and x->q are independent
        s, t = (b1 * g22 - b2 * g12) / det, (g11 * b2 - g12 * b1) / det
        (ux, vx), (up, vp), (uq, vq), (ud, vd) = (tuple(map(Fraction, uv)) for uv in (uv_a[x], uv_a[p], uv_a[q],
                                                                                         uv_b[d]))
        du = (ux + s * (up - ux) + t * (uq - ux) - ud) * w
        dv = (vx + s * (vp - vx) + t * (vq - vx) - vd) * h
        tol = Fraction(uv_tol) * min(w, h) if relative else Fraction(uv_tol)
        if du * du + dv * dv > tol * tol:
            return None
    uvs = (uv_a[x], uv_a[p], uv_b[d], uv_a[q])
    size = sizes[a.texture] if a.texture >= 0 else None
    return _Poly(corners, uvs, a.texture, a.double, _coef(corners, uvs, size), a.sources + b.sources)


def _polygons(m: pol.Model, quad: tuple = (QUAD_ANGLE, QUAD_UV, False)) -> list[_Poly]:
    info = _walk(m)
    try:
        tris = list(pol.triangles(m))
    except DecodeError as err:
        raise ConvertError(f"model: {err}") from err
    sizes = [(int(t.width), int(t.height)) for t in m.textures]
    polys: list[_Poly] = []
    pending: _Poly | None = None
    pending_key = None
    for i, (tri, inf) in enumerate(zip(tris, info)):
        t = _triangle(tri, inf, sizes, i)
        if t is None:
            continue
        if pending is not None and pending_key == inf[0] and pending.sources[-1] == i - 1:
            q = _merge(pending, t, sizes, quad)
            if q is not None:
                polys.append(q)
                pending = None
                continue
        if pending is not None:
            polys.append(pending)
        pending, pending_key = t, inf[0]
    if pending is not None:
        polys.append(pending)
    return polys


# ---- texture points past their range (Phase 11 Task 3) ----

POINT_UNIT = 8                  # texture point words per texel (textures.POINT_UNIT)


def band(size) -> int:
    """The most repeats of a texture `size` texels on an axis that a cut piece spans: the largest even number whose
    texture points fit the u16 (texture.POINT_MAX, in 1/8 texel). Even, so a mirrored axis (a two-repeat period) cuts
    on its period too. 0 when not even two repeats fit (no side the converter makes: at most 256 texels, 30 repeats).
    ConvertError for a size that is not a positive int (bool refused)."""
    if not is_int(size) or size <= 0:
        raise ConvertError(f"texture size {size!r:.30} is not a positive int")
    return (T.POINT_MAX // (POINT_UNIT * int(size))) & ~1


def _axis(uv, k: int) -> float:
    """A corner's texture coordinate on axis k in repeats as textures.polygon_points scales it: u, or 1 - v (rows are
    stored top first)."""
    return float(uv[0]) if k == 0 else 1.0 - float(uv[1])


def _fits(uvs, k: int, size: int) -> bool:
    """textures.polygon_points' range check on axis k (unmirrored): the points, rounded to 1/8 texel and shifted by
    whole repeats to the least in the first, within texture.POINT_MAX."""
    repeat = POINT_UNIT * size
    words = [math.floor(repeat * _axis(uv, k) + 0.5) for uv in uvs]
    return max(words) - (min(words) // repeat) * repeat <= T.POINT_MAX


def _slices(corners: list, uvs: list, k: int, step: int) -> list[tuple[list, list]]:
    """The convex polygon (corners, uvs) cut along axis k at every multiple of `step` repeats strictly inside it: its
    slices in band order, each (corners, uvs) in the polygon's order (`convert_model`'s rule)."""
    s = [_axis(uv, k) for uv in uvs]
    cuts = [j * step for j in range(math.floor(min(s) / step), math.ceil(max(s) / step) + 1)
            if min(s) < j * step < max(s)]
    if not cuts:
        return [(corners, uvs)]
    n, ring = len(corners), []                          # (coordinate, corner, uv) round the boundary
    for i in range(n):
        j = (i + 1) % n
        ring.append((s[i], corners[i], uvs[i]))
        a, b = (i, j) if s[i] < s[j] else (j, i)        # the ends in coordinate order: a neighbour cut too agrees
        on = []
        for c in (c for c in cuts if s[a] < c < s[b]):
            t = (c - s[a]) / (s[b] - s[a])
            pa, pb, ua, ub = corners[a], corners[b], uvs[a], uvs[b]
            point = tuple(float(x + t * (y - x)) for x, y in zip(pa, pb))
            uv = (float(c), float(ua[1] + t * (ub[1] - ua[1]))) if k == 0 else (
                float(ua[0] + t * (ub[0] - ua[0])), float(1 - c))
            on.append((float(c), point, uv))
        ring += on if a == i else on[::-1]
    out = []
    bounds = [-math.inf] + cuts + [math.inf]
    for lo, hi in zip(bounds, bounds[1:]):
        sel = [(c, u) for v, c, u in ring if lo <= v <= hi]
        sel = [q for i, q in enumerate(sel) if q[0] != sel[i - 1][0]] if len(sel) > 1 else sel
        if len(sel) >= 3:
            out.append(([c for c, _ in sel], [u for _, u in sel]))
    return out


def overhang(corners) -> float:
    """How far (world units, the largest of the four sides) the polygon with these arcade corners reaches past the
    cell of its centroid (`cell_of`'s cell, edges CELL_EDGES); 0.0 when it lies inside."""
    c = np.asarray(corners, dtype=float)
    cell = cell_of(float(c[:, 0].mean()), float(c[:, 2].mean()))
    x0, z0 = CELL_EDGES[cell & 15], CELL_EDGES[cell >> 4]
    return float(max(0.0, x0 - c[:, 0].min(), c[:, 0].max() - (x0 + CELL),
                     z0 - c[:, 2].min(), c[:, 2].max() - (z0 + CELL)))


def _cell_slices(corners: list, uvs: list, k: int) -> list[tuple[list, list]]:
    """The convex polygon (corners, uvs) cut along world axis k (0 x, 2 z) at every cell edge (CELL_EDGES) strictly
    inside it: its slices in coordinate order, each (corners, uvs) in the polygon's order (the module docstring's
    rule, as `_slices` cuts on a texture axis)."""
    s = [float(c[k]) for c in corners]
    cuts = [e for e in CELL_EDGES if min(s) < e < max(s)]
    if not cuts:
        return [(corners, uvs)]
    n, ring = len(corners), []
    for i in range(n):
        j = (i + 1) % n
        ring.append((s[i], corners[i], uvs[i]))
        a, b = (i, j) if s[i] < s[j] else (j, i)        # the ends in coordinate order: a neighbour cut too agrees
        on = []
        for e in (e for e in cuts if s[a] < e < s[b]):
            t = (e - s[a]) / (s[b] - s[a])
            point = tuple(float(e) if q == k else float(x + t * (y - x))
                          for q, (x, y) in enumerate(zip(corners[a], corners[b])))
            uv = tuple(float(x + t * (y - x)) for x, y in zip(uvs[a], uvs[b]))
            on.append((float(e), point, uv))
        ring += on if a == i else on[::-1]
    out = []
    bounds = [-math.inf] + cuts + [math.inf]
    for lo, hi in zip(bounds, bounds[1:]):
        sel = [(c, u) for v, c, u in ring if lo <= v <= hi]
        sel = [q for i, q in enumerate(sel) if q[0] != sel[i - 1][0]] if len(sel) > 1 else sel
        if len(sel) >= 3:
            out.append(([c for c, _ in sel], [u for _, u in sel]))
    return out


def _cell_cut(p: _Poly, reach: float | None, sizes) -> list[_Poly]:
    """`p`, or its pieces cut on the cell grid when it reaches more than `reach` units past its own cell
    (`overhang`; None cuts nothing): the module docstring's rule."""
    if reach is None or overhang(p.corners) <= reach:
        return [p]
    parts = [(list(p.corners), list(p.uvs))]
    for k in (0, 2):
        parts = [q for c, u in parts for q in _cell_slices(c, u, k)]
    out = []
    for corners, uvs in parts:
        for idx in ([0, i, i + 1, i + 2] if i + 2 < len(corners) else [0, i, i + 1]
                    for i in range(1, len(corners) - 1, 2)):
            c, u = tuple(corners[i] for i in idx), tuple(uvs[i] for i in idx)
            if not _newell(c).any():
                continue                                # a sliver with no area
            coef = _coef(c, u, sizes[p.texture]) if p.texture >= 0 else p.coef
            out.append(_Poly(c, u, p.texture, p.double, coef, p.sources))
    return out


def _cut(p: _Poly, points: dict | None, sizes) -> list[_Poly]:
    """`p`, or its pieces when its texture points would pass their range on the converted texture (`points`:
    texture -> (width, height); `convert_model`'s rule). ConvertError for a texture too small a band for even two
    repeats or a piece that still does not fit."""
    if p.texture < 0 or not points or p.texture not in points:
        return [p]
    parts = [(list(p.corners), list(p.uvs))]
    for k, size in enumerate(points[p.texture]):
        if all(_fits(uvs, k, size) for _, uvs in parts):
            continue
        step = band(size)
        if not step:
            raise ConvertError(f"texture {p.texture}: {size} texels on {'uv'[k]}: no two repeats fit a texture point "
                               f"(0-{T.POINT_MAX:#x}, 1/8 texel), so a polygon past it cannot be cut to fit")
        parts = [q for c, uvs in parts for q in (_slices(c, uvs, k, step) if not _fits(uvs, k, size) else
                                                  [(c, uvs)])]
    if len(parts) == 1 and parts[0][0] == list(p.corners):
        return [p]
    out = []
    size = sizes[p.texture]
    for corners, uvs in parts:
        for idx in ([0, i, i + 1, i + 2] if i + 2 < len(corners) else [0, i, i + 1]
                    for i in range(1, len(corners) - 1, 2)):
            c, u = tuple(corners[i] for i in idx), tuple(uvs[i] for i in idx)
            if not _newell(c).any():
                continue                                # a sliver with no area
            if not all(_fits(u, k, s_) for k, s_ in enumerate(points[p.texture])):
                raise ConvertError(f"texture {p.texture}: a cut piece's points {u!r:.80} still pass the range "
                                   f"(0-{T.POINT_MAX:#x}, 1/8 texel)")
            out.append(_Poly(c, u, p.texture, p.double, _coef(c, u, size), p.sources))
    return out


# ---- records ----

DEPTH_TEST = 1 << 7
"""Attr bit 7 (v0.3): asks the custom MAME's renderer for a per-pixel depth test on this polygon (tools/
mame-daytona-rom-space.patch; docs/formats/model2-polygon-rom.md, "The depth test"). No record of the original set has
it (101,462 records of the three courses read), and stock MAME and the hardware ignore it."""


def _attr(n: int, link: int, p: _Poly, depth: bool = False) -> int:
    return ((1 if n == 4 else 2) | link << 8 | ZSORT << 10 | STEP << 12 | (DOUBLE_SIDED if p.double else 0)
            | TEX_PARAM << 18 | link << 23 | p.coef << 27 | (DEPTH_TEST if depth else 0))


def _f32_point(p) -> tuple[float, float, float]:
    return struct.unpack("<3f", struct.pack("<3f", *p))


def _rotate(p: _Poly, k: int) -> _Poly:
    return p._replace(corners=p.corners[k:] + p.corners[:k], uvs=p.uvs[k:] + p.uvs[:k])


def _chain(polys: list[_Poly]) -> list[tuple[_Poly, bool]]:
    """The cell's polygons in drawing order, each rotated so that its first two corners are (P1(n-1), P0(n-1)), with
    True when it is joined to the one before (else a move record or the seeds precede it). The link of each is
    implied: the exit its successor starts on, or DEAD_LINK."""
    edges: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(polys):
        n = len(p.corners)
        for k in range(n):
            edges[(p.corners[k], p.corners[(k + 1) % n])].append(i)
    used = [False] * len(polys)

    def successor(p: _Poly):
        best = None
        for _, (a, b) in _EXITS[len(p.corners)]:
            key = (p.corners[a], p.corners[b])
            for j in edges.get(key, ()):
                if not used[j] and (best is None or j < best[0]):
                    best = (j, key)
                    break
        return best

    out: list[tuple[_Poly, bool]] = []
    nxt = 0
    prev: _Poly | None = None
    while True:
        found = successor(prev) if prev is not None else None
        if found is not None:
            j, (a, b) = found
            p = polys[j]
            n = len(p.corners)
            k = next(k for k in range(n) if p.corners[k] == a and p.corners[(k + 1) % n] == b)
            used[j] = True
            prev = _rotate(p, k)
            out.append((prev, True))
            continue
        while nxt < len(polys) and used[nxt]:
            nxt += 1
        if nxt == len(polys):
            return out
        used[nxt] = True
        p = polys[nxt]
        starts = [_rotate(p, k) for k in range(len(p.corners))]
        prev = next((s for s in starts if successor(s) is not None), starts[0])
        out.append((prev, False))


def _link(p: _Poly, nxt: _Poly | None) -> int:
    """The link mode that makes `nxt` start on its first two corners after `p`."""
    if nxt is not None:
        for link, (a, b) in _EXITS[len(p.corners)]:
            if (p.corners[a], p.corners[b]) == nxt.corners[:2]:
                return link
    return DEAD_LINK


def _pieces(polys: list[_Poly], depth: bool = False) -> tuple[Piece, ...]:
    order = _chain(polys)
    pieces = []
    recs: list = []
    meta: list = []
    seeds = None

    def close():
        if recs:
            obj = P.Object(seeds[0], seeds[1], list(recs), 6 + 10 * len(recs) + len(END), END)
            pieces.append(Piece(obj, tuple(t for t, _, _ in meta), tuple(u for _, u, _ in meta),
                                tuple(s for _, _, s in meta)))

    for k, (p, joined) in enumerate(order):
        a, b = (_f32_point(c) for c in p.corners[:2])
        need = 1 if joined or not recs else 2
        if len(recs) + need > MAX_RECORDS:
            close()
            recs, meta = [], []
        if not recs:
            seeds = (b, a)                          # seed 0 is P0(n-1), seed 1 is P1(n-1)
        elif not joined:
            recs.append(P.Record(MOVE_ATTR, (0.0, 0.0, 0.0), b, a))
            meta.append((-1, None, ()))
        nxt = order[k + 1][0] if k + 1 < len(order) and order[k + 1][1] else None
        n = len(p.corners)
        face = _newell(p.corners)
        normal = _f32_point(-face / np.linalg.norm(face))
        rest = [_f32_point(c) for c in p.corners[2:]]
        recs.append(P.Record(_attr(n, _link(p, nxt), p, depth), normal, rest[0], rest[1] if n == 4 else None))
        meta.append((p.texture, p.uvs, p.sources))
    close()
    return tuple(pieces)


def triangle_meshes(m) -> tuple[tuple[int, int], ...]:
    """Per triangle of `pol.triangles(m)`, in its order: the (object, mesh) indices of the mesh it belongs to. A
    Piece's `sources` index this, so each record's mesh (its texture mode, blending and colour) can be read back.
    ConvertError as `convert_model` for a model it refuses."""
    return tuple(key[:2] for key, *_ in _walk(m))


def drop_textures(m, textures) -> pol.Model:
    """`m` without the meshes drawn with any of `textures` (Dreamcast texture indices into `m.textures`): every other
    mesh and its strips as they are, in order (re-uses reach only their own mesh's vertices, so the rest still
    resolve), each object's strip-entry count that of its kept meshes, the texture table unchanged. ConvertError for an
    `m` that is not a pol.Model of the pol types (`convert_model`'s checks), or `textures` that are not a sequence of
    integer indices into the texture table (bool refused; -1, no texture, is not one)."""
    _walk(m)
    if not isinstance(textures, (tuple, list)) or not all(is_int(t) and 0 <= t < len(m.textures) for t in textures):
        raise ConvertError(f"textures {textures!r:.60} are not indices into the {len(m.textures)}-entry texture table")
    drop = {int(t) for t in textures}
    return _without(m, {(oi, mi) for oi, ob in enumerate(m.objects) for mi, me in enumerate(ob.meshes)
                        if me.texture in drop})


def drop_meshes(m, meshes) -> pol.Model:
    """`m` without the named meshes ((object, mesh) index pairs into `m`): every other mesh as `drop_textures` keeps
    them; an object whose every mesh is dropped stays, with no mesh (object indices do not move). ConvertError for an
    `m` that `drop_textures` refuses, or `meshes` that are not a sequence of (object, mesh) integer pairs naming a mesh
    of `m` (bool refused)."""
    _walk(m)
    ok = isinstance(meshes, (tuple, list)) and all(
        isinstance(k, tuple) and len(k) == 2 and all(is_int(i) for i in k) and 0 <= k[0] < len(m.objects)
        and 0 <= k[1] < len(m.objects[k[0]].meshes) for k in meshes)
    if not ok:
        raise ConvertError(f"meshes {meshes!r:.60} are not (object, mesh) pairs of the model")
    return _without(m, {(int(oi), int(mi)) for oi, mi in meshes})


def _without(m: pol.Model, drop: set) -> pol.Model:
    if not drop:
        return m
    objects = []
    for oi, ob in enumerate(m.objects):
        meshes = tuple(me for mi, me in enumerate(ob.meshes) if (oi, mi) not in drop)
        count = sum(len(st.entries) for me in meshes for st in me.strips)
        objects.append(pol.Object(ob.kind, ob.sphere, meshes, count) if len(meshes) != len(ob.meshes) else ob)
    return pol.Model(m.header, tuple(objects), m.textures)


ROAD_COPY_FLAT = 0.8
"""A polygon is ground for `road_copy` when its normal's vertical component is at least this share of its length."""


def _road_copy(road_copy):
    """`convert_model`'s `road_copy` checked: None, or (road centres as a list of at least three (x, z) pairs, margin,
    band) with margin and band finite numbers >= 0; returned as (centres, margin, band) of floats."""
    if road_copy is None:
        return None
    ok = isinstance(road_copy, (tuple, list)) and len(road_copy) == 3
    if ok:
        road, margin, band = road_copy
        ok = (isinstance(road, (tuple, list)) and len(road) >= 3 and all(
            isinstance(q, (tuple, list)) and len(q) == 2 and all(is_number(v) and math.isfinite(v) for v in q)
            for q in road) and all(is_number(v) and math.isfinite(v) and v >= 0 for v in (margin, band)))
    if not ok:
        raise ConvertError(f"road_copy {road_copy!r:.60} is not (road centres: three or more (x, z), margin >= 0, "
                           f"band >= 0)")
    return [(float(x), float(z)) for x, z in road], float(margin), float(band)


def _road_distance(x: float, z: float, road) -> float:
    """The distance from (x, z) to the closed polyline `road` (plain float arithmetic)."""
    best = math.inf
    for k in range(len(road)):
        (ax, az), (bx, bz) = road[k], road[(k + 1) % len(road)]
        dx, dz = bx - ax, bz - az
        ll = dx * dx + dz * dz
        t = 0.0 if not ll else min(1.0, max(0.0, ((x - ax) * dx + (z - az) * dz) / ll))
        best = min(best, math.hypot(x - ax - t * dx, z - az - t * dz))
    return best


def _copy_cells(p: _Poly, home: int, copy) -> list[int]:
    """The neighbouring cells `p` (filed in `home`) is copied to under `road_copy` (`copy`, checked): none unless it is
    near-flat (ROAD_COPY_FLAT) with its centroid within band of the road; else each neighbour (edge or diagonal) whose
    square (CELL_EDGES) its corners' extent comes within margin of."""
    road, margin, band = copy
    c = p.corners
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = c[0], c[1], c[2]
    ux, uy, uz, vx, vy, vz = x1 - x0, y1 - y0, z1 - z0, x2 - x0, y2 - y0, z2 - z0
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not length or abs(ny) < ROAD_COPY_FLAT * length:
        return []
    xs, zs = [q[0] for q in c], [q[2] for q in c]
    if _road_distance(sum(xs) / len(xs), sum(zs) / len(zs), road) > band:
        return []
    row, col = home >> 4, home & 15
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not (dr or dc) or not (0 <= row + dr < 16 and 0 <= col + dc < 16):
                continue
            lo_x, hi_x = CELL_EDGES[col + dc], CELL_EDGES[col + dc + 1]
            lo_z, hi_z = CELL_EDGES[row + dr], CELL_EDGES[row + dr + 1]
            if max(xs) + margin >= lo_x and min(xs) - margin <= hi_x and \
                    max(zs) + margin >= lo_z and min(zs) - margin <= hi_z:
                out.append((row + dr) << 4 | (col + dc))
    return out


def convert_model(m, quad=None, points=None, cell_cut=None, road_copy=None,
                  depth: bool = False) -> dict[int, tuple[Piece, ...]]:
    """The course model `m` as polygon objects per scenery cell: cell (row << 4 | col) -> its Pieces (one object
    each, at most MAX_RECORDS records; several when the cell has more), only the cells that hold a polygon. Every
    non-degenerate triangle of `pol.triangles(m)` is drawn by exactly one record of one Piece (`Piece.sources`), or by
    the pieces of its polygon where that is cut (`points`). The
    rules are the module docstring's. ConvertError for an `m` that is not a pol.Model of the pol types, a vertex
    whose position, (u, v) or normal is not finite (or does not fit single precision; a raw-normal mesh's normal
    words not three u32), a vertex outside WORLD, a texture index past the texture table, a texture entry without a
    positive size, a re-use whose word is not 0x5ff..... or that lands on no earlier vertex, or a triangle list
    whose entry count is not a multiple of 3, or a `quad` that is neither None nor two positive finite numbers (the
    degrees under 90).

    `quad`: per course (build.QUAD; Phase 10 Task 3, fix round 1's ruling I1), the quad tolerances (degrees between
    the two triangles' normals, under 90 since `_merge` compares the fold squared; the fourth corner's shear off the
    affine prediction as a fraction of its texture's smaller side) in place of QUAD_ANGLE and QUAD_UV (absolute
    texels); None keeps those.

    `points`: the converted textures' sizes, texture index -> (width, height) in texels (build.model_scenery passes
    its headers'), with which a polygon whose texture points would pass their u16 range is cut (the module docstring,
    "texture points past their range"; Phase 11 Task 3); None (or a texture not named) cuts nothing. ConvertError
    also for `points` that is neither None nor a dict of texture indices to two positive ints, and for a polygon that
    cannot be cut to fit.

    `cell_cut`: per course (build.CELL_CUT; v0.1.2), the world units a polygon may reach past its own cell before it
    is cut on the cell grid (the module docstring, "long polygons cut on the cell grid"), after the texture point cut;
    None cuts nothing. ConvertError also for a `cell_cut` that is neither None nor a finite number >= 0.

    `road_copy`: per course (build.ROAD_COPY; v0.2), (the road line's centres (x, z), margin, band): a near-flat
    polygon whose centroid is within `band` units of the road line is also filed in each neighbouring cell its extent
    comes within `margin` units of (`_copy_cells`; the module docstring, "the road copied across cell edges"), so such
    a triangle is drawn by one record in each of those cells; None copies nothing. ConvertError also for a bad
    `road_copy` (`_road_copy`).

    `depth` (v0.3): every polygon record carries DEPTH_TEST (the move records do not: they draw nothing), so the
    custom MAME resolves the model's overlapping layers per pixel as the Dreamcast did; nothing else changes.
    ConvertError for a `depth` that is not a bool."""
    if not isinstance(depth, bool):
        raise ConvertError(f"depth {depth!r:.40} is not a bool")
    if cell_cut is not None and not (is_number(cell_cut) and math.isfinite(cell_cut) and cell_cut >= 0):
        raise ConvertError(f"cell_cut {cell_cut!r:.40} is not a finite number of world units >= 0")
    if points is not None and not (isinstance(points, dict) and all(
            is_int(t) and isinstance(v, tuple) and len(v) == 2 and all(is_int(x) and x > 0 for x in v)
            for t, v in points.items())):
        raise ConvertError(f"points {points!r:.60} is not a dict of texture index -> (width, height)")
    copy = _road_copy(road_copy)
    polys = _polygons(m, _quad(quad))
    sizes = [(int(t.width), int(t.height)) for t in m.textures]
    cells: dict[int, list[_Poly]] = defaultdict(list)
    for p in (r for p0 in polys for q in _cut(p0, points, sizes) for r in _cell_cut(q, cell_cut, sizes)):
        c = np.asarray(p.corners, dtype=float)
        home = cell_of(float(c[:, 0].mean()), float(c[:, 2].mean()))
        cells[home].append(p)
        for other in (_copy_cells(p, home, copy) if copy else ()):
            cells[other].append(p)
    return {cell: _pieces(cells[cell], depth) for cell in sorted(cells)}


def _quad(quad) -> tuple:
    """`convert_model`'s `quad` checked, as `_merge` takes it: (QUAD_ANGLE, QUAD_UV, False) for None, else (degrees,
    fraction, True)."""
    if quad is None:
        return QUAD_ANGLE, QUAD_UV, False
    if not (isinstance(quad, (tuple, list)) and len(quad) == 2 and all(
            is_number(x) and math.isfinite(x) and x > 0 for x in quad) and quad[0] < 90):
        raise ConvertError(f"quad {quad!r:.60} is not (degrees under 90, fraction of the texture's side), two positive "
                           f"finite numbers")
    return float(quad[0]), float(quad[1]), True


# ---- thinning (Phase 7, Task 7) ----

THIN_TURN = 60.0
"""Degrees: a collapse may not turn any triangle it moves further than this from its own facing (nor flatten one)."""


def _keep_of(keep, oi: int) -> float:
    try:
        f = keep(oi)
    except Exception as err:                        # the caller's callable: whatever it raises becomes ours
        raise ConvertError(f"keep({oi}) raised {type(err).__name__}: {err}") from err
    if not is_number(f) or not 0 < f <= 1:
        raise ConvertError(f"keep({oi}) is {f!r:.30}, not a number in (0, 1]")
    return float(f)


def _quadric(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The area-weighted plane quadric of a triangle: n the (unnormalised) cross product, |n| twice its area."""
    a = np.linalg.norm(n)
    if a == 0:
        return np.zeros((4, 4))
    plane = np.append(n / a, -(n / a) @ p)
    return (a / 2) * np.outer(plane, plane)


def _decimate(tris: list, verts: list, locked: set, target: int) -> None:
    """Half-edge collapses (`thin`'s rules) on one object's triangles, `tris` (three vertex ids each, into `verts`;
    edited in place, None for a removed one), until at most `target` non-degenerate ones remain or no collapse is
    allowed. `locked` vertices never move."""
    p = np.array([_pos(v) for v in verts], dtype=float)
    around: list[set[int]] = [set() for _ in verts]          # vertex -> the triangles at it
    q = np.zeros((len(verts), 4, 4))
    alive = 0
    for ti, t in enumerate(tris):
        n = np.cross(p[t[1]] - p[t[0]], p[t[2]] - p[t[0]])
        alive += bool(n.any())
        for vid in t:
            around[vid].add(ti)
            q[vid] += _quadric(p[t[0]], n)
    version = [0] * len(verts)
    heap: list = []
    cos_turn = math.cos(math.radians(THIN_TURN))

    def neighbours(u: int) -> set[int]:
        return {w for ti in around[u] for w in tris[ti]} - {u}

    def push(u: int, v: int) -> None:
        if u not in locked:
            h = np.append(p[v], 1.0)
            heapq.heappush(heap, (float(h @ q[u] @ h), u, v, version[u], version[v]))

    def allowed(u: int, v: int) -> bool:
        shared = [ti for ti in around[u] if v in tris[ti]]
        if neighbours(u) & neighbours(v) != {w for ti in shared for w in tris[ti]} - {u, v}:
            return False
        for ti in around[u] - set(shared):
            t = tris[ti]
            old = np.cross(p[t[1]] - p[t[0]], p[t[2]] - p[t[0]])
            c = [v if w == u else w for w in t]
            new = np.cross(p[c[1]] - p[c[0]], p[c[2]] - p[c[0]])
            if not new @ old > cos_turn * np.linalg.norm(new) * np.linalg.norm(old):
                return False
        return True

    for u in range(len(verts)):
        for v in neighbours(u):
            push(u, v)
    while alive > target and heap:
        _, u, v, vu, vv = heapq.heappop(heap)
        if vu != version[u] or vv != version[v] or not around[u] or v not in neighbours(u) or not allowed(u, v):
            continue
        for ti in list(around[u]):
            t = tris[ti]
            if v in t:
                for w in t:
                    around[w].discard(ti)
                tris[ti] = None
                alive -= 1
            else:
                t[t.index(u)] = v
                around[v].add(ti)
        around[u] = set()
        q[v] += q[u]
        version[u] += 1
        version[v] += 1
        for w in neighbours(v):
            push(v, w)
            push(w, v)


def _rebuild(ob: pol.Object, tris: list, meta: list, verts: list, whole: dict) -> pol.Object:
    """The object with each mesh's kept triangles as triangle lists (`thin`), a mesh whose triangles are its own
    (`whole`: mesh -> its triangles' corners as read) keeping its strips."""
    kept: dict[int, list] = defaultdict(list)
    for t, (mi, flags, _) in zip(tris, meta):
        if t is not None:
            kept[mi].append((flags, tuple(verts[i] for i in t)))
    meshes = []
    for mi, me in enumerate(ob.meshes):
        rows = kept.get(mi, [])
        if [c for _, c in rows] == whole.get(mi, []):
            meshes.append(me)
            continue
        strips = []
        for flags, corners in rows:
            if strips and strips[-1][0] == flags:
                strips[-1][1].extend(corners)
            else:
                strips.append((flags, list(corners)))
        meshes.append(dataclasses.replace(me, strips=tuple(pol.Strip(flags | pol.LIST_FLAG, tuple(e))
                                                                for flags, e in strips)))
    return pol.Object(ob.kind, ob.sphere, tuple(meshes), sum(len(st.entries) for me in meshes for st in me.strips))


def thin(m, keep, protected=frozenset()) -> pol.Model:
    """`m` with each object's triangles thinned to the fraction `keep(object index)` (a number in (0, 1]; at 1 the
    object is returned as it is) by half-edge collapse (Phase 7 Task 7, docs/spike/2026-11-convert.md, "Thinning"):

    - an object's triangles are those of `pol.triangles(m)`; its degenerate ones (`degenerate`, after the transform)
      are dropped and the rest, n, are thinned to at most round(keep * n), or as far as the rules allow;
    - a vertex is a mesh's (position, normal, u, v): one never moves when it belongs to a `protected` triangle (indices
      into `pol.triangles(m)`: build.road_triangles' road surface and the triangles near the road line), lies on its
      mesh's border or a non-manifold edge, or shares its position with another vertex (a texture seam, another mesh
      or object): so a protected triangle is kept with its corners, order and winding, and no hole opens between
      meshes or objects;
    - each collapse moves one vertex onto a neighbour of the same mesh (its triangles keep their corners' own (u, v)
      and normals; the two triangles on the edge go), the one of least quadric error (the moved vertex's area-weighted
      plane quadrics, Garland and Heckbert, at the neighbour's position); allowed when the two share exactly the
      neighbours the edge's triangles have (the surface stays a manifold) and no moved triangle turns further than
      THIN_TURN degrees or flattens;
    - a thinned object's meshes keep their index, texture, parameters and colours; their kept triangles are written
      in their original order as triangle lists (pol.LIST_FLAG), one strip per run of the same strip flags (the cull
      mode kept per triangle); a mesh left whole keeps its strips.

    ConvertError for an `m` that `convert_model` refuses, a `keep` that is not callable or does not give a number in
    (0, 1] for every object (bool refused; what it raises is re-raised as ConvertError), or `protected` that is not a
    set, frozenset, tuple or list of triangle indices of `m`."""
    info = _walk(m)
    if not callable(keep):
        raise ConvertError(f"keep {keep!r:.30} is not callable (object index -> kept fraction)")
    if not isinstance(protected, (set, frozenset, tuple, list)) or not all(
            is_int(i) and 0 <= i < len(info) for i in protected):
        raise ConvertError(f"protected {protected!r:.60} is not a collection of triangle indices into the "
                           f"{len(info)} triangles")
    guard = {int(i) for i in protected}
    try:
        tris = list(pol.triangles(m))
    except DecodeError as err:
        raise ConvertError(f"model: {err}") from err
    fractions = [_keep_of(keep, oi) for oi in range(len(m.objects))]
    owners: dict[tuple, set] = defaultdict(set)         # position -> the (object, mesh, vertex) keys there
    for (a, b, c, oi), inf in zip(tris, info):
        for v in (a, b, c):
            owners[_pos(v)].add((oi, inf[0][1], v))
    objects = list(m.objects)
    for oi, f in enumerate(fractions):
        if f < 1:
            rows = [(k, t, inf) for k, (t, inf) in enumerate(zip(tris, info)) if t[3] == oi]
            objects[oi] = _thin_object(m.objects[oi], f, rows, guard, owners)
    return pol.Model(m.header, tuple(objects), m.textures)


def _thin_object(ob: pol.Object, f: float, rows: list, guard: set, owners: dict) -> pol.Object:
    """One object thinned (`thin`'s rules). `rows`: (index, triangle, _walk info) of its triangles, in order."""
    ids: dict[tuple, int] = {}                          # (mesh, vertex) -> id
    verts: list[pol.Vertex] = []
    tris, meta = [], []                                 # per kept triangle: [ids], (mesh, flags, index)
    locked: set[int] = set()
    whole: dict[int, list] = defaultdict(list)
    for k, t, (key, flags, _, _) in rows:
        mi = key[1]
        whole[mi].append(tuple(t[:3]))
        corners = [_pos(v) for v in t[:3]]
        if degenerate(corners) and k not in guard:
            continue
        tri = []
        for v in t[:3]:
            vid = ids.setdefault((mi, v), len(verts))
            if vid == len(verts):
                verts.append(v)
            tri.append(vid)
        if len(set(tri)) < 3:                           # a protected degenerate one: kept as it is, never touched
            locked.update(tri)
        tris.append(tri)
        meta.append((mi, flags, k))
        if k in guard:
            locked.update(tri)
    n = sum(1 for t in tris if not degenerate([_pos(verts[i]) for i in t]))
    target = round(f * n)
    edges: dict[tuple[int, int], int] = defaultdict(int)
    for t in tris:
        for e in range(3):
            edges[tuple(sorted((t[e], t[(e + 1) % 3])))] += 1
    for (a, b), count in edges.items():
        if count != 2:
            locked.update((a, b))
    for vid, v in enumerate(verts):
        if len(owners[_pos(v)]) > 1:
            locked.add(vid)
    _decimate(tris, verts, locked, target)
    return _rebuild(ob, tris, meta, verts, whole)


# ---- re-meshing (Phase 10, Task 3) ----

REMESH_TURN = 60.0
"""Degrees: a re-mesh collapse may not turn any triangle it moves further than this from its own facing (nor flatten
one), as THIN_TURN."""
WELD = 0.01
"""World units: vertex positions this close are one point of the surface (`remesh`; CS4 has 506 pairs of distinct
positions closer than this, seams whose shared corners differ in the last bits). Fix round 1's ruling C1."""
REMESH_LEAF = 0.25
"""World units: `covered` splits a region no smaller than this; a smaller one is checked at its corners only."""
_REGIONS_MAX = 200000            # regions `covered` keeps at once before it gives up (refuses)


def _closest(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distances from the points x[i] (or one point x) to the triangles (a[i], b[i], c[i]) and the barycentrics of the
    closest points (Ericson, Real-Time Collision Detection 5.1.5, over arrays). Each region's formula is evaluated on
    its own elements only (fix round 2: the same arithmetic per element, a fraction of the work)."""
    def dot(u, v):
        return u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1] + u[:, 2] * v[:, 2]
    ab, ac, ap, bp, cp = b - a, c - a, x - a, x - b, x - c
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    k = len(a)
    bar = np.zeros((k, 3))
    left = np.ones(k, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for mask, col in (((d1 <= 0) & (d2 <= 0), 0), ((d3 >= 0) & (d4 <= d3), 1), ((d6 >= 0) & (d5 <= d6), 2)):
            sel = mask & left
            bar[sel, col] = 1.0
            left &= ~sel
        vc = d1 * d4 - d3 * d2
        sel = np.flatnonzero(left & (vc <= 0) & (d1 >= 0) & (d3 <= 0))
        v = d1[sel] / (d1[sel] - d3[sel])
        bar[sel, 0], bar[sel, 1], bar[sel, 2] = 1 - v, v, 0 * v
        left[sel] = False
        vb = d5 * d2 - d1 * d6
        sel = np.flatnonzero(left & (vb <= 0) & (d2 >= 0) & (d6 <= 0))
        w = d2[sel] / (d2[sel] - d6[sel])
        bar[sel, 0], bar[sel, 1], bar[sel, 2] = 1 - w, 0 * w, w
        left[sel] = False
        va = d3 * d6 - d5 * d4
        sel = np.flatnonzero(left & (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))
        w = (d4[sel] - d3[sel]) / ((d4[sel] - d3[sel]) + (d5[sel] - d6[sel]))
        bar[sel, 0], bar[sel, 1], bar[sel, 2] = 0 * w, 1 - w, w
        left[sel] = False
        sel = np.flatnonzero(left)
        den = 1 / (va[sel] + vb[sel] + vc[sel])
        v, w = vb[sel] * den, vc[sel] * den
        bar[sel, 0], bar[sel, 1], bar[sel, 2] = 1 - v - w, v, w
    bar = np.nan_to_num(bar)
    q = bar[:, :1] * a + bar[:, 1:2] * b + bar[:, 2:] * c
    return np.linalg.norm(q - x, axis=1), bar


def _distances(x: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Distances from the points x (n, 3) to the triangles tri (k, 3, 3): (n, k) (`_closest` over every pair)."""
    n, k = len(x), len(tri)
    out = np.empty((n, k))
    step = max(1, 20000 // max(k, 1))
    for i in range(0, n, step):
        xs = np.repeat(x[i:i + step], k, axis=0)
        t = np.tile(tri, (len(x[i:i + step]), 1, 1))
        out[i:i + step] = _closest(xs, t[:, 0], t[:, 1], t[:, 2])[0].reshape(-1, k)
    return out


def covered(tris, cover, eps, leaf=REMESH_LEAF) -> tuple[bool, float]:
    """Whether every point of the triangles `tris` (n, 3, 3) lies within `eps` of the union of the triangles `cover`
    (k, 3, 3), and the largest corner distance met (a lower bound of the deviation). Each triangle is a region; a region
    whose three corners all lie within `eps` of one triangle of `cover` is covered (the distance to a triangle is
    convex, so its maximum over the region is at a corner: exact); a region with a corner farther than `eps` from all of
    them is not; any other is split in four at its edges' midpoints, down to regions of `leaf` world units, which are
    taken at their corners. ConvertError for arrays that are not (n, 3, 3) finite numbers or an eps or leaf that is not
    a positive finite number."""
    work, cover = np.asarray(tris, dtype=float), np.asarray(cover, dtype=float)
    for name, a in (("tris", work), ("cover", cover)):
        if a.ndim != 3 or a.shape[1:] != (3, 3) or not np.isfinite(a).all():
            raise ConvertError(f"{name} is not an (n, 3, 3) array of finite numbers")
    for name, v in (("eps", eps), ("leaf", leaf)):
        if not is_number(v) or not v > 0:
            raise ConvertError(f"{name} {v!r:.30} is not a positive finite number")
    if not len(work):
        return True, 0.0
    if not len(cover):
        return False, math.inf
    worst = 0.0
    clo, chi = cover.min(axis=1) - eps, cover.max(axis=1) + eps
    while len(work):
        # only the (region, cover triangle) pairs whose boxes come within eps can cover a corner
        wlo, whi = work.min(axis=1), work.max(axis=1)
        ri, ki = np.nonzero(((wlo[:, None] <= chi[None]) & (whi[:, None] >= clo[None])).all(axis=2))
        near = np.full((len(work), 3), np.inf)
        whole = np.zeros(len(work), dtype=bool)
        if len(ri):
            t = np.repeat(cover[ki], 3, axis=0)
            d = _closest(work[ri].reshape(-1, 3), t[:, 0], t[:, 1], t[:, 2])[0].reshape(-1, 3)
            np.minimum.at(near, ri, d)
            whole[ri[d.max(axis=1) <= eps]] = True
        worst = max(worst, float(near.max()))
        if (near > eps).any():
            return False, worst
        open_ = work[~whole]
        edge = np.linalg.norm(open_ - np.roll(open_, -1, axis=1), axis=2).max(axis=1)
        open_ = open_[edge > leaf]
        if len(open_) * 4 > _REGIONS_MAX:
            return False, worst
        a, b, c = open_[:, 0], open_[:, 1], open_[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        work = np.concatenate([np.stack(r, axis=1) for r in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return True, worst


def weld(points, tol=WELD) -> np.ndarray:
    """Per point (n, 3) the index of the first point of its group: points closer than `tol` are one group, and so,
    transitively, are their groups' (union-find over a grid of `tol`). ConvertError for points that are not (n, 3)
    finite numbers or a tol that is not a positive finite number."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1:] != (3,) or not np.isfinite(pts).all():
        raise ConvertError("points are not an (n, 3) array of finite numbers")
    if not is_number(tol) or not tol > 0:
        raise ConvertError(f"tol {tol!r:.30} is not a positive finite number")
    parent = list(range(len(pts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    grid: dict[tuple, list[int]] = defaultdict(list)
    cells = np.floor(pts / tol).astype(np.int64)
    for i, (p, c) in enumerate(zip(pts, cells)):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        if np.linalg.norm(pts[j] - p) <= tol:
                            ri, rj = root(i), root(j)
                            if ri != rj:
                                parent[max(ri, rj)] = min(ri, rj)
        grid[(c[0], c[1], c[2])].append(i)
    return np.array([root(i) for i in range(len(pts))], dtype=np.int64)


def remesh(m, eps) -> pol.Model:
    """`m` with fewer triangles and no surface lost: every mesh's surface within `eps` world units of its original both
    ways (Phase 10 Task 3, the strip re-mesh; fix round 1's ruling C1: National Park Speedway's track is long strips,
    whose cross-sections and lane strips this drops where they are flat enough; docs/spike/2026-11-courses-4-7.md,
    "National Park Speedway"):

    - the surface is every non-degenerate triangle of `pol.triangles(m)`; corners closer than WELD are one point
      (`weld`, across meshes and objects); a vertex is a (object, mesh, point, u, v) (a double-sided triangle's,
      strip modes 0-1, also its normal), so it belongs to one mesh;
    - each step is a half-edge collapse of every vertex at one point onto a neighbouring point (no vertex is made: the
      moved triangles keep their other corners and those corners' (u, v)), so where meshes meet all move together;
      allowed when each moved vertex has exactly one neighbour of its own mesh at the target, the two points share
      exactly the neighbours of the edge's triangles (the surface keeps its topology), no moved triangle turns more
      than REMESH_TURN degrees or flattens, no crack opens (every edge at the target that one triangle uses afterwards,
      all meshes counted, lies within WELD of the original surface's outline), and the deviation stays within `eps`
      both ways, each mesh against itself only (`covered`): every original triangle of a mesh the collapse touches,
      gone and moved triangles' included, within `eps` of the mesh's triangles afterwards, and every moved triangle
      within `eps` of the mesh's original triangles;
    - the collapse of least deviation first (estimated by the moved vertices' distance), until none is allowed;
    - a mesh with a triangle removed or moved is written as triangle lists in its triangles' original order, one strip
      per run of the same strip flags (the cull mode kept per triangle, as `thin`); a mesh left whole keeps its strips;
      object and mesh indices, textures, parameters and colours are kept.

    ConvertError for an `m` that `convert_model` refuses or an `eps` that is not a positive finite number."""
    info = _walk(m)
    if not is_number(eps) or not eps > 0:
        raise ConvertError(f"eps {eps!r:.30} is not a positive finite number")
    eps = float(eps)
    try:
        raw = list(pol.triangles(m))
    except DecodeError as err:
        raise ConvertError(f"model: {err}") from err
    corners: list[list[pol.Vertex]] = []
    meta: list[tuple] = []
    for k, (t, ((oi, mi, _), flags, _, _)) in enumerate(zip(raw, info)):
        if not degenerate([_pos(v) for v in t[:3]]):
            corners.append(list(t[:3]))
            meta.append((oi, mi, flags, k))
    if not corners:
        return m
    flat = np.array([_pos(v) for cs in corners for v in cs], dtype=float)
    node_of = weld(flat)
    ids: dict[tuple, int] = {}
    verts: list[pol.Vertex] = []
    vnode: list[int] = []
    tris: list[list[int]] = []
    for t, (cs, (oi, mi, flags, _)) in enumerate(zip(corners, meta)):
        row = []
        for j, v in enumerate(cs):
            node = int(node_of[3 * t + j])
            key = (oi, mi, node, float(v.u), float(v.v)) + ((tuple(map(float, v.normal)),) if flags & 3 < 2 else ())
            if key not in ids:
                ids[key] = len(verts)
                verts.append(v)
                vnode.append(node)
            row.append(ids[key])
        tris.append(row)
    geom = flat.reshape(-1, 3, 3).copy()                  # per triangle: its corners as written
    mesh_of = [(oi, mi) for oi, mi, _, _ in meta]
    original: dict[tuple, np.ndarray] = defaultdict(list)
    for t, key in enumerate(mesh_of):
        original[key].append(geom[t])
    original = {k: np.array(v) for k, v in original.items()}
    tri_of: dict[tuple, list] = defaultdict(list)         # per mesh: its original triangles' indices, in order
    for t, key in enumerate(mesh_of):
        tri_of[key].append(t)
    outward: dict[tuple, tuple] = {}

    def intact(key, i) -> bool:
        t = tri_of[key][i]
        return alive[t] and not changed[t]
    obox = {k: (v.min(axis=1), v.max(axis=1)) for k, v in original.items()}
    current: dict[tuple, set] = defaultdict(set)
    for t, key in enumerate(mesh_of):
        current[key].add(t)
    alive = [True] * len(tris)
    changed = [False] * len(tris)
    vt: dict[int, set] = defaultdict(set)
    at: dict[int, set] = defaultdict(set)
    for t, row in enumerate(tris):
        for a in row:
            vt[a].add(t)
    for a, node in enumerate(vnode):
        at[node].add(a)
    count: dict[tuple, int] = defaultdict(int)
    for row in tris:
        for j in range(3):
            count[tuple(sorted((vnode[row[j]], vnode[row[(j + 1) % 3]])))] += 1
    rim = {e for e, n in count.items() if n == 1}
    segs = np.array([[geom[t][j], geom[t][(j + 1) % 3]] for t, row in enumerate(tris) for j in range(3)
                     if tuple(sorted((vnode[row[j]], vnode[row[(j + 1) % 3]]))) in rim]).reshape(-1, 2, 3)
    cos_turn = math.cos(math.radians(REMESH_TURN))

    def around(p) -> set:
        return set().union(*(vt[a] for a in at[p])) if at[p] else set()

    def neighbours(p) -> set:
        return {vnode[a] for t in around(p) for a in tris[t]} - {p}

    def on_rim(a, b) -> bool:
        if not len(segs):
            return False
        pts = a[None] + np.linspace(0.0, 1.0, 9)[:, None] * (b - a)[None]
        lo, hi = pts.min(axis=0) - WELD, pts.max(axis=0) + WELD
        sel = segs[((np.minimum(segs[:, 0], segs[:, 1]) <= hi) & (np.maximum(segs[:, 0], segs[:, 1]) >= lo)).all(1)]
        if not len(sel):
            return False
        e = sel[:, 1] - sel[:, 0]
        s = np.clip(((pts[:, None] - sel[None, :, 0]) * e[None]).sum(2) / np.maximum((e * e).sum(1), 1e-30)[None],
                    0.0, 1.0)
        return bool((np.linalg.norm(pts[:, None] - (sel[None, :, 0] + s[..., None] * e[None]), axis=2).min(1)
                     <= WELD).all())

    def in_box(arr, lo, hi) -> np.ndarray:
        return ((arr.min(axis=1) <= hi) & (arr.max(axis=1) >= lo)).all(axis=1)

    def evaluate(p, q, full: bool):
        tp = around(p)
        gone = {t for t in tp if any(vnode[a] == q for a in tris[t])}
        if not gone:
            return None
        onto = {}
        for a in at[p]:
            if not vt[a]:
                continue
            targets = {b for t in vt[a] for b in tris[t] if vnode[b] == q}
            if len(targets) != 1:
                return None
            onto[a] = next(iter(targets))
        third = {vnode[a] for t in gone for a in tris[t]} - {p, q}
        if neighbours(p) & neighbours(q) != third:
            return None
        moved = {}
        for t in tp - gone:
            row = [onto.get(a, a) for a in tris[t]]
            g = np.array([geom[t][j] if row[j] == tris[t][j] else _pos(verts[row[j]]) for j in range(3)])
            n0 = np.cross(geom[t][1] - geom[t][0], geom[t][2] - geom[t][0])
            n1 = np.cross(g[1] - g[0], g[2] - g[0])
            l0, l1 = np.linalg.norm(n0), np.linalg.norm(n1)
            if l1 <= 1e-9 * max(l0, 1e-30) or n0 @ n1 < cos_turn * l0 * l1:
                return None
            moved[t] = (row, g)
        after = (around(q) - gone) | set(moved)
        edges: dict[tuple, list] = defaultdict(list)
        for t in after:
            row, g = moved.get(t, (tris[t], geom[t]))
            for j in range(3):
                e = tuple(sorted((vnode[row[j]], vnode[row[(j + 1) % 3]])))
                if q in e:
                    edges[e].append((g[j], g[(j + 1) % 3]))
        for e, uses in edges.items():
            if len(uses) == 1 and e not in rim and not on_rim(*uses[0]):
                return None                               # a crack
        # estimate: the moved vertices' distance to their mesh's triangles at the target
        cost = 0.0
        for a in onto:
            mine = [moved.get(t, (tris[t], geom[t]))[1] for t in after if mesh_of[t] == mesh_of[next(iter(vt[a]))]]
            if not mine:
                return None
            cost = max(cost, float(_distances(np.array([_pos(verts[a])]), np.array(mine)).min()))
            if cost > eps:
                return None
        if not full:
            return cost, gone, moved
        for t, (row, g) in moved.items():
            key = mesh_of[t], tuple(row)
            if key not in outward:                        # a triangle's distance to its mesh's original: fixed
                lo, hi = g.min(axis=0) - eps, g.max(axis=0) + eps
                outward[key] = covered(g[None], original[key[0]][in_box(original[key[0]], lo, hi)], eps)
            ok, w = outward[key]
            if not ok:
                return None
            cost = max(cost, w)
        for key in {mesh_of[t] for t in tp}:
            g = np.array([geom[t] for t in tp if mesh_of[t] == key])
            lo, hi = g.min(axis=(0, 1)) - eps, g.max(axis=(0, 1)) + eps
            olo, ohi = obox[key]
            near = np.flatnonzero(((olo <= hi) & (ohi >= lo)).all(axis=1))
            near = near[[not intact(key, i) or tri_of[key][i] in gone or tri_of[key][i] in moved for i in near]]
            if not len(near):
                continue                                  # every original there is still drawn as it was
            src = original[key][near]
            lo, hi = src.min(axis=(0, 1)) - eps, src.max(axis=(0, 1)) + eps
            dst = [moved[t][1] if t in moved else geom[t] for t in current[key] - gone]
            dst = np.array(dst).reshape(-1, 3, 3)
            ok, w = covered(src, dst[in_box(dst, lo, hi)], eps)
            if not ok:
                return None
            cost = max(cost, w)
        return cost, gone, moved

    version: dict[int, int] = defaultdict(int)
    heap: list = []

    def push(p, full_cost=None):
        for q in neighbours(p):
            for a, b in ((p, q), (q, p)):
                r = evaluate(a, b, False)
                if r is not None:
                    heapq.heappush(heap, (r[0], 0, version[a], version[b], a, b))

    for p in sorted(at):
        for q in sorted(neighbours(p)):
            r = evaluate(p, q, False)
            if r is not None:
                heapq.heappush(heap, (r[0], 0, 0, 0, p, q))
    while heap:
        cost, full, vp, vq, p, q = heapq.heappop(heap)
        if vp != version[p] or vq != version[q] or not at[p]:
            continue
        r = evaluate(p, q, True)
        if r is None:
            continue
        if not full:
            heapq.heappush(heap, (r[0], 1, vp, vq, p, q))
            continue
        _, gone, moved = r
        for t in gone:
            alive[t] = False
            current[mesh_of[t]].discard(t)
            for a in tris[t]:
                vt[a].discard(t)
        for t, (row, g) in moved.items():
            for a in tris[t]:
                vt[a].discard(t)
            corners[t] = [verts[b] if a != b else v for a, b, v in zip(tris[t], row, corners[t])]
            tris[t] = row
            geom[t] = g
            changed[t] = True
            for a in row:
                vt[a].add(t)
        at[p] = set()
        touched = {q} | neighbours(q)
        for x in touched:
            version[x] += 1
        for x in sorted(touched):
            push(x)
    rows: dict[tuple, list] = defaultdict(list)
    dirty = set()
    for t, (oi, mi, flags, _) in enumerate(meta):
        if alive[t]:
            rows[(oi, mi)].append((flags, corners[t]))
        if not alive[t] or changed[t]:
            dirty.add((oi, mi))
    objects = []
    for oi, ob in enumerate(m.objects):
        if not any((oi, mi) in dirty for mi in range(len(ob.meshes))):
            objects.append(ob)
            continue
        meshes = []
        for mi, me in enumerate(ob.meshes):
            if (oi, mi) not in dirty:
                meshes.append(me)
                continue
            strips: list = []
            for flags, cs in rows.get((oi, mi), []):
                if strips and strips[-1][0] == flags:
                    strips[-1][1].extend(cs)
                else:
                    strips.append((flags, list(cs)))
            meshes.append(dataclasses.replace(me, strips=tuple(pol.Strip(flags | pol.LIST_FLAG, tuple(e))
                                                                    for flags, e in strips)))
        objects.append(pol.Object(ob.kind, ob.sphere, tuple(meshes),
                                  sum(len(st.entries) for me in meshes for st in me.strips)))
    return pol.Model(m.header, tuple(objects), m.textures)
