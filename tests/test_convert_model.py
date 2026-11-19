"""Tests for the course model converted into Model 2 polygon objects (dtools/convert/model.py).

Review Focus 1 (a polygon drawn twice or not at all): every Dreamcast triangle of the course model lands in exactly one
arcade polygon of exactly one scenery cell's object, counted by decoding the written objects."""

import dataclasses
import math
import random
import struct

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import polyrom as P
from dtools.convert import ConvertError, model
from dtools.dc import pol
from dtools.dc.pol import Mesh, Model, Object, Strip, TexEntry, Vertex
from tests.test_convert_build import _beginner, _romset
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

UP = (0.0, 1.0, 0.0)
LIST = pol.LIST_FLAG
LINK_BITS = 0x3 << 8 | 0x3 << 23


def V(x, y, z, u=0.0, v=0.0, n=UP) -> Vertex:
    return Vertex(float(x), float(y), float(z), tuple(n), float(u), float(v))


def mesh(strips, texture=-1, params=(0x8000002C, 0x83000000, 0x2008216D, 0xCC000000)) -> Mesh:
    return Mesh(params, (0.0, 0.0, 0.0, 1.0), texture, 0, (1.0,) * 9, tuple(strips))


def make_model(meshes, textures=(TexEntry(64, 64, 0x101, 0, 0), TexEntry(32, 32, 0x101, 0, 0))) -> Model:
    total = sum(len(s.entries) for me in meshes for s in me.strips)
    obj = Object(3, (0.0, 0.0, 0.0, 1000.0), tuple(meshes), total)
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    return pol.fix_header(Model(hdr, (obj,), tuple(textures)))


def synthetic() -> Model:
    """In the Dreamcast's frame, every vertex normal on the side the face is seen from (as the real files):
    - a strip (cull mode 2: clockwise about the normal), a ribbon along +z whose heights zig-zag (no two of its
      triangles coplanar), textured (texture 0, 64 x 64); all but one triangle in cell 0x78 (arcade z = -z < 0);
    - a fan as a triangle list (cull mode 3: anticlockwise about the normal), its rim heights alternating;
    - two disjoint triangles (a list, cull mode 2), the second far away in another cell (arcade z = -Dreamcast z);
    - a flat strip (cull mode 1: double-sided) whose UVs are affine in the plane: its four triangles become two quads
      (texture 1, 32 x 32)."""
    ys = (0.0, 0.3, 0.1, 0.5, 0.0, 0.9)
    ribbon = Strip(0x12, tuple(V(2 + (k & 1), ys[k], 3 + k // 2, u=k & 1, v=(k // 2) / 4) for k in range(6)))
    centre = V(10, 0, 10)
    rim = [V(10 + 3 * math.cos(-a), (k & 1) * 1.0, 10 + 3 * math.sin(-a)) for k, a in
           enumerate(math.radians(60 * j) for j in range(6))]
    fan = Strip(LIST | 3, tuple(v for k in range(5) for v in (centre, rim[k], rim[k + 1])))
    lone = Strip(LIST | 2, (V(20, 0, 20), V(21, 0, 20), V(20, 0, 21),
                            V(300, 1, -300), V(301, 1, -300), V(300, 1, -299)))
    flat = Strip(0x11, tuple(V(5 + (k & 1), 5.0, 5 + k // 2, u=(5 + (k & 1)) / 2, v=(5 + k // 2) / 2)
                             for k in range(6)))
    return make_model([mesh([ribbon], texture=0), mesh([fan]), mesh([lone]), mesh([flat], texture=1)])


def arcade(v: Vertex) -> tuple:
    """The arcade position of a Dreamcast vertex: single precision, z negated."""
    x, y, z = struct.unpack("<3f", struct.pack("<3f", v.x, v.y, v.z))
    return (x, y, 0.0 - z)


def newell(points) -> np.ndarray:
    p = np.array(points, dtype=float)
    return sum(np.cross(p[k], p[(k + 1) % len(p)]) for k in range(len(p)))


def canon(tri) -> tuple:
    """A triangle's corners rotated to start at the least: equal for the same winding, whatever the start."""
    k = min(range(3), key=lambda i: tri[i])
    return tuple(tri[k:]) + tuple(tri[:k])


def split(points, n: int) -> list[list[tuple]]:
    """The triangles of a drawn polygon: itself, or a quad's two along either diagonal (both candidates)."""
    if n == 1:
        return [[canon(points)]]
    q = points
    return [[canon((q[0], q[1], q[2])), canon((q[0], q[2], q[3]))],
            [canon((q[1], q[2], q[3])), canon((q[1], q[3], q[0]))]]


def words_of(obj: P.Object) -> bytes:
    return struct.pack(f"<{obj.word_count}I", *P.encode_object(obj))


def check_pieces(m: Model, pieces) -> dict:
    """Review Focus 1 on a converted model, by decoding the written objects: every polygon a record draws is its
    source triangles (z negated, the same corners, the arcade's winding: corners anticlockwise about the face's
    outside, the stored normal the inside), every non-degenerate triangle in exactly one drawn record of one cell,
    every object within MAX_RECORDS and its polygons' centroids in its cell. Returns counts."""
    tris = list(pol.triangles(m))
    meta = tri_meta(m)
    owner: dict[int, int] = {}
    counts = dict(records=0, drawn=0, quads=0, moves=0, pieces=0, words=0)
    objects = set()
    for cell, ps in pieces.items():
        assert isinstance(cell, int) and 0 <= cell < 256
        for pc in ps:
            assert id(pc.obj) not in objects                # no object in two cells
            objects.add(id(pc.obj))
            obj = P.decode_object(words_of(pc.obj), 0, mode=1)
            assert obj == pc.obj
            assert 0 < len(obj.records) <= model.MAX_RECORDS
            assert len(pc.textures) == len(pc.uvs) == len(pc.sources) == len(obj.records)
            counts["pieces"] += 1
            counts["records"] += len(obj.records)
            counts["words"] += obj.word_count
            for poly, rec, src, tex, uvs in zip(P.polygons(obj), obj.records, pc.sources, pc.textures, pc.uvs):
                if not poly.drawn:
                    assert rec.attr == model.MOVE_ATTR and src == () and uvs is None and tex == -1
                    counts["moves"] += 1
                    continue
                counts["drawn"] += 1
                counts["quads"] += rec.is_quad
                assert len(src) == (2 if rec.is_quad else 1)
                for i in src:
                    assert i not in owner, f"triangle {i} drawn by cells {owner[i]} and {cell}"
                    owner[i] = cell
                want = sorted(canon(seen_tri(tris[i], meta[i])) for i in src)
                got = [sorted(c) for c in split(poly.points, len(src))]
                assert want in got, (cell, src)             # the winding the strip's cull mode implies
                assert {(rec.attr >> 17) & 1} == {int(meta[i][0] & 3 < 2) for i in src}
                face = newell(poly.points)
                n = np.array(rec.normal)
                assert abs(np.linalg.norm(n) - 1) < 1e-4
                assert np.allclose(n, -face / np.linalg.norm(face), atol=1e-4)
                for i in src:                     # each triangle's seen side, against the stored normal
                    t = newell(seen_tri(tris[i], meta[i]))
                    assert t @ n / np.linalg.norm(t) < -(1 - 1e-4)
                pts = np.array(poly.points)
                assert C.cell_of(pts[:, 0].mean(), pts[:, 2].mean()) == cell == model.cell_of(*pts[:, [0, 2]].mean(0))
                assert (pts[:, [0, 2]] >= -1024).all() and (pts[:, [0, 2]] <= 1023).all()
                assert ((rec.attr >> 8) & 3) == ((rec.attr >> 23) & 3)
                assert (rec.attr >> 12) & 0x1F == 1 and (rec.attr >> 10) & 3 == 2 and rec.tex_param == 2
                # uvs: each corner's Dreamcast (u, v), in the rasterizer's corner order
                verts = {arcade(v): (v.u, v.v) for i in src for v in tris[i][:3]}
                assert uvs == tuple(verts[p] for p in poly.points)
                assert tex in {meta[i][2] for i in src}
    counts["owned"] = len(owner)
    return counts


def tri_meta(m: Model) -> list[tuple[int, bool, int]]:
    """Per triangle of pol.triangles(m): (its strip's flags, raw normals in its mesh, its mesh's texture)."""
    out = []
    for ob in m.objects:
        for me in ob.meshes:
            raw = (me.params[0] & 0x30) == 0x10
            for st in me.strips:
                n = len(st.entries) // 3 if st.flags & LIST else max(len(st.entries) - 2, 0)
                out += [(st.flags, raw, me.texture)] * n
    return out


def stored_normal(v: Vertex, raw: bool) -> np.ndarray:
    """A vertex's stored normal in the arcade's frame (z negated); a raw one's signed bytes of its first word."""
    n = struct.unpack("3b", struct.pack(">I", v.normal[0])[1:]) if raw else v.normal
    return np.array((n[0], n[1], -n[2]), dtype=float)


def seen_tri(tri, meta) -> tuple:
    """A triangle's arcade corners wound as the strip's cull mode implies (anticlockwise about the side seen):
    mode 2 kept, mode 3 reversed, modes 0-1 (double-sided) facing the side its stored normals point to."""
    flags, raw, _ = meta
    pts = tuple(arcade(v) for v in tri[:3])
    rev = (pts[0], pts[2], pts[1])
    mode = flags & 3
    if mode == 2:
        return pts
    if mode == 3:
        return rev
    seen = sum(stored_normal(v, raw) for v in tri[:3])
    return pts if newell(pts) @ seen >= 0 else rev


# ---- the cell rule ----

def test_cell_of():
    """The game's 0x172a0: round to nearest (even), + 1024, >> 7, row << 4 | col (row from z); refused outside the
    world -1024..1023, or for a value that is not a finite number."""
    assert model.CELL == 128
    assert model.cell_of(-1024, -1024) == 0
    assert model.cell_of(1023, 1023) == 255
    assert model.cell_of(-896.5, -1024) == 1            # rounds to -896 (half to even): column 1
    assert model.cell_of(-897.5, -1024) == 0            # rounds to -898: column 0
    assert model.cell_of(-896.51, -1024) == 0
    assert model.cell_of(0.0, -1024) == 8
    assert model.cell_of(-1024, 0.0) == 0x80
    assert model.cell_of(np.float32(300.0), np.int64(-300)) == C.cell_of(300.0, -300.0) == 0x5A
    rng = random.Random(5)
    for _ in range(500):
        x, z = rng.uniform(-1024, 1023), rng.uniform(-1024, 1023)
        assert model.cell_of(x, z) == C.cell_of(x, z)
    for x, z in ((1023.5, 0), (0, -1024.5), (float("nan"), 0), (0, float("inf")), (True, 0), (0, None), ("1", 0),
                 (10 ** 400, 0)):
        with pytest.raises(ConvertError, match="cell"):
            model.cell_of(x, z)


# ---- synthetic ----

def test_model_round_trip_synthetic():
    """A strip, a fan and two disjoint triangles (and a flat double-sided strip): polygons(decode_object(
    encode_object(o))) gives back the input triangles, z-negated, in the arcade's winding (Beginner's: the corners
    anticlockwise about the side the face is seen from, the stored normal the other way), normals within 1e-4; one
    link-0 move record per break in the chain; the flat strip's four triangles drawn as two double-sided quads."""
    m = synthetic()
    pieces = model.convert_model(m)
    assert sorted(pieces) == [model.cell_of(10, -10), model.cell_of(300, 300)] == [0x78, 0xAA]
    counts = check_pieces(m, pieces)
    tris = list(pol.triangles(m))
    assert counts["owned"] == len(tris) == 4 + 5 + 2 + 4
    assert (counts["drawn"], counts["quads"], counts["moves"], counts["pieces"]) == (13, 2, 3, 2)
    # the face is seen from the Dreamcast normal's side (z negated): Newell of the corners along it, normal against it
    for ps in pieces.values():
        for pc in ps:
            for poly, rec, src in zip(P.polygons(pc.obj), pc.obj.records, pc.sources):
                for i in src:
                    n = tris[i][0].normal
                    seen = np.array((n[0], n[1], -n[2]))
                    assert newell(poly.points) @ seen > 0 and np.array(rec.normal) @ seen < 0
    # links: one move per chain break; joined triangles on links 1 and 3; the quads double-sided, the rest not
    [pc] = pieces[0x78]
    recs = pc.obj.records
    assert [r.link for r in recs].count(0) == 3
    for r, nxt in zip(recs, recs[1:]):
        if r.link and nxt.link and not r.is_quad:
            assert r.link in (1, 3)
    assert {(r.attr >> 17) & 1 for r in recs if r.is_quad and r.link} == {1}
    assert {(r.attr >> 17) & 1 for r in recs if not r.is_quad} == {0}
    # untextured polygons carry coefficient 0; the textured ones floor(3 log2(texels per unit))
    for pc in pieces[0x78]:
        for rec, src, tex in zip(pc.obj.records, pc.sources, pc.textures):
            if not src:
                continue
            if tex < 0:
                assert rec.coef == 0
            elif rec.is_quad:                       # 32 x 32 texture, u = x / 2, v = z / 2: 16 texels per unit
                assert rec.coef == math.floor(3 * math.log2(16)) == 12
            else:                                   # 64 x 64, u = x, v = z / 4: 64 x 16 texels on a slanted unit
                assert rec.coef == model.lod_coef(rec_density(src, m)) in (13, 14)


def rec_density(src, m) -> float:
    """Texels per world unit of the triangles `src` of `m`: sqrt(texel area / world area)."""
    tris = list(pol.triangles(m))
    tex = m.textures[tri_meta(m)[src[0]][2]]
    world = texel = 0.0
    for i in src:
        a, b, c = (np.array(arcade(v)) for v in tris[i][:3])
        world += np.linalg.norm(np.cross(b - a, c - a)) / 2
        (ua, va), (ub, vb), (uc, vc) = ((v.u * tex.width, v.v * tex.height) for v in tris[i][:3])
        texel += abs((ub - ua) * (vc - va) - (vb - va) * (uc - ua)) / 2
    return math.sqrt(texel / world)


def test_model_lod_coef():
    """floor(3 log2(texels per world unit)), clamped to the 5 bits 0-31: Beginner's rule (docs/spike, "Model")."""
    assert [model.lod_coef(d) for d in (1.0, 2.0, 8.0, 32.0, 0.5, 2 ** 11, 10.67)] == [0, 3, 9, 15, 0, 31, 10]
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, None, "2"):
        with pytest.raises(ConvertError, match="density"):
            model.lod_coef(bad)


def test_model_split_by_max_records(monkeypatch):
    """A cell with more records than MAX_RECORDS is split into several objects, each starting from its seeds (no
    move record), together drawing every triangle once."""
    monkeypatch.setattr(model, "MAX_RECORDS", 4)
    m = synthetic()
    pieces = model.convert_model(m)
    counts = check_pieces(m, pieces)
    assert counts["owned"] == 15
    assert len(pieces[0x78]) >= 3 and all(len(pc.obj.records) <= 4 for pc in pieces[0x78])
    assert all(pc.obj.records[0].link != 0 for pc in pieces[0x78])


def test_model_drops_degenerate():
    """A triangle with two equal corners, or three collinear ones, is not drawn (and not in any object)."""
    m = make_model([mesh([Strip(LIST | 2, (V(0, 0, 0), V(1, 0, 0), V(0, 0, 1),
                                           V(0, 0, 0), V(0, 0, 0), V(1, 0, 0),
                                           V(0, 0, 0), V(1, 0, 0), V(2, 0, 0)))])])
    pieces = model.convert_model(m)
    assert [pc.sources for pc in pieces[0x88]] == [((0,),)]
    assert model.convert_model(make_model([mesh([Strip(LIST | 2, (V(0, 0, 0),) * 3)])])) == {}


def test_drop_textures():
    """drop_textures removes every mesh drawn with one of the named textures (Task 5 ruling 2: the alpha-blended
    decals); the other meshes, their triangles (in pol.triangles order) and the texture table are unchanged, and each
    object's strip-entry count is its kept meshes'. triangle_meshes names each triangle's (object, mesh) in
    pol.triangles order. Refused: junk, bool, a texture outside the table, -1 (no texture)."""
    m = synthetic()                     # ribbon (texture 0, 4 triangles), fan (5), lone (2), flat (texture 1, 4)
    tris = list(pol.triangles(m))
    assert model.triangle_meshes(m) == ((0, 0),) * 4 + ((0, 1),) * 5 + ((0, 2),) * 2 + ((0, 3),) * 4
    kept = model.drop_textures(m, (0,))
    assert [me.texture for me in kept.objects[0].meshes] == [-1, -1, 1]
    assert kept.objects[0].meshes == m.objects[0].meshes[1:] and kept.textures == m.textures
    assert kept.objects[0].strip_count == sum(len(st.entries) for me in kept.objects[0].meshes for st in me.strips)
    assert list(pol.triangles(kept)) == tris[4:]
    assert model.triangle_meshes(kept) == ((0, 0),) * 5 + ((0, 1),) * 2 + ((0, 2),) * 4
    assert model.drop_textures(m, ()) == m and model.drop_textures(m, [1, 0]) == model.drop_textures(m, (0, 1))
    assert [me.texture for me in model.drop_textures(m, (0, 1)).objects[0].meshes] == [-1, -1]
    assert check_pieces(kept, model.convert_model(kept))["owned"] == 11
    for bad in ((True,), (2,), (-1,), ("0",), None, 0, (0.0,), (np.int64(0), None)):
        with pytest.raises(ConvertError, match="texture"):
            model.drop_textures(m, bad)
    for bad in (None, "m", dataclasses.replace(m, objects=None), dataclasses.replace(m, objects=(None,)),
                dataclasses.replace(m, textures=None)):
        with pytest.raises(ConvertError):
            model.drop_textures(bad, (0,))
        with pytest.raises(ConvertError):
            model.triangle_meshes(bad)


def test_drop_meshes():
    """drop_meshes removes the named (object, mesh) meshes and keeps the rest as drop_textures does (Task 6: the sky
    domes and the shadow volumes); an object whose every mesh goes stays, with none, so object indices do not move.
    Refused: junk, bool indices, a pair naming no mesh."""
    m = synthetic()
    tris = list(pol.triangles(m))
    kept = model.drop_meshes(m, [(0, 1), (0, 3)])
    assert kept.objects[0].meshes == (m.objects[0].meshes[0], m.objects[0].meshes[2]) and kept.textures == m.textures
    assert kept.objects[0].strip_count == sum(len(st.entries) for me in kept.objects[0].meshes for st in me.strips)
    assert list(pol.triangles(kept)) == tris[:4] + tris[9:11]
    assert model.drop_meshes(m, ()) == m and model.drop_meshes(m, ((0, 3), (0, 1))) == kept
    assert model.drop_meshes(m, [(0, 0)]) == model.drop_textures(m, (0,))
    two = dataclasses.replace(m, objects=m.objects * 2)
    gone = model.drop_meshes(two, [(0, k) for k in range(4)])
    assert len(gone.objects) == 2 and gone.objects[0].meshes == () and gone.objects[0].strip_count == 0
    assert [t[:3] for t in pol.triangles(gone)] == [t[:3] for t in tris]
    assert model.triangle_meshes(gone) == tuple((1, k) for _, k in model.triangle_meshes(m))
    for bad in (None, 0, [(0,)], [(0, 4)], [(1, 0)], [(-1, 0)], [(True, 0)], [(0, 1.0)], [[0, 1]], "01"):
        with pytest.raises(ConvertError, match="meshes"):
            model.drop_meshes(m, bad)
    with pytest.raises(ConvertError):
        model.drop_meshes(None, [(0, 0)])


# ---- Beginner ----

def _beginner_records(rs, c):
    """(record, its polygon, its texture header, its (u, v) per corner in texels) of Beginner's drawn, textured cell
    records, from each cell placement (texture points 2 words per vertex, v then u in 1/8 texel, tpa advancing over
    every record, culled ones too)."""
    from dtools.arcade import texture as T
    tw = np.frombuffer(rs.textures, "<u2")
    seen = set()
    for ptr in c.cells:
        for pl in c.object_lists[ptr]:
            if (pl.oba, pl.tpa) in seen:
                continue
            seen.add((pl.oba, pl.tpa))
            o = c.objects[pl.oba & 0x7FFFFF]
            t = pl.tpa
            for rec, poly, h in zip(o.records, P.polygons(o), T.object_headers(rs, pl.tha, o)):
                n = 4 if rec.is_quad else 3
                w = tw[t:t + 2 * n]
                t += 2 * n
                if poly.drawn and h.textured:
                    yield rec, poly, h, tuple((w[2 * k + 1] / 8, w[2 * k] / 8) for k in range(n))


def _planar_affine(poly, uv) -> bool:
    q = np.array(poly.points, dtype=float)
    if len(q) == 3:
        return True
    n1, n2 = np.cross(q[1] - q[0], q[2] - q[0]), np.cross(q[2] - q[0], q[3] - q[0])
    if n1 @ n2 / np.linalg.norm(n1) / np.linalg.norm(n2) < math.cos(math.radians(0.1)):
        return False
    a = np.array([q[1] - q[0], q[2] - q[0]]).T
    st = np.linalg.lstsq(a, q[3] - q[0], rcond=None)[0]
    u = np.array(uv)
    return np.linalg.norm(u[0] + st[0] * (u[1] - u[0]) + st[1] * (u[2] - u[0]) - u[3]) < 0.01


def test_model_beginner_style():
    """The attribute bits, link use and winding of a converted Beginner-like polygon equal those of a real Beginner
    record of the same kind: a Beginner record's polygon (planar, its UVs affine), turned back into the Dreamcast's
    two triangles (z negated, normal on the side it is seen from, cull mode 2, or 1 when double-sided), converts to
    one record with Beginner's attribute (a lone polygon ends with link 2, as Beginner's quads in a strip do), the
    same corners in the same cyclic order, its normal within 1e-4 and its texture points' (u, v) per corner."""
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    c = _beginner()
    want = {(True, 0): 0, (True, 1): 0, (False, 0): 0}
    for rec, poly, h, uv in _beginner_records(rs, c):
        kind = (rec.is_quad, (rec.attr >> 17) & 1)
        if want.get(kind, 9) >= 3 or rec.tex_param != 2 or (rec.attr >> 10) & 3 != 2:
            continue
        if rec.is_quad and rec.link != 2:
            continue
        if not _planar_affine(poly, uv):
            continue
        q = poly.points
        world = sum(np.linalg.norm(np.cross(np.subtract(q[k], q[0]), np.subtract(q[k + 1], q[0]))) / 2
                    for k in range(1, len(q) - 1))
        texel = sum(abs((uv[k][0] - uv[0][0]) * (uv[k + 1][1] - uv[0][1])
                        - (uv[k][1] - uv[0][1]) * (uv[k + 1][0] - uv[0][0])) / 2 for k in range(1, len(q) - 1))
        f = 3 * math.log2(math.sqrt(texel / world))
        if not 0.2 < f - math.floor(f) < 0.8 or math.floor(f) != rec.coef:
            continue                                # Beginner's coefficient off the rule (8 %), or near a step
        want[kind] += 1
        nx, ny, nz = rec.normal
        n_dc = (-nx, -ny, nz)                       # the visible side (against the stored normal), z negated
        verts = [V(p[0], p[1], 0.0 - p[2], u=u / h.width, v=v / h.height, n=n_dc) for p, (u, v) in zip(q, uv)]
        tris = [verts[0], verts[1], verts[2]] + ([verts[0], verts[2], verts[3]] if rec.is_quad else [])
        flags = LIST | (1 if kind[1] else 2)
        m = make_model([mesh([Strip(flags, tuple(tris))], texture=0)], (TexEntry(h.width, h.height, 0x101, 0, 0),))
        pieces = model.convert_model(m)
        [cell] = pieces
        [pc] = pieces[cell]
        [got] = pc.obj.records
        [gp] = P.polygons(pc.obj)
        if rec.is_quad:
            assert got.attr == rec.attr, (hex(got.attr), hex(rec.attr))
        else:                                       # a lone triangle ends on link 2, Beginner's on its strip's link
            assert got.attr & ~LINK_BITS == rec.attr & ~LINK_BITS and got.link == 2, (hex(got.attr), hex(rec.attr))
        k = gp.points.index(q[0])
        assert gp.points[k:] + gp.points[:k] == q                       # the same corners, the same winding
        assert np.allclose(got.normal, rec.normal, atol=1e-4)
        uvs = dict(zip(q, ((u / h.width, v / h.height) for u, v in uv)))
        assert pc.uvs == (tuple(uvs[p] for p in gp.points),)
        assert pc.textures == (0,)
    assert want == {(True, 0): 3, (True, 1): 3, (False, 0): 3}


# ---- thinning (Task 7) ----

def terrain(n: int = 16, x0: float = 0.0, texture: int = 0) -> Object:
    """An n x n vertex height field (Dreamcast frame, one unit apart from (x0, 0)), smooth rolling heights, as one
    textured mesh of row strips (cull mode 2, normals up; u, v = x / 8, z / 8), the strips' triangles alternating
    as pol.triangles reads them."""
    def h(i, j):
        return 2.0 * math.sin(i / 3.0) * math.cos(j / 4.0) + 0.05 * i * j / n

    strips = []
    for j in range(n - 1):
        ents = []
        for i in range(n):
            for jj in (j + 1, j):
                ents.append(V(x0 + i, h(i, jj), jj, u=(x0 + i) / 8, v=jj / 8))
        strips.append(Strip(0x12, tuple(ents)))
    me = mesh(strips, texture=texture)
    return Object(1, (x0 + n / 2, 0.0, n / 2, float(n)), (me,), sum(len(s.entries) for s in strips))


def two_terrains() -> Model:
    objs = (terrain(16), terrain(8, 40.0, texture=1))
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    return pol.fix_header(Model(hdr, objs, (TexEntry(64, 64, 0x101, 0, 0), TexEntry(32, 32, 0x101, 0, 0))))


def _tri_key(t) -> tuple:
    return tuple((v.x, v.y, v.z, v.u, v.v) for v in t[:3])


def test_thin_keeps_road():
    """model.thin (Task 7): per object the kept fraction of its triangles by half-edge collapse (every kept corner an
    original vertex), at most round(keep * n) and at most one under it; the protected triangles (the road surface and
    those near the road line, build.road_triangles) kept as they were, corners, order and winding; the mesh's border
    kept (no hole); no triangle turned over; an object kept at 1 (and keep 1 everywhere) unchanged; the thinned model
    converts (Review Focus 1 on it, check_pieces)."""
    m = two_terrains()
    assert model.thin(m, lambda oi: 1.0) == m
    tris = list(pol.triangles(m))
    first = [i for i, t in enumerate(tris) if t[3] == 0]
    road = {i for i in first if 5.0 <= np.mean([v.z for v in tris[i][:3]]) <= 7.0}     # a band across the field
    assert len(first) == 450 and len(road) == 60
    thinned = model.thin(m, lambda oi: 0.5 if oi == 0 else 1.0, protected=road)
    assert thinned.objects[1] == m.objects[1]
    out = [t for t in pol.triangles(thinned) if t[3] == 0]
    assert round(0.5 * 450) - 1 <= len(out) <= round(0.5 * 450)
    keys = [_tri_key(t) for t in out]
    for i in sorted(road):
        assert _tri_key(tris[i]) in keys
    assert [k for k in keys if k in {_tri_key(tris[i]) for i in road}] == [_tri_key(tris[i]) for i in sorted(road)]
    originals = {(v.x, v.y, v.z, v.u, v.v) for t in tris[:450] for v in t[:3]}
    assert all(c in originals for k in keys for c in k)
    border = {(x, z) for x, _, z, _, _ in originals if x in (0, 15) or z in (0, 15)}
    assert border <= {(x, z) for k in keys for x, _, z, _, _ in k}
    assert all(newell([(v.x, v.y, v.z) for v in t[:3]])[1] < 0 for t in out)   # clockwise about +y, as the input
    assert all(newell([(v.x, v.y, v.z) for v in t[:3]])[1] < 0 for t in tris)
    check_pieces(thinned, model.convert_model(thinned))
    assert len(list(pol.triangles(model.thin(m, lambda oi: 0.01)))) < 200   # down to what the border allows
    for keep in (None, 0.5, lambda oi: True, lambda oi: 0.0, lambda oi: 1.5, lambda oi: math.nan, lambda oi: "0.5",
                 lambda oi: {}[oi], lambda oi, extra: 0.5):
        with pytest.raises(ConvertError):
            model.thin(m, keep)
    for protected in (None, 3, {-1}, {True}, {len(tris)}, {"1"}, {1.5}):
        with pytest.raises(ConvertError):
            model.thin(m, lambda oi: 0.5, protected=protected)
    with pytest.raises(ConvertError):
        model.thin("m", lambda oi: 0.5)


# ---- Desert City ----

CS3_TRIANGLES = 22837            # CS3_POL.BIN (the brief's 8,961 is CS0's, Beginner's)


def test_model_desert_city(disc):  # noqa: F811
    """Review Focus 1 on Desert City (CS3_POL.BIN) as the scenery draws it (build.kept_model): the meshes of the
    dropped decals (build.decals: the shadows 55, 56, 74 and the wholly transparent 81; Task 5 ruling 2), of the two
    sky domes (build.sky_domes: objects 0 and 1), of the 20 shadow volumes (build.shadow_volumes; Task 6; v0.1.1
    adds 22:1-4, kind 3, in prop 22) and of the run-time props at the origin (build.origin_props: objects 2-4, 8-28 and 54; Task 7) removed,
    the kept triangles less the degenerate ones each appear exactly once over all cells' objects, counted by decoding
    the written objects; every object within MAX_RECORDS; every cell in 0-255; the world inside -1024..1023; the
    counts the discovery log gives (docs/spike, "Model", "Scenery assembled" and "Desert City seen")."""
    from dtools.convert import build
    from dtools.dc.disc import course_files
    name = course_files(3, kind="model")
    m = pol.decode_pol(disc.read(name), name)
    tris = list(pol.triangles(m))
    assert len(tris) == CS3_TRIANGLES
    blob = disc.read(course_files(3, kind="textures"))
    dropped = build.decals(m, blob)
    assert dropped == (55, 56, 74, 81)
    assert build.sky_domes(m) == DESERT_SKY and build.shadow_volumes(m) == DESERT_VOLUMES
    decals = model.drop_textures(m, dropped)
    gone = [me for ob in m.objects for me in ob.meshes if me.texture in dropped]
    assert (len(gone), len(tris) - len(list(pol.triangles(decals)))) == DESERT_DROPPED
    assert build.origin_props(m) == DESERT_PROPS
    kept = build.kept_model(m, blob)
    keys = {(oi, mi) for oi, ob in enumerate(m.objects) for mi, me in enumerate(ob.meshes)
            if me.texture in dropped or oi in DESERT_SKY + DESERT_PROPS} | set(DESERT_VOLUMES)
    assert kept == model.drop_meshes(m, sorted(keys))
    ktris = list(pol.triangles(kept))
    assert len(ktris) == len(tris) - DESERT_DROPPED[1] - DESERT_SKY_VOLUME_TRIANGLES - DESERT_PROP_TRIANGLES
    pieces = model.convert_model(kept)
    counts = check_pieces(kept, pieces)
    degenerate = len(ktris) - counts["owned"]
    assert degenerate == model_degenerate(ktris)
    assert (degenerate, len(pieces)) == DESERT_COUNTS[:2]
    assert (counts["drawn"], counts["quads"], counts["moves"], counts["pieces"], counts["words"]) == DESERT_COUNTS[2:]
    assert 4 * counts["words"] <= 1_831_884                    # the polygon run left after the HUD outline
    # the whole model (Phase 7 Task 2's counts) converts too: 41 degenerate, 18,059 drawn, 21,136 records
    full = check_pieces(m, model.convert_model(m))
    assert (CS3_TRIANGLES - full["owned"], full["drawn"], full["records"]) == (41, 18059, 21136)


DESERT_DROPPED = (77, 1883)
"""Measured: the dropped decals' meshes and triangles (55: 12 meshes, 62 triangles; 56: 18, 370; 74: 31, 657; 81: 16,
794)."""
DESERT_SKY = (0, 1)
DESERT_VOLUMES = ((22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (24, 0), (30, 0), (30, 1), (32, 0), (32, 1), (35, 0),
                  (37, 0), (39, 0), (41, 0), (45, 0), (51, 0), (63, 0), (65, 0), (67, 0), (67, 1))
DESERT_SKY_VOLUME_TRIANGLES = 1536
"""Measured (Task 6): the triangles of the two domes' four meshes and of the 16 volumes."""
DESERT_PROPS = (2, 3, 4) + tuple(range(8, 29)) + (54,)
DESERT_PROP_TRIANGLES = 2398
"""Measured (Task 7): the props' triangles not already dropped as a volume (22:0 and 24:0 are both)."""
DESERT_COUNTS = (28, 69, 13522, 3470, 2220, 69, 158110)
"""Measured on the kept model (Task 7: the props dropped; Task 6: 70 cells, 15,415 drawn, 3,975 quads, 2,627 moves, 70
objects, 181,120 words): degenerate triangles, cells, drawn polygons, quads, move records, objects, words."""


def model_degenerate(tris) -> int:
    """Triangles with zero area once rounded (model.degenerate, the converter's own test)."""
    return sum(model.degenerate([arcade(v) for v in t[:3]]) for t in tris)


# ---- refusals ----

def test_model_refuses():
    """A non-finite vertex, a model outside the world, junk: ConvertError."""
    m = synthetic()
    ob = m.objects[0]

    def with_entry(mi, k, value):
        me = ob.meshes[mi]
        st = me.strips[0]
        st2 = dataclasses.replace(st, entries=st.entries[:k] + (value,) + st.entries[k + 1:])
        me2 = dataclasses.replace(me, strips=(st2,) + me.strips[1:])
        return dataclasses.replace(m, objects=(dataclasses.replace(ob, meshes=ob.meshes[:mi] + (me2,)
                                                                   + ob.meshes[mi + 1:]),))

    def with_mesh(mi, **kw):
        me = dataclasses.replace(ob.meshes[mi], **kw)
        return dataclasses.replace(m, objects=(dataclasses.replace(ob, meshes=ob.meshes[:mi] + (me,)
                                                                   + ob.meshes[mi + 1:]),))

    v = ob.meshes[0].strips[0].entries[1]
    cases = [
        (with_entry(0, 1, dataclasses.replace(v, x=float("nan"))), "vertex"),
        (with_entry(0, 1, dataclasses.replace(v, y=float("inf"))), "vertex"),
        (with_entry(0, 1, dataclasses.replace(v, z=1e39)), "vertex"),
        (with_entry(0, 1, dataclasses.replace(v, x=True)), "vertex"),
        (with_entry(0, 1, dataclasses.replace(v, x=1023.5)), "outside"),
        (with_entry(0, 1, dataclasses.replace(v, x=-1024.5)), "outside"),
        (with_entry(0, 1, dataclasses.replace(v, z=-1023.5)), "outside"),        # the arcade's z 1023.5
        (with_entry(0, 1, dataclasses.replace(v, z=1024.5)), "outside"),
        (with_entry(0, 1, dataclasses.replace(v, u=float("nan"))), "u, v"),
        (with_entry(0, 1, dataclasses.replace(v, v=None)), "u, v"),
        (with_entry(0, 1, dataclasses.replace(v, normal=(0.0, float("nan"), 0.0))), "normal"),
        (with_entry(0, 1, dataclasses.replace(v, normal=None)), "normal"),
        (with_entry(0, 1, None), "entry"),
        (with_entry(0, 1, pol.Reuse(0x5FF00000, -4)), "re-use"),
        (with_entry(0, 1, pol.Reuse(None, -40)), "re-use word"),
        (with_entry(0, 1, pol.Reuse(0x12345678, -40)), "re-use word"),
        (with_mesh(1, strips=(Strip(LIST | 3, ob.meshes[1].strips[0].entries[:4]),)), "multiple of 3"),
        (with_mesh(0, texture=99), "texture"),
        (with_mesh(0, texture=True), "texture"),
        (with_mesh(0, texture=-2), "texture"),
        (with_mesh(0, params=None), "param"),
        (with_mesh(0, params=(1, 2)), "param"),
        (with_mesh(0, strips=None), "strip"),
        (with_mesh(0, strips=(Strip(None, ()),)), "flags"),
        (with_mesh(0, strips=(Strip(2, None),)), "entries"),
        (dataclasses.replace(m, objects=(dataclasses.replace(ob, meshes=None),)), "meshes"),
        (dataclasses.replace(m, objects=(None,)), "Object"),
        (dataclasses.replace(m, objects=None), "objects"),
        (dataclasses.replace(m, textures=(TexEntry(0, 64, 0, 0, 0),) + m.textures[1:]), "texture 0"),
        (dataclasses.replace(m, textures=(TexEntry(True, 64, 0, 0, 0),) + m.textures[1:]), "texture 0"),
        (dataclasses.replace(m, textures=(None,) + m.textures[1:]), "texture 0"),
        (dataclasses.replace(m, textures=None), "textures"),
    ]
    raw_mesh = with_mesh(1, params=(0x8000001C,) + ob.meshes[1].params[1:])     # raw normals: three u32 words
    cases.append((raw_mesh, "normal"))
    for bad, match in cases:
        with pytest.raises(ConvertError, match=match):
            model.convert_model(bad)
    for bad in (None, "CS3_POL.BIN", dataclasses.astuple(m), b"\0" * 64):
        with pytest.raises(ConvertError, match="Model"):
            model.convert_model(bad)


def test_model_raw_normals():
    """A raw-normal mesh (params[0] & 0x30 == 0x10): the first normal word packs the normal as signed bytes
    (x << 16 | y << 8 | z); it decides the side of a double-sided (cull mode 1) triangle as a float normal does, and
    the record's normal is the face's (as every record's)."""
    up = (0x007F00, 0x808080, 0)                      # (0, 127, 0)
    down = (0x008100, 0x808080, 0)                    # (0, -127, 0)
    raw = (0x8000001C, 0x83000000, 0x2008A152, 0xCC000000)
    for words, sign in ((up, 1), (down, -1)):
        tri = Strip(LIST | 1, (V(0, 0, 0, n=words), V(1, 0, 0, n=words), V(0, 0, 1, n=words)))
        [pc] = model.convert_model(make_model([mesh([tri], params=raw)]))[0x88]
        [rec] = pc.obj.records
        assert rec.normal == (0.0, -sign * 1.0, 0.0) and (rec.attr >> 17) & 1


def test_model_never_raises_other_errors():
    """Fuzz: the synthetic model with a vertex, strip, mesh, object or texture field swapped for junk converts or is
    refused with ConvertError, never another error; whatever converts encodes."""
    m = synthetic()
    rng = random.Random(29)
    junk = (None, True, np.bool_(False), -1, 0, 1, 2, 3, 8, 11, 0x10000, 2 ** 40, 10 ** 400, float("nan"),
            float("inf"), 1e39, -2000.0, 1100.0, 0.5, "x", (), (1,), (1.0, 2.0, 3.0), np.int64(3), np.float32(2.0),
            (0x7F00, 0, 0), pol.Reuse(0x5FF00000, -40), V(0, 0, 0))
    converted = refused = 0
    for _ in range(1500):
        ob = m.objects[0]
        mi = rng.randrange(len(ob.meshes))
        me = ob.meshes[mi]
        st = me.strips[0]
        k = rng.randrange(len(st.entries))
        which = rng.choice(("vertex", "vertex", "vertex", "strip", "mesh", "texentry", "object"))
        if which == "vertex":
            f = rng.choice(("x", "y", "z", "normal", "u", "v"))
            ent = dataclasses.replace(st.entries[k], **{f: rng.choice(junk)})
            st = dataclasses.replace(st, entries=st.entries[:k] + (ent,) + st.entries[k + 1:])
        elif which == "strip":
            st = dataclasses.replace(st, **{rng.choice(("flags", "entries")): rng.choice(junk)})
        if which in ("vertex", "strip"):
            me = dataclasses.replace(me, strips=(st,))
        elif which == "mesh":
            me = dataclasses.replace(me, **{rng.choice(("params", "texture", "kind", "strips")): rng.choice(junk)})
        bad = dataclasses.replace(m, objects=(dataclasses.replace(ob, meshes=ob.meshes[:mi] + (me,)
                                                                  + ob.meshes[mi + 1:]),))
        if which == "texentry":
            t = dataclasses.replace(m.textures[0], **{rng.choice(("width", "height")): rng.choice(junk)})
            bad = dataclasses.replace(m, textures=(t,) + m.textures[1:])
        elif which == "object":
            bad = dataclasses.replace(m, objects=(rng.choice(junk),))
        try:
            pieces = model.convert_model(bad)
        except ConvertError:
            refused += 1
            continue
        converted += 1
        for ps in pieces.values():
            for pc in ps:
                P.encode_object(pc.obj)
    assert refused > 900 and converted > 300, (refused, converted)     # measured 1,098 and 402 (kind is raw,
    #                                                                    many swaps are legal values)


# ---- the seen side against the arcade's own Beginner ----

def _contains(p: P.Poly, pt: np.ndarray, tol: float) -> bool:
    """`pt` lies on the plane of drawn polygon `p` (within `tol`) and inside it (its fan's triangles)."""
    q = np.array(p.points, dtype=float)
    if abs((pt - q[0]) @ np.array(p.normal)) > tol:
        return False
    for k in range(1, len(q) - 1):
        v0, v1, v2 = q[k] - q[0], q[k + 1] - q[0], pt - q[0]
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if den and (b := (d11 * d20 - d01 * d21) / den) >= -1e-6 and (c := (d00 * d21 - d01 * d20) / den) >= -1e-6 \
                and b + c <= 1 + 1e-6:
            return True
    return False


def test_model_seen_side_matches_beginner(disc):  # noqa: F811
    """The seen-side rule checked against the arcade, independently of the Beginner-style test's own convention: CS0
    (the Dreamcast's Beginner) converted, each textured drawn polygon matched to the arcade Beginner's drawn cell
    polygons that contain all its corners (on their plane within 0.01, inside them). Every matched single-sided
    polygon has a matched arcade polygon with the same stored normal (dot > 0.99) and none has only opposite ones;
    the double-sided ones are mostly double-sided on the arcade too. The counts are pinned (the meshes differ in
    most places, so most polygons match nothing). Untextured faces are left out: CS0 has two untextured faces on the
    road (y 0.0026, seen from below) that coincide with the arcade's road seen from above."""
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    from dtools.dc.disc import course_files
    c = _beginner()
    arcade_polys = [p for i in sorted({pl.oba & 0x7FFFFF for ptr in set(c.cells) for pl in c.object_lists[ptr]})
                    for p in P.polygons(c.objects[i]) if p.drawn]
    grid: dict[tuple, list[int]] = {}
    for k, p in enumerate(arcade_polys):
        q = np.array(p.points)
        lo, hi = np.floor(q[:, [0, 2]].min(0) / 8).astype(int), np.floor(q[:, [0, 2]].max(0) / 8).astype(int)
        for x in range(lo[0], hi[0] + 1):
            for z in range(lo[1], hi[1] + 1):
                grid.setdefault((x, z), []).append(k)
    name = course_files(0, kind="model")
    pieces = model.convert_model(pol.decode_pol(disc.read(name), name))
    counts = dict(single=0, single_same=0, single_opposite=0, double=0, double_on_arcade=0)
    for ps in pieces.values():
        for pc in ps:
            for poly, rec, tex in zip(P.polygons(pc.obj), pc.obj.records, pc.textures):
                if not poly.drawn or tex < 0:
                    continue
                q = np.array(poly.points)
                centre = q.mean(0)
                near = {k for pt in q for k in grid.get(tuple(np.floor(pt[[0, 2]] / 8).astype(int)), ())}
                hits = [arcade_polys[k] for k in sorted(near)
                        if all(_contains(arcade_polys[k], pt * 0.999 + centre * 0.001, 0.01) for pt in q)]
                if not hits:
                    continue
                dots = [np.array(rec.normal) @ np.array(h.normal) for h in hits]
                if (rec.attr >> 17) & 1:
                    counts["double"] += 1
                    counts["double_on_arcade"] += any((h.attr >> 17) & 1 for h in hits)
                else:
                    counts["single"] += 1
                    counts["single_same"] += any(d > 0.99 for d in dots)
                    counts["single_opposite"] += not any(d > 0.99 for d in dots)
    assert counts == SEEN_SIDE_COUNTS


SEEN_SIDE_COUNTS = dict(single=2076, single_same=2076, single_opposite=0, double=104, double_on_arcade=94)
"""Measured: CS0's textured drawn polygons whose corners all lie in an arcade Beginner polygon (2,180 of them; the
review's probe, matched another way, counted 2,393 single-sided and 279 of 292 double-sided)."""


# ---- texture points (Task 4) ----

def _tex_header(w=64, h=32, **kw):
    from dtools.arcade import texture as T
    base = dict(tha=0, step=0, width=w, height=h, tex_x=0, tex_y=0, sheet=0, textured=True, translucent=False,
                checker=False, wrap_x=False, wrap_y=False, mirror_x=False, mirror_y=False, utex=False,
                utex_min_lod=1, utex_x=0, utex_y=0, luma_base=0, colour=1, raw=(0, 1, 0))
    base.update(kw)
    return T.TexHeader(**base)


def test_polygon_points_rule():
    """The UV rule: u' = width * u, v' = height * (1 - v) (rows flipped top-first by Task 3), in 1/8 texel rounded to
    nearest, as (v', u') per corner; then each axis shifted by whole texture repeats so its least point is in
    0 .. 8 * size - 1; the axis needs wrap where a point then lies past 8 * size (a point on the far edge is inside).
    A span past the u16 range is refused."""
    from dtools.convert import textures
    pp = textures.polygon_points
    full = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert pp(full, 64, 32) == (((256, 0), (256, 512), (0, 512), (0, 0)), (False, False))
    assert pp(((0.5, 0.5), (0.25, 0.75), (1 / 64, 1 / 3)), 64, 32) == \
        (((128, 256), (64, 128), (171, 8)), (False, False))         # 85.33 -> 85 ... rounded to nearest 1/8 texel
    # u in -1..0 is one repeat to the left: shifted by +1 repeat, inside; v in 1..2 likewise (v' = 32 (1 - v) <= 0)
    assert pp(((-1.0, 1.0), (0.0, 1.0), (0.0, 2.0)), 64, 32) == (((256, 0), (256, 512), (0, 512)), (False, False))
    # u 1.5 .. 2.25: least 768 -> 256 (one repeat off), greatest 1152 -> 640 > 512: wrap x
    assert pp(((1.5, 0.0), (2.25, 0.0), (2.0, 0.5)), 64, 32) == (((256, 256), (256, 640), (128, 512)), (True, False))
    # v -0.5 .. 0.5: v' 16 .. 48 texels -> 128 .. 384 words, past 256: wrap y; u stays
    assert pp(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5)), 64, 32) == (((384, 0), (384, 512), (128, 512)), (False, True))
    # a span wider than a repeat needs wrap whatever the shift
    assert pp(((0.0, 0.0), (3.0, 0.0), (3.0, 1.0)), 32, 32)[1] == (True, False)
    # mirrored axis: shifted by whole mirror periods (2 repeats), so the mirror phase is kept; no wrap bit needed
    assert pp(((-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 64, 32, mirror=(True, False)) == \
        (((0, 512), (0, 1024), (0, 1024)), (False, False))
    assert pp(((255.0, 0.0), (255.9, 0.0), (255.5, 1.0)), 32, 32)[0][0] == (256, 0)
    with pytest.raises(ConvertError, match="u16"):
        pp(((0.0, 0.0), (300.0, 0.0), (0.0, 1.0)), 32, 32)          # 300 * 32 * 8 = 76,800 > 65,535
    with pytest.raises(ConvertError, match="u16"):
        pp(((0.0, 0.0), (0.0, -300.0), (0.0, 1.0)), 32, 32)


def test_polygon_points_refuse():
    """Junk UVs (not 3 or 4 pairs of finite numbers; bool), a size that is not 32 << n, a mirror that is not two
    bools: ConvertError, never a raw error."""
    from dtools.convert import textures
    tri = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    for uvs in (None, (), tri[:2], tri * 2, ((0.0, 0.0, 0.0),) * 3, ((math.nan, 0.0),) + tri[1:],
                ((math.inf, 0.0),) + tri[1:], ((1e308, 0.0),) + tri[1:], ((True, 0.0),) + tri[1:],
                (("0", 0.0),) + tri[1:], "abc",
                np.zeros((3, 2))):
        with pytest.raises(ConvertError):
            textures.polygon_points(uvs, 64, 64)
    for w, h in ((48, 64), (64, 0), (True, 64), (64, 8192), (64.0, 64), (None, 64)):
        with pytest.raises(ConvertError, match="32 << n"):
            textures.polygon_points(tri, w, h)
    for mirror in ((1, 0), (True,), None, "ab"):
        with pytest.raises(ConvertError, match="mirror"):
            textures.polygon_points(tri, 64, 64, mirror=mirror)


def test_course_points_synthetic():
    """course_points over convert_model's pieces: per piece one entry per record (encode_texture_points with the
    piece's object accepts them), a move record's 4 points and an untextured record's 3 or 4 all zero (as Beginner's:
    4,712 of 4,713 moves and 4,521 of 4,566 untextured drawn records), a textured record's the UV rule on its
    texture's header; the headers returned with wrap_x / wrap_y set exactly where one of the texture's polygons needs
    it (wrapped is safe for an unwrapped use too), every other field as given, the input untouched."""
    from dtools.arcade import texture as T
    from dtools.convert import textures
    m = synthetic()
    pieces = model.convert_model(m)
    hdrs = {0: _tex_header(64, 64), 1: _tex_header(32, 32, tex_x=64)}
    pts, out = textures.course_points(pieces, hdrs)
    assert set(pts) == set(pieces) and hdrs[0].wrap_x is False                  # the input is not changed
    need = {0: [False, False], 1: [False, False]}
    seen = dict(move=0, untextured=0, textured=0)
    for cell, ps in pieces.items():
        assert len(pts[cell]) == len(ps)
        for pc, pp in zip(ps, pts[cell]):
            T.encode_texture_points(pp, pc.obj)
            for r, t, uv, q in zip(pc.obj.records, pc.textures, pc.uvs, pp):
                if uv is None or t < 0:
                    assert q == ((0, 0),) * (4 if r.is_quad else 3)
                    seen["move" if uv is None else "untextured"] += 1
                    continue
                seen["textured"] += 1
                h = hdrs[t]
                want, wraps = textures.polygon_points(uv, h.width, h.height)
                assert q == want
                need[t] = [a or b for a, b in zip(need[t], wraps)]
    assert seen["move"] and seen["untextured"] and seen["textured"]
    for t, h in out.items():
        assert (h.wrap_x, h.wrap_y) == tuple(need[t])
        assert dataclasses.replace(h, wrap_x=False, wrap_y=False) == hdrs[t]
    assert need[0] == [False, False] and need[1] == [False, False]         # synthetic UVs lie in 0..1
    # a texture whose UVs leave 0..1: its header gets wrap on that axis only
    far = make_model([mesh([Strip(0x12, (V(0, 0, 0, 0.0, 0.0), V(1, 0, 0, 2.5, 0.0), V(0, 0, 1, 0.0, 1.0)))],
                           texture=1)])
    _, out = textures.course_points(model.convert_model(far), hdrs)
    assert (out[1].wrap_x, out[1].wrap_y, out[0].wrap_x) == (True, False, False)


def test_course_points_refuse():
    """Pieces or headers that are not convert_model's and course_textures' kinds, a texture index without a header,
    UVs whose count is not the record's, a polygon past the u16 range: ConvertError naming the cell and record."""
    from dtools.convert import textures
    pieces = model.convert_model(synthetic())
    hdrs = {0: _tex_header(64, 64), 1: _tex_header(32, 32, tex_x=64)}
    cell, ps = next(iter(pieces.items()))
    pc = ps[0]
    k = next(i for i, t in enumerate(pc.textures) if t >= 0)
    bad_uv = pc.uvs[:k] + (pc.uvs[k][:2],) + pc.uvs[k + 1:]
    far = pc.uvs[:k] + (((0.0, 0.0), (900.0, 0.0)) + pc.uvs[k][2:],) + pc.uvs[k + 1:]
    for bad_pieces, bad_hdrs, match in [
            (None, hdrs, "pieces"), ({cell: (None,)}, hdrs, "Piece"), ({cell: pc}, hdrs, "pieces"),
            ({"c": ps}, hdrs, "cell"), (pieces, None, "headers"), (pieces, {0: hdrs[0]}, "texture 1"),
            (pieces, {**hdrs, 1: "h"}, "TexHeader"),
            (pieces, {**hdrs, 1: _tex_header(32, 32, textured=False)}, "TexHeader"),
            ({cell: (pc._replace(uvs=bad_uv),) + ps[1:]}, hdrs, f"cell {cell:#x}"),
            ({cell: (pc._replace(uvs=far),) + ps[1:]}, hdrs, "u16"),
            ({cell: (pc._replace(textures=pc.textures[:-1]),) + ps[1:]}, hdrs, "records"),
            ({cell: (pc._replace(textures=pc.textures[:k] + (True,) + pc.textures[k + 1:]),) + ps[1:]}, hdrs,
             "texture")]:
        with pytest.raises(ConvertError, match=match):
            textures.course_points(bad_pieces, bad_hdrs)


DESERT_POINTS = dict(textured=16865, wrapped_x=5100, wrapped_y=3062, textures_wrapped=49)
"""Measured on Desert City: textured drawn polygons, how many need wrap in x and in y, the textures given a wrap bit."""


def test_converted_points_inside(disc):  # noqa: F811
    """Review Focus 3 on Desert City: every converted texture point lies inside its texture (0 .. 8 * size, 1/8
    texel) or its header has the wrap (or mirror) bit for that axis; every piece's points encode with its object (one
    entry per record, 3 or 4 points by the record) and decode back from a textures blob (texture.texture_points);
    move and untextured records hold zeros; the counts (DESERT_POINTS)."""
    from dtools.arcade import texture as T
    from dtools.arcade.romset import RomSet
    from dtools.convert import textures
    from dtools.dc.disc import course_files
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    mn, tn = course_files(3, kind="model"), course_files(3, kind="textures")
    m = pol.decode_pol(disc.read(mn), mn)
    headers, _ = textures.course_textures(m, disc.read(tn), rs)
    pieces = model.convert_model(m)
    pts, out = textures.course_points(pieces, headers)
    counts = dict(textured=0, wrapped_x=0, wrapped_y=0)
    blob, at = bytearray(), {}
    for cell, ps in pieces.items():
        for pc, pp in zip(ps, pts[cell]):
            at[(cell, len(blob) // 2)] = (pc, pp)
            blob += T.encode_texture_points(pp, pc.obj)
            for t, uv, q in zip(pc.textures, pc.uvs, pp):
                if t < 0 or uv is None:
                    assert not any(map(any, q))
                    continue
                h = out[t]
                counts["textured"] += 1
                vs, us = [v for v, _ in q], [u for _, u in q]
                inside_x, inside_y = max(us) <= 8 * h.width, max(vs) <= 8 * h.height
                assert min(us) >= 0 and min(vs) >= 0
                assert inside_x or h.wrap_x or h.mirror_x, (cell, t, q)
                assert inside_y or h.wrap_y or h.mirror_y, (cell, t, q)
                counts["wrapped_x"] += not inside_x
                counts["wrapped_y"] += not inside_y
    trs = RomSet(program=b"", data=b"", copro_data=b"", polygons=b"", textures=bytes(blob), extra={})
    for (cell, w), (pc, pp) in at.items():
        assert T.texture_points(trs, w, pc.obj) == pp
    counts["textures_wrapped"] = sum(h.wrap_x or h.wrap_y for h in out.values())
    print(f"\nDesert City texture points: {counts}, {len(blob):,} bytes")
    assert counts == DESERT_POINTS
    assert all(dataclasses.replace(out[t], wrap_x=False, wrap_y=False) == headers[t] for t in headers)


# ---- per-course quad tolerances and the re-mesh (Phase 10 Task 3) ----

def _trapezoid(texture: int = 0) -> Model:
    """Two triangles of one strip (cull mode 2) making a planar trapezoid whose (u, v) are not affine: v is the same
    along each of the two parallel edges, which differ in length. The fourth corner is a quarter of the texture's side
    off the affine prediction (texture 0, 64 x 64: 16 texels; texture 1, 32 x 32: 8 texels)."""
    st = Strip(0x12, (V(0, 0, 0, u=0, v=0), V(4, 0, 0, u=1, v=0), V(0.5, 0, 4, u=0, v=1), V(3.5, 0, 4, u=1, v=1)))
    return make_model([mesh([st], texture=texture)])


def test_convert_model_quad():
    """convert_model's `quad` (build.QUAD, per course; fix round 1's ruling I1): (degrees between the two normals, the
    fourth corner's shear off the affine prediction as a fraction of its texture's smaller side). None keeps QUAD_ANGLE
    and QUAD_UV (absolute texels: the same pieces); the shear bound scales with the texture (a quarter-side shear merges
    at 0.26 and not at 0.24 on a 64- and a 32-texel texture alike); the fold bound as QUAD_ANGLE; Review Focus 1 holds
    either way; a `quad` that is not two positive finite numbers is refused."""
    m = _trapezoid()
    assert model.convert_model(m, None) == model.convert_model(m)
    assert check_pieces(m, model.convert_model(m))["quads"] == 0
    for tex in (0, 1):
        t = _trapezoid(tex)
        assert check_pieces(t, model.convert_model(t, (0.5, 0.24)))["quads"] == 0, tex
        assert check_pieces(t, model.convert_model(t, (0.5, 0.26)))["quads"] == 1, tex
    fold = make_model([mesh([Strip(0x12, (V(0, 0, 0), V(4, 0, 0), V(0, 0, 4), V(4, 1, 4)))])])  # ~14 degrees

    def quads(pieces):                                  # a folded quad's normal is neither triangle's: no check_pieces
        return sum(len(s) == 2 for ps in pieces.values() for pc in ps for s in pc.sources)
    assert quads(model.convert_model(fold)) == quads(model.convert_model(fold, (10, 0.125))) == 0
    [[pc]] = model.convert_model(fold, (20, 0.125)).values()
    assert quads({0: [pc]}) == 1 and [len(s) for s in pc.sources] == [2]
    for bad in ((1,), (1, 2, 3), (0, 1), (1, -1), (math.nan, 1), (1, math.inf), (True, 1), ("1", 1), 1, "q",
                (90, 0.125), (120.0, 0.125)):      # the fold test squares cos(angle): angle under 90 degrees
        with pytest.raises(ConvertError):
            model.convert_model(m, bad)



def _tie(shear: float) -> tuple:
    """Two triangles (x, p, q) and (q, p, d) of a parallelogram (d = p + q - x, every coordinate a multiple of 1/64:
    exact in single and double precision) on a 32 x 32 texture, d's (u, v) `shear` off the affine prediction in u."""
    x, e1, e2 = (-45.3125, 10.34375, 45.96875), (-0.75, 0.015625, -0.53125), (0.96875, 0.796875, 0.875)
    p, q = tuple(a + b for a, b in zip(x, e1)), tuple(a + b for a, b in zip(x, e2))
    d = tuple(a + b + c for a, b, c in zip(x, e1, e2))
    ux, up, uq = (0.0703125, 0.9765625), (0.0078125, 0.9921875), (0.9296875, 0.9765625)
    ud = (up[0] + uq[0] - ux[0] + shear, up[1] + uq[1] - ux[1])
    return (model._Poly((x, p, q), (ux, up, uq), 0, False, 0, (0,)),
            model._Poly((q, p, d), (uq, up, ud), 0, False, 0, (1,)))


def test_merge_shear_tie_is_exact():
    """Phase 13 Task 2 round 2: the quad tests decide in exact arithmetic, the same on every machine. A fourth corner
    exactly on the shear bound (1/8 of a 32-texel side: 4 texels) merges ("at most"), whatever a least-squares solve's
    rounding would say (numpy's lstsq put this corner past the bound on macOS arm64; LAPACK builds round differently,
    and National Park Speedway's ties merged on one machine and not on another); 2^-20 of the texture past the bound
    it does not, 2^-20 inside it does; under a 0.124 bound it does not; as an absolute 4 texels it does."""
    sizes, quad = [(32, 32)], (10.0, 0.125, True)
    a, b = _tie(0.125)
    q = model._merge(a, b, sizes, quad)
    assert q is not None and q.sources == (0, 1) and q.corners == (a.corners[0], a.corners[1], b.corners[2],
                                                                   a.corners[2])
    assert model._merge(*_tie(0.125 + 2.0 ** -20), sizes, quad) is None
    assert model._merge(*_tie(0.125 - 2.0 ** -20), sizes, quad) is not None
    assert model._merge(*_tie(0.125), sizes, (10.0, 0.124, True)) is None
    assert model._merge(*_tie(0.125), sizes, (10.0, 4.0, False)) is not None      # absolute texels: the same bound


def _road(bend: float = 0.0) -> Model:
    """A road of two lane strips (texture 0, cull mode 2) along +z, 9 rows 2 units apart, x 0-4 and 4-8 (u 0-0.5,
    0.5-1, continuous across the shared line; v = z / 16), and a verge strip (texture 1) sharing the road's x = 8
    edge out to x = 12. `bend`: each row's x shifted by bend * (z - 8)^2 (0: straight and flat, so every surface is
    one plane)."""
    def at(x, z, u, tex_v):
        return V(x + bend * (z - 8) ** 2, 0, z, u=u, v=tex_v)
    lanes = [Strip(0x12, tuple(at(x0 + 4 * (k & 1), 2 * (k // 2), (x0 + 4 * (k & 1)) / 8, 2 * (k // 2) / 16)
                              for k in range(18))) for x0 in (0, 4)]
    verge = Strip(0x12, tuple(at(8 + 4 * (k & 1), 2 * (k // 2), (k & 1) / 2, 2 * (k // 2) / 16) for k in range(18)))
    return make_model([mesh(lanes, texture=0), mesh([verge], texture=1)])


def test_remesh():
    """model.remesh (Phase 10 Task 3, lever 3): on a flat straight road the lane strips merge across and the
    cross-sections go (far fewer triangles), every kept corner an original vertex with its (u, v), the outline kept,
    the verge moving with the road along their shared edge (each mesh's vertices on it the same set: no crack), every
    original vertex within eps of the new surface, the winding kept; on a bent road whose rows bow more than eps the
    cross-sections stay; a model with nothing to collapse comes back equal; the result converts (Review Focus 1); an
    eps that is not a positive finite number is refused."""
    m = _road()
    tris = list(pol.triangles(m))
    assert len(tris) == 48
    out = model.remesh(m, 0.1)
    new = list(pol.triangles(out))
    assert len(new) <= 8
    originals = {(v.x, v.y, v.z, v.u, v.v) for t in tris for v in t[:3]}
    assert all(c in originals for t in new for c in _tri_key(t))
    corners = {(c[0], c[2]) for t in new for c in _tri_key(t)}
    assert {(0.0, 0.0), (0.0, 16.0), (12.0, 0.0), (12.0, 16.0)} <= corners
    for mi in (0, 1):
        assert out.objects[0].meshes[mi].texture == m.objects[0].meshes[mi].texture
    keys = model.triangle_meshes(out)
    on_edge = [{c[2] for t, key in zip(new, keys) if key[1] == mi for c in _tri_key(t) if c[0] == 8.0} for mi in (0, 1)]
    assert on_edge[0] == on_edge[1]
    assert all(newell([(v.x, v.y, v.z) for v in t[:3]])[1] < 0 for t in new)
    check_pieces(out, model.convert_model(out))
    bent = _road(0.02)                                  # a dropped row would bow the edge 0.08 units, past eps 0.01
    kept_rows = {round(v.z, 3) for t in pol.triangles(model.remesh(bent, 0.01)) for v in t[:3]}
    assert kept_rows == {float(2 * r) for r in range(9)}
    one = make_model([mesh([Strip(LIST | 2, (V(0, 0, 0), V(1, 0, 0), V(0, 0, 1)))])])
    assert model.remesh(one, 0.1) == one
    for bad in (0, -1, math.nan, math.inf, True, "0.1", None):
        with pytest.raises(ConvertError):
            model.remesh(m, bad)
    with pytest.raises(ConvertError):
        model.remesh("m", 0.1)


# ---- the re-mesh's bound, measured independently (Phase 10 Task 3 fix round 1: C1, I2, M1) ----

def _point_tri(x: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Distances from points x (n, 3) to the triangles tri (k, 3, 3): (n, k). Inside the triangle's prism the distance
    to its plane, else to the nearest edge (segment)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    rel = x[:, None, :] - a[None]
    h = (rel * n[None]).sum(2)
    p = x[:, None, :] - h[..., None] * n[None]                      # projections on the planes
    inside = np.ones(h.shape, dtype=bool)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= (np.cross(v - u, p - u[None]) * n[None]).sum(2) >= -1e-12
    best = np.where(inside, np.abs(h), np.inf)
    for u, v in ((a, b), (b, c), (c, a)):
        e = v - u
        t = np.clip(((x[:, None, :] - u[None]) * e[None]).sum(2) / (e * e).sum(1)[None], 0, 1)
        q = u[None] + t[..., None] * e[None]
        best = np.minimum(best, np.linalg.norm(x[:, None, :] - q, axis=2))
    return best


def _by_mesh(m) -> dict:
    """(object, mesh) -> its non-degenerate triangles' corners (n, 3, 3), Dreamcast frame."""
    out: dict = {}
    for t, ((oi, mi, _), *_rest) in zip(pol.triangles(m), model._walk(m)):
        p = np.array([(v.x, v.y, v.z) for v in t[:3]], dtype=float)
        if np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) > 1e-9:
            out.setdefault((oi, mi), []).append(p)
    return {k: np.array(v) for k, v in out.items()}


def _far(src: np.ndarray, dst: np.ndarray, spacing: float) -> float:
    """The largest distance from a point of the triangles `src` to the union of `dst`: every corner, points along every
    edge and an interior grid, spacing at most `spacing` (at least 10 steps per triangle)."""
    worst = 0.0
    lo, hi = dst.min(1), dst.max(1)
    for p in src:
        n = max(10, math.ceil(np.max(np.linalg.norm(p - np.roll(p, -1, 0), axis=1)) / spacing))
        lat = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], dtype=float) / n
        pts = lat @ p
        sel = np.flatnonzero(((lo <= pts.max(0) + 1.0) & (hi >= pts.min(0) - 1.0)).all(1))
        if not len(sel):
            return math.inf
        d = np.full(len(pts), np.inf)
        for k in range(0, len(sel), 64):
            d = np.minimum(d, _point_tri(pts, dst[sel[k:k + 64]]).min(1))
        worst = max(worst, float(d.max()))
    return worst


def two_way(m0, m1, spacing: float = 0.25) -> tuple[float, float]:
    """(original -> new, new -> original): the largest distance from the surface of each (object, mesh) of `m0` to the
    same mesh's surface in `m1`, and back (`_far`); a mesh only ever covered by its own triangles."""
    a, b = _by_mesh(m0), _by_mesh(m1)
    assert set(b) <= set(a)
    return (max(_far(a[k], b[k], spacing) if k in b else math.inf for k in a),
            max(_far(b[k], a[k], spacing) for k in b))


def cracks(m0, m1, weld: float = 0.01) -> int:
    """Edges of `m1` on its outline (positions welded within `weld`, every mesh together: an edge one triangle uses)
    with a point farther than `weld` from `m0`'s outline: openings the re-mesh made."""
    def outline(m):
        tris = [t for v in _by_mesh(m).values() for t in v]
        pts = np.array([c for t in tris for c in t])
        key = {}
        for i, p in enumerate(pts):
            cell = tuple(np.floor(p / weld).astype(int))
            hit = next((key[c] for c in ((cell[0] + dx, cell[1] + dy, cell[2] + dz) for dx in (-1, 0, 1)
                                         for dy in (-1, 0, 1) for dz in (-1, 0, 1))
                        if c in key and np.linalg.norm(pts[key[c]] - p) <= weld), None)
            key.setdefault(cell, i if hit is None else hit)
        node = [key[tuple(np.floor(p / weld).astype(int))] if np.linalg.norm(
            pts[key[tuple(np.floor(p / weld).astype(int))]] - p) <= weld else i for i, p in enumerate(pts)]
        count: dict = {}
        for t in range(len(tris)):
            for j in range(3):
                e = tuple(sorted((node[3 * t + j], node[3 * t + (j + 1) % 3])))
                count.setdefault(e, []).append((pts[3 * t + j], pts[3 * t + (j + 1) % 3]))
        return [segs[0] for e, segs in count.items() if len(segs) == 1 and e[0] != e[1]]
    old, new = outline(m0), outline(m1)
    if not new:
        return 0
    seg = np.array([[a, b] for a, b in old]) if old else np.zeros((0, 2, 3))
    bad = 0
    for a, b in new:
        pts = a[None] + np.linspace(0, 1, 9)[:, None] * (b - a)[None]
        if not len(seg):
            bad += 1
            continue
        e = seg[:, 1] - seg[:, 0]
        t = np.clip(((pts[:, None] - seg[None, :, 0]) * e[None]).sum(2) / np.maximum((e * e).sum(1), 1e-30)[None], 0, 1)
        d = np.linalg.norm(pts[:, None] - (seg[None, :, 0] + t[..., None] * e[None]), axis=2).min(1)
        bad += bool((d > weld).any())
    return bad


def _arc_road(radius: float, seam_offset: float = 0.0, objects: bool = False) -> Model:
    """A road strip (texture 0) along an arc of `radius` (Dreamcast frame, about +z), 13 rows 2 units apart, 4 units
    wide, and a verge strip (texture 1) beside it sharing the road's outer edge, that edge's verge copies shifted by
    `seam_offset` in x (a near-coincident seam). `objects`: the verge in an object of its own."""
    def at(k, w, shift=0.0):
        a = 2 * k / radius
        r = radius + w
        return r * math.cos(a) - radius + shift, r * math.sin(a)

    def strip(w0, w1, shift_outer, tex):
        es = []
        for k in range(13):
            for w, s in ((w0, 0.0 if w0 != 4 else shift_outer), (w1, shift_outer if w1 == 4 else 0.0)):
                x, z = at(k, w, s)
                es.append(V(x, 0, z, u=w / 8, v=k / 8))
        return Strip(0x12, tuple(es))
    road = mesh([strip(0, 4, 0.0, 0)], texture=0)
    verge = mesh([strip(4, 8, seam_offset, 1)], texture=1)
    if not objects:
        return make_model([road, verge])
    m = make_model([road])
    obj = Object(3, (0.0, 0.0, 0.0, 1000.0), (verge,), sum(len(s.entries) for s in verge.strips))
    return pol.fix_header(Model(m.header, m.objects + (obj,), m.textures))


def test_remesh_two_way_bound():
    """model.remesh on a gently curved road and verge (radius 200: a dropped row bows the chord 0.01, rows go): fewer
    triangles, and the surface within eps both ways, mesh by mesh (two_way: every corner, points along every edge and
    an interior grid), with no crack; on a tighter curve (radius 20: a dropped row bows 0.1+) the bound still holds."""
    for radius in (200.0, 20.0):
        m = _arc_road(radius)
        out = model.remesh(m, 0.1)
        there, back = two_way(m, out)
        assert there <= 0.1 and back <= 0.1, (radius, there, back)
        assert cracks(m, out) == 0, radius
    assert len(list(pol.triangles(model.remesh(_arc_road(200.0), 0.1)))) < len(list(pol.triangles(_arc_road(200.0))))


def test_remesh_near_coincident_seam():
    """C1: the verge's copies of the road's outer edge lie 1e-5 off the road's (a seam whose corners differ in the last
    bits, as CS4's 506 pairs closer than 0.01): welded, the two move together or not at all: no crack, both within eps
    both ways."""
    m = _arc_road(60.0, seam_offset=1e-5)
    out = model.remesh(m, 0.1)
    there, back = two_way(m, out)
    assert there <= 0.1 and back <= 0.1, (there, back)
    assert cracks(m, out) == 0


def test_remesh_no_cover_across_objects():
    """C1: a panel (two triangles, texture 0) whose corner lies on a triangle of another object drawn alike, which also
    has a vertex at the panel's next corner (CS4's tree-line panels, objects 22 and 23): the other object's triangle
    does not count as cover, so the corner (one triangle uses it) is not collapsed and neither loses surface."""
    a = mesh([Strip(LIST | 2, (V(0, 0, 0, u=0, v=0), V(10, 0, 0, u=1, v=0), V(0, 0, 10, u=0, v=1),
                               V(10, 0, 0, u=1, v=0), V(10, 0, 10, u=1, v=1), V(0, 0, 10, u=0, v=1)))], texture=0)
    b = mesh([Strip(LIST | 2, (V(-10, 0, -10, u=-1, v=-1), V(10, 0, 0, u=1, v=0), V(-10, 0, 20, u=-1, v=2)))],
             texture=0)                     # mapped as the panel is: u = x / 10, v = z / 10
    m = make_model([a])
    m = pol.fix_header(Model(m.header, m.objects + (Object(3, (0.0, 0.0, 0.0, 1000.0), (b,), 3),), m.textures))
    out = model.remesh(m, 0.1)
    there, back = two_way(m, out)
    assert there <= 0.1 and back <= 0.1, (there, back)
    assert len(list(pol.triangles(out))) == 3


REMESH4_SHA1 = "0ad9bc0bc3414e3e8c7fd2f3ae7e5168cc5b1201"
"""The sha1 of repr(model.remesh(CS4's kept model, 0.1)), the kept model without build.NAMED_PROPS[4] (Task 4's fix
round 1: object 3, the big wheel's frame), as 0f0a36c's code computes it (git archive, its own empty cache): 13,263
triangles, every object's re-mesh the same as before but object 3's 98 gone. Before the fix it was c38cb5e2...
(5833095's, the set Task 3's gate measured). v0.1.1 moved it from fcea416f...: build.shadow_volumes also drops
CS4's 23 volumes in per-volume ID colours (13:1-14, 45:0-4, 46:0-3), no longer re-meshed: 12,837 triangles out
(16,543 in), within 0.0991 and 0.0998 both ways."""


def test_remesh_cs4_bound(disc):  # noqa: F811
    """C1 / I2 on the real data: CS4's kept model re-meshed to build.REMESH[4] (0.1) is within 0.1 of the original both
    ways, mesh by mesh (two_way: corners, edge points and an interior grid at most 0.5 apart), and opens no crack. The
    output is the reference code's byte for byte (REMESH4_SHA1), computed here without build.remeshed's cache. The kept
    model is the one converted_set re-meshes: without NAMED_PROPS[4]."""
    import hashlib
    from dtools.convert import build
    from dtools.dc.data import decode_data
    from dtools.dc.disc import course_files
    mn, tn, dn = (course_files(4, kind=k) for k in ("model", "textures", "data"))
    kept = build.kept_model(pol.decode_pol(disc.read(mn), mn), disc.read(tn), decode_data(disc.read(dn), dn),
                            props=build.NAMED_PROPS[4])
    out = model.remesh(kept, build.REMESH[4])
    assert hashlib.sha1(repr(out).encode()).hexdigest() == REMESH4_SHA1
    there, back = two_way(kept, out, 0.5)
    print(f"\nCS4 re-mesh: {len(list(pol.triangles(kept)))} -> {len(list(pol.triangles(out)))} triangles; "
          f"original -> new {there:.4f}, new -> original {back:.4f}")
    assert there <= 0.1 and back <= 0.1, (there, back)
    assert cracks(kept, out) == 0


def test_remesh_valence_one_corner_one_mesh():
    """N3: a corner one triangle uses, inside a single mesh. A panel's corner (two triangles, 10 x 10) is kept: its
    triangle's surface is nowhere else in the mesh. A sliver's corner 0.05 off its neighbour's edge may go: its surface
    lies within eps of that edge. Either way the mesh stays within eps of itself both ways and opens no crack."""
    panel = mesh([Strip(LIST | 2, (V(0, 0, 0, u=0, v=0), V(10, 0, 0, u=1, v=0), V(0, 0, 10, u=0, v=1),
                                   V(10, 0, 0, u=1, v=0), V(10, 0, 10, u=1, v=1), V(0, 0, 10, u=0, v=1)))], texture=0)
    m = make_model([panel])
    out = model.remesh(m, 0.1)
    assert len(list(pol.triangles(out))) == 2 and two_way(m, out) == (0.0, 0.0) and cracks(m, out) == 0
    sliver = mesh([Strip(LIST | 2, (V(0, 0, 0, u=0, v=0), V(10, 0, 0, u=1, v=0), V(0, 0, 10, u=0, v=1),
                                    V(10, 0, 0, u=1, v=0), V(5.05, 0, 5, u=0.505, v=0.5), V(0, 0, 10, u=0, v=1)))],
                  texture=0)
    m = make_model([sliver])
    out = model.remesh(m, 0.1)
    there, back = two_way(m, out, 0.05)
    assert there <= 0.1 and back <= 0.1 and cracks(m, out) == 0, (there, back)


def test_remesh_seam_between_objects():
    """N3: the near-coincident seam (copies 1e-5 apart) between two objects (the road and its verge in objects of their
    own, as CS4's road objects and their neighbours): welded across the objects, both move together or not at all; no
    crack, each mesh within eps of itself both ways."""
    m = _arc_road(60.0, seam_offset=1e-5, objects=True)
    assert len(m.objects) == 2
    out = model.remesh(m, 0.1)
    there, back = two_way(m, out)
    assert there <= 0.1 and back <= 0.1, (there, back)
    assert cracks(m, out) == 0


# ---- texture points past their range (Phase 11 Task 3: discovery log docs/spike/2026-11-courses-4-7.md, "Rin Rin
# Rink") ----

def _strip_model(length=650.0, repeats=300.0, axis=0, texture=1):
    """One flat, double-sided strip of two triangles (they merge into one quad) along +x at y 0, z 5..6 (the arcade's
    z -5..-6), `length` units long, its texture (texture 1: 32 x 32) repeating `repeats` times along it on `axis`
    (0: u runs 0..repeats, v 0..1 across; 1: v runs 1..1 - repeats, u across)."""
    def uv(t, side):
        return (t * repeats, float(side)) if axis == 0 else (float(side), 1.0 - t * repeats)
    corners = [V(length * t, 0.0, 5 + side, *uv(t, side)) for t, side in ((0, 0), (0, 1), (1, 0), (1, 1))]
    return make_model([mesh([Strip(0x11, tuple(corners))], texture=texture)])


def _drawn(pieces):
    """(corners, uvs) of every drawn record of `pieces`, in order."""
    out = []
    for ps in pieces.values():
        for pc in ps:
            for poly, uvs, src in zip(P.polygons(pc.obj), pc.uvs, pc.sources):
                if src:
                    out.append((poly.points, uvs))
    return out


def _area(points) -> float:
    return float(np.linalg.norm(model._newell([tuple(map(float, p)) for p in points]))) / 2


def test_texture_points_cut():
    """convert_model's cut (module docstring, "texture points past their range"): a strip whose texture repeats 300
    times along it (u 0..300 on a 32-texel texture: 76,800 eighths of a texel, past the u16 texture point) is one
    quad textures.polygon_points refuses; with the converted sizes (`points`) it is cut at u 254 (model.band(32): the
    largest even number of repeats that fits, 254 x 256 = 65,024) into two quads:

    - each piece's points fit (polygon_points accepts it, mirrored too: the band is even);
    - the pieces cover the strip exactly: their areas add up to its area, their corners are its corners and the cut
      points, which lie on its long edges at x 650 x 254 / 300 and are the same f32 point in both pieces;
    - the mapping is the same picture: at the cut both pieces carry u 254 (the same texel: 65,024 words and 0 after
      their shifts, both a whole number of repeats) and v as the strip's; each corner's (u, v) is the strip's affine
      mapping at that point; the same along v (axis 1);
    - a strip that fits is not cut (the same pieces with or without `points`); a texture too large for two repeats
      in a point (4,096 texels) refuses; `points` that is not a dict of texture -> (width, height) refuses."""
    from dtools.convert import textures
    m = _strip_model()
    plain = _drawn(model.convert_model(m))
    assert len(plain) == 1 and len(plain[0][0]) == 4
    with pytest.raises(ConvertError, match="u16"):
        textures.polygon_points(plain[0][1], 32, 32)
    assert model.band(32) == 254 and model.band(64) == 126 and model.band(256) == 30 and model.band(4096) == 0
    cut = _drawn(model.convert_model(m, points={1: (32, 32)}))
    assert len(cut) == 2 and all(len(c) == 4 for c, _ in cut)
    for c, uvs in cut:
        textures.polygon_points(uvs, 32, 32)
        textures.polygon_points(uvs, 32, 32, (True, False))
        for p_, (u, v) in zip(c, uvs):
            assert math.isclose(u, p_[0] / 650.0 * 300.0, abs_tol=1e-4) and v == float(-p_[2] - 5 > 0.5), (p_, u, v)
    assert math.isclose(sum(_area(c) for c, _ in cut), _area(plain[0][0]), rel_tol=1e-9)
    xc = 650.0 * 254 / 300
    shared = set(map(tuple, cut[0][0])) & set(map(tuple, cut[1][0]))
    assert len(shared) == 2 and all(math.isclose(p_[0], xc, rel_tol=1e-6) for p_ in shared)
    assert set(map(tuple, cut[0][0])) | set(map(tuple, cut[1][0])) == set(map(tuple, plain[0][0])) | shared
    at_cut = [[uv for p_, uv in zip(c, uvs) if tuple(p_) in shared] for c, uvs in cut]
    assert sorted(at_cut[0]) == sorted(at_cut[1]) == [(254.0, 0.0), (254.0, 1.0)]
    words = [[q[1] for q, p_ in zip(textures.polygon_points(uvs, 32, 32)[0], c) if tuple(p_) in shared]
             for c, uvs in cut]
    assert sorted(w % 256 for w in words[0] + words[1]) == [0] * 4 and sorted(set(words[0] + words[1])) == [0, 65024]
    # along v
    cut_v = _drawn(model.convert_model(_strip_model(axis=1), points={1: (32, 32)}))
    assert len(cut_v) == 2
    for c, uvs in cut_v:
        textures.polygon_points(uvs, 32, 32)
    # a strip that fits is not cut
    short = _strip_model(repeats=200.0)
    assert model.convert_model(short, points={1: (32, 32)}) == model.convert_model(short)
    with pytest.raises(ConvertError, match="no two repeats"):
        model.convert_model(m, points={1: (4096, 32)})
    for bad in ([], {1: (32,)}, {1: (32, 0)}, {1: (32.0, 32)}, {"1": (32, 32)}, {1: (True, 32)}):
        with pytest.raises(ConvertError):
            model.convert_model(m, points=bad)


def _long_strip(x0=-600.0, x1=-360.0, z0=136.0, z1=164.0) -> Model:
    """A flat double-sided textured strip (two triangles, one quad), arcade x x0..x1 and z z0..z1 (Dreamcast z
    negated), u = x / 16 and v = Dreamcast z / 16: Mermaid Lake's start-straight strip in shape and place."""
    pts = [(x0, -z0), (x0, -z1), (x1, -z0), (x1, -z1)]
    return make_model([mesh([Strip(0x11, tuple(V(x, 41.0, z, u=x / 16, v=z / 16) for x, z in pts))], texture=0)])


def _cell_rows(pieces):
    """[(cell, points, uvs)] of every drawn record."""
    return [(cell, poly.points, uv) for cell, ps in pieces.items() for pc in ps
            for poly, uv in zip(P.polygons(pc.obj), pc.uvs) if poly.drawn]


def _flat_area(points) -> float:
    return float(np.linalg.norm(newell(points))) / 2


def test_overhang():
    """model.overhang: how far a polygon reaches past the cell of its centroid (edges at 128 j - 1024.5)."""
    assert model.CELL_EDGES[0] == -1024.5 and model.CELL_EDGES[16] == 1023.5 and len(model.CELL_EDGES) == 17
    assert model.overhang(((0, 0, 0), (10, 0, 0), (0, 0, 10))) == 0.0
    # centroid x -480 (column 4: -512.5..-384.5), reaching -600 (87.5 past) and -360 (24.5 past)
    assert model.overhang(((-600, 0, 140), (-360, 0, 140), (-360, 0, 160), (-600, 0, 160))) == pytest.approx(87.5)
    # centroid z 193.3 (row 9: 127.5..255.5), reaching z 300: 44.5 past
    assert model.overhang(((-500, 0, 140), (-390, 0, 140), (-390, 0, 300))) == pytest.approx(300 - 255.5)


def test_cell_cut_long_strip():
    """convert_model's `cell_cut` (build.CELL_CUT, v0.1.2): uncut, the strip lies in the cell of its centroid (0x94,
    Mermaid Lake's); cut at 32 units it is cut at x -512.5 and -384.5 into pieces in cells 0x93, 0x94 and 0x95, each
    inside its cell, together covering the strip's area exactly, every corner's (u, v) still u = x / 16, v = -z / 16
    (the picture unchanged) and every cut corner exactly on the edge. A reach the strip does not pass cuts nothing;
    a bad `cell_cut` is refused."""
    m = _long_strip()
    plain = model.convert_model(m)
    assert sorted(plain) == [0x94] and len(_cell_rows(plain)) == 1
    cut = model.convert_model(m, cell_cut=32.0)
    rows = _cell_rows(cut)
    assert sorted(cut) == [0x93, 0x94, 0x95] and len(rows) >= 3
    assert sum(_flat_area(p) for _, p, _ in rows) == pytest.approx(240 * 28, rel=1e-6)
    for cell, pts, uvs in rows:
        assert model.overhang(pts) < 1e-3, (cell, pts)
        assert model.cell_of(*np.array(pts)[:, [0, 2]].mean(0)) == cell
        for (x, _, z), (u, v) in zip(pts, uvs):
            assert u == pytest.approx(x / 16, abs=1e-4) and v == pytest.approx(-z / 16, abs=1e-4)
            assert x in (-600.0, -360.0, -512.5, -384.5)
    assert model.convert_model(m, cell_cut=100.0) == plain          # reaches 87.5: not cut
    assert model.convert_model(m, cell_cut=87.0) != plain
    for bad in (-1, math.nan, math.inf, True, "32", (32,)):
        with pytest.raises(ConvertError):
            model.convert_model(m, cell_cut=bad)


def test_cell_cut_spans_both_axes():
    """A polygon crossing a cell edge in x and in z is cut on both: four pieces' cells, each piece inside its cell,
    the area kept."""
    m = _long_strip(x0=-560.0, x1=-460.0, z0=100.0, z1=200.0)       # x crosses -512.5, z crosses 127.5
    rows = _cell_rows(model.convert_model(m, cell_cut=0.0))
    assert {c for c, _, _ in rows} == {0x83, 0x84, 0x93, 0x94}
    assert sum(_flat_area(p) for _, p, _ in rows) == pytest.approx(100 * 100, rel=1e-6)
    assert all(model.overhang(p) < 1e-3 for _, p, _ in rows)


def _wall(x: float, z0: float, z1: float):
    """A vertical double-sided strip at arcade x `x` from z z0 to z1, 20 units tall (a wall, not ground)."""
    pts = [(-z0, 41.0), (-z0, 21.0), (-z1, 41.0), (-z1, 21.0)]
    return make_model([mesh([Strip(0x11, tuple(V(x, y, z, u=z / 16, v=y / 16) for z, y in pts))], texture=0)])


def test_road_copy():
    """convert_model's `road_copy` (build.ROAD_COPY, v0.2: Pascal's Rin Rin Rink report, the road vanishing where the
    game's culling drops the cell the camera still sees): a near-flat polygon whose centroid is within `band` units
    of the road line and which comes within `margin` units of a neighbouring cell is filed in that cell too (the same
    records), so it draws when either cell is open. A strip in cell 0x94, 7.5 units from its west edge (x -512.5) and
    far from its rows' edges, by a road along z 180: in 0x94 and 0x93, the same polygon; with the road far away, or
    a margin it does not reach, or as a wall, only in 0x94; None copies nothing (the default). A bad `road_copy` is
    refused."""
    road = [(-700.0 + 20 * k, 180.0) for k in range(20)]
    m = _long_strip(x0=-505.0, x1=-495.0, z0=170.0, z1=190.0)
    plain = model.convert_model(m)
    assert sorted(plain) == [0x94]
    copied = model.convert_model(m, road_copy=(road, 16.0, 10.0))
    rows = _cell_rows(copied)
    assert sorted(copied) == [0x93, 0x94] and len(rows) == 2 and rows[0][1:] == rows[1][1:]
    assert model.convert_model(m, road_copy=([(x, z + 300.0) for x, z in road], 16.0, 10.0)) == plain
    assert model.convert_model(m, road_copy=(road, 6.0, 10.0)) == plain           # 7.5 from the edge: not within 6
    assert model.convert_model(m, road_copy=None) == plain
    w = _wall(-505.0, 170.0, 190.0)
    assert sorted(model.convert_model(w, road_copy=(road, 16.0, 10.0))) == sorted(model.convert_model(w))
    for bad in ((road, -1.0, 10.0), (road, 16.0, math.nan), (road[:2], 16.0, 10.0), ("road", 16.0, 10.0),
                (road, 16.0), (road, True, 10.0), 5):
        with pytest.raises(ConvertError):
            model.convert_model(m, road_copy=bad)
