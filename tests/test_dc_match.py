"""Tests for the Dreamcast-to-arcade course match (dtools/dc/match.py)."""

import math
import re
import warnings
from dataclasses import replace

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import golden, tgp
from dtools.dc import data, match, pol
from dtools.dc.data import Cell, CourseData, Lane, LinePoint, RoadPoint, Step
from dtools.dc.errors import DcError
from dtools.dc.match import MatchError, Transform
from dtools.dc.pol import Mesh, Model, Object, Strip, TexEntry, Vertex

# ---- synthetic courses ----

TRUE = Transform(1.0275, math.radians(0.6), 3.25, -2.5)       # the kind of transform Phase 1 expected


def _loop(n=40, width=17.0):
    """An arcade road of n quads round a lopsided loop (no mirror symmetry): corners 0, 1 across the forward end,
    2, 3 across the back end (the arcade's layout), y rolling gently."""
    th = np.linspace(0.0, 2.0 * math.pi, n + 1)
    r = 200.0 + 40.0 * np.cos(th) + 25.0 * np.sin(2.0 * th)
    cx, cz = 30.0 + r * np.cos(th), -50.0 + 0.7 * r * np.sin(th)
    d = np.gradient(np.c_[cx, cz], axis=0)
    side = np.c_[-d[:, 1], d[:, 0]] / np.linalg.norm(d, axis=1)[:, None] * width / 2
    y = 2.0 * np.sin(th)
    out_ = [(float(cx[k] + side[k, 0]), float(y[k]), float(cz[k] + side[k, 1])) for k in range(n + 1)]
    in_ = [(float(cx[k] - side[k, 0]), float(y[k]), float(cz[k] - side[k, 1])) for k in range(n + 1)]
    polys = []
    for k in range(n):
        corners = (out_[k + 1], in_[k + 1], in_[k], out_[k])
        centre = tuple(float(v) for v in np.mean(corners, axis=0))
        polys.append(tgp.GroundPoly(corners, centre=centre, direction=0))
    return tuple(polys)


def _arcade(polys, nsteps=None):
    nsteps = len(polys) if nsteps is None else nsteps
    g = tgp.Ground(course=0, grid=(), attributes=bytes(len(polys)), polys=tuple(polys), lists={}, base=0, raw=(),
                   grid_at=0, attributes_at=0, list_base=0, pool_base=0)
    entry = C.CourseEntry(0, 0, 0, 0, 0, 0, 0, (0, 0))
    return golden.Course(course=0, entry=entry, steps=tuple(C.Step(k, 1) for k in range(nsteps)), ground_data=g)


def _inverse(t: Transform, p):
    """The Dreamcast point that `t` takes to arcade point p."""
    x, z = p[0] - t.tx, p[2] - t.tz
    c, s = math.cos(t.angle), math.sin(t.angle)
    x, z = (c * x + s * z) / t.scale, (-s * x + c * z) / t.scale
    return (x, p[1] / t.scale, -z if t.flip_z else z)


def _dc(polys, t=TRUE, mirror=False):
    """Course data whose cells are the arcade polygons taken back through t; `mirror` leaves z un-negated."""
    def back(p):
        q = _inverse(t, p)
        return (q[0], q[1], -q[2]) if mirror else q
    cells = tuple(Cell(tuple(back(p) for p in poly.corners), back(poly.centre), 0) for poly in polys)
    steps = tuple(Step(0, 1) for _ in cells)
    return CourseData((0, 4, 5, 3, 1, 6, 7, 2), (), (), (), bytes([1]) * len(cells), steps, cells,
                      (0,) * data.GRID_CELLS, ((),))


def test_transform_apply():
    t = Transform(2.0, math.pi / 2, 10.0, -5.0)
    # z negated first: (1, 3, -2); scaled: (2, 6, -4); turned a quarter (x' = c x - s z, z' = s x + c z): (4, 6, 2)
    assert t.apply((1.0, 3.0, 2.0)) == pytest.approx((14.0, 6.0, -3.0))
    assert Transform(2.0, math.pi / 2, 10.0, -5.0, flip_z=False).apply((1, 3, 2)) == pytest.approx((6.0, 6.0, -3.0))
    assert Transform(1.0, 0.0, 0.0, 0.0).apply(np.array([1.5, 2.0, 3.0], dtype=np.float32)) == (1.5, 2.0, -3.0)
    for p in [(0.0, 0.0, 0.0), (12.0, -3.0, 40.0)]:
        assert TRUE.apply(_inverse(TRUE, p)) == pytest.approx(p)


@pytest.mark.parametrize("args", [
    (True, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0), (math.nan, 0.0, 0.0, 0.0),
    (1.0, math.inf, 0.0, 0.0), (1.0, 0.0, "1", 0.0), (1.0, 0.0, 0.0, None), (1.0, 0.0, 0.0, 10 ** 400),
    (1.0, 0.0, 0.0, 0.0, 1), (1.0, 0.0, 0.0, 0.0, None)])
def test_transform_refuses(args):
    with pytest.raises(DcError):
        Transform(*args)


@pytest.mark.parametrize("p", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (1.0, True, 3.0), (1.0, math.nan, 0.0), "abc",
                               None, 3.0, (1.0, 2.0, "3")])
def test_transform_apply_refuses(p):
    with pytest.raises(DcError):
        TRUE.apply(p)


def test_fit_synthetic():
    """A known transform (z negation included) recovered to 1e-4; the mirror refused by the gate (Review Focus 5)."""
    polys = _loop()
    arcade = _arcade(polys)
    t, res = match.fit_course(_dc(polys), arcade, gate=1e-3)
    assert t.flip_z is True
    assert (t.scale, t.angle, t.tx, t.tz) == pytest.approx((TRUE.scale, TRUE.angle, TRUE.tx, TRUE.tz), abs=1e-4)
    assert res["rms"] < 1e-4 and res["max"] < 1e-4
    assert set(res) >= {"rms", "max", "rms_x", "rms_y", "rms_z", "points"}
    assert res["points"] == 5 * len(polys)                       # four corners and the centre of every cell
    # the same course mirrored (z left as the arcade's): no rotation and scale fits it, so the gate refuses it
    with pytest.raises(MatchError, match=r"RMS [0-9.]+ .*gate 2") as e:
        match.fit_course(_dc(polys, mirror=True), arcade, gate=2.0)
    rms = float(re.search(r"RMS ([0-9.]+)", str(e.value)).group(1))
    assert rms > 50.0                                            # not a near miss
    assert isinstance(e.value, DcError)
    # a loose gate would take a mirror fitted by reflection; the fit never reflects, so even a loose gate fails
    with pytest.raises(MatchError):
        match.fit_course(_dc(polys, mirror=True), arcade, gate=20.0)


def test_fit_refuses_bad_input():
    polys = _loop(12)
    arcade, dc = _arcade(polys), _dc(polys)
    for gate in (True, 0.0, -1.0, math.nan, math.inf, "1", None):
        with pytest.raises(DcError):
            match.fit_course(dc, arcade, gate=gate)
    for bad in (None, "course", polys):
        with pytest.raises(DcError):
            match.fit_course(bad, arcade, gate=1.0)
        with pytest.raises(DcError):
            match.fit_course(dc, bad, gate=1.0)
    with pytest.raises(DcError, match="ground"):                 # an arcade course without its ground
        match.fit_course(dc, replace(arcade, ground_data=None), gate=1.0)
    for bad in (replace(arcade, ground_data=polys), replace(arcade, steps=None),
                replace(arcade, ground_data=replace(arcade.ground_data, polys=None)),
                replace(arcade, ground_data=replace(arcade.ground_data, polys=(None,) * 12)),
                replace(arcade, ground_data=replace(arcade.ground_data, polys=(
                    tgp.GroundPoly(polys[0].corners[:3]),) + polys[1:])),
                replace(arcade, ground_data=replace(arcade.ground_data, polys=(
                    tgp.GroundPoly(polys[0].corners[:3] + ((True, 0.0, 0.0),)),) + polys[1:]))):
        with pytest.raises(DcError):
            match.fit_course(dc, bad, gate=1.0)
    with pytest.raises(MatchError, match="12 .*11|11 .*12"):     # the road counts disagree
        match.fit_course(dc, _arcade(polys, nsteps=11), gate=1.0)
    with pytest.raises(DcError):                                 # more steps than polygons
        match.fit_course(dc, _arcade(polys, nsteps=13), gate=1.0)
    with pytest.raises(DcError):                                 # a non-finite cell
        bad = replace(dc, cells=(Cell(dc.cells[0].corners, (math.nan, 0.0, 0.0), 0),) + dc.cells[1:])
        match.fit_course(bad, arcade, gate=1.0)
    one = _loop(1)
    with pytest.raises(MatchError):                              # every point the same: no scale to fit
        flat = tgp.GroundPoly(((1.0, 0.0, 1.0),) * 4, centre=(1.0, 0.0, 1.0))
        match.fit_course(_dc((flat,)), _arcade((flat,)), gate=1.0)
    assert match.fit_course(_dc(one), _arcade(one), gate=1e-3)[1]["points"] == 5


@pytest.mark.parametrize("degrees", [179.9, -90.0, 45.0, -1.5])
def test_fit_points_turned(degrees):
    """A rotation well past 1 degree (a half turn less 0.1, a quarter turn back) recovered, z negation included."""
    true = Transform(1.0125, math.radians(degrees), -40.0, 12.5)
    src = [(0.0, 0.0, 0.0), (10.0, 1.0, 0.0), (0.0, 2.0, 10.0), (-7.5, -1.0, 3.25)]
    t, res = match.fit_points(src, [true.apply(p) for p in src], gate=1e-6)
    assert (t.scale, t.tx, t.tz, t.flip_z) == pytest.approx((true.scale, true.tx, true.tz, True), abs=1e-9)
    assert math.remainder(t.angle - true.angle, 2.0 * math.pi) == pytest.approx(0.0, abs=1e-12)
    assert res["rms"] < 1e-9


def test_fit_points():
    src = [(0.0, 0.0, 0.0), (10.0, 1.0, 0.0), (0.0, 2.0, 10.0)]
    t, res = match.fit_points(src, [TRUE.apply(p) for p in src], gate=1e-6)
    assert (t.scale, t.angle, t.tx, t.tz) == pytest.approx((TRUE.scale, TRUE.angle, TRUE.tx, TRUE.tz), abs=1e-9)
    t, _ = match.fit_points(np.array(src), np.array(src), gate=1e-6, flip_z=False)
    assert (t.scale, t.angle, t.tx, t.tz, t.flip_z) == (1.0, 0.0, 0.0, 0.0, False)
    for a, b, kw in [(src, src[:2], {}), ([], [], {}), (src, src, {"flip_z": 1}), (np.zeros((3, 2)), src, {}),
                     (np.ones((3, 3), dtype=bool), src, {}), (src, [(0.0, 0.0, math.inf)] * 3, {}),
                     ([(0.0, 0.0)] * 3, src, {}), ("abc", src, {}), (None, src, {}), (src, src, {"gate": True})]:
        with pytest.raises(DcError):
            match.fit_points(a, b, **{"gate": 1.0, **kw})
    big = [(1e300, 0.0, 0.0), (-1e300, 0.0, 1e300), (0.0, 0.0, -1e300)]       # the fit's sums overflow
    with pytest.raises(MatchError, match="overflow"):
        match.fit_points(big, big, gate=1.0)
    with pytest.raises(DcError, match="float range"):                         # a point taken past the float range
        Transform(1e10, 0.0, 0.0, 0.0).apply((1e300, 0.0, 0.0))


def test_lap_order_and_width_synthetic():
    polys = _loop(40)
    arcade = _arcade(polys)
    t, _ = match.fit_course(_dc(polys), arcade, gate=1e-3)
    assert match.lap_order(_dc(polys), arcade, t) == (0, 1)
    # the Dreamcast's lap starting at the arcade's cell 7: rotate its cells (the fit is done on the true order)
    turned = _dc(polys[7:] + polys[:7])
    assert match.lap_order(turned, arcade, t) == (7, 1)
    # driven the other way round, starting at cell 30
    rev = tuple(polys[(30 - k) % 40] for k in range(40))
    assert match.lap_order(_dc(rev), arcade, t) == (30, -1)
    w_dc, w_arcade = match.road_width(_dc(polys), arcade, t)
    assert w_arcade == pytest.approx(17.0, rel=0.02)
    assert w_dc == pytest.approx(w_arcade, rel=1e-5)
    # at the wrong scale the Dreamcast's width (in arcade units) shows it
    w2, _ = match.road_width(_dc(polys), arcade, replace(t, scale=1.1 * t.scale))
    assert w2 / w_arcade == pytest.approx(1.1, rel=1e-5)
    for f in (match.lap_order, match.road_width):
        with pytest.raises(DcError):
            f(_dc(polys), arcade, (1.0, 0.0, 0.0, 0.0))
        with pytest.raises(DcError, match="float range"):          # a transform taking the road past the floats
            f(_dc(polys), arcade, Transform(1e307, 0.0, 0.0, 0.0))
        with pytest.raises(MatchError):                            # no road
            f(replace(_dc(polys), steps=()), arcade, t)
        with pytest.raises(DcError):
            f(None, arcade, t)
        with pytest.raises(DcError):
            f(_dc(polys), None, t)


# ---- the disc and the arcade ----

@pytest.fixture(scope="module")
def disc(request):
    from dtools import config
    from dtools.dc.disc import DcDisc
    p = config.dc_path()
    if p is None or not p.is_file():
        pytest.skip("set DAYTONA_DC to the Dreamcast disc .cue or .7z")
    d = DcDisc.open(p)
    request.addfinalizer(d.close)
    return d


@pytest.fixture(scope="module")
def rs():
    from dtools import config
    from dtools.arcade.romset import RomSet
    d = config.rom_dir()
    if d is None or not (d / "daytona.zip").is_file():
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return RomSet.from_zip(d / "daytona.zip")


def _course(disc, rs, n):
    from dtools.dc.disc import course_files
    name = course_files(n, kind="data")
    return data.decode_data(disc.read(name), name), golden.decode_course(rs, n)


# Measured (Task 7): course n -> (RMS gate, median road cell width in arcade units). The measured RMS is 0.0 on all
# three (the road cells are the arcade's polygons with z negated, Task 5), the fit exactly scale 1, angle 0, t 0; the
# gate 1e-6 is that with room for float rounding. Phase 1's naming (n 0-2 = arcade ids 0-2) holds for all three.
MEASURED = {0: (1e-6, 16.9996), 1: (1e-6, 23.9997), 2: (1e-6, 20.0)}


def test_beginner_match(disc, rs):
    d, c = _course(disc, rs, 0)
    gate, width = MEASURED[0]
    t, res = match.fit_course(d, c, gate=gate)
    assert res["rms"] < gate <= 2.0
    assert 1.00 <= t.scale <= 1.05 and abs(math.degrees(t.angle)) < 1.0
    assert (t.scale, t.angle, t.tx, t.tz, t.flip_z) == (1.0, 0.0, 0.0, 0.0, True)          # measured: exact
    assert (res["rms"], res["max"]) == (0.0, 0.0) and res["points"] == 5 * 503
    assert match.lap_order(d, c, t) == (0, 1)                    # the Dreamcast's cell 0 is the arcade's, same way
    w_dc, w_arcade = match.road_width(d, c, t)
    assert w_arcade == pytest.approx(width, abs=1e-3)
    assert w_dc / w_arcade == pytest.approx(1.0, abs=1e-6)


def test_expert_advanced_match(disc, rs):
    """Courses 1 and 2 against arcade ids 1 and 2: the cross-check that Phase 1's geometry naming is right."""
    for n, count in ((1, 1064), (2, 818)):
        d, c = _course(disc, rs, n)
        gate, width = MEASURED[n]
        t, res = match.fit_course(d, c, gate=gate)
        assert res["rms"] < gate and res["points"] == 5 * count, n
        assert (t.scale, t.angle, t.tx, t.tz, t.flip_z) == (1.0, 0.0, 0.0, 0.0, True), n
        assert match.lap_order(d, c, t) == (0, 1), n
        w_dc, w_arcade = match.road_width(d, c, t)
        assert w_arcade == pytest.approx(width, abs=1e-3) and w_dc / w_arcade == pytest.approx(1.0, abs=1e-6), n
    # the counts alone keep the courses apart: course 1's data against arcade id 2 is refused
    d1, _ = _course(disc, rs, 1)
    with pytest.raises(MatchError, match="1064 road cells .* 818 road polygons"):
        match.fit_course(d1, golden.decode_course(rs, 2), gate=2.0)


def test_beginner_mirror_refused(disc, rs):
    """Review Focus 5 on the real course: the arcade's own cells (z not negated) offered as Dreamcast cells are the
    Beginner mirrored, and the fit refuses them, naming the RMS and the gate."""
    d, c = _course(disc, rs, 0)
    mirrored = tuple(Cell(p.corners, p.centre, 0) for p in c.ground_data.polys[:len(c.steps)])
    with pytest.raises(MatchError, match=r"RMS [0-9.]+.*gate 2") as e:
        match.fit_course(replace(d, cells=mirrored + d.cells[len(mirrored):]), c, gate=2.0)
    assert float(re.search(r"RMS ([0-9.]+)", str(e.value)).group(1)) > 100.0     # not a near miss: measured 188.1


# ---- the CS0 model against the arcade's ground (two checks ruled into Task 7) ----

def _inside(p, quads):
    """(len(p), len(quads)) bool: point p (x, z) inside convex quad (four (x, z)), either winding."""
    s = []
    for k in range(4):
        a, b = quads[:, k], quads[:, (k + 1) % 4]
        s.append((b[None, :, 0] - a[None, :, 0]) * (p[:, None, 1] - a[None, :, 1])
                 - (b[None, :, 1] - a[None, :, 1]) * (p[:, None, 0] - a[None, :, 0]))
    s = np.stack(s)
    return np.all(s <= 1e-6, axis=0) | np.all(s >= -1e-6, axis=0)


@pytest.fixture(scope="module")
def cs0_road(disc, rs):
    """CS0's road surface in the arcade's frame: the course 0 fit `t`, the arcade's road polygons' corners (n, 4, 3),
    every CS0 triangle (DC vertices, arcade vertices, the vertices' stored normals or None) and the indices of the
    road-surface triangles: centroid (after t) over an arcade road polygon and within 0.5 of that polygon's plane."""
    from dtools.dc import pol
    from dtools.dc.disc import course_files
    d, c = _course(disc, rs, 0)
    t, _ = match.fit_course(d, c, gate=MEASURED[0][0])
    quads = np.array([p.corners for p in c.ground_data.polys[:len(c.steps)]])
    name = course_files(0, kind="model")
    m = pol.decode_pol(disc.read(name), name)
    tris = list(pol.triangles(m))
    # each triangle's place in its strip, in pol.triangles' order: "even", "odd" (swapped) or "list"
    kinds = []
    for ob in m.objects:
        for me in ob.meshes:
            for st in me.strips:
                n = len(st.entries)
                if st.flags & pol.LIST_FLAG:
                    kinds += ["list"] * (n // 3)
                else:                                           # a strip of n entries: n - 2 triangles
                    kinds += ["odd" if k & 1 else "even" for k in range(n - 2)]
    assert len(kinds) == len(tris)
    dc_v = np.array([[(v.x, v.y, v.z) for v in tri[:3]] for tri in tris])
    ar_v = np.array([[t.apply(p) for p in tri] for tri in dc_v])
    normals = [np.array([v.normal for v in tri[:3]], dtype=float)
               if all(isinstance(x, float) for v in tri[:3] for x in v.normal) else None for tri in tris]
    cen = ar_v.mean(axis=1)
    ins = _inside(cen[:, [0, 2]], quads[:, :, [0, 2]])
    road = []
    for k in np.nonzero(ins.any(axis=1))[0]:
        q = quads[int(ins[k].argmax())]
        a, b, e = np.linalg.lstsq(np.c_[q[:, 0], q[:, 2], np.ones(4)], q[:, 1], rcond=None)[0]
        if abs(cen[k, 1] - (a * cen[k, 0] + b * cen[k, 2] + e)) < 0.5:
            road.append(int(k))
    return t, quads, dc_v, ar_v, normals, np.array(road), np.array(kinds)


def _edge_distance(p, quads):
    """The xz distance of each point p (x, z) to the nearest edge of the quads."""
    a = np.concatenate([quads[:, k] for k in range(4)])
    b = np.concatenate([quads[:, (k + 1) % 4] for k in range(4)])
    ab = b - a
    u = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0.0, 1.0)
    return np.sqrt(((p[:, None] - (a[None] + u[..., None] * ab[None])) ** 2).sum(-1)).min(axis=1)


def test_cs0_model_scale(cs0_road):
    """Phase 1's s = 1.01-1.04 (from the models) does not hold: through the course 0 fit (scale 1), CS0's road surface
    lies on the arcade's road polygons. Its vertices sit on the polygons' edges, a similarity fitted to those at the
    polygons' corners is scale 1, and its triangles ride a few hundredths above the arcade's ground."""
    t, quads, dc_v, ar_v, _, road, _ = cs0_road
    assert len(road) == 2034
    verts = np.unique(ar_v[road].reshape(-1, 3), axis=0)
    assert len(verts) == 2038
    on_edge = _edge_distance(verts[:, [0, 2]], quads[:, :, [0, 2]]) < 1e-3
    assert on_edge.sum() == 1802                                 # 88 %
    # at Phase 1's scale they would not be
    t103 = replace(t, scale=1.03)
    dc_verts = np.unique(dc_v[road].reshape(-1, 3), axis=0)
    at103 = np.array([t103.apply(p) for p in dc_verts])
    assert (_edge_distance(at103[:, [0, 2]], quads[:, :, [0, 2]]) < 1e-3).sum() < 0.01 * len(verts)    # measured: 2
    # the vertices at an arcade road corner (within 0.01) span the course; fitted from the Dreamcast frame: scale 1
    corners = quads.reshape(-1, 3)
    dist = np.sqrt(((verts[:, None] - corners[None]) ** 2).sum(-1))
    at = dist.min(axis=1) < 0.01
    assert at.sum() == 493 and np.ptp(verts[at, 0]) > 500.0
    src = [(x, y, -z) for x, y, z in verts[at]]                 # back to the Dreamcast frame (t is z negation only)
    s, res = match.fit_points(src, corners[dist.argmin(axis=1)[at]], gate=0.01)
    assert abs(s.scale - 1.0) < 1e-5 and abs(math.degrees(s.angle)) < 1e-4 and res["rms"] < 0.002     # 1.2e-3
    # height: the model's road over the arcade's ground (centroid less the polygon's plane), median +0.04
    cen = ar_v[road].mean(axis=1)
    ins = _inside(cen[:, [0, 2]], quads[:, :, [0, 2]]).argmax(axis=1)
    dy = []
    for k, j in enumerate(ins):
        q = quads[j]
        a, b, e = np.linalg.lstsq(np.c_[q[:, 0], q[:, 2], np.ones(4)], q[:, 1], rcond=None)[0]
        dy.append(cen[k, 1] - (a * cen[k, 0] + b * cen[k, 2] + e))
    assert 0.03 < float(np.median(dy)) < 0.05 and float(np.min(dy)) > -0.08


def test_cs0_strip_winding(cs0_road):
    """pol.triangles' winding (odd strip triangles swap their first two vertices) on CS0's road surface: after the
    course 0 transform (a mirror: z negated), the face normal (b - a) x (c - a) agrees with the vertices' stored
    normals (transformed alike) on 94 % and points up on 91 %, even and odd triangles alike. In the Dreamcast's own
    frame the same normals are the other way round (6 % agree): seen from the side its stored normal points to, a
    triangle runs clockwise."""
    t, _, dc_v, ar_v, normals, road, kinds = cs0_road
    assert all(normals[k] is not None for k in road)            # no raw-normal mesh on the road
    turn = Transform(1.0, t.angle, 0.0, 0.0)                     # a direction: mirrored and turned, not moved

    def counts(v, flip, sel):
        face = np.cross(v[sel, 1] - v[sel, 0], v[sel, 2] - v[sel, 0])
        stored = np.array([np.mean([turn.apply(n) if flip else n for n in normals[k]], axis=0) for k in sel])
        return int(((face * stored).sum(axis=1) > 0).sum()), int((face[:, 1] > 0).sum())

    assert counts(ar_v, True, road) == (1918, 1856)              # of 2034: 94.3 % agree, 91.2 % up
    assert counts(dc_v, False, road)[0] == 2034 - 1918           # the Dreamcast frame: reversed
    # the strip alternation is right: even and odd (swapped) triangles agree alike; a wrong swap would split them
    even, odd = road[kinds[road] == "even"], road[kinds[road] == "odd"]
    assert (len(even), len(odd)) == (1046, 982)
    assert (counts(ar_v, True, even)[0], counts(ar_v, True, odd)[0]) == (988, 925)


def _nearest(p: np.ndarray, q: np.ndarray, tol: float) -> np.ndarray:
    """For each point of p (n, 3), its distance to the nearest point of q within `tol` in x (inf when none)."""
    qs = q[np.argsort(q[:, 0])]
    lo = np.searchsorted(qs[:, 0], p[:, 0] - tol)
    hi = np.searchsorted(qs[:, 0], p[:, 0] + tol, "right")
    out = np.full(len(p), np.inf)
    for i in np.nonzero(hi > lo)[0]:
        out[i] = np.sqrt(((qs[lo[i]:hi[i]] - p[i]) ** 2).sum(axis=1)).min()
    return out


def test_course3_model_and_data_share_the_frame(disc):
    """M3 converts course 3 (Desert City), which has no arcade twin, so the CS0-only frame pin is repeated on it: CS3's
    and VCS3's vertices lie on CS_DATA3's road-cell corners (< 0.01, the Dreamcast frame, no transform) all over the
    course, and none do with z negated."""
    from dtools.dc.disc import course_files
    name = course_files(3, kind="data")
    d = data.decode_data(disc.read(name), name)
    corners = np.unique(np.array([c for cell in data.road_cells(d) for c in cell.corners]), axis=0)
    assert len(corners) == 1056
    for kind, on, within_1e4 in (("model", 659, 488), ("model_v", 574, 472)):
        mname = course_files(3, kind=kind)
        m = pol.decode_pol(disc.read(mname), mname)
        verts = np.unique(np.array([(e.x, e.y, e.z) for ob in m.objects for me in ob.meshes for st in me.strips
                                    for e in st.entries if isinstance(e, Vertex)]), axis=0)
        dist = _nearest(verts, corners, 0.01)
        hit = dist < 0.01
        assert (int(hit.sum()), int((dist < 1e-4).sum())) == (on, within_1e4), mname    # of 16,895 / 10,350
        assert np.ptp(verts[hit, 0]) > 700.0 and np.ptp(verts[hit, 2]) > 1100.0              # the whole course
        assert (_nearest(verts * np.array([1.0, 1.0, -1.0]), corners, 0.01) < 0.01).sum() == 0


def test_course3_line_heading_rule(disc):
    """The heading rule (arcade = Dreamcast - 0x1fff) on course 3, whose headings the Dreamcast's tool recomputed:
    every AI line point's heading, less 0x1fff, is the direction to the next point in the arcade's convention
    (0x4000 per turn; travel (x, z) along (-sin, cos); the arcade's z is the Dreamcast's negated) within 2.2 units,
    0.05 degrees; the mean residual is about +1 unit (1/16384 turn), so the tool's own offset was 0x2000."""
    from dtools.dc.disc import course_files
    name = course_files(3, kind="data")
    d = data.decode_data(disc.read(name), name)
    turn = 0x4000
    res = []
    for line in d.lines:
        a = np.array([(p.x, -p.z, p.heading) for p in line], dtype=float)       # arcade x, z; Dreamcast heading
        dx, dz = np.diff(a[:, 0]), np.diff(a[:, 1])
        geo = np.arctan2(-dx, dz) / (2.0 * math.pi) * turn
        res.append((a[:-1, 2] - 0x1FFF - geo + turn / 2) % turn - turn / 2)
    r = np.concatenate(res)
    assert len(r) == 8 * (len(d.lines[0]) - 1)
    assert float(np.abs(r).max()) < 2.2                         # measured 2.12 (0.047 degrees)
    assert 0.85 < float(r.mean()) < 1.0                         # measured 0.92


# ---- the M (mirror) files against the mirror rules (Task 8) ----

DC_CELL_ORDER = (1, 0, 3, 2)


def _s16(v: int) -> int:
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def _raw(x, y, z):
    """A raw-normal vertex's three words: (x, y, z) as signed bytes in word 0 (x << 16 | y << 8 | z), a colour twice."""
    return ((x & 0xFF) << 16 | (y & 0xFF) << 8 | (z & 0xFF), 0xFFB3B3B2, 0xFFB3B3B2)


def _plain_model() -> Model:
    """One object: a strip of one triangle (float normals, strip flags 0x92: cull bits 2) and a triangle list of one
    (raw normals, flags 0x19: cull bits 1)."""
    def V(x, y, z, n):
        return Vertex(x, y, z, n, 0.25, 0.5)
    strip = Strip(0x92, (V(1.0, 2.0, -3.0, (0.0, 0.6, 0.8)), V(4.0, 2.5, 1.0, (0.0, 1.0, 0.0)),
                         V(2.0, 0.0, 5.0, (0.6, 0.0, -0.8))))
    tri = Strip(0x19, (V(7.0, 1.0, 2.0, _raw(0, 127, 0)), V(8.0, 1.0, -6.0, _raw(90, 0, -90)),
                       V(9.0, 3.0, 4.0, _raw(0, 90, 90))))
    mesh_a = Mesh((0x8000002C, 0x83000000, 0x2008216D, 0xCC000000), (0.0, 1.0, 2.0, 3.0), 0, 7, (1.0,) * 9, (strip,))
    mesh_b = Mesh((0x8000001C, 0x83000000, 0x2008A152, 0xCC000000), (0.0, 0.0, 0.0, 1.0), -1, -3, (1.0,) * 9, (tri,))
    obj = Object(3, (0.5, 0.25, 0.125, 9.0), (mesh_a, mesh_b), 6)
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    return pol.fix_header(Model(hdr, (obj,), (TexEntry(64, 64, 0x101, 0, 0),)))


def _plain_data() -> CourseData:
    """Four non-planar cells (two road), three road points (one with an asymmetric lane range), eight lines of nine
    points whose every field varies, and a grid whose lists sit on rows 10, 60 and 127."""
    def lp(k, i):
        return LinePoint(10.0 * k + i, 0.5 * i, 3.0 * k - 2.0 * i + 0.25, 1.25 + 0.125 * k, 27 - 5 * i,
                         0x0AAA + 7 * k - 3 * i, 4 + k - i, i - 2 * k, 0x180 + 0x40 * k + i, i % 2)

    def lane(i, j):
        return Lane(5.0 * i + j, 0.25 * j, 2.0 * j - i + 0.5, 3 * j - i, 0x0AA9 + 11 * j + i, 13 * j - 5 * i, 0)
    road = tuple(RoadPoint(lo, hi, curve, tuple(lane(i, j) for j in range(8)))
                 for i, (lo, hi, curve) in enumerate(((1, 6, 0x111), (0, 6, -0x222), (1, 7, 0x333))))
    cells = []
    for c in range(4):
        corners = ((0.0, 0.5 * c, 4.0 * c), (6.0, 0.25, 4.0 * c + 1.0), (6.5, 0.75, 4.0 * c + 4.0),
                   (0.5, 1.0 * c, 4.0 * c + 3.5))
        cells.append(Cell(corners, (3.25, 0.5, 4.0 * c + 2.125), (0x2AAB, 0x9000, 0x0001, 0xFFFF)[c]))
    grid = [0] * data.GRID_CELLS
    grid[10 * 128 + 5], grid[60 * 128 + 7], grid[127 * 128 + 0] = 1, 2, 3
    return CourseData((0, 4, 5, 3, 1, 6, 7, 2), tuple(tuple(lp(k, i) for i in range(9)) for k in range(8)),
                      tuple(range(8)), road, bytes([1, 1, 5, 8]), (Step(0, 1), Step(1, 0x42)), tuple(cells),
                      tuple(grid), ((), (0, 1), (2,), (3, 0)))


def _mirror(plain, *, c=0.0, corners=DC_CELL_ORDER, lines=lambda k: 7 - k, heading=0x1FFF, lane=0x700,
            speed=16 / 15, slots=True, lane_range="kept", cull=True, winding="kept", roll=None):
    """A mirrored pair built by hand: world z -> c - z (normals -z, a raw normal's z byte negated), cells' corners in
    `corners`, direction 0x8000 - d, grid row r -> 127 - r, M line k = plain line lines(k) with heading `heading` -
    h, lane `lane` - lane, roll and steer negated (roll: a function of the plain roll instead), speed scaled, pitch and
    step kept; the road's slots reversed (or kept) with each slot's attitude from the slot it takes its position from,
    lane_lo and lane_hi kept or reversed, curve negated; the strips' cull bits 2 and 3 swapped (`cull`), the triangles'
    vertex order kept or reversed."""
    model, d = plain

    def z(v):
        return float(np.float32(c - v)) if c else -v

    def w(p):
        return (p[0], p[1], z(p[2]))
    roll = roll or (lambda r: -r)

    def vertex(v, raw):
        n = v.normal
        if raw:
            b = n[0] & 0xFF
            n = ((n[0] & ~0xFF) | (-(b - 256 if b & 0x80 else b) & 0xFF), n[1], n[2])
        else:
            n = (n[0], n[1], -n[2])
        return replace(v, z=z(v.z), normal=n)
    objects = []
    for ob in model.objects:
        meshes = []
        for me in ob.meshes:
            raw = pol._raw_normal(me.params)
            strips = []
            for st in me.strips:
                ents = tuple(vertex(e, raw) for e in st.entries)
                flags = st.flags ^ 1 if cull and st.flags & 3 in (2, 3) else st.flags
                strips.append(Strip(flags, ents[::-1] if winding == "reversed" else ents))
            meshes.append(replace(me, strips=tuple(strips)))
        objects.append(replace(ob, meshes=tuple(meshes)))
    m = pol.fix_header(replace(model, objects=tuple(objects)))
    cells = tuple(Cell(tuple(w(cl.corners[k]) for k in corners), w(cl.centre), (0x8000 - cl.direction) & 0xFFFF)
                  for cl in d.cells)
    grid = tuple(d.grid[(127 - k // 128) * 128 + k % 128] for k in range(data.GRID_CELLS))

    def point(p):
        return replace(p, z=z(p.z), speed=float(np.float32(p.speed * speed)), heading=_s16(heading - p.heading),
                       roll=_s16(roll(p.roll)), steer=_s16(-p.steer), lane=_s16(lane - p.lane))
    ai = tuple(tuple(point(p) for p in d.lines[lines(k)]) for k in range(8))

    def road_point(r):
        lo, hi = (r.lane_lo, r.lane_hi) if lane_range == "kept" else (7 - r.lane_hi, 7 - r.lane_lo)
        src = [r.lanes[7 - k if slots else k] for k in range(8)]
        return RoadPoint(lo, hi, _s16(-r.curve), tuple(
            Lane(s.x, s.y, z(s.z), s.pitch, _s16(heading - s.heading), _s16(roll(s.roll)), s.pad) for s in src))
    return m, replace(d, cells=cells, grid=grid, lines=ai, road=tuple(road_point(r) for r in d.road))


def _rules(r) -> dict:
    """Every rule the report names, flat: 'family.field' -> rule (or order)."""
    out = {}
    for fam, v in r.items():
        for k, x in v.items():
            if isinstance(x, dict) and "rule" in x:
                out[f"{fam}.{k}"] = x["rule"]
            elif k in ("order", "slots"):
                out[f"{fam}.{k}"] = x
    return out


DC_RULES = {
    "model.cull": "swap 2, 3",
    "cells.order": DC_CELL_ORDER, "cells.direction": "0x8000 - v", "cells.grid": "127 - r", "cells.steps": "kept",
    "cells.attributes": "kept",
    "lines.order": "reversed", "lines.speed": "x 1.06667", "lines.pitch": "kept", "lines.heading": "0x1fff - v",
    "lines.roll": "negated", "lines.steer": "negated", "lines.lane": "0x700 - v", "lines.step": "kept",
    "road.slots": "reversed", "road.lane_range": "kept", "road.curve": "negated", "road.pitch": "kept",
    "road.heading": "0x1fff - v", "road.roll": "negated",
}


def test_mirror_report_synthetic():
    """A pair built under the Dreamcast's rules, and one under the arcade's (Phase 4): each rule named, the plane
    measured, every point exact under it."""
    plain = (_plain_model(), _plain_data())
    r = match.mirror_report(plain, _mirror(plain))
    assert _rules(r) == DC_RULES
    assert r["model"] == {"vertices": 6, "exact": 6, "positions": 6, "plane": 0.0, "triangles": 2,
                          "winding": {"kept": 2, "reversed": 0, "unmatched": 0, "degenerate": 0},
                          "cull": {"rule": "swap 2, 3", "pairs": {(1, 1): 1, (2, 3): 1}, "strips": 2,
                                   "other_bits": 0}}
    cells = r["cells"]
    assert (cells["sizes"], cells["exact"], cells["plane"], cells["offset"]) == ((4, 4), 4, 0.0, 0.0)
    assert cells["direction"] == {"rule": "0x8000 - v", "near": "0x8000 - v", "exact": 4, "count": 4, "max_error": 0,
                                 "high_kept": 4}
    assert cells["grid"] == {"rule": "127 - r", "exact": data.GRID_CELLS, "count": data.GRID_CELLS}
    lines = r["lines"]
    assert (lines["map"], lines["points"], lines["exact"], lines["offset"], lines["plane"]) == (
        (7, 6, 5, 4, 3, 2, 1, 0), 72, 72, 0.0, 0.0)
    assert lines["heading"]["exact"] == 72 and lines["speed"]["exact"] == 72
    road = r["road"]
    assert (road["sizes"], road["points"], road["exact"], road["offset"], road["plane"]) == ((3, 3), 20, 20, 0.0, 0.0)
    assert road["lane_range"] == {"rule": "kept", "count": 3, "asymmetric": 2, "kept": 3, "reversed": 1}

    # the arcade's rules: plane z = -0.5 (z -> -1 - z), ground quads (0, 3, 2, 1), lines in place, lane 0x6ff - lane,
    # heading 0x2000 - h, speed kept, lane_lo and lane_hi reversed, no cull bits (vertex order kept)
    r = match.mirror_report(plain, _mirror(plain, c=-1.0, corners=(0, 3, 2, 1), lines=lambda k: k, heading=0x2000,
                                           lane=0x6FF, speed=1.0, lane_range="reversed", cull=False))
    assert _rules(r) == {**DC_RULES, "model.cull": "kept", "cells.order": (0, 3, 2, 1), "lines.order": "kept",
                         "lines.speed": "kept", "lines.heading": "0x2000 - v", "lines.lane": "0x6ff - v",
                         "road.lane_range": "reversed", "road.heading": "0x2000 - v"}
    assert [r[f]["plane"] for f in ("model", "cells", "lines", "road")] == [-1.0] * 4
    assert (r["model"]["exact"], r["cells"]["exact"], r["lines"]["exact"], r["road"]["exact"]) == (6, 4, 72, 20)
    assert r["model"]["winding"]["kept"] == 2


def test_mirror_report_rule_variants():
    """The other names: lines rotated by one, slots kept, a recomputed roll ('other', with the nearest rule and its
    error), triangles wound the other way, a lane range neither kept nor reversed, z not mirrored at all."""
    plain = (_plain_model(), _plain_data())
    m = _mirror(plain, lines=lambda k: (k + 1) % 8, slots=False, winding="reversed",
                roll=lambda v: -v + int(v % 4 == 0))
    r = match.mirror_report(plain, m)
    assert (r["lines"]["order"], r["lines"]["map"]) == ("rotated", (1, 2, 3, 4, 5, 6, 7, 0))
    assert r["road"]["slots"] == "kept"
    roll = r["lines"]["roll"]
    assert (roll["rule"], roll["near"], roll["max_error"]) == ("other", "negated", 1) and roll["exact"] < roll["count"]
    assert r["model"]["winding"] == {"kept": 0, "reversed": 2, "unmatched": 0, "degenerate": 0}
    d = m[1]
    odd = replace(d, road=(replace(d.road[0], lane_lo=2),) + d.road[1:])
    assert match.mirror_report(plain, (m[0], odd))["road"]["lane_range"]["rule"] == "other"
    # all-zero fields are named so; an unpaired line (another length) leaves its order 'other', the offset infinite
    zero = tuple(tuple(replace(p, steer=0) for p in line) for line in plain[1].lines)
    pz = (plain[0], replace(plain[1], lines=zero))
    assert match.mirror_report(pz, _mirror(pz))["lines"]["steer"]["rule"] == "zero"
    short = _mirror(plain)
    short = (short[0], replace(short[1], lines=short[1].lines[:7] + (short[1].lines[7][:8],)))
    r = match.mirror_report(plain, short)
    assert r["lines"]["order"] == "other" and r["lines"]["offset"] == math.inf and r["lines"]["points"] == 71
    # the plain pair against itself: nothing mirrored (z kept: no plane fits every point, no strip pairs with one)
    r = match.mirror_report(plain, plain)
    assert (r["cells"]["exact"], r["model"]["exact"], r["model"]["cull"]["rule"]) == (0, 0, "none")
    assert r["lines"]["exact"] == 0 and r["road"]["exact"] == 0


def test_mirror_report_refuses():
    plain = (_plain_model(), _plain_data())
    m = _mirror(plain)
    for a, b in [(None, m), (plain, None), (plain[:1], m), (plain + (1,), m), ((plain[1], plain[0]), m),
                 ("ab", m), (plain, (m[0], None)), (plain, (True, m[1])), ([plain[0], 1], m)]:
        with pytest.raises(DcError):
            match.mirror_report(a, b)
    bad_v = replace(plain[0].objects[0].meshes[0].strips[0].entries[0], x=math.nan)
    bad_model = replace(plain[0], objects=(replace(plain[0].objects[0], meshes=(replace(
        plain[0].objects[0].meshes[0], strips=(Strip(0x92, (bad_v,) + plain[0].objects[0].meshes[0].strips[0]
                                                    .entries[1:]),)),) + plain[0].objects[0].meshes[1:]),))
    bad_line = replace(plain[1], lines=((replace(plain[1].lines[0][0], heading=True),) + plain[1].lines[0][1:],)
                       + plain[1].lines[1:])
    for bad in [(bad_model, plain[1]), (plain[0], bad_line), (plain[0], replace(plain[1], cells=None)),
                (plain[0], replace(plain[1], grid=(0,) * 3))]:
        with pytest.raises(DcError):
            match.mirror_report(bad, m)
        with pytest.raises(DcError):
            match.mirror_report(plain, bad)
    # numpy numbers are taken as the encoders take them
    npd = replace(plain[1], lines=tuple(tuple(replace(p, heading=np.int16(p.heading), x=np.float32(p.x)) for p in line)
                                        for line in plain[1].lines))
    assert _rules(match.mirror_report((plain[0], npd), m)) == DC_RULES


def _side(disc, n, v):
    from dtools.dc.disc import course_files
    model, name = course_files(n, v, "model"), course_files(n, v, "data")
    return pol.decode_pol(disc.read(model), model), data.decode_data(disc.read(name), name)


# Measured (Task 8): plain -> M on the disc, through `match.mirror_report`. The rules (`_rules`) of courses 0-2 (the
# arcade's courses), then each course's differences; courses 3-7 recompute most line and road angles (OTHER).
M_RULES = {
    "model.cull": "swap 2, 3",
    "cells.order": DC_CELL_ORDER, "cells.direction": "0x8000 - v", "cells.grid": "127 - r", "cells.steps": "kept",
    "cells.attributes": "kept",
    "lines.order": "reversed", "lines.speed": "x 1.06667", "lines.pitch": "kept", "lines.heading": "0x1fff - v",
    "lines.roll": "negated", "lines.steer": "negated", "lines.lane": "0x700 - v", "lines.step": "kept",
    "road.slots": "reversed", "road.lane_range": "symmetric", "road.curve": "negated", "road.pitch": "kept",
    "road.heading": "0x1fff - v", "road.roll": "negated",
}
OTHER = {**{f"lines.{f}": "other" for f in ("speed", "heading", "roll", "steer", "lane")},
         **{f"road.{f}": "other" for f in ("curve", "heading", "roll")}}
M_DIFF = {
    0: {"cells.direction": "other"},                            # 1,176 of 1,184
    1: {"road.lane_range": "kept"},                             # 38 asymmetric ranges kept, not reversed
    2: {},
    3: {**OTHER, "cells.direction": "other"},
    4: {**OTHER, "lines.order": "rotated", "lines.pitch": "other", "road.pitch": "other"},
    5: {**OTHER, "lines.order": "rotated", "lines.steer": "zero"},
    6: {**OTHER, "lines.order": "rotated", "cells.direction": "other"},     # 3,237 of 3,238
    7: {**OTHER, "lines.order": "rotated"},
}
# course -> (model: vertices, exact, positions, triangles kept / reversed / unmatched; cells: exact, count, direction
# exact; lines: exact, points; road: exact, points)
M_COUNTS = {
    0: ((11867, 10636, 11279, 8647, 0, 312), (1182, 1184, 1176), (1568, 1568), (1176, 1176)),
    1: ((62842, 61776, 62578, 48232, 0, 187), (7235, 7239, 7239), (7928, 7928), (6618, 6618)),
    2: ((34761, 34643, 34761, 28800, 0, 0), (2984, 2986, 2986), (3896, 3896), (3494, 3494)),
    3: ((26389, 26389, 26389, 22800, 0, 0), (763, 3073, 128), (3720, 3720), (0, 2790)),
    4: ((30376, 28836, 29208, 22371, 4, 831), (2169, 2169, 2169), (3352, 3408), (0, 2556)),
    5: ((33998, 32548, 33933, 24941, 8, 92), (874, 874, 874), (1040, 1040), (0, 780)),
    6: ((31371, 30216, 30725, 23711, 0, 500), (3238, 3238, 3237), (4000, 4000), (0, 3000)),
    7: ((33402, 33216, 33402, 25501, 0, 0), (2718, 2718, 2718), (4000, 4000), (0, 3000)),
}
LINE_OFFSET = 0.81          # the lines' largest positional offset, frozen from the measurement: 0.807 (course 4)
ROAD_OFFSET = 0.016         # the road's on courses 3-7: 0.0159 (course 4); 0.0 on 0-2

# R -> MR: the same report; rules as courses 0-2's, then each course's differences
MR_RULES = {**M_RULES, "lines.order": "kept", "lines.speed": "kept", "lines.heading": "other", "road.slots": "kept",
            "road.curve": "0x8000 - v", "road.heading": "0x2000 - v"}
MR_OTHER = {**OTHER, "lines.order": "rotated", "road.slots": "reversed"}
MR_DIFF = {
    0: {"cells.direction": "other"}, 1: {"road.lane_range": "kept"}, 2: {},
    3: {**MR_OTHER, "cells.direction": "other"},
    4: {**MR_OTHER, "lines.pitch": "other", "road.pitch": "other"},
    5: {**MR_OTHER, "lines.pitch": "other", "lines.steer": "zero"},
    6: {**MR_OTHER, "lines.pitch": "other"}, 7: {**MR_OTHER, "lines.pitch": "other"},
}
MR_ROAD_OFFSET = 0.1        # 0.098 (course 7); 0.0 on 0-2


def _sums(p, m, period):
    """(p + m) modulo `period`, centred on 0x2000: a heading pair's distance from the rule 0x2000 - h."""
    return [(a + b - 0x2000 + period // 2) % period - period // 2 for a, b in zip(p, m)]


def test_mirror_files_all(disc):
    """Every M file pair (model, data) against its plain pair: each family's rule named and pinned per course, the
    plane z = 0 (z -> -z) everywhere, the lines exact once paired in the measured order."""
    for n in range(8):
        plain, mir = _side(disc, n, ""), _side(disc, n, "M")
        r = match.mirror_report(plain, mir)
        assert _rules(r) == {**M_RULES, **M_DIFF[n]}, n
        assert [r[f]["plane"] for f in ("model", "cells", "lines", "road")] == [0.0] * 4, n
        m, c, ln, rd = r["model"], r["cells"], r["lines"], r["road"]
        w = m["winding"]
        assert ((m["vertices"], m["exact"], m["positions"], w["kept"], w["reversed"], w["unmatched"]),
                (c["exact"], c["count"], c["direction"]["exact"]), (ln["exact"], ln["points"]),
                (rd["exact"], rd["points"])) == M_COUNTS[n], n
        assert c["sizes"] == (c["count"],) * 2 and rd["sizes"][0] == rd["sizes"][1] and m["cull"]["other_bits"] <= 13
        assert ln["offset"] <= LINE_OFFSET <= 10.0 and (ln["offset"] == 0.0) == (n != 4), n
        assert rd["offset"] == 0.0 if n < 3 else 0.0 < rd["offset"] <= ROAD_OFFSET, n
        if n >= 3:
            # the recomputed headings: within 2 of the rule 0x2000 - h (modulo a turn), the AI lines of course 4 aside
            lp = [(p, q) for k, j in enumerate(ln["map"]) for p, q in zip(plain[1].lines[j], mir[1].lines[k])]
            rp = [(a.lanes[7 - k], b.lanes[k]) for a, b in zip(plain[1].road, mir[1].road)
                  for k in range(b.lane_lo, b.lane_hi + 1)]
            for pairs, bound in ((rp, 2), (lp, 181 if n == 4 else 2)):
                err = _sums([p.heading for p, _ in pairs], [q.heading for _, q in pairs], match.HEADING_TURN)
                assert max(map(abs, err)) == bound, n
    # courses 0-2: each slot's attitude comes from the plain slot its position comes from (7 - k); the arcade's rule
    # (`arcade.mirror.mirror_road_point`: slot min(8 - k, lane_hi)) gives another on these used slots
    differ = []
    for n in range(3):
        d = _side(disc, n, "")[1]
        differ.append(sum(
            (r.lanes[7 - k].pitch, r.lanes[7 - k].roll) != (r.lanes[min(8 - k, r.lane_hi)].pitch,
                                                            r.lanes[min(8 - k, r.lane_hi)].roll)
            for r in d.road for k in range(7 - r.lane_hi, 8 - r.lane_lo)))
    assert differ == [279, 597, 571]
    # Phase 1's "up to 10 units apart": line k against plain line k, not the mirrored line 7 - k
    plain, mir = _side(disc, 0, "")[1], _side(disc, 0, "M")[1]
    far = max(math.dist((p.x, p.y, -p.z), (q.x, q.y, q.z)) for k in range(8)
              for p, q in zip(plain.lines[k], mir.lines[k]))
    assert 10.0 < far < 11.0                                    # measured 10.89


def test_mirror_files_mr(disc):
    """The MR files against the R files, the same way: positions exact in the measured line order on every course."""
    for n in range(8):
        r = match.mirror_report(_side(disc, n, "R"), _side(disc, n, "MR"))
        assert _rules(r) == {**MR_RULES, **MR_DIFF[n]}, n
        assert [r[f]["plane"] for f in ("model", "cells", "lines", "road")] == [0.0] * 4, n
        assert r["lines"]["exact"] == r["lines"]["points"] and r["lines"]["offset"] == 0.0, n
        assert r["road"]["offset"] == 0.0 if n < 3 else r["road"]["offset"] <= MR_ROAD_OFFSET, n


def test_mirror_report_far_plane():
    """Coordinates near the f32 range: where c - z leaves f32 the point matches nothing, never an error (warnings as
    errors included)."""
    plain = (_plain_model(), _plain_data())
    m = _mirror(plain, c=3e38)                                  # every mirrored z near 3e38: the plane offset 3e38
    ob = plain[0].objects[0]
    st = ob.meshes[0].strips[0]
    far = pol.fix_header(replace(plain[0], objects=(replace(ob, meshes=(replace(ob.meshes[0], strips=(replace(
        st, entries=(replace(st.entries[0], z=-3e38),) + st.entries[1:]),)),) + ob.meshes[1:]),)))
    z = float(np.float32(3e38))
    lines = tuple(tuple(replace(p, z=-z if i % 2 else z) for i, p in enumerate(line)) for line in plain[1].lines)
    up = tuple(tuple(replace(p, z=z) for p in line) for line in plain[1].lines)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = match.mirror_report((far, plain[1]), m)             # 3e38 - (-3e38) leaves f32
        assert r["model"]["plane"] == pytest.approx(3e38) and (r["model"]["exact"], r["model"]["positions"]) == (5, 5)
        r = match.mirror_report((plain[0], replace(plain[1], lines=lines)), (plain[0], replace(plain[1], lines=up)))
        # five of a line's nine z' + z are 6e38: c = 6e38, and c - (-3e38) leaves f32 on the other four
        assert r["lines"]["plane"] == pytest.approx(6e38) and (r["lines"]["exact"], r["lines"]["points"]) == (40, 72)
