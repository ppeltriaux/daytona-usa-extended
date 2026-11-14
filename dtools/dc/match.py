"""The Dreamcast courses matched to the arcade's: the transform from a Dreamcast course's frame to the arcade's.

The fit (`fit_course`) is a least-squares similarity on corresponding road cells: the Dreamcast's road cell k (its four
corners and its centre, `data.road_cells`) against the arcade's ground road polygon k (`golden.Course.ground_data`,
the first len(steps) polygons), both in step order. The Dreamcast's z is the arcade's negated, and the fit applies that
negation explicitly before it fits a scale, a rotation about y and a translation; it never fits a reflection, so a
course mirrored the wrong way cannot pass a loose residual (Review Focus 5). A fit whose RMS is over its gate is refused
(`MatchError`, naming both), never returned.

Measured (Phase 5, Task 7; docs/spike/2026-11-dc-decode.md, "Match"): courses 0-2 fit arcade ids 0-2 at scale 1,
angle 0, translation 0, residual 0; the CS0 model's road surface lies in the same frame at scale 1.

`mirror_report` (Task 8) sets a mirrored course (the `M` and `MR` files) against its plain one family by family: the
mirror plane, the corner, line and lane-slot orders, the strips' winding and cull bits, and each field's rule. What it
measured, and where the Dreamcast's mirror agrees with the arcade's (`dtools/arcade/mirror.py`) and where it differs:
docs/spike/2026-11-dc-decode.md and docs/formats/dreamcast-disc.md, "Mirror files"."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dtools.arcade import golden, tgp
from dtools.arcade.tablebase import _seq, is_number
from dtools.dc import data, pol
from dtools.dc.errors import DcError


class MatchError(DcError):
    """A fit refused: its residual is over the gate (named, with the RMS), or the courses do not correspond."""


def _short(v) -> str:
    r = repr(v)
    return r if len(r) <= 40 else r[:37] + "..."


@dataclass(frozen=True)
class Transform:
    """Dreamcast point -> arcade point: z negated first (when flip_z), then scaled by `scale` (y too, with no offset),
    turned by `angle` radians about y (x' = cos a x - sin a z, z' = sin a x + cos a z) and moved by (tx, 0, tz)."""
    scale: float
    angle: float
    tx: float
    tz: float
    flip_z: bool = True

    def __post_init__(self) -> None:
        for name in ("scale", "angle", "tx", "tz"):
            v = getattr(self, name)
            if not is_number(v):
                raise DcError(f"transform: {name} {_short(v)} is not a finite number")
            object.__setattr__(self, name, float(v))
        if not self.scale > 0.0:
            raise DcError(f"transform: scale {self.scale!r} is not positive")
        if not isinstance(self.flip_z, bool):
            raise DcError(f"transform: flip_z {_short(self.flip_z)} is not a bool")

    def apply(self, p) -> tuple[float, float, float]:
        """The arcade point of Dreamcast point p, three finite numbers (bool refused); DcError otherwise."""
        x, y, z = _point(p, "transform: the point")
        out = tuple(float(v) for v in self._apply(np.array([[x, y, z]]))[0])
        if not all(map(math.isfinite, out)):
            raise DcError(f"transform: the point {_short(p)} goes past the float range ({out!r})")
        return out

    def _apply(self, pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], -pts[:, 2] if self.flip_z else pts[:, 2]
        c, s = math.cos(self.angle), math.sin(self.angle)
        k = self.scale
        with np.errstate(all="ignore"):         # an overflow is an inf, checked by the callers
            return np.stack([k * (c * x - s * z) + self.tx, k * y, k * (s * x + c * z) + self.tz], axis=1)


def _point(p, what: str) -> tuple[float, float, float]:
    if isinstance(p, np.ndarray):
        p = p.tolist() if p.ndim == 1 else None
    if not isinstance(p, (tuple, list)) or len(p) != 3 or not all(is_number(v) for v in p):
        raise DcError(f"{what} {_short(p)} is not three finite numbers")
    return float(p[0]), float(p[1]), float(p[2])


def _points(ps, what: str) -> np.ndarray:
    """(n, 3) float64 of a sequence of points (or an (n, 3) array); DcError for anything else."""
    if isinstance(ps, np.ndarray):
        if ps.ndim != 2 or ps.shape[1] != 3 or ps.dtype.kind not in "iuf":
            raise DcError(f"{what}: an array of shape {ps.shape} ({ps.dtype}) is not (n, 3) numbers")
        out = ps.astype(np.float64)
        if not np.isfinite(out).all():
            raise DcError(f"{what}: a point is not finite")
        return out
    if not isinstance(ps, (tuple, list)):
        raise DcError(f"{what}: {_short(ps)} is not a sequence of points")
    return np.array([_point(p, f"{what}: point {k}") for k, p in enumerate(ps)], dtype=np.float64).reshape(-1, 3)


def _gate(gate) -> float:
    if not is_number(gate) or not gate > 0:
        raise DcError(f"match: the gate {_short(gate)} is not a positive finite number")
    return float(gate)


def fit_points(src, dst, gate: float, flip_z: bool = True) -> tuple[Transform, dict]:
    """The least-squares Transform taking Dreamcast points `src` to arcade points `dst` (corresponding, same count):
    z negated first when `flip_z`, then a scale and rotation about y fitted in (x, z) without reflection, y scaled.
    Returns it with the residuals (arcade units): `rms` (3D), `max` (3D), `rms_x`, `rms_y`, `rms_z` and `points`.
    MatchError when the RMS is over `gate` (naming both), or the points have no spread to fit; DcError for bad input."""
    g = _gate(gate)
    if not isinstance(flip_z, bool):
        raise DcError(f"match: flip_z {_short(flip_z)} is not a bool")
    a, b = _points(src, "match: the Dreamcast points"), _points(dst, "match: the arcade points")
    if len(a) != len(b) or not len(a):
        raise MatchError(f"match: {len(a)} Dreamcast points against {len(b)} arcade points")
    m = a * (1.0, 1.0, -1.0) if flip_z else a
    with np.errstate(all="ignore"):             # points near the float range overflow: refused below, as inf / nan
        p, q = m[:, [0, 2]], b[:, [0, 2]]
        mp, mq = p.mean(axis=0), q.mean(axis=0)
        u, v = p - mp, q - mq
        norm = float((u * u).sum())
        # the similarity (x, z) -> (c x - s z, s x + c z) + t, c = k cos a, s = k sin a, by least squares
        c = float((u * v).sum()) / norm if norm > 0.0 else 0.0
        s = float((u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()) / norm if norm > 0.0 else 0.0
        tx, tz = (float(w) for w in mq - np.array([c * mp[0] - s * mp[1], s * mp[0] + c * mp[1]]))
    if not all(map(math.isfinite, (norm, c, s, tx, tz))):
        raise MatchError(f"match: the {len(a)} points overflow the fit (coordinates near the float range)")
    if not norm > 1e-12 * len(a):
        raise MatchError(f"match: the {len(a)} Dreamcast points have no spread in x and z, nothing to fit")
    k, angle = math.hypot(c, s), math.atan2(s, c)
    if not 0.0 < k < math.inf:
        raise MatchError(f"match: the best fit has scale {k!r} (the points do not correspond); not under the gate "
                         f"{g:g}")
    t = Transform(k, angle, tx, tz, flip_z)
    with np.errstate(all="ignore"):
        r = t._apply(a) - b
        d = np.sqrt((r * r).sum(axis=1))
        res = {"rms": float(np.sqrt((d * d).mean())), "max": float(d.max()), "points": len(a)}
        for i, axis in enumerate("xyz"):
            res[f"rms_{axis}"] = float(np.sqrt((r[:, i] ** 2).mean()))
    if not res["rms"] <= g:
        raise MatchError(f"match: RMS {res['rms']:.6g} (max {res['max']:.6g}) over {len(a)} points is over the gate "
                         f"{g:g}; fit scale {k:.6g}, angle {math.degrees(angle):.4g} deg")
    return t, res


def _arcade_road(arcade) -> tuple:
    if not isinstance(arcade, golden.Course):
        raise DcError(f"match: {type(arcade).__name__} is not a golden.Course")
    g = arcade.ground_data
    if g is None:
        raise DcError(f"match: arcade course {arcade.course} has no ground (ground_data is None)")
    if not isinstance(g, tgp.Ground):
        raise DcError(f"match: arcade course {arcade.course}: ground_data {type(g).__name__} is not a tgp.Ground")
    polys = _seq(g.polys, None, "ground_data.polys", f"match: arcade course {arcade.course}", DcError)
    n = len(_seq(arcade.steps, None, "steps", f"match: arcade course {arcade.course}", DcError))
    if n > len(polys):
        raise DcError(f"match: arcade course {arcade.course} has {n} steps, more than its {len(polys)} ground polygons")
    return polys[:n]


def _poly_points(poly, what: str) -> list:
    corners = getattr(poly, "corners", None)
    if not isinstance(corners, (tuple, list)) or len(corners) != 4:
        raise DcError(f"{what}: {_short(poly)} has not four corners")
    pts = [_point(p, f"{what}: a corner") for p in corners]
    centre = getattr(poly, "centre", None)
    centre = centre if centre is not None else tuple(sum(p[i] for p in pts) / 4 for i in range(3))
    return pts + [_point(centre, f"{what}: the centre")]


def _road(dc, arcade) -> tuple[np.ndarray, np.ndarray]:
    """The Dreamcast's road cells and the arcade's road polygons, each (n, 5, 3): four corners and the centre, in
    step order (the counts may differ)."""
    if not isinstance(dc, data.CourseData):
        raise DcError(f"match: {type(dc).__name__} is not a data.CourseData")
    cells = data.road_cells(dc)
    polys = _arcade_road(arcade)
    a = np.array([_poly_points(c, f"match: Dreamcast road cell {k}") for k, c in enumerate(cells)]).reshape(-1, 5, 3)
    b = np.array([_poly_points(p, f"match: arcade road polygon {k}") for k, p in enumerate(polys)]).reshape(-1, 5, 3)
    return a, b


def fit_course(dc: data.CourseData, arcade: golden.Course, gate: float) -> tuple[Transform, dict]:
    """The Transform from Dreamcast course `dc` to arcade course `arcade`: `fit_points` on the road cells' four
    corners and centre against the arcade's ground road polygons', cell k to polygon k in step order, z negated.
    MatchError when the road counts differ or the RMS is over `gate`; DcError for inputs of the wrong kind."""
    g = _gate(gate)
    a, b = _road(dc, arcade)
    if len(a) != len(b):
        raise MatchError(f"match: the Dreamcast's {len(a)} road cells against arcade course {arcade.course}'s "
                         f"{len(b)} road polygons: not the same course")
    return fit_points(a.reshape(-1, 3), b.reshape(-1, 3), g)


def _road_through(dc, arcade, t) -> tuple[np.ndarray, np.ndarray]:
    """`_road`, the Dreamcast's taken through `t` into the arcade's frame. MatchError for an empty road."""
    if not isinstance(t, Transform):
        raise DcError(f"match: {type(t).__name__} is not a match.Transform")
    a, b = _road(dc, arcade)
    if not len(a) or not len(b):
        raise MatchError(f"match: {len(a)} Dreamcast road cells, {len(b)} arcade road polygons")
    a = t._apply(a.reshape(-1, 3)).reshape(-1, 5, 3)
    if not np.isfinite(a).all():
        raise DcError(f"match: the Dreamcast's road goes past the float range through {t}")
    return a, b


def lap_order(dc: data.CourseData, arcade: golden.Course, t: Transform) -> tuple[int, int]:
    """(k0, direction): the arcade road polygon (step index) nearest the Dreamcast's road cell 0 through `t`, and +1
    when the Dreamcast's cells run the arcade's step order forward, -1 when backward (by the majority of consecutive
    cells whose nearest polygons are one step apart). MatchError when neither direction holds for most cells."""
    a, b = _road_through(dc, arcade, t)
    ca, cb = a[:, 4, [0, 2]], b[:, 4, [0, 2]]
    near = np.array([int(((cb - p) ** 2).sum(axis=1).argmin()) for p in ca])
    n = len(cb)
    if len(near) < 2:
        return int(near[0]), 1
    step = np.diff(near) % n
    # n = 2 (two arcade polygons): one step forward is one step back (1 = n - 1), so fwd == back and the +1 below is
    # arbitrary; n = 1: every step is 0, neither
    fwd, back = int((step == 1 % n).sum()), int((step == (n - 1) % n).sum())
    if max(fwd, back) <= (len(near) - 1) // 2:
        raise MatchError(f"match: the Dreamcast's road does not follow the arcade's step order ({fwd} cells forward, "
                         f"{back} backward of {len(near) - 1})")
    return int(near[0]), 1 if fwd >= back else -1


def _widths(q: np.ndarray) -> np.ndarray:
    """Per quad (n, 5, 3): the mean length of its edges 0-1 and 3-2, across the road (the arcade's layout: corners 0,
    1 the forward end, 3, 2 the back end; edges 1-2 and 0-3 run along the road, about one step long)."""
    return (np.linalg.norm(q[:, 1] - q[:, 0], axis=1) + np.linalg.norm(q[:, 2] - q[:, 3], axis=1)) / 2


def road_width(dc: data.CourseData, arcade: golden.Course, t: Transform) -> tuple[float, float]:
    """(Dreamcast, arcade): the median road cell width, the Dreamcast's through `t` (arcade units). A cell's width is
    the mean length of its two edges across the road, 0-1 and 3-2. A cell is not always the road's whole width: where
    the road is split across (kerb cells of 2 units on courses 0 and 2), each piece is a cell of its own."""
    a, b = _road_through(dc, arcade, t)
    return float(np.median(_widths(a))), float(np.median(_widths(b)))


# ---- the M (mirror) files against the mirror rules (Phase 5, Task 8) ----

CORNER_ORDERS = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2), (1, 0, 3, 2), (0, 3, 2, 1), (3, 2, 1, 0),
                 (2, 1, 0, 3))
"""A cell's corner orders `mirror_report` tries (a mirrored cell's corner k is the plain cell's corner order[k]): the
quad's turns and its reflections. The arcade's ground mirror uses (0, 3, 2, 1) (`arcade.mirror.QUAD_ORDER`; (1, 0, 3,
2), `TRIANGLE_ORDER`, for a triangle)."""

LINE_ORDERS = {"kept": lambda k: k, "reversed": lambda k: 7 - k, "rotated": lambda k: (k + 1) % 8}
"""The named orders of the 8 AI lines (and, kept and reversed, of a road point's 8 lane slots): the mirrored file's
line k is the plain file's line order(k)."""

HEADING_TURN = 0x4000
"""A heading's whole turn: the yaw is 4 h (0x10000 a turn; `arcade.mirror.mirror_heading`), so headings are compared
modulo 0x4000."""

SPEED_TOLERANCE = 1e-6
"""The relative error within which a line's speeds are a constant multiple of the plain file's (f32 rounding)."""


def _side(pair, what: str) -> tuple:
    """(Model, CourseData) of one side of `mirror_report`, both checked whole by their encoders (DcError for anything
    the format cannot hold, so the report's arithmetic meets only well-formed values)."""
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise DcError(f"mirror report: the {what} side {_short(pair)} is not a (pol.Model, data.CourseData) pair")
    model, d = pair
    if not isinstance(model, pol.Model):
        raise DcError(f"mirror report: the {what} model is a {type(model).__name__}, not a pol.Model")
    if not isinstance(d, data.CourseData):
        raise DcError(f"mirror report: the {what} course data is a {type(d).__name__}, not a data.CourseData")
    try:
        pol.encode_pol(model)
        data.encode_data(d)
    except DcError as e:
        raise DcError(f"mirror report: the {what} side: {e}") from e
    return model, d


def _plane(sums) -> float | None:
    """The mirror plane's offset c (a point's z becomes c - z): the median of z' + z over the pairs, to 0.001 (0.0 for
    the plain negation, -1.0 for the arcade's plane z = -0.5). None without pairs."""
    s = np.asarray(sums, dtype=np.float64)
    if not len(s):
        return None
    with np.errstate(all="ignore"):
        c = float(np.median(s))
    return round(c, 3) + 0.0 if math.isfinite(c) else c


def _mirror_z(z, c: float | None):
    """c - z, rounded to f32 as a stored coordinate is (exact for c = 0: the negation)."""
    c = c or 0.0
    with np.errstate(all="ignore"):
        return -z if c == 0.0 else (c - np.asarray(z, dtype=np.float64)).astype(np.float32).astype(np.float64)


def _positions(p: np.ndarray, m: np.ndarray, c: float | None) -> tuple[int, float]:
    """(exactly mirrored, the largest distance): paired points (n, 3), plain p and mirrored m, under z -> c - z."""
    if not len(p):
        return 0, 0.0
    q = p.copy()
    q[:, 2] = _mirror_z(p[:, 2], c)
    exact = int((q == m).all(axis=1).sum())
    with np.errstate(all="ignore"):
        d = np.sqrt(((q - m) ** 2).sum(axis=1))
    return exact, float(d.max())


def _int_rule(p, m, period: int = 0x10000) -> dict:
    """The rule taking the plain values p to the mirrored m (integers, compared modulo `period`): 'kept', 'negated',
    'c - v' (c the commonest p + m, in hex) or 'zero' (every value 0) when it holds for every pair; else 'other'.
    `near` names the candidate with the smallest mean error (modulo the period, signed), `exact` counts the pairs it
    holds for and `max_error` is its largest error's magnitude."""
    p, m = np.asarray(p, dtype=np.int64), np.asarray(m, dtype=np.int64)
    n = len(p)
    if not n:
        return {"rule": "none", "near": "none", "exact": 0, "count": 0, "max_error": 0}
    half = period // 2

    def err(x):
        return (x + half) % period - half
    vals, counts = np.unique((p + m) % period, return_counts=True)
    c = int(vals[counts.argmax()])
    cands = [("kept", err(m - p)), ("negated", err(m + p))]
    if c:
        cands.append((f"{c:#x} - v", err(m + p - c)))
    name, e = min(cands, key=lambda t: float(np.abs(t[1]).mean()))   # the first of the best: kept, negated, c - v
    exact = int((e == 0).sum())
    if exact == n and not (p % period).any() and not (m % period).any():
        name = "zero"
    return {"rule": name if exact == n else "other", "near": name, "exact": exact, "count": n,
            "max_error": int(np.abs(e).max())}


def _scale_rule(p, m) -> dict:
    """The rule taking the plain speeds p to the mirrored m: 'kept' (equal), 'x r' (every m within SPEED_TOLERANCE of
    r p, r the median ratio, 6 digits) or 'other' (near 'kept'; max_error the largest |m / p - 1|)."""
    p, m = np.asarray(p, dtype=np.float64), np.asarray(m, dtype=np.float64)
    n = len(p)
    if not n:
        return {"rule": "none", "near": "none", "exact": 0, "count": 0, "max_error": 0.0}
    kept = int((p == m).sum())
    if kept == n:
        return {"rule": "kept", "near": "kept", "exact": n, "count": n, "max_error": 0.0}
    nz = p != 0.0
    with np.errstate(all="ignore"):
        ratio = np.where(nz, m / np.where(nz, p, 1.0), 0.0)
        r = float(np.median(ratio[nz])) if nz.any() else math.nan
        if math.isfinite(r) and r > 0.0 and r != 1.0:
            rel = np.where(nz, np.abs(ratio - r) / r, np.where(m == 0.0, 0.0, np.inf))
            if (rel <= SPEED_TOLERANCE).all():
                return {"rule": f"x {r:.6g}", "near": f"x {r:.6g}", "exact": n, "count": n,
                        "max_error": float(rel.max())}
        off = np.where(nz, np.abs(ratio - 1.0), np.abs(m))
    return {"rule": "other", "near": "kept", "exact": kept, "count": n, "max_error": float(off.max())}


def _same_rule(a, b) -> dict:
    """'kept' when the sequences a and b are equal item for item (and as long), else 'other'."""
    exact = sum(x == y for x, y in zip(a, b))
    return {"rule": "kept" if exact == len(a) == len(b) else "other", "exact": exact, "count": max(len(a), len(b))}


def _vertices(model) -> list:
    """Every vertex entry of `model` as (Vertex, raw normal?) in file order."""
    return [(e, pol._raw_normal(me.params)) for ob in model.objects for me in ob.meshes for st in me.strips
            for e in st.entries if isinstance(e, pol.Vertex)]


def _mirror_normal(normal, raw: bool):
    """A stored normal mirrored: (x, y, -z) for floats; for a raw-normal mesh's three words (word 0 x << 16 | y << 8 |
    z, signed bytes at length 127; words 1 and 2 a colour), z's byte negated."""
    if not raw:
        return float(normal[0]), float(normal[1]), -float(normal[2])
    w = int(normal[0])
    b = w & 0xFF
    return (w & ~0xFF) | (-(b - 0x100 if b & 0x80 else b) & 0xFF), int(normal[1]), int(normal[2])


def _canon(t: tuple) -> tuple:
    return min(t, t[1:] + t[:1], t[2:] + t[:2])


def _model_report(a, b) -> dict:
    va, vb = _vertices(a), _vertices(b)
    by_xy: dict = {}
    for v, _ in va:
        by_xy.setdefault((float(v.x), float(v.y)), set()).add(float(v.z))
    sums = [float(v.z) + next(iter(zs)) for v, _ in vb if len(zs := by_xy.get((float(v.x), float(v.y)), ())) == 1]
    c = _plane(sums)

    def mz(z):
        if not c:
            return -float(z)
        with np.errstate(all="ignore"):         # c - z past the f32 range: inf, matching nothing
            return float(np.float32(c - float(z)))

    def key(v, raw, mirror):
        z = mz(v.z) if mirror else float(v.z)
        n = _mirror_normal(v.normal, raw) if mirror else tuple(
            int(w) if raw else float(w) for w in v.normal)
        return float(v.x), float(v.y), z, n, float(v.u), float(v.v)
    full = {key(v, raw, True) for v, raw in va}
    pos = {k[:3] for k in full}
    exact = sum(key(v, raw, False) in full for v, raw in vb)
    positions = sum(key(v, raw, False)[:3] in pos for v, raw in vb)

    def tri_pos(t, mirror):
        return tuple((float(v.x), float(v.y), mz(v.z) if mirror else float(v.z)) for v in t[:3])
    plain = {_canon(tri_pos(t, True)) for t in pol.triangles(a)}
    wind = {"kept": 0, "reversed": 0, "unmatched": 0, "degenerate": 0}
    ntri = 0
    for t in pol.triangles(b):
        ntri += 1
        k = tri_pos(t, False)
        if len(set(k)) < 3:
            wind["degenerate"] += 1
        elif _canon(k) in plain:
            wind["kept"] += 1
        elif _canon(k[::-1]) in plain:
            wind["reversed"] += 1
        else:
            wind["unmatched"] += 1

    def strips(model, mirror) -> dict:
        """Strip flags by the strip's vertex positions (sorted), for the strips whose positions no other strip of the
        model shares (a model draws some geometry twice, with other flags)."""
        out: dict = {}
        for ob in model.objects:
            for me in ob.meshes:
                for st in me.strips:
                    k = tuple(sorted((float(e.x), float(e.y), mz(e.z) if mirror else float(e.z))
                                     for e in st.entries if isinstance(e, pol.Vertex)))
                    if k:
                        out.setdefault(k, []).append(int(st.flags))
        return {k: v[0] for k, v in out.items() if len(v) == 1}
    flags_a, flags_b = strips(a, True), strips(b, False)
    pairs: dict = {}
    other_bits = 0
    for k, f in flags_b.items():
        if k in flags_a:
            pa = flags_a[k]
            pairs[(pa & 3, f & 3)] = pairs.get((pa & 3, f & 3), 0) + 1
            other_bits += (pa & ~3) != (f & ~3)
    swap = {2: 3, 3: 2}
    if not pairs:
        cull = "none"
    elif all(q == p for p, q in pairs):
        cull = "kept"
    elif all(q == swap.get(p, p) for p, q in pairs):
        cull = "swap 2, 3"
    else:
        cull = "other"
    return {"vertices": len(vb), "exact": exact, "positions": positions, "plane": c, "triangles": ntri,
            "winding": wind, "cull": {"rule": cull, "pairs": dict(sorted(pairs.items())), "strips": sum(pairs.values()),
                                       "other_bits": other_bits}}


def _cell_points(cells) -> np.ndarray:
    return np.array([[tuple(map(float, p)) for p in c.corners] + [tuple(map(float, c.centre))] for c in cells],
                    dtype=np.float64).reshape(-1, 5, 3)


def _grid_rule(a, b) -> dict:
    n = data.GRID_CELLS
    la = [frozenset(a.lists[int(g)]) for g in a.grid]
    lb = [frozenset(b.lists[int(g)]) for g in b.grid]
    flip = sum(lb[k] == la[(data.GRID_SIZE - 1 - k // data.GRID_SIZE) * data.GRID_SIZE + k % data.GRID_SIZE]
               for k in range(n))
    kept = sum(x == y for x, y in zip(la, lb))
    rule = "127 - r" if flip == n else "kept" if kept == n else "other"
    return {"rule": rule, "exact": kept if rule == "kept" else flip if rule == "127 - r" else max(flip, kept),
            "count": n}


def _cells_report(a, b) -> dict:
    n = min(len(a.cells), len(b.cells))
    pa, pb = _cell_points(a.cells[:n]), _cell_points(b.cells[:n])
    c = _plane(pb[:, 4, 2] + pa[:, 4, 2])
    q = pa.copy()
    q[:, :, 2] = _mirror_z(pa[:, :, 2], c)
    best, exact = CORNER_ORDERS[0], -1
    for order in CORNER_ORDERS:
        ok = int((q[:, list(order) + [4]] == pb).all(axis=(1, 2)).sum())
        if ok > exact:
            best, exact = order, ok
    if n:
        with np.errstate(all="ignore"):
            offset = float(np.sqrt(((q[:, list(best) + [4]] - pb) ** 2).sum(axis=2)).max())
    else:
        offset = 0.0
    da = [int(x.direction) for x in a.cells[:n]]
    db = [int(x.direction) for x in b.cells[:n]]
    direction = _int_rule([x & 0xFFFF for x in da], [x & 0xFFFF for x in db])
    direction["high_kept"] = sum(x >> 16 == y >> 16 for x, y in zip(da, db))
    return {"sizes": (len(a.cells), len(b.cells)), "order": best, "exact": exact, "count": n, "plane": c,
            "offset": offset, "direction": direction, "grid": _grid_rule(a, b),
            "steps": _same_rule([(int(s.segment), int(s.flags)) for s in a.steps],
                                [(int(s.segment), int(s.flags)) for s in b.steps]),
            "attributes": _same_rule(bytes(a.attributes), bytes(b.attributes))}


def _line_points(line) -> np.ndarray:
    return np.array([(float(p.x), float(p.y), float(p.z)) for p in line], dtype=np.float64).reshape(-1, 3)


def _lines_report(a, b) -> dict:
    pa = [_line_points(x) for x in a.lines]
    pb = [_line_points(x) for x in b.lines]
    nb, na = len(pb), len(pa)
    dist = np.full((nb, na), np.inf)
    with np.errstate(all="ignore"):
        for k in range(nb):
            for j in range(na):
                if len(pb[k]) == len(pa[j]) and len(pb[k]):
                    q = pa[j] * (1.0, 1.0, -1.0)
                    dist[k, j] = float(np.sqrt(((q - pb[k]) ** 2).sum(axis=1)).max())
    nearest = tuple(int(dist[k].argmin()) for k in range(nb)) if na else ()
    order, line_map = "other", nearest
    if nb == na == 8:
        named = {name: max(dist[k, f(k)] for k in range(8)) for name, f in LINE_ORDERS.items()}
        name = min(named, key=named.get)                      # the first of the best: kept, reversed, rotated
        f = LINE_ORDERS[name]
        if math.isfinite(named[name]) and all(dist[k, f(k)] <= dist[k].min() for k in range(8)):
            order, line_map = name, tuple(f(k) for k in range(8))
    pairs = [(p, q) for k, j in enumerate(line_map) for p, q in zip(a.lines[j], b.lines[k])]
    pp = np.array([(float(p.x), float(p.y), float(p.z)) for p, _ in pairs], dtype=np.float64).reshape(-1, 3)
    pm = np.array([(float(q.x), float(q.y), float(q.z)) for _, q in pairs], dtype=np.float64).reshape(-1, 3)
    c = _plane(pm[:, 2] + pp[:, 2])
    exact, offset = _positions(pp, pm, c)
    if any(len(pb[k]) != len(pa[j]) for k, j in enumerate(line_map)) or len(line_map) != nb:
        offset = math.inf
    out = {"order": order, "map": line_map, "points": len(pairs), "exact": exact, "offset": offset, "plane": c,
           "speed": _scale_rule([float(p.speed) for p, _ in pairs], [float(q.speed) for _, q in pairs])}
    for f in ("pitch", "heading", "roll", "steer", "lane", "step"):
        out[f] = _int_rule([int(getattr(p, f)) for p, _ in pairs], [int(getattr(q, f)) for _, q in pairs],
                           HEADING_TURN if f == "heading" else 0x10000)
    return out


def _road_report(a, b) -> dict:
    n = min(len(a.road), len(b.road))
    best = None
    for name in ("kept", "reversed"):
        f = LINE_ORDERS[name]
        pairs = [(ra.lanes[f(k)], rb.lanes[k]) for ra, rb in zip(a.road[:n], b.road[:n])
                 for k in range(int(rb.lane_lo), int(rb.lane_hi) + 1)]
        pp = np.array([(float(p.x), float(p.y), float(p.z)) for p, _ in pairs], dtype=np.float64).reshape(-1, 3)
        pm = np.array([(float(q.x), float(q.y), float(q.z)) for _, q in pairs], dtype=np.float64).reshape(-1, 3)
        _, d = _positions(pp, pm, 0.0)
        if best is None or d < best[0]:
            best = (d, name, pairs, pp, pm)
    _, slots, pairs, pp, pm = best
    c = _plane(pm[:, 2] + pp[:, 2])
    exact, offset = _positions(pp, pm, c)
    ranges = [((int(ra.lane_lo), int(ra.lane_hi)), (int(rb.lane_lo), int(rb.lane_hi)))
              for ra, rb in zip(a.road[:n], b.road[:n])]
    kept = sum(p == q for p, q in ranges)
    rev = sum(q == (7 - p[1], 7 - p[0]) for p, q in ranges)
    asym = sum(p[0] + p[1] != 7 for p, _ in ranges)
    rng = ("symmetric" if not asym and kept == n else "kept" if kept == n else "reversed" if rev == n else "other")
    out = {"sizes": (len(a.road), len(b.road)), "slots": slots, "points": len(pairs), "exact": exact,
           "offset": offset, "plane": c,
           "lane_range": {"rule": rng, "count": n, "asymmetric": asym, "kept": kept, "reversed": rev},
           "curve": _int_rule([int(r.curve) for r in a.road[:n]], [int(r.curve) for r in b.road[:n]])}
    for f in ("pitch", "heading", "roll"):
        out[f] = _int_rule([int(getattr(p, f)) for p, _ in pairs], [int(getattr(q, f)) for _, q in pairs],
                           HEADING_TURN if f == "heading" else 0x10000)
    return out


def mirror_report(plain: tuple[pol.Model, data.CourseData], mirrored: tuple[pol.Model, data.CourseData]) -> dict:
    """How a mirrored course (a Dreamcast `M` or `MR` file pair) relates to its plain one, field by field.

    `plain` and `mirrored` are each (pol.Model, data.CourseData), checked whole by their encoders. A mirror plane's
    offset c (a point's z becomes c - z; 0.0 the Dreamcast's z negation, -1.0 the arcade's plane z = -0.5) is measured
    per family as the median of z' + z over its pairs (to 0.001), and points are exact when x and y are equal and
    z' = c - z (f32). Rules: 'kept', 'negated', 'c - v' (hex), 'zero', 'x r' (a scale), 'other' (with the nearest
    rule, the pairs it holds for and its largest error), 'none' (no pairs); orders 'kept', 'reversed', 'rotated'
    (LINE_ORDERS) or 'other'. Returns, per family:

    - model: vertex entries (`vertices`), how many equal a plain vertex mirrored (position, normal z negated or a raw
      normal's z byte, u, v: `exact`) and by position alone (`positions`), the `plane`, the mirrored model's
      `triangles` and their `winding` against the plain ones mirrored (vertex order kept, reversed, unmatched,
      degenerate), and the strips' `cull` bits (flags & 3; strips paired by their vertex positions, where no other
      strip of either model shares them): kept, 'swap 2, 3' (1 and 0 kept) or other, with the (plain, mirrored)
      `pairs`, the `strips` paired and how many of them change other flag bits (`other_bits`);
    - cells (index for index): `sizes`, the corner `order` exact on the most cells (CORNER_ORDERS), `exact`, `plane`,
      `offset` (the largest distance), the `direction` rule (low half; `high_kept`), the `grid` ('127 - r': every
      square's list is the plain square's at row 127 - r, as a set), `steps` and `attributes` (kept or other);
    - lines: the `order` and `map` (mirrored line k is the plain line map[k]: the named order every line is nearest
      to, else 'other' and each line's nearest, positions mirrored), `points` paired, `exact`, `offset`, `plane`,
      and the rules of speed, pitch, heading (modulo HEADING_TURN), roll, steer, lane and step;
    - road (point for point): the lane `slots` order (kept or reversed, the nearer over the used slots), `points`,
      `exact`, `offset`, `plane`, the `lane_range` (lane_lo, lane_hi: kept, reversed to 7 - hi, 7 - lo, symmetric
      when every range is its own reversal, or other), and the rules of curve and of each used slot's pitch, heading
      and roll (from the slot its position comes from).

    DcError for a side that is not such a pair or that its encoder refuses."""
    a_model, a = _side(plain, "plain")
    b_model, b = _side(mirrored, "mirrored")
    return {"model": _model_report(a_model, b_model), "cells": _cells_report(a, b), "lines": _lines_report(a, b),
            "road": _road_report(a, b)}
