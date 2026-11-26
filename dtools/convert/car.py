"""A Dreamcast car set as Model 2 objects with true-colour textures (v0.4).

`convert_car` composes a level-0 set (`dtools.dc.car.car_parts`) into what the arcade draws for the player's car
(docs/spike/2026-11-cars.md, "The draw routine"): a body list, a cockpit list for the driver's seat view, and a left
and a right wheel, each a tuple of CarPieces (a polygon object, and per record its baked texture and texture points).

- **Frame.** The two games' frames have opposite handedness (the courses are converted with z negated); a car keeps
  its front on +z and its top on +y, so x is mirrored: arcade (x, y, z) = Dreamcast (-x, y, z). The model is turned
  half round the y axis before `model.convert_object` (which negates z), a proper rotation, so every winding and
  normal rule of the course converter holds.
- **Textures.** Every polygon is true colour (`model.RGB_TEX`): the Model 2's own textures hold 16 shades of one
  colour a polygon and its sheets are full (Pascal's choice, 26 Nov 2026). A mesh's texture is baked with its
  shading (`bake`): the texels, the texels times the face colour, or the texels over the face colour (the body's
  paint under its reflection map); an untextured mesh takes a flat texture of its face colour (`flat`). A mesh that
  alpha-blends a texture with no opaque texel majority is drawn checkered (the Model 2's own translucency).
- Texture points are `textures.polygon_points`' on the baked size."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np

from dtools.arcade import polyrom as P
from dtools.convert import ConvertError, model, textures
from dtools.dc import car as dcar, pol, tex
from dtools.dc.errors import DcError

MIN_SIDE = 32                   # the texture header's least side
ALPHA_CUT = 128                 # alpha under this is a transparent texel
SHADINGS = (0, 1, 2, 3)         # the PowerVR's texture/shading instruction: decal, modulate, decal alpha, modulate alpha
CHECKER_ALPHA = 0.9             # a blended texture whose opaque texels' mean alpha is under this is drawn checkered


class CarPiece(NamedTuple):
    """One polygon object and, per record (move records included): the index of its baked texture in
    `ConvertedCar.textures` (-1 for a move record), its (v, u) texture points (zeros for a move record, four of them)
    and whether it is drawn checkered."""
    obj: P.Object
    textures: tuple[int, ...]
    points: tuple[tuple[tuple[int, int], ...], ...]
    checker: tuple[bool, ...]


class ConvertedCar(NamedTuple):
    textures: tuple[np.ndarray, ...]        # each (height, width) uint16 ARGB1555, top row first
    body: tuple[CarPiece, ...]
    cockpit: tuple[CarPiece, ...]
    wheel_left: tuple[CarPiece, ...]        # at arcade x < 0
    wheel_right: tuple[CarPiece, ...]
    hubs: tuple[tuple[float, float, float], ...]     # arcade frame: front x < 0, front x > 0, rear x < 0, rear x > 0


def _words(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    c = np.clip(np.rint(rgb * (31.0 / 255.0)), 0, 31).astype(np.uint16)
    w = (c[..., 0] << 10) | (c[..., 1] << 5) | c[..., 2] | 0x8000
    return np.where(alpha >= ALPHA_CUT, w, 0).astype(np.uint16)


def _grow(a: np.ndarray) -> np.ndarray:
    """`a` with a side under MIN_SIDE stretched up to it (each texel repeated): the texture points are scaled to the
    stretched size, so the picture is drawn once."""
    h, w = a.shape
    return np.repeat(np.repeat(a, max(1, MIN_SIDE // h), 0), max(1, MIN_SIDE // w), 1)


def bake(rgba, shading: int, colour) -> np.ndarray:
    """A decoded Dreamcast texture (`tex.to_rgba`: (h, w, 4) uint8, the stored row order) as (h, w) ARGB1555 words,
    top row first (the stored rows reversed, as `textures.polygon_points` flips v), baked with the mesh's `shading`
    (SHADINGS) and face `colour` (three numbers 0-1); a side under MIN_SIDE stretched up to it. ConvertError for a
    picture that is not (h, w, 4) uint8 with power-of-two sides, or a shading outside SHADINGS."""
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4 or a.dtype != np.uint8 or any(s < 1 or s & (s - 1) for s in a.shape[:2]):
        raise ConvertError(f"bake: a picture of shape {a.shape} {a.dtype} is not (h, w, 4) uint8 with power-of-two sides")
    if shading not in SHADINGS or isinstance(shading, bool):
        raise ConvertError(f"bake: shading {shading!r} is not one of {SHADINGS}")
    col = np.asarray(colour, float).reshape(3)
    rgb, alpha = a[..., :3].astype(float), a[..., 3].astype(float)
    if shading in (1, 3):
        rgb = rgb * col
    elif shading == 2:
        f = alpha[..., None] / 255.0
        rgb, alpha = rgb * f + 255.0 * col * (1.0 - f), np.full_like(alpha, 255.0)
    return _grow(_words(rgb, alpha)[::-1])


def flat(colour) -> np.ndarray:
    """A MIN_SIDE square of one colour (three numbers 0-1): an untextured mesh's texture."""
    col = np.asarray(colour, float).reshape(3)
    return _words(np.broadcast_to(255.0 * col, (MIN_SIDE, MIN_SIDE, 3)), np.full((MIN_SIDE, MIN_SIDE), 255.0))


def _turned(m: pol.Model, indices, offsets=None) -> pol.Model:
    """The objects `indices` of `m` as one model, turned half round the y axis (x and z negated, the normals too)."""
    def vert(v):
        if not isinstance(v, pol.Vertex):
            return v
        n = v.normal
        if all(isinstance(c, float) for c in n):
            n = (0.0 - n[0], n[1], 0.0 - n[2])
        return replace(v, x=0.0 - v.x, z=0.0 - v.z, normal=n)
    objs = []
    for i in indices:
        o = m.objects[i]
        objs.append(replace(o, meshes=tuple(replace(me, strips=tuple(replace(s, entries=tuple(vert(e) for e in s.entries))
                                                                   for s in me.strips)) for me in o.meshes)))
    return replace(m, objects=tuple(objs))


def _part(m: pol.Model, indices, images, bank: dict, baked: list, name: str) -> tuple[CarPiece, ...]:
    from dtools.convert import build
    sub = _turned(m, indices)
    where = model.triangle_meshes(sub)
    meshes = {(oi, mi): me for oi, ob in enumerate(sub.objects) for mi, me in enumerate(ob.meshes)}
    face = build._face_colours(sub)
    sizes = {t: (max(MIN_SIDE, e.width), max(MIN_SIDE, e.height)) for t, e in enumerate(sub.textures)}
    out = []
    for pc in model.convert_object(sub, sizes, rgb=True):
        texs, pts, chk = [], [], []
        for r, t, uv, src in zip(pc.obj.records, pc.textures, pc.uvs, pc.sources):
            if not src:
                texs.append(-1), pts.append(((0, 0),) * 4), chk.append(False)
                continue
            me = meshes[where[src[0]]]
            colour = tuple(round(float(c), 4) for c in face[id(me)])
            if t < 0:
                key, checker = ("flat", colour), False
                if key not in bank:
                    bank[key] = len(baked)
                    baked.append(flat(colour))
                texs.append(bank[key]), pts.append(((0, 0),) * len(uv)), chk.append(checker)
                continue
            if not 0 <= t < len(images) or images[t] is None:
                raise ConvertError(f"{name}: a polygon names texture {t}, which was not decoded")
            shading = (me.params[2] >> build.TSP_SHADING) & 3
            key = (t, shading, colour if shading else None)
            if key not in bank:
                bank[key] = len(baked)
                baked.append(bake(images[t], shading, colour))
            img = np.asarray(images[t])
            seen = img[..., 3][img[..., 3] >= ALPHA_CUT]
            checker = bool(build._blends(me) and shading != 2 and seen.size and seen.mean() / 255.0 < CHECKER_ALPHA)
            h, w = baked[bank[key]].shape
            texs.append(bank[key]), pts.append(textures.polygon_points(uv, w, h)[0]), chk.append(checker)
        out.append(CarPiece(pc.obj, tuple(texs), tuple(pts), tuple(chk)))
    return tuple(out)


def convert_car(m: pol.Model, images, hubs, name: str = "car set") -> ConvertedCar:
    """The level-0 set `m` (its textures decoded to `images`, RGBA in the stored row order, one per texture-table
    entry; `hubs` its four wheel hubs, `dcar.wheel_positions`) as a ConvertedCar. ConvertError naming `name` for a
    set `dcar.car_parts` refuses, images that are not one per texture, hubs that are not four (x, y, z), and as
    `model.convert_object` and `bake`."""
    try:
        parts = dcar.car_parts(m, name)
    except DcError as err:
        raise ConvertError(str(err)) from err
    if not isinstance(images, (list, tuple)) or len(images) != len(m.textures):
        raise ConvertError(f"{name}: {len(images) if isinstance(images, (list, tuple)) else 'no'} images for "
                           f"{len(m.textures)} textures")
    if not isinstance(hubs, (list, tuple)) or len(hubs) != 4 or any(len(tuple(h)) != 3 for h in hubs):
        raise ConvertError(f"{name}: hubs {hubs!r:.60} are not four (x, y, z)")
    bank, baked = {}, []
    body = _part(m, (parts.lower, parts.inner, parts.body, parts.trim, *parts.glass, parts.numbers[0]), images, bank,
                 baked, name)
    cockpit = _part(m, parts.cockpit, images, bank, baked, name)
    # the Dreamcast's left wheels are at x > 0: the arcade's x < 0 after the mirror
    left = _part(m, (parts.wheels_left[0],), images, bank, baked, name)
    right = _part(m, (parts.wheels_right[0],), images, bank, baked, name)
    return ConvertedCar(tuple(baked), body, cockpit, left, right,
                        tuple((0.0 - float(h[0]), float(h[1]), float(h[2])) for h in hubs))


def car_from_disc(disc, car: int) -> ConvertedCar:
    """Car 0-4 of `disc` (a `DcDisc`) converted: its level-0 files and its hubs from `1ST_READ.BIN`."""
    try:
        pn, tn = dcar.car_files(car, 0)
        m = pol.decode_pol(disc.read(pn), pn)
        images = [tex.to_rgba(t) for t in tex.decode_tex(disc.read(tn), m.textures, tn)]
        hubs = dcar.wheel_positions(disc.read("1ST_READ.BIN"), car)
    except DcError as err:
        raise ConvertError(str(err)) from err
    return convert_car(m, images, hubs, pn)


class EnemySet(NamedTuple):
    """An opponents' set converted: the baked textures its objects share; each needed object's pieces, by its index
    in the set (a body, the floor, the cabin, a panel: each converted once, whichever cars use it); and per car the
    objects of its three levels of detail (`dcar.EnemyParts`)."""
    textures: tuple[np.ndarray, ...]
    objects: dict[int, tuple[CarPiece, ...]]
    cars: tuple[dcar.EnemyParts, ...]


def enemies_from_disc(disc, k: int) -> EnemySet:
    """Opponents' set 0-3 of `disc` converted (x mirrored, true colour, depth tested, as `convert_car`'s parts).
    ConvertError for what the readers or `_part` refuse."""
    try:
        pn, tn = dcar.enemy_files(k)
        m = pol.decode_pol(disc.read(pn), pn)
        images = [tex.to_rgba(t) for t in tex.decode_tex(disc.read(tn), m.textures, tn)]
        cars = tuple(dcar.enemy_parts(m, c, pn) for c in range(dcar.ENEMY_CARS))
    except DcError as err:
        raise ConvertError(str(err)) from err
    bank, baked, objects = {}, [], {}
    for idx in sorted({i for p in cars for level in p for i in level}):
        objects[idx] = _part(m, (idx,), images, bank, baked, pn)
    return EnemySet(tuple(baked), objects, cars)
