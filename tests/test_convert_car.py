"""A Dreamcast car set converted to Model 2 objects with true-colour textures (v0.4; dtools/convert/car.py,
docs/formats/model2-polygon-rom.md "True-colour textures")."""
import numpy as np
import pytest

from dtools.convert import ConvertError, car as ccar, model
from dtools.dc import car as dcar, pol
from dtools.dc.pol import Mesh, Model, Object, Strip, TexEntry
from tests.test_convert_model import V, _wall, mesh
from tests.test_convert_tables import disc  # noqa: F401

DECAL = (0x8000002C, 0x83000000, 0x2008210D, 0xCC000000)        # TSP shading 0: the texture as it is
MODULATE = (0x8000002C, 0x83000000, 0x2008214D, 0xCC000000)     # 1: the texture times the face colour
DECAL_ALPHA = (0x8000002C, 0x83000000, 0x2008218D, 0xCC000000)  # 2: the texture over the face colour by its alpha


def test_rgb_flag():
    """model.convert_object: a whole model as Pieces, not filed by cell, every drawn record carrying DEPTH_TEST and,
    with `rgb`, RGB_TEX (attr bit 6, the custom MAME's true-colour texture; no original record sets it: bits used
    0x21f03). The polygons are convert_model's."""
    assert model.RGB_TEX == 0x40 and not model.RGB_TEX & (0x21F03 | model.DEPTH_TEST) and model.RGB_TEX & 0x3FFFF
    m = _wall(-505.0, 170.0, 190.0)
    [cell] = model.convert_model(m, depth=True).values()
    plain, rgb = model.convert_object(m), model.convert_object(m, rgb=True)
    assert [p.sources for p in plain] == [p.sources for p in cell] == [p.sources for p in rgb]
    for a, b in zip(plain, rgb):
        for ra, rb in zip(a.obj.records, b.obj.records):
            move = ra.attr == model.MOVE_ATTR
            assert rb.attr == (ra.attr if move else ra.attr | model.RGB_TEX)
            assert move or ra.attr & model.DEPTH_TEST
    with pytest.raises(ConvertError, match="rgb"):
        model.convert_object(m, rgb=1)


def test_bake():
    """car.bake: a Dreamcast texture (RGBA, the stored row order) as ARGB1555 words, top row first for the arcade's
    points (the stored rows reversed), by the mesh's shading: decal the texels, modulate the texels times the face
    colour, decal-alpha the texels over the face colour (opaque). Alpha under a half is transparent (bit 15 clear).
    A side under 32 is repeated up to 32."""
    rgba = np.zeros((32, 64, 4), np.uint8)
    rgba[0, 0] = (255, 0, 0, 255)
    rgba[31, 0] = (0, 255, 0, 255)
    rgba[5, 5] = (0, 0, 255, 100)
    t = ccar.bake(rgba, 0, (1.0, 1.0, 1.0))
    assert t.shape == (32, 64) and t.dtype == np.uint16
    assert t[31, 0] == 0x8000 | 31 << 10 and t[0, 0] == 0x8000 | 31 << 5          # the rows reversed
    assert t[26, 5] == 0 and t[1, 1] == 0                  # alpha 100 and 0: transparent, the whole word clear
    half = ccar.bake(rgba, 1, (0.5, 1.0, 1.0))
    assert half[31, 0] == 0x8000 | 15 << 10 or half[31, 0] == 0x8000 | 16 << 10
    over = ccar.bake(rgba, 2, (0.0, 0.0, 1.0))
    assert over[1, 1] == 0x8000 | 31 and over[31, 0] == 0x8000 | 31 << 10        # the face colour shows through
    rows = np.zeros((16, 64, 4), np.uint8)
    rows[..., 3] = 255
    rows[0, :, 0] = 255                                   # the stored first row red: the arcade's last
    small = ccar.bake(rows, 0, (1.0, 1.0, 1.0))
    assert small.shape == (32, 64)                        # stretched, not repeated: the picture once
    assert (small[30:] == (0x8000 | 31 << 10)).all() and (small[:30] == 0x8000).all()
    flat = ccar.flat((1.0, 0.0, 0.0))
    assert flat.shape == (32, 32) and (flat == (0x8000 | 31 << 10)).all()
    for bad in ((np.zeros((32, 48, 4), np.uint8), 0), (np.zeros((32, 32, 3), np.uint8), 0), (rgba, 7)):
        with pytest.raises(ConvertError):
            ccar.bake(bad[0], bad[1], (1.0, 1.0, 1.0))


def _quad(x, z, tex, params=DECAL, colours=(1.0,) * 9):
    st = Strip(0x11, (V(x, 1, z, u=0, v=0), V(x + 1, 1, z, u=1, v=0), V(x, 1, z + 1, u=0, v=1), V(x + 1, 1, z + 1, u=1, v=1)))
    return Mesh(params, (0.0, 0.0, 0.0, 1.0), tex, 0, colours, (st,))


def _set():
    """A level-0 set of 80 one-quad objects: object i at x = i (wheels round the origin), texture i % 2; the body's
    quad is decal-alpha over a red face colour, the lower body untextured green."""
    red = (0.75, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    green = (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    objs = []
    for i in range(dcar.SET_OBJECTS):
        if i == 0:
            me = _quad(0, 0, 0, DECAL_ALPHA, red)
        elif i == 9:
            me = _quad(9, 0, -1, mesh([]).params, green)
        elif 58 <= i < 68:
            me = _quad(-0.5, -0.5, 1)
        else:
            me = _quad(i, 0, i % 2)
        objs.append(Object(dcar.SET_KINDS[i], (0.0, 0.0, 0.0, 1000.0), (me,), 4))
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    return pol.fix_header(Model(hdr, tuple(objs), (TexEntry(64, 64, 0x102, 0, 0), TexEntry(32, 32, 0x101, 0, 0))))


def test_convert_car_synthetic():
    """convert_car: the body list (the undamaged shell, the lower body, the inside, the decal layer, the glass, the
    first number panel), the cockpit list, one left and one right wheel (the still ones), each a tuple of CarPieces
    whose drawn records name a baked texture and carry their (v, u) points; x is mirrored (the two games' frames have
    opposite handedness; front stays +z, up +y), so object i's quad lies at arcade x -(i + 1) .. -i."""
    m = _set()
    images = [np.full((64, 64, 4), (10, 20, 30, 128), np.uint8), np.full((32, 32, 4), 255, np.uint8)]
    hubs = ((0.8, 0.34, 1.4), (-0.8, 0.34, 1.4), (0.8, 0.34, -1.3), (-0.8, 0.34, -1.3))
    c = ccar.convert_car(m, images, hubs, "SYN")
    parts = dcar.car_parts(m)

    def quads(pieces):
        return sum(1 for pc in pieces for r in pc.obj.records if r.attr != model.MOVE_ATTR)

    def xs(pieces):
        pts = [p for pc in pieces for q in __import__("dtools.arcade.polyrom", fromlist=["polygons"]).polygons(pc.obj)
               if q.drawn for p in q.points]
        return min(p[0] for p in pts), max(p[0] for p in pts)

    assert quads(c.body) == 7 and quads(c.cockpit) == len(parts.cockpit) and quads(c.wheel_left) == quads(c.wheel_right) == 1
    assert xs(c.body) == (-33.0, 0.0)                                 # objects 0 .. 32 (the first number panel)
    assert xs(c.wheel_left) == (-0.5, 0.5)
    for pieces in (c.body, c.cockpit, c.wheel_left, c.wheel_right):
        for pc in pieces:
            assert len(pc.textures) == len(pc.points) == len(pc.obj.records)
            for r, t, pts in zip(pc.obj.records, pc.textures, pc.points):
                if r.attr == model.MOVE_ATTR:
                    assert t == -1
                    continue
                assert r.attr & model.RGB_TEX and r.attr & model.DEPTH_TEST and 0 <= t < len(c.textures)
                assert len(pts) in (3, 4) and all(0 <= v <= 0xFFFF and 0 <= u <= 0xFFFF for v, u in pts)
    # the body's decal-alpha quad over red: opaque, between the texel and the face colour; the lower body flat green
    body_tex = {c.textures[t].tobytes() for pc in c.body for t in pc.textures if t >= 0}
    over = ccar.bake(images[0], 2, (1.0, 0.0, 0.0))
    assert over.tobytes() in body_tex and (over & 0x8000).all()
    assert ccar.flat((0.0, 1.0, 0.0)).tobytes() in body_tex
    assert c.hubs == ((-0.8, 0.34, 1.4), (0.8, 0.34, 1.4), (-0.8, 0.34, -1.3), (0.8, 0.34, -1.3))   # x mirrored
    assert len({t.tobytes() for t in c.textures}) == len(c.textures)   # each baked texture once
    with pytest.raises(ConvertError, match="SYN"):
        ccar.convert_car(m, images[:1], hubs, "SYN")
    with pytest.raises(ConvertError):
        ccar.convert_car(m, images, hubs[:3], "SYN")


def test_grasshopper_on_disc(disc):  # noqa: F811
    c = ccar.car_from_disc(disc, 1)
    n = lambda pieces: sum(1 for pc in pieces for r in pc.obj.records if r.attr != model.MOVE_ATTR)  # noqa: E731
    assert 800 <= n(c.body) <= 2200 and 60 <= n(c.wheel_left) <= 160 and n(c.wheel_left) == n(c.wheel_right)
    assert 200 <= n(c.cockpit) <= 800
    pts = np.array([p for pc in c.body for q in __import__("dtools.arcade.polyrom", fromlist=["polygons"]).polygons(pc.obj)
                    if q.drawn for p in q.points])
    assert pts[:, 0].min() > -1.2 and pts[:, 0].max() < 1.2 and pts[:, 2].min() > -2.6 and pts[:, 2].max() < 2.6
    assert sum(t.size for t in c.textures) < 400_000
    print(f"\nGrasshopper: body {n(c.body)} polygons, cockpit {n(c.cockpit)}, wheel {n(c.wheel_left)}; "
          f"{len(c.textures)} textures, {sum(t.size for t in c.textures):,} texels")
