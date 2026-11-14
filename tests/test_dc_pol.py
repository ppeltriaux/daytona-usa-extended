"""Tests for the Dreamcast course model decoder and encoder (dtools/dc/pol.py)."""

import random
import struct
from dataclasses import replace

import numpy as np
import pytest

from dtools.dc import pol
from dtools.dc.disc import DcDisc
from dtools.dc.errors import DcError
from dtools.dc.pol import Mesh, Model, Object, Reuse, Strip, TexEntry, Vertex

OBJ_START = 0x30 + 4 + 16 + 4                     # one object, one texture, 4 bytes of padding
STRIP_BASE = OBJ_START + 0x18 + 0x50            # the first mesh's strip data


def V(x, n=(0.0, 1.0, 0.0), u=0.25, v=0.5):
    return Vertex(x, 2.0, -3.0, n, u, v)


def make_model(**kw) -> Model:
    """One object with two meshes: a strip (3 vertices, a re-use of the second, a fourth vertex) and a triangle
    list with raw normal words."""
    strip_a = Strip(0x13, (V(1.0), V(2.0), V(3.0), Reuse(0x5FF00000 | 0xFFFB8, -72), V(4.0)))
    # v1 is at STRIP_BASE + 8 + 32; the re-use sits at STRIP_BASE + 8 + 96, its end 8 later
    mesh_a = Mesh((0x8000002C, 0x83000000, 0x2008216D, 0xCC000000), (0.0, 1.0, 2.0, 3.0), 0, 7,
                  (1.0,) * 9, (strip_a,))                   # texture 0 (+0x20), kind 7 (+0x24, raw)
    raw = (0x10203040, 0xFFFFFFFF, 7)
    strip_b = Strip(0x1B, (V(5.0, raw), V(6.0, raw), V(7.0, raw)))
    mesh_b = Mesh((0x8000001C, 0x83000000, 0x2008A152, 0xCC000000), (0.0, 0.0, 0.0, 1.0), -1, -3,
                  (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), (strip_b,))
    obj = Object(3, (0.5, 0.25, 0.125, 9.0), (mesh_a, mesh_b), 8)
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    base = Model(hdr, (obj,), (TexEntry(64, 64, 0x101, 0, 0),))
    base = replace(base, **kw)
    return pol.fix_header(base)


def test_pol_round_trip_synthetic():
    m = make_model()
    blob = pol.encode_pol(m)
    assert len(blob) == m.header[6]
    back = pol.decode_pol(blob, "SYN_POL.BIN")
    assert back == m
    assert pol.encode_pol(back) == blob
    assert pol.vertex_count(back) == 7
    tris = list(pol.triangles(back))
    # strip of 5 entries: 3 triangles (the second with swapped winding, the re-use resolved), list: 1 triangle
    assert len(tris) == 4
    a, b, c, oi = tris[1]
    assert (a.x, b.x, c.x, oi) == (3.0, 2.0, 2.0, 0)
    assert tris[0][:3] == (V(1.0), V(2.0), V(3.0))
    assert tris[2][:3] == (V(3.0), V(2.0), V(4.0))


def test_pol_accepts_numpy_numbers():
    m = make_model()
    obj = m.objects[0]
    mesh = replace(obj.meshes[0], sphere=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32), kind=np.int32(7),
                   texture=np.int64(0))
    m2 = replace(m, objects=(replace(obj, meshes=(mesh, obj.meshes[1])),))
    assert pol.encode_pol(m2) == pol.encode_pol(m)


def _with_mesh_a(m: Model, **kw) -> Model:
    obj = m.objects[0]
    return replace(m, objects=(replace(obj, meshes=(replace(obj.meshes[0], **kw), obj.meshes[1])),))


def _with_strip_a(m: Model, strip: Strip) -> Model:
    return _with_mesh_a(m, strips=(strip,))


def _refuses(m: Model, match: str):
    with pytest.raises(DcError, match=match):
        pol.encode_pol(m)


def test_pol_refuses():
    m = make_model()
    good = m.objects[0].meshes[0].strips[0].entries
    # Review Focus 2: coordinates the format cannot hold
    for bad in (float("nan"), float("inf"), 1e39, True, "1"):
        _refuses(_with_strip_a(m, Strip(0x13, (replace(good[0], x=bad),) + good[1:])), "position|finite|single")
    _refuses(_with_strip_a(m, Strip(0x13, (replace(good[0], normal=(0.0, float("nan"), 0.0)),) + good[1:])), "normal")
    _refuses(_with_mesh_a(m, sphere=(0.0, 0.0, float("nan"), 1.0)), "sphere")
    _refuses(_with_mesh_a(m, colours=(1.0,) * 8), "colours")
    # a re-use that does not land on an earlier vertex: off by 8 (inside a vertex), forward, before the data
    for off in (-64, -80, 0, 8, -10_000):
        ent = good[:3] + (Reuse(0x5FFFFFB8, off),) + good[4:]
        _refuses(_with_strip_a(m, Strip(0x13, ent)), "not an earlier vertex")
    _refuses(_with_strip_a(m, Strip(0x13, good[:3] + (Reuse(0x12345678, -72),) + good[4:])), "re-use word")
    # Review Focus 2: a texture index (+0x20) past the table; -1 is the only value besides an index
    for tex in (1, 117, -2, -3, -4):
        _refuses(_with_mesh_a(m, texture=tex), "texture index")
    # the +0x24 word (`kind`) is raw: any 32-bit value, a number past the table included
    for kind in (117, -3, -(1 << 31)):
        assert pol.decode_pol(pol.encode_pol(_with_mesh_a(m, kind=kind))).objects[0].meshes[0].kind == kind
    # a strip count that disagrees
    obj = m.objects[0]
    _refuses(replace(m, objects=(replace(obj, strip_count=7),)), "strip count")
    # a list that is not whole triangles; a mesh whose first parameter word is zero
    _refuses(_with_strip_a(m, Strip(0x1B, good[:4])), "multiple of 3")
    _refuses(_with_mesh_a(m, params=(0, 1, 2, 3)), "parameter word 0")
    # a normal that does not fit its mesh's format
    raw_mesh = m.objects[0].meshes[1]
    bad_raw = replace(raw_mesh, strips=(Strip(0x1B, (replace(raw_mesh.strips[0].entries[0], normal=(0.5, 0.0, 0.0)),)
                                              + raw_mesh.strips[0].entries[1:]),))
    _refuses(replace(m, objects=(replace(obj, meshes=(obj.meshes[0], bad_raw)),)), "raw normal")
    # header words that disagree with the contents
    _refuses(replace(m, header=m.header[:6] + (m.header[6] + 4,) + m.header[7:]), "file size")
    _refuses(replace(m, header=m.header[:3] + (5,) + m.header[4:]), "header words 3-5")
    # a vertex whose x has a re-use entry's bit pattern would read back as a re-use
    pattern = struct.unpack("<f", struct.pack("<I", 0x5FF12345))[0]
    _refuses(_with_strip_a(m, Strip(0x13, (replace(good[0], x=pattern),) + good[1:])), "re-use entry")
    # types
    _refuses(replace(m, objects=(None,)), "not an Object")
    _refuses(replace(m, textures=("x",)), "not a TexEntry")
    with pytest.raises(DcError):
        pol.encode_pol("model")


B_BASE = STRIP_BASE + 8 + 4 * 32 + 8 + 0x50       # mesh 2's strip data, after mesh 1's five entries
B_REUSE = B_BASE + 8 + 2 * 32                           # where mesh 2's third entry sits
A_V0 = STRIP_BASE + 8                                   # mesh 1's first vertex


def _two_mesh_model(offset: int) -> Model:
    """Mesh 2's strip is two vertices and a re-use with the given offset."""
    m = make_model()
    obj = m.objects[0]
    mesh_a, mesh_b = obj.meshes
    strip = Strip(0x13, mesh_b.strips[0].entries[:2] + (Reuse(0x5FFFFFB8, offset),))
    mesh_b = replace(mesh_b, strips=(strip,))
    return replace(m, objects=(replace(obj, meshes=(mesh_a, mesh_b), strip_count=8),))


def test_pol_reuse_stays_in_its_mesh():
    inside = pol.fix_header(_two_mesh_model(B_BASE + 8 - (B_REUSE + 8)))        # mesh 2's own first vertex
    blob = pol.encode_pol(inside)
    assert pol.decode_pol(blob) == inside
    cross = A_V0 - (B_REUSE + 8)                                                # mesh 1's first vertex
    # encode
    with pytest.raises(DcError, match=rf"offset {B_REUSE:#x}.*earlier vertex of this mesh"):
        pol.encode_pol(pol.fix_header(_two_mesh_model(cross)))
    # decode: the same bytes with the offset patched
    b = bytearray(blob)
    struct.pack_into("<i", b, B_REUSE + 4, cross)
    with pytest.raises(DcError, match=rf"CROSS_POL.BIN.*offset {B_REUSE:#x}.*earlier vertex of this mesh"):
        pol.decode_pol(bytes(b), "CROSS_POL.BIN")
    # triangles() on a model that skipped the encoder
    with pytest.raises(DcError, match="earlier vertex of this mesh"):
        list(pol.triangles(pol.fix_header(_two_mesh_model(cross))))


def test_pol_walkers_refuse_what_is_not_a_model():
    """triangles and vertex_count raise only DcError: on a non-Model, members of the wrong types, a strip's flags or a
    re-use's offset that is not an integer (they used to raise AttributeError or TypeError)."""
    m = make_model()
    obj = m.objects[0]
    strip = obj.meshes[0].strips[0]
    bad = [None, 5, "model", replace(m, objects=None), replace(m, objects=(5,)), replace(m, textures=None),
           replace(m, objects=(replace(obj, meshes=(None,)),)),
           replace(m, objects=(replace(obj, meshes=7),)),
           _with_mesh_a(m, strips=("strip",)),
           _with_strip_a(m, Strip("0x13", strip.entries)),
           _with_strip_a(m, Strip(True, strip.entries)),
           _with_strip_a(m, Strip(0x13, None)),
           _with_strip_a(m, Strip(0x13, strip.entries[:3] + ("vertex",) + strip.entries[4:])),
           _with_strip_a(m, Strip(0x13, strip.entries[:3] + (Reuse(0x5FFFFFB8, "-72"),) + strip.entries[4:])),
           _with_strip_a(m, Strip(0x13, strip.entries[:3] + (Reuse(0x5FFFFFB8, -72.0),) + strip.entries[4:]))]
    for b in bad:
        with pytest.raises(DcError):
            pol.triangles(b)                    # refused at the call, before iterating
        with pytest.raises(DcError):
            pol.vertex_count(b)
    # numpy integers are integers
    ok = _with_strip_a(m, Strip(np.uint32(0x13), strip.entries[:3] + (Reuse(0x5FFFFFB8, np.int64(-72)),)
                                + strip.entries[4:]))
    assert pol.vertex_count(ok) == 7 and len(list(pol.triangles(ok))) == 4


def test_pol_decode_refuses_a_name_that_is_not_a_str():
    blob = pol.encode_pol(make_model())
    for name in (1.5, None, b"X_POL.BIN", 3):
        with pytest.raises(DcError, match="name"):
            pol.decode_pol(blob, name)
    assert pol.decode_pol(blob, "X_POL.BIN") == make_model()


def test_pol_decode_refuses_malformed():
    """Review Focus 1: a truncated file, or an offset, count or size past the end, is a DcError naming the offset."""
    blob = pol.encode_pol(make_model())
    for n in range(len(blob)):
        with pytest.raises(DcError):
            pol.decode_pol(blob[:n], "CUT_POL.BIN")
    with pytest.raises(DcError):
        pol.decode_pol(blob + b"\0\0\0\0", "LONG_POL.BIN")

    def patched(off: int, value: int, fmt: str = "<I") -> bytes:
        b = bytearray(blob)
        struct.pack_into(fmt, b, off, value)
        return bytes(b)

    cases = {
        "object offset past the end": (0x30, 0x100001, r"object 0 at table offset 0x30"),
        "object offset not the next": (0x30, OBJ_START + 8 + 1, r"expected 0x49"),
        "object offset bit 0 clear": (0x30, OBJ_START, r"bit 0"),
        "object count past the file": (0x0C, 0x10000 | 0xFFFF, r"object table"),
        "texture count past the file": (0x0C, 0xFFFF0001, r"texture table"),
        "texture table offset": (0x14, 0x40, r"texture table"),
        "file size": (0x18, 0x10, r"file is|bytes"),
        "object first word": (OBJ_START, 2, r"first word"),
        "mesh strip size past the end": (OBJ_START + 0x18 + 0x4C, 0x7FFFFFF0, r"mesh strip data"),
        "mesh strip size short": (OBJ_START + 0x18 + 0x4C, 0x20, r"runs past"),
        "strip count past the data": (STRIP_BASE + 4, 0xFFFFFF, r"runs past"),
        "texture index": (OBJ_START + 0x18 + 0x20, 5, r"texture index 5"),
        "texture index -3": (OBJ_START + 0x18 + 0x20, -3 & 0xFFFFFFFF, r"texture index -3"),
        "object strip count": (len(blob) - 4, 99, r"strip count"),
    }
    for what, (off, value, match) in cases.items():
        b = patched(off, value)
        with pytest.raises(DcError, match=match) as exc:
            pol.decode_pol(b, "BAD_POL.BIN")
        assert "BAD_POL.BIN" in str(exc.value), what
        assert "0x" in str(exc.value), what
    # the padding before the first object must be zero
    with pytest.raises(DcError, match="padding"):
        pol.decode_pol(patched(0x44, 1), "BAD_POL.BIN")
    # a re-use pointing at a non-vertex
    b = patched(STRIP_BASE + 8 + 96 + 4, -64 & 0xFFFFFFFF)
    with pytest.raises(DcError, match="not an earlier vertex"):
        pol.decode_pol(b, "BAD_POL.BIN")
    # a NaN coordinate
    b = patched(STRIP_BASE + 8, 0x7FC00000)
    with pytest.raises(DcError, match="not finite"):
        pol.decode_pol(b, "BAD_POL.BIN")
    with pytest.raises(DcError):
        pol.decode_pol("text")


def test_pol_decode_never_raises_other_errors():
    blob = pol.encode_pol(make_model())
    rng = random.Random(3)
    for _ in range(3000):
        b = bytearray(blob)
        for _ in range(rng.randint(1, 4)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        try:
            m = pol.decode_pol(bytes(b), "FUZZ_POL.BIN")
        except DcError:
            continue
        list(pol.triangles(m))          # whatever decodes is usable
        assert pol.encode_pol(m) == bytes(b)


def test_pol_names():
    names = pol.pol_names()
    assert len(names) == 64 == len(set(names))
    assert names[0] == "CS0_POL.BIN" and names[1] == "VCS0_POL.BIN" and names[-1] == "VCS7MR_POL.BIN"


# ---- the disc ----

@pytest.fixture(scope="module")
def disc(request):
    from dtools import config
    p = config.dc_path()
    if p is None or not p.is_file():
        pytest.skip("set DAYTONA_DC to the Dreamcast disc .cue or .7z")
    d = DcDisc.open(p)
    request.addfinalizer(d.close)
    return d


def test_pol_lossless_all(disc):
    assert pol.verify_lossless_pol(disc) == []


def test_pol_counts_real(disc):
    m = pol.decode_pol(disc.read("CS0_POL.BIN"), "CS0_POL.BIN")
    assert len(m.objects) == 116 and len(m.textures) == 117
    assert pol.vertex_count(m) == 11867
    assert sum(1 for _ in pol.triangles(m)) == 8961
    # the mesh's +0x20 word is the texture index: every texture named, nothing else but -1; +0x24 is raw
    assert {me.texture for ob in m.objects for me in ob.meshes} == set(range(117)) | {-1}
    assert {me.kind for ob in m.objects for me in ob.meshes} <= {0, 5, 2, -1, -3}
    total = sum(len(pol.decode_pol(disc.read(n), n).objects) for n in pol.pol_names())
    assert total == 8481
