"""The Dreamcast course models, `CSn{v}_POL.BIN` and `VCSn{v}_POL.BIN`: a lossless decoder and encoder.

Layout (docs/formats/dreamcast-disc.md, "CSnv_POL.BIN and VCSnv_POL.BIN"; docs/spike/2026-11-dc-decode.md, "Models"):
a 0x30-byte header of twelve little-endian words, the object table (`nobj` words, each an object offset | 1), the
texture table (`ntex` x 16 bytes), then the objects back to back to the end of the file. An object is two words, a
bounding sphere, meshes until a zero word, then that word and the object's number of strip entries. A mesh is a
0x50-byte header and its strips; a strip is a flag word, a count and entries, each a 32-byte vertex or an 8-byte
re-use of an earlier vertex.

Words whose meaning is not known stay raw: the header words, the mesh's four parameter words and its `kind` (+0x24;
the texture index is the +0x20 word, `Mesh.texture`), the re-use's first word, the texture table's last word. Every
malformed input is a `DcError` naming the file and offset; the encoder refuses what the format cannot hold, as the
arcade encoders do."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from typing import Iterator

from dtools.dc.errors import DcError
from dtools.arcade.tablebase import _f32, _int, _seq

HEADER_WORDS = 12
HEADER_SIZE = 0x30
TEX_ENTRY_SIZE = 16
OBJECT_HEAD = 0x18          # object: w0, w1, sphere (4 f32)
MESH_HEAD = 0x50
VERTEX_SIZE = 32
REUSE_SIZE = 8
STRIP_HEAD = 8
LIST_FLAG = 0x08            # strip flag bit 3: a triangle list, `count` triangles; clear: a strip of `count` entries
RAW_NORMAL = 0x30           # mesh params[0] & 0x30 == 0x10: the vertex's three normal words are not floats
RAW_NORMAL_BITS = 0x10
NO_TEXTURE = -1             # the mesh's texture index (+0x20) when it names no texture-table entry

_VERTEX_F = struct.Struct("<8f")
_VERTEX_I = struct.Struct("<3f3I2f")


@dataclass(frozen=True)
class Vertex:
    """x, y, z, normal (three floats; three raw u32 words when the mesh's params[0] & 0x30 == 0x10), u, v."""
    x: float
    y: float
    z: float
    normal: tuple
    u: float
    v: float


@dataclass(frozen=True)
class Reuse:
    """An 8-byte re-use of an earlier vertex: the entry's first word (0x5ffxxxxx; its low bits are not understood)
    and the signed byte offset from the end of this entry to the vertex."""
    raw: int
    offset: int


@dataclass(frozen=True)
class Strip:
    flags: int
    entries: tuple


@dataclass(frozen=True)
class Mesh:
    """`texture` is the +0x20 word: -1 or an index into the texture table (every mesh of the 64 files; Task 4).
    `kind` is the +0x24 word, kept raw (0, 5, 2, -1 or -3 in the files; meaning not learned)."""
    params: tuple[int, int, int, int]
    sphere: tuple[float, float, float, float]
    texture: int
    kind: int
    colours: tuple[float, ...]
    strips: tuple[Strip, ...]


@dataclass(frozen=True)
class Object:
    kind: int
    sphere: tuple[float, float, float, float]
    meshes: tuple[Mesh, ...]
    strip_count: int


@dataclass(frozen=True)
class TexEntry:
    width: int
    height: int
    fmt: int
    offset: int
    raw: int


@dataclass(frozen=True)
class Model:
    header: tuple[int, ...]
    objects: tuple[Object, ...]
    textures: tuple[TexEntry, ...]


def _pad(end: int) -> int:
    """Zero bytes between the texture table and the first object: objects start on an 8-byte boundary (0 or 4 bytes
    in the 64 files)."""
    return -end % 8


def _is_ref(w: int) -> bool:
    return (w >> 28) == 5 and (w & 0x0FF00000) == 0x0FF00000


def _raw_normal(params) -> bool:
    return (params[0] & RAW_NORMAL) == RAW_NORMAL_BITS


def _count(strip: Strip) -> int:
    n = len(strip.entries)
    return n // 3 if strip.flags & LIST_FLAG else n


# ---- decode ----

def decode_pol(blob: bytes, name: str = "") -> Model:
    """The model in `blob`. DcError (naming `name` and the offset) for a file that is truncated, whose offsets, counts
    or sizes point past its end or disagree, or whose floats are not finite."""
    if not isinstance(name, str):
        raise DcError(f"model: the name {type(name).__name__} is not a str")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise DcError(f"{name or 'model'}: {type(blob).__name__} is not bytes")
    b = bytes(blob)
    who = name or "model"
    size = len(b)
    isfinite = math.isfinite

    def need(off: int, n: int, what: str) -> None:
        if off < 0 or off + n > size:
            raise DcError(f"{who}: {what} at offset {off:#x} needs {n} bytes, the file ends at {size:#x}")

    def u32(off: int) -> int:
        return struct.unpack_from("<I", b, off)[0]

    def f32s(fmt: str, off: int, what: str) -> tuple:
        vals = struct.unpack_from(fmt, b, off)
        if not all(isfinite(x) for x in vals):
            raise DcError(f"{who}: {what} at offset {off:#x} is not finite: {vals!r}")
        return vals

    need(0, HEADER_SIZE, "header")
    hdr = struct.unpack_from("<12I", b, 0)
    nobj, ntex = hdr[3] & 0xFFFF, hdr[3] >> 16
    if hdr[4] != HEADER_SIZE or hdr[5] != HEADER_SIZE + 4 * nobj:
        raise DcError(f"{who}: header at offset 0x10: object table {hdr[4]:#x}, texture table {hdr[5]:#x}; "
                      f"expected {HEADER_SIZE:#x} and {HEADER_SIZE + 4 * nobj:#x} for {nobj} objects")
    if hdr[6] != size:
        raise DcError(f"{who}: header at offset 0x18 says the file is {hdr[6]:#x} bytes, it is {size:#x}")
    need(hdr[4], 4 * nobj, "object table")
    need(hdr[5], TEX_ENTRY_SIZE * ntex, "texture table")
    textures = []
    for i in range(ntex):
        w, h, fmt, off, raw = struct.unpack_from("<HHIII", b, hdr[5] + TEX_ENTRY_SIZE * i)
        textures.append(TexEntry(w, h, fmt, off, raw))

    table_end = hdr[5] + TEX_ENTRY_SIZE * ntex
    cursor = table_end + _pad(table_end)
    need(table_end, cursor - table_end, "padding before the first object")
    if any(b[table_end:cursor]):
        raise DcError(f"{who}: the padding at offset {table_end:#x} is not zero")
    objects = []
    for oi in range(nobj):
        entry = hdr[4] + 4 * oi
        o = u32(entry)
        if not o & 1 or (o & ~1) != cursor:
            raise DcError(f"{who}: object {oi} at table offset {entry:#x} is {o:#x}; expected {cursor | 1:#x} "
                          f"(objects follow one another, bit 0 set)")
        need(cursor, OBJECT_HEAD, f"object {oi} header")
        w0, w1 = struct.unpack_from("<2I", b, cursor)
        if w0 != 1:
            raise DcError(f"{who}: object {oi} at offset {cursor:#x}: first word {w0:#x}, expected 1")
        osphere = f32s("<4f", cursor + 8, f"object {oi} sphere")
        q = cursor + OBJECT_HEAD
        meshes = []
        entries_total = 0
        while True:
            need(q, 4, f"object {oi} mesh list")
            if u32(q) == 0:
                break
            need(q, MESH_HEAD, f"object {oi} mesh header")
            params = struct.unpack_from("<4I", b, q)
            msphere = f32s("<4f", q + 0x10, f"object {oi} mesh sphere")
            tex, kind = struct.unpack_from("<ii", b, q + 0x20)
            if tex != NO_TEXTURE and not 0 <= tex < ntex:
                raise DcError(f"{who}: object {oi} mesh at offset {q:#x}: texture index {tex} is outside the "
                              f"{ntex}-entry texture table")
            colours = f32s("<9f", q + 0x28, f"object {oi} mesh colours")
            ssize = u32(q + 0x4C)
            s, e = q + MESH_HEAD, q + MESH_HEAD + ssize
            need(s, ssize, f"object {oi} mesh strip data ({ssize:#x} bytes)")
            vat: set[int] = set()           # re-uses reach only this mesh's own vertices
            vs = _VERTEX_I if _raw_normal(params) else _VERTEX_F
            strips = []
            while s < e:
                if s + STRIP_HEAD > e:
                    raise DcError(f"{who}: object {oi} strip header at offset {s:#x} runs past the mesh's "
                                  f"strip data (ends {e:#x})")
                fl, c = struct.unpack_from("<II", b, s)
                s += STRIP_HEAD
                entries = []
                for _ in range(c * 3 if fl & LIST_FLAG else c):
                    if s + REUSE_SIZE > e:
                        raise DcError(f"{who}: object {oi} strip entry at offset {s:#x} runs past the mesh's "
                                      f"strip data (ends {e:#x})")
                    w = u32(s)
                    if _is_ref(w):
                        off = struct.unpack_from("<i", b, s + 4)[0]
                        if s + REUSE_SIZE + off not in vat:
                            raise DcError(f"{who}: object {oi} re-use at offset {s:#x}: offset {off} lands on "
                                          f"{s + REUSE_SIZE + off:#x}, which is not an earlier vertex of this mesh")
                        entries.append(Reuse(w, off))
                        s += REUSE_SIZE
                    else:
                        if s + VERTEX_SIZE > e:
                            raise DcError(f"{who}: object {oi} vertex at offset {s:#x} runs past the mesh's "
                                          f"strip data (ends {e:#x})")
                        v = vs.unpack_from(b, s)
                        if vs is _VERTEX_F:
                            if not all(isfinite(x) for x in v):
                                raise DcError(f"{who}: vertex at offset {s:#x} is not finite: {v!r}")
                        elif not all(isfinite(x) for x in v[:3] + v[6:]):
                            raise DcError(f"{who}: vertex at offset {s:#x} is not finite: {v!r}")
                        vat.add(s)
                        entries.append(Vertex(v[0], v[1], v[2], v[3:6], v[6], v[7]))
                        s += VERTEX_SIZE
                    entries_total += 1
                strips.append(Strip(fl, tuple(entries)))
            meshes.append(Mesh(params, msphere, tex, kind, colours, tuple(strips)))
            q = e
        need(q, 8, f"object {oi} terminator")
        tail = u32(q + 4)
        if tail != entries_total:
            raise DcError(f"{who}: object {oi} at offset {q + 4:#x}: strip count {tail}, its strips hold "
                          f"{entries_total} entries")
        objects.append(Object(w1, osphere, tuple(meshes), tail))
        cursor = q + 8
    if cursor != size:
        raise DcError(f"{who}: the objects end at offset {cursor:#x}, the file at {size:#x}")
    return Model(tuple(hdr), tuple(objects), tuple(textures))


# ---- encode ----

def fix_header(m: Model) -> Model:
    """`m` with header words 3-6 (counts, table offsets, file size) set from its contents."""
    if not isinstance(m, Model):
        raise DcError(f"{type(m).__name__} is not a Model")
    h = list(_seq(m.header, HEADER_WORDS, "header", "model", DcError))
    nobj, ntex = len(m.objects), len(m.textures)
    if nobj > 0xFFFF or ntex > 0xFFFF:
        raise DcError(f"model: {nobj} objects and {ntex} textures do not fit 16 bits each")
    h[3] = ntex << 16 | nobj
    h[4] = HEADER_SIZE
    h[5] = HEADER_SIZE + 4 * nobj
    h[6] = len(_encode(replace(m, header=tuple(h[:6]) + (0,) + tuple(h[7:]))))
    return replace(m, header=tuple(h))


def _u32(v, name: str, what: str) -> int:
    return _int(v, name, 0, 0xFFFFFFFF, what, DcError)


def encode_pol(m: Model) -> bytes:
    """The file bytes of `m`. DcError for anything the format cannot hold or that disagrees: a non-finite or
    out-of-single-precision float, a re-use whose offset does not land on an earlier vertex, a texture index past the
    table, an object's strip count that differs from its entries, header words 3-6 that differ from the contents."""
    out = _encode(m)
    h = m.header
    if h[6] != len(out):
        raise DcError(f"model: header word 6 (file size) is {h[6]:#x}, the encoded file is {len(out):#x} bytes "
                      f"(fix_header sets it)")
    return out


def _encode(m: Model) -> bytes:
    if not isinstance(m, Model):
        raise DcError(f"{type(m).__name__} is not a Model")
    hdr = tuple(_u32(w, f"word {i}", "model header")
                for i, w in enumerate(_seq(m.header, HEADER_WORDS, "header", "model", DcError)))
    objects = _seq(m.objects, None, "objects", "model", DcError)
    textures = _seq(m.textures, None, "textures", "model", DcError)
    nobj, ntex = len(objects), len(textures)
    if nobj > 0xFFFF or ntex > 0xFFFF:
        raise DcError(f"model: {nobj} objects and {ntex} textures do not fit 16 bits each")
    want = (ntex << 16 | nobj, HEADER_SIZE, HEADER_SIZE + 4 * nobj)
    if hdr[3:6] != want:
        raise DcError(f"model: header words 3-5 are {tuple(hex(w) for w in hdr[3:6])}, the contents give "
                      f"{tuple(hex(w) for w in want)} (fix_header sets them)")
    table_end = hdr[5] + TEX_ENTRY_SIZE * ntex
    pad = _pad(table_end)
    table_end += pad
    tex_blob = bytearray()
    for i, t in enumerate(textures):
        if not isinstance(t, TexEntry):
            raise DcError(f"texture {i}: {type(t).__name__} is not a TexEntry")
        what = f"texture {i}"
        tex_blob += struct.pack("<HHIII", _int(t.width, "width", 0, 0xFFFF, what, DcError),
                                _int(t.height, "height", 0, 0xFFFF, what, DcError),
                                _u32(t.fmt, "fmt", what), _u32(t.offset, "offset", what), _u32(t.raw, "raw", what))
    body = bytearray()
    starts = []
    for oi, ob in enumerate(objects):
        if not isinstance(ob, Object):
            raise DcError(f"object {oi}: {type(ob).__name__} is not an Object")
        what = f"object {oi}"
        starts.append(table_end + len(body))
        body += struct.pack("<2I", 1, _u32(ob.kind, "kind", what))
        body += _f32(ob.sphere, 4, "sphere", what, DcError)
        total = 0
        for mi, me in enumerate(_seq(ob.meshes, None, "meshes", what, DcError)):
            if not isinstance(me, Mesh):
                raise DcError(f"{what} mesh {mi}: {type(me).__name__} is not a Mesh")
            mwhat = f"{what} mesh {mi}"
            params = tuple(_u32(p, "parameter word", mwhat) for p in _seq(me.params, 4, "params", mwhat, DcError))
            if params[0] == 0:
                raise DcError(f"{mwhat}: parameter word 0 is zero, which ends the object's mesh list")
            tex = _int(me.texture, "texture", -(1 << 31), (1 << 31) - 1, mwhat, DcError)
            if tex != NO_TEXTURE and not 0 <= tex < ntex:
                raise DcError(f"{mwhat}: texture index {tex} is outside the {ntex}-entry texture table")
            sdata = bytearray()
            vat: set[int] = set()           # re-uses reach only this mesh's own vertices
            base = table_end + len(body) + MESH_HEAD
            raw_normal = _raw_normal(params)
            for si, st in enumerate(_seq(me.strips, None, "strips", mwhat, DcError)):
                if not isinstance(st, Strip):
                    raise DcError(f"{mwhat} strip {si}: {type(st).__name__} is not a Strip")
                swhat = f"{mwhat} strip {si}"
                flags = _u32(st.flags, "flags", swhat)
                entries = _seq(st.entries, None, "entries", swhat, DcError)
                if flags & LIST_FLAG and len(entries) % 3:
                    raise DcError(f"{swhat}: a triangle list holds {len(entries)} entries, not a multiple of 3")
                sdata += struct.pack("<II", flags, len(entries) // 3 if flags & LIST_FLAG else len(entries))
                for ei, ent in enumerate(entries):
                    pos = base + len(sdata)
                    ewhat = f"{swhat} entry {ei} (offset {pos:#x})"
                    if isinstance(ent, Reuse):
                        raw = _u32(ent.raw, "raw word", ewhat)
                        off = _int(ent.offset, "offset", -(1 << 31), (1 << 31) - 1, ewhat, DcError)
                        if not _is_ref(raw):
                            raise DcError(f"{ewhat}: re-use word {raw:#x} is not 0x5ff..... (it would read as a "
                                          f"vertex)")
                        if pos + REUSE_SIZE + off not in vat:
                            raise DcError(f"{ewhat}: re-use offset {off} lands on {pos + REUSE_SIZE + off:#x}, "
                                          f"which is not an earlier vertex of this mesh")
                        sdata += struct.pack("<Ii", raw, off)
                    elif isinstance(ent, Vertex):
                        sdata += _vertex(ent, raw_normal, ewhat)
                        vat.add(pos)
                    else:
                        raise DcError(f"{ewhat}: {type(ent).__name__} is neither a Vertex nor a Reuse")
                    total += 1
            body += struct.pack("<4I", *params)
            body += _f32(me.sphere, 4, "sphere", mwhat, DcError)
            body += struct.pack("<ii", tex, _int(me.kind, "kind", -(1 << 31), (1 << 31) - 1, mwhat, DcError))
            body += _f32(me.colours, 9, "colours", mwhat, DcError)
            body += struct.pack("<I", len(sdata))
            body += sdata
        if _int(ob.strip_count, "strip count", 0, 0xFFFFFFFF, what, DcError) != total:
            raise DcError(f"{what}: strip count {ob.strip_count} but its strips hold {total} entries")
        body += struct.pack("<2I", 0, total)
    head = struct.pack("<12I", *hdr)
    table = b"".join(struct.pack("<I", s | 1) for s in starts)
    return head + table + bytes(tex_blob) + bytes(pad) + bytes(body)


def _vertex(v: Vertex, raw_normal: bool, what: str) -> bytes:
    xyz = _f32((v.x, v.y, v.z), 3, "position", what, DcError)
    uv = _f32((v.u, v.v), 2, "u, v", what, DcError)
    normal = _seq(v.normal, 3, "normal", what, DcError)
    if raw_normal:
        nb = struct.pack("<3I", *(_u32(w, "raw normal word", what) for w in normal))
    else:
        nb = _f32(normal, 3, "normal", what, DcError)
    blob = xyz + nb + uv
    if _is_ref(struct.unpack_from("<I", blob)[0]):
        raise DcError(f"{what}: x = {v.x!r} has the bit pattern of a re-use entry")
    return blob


# ---- triangles ----

def _walkable(m) -> None:
    """DcError unless `m` is what triangles and vertex_count walk: a Model whose textures, objects, meshes, strips
    and entries are sequences of the pol types, each strip's flags an integer and each re-use's offset an integer."""
    if not isinstance(m, Model):
        raise DcError(f"{type(m).__name__} is not a Model")
    _seq(m.textures, None, "textures", "model", DcError)
    for oi, ob in enumerate(_seq(m.objects, None, "objects", "model", DcError)):
        if not isinstance(ob, Object):
            raise DcError(f"object {oi}: {type(ob).__name__} is not an Object")
        what = f"object {oi}"
        for mi, me in enumerate(_seq(ob.meshes, None, "meshes", what, DcError)):
            if not isinstance(me, Mesh):
                raise DcError(f"{what} mesh {mi}: {type(me).__name__} is not a Mesh")
            mwhat = f"{what} mesh {mi}"
            for si, st in enumerate(_seq(me.strips, None, "strips", mwhat, DcError)):
                if not isinstance(st, Strip):
                    raise DcError(f"{mwhat} strip {si}: {type(st).__name__} is not a Strip")
                swhat = f"{mwhat} strip {si}"
                _u32(st.flags, "flags", swhat)
                for ei, e in enumerate(_seq(st.entries, None, "entries", swhat, DcError)):
                    if isinstance(e, Reuse):
                        _int(e.offset, "offset", -(1 << 31), (1 << 31) - 1, f"{swhat} entry {ei}", DcError)
                    elif not isinstance(e, Vertex):
                        raise DcError(f"{swhat} entry {ei}: {type(e).__name__} is neither a Vertex nor a Reuse")


def triangles(m: Model) -> Iterator[tuple[Vertex, Vertex, Vertex, int]]:
    """Every triangle of `m` as (a, b, c, object index), re-uses resolved; strips alternate winding (every odd
    triangle swaps its first two vertices), triangle lists are as stored. Degenerate triangles are kept. DcError (at
    the call) for an `m` that is not a Model of the pol types (`_walkable`); while iterating, for a re-use that does
    not land on an earlier vertex of its mesh."""
    _walkable(m)
    return _triangles(m)


def _triangles(m: Model) -> Iterator[tuple[Vertex, Vertex, Vertex, int]]:
    pos = HEADER_SIZE + 4 * len(m.objects) + TEX_ENTRY_SIZE * len(m.textures)
    pos += _pad(pos)
    for oi, ob in enumerate(m.objects):
        pos += OBJECT_HEAD
        for me in ob.meshes:
            pos += MESH_HEAD
            at: dict[int, Vertex] = {}      # re-uses reach only this mesh's own vertices
            for st in me.strips:
                pos += STRIP_HEAD
                ents = []
                for e in st.entries:
                    if isinstance(e, Reuse):
                        target = pos + REUSE_SIZE + e.offset
                        if target not in at:
                            raise DcError(f"object {oi}: re-use at offset {pos:#x} lands on {target:#x}, which is "
                                          f"not an earlier vertex of this mesh")
                        ents.append(at[target])
                        pos += REUSE_SIZE
                    else:
                        at[pos] = e
                        ents.append(e)
                        pos += VERTEX_SIZE
                if st.flags & LIST_FLAG:
                    for k in range(0, len(ents) - len(ents) % 3, 3):
                        yield ents[k], ents[k + 1], ents[k + 2], oi
                else:
                    for k in range(len(ents) - 2):
                        if k & 1:
                            yield ents[k + 1], ents[k], ents[k + 2], oi
                        else:
                            yield ents[k], ents[k + 1], ents[k + 2], oi
        pos += 8


def vertex_count(m: Model) -> int:
    """The stored vertices of `m` (re-uses not counted). DcError for an `m` that is not a Model of the pol types."""
    _walkable(m)
    return sum(isinstance(e, Vertex) for ob in m.objects for me in ob.meshes for st in me.strips for e in st.entries)


# ---- the disc ----

def pol_names() -> list[str]:
    """The 64 model file names: 8 courses x 4 variants x (CS, VCS)."""
    from dtools.dc.disc import COURSE_COUNT, VARIANTS, course_files
    return [course_files(n, v, kind) for n in range(COURSE_COUNT) for v in VARIANTS for kind in ("model", "model_v")]


def verify_lossless_pol(disc) -> list[str]:
    """One line per model file that `encode_pol(decode_pol(file))` does not reproduce byte for byte, or that does not
    decode ([] when all 64 are lossless)."""
    out = []
    for name in pol_names():
        blob = disc.read(name)
        try:
            back = encode_pol(decode_pol(blob, name))
        except DcError as exc:
            out.append(f"{name}: {exc}")
            continue
        if back != blob:
            k = next((k for k, (a, b) in enumerate(zip(blob, back)) if a != b), min(len(blob), len(back)))
            out.append(f"{name}: first difference at offset {k:#x} (file {len(blob):#x} bytes, encoded {len(back):#x})")
    return out
