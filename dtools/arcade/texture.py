"""Decode and encode Model 2 (original) texture headers and texels for Daytona USA.

Ports MAME 0.289 src/mame/sega/model2_v.cpp (`model2_3d_process_polygon`, `model2_3d_render`) and model2rd.ipp
(`get_texel`, `fetch_bilinear_texel`, `draw_scanline_tex`), BSD-3-Clause, copyright the MAME team; the game-side
upload is the i960 routine at program 0x146c (see docs/formats/model2-textures.md, which this module implements).

- The **textures region** (`RomSet.textures`, MAME's "textures") holds texture *points* (per-vertex u, v) and
  texture *headers*, as 16-bit words. `tpa` / `tha` are word indices into it (bit 23 set selects the rasterizer's
  own texture RAM instead, filled by geometry command 0x04; not decodable from ROM).
- The **texels** live in two 1 MB texture-RAM sheets (i960 0x12000000 / 0x12400000) that the game fills from
  **data ROM**: sheet 0 from a per-course 1 MB block, sheet 1 from a common block. MAME has one texel format: 4 bits
  per texel, a 2x2 texel block per 16-bit word.
- The **colour** path (texel -> luma RAM -> palette entry -> colour translation RAM -> MAME's gamma) uses tables the
  game builds in RAM at boot; all three come from ROM (luma tables and palette are copied from data ROM, the
  translation table is computed by program code) and are rebuilt here by `colour_tables`.
"""

from __future__ import annotations

import dataclasses
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dtools.arcade import course as course_mod
from dtools.arcade import polyrom
from dtools.arcade.romset import DATA_COPY_END, RomSet, RomSetError
from dtools.arcade.tablebase import DATA_BASE, DATA_WINDOW, DecodeError, _int, is_int


class TextureError(DecodeError):
    pass


TEX_ADDR_MASK = 0x7FFFFF             # model2_v.cpp: texture_rom_mask = bytes / 2 - 1 for the 16 MB region
TEX_RAM_SELECT = 0x800000            # tpa / tha bit 23: rasterizer texture RAM (raster->texture_ram), not the ROM

# The game's sheet upload (program ROM, i960 0x1438 / 0x1454 / 0x146c).
COURSE_SHEETS_AT = 0x15A0            # u32[4] i960 addresses, indexed by the course id byte at 0x00501460
COMMON_SHEET_AT = 0x1460             # displacement word of `lda 0x2500000,g5` at 0x145c
SHEET_BLOCK = 0x100000               # bytes of data ROM consumed per sheet upload
SHEET_WORDS = 0x80000                # u16 words per sheet (1 MB)
LEVEL0_WORDS = 0x60000               # words copied straight (level 0); the rest is the mip area
MIP_TABLE = (0x200, 0x100, 0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1)   # program 0x1578, the run-length table
SHEET_W, SHEET_H = 2048, 1024        # the logical sheet (model2rd.ipp get_texel: stored as 1024 x 2048)

# Colour tables (data ROM offsets).
LUMA_COUNT_AT = 0x802FC4             # u32: number of 128-byte luma tables (routine at program 0x1400)
LUMA_TABLES_AT = 0x802FC8
LUMA_SIZE = 0x8000                   # model2.cpp: m_lumaram, 0x8000 bytes
PALETTE_START, PALETTE_END = 0x84F4EC, 0x84FCCA   # copied to palette 0x01802000 (entry 0x1000) by program 0x5c7c
PALETTE_ENTRIES = 0x2000             # u16 palette RAM 0x01800000-0x01803fff
POLY_PALETTE = 0x1000                # model2rd.ipp: m_palram[colorbase + 0x1000]
XLAT_ENTRIES = 0x6000                # u16 colour translation RAM 0x01810000-0x0181bfff (r, g, b at +0, +0x2000, +0x4000)
LIGHT_MAX = 0xFF                     # object.luma: (command_buffer[9] >> 15) & 0xff


@dataclass(frozen=True)
class TexHeader:
    """One polygon's texture header: the four 16-bit words at `tha`, plus the attr's header step. Field names follow
    model2_v.cpp `model2_3d_render` (lines 567-601). `raw` holds the bits MAME does not read: (h1 >> 8, h2 bit 11,
    h3 & 0x3f)."""
    tha: int                 # the header's address as sent (u16 word index into the textures region)
    step: int                # attr bits 12-16, signed: the header address moves by step * 4 words after the polygon
    width: int               # 32 << h0[0:3]
    height: int              # 32 << h0[3:6]
    tex_x: int               # 32 * h2[0:6]   logical sheet x of texel (0, 0)
    tex_y: int               # 32 * h2[6:11]  logical sheet y
    sheet: int               # h2 bit 12: level 0 in texture RAM 1 (else 0); odd mip levels in the other sheet
    textured: bool           # h0 bit 14 (renderer bit 1)
    translucent: bool        # h0 bit 13 (renderer bit 0): texel 0xf is transparent
    checker: bool            # h0 bit 15: every other pixel drawn
    wrap_x: bool             # h0 bit 6 (smooth wrap; ignored when mirrored)
    wrap_y: bool             # h0 bit 7
    mirror_x: bool           # h0 bit 8
    mirror_y: bool           # h0 bit 9
    utex: bool               # h0 bit 12: microtexture
    utex_min_lod: int        # h0 bits 10-11
    utex_x: int              # 128 * h2 bit 13
    utex_y: int              # 128 * h2 bits 14-15
    luma_base: int           # h1 bits 0-7: luma table (128 bytes each)
    colour: int              # h3 bits 6-15: palette entry POLY_PALETTE + colour
    raw: tuple[int, int, int]

    @property
    def next_tha(self) -> int:
        return self.tha + 4 * self.step

    @property
    def words(self) -> tuple[int, int, int, int]:
        return struct.unpack("<4H", encode_tex_header(self))


def _u32(value, name: str) -> int:
    if not is_int(value) or not 0 <= value <= 0xFFFFFFFF:
        raise TextureError(f"{name} {value!r} is not a u32")
    return int(value)


def tex_header(rs: RomSet, tha: int, attr: int = 0) -> TexHeader:
    """The header at `tha` (a geometry `tha` word: bit 23 selects texture RAM, bits 0-22 the ROM word index) for a
    polygon whose attribute is `attr` (only bits 12-16, the header step, are used). TextureError for a texture-RAM
    header or one outside the textures region."""
    tha, attr = _u32(tha, "tha"), _u32(attr, "attr")
    if tha & TEX_RAM_SELECT:
        raise TextureError(f"tha {tha:#x} selects the rasterizer's texture RAM (bit 23), uploaded by geometry "
                           f"command 0x04 at run time; it is not in ROM")
    at = tha & TEX_ADDR_MASK
    if 2 * at + 8 > len(rs.textures):
        raise TextureError(f"tha {tha:#x}: the header at word {at:#x} is outside the textures region "
                           f"({len(rs.textures):#x} bytes)")
    h0, h1, h2, h3 = struct.unpack_from("<4H", rs.textures, 2 * at)
    step = (attr >> 12) & 0x1F
    if step & 0x10:
        step -= 32
    return TexHeader(
        tha=tha, step=step, width=32 << (h0 & 7), height=32 << ((h0 >> 3) & 7),
        tex_x=32 * (h2 & 0x3F), tex_y=32 * ((h2 >> 6) & 0x1F), sheet=(h2 >> 12) & 1,
        textured=bool(h0 >> 14 & 1), translucent=bool(h0 >> 13 & 1), checker=bool(h0 >> 15 & 1),
        wrap_x=bool(h0 >> 6 & 1), wrap_y=bool(h0 >> 7 & 1), mirror_x=bool(h0 >> 8 & 1), mirror_y=bool(h0 >> 9 & 1),
        utex=bool(h0 >> 12 & 1), utex_min_lod=(h0 >> 10) & 3, utex_x=128 * ((h2 >> 13) & 1),
        utex_y=128 * ((h2 >> 14) & 3), luma_base=h1 & 0xFF, colour=h3 >> 6, raw=(h1 >> 8, (h2 >> 11) & 1, h3 & 0x3F))


def _field(value, name: str, lo: int, hi: int) -> int:
    return _int(value, name, lo, hi, error=TextureError)


def _flag(value, name: str) -> int:
    if value not in (0, 1) or not isinstance(value, (bool, int, np.integer)):
        raise TextureError(f"{name} {value!r} is not a flag (False/True)")
    return int(value)


def _log2_size(value, name: str) -> int:
    if is_int(value):
        for n in range(8):
            if value == 32 << n:
                return n
    raise TextureError(f"{name} {value!r} is not 32 << n for n in 0..7 (32-4096)")


def _multiple(value, name: str, unit: int, count: int) -> int:
    v = _field(value, name, 0, unit * (count - 1))
    if v % unit:
        raise TextureError(f"{name} {value!r} is not a multiple of {unit}")
    return v // unit


def encode_tex_header(h: TexHeader) -> bytes:
    """The 8 bytes (four LE u16) of `h` as stored in the textures region; `tha` and `step` are not part of it.
    TextureError for a field that does not fit its bits."""
    if not isinstance(h.raw, (tuple, list)) or len(h.raw) != 3:
        raise TextureError(f"raw {h.raw!r} must be (h1 >> 8, h2 bit 11, h3 & 0x3f)")
    h0 = (_log2_size(h.width, "width") | _log2_size(h.height, "height") << 3 | _flag(h.wrap_x, "wrap_x") << 6
          | _flag(h.wrap_y, "wrap_y") << 7 | _flag(h.mirror_x, "mirror_x") << 8 | _flag(h.mirror_y, "mirror_y") << 9
          | _field(h.utex_min_lod, "utex_min_lod", 0, 3) << 10 | _flag(h.utex, "utex") << 12
          | _flag(h.translucent, "translucent") << 13 | _flag(h.textured, "textured") << 14
          | _flag(h.checker, "checker") << 15)
    h1 = _field(h.luma_base, "luma_base", 0, 0xFF) | _field(h.raw[0], "raw[0] (h1 >> 8)", 0, 0xFF) << 8
    h2 = (_multiple(h.tex_x, "tex_x", 32, 64) | _multiple(h.tex_y, "tex_y", 32, 32) << 6
          | _field(h.raw[1], "raw[1] (h2 bit 11)", 0, 1) << 11 | _field(h.sheet, "sheet", 0, 1) << 12
          | _multiple(h.utex_x, "utex_x", 128, 2) << 13 | _multiple(h.utex_y, "utex_y", 128, 4) << 14)
    h3 = _field(h.raw[2], "raw[2] (h3 & 0x3f)", 0, 0x3F) | _field(h.colour, "colour", 0, 0x3FF) << 6
    return struct.pack("<4H", h0, h1, h2, h3)


def object_headers(rs: RomSet, tha: int, obj) -> list[TexHeader]:
    """One header per record of a decoded object (`polyrom.Object`), in record order: the rasterizer reads the header
    at the current address for every polygon, culled or not, then adds the record's step * 4 (model2_v.cpp
    lines 404-423)."""
    out = []
    for rec in obj.records:
        h = tex_header(rs, tha, rec.attr)
        out.append(h)
        tha = (h.next_tha) & 0xFFFFFFFF
    return out


POINT_MAX = 0xFFFF                   # a texture point word: unsigned, 1/8 texel


def _vertex_counts(obj, what: str) -> list[int]:
    """3 or 4 per record of a decoded object (attr bit 0: a quad reads 4 points, a triangle 3), move records included.
    TextureError for an `obj` that is not a polyrom.Object of records with u32 attributes."""
    if not isinstance(obj, polyrom.Object):
        raise TextureError(f"{what}: {obj!r:.40} is not a polyrom.Object")
    if not isinstance(obj.records, (list, tuple)):
        raise TextureError(f"{what}: the object's records {obj.records!r:.40} are not a list of polyrom.Record")
    out = []
    for k, r in enumerate(obj.records):
        if not isinstance(r, polyrom.Record) or not is_int(r.attr) or not 0 <= r.attr <= 0xFFFFFFFF:
            raise TextureError(f"{what}: record {k} {r!r:.40} is not a polyrom.Record with a u32 attr")
        out.append(4 if r.attr & 1 else 3)
    return out


def texture_points(rs: RomSet, tpa: int, obj) -> tuple[tuple[tuple[int, int], ...], ...]:
    """The texture points a decoded object's polygons read from `tpa` (a geometry `tpa` word: bit 23 selects texture
    RAM, bits 0-22 the ROM word index): per record, in record order, its 3 or 4 vertices' (v, u), unsigned, in 1/8
    texel (model2_v.cpp lines 383-402: 2 words per vertex, v then u; `tpa` advances by 2 * vertices words). Every
    record reads them, a culled link-0 (move) record included, and an untextured one (docs/formats/model2-textures.md,
    "Texture points"). TextureError for an `rs` that is not a RomSet, a `tpa` that is not a u32 or selects texture RAM,
    an `obj` that is not a polyrom.Object, or points past the end of the textures region."""
    if not isinstance(rs, RomSet):
        raise TextureError(f"{rs!r:.40} is not a RomSet")
    tpa = _u32(tpa, "tpa")
    if tpa & TEX_RAM_SELECT:
        raise TextureError(f"tpa {tpa:#x} selects the rasterizer's texture RAM (bit 23), uploaded by geometry command "
                           f"0x04 at run time; it is not in ROM")
    counts = _vertex_counts(obj, f"texture points at tpa {tpa:#x}")
    at, n = tpa & TEX_ADDR_MASK, 2 * sum(counts)
    if 2 * (at + n) > len(rs.textures):
        raise TextureError(f"tpa {tpa:#x}: {n} point words from word {at:#x} run outside the textures region "
                           f"({len(rs.textures):#x} bytes)")
    words = struct.unpack_from(f"<{n}H", rs.textures, 2 * at)
    out, k = [], 0
    for c in counts:
        out.append(tuple((words[k + 2 * j], words[k + 2 * j + 1]) for j in range(c)))
        k += 2 * c
    return tuple(out)


def encode_texture_points(points, obj=None) -> bytes:
    """The inverse of `texture_points`: per polygon 3 or 4 (v, u) pairs of integers 0-0xffff (Python or numpy, not
    bool), as LE u16 words in order. With `obj` (a polyrom.Object) each polygon must have its record's vertex count
    (4 for a quad, 3 for a triangle) and there must be one per record. TextureError otherwise (a tuple or list of
    polygons, each a tuple or list of pairs)."""
    counts = _vertex_counts(obj, "encode_texture_points") if obj is not None else None
    if not isinstance(points, (tuple, list)):
        raise TextureError(f"texture points {points!r:.40} are not a tuple or list of polygons")
    if counts is not None and len(points) != len(counts):
        raise TextureError(f"{len(points)} polygons of texture points for an object of {len(counts)} records")
    words = []
    for k, poly in enumerate(points):
        if not isinstance(poly, (tuple, list)):
            raise TextureError(f"texture points: polygon {k} {poly!r:.40} is not a tuple or list of (v, u) pairs")
        if len(poly) not in (3, 4):
            raise TextureError(f"texture points: polygon {k} has {len(poly)} vertices, not 3 or 4")
        if counts is not None and len(poly) != counts[k]:
            raise TextureError(f"texture points: polygon {k} has {len(poly)} vertices; its record reads "
                               f"{counts[k]} vertices")
        for pair in poly:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TextureError(f"texture points: polygon {k}: {pair!r:.40} is not a (v, u) pair")
            for w in pair:
                if not is_int(w) or not 0 <= w <= POINT_MAX:
                    raise TextureError(f"texture points: polygon {k}: {w!r:.40} is not a u16 (0-{POINT_MAX:#x}, "
                                       f"1/8 texel)")
                words.append(int(w))
    return struct.pack(f"<{len(words)}H", *words)


# --- texels ---------------------------------------------------------------------------------------------------

def sheet_source(rs: RomSet, sheet: int, course: int | None = None) -> int:
    """Data-ROM offset of the 1 MB block the game uploads into texture RAM `sheet`: sheet 0 from the course table at
    program COURSE_SHEETS_AT (by course id; on a grown set the sixteen-entry table course.id_table(rs,
    "COURSE_SHEETS")), sheet 1 from the common block (program COMMON_SHEET_AT). TextureError for a sheet 0 without a
    course id of the set (0-2, or 0-15 grown; bool refused), a pointer word the set does not hold, or a pointer that
    is not a 1 MB block of the set's data ROM; for `rs` not a RomSet."""
    C = course_mod
    if not isinstance(rs, RomSet):
        raise TextureError(f"{type(rs).__name__} is not a RomSet")
    if sheet == 1:
        at = COMMON_SHEET_AT
    elif sheet == 0:
        n = C.id_count(rs)
        if not is_int(course) or not 0 <= course < n:
            raise TextureError(f"texture sheet 0 is the course's: give a course id 0-{n - 1}, not {course!r}")
        at = C.id_table(rs, "COURSE_SHEETS") + 4 * int(course)
    else:
        raise TextureError(f"sheet {sheet!r} is not 0 or 1")
    try:
        (ptr,) = struct.unpack("<I", rs.read_i960(at, 4))
    except RomSetError as e:
        raise TextureError(f"sheet {sheet} source pointer at i960 {at:#x}: {e}") from None
    off = ptr - DATA_BASE
    if not 0 <= off <= len(rs.data) - SHEET_BLOCK:
        raise TextureError(f"sheet {sheet} source pointer at i960 {at:#x} is {ptr:#x}, not a 1 MB data-ROM block")
    return off


def custom_sheet_blocks(rs: RomSet) -> tuple[tuple[str, int, int, str], ...]:
    """("data", offset, SHEET_BLOCK, owner) of every course sheet block a grown set's ids 3-15 name (`sheet_source(rs,
    0, id)`, COURSE_SHEETS[id]) that lies in the custom data area (from romset.DATA_COPY_END): a new id's own 1 MB
    texel block, Desert City's at id 3. The block is owned whatever its bytes (texels hold long 0xff runs), so the
    free-space writers and allocators treat it as busy (tgp.write_free, golden.owned_spans). A block several ids name
    is listed once, the owner naming them all; ascending; () for a set that is not grown. TextureError as
    `sheet_source` (a pointer word that is not a 1 MB block of the set's data ROM; `rs` not a RomSet)."""
    C = course_mod
    if not isinstance(rs, RomSet):
        raise TextureError(f"{type(rs).__name__} is not a RomSet")
    try:
        if not C.grown(rs):
            return ()
    except DecodeError as e:
        raise TextureError(f"course sheet blocks: {e}") from e
    ids: dict[int, list[int]] = {}
    for k in range(C.COURSE_COUNT, C.COURSE_IDS):
        off = sheet_source(rs, 0, k)
        if off >= DATA_COPY_END:
            ids.setdefault(off, []).append(k)
    return tuple(("data", off, SHEET_BLOCK, "the course sheet block of "
                  + ", ".join(f"COURSE_SHEETS[{k}]" for k in ks)) for off, ks in sorted(ids.items()))


def encode_sheet_source(off: int) -> bytes:
    """A sheet-source word (COURSE_SHEETS_AT[id] or the displacement at COMMON_SHEET_AT): the i960 address of a 1 MB
    data-ROM block at offset `off`. TextureError for a block that does not fit data ROM."""
    if not is_int(off) or not 0 <= off <= DATA_WINDOW - SHEET_BLOCK:
        raise TextureError(f"sheet source {off!r} is not the data-ROM offset of a 1 MB block")
    return struct.pack("<I", DATA_BASE + int(off))


_GEOMETRY = ("width", "height", "tex_x", "tex_y", "sheet", "utex_x", "utex_y")


def _plain(h: TexHeader) -> TexHeader:
    """`h` checked by `encode_tex_header` (TextureError for a field that does not fit its bits), with its geometry
    fields as Python ints: a numpy integer is accepted, but never used in arithmetic (a numpy uint8 tex_x overflows or
    wraps, a numpy width has no bit_length)."""
    if not isinstance(h, TexHeader):
        raise TextureError(f"{h!r:.40} is not a texture.TexHeader")
    encode_tex_header(h)
    return dataclasses.replace(h, **{f: int(getattr(h, f)) for f in _GEOMETRY})


def _check_textured(h: TexHeader) -> None:
    if not isinstance(h, TexHeader):
        raise TextureError(f"{h!r:.40} is not a texture.TexHeader")
    if not h.textured:
        raise TextureError(f"header at {h.tha:#x} is not textured (h0 bit 14 clear): it has no texels")


def texel_spans(rs: RomSet, h: TexHeader, course: int | None = None) -> list[tuple[int, int]]:
    """(data-ROM offset, byte count) runs that hold `h`'s level-0 texels, in the order `encode_texels` writes them:
    for each pair of texel rows, the part at logical x < 1024 then the part at x >= 1024. Logical (x, y) is stored
    at (x, y) for x < 1024 and at (x - 1024, y ^ 1024) otherwise (model2rd.ipp get_texel); stored (x, y) is sheet
    word (y // 2) * 512 + x // 2, which is the same word of the sheet's data-ROM block (the game copies the first
    0x60000 words as they are). TextureError for an untextured header, a header field that `encode_tex_header`
    refuses (a size not 32 << n, a position not a multiple of 32, ...), a texture outside the 2048 x 1024 sheet, or
    one reaching into the mip area (x >= 1024 and y >= 512), which holds no level-0 texels."""
    return texel_spans_at(h, sheet_source(rs, h.sheet, course))


def texel_spans_at(h: TexHeader, base: int) -> list[tuple[int, int]]:
    """`texel_spans` for a sheet whose 1 MB data-ROM block starts at offset `base` (`sheet_source`'s value): for a
    caller that holds the block offsets rather than a RomSet (a Course re-emitting its texels)."""
    _check_textured(h)
    h = _plain(h)                         # the geometry must be a header's: sizes 32 << n, positions 32-aligned
    x0, y0, w, hh = h.tex_x, h.tex_y, h.width, h.height
    if x0 + w > SHEET_W or y0 + hh > SHEET_H:
        raise TextureError(f"header at {h.tha:#x}: texture {w}x{hh} at ({x0}, {y0}) is outside the "
                           f"{SHEET_W}x{SHEET_H} sheet")
    if x0 + w > 1024 and y0 + hh > 512:
        raise TextureError(f"header at {h.tha:#x}: texture {w}x{hh} at ({x0}, {y0}) reaches into the mip area "
                           f"(x >= 1024, y >= 512), which holds mip levels, not level-0 texels")
    spans = []
    for y in range(y0, y0 + hh, 2):
        for xa, xb in ((x0, min(x0 + w, 1024)), (max(x0, 1024), x0 + w)):
            if xa >= xb:
                continue
            xs, ys = (xa, y) if xa < 1024 else (xa - 1024, y ^ 1024)
            spans.append((base + 2 * ((ys // 2) * 512 + xs // 2), xb - xa))
    return spans


def _unpack(blob: bytes, w: int, hh: int) -> np.ndarray:
    """Packed rows (w // 2 LE u16 per two texel rows) -> (hh, w) texel indices."""
    b = np.frombuffer(blob, np.uint8).reshape(hh // 2, w // 2, 2)
    out = np.empty((hh, w), np.uint8)
    out[0::2, 0::2], out[0::2, 1::2] = b[:, :, 1] >> 4, b[:, :, 1] & 0xF    # even y: the high byte
    out[1::2, 0::2], out[1::2, 1::2] = b[:, :, 0] >> 4, b[:, :, 0] & 0xF    # odd y: the low byte
    return out


def decode_indices(rs: RomSet, h: TexHeader, course: int | None = None) -> np.ndarray:
    """`h`'s level-0 texels as raw 4-bit indices, shape (height, width), uint8, read from data ROM. `course` (0-2)
    is required for sheet 0. TextureError as `texel_spans`."""
    spans = texel_spans(rs, h, course)
    return _unpack(b"".join(rs.data[o:o + n] for o, n in spans), h.width, h.height)


def encode_texels(h: TexHeader, indices) -> bytes:
    """The inverse of `decode_indices`: `indices` (height x width integers 0-15) packed as the sheet stores them,
    i.e. the bytes of `texel_spans` concatenated. TextureError for an untextured header, a wrong shape, a
    non-integer array or a value outside 0-15."""
    _check_textured(h)
    h = _plain(h)
    a = np.asarray(indices)
    if a.shape != (h.height, h.width):
        raise TextureError(f"indices shape {a.shape} is not (height, width) = {(h.height, h.width)}")
    if not np.issubdtype(a.dtype, np.integer):
        raise TextureError(f"indices must be an integer array, not {a.dtype}")
    if a.size and (int(a.min()) < 0 or int(a.max()) > 15):
        raise TextureError(f"indices must be 0-15 (4 bits per texel), got {int(a.min())}..{int(a.max())}")
    a = a.astype(np.uint8)
    out = np.empty((h.height // 2, h.width // 2, 2), np.uint8)
    out[:, :, 1] = a[0::2, 0::2] << 4 | a[0::2, 1::2]
    out[:, :, 0] = a[1::2, 0::2] << 4 | a[1::2, 1::2]
    return out.tobytes()


MIPS_MODES = ("keep", "decimate")


def with_texels(rs: RomSet, h: TexHeader, indices, course: int | None = None, *, mips: str) -> RomSet:
    """A copy of `rs` with `indices` written as `h`'s level-0 texels in data ROM, and the mip levels as `mips` says
    (required, no default: the game copies its mip levels from ROM and never computes them, so the caller decides):

    - "keep": the mip area is left as it is, so the texture's mip levels (seen at a distance) still show the OLD
      texels;
    - "decimate": every mip level the texture uses (`mip_levels(h)`) is rewritten from `indices` by nearest
      sampling, level k taking every 2^k-th texel in x and y (4-bit indices cannot be averaged). Sega's own filter
      is not known; this is our choice.

    TextureError as `encode_texels` / `texel_spans` / `mip_texels`, or for any other `mips`."""
    if mips not in MIPS_MODES:
        raise TextureError(f"mips {mips!r} is not one of {MIPS_MODES}")
    blob = encode_texels(h, indices)
    buf, pos = bytearray(rs.data), 0
    spans = texel_spans(rs, h, course)
    for off, n in spans:
        buf[off:off + n] = blob[pos:pos + n]
        pos += n
    lo, hi = min(off for off, _ in spans), max(off + n for off, n in spans)
    if mips == "decimate":
        a = np.asarray(indices).astype(np.uint8)
        for level in mip_levels(h):
            off, high = mip_texels(rs, h, level, course)
            vals = a[::1 << level, ::1 << level]
            for o, hb, v in zip(off.ravel().tolist(), high.ravel().tolist(), vals.ravel().tolist()):
                buf[o] = (buf[o] & 0x0F) | (v << 4) if hb else (buf[o] & 0xF0) | v
            lo, hi = min(lo, int(off.min())), max(hi, int(off.max()) + 1)
    # one patch: every byte lies inside one sheet block (mip_texels refuses another block), and RomSet.with_patch
    # keeps the copied 1 MB blocks above 0x800000 in step should a block ever sit there
    return rs.with_patch("data", lo, bytes(buf[lo:hi]))


def mip_levels(h: TexHeader) -> range:
    """The mip levels MAME samples for `h` besides level 0: 1 .. log2(min(width, height)) - 1, down to 2 x 2
    (model2rd.ipp draw_scanline_tex: max_level = 30 - countl_zero(min(width, height)))."""
    _check_textured(h)
    h = _plain(h)
    return range(1, min(h.width, h.height).bit_length() - 1)   # 32 -> 1..4 (16, 8, 4, 2 texels)


def mip_texels(rs: RomSet, h: TexHeader, level: int, course: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Where texel (u, v) of mip `level` of `h` lives in data ROM: (byte offset, high nibble?) arrays of shape
    (height >> level, width >> level). MAME reads it at logical (((tex_x - 2048) >> level) & 2047 + u,
    ((tex_y - 1024) >> level) & 1023 + v) in sheet `sheet ^ (level & 1)` (fetch_bilinear_texel), word and nibble
    as get_texel (nibble chosen by the parity of u, v). That sheet word comes from the same word of one of the two
    uploaded blocks (the inverse of the upload's mip placement, `_mip_other`). TextureError for a level outside
    `mip_levels(h)` or a texel whose word comes from the other sheet's block (a write there would change another
    texture's block); no course texture has one."""
    return mip_texels_at(h, level, sheet_source(rs, h.sheet, course))


def mip_texels_at(h: TexHeader, level: int, base: int) -> tuple[np.ndarray, np.ndarray]:
    """`mip_texels` for a sheet block at data-ROM offset `base` (see `texel_spans_at`)."""
    if level not in mip_levels(h):
        raise TextureError(f"header at {h.tha:#x}: level {level!r} is not one of its mip levels "
                           f"{list(mip_levels(h))}")
    h = _plain(h)
    texel_spans_at(h, base)               # level 0 must be valid (inside the sheet, outside the mip area)
    bx, by = ((h.tex_x - 2048) >> level) & 2047, ((h.tex_y - 1024) >> level) & 1023
    v, u = np.mgrid[0:h.height >> level, 0:h.width >> level]
    x2, y2 = bx + u, by + v               # always x2 >= 1024: stored at (x2 - 1024, y2 ^ 1024)
    word = ((y2 ^ 1024) // 2) * 512 + (x2 - 1024) // 2
    target = h.sheet ^ (level & 1)
    other = _mip_other()[word - LEVEL0_WORDS]
    source_sheet = np.where(other, target ^ 1, target)
    if (source_sheet != h.sheet).any():
        raise TextureError(f"header at {h.tha:#x}: mip level {level} has texels whose words are uploaded from the "
                           f"other sheet's block; not written")
    shift = np.where(v & 1, 0, 8) + np.where(u & 1, 0, 4)
    off = base + 2 * word + (shift >= 8)
    return off, (shift & 4) != 0


def mip_indices(rs: RomSet, h: TexHeader, level: int, course: int | None = None) -> np.ndarray:
    """Mip `level` of `h` as stored: raw 4-bit indices, shape (height >> level, width >> level), read from data ROM
    at `mip_texels`. TextureError as `mip_texels`."""
    off, high = mip_texels(rs, h, level, course)
    b = np.frombuffer(rs.data, np.uint8)[off]
    return np.where(high, b >> 4, b & 0xF).astype(np.uint8)


def encode_mip_bytes(h: TexHeader, level: int, indices, base: int) -> list[tuple[int, bytes]]:
    """The inverse of `mip_indices` for a sheet block at data-ROM offset `base`: (offset, bytes) runs, merged and
    sorted. Every byte must hold two texels of this level (its high and low nibble); TextureError for a byte that
    would hold only one (the other nibble belongs to a neighbour: a level whose position is odd, from level 5 of
    textures 128 or more on both sides; none on the real set), a wrong shape, non-integers or values outside 0-15,
    and as `mip_texels`."""
    off, high = mip_texels_at(h, level, base)
    a = np.asarray(indices)
    if a.shape != off.shape:
        raise TextureError(f"mip level {level} indices shape {a.shape} is not {off.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        raise TextureError(f"mip indices must be an integer array, not {a.dtype}")
    if a.size and (int(a.min()) < 0 or int(a.max()) > 15):
        raise TextureError(f"mip indices must be 0-15 (4 bits per texel), got {int(a.min())}..{int(a.max())}")
    order = np.argsort(off.ravel(), kind="stable")
    o, hb, v = off.ravel()[order], high.ravel()[order], a.astype(np.uint8).ravel()[order]
    if len(o) % 2 or (o[0::2] != o[1::2]).any() or (hb[0::2] == hb[1::2]).any():
        raise TextureError(f"header at {h.tha:#x}: mip level {level} shares bytes with neighbouring texels "
                           f"(a byte holds one nibble of this level); not encodable on its own")
    hi_v = np.where(hb[0::2], v[0::2], v[1::2])
    lo_v = np.where(hb[0::2], v[1::2], v[0::2])
    offs, vals = o[0::2].tolist(), (hi_v << 4 | lo_v).astype(np.uint8)
    runs: list[tuple[int, bytes]] = []
    start = 0
    for i in range(1, len(offs) + 1):
        if i == len(offs) or offs[i] != offs[i - 1] + 1:
            runs.append((offs[start], vals[start:i].tobytes()))
            start = i
    return runs


def mip_spans(rs: RomSet, h: TexHeader, level: int, course: int | None = None) -> list[tuple[int, int]]:
    """The data-ROM byte runs (offset, count) that hold mip `level` of `h` (see `mip_texels`), merged and sorted.
    The bytes are shared nibble-wise with neighbouring texels only where the level's position is odd."""
    off, _ = mip_texels(rs, h, level, course)
    spans: list[tuple[int, int]] = []
    for o in np.unique(off).tolist():
        if spans and spans[-1][0] + spans[-1][1] == o:
            spans[-1] = (spans[-1][0], spans[-1][1] + 1)
        else:
            spans.append((o, 1))
    return spans


def upload_sheets(rs: RomSet, course: int) -> tuple[np.ndarray, np.ndarray]:
    """Texture RAM 0 and 1 (u16 words, SHEET_WORDS each, in the stored 1024 x 2048 layout) as the game leaves them
    after the course load: the common block uploaded at boot (program 0x1454, into sheet 1), then the course block
    (0x1438, sheet 0). Each upload (routine 0x146c) copies 0x60000 words into its sheet, then fills the mip area
    (words 0x60000-0x7ffff), alternating sheets in runs from MIP_TABLE. A port of that routine; matches MAME's texture
    RAM byte for byte (docs/formats/model2-textures.md)."""
    sheets = (np.zeros(SHEET_WORDS, np.uint16), np.zeros(SHEET_WORDS, np.uint16))
    for sheet, src in ((1, sheet_source(rs, 1)), (0, sheet_source(rs, 0, course))):
        _upload(sheets, np.frombuffer(rs.data, "<u2", SHEET_BLOCK // 2, src), sheet)
    return sheets


def _upload(sheets, src: np.ndarray, sheet: int) -> None:
    """Routine 0x146c: level 0 straight into `sheet`; mip word w (source word w too: the source and destination
    pointers advance together) into the other sheet where `_mip_other()` is set, else into `sheet`."""
    sheets[sheet][:LEVEL0_WORDS] = src[:LEVEL0_WORDS]
    other, mip = _mip_other(), src[LEVEL0_WORDS:]
    sheets[sheet ^ 1][LEVEL0_WORDS:][other] = mip[other]
    sheets[sheet][LEVEL0_WORDS:][~other] = mip[~other]


_MIP_OTHER: np.ndarray | None = None


def _mip_other() -> np.ndarray:
    """bool[0x20000], per mip-area word (sheet word 0x60000 + i): True when routine 0x146c writes it to the sheet
    other than the one it uploads. The loop 0x14f0-0x1570: 256 rows of 512 words, 128 rows then 64, 32, 16, 8, 4, 2,
    1, 1 (one batch per r9 = 0..8). Batch 0 goes to the other sheet; in later batches a row starts on the other sheet
    and flips after each run, the runs being MIP_TABLE[1], then MIP_TABLE[k], doubled once k is past r9."""
    global _MIP_OTHER
    if _MIP_OTHER is None:
        out = np.empty(SHEET_WORDS - LEVEL0_WORDS, bool)
        cur, rows = 0, 128
        for level in range(9):            # r9 0..8; the row count halves to 1 and stays there
            for _ in range(rows):
                other, left, k = True, MIP_TABLE[1], 1
                for _ in range(512):
                    out[cur] = other
                    cur += 1
                    if level and left is not None:
                        left -= 1
                        if left == 0:    # end of a run: next run length (doubled past the level), other sheet
                            k += 1
                            # past the table the routine reads the course-sheet pointers at 0x15a0 (about
                            # 0x2200000): a run that never ends inside the 512-word row
                            left = MIP_TABLE[k] << (k > level) if k < len(MIP_TABLE) else None
                            other = not other
            if rows > 1:
                rows >>= 1
        if cur != len(out):
            raise AssertionError(f"mip pattern covers {cur} words")
        _MIP_OTHER = out
    return _MIP_OTHER


# --- colour ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ColourTables:
    luma: np.ndarray       # uint8[LUMA_SIZE]: luma RAM (texel -> luma 0-63)
    palette: np.ndarray    # uint16[PALETTE_ENTRIES]: palette RAM (x bbbbb ggggg rrrrr)
    xlat: np.ndarray       # uint16[XLAT_ENTRIES]: colour translation RAM, r / g / b tables of 32 rows x 256


def colour_xlat(flag: bool = True) -> np.ndarray:
    """The colour translation RAM as program 0x0a74-0x0b40 builds it: for each of the three tables, 32 rows (a 5-bit
    palette component c) of 64 shaded entries (luma l: (c * l * k + base) >> 6, 0 when c * l = 0, 0xffff from 256)
    then 192 copies of the tile-layer entry ((k * c + bias) >> 1). The constants depend on the half-word at work RAM
    0x005fe5e2: non-zero (k 10, base 0x1920, bias 0xc9) is what MAME's run had; zero gives k 12, base 0x1160,
    bias 0x8b."""
    k, rows_end, base, bias = (10, 640, 0x1920, 0xC9) if flag else (12, 768, 0x1160, 0x8B)
    out = []
    for c in range(32):
        for m in range(0, rows_end, k):
            v = c * m
            v = (v + base if v else 0) >> 6
            out.append(v if v < 256 else 0xFFFF)
        out.extend([(k * c + bias) >> 1 if c else 0] * 192)
    t = np.array(out, np.uint16)
    return np.concatenate([t, t, t])


def colour_tables(rs: RomSet, xlat_flag: bool = True) -> ColourTables:
    """The luma RAM, palette RAM (polygon entries only) and colour translation RAM rebuilt from ROM: the luma tables
    from data LUMA_TABLES_AT (count at LUMA_COUNT_AT; program 0x1400), the polygon palette from data
    PALETTE_START-PALETTE_END into entries 0x1000+ (program 0x5c7c), the translation table from `colour_xlat`. A few
    palette entries the game rewrites at run time are not modelled. TextureError for a luma count past the RAM."""
    (n,) = struct.unpack_from("<I", rs.data, LUMA_COUNT_AT)
    if not 0 <= n * 128 <= LUMA_SIZE:
        raise TextureError(f"luma table count at data {LUMA_COUNT_AT:#x} is {n}, past the {LUMA_SIZE:#x}-byte luma RAM")
    luma = np.zeros(LUMA_SIZE, np.uint8)
    luma[:n * 128] = np.frombuffer(rs.data, np.uint8, n * 128, LUMA_TABLES_AT)
    pal = np.zeros(PALETTE_ENTRIES, np.uint16)
    words = np.frombuffer(rs.data, "<u2", (PALETTE_END - PALETTE_START) // 2, PALETTE_START)
    pal[POLY_PALETTE:POLY_PALETTE + len(words)] = words
    return ColourTables(luma=luma, palette=pal, xlat=colour_xlat(xlat_flag))


def load_dump(directory: Path, frame: int) -> ColourTables:
    """The tables dumped by dtools/mame/lua/texdump.lua at `frame` (luma.<f>.bin, pal.<f>.bin, xlat.<f>.bin).
    TextureError naming a file of the wrong size."""
    def read(name: str, size: int) -> bytes:
        p = Path(directory) / f"{name}.{frame}.bin"
        blob = p.read_bytes()
        if len(blob) != size:
            raise TextureError(f"{p}: {len(blob)} bytes, expected {size}")
        return blob
    luma = np.frombuffer(read("luma", 4 * LUMA_SIZE), np.uint8)[0::4].copy()
    pal = np.frombuffer(read("pal", 2 * PALETTE_ENTRIES), "<u2").copy()
    xlat = np.frombuffer(read("xlat", 2 * XLAT_ENTRIES), "<u2").copy()
    return ColourTables(luma=luma, palette=pal, xlat=xlat)


def _gamma() -> np.ndarray:
    """MAME's display gamma (model2_v.cpp video_start): monitor emulation, not game data."""
    return np.array([int(max((i - 64.0) * 255.0 / 191.0, 0.0)) for i in range(256)], np.uint8)


def decode_texels(rs: RomSet, h: TexHeader, course: int | None = None, tables: ColourTables | None = None,
                  light: int = LIGHT_MAX) -> np.ndarray:
    """`h`'s level-0 texels as RGBA, shape (height, width, 4), uint8, coloured the way model2rd.ipp
    draw_scanline_tex colours an unfiltered texel: t = index << 4; luma = luma[luma_base * 128 + (t >> 1)] *
    light // 256, capped at 63; r, g, b = xlat[component table + (palette component) * 256 + luma] & 0xff through
    MAME's gamma. `light` stands for the polygon's lighting (object.luma, 0-255), which the geometry engine computes
    per polygon at run time. Alpha is 0 for index 15 in a translucent texture, else 255. `tables` defaults to
    `colour_tables(rs)`. TextureError as `decode_indices`, or for `light` outside 0-255."""
    light = _field(light, "light", 0, LIGHT_MAX)
    idx = decode_indices(rs, h, course)
    t = tables if tables is not None else colour_tables(rs)
    luma = (t.luma[h.luma_base * 128 + ((idx.astype(np.int32) << 4) >> 1)].astype(np.int32) * light // 256)
    luma = np.minimum(luma, 0x3F)
    c = int(t.palette[h.colour + POLY_PALETTE]) & 0x7FFF
    g = _gamma()
    rgba = np.empty(idx.shape + (4,), np.uint8)
    for ch, (table, comp) in enumerate(((0, c & 0x1F), (0x2000, (c >> 5) & 0x1F), (0x4000, (c >> 10) & 0x1F))):
        rgba[..., ch] = g[t.xlat[table + comp * 256 + luma] & 0xFF]
    rgba[..., 3] = np.where(h.translucent & (idx == 15), 0, 255)
    return rgba


# --- contact sheet --------------------------------------------------------------------------------------------

def course_headers(rs: RomSet, course: int) -> list[TexHeader]:
    """The distinct textured headers (by their four words) of the course's scenery objects, in placement and record
    order; every record's header is read, culled link-0 records included."""
    entry = course_mod.course_table(rs)[course]
    seen, out = set(), []
    for p in course_mod.scenery(rs, entry, allow_unmodelled=True):
        obj = polyrom.decode_object(rs.polygons, course_mod.check_rom_object(p.oba), p.obc, 1)
        for h in object_headers(rs, p.tha, obj):
            if h.textured and h.words not in seen:
                seen.add(h.words)
                out.append(h)
    return out


def contact_sheet(tiles: list[np.ndarray], width: int = 2048, gap: int = 4) -> np.ndarray:
    """RGBA tiles packed left to right in rows (tallest first) on a grey checkerboard, as one RGB uint8 image."""
    order = sorted(range(len(tiles)), key=lambda i: (-tiles[i].shape[0], -tiles[i].shape[1]))
    places, x, y, row_h = {}, gap, gap, 0
    for i in order:
        th, tw = tiles[i].shape[:2]
        if x + tw + gap > width and x > gap:
            x, y, row_h = gap, y + row_h + gap, 0
        places[i] = (x, y)
        x, row_h = x + tw + gap, max(row_h, th)
    height = y + row_h + gap
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.where(((yy // 8 + xx // 8) & 1)[..., None] == 1, 96, 160).astype(np.uint8).repeat(3, axis=2)
    for i, (px, py) in places.items():
        t = tiles[i]
        th, tw = t.shape[:2]
        a = t[..., 3:4] > 0
        img[py:py + th, px:px + tw] = np.where(a, t[..., :3], img[py:py + th, px:px + tw])
    return img


def main(argv: list[str] | None = None) -> None:
    """python -m dtools.arcade.texture <out.png> [course] [light]: the contact sheet of a course's scenery
    textures (needs DAYTONA_ROMDIR and Pillow; writes game-derived pixels: keep it under out/)."""
    import sys

    from PIL import Image

    from dtools import config

    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0])
    course = int(args[1]) if len(args) > 1 else 0
    light = int(args[2]) if len(args) > 2 else LIGHT_MAX
    rs = RomSet.from_zip(config.rom_dir() / "daytona.zip")
    tables = colour_tables(rs)
    heads = course_headers(rs, course)
    img = contact_sheet([decode_texels(rs, h, course, tables, light) for h in heads])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    print(f"{len(heads)} textures -> {out} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
