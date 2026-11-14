"""The Dreamcast course textures (`dtools.dc.tex`) as Model 2 4-bit texels with one hue and one luma table each, packed
into the course texel sheet: `convert_texture`, `pack`, `course_textures`, `course_texels`, `write_texels`; and the
course model's texture points on them: `polygon_points`, `course_points`.

The rules (docs/spike/2026-11-convert.md, "Textures (M3b)"; the colour model is docs/formats/model2-textures.md,
"Colour"):

- rows: `tex.to_rgba` gives the rows as stored, bottom row first; `convert_texture` flips them once, so its indices
  are top row first (sheet row tex_y is the picture's top). The UV side of the flip is `polygon_points`'
  (v' = height x (1 - v)).
- sizes: a side below 32 is enlarged to 32 by repeating texels (nearest), a side past 256 halved to 256 (each texel the
  mean of 2 x 2, alpha too); the arcade's sizes are `32 << n`. `course_textures` halves the largest further (largest
  area first, then the lowest index) only when the set does not fit the sheet.
- translucency: a texel whose alpha is below 50 % (alpha < OPAQUE) becomes index 15, which a translucent header
  draws transparent; a texture with any such texel is translucent and keeps 15 for them (its opaque texels use 0-14).
- colour (the ruling: one hue, luminance as intensity, the palette not extended): a texel's intensity is its Rec. 601
  luminance. For a palette entry and a luma table (0 or 1, the two course headers use), each texel takes the index
  whose decoded colour has the nearest luminance (the decoded colour as `texture.decode_texels` draws it at full
  light, 255). The entry and table are those whose decoded texture has the mean colour nearest (Euclidean RGB) the
  Dreamcast texture's mean colour, over the opaque texels: the entry nearest the texture's mean colour in the game's
  own colour model. Ties go to the lowest entry, then table 0. The four entries the game rewrites at run time
  (REWRITTEN) are never chosen. A texture's second colours are lost.
- the sheet: level 0 in sheet 0 (the slot's course block), outside the mip area (x >= 1024 and y >= 512), each
  texture aligned to its own width and height (so its mip levels start on even texels and never share a byte with a
  neighbour's); mip levels by decimation (level k takes every 2^k-th texel, as texture.with_texels "decimate").
- headers: textured, translucent as above, no wrap or mirror (`course_points` sets wrap from the texture points),
  no microtexture, and the bits MAME does not read as on every Beginner textured header: utex_min_lod 1, raw (0, 1, 0).

Every public function validates its inputs and raises ConvertError; a DcError from `tex.decode_tex` and a
TextureError from the arcade encoders pass through (both DecodeErrors)."""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping

import numpy as np

from dtools.arcade import golden as G
from dtools.arcade import texture as T
from dtools.arcade.romset import RomSet
from dtools.arcade.tablebase import is_int, is_number
from dtools.convert import ConvertError
from dtools.dc import pol, tex

MIN_SIDE, MAX_SIDE = 32, 256
"""Converted sides: 32 << n up to 256 (the Dreamcast's 8 and 16 enlarged, 512 and 1024 halved)."""
OPAQUE = 128
"""Alpha (0-255) at or above which a texel is opaque: alpha below 50 % is transparent (index 15)."""
TRANSPARENT = 15
"""The index a translucent header draws transparent (model2rd.ipp: texel 0xf)."""
LUMA_TABLES = (0, 1)
"""The luma tables course headers use: 0 rises (index i -> luma 4i), 1 falls (63 - 4i)."""
PALETTE_COLOURS = (T.PALETTE_END - T.PALETTE_START) // 2
"""Polygon palette entries the game loads (1,007: colour 0x000-0x3ee)."""
REWRITTEN = (0x0D3, 0x1AE, 0x1B5, 0x1B6)
"""Palette entries the game rewrites at run time (docs/formats/model2-textures.md): their colour is not the ROM's,
so they are never chosen."""
REC601 = np.array([0.299, 0.587, 0.114])
LIGHT = T.LIGHT_MAX
"""The polygon light the colour model assumes: full (decode_texels' default)."""

GRID = 32
GRID_W, GRID_H = T.SHEET_W // GRID, T.SHEET_H // GRID
MIP_X, MIP_Y = 1024, 512
"""The mip area: logical x >= MIP_X and y >= MIP_Y of the sheet holds mip levels, never level-0 texels."""
LEVEL0_TEXELS = T.SHEET_W * T.SHEET_H - (T.SHEET_W - MIP_X) * (T.SHEET_H - MIP_Y)
"""Level-0 texels a sheet holds: 1,572,864."""
MAX_PACKED_SIDE = MAX_SIDE
"""`pack` takes sides 32 << n up to 256, all that `convert_texture` produces."""

_HEADER = dict(tha=0, step=0, sheet=0, textured=True, checker=False, wrap_x=False, wrap_y=False, mirror_x=False,
               mirror_y=False, utex=False, utex_min_lod=1, utex_x=0, utex_y=0, raw=(0, 1, 0))


# --- the colour model --------------------------------------------------------------------------------------------

def _check_tables(tables) -> None:
    if not isinstance(tables, T.ColourTables):
        raise ConvertError(f"tables {type(tables).__name__} is not a texture.ColourTables")
    for name, size, top in (("luma", T.LUMA_SIZE, 0xFF), ("palette", T.PALETTE_ENTRIES, 0xFFFF),
                            ("xlat", T.XLAT_ENTRIES, 0xFFFF)):
        a = getattr(tables, name)
        if not isinstance(a, np.ndarray) or a.shape != (size,) or a.dtype.kind not in "ui":
            raise ConvertError(f"tables.{name} is not an integer array of {size} entries")
        lo, hi = int(a.min()), int(a.max())
        if lo < 0 or hi > top:
            raise ConvertError(f"tables.{name} values {lo}..{hi} are not 0-{top:#x} (texture.colour_tables' ranges)")


def _gamma() -> np.ndarray:
    """MAME's display gamma (texture._gamma): max((i - 64) * 255 / 191, 0), truncated."""
    i = np.arange(256, dtype=np.float64)
    return np.maximum((i - 64.0) * 255.0 / 191.0, 0.0).astype(np.int64)


def entry_colours(tables) -> np.ndarray:
    """(PALETTE_COLOURS, 2, 16, 3) float: the RGB of index i of a texture with colour entry p and luma table b
    (LUMA_TABLES), as texture.decode_texels draws an unfiltered texel at full light: luma = min(luma[b * 128 + (i << 4
    >> 1)] * 255 // 256, 63); each channel gamma[xlat[table + component * 256 + luma] & 0xff]. ConvertError for tables
    that are not ColourTables."""
    _check_tables(tables)            # luma 0-0xff, palette and xlat 0-0xffff: every index below stays in its table
    pal = tables.palette[T.POLY_PALETTE:T.POLY_PALETTE + PALETTE_COLOURS].astype(np.int64) & 0x7FFF
    comp = np.stack([pal & 31, (pal >> 5) & 31, (pal >> 10) & 31], -1)                   # (N, 3) r, g, b
    luma = np.array([[int(tables.luma[b * 128 + ((i << 4) >> 1)]) for i in range(16)] for b in LUMA_TABLES])
    luma = np.minimum(luma * LIGHT // 256, 0x3F)                                           # (2, 16)
    xlat, g = tables.xlat.astype(np.int64), _gamma()
    out = np.empty((PALETTE_COLOURS, len(LUMA_TABLES), 16, 3))
    for ch in range(3):
        out[..., ch] = g[xlat[0x2000 * ch + comp[:, ch, None, None] * 256 + luma[None]] & 0xFF]
    return out


FLAT_LUMA = 0x3F
"""The luma a flat (untextured) polygon draws with at full light: model2rd.ipp draw_scanline_solid, object.luma >> 2."""


def flat_colours(tables) -> np.ndarray:
    """(PALETTE_COLOURS, 3) float: the RGB a flat (untextured) polygon of palette entry p draws at full light, as
    model2rd.ipp draw_scanline_solid colours it: each channel gamma[xlat[table + component * 256 + FLAT_LUMA] & 0xff].
    ConvertError for tables that are not ColourTables."""
    _check_tables(tables)
    pal = tables.palette[T.POLY_PALETTE:T.POLY_PALETTE + PALETTE_COLOURS].astype(np.int64) & 0x7FFF
    xlat, g = tables.xlat.astype(np.int64), _gamma()
    return np.stack([g[xlat[0x2000 * ch + ((pal >> (5 * ch)) & 31) * 256 + FLAT_LUMA] & 0xFF] for ch in range(3)],
                    -1).astype(float)


def flat_entry(rgb, tables) -> int:
    """The palette entry an untextured polygon of colour `rgb` (three numbers 0-255) is drawn with: the entry whose
    flat colour (`flat_colours`) is nearest `rgb` (Euclidean), ties to the lowest; the entries the game rewrites at
    run time (REWRITTEN) never. ConvertError for an `rgb` that is not a tuple or list of three finite numbers 0-255
    (bool refused), or tables that are not ColourTables."""
    if (not isinstance(rgb, (tuple, list)) or len(rgb) != 3
            or not all(is_number(c) and 0 <= float(c) <= 255 for c in rgb)):
        raise ConvertError(f"colour {rgb!r:.60} is not three numbers 0-255 (r, g, b)")
    d = np.linalg.norm(flat_colours(tables) - np.array([float(c) for c in rgb]), axis=-1)
    d[list(REWRITTEN)] = np.inf
    return int(np.argmin(d))


def tint_entry(indices, header, tint, tables) -> int:
    """The palette entry a converted texture (`indices`, its `header`, `course_textures`' pair) takes where a Dreamcast
    mesh modulates it by a face colour `tint` (three numbers 0-1): the entry whose decoding of these indices with the
    header's luma table (`entry_colours`: full light, the opaque texels; index 15 left out of a translucent texture) has
    the mean colour nearest (Euclidean) the header's own decoded mean times `tint`, ties to the lowest, the REWRITTEN
    entries never: Task 3's rule (nearest in the game's own colour model) with the texels fixed. The luma table is the
    header's: the variants share the texels, whose indices were assigned for it (table 1 falls where 0 rises, so the
    other would invert the picture). An untinted texture (tint 1, 1, 1) keeps its header's entry. ConvertError for
    indices that are not an integer (height, width) array of 0-15, a header `_check_header` refuses or whose luma
    table is not 0 or 1, a tint that is not a tuple or list of three finite numbers 0-1 (bool refused), tables that
    are not ColourTables, or a texture with no opaque texel."""
    _check_header(header, "header")
    if not isinstance(indices, np.ndarray) or indices.dtype.kind not in "ui" or \
            indices.shape != (header.height, header.width):
        raise ConvertError(f"indices are not an integer array of shape {(header.height, header.width)!r}")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) > 15):
        raise ConvertError("indices are not 0-15")
    if header.luma_base not in LUMA_TABLES:
        raise ConvertError(f"header luma table {header.luma_base!r} is not one of {LUMA_TABLES}")
    if (not isinstance(tint, (tuple, list)) or len(tint) != 3
            or not all(is_number(c) and 0 <= float(c) <= 1 for c in tint)):
        raise ConvertError(f"tint {tint!r:.60} is not three numbers 0-1 (r, g, b)")
    colours = entry_colours(tables)[:, LUMA_TABLES.index(header.luma_base)]           # (N, 16, 3)
    hist = np.bincount(indices.ravel().astype(np.int64), minlength=16).astype(float)
    if header.translucent:
        hist[TRANSPARENT] = 0
    if not hist.sum():
        raise ConvertError("the texture has no opaque texel: nothing to tint")
    if all(float(c) == 1.0 for c in tint):
        return int(header.colour)
    means = hist @ colours / hist.sum()                                                # (N, 3)
    d = np.linalg.norm(means - means[header.colour] * np.array([float(c) for c in tint]), axis=-1)
    d[list(REWRITTEN)] = np.inf
    return int(np.argmin(d))


def _thresholds(levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For level luminances (..., n): the levels' order by luminance (stable) and the n - 1 midpoints between
    neighbours in that order. A luminance y takes level order[k], k the number of midpoints below y (strictly)."""
    order = np.argsort(levels, axis=-1, kind="stable")
    s = np.take_along_axis(levels, order, -1)
    return order, (s[..., 1:] + s[..., :-1]) / 2


def _choose(px: np.ndarray, colours: np.ndarray, n_levels: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """(entry, table, the levels' order, its midpoints) whose decoded texels' mean colour is nearest px's mean."""
    y = np.sort(px @ REC601)
    n = len(y)
    lv = colours[:, :, :n_levels]                                                  # (N, 2, L, 3)
    order, thr = _thresholds(lv @ REC601)
    cuts = np.searchsorted(y, thr.ravel(), side="right").reshape(thr.shape)        # texels at or below each midpoint
    counts = np.diff(np.concatenate([np.zeros(thr.shape[:-1] + (1,), np.int64), cuts,
                                     np.full(thr.shape[:-1] + (1,), n, np.int64)], -1), axis=-1)
    sorted_lv = np.take_along_axis(lv, order[..., None], 2)
    mean = (counts[..., None] * sorted_lv).sum(2) / n                              # (N, 2, 3)
    dist = np.linalg.norm(mean - px.mean(0), axis=-1)
    dist[list(REWRITTEN)] = np.inf
    k = int(np.argmin(dist))                                                       # lowest entry, then table 0
    p, b = divmod(k, len(LUMA_TABLES))
    return p, LUMA_TABLES[b], order[p, b], thr[p, b]


# --- convert_texture ---------------------------------------------------------------------------------------------

def _check_rgba(rgba) -> np.ndarray:
    if not isinstance(rgba, np.ndarray):
        raise ConvertError(f"rgba {type(rgba).__name__} is not a numpy array (tex.to_rgba's (h, w, 4) uint8)")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ConvertError(f"rgba shape {rgba.shape} is not (height, width, 4)")
    if rgba.dtype.kind not in "ui":
        raise ConvertError(f"rgba dtype {rgba.dtype} is not an integer type (bool and floats refused)")
    if rgba.size and (int(rgba.min()) < 0 or int(rgba.max()) > 255):
        raise ConvertError(f"rgba values {int(rgba.min())}..{int(rgba.max())} are not 0-255")
    for side, name in ((rgba.shape[0], "height"), (rgba.shape[1], "width")):
        if side < 1 or side > tex.MAX_SIDE or side & (side - 1):
            raise ConvertError(f"rgba {name} {side} is not a power of two 1-{tex.MAX_SIDE}")
    return rgba.astype(np.float64)


def _halve(a: np.ndarray, axis: int) -> np.ndarray:
    """Each pair of texels along `axis` as their mean (alpha too), rounded half up."""
    s = a.take(range(0, a.shape[axis], 2), axis) + a.take(range(1, a.shape[axis], 2), axis)
    return np.floor(s / 2 + 0.5)


def _resize(a: np.ndarray, halvings: int = 0) -> np.ndarray:
    """Sides past MAX_SIDE halved down to it, then both sides halved `halvings` more times (a side stops at
    MIN_SIDE), then sides below MIN_SIDE repeated up to it."""
    for axis in (0, 1):
        while a.shape[axis] > MAX_SIDE:
            a = _halve(a, axis)
        for _ in range(halvings):
            if a.shape[axis] > MIN_SIDE:
                a = _halve(a, axis)
        if a.shape[axis] < MIN_SIDE:
            a = a.repeat(MIN_SIDE // a.shape[axis], axis)
    return a


def convert_texture(rgba, tables) -> tuple[np.ndarray, int, int]:
    """A Dreamcast texture as arcade texels: (indices, colour, luma table). `rgba` is (h, w, 4) uint8 as
    `tex.to_rgba` gives it (rows as stored, bottom row first; sides powers of two); `tables` the game's colour
    tables (`texture.colour_tables`). The indices are (h', w') uint8 0-15, **top row first** (the rows flipped
    once here), sides 32 << n up to 256 (smaller enlarged, larger halved); colour the palette entry
    (header colour, 0-0x3ee) and luma table (0 or 1) whose decoded mean colour is nearest the texture's (the module
    docstring's rule); index 15 for alpha below 50 % (a translucent texture), else 0-14 in such a texture and 0-15
    otherwise. ConvertError for an rgba that is not such an array or tables that are not ColourTables."""
    idx, p, b, _ = _convert(rgba, tables)
    return idx, p, b


def _convert(rgba, tables, halvings: int = 0) -> tuple[np.ndarray, int, int, bool]:
    """convert_texture's (indices, colour, luma table) and whether the texture is translucent (a texel's alpha below
    50 % after resizing), with `halvings` extra halvings of both sides (`_resize`)."""
    a = _resize(_check_rgba(rgba)[::-1], halvings)
    colours = entry_colours(tables)
    opaque = a[..., 3] >= OPAQUE
    translucent = not opaque.all()
    px = a[..., :3][opaque] if opaque.any() else a[..., :3].reshape(-1, 3)
    p, b, order, thr = _choose(px, colours, TRANSPARENT if translucent else 16)
    idx = order[np.searchsorted(thr, a[..., :3] @ REC601, side="left")].astype(np.uint8)
    idx[~opaque] = TRANSPARENT
    return idx, p, b, translucent


# --- pack --------------------------------------------------------------------------------------------------------

def _side(value, what: str, top: int) -> int:
    if is_int(value):
        for n in range(8):
            if value == 32 << n <= top:
                return int(value)
    raise ConvertError(f"{what} {value!r} is not 32 << n up to {top}")


def pack(sizes) -> list[tuple[int, int]]:
    """Level-0 positions (tex_x, tex_y) in the course sheet for textures of `sizes` ((width, height), each 32 << n up
    to 256), in their order: inside the 2048 x 1024 sheet, outside the mip area (x >= 1024 and y >= 512), no two
    overlapping, each aligned to its own width and height (so 32-aligned, and its mip levels never share a byte with a
    neighbour's). Largest area first (then the longer side, then input order), each at the first free aligned place
    scanning rows top down. Square sets pack exactly: a set of squares is placed whenever its area fits the level-0
    area (power-of-two squares, largest first, each aligned to its size, never fragment). Sets with rectangles are
    best effort: a set that could be tiled may still be refused when nearly full. ConvertError for a size that is not
    one, or when they do not fit."""
    if not isinstance(sizes, (list, tuple)):
        raise ConvertError(f"sizes {type(sizes).__name__} is not a list of (width, height)")
    dims = []
    for i, s in enumerate(sizes):
        if not isinstance(s, (list, tuple)) or len(s) != 2:
            raise ConvertError(f"sizes[{i}] {s!r:.40} is not (width, height)")
        dims.append((_side(s[0], f"sizes[{i}] width", MAX_PACKED_SIDE) // GRID,
                     _side(s[1], f"sizes[{i}] height", MAX_PACKED_SIDE) // GRID))
    free = np.ones((GRID_H, GRID_W), bool)
    free[MIP_Y // GRID:, MIP_X // GRID:] = False
    out: list[tuple[int, int] | None] = [None] * len(dims)
    for i in sorted(range(len(dims)), key=lambda k: (-dims[k][0] * dims[k][1], -max(dims[k]), k)):
        w, h = dims[i]
        slots = free.reshape(GRID_H // h, h, GRID_W // w, w).all(axis=(1, 3))     # aligned places wholly free
        if slots.any():
            sy, sx = divmod(int(np.argmax(slots)), slots.shape[1])                  # the first, rows top down
            out[i] = (sx * w, sy * h)
        else:
            used = sum(a * b for a, b in dims) * GRID * GRID
            raise ConvertError(f"{len(dims)} textures ({used:,} texels) do not fit the course sheet's level-0 area "
                               f"({LEVEL0_TEXELS:,} texels): no place for texture {i} ({w * GRID} x {h * GRID})")
        x, y = out[i]
        free[y:y + h, x:x + w] = False
    return [(x * GRID, y * GRID) for x, y in out]


# --- a course's textures -----------------------------------------------------------------------------------------

def _header(w: int, h: int, x: int, y: int, colour: int, luma: int, translucent: bool) -> T.TexHeader:
    return T.TexHeader(width=w, height=h, tex_x=x, tex_y=y, translucent=translucent, luma_base=luma, colour=colour,
                       **_HEADER)


def course_textures(m, tex_blob, rs) -> tuple[dict[int, T.TexHeader], dict[int, np.ndarray]]:
    """Every texture of the course model `m` (its `textures` table; `tex_blob` the matching `_TEX.BIN`) converted
    (`convert_texture`, the colour tables of `rs`) and packed into sheet 0 (`pack`): (Dreamcast texture index ->
    header, Dreamcast texture index -> indices, top row first). When the set does not fit, the largest texture (largest
    area, then the lowest index) is halved (2 x 2 mean) and the set packed again, until it fits; a reduced texture's
    header is smaller than its Dreamcast size (a size that is not max(32, the Dreamcast's) was reduced). The headers'
    tha and step are 0 (the caller places them). ConvertError for an `m` that is not a pol.Model, a blob that is not
    bytes, an `rs` that is not a RomSet, or a set that does not fit at 32 x 32 each; DcError from tex.decode_tex."""
    if not isinstance(m, pol.Model):
        raise ConvertError(f"m {type(m).__name__} is not a pol.Model")
    if not isinstance(tex_blob, (bytes, bytearray, memoryview)):
        raise ConvertError(f"tex_blob {type(tex_blob).__name__} is not bytes")
    if not isinstance(rs, RomSet):
        raise ConvertError(f"rs {type(rs).__name__} is not a RomSet")
    if not isinstance(m.textures, tuple):
        raise ConvertError(f"m.textures {type(m.textures).__name__} is not a tuple of TexEntry")
    tables = T.colour_tables(rs)
    stored = [tex.to_rgba(t) for t in tex.decode_tex(bytes(tex_blob), m.textures, "textures")]
    conv = [_convert(a, tables) for a in stored]
    halvings = [0] * len(stored)
    while True:
        sizes = [(c[0].shape[1], c[0].shape[0]) for c in conv]
        try:
            places = pack(sizes)
            break
        except ConvertError:
            big = [i for i, (w, h) in enumerate(sizes) if max(w, h) > MIN_SIDE]
            if not big:
                raise
            i = min(big, key=lambda k: (-sizes[k][0] * sizes[k][1], k))
            halvings[i] += 1
            conv[i] = _convert(stored[i], tables, halvings[i])
    headers, indices = {}, {}
    for i, ((idx, colour, luma, translucent), (x, y)) in enumerate(zip(conv, places)):
        headers[i] = _header(idx.shape[1], idx.shape[0], x, y, colour, luma, translucent)
        indices[i] = idx
    return headers, indices


def _check_header(h, what: str) -> None:
    """`h` a textured TexHeader whose fields are scalars of their kinds (flags bool or 0/1 integers, the rest integers,
    raw three integers) and that texture.mip_levels accepts (encode_tex_header's field checks). ConvertError
    otherwise."""
    if not isinstance(h, T.TexHeader) or h.textured is not True:
        raise ConvertError(f"{what} {h!r:.40} is not a textured texture.TexHeader")
    for f in dataclasses.fields(h):
        v = getattr(h, f.name)
        if f.name == "raw":
            if not isinstance(v, tuple) or len(v) != 3 or not all(is_int(x) for x in v):
                raise ConvertError(f"{what}: raw {v!r:.40} is not three integers")
        elif not (is_int(v) or isinstance(v, bool)):
            raise ConvertError(f"{what}: {f.name} {v!r:.40} is not an integer or bool")
    try:
        T.mip_levels(h)
    except T.TextureError as e:
        raise ConvertError(f"{what}: {e}") from None


def course_texels(headers, indices) -> tuple[G.Texels, ...]:
    """golden.Texels per texture of `headers` / `indices` (the same keys, `course_textures`' output), in `headers`'
    order: level 0 the indices, each mip level of texture.mip_levels by decimation (level k every 2^k-th texel in x
    and y). ConvertError for mappings whose keys differ, a value that is not a textured TexHeader, or indices that are
    not an integer (height, width) array of 0-15. The header is checked first (`_check_header`)."""
    if not isinstance(headers, Mapping) or not isinstance(indices, Mapping):
        raise ConvertError("headers and indices must be mappings (Dreamcast texture index -> header / indices)")
    if set(headers) != set(indices):
        raise ConvertError(f"headers' keys {sorted(map(repr, headers))[:4]} are not indices' "
                           f"{sorted(map(repr, indices))[:4]}")
    out = []
    for k, h in headers.items():
        _check_header(h, f"headers[{k!r}]")
        a = indices[k]
        if not isinstance(a, np.ndarray) or a.dtype.kind not in "ui" or a.shape != (h.height, h.width):
            raise ConvertError(f"indices[{k!r}] is not an integer array of shape {(h.height, h.width)!r}")
        if a.size and (int(a.min()) < 0 or int(a.max()) > 15):
            raise ConvertError(f"indices[{k!r}] values {int(a.min())}..{int(a.max())} are not 0-15")
        a = a.astype(np.uint8)
        out.append(G.Texels(h, a, tuple(a[::1 << lv, ::1 << lv].copy() for lv in T.mip_levels(h))))
    return tuple(out)


def write_texels(rs, texels, course) -> RomSet:
    """A copy of `rs` with every golden.Texels of `texels` written through golden.texel_writes (the writer
    golden.tables uses) into the course sheet's data-ROM block (texture.sheet_source(rs, 0, course)). ConvertError for
    an `rs` that is not a RomSet, texels that are not a tuple or list of golden.Texels, a header `_check_header`
    refuses or in sheet 1 (the common block is never written), a course that is not 0-2, or two textures writing different bytes to one place (overlapping texels or mip levels);
    TextureError from the encoders (a texture outside the sheet or in the mip area, a wrong shape)."""
    if not isinstance(rs, RomSet):
        raise ConvertError(f"rs {type(rs).__name__} is not a RomSet")
    if not isinstance(texels, (tuple, list)) or not all(isinstance(t, G.Texels) for t in texels):
        raise ConvertError(f"texels {texels!r:.40} is not a tuple or list of golden.Texels")
    if not is_int(course) or course not in (0, 1, 2):
        raise ConvertError(f"course {course!r} is not a course id 0-2")
    buf = bytearray(rs.data)
    written = np.zeros(len(buf), bool)
    lo, hi = len(buf), 0
    for n, tx in enumerate(texels):
        _check_header(tx.header, f"texels[{n}].header")
        if not isinstance(tx.mips, tuple):
            raise ConvertError(f"texels[{n}] mips {type(tx.mips).__name__} is not a tuple of levels")
        if tx.header.sheet != 0:
            raise ConvertError(f"texels[{n}] is in sheet {tx.header.sheet}: the converter writes the course sheet (0) "
                               f"only, never the common block")
        base = T.sheet_source(rs, 0, course)
        for off, blob, _ in G.texel_writes(tx, base):
            end = off + len(blob)
            if written[off:end].any():
                old = np.frombuffer(bytes(buf[off:end]), np.uint8)
                clash = written[off:end] & (old != np.frombuffer(blob, np.uint8))
                if clash.any():
                    at = off + int(np.argmax(clash))
                    raise ConvertError(f"texels[{n}] ({tx.header.width} x {tx.header.height} at "
                                       f"({tx.header.tex_x}, {tx.header.tex_y})) overlap an earlier texture's at data "
                                       f"{at:#x} with different bytes")
            buf[off:end] = blob
            written[off:end] = True
            lo, hi = min(lo, off), max(hi, end)
    return rs if hi <= lo else rs.with_patch("data", lo, bytes(buf[lo:hi]))


# --- texture points ----------------------------------------------------------------------------------------------

POINT_UNIT = 8
"""Texture point words per texel: a point is unsigned, in 1/8 texel (docs/formats/model2-textures.md)."""


def _sizes(width, height) -> tuple[int, int]:
    for name, s in (("width", width), ("height", height)):
        if not is_int(s) or int(s) not in tuple(32 << n for n in range(8)):
            raise ConvertError(f"texture {name} {s!r:.30} is not 32 << n for n in 0..7 (32-4096)")
    return int(width), int(height)


def polygon_points(uvs, width, height, mirror=(False, False)) -> tuple[tuple[tuple[int, int], ...],
                                                                        tuple[bool, bool]]:
    """One textured polygon's texture points from its Dreamcast (u, v) per corner (3 or 4 pairs, `model.Piece.uvs`'
    order) on a converted texture of `width` x `height` texels: ((v', u') per corner, (wrap x needed, wrap y needed)).

    - scale and flip: u' = width * u and v' = height * (1 - v) texels (Task 3 stores the rows top first: a Dreamcast
      stored row r is arcade row height - 1 - r, docs/spike/2026-11-convert.md, "Rows"), in 1/8 texel rounded to
      nearest (half up);
    - shift: each axis moved by whole texture repeats (8 * size words; two repeats on a `mirror`ed axis, so the mirror
      phase is kept) until its least point lies in 0 .. 8 * size - 1. The rasterizer repeats every texture
      (model2rd.ipp fetch_bilinear_texel: u & (width - 1)), so the picture is unchanged;
    - wrap: an axis (not mirrored) needs its header's wrap bit where a point then lies past 8 * size (a point on the
      far edge, 8 * size, is inside): the polygon leaves its texture, and without the bit the bilinear filter clamps
      at the texture's edge instead of blending with the next repeat.

    ConvertError for uvs that are not 3 or 4 pairs of finite numbers (bool refused), a size that is not 32 << n, a
    `mirror` that is not two bools, or a polygon whose shifted points pass the u16 range (65,535 / 8 = 8,191.875
    texels)."""
    w, h = _sizes(width, height)
    if (not isinstance(mirror, tuple) or len(mirror) != 2 or not all(isinstance(m, bool) for m in mirror)):
        raise ConvertError(f"mirror {mirror!r:.30} is not two bools (x, y)")
    if not isinstance(uvs, (tuple, list)) or len(uvs) not in (3, 4):
        raise ConvertError(f"uvs {uvs!r:.60} are not 3 or 4 (u, v) pairs")
    for p in uvs:
        if not isinstance(p, (tuple, list)) or len(p) != 2 or not all(is_number(x) for x in p):
            raise ConvertError(f"uvs {uvs!r:.60}: {p!r:.40} is not a (u, v) pair of finite numbers")
    axes, wraps = [], []
    for k, (size, flip, mirrored) in enumerate(((w, False, mirror[0]), (h, True, mirror[1]))):
        repeat = POINT_UNIT * size
        scaled = [POINT_UNIT * size * ((1.0 - float(p[k])) if flip else float(p[k])) for p in uvs]
        if not all(math.isfinite(x) for x in scaled):
            raise ConvertError(f"uvs {uvs!r:.80} on a {w} x {h} texture: {'uv'[k]} is past the float range once "
                               f"scaled")
        words = [math.floor(x + 0.5) for x in scaled]
        period = 2 * repeat if mirrored else repeat
        shift = (min(words) // period) * period
        words = [x - shift for x in words]
        if max(words) > T.POINT_MAX:
            raise ConvertError(f"uvs {uvs!r:.80} on a {w} x {h} texture: {'uv'[k]} spans {words[0]}..{max(words)} "
                               f"(1/8 texel) once shifted, past the u16 range (0-{T.POINT_MAX:#x})")
        axes.append(words)
        wraps.append(not mirrored and max(words) > repeat)
    return tuple(zip(axes[1], axes[0])), (wraps[0], wraps[1])


def course_points(pieces, headers) -> tuple[dict[int, tuple[tuple[tuple[tuple[int, int], ...], ...], ...]],
                                            dict[int, T.TexHeader]]:
    """The texture points of the course model's pieces (`model.convert_model`'s cell -> Pieces) on the converted
    textures (`course_textures`' Dreamcast texture index -> header): (cell -> per piece, per record of its object, the
    record's 3 or 4 (v, u) points: `texture.encode_texture_points(points, piece.obj)` writes them at the piece's
    tpa), and the headers with wrap_x / wrap_y set for each texture one of whose polygons needs it (`polygon_points`;
    a texture used both wrapped and not gets the bit: wrapped is safe for both, only the bilinear blend at the
    edge differs). Every other header field is kept; this function never sets a mirror bit itself: the
    Dreamcast's flip bits are carried by `model_scenery` as mirror_x / mirror_y header variants (Tasks 4-5, see
    docs/spike/2026-11-convert.md, "Texture points (M3b)"), and a header's own mirror bit is honoured here.

    A textured record's points are `polygon_points` on its texture's header size. A move record (uvs None) reads 4
    points and an untextured record (texture -1) 3 or 4: all 0, as Beginner's (4,712 of its 4,713 move records and
    4,521 of its 4,566 untextured drawn records hold zeros).

    ConvertError for pieces that are not a dict of cell (0-255) -> tuple of model.Piece, a piece whose textures or
    uvs are not one per record (a texture an integer, -1 or a key of `headers`; uvs None exactly for a move record,
    else as many pairs as the record has vertices), headers that are not a dict of textured texture.TexHeader, and
    as `polygon_points`, naming the cell, piece and record."""
    from dtools.convert import model
    if not isinstance(pieces, dict):
        raise ConvertError(f"pieces {pieces!r:.40} are not a dict of cell -> tuple of model.Piece (convert_model's)")
    if not isinstance(headers, dict):
        raise ConvertError(f"headers {headers!r:.40} are not a dict of Dreamcast texture index -> TexHeader")
    for k, h in headers.items():
        _check_header(h, f"headers[{k!r}]")
    need = {k: [False, False] for k in headers}
    out = {}
    for cell, ps in pieces.items():
        if not is_int(cell) or not 0 <= cell <= 255:
            raise ConvertError(f"pieces: cell {cell!r:.30} is not a scenery cell 0-255")
        if not isinstance(ps, tuple) or isinstance(ps, model.Piece):
            raise ConvertError(f"pieces[{cell:#x}] {ps!r:.40} are not a tuple of model.Piece")
        cell_points = []
        for i, pc in enumerate(ps):
            what = f"cell {cell:#x} piece {i}"
            if not isinstance(pc, model.Piece):
                raise ConvertError(f"{what}: {pc!r:.40} is not a model.Piece")
            try:
                counts = [4 if r.is_quad else 3 for r in pc.obj.records]
            except Exception:
                raise ConvertError(f"{what}: its obj {pc.obj!r:.40} is not a polyrom.Object of records") from None
            if (not isinstance(pc.textures, tuple) or not isinstance(pc.uvs, tuple)
                    or not len(pc.textures) == len(pc.uvs) == len(counts)):
                raise ConvertError(f"{what}: textures and uvs are not tuples of one entry per record "
                                   f"({len(counts)} records)")
            recs = []
            for j, (n, t, uv) in enumerate(zip(counts, pc.textures, pc.uvs)):
                rwhat = f"{what} record {j}"
                if not is_int(t) or not (t == -1 or t in headers):
                    raise ConvertError(f"{rwhat}: texture {t!r:.30} is neither -1 nor a texture with a header")
                if uv is None or t == -1:
                    if uv is None and t != -1:
                        raise ConvertError(f"{rwhat}: texture {t} on a move record (uvs None)")
                    if uv is not None and (not isinstance(uv, tuple) or len(uv) != n):
                        raise ConvertError(f"{rwhat}: uvs {uv!r:.40} are not {n} pairs (the record's vertices)")
                    recs.append(((0, 0),) * n)
                    continue
                if not isinstance(uv, tuple) or len(uv) != n:
                    raise ConvertError(f"{rwhat}: uvs {uv!r:.40} are not {n} pairs (the record's vertices)")
                h = headers[t]
                try:
                    q, wraps = polygon_points(uv, h.width, h.height, (bool(h.mirror_x), bool(h.mirror_y)))
                except ConvertError as e:
                    raise ConvertError(f"{rwhat} (texture {t}): {e}") from None
                need[t] = [a or b for a, b in zip(need[t], wraps)]
                recs.append(q)
            cell_points.append(tuple(recs))
        out[cell] = tuple(cell_points)
    wrapped = {k: dataclasses.replace(h, wrap_x=bool(h.wrap_x) or need[k][0], wrap_y=bool(h.wrap_y) or need[k][1])
               for k, h in headers.items()}
    return out, wrapped
