"""The Dreamcast course textures, `CSn{v}_TEX.BIN` and `VCSn{v}_TEX.BIN`, and the map sheets `GAME_MAP*.BIN`: a
lossless decoder and encoder.

A `_TEX.BIN` has no header: it is the texture data that its `_POL.BIN`'s texture table points at (CS with CS, VCS with
VCS; docs/formats/dreamcast-disc.md, "Textures"). Each table entry gives width, height, a format word and an offset.
The format word is the PVR file's: low byte the pixel format (`PIXEL_FORMATS`: the PowerVR texture control word's
codes 0-2), next byte the layout (`LAYOUTS`: 1 twiddled, 2 twiddled with mipmaps, 3 VQ, 4 VQ with mipmaps, 0x0d a
twiddled rectangle, the car sets'); the
other bytes are zero. Words outside these are refused, not guessed.

Stored forms (all 16-bit texels, little-endian, PowerVR twiddled order: bit 2i of the index is y bit i, bit 2i+1 is
x bit i; a rectangle is squares of its short side one after the other):
- twiddled: w * h texels.
- twiddled_mip: 2 bytes (zero in the files), then the levels from 1 x 1 up to w x w, each twiddled.
- vq: a codebook of 256 entries of 4 texels (a 2 x 2 block in twiddled order, as (row, column): (0,0), (1,0), (0,1),
  (1,1)), then one index byte per block, the (w/2) x (h/2) grid twiddled.
- vq_mip: the codebook, then the index levels from 1 x 1 (one byte, the entry's first texel) up to w x w.
Textures are placed at their offsets; the bytes between them (padding to 32 bytes in the files) must be zero.

`to_rgba`, `texels` and `vq_texels` return the picture as stored: row 0 is the first stored row (the visual check
found the pictures stored bottom row first; docs/spike/2026-11-dc-decode.md, "Textures"). Every malformed input is a
`DcError` naming the file or texture and the offset or word."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dtools.dc.errors import DcError
from dtools.dc.pol import TexEntry
from dtools.arcade.tablebase import _int, _seq, is_int

PIXEL_FORMATS = {0: "ARGB1555", 1: "RGB565", 2: "ARGB4444"}
LAYOUTS = {1: "twiddled", 2: "twiddled_mip", 3: "vq", 4: "vq_mip", 0x0D: "twiddled_rect"}
"""0x0d (the car sets, v0.4): a twiddled rectangle, stored as "twiddled" is; its own name so the word re-encodes."""
MIPPED = ("twiddled_mip", "vq_mip")
VQ = ("vq", "vq_mip")
CODEBOOK_SIZE = 256 * 8         # 256 entries of four 16-bit texels
MIP_PREAMBLE = 2                # bytes before a twiddled mip chain's 1 x 1 level
MIN_SIDE, MAX_SIDE = 8, 1024    # the PowerVR's texture sides

_PIXEL_CODE = {v: k for k, v in PIXEL_FORMATS.items()}
_LAYOUT_CODE = {v: k for k, v in LAYOUTS.items()}


@dataclass(frozen=True)
class VqData:
    """A VQ texture's stored form: the codebook (2048 bytes) and the index bytes (every level, smallest first)."""
    codebook: bytes
    indices: bytes


@dataclass(frozen=True)
class Texture:
    """One texture as stored: its table entry, layout and pixel format (named from the entry's word), the data
    (bytes; VqData for VQ) and the number of mip levels (1 without mipmaps)."""
    entry: TexEntry
    layout: str
    pixel: str
    data: bytes | VqData
    mips: int


# ---- twiddling ----

def _side_ok(n) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1 and n & (n - 1) == 0


def _as_int(v):
    """`v` as a Python int when it is a Python or numpy integer (not bool); else `v` unchanged."""
    return int(v) if is_int(v) else v


def _sides(w, h, who: str) -> tuple[int, int]:
    w2, h2 = _as_int(w), _as_int(h)
    if not (_side_ok(w2) and _side_ok(h2)):
        raise DcError(f"{who}: size {w!r} x {h!r} is not a power-of-two size")
    return w2, h2


def _numbers(values, who: str, ndim: int) -> np.ndarray:
    """`values` as an `ndim`-D numeric array (integers or floats, not bool), else DcError naming what was given."""
    try:
        a = np.asarray(values)
    except (ValueError, TypeError) as exc:
        raise DcError(f"{who}: {type(values).__name__} is not an array of numbers ({exc})") from None
    if a.ndim != ndim or a.dtype.kind not in "uif":
        raise DcError(f"{who}: {type(values).__name__} of shape {a.shape} and dtype {a.dtype} is not a {ndim}-D "
                      f"array of numbers")
    return a


def twiddle_index(w: int, h: int) -> np.ndarray:
    """The stored index of each texel (y, x) of a twiddled w x h texture, shape (h, w). w and h powers of two."""
    w, h = _sides(w, h, "twiddle")
    y, x = np.mgrid[0:h, 0:w]
    m = min(w, h)
    bits = m.bit_length() - 1
    idx = np.zeros((h, w), dtype=np.int64)
    for i in range(bits):
        idx |= ((y >> i) & 1) << (2 * i) | ((x >> i) & 1) << (2 * i + 1)
    big = (y if h > w else x) >> bits
    return idx + big * (m * m)


def untwiddle(stream, w: int, h: int) -> np.ndarray:
    """The (h, w) picture of a twiddled stream of w * h values."""
    w, h = _sides(w, h, "untwiddle")
    s = _numbers(stream, "untwiddle", 1)
    if len(s) != w * h:
        raise DcError(f"untwiddle: {len(s)} values for a {w} x {h} texture")
    return s[twiddle_index(w, h)]


def twiddle(img) -> np.ndarray:
    """The twiddled stream of an (h, w) picture."""
    a = _numbers(img, "twiddle", 2)
    w, h = _sides(a.shape[1], a.shape[0], "twiddle")
    out = np.empty(w * h, dtype=a.dtype)
    out[twiddle_index(w, h)] = a
    return out


# ---- formats ----

def parse_fmt(word, who: str = "texture") -> tuple[str, str]:
    """(layout, pixel) of a format word; DcError naming `who` and the word for one the files do not use."""
    word = _as_int(word)
    if not isinstance(word, int) or isinstance(word, bool):
        raise DcError(f"{who}: format word {word!r} is not an integer")
    pixel, layout, rest = word & 0xFF, (word >> 8) & 0xFF, word >> 16
    if rest or pixel not in PIXEL_FORMATS or layout not in LAYOUTS:
        raise DcError(f"{who}: format word {word:#x} is not a known texture format (low byte pixel format "
                      f"{sorted(PIXEL_FORMATS)}, next byte layout {sorted(LAYOUTS)}, no other bits)")
    return LAYOUTS[layout], PIXEL_FORMATS[pixel]


def _levels(layout: str, w: int) -> int:
    return w.bit_length() if layout in MIPPED else 1


def _check_size(layout: str, w, h, who: str) -> tuple[int, int]:
    w, h = _as_int(w), _as_int(h)
    if not (_side_ok(w) and _side_ok(h)) or not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
        raise DcError(f"{who}: size {w!r} x {h!r} is not a texture size (powers of two, {MIN_SIDE}-{MAX_SIDE})")
    if layout in MIPPED and w != h:
        raise DcError(f"{who}: size {w} x {h}: a {layout} texture is square")
    return w, h


def _mip_sides(w: int) -> list[int]:
    """1, 2, 4, ..., w: the levels in stored order."""
    return [1 << k for k in range(w.bit_length())]


def stored_size(layout: str, w: int, h: int) -> int:
    """Bytes of a texture's stored form (without padding). DcError for a layout that is not one of LAYOUTS' names,
    or a size that is not a texture size of that layout (powers of two, MIN_SIDE-MAX_SIDE; square when mipmapped)."""
    if not isinstance(layout, str) or layout not in _LAYOUT_CODE:
        raise DcError(f"stored_size: layout {layout!r} is not one of {tuple(LAYOUTS.values())}")
    w, h = _check_size(layout, w, h, f"stored_size ({layout})")
    return _stored_size(layout, w, h)


def _stored_size(layout: str, w: int, h: int) -> int:
    if layout in ("twiddled", "twiddled_rect"):
        return 2 * w * h
    if layout == "twiddled_mip":
        return MIP_PREAMBLE + sum(2 * s * s for s in _mip_sides(w))
    if layout == "vq":
        return CODEBOOK_SIZE + (w // 2) * (h // 2)
    return CODEBOOK_SIZE + sum(max(1, (s // 2) ** 2) for s in _mip_sides(w))


# ---- decode ----

def decode_tex(blob: bytes, entries: Sequence[TexEntry], name: str = "") -> tuple[Texture, ...]:
    """The textures of `blob` named by `entries` (the matching `_POL.BIN`'s texture table), in table order.
    DcError (naming `name`, the texture and the offset or word) for an unknown format word, a size the layout cannot
    hold, a texture that runs past the end of the file or overlaps another, or a non-zero byte outside the
    textures."""
    if not isinstance(name, str):
        raise DcError(f"textures: the name {type(name).__name__} is not a str")
    who = name or "textures"
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise DcError(f"{who}: {type(blob).__name__} is not bytes")
    b = bytes(blob)
    size = len(b)
    entries = _seq(entries, None, "entries", who, DcError)
    out = []
    spans = []
    for i, e in enumerate(entries):
        what = f"{who}: texture {i}"
        if not isinstance(e, TexEntry):
            raise DcError(f"{what}: {type(e).__name__} is not a TexEntry")
        layout, pixel = parse_fmt(e.fmt, what)
        what = f"{what} (format word {int(e.fmt):#x})"
        w, h = _check_size(layout, e.width, e.height, what)
        n = _stored_size(layout, w, h)
        off = _as_int(e.offset)
        if isinstance(off, bool) or not isinstance(off, int):
            raise DcError(f"{what}: offset {off!r} is not an integer")
        if off < 0 or off + n > size:
            raise DcError(f"{what} ({w} x {h}, {layout} {pixel}) at offset {off:#x} needs {n:#x} bytes, "
                          f"the file ends at {size:#x}")
        raw = b[off:off + n]
        data = VqData(raw[:CODEBOOK_SIZE], raw[CODEBOOK_SIZE:]) if layout in VQ else raw
        out.append(Texture(e, layout, pixel, data, _levels(layout, w)))
        spans.append((off, off + n, i))
    _check_spans(spans, who)
    covered = np.zeros(size, dtype=bool)
    for s, e, _ in spans:
        covered[s:e] = True
    stray = np.flatnonzero(~covered & (np.frombuffer(b, np.uint8) != 0))
    if len(stray):
        raise DcError(f"{who}: byte {b[stray[0]]:#04x} at offset {int(stray[0]):#x} lies outside every texture and "
                      f"is not zero (the gaps are not kept)")
    return tuple(out)


def _check_spans(spans: list[tuple[int, int, int]], who: str) -> None:
    """Textures may share their exact bytes (same offset and length), never overlap otherwise."""
    last = None
    for s, e, i in sorted(spans):
        if last is not None and s < last[1] and (s, e) != last[:2]:
            raise DcError(f"{who}: texture {i} overlaps texture {last[2]} ({last[0]:#x}-{last[1]:#x}) at offset "
                          f"{s:#x}")
        if last is None or e > last[1]:
            last = (s, e, i)


# ---- encode ----

def _checked(t, what: str) -> tuple[str, str, int, int]:
    """(layout, pixel, width, height) of `t` after checking its entry's word and size, its layout, pixel format and
    mip count against them."""
    if not isinstance(t, Texture):
        raise DcError(f"{what}: {type(t).__name__} is not a Texture")
    e = t.entry
    if not isinstance(e, TexEntry):
        raise DcError(f"{what}: entry {type(e).__name__} is not a TexEntry")
    layout, pixel = parse_fmt(e.fmt, what)
    what = f"{what} (format word {int(e.fmt):#x})"
    if (t.layout, t.pixel) != (layout, pixel):
        raise DcError(f"{what}: layout {t.layout!r} and pixel {t.pixel!r} disagree with the word ({layout} {pixel})")
    w, h = _check_size(layout, e.width, e.height, what)
    want = _levels(layout, w)
    mips = _int(t.mips, "mips", 1, MAX_SIDE.bit_length(), what, DcError)
    if mips != want:
        raise DcError(f"{what}: mip chain of {mips} levels, a {w} x {h} {layout} texture has {want}")
    return layout, pixel, w, h


def _check_texture(t, i: int) -> tuple[bytes, int]:
    """The stored bytes of `t` and its offset, after checking it against its entry."""
    what = f"texture {i}"
    layout, pixel, w, h = _checked(t, what)
    e = t.entry
    off = _int(e.offset, "offset", 0, 0xFFFFFFFF, what, DcError)
    if layout in VQ:
        if not isinstance(t.data, VqData):
            raise DcError(f"{what}: {layout} data is {type(t.data).__name__}, not VqData")
        cb, ix = t.data.codebook, t.data.indices
        if not isinstance(cb, (bytes, bytearray)) or not isinstance(ix, (bytes, bytearray)):
            raise DcError(f"{what}: the codebook and indices are not bytes")
        if len(cb) != CODEBOOK_SIZE:
            raise DcError(f"{what}: the codebook is {len(cb)} bytes, a VQ codebook is {CODEBOOK_SIZE}")
        raw = bytes(cb) + bytes(ix)
    else:
        if not isinstance(t.data, (bytes, bytearray)):
            raise DcError(f"{what}: {layout} data is {type(t.data).__name__}, not bytes")
        raw = bytes(t.data)
    n = _stored_size(layout, w, h)
    if len(raw) != n:
        raise DcError(f"{what}: {len(raw)} bytes of data, a {w} x {h} {layout} texture is {n} bytes")
    return raw, off


def encode_tex(textures: Sequence[Texture], size: int) -> bytes:
    """The `size`-byte file holding `textures` at their entries' offsets, zero elsewhere. DcError for a texture whose
    data, layout, pixel format or mip count disagrees with its entry, that runs past `size` or overlaps another."""
    size = _int(size, "size", 0, 0xFFFFFFFF, "textures", DcError)
    textures = _seq(textures, None, "textures", "textures", DcError)
    out = bytearray(size)
    spans = []
    placed: dict[tuple[int, int], tuple[bytes, int]] = {}
    for i, t in enumerate(textures):
        raw, off = _check_texture(t, i)
        if off + len(raw) > size:
            raise DcError(f"texture {i} at offset {off:#x} ({len(raw):#x} bytes) runs past the file's {size:#x} bytes")
        key = (off, off + len(raw))
        if key in placed and placed[key][0] != raw:
            raise DcError(f"texture {i} at offset {off:#x} shares its bytes with texture {placed[key][1]} but holds "
                          f"different data")
        placed.setdefault(key, (raw, i))
        spans.append(key + (i,))
        out[off:off + len(raw)] = raw
    _check_spans(spans, "textures")
    return bytes(out)


# ---- pixels ----

def texels_to_rgba(texels, pixel: str) -> np.ndarray:
    """RGBA (..., 4) uint8 of 16-bit texels; 5- and 6-bit channels widened by repeating their top bits. DcError for
    values that are not 16-bit texels (not integers, negative, past 0xFFFF) or an unknown pixel format."""
    try:
        a = np.asarray(texels)
    except (ValueError, TypeError) as exc:
        raise DcError(f"texels: {type(texels).__name__} is not an array of 16-bit texels ({exc})") from None
    if a.dtype.kind not in "ui" or (a.size and (int(a.min()) < 0 or int(a.max()) > 0xFFFF)):
        raise DcError(f"texels: {type(texels).__name__} of dtype {a.dtype} does not hold 16-bit texels "
                      f"(integers 0-0xFFFF)")
    p = a.astype(np.uint32)
    if pixel == "ARGB1555":
        a = ((p >> 15) & 1) * 255
        r, g, b = (p >> 10) & 31, (p >> 5) & 31, p & 31
        r, g, b = (r << 3 | r >> 2), (g << 3 | g >> 2), (b << 3 | b >> 2)
    elif pixel == "RGB565":
        r, g, b = (p >> 11) & 31, (p >> 5) & 63, p & 31
        r, g, b = (r << 3 | r >> 2), (g << 2 | g >> 4), (b << 3 | b >> 2)
        a = np.full_like(p, 255)
    elif pixel == "ARGB4444":
        a, r, g, b = ((p >> 12) & 15) * 17, ((p >> 8) & 15) * 17, ((p >> 4) & 15) * 17, (p & 15) * 17
    else:
        raise DcError(f"pixel format {pixel!r} is not one of {list(PIXEL_FORMATS.values())}")
    return np.stack([r, g, b, a], -1).astype(np.uint8)


def _what(t) -> str:
    if not isinstance(t, Texture) or not isinstance(t.entry, TexEntry):
        return "texture"
    e = t.entry
    return f"texture {e.width!r} x {e.height!r} at offset {e.offset!r}"


def _level(t: Texture, level) -> tuple[int, int, int, int, int]:
    """(level, level width, level height, full width, full height) after checking `t` (as the encoder does) and
    `level`."""
    what = _what(t)
    _, _, w, h = _checked(t, what)
    mips = int(t.mips)
    level = _as_int(level)
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < mips:
        raise DcError(f"{what}: level {level!r} is outside its {mips} mip level(s)")
    return level, w >> level, h >> level, w, h


def _vq_parts(t: Texture, what: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(t.data, VqData):
        raise DcError(f"{what}: {t.layout} data is {type(t.data).__name__}, not VqData")
    cb, ix = t.data.codebook, t.data.indices
    if not isinstance(cb, (bytes, bytearray)) or not isinstance(ix, (bytes, bytearray)) or len(cb) % 8:
        raise DcError(f"{what}: the codebook ({type(cb).__name__}) is not whole 8-byte entries")
    n = _stored_size(t.layout, w, h) - CODEBOOK_SIZE
    if len(ix) != n:
        raise DcError(f"{what}: {len(ix)} index bytes, a {w} x {h} {t.layout} texture has {n}")
    codebook = np.frombuffer(bytes(cb), "<u2").reshape(-1, 4)
    indices = np.frombuffer(bytes(ix), np.uint8)
    if len(indices) and int(indices.max()) >= len(codebook):
        k = int(np.argmax(indices >= len(codebook)))
        raise DcError(f"{what}: index byte {k} ({int(indices[k])}) is past its codebook of {len(codebook)} entries")
    return codebook, indices


def vq_texels(t: Texture, level: int = 0) -> np.ndarray:
    """The (h, w) 16-bit texels of a VQ texture's level, as stored (row 0 first). A 1 x 1 level is its entry's
    first texel."""
    level, lw, lh, w, h = _level(t, level)
    what = _what(t)
    if t.layout not in VQ:
        raise DcError(f"{what}: layout {t.layout} is not VQ")
    codebook, indices = _vq_parts(t, what, w, h)
    start = sum(max(1, (s // 2) ** 2) for s in _mip_sides(w) if s < lw) if t.layout == "vq_mip" else 0
    if lw == 1:
        return codebook[indices[start], :1].reshape(1, 1).copy()
    grid = untwiddle(indices[start:start + (lw // 2) * (lh // 2)], lw // 2, lh // 2)
    blocks = codebook[grid]                         # (lh/2, lw/2, 4)
    out = np.empty((lh, lw), dtype=np.uint16)
    for j, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        out[dy::2, dx::2] = blocks[..., j]
    return out


def texels(t: Texture, level: int = 0) -> np.ndarray:
    """The (h, w) 16-bit texels of a texture's level, as stored (row 0 first)."""
    level, lw, lh, w, h = _level(t, level)
    what = _what(t)
    if t.layout in VQ:
        return vq_texels(t, level)
    if not isinstance(t.data, (bytes, bytearray)):
        raise DcError(f"{what}: {t.layout} data is {type(t.data).__name__}, not bytes")
    n = _stored_size(t.layout, w, h)
    if len(t.data) != n:
        raise DcError(f"{what}: {len(t.data)} bytes of data, a {w} x {h} {t.layout} texture is {n}")
    mip = t.layout == "twiddled_mip"
    stream = np.frombuffer(bytes(t.data), "<u2", offset=MIP_PREAMBLE if mip else 0)
    start = sum(s * s for s in _mip_sides(w) if s < lw) if mip else 0
    return untwiddle(stream[start:start + lw * lh], lw, lh)


def to_rgba(t: Texture, level: int = 0) -> np.ndarray:
    """The (h, w, 4) uint8 RGBA of a texture's mip level (0 the full size), rows as stored: the textures are stored
    bottom row first, so row 0 is the picture's bottom row (`rgba[::-1]` shows it upright). The flip is left to the
    converter (M3, which converts Desert City's textures first), with the UV convention."""
    return texels_to_rgba(texels(t, level), t.pixel)


# ---- the disc ----

def tex_names() -> list[tuple[str, str]]:
    """The 64 (model, texture) file pairs: 8 courses x 4 variants x (CS, VCS), in `pol.pol_names()` order."""
    from dtools.dc.pol import pol_names
    return [(p, p[:-len("_POL.BIN")] + "_TEX.BIN") for p in pol_names()]


def verify_lossless_tex(disc) -> list[str]:
    """One line per texture file that `encode_tex(decode_tex(file))` does not reproduce byte for byte, or that does
    not decode with its model's texture table ([] when all 64 are lossless)."""
    from dtools.dc.pol import decode_pol
    out = []
    for pname, tname in tex_names():
        blob = disc.read(tname)
        try:
            entries = decode_pol(disc.read(pname), pname).textures
            back = encode_tex(decode_tex(blob, entries, tname), len(blob))
        except DcError as exc:
            out.append(f"{tname}: {exc}")
            continue
        if back != blob:
            k = next((k for k, (a, c) in enumerate(zip(blob, back)) if a != c), min(len(blob), len(back)))
            out.append(f"{tname}: first difference at offset {k:#x}")
    return out


# ---- GAME_MAP ----

SHEET_W, SHEET_H, MAP_SIDE = 32, 64, 64
_SHEET_BYTES = 2 * SHEET_W * SHEET_H
_MAP_BYTES = 2 * MAP_SIDE * MAP_SIDE


def game_map(blob: bytes, name: str = "GAME_MAP.BIN") -> tuple[np.ndarray, ...]:
    """`GAME_MAP.BIN` / `GAME_MAP2.BIN`: the 32 x 64 sheet (shape (64, 32)) then the 64 x 64 maps, each untwiddled,
    16-bit ARGB4444 texels (uint16) as stored. DcError naming `name` for a size that is not the sheet and whole maps
    (`GAME_MAP3.BIN` is one 256 x 256 sheet: `game_map3`)."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise DcError(f"{name}: {type(blob).__name__} is not bytes")
    b = bytes(blob)
    if len(b) < _SHEET_BYTES or (len(b) - _SHEET_BYTES) % _MAP_BYTES:
        raise DcError(f"{name}: {len(b)} bytes is not a {SHEET_W} x {SHEET_H} sheet ({_SHEET_BYTES}) and whole "
                      f"{MAP_SIDE} x {MAP_SIDE} maps ({_MAP_BYTES} each); the data ends at offset {len(b):#x}")
    t = np.frombuffer(b, "<u2")
    sheet = SHEET_W * SHEET_H
    out = [untwiddle(t[:sheet], SHEET_W, SHEET_H).astype(np.uint16)]
    for k in range((len(t) - sheet) // (MAP_SIDE * MAP_SIDE)):
        s = sheet + k * MAP_SIDE * MAP_SIDE
        out.append(untwiddle(t[s:s + MAP_SIDE * MAP_SIDE], MAP_SIDE, MAP_SIDE).astype(np.uint16))
    return tuple(out)


def encode_game_map(images: Sequence[np.ndarray]) -> bytes:
    """The file bytes of a sheet and maps as `game_map` returns them. DcError for a wrong shape or a value that is not
    a 16-bit texel."""
    images = _seq(images, None, "images", "GAME_MAP", DcError)
    if not images:
        raise DcError("GAME_MAP: no sheet")
    parts = []
    for k, img in enumerate(images):
        what = "the sheet" if k == 0 else f"map {k - 1}"
        want = (SHEET_H, SHEET_W) if k == 0 else (MAP_SIDE, MAP_SIDE)
        a = np.asarray(img) if isinstance(img, np.ndarray) else None
        if a is None or a.shape != want:
            raise DcError(f"GAME_MAP: {what} is {getattr(a, 'shape', type(img).__name__)}, expected {want}")
        if a.dtype.kind not in "ui" or (a.size and (int(a.min()) < 0 or int(a.max()) > 0xFFFF)):
            raise DcError(f"GAME_MAP: {what} ({a.dtype}) does not hold 16-bit texels")
        parts.append(twiddle(a.astype("<u2")).tobytes())
    return b"".join(parts)


def game_map_rgba(images: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """RGBA of `game_map`'s (or `game_map3`'s) images (ARGB4444). DcError for anything that is not 16-bit texels."""
    images = _seq(images, None, "images", "GAME_MAP", DcError)
    return tuple(texels_to_rgba(img, "ARGB4444") for img in images)


SHEET3_SIDE = 256
_SHEET3_BYTES = 2 * SHEET3_SIDE * SHEET3_SIDE


def game_map3(blob: bytes, name: str = "GAME_MAP3.BIN") -> np.ndarray:
    """`GAME_MAP3.BIN`: one 256 x 256 twiddled ARGB4444 sheet, as (256, 256) uint16 texels, untwiddled, rows as
    stored (bottom row first). The 256 x 256 reading rests on the picture looking right *(by hand)*. DcError naming
    `name` and the offset for any other length."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise DcError(f"{name}: {type(blob).__name__} is not bytes")
    b = bytes(blob)
    if len(b) != _SHEET3_BYTES:
        raise DcError(f"{name}: {len(b)} bytes is not a {SHEET3_SIDE} x {SHEET3_SIDE} sheet ({_SHEET3_BYTES} bytes); "
                      f"the data ends at offset {len(b):#x}")
    return untwiddle(np.frombuffer(b, "<u2"), SHEET3_SIDE, SHEET3_SIDE).astype(np.uint16)


def encode_game_map3(sheet: np.ndarray) -> bytes:
    """The file bytes of a 256 x 256 sheet as `game_map3` returns it. DcError for a wrong shape or a value that is
    not a 16-bit texel."""
    a = sheet if isinstance(sheet, np.ndarray) else None
    if a is None or a.shape != (SHEET3_SIDE, SHEET3_SIDE):
        raise DcError(f"GAME_MAP3: the sheet is {getattr(a, 'shape', type(sheet).__name__)}, expected "
                      f"{(SHEET3_SIDE, SHEET3_SIDE)}")
    if a.dtype.kind not in "ui" or (a.size and (int(a.min()) < 0 or int(a.max()) > 0xFFFF)):
        raise DcError(f"GAME_MAP3: the sheet ({a.dtype}) does not hold 16-bit texels")
    return twiddle(a.astype("<u2")).tobytes()
