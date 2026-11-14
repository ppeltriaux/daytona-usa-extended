"""System 24 tile-layer data of the arcade Daytona USA (Sega Model 2): tile maps, chars, char and palette lists,
upload sets (and the backdrop segments an upload set can carry). Layouts: docs/formats/model2-course-tables.md,
"Tile layers and uploads" and "Backdrop".

The screens are drawn on the System 24 tilemap chip (MAME segaic24.cpp): tile RAM 0x01000000-0x0100ffff holds four
64 x 64 layers of u16 tiles (layer k at word 0x1000 * k), char RAM 0x01080000-0x010fffff 0x4000 chars of 8 x 8
pixels at 4 bpp (32 bytes each). A tile is char `v & 0x3fff`, colour `(v >> 7) & 0xff` (16 pens at palette entry
colour * 16; the two fields overlap, so a char's palette follows from its index), priority bit 15. Tiles with bit 15
clear are drawn behind the polygons (layers 3 and 2 opaque, then 1 and 0), tiles with it set in front
(model2_v.cpp screen_update; segaic24.cpp draw_common).

Uploads: an upload set is two (or ten) words: a char list (src, dst) pairs of data-ROM char blocks (u32 count, then
count chars) and char RAM addresses, ended by a zero src (copied by 0x1dd70); a palette list (dst, u32 count, count
u16 entries) ended by a zero dst (0x1dde0); for a course, eight backdrop segments. Later uploads overwrite earlier
ones; char RAM is never cleared.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

import numpy as np

from dtools.arcade.romset import RomSet
from dtools.arcade.tablebase import (DATA_BASE, PROGRAM_WINDOW, CourseError, _S16, _U16, _data_off, _data_ptr_ok,
                                     _int, _read, _seq, _u32, data_size, is_int)

TILE_RAM = (0x01000000, 0x01010000)
TILE_LAYER = 64                 # tiles per row and rows per layer
CHAR_RAM = (0x01080000, 0x01100000)
CHAR_SIZE = 32
CHAR_COUNT = 0x4000
CHAR_MASK = 0x3FFF
PALETTE_RAM = (0x01800000, 0x01804000)
BACKDROP_SEGMENTS = 8
BACKDROP_SEGMENT_COLS = 32      # 0x1dfd0: column = (x >> 8) & 31, segment = (x >> 13) & 7
BACKDROP_FIRST_ROW = 6          # the band starts at layer-2 row 6 (0x1e068: 3 << 8 bytes)


class TileMap(NamedTuple):
    """A tile-map blob as the blit 0x19304 reads it: s16 base, s16 wide, u32 rows, u32 cols, then rows * cols
    entries (s16 when wide, else s8), each stored as (entry + base) & 0xffff, row-major, 64 tiles per tile-RAM row."""
    base: int
    wide: bool
    rows: int
    cols: int
    entries: tuple[int, ...]

    def tiles(self) -> tuple[int, ...]:
        return tuple((e + self.base) & 0xFFFF for e in self.entries)

    def tile(self, row: int, col: int) -> int:
        return (self.entries[row * self.cols + col] + self.base) & 0xFFFF


def tile_map(rs: RomSet, addr: int) -> TileMap:
    """The tile map at i960 address `addr` (data ROM). CourseError outside data ROM or for a size past the 64 x 64
    layer."""
    _data_off(addr, 12, "tile map", rs)
    base, wide, rows, cols = struct.unpack("<hhII", _read(rs, addr, 12, "tile map"))
    if not 1 <= rows <= TILE_LAYER or not 1 <= cols <= TILE_LAYER:
        raise CourseError(f"tile map at {addr:#x}: {rows} rows x {cols} cols is outside 1..{TILE_LAYER}")
    size = 2 if wide else 1
    _data_off(addr + 12, size * rows * cols, f"tile map at {addr:#x}", rs)
    body = _read(rs, addr + 12, size * rows * cols, "tile map")
    return TileMap(base, bool(wide), rows, cols, struct.unpack(f"<{rows * cols}{'h' if wide else 'b'}", body))


def encode_tile_map(m: TileMap) -> bytes:
    """The blob of `m`. CourseError for a base outside s16, rows or cols outside 1..64, an entry count that is not
    rows * cols, an entry outside s16 (wide) or s8."""
    what = "tile map"
    base = _int(m.base, "base", *_S16, what)
    rows = _int(m.rows, "rows", 1, TILE_LAYER, what)
    cols = _int(m.cols, "cols", 1, TILE_LAYER, what)
    lo, hi = _S16 if m.wide else (-0x80, 0x7F)
    entries = _seq(m.entries, None, "entries", what)
    if len(entries) != rows * cols:
        raise CourseError(f"{what}: {len(entries)} entries for {rows * cols} entries ({rows} x {cols})")
    entries = [_int(e, "entry", lo, hi, what) for e in entries]
    return struct.pack("<hhII", base, 1 if m.wide else 0, rows, cols) + \
        struct.pack(f"<{len(entries)}{'h' if m.wide else 'b'}", *entries)


def decode_chars(data: bytes) -> np.ndarray:
    """Char RAM bytes as (n, 8, 8) pixel indices 0-15. Each row is two u16 words; pixel 0 is the high nibble of the
    first (segaic24.cpp char_layout, read with xor 8 on a little-endian host). CourseError for a length not a
    multiple of 32."""
    if len(data) % CHAR_SIZE:
        raise CourseError(f"char data of {len(data)} bytes is not a multiple of {CHAR_SIZE}")
    w = np.frombuffer(bytes(data), "<u2").reshape(-1, 8, 2).astype(np.uint16)
    shifts = np.array([12, 8, 4, 0], np.uint16)
    px = (w[..., :, None] >> shifts) & 0xF                    # (n, 8, 2, 4)
    return px.reshape(-1, 8, 8).astype(np.uint8)


def encode_chars(pixels) -> bytes:
    """The char RAM bytes of (n, 8, 8) pixel indices. CourseError for another shape or values outside 0-15."""
    a = np.asarray(pixels)
    if a.ndim != 3 or a.shape[1:] != (8, 8):
        raise CourseError(f"chars of shape {a.shape}: expected (n, 8, 8)")
    if a.size and (not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 15):
        raise CourseError("char pixels must be integers 0-15")
    q = a.astype(np.uint16).reshape(-1, 8, 2, 4)
    w = (q[..., 0] << 12) | (q[..., 1] << 8) | (q[..., 2] << 4) | q[..., 3]
    return w.astype("<u2").tobytes()


class CharUpload(NamedTuple):
    src: int                    # i960 address of the block (u32 count, then the chars)
    dst: int                    # char RAM address
    count: int                  # chars

    @property
    def first_char(self) -> int:
        return (self.dst - CHAR_RAM[0]) // CHAR_SIZE

    def data(self, rs: RomSet) -> bytes:
        return _read(rs, self.src + 4, CHAR_SIZE * self.count, "char block")


class PaletteUpload(NamedTuple):
    dst: int                    # palette RAM address
    words: tuple[int, ...]      # u16 entries


def _check_char_dst(dst: int, count: int, what: str) -> None:
    if not is_int(dst) or (dst - CHAR_RAM[0]) % CHAR_SIZE or not CHAR_RAM[0] <= dst or \
            dst + CHAR_SIZE * count > CHAR_RAM[1]:
        raise CourseError(f"{what}: destination {dst!r} ({count} chars) is not char-aligned inside char RAM "
                          f"{CHAR_RAM[0]:#x}-{CHAR_RAM[1] - 1:#x}")


def _check_palette_dst(dst: int, n: int, what: str) -> None:
    if not is_int(dst) or dst % 2 or not PALETTE_RAM[0] <= dst or dst + 2 * n > PALETTE_RAM[1]:
        raise CourseError(f"{what}: destination {dst!r} ({n} entries) is outside palette RAM "
                          f"{PALETTE_RAM[0]:#x}-{PALETTE_RAM[1] - 1:#x}")


_MAX_LIST = 256                 # entries; the real lists have up to 30
def _rom_ranges(rs: RomSet | None) -> tuple[tuple[int, int], ...]:
    """The i960 ROM ranges: program, the program window, data ROM (`rs`'s image: data_size)."""
    return (0x00000000, 0x00040000), PROGRAM_WINDOW, (DATA_BASE, DATA_BASE + data_size(rs))


def _rom_ptr(ptr, size: int, what: str, rs: RomSet | None = None) -> int:
    """`ptr` checked to address `size` bytes of ROM: program, the program window or data ROM (a char block may
    sit in any of them: BOOT_CHAR_LIST's blocks are in the program window)."""
    if not is_int(ptr) or not any(lo <= ptr and ptr + size <= hi for lo, hi in _rom_ranges(rs)):
        raise CourseError(f"{what} {ptr!r} ({size} bytes) is outside the ROM regions (program, window, data)")
    return int(ptr)


def char_list(rs: RomSet, addr: int) -> tuple[CharUpload, ...]:
    """The (src, dst) pairs of the char list at `addr` (0x1dd70). CourseError for a source block outside ROM or a
    destination outside char RAM, naming the word's address."""
    out = []
    at = addr
    while True:
        src = _u32(rs, at)
        if src == 0:
            return tuple(out)
        if len(out) == _MAX_LIST:
            raise CourseError(f"char list at {addr:#x}: no end after {_MAX_LIST} entries")
        dst = _u32(rs, at + 4)
        _rom_ptr(src, 4, f"char list at {addr:#x}: block (word at {at:#x})", rs)
        n = _u32(rs, src)
        _rom_ptr(src, 4 + CHAR_SIZE * n, f"char list at {addr:#x}: block of {n} chars (word at {at:#x})", rs)
        _check_char_dst(dst, n, f"char list: word at {at + 4:#x}")
        out.append(CharUpload(src, dst, n))
        at += 8


def palette_list(rs: RomSet, addr: int) -> tuple[PaletteUpload, ...]:
    """The (dst, entries) runs of the palette list at `addr` (0x1dde0: copies count / 2 words). CourseError for a run
    outside palette RAM or an odd count, naming its address."""
    out = []
    at = addr
    while True:
        dst = _u32(rs, at)
        if dst == 0:
            return tuple(out)
        if len(out) == _MAX_LIST:
            raise CourseError(f"palette list at {addr:#x}: no end after {_MAX_LIST} entries")
        n = _u32(rs, at + 4)
        _check_palette_dst(dst, n, f"palette list: run at {at:#x}")
        if n % 2:
            raise CourseError(f"palette list: run at {at:#x} has an odd count {n} (copied as {n // 2} words)")
        out.append(PaletteUpload(dst, struct.unpack(f"<{n}H", _read(rs, at + 8, 2 * n, "palette list"))))
        at += 8 + 2 * n


class UploadSet(NamedTuple):
    addr: int                   # i960 address of the set
    char_list: int
    palette_list: int
    chars: tuple[CharUpload, ...]
    palette: tuple[PaletteUpload, ...]
    segments: tuple[int, ...]   # backdrop segment addresses (a course's set), else ()


def upload_set(rs: RomSet, addr: int, segments: bool) -> UploadSet:
    """The upload set at `addr`; with `segments`, its eight backdrop segment pointers too (checked to decode)."""
    cl, pl = _u32(rs, addr), _u32(rs, addr + 4)
    segs = ()
    if segments:
        segs = tuple(_u32(rs, addr + 8 + 4 * k) for k in range(BACKDROP_SEGMENTS))
        for s in segs:
            backdrop_segment(rs, s)
    return UploadSet(addr, cl, pl, char_list(rs, cl), palette_list(rs, pl), segs)


def encode_char_block(data: bytes) -> bytes:
    """A char block: u32 count, then the chars. CourseError for no chars, a length not a multiple of 32, or more than
    CHAR_COUNT chars (char RAM holds no more)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CourseError(f"char block data {type(data).__name__} is not bytes")
    if not data or len(data) % CHAR_SIZE:
        raise CourseError(f"char block of {len(data)} bytes: empty or not a multiple of {CHAR_SIZE}")
    if len(data) > CHAR_SIZE * CHAR_COUNT:
        raise CourseError(f"char block of {len(data) // CHAR_SIZE} chars: char RAM holds {CHAR_COUNT}")
    return struct.pack("<I", len(data) // CHAR_SIZE) + bytes(data)


def encode_char_list(pairs) -> bytes:
    """A char list of (src block address, char RAM destination) pairs plus the zero end. CourseError for a source
    outside ROM or a destination that is not char-aligned inside char RAM (the block's own count is checked by
    char_list when read back)."""
    out = b""
    for pair in _seq(pairs, None, "pairs", "char list"):
        src, dst = _seq(pair, 2, "pair (src, dst)", "char list")
        _rom_ptr(src, 4, "char list: source")
        _check_char_dst(dst, 1, "char list")
        out += struct.pack("<II", src, dst)
    return out + b"\0\0\0\0"


def encode_palette_list(runs) -> bytes:
    """A palette list plus the zero end. CourseError for a run outside palette RAM, an odd count (the copy moves
    words) or an entry outside u16."""
    out = b""
    for r in _seq(runs, None, "runs", "palette list"):
        words = [_int(w, "entry", *_U16, "palette list (u16)") for w in _seq(r.words, None, "words", "palette list")]
        n = len(words)
        _check_palette_dst(r.dst, n, "palette list")
        if n % 2:
            raise CourseError(f"palette list: run at {r.dst:#x} has {n} entries; the copy needs an even count")
        out += struct.pack(f"<II{n}H", r.dst, n, *words)
    return out + b"\0\0\0\0"


def encode_upload_set(char_list_addr: int, palette_list_addr: int, segments=()) -> bytes:
    """Two words (a screen's set) or ten (a course's, with BACKDROP_SEGMENTS segment addresses). CourseError for an
    address outside data ROM or another segment count."""
    segments = _seq(segments, None, "segments", "upload set")
    if segments and len(segments) != BACKDROP_SEGMENTS:
        raise CourseError(f"upload set: {len(segments)} segments, expected 0 or {BACKDROP_SEGMENTS}")
    for a in (char_list_addr, palette_list_addr) + segments:
        _data_ptr_ok(a, "upload set: pointer", 4)
    return struct.pack(f"<{2 + len(segments)}I", char_list_addr, palette_list_addr, *segments)


class BackdropSegment(NamedTuple):
    flags: int                  # word 0; 0x10000 on every real segment, no reader found
    rows: int
    cols: int
    tiles: tuple[int, ...]      # u16 tile-RAM values, row-major


_BACKDROP_MAX_ROWS = TILE_LAYER - BACKDROP_FIRST_ROW


def backdrop_segment(rs: RomSet, addr: int) -> BackdropSegment:
    _data_off(addr, 12, "backdrop segment", rs)
    flags, rows, cols = struct.unpack("<III", _read(rs, addr, 12, "backdrop segment"))
    if cols != BACKDROP_SEGMENT_COLS or not 1 <= rows <= _BACKDROP_MAX_ROWS:
        raise CourseError(f"backdrop segment at {addr:#x}: {rows} rows x {cols} cols; the streamer needs "
                          f"{BACKDROP_SEGMENT_COLS} columns and 1..{_BACKDROP_MAX_ROWS} rows")
    _data_off(addr + 12, 2 * rows * cols, f"backdrop segment at {addr:#x}", rs)
    return BackdropSegment(flags, rows, cols, struct.unpack(f"<{rows * cols}H",
                                                            _read(rs, addr + 12, 2 * rows * cols, "segment")))


def encode_backdrop_segment(s: BackdropSegment) -> bytes:
    """CourseError for flags outside u32, columns other than 32, rows outside 1..58 (the band from layer row 6), a
    tile count that is not rows * 32 or a tile outside u16."""
    what = "backdrop segment"
    flags = _int(s.flags, "flags", 0, 0xFFFFFFFF, what)
    if s.cols != BACKDROP_SEGMENT_COLS or not is_int(s.cols):
        raise CourseError(f"{what}: {s.cols!r} columns; the streamer (0x1dfd0) needs {BACKDROP_SEGMENT_COLS} columns")
    rows = _int(s.rows, "rows", 1, _BACKDROP_MAX_ROWS, what)
    tiles = [_int(t, "tile", *_U16, f"{what} (u16)")
             for t in _seq(s.tiles, rows * BACKDROP_SEGMENT_COLS, f"tiles ({rows} x {BACKDROP_SEGMENT_COLS})", what)]
    return struct.pack(f"<III{len(tiles)}H", flags, rows, BACKDROP_SEGMENT_COLS, *tiles)
