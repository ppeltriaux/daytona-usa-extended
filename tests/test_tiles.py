"""Tile-layer primitives (dtools/arcade/tiles.py): tile maps, chars, char and palette lists, upload sets, backdrop
segments. Synthetic data only; the real-ROM checks are in tests/test_course.py (select_assets, backdrop)."""

import struct

import numpy as np
import pytest

from dtools.arcade.romset import RomSet
from dtools.arcade.tablebase import CourseError
from dtools.arcade.tiles import (BACKDROP_SEGMENTS, BackdropSegment, CharUpload, PaletteUpload, TileMap, char_list,
                                 decode_chars, encode_backdrop_segment, encode_char_block, encode_char_list,
                                 encode_chars, encode_palette_list, encode_tile_map, encode_upload_set, palette_list,
                                 tile_map, upload_set)

DATA = 0x02000000


def _rs(program: bytes | bytearray, data: bytes | bytearray) -> RomSet:
    return RomSet(program=bytes(program), data=bytes(data), copro_data=b"", polygons=b"", textures=b"", extra={})


def _tile_map_bytes(base: int, wide: bool, rows: int, cols: int, entries) -> bytes:
    fmt = "h" if wide else "b"
    return struct.pack("<hhII", base, 1 if wide else 0, rows, cols) + struct.pack(f"<{rows * cols}{fmt}", *entries)


def test_tile_map_round_trip_synthetic():
    data = bytearray(16 << 20)
    wide = _tile_map_bytes(-0x8000, True, 2, 3, [0x3480, 0x3481, -1, 0, 5, 0x7FFF])
    narrow = _tile_map_bytes(0x3000, False, 1, 4, [0, 1, -128, 127])
    data[0x100:0x100 + len(wide)] = wide
    data[0x200:0x200 + len(narrow)] = narrow
    rs = _rs(bytearray(256 * 1024), data)
    m = tile_map(rs, DATA + 0x100)
    assert (m.base, m.wide, m.rows, m.cols) == (-0x8000, True, 2, 3)
    assert m.tiles() == (0xB480, 0xB481, 0x7FFF, 0x8000, 0x8005, 0xFFFF)   # (entry + base) & 0xffff, row-major
    assert m.tile(1, 2) == 0xFFFF
    assert encode_tile_map(m) == wide
    n = tile_map(rs, DATA + 0x200)
    assert n.tiles() == (0x3000, 0x3001, 0x2F80, 0x307F)
    assert encode_tile_map(n) == narrow


@pytest.mark.parametrize("m, match", [
    (TileMap(0x8000, True, 1, 1, (0,)), "base"),
    (TileMap(0, False, 1, 1, (128,)), "entry"),
    (TileMap(0, True, 1, 1, (-0x8001,)), "entry"),
    (TileMap(0, True, 1, 2, (0,)), "2 entries"),
    (TileMap(0, True, 0, 0, ()), "rows"),
    (TileMap(0, True, 1, 65, (0,) * 65), "cols"),
])
def test_encode_tile_map_refuses_what_does_not_fit(m, match):
    with pytest.raises(CourseError, match=match):
        encode_tile_map(m)


def test_tile_map_refuses_bad_headers():
    data = bytearray(16 << 20)
    data[0:12] = struct.pack("<hhII", 0, 1, 65, 18)                  # 65 rows: past the 64-row layer
    data[-0x10:-0x4] = struct.pack("<hhII", 0, 1, 10, 18)         # a good header whose body runs past the image
    rs = _rs(bytearray(256 * 1024), data)
    with pytest.raises(CourseError, match=r"0x2000000.*rows"):
        tile_map(rs, DATA)
    with pytest.raises(CourseError, match="outside data ROM"):
        tile_map(rs, 0x00238634)
    with pytest.raises(CourseError, match="outside data ROM"):
        tile_map(rs, DATA + (16 << 20) - 0x10)                       # runs past the image


def test_chars_round_trip_and_refusals():
    px = np.arange(2 * 64).reshape(2, 8, 8) % 16
    b = encode_chars(px)
    assert len(b) == 64
    # pixel 0 of a row is the high nibble of the row's first u16 (segaic24.cpp char_layout, xor 8 on little-endian)
    assert struct.unpack_from("<H", b, 0)[0] == (px[0, 0, 0] << 12 | px[0, 0, 1] << 8 | px[0, 0, 2] << 4 | px[0, 0, 3])
    assert (decode_chars(b) == px).all()
    with pytest.raises(CourseError, match="0-15"):
        encode_chars(np.full((1, 8, 8), 16))
    with pytest.raises(CourseError, match="shape"):
        encode_chars(np.zeros((1, 8, 7), int))
    with pytest.raises(CourseError, match="32"):
        decode_chars(b"\0" * 33)


def _lists_rs():
    """A data image with a char list (two blocks), a palette list and an upload set pointing at them."""
    data = bytearray(16 << 20)
    blk_a = encode_char_block(bytes(range(64)))                     # 2 chars
    blk_b = encode_char_block(b"\x11" * 32)                         # 1 char
    data[0x1000:0x1000 + len(blk_a)] = blk_a
    data[0x1100:0x1100 + len(blk_b)] = blk_b
    cl = encode_char_list([(DATA + 0x1000, 0x010E0000), (DATA + 0x1100, 0x010E9000)])
    data[0x2000:0x2000 + len(cl)] = cl
    pl = encode_palette_list([PaletteUpload(0x01800C00, (1, 2, 3, 4)), PaletteUpload(0x01801C00, ())])
    data[0x3000:0x3000 + len(pl)] = pl
    seg = encode_backdrop_segment(BackdropSegment(0x10000, 2, 32, tuple(range(0x3000, 0x3040))))
    segs = []
    for k in range(BACKDROP_SEGMENTS):
        data[0x4000 + 0x100 * k:0x4000 + 0x100 * k + len(seg)] = seg
        segs.append(DATA + 0x4000 + 0x100 * k)
    us = encode_upload_set(DATA + 0x2000, DATA + 0x3000, segs)
    data[0x5000:0x5000 + len(us)] = us
    us2 = encode_upload_set(DATA + 0x2000, DATA + 0x3000)
    data[0x5100:0x5100 + len(us2)] = us2
    return _rs(bytearray(256 * 1024), data), cl, pl, us, us2


def test_upload_lists_round_trip_synthetic():
    rs, cl, pl, us, us2 = _lists_rs()
    chars = char_list(rs, DATA + 0x2000)
    assert chars == (CharUpload(DATA + 0x1000, 0x010E0000, 2), CharUpload(DATA + 0x1100, 0x010E9000, 1))
    assert [c.first_char for c in chars] == [0x3000, 0x3480]
    assert chars[0].data(rs) == bytes(range(64))
    assert encode_char_list([(c.src, c.dst) for c in chars]) == cl
    pal = palette_list(rs, DATA + 0x3000)
    assert pal == (PaletteUpload(0x01800C00, (1, 2, 3, 4)), PaletteUpload(0x01801C00, ()))
    assert encode_palette_list(pal) == pl
    u = upload_set(rs, DATA + 0x5000, segments=True)
    assert (u.char_list, u.palette_list, u.chars, u.palette) == (DATA + 0x2000, DATA + 0x3000, chars, pal)
    assert len(u.segments) == BACKDROP_SEGMENTS
    assert encode_upload_set(u.char_list, u.palette_list, u.segments) == us
    v = upload_set(rs, DATA + 0x5100, segments=False)
    assert v.segments == () and encode_upload_set(v.char_list, v.palette_list) == us2


@pytest.mark.parametrize("call, match", [
    (lambda: encode_char_block(b""), "empty"),
    (lambda: encode_char_block(b"\0" * 33), "32"),
    (lambda: encode_char_list([(0x00500000, 0x010E0000)]), "outside the ROM"),
    (lambda: encode_char_list([(DATA, 0x01100000)]), "char RAM"),
    (lambda: encode_char_list([(DATA, 0x010E0010)]), "char RAM"),
    (lambda: encode_palette_list([PaletteUpload(0x01804000, (1, 2))]), "palette"),
    (lambda: encode_palette_list([PaletteUpload(0x01803FFC, (1, 2, 3, 4))]), "palette"),
    (lambda: encode_palette_list([PaletteUpload(0x01800000, (1, 2, 3))]), "even"),
    (lambda: encode_palette_list([PaletteUpload(0x01800000, (1, 0x10000))]), "u16"),
    (lambda: encode_upload_set(DATA, DATA, [DATA] * 7), "8"),
    (lambda: encode_upload_set(DATA, 0x10), "outside data ROM"),
    (lambda: encode_backdrop_segment(BackdropSegment(0x10000, 2, 31, (0,) * 62)), "32 columns"),
    (lambda: encode_backdrop_segment(BackdropSegment(0x10000, 59, 32, (0,) * 59 * 32)), "rows"),
    (lambda: encode_backdrop_segment(BackdropSegment(0x10000, 2, 32, (0,) * 63)), "64"),
    (lambda: encode_backdrop_segment(BackdropSegment(0x10000, 1, 32, (0x10000,) * 32)), "u16"),
])
def test_upload_encoders_refuse_what_does_not_fit(call, match):
    with pytest.raises(CourseError, match=match):
        call()


def test_upload_decoders_refuse_bad_lists():
    rs, *_ = _lists_rs()
    data = bytearray(rs.data)
    struct.pack_into("<I", data, 0x2004, 0x01100000)                  # first destination past char RAM
    with pytest.raises(CourseError, match=r"0x2002004.*char RAM"):
        char_list(_rs(rs.program, data), DATA + 0x2000)
    data = bytearray(rs.data)
    struct.pack_into("<I", data, 0x3004, 0x3000)                      # 0x3000 entries: past the palette
    with pytest.raises(CourseError, match=r"0x2003000.*palette"):
        palette_list(_rs(rs.program, data), DATA + 0x3000)
    data = bytearray(rs.data)
    struct.pack_into("<I", data, 0x4004, 59)                          # 59 rows
    with pytest.raises(CourseError, match=r"0x2004000.*rows"):
        upload_set(_rs(rs.program, data), DATA + 0x5000, segments=True)
