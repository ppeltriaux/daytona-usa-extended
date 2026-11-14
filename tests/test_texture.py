import struct

import numpy as np
import pytest

from dtools.arcade import course as course_mod
from dtools.arcade import polyrom
from dtools.arcade.romset import RomSet
from dtools.arcade.texture import (COMMON_SHEET_AT, COURSE_SHEETS_AT, LUMA_COUNT_AT, LUMA_TABLES_AT, PALETTE_END,
                                   PALETTE_START, SHEET_BLOCK, ColourTables, TexHeader, TextureError, colour_tables,
                                   colour_xlat, contact_sheet, course_headers, decode_indices, decode_texels, encode_tex_header, encode_texels,
                                   load_dump, mip_levels, mip_spans, mip_texels, object_headers, sheet_source, tex_header, texel_spans, upload_sheets,
                                   with_texels)

DATA = 0x02000000
SRC = {0: 0x200000, 1: 0x400000, 2: 0x300000}      # course sheet blocks (the program table at 0x15a0)
COMMON = 0x500000


def _program() -> bytearray:
    p = bytearray(256 * 1024)
    struct.pack_into("<4I", p, COURSE_SHEETS_AT, DATA + SRC[0], DATA + SRC[1], DATA + SRC[2], DATA + SRC[0])
    struct.pack_into("<I", p, COMMON_SHEET_AT, DATA + COMMON)
    return p


def _rs(textures: bytes = b"", data: bytes | None = None, program: bytes | None = None) -> RomSet:
    return RomSet(program=bytes(program if program is not None else _program()),
                  data=bytes(data if data is not None else bytearray(16 << 20)),
                  copro_data=b"", polygons=b"", textures=bytes(textures), extra={})


def _tex_rom(at: int, words) -> bytes:
    t = bytearray(16 << 20)
    struct.pack_into(f"<{len(words)}H", t, 2 * at, *words)
    return bytes(t)


def _hdr(**kw) -> TexHeader:
    base = dict(tha=0x100, step=0, width=64, height=32, tex_x=64, tex_y=96, sheet=0, textured=True,
                translucent=False, checker=False, wrap_x=False, wrap_y=False, mirror_x=False, mirror_y=False,
                utex=False, utex_min_lod=0, utex_x=0, utex_y=0, luma_base=3, colour=0x229, raw=(0, 1, 0))
    base.update(kw)
    return TexHeader(**base)


# --- header -----------------------------------------------------------------------------------------------------

def test_tex_header_fields():
    # h0 0x44c9: width 64 (bits 0-2 = 1), height 64 (bits 3-5 = 1), wrap x/y (6, 7), utex min lod 1 (10-11),
    # textured (14). h1 0x0007: luma table 7. h2 0x1bac: x 0x2c*32, y 0x0e*32, bit 11 set, sheet 1 (bit 12).
    # h3 0x8a40: colour 0x229.
    rs = _rs(_tex_rom(0x1da0, [0x44c9, 0x0007, 0x1bac, 0x8a40]))
    h = tex_header(rs, 0x1da0, attr=0x1f << 12)                  # header step -1
    assert (h.width, h.height, h.tex_x, h.tex_y, h.sheet) == (64, 64, 1408, 448, 1)
    assert h.textured and not h.translucent and not h.checker
    assert (h.wrap_x, h.wrap_y, h.mirror_x, h.mirror_y) == (True, True, False, False)
    assert (h.utex, h.utex_min_lod, h.utex_x, h.utex_y) == (False, 1, 0, 0)
    assert (h.luma_base, h.colour, h.raw) == (7, 0x229, (0, 1, 0))
    assert h.tha == 0x1da0 and h.step == -1 and h.next_tha == 0x1da0 - 4
    assert h.words == (0x44c9, 0x0007, 0x1bac, 0x8a40)


def test_tex_header_step_sign():
    rs = _rs(_tex_rom(0x40, [0, 0, 0, 0]))
    assert tex_header(rs, 0x40, 0x0f << 12).step == 15
    assert tex_header(rs, 0x40, 0x10 << 12).step == -16
    assert tex_header(rs, 0x40, 0x00001101).step == 1           # only bits 12-16 count


def test_tex_header_masks_rom_address():
    rs = _rs(_tex_rom(0x10, [0x4000, 0, 0, 0x40]))
    assert tex_header(rs, 0x10).words == tex_header(rs, 0x0f000010).words   # bits 24-31 are not part of the address
    assert tex_header(rs, 0x10).colour == 1


def test_tex_header_ram_select_refused():
    rs = _rs(_tex_rom(0, [0, 0, 0, 0]))
    with pytest.raises(TextureError, match=r"0x800010.*texture RAM"):
        tex_header(rs, 0x800010)


def test_tex_header_bad_inputs():
    rs = _rs(_tex_rom(0, [0, 0, 0, 0]))
    with pytest.raises(TextureError, match="tha"):
        tex_header(rs, -1)
    with pytest.raises(TextureError, match="attr"):
        tex_header(rs, 0, attr=1 << 32)
    with pytest.raises(TextureError, match="textures region"):
        tex_header(_rs(b"\0" * 64), 0x40)                      # a short region: the header is not inside it


def test_encode_tex_header_round_trip():
    words = [(0x44c9, 0x0007, 0x1bac, 0x8a40), (0xffff, 0xffff, 0xffff, 0xffff), (0, 0, 0, 0),
             (0xa0d2, 0x1203, 0x0fff, 0x003f)]
    for w in words:
        rs = _rs(_tex_rom(0x200, w))
        h = tex_header(rs, 0x200)
        assert encode_tex_header(h) == struct.pack("<4H", *w)


@pytest.mark.parametrize("field,value,match", [
    ("width", 48, "width"), ("width", 8192, "width"), ("height", 16, "height"),
    ("tex_x", 2048, "tex_x"), ("tex_x", 33, "tex_x"), ("tex_y", 1024, "tex_y"), ("tex_y", -32, "tex_y"),
    ("sheet", 2, "sheet"), ("utex_min_lod", 4, "utex_min_lod"), ("utex_x", 64, "utex_x"), ("utex_y", 512, "utex_y"),
    ("luma_base", 256, "luma_base"), ("colour", 1024, "colour"), ("colour", -1, "colour"),
    ("raw", (256, 0, 0), "raw"), ("raw", (0, 2, 0), "raw"), ("raw", (0, 0, 64), "raw"), ("raw", (0, 0), "raw"),
    ("textured", 2, "textured"),
])
def test_encode_tex_header_refuses(field, value, match):
    with pytest.raises(TextureError, match=match):
        encode_tex_header(_hdr(**{field: value}))


def test_object_headers_walk():
    # three records: steps +1, -1, 0 -> headers at tha, tha+4, tha
    t = _tex_rom(0x100, [0x4000, 0, 0, 0x40, 0x4000, 0, 0, 0x80])
    rs = _rs(t)
    pt = (0.0, 0.0, 0.0)
    recs = [polyrom.Record(attr=0x01 | (0x01 << 12) | 0x100, normal=pt, p0=pt, p1=pt),
            polyrom.Record(attr=0x01 | (0x1f << 12) | 0x100, normal=pt, p0=pt, p1=pt),
            polyrom.Record(attr=0x02 | 0x100, normal=pt, p0=pt, p1=None)]
    obj = polyrom.Object(seed0=pt, seed1=pt, records=recs, word_count=0)
    hs = object_headers(rs, 0x100, obj)
    assert [h.tha for h in hs] == [0x100, 0x104, 0x100]
    assert [h.colour for h in hs] == [1, 2, 1]
    assert [h.step for h in hs] == [1, -1, 0]


# --- texels -----------------------------------------------------------------------------------------------------

def test_sheet_source():
    rs = _rs()
    assert [sheet_source(rs, 0, c) for c in (0, 1, 2)] == [SRC[0], SRC[1], SRC[2]]
    assert sheet_source(rs, 1) == COMMON and sheet_source(rs, 1, 2) == COMMON
    with pytest.raises(TextureError, match="course"):
        sheet_source(rs, 0)                                       # the course sheet needs the course
    with pytest.raises(TextureError, match="course"):
        sheet_source(rs, 0, 3)
    p = _program()
    struct.pack_into("<I", p, COURSE_SHEETS_AT + 4, 0x00238634)   # a pointer that is not a data-ROM block
    with pytest.raises(TextureError, match=r"0x15a4.*0x238634"):
        sheet_source(_rs(program=p), 0, 1)


def test_texel_packing_matches_mame_get_texel():
    # One u16 at the texture's origin: MAME get_texel takes bits 12-15 for (even x, even y), 8-11 (odd x, even y),
    # 4-7 (even x, odd y), 0-3 (odd x, odd y).
    h = _hdr(width=32, height=32, tex_x=64, tex_y=96, sheet=0)
    data = bytearray(16 << 20)
    off = SRC[0] + 2 * ((96 // 2) * 512 + 64 // 2)
    struct.pack_into("<H", data, off, 0xABCD)
    idx = decode_indices(_rs(data=data), h, course=0)
    assert idx.shape == (32, 32) and idx.dtype == np.uint8
    assert idx[0, 0] == 0xA and idx[0, 1] == 0xB and idx[1, 0] == 0xC and idx[1, 1] == 0xD
    assert int(idx.sum()) == 0xA + 0xB + 0xC + 0xD


def test_texel_spans_right_half_maps_to_y_xor_1024():
    # logical x >= 1024 is stored at (x - 1024, y ^ 1024): a 64x32 texture at (1024, 0) sits at storage y 1024.
    h = _hdr(width=64, height=32, tex_x=1024, tex_y=0, sheet=1)
    spans = texel_spans(_rs(), h)
    assert len(spans) == 16 and spans[:2] == [(COMMON + 2 * ((1024 // 2) * 512), 64), (COMMON + 2 * ((1026 // 2) * 512), 64)]


def test_texel_spans_straddling_x_1024():
    h = _hdr(width=64, height=32, tex_x=992, tex_y=32, sheet=0)
    spans = texel_spans(_rs(), h, course=2)
    base = SRC[2]
    assert len(spans) == 32 and spans[:2] == [(base + 2 * (16 * 512 + 496), 32), (base + 2 * ((1056 // 2) * 512 + 0), 32)]


@pytest.mark.parametrize("hkw,course", [
    (dict(width=64, height=32, tex_x=64, tex_y=96, sheet=0), 0),
    (dict(width=128, height=64, tex_x=1024, tex_y=448, sheet=1), None),
    (dict(width=64, height=64, tex_x=992, tex_y=0, sheet=0), 1),
    (dict(width=32, height=1024, tex_x=0, tex_y=0, sheet=0), 2),
])
def test_texels_round_trip(hkw, course):
    h = _hdr(**hkw)
    rng = np.random.default_rng(hkw["tex_x"] + hkw["height"])
    idx = rng.integers(0, 16, size=(h.height, h.width), dtype=np.uint8)
    blob = encode_texels(h, idx)
    assert len(blob) == h.width * h.height // 2
    rs = with_texels(_rs(), h, idx, course=course, mips="keep")
    assert np.array_equal(decode_indices(rs, h, course=course), idx)
    # the encoded bytes are the spans' bytes, in order
    assert b"".join(rs.data[o:o + n] for o, n in texel_spans(rs, h, course)) == blob
    # nothing outside the spans changed
    changed = np.nonzero(np.frombuffer(rs.data, np.uint8))[0]
    covered = np.zeros(16 << 20, bool)
    for o, n in texel_spans(rs, h, course):
        covered[o:o + n] = True
    assert covered[changed].all()


def test_encode_texels_refuses():
    h = _hdr(width=32, height=32)
    with pytest.raises(TextureError, match="shape"):
        encode_texels(h, np.zeros((32, 64), np.uint8))
    with pytest.raises(TextureError, match="0-15"):
        encode_texels(h, np.full((32, 32), 16, np.int32))
    with pytest.raises(TextureError, match="0-15"):
        encode_texels(h, np.full((32, 32), -1, np.int32))
    with pytest.raises(TextureError, match="integer"):
        encode_texels(h, np.zeros((32, 32), np.float32))
    with pytest.raises(TextureError, match="not textured"):
        encode_texels(_hdr(textured=False), np.zeros((32, 64), np.uint8))


@pytest.mark.parametrize("hkw,match", [
    (dict(width=64, height=64, tex_x=1024, tex_y=480), "mip"),      # y 480-543 at x >= 1024: the mip area
    (dict(width=128, height=32, tex_x=1984, tex_y=0), "outside"),   # x past 2048
    (dict(width=32, height=64, tex_x=0, tex_y=992), "outside"),     # y past 1024
    (dict(textured=False), "not textured"),
    (dict(tex_x=33), "tex_x"),                                       # not 32-aligned: texels would land one off
    (dict(width=48), "width"),                                       # not 32 << n
    (dict(tex_x=1001, width=32), "tex_x"),                           # unaligned and straddling x = 1024
    (dict(height=8), "height"),
])
def test_texel_spans_refuses(hkw, match):
    with pytest.raises(TextureError, match=match):
        texel_spans(_rs(), _hdr(**hkw), course=0)


@pytest.mark.parametrize("hkw", [dict(tex_x=33), dict(width=48), dict(tex_x=1001, width=32)])
def test_with_texels_refuses_bad_geometry(hkw):
    h = _hdr(**hkw)
    with pytest.raises(TextureError):
        with_texels(_rs(), h, np.zeros((h.height, h.width), np.uint8), course=0, mips="keep")


def test_with_texels_mips_is_required():
    h = _hdr(width=32, height=32)
    idx = np.zeros((32, 32), np.uint8)
    with pytest.raises(TypeError):
        with_texels(_rs(), h, idx, course=0)                         # no default: the caller must choose
    for bad in ("average", None, "", "KEEP"):
        with pytest.raises(TextureError, match="mips"):
            with_texels(_rs(), h, idx, course=0, mips=bad)


def test_mip_levels():
    assert list(mip_levels(_hdr(width=32, height=32))) == [1, 2, 3, 4]        # down to 2 x 2
    assert list(mip_levels(_hdr(width=256, height=64))) == [1, 2, 3, 4, 5]
    with pytest.raises(TextureError, match="level"):
        mip_texels(_rs(), _hdr(width=32, height=32), 0, course=0)
    with pytest.raises(TextureError, match="level"):
        mip_texels(_rs(), _hdr(width=32, height=32), 5, course=0)


def _get_texel(words: np.ndarray, base_x: int, base_y: int, x: int, y: int) -> int:
    """model2rd.ipp get_texel on a sheet given as u16 words (sheet[offset >> 1] >> 16 * (offset & 1))."""
    x2, y2 = base_x + x, base_y + y
    if x2 >= 1024:
        x2 -= 1024
        y2 ^= 1024
    texel = int(words[(y2 // 2) * 512 + x2 // 2])
    if (y & 1) == 0:
        texel >>= 8
    if (x & 1) == 0:
        texel >>= 4
    return texel & 0xF


def _random_blocks() -> bytearray:
    data = bytearray(16 << 20)
    rng = np.random.default_rng(7)
    for src in (SRC[0], COMMON):
        data[src:src + SHEET_BLOCK] = rng.integers(0, 256, SHEET_BLOCK, dtype=np.uint8).tobytes()
    return data


@pytest.mark.parametrize("hkw,course", [
    (dict(width=64, height=32, tex_x=64, tex_y=96, sheet=0), 0),
    (dict(width=128, height=64, tex_x=1024, tex_y=448, sheet=1), None),
    (dict(width=256, height=256, tex_x=512, tex_y=256, sheet=0), 0),
])
def test_with_texels_mips_keep_and_decimate(hkw, course):
    h = _hdr(**hkw)
    rs = _rs(data=_random_blocks())
    idx = np.random.default_rng(3).integers(0, 16, size=(h.height, h.width), dtype=np.uint8)
    before = upload_sheets(rs, 0)

    kept = with_texels(rs, h, idx, course=course, mips="keep")
    after = upload_sheets(kept, 0)
    for s in (0, 1):
        assert np.array_equal(after[s][0x60000:], before[s][0x60000:])          # mip area untouched

    dec = with_texels(rs, h, idx, course=course, mips="decimate")
    sheets = upload_sheets(dec, 0)                                              # the game's own upload path
    for level in mip_levels(h):
        bx, by = ((h.tex_x - 2048) >> level) & 2047, ((h.tex_y - 1024) >> level) & 1023
        words = sheets[h.sheet ^ (level & 1)]
        got = np.array([[_get_texel(words, bx, by, u, v) for u in range(h.width >> level)]
                        for v in range(h.height >> level)])
        assert np.array_equal(got, idx[::1 << level, ::1 << level]), level
    assert np.array_equal(decode_indices(dec, h, course), idx)
    # only level-0 and mip bytes changed, all in the header's own sheet block
    changed = np.nonzero(np.frombuffer(dec.data, np.uint8) != np.frombuffer(rs.data, np.uint8))[0]
    covered = np.zeros(16 << 20, bool)
    for o, n in texel_spans(rs, h, course) + [sp for k in mip_levels(h) for sp in mip_spans(rs, h, k, course)]:
        covered[o:o + n] = True
    assert covered[changed].all()
    block = sheet_source(rs, h.sheet, course)
    assert block <= changed.min() and changed.max() < block + SHEET_BLOCK


def test_mip_spans_merged():
    h = _hdr(width=64, height=32, tex_x=64, tex_y=96, sheet=0)
    spans = mip_spans(_rs(), h, 1, course=0)
    assert spans == sorted(spans) and all(a + n < b for (a, n), (b, _) in zip(spans, spans[1:]))
    assert sum(n for _, n in spans) == 32 * 16 // 2                    # 32 x 16 texels, two per byte


def test_upload_sheets_layout():
    data = bytearray(16 << 20)
    rng = np.random.default_rng(1)
    data[SRC[0]:SRC[0] + SHEET_BLOCK] = rng.integers(0, 256, SHEET_BLOCK, dtype=np.uint8).tobytes()
    data[COMMON:COMMON + SHEET_BLOCK] = rng.integers(0, 256, SHEET_BLOCK, dtype=np.uint8).tobytes()
    s0, s1 = upload_sheets(_rs(data=data), course=0)
    assert s0.dtype == np.uint16 and s0.shape == s1.shape == (0x80000,)
    # level 0: the first 0xc0000 bytes of each block, as they are
    assert s0[:0x60000].tobytes() == bytes(data[SRC[0]:SRC[0] + 0xC0000])
    assert s1[:0x60000].tobytes() == bytes(data[COMMON:COMMON + 0xC0000])
    # the first mip rows (128 x 512 words) go to the other sheet: the common block's to sheet 0, the course's to 1
    assert s0[0x60000:0x70000].tobytes() == bytes(data[COMMON + 0xC0000:COMMON + 0xE0000])
    assert s1[0x60000:0x70000].tobytes() == bytes(data[SRC[0] + 0xC0000:SRC[0] + 0xE0000])
    # every word of both sheets is written, from exactly 1 MB of each block
    both = np.sort(np.concatenate([s0, s1]))
    src = np.frombuffer(bytes(data[SRC[0]:SRC[0] + SHEET_BLOCK]) + bytes(data[COMMON:COMMON + SHEET_BLOCK]), "<u2")
    assert np.array_equal(both, np.sort(src))


# --- colour -------------------------------------------------------------------------------------------------------

def test_colour_xlat_generator():
    x = colour_xlat(flag=True)
    assert x.shape == (0x6000,) and x.dtype == np.uint16
    assert np.array_equal(x[:0x2000], x[0x2000:0x4000]) and np.array_equal(x[:0x2000], x[0x4000:])
    # row c, luma l < 64: (c * 10 * l + 0x1920) >> 6 when c * l != 0, else 0; 0xffff when >= 256
    assert x[0 * 256 + 5] == 0
    assert x[3 * 256 + 2] == (3 * 20 + 0x1920) >> 6
    assert x[31 * 256 + 63] == 0xFFFF
    # entries 64-255 of row c: (10 c + 0xc9) >> 1 (0 for c = 0); MAME's palette reads entry 64
    assert x[7 * 256 + 64] == (70 + 0xC9) >> 1 and x[0 * 256 + 200] == 0
    y = colour_xlat(flag=False)
    assert y[3 * 256 + 2] == (3 * 24 + 0x1160) >> 6 and y[7 * 256 + 64] == (84 + 0x8B) >> 1


def _tables_rom() -> RomSet:
    data = bytearray(16 << 20)
    struct.pack_into("<I", data, LUMA_COUNT_AT, 2)
    data[LUMA_TABLES_AT:LUMA_TABLES_AT + 256] = bytes(range(256))
    pal = (np.arange(PALETTE_END - PALETTE_START) % 256).astype(np.uint8)
    data[PALETTE_START:PALETTE_END] = pal.tobytes()
    return _rs(data=data)


def test_colour_tables_from_rom():
    t = colour_tables(_tables_rom())
    assert t.luma.shape == (0x8000,) and t.luma[:256].tolist() == list(range(256)) and not t.luma[256:].any()
    n = (PALETTE_END - PALETTE_START) // 2
    assert t.palette.shape == (0x2000,) and n == 1007
    assert t.palette[0x1000] == 0x0100 and t.palette[0x1001] == 0x0302 and not t.palette[0x1000 + n:].any()
    assert np.array_equal(t.xlat, colour_xlat(flag=True))


def test_colour_tables_bad_luma_count():
    rs = _tables_rom()
    rs = rs.with_patch("data", LUMA_COUNT_AT, struct.pack("<I", 257))
    with pytest.raises(TextureError, match="luma"):
        colour_tables(rs)


def test_decode_texels_rgba():
    luma = np.zeros(0x8000, np.uint8)
    luma[3 * 128 + 5 * 8] = 40                        # table 3, texel 5 -> luma 40
    luma[3 * 128 + 15 * 8] = 63
    pal = np.zeros(0x2000, np.uint16)
    pal[0x1000 + 0x22] = (1 << 10) | (2 << 5) | 31    # b 1, g 2, r 31
    xlat = np.zeros(0x6000, np.uint16)
    xlat[31 * 256 + 39] = 200                          # r row 31; luma 40 * light 255 // 256 = 39
    xlat[0x2000 + 2 * 256 + 39] = 100                  # g row 2
    xlat[0x4000 + 1 * 256 + 39] = 64                   # b row 1
    tables = ColourTables(luma=luma, palette=pal, xlat=xlat)
    h = _hdr(width=32, height=32, tex_x=0, tex_y=0, sheet=0, luma_base=3, colour=0x22, translucent=True)
    data = bytearray(16 << 20)
    struct.pack_into("<H", data, SRC[0], 0x5F55)       # (0,0)=5 (1,0)=f (0,1)=5 (1,1)=5
    rgba = decode_texels(_rs(data=data), h, course=0, tables=tables, light=255)
    assert rgba.shape == (32, 32, 4) and rgba.dtype == np.uint8
    gamma = lambda v: int(max((v - 64.0) * 255.0 / 191.0, 0.0))
    assert tuple(rgba[0, 0]) == (gamma(200), gamma(100), gamma(64), 255)
    assert rgba[0, 1, 3] == 0                          # 0xf is transparent in a translucent texture
    opaque = decode_texels(_rs(data=data), _hdr(width=32, height=32, tex_x=0, tex_y=0, luma_base=3, colour=0x22),
                           course=0, tables=tables, light=255)
    assert opaque[0, 1, 3] == 255
    with pytest.raises(TextureError, match="light"):
        decode_texels(_rs(data=data), h, course=0, tables=tables, light=257)


def test_load_dump(tmp_path):
    (tmp_path / "luma.7.bin").write_bytes(bytes([9, 1, 2, 3]) * 0x8000)
    (tmp_path / "pal.7.bin").write_bytes(struct.pack("<H", 0x1234) * 0x2000)
    (tmp_path / "xlat.7.bin").write_bytes(struct.pack("<H", 0x55) * 0x6000)
    t = load_dump(tmp_path, 7)
    assert t.luma.shape == (0x8000,) and (t.luma == 9).all()        # byte 4k of the dump is table byte k
    assert (t.palette == 0x1234).all() and (t.xlat == 0x55).all()
    (tmp_path / "pal.7.bin").write_bytes(b"\0" * 10)
    with pytest.raises(TextureError, match="pal.7.bin"):
        load_dump(tmp_path, 7)


def test_contact_sheet():
    a = np.zeros((32, 64, 4), np.uint8)
    a[..., 0], a[..., 3] = 200, 255
    b = np.zeros((64, 32, 4), np.uint8)                # fully transparent: the checkerboard shows through
    img = contact_sheet([a, b], width=80, gap=4)
    assert img.dtype == np.uint8 and img.shape[1] == 80 and img.shape[2] == 3
    assert img.shape[0] == 4 + 64 + 4 + 32 + 4          # tallest first; the 64-wide tile does not fit beside it
    assert tuple(img[4 + 64 + 4, 4]) == (200, 0, 0)
    assert set(np.unique(img[4:68, 4:36])) <= {96, 160}


# --- real ROM ---------------------------------------------------------------------------------------------------

@pytest.fixture
def romset(romset_zip):
    return _real_cache(romset_zip)


_CACHE = {}


def _real_cache(zp):
    if zp not in _CACHE:
        _CACHE[zp] = RomSet.from_zip(zp)
    return _CACHE[zp]


TEX_POPULATED = ((0, 0x200000), (0x400000, 0x600000))   # u16-word ranges of the textures region with ROMs in them


def _beginner_headers(rs):
    entry = course_mod.course_table(rs)[0]
    out = []
    for p in course_mod.scenery(rs, entry, allow_unmodelled=True):
        obj = polyrom.decode_object(rs.polygons, course_mod.check_rom_object(p.oba), p.obc, 1)
        out.extend((h, r) for h, r in zip(object_headers(rs, p.tha, obj), obj.records))
    return out


def test_real_sheet_sources(romset):
    assert [sheet_source(romset, 0, c) for c in (0, 1, 2)] == [0x200000, 0x400000, 0x300000]
    assert sheet_source(romset, 1) == 0x500000


def test_real_every_beginner_header_inside_textures_region(romset):
    pairs = _beginner_headers(romset)
    assert len(pairs) > 1000
    textured = set()
    for h, _ in pairs:
        assert not h.tha & 0x800000                                # a ROM header, not rasterizer texture RAM
        lo = h.tha & 0x7FFFFF
        assert any(a <= lo and lo + 4 <= b for a, b in TEX_POPULATED), hex(h.tha)
        assert encode_tex_header(h) == romset.textures[2 * lo:2 * lo + 8]
        if h.textured:
            textured.add(h.words)
            assert texel_spans(romset, h, course=0)                 # inside the sheet's level-0 area
            for level in mip_levels(h):                             # mips in the header's own block
                mip_texels(romset, h, level, course=0)
    assert len(textured) > 200


def test_real_beginner_texels_round_trip(romset):
    seen = set()
    for h, _ in _beginner_headers(romset):
        if not h.textured or h.words in seen:
            continue
        seen.add(h.words)
        idx = decode_indices(romset, h, course=0)
        assert idx.shape == (h.height, h.width) and int(idx.max()) <= 15
        spans = texel_spans(romset, h, course=0)
        assert encode_texels(h, idx) == b"".join(romset.data[o:o + n] for o, n in spans)


def test_real_course_headers(romset):
    heads = course_headers(romset, 0)
    assert len({h.words for h in heads}) == len(heads) > 200 and all(h.textured for h in heads)
    assert {h.words for h in heads} == {h.words for h, _ in _beginner_headers(romset) if h.textured}


def test_real_colour_tables(romset):
    t = colour_tables(romset)
    assert int(romset.data[LUMA_COUNT_AT]) == 66
    assert int(t.luma.max()) == 63 and not t.luma[66 * 128:].any()
    assert t.palette[0x1000:0x1000 + 1007].any() and not t.palette[:0x1000].any()


@pytest.mark.mame
@pytest.mark.parametrize("course", [0, 1, 2])
def test_upload_sheets_equal_mame_texture_ram(tmp_path, romset_zip, course):
    """Integration (MAME, one run per course): texture RAM 0 and 1 dumped by texdump.lua at frame 1335, after the
    course load, equal upload_sheets(rs, course) byte for byte."""
    from dtools.mame.runner import LUA_DIR, run_mame
    run_mame(LUA_DIR / "autoplay.lua", 24, tmp_path, romset_zip.parent,
             env={"DAYTONA_COURSE": str(course_mod.PANEL_COURSES[course]),      # the course's panel
                  "DAYTONA_TEXDUMP_DIR": str(tmp_path),
                  "DAYTONA_TEXDUMP_AT": "1335", "DAYTONA_SNAPS": "1335"})
    sheets = upload_sheets(RomSet.from_zip(romset_zip), course)
    for k in (0, 1):
        assert (tmp_path / f"tex{k}.1335.bin").read_bytes() == sheets[k].astype("<u2").tobytes(), (course, k)


def test_mip_indices_and_encode_mip_bytes_round_trip():
    from dtools.arcade.texture import encode_mip_bytes, mip_indices, mip_texels_at, texel_spans_at
    rng = np.random.default_rng(9)
    rs = _rs(data=rng.integers(0, 256, 16 << 20, dtype=np.uint8).tobytes())
    h = _hdr(width=64, height=32, tex_x=64, tex_y=96)
    assert texel_spans_at(h, SRC[0]) == texel_spans(rs, h, course=0)
    for level in mip_levels(h):
        assert all((a == b).all() for a, b in zip(mip_texels_at(h, level, SRC[0]), mip_texels(rs, h, level, 0)))
        m = mip_indices(rs, h, level, course=0)
        assert m.shape == (32 >> level, 64 >> level) and int(m.max()) <= 15
        runs = encode_mip_bytes(h, level, m, SRC[0])
        assert sum(len(b) for _, b in runs) == m.size // 2
        assert all(rs.data[o:o + len(b)] == b for o, b in runs)
    big = _hdr(width=128, height=128, tex_x=32, tex_y=0)         # level 5 starts at an odd column: shared bytes
    with pytest.raises(TextureError, match="shares bytes"):
        encode_mip_bytes(big, 5, np.zeros((4, 4), np.uint8), SRC[0])


# --- texture points ---------------------------------------------------------------------------------------------

def _points_obj(attrs) -> polyrom.Object:
    pt = (0.0, 0.0, 0.0)
    recs = [polyrom.Record(attr=a, normal=pt, p0=pt, p1=pt if isinstance(a, int) and a & 1 else None) for a in attrs]
    return polyrom.Object(seed0=pt, seed1=pt, records=recs, word_count=0)


# a drawn quad, a drawn triangle, a link-0 (move) quad: 8 + 6 + 8 words, read for every record (model2_v.cpp 383-402)
POINT_ATTRS = (0x01 | 0x200, 0x02 | 0x100, 0x00040401)
POINT_WORDS = (0, 0, 0x200, 0, 0x200, 0x200, 0, 0x200,          # Beginner tpa 0x550a8: a 64 x 64 texture's corners
               8, 16, 24, 32, 40, 0xFFFF,
               0, 0, 0, 0, 0, 0, 0, 0)


def test_texture_points_decode_and_encode():
    """texture_points reads 2 words per vertex (v then u), 3 or 4 vertices per record by attr bit 0, move records
    included, from tpa (the ROM word index: tpa & 0x7fffff); encode_texture_points writes the same words back."""
    from dtools.arcade.texture import encode_texture_points, texture_points
    rs = _rs(_tex_rom(0x123, POINT_WORDS))
    obj = _points_obj(POINT_ATTRS)
    pts = texture_points(rs, 0x123, obj)
    assert pts == (((0, 0), (0x200, 0), (0x200, 0x200), (0, 0x200)),
                   ((8, 16), (24, 32), (40, 0xFFFF)),
                   ((0, 0),) * 4)
    assert texture_points(rs, 0x7F000123, obj) == pts               # bits 24-31 above the ROM mask are ignored
    blob = encode_texture_points(pts)
    assert blob == struct.pack(f"<{len(POINT_WORDS)}H", *POINT_WORDS)
    assert encode_texture_points(pts, obj) == blob
    assert encode_texture_points([[list(c) for c in p] for p in pts]) == blob           # lists as well as tuples
    assert encode_texture_points(tuple(tuple((np.uint16(v), np.int64(u)) for v, u in p) for p in pts)) == blob
    assert texture_points(rs, 0x123, _points_obj(())) == () and encode_texture_points(()) == b""


def test_texture_points_refuse():
    """A value past u16 (or negative, a bool, a float), a vertex count other than the object's records' (or not 3 or
    4), junk, a texture-RAM tpa or points past the textures region: TextureError, never a raw error."""
    from dtools.arcade.texture import encode_texture_points, texture_points
    rs = _rs(_tex_rom(0x123, POINT_WORDS))
    obj = _points_obj(POINT_ATTRS)
    pts = texture_points(rs, 0x123, obj)
    quad = ((0, 0),) * 4
    for bad, match in [
            (((0, 0x10000),) * 4, "u16"), (((-1, 0),) * 4, "u16"), (((True, 0),) * 4, "u16"),
            (((0.0, 0),) * 4, "u16"), (((0, 0),) * 2, "3 or 4"), (((0, 0),) * 5, "3 or 4"),
            (((0, 0, 0),) * 4, "pair"), ((0, 0, 0, 0), "pair"), (None, "polygons"), ("abcd", "pair"),
            (b"\0\0\0\0", "polygons"), ({0: quad}, "polygons"), (np.zeros((1, 4, 2), int), "polygons")]:
        with pytest.raises(TextureError, match=match):
            encode_texture_points((bad,) if match != "polygons" else bad)
    with pytest.raises(TextureError, match="3 vertices"):
        encode_texture_points((quad, quad, quad), obj)                  # the triangle record given 4 vertices
    with pytest.raises(TextureError, match="records"):
        encode_texture_points(pts[:2], obj)
    with pytest.raises(TextureError, match="polyrom.Object"):
        encode_texture_points(pts, "obj")
    for args, match in [((None, 0x123, obj), "RomSet"), ((rs, True, obj), "tpa"), ((rs, -1, obj), "tpa"),
                        ((rs, 0x123, None), "polyrom.Object"), ((rs, 0x800123, obj), "texture RAM"),
                        ((rs, 0x7FFFFF, obj), "outside the textures region"),
                        ((rs, 0x123, _points_obj((1, "x"))), "attr"),
                        ((rs, 0x123, polyrom.Object((0.0,) * 3, (0.0,) * 3, None, 0)), "records")]:
        with pytest.raises(TextureError, match=match):
            texture_points(*args)


def test_header_numpy_ints():
    """A header whose fields are numpy integers (a converted header) works as one of Python ints: mip_levels,
    texel_spans_at and mip_texels_at give the same values (no AttributeError from bit_length, no OverflowError or
    silent wrap from a numpy uint8 position); a numpy value past its field is a TextureError."""
    import dataclasses
    from dtools.arcade.texture import encode_mip_bytes, mip_texels_at, texel_spans_at
    h = _hdr(width=64, height=32, tex_x=64, tex_y=96)
    n = dataclasses.replace(h, width=np.int64(64), height=np.uint8(32), tex_x=np.uint8(64), tex_y=np.int16(96),
                            sheet=np.uint8(0))
    assert list(mip_levels(n)) == list(mip_levels(h))
    assert texel_spans_at(n, SRC[0]) == texel_spans_at(h, SRC[0])
    assert all(isinstance(o, int) and isinstance(k, int) for o, k in texel_spans_at(n, SRC[0]))
    for level in mip_levels(h):
        assert all((a == b).all() for a, b in zip(mip_texels_at(n, level, SRC[0]), mip_texels_at(h, level, SRC[0])))
        m = np.zeros((32 >> level, 64 >> level), np.uint8)
        assert encode_mip_bytes(n, level, m, SRC[0]) == encode_mip_bytes(h, level, m, SRC[0])
    assert encode_tex_header(n) == encode_tex_header(h) and encode_texels(n, np.zeros((32, 64), np.uint8))
    for bad in (dict(width=np.uint8(255)), dict(tex_x=np.uint8(255)), dict(tex_x=np.int64(1 << 40))):
        with pytest.raises(TextureError):
            mip_levels(dataclasses.replace(h, **bad))
    with pytest.raises(TextureError, match="TexHeader"):
        mip_levels("header")


@pytest.mark.parametrize("course", [0, 1, 2])
def test_texture_points_round_trip(romset, course):
    """Every placement golden decodes an object for (the cell, actor, billboard, switch and spawn lists, and the HUD
    outline) on ids 0-2: its points decoded at its tpa and encoded back equal the ROM's bytes; each placement's tpa
    except the last in its run is the previous one plus all its records' points (move records included). Beginner's
    move records hold zeros (all but one of 4,713), as do its untextured drawn records (4,521 of 4,566)."""
    from dtools.arcade import golden
    from dtools.arcade.texture import TEX_ADDR_MASK, encode_texture_points, texture_points
    c = golden.decode_course(romset, course)
    places = [(p, c.objects[course_mod.check_rom_object(p.oba)]) for k, lst in c.object_lists.items()
              for p in lst if course_mod.check_rom_object(p.oba) in c.objects]
    places += [(p, c.hud_map.objects[course_mod.check_rom_object(p.oba)]) for p in c.hud_map.entries]
    assert len(places) == {0: 539, 1: 430, 2: 390}[course]
    moves = untextured = 0
    for p, obj in places:
        pts = texture_points(romset, p.tpa, obj)
        at = 2 * (p.tpa & TEX_ADDR_MASK)
        blob = encode_texture_points(pts, obj)
        assert blob == romset.textures[at:at + len(blob)], hex(p.tpa)
        assert len(blob) == 4 * sum(4 if r.is_quad else 3 for r in obj.records)
        if course == 0:
            for r, h, q in zip(obj.records, object_headers(romset, p.tha, obj), pts):
                zero = not any(map(any, q))
                moves += r.link == 0 and zero
                untextured += r.link != 0 and not h.textured and zero
    if course == 0:
        assert (moves, untextured) == (4712, 4521)
