"""Tests for the Dreamcast course textures and GAME_MAP decoders and encoders (dtools/dc/tex.py)."""

import random
import struct
from dataclasses import replace

import numpy as np
import pytest

from dtools.dc import pol, tex
from dtools.dc.disc import DcDisc
from dtools.dc.errors import DcError
from dtools.dc.pol import TexEntry
from dtools.dc.tex import Texture, VqData


# ---- twiddling ----

def test_twiddle_synthetic():
    # 8 x 8: bit 2i of the stored index is y bit i, bit 2i+1 is x bit i
    idx = tex.twiddle_index(8, 8)
    assert idx.shape == (8, 8)
    assert (idx[0, 0], idx[1, 0], idx[0, 1], idx[1, 1]) == (0, 1, 2, 3)
    assert idx[2, 0] == 4 and idx[0, 2] == 8 and idx[0, 4] == 32 and idx[4, 0] == 16 and idx[7, 7] == 63
    assert sorted(idx.ravel().tolist()) == list(range(64))
    # 8 wide, 16 tall: two 8 x 8 squares, the top one first; 16 wide, 8 tall: the left one first
    tall = tex.twiddle_index(8, 16)
    assert tall.shape == (16, 8)
    assert (tall[:8] == idx).all() and (tall[8:] == idx + 64).all()
    wide = tex.twiddle_index(16, 8)
    assert wide.shape == (8, 16)
    assert (wide[:, :8] == idx).all() and (wide[:, 8:] == idx + 64).all()
    # both ways: untwiddle then twiddle gives the stream back, twiddle then untwiddle the picture
    for w, h in ((8, 8), (8, 16), (16, 8)):
        stream = np.arange(w * h, dtype=np.uint16) * 7 + 3
        img = tex.untwiddle(stream, w, h)
        assert img.shape == (h, w)
        assert img[0, 1] == stream[2] and img[1, 0] == stream[1]
        assert (tex.twiddle(img) == stream).all()
        pic = np.arange(w * h, dtype=np.uint16).reshape(h, w)
        assert (tex.untwiddle(tex.twiddle(pic), w, h) == pic).all()


def _entry(w, h, fmt, offset=0):
    return TexEntry(w, h, fmt, offset, 0)


def test_twiddled_formats_synthetic():
    # one 8 x 8 texture per pixel format; the texel colours land where the twiddle puts them
    texels = np.arange(64, dtype="<u2") * 0x0421 | 0x8000
    blob = texels.tobytes()
    for fmt, pixel in ((0x100, "ARGB1555"), (0x101, "RGB565"), (0x102, "ARGB4444")):
        (t,) = tex.decode_tex(blob, (_entry(8, 8, fmt),), "SYN_TEX.BIN")
        assert (t.layout, t.pixel, t.mips) == ("twiddled", pixel, 1)
        assert t.data == blob
        rgba = tex.to_rgba(t)
        assert rgba.shape == (8, 8, 4) and rgba.dtype == np.uint8
        assert (rgba[0, 1] == tex.texels_to_rgba(np.array([texels[2]]), pixel)[0]).all()
        assert tex.encode_tex((t,), len(blob)) == blob
    v = np.array([0x8000 | 31 << 10 | 1 << 5 | 16, 0x7C00], dtype=np.uint16)
    assert tex.texels_to_rgba(v, "ARGB1555").tolist() == [[255, 8, 132, 255], [255, 0, 0, 0]]
    v = np.array([31 << 11 | 63 << 5 | 0, 0x0841], dtype=np.uint16)
    assert tex.texels_to_rgba(v, "RGB565").tolist() == [[255, 255, 0, 255], [8, 8, 8, 255]]
    v = np.array([0xF4A1], dtype=np.uint16)
    assert tex.texels_to_rgba(v, "ARGB4444").tolist() == [[68, 170, 17, 255]]


def _mip_texels(n):
    """A twiddled mip chain for an n x n texture: 2 bytes of preamble, then the levels 1 x 1 .. n x n."""
    out = [np.zeros(1, "<u2")]
    s, k = 1, 1
    while s <= n:
        out.append(np.full(s * s, k, "<u2"))
        s, k = s * 2, k + 1
    return np.concatenate(out).tobytes()


def test_twiddled_mip_synthetic():
    blob = _mip_texels(8)
    assert len(blob) == 2 + 2 * (1 + 4 + 16 + 64)
    padded = blob + bytes(-len(blob) % 32)
    (t,) = tex.decode_tex(padded, (_entry(8, 8, 0x201),), "SYN_TEX.BIN")
    assert (t.layout, t.pixel, t.mips, t.data) == ("twiddled_mip", "RGB565", 4, blob)
    # level 0 is the full size (stored last), level 3 the 1 x 1 (stored first, after the preamble)
    for level, size in enumerate((8, 4, 2, 1)):
        rgba = tex.to_rgba(t, level)
        assert rgba.shape == (size, size, 4)
        want = tex.texels_to_rgba(np.array([4 - level], dtype=np.uint16), "RGB565")[0]
        assert (rgba == want).all()
    assert tex.encode_tex((t,), len(padded)) == padded
    with pytest.raises(DcError, match="level"):
        tex.to_rgba(t, 4)
    with pytest.raises(DcError, match="level"):
        tex.to_rgba(t, -1)


def _vq(n, mip=False):
    """A codebook whose entry k holds texels 4k .. 4k+3 (in twiddled order: (0,0), (1,0), (0,1), (1,1)); the
    index grid (n/2 x n/2, twiddled) names entry i at stored position i."""
    codebook = np.arange(1024, dtype="<u2").tobytes()
    if not mip:
        return VqData(codebook, bytes(range((n // 2) ** 2)))
    idx = [bytes([200])]                          # 1 x 1
    s = 2
    while s <= n:
        idx.append(bytes([(s + j) % 256 for j in range((s // 2) ** 2)]))
        s *= 2
    return VqData(codebook, b"".join(idx))


def test_vq_synthetic():
    vq = _vq(8)
    blob = vq.codebook + vq.indices
    assert len(blob) == 2048 + 16
    (t,) = tex.decode_tex(blob, (_entry(8, 8, 0x302),), "SYN_TEX.BIN")
    assert (t.layout, t.pixel, t.mips) == ("vq", "ARGB4444", 1)
    assert t.data == vq
    texels = tex.vq_texels(t)
    assert texels.shape == (8, 8)
    # block (0, 0) is entry 0: texels 0..3 at (0,0), (1,0), (0,1), (1,1); block (0, 1) (stored second) is entry 2
    assert texels[:2, :2].tolist() == [[0, 2], [1, 3]]
    assert texels[:2, 2:4].tolist() == [[8, 10], [9, 11]]
    assert texels[2:4, :2].tolist() == [[4, 6], [5, 7]]
    assert tex.encode_tex((t,), len(blob)) == blob
    assert tex.to_rgba(t).shape == (8, 8, 4)

    vqm = _vq(8, mip=True)
    blob = vqm.codebook + vqm.indices
    assert len(vqm.indices) == 1 + 1 + 4 + 16
    (t,) = tex.decode_tex(blob, (_entry(8, 8, 0x400),), "SYN_TEX.BIN")
    assert (t.layout, t.pixel, t.mips) == ("vq_mip", "ARGB1555", 4)
    assert tex.vq_texels(t, 3).tolist() == [[800]]                 # 1 x 1: entry 200, its first texel
    assert tex.vq_texels(t, 2).tolist() == [[8, 10], [9, 11]]      # 2 x 2: one index, entry 2
    assert tex.vq_texels(t, 1)[:2, :2].tolist() == [[16, 18], [17, 19]]   # 4 x 4 starts at index byte 2: entry 4
    assert tex.vq_texels(t, 0)[:2, :2].tolist() == [[32, 34], [33, 35]]   # 8 x 8 starts at byte 6: entry 8
    assert tex.encode_tex((t,), len(blob)) == blob


def test_unknown_format_refused():
    # Review Focus 3: a format word the files do not use names the texture and the word
    blob = bytes(4096)
    for fmt in (0x103, 0x105, 0x106, 0x000, 0x500, 0x901, 0x10101, 0x80000101):
        with pytest.raises(DcError, match=rf"texture 1.*{fmt:#x}"):
            tex.decode_tex(blob, (_entry(8, 8, 0x101), _entry(8, 8, fmt, 128)), "SYN_TEX.BIN")
    # sizes the layouts cannot hold: not a power of two, too small, a non-square mip chain or VQ
    for w, h, fmt in ((12, 8, 0x101), (8, 4, 0x101), (8, 16, 0x201), (16, 8, 0x402), (2048, 8, 0x101)):
        with pytest.raises(DcError, match=f"texture 0.*{w} x {h}"):
            tex.decode_tex(bytes(1 << 16), (_entry(w, h, fmt),), "SYN_TEX.BIN")
    # a mip chain longer than the texture, a VQ index past its codebook
    (t,) = tex.decode_tex(_mip_texels(8), (_entry(8, 8, 0x201),), "SYN_TEX.BIN")
    for mips in (5, 9, 0):
        with pytest.raises(DcError, match="mip"):
            tex.to_rgba(replace(t, mips=mips))
        with pytest.raises(DcError, match="mip"):
            tex.encode_tex((replace(t, mips=mips),), 4096)
    vq = _vq(8)
    (v,) = tex.decode_tex(vq.codebook + vq.indices, (_entry(8, 8, 0x301),), "SYN_TEX.BIN")
    short = replace(v, data=VqData(vq.codebook[:64], vq.indices))       # 8 entries; the indices reach 15
    with pytest.raises(DcError, match="codebook"):
        tex.to_rgba(short)
    with pytest.raises(DcError, match="codebook"):
        tex.encode_tex((short,), 4096)
    # a Texture whose layout or pixel disagrees with its entry's word
    with pytest.raises(DcError, match="0x301"):
        tex.encode_tex((replace(v, layout="twiddled"),), 4096)
    with pytest.raises(DcError, match="0x301"):
        tex.encode_tex((replace(v, pixel="ARGB4444"),), 4096)
    with pytest.raises(DcError, match="layout"):
        tex.to_rgba(replace(v, layout="stride"))


def test_tex_accepts_numpy_numbers_and_shared_bytes():
    blob = _mip_texels(8)
    (t,) = tex.decode_tex(blob, (_entry(8, 8, 0x201),), "SYN_TEX.BIN")
    e = TexEntry(np.uint16(8), np.int32(8), np.uint32(0x201), np.int64(0), 0)
    tn = replace(t, entry=e, mips=np.int8(4))
    assert tex.encode_tex((tn,), np.int64(len(blob))) == blob
    assert (tex.to_rgba(tn, np.int32(1)) == tex.to_rgba(t, 1)).all()
    assert tex.decode_tex(blob, (e,), "SYN_TEX.BIN")[0].data == blob
    # two entries may name the same bytes; the encoder refuses them if their data differ
    two = tex.decode_tex(blob, (_entry(8, 8, 0x201), _entry(8, 8, 0x201)), "SYN_TEX.BIN")
    assert tex.encode_tex(two, len(blob)) == blob
    other = replace(two[1], data=bytes(len(blob)))
    with pytest.raises(DcError, match="shares its bytes"):
        tex.encode_tex((two[0], other), len(blob))
    # to_rgba reads the entry's word too: an unknown one is refused, naming the word
    with pytest.raises(DcError, match="0x203"):
        tex.to_rgba(replace(t, entry=_entry(8, 8, 0x203)))


def test_tex_truncated_refused():
    # Review Focus 1: a truncated file, an offset past the end, overlapping textures: DcError naming file and offset
    blob = _mip_texels(8) + bytes(20) + bytes(128)
    entries = (_entry(8, 8, 0x201), _entry(8, 8, 0x101, 192))
    tex.decode_tex(blob, entries, "SYN_TEX.BIN")
    with pytest.raises(DcError, match=r"SYN_TEX\.BIN.*texture 1.*0xc0"):
        tex.decode_tex(blob[:-1], entries, "SYN_TEX.BIN")
    with pytest.raises(DcError, match=r"SYN_TEX\.BIN.*texture 0.*0x0"):
        tex.decode_tex(blob[:100], entries, "SYN_TEX.BIN")
    with pytest.raises(DcError, match=r"SYN_TEX\.BIN.*0x10000000"):
        tex.decode_tex(blob, (_entry(8, 8, 0x101, 0x10000000),), "SYN_TEX.BIN")
    with pytest.raises(DcError, match=r"SYN_TEX\.BIN.*overlap.*0x40"):
        tex.decode_tex(blob, (_entry(8, 8, 0x201), _entry(8, 8, 0x101, 64)), "SYN_TEX.BIN")
    # bytes between the textures that are not zero cannot be kept
    bad = bytearray(blob)
    bad[180] = 1
    with pytest.raises(DcError, match=r"SYN_TEX\.BIN.*0xb4"):
        tex.decode_tex(bytes(bad), entries, "SYN_TEX.BIN")
    # the encoder: a texture past the size, overlapping, data of the wrong length
    ts = tex.decode_tex(blob, entries, "SYN_TEX.BIN")
    assert tex.encode_tex(ts, len(blob)) == blob
    with pytest.raises(DcError, match="past"):
        tex.encode_tex(ts, len(blob) - 1)
    with pytest.raises(DcError, match="overlap"):
        tex.encode_tex((ts[0], replace(ts[1], entry=replace(ts[1].entry, offset=64))), len(blob))
    with pytest.raises(DcError, match="bytes"):
        tex.encode_tex((replace(ts[1], data=ts[1].data[:-2]),), len(blob))
    for size in (True, -1, 1.5, "8"):
        with pytest.raises(DcError):
            tex.encode_tex(ts, size)
    with pytest.raises(DcError):
        tex.decode_tex("text", entries, "SYN_TEX.BIN")
    with pytest.raises(DcError):
        tex.decode_tex(blob, ("x",), "SYN_TEX.BIN")
    with pytest.raises(DcError):
        tex.encode_tex(("x",), 16)


def test_tex_decode_never_raises_other_errors():
    vq = _vq(8, mip=True)
    blob = _mip_texels(8) + bytes(20) + vq.codebook + vq.indices + bytes(10)
    rng = random.Random(4)
    for _ in range(2000):
        w = rng.choice((8, 16, 3, 0))
        h = rng.choice((8, 16, w))
        fmt = rng.choice((0x201, 0x101, 0x402, 0x300, 0x203, rng.randrange(1 << 16)))
        off = rng.choice((0, 192, 191, len(blob) - 128, rng.randrange(1 << 32)))
        b = bytearray(blob)
        for _ in range(rng.randint(0, 3)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        cut = rng.choice((len(b), rng.randrange(len(b) + 1)))
        try:
            ts = tex.decode_tex(bytes(b[:cut]), (_entry(w, h, fmt, off),), "FUZZ_TEX.BIN")
        except DcError:
            continue
        for t in ts:
            for level in range(t.mips):
                tex.to_rgba(t, level)
        assert tex.encode_tex(ts, cut) == bytes(b[:cut])


def test_tex_names():
    names = tex.tex_names()
    assert len(names) == 64 == len(set(names))
    assert names[0] == ("CS0_POL.BIN", "CS0_TEX.BIN") and names[1] == ("VCS0_POL.BIN", "VCS0_TEX.BIN")
    assert names[-1] == ("VCS7MR_POL.BIN", "VCS7MR_TEX.BIN")


# ---- GAME_MAP ----

def _game_map_blob(nmaps):
    rng = np.random.default_rng(5)
    return rng.integers(0, 1 << 16, 2048 + 4096 * nmaps, dtype=np.uint16).astype("<u2").tobytes()


def test_game_map_synthetic():
    blob = _game_map_blob(9)
    images = tex.game_map(blob, "GAME_MAP.BIN")
    assert len(images) == 10
    assert images[0].shape == (64, 32) and all(m.shape == (64, 64) for m in images[1:])
    assert images[0].dtype == np.uint16
    raw = np.frombuffer(blob, "<u2")
    assert images[0][0, 1] == raw[2] and images[0][32, 0] == raw[1024]
    assert images[1][1, 0] == raw[2048 + 1]
    assert tex.encode_game_map(images) == blob
    assert tex.encode_game_map(tex.game_map(_game_map_blob(8))) == _game_map_blob(8)
    for cut in (0, 4095, 4096 + 8191, len(blob) - 2):
        with pytest.raises(DcError, match="GAME_MAP"):
            tex.game_map(blob[:cut], "GAME_MAP.BIN")
    with pytest.raises(DcError, match="sheet"):
        tex.encode_game_map((images[1],) + images[1:])
    with pytest.raises(DcError, match="map 1"):
        tex.encode_game_map(images[:2] + (images[2][:32],) + images[3:])
    with pytest.raises(DcError):
        tex.encode_game_map((images[0], images[1].astype(np.float32)))
    with pytest.raises(DcError):
        tex.encode_game_map("maps")


def test_game_map3_synthetic():
    rng = np.random.default_rng(6)
    blob = rng.integers(0, 1 << 16, 256 * 256, dtype=np.uint16).astype("<u2").tobytes()
    sheet = tex.game_map3(blob, "GAME_MAP3.BIN")
    assert sheet.shape == (256, 256) and sheet[0, 1] == np.frombuffer(blob, "<u2")[2]
    assert tex.encode_game_map3(sheet) == blob
    # Review Focus 1: a truncated, odd-length or long blob names the file and the offset, never numpy's ValueError
    for cut in (0, 1, 4097, len(blob) - 1, len(blob) - 2):
        with pytest.raises(DcError, match=rf"GAME_MAP3\.BIN.*{cut:#x}"):
            tex.game_map3(blob[:cut], "GAME_MAP3.BIN")
    with pytest.raises(DcError, match="GAME_MAP3"):
        tex.game_map3(blob + b"\0\0", "GAME_MAP3.BIN")
    with pytest.raises(DcError):
        tex.game_map3("text")
    neg = sheet.astype(np.int32)
    neg[3, 4] = -1
    for bad in (sheet[:128], sheet.astype(np.float32), neg, "x"):
        with pytest.raises(DcError):
            tex.encode_game_map3(bad)


def test_public_helpers_refuse_bad_input():
    # untwiddle / twiddle / twiddle_index: sizes and streams that are not numbers name what they were given
    for args in (([1, 2], "a", "b"), ([1, 2], 2, None), ([1, 2], 1.0, 2), (["a", "b"], 2, 1), ([[1], [2, 3]], 2, 1),
                 ([1, 2, 3], 2, 2), (np.zeros((2, 1)), 2, 1)):
        with pytest.raises(DcError, match="untwiddle"):
            tex.untwiddle(*args)
    with pytest.raises(DcError, match="'a'"):
        tex.untwiddle([1, 2], "a", "b")
    for bad in ("ab", [[1], [2, 3]], np.zeros(4), np.array([["a", "b"]]), np.zeros((3, 2))):
        with pytest.raises(DcError, match="twiddle"):
            tex.twiddle(bad)
    for w, h in (("8", 8), (8, True), (0, 8), (6, 8)):
        with pytest.raises(DcError, match="twiddle"):
            tex.twiddle_index(w, h)
    # texels_to_rgba: not 16-bit texels (negative, too big, floats, strings, ragged)
    for bad in (np.array([-1]), [0x10000], np.array([1.5]), ["ab"], np.array(["ab"]), [[1], [2, 3]], None):
        with pytest.raises(DcError, match="texel"):
            tex.texels_to_rgba(bad, "RGB565")
    with pytest.raises(DcError, match="pixel format"):
        tex.texels_to_rgba([1], "YUV422")
    # game_map_rgba: strings or non-images
    for bad in ("maps", ["ab"], [np.array([-1])], [[[1], [2, 3]]]):
        with pytest.raises(DcError):
            tex.game_map_rgba(bad)
    assert tex.game_map_rgba((np.zeros((2, 2), np.uint16),))[0].shape == (2, 2, 4)


def test_decode_tex_refuses_a_name_that_is_not_a_str():
    for name in (1.5, None, b"X_TEX.BIN", 3):
        with pytest.raises(DcError, match="name"):
            tex.decode_tex(b"", [], name)
        with pytest.raises(DcError, match="name"):
            tex.decode_tex(bytes(2 * 8 * 8), (_entry(8, 8, 0x101),), name)
    assert tex.decode_tex(b"", [], "X_TEX.BIN") == ()


def test_stored_size_checks_its_input():
    assert tex.stored_size("twiddled", 8, 16) == 2 * 8 * 16
    assert tex.stored_size("twiddled_mip", 8, 8) == tex.MIP_PREAMBLE + 2 * (1 + 4 + 16 + 64)
    assert tex.stored_size("vq", 16, 16) == tex.CODEBOOK_SIZE + 64
    assert tex.stored_size("vq_mip", np.int64(16), np.uint16(16)) == tex.CODEBOOK_SIZE + 1 + 1 + 4 + 16 + 64
    for layout in ("bogus", "VQ", None, 3, ("vq",)):
        with pytest.raises(DcError, match="layout"):
            tex.stored_size(layout, 8, 8)
    for w, h in ((1.5, 8), ("8", 8), (8, None), (True, 8), (6, 8), (4, 4), (2048, 8), (-8, 8), (8, 10 ** 400)):
        with pytest.raises(DcError, match="size"):
            tex.stored_size("twiddled", w, h)
    with pytest.raises(DcError, match="square"):
        tex.stored_size("vq_mip", 8, 16)


def test_mips_and_errors_name_the_word():
    # Review Focus 3: the encoder's mip-count and size errors name the texture and its format word
    (t,) = tex.decode_tex(_mip_texels(8), (_entry(8, 8, 0x201),), "SYN_TEX.BIN")
    for mips in (np.array([4, 4]), 4.0, "4", True, None, 12, 0, 5):
        with pytest.raises(DcError, match=r"texture 0.*0x201.*mip"):
            tex.encode_tex((replace(t, mips=mips),), 4096)
        with pytest.raises(DcError, match=r"0x201.*mip"):
            tex.to_rgba(replace(t, mips=mips))
    assert tex.encode_tex((replace(t, mips=np.int16(4)),), len(_mip_texels(8))) == _mip_texels(8)
    for w, h in ((12, 8), (8, 16), (4096, 4096)):
        with pytest.raises(DcError, match=rf"texture 0.*0x201.*{w} x {h}"):
            tex.encode_tex((replace(t, entry=_entry(w, h, 0x201)),), 4096)
    with pytest.raises(DcError, match=r"SYN_TEX\.BIN: texture 0.*0x201.*12 x 8"):
        tex.decode_tex(bytes(4096), (_entry(12, 8, 0x201),), "SYN_TEX.BIN")
    assert not hasattr(tex, "fmt_word")


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


def test_tex_lossless_all(disc):
    assert tex.verify_lossless_tex(disc) == []


def test_game_map_lossless(disc):
    gm = disc.read("GAME_MAP.BIN")
    images = tex.game_map(gm, "GAME_MAP.BIN")
    assert len(images) == 10                        # the 32 x 64 sheet, nine 64 x 64 maps
    assert tex.encode_game_map(images) == gm
    assert (images[8] == images[9]).all()           # maps 7 and 8 are identical
    gm2 = disc.read("GAME_MAP2.BIN")
    images2 = tex.game_map(gm2, "GAME_MAP2.BIN")
    assert len(images2) == 9                        # the same sheet, then eight maps of their own
    assert tex.encode_game_map(images2) == gm2
    assert (images2[0] == images[0]).all()
    assert not any((a == b).all() for a in images2[1:] for b in images[1:])
    gm3 = disc.read("GAME_MAP3.BIN")
    with pytest.raises(DcError, match="GAME_MAP3"):
        tex.game_map(gm3, "GAME_MAP3.BIN")          # not this layout: one 256 x 256 sheet
    sheet = tex.game_map3(gm3)
    assert sheet.shape == (256, 256) and sheet.dtype == np.uint16
    assert tex.encode_game_map3(sheet) == gm3


def test_to_rgba_real(disc):
    m = pol.decode_pol(disc.read("CS0_POL.BIN"), "CS0_POL.BIN")
    ts = tex.decode_tex(disc.read("CS0_TEX.BIN"), m.textures, "CS0_TEX.BIN")
    assert len(ts) == 117
    flat = 0
    for i, t in enumerate(ts):
        rgba = tex.to_rgba(t)
        assert rgba.shape == (t.entry.height, t.entry.width, 4), i
        for level in range(t.mips):
            assert tex.to_rgba(t, level).shape == (t.entry.height >> level, t.entry.width >> level, 4)
        flat += bool((rgba == rgba[0, 0]).all())
    assert flat == 0                                # measured: no CS0 texture is one flat colour


def test_twiddled_rect_synthetic():
    """Layout 0x0d (the car sets' `0xd01`): a twiddled rectangle, stored as the plain twiddled layout is (two bytes a
    texel, no mipmaps), named `twiddled_rect` so the format word re-encodes as read. A 32 x 8 picture: texel (y, x)
    of the left 8 x 8 block is where the twiddle puts it, the next block follows it."""
    texels = (np.arange(32 * 8, dtype="<u2") | 0x8000)
    blob = texels.tobytes()
    (t,) = tex.decode_tex(blob, (_entry(32, 8, 0xD01),), "SYN_TEX.BIN")
    assert (t.layout, t.pixel, t.mips) == ("twiddled_rect", "RGB565", 1) and t.data == blob
    got = tex.texels(t)
    assert got.shape == (8, 32)
    assert got[0, 1] == texels[2] and got[1, 0] == texels[1] and got[0, 8] == texels[64] and got[7, 31] == texels[255]
    assert tex.to_rgba(t).shape == (8, 32, 4)
    assert tex.encode_tex((t,), len(blob)) == blob
    assert tex.stored_size("twiddled_rect", 64, 16) == 2 * 64 * 16
    assert tex.parse_fmt(0xD01) == ("twiddled_rect", "RGB565")
