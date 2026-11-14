"""Tests for the Dreamcast background screens LG_BGn, CHP_BGn, NAME_BGnn (dtools/dc/screens.py)."""

import random
from dataclasses import replace

import numpy as np
import pytest

from dtools.dc import screens
from dtools.dc.disc import DcDisc
from dtools.dc.errors import DcError
from dtools.dc.screens import Screen
from dtools.dc.tex import texels_to_rgba

SIZE = 640 * 480 * 2


def _picture(seed=6):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 0x10000, size=(480, 640), dtype=np.uint16)


def test_screen_round_trip_synthetic():
    pic = _picture()
    s = screens.screen_from_texels(pic, "bg_lg", 3)
    assert (s.kind, s.n, s.width, s.height, s.layout, s.pixel) == ("bg_lg", 3, 640, 480, "twiddled_bands", "RGB565")
    blob = screens.encode_screen(s)
    assert isinstance(blob, bytes) and len(blob) == SIZE
    back = screens.decode_screen(blob, "bg_lg", 3)
    assert back == s
    assert (screens.texels(back) == pic).all()
    rgba = screens.to_rgba(back)
    assert rgba.shape == (480, 640, 4) and rgba.dtype == np.uint8
    assert (rgba == texels_to_rgba(pic, "RGB565")).all()
    # the layout: eight twiddled pieces, bands of 256, 128, 64, 32 rows from the top, each 512 then 128 wide; each
    # piece stored bottom row first (its first texel is its bottom-left corner, the second the texel above it)
    stream = np.frombuffer(blob, "<u2")
    assert stream[0] == pic[255, 0] and stream[1] == pic[254, 0] and stream[2] == pic[255, 1]
    assert stream[512 * 256] == pic[255, 512]                       # piece 1: 128 x 256 at x 512
    assert stream[512 * 256 + 128 * 256] == pic[383, 0]             # piece 2: 512 x 128 at y 256
    assert stream[640 * 480 - 128 * 32] == pic[479, 512]            # piece 7: 128 x 32 at (512, 448)
    assert stream[-1] == pic[448, 639]                              # its top-right corner, stored last
    assert [p[:4] for p in screens.PIECES] == [(0, 0, 512, 256), (0, 512, 128, 256), (256, 0, 512, 128),
                                               (256, 512, 128, 128), (384, 0, 512, 64), (384, 512, 128, 64),
                                               (448, 0, 512, 32), (448, 512, 128, 32)]
    # every kind and course; numpy numbers accepted; bytearray and memoryview blobs
    for kind in screens.KINDS:
        for n in (0, np.int64(7)):
            t = screens.decode_screen(bytearray(blob), kind, n)
            assert t.kind == kind and t.n == int(n) and type(t.n) is int and t.data == blob
            assert screens.encode_screen(t) == blob
    assert screens.decode_screen(memoryview(blob), "bg_name", 1).data == blob
    assert screens.screen_names()[:3] == [("bg_lg", 0, "LG_BG0.BIN"), ("bg_chp", 0, "CHP_BG0.BIN"),
                                          ("bg_name", 0, "NAME_BG00.BIN")]
    assert len(screens.screen_names()) == 24


def test_screen_refuses():
    blob = screens.encode_screen(screens.screen_from_texels(_picture(), "bg_chp", 2))
    # wrong size: DcError naming the file and the offset
    with pytest.raises(DcError, match=r"CHP_BG2\.BIN.*0x95ffe"):
        screens.decode_screen(blob[:-2], "bg_chp", 2)
    with pytest.raises(DcError, match=r"NAME_BG05\.BIN.*0x96000"):
        screens.decode_screen(blob + b"\0\0", "bg_name", 5)
    with pytest.raises(DcError, match=r"LG_BG0\.BIN.*0x0"):
        screens.decode_screen(b"", "bg_lg", 0)
    # kind, course and blob
    for kind, n in (("bg", 0), ("model", 0), (None, 0), ("bg_lg", 8), ("bg_lg", -1), ("bg_lg", True),
                    ("bg_lg", 1.0), ("bg_lg", "1"), ("bg_lg", np.bool_(True))):
        with pytest.raises(DcError):
            screens.decode_screen(blob, kind, n)
    for bad in ("x" * SIZE, None, list(blob[:4]), np.zeros(SIZE, np.uint8)):
        with pytest.raises(DcError):
            screens.decode_screen(bad, "bg_lg", 0)
    # the encoder refuses a screen the format cannot hold
    s = screens.decode_screen(blob, "bg_chp", 2)
    for bad in (replace(s, data=blob[:-2]), replace(s, data=None), replace(s, data=list(blob)),
                replace(s, width=512), replace(s, height=True), replace(s, width=640.0),
                replace(s, layout="unknown"), replace(s, layout="twiddled"), replace(s, pixel="ARGB1555"),
                replace(s, kind="bg"), replace(s, n=8), replace(s, n=False), "screen", None):
        with pytest.raises(DcError):
            screens.encode_screen(bad)
        with pytest.raises(DcError):
            screens.to_rgba(bad)
        with pytest.raises(DcError):
            screens.texels(bad)
    # pictures that are not 480 x 640 16-bit texels
    pic = _picture()
    neg, big = pic.astype(np.int32), pic.astype(np.uint32)
    neg[7, 7], big[7, 7] = -1, 0x10000
    for bad in (pic[:479], pic.T, pic.astype(np.float32), pic.astype(bool), neg, big, pic.ravel(), [[1, 2], [3]],
                None, "pic"):
        with pytest.raises(DcError):
            screens.screen_from_texels(bad, "bg_lg", 0)
    with pytest.raises(DcError):
        screens.screen_from_texels(pic, "bg_lg", 8)
    assert screens.screen_from_texels(pic.astype(np.int64), "bg_lg", 0).data == \
        screens.screen_from_texels(pic, "bg_lg", 0).data


def test_screen_decode_never_raises_other_errors():
    rng = random.Random(6)
    for _ in range(50):
        n = rng.choice((0, 1, 2, SIZE - 1, SIZE + 1, rng.randrange(SIZE * 2)))
        blob = bytes(rng.getrandbits(8) for _ in range(min(n, 64))) * (n // 64 + 1)
        blob = blob[:n]
        try:
            s = screens.decode_screen(blob, rng.choice(screens.KINDS), rng.randrange(8))
        except DcError:
            continue
        assert screens.encode_screen(s) == blob


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


def test_screens_lossless_all(disc):
    assert screens.verify_lossless_screens(disc) == []
    assert {len(disc.read(name)) for _, _, name in screens.screen_names()} == {SIZE}


def _seam_ratio(pic):
    """Mean green step across the pieces' seams (rows 255/256, 383/384, 447/448, columns 511/512) over the mean step
    everywhere else."""
    g = (pic.astype(np.int32) >> 5) & 63
    dv, dh = np.abs(np.diff(g, axis=0)), np.abs(np.diff(g, axis=1))
    rows = [255, 383, 447]
    seam = np.concatenate([dv[r] for r in rows] + [dh[:, 511]]).mean()
    inner = np.concatenate([np.delete(dv, rows, axis=0).ravel(), np.delete(dh, 511, axis=1).ravel()]).mean()
    return seam / inner


def test_screens_decode_real(disc):
    for kind, n, name in screens.screen_names():
        s = screens.decode_screen(disc.read(name), kind, n)
        rgba = screens.to_rgba(s)
        assert rgba.shape == (480, 640, 4), name
        assert not (rgba == rgba[0, 0]).all(), name
        # measured: the seams between the pieces are no rougher than the picture (ratio 0.33-1.79 on the 24
        # screens; 5.0-22.8 with the pieces read top row first)
        assert _seam_ratio(screens.texels(s)) < 2.5, name
