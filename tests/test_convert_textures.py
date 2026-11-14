"""Tests for the Dreamcast textures converted into arcade 4-bit texels (dtools/convert/textures.py).

Review Focus 2 (a texture whose texels or mip levels overlap another's, or reach into the mip area or sheet 1): the
packer refuses overlaps and the written block's `upload_sheets` reads every texture back, every mip level included."""

import functools
import random

import numpy as np
import pytest

from dtools.arcade import golden as G
from dtools.arcade import texture as T
from dtools.arcade.errors import DecodeError
from dtools.convert import ConvertError, textures
from dtools.dc import pol, tex
from tests.test_convert_build import _beginner, _romset
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_texture import _random_blocks, _rs

REC601 = np.array([0.299, 0.587, 0.114])


@pytest.fixture
def rs():
    got = _romset()
    if got is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return got


@functools.lru_cache(maxsize=1)
def _tables():
    return T.colour_tables(_romset())


def _header(w, h, colour, luma, x=0, y=0, translucent=False) -> T.TexHeader:
    return T.TexHeader(tha=0, step=0, width=w, height=h, tex_x=x, tex_y=y, sheet=0, textured=True,
                       translucent=translucent, checker=False, wrap_x=False, wrap_y=False, mirror_x=False,
                       mirror_y=False, utex=False, utex_min_lod=1, utex_x=0, utex_y=0, luma_base=luma,
                       colour=colour, raw=(0, 1, 0))


def _decode(rs, idx, colour, luma, translucent=False) -> np.ndarray:
    """`idx` drawn the game's way: written into Beginner's sheet block, read back by texture.decode_texels."""
    h = _header(idx.shape[1], idx.shape[0], colour, luma, translucent=translucent)
    return T.decode_texels(T.with_texels(rs, h, idx, course=0, mips="keep"), h, 0, _tables())


def _rgba(rgb, alpha=255) -> np.ndarray:
    a = np.empty(rgb.shape[:2] + (4,), np.uint8)
    a[..., :3], a[..., 3] = rgb, alpha
    return a


# --- convert_texture ---------------------------------------------------------------------------------------------

def test_convert_texture_grey(rs):
    """A grey ramp (0-252 across x) decodes, through decode_texels with the chosen table and entry, to the ramp within
    the 16 levels: every texel's decoded luminance is the nearest of the 16 the entry gives, and they rise along x."""
    ramp = np.repeat((np.arange(64) * 4)[None, :], 64, 0).astype(np.uint8)
    idx, colour, luma = textures.convert_texture(_rgba(np.repeat(ramp[..., None], 3, 2)), _tables())
    assert idx.shape == (64, 64) and idx.dtype == np.uint8 and luma in (0, 1)
    got = _decode(rs, idx, colour, luma)[..., :3] @ REC601
    levels = _decode(rs, np.repeat(np.arange(16, dtype=np.uint8).repeat(2)[None, :], 32, 0), colour, luma)[0, ::2, :3]
    ly = levels @ REC601
    want = ramp.astype(float)
    nearest = np.abs(want[..., None] - ly).min(-1)
    assert np.allclose(np.abs(got - want), nearest)                 # each texel at its nearest level
    assert (np.diff(got, axis=1) >= -1e-9).all()                    # the ramp rises
    gaps = np.diff(np.sort(ly))
    assert np.abs(got - want).max() <= gaps.max() / 2 + 1e-9
    assert np.ptp(levels, axis=1).max() <= 12                       # a grey entry: r, g, b within 12 at every level


def test_convert_texture_flips_rows(rs):
    """The rows come in as stored (bottom row first, tex.to_rgba) and leave top row first: stored row 0 is the last."""
    rgb = np.full((32, 32, 3), 230, np.uint8)
    rgb[0] = 0                                                      # the picture's bottom row, stored first
    idx, colour, luma = textures.convert_texture(_rgba(rgb), _tables())
    dec = _decode(rs, idx, colour, luma)[..., :3] @ REC601
    assert dec[-1].max() < 40 and dec[:-1].min() > 150


def test_convert_texture_enlarges_and_reduces(rs):
    """Sides below 32 are enlarged to 32 by repeating texels (nearest), sides past 256 halved to 256 (2 x 2 mean)."""
    for side, want in ((8, 32), (16, 32), (32, 32), (256, 256), (512, 256), (1024, 256)):
        rgb = np.random.default_rng(side).integers(0, 256, (side, side, 3), dtype=np.uint8)
        idx, _, _ = textures.convert_texture(_rgba(rgb), _tables())
        assert idx.shape == (want, want)
    rgb = np.random.default_rng(16).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    small = textures.convert_texture(_rgba(rgb), _tables())
    big = textures.convert_texture(_rgba(rgb.repeat(2, 0).repeat(2, 1)), _tables())
    assert np.array_equal(small[0], big[0]) and small[1:] == big[1:]
    flat = np.full((512, 512, 3), 90, np.uint8)
    flat[::2] = 110                                                 # rows alternate: the 2 x 2 mean is 100
    assert np.array_equal(textures.convert_texture(_rgba(flat), _tables())[0],
                          textures.convert_texture(_rgba(np.full((256, 256, 3), 100, np.uint8)), _tables())[0])
    rect = textures.convert_texture(_rgba(np.zeros((16, 512, 3), np.uint8)), _tables())[0]
    assert rect.shape == (32, 256)


def _entry_colours(rs) -> np.ndarray:
    """(entries, tables 0-1, 16 levels, rgb) through decode_texels: every palette entry's 16 colours per table."""
    ramp = np.repeat(np.arange(16, dtype=np.uint8).repeat(2)[None, :], 32, 0)
    hs = [_header(32, 32, 0, 0)]
    base = T.with_texels(rs, hs[0], ramp, course=0, mips="keep")
    out = np.empty((textures.PALETTE_COLOURS, 2, 16, 3))
    for p in range(textures.PALETTE_COLOURS):
        for b in (0, 1):
            out[p, b] = T.decode_texels(base, _header(32, 32, p, b), 0, _tables())[0, ::2, :3]
    return out


@pytest.mark.parametrize("colours", [[(200, 40, 40), (40, 160, 60), (60, 90, 200), (128, 128, 128),
                                      (230, 200, 120), (17, 17, 17), (250, 250, 250), (90, 60, 30)]])
def test_convert_texture_hue(rs, colours):
    """A flat colour picks the palette entry whose decoded colour is nearest, and no other entry is nearer: for each
    entry and table the texel's colour is the level whose luminance is nearest the input's (the luminance is the
    intensity); the converter's choice, decoded by decode_texels, is at least as near as every entry's (the four
    entries the game rewrites at run time are never chosen)."""
    every = _entry_colours(rs)
    allowed = np.ones(textures.PALETTE_COLOURS, bool)
    allowed[list(textures.REWRITTEN)] = False
    for c in colours:
        flat = np.array(c, float)
        idx, colour, luma = textures.convert_texture(_rgba(np.full((32, 32, 3), c, np.uint8)), _tables())
        assert allowed[colour] and len(np.unique(idx)) == 1
        got = _decode(rs, idx, colour, luma)[..., :3]
        assert (got == got[0, 0]).all()
        mine = np.linalg.norm(got[0, 0] - flat)
        ly = every @ REC601                                         # (entries, 2, 16)
        pick = np.abs(ly - flat @ REC601).argmin(-1)                # the luminance-nearest level per entry, table
        theirs = np.linalg.norm(np.take_along_axis(every, pick[..., None, None], 2)[:, :, 0] - flat, axis=-1)
        assert mine <= theirs[allowed].min() + 1e-9, (c, colour, mine, theirs[allowed].min())


def _flat_rgb(tables, p: int) -> np.ndarray:
    """MAME's draw_scanline_solid for palette entry `p` at full light (object.luma 255, luma = 255 >> 2 = 63): each
    channel gamma[xlat[table + component * 256 + 63] & 0xff], written out here from model2rd.ipp."""
    c = int(tables.palette[T.POLY_PALETTE + p]) & 0x7FFF
    g = np.maximum((np.arange(256) - 64.0) * 255.0 / 191.0, 0.0).astype(np.int64)
    return np.array([g[int(tables.xlat[0x2000 * ch + ((c >> (5 * ch)) & 31) * 256 + 63]) & 0xFF] for ch in range(3)],
                    float)


def test_flat_entry(rs):
    """An untextured polygon (Task 5 ruling 3): its flat colour is the palette entry whose colour, as MAME draws a flat
    polygon at full light (draw_scanline_solid: luma 63), is nearest the Dreamcast mesh colour (Euclidean RGB, ties to
    the lowest entry; the four rewritten entries never). Brute force over every entry; black, white and the pure
    primaries; refusals."""
    t = _tables()
    every = np.array([_flat_rgb(t, p) for p in range(textures.PALETTE_COLOURS)])
    assert np.array_equal(textures.flat_colours(t), every)
    allowed = np.ones(textures.PALETTE_COLOURS, bool)
    allowed[list(textures.REWRITTEN)] = False
    rng = np.random.default_rng(11)
    samples = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (229.5, 229.5, 229.5),
               (76.5, 51.0, 0.0)] + [tuple(rng.uniform(0, 255, 3)) for _ in range(40)]
    for c in samples:
        got = textures.flat_entry(c, t)
        d = np.linalg.norm(every - np.array(c, float), axis=-1)
        d[~allowed] = np.inf
        assert got == int(np.argmin(d)), (c, got)
        assert allowed[got]
    assert textures.flat_entry((np.float32(10.0), np.int64(20), 30), t) == textures.flat_entry((10, 20, 30), t)
    for bad in (None, (1, 2), (1, 2, 3, 4), (True, 0, 0), (-1, 0, 0), (0, 256, 0), (0, float("nan"), 0), "rgb",
                np.zeros(3)):
        with pytest.raises(ConvertError, match="colour"):
            textures.flat_entry(bad, t)
    with pytest.raises(ConvertError, match="ColourTables"):
        textures.flat_entry((0, 0, 0), None)
    with pytest.raises(ConvertError, match="ColourTables"):
        textures.flat_colours("tables")


def test_tint_entry(rs):
    """Fix round 1 ruling: a texture tinted by a Dreamcast face colour takes the palette entry whose decoding of its
    own indices (its header's luma table, opaque texels, full light) has the mean colour nearest its untinted decoded
    mean times the tint; brute force over every entry through decode_texels (`_entry_colours`); an untinted texture
    keeps its entry; a darker tint gives a darker mean; refusals."""
    every = _entry_colours(rs)
    allowed = np.ones(textures.PALETTE_COLOURS, bool)
    allowed[list(textures.REWRITTEN)] = False
    rng = np.random.default_rng(5)
    rgba = _rgba(np.clip(rng.normal(150, 40, (64, 64, 3)), 0, 255).astype(np.uint8))
    rgba[:8, :8, 3] = 0                                                     # a translucent corner
    idx, colour, luma = textures.convert_texture(rgba, _tables())
    h = _header(64, 64, colour, luma, translucent=True)
    hist = np.bincount(idx.ravel(), minlength=16).astype(float)
    hist[15] = 0
    means = hist @ every[:, luma] / hist.sum()                              # (entries, 3)
    assert textures.tint_entry(idx, h, (1, 1, 1), _tables()) == colour
    picked = {}
    for tint in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.2, 0.2, 0.2), (0.5, 0.5, 0.5), (0.3, 0.9, 0.4)):
        got = textures.tint_entry(idx, h, tint, _tables())
        d = np.linalg.norm(means - means[colour] * np.array(tint), axis=-1)
        d[~allowed] = np.inf
        assert got == int(np.argmin(d)) and allowed[got], tint
        picked[tint] = got
    assert picked[(0.0, 0.0, 1.0)] != picked[(1.0, 0.0, 0.0)]
    assert means[picked[(0.2, 0.2, 0.2)]].sum() < means[picked[(0.5, 0.5, 0.5)]].sum() < means[colour].sum()
    for bad in ((1, 1), (True, 1, 1), (1.5, 1, 1), (-0.1, 1, 1), (float("nan"), 1, 1), None, "rgb"):
        with pytest.raises(ConvertError, match="tint"):
            textures.tint_entry(idx, h, bad, _tables())
    for bad_idx in (idx[:32], idx.astype(float), idx + 16, None):
        with pytest.raises(ConvertError, match="indices"):
            textures.tint_entry(bad_idx, h, (1, 1, 1), _tables())
    with pytest.raises(ConvertError):
        textures.tint_entry(idx, None, (1, 1, 1), _tables())
    with pytest.raises(ConvertError, match="opaque"):
        textures.tint_entry(np.full((64, 64), 15, np.uint8), h, (1, 1, 1), _tables())
    with pytest.raises(ConvertError, match="ColourTables"):
        textures.tint_entry(idx, h, (0.5, 1, 1), None)


def test_convert_texture_never_rewritten(rs):
    """The four entries the game rewrites at run time are never chosen, not even for a colour only they decode to
    (each of their 32 colours as a flat texture; some, e.g. entry 0x1ae's, no other entry decodes to)."""
    every = _entry_colours(rs)
    for p in textures.REWRITTEN:
        for b, i in ((b, i) for b in (0, 1) for i in range(16)):
            c = np.clip(np.round(every[p, b, i]), 0, 255).astype(np.uint8)
            _, colour, _ = textures.convert_texture(_rgba(np.full((32, 32, 3), c, np.uint8)), _tables())
            assert colour not in textures.REWRITTEN, (p, b, i, c)


def test_convert_texture_translucent(rs):
    """Alpha below 50 % becomes index 15 (transparent in a translucent texture) and no opaque texel uses 15."""
    rgb = np.repeat((np.arange(32) * 8)[None, :, None], 32, 0).repeat(3, 2).astype(np.uint8)
    alpha = np.full((32, 32), 255, np.uint8)
    alpha[:, :8], alpha[:, 8:12] = 0, 127
    alpha[:, 12:16] = 128
    idx, colour, luma = textures.convert_texture(_rgba(rgb, alpha), _tables())
    assert (idx[:, :12] == 15).all() and (idx[:, 12:] != 15).all()
    dec = _decode(rs, idx, colour, luma, translucent=True)
    assert (dec[:, :12, 3] == 0).all() and (dec[:, 12:, 3] == 255).all()
    # a fully transparent texture: every texel 15
    idx, _, _ = textures.convert_texture(_rgba(rgb, 0), _tables())
    assert (idx == 15).all()


# --- pack --------------------------------------------------------------------------------------------------------

def _check_packed(sizes, places):
    assert len(places) == len(sizes)
    grid = np.zeros((T.SHEET_H, T.SHEET_W), np.int16)
    for (w, h), (x, y) in zip(sizes, places):
        assert x % 32 == 0 and y % 32 == 0 and x % w == 0 and y % h == 0          # aligned (to its own size too)
        assert 0 <= x and x + w <= T.SHEET_W and 0 <= y and y + h <= T.SHEET_H
        assert not (x + w > 1024 and y + h > 512)                                   # outside the mip area
        grid[y:y + h, x:x + w] += 1
    assert grid.max() <= 1                                                          # no overlap


def _random_sizes(rng: random.Random, n: int) -> list[tuple[int, int]]:
    out = []
    for _ in range(n):
        if rng.random() < 0.7:
            s = 32 << rng.randrange(4)
            out.append((s, s))
        else:
            out.append((32 << rng.randrange(4), 32 << rng.randrange(4)))
    return out


def _square_tiling(rng: random.Random) -> list[tuple[int, int]]:
    """A random full tiling of the level-0 area by aligned squares 32-256: its 24 cells of 256, each split at random
    into quadrants down to 32."""
    out = []

    def split(side):
        if side > 32 and rng.random() < 0.6:
            for _ in range(4):
                split(side // 2)
        else:
            out.append((side, side))
    for _ in range(textures.LEVEL0_TEXELS // (256 * 256)):
        split(256)
    rng.shuffle(out)
    return out


def test_pack_squares_exact():
    """Square sets pack exactly: a random full tiling by aligned squares always packs (in any order), and one more
    32 x 32 is refused."""
    rng = random.Random(23)
    for _ in range(100):
        full = _square_tiling(rng)
        assert sum(w * h for w, h in full) == textures.LEVEL0_TEXELS
        _check_packed(full, textures.pack(full))
        with pytest.raises(ConvertError, match="do not fit"):
            textures.pack(full + [(32, 32)])


def test_pack():
    """Review Focus 2: random size sets pack without overlap, inside the sheet and outside the mip area; a set one
    texture too many is refused. Squares are exact (refused only past the area); sets with rectangles are best effort,
    and in this sample refused only when at least 95 % full."""
    rng = random.Random(11)
    placed = refused = 0
    for _ in range(300):
        sizes = _random_sizes(rng, rng.randrange(1, 120))
        try:
            places = textures.pack(sizes)
        except ConvertError:
            refused += 1
            area = sum(w * h for w, h in sizes)
            assert area > 0.95 * textures.LEVEL0_TEXELS                     # best effort: refused only when crowded
            if all(w == h for w, h in sizes):
                assert area > textures.LEVEL0_TEXELS                        # squares: only when they cannot fit
            continue
        placed += 1
        _check_packed(sizes, places)
    assert placed > 150 and refused > 50
    for _ in range(200):                                                    # squares only
        sizes = [(32 << k, 32 << k) for k in (rng.randrange(4) for _ in range(rng.randrange(1, 150)))]
        if sum(w * h for w, h in sizes) <= textures.LEVEL0_TEXELS:
            _check_packed(sizes, textures.pack(sizes))
        else:
            with pytest.raises(ConvertError, match="do not fit"):
                textures.pack(sizes)
    # exact fills, then one more
    for full in ([(256, 256)] * 24, [(32, 32)] * 1536, [(64, 256)] * 96, [(256, 32)] * 192):
        _check_packed(full, textures.pack(full))
        with pytest.raises(ConvertError, match="do not fit"):
            textures.pack(full + [(32, 32)])
    assert textures.pack([]) == []
    for big in ((512, 512), (256, 512), (1024, 32)):                        # sides past 256: not a converted size
        with pytest.raises(ConvertError, match="up to 256"):
            textures.pack([big])


def test_pack_written_reads_back():
    """Review Focus 2 through the game's upload: a random packed set written by the converter's writer into a
    synthetic set, uploaded by `upload_sheets`, reads back every texture's level 0 and every mip level; sheet 1's level
    0 and the common block are untouched."""
    rng = random.Random(5)
    sizes = _random_sizes(rng, 60)
    places = textures.pack(sizes)
    _check_packed(sizes, places)
    nrng = np.random.default_rng(5)
    headers = {i: _header(w, h, i, i & 1, x, y) for i, ((w, h), (x, y)) in enumerate(zip(sizes, places))}
    idx = {i: nrng.integers(0, 16, (h.height, h.width), dtype=np.uint8) for i, h in headers.items()}
    rs0 = _rs(data=_random_blocks())
    rs1 = textures.write_texels(rs0, textures.course_texels(headers, idx), 0)
    _check_read_back(rs0, rs1, headers, idx)


def _read(words: np.ndarray, h: T.TexHeader, level: int) -> np.ndarray:
    """model2rd.ipp get_texel over a whole level (vectorised; test_texture._get_texel's rule)."""
    bx, by = ((h.tex_x - 2048) >> level) & 2047, ((h.tex_y - 1024) >> level) & 1023
    if level == 0:
        bx, by = h.tex_x, h.tex_y
    v, u = np.mgrid[0:h.height >> level, 0:h.width >> level]
    x2, y2 = bx + u, by + v
    right = x2 >= 1024
    x2, y2 = np.where(right, x2 - 1024, x2), np.where(right, y2 ^ 1024, y2)
    word = words[(y2 // 2) * 512 + x2 // 2].astype(np.int64)
    shift = np.where(v & 1, 0, 8) + np.where(u & 1, 0, 4)
    return ((word >> shift) & 0xF).astype(np.uint8)


def _check_read_back(rs0, rs1, headers, idx):
    before, after = T.upload_sheets(rs0, 0), T.upload_sheets(rs1, 0)
    mine = np.zeros(T.SHEET_WORDS, bool)                    # sheet 1 words that our odd mip levels may change
    for i, h in headers.items():
        assert np.array_equal(_read(after[0], h, 0), idx[i]), i
        for level in T.mip_levels(h):
            assert np.array_equal(_read(after[level & 1], h, level), idx[i][::1 << level, ::1 << level]), (i, level)
            if level & 1:
                bx, by = ((h.tex_x - 2048) >> level) & 2047, ((h.tex_y - 1024) >> level) & 1023
                v, u = np.mgrid[0:h.height >> level, 0:h.width >> level]
                mine[((by + v) ^ 1024) // 2 * 512 + (bx + u - 1024) // 2] = True
    assert np.array_equal(after[1][:T.LEVEL0_WORDS], before[1][:T.LEVEL0_WORDS])     # sheet 1's level 0 untouched
    assert np.array_equal(after[1][~mine], before[1][~mine])
    common = T.sheet_source(rs0, 1)
    assert rs1.data[common:common + T.SHEET_BLOCK] == rs0.data[common:common + T.SHEET_BLOCK]
    diff = np.flatnonzero(np.frombuffer(rs1.data, np.uint8) != np.frombuffer(rs0.data, np.uint8))
    block = T.sheet_source(rs0, 0, 0)
    assert diff.size == 0 or (block <= diff.min() and diff.max() < block + T.SHEET_BLOCK)


# --- Desert City ---------------------------------------------------------------------------------------------------

DESERT_TEXTURES = 86
DESERT_TRANSLUCENT = 21
DESERT_OCCUPANCY = 822_272         # level-0 texels of the 86 (52.3 % of the sheet's 1,572,864)
# Frozen over the 85 textures with an opaque texel: |mean decoded luminance - the Dreamcast's| and the distance of the
# decoded mean colour from the Dreamcast's, as (median, 90th percentile, largest). Measured 0.60, 7.26, 30.30 and
# 3.32, 27.22, 53.40; the largest is texture 16 (the sky).
MEAN_LUMA_TOLERANCE = (0.65, 7.5, 30.5)
MEAN_COLOUR_TOLERANCE = (3.5, 27.5, 53.5)


@pytest.fixture(scope="module")
def desert_textures(disc):  # noqa: F811
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    from dtools.dc.disc import course_files
    mn, tn = course_files(3, kind="model"), course_files(3, kind="textures")
    m = pol.decode_pol(disc.read(mn), mn)
    blob = disc.read(tn)
    headers, idx = textures.course_textures(m, blob, rs)
    return rs, m, tex.decode_tex(blob, m.textures, tn), headers, idx


def test_course_textures_desert_city(desert_textures):
    """All of CS3's 86 textures convert and pack (none enlarged or reduced: 32-256 already), translucent ones where
    the Dreamcast's alpha is; the written block, uploaded by upload_sheets, reads each texture's level 0 back equal to
    its indices and each mip level equal to the decimation; the decoded mean luminance and colour within frozen
    tolerances of the Dreamcast's."""
    rs, m, dc, headers, idx = desert_textures
    assert sorted(headers) == sorted(idx) == list(range(DESERT_TEXTURES)) == list(range(len(m.textures)))
    sizes = [(h.width, h.height) for h in headers.values()]
    _check_packed(sizes, [(h.tex_x, h.tex_y) for h in headers.values()])
    for i, h in headers.items():
        assert (h.width, h.height) == (max(32, m.textures[i].width), max(32, m.textures[i].height))
        assert h.sheet == 0 and h.textured and h.luma_base in (0, 1) and h.colour not in textures.REWRITTEN
        assert idx[i].shape == (h.height, h.width) and idx[i].dtype == np.uint8
        a = tex.to_rgba(dc[i])
        assert h.translucent == bool((a[..., 3] < 128).any())
        if h.translucent:
            assert np.array_equal(idx[i][::-1] == 15, a[..., 3] < 128)
    assert sum(h.translucent for h in headers.values()) == DESERT_TRANSLUCENT
    assert sum(h.width * h.height for h in headers.values()) == DESERT_OCCUPANCY
    texels = textures.course_texels(headers, idx)
    assert len(texels) == DESERT_TEXTURES
    out = textures.write_texels(rs, texels, 0)
    _check_read_back(rs, out, headers, idx)
    tables = _tables()
    dys, dcs = [], []
    for i, h in headers.items():
        a = tex.to_rgba(dc[i]).astype(float)
        op = a[..., 3] >= 128
        if not op.any():
            continue
        d = T.decode_texels(out, h, 0, tables)[..., :3].astype(float)[::-1]      # back to the stored row order
        dm, am = d[op].mean(0), a[..., :3][op].mean(0)
        dys.append(abs((dm - am) @ REC601))
        dcs.append(float(np.linalg.norm(dm - am)))
    assert len(dys) == DESERT_TEXTURES - 1                                      # 81 is wholly transparent
    for got, tol in ((dys, MEAN_LUMA_TOLERANCE), (dcs, MEAN_COLOUR_TOLERANCE)):
        stats = (float(np.median(got)), float(np.percentile(got, 90)), max(got))
        assert all(v <= t for v, t in zip(stats, tol)), (stats, tol)


def test_beginner_textures_round_trip(rs):
    """Beginner's own headers and texels (both sheets, every mip level as stored) re-encoded through the same writer
    equal the ROM's bytes, into a set whose two blocks were wiped first: sheet 0's through write_texels, sheet 1's
    (which write_texels refuses: the converter never writes the common block) through golden.texel_writes itself."""
    b = _beginner()
    wiped = rs
    for src in {b.sheet_sources[0], b.sheet_sources[1]}:
        wiped = wiped.with_patch("data", src, b"\xa5" * T.SHEET_BLOCK)
    course = tuple(tx for tx in b.texels if tx.header.sheet == 0)
    common = [tx for tx in b.texels if tx.header.sheet == 1]
    assert course and common
    with pytest.raises(ConvertError, match="sheet 1"):
        textures.write_texels(wiped, common[:1], 0)
    out = textures.write_texels(wiped, course, 0)
    buf = bytearray(out.data)
    for tx in common:
        for off, blob, _ in G.texel_writes(tx, b.sheet_sources[1]):
            buf[off:off + len(blob)] = blob
    src = b.sheet_sources[1]
    out = out.with_patch("data", src, bytes(buf[src:src + T.SHEET_BLOCK]))
    written = np.zeros(len(rs.data), bool)
    for tx in b.texels:
        for off, blob, _ in G.texel_writes(tx, b.sheet_sources[tx.header.sheet]):
            written[off:off + len(blob)] = True
    got, want = np.frombuffer(out.data, np.uint8), np.frombuffer(rs.data, np.uint8)
    assert written.sum() > 0 and np.array_equal(got[written], want[written])
    assert (got[~written] == np.frombuffer(wiped.data, np.uint8)[~written]).all()


# --- malformed input ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,value", [
    ("luma", 2 ** 62), ("luma", -1), ("luma", 256), ("palette", -1), ("palette", 0x10000), ("palette", 2 ** 63),
    ("xlat", -1), ("xlat", 0x10000), ("xlat", 2 ** 64 - 1),
])
def test_tables_values_checked(name, value):
    """Colour tables with a value outside colour_tables' ranges (luma 0-0xff, palette and xlat 0-0xffff) are refused,
    not indexed past their ends or read as wrong colours; int64 and uint64 alike."""
    t = T.ColourTables(np.zeros(T.LUMA_SIZE, np.uint8), np.zeros(T.PALETTE_ENTRIES, np.uint16), T.colour_xlat())
    a = getattr(t, name).astype(np.uint64 if value >= 2 ** 63 else np.int64)
    a[7] = value
    bad = T.ColourTables(**{**t.__dict__, name: a})
    for call in (lambda: textures.entry_colours(bad),
                 lambda: textures.convert_texture(np.zeros((32, 32, 4), np.uint8), bad)):
        with pytest.raises(ConvertError, match=f"tables.{name}"):
            call()
    textures.entry_colours(t)                                                   # in range: accepted


@pytest.mark.parametrize("field,value", [
    ("width", np.array([32, 32])), ("height", np.array(32)), ("translucent", np.array([True, False])),
    ("tex_x", 3.0), ("colour", None), ("raw", (0, np.array([1]), 0)), ("raw", [0, 1, 0]), ("sheet", "0"),
    ("width", 48), ("luma_base", 300),
])
def test_course_texels_checks_header_first(field, value):
    """A header field that is an array (or any non-scalar, or out of its range) is a ConvertError from course_texels
    and write_texels, raised before the indices' shape is compared."""
    import dataclasses
    h = dataclasses.replace(_header(32, 32, 1, 0), **{field: value})
    with pytest.raises(ConvertError):
        textures.course_texels({0: h}, {0: np.zeros((32, 32), np.uint8)})
    with pytest.raises(ConvertError):
        textures.write_texels(_rs(), (G.Texels(h, np.zeros((32, 32), np.uint8), ()),), 0)


def test_write_texels_refuses_sheet_1():
    """The converter writes the course sheet only: a texture in sheet 1 (the common block) is refused."""
    import dataclasses
    h = dataclasses.replace(_header(32, 32, 1, 0), sheet=1)
    with pytest.raises(ConvertError, match="sheet 1"):
        textures.write_texels(_rs(), textures.course_texels({0: h}, {0: np.zeros((32, 32), np.uint8)}), 0)


def _junk(rng: random.Random):
    pick = rng.randrange(14)
    if pick == 0:
        return None
    if pick == 1:
        return rng.choice([True, False, 1.5, float("nan"), "x", b"x", -1, 2 ** 40])
    if pick == 2:
        return np.zeros((rng.randrange(0, 70), rng.randrange(0, 70), rng.choice([3, 4, 5])), np.uint8)
    if pick == 3:
        return np.zeros((32, 32, 4), rng.choice([bool, np.float32, np.int8, np.int64, object, np.complex64]))
    if pick == 4:
        a = np.zeros((32, 32, 4), np.int64)
        a[0, 0, 0] = rng.choice([-1, 256, 2 ** 40])
        return a
    if pick == 5:
        return np.zeros(rng.choice([(32,), (32, 32), (2, 32, 32, 4)]), np.uint8)
    if pick == 6:
        return [[(rng.randrange(-40, 300), True)]]
    if pick == 7:
        return [(rng.choice([32, 64, 48, 0, -32, 2048, True, 32.0]), rng.choice([32, 64, 33, None]))]
    if pick == 8:
        return [(32, 32, 32)]
    if pick == 9:
        return {rng.randrange(5): rng.choice([None, 3, np.zeros((32, 32), np.uint8)])}
    if pick == 10:
        return rng.choice([(G.Texels(None, None, None),),
                           {0: _header(32, 32, 1, 0).__class__(**{**_header(32, 32, 1, 0).__dict__,
                                                                  rng.choice(["width", "tex_y", "wrap_x", "colour"]):
                                                                  np.array([1, 2])})}])
    if pick == 11:
        return rng.choice([T.ColourTables(np.zeros(3), np.zeros(3), np.zeros(3)), object()])
    if pick == 12:
        return rng.choice([0, 1, 2, 3, -1, True, None])
    return np.full((64, 64, 4), 7, np.uint8)


def test_textures_fuzz():
    """Malformed inputs to every public function raise a DecodeError (ConvertError, or a DcError / TextureError that
    the converter lets through), never a raw Python or numpy error."""
    rng = random.Random(1)
    tables = _tables() if _romset() else T.ColourTables(np.zeros(T.LUMA_SIZE, np.uint8),
                                                         np.zeros(T.PALETTE_ENTRIES, np.uint16),
                                                         T.colour_xlat())
    rs0 = _rs(data=bytearray(16 << 20))
    good_h = {0: _header(32, 32, 1, 0)}
    good_i = {0: np.zeros((32, 32), np.uint8)}
    calls = [
        lambda: textures.convert_texture(_junk(rng), tables if rng.random() < 0.7 else _junk(rng)),
        lambda: textures.pack(_junk(rng)),
        lambda: textures.pack([_junk(rng)]),
        lambda: textures.course_texels(_junk(rng), good_i),
        lambda: textures.course_texels(good_h, _junk(rng)),
        lambda: textures.course_texels({0: _header(64, 32, 1, 0)}, good_i),
        lambda: textures.write_texels(rs0, _junk(rng), 0),
        lambda: textures.write_texels(_junk(rng), (), 0),
        lambda: textures.write_texels(rs0, textures.course_texels(good_h, good_i), _junk(rng)),
        lambda: textures.course_textures(_junk(rng), b"", rs0),
        lambda: textures.course_textures(pol.Model((), (), ()), _junk(rng), rs0),
        lambda: textures.course_textures(pol.Model((), (), ()), b"", _junk(rng)),
    ]
    for _ in range(3000):
        try:
            rng.choice(calls)()
        except DecodeError:
            pass
    with pytest.raises(ConvertError):
        textures.convert_texture(np.zeros((32, 32, 4), bool), tables)
    with pytest.raises(ConvertError):
        textures.pack([(True, 32)])
    with pytest.raises(ConvertError):
        textures.pack([(48, 32)])
    # two textures written over each other: refused
    with pytest.raises(ConvertError, match="overlap"):
        textures.write_texels(rs0, textures.course_texels({0: _header(64, 64, 1, 0), 1: _header(32, 32, 2, 0)},
                                                          {0: np.zeros((64, 64), np.uint8),
                                                           1: np.ones((32, 32), np.uint8)}), 0)
