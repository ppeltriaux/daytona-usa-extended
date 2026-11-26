"""The attract logo and "Extended" drawn into it (dtools/arcade/title.py; docs/formats/model2-attract-logo.md; discovery
log docs/spike/2026-11-release.md, "The title"). Offline: the logo decoded from the set against the RAM dumps of Step 1
(out/phase12/title/dump/<set>/3450.*, game data, never committed: those checks skip without them), the four variants
and the mock-ups."""

import builtins
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from dtools.arcade import tiles as TL
from dtools.arcade import title as TT
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet
from dtools.config import REPO_ROOT
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

DUMPS = REPO_ROOT / "out" / "phase12" / "title" / "dump"
LOGO_FRAME = 3450       # a frame of the first logo (3265-3668 on a stock boot), the wipe done


@lru_cache(maxsize=1)
def _stock(path: str) -> RomSet:
    return RomSet.from_zip(Path(path))


@pytest.fixture
def rs(romset_zip) -> RomSet:
    return _stock(str(romset_zip))


def _dump(name: str):
    d = DUMPS / name
    if not (d / f"{LOGO_FRAME}.tile").is_file():
        pytest.skip(f"no RAM dump at {d} (Phase 12 Task 3 Step 1; game data, not in the repository)")
    tile = np.fromfile(d / f"{LOGO_FRAME}.tile", "<u2")
    chars = TL.decode_chars((d / f"{LOGO_FRAME}.char").read_bytes())
    pal = np.fromfile(d / f"{LOGO_FRAME}.pal", "<u2")
    return tile, chars, pal


def _layer(tile, chars, layer: int) -> np.ndarray:
    """(512, 512) palette indices of a tile layer as the dump holds it (0 where the pen is 0, transparent)."""
    m = tile[0x1000 * layer:0x1000 * (layer + 1)].reshape(64, 64)
    px = chars[m & TL.CHAR_MASK]
    pens = ((m >> 7) & 0xFF).astype(np.int32)[:, :, None, None] * 16 + px
    return np.where(px == 0, 0, pens).transpose(0, 2, 1, 3).reshape(512, 512)


def test_logo_decodes(rs):
    """Step 1's answer: the logo is two tile maps the logo state 0x222fd0 blits, layer 0's black letters and flags
    (data 0x0282a270, 34 x 62 from tile row 7) over layer 2's field, grey shadow and checker bands (0x0282499e, 48 x 62
    from row 0), their chars the boot upload set's (0x1800+511, palettes 0x30-0x33; 0x1a00+379, 0x34-0x36). `render`
    is the dump's layers 2 and 0 at the logo frame pixel for pixel, through the palette the dump holds, on every cell
    the HUD does not draw (layer 0's CREDIT / INSERT COIN / SEGA tiles, tile rows 38-46 only); the M5b set's dump is
    the stock one's."""
    logo = TT.logo(rs)
    assert logo.kind == "tiles"
    assert [(g.layer, g.row, g.col, g.map_at, g.map.rows, g.map.cols) for g in logo.layers] == \
        [(0, 7, 0, 0x0282A270, 34, 62), (2, 0, 0, 0x0282499E, 48, 62)]
    assert logo.frames == ((3265, 3668), (6432, 6835))
    assert {c >> 7 for c in logo.chars} == set(range(0x30, 0x37))
    assert min(logo.chars) == 0x1800 and max(logo.chars) == 0x1B7A and len(logo.chars) == 511 + 379
    tile, chars, pal = _dump("stock")
    for c, px in logo.chars.items():
        assert np.array_equal(px, chars[c]), hex(c)
    for colour, words in logo.palette.items():
        # pens 1-15: pen 0 is never drawn (transparent); the boot writes 0xf0dc there in 0x30-0x33, the dump holds 0
        assert tuple(int(w) for w in pal[16 * colour + 1:16 * colour + 16]) == words[1:], hex(colour)
    got = TT.render(rs, logo)
    assert got.shape == (TT.SCREEN_H, TT.SCREEN_W, 3) and got.dtype == np.uint8
    l0, l2 = _layer(tile, chars, 0), _layer(tile, chars, 2)
    want = TT.palette_rgb(pal)[np.where(l0 != 0, l0, l2)][:TT.SCREEN_H, :TT.SCREEN_W]
    # the HUD: layer-0 cells whose dump tile is not the logo's (its map tile there, else a transparent char)
    g0 = logo.layers[0]
    expect = np.full((64, 64), -1, np.int64)
    expect[g0.row:g0.row + g0.map.rows, g0.col:g0.col + g0.map.cols] = np.array(g0.map.tiles()).reshape(
        g0.map.rows, g0.map.cols)
    dumped = tile[:0x1000].reshape(64, 64).astype(np.int64)
    blank = np.array([not chars[t & TL.CHAR_MASK].any() for t in dumped.ravel()]).reshape(64, 64)
    hud = ~((dumped == expect) | ((expect == -1) & blank))
    assert hud.any() and np.nonzero(hud)[0].min() >= 38                  # the HUD's rows, nothing in the logo's
    keep = ~np.kron(hud, np.ones((8, 8), bool))[:TT.SCREEN_H, :TT.SCREEN_W]
    assert keep.mean() > 0.75
    assert np.array_equal(got[keep], want[keep])
    m5b = _dump("m5b")
    assert all(np.array_equal(a, b) for a, b in zip((tile, chars, pal), m5b))


def test_logo_refuses(rs):
    """A set whose logo state does not blit the logo maps (an lda site rewritten) is a TitleError, as is a set that
    is not a RomSet; an unknown variant too."""
    with pytest.raises(TT.TitleError):
        TT.logo(rs.with_patch("program", TT.LOGO_SITES[0], b"\0\0\0\0"))
    with pytest.raises(TT.TitleError):
        TT.logo("daytona.zip")
    with pytest.raises(TT.TitleError):
        TT.extended_pixels(rs, "v5")
    assert issubclass(TT.TitleError, DecodeError)


def _cells(changed: np.ndarray) -> set[tuple[int, int]]:
    ys, xs = np.nonzero(changed)
    return set(zip((ys // 8).tolist(), (xs // 8).tolist()))


HUD_REFUSED = ("v1", "v2", "v4")
"""The variants whose shadow's last row reaches tile row 38, the HUD's INSERT COIN(S) band (Task 3's review, I1)."""


def test_variants(rs, monkeypatch):
    """Four variants, Pascal's pick v3 (TT.PICKED). Each `extended_pixels` the HUD leaves room for uses only palette
    indices the logo itself shows, changes pixels only inside its box, only in cells where layer 0 holds the logo map's
    transparent char (so the word goes into layer 0 alone, the field and the select's background untouched) and none
    of the HUD's (TT.HUD_ROWS x TT.HUD_COLS, INSERT COIN(S)), within the budget of Step 2 (TT.BUDGET_CHARS new chars,
    the boot room TT.ROOM), is deterministic and reads nothing but the set (no file is opened). v1, v2 and v4 reach
    the HUD's row 38 (Task 3's review, I1) and are refused (TitleError naming the HUD), as is any variant that does:
    their words, drawn without the check, differ from v3's and one another visibly."""
    assert list(TT.VARIANTS) == ["v1", "v2", "v3", "v4"] and TT.PICKED == "v3"
    assert (TT.HUD_ROWS, TT.HUD_COLS) == ((38, 41), (12, 50))
    logo = TT.logo(rs)
    base = TT.logo_pixels(rs, logo)
    used = set(np.unique(base).tolist())
    g0 = logo.layers[0]
    l0 = np.array(g0.map.tiles()).reshape(g0.map.rows, g0.map.cols)
    assert sum(hi - lo for lo, hi in TT.ROOM) == TT.BUDGET_CHARS
    words = {name: TT._word(logo, v) for name, v in TT.VARIANTS.items()}
    hud = {name: {(r, c) for r, c in _cells(w != 0) if TT.HUD_ROWS[0] <= r < TT.HUD_ROWS[1] and
                  TT.HUD_COLS[0] <= c < TT.HUD_COLS[1]} for name, w in words.items()}
    assert [n for n in TT.VARIANTS if hud[n]] == list(HUD_REFUSED)
    for name in HUD_REFUSED:
        with pytest.raises(TT.TitleError, match="HUD"):
            TT.extended_pixels(rs, name)
        with pytest.raises(TT.TitleError, match="HUD"):
            TT.new_chars(rs, name)
    names = list(words)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert ((words[a] != 0) != (words[b] != 0)).sum() > 500, (a, b)
    assert len({(v.size, v.case, v.slant, v.box) for v in TT.VARIANTS.values()}) == 4
    # a variant of the picked one moved down into the HUD's band is refused
    v = TT.VARIANTS["v3"]
    low = v._replace(box=(v.box[0], 279, v.box[2], 305))
    monkeypatch.setitem(TT.VARIANTS, "low", low)
    with pytest.raises(TT.TitleError, match="HUD"):
        TT.extended_pixels(rs, "low")
    monkeypatch.delitem(TT.VARIANTS, "low")
    out = {}
    for name in [n for n in TT.VARIANTS if n not in HUD_REFUSED]:
        v = TT.VARIANTS[name]
        px = TT.extended_pixels(rs, name)
        assert px.shape == base.shape and px.dtype == base.dtype
        assert set(np.unique(px).tolist()) <= used, name
        changed = px != base
        x0, y0, x1, y1 = v.box
        inside = np.zeros_like(changed)
        inside[y0:y1, x0:x1] = True
        assert changed.sum() > 1000 and not (changed & ~inside).any(), name
        for r, c in _cells(changed):
            assert l0[r - g0.row, c - g0.col] == TT.blank_tile(rs, logo), (name, r, c)
            assert not (TT.HUD_ROWS[0] <= r < TT.HUD_ROWS[1] and TT.HUD_COLS[0] <= c < TT.HUD_COLS[1]), (name, r, c)
        chars = TT.new_chars(rs, name)
        assert 0 < len(chars) <= TT.BUDGET_CHARS, name
        assert len({c.tobytes() for c in chars}) == len(chars)
        out[name] = px
    assert list(out) == ["v3"]
    with monkeypatch.context() as mp:
        def no_open(*a, **k):
            raise AssertionError("extended_pixels opened a file")
        mp.setattr(builtins, "open", no_open)
        mp.setattr(Path, "read_bytes", no_open)
        for name in out:
            assert np.array_equal(TT.extended_pixels(rs, name), out[name])


def test_mockup(rs):
    """The mock-up is the logo as `render` draws it beside the variant's pixels through the set's own palette, both
    at 2x (nearest), a dark gap between."""
    logo = TT.logo(rs)
    m = TT.mockup(rs, TT.PICKED)
    h, w = 2 * TT.SCREEN_H, 2 * TT.SCREEN_W
    assert m.shape == (h, 2 * w + 2 * TT.MOCKUP_GAP, 3) and m.dtype == np.uint8
    left = TT.render(rs, logo).repeat(2, 0).repeat(2, 1)
    assert np.array_equal(m[:, :w], left)
    right = TT.palette_rgb(TT.palette_ram(logo))[TT.extended_pixels(rs, TT.PICKED)].repeat(2, 0).repeat(2, 1)
    assert np.array_equal(m[:, w + 2 * TT.MOCKUP_GAP:], right)


def test_drawn_refuses_taken_cells(rs):
    """`extended_pixels` and `new_chars` draw only into layer 0's transparent cells: with one cell under the picked
    word given another tile of the logo map (the logo's ink, a flag), both are a TitleError naming that cell (Task 3's
    review, M1). On the titled set (the word's own tiles there) too: test_extended_title_writes."""
    logo = TT.logo(rs)
    g0 = logo.layers[0]
    word = TT.extended_pixels(rs, TT.PICKED) != TT.logo_pixels(rs, logo)
    r, c = min(_cells(word))
    at = g0.map_at - TT.DATA_BASE + 12 + 2 * ((r - g0.row) * g0.map.cols + (c - g0.col))
    blank = TT.blank_tile(rs, logo)
    inked = sorted({t for t in g0.map.tiles() if logo.chars[t & TL.CHAR_MASK].any()})
    assert blank not in inked and len(inked) > 100
    for tile in (inked[0], inked[-1]):
        taken = rs.with_patch("data", at, struct.pack("<h", tile))
        assert TT.logo(taken).layers[0].map.tile(r - g0.row, c - g0.col) == tile
        for f in (TT.extended_pixels, TT.new_chars):
            with pytest.raises(TT.TitleError, match=rf"tile \({r}, {c}\) is not a transparent cell"):
                f(taken, TT.PICKED)


def test_arguments_refused(rs):
    """The public functions taking a logo, a variant or palette words refuse anything else with a TitleError (a
    DecodeError), never a raw exception (Task 3's review, M2): `render`, `logo_pixels`, `palette_ram`, `blank_tile`
    (a non-Logo, a Logo whose fields are not `logo`'s, a non-RomSet), `coverage` (a non-Variant, a Variant of the wrong
    types), `palette_rgb` (strings, floats, bools, ragged lists, values outside u16), `title_palette`, `titled`,
    `title_spans` and `extended_title`."""
    logo = TT.logo(rs)
    bad_logos = ["logo", None, (), logo._replace(kind="poly"), logo._replace(layers=()),
                 logo._replace(chars={}), logo._replace(palette={0x30: (0,) * 15}),
                 logo._replace(layers=(logo.layers[0]._replace(row=40),)),
                 logo._replace(chars={k: np.zeros((4, 4), np.uint8) for k in logo.chars})]
    for bad in bad_logos:
        for f in (lambda x: TT.render(rs, x), lambda x: TT.logo_pixels(rs, x), TT.palette_ram,
                  lambda x: TT.blank_tile(rs, x), lambda x: TT.title_palette(x, np.ones((1, 8, 8), int))):
            with pytest.raises(TT.TitleError):
                f(bad)
    for f in (TT.render, TT.logo_pixels, TT.blank_tile):
        with pytest.raises(TT.TitleError):
            f("daytona.zip", logo)
    v = TT.VARIANTS[TT.PICKED]
    for bad in ("v3", None, tuple(v), v._replace(box=(8, 229, 248)), v._replace(box=(8.0, 229, 248, 255)),
                v._replace(size="24"), v._replace(size=True), v._replace(case=None), v._replace(slant="0.3"),
                v._replace(slant=True)):
        with pytest.raises(TT.TitleError):
            TT.coverage(bad)
    for bad in ("0x8000", [1.5], [True], [[1, 2], [3]], [-1], [0x10000], None, object()):
        with pytest.raises(TT.TitleError):
            TT.palette_rgb(bad)
    assert TT.palette_rgb([0x8000, 0x7FFF]).shape == (2, 3)
    for bad in (np.zeros((0, 8, 8), int), np.zeros((1, 8, 8), int), np.ones((1, 8, 7), int),
                np.full((1, 8, 8), 0x2000), np.full((1, 8, 8), 1.0), "chars"):
        with pytest.raises(TT.TitleError):
            TT.title_palette(logo, bad)
    for f in (TT.titled, TT.title_spans, TT.extended_title):
        with pytest.raises(TT.TitleError):
            f("daytona.zip")


def test_title_palette(rs):
    """The picked word's chars hold indices of two logo palettes (layer 0's black and grey ramp in 0x30, layer 2's
    shadow grey in 0x34); `title_palette` puts them in one: layer 0's palette with its pens kept and the shadow grey on
    the highest pen the word does not use (pen 15, no ramp step being that grey). Through it every pixel of every char
    shows the palette word its index showed, pen 0 exactly where the index is 0 (Task 3's review, M3). Sixteen
    colours or more do not fit 15 pens (TitleError)."""
    logo = TT.logo(rs)
    chars = np.stack(TT.new_chars(rs, TT.PICKED))
    used = set(np.unique(chars).tolist()) - {0}
    assert {i >> 4 for i in used} == {0x30, 0x34}
    words, pens = TT.title_palette(logo, chars)
    assert len(words) == 16 and pens.shape == chars.shape and pens.dtype == np.uint8
    ram = TT.palette_ram(logo)
    assert np.array_equal(np.array(words)[pens][chars != 0], ram[chars][chars != 0])
    assert np.array_equal(pens == 0, chars == 0)
    assert {int(p) for p, i in zip(pens.ravel(), chars.ravel()) if i >> 4 == 0x34} == {15}
    assert all(words[p] == ram[0x300 + p] for p in range(15))
    assert all(int(p) == i & 15 for p, i in zip(pens.ravel(), chars.ravel()) if i >> 4 == 0x30)
    # sixteen distinct words (ramp 0x301-0x308, the flags 0x30a-0x30d, then layer 2's) cannot share 15 pens
    many = sorted({i for i in range(0x300, 0x370) if ram[i] and i & 15} - {0x309, 0x30e, 0x30f},
                  key=lambda i: (i >> 4 != 0x30, i))
    distinct = list({int(ram[i]): i for i in reversed(many)}.values())[:16]
    assert len(distinct) == 16
    with pytest.raises(TT.TitleError, match="15 pens"):
        TT.title_palette(logo, np.array(distinct, int).repeat(4).reshape(1, 8, 8))


def test_extended_title_refuses(rs, monkeypatch):
    """`extended_title` refuses (TitleError, before writing): a set that is not custom (TITLE_RUN is the custom data
    area), a TITLE_RUN byte not 0xff, an owned span over TITLE_RUN (golden.owned_spans), a ROOM run the boot uploads
    (ROOM patched to the logo's own block), an unknown or HUD-refused variant; on the custom stock set it writes, and
    a second call is refused (`titled`)."""
    from dtools.arcade import golden as G
    from dtools.arcade.romset import CUSTOM_FREE
    assert TT.TITLE_RUN == CUSTOM_FREE["data"][1] and TT.BOOT_WORD == 0x02700000 and TT.MAP_WORD == 0x22FF0
    with pytest.raises(TT.TitleError, match="custom"):
        TT.extended_title(rs)
    cs = rs.as_custom()
    assert not TT.titled(cs) and TT.title_spans(cs) == []
    with pytest.raises(TT.TitleError, match="not free"):
        TT.extended_title(cs.with_patch("data", TT.TITLE_RUN[0] + 0x1000, b"\0"))
    for name in ("v5", "v1", None):
        with pytest.raises(TT.TitleError):
            TT.extended_title(cs, name)
    with monkeypatch.context() as mp:
        mp.setattr(TT, "ROOM", ((0x1800, 0x1A00),))
        with pytest.raises(TT.TitleError, match="uploaded by the boot"):
            TT.extended_title(cs)
    with monkeypatch.context() as mp:
        real = G.owned_spans
        mp.setattr(G, "owned_spans", lambda s: sorted(real(s) + [("data", TT.TITLE_RUN[0] + 16, 4, "a test's table")],
                                                         key=lambda sp: sp[:2]))   # sorted, as owned_spans' (v0.4:
            #                                                 reserved spans now follow the title's run)
        G._OWNED[0] = None
        with pytest.raises(TT.TitleError, match="a test's table"):
            TT.extended_title(cs)
    G._OWNED[0] = None
    out = TT.extended_title(cs)
    assert TT.titled(out) and len(TT.title_spans(out)) == 3
    with pytest.raises(TT.TitleError, match="already"):
        TT.extended_title(out)


def _hole_uploads(rs) -> tuple[list, list]:
    """Every upload set `rs` names (the boot's char list and BOOT_SCREEN sets, SELECT_SCREEN's, the boot table's other
    screens 0x2600000[2..6], each course table entry's upload list): (its address, the ROOM chars it writes), (its
    address, the palettes 0x85-0xf5 it writes), the sets writing none left out."""
    room = {c for lo, hi in TT.ROOM for c in range(lo, hi)}
    lists = TT.upload_sets(rs)
    chars, pals = [], []
    for a, cs, ps in lists:
        got = {c.first_char + k for c in cs for k in range(c.count)} & room
        if got:
            chars.append((a, got))
        cols = {(p.dst - TL.PALETTE_RAM[0]) // 2 + k >> 4 for p in ps for k in range(len(p.words))}
        if cols & set(range(0x85, 0xF6)):
            pals.append((a, cols & set(range(0x85, 0xF6))))
    return chars, pals


def test_extended_title_writes(romset_zip, disc):  # noqa: F811
    """On the M5b playtest set (tools/desert_city_playtest.py `m5b_set`, the release chain with Phase 11's row):
    `titled` false, then true on `extended_title`'s set; every byte that changes was 0xff (in TITLE_RUN) or is one of
    the two repointed words, the three `title_spans`, which golden.owned_spans owns; TITLE_RUN was 0xff and owned by
    nothing before; `render(out, logo(out))` is `extended_pixels(PICKED)` through the set's palette, pixel for pixel
    (layer 2's map, the select's background, unchanged; layer 0's the copy in TITLE_RUN); the boot loads the word's
    chars into one ROOM run with palette 0x80 + its block, `title_palette`'s, and of every upload set the set names
    only the title's copy writes ROOM chars or palettes 0x85-0xf5 (on the input, none); a second call is a TitleError,
    as is `extended_pixels` on the titled set (its cells are taken)."""
    from dtools.arcade import golden as G
    from dtools.arcade.romset import REGIONS
    from tests.test_release import _m5b
    ref = _m5b(romset_zip, disc)
    assert not TT.titled(ref) and TT.title_spans(ref) == []
    assert _hole_uploads(ref) == ([], [])
    out = TT.extended_title(ref)
    assert TT.titled(out)
    spans = TT.title_spans(out)
    (_, run_at, run_n, _), word, mapw = spans
    assert word[:3] == ("data", TT.BOOT_WORD - TT.DATA_BASE, 4) and mapw[:3] == ("program", TT.MAP_WORD, 4)
    assert run_at == TT.TITLE_RUN[0] and 0 < run_n <= TT.TITLE_RUN[1]
    assert ref.data[run_at:run_at + TT.TITLE_RUN[1]] == b"\xff" * TT.TITLE_RUN[1]
    assert G.owner(ref, "data", *TT.TITLE_RUN) is None
    owned = G.owned_spans(out)
    assert all(sp in owned for sp in spans)
    words = {(r, o + i) for r, o, n, _ in spans[1:] for i in range(n)}
    changed = 0
    for region in REGIONS:
        a, b = (np.frombuffer(getattr(s, region), np.uint8) for s in (ref, out))
        d = np.flatnonzero(a != b)
        changed += d.size
        assert all((region, int(i)) in words or (region == "data" and run_at <= i < run_at + run_n and a[i] == 0xFF)
                   for i in d), region
    assert changed > 5000
    lg, lo = TT.logo(ref), TT.logo(out)
    assert lo.layers[1] == lg.layers[1] and lo.layers[0].map_at == TT.DATA_BASE + TT.TITLE_RUN[0]
    want = TT.palette_rgb(TT.palette_ram(lg))[TT.extended_pixels(ref, TT.PICKED)]
    assert np.array_equal(TT.render(out, lo), want)
    chars, pals = _hole_uploads(out)
    upload = TT._u32(out, TT.BOOT_WORD)
    n = len(TT.new_chars(ref, TT.PICKED))
    assert [a for a, _ in chars] == [upload] and len(chars[0][1]) == n
    (lo_char,) = {c >> 7 for c in chars[0][1]}
    assert pals == [(upload, {0x80 + lo_char})]
    words16, _ = TT.title_palette(lg, np.stack(TT.new_chars(ref, TT.PICKED)))
    assert TT.palette_ram(lo)[16 * (0x80 + lo_char):16 * (0x81 + lo_char)].tolist() == list(words16)
    with pytest.raises(TT.TitleError, match="already"):
        TT.extended_title(out)
    with pytest.raises(TT.TitleError, match="not a transparent cell"):
        TT.extended_pixels(out, TT.PICKED)
