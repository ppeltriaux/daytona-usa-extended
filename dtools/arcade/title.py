"""The attract logo ("DAYTONA USA") and "Extended" drawn into it in the logo's own style. Layout and the runs:
docs/formats/model2-attract-logo.md; discovery log docs/spike/2026-11-release.md, "The title".

The logo is the attract loop's title screen (once per loop: frames 3265-3668 and 6432-6835 on a stock boot), drawn on
the System 24 tilemap chip, no polygons. Its first state (program window `LOGO_STATE`) blits two tile maps with
`0x19304` (`tiles.TileMap`): layer 0, from tile row 7, the black italic letters, the trademark and the four flags
(data `0x0282a270`, 34 x 62); layer 2, from row 0, the white field with the letters' light grey shadow and the
checker bands (`0x0282499e`, 48 x 62; Circuit Select's background is the same map, blitted by 0x226a4c). Their chars
are loaded once at boot by the second boot upload set (data word `0x02700000`): 0x1800+511 (palettes 0x30-0x33, four
copies of one: black, a seven-step grey ramp to white, the flags' red, blue, green and yellow) and 0x1a00+379
(0x34-0x36: the shadow grey and white). Layers 0 and 1 are in the chip's split mode: the letters wipe in from the left
over the static field (word 0x5000 from 0x100 to 0x400), then rest unscrolled.

"Extended" is a word of black letters with the logo's grey anti-aliasing ramp and its 1-2 px grey drop shadow, in the
logo's slant and weight, in cells of layer 0 that hold the map's transparent char and that the HUD does not write
(so it goes into layer 0's map alone; the field and the select's background stay as they are). The letters are drawn
by this module (vector shapes, supersampled): the logo's own letters D, T, N are fused with their neighbours in
"DAYTONA" (one connected run of ink, no column free of it even unslanted), so none can be lifted whole; the shapes
take the logo's measured slant, stem weight and small counters instead. `extended_pixels` gives the logo's palette
indices with the word drawn; `mockup` shows it through the set's own palette.

`extended_title` writes Pascal's pick (PICKED) into a set: the word's chars into boot holes below 0x3000 (ROOM), drawn
with tile bit 14 through one palette per hole block (`title_palette`), loaded by a copy of the boot's second upload
set; that copy and a copy of the layer-0 map holding the word go into the free custom data run TITLE_RUN, and two
words are repointed (the boot's upload word BOOT_WORD, the logo state's map word MAP_WORD). No instruction is added
or changed; `titled` and `title_spans` read it back (golden.owned_spans owns the spans).
"""
from __future__ import annotations

import struct
from typing import NamedTuple

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet, RomSetError
from dtools.arcade.tablebase import DATA_BASE, CourseError, _u32, is_int, is_number
from dtools.arcade.texture import colour_xlat


class TitleError(DecodeError):
    pass


SCREEN_H, SCREEN_W = 384, 496
LOGO_STATE = 0x00222FD0
"""Program window: the logo screen's first state (0x222fb8[0]): one blit to layer 2, then LOGO_SITES' two."""
LOGO_SITES = (0x00022FE4, 0x00022FF8)
"""Program offsets of the logo state's two blits, each `lda <tile RAM>,g1; lda <map>,g0; call 0x19304`: layer 0's
letters, then layer 2's field (the map words at +4 and +12)."""
BLIT = 0x00019304               # (map, tile RAM): copies a tile map (select.BLIT)
_LDA_G1, _LDA_G0 = 0x8C883000, 0x8C803000
_WINDOW = 0x200000               # program offset + this: the window address the code runs at
LOGO_FRAMES = ((3265, 3668), (6432, 6835))
"""The frames the logo shows on a stock boot with no coin (MAME 0.289; the M5b set's are the same)."""
LOGO_LAYER_ORDER = (2, 0)        # drawn back to front (layer 2 opaque under layer 0)

ROOM = ((0x04D9, 0x0500), (0x0578, 0x0580), (0x063C, 0x0680), (0x0698, 0x0700), (0x0718, 0x0780), (0x07B0, 0x0800),
        (0x09CC, 0x0A00), (0x0B68, 0x0B80), (0x0BD4, 0x0C00), (0x0EC1, 0x0F00), (0x10C5, 0x1100), (0x1178, 0x1180),
        (0x1349, 0x1380), (0x14CC, 0x1500), (0x17C9, 0x1800), (0x1D0D, 0x1D80), (0x215F, 0x2180), (0x21E4, 0x2200),
        (0x2291, 0x2300), (0x2359, 0x2380), (0x248E, 0x2500), (0x25A1, 0x2600), (0x2DEF, 0x2E00), (0x2FB2, 0x3000))
"""The chars the title may take (lo, end): the holes the boot uploads leave below 0x3000 (no later upload reaches below
0x3000: screens and courses load from 0x3000, the select's generated chars from 0x3b00), less the three holes of
blocks 2-4, whose bit-14 palettes (0x82-0x84) the boot writes. A tile with bit 14 set draws its char with palette
0x80 + (char >> 7), which nothing writes or shows (a whole M5b session: boot, the logo, the select, Beginner's race to
its time-over, the attract again; no char of these holes written, no palette 0x85-0xf5 written, none on screen)."""
BUDGET_CHARS = sum(hi - lo for lo, hi in ROOM)
"""The most new chars the title may load: all of ROOM (1,445). Loaded at boot by a copy of the boot upload set, every
hole char (1,583, ROOM and blocks 2-4) kept the frame counters 0x5010b0 / 0x540058 on every frame to 1200 and
Beginner from the select call for call with the M5b set (scratch runs), so the room, not the boot's time, is the
limit."""

WORD = "Extended"
LOGO_SLANT = 0.172
"""The logo's slant: px to the right per px up (the D's stem: 11 px over 64 rows)."""
SHADOW_DROP = (1, 2)            # the shadow: the ink (half covered or more) dropped by these px, grey where no ink
SUPERSAMPLE = 8
MOCKUP_GAP = 8                   # px (before the 2x) between the original and the variant
_GAP = 6                         # design units between letters (cap height 100)


class Variant(NamedTuple):
    box: tuple[int, int, int, int]      # screen x0, y0, x1, y1 (end exclusive): the word and its shadow centred in it
    size: int                           # cap height, px
    case: str                           # "upper" (EXTENDED) or "title" (Extended)
    slant: float                        # px right per px up


VARIANTS = {
    "v1": Variant((8, 277, 248, 305), 26, "upper", LOGO_SLANT),
    "v2": Variant((4, 269, 250, 305), 34, "title", LOGO_SLANT),
    "v3": Variant((8, 229, 248, 255), 24, "title", 0.30),
    "v4": Variant((4, 273, 250, 305), 30, "upper", 0.30),
}
"""USA's baseline is screen row 303 (its ink ends on 302, its shadow on 305); a box as tall as the cap height and the
shadow puts the word's baseline on its bottom less the shadow. v1 EXTENDED, small, on USA's baseline, the logo's
slant; v2 Extended as large as the room left of USA allows, on USA's baseline; v3 Extended small and steeper, its cap
line on USA's (row 229), under DAYTONA's left half; v4 EXTENDED larger and steeper on USA's baseline. Since Task 3's
review v1, v2 and v4 are refused (`_drawn`): their shadow's last row reaches tile row 38, the HUD's (HUD_ROWS); they
stay as the record of what Pascal chose from."""

PICKED = "v3"
"""Pascal's pick at Task 3's STOP (13 Nov 2026): "v3 Extended, under DAYTONA". What `extended_title` writes."""

HUD_ROWS, HUD_COLS = (38, 41), (12, 50)
"""The layer-0 cells (tile rows, tile columns; end exclusive) the HUD writes while the logo shows: the INSERT COIN(S)
band (at frame 3450 tile RAM holds the HUD's tiles 0x8280-0x8285 there in place of the map's transparent tile; the
text blinks). The HUD's other rows (41-46: CREDIT, the copyright, the SEGA logo) lie below the logo map, which ends
on row 40. `_drawn` refuses a word cell here: the HUD would overwrite it."""

TITLE_RUN = (0x01F20000, 0x000D0000)      # to the Car Select screen's graphics (v0.4: romset.RESERVED; 0xE0000 before)
"""(data image offset, length): the bytes the title may take, the custom data run after the select's reserved data
(romset.CUSTOM_FREE["data"][1]), outside build.ID_DATA_FREE (the converter, the views and the mirrors allocate
there, never here). 0xff and owned by nothing on the M5b and release chains before the title."""
BOOT_WORD = C.BOOT_SCREEN[1]
"""The data word (i960 address) naming the boot's second upload set (0x02700000 -> 0x02700008; the select's data ends
right before it): repointed to the title's copy of that set."""
MAP_WORD = LOGO_SITES[0] + 12
"""Program offset of the 32-bit displacement word (0x22ff0) of the two-word `lda 0x282a270,g0` at 0x22fec, which holds
layer 0's map address: repointed to the title's copy of the map. The opcode word stays as it is."""


class LogoLayer(NamedTuple):
    layer: int                  # tile layer 0-3
    row: int                    # the map's first tile row
    col: int                    # its first tile column
    map_at: int                 # i960 address of the tile map
    map: TL.TileMap


class Logo(NamedTuple):
    kind: str                                   # "tiles"
    layers: tuple[LogoLayer, ...]               # in the game's blit order: layer 0's letters, layer 2's field
    chars: dict[int, np.ndarray]                # char -> (8, 8) pens, every char the maps draw, as the boot loads it
    palette: dict[int, tuple[int, ...]]         # colour -> its 16 palette words, every colour the maps draw
    frames: tuple[tuple[int, int], ...]


def _check_rs(rs) -> None:
    if not isinstance(rs, RomSet):
        raise TitleError(f"{type(rs).__name__} is not a RomSet")


def _check_logo(lg) -> None:
    """TitleError unless `lg` is a Logo as `logo` builds it: kind "tiles", its layers LogoLayers with a TileMap that
    fits the layer, every char its maps draw in `chars` as (8, 8) pens, every palette 16 words."""
    if not isinstance(lg, Logo):
        raise TitleError(f"{type(lg).__name__} is not a Logo")
    if lg.kind != "tiles" or not isinstance(lg.layers, tuple) or not lg.layers or \
            not isinstance(lg.chars, dict) or not isinstance(lg.palette, dict):
        raise TitleError("the Logo is not a tile logo (kind, layers, chars, palette)")
    for g in lg.layers:
        if not isinstance(g, LogoLayer) or not isinstance(g.map, TL.TileMap) or \
                not all(is_int(v) for v in (g.layer, g.row, g.col, g.map.rows, g.map.cols)) or \
                not 0 <= g.layer < 4 or not 0 <= g.row <= TL.TILE_LAYER - g.map.rows or \
                not 0 <= g.col <= TL.TILE_LAYER - g.map.cols or len(g.map.entries) != g.map.rows * g.map.cols:
            raise TitleError(f"the Logo's layer {g!r:.60} is not a LogoLayer whose map fits its tile layer")
        missing = {t & TL.CHAR_MASK for t in g.map.tiles()} - set(lg.chars)
        if missing:
            raise TitleError(f"the Logo's layer {g.layer} draws chars it does not hold ({len(missing)})")
    for px in lg.chars.values():
        if not isinstance(px, np.ndarray) or px.shape != (8, 8):
            raise TitleError("a Logo char is not (8, 8) pens")
    for words in lg.palette.values():
        if not isinstance(words, tuple) or len(words) != 16 or not all(is_int(w) and 0 <= w <= 0xFFFF for w in words):
            raise TitleError("a Logo palette is not 16 u16 words")


def _site(rs: RomSet, at: int) -> tuple[int, int]:
    """(tile RAM address, map address) of the blit at program offset `at`. TitleError when the three instructions
    are not lda g1, lda g0, call 0x19304."""
    w = struct.unpack_from("<5I", rs.program, at)
    target = (_WINDOW + at + 16 + ((w[4] & 0xFFFFFC) - ((w[4] & 0x800000) << 1))) & 0xFFFFFFFF
    if w[0] != _LDA_G1 or w[2] != _LDA_G0 or w[4] >> 24 != 0x09 or target != BLIT:
        raise TitleError(f"the logo blit at program {at:#x} is not `lda <tile RAM>,g1; lda <map>,g0; call "
                         f"{BLIT:#x}` (words {' '.join(f'{x:08x}' for x in w)})")
    return w[1], w[3]


def _boot(rs: RomSet) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    """What the boot leaves in char and palette RAM: char -> i960 address of its 32 bytes (the last upload wins), the
    palette words and which entries were written."""
    src: dict[int, int] = {}
    pal = np.zeros(0x2000, np.uint16)
    written = np.zeros(0x2000, bool)
    for c in TL.char_list(rs, C.BOOT_CHAR_LIST):
        for k in range(c.count):
            src[c.first_char + k] = c.src + 4 + TL.CHAR_SIZE * k
    for slot in C.BOOT_SCREEN:
        u = TL.upload_set(rs, _u32(rs, slot), segments=False)
        for c in u.chars:
            for k in range(c.count):
                src[c.first_char + k] = c.src + 4 + TL.CHAR_SIZE * k
        for p in u.palette:
            e = (p.dst - TL.PALETTE_RAM[0]) // 2
            pal[e:e + len(p.words)] = p.words
            written[e:e + len(p.words)] = True
    return src, pal, written


def logo(rs: RomSet) -> Logo:
    """The attract logo of `rs` (the stock set, M5b's or a release set): its two maps from the logo state's blits,
    the chars and palettes the boot loads for them. TitleError for a set whose logo state is not the expected blits,
    a map that does not decode or fit its layer, or a char or palette the boot does not load."""
    _check_rs(rs)
    layers = []
    try:
        for at in LOGO_SITES:
            dst, at_map = _site(rs, at)
            off = dst - TL.TILE_RAM[0]
            if not 0 <= off < 0x8000 or off % 2:
                raise TitleError(f"the logo blit at {at:#x} writes {dst:#x}, not a tile layer")
            layer, rest = divmod(off, 0x2000)
            row, col = rest // 0x80, rest % 0x80 // 2
            m = TL.tile_map(rs, at_map)
            if row + m.rows > TL.TILE_LAYER or col + m.cols > TL.TILE_LAYER:
                raise TitleError(f"the logo map at {at_map:#x} ({m.rows} x {m.cols}) runs off layer {layer}")
            layers.append(LogoLayer(layer, row, col, at_map, m))
        src, pal, written = _boot(rs)
        chars, palette = {}, {}
        for g in layers:
            for t in set(g.map.tiles()):
                ch, colour = t & TL.CHAR_MASK, (t >> 7) & 0xFF
                if ch not in src:
                    raise TitleError(f"logo char {ch:#x} (map {g.map_at:#x}) is not loaded at boot")
                if not written[16 * colour:16 * colour + 16].all():
                    raise TitleError(f"logo palette {colour:#x} (map {g.map_at:#x}) is not written at boot")
                if ch not in chars:
                    chars[ch] = TL.decode_chars(rs.read_i960(src[ch], TL.CHAR_SIZE))[0]
                palette[colour] = tuple(int(w) for w in pal[16 * colour:16 * colour + 16])
    except (CourseError, RomSetError) as e:
        raise TitleError(f"the logo does not decode: {e}") from None
    return Logo("tiles", tuple(layers), dict(sorted(chars.items())), dict(sorted(palette.items())), LOGO_FRAMES)


def palette_ram(lg: Logo) -> np.ndarray:
    """The tile palette RAM (0x2000 u16 entries) holding the logo's colours, the rest 0. TitleError for `lg` not a
    Logo (`logo`'s)."""
    _check_logo(lg)
    out = np.zeros(0x2000, np.uint16)
    for colour, words in lg.palette.items():
        out[16 * colour:16 * colour + 16] = words
    return out


def palette_rgb(words) -> np.ndarray:
    """(n, 3) uint8: tile palette words as MAME 0.289 shows them (model2.cpp palette_w: each 5-bit component through
    the colour translation table's tile entry, `texture.colour_xlat`, then the gamma table of model2_v.cpp).
    TitleError for `words` that are not integers 0-0xffff (bool refused)."""
    try:
        w = np.asarray(words)
    except (TypeError, ValueError) as e:
        raise TitleError(f"palette words {words!r:.60}: {e}") from None
    if w.dtype == bool or not np.issubdtype(w.dtype, np.integer) or (w.size and (w.min() < 0 or w.max() > 0xFFFF)):
        raise TitleError(f"palette words {words!r:.60} are not u16 integers")
    w = w.astype(np.int64)
    xlat = colour_xlat(True).astype(np.float64)

    def comp(c):
        v = xlat[0x40 + (c << 8)]
        return np.maximum((v - 64.0) * 255.0 / 191.0, 0).astype(np.uint8)
    return np.stack([comp(w & 31), comp((w >> 5) & 31), comp((w >> 10) & 31)], -1)


def _layer_pixels(lg: Logo, g: LogoLayer) -> np.ndarray:
    """(SCREEN_H, SCREEN_W) palette indices of one logo layer at rest (no scroll), 0 where transparent."""
    out = np.zeros((TL.TILE_LAYER * 8, TL.TILE_LAYER * 8), np.uint16)
    tiles = np.array(g.map.tiles(), np.int64).reshape(g.map.rows, g.map.cols)
    px = np.stack([lg.chars[t & TL.CHAR_MASK] for t in tiles.ravel()]).reshape(g.map.rows, g.map.cols, 8, 8)
    pens = ((tiles >> 7) & 0xFF)[:, :, None, None] * 16 + px
    pens = np.where(px == 0, 0, pens).transpose(0, 2, 1, 3).reshape(8 * g.map.rows, 8 * g.map.cols)
    out[8 * g.row:8 * (g.row + g.map.rows), 8 * g.col:8 * (g.col + g.map.cols)] = pens
    return out[:SCREEN_H, :SCREEN_W]


def _layer(lg: Logo, layer: int) -> LogoLayer:
    for g in lg.layers:
        if g.layer == layer:
            return g
    raise TitleError(f"the logo has no layer {layer}")


def logo_pixels(rs: RomSet, lg: Logo) -> np.ndarray:
    """(SCREEN_H, SCREEN_W) uint16 palette indices of the logo as the screen shows it at rest: layer 2, then layer 0
    over it where its pen is not 0. TitleError for `rs` not a RomSet or `lg` not a Logo."""
    _check_rs(rs)
    _check_logo(lg)
    out = np.zeros((SCREEN_H, SCREEN_W), np.uint16)
    for layer in LOGO_LAYER_ORDER:
        px = _layer_pixels(lg, _layer(lg, layer))
        out = np.where(px != 0, px, out)
    return out


def render(rs: RomSet, lg: Logo) -> np.ndarray:
    """(SCREEN_H, SCREEN_W, 3) uint8: the logo as on screen (the logo's two layers; no HUD, no polygons). TitleError
    for `rs` not a RomSet or `lg` not a Logo."""
    _check_rs(rs)
    _check_logo(lg)
    return palette_rgb(palette_ram(lg))[logo_pixels(rs, lg)]


def blank_tile(rs: RomSet, lg: Logo) -> int:
    """Layer 0's transparent tile: the map tile drawn most whose char has no pixel. TitleError when there is none, and
    for `rs` not a RomSet or `lg` not a Logo."""
    _check_rs(rs)
    _check_logo(lg)
    g = _layer(lg, 0)
    tiles, counts = np.unique(np.array(g.map.tiles()), return_counts=True)
    for t in tiles[np.argsort(-counts, kind="stable")]:
        if not lg.chars[int(t) & TL.CHAR_MASK].any():
            return int(t)
    raise TitleError("layer 0's logo map has no transparent tile")


# --- the letters -------------------------------------------------------------------------------------------------
#
# Each glyph: (advance, shapes), in design units: cap height 100, y 0 at the cap line and 100 on the baseline, x from
# the glyph's left. Shapes ("+" add, "-" cut, in order): ("p", points) a polygon, ("e", cx, cy, rx, ry) an ellipse.
# The logo's proportions: stems about 0.39 of the cap height, horizontals 0.27, counters 0.12-0.25, wide letters.

def _r(x0, y0, x1, y1):
    return ("p", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


_X_UP = 74                       # lower-case x-height
GLYPHS = {
    "E": (80, [("+", _r(0, 0, 80, 26)), ("+", _r(0, 0, 40, 100)), ("+", _r(0, 38, 72, 62)),
               ("+", _r(0, 74, 82, 100))]),
    "X": (94, [("+", ("p", ((0, 0), (40, 0), (94, 100), (54, 100)))),
               ("+", ("p", ((54, 0), (94, 0), (40, 100), (0, 100))))]),
    "T": (84, [("+", _r(0, 0, 84, 27)), ("+", _r(22, 0, 62, 100))]),
    "N": (92, [("+", _r(0, 0, 34, 100)), ("+", _r(58, 0, 92, 100)),
               ("+", ("p", ((0, 0), (36, 0), (92, 100), (56, 100))))]),
    "D": (90, [("+", _r(0, 0, 50, 100)), ("+", ("e", 50, 50, 40, 50)),
               ("-", _r(40, 26, 50, 74)), ("-", ("e", 50, 50, 13, 24))]),
    "x": (82, [("+", ("p", ((0, 26), (36, 26), (82, 100), (46, 100)))),
               ("+", ("p", ((46, 26), (82, 26), (36, 100), (0, 100))))]),
    "t": (58, [("+", _r(10, 6, 44, 100)), ("+", _r(0, 26, 58, 47))]),
    "e": (82, [("+", ("e", 41, 63, 41, 37)), ("-", ("e", 42, 52, 15, 9)),
               ("-", ("p", ((40, 70), (90, 70), (90, 83), (44, 83))))]),
    "n": (80, [("+", _r(0, 26, 33, 100)), ("+", ("e", 48, 58, 32, 32)), ("+", _r(48, 58, 80, 100)),
               ("+", _r(0, 26, 48, 58)), ("-", ("e", 40, 62, 8, 8)), ("-", _r(33, 62, 48, 100))]),
    "d": (84, [("+", ("e", 38, 63, 38, 37)), ("+", _r(50, 0, 84, 100)), ("-", ("e", 34, 63, 12, 19))]),
}


def _inside(shape, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if shape[0] == "e":
        _, cx, cy, rx, ry = shape
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    pts = shape[1]
    inside = np.zeros(x.shape, bool)
    for (xa, ya), (xb, yb) in zip(pts, pts[1:] + pts[:1]):
        if ya == yb:
            continue
        cross = (ya > y) != (yb > y)
        xi = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside ^= cross & (x < xi)
    return inside


def word_text(case: str) -> str:
    if case == "upper":
        return WORD.upper()
    if case == "title":
        return WORD[0].upper() + WORD[1:].lower()
    raise TitleError(f"case {case!r}: expected 'upper' or 'title'")


def coverage(v: Variant) -> np.ndarray:
    """(y1 - y0, x1 - x0) float: the word's ink coverage in `v.box` (0-1, SUPERSAMPLE x SUPERSAMPLE samples a px),
    centred in the box with room for its shadow below. TitleError when it does not fit, and for `v` not a Variant of
    ints (box, size), a case and a number (slant)."""
    if not isinstance(v, Variant) or not isinstance(v.box, tuple) or len(v.box) != 4 or \
            not all(is_int(a) for a in (*v.box, v.size)) or not isinstance(v.case, str) or not is_number(v.slant):
        raise TitleError(f"{v!r:.80} is not a Variant (box of four ints, int size, case, number slant)")
    x0, y0, x1, y1 = v.box
    if not (0 <= x0 < x1 <= SCREEN_W and 0 <= y0 < y1 <= SCREEN_H) or not 8 <= v.size <= 100 or \
            not 0 <= v.slant <= 1:
        raise TitleError(f"variant {v!r}: box, size or slant out of range")
    text = word_text(v.case)
    scale = v.size / 100.0
    advances = [GLYPHS[ch][0] for ch in text]
    width_d = sum(advances) + _GAP * (len(text) - 1)
    width = width_d * scale + v.slant * v.size
    h, w = y1 - y0, x1 - x0
    drop = max(SHADOW_DROP)
    if width > w or v.size + drop > h:
        raise TitleError(f"variant {v!r}: the word ({width:.0f} x {v.size + drop} px) does not fit its box ({w} x {h})")
    left = (w - width) / 2.0                     # box-relative
    base = (h - v.size - drop) // 2 + v.size     # the baseline: box-relative row boundary
    ss = SUPERSAMPLE
    sy = (np.arange(h * ss) + 0.5) / ss
    sx = (np.arange(w * ss) + 0.5) / ss
    yy, xx = np.meshgrid(sy, sx, indexing="ij")
    yd = 100.0 - (base - yy) / scale
    xs = (xx - left - v.slant * (base - yy)) / scale
    ink = np.zeros(yy.shape, bool)
    at = 0.0
    for ch, adv in zip(text, advances):
        xd = xs - at
        near = (xd > -1) & (xd < adv + 1) & (yd > -1) & (yd < 101)
        g = np.zeros(yy.shape, bool)
        for op, shape in GLYPHS[ch][1]:
            hit = near & _inside(shape, xd, yd)
            g = g | hit if op == "+" else g & ~hit
        ink |= g
        at += adv + _GAP
    return ink.reshape(h, ss, w, ss).mean(axis=(1, 3))


class _Style(NamedTuple):
    ramp: tuple[tuple[int, float], ...]     # (palette index, ink coverage), black first
    shadow: int
    field: int


def _style(lg: Logo) -> _Style:
    """The logo's own pens: layer 0's greys (its black and anti-aliasing ramp, by brightness against the field), the
    field's white and the shadow's grey (layer 2's pen most drawn under layer 0's black)."""
    rgb = palette_rgb(palette_ram(lg))
    l0 = _layer_pixels(lg, _layer(lg, 0))
    l2 = _layer_pixels(lg, _layer(lg, 2))
    vals, counts = np.unique(l2, return_counts=True)
    field = int(vals[np.argmax(counts)])
    white = float(rgb[field].mean())
    used0 = [int(p) for p in np.unique(l0) if p]
    greys = [p for p in used0 if rgb[p][0] == rgb[p][1] == rgb[p][2] and rgb[p][0] < white]
    if not greys:
        raise TitleError("layer 0 of the logo has no grey pens")
    black = min(greys, key=lambda p: (int(rgb[p][0]), p))
    under = l2[(l0 == black)]
    vals, counts = np.unique(under[under != field], return_counts=True)
    if not len(vals):
        raise TitleError("the logo's letters have no shadow on layer 2")
    shadow = int(vals[np.argmax(counts)])
    ramp = tuple(sorted(((p, 1.0 - float(rgb[p][0]) / white) for p in greys), key=lambda t: (-t[1], t[0])))
    return _Style(ramp, shadow, field)


def _word(lg: Logo, v: Variant) -> np.ndarray:
    """The word's layer-0 pixels in the full screen (palette indices, 0 transparent): ink by the ramp, shadow."""
    st = _style(lg)
    cov = coverage(v)
    pens = np.array([0] + [p for p, _ in st.ramp], np.int64)
    levels = np.array([0.0] + [c for _, c in st.ramp])
    ink = pens[np.argmin(np.abs(cov[..., None] - levels), axis=-1)]
    solid = cov >= 0.5
    shade = np.zeros_like(solid)
    for d in SHADOW_DROP:
        shade[d:] |= solid[:-d]
    word = np.where(ink != 0, ink, np.where(shade, st.shadow, 0))
    x0, y0, x1, y1 = v.box
    out = np.zeros((SCREEN_H, SCREEN_W), np.uint16)
    out[y0:y1, x0:x1] = word
    return out


def _variant(name) -> Variant:
    if not isinstance(name, str) or name not in VARIANTS:
        raise TitleError(f"variant {name!r}: expected one of {', '.join(VARIANTS)}")
    return VARIANTS[name]


def _drawn(rs: RomSet, name) -> tuple[Logo, np.ndarray, np.ndarray]:
    """(the logo, its pixels, the word's layer-0 pixels) of variant `name` on `rs`. TitleError for a word cell off
    layer 0's map, on a cell whose map tile is not the transparent one (a word already there, or the logo's ink), or
    on a HUD cell (HUD_ROWS x HUD_COLS)."""
    v = _variant(name)
    lg = logo(rs)
    base = logo_pixels(rs, lg)
    word = _word(lg, v)
    g0 = _layer(lg, 0)
    blank = blank_tile(rs, lg)
    tiles = np.array(g0.map.tiles(), np.int64).reshape(g0.map.rows, g0.map.cols)
    for r, c in sorted({(y // 8, x // 8) for y, x in zip(*np.nonzero(word))}):
        mr, mc = r - g0.row, c - g0.col
        if HUD_ROWS[0] <= r < HUD_ROWS[1] and HUD_COLS[0] <= c < HUD_COLS[1]:
            raise TitleError(f"variant {name}: tile ({r}, {c}) is the HUD's (INSERT COIN(S), rows {HUD_ROWS[0]}-"
                             f"{HUD_ROWS[1] - 1}, columns {HUD_COLS[0]}-{HUD_COLS[1] - 1}), which overwrites it")
        if not (0 <= mr < g0.map.rows and 0 <= mc < g0.map.cols) or tiles[mr, mc] != blank:
            raise TitleError(f"variant {name}: tile ({r}, {c}) is not a transparent cell of layer 0's logo map")
    return lg, base, word


def extended_pixels(rs: RomSet, variant: str) -> np.ndarray:
    """(SCREEN_H, SCREEN_W) uint16: the logo's palette indices (`logo_pixels`) with "Extended" drawn as `variant`
    (VARIANTS) has it: only indices the logo shows, only inside the variant's box, only in cells where layer 0 holds
    its transparent char. Reads nothing but `rs`. TitleError for an unknown variant or a word that does not fit."""
    lg, base, word = _drawn(rs, variant)
    return np.where(word != 0, word, base).astype(np.uint16)


def _cells(word: np.ndarray) -> tuple[list[tuple[tuple[int, int], int]], list[np.ndarray]]:
    """The word's cells in row-major order, each with the index of its char, and the distinct chars (8, 8 palette
    indices, 0 transparent) in the order they first show."""
    cells, chars, index = [], [], {}
    for r, c in sorted({(y // 8, x // 8) for y, x in zip(*np.nonzero(word))}):
        cell = word[8 * r:8 * r + 8, 8 * c:8 * c + 8]
        key = cell.tobytes()
        if key not in index:
            index[key] = len(chars)
            chars.append(cell.copy())
        cells.append(((r, c), index[key]))
    return cells, chars


def new_chars(rs: RomSet, variant: str) -> list[np.ndarray]:
    """The distinct layer-0 chars the word needs, as (8, 8) screen palette indices (0 transparent), in the order of
    the cells (row-major): what `extended_title` loads into ROOM, after `title_palette` puts their colours in one
    palette. Within BUDGET_CHARS."""
    _, _, word = _drawn(rs, variant)
    out = _cells(word)[1]
    if len(out) > BUDGET_CHARS:
        raise TitleError(f"variant {variant}: {len(out)} new chars, past the budget of {BUDGET_CHARS}")
    return out


def mockup(rs: RomSet, variant: str) -> np.ndarray:
    """(2 SCREEN_H, 4 SCREEN_W + 2 MOCKUP_GAP, 3) uint8: the logo (`render`) and the variant (`extended_pixels`
    through the set's palette) side by side, each at 2x (nearest), a dark gap between."""
    lg = logo(rs)
    left = render(rs, lg)
    right = palette_rgb(palette_ram(lg))[extended_pixels(rs, variant)]
    gap = np.full((SCREEN_H, MOCKUP_GAP, 3), 32, np.uint8)
    return np.hstack([left, gap, right]).repeat(2, 0).repeat(2, 1)


# --- the title written (Task 4) -----------------------------------------------------------------------------------

def title_palette(lg: Logo, chars) -> tuple[tuple[int, ...], np.ndarray]:
    """The word's chars in one 16-pen palette: (its 16 words, the chars as (n, 8, 8) uint8 pens). `chars` are screen
    palette indices (`new_chars`) from two of the logo's palettes, layer 0's (black and the grey ramp, in 0x30-0x33)
    and layer 2's shadow grey (0x34-0x36), but a char draws through one palette, its hole block's (tile bit 14:
    0x80 + block). So the palette is the logo palette holding most of the word's indices (layer 0's), its pens kept;
    each index of another palette goes to a pen of it already holding the same word, else to the highest pen the word
    does not use (the shadow grey: pen 15 when no ramp step matches), that pen's word set. Pen 0 stays transparent.
    Every pixel then shows the palette word its index showed. TitleError for `lg` not a Logo, `chars` not a non-empty
    (n, 8, 8) stack of indices 0-0x1fff with some ink, and for more colours than 15 pens."""
    _check_logo(lg)
    try:
        a = np.asarray(chars)
    except (TypeError, ValueError) as e:
        raise TitleError(f"chars: {e}") from None
    if a.ndim != 3 or a.shape[1:] != (8, 8) or not len(a) or a.dtype == bool or \
            not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() >= 0x2000:
        raise TitleError(f"chars of shape {a.shape} ({a.dtype}): expected (n, 8, 8) palette indices 0-0x1fff")
    ram = palette_ram(lg)
    used = [int(i) for i in np.unique(a) if i]
    if not used:
        raise TitleError("the chars have no ink")
    colours = [i >> 4 for i in used]
    base = max(sorted(set(colours)), key=colours.count)
    words = [int(w) for w in ram[16 * base:16 * base + 16]]
    pen = {i: i & 15 for i in used if i >> 4 == base}
    taken = set(pen.values())
    for i in used:
        if i in pen:
            continue
        same = [p for p in range(1, 16) if words[p] == int(ram[i])]
        free = [p for p in range(15, 0, -1) if p not in taken]
        if not same and not free:
            raise TitleError(f"the chars' {len(used)} colours do not fit one palette's 15 pens")
        p = same[0] if same else free[0]
        words[p] = int(ram[i])
        pen[i] = p
        taken.add(p)
    lut = np.zeros(0x2000, np.uint8)
    for i, p in pen.items():
        lut[i] = p
    return tuple(words), lut[a]


def _room_runs(n: int) -> list[tuple[int, int]]:
    """(first char, count) of the ROOM runs `n` chars go into: the first run (ROOM's order) that holds them all, else
    ROOM's runs in order until they are placed."""
    for lo, hi in ROOM:
        if hi - lo >= n:
            return [(lo, n)]
    out = []
    for lo, hi in ROOM:
        k = min(n, hi - lo)
        out.append((lo, k))
        n -= k
        if not n:
            return out
    raise TitleError(f"{n} chars more than ROOM holds ({BUDGET_CHARS})")


def upload_sets(rs: RomSet) -> list[tuple[int, list, list]]:
    """Every upload set `rs` names, as (its address, its char runs, its palette runs): the boot's char list (no
    palettes), the BOOT_SCREEN and SELECT_SCREEN sets, the boot table's other screens 0x2600000[2..6] and each course
    table entry's upload list. Decode errors (CourseError, RomSetError) pass through."""
    out = [(C.BOOT_CHAR_LIST, TL.char_list(rs, C.BOOT_CHAR_LIST), [])]
    addrs = {_u32(rs, s): False for s in C.BOOT_SCREEN + C.SELECT_SCREEN}
    addrs.update({_u32(rs, 0x2600000 + 4 * k): False for k in range(2, 7)})
    addrs.update({DATA_BASE + e.upload_list: True for e in C.course_table(rs)})
    for a, segments in addrs.items():
        u = TL.upload_set(rs, a, segments)
        out.append((a, list(u.chars), list(u.palette)))
    return out


def _uploaded(rs: RomSet) -> tuple[set[int], set[int]]:
    """The chars and palette entries every upload set `rs` names writes (`upload_sets`)."""
    chars, pal = set(), set()
    for _, cs, ps in upload_sets(rs):
        for c in cs:
            chars.update(range(c.first_char, c.first_char + c.count))
        for p in ps:
            e = (p.dst - TL.PALETTE_RAM[0]) // 2
            pal.update(range(e, e + len(p.words)))
    return chars, pal


def extended_title(rs: RomSet, variant: str = PICKED) -> RomSet:
    """`rs` with "Extended" (VARIANTS[variant], Pascal's pick by default) in its attract logo, as `extended_pixels`
    draws it on `rs` (whose word cells must still be blank), written with no instruction added:

    - the word's distinct chars (`new_chars`) in ROOM (`_room_runs`), their pens in one palette (`title_palette`), that
      palette at 0x80 + block for each hole block they take (tile bit 14);
    - in TITLE_RUN, from its start: a copy of layer 0's logo map with the word's cells `0x4000 | char`, the char blocks,
      then a copy of the boot's second upload set (its char list plus the new chars' pairs, its palette list plus the
      new palettes) whose pairs, runs and the set itself follow;
    - BOOT_WORD names the copied set and MAP_WORD the copied map (layer 2's map, also the select's background, is not
      touched).

    Every byte that changes was 0xff (TITLE_RUN) or is one of the two words (`title_spans`). TitleError for `rs` not a
    custom RomSet (RomSet.as_custom: TITLE_RUN is its data area), a set already titled (`titled`), an unknown variant
    or one `_drawn` refuses, TITLE_RUN's bytes not 0xff or owned (golden.owned_spans), the chosen chars or palettes
    uploaded by any set the ROM names (`upload_sets`), a boot upload set or map that does not decode. The two words are
    pointer slots golden's course decode already owns (BOOT_WORD a "screen upload pointer" of every course, read like
    the select's own repointed SELECT_UPLOAD_WORD; MAP_WORD the logo map's address `logo` decodes): the code checks that they
    decode (the boot word an upload set, the map word layer 0's map), not that they are free."""
    from dtools.arcade import golden as G         # golden imports this module (owned_spans: title_spans)
    _check_rs(rs)
    if titled(rs):
        raise TitleError("the set already carries the title (its boot upload word names TITLE_RUN)")
    if not rs.custom:
        raise TitleError("the set is not a custom one (RomSet.as_custom): TITLE_RUN lies in the custom data area")
    lg, _, word = _drawn(rs, variant)
    cells, chars = _cells(word)
    runs = _room_runs(len(chars))
    nums = [c for lo, k in runs for c in range(lo, lo + k)]
    colours = sorted({0x80 + (c >> 7) for c in nums})
    try:
        busy_chars, busy_pal = _uploaded(rs)
        boot = TL.upload_set(rs, _u32(rs, BOOT_WORD), segments=False)
    except (CourseError, RomSetError) as e:
        raise TitleError(f"the boot or select uploads do not decode: {e}") from None
    taken = sorted(busy_chars.intersection(nums))
    if taken:
        raise TitleError(f"chars {taken[0]:#x}.. ({len(taken)}) of the title's room are uploaded by the boot or the "
                         f"select")
    if any(16 * col + p in busy_pal for col in colours for p in range(16)):
        raise TitleError(f"a palette of {', '.join(f'{c:#x}' for c in colours)} is uploaded by the boot or the select")
    words, pens = title_palette(lg, np.stack(chars))
    g0 = _layer(lg, 0)
    tiles = list(g0.map.tiles())
    for (r, c), k in cells:
        tiles[(r - g0.row) * g0.map.cols + (c - g0.col)] = 0x4000 | nums[k]
    at = TITLE_RUN[0]
    try:
        blob = bytearray(TL.encode_tile_map(TL.TileMap(0, True, g0.map.rows, g0.map.cols,
                                                       tuple(t - 0x10000 if t & 0x8000 else t for t in tiles))))
        pairs = [(u.src, u.dst) for u in boot.chars]
        i = 0
        for lo, k in runs:
            pairs.append((DATA_BASE + at + len(blob), TL.CHAR_RAM[0] + TL.CHAR_SIZE * lo))
            blob += TL.encode_char_block(TL.encode_chars(pens[i:i + k]))
            i += k
        char_list = DATA_BASE + at + len(blob)
        blob += TL.encode_char_list(pairs)
        palette_list = DATA_BASE + at + len(blob)
        blob += TL.encode_palette_list(list(boot.palette) + [TL.PaletteUpload(TL.PALETTE_RAM[0] + 32 * col, words)
                                                              for col in colours])
        upload = DATA_BASE + at + len(blob)
        blob += TL.encode_upload_set(char_list, palette_list)
    except CourseError as e:
        raise TitleError(f"the title does not encode: {e}") from None
    if len(blob) > TITLE_RUN[1] or rs.data[at:at + len(blob)] != b"\xff" * len(blob):
        raise TitleError(f"data {at:#x}+{len(blob):#x} (the title) is not free: not all 0xff")
    who = G.owner(rs, "data", at, len(blob))
    if who is not None:
        raise TitleError(f"data {at:#x}+{len(blob):#x} (the title) is owned by {who}")
    try:
        return rs.with_patches({("data", at): bytes(blob),
                                ("data", BOOT_WORD - DATA_BASE): struct.pack("<I", upload),
                                ("program", MAP_WORD): struct.pack("<I", DATA_BASE + at)})
    except RomSetError as e:
        raise TitleError(f"the title does not fit the set: {e}") from None


def titled(rs: RomSet) -> bool:
    """Whether `rs` carries the title (`extended_title`): its boot upload word (BOOT_WORD) names an upload set in
    TITLE_RUN. False for a set whose data image does not reach the word. TitleError for `rs` not a RomSet."""
    _check_rs(rs)
    off = BOOT_WORD - DATA_BASE
    if len(rs.data) < off + 4:
        return False
    w = struct.unpack_from("<I", rs.data, off)[0]
    return DATA_BASE + TITLE_RUN[0] <= w < DATA_BASE + TITLE_RUN[0] + TITLE_RUN[1]


def title_spans(rs: RomSet) -> list[tuple[str, int, int, str]]:
    """(region, offset, length, what) of the bytes the title owns on `rs`, [] when it is not titled: TITLE_RUN from its
    start to the end of the copied upload set (the map, the chars, the lists, the set), the boot upload word and the
    logo's map word. TitleError for `rs` not a RomSet, or titled with a map word that does not name the title's map
    or an upload set that runs out of TITLE_RUN."""
    if not titled(rs):
        return []
    upload = struct.unpack_from("<I", rs.data, BOOT_WORD - DATA_BASE)[0]
    end = upload + 8 - DATA_BASE
    if struct.unpack_from("<I", rs.program, MAP_WORD)[0] != DATA_BASE + TITLE_RUN[0] or \
            end > TITLE_RUN[0] + TITLE_RUN[1]:
        raise TitleError("the set's boot upload word names TITLE_RUN but its logo map word or upload set is not the "
                         "title's")
    return [("data", TITLE_RUN[0], end - TITLE_RUN[0], "the Extended title's map, chars and boot upload set "
             "(title.extended_title)"),
            ("data", BOOT_WORD - DATA_BASE, 4, "the boot upload word, repointed to the title's copy (title.BOOT_WORD)"),
            ("program", MAP_WORD, 4, "the logo's layer-0 map word, repointed to the title's map (title.MAP_WORD)")]
