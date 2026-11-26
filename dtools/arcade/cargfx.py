"""The Car Select screen's tile art (v0.4 stage 2; dtools/arcade/carscreen.py), generated from the set's own
select-screen art at build time as the Course Direction screen's is (dirgfx.py: nothing of the game is stored here).

The banner "Car Select" in the banners' cursive: C, r, S, e, c, t are dirgfx.LETTERS'; the l and the first e of
"Select" are cut from Circuit Select's banner beside them (LETTERS); the a, which neither banner has, is Circuit's
c closed by the right stem of its u (A_STEM, A_OVERLAP, A_TOP; checked by eye). On the Circuit Select
banner's frame, as dirgfx.banner_pens.

The cars' names in the arcade's 8 x 8 font (dirgfx.caption_pens' at scale 1): one tile map each, their chars in one
block of 128 (NAME_FIRST), drawn twice with two palettes: the chosen one's yellow, the others' white.

CargfxError (a DecodeError) for a set whose banners do not decode."""

from __future__ import annotations

import numpy as np

from dtools.arcade import dirgfx as G
from dtools.arcade import select as S
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet


class CargfxError(DecodeError):
    pass


BANNER_TEXT = "Car Select"
LETTERS = {**G.LETTERS, "l": ("circuit", 224, 238), "E": ("circuit", 204, 224)}
"""dirgfx's letters, the l of Circuit Select's "Select", and (E) the e before it: its tail joins the l, where
dirgfx's e (the second) joins the c."""
A_STEM = (110, 122)                 # the u's right stem and tail (Circuit Select's banner columns)
A_OVERLAP = 4                       # columns the stem overlaps the o's right edge
A_TOP = 0                           # the stem starts this many rows under the o's top (the i's dot is above)
CHAR_FIRST = G.CHAR_FIRST           # the banner's chars: two blocks from 0x3d00, the banners' palette
R_DOT = (18, 6)                     # rows, columns at the r's top left cleared: the dot of the i before it
NAMES = ("HORNET", "GRASSHOPPER", "FALCON", "LIGHTNING", "UNICORN")
NAME_FIRST = (0x3E00, 0x3E80)       # the names' chars: a block with the unchosen palette, a block with the chosen
NAME_FILL = 4                       # the names' fill pen (dirgfx.CAPTION_FILL), 2 their outline
NAME_COLS = 12                      # tiles a name's map is wide (the longest, GRASSHOPPER, is 11 letters), centred


def banner_pens(rs: RomSet) -> np.ndarray:
    """The pens of the "Car Select" banner (48 rows, a multiple of 8 columns)."""
    try:
        banners = {"circuit": G.render_map(rs, G.CIRCUIT_BANNER)[1], "mission": G.render_map(rs, G.MISSION_BANNER)[1]}
    except G.DirgfxError as e:
        raise CargfxError(str(e)) from e
    top, bottom = G.LETTER_ROWS

    def glyph(ch):
        src, a, b = LETTERS[ch]
        return G._despeck(banners[src][top:bottom, a:b])

    def letter_a():
        o = glyph("c")                                   # the bowl: the c, which the stem closes
        stem = banners["circuit"][top:bottom, A_STEM[0]:A_STEM[1]].copy()
        stem[:int(np.nonzero((o != G.BG).any(1))[0][0]) + A_TOP] = G.BG    # the next letter's dot, above the bowl
        stem = G._despeck(stem)
        g = np.full((bottom - top, max(o.shape[1], o.shape[1] + stem.shape[1] - A_OVERLAP)), G.BG, np.uint8)
        G._over(g, o, 0)
        G._over(g, stem, o.shape[1] - A_OVERLAP)
        return G._despeck(g)

    text = "Car SElect"                                   # E: the e that joins the l
    def letter_r():
        g = glyph("r").copy()
        g[:R_DOT[0], :R_DOT[1]] = G.BG
        return G._despeck(g)

    glyphs = [None if ch == " " else letter_a() if ch == "a" else letter_r() if ch == "r" else glyph(ch)
              for ch in text]
    width_text = sum(G.SPACE if g is None else g.shape[1] for g in glyphs)
    width = (width_text + 2 * G.FRAME_ENDS + 4 + 7) // 8 * 8
    frame = banners["circuit"]
    out = np.concatenate([frame[:, :G.FRAME_ENDS], np.repeat(frame[:, G.FRAME_FILL:G.FRAME_FILL + 1],
                                                             width - 2 * G.FRAME_ENDS, 1), frame[:, -G.FRAME_ENDS:]], 1)
    x = (width - width_text) // 2
    body = out[top:bottom]
    for g in glyphs:
        if g is not None:
            G._over(body, g, x)
        x += G.SPACE if g is None else g.shape[1]
    return out


def name_pens(rs: RomSet, k: int) -> np.ndarray:
    """Car `k`'s name (NAMES) in the arcade's 8 x 8 font, centred in 8 x (8 * NAME_COLS) pixels: pen NAME_FILL the
    fill, 2 the outline, 0 elsewhere. CargfxError for a `k` outside NAMES."""
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < len(NAMES):
        raise CargfxError(f"car {k!r} is not 0..{len(NAMES) - 1}")
    font = S.panel_art(rs).font
    cells = [np.where(font[ch] == S.FONT_FILL, NAME_FILL, np.where(font[ch] != 0, 2, 0)).astype(np.uint8)
             for ch in NAMES[k]]
    text = np.concatenate(cells, 1)
    out = np.zeros((8, 8 * NAME_COLS), np.uint8)
    at = (out.shape[1] - text.shape[1]) // 2 // 8 * 8      # on a tile boundary, so the letters' chars are shared
    out[:, at:at + text.shape[1]] = text
    return out


BAR_LABELS = ("SPEED", "ACCEL")
BAR_SEGMENTS = 8
BAR_COLS = 5 + 1 + BAR_SEGMENTS      # a label, a gap, the segments
TOP_SPEED = (325, 305, 316, 335, 349)
"""Each car's top speed, km/h, measured in MAME on Three Seven Speedway, automatic (docs/follow-ups/v0.4.md, stage
3): the SPEED bar. The ACCEL bar is the car's first-gear drive (its ratio times its torque factor), from the set."""
SPEED_RANGE = (280.0, 350.0)
ACCEL_RANGE = (8.2, 10.7)


def segments(value: float, lo: float, hi: float) -> int:
    """`value` on a bar of BAR_SEGMENTS between `lo` (none) and `hi` (all), at least one."""
    return max(1, min(BAR_SEGMENTS, round(BAR_SEGMENTS * (value - lo) / (hi - lo))))


def bars_pens(rs: RomSet, filled) -> np.ndarray:
    """Two bars, one per BAR_LABELS, 8 pixels high each with a blank row of 8 between: the label in the 8 x 8 font,
    then BAR_SEGMENTS squares, the first `filled[k]` of them full (pen NAME_FILL inside an outline of pen 2), the rest
    hollow. CargfxError for `filled` that is not one count 0..BAR_SEGMENTS per label."""
    if (not isinstance(filled, (tuple, list)) or len(filled) != len(BAR_LABELS)
            or not all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= BAR_SEGMENTS for n in filled)):
        raise CargfxError(f"bars {filled!r:.40} are not {len(BAR_LABELS)} counts 0..{BAR_SEGMENTS}")
    font = S.panel_art(rs).font
    full = np.full((8, 8), 2, np.uint8)
    full[1:7, 1:7] = NAME_FILL
    hollow = np.full((8, 8), 2, np.uint8)
    hollow[1:7, 1:7] = 0
    full[:, 7] = hollow[:, 7] = 0                       # a pixel between the squares
    full[7], hollow[7] = 0, 0
    rows = []
    for label, n in zip(BAR_LABELS, filled):
        cells = [np.where(font[ch] == S.FONT_FILL, NAME_FILL, np.where(font[ch] != 0, 2, 0)).astype(np.uint8)
                 for ch in label]
        cells.append(np.zeros((8, 8), np.uint8))
        cells += [full] * n + [hollow] * (BAR_SEGMENTS - n)
        rows.append(np.concatenate(cells, 1))
    return np.concatenate([rows[0], np.zeros_like(rows[0]), rows[1]], 0)


def name_palette(chosen: bool) -> tuple[int, ...]:
    """The names' block: pen 2 black, pen NAME_FILL yellow when `chosen`, else white."""
    return (0, 0, G.BLACK, 0, G.YELLOW if chosen else G.WHITE) + (0,) * 11
