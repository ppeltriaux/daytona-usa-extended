"""The Course Direction screen's graphics (M9 stage 3, dtools/arcade/dirscreen.py; Pascal's mockup C), generated from
the set's own select-screen art at build time (nothing of the game is stored in the repository).

The banner "Course Direction" in the cursive of Circuit Select's and Mission Select's banners (their tile maps
CIRCUIT_BANNER and MISSION_BANNER, rendered from the chars Circuit Select's uploads put in char RAM, course.select_assets):
each letter cut from one of them at the joins (LETTERS, columns checked by eye against both banners), the D made of the
S's upper curve over the C mirrored (neither banner has a D), on the Circuit Select banner's frame widened to fit
(its left and right ends kept, a plain interior column repeated between). Its pens use the banners' one palette (both
banners' four colours hold the same 16 words).

The chars go to char RAM from CHAR_FIRST (0x3d00: no boot or select upload covers 0x3a82-0x3fff, and the release row's
generated select chars end below it, select.graphics), so their colours are 0x7a on (a char's colour is its index >> 7,
tiles.py); each block of 128 of them gets the palette of what it holds.

DirgfxError (a DecodeError) for a set whose banners do not decode."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import select as S
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet


class DirgfxError(DecodeError):
    pass


CIRCUIT_BANNER = 0x0282B712         # Circuit Select's banner map (blitted at 0x226a78 to 0x1000096)
MISSION_BANNER = 0x02828944         # Mission Select's (0x225b00)
BANNER_COLOUR = 30                  # one of the banners' colours (30-33, 74-75: the same 16 words)
BG = 14                             # the banner's red
LETTER_ROWS = (5, 42)               # the banners' interior rows the letters use
FRAME_ENDS = 10                     # columns of each frame end kept whole
FRAME_FILL = 160                    # a Circuit Select banner column with no letter (between its two words)
LETTERS = {                         # letter: (banner, first column, end column), cut at the joins
    "C": ("circuit", 10, 50), "u": ("circuit", 98, 122), "r": ("circuit", 62, 80), "e": ("circuit", 238, 258),
    "c": ("circuit", 258, 277), "t": ("circuit", 277, 297), "o": ("mission", 128, 147), "s": ("mission", 98, 116),
    "i": ("mission", 116, 128), "n": ("mission", 147, 166), "S": ("circuit", 169, 204),
}
D_CURVE_ROWS = 18                   # the S's upper rows that make the D's top
D_HOOK = (10, 12)                   # rows, columns of the mirrored C's hook cleared under the curve
SPACE = 12                          # pixels between the words
SPECK = 6                           # ink islands smaller than this at a cut are a neighbour's crumbs: cleared
BANNER_TEXT = "Course Direction"
CHAR_FIRST = 0x3D00
CHAR_END = 0x4000
BLOCK = 128


def render_map(rs: RomSet, addr: int) -> tuple[TL.TileMap, np.ndarray]:
    """(the tile map at `addr`, its pens): each tile's char as Circuit Select's uploads leave it in char RAM.
    DirgfxError for a map or char that does not decode."""
    try:
        m = TL.tile_map(rs, addr)
        source = C.select_assets(rs).char_source
    except DecodeError as e:
        raise DirgfxError(f"the map at {addr:#x}: {e}") from e
    pens = np.zeros((8 * m.rows, 8 * m.cols), np.uint8)
    for r in range(m.rows):
        for c in range(m.cols):
            ch = m.tile(r, c) & TL.CHAR_MASK
            if ch not in source:
                raise DirgfxError(f"the map at {addr:#x}: tile ({r}, {c}) names char {ch:#x}, which no select upload "
                                  f"provides")
            pens[8 * r:8 * r + 8, 8 * c:8 * c + 8] = TL.decode_chars(rs.read_i960(source[ch], TL.CHAR_SIZE))[0]
    return m, pens


def _despeck(g: np.ndarray) -> np.ndarray:
    """`g` with ink islands (8-connected) smaller than SPECK pixels made background."""
    ink, seen, out = g != BG, np.zeros(g.shape, bool), g.copy()
    for y, x in zip(*np.nonzero(ink)):
        if seen[y, x]:
            continue
        q, comp = deque([(y, x)]), []
        seen[y, x] = True
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for ny in (cy - 1, cy, cy + 1):
                for nx in (cx - 1, cx, cx + 1):
                    if 0 <= ny < g.shape[0] and 0 <= nx < g.shape[1] and ink[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        if len(comp) < SPECK:
            for cy, cx in comp:
                out[cy, cx] = BG
    return out


def _over(dst: np.ndarray, src: np.ndarray, at: int) -> None:
    ink = src != BG
    dst[:, at:at + src.shape[1]][ink] = src[ink]


def banner_pens(rs: RomSet) -> np.ndarray:
    """The pens of the "Course Direction" banner (48 rows, a multiple of 8 columns): the module docstring's rule."""
    banners = {"circuit": render_map(rs, CIRCUIT_BANNER)[1], "mission": render_map(rs, MISSION_BANNER)[1]}
    top, bottom = LETTER_ROWS

    def glyph(ch):
        src, a, b = LETTERS[ch]
        return _despeck(banners[src][top:bottom, a:b])

    def letter_d():
        bowl = glyph("C")[:, ::-1].copy()
        bowl[:D_HOOK[0], :D_HOOK[1]] = BG
        curve = glyph("S").copy()
        curve[D_CURVE_ROWS:] = BG
        g = np.full((bottom - top, bowl.shape[1] + 4), BG, np.uint8)
        _over(g, bowl, g.shape[1] - bowl.shape[1])
        _over(g, curve[:, :g.shape[1]], 0)
        return _despeck(g)

    glyphs = [None if ch == " " else letter_d() if ch == "D" else glyph(ch) for ch in BANNER_TEXT]
    text = sum(SPACE if g is None else g.shape[1] for g in glyphs)
    width = (text + 2 * FRAME_ENDS + 4 + 7) // 8 * 8
    frame = banners["circuit"]
    out = np.concatenate([frame[:, :FRAME_ENDS], np.repeat(frame[:, FRAME_FILL:FRAME_FILL + 1],
                                                           width - 2 * FRAME_ENDS, 1), frame[:, -FRAME_ENDS:]], 1)
    x = (width - text) // 2
    body = out[top:bottom]
    for g in glyphs:
        if g is not None:
            _over(body, g, x)
        x += SPACE if g is None else g.shape[1]
    return out


class Art(NamedTuple):
    """Chars (n, 8, 8) from a first char, and a tile map of them (char indices, row-major)."""
    chars: np.ndarray
    rows: int
    cols: int
    tiles: tuple[int, ...]


def tiled(pens: np.ndarray, first: int) -> Art:
    """`pens` (a multiple of 8 each way) cut into 8 x 8 chars, identical ones shared, numbered from `first`."""
    rows, cols = pens.shape[0] // 8, pens.shape[1] // 8
    index: dict[bytes, int] = {}
    chars, tiles = [], []
    for r in range(rows):
        for c in range(cols):
            cell = pens[8 * r:8 * r + 8, 8 * c:8 * c + 8]
            key = cell.tobytes()
            if key not in index:
                index[key] = first + len(chars)
                chars.append(cell)
            tiles.append(index[key])
    return Art(np.array(chars, np.uint8), rows, cols, tuple(tiles))


def banner_palette(rs: RomSet) -> tuple[int, ...]:
    return S.palette_of(rs, BANNER_COLOUR)


# --- the panel (the chosen entry's select panel, its band reading REVERSE x N LAPS in red) -------------------------

PANEL_ROWS, PANEL_COLS = 10, 18     # a select panel's map (course.PANEL_COLS)
BAND_Y = (8 + S.BAND[0], 8 + S.BAND[1])     # the panel's pixel rows of the band (the border's row above its interior)
INTERIOR_X = (8, 136)               # the panel's pixel columns inside its frame
RED = 0x801F                        # REVERSE's colour word (System 24: bit 15 set as the panels' own words, red 31)
V_ROWS = ("##.##",) * 8 + (".###.",) * 3 + ("..#..",) * 3
"""The V, which Sega's bands never needed: two-pixel strokes straight for eight rows, then meeting as the font's U
narrows at its bottom; five wide like the narrowed letters (NARROW_COLUMN)."""
NARROW_COLUMN = 3
"""REVERSE's letters are the band font's six-wide letters with this column of their glyph (margin, then six) dropped:
five wide. Six-wide, REVERSE (48 pixels with its gaps) does not fit before the times sign on a band whose lap count
has two digits (38-40 pixels there from column 15, where Sega's words start, to the sign's gap); five-wide it does,
the same on every panel."""


def _runs(cols) -> list[tuple[int, int]]:
    out, start = [], None
    for x, v in enumerate(cols):
        if v and start is None:
            start = x
        if not v and start is not None:
            out.append((start, x))
            start = None
    if start is not None:
        out.append((start, len(cols)))
    return out


def _words(rs: RomSet, m: TL.TileMap) -> tuple[np.ndarray, np.ndarray]:
    """(pens, colour words) of a select tile map as Circuit Select shows it (each tile's char from its uploads, its
    colour's palette)."""
    source = C.select_assets(rs).char_source
    pens = np.zeros((8 * m.rows, 8 * m.cols), np.uint8)
    words = np.zeros(pens.shape, np.int64)
    palettes: dict[int, np.ndarray] = {}
    for r in range(m.rows):
        for c in range(m.cols):
            v = m.tile(r, c)
            px = TL.decode_chars(rs.read_i960(source[v & TL.CHAR_MASK], TL.CHAR_SIZE))[0]
            colour = (v >> 7) & 0xFF
            if colour not in palettes:
                palettes[colour] = np.array(S.palette_of(rs, colour), np.int64)
            pens[8 * r:8 * r + 8, 8 * c:8 * c + 8] = px
            words[8 * r:8 * r + 8, 8 * c:8 * c + 8] = palettes[colour][px]
    return pens, words


def panel_words(rs: RomSet, entry: int, mode: int) -> tuple[np.ndarray, np.ndarray]:
    """(pens, colour words) of `entry`'s panel on a patched set's Circuit Select in game mode `mode` (its normal map,
    the name window's rows from its layer-1 map, select.WINDOW_ROWS)."""
    row = S.select_row(rs)
    pens, words = _words(rs, TL.tile_map(rs, row.records[mode][entry].maps[0]))
    w = S.window_of(S.ENTRIES[entry])
    if w != S.WINDOW_NONE:
        npens, nwords = _words(rs, TL.tile_map(rs, row.names[entry]))
        x0, x1 = INTERIOR_X
        for tr in S.WINDOW_ROWS[w]:
            pens[8 * tr:8 * tr + 8, x0:x1] = npens[8 * tr:8 * tr + 8, x0:x1]
            words[8 * tr:8 * tr + 8, x0:x1] = nwords[8 * tr:8 * tr + 8, x0:x1]
    return pens, words


def _band_parts(words: np.ndarray):
    """(background, text word, the length word's letter runs, the times sign's run) of a panel's band: the text the
    white, the digits the accent (the most orange), the runs of white before the digits the word's letters then the
    times sign. DirgfxError when it is not so."""
    band = words[BAND_Y[0]:BAND_Y[1]]
    vals, counts = np.unique(band, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    grey = [int(v) for v in vals if (v & 31) == (v >> 5 & 31) == (v >> 10 & 31)]
    text = max(grey, key=lambda v: v & 31)
    accent = max((int(v) for v in vals if v != text), key=lambda v: (v & 31) - (v >> 10 & 31))
    tcols, acols = (band == text).any(0), (band == accent).any(0)
    for cols in (tcols, acols):
        cols[:INTERIOR_X[0]] = cols[INTERIOR_X[1]:] = False
    if not acols.any():
        raise DirgfxError("a panel band without digits")
    digits0 = int(np.nonzero(acols)[0][0])
    pre = [r for r in _runs(tcols) if r[1] <= digits0]
    if len(pre) < 2:
        raise DirgfxError("a panel band without a length word and a times sign")
    return bg, text, pre[:-1], pre[-1]


def _glyphs(words: np.ndarray, letters: str) -> tuple[dict[str, np.ndarray], int]:
    bg, text, runs, _ = _band_parts(words)
    if len(runs) != len(letters):
        raise DirgfxError(f"a band word of {len(runs)} letters, expected {letters}")
    out = {}
    for ch, (a, b) in zip(letters, runs):
        g = words[BAND_Y[0]:BAND_Y[1], a - 1:b + 1].copy()
        g[g != text] = -1
        out[ch] = g
    return out, text


def reverse_letters(rs: RomSet) -> tuple[dict[str, np.ndarray], int]:
    """({R, E, V, S: text-pixel glyphs, -1 elsewhere, five wide}, the text word): R and S from Three Seven Speedway's
    SHORT, E from Dinosaur Canyon's MEDIUM (game mode 0), each narrowed (NARROW_COLUMN), V drawn (V_ROWS) on the
    font's rows."""
    short, text = _glyphs(panel_words(rs, 0, 0)[1], "SHORT")
    medium, _ = _glyphs(panel_words(rs, 1, 0)[1], "MEDIUM")
    out = {ch: np.delete(g, NARROW_COLUMN, 1) for ch, g in (("R", short["R"]), ("S", short["S"]), ("E", medium["E"]))}
    top = int(np.nonzero((medium["U"] == text).any(1))[0].min())
    v = np.full(out["E"].shape, -1, np.int64)
    for y, line in enumerate(V_ROWS):
        for x, ch in enumerate(line):
            if ch == "#":
                v[top + y, 1 + x] = text
    out["V"] = v
    return out, text


def reverse_panel(words: np.ndarray, letters: dict[str, np.ndarray], text: int) -> np.ndarray:
    """`words` (a panel) with its band's length word replaced by REVERSE in RED, ending select.BAND_WORD_GAP before the
    times sign. DirgfxError when it would leave the panel's interior."""
    out = words.copy()
    bg, _, runs, times = _band_parts(words)
    out[BAND_Y[0]:BAND_Y[1], runs[0][0] - 1:runs[-1][1] + 1] = bg
    seq = [letters[c] for c in "REVERSE"]
    width = sum(g.shape[1] - 1 for g in seq) + 1
    x = times[0] - S.BAND_WORD_GAP - width
    if x < INTERIOR_X[0]:
        raise DirgfxError(f"REVERSE ({width} pixels) does not fit before the times sign at {times[0]}")
    for g in seq:
        region = out[BAND_Y[0]:BAND_Y[1], x:x + g.shape[1]]
        region[g == text] = RED
        x += g.shape[1] - 1
    return out


PANEL_FIRST = 0x3E00                # the panel's chars: two blocks (0x7c, 0x7d), its own palette
BAND_TILE_ROWS = (6, 9)             # the panel's tile rows the band swap rewrites (BAND_Y within them)


class PanelArt(NamedTuple):
    chars: np.ndarray               # (n, 8, 8) pens from PANEL_FIRST
    palette: tuple[int, ...]        # 16 words: pen 0 transparent, then the panel's colours and RED
    panel: tuple[int, ...]          # PANEL_ROWS x PANEL_COLS chars: the panel, its band normal
    normal_band: tuple[int, ...]    # the band's tile rows (BAND_TILE_ROWS) x PANEL_COLS, normal
    reverse_band: tuple[int, ...]   # the same rows, REVERSE x N LAPS


def panel_art(rs: RomSet, entry: int, mode: int, letters=None) -> PanelArt:
    """`entry`'s panel in game mode `mode` and its REVERSE band as chars from PANEL_FIRST in one palette (the panel's
    colour words plus RED; pen 0 where the panel is transparent). `letters` (reverse_letters' result) saves its
    re-reading. DirgfxError for more than 15 colours or more chars than PANEL_FIRST's two blocks hold."""
    letters, text = letters or reverse_letters(rs)
    pens, words = panel_words(rs, entry, mode)
    rev = reverse_panel(words, letters, text)
    used = sorted(set(words[pens != 0].tolist()) | set(rev[pens != 0].tolist()))
    if len(used) > 15:
        raise DirgfxError(f"entry {entry} mode {mode}: {len(used)} colours")
    pen = {w: k + 1 for k, w in enumerate(used)}
    def to_pens(w):
        out = np.zeros(w.shape, np.uint8)
        for word, k in pen.items():
            out[w == word] = k
        out[pens == 0] = 0
        return out
    normal, reverse = to_pens(words), to_pens(rev)
    lo, hi = 8 * BAND_TILE_ROWS[0], 8 * BAND_TILE_ROWS[1]
    both = np.concatenate([normal, reverse[lo:hi]], 0)
    art = tiled(both, PANEL_FIRST)
    if len(art.chars) > 2 * BLOCK:
        raise DirgfxError(f"entry {entry} mode {mode}: {len(art.chars)} chars, more than two blocks")
    n = PANEL_ROWS * PANEL_COLS
    rows = art.tiles[:n]
    band_n = (BAND_TILE_ROWS[1] - BAND_TILE_ROWS[0]) * PANEL_COLS
    return PanelArt(art.chars, (0,) + tuple(used) + (0,) * (15 - len(used)), rows,
                    rows[BAND_TILE_ROWS[0] * PANEL_COLS:BAND_TILE_ROWS[1] * PANEL_COLS], art.tiles[n:n + band_n])


CAPTION_FIRST = (0x3F00, 0x3F80)    # the NORMAL side's chars (block 0x7e: caption, arrow, frame), REVERSE's (0x7f)
WHITE, YELLOW, BLACK, RED = 0xFFFF, 0x83FF, 0x8000, 0x801F
CAPTION_SCALE = 2                   # each font pixel 2 x 2
CAPTION_FILL = 4                    # the caption's fill pen (the arrow 1, black 2, the frame's yellow 3)
PRIORITY = 0x8000                   # a tile's bit 15: drawn over the polygons (tiles.py)


def caption_pens(rs: RomSet, text: str) -> np.ndarray:
    """`text` in the arcade's 8 x 8 font (select.panel_art's), each pixel CAPTION_SCALE square: pen CAPTION_FILL
    the letters' fill (select.FONT_FILL), pen 2 their outline, 0 elsewhere."""
    font = S.panel_art(rs).font
    cells = []
    for ch in text:
        g = font[ch]
        cell = np.where(g == S.FONT_FILL, CAPTION_FILL, np.where(g != 0, 2, 0)).astype(np.uint8)
        cells.append(np.kron(cell, np.ones((CAPTION_SCALE, CAPTION_SCALE), np.uint8)))
    return np.concatenate(cells, 1)


def side_palette(side: int, chosen: bool) -> tuple[int, ...]:
    """Side `side`'s block (0 NORMAL, 1 REVERSE): pen 1 its arrow (WHITE, RED), 2 BLACK, 3 the frame's YELLOW, 4 the
    caption's fill (YELLOW when `chosen`, else WHITE)."""
    return (0, (WHITE, RED)[side], BLACK, YELLOW, YELLOW if chosen else WHITE) + (0,) * 11


# --- the arrows and the frame over the two maps (drawn here: not the game's art) -----------------------------------

AREA_ROWS, AREA_COLS = 15, 16       # each map's area of tiles: the frame round it, the arrow inside
ARROW = 96                          # the arrow's square, pixels, centred in the area
ARROW_RADIUS, ARROW_WIDTH = 34, 7   # the ring's radius and line width, pixels
ARROW_ARC = (100, 380)              # degrees the ring runs (0 = right, growing clockwise on screen): a gap at the top
ARROW_HEAD = 15                     # the head's half width, pixels
FRAME_WIDTH = 3                     # the yellow frame's line, inside a black pixel each side
OUTLINE, FILL_WHITE, FILL_RED, FILL_YELLOW = "outline", "white", "red", "yellow"


def arrow_pens(clockwise: bool) -> np.ndarray:
    """(ARROW, ARROW) of 0 (transparent), 1 (the arrow's fill) and 2 (its one-pixel outline): a ring running ARROW_ARC
    with a head at its end, round the way the lap goes on screen (`clockwise`: y down; the anticlockwise one the
    clockwise mirrored)."""
    y, x = np.mgrid[0:ARROW, 0:ARROW].astype(float) + 0.5
    c = ARROW / 2
    r = np.hypot(x - c, y - c)
    ang = (np.degrees(np.arctan2(y - c, x - c)) - ARROW_ARC[0]) % 360
    span = ARROW_ARC[1] - ARROW_ARC[0] - 12
    fill = (np.abs(r - ARROW_RADIUS) <= ARROW_WIDTH / 2) & (ang <= span)
    end = np.radians(ARROW_ARC[0] + span)
    tip_dir = np.array([-np.sin(end), np.cos(end)])          # the tangent, clockwise
    base = c + ARROW_RADIUS * np.array([np.cos(end), np.sin(end)])
    out_dir = np.array([np.cos(end), np.sin(end)])
    a = base + out_dir * ARROW_HEAD
    b = base - out_dir * ARROW_HEAD
    t = base + tip_dir * (ARROW_HEAD + 4)
    def side(p, q):
        return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])
    s1, s2, s3 = side(a, b), side(b, t), side(t, a)
    head = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    fill |= head
    grown = fill.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(fill, dy, 0), dx, 1)
    pens = np.where(fill, 1, np.where(grown, 2, 0)).astype(np.uint8)
    return pens if clockwise else pens[:, ::-1].copy()


def area_pens(clockwise: bool, framed: bool) -> np.ndarray:
    """(8 * AREA_ROWS, 8 * AREA_COLS) of 0, 1 (the arrow's fill), 2 (black) and 3 (yellow): the arrow centred, and with
    `framed` the frame on the area's edge (FRAME_WIDTH of yellow between black pixels)."""
    h, w = 8 * AREA_ROWS, 8 * AREA_COLS
    out = np.zeros((h, w), np.uint8)
    y0, x0 = (h - ARROW) // 2, (w - ARROW) // 2
    out[y0:y0 + ARROW, x0:x0 + ARROW] = arrow_pens(clockwise)
    if framed:
        for k, pen in ((0, 2), *((i, 3) for i in range(1, FRAME_WIDTH + 1)), (FRAME_WIDTH + 1, 2)):
            out[k, k:w - k] = out[h - 1 - k, k:w - k] = pen
            out[k:h - k, k] = out[k:h - k, w - 1 - k] = pen
    return out


def course_clockwise(rs: RomSet, course: int) -> bool:
    """Whether course `course`'s lap runs clockwise on its select map as the screen shows it: its road in the map
    quad's frame (selmap.fit, the originals' turn), shown mirrored left to right (measured: National Park Speedway's
    map on the screen against the fit, 18 Nov 2026), the shoelace sum positive with y down."""
    from dtools.arcade import selmap
    q = np.array(selmap.fit(C.road_line(rs, C.course_table(rs)[course])))
    sx, sy = -q[:, 0], q[:, 1]
    return float(np.sum(sx * np.roll(sy, -1) - np.roll(sx, -1) * sy)) > 0


def tiled_many(pens: list[np.ndarray], first: int) -> tuple[np.ndarray, list[Art]]:
    """Each of `pens` cut into 8 x 8 chars as `tiled` cuts one, identical chars shared across all of them (numbered
    from `first` in order of first use): (the chars, an Art per pens, each holding all the chars)."""
    index: dict[bytes, int] = {}
    chars, arts = [], []
    for p in pens:
        rows, cols = p.shape[0] // 8, p.shape[1] // 8
        tiles = []
        for r in range(rows):
            for c in range(cols):
                cell = p[8 * r:8 * r + 8, 8 * c:8 * c + 8]
                key = cell.tobytes()
                if key not in index:
                    index[key] = first + len(chars)
                    chars.append(cell)
                tiles.append(index[key])
        arts.append((rows, cols, tuple(tiles)))
    out = np.array(chars, np.uint8).reshape(-1, 8, 8)
    return out, [Art(out, r, c, t) for r, c, t in arts]


class Side(NamedTuple):
    """A side's chars (from CAPTION_FIRST[side]) and its maps: the caption, the area without and with the frame
    (the area's tiles with PRIORITY: over the map)."""
    chars: np.ndarray
    caption: Art
    plain: Art
    framed: Art


def side_art(rs: RomSet, side: int, clockwise: bool) -> Side:
    """Side `side`'s art (0 NORMAL, 1 REVERSE) with its arrow `clockwise` or not: the caption first, so its chars
    (and map) are the same for either arrow. DirgfxError past the side's block."""
    chars, (cap, plain, framed) = tiled_many([caption_pens(rs, ("NORMAL", "REVERSE")[side]),
                                              area_pens(clockwise, False), area_pens(clockwise, True)],
                                             CAPTION_FIRST[side])
    if len(chars) > BLOCK:
        raise DirgfxError(f"side {side}'s chars ({len(chars)}) fill more than a block")
    pri = [a._replace(tiles=tuple(t | PRIORITY for t in a.tiles)) for a in (plain, framed)]
    return Side(chars, cap, *pri)
