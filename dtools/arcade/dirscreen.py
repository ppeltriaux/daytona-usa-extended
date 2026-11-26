"""The Course Direction screen (M9 stage 3; spec docs/superpowers/specs/2026-11-16-reverse-courses-design.md §5.3,
Pascal's mockup C, plan docs/superpowers/plans/2026-11-18-reverse-stage3-direction.md): after Circuit Select, a
NORMAL / REVERSE choice between the chosen course n and its reverse id n + 8, before Mission Select (the AT / MT
screen).

The select's state machine (byte 0x62(g13), g13 = 0x504380) runs one handler per state from a table of 17 words
(STOCK_TABLE, single cabinet; the linked cabinets' table LINKED_TABLE follows it, so it cannot grow in place): 0-3
Circuit Select, 4-7 the wipe into Mission Select (7 ends with `mov 8` and Mission Select's layers), 8-11 Mission Select,
12-16 the wipe and the race load. patch_direction copies the single table to TABLE_AT with three states more and points
the dispatcher's load at it (DISPATCH_WORD): state 7's handler is wrapped (`dir7`: the stock handler, then, when it
moved on to 8 with a course id below 8 and this cabinet racing alone, state 17 instead), 17 sets the screen up, 18 runs it, 19 holds after the choice
and goes on to Mission Select (state 8), with the course id + 8 and widen_ids' shadows refreshed (course.WIDE_CODE's
`update`) for REVERSE. Both dispatchers' tables are copied so (TABLE_AT, LINKED_TABLE_AT): MAME's default cabinet
(TWIN) runs the linked one even alone; the wrapper skips the screen when the set is linked and the vote counted
another cabinet (VOTES > 1), so real linked play keeps the stock path. The attract demo never runs the select. The dispatcher draws the timer only in states 0-4 and 9 and runs the select's per-frame calls (0x17e28, the
maps, the prompts) only below 12, so 18 draws the timer itself (0x226d40).

The choice (Virtua Racing's D6): NORMAL until the wheel has been at centre or left (w <= ARM_AT) on this screen; armed,
w > CHOOSE_AT (Mission Select's own threshold) is REVERSE. w as Mission Select reads it: (0x501050 + 0x50105c) >> 1.
The gas counts once the timer is at most GAS_FROM (0x225cb0's rule); the timer's end takes what is chosen.

DirectionError (a DecodeError) for a set the patch does not fit."""

from __future__ import annotations

import struct
from typing import NamedTuple

from dtools.arcade import course as C
from dtools.arcade import dirgfx as G
from dtools.arcade import i960
from dtools.arcade import select as S
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet


class DirectionError(DecodeError):
    pass


STOCK_TABLE = 0x0022558C            # the single cabinet's state handlers, u32[17]
LINKED_TABLE = 0x002255D0           # the linked cabinets', u32[17]
STOCK_STATES = 17
DISPATCH_LOAD = 0x0022547C          # `ld 0x22558c[r3*4],r4` (MEMB, 8 bytes)
DISPATCH_WORD = DISPATCH_LOAD + 4   # its displacement word
LINKED_LOAD = 0x002254B4            # the linked dispatcher's `ld 0x2255d0[r3*4],r4`
LINKED_WORD = LINKED_LOAD + 4
LINKED = 0x005FE588                 # byte: the link setting (MAME's default cabinet, TWIN, is linked: 1)
VOTES = 0x00501B03                  # byte: the select vote's count (select.VOTE_RAM): 1 when this cabinet races alone
STATE_SETUP, STATE_RUN, STATE_HOLD = 17, 18, 19
STATES = STATE_HOLD + 1
STOCK_STATE7 = 0x00225AB8           # the wipe into Mission Select; ends with `mov 8`
MISSION_SETUP = 8
CODE_AT = 0x0001EF00                # in the select patch's reserved program run (select.CODE_FREE), past its code
DIR_RAM = 0x00501B26                # the select's reserved RAM past its own bytes (select.FOLLOW_RAM is 0x501b25)
CHOICE = DIR_RAM                    # byte: 0 NORMAL, 1 REVERSE
ARMED = DIR_RAM + 1                 # byte: 1 once the wheel has been at centre or left on this screen
TIMER = 0x005010A8                  # the select's timer (frames; shown >> 6)
SCREEN_FRAMES = 640                 # 10 s, as state 4 sets for Mission Select
GAS_FROM = 0x270                    # the gas counts from here down (16 frames in), as 0x225cb0
WHEEL_A, WHEEL_B = 0x00501050, 0x0050105C
ARM_AT = 16                         # w at most this: centre or left (the idle paddle 0x80 reads about 0)
CHOOSE_AT = 30                      # w past this: right (Mission Select's Manual threshold, 0x225c70)
GAS = 0x00501068                    # the gas pedal's press this frame
HOLD = 0x56                         # 0x56(g13): the select's hold counter (states 10-11 use it)
HOLD_FRAMES = 32
CONFIRM_SOUND = 0x00AE2158          # Mission Select's confirm (0x225d40)
SOUND = 0x0001A064
DRAW_TIMER = 0x00226D40
REVERSE_OFFSET = 8
UPLOAD_CHARS, UPLOAD_PALETTES = 0x0001DD70, 0x0001DDE0     # a char list, a palette list (g0)
BLIT = S.BLIT                       # (g0 map, g1 tile-RAM address)
BANNER_DST = TL.TILE_RAM[0] + 2 * (1 * TL.TILE_LAYER + 8)  # layer 0, row 1, column 8: centred where the old one sat
MISSION_LAYERS = 0x00225AD0         # state 7's tail after `mov 8`: Mission Select's background to layer 2, layer 0
                                    # cleared (0x1bb2c), its boxes (0x226ab8) and banner: called, it returns here
DATA_AT = C.DATA_BASE + 0x01FC0000  # the screen's graphics: the custom data run's last 256 KB (title.TITLE_RUN's end,
DATA_SPAN = 0x30000                 # which the title's bytes, from that run's start, never reach)


PANEL_DST = TL.TILE_RAM[0] + 2 * (8 * TL.TILE_LAYER + 22)   # layer 0, row 8, column 22: the panel at the top centre
BAND_DST = PANEL_DST + 2 * G.BAND_TILE_ROWS[0] * TL.TILE_LAYER
COURSE_BOX, BOX_DST = 0x02835318, 0x0100128C   # Circuit Select's left box, SELECT COURSE BY STEER, where Mission
                                    # Select's (0x283593c, SELECT MISSION BY STEER) goes: its chars are still loaded
FOOTER_ROWS = (44, 48)              # layer 0's rows of Mission Select's footer (the time-attack line): cleared
HOLE = S.CHAR_HOLE                  # the select's all-transparent char (still in char RAM)
def _cell(row: int, col: int) -> int:
    """Layer 0's tile (row, col) in tile RAM."""
    return TL.TILE_RAM[0] + 2 * (row * TL.TILE_LAYER + col)


AREA_ROW, AREA_COLS = 18, (3, 43)   # the two maps' areas (dirgfx.AREA_ROWS x AREA_COLS tiles): the arrow, the frame
AREA_DST = tuple(_cell(AREA_ROW, c) for c in AREA_COLS)
CAPTION_DST = (_cell(34, 5), _cell(34, 44))         # under the areas, centred (NORMAL 96 pixels, REVERSE 112)
SIDE_RECORD = 20                    # an arrow variant: the char list (both sides), then the area maps [choice][side]
GAME_MODE = 0x005FE5E0              # u16: the game mode 0-2 (the select draws that mode's panel records)
DIR_PTR = DIR_RAM + 2               # u32 (0x501b28): the chosen entry's directory record
SHOWN = DIR_RAM + 6                 # byte: the choice the band shows (0xff: none yet)
DIR_OBJECTS = DIR_RAM + 10          # u32 (0x501b30): the chosen entry's map object list
DIR_SIDES = DIR_RAM + 14            # u32 (0x501b34): the chosen entry's arrow variant (SIDE_RECORD bytes)
SAVED_TIMER = DIR_RAM + 18          # u32 (0x501b38): Mission Select's timer as state 7 left it (320 on the linked
                                    # path, 640 on the single-cabinet one), restored when the screen hands over
FRAME_SETUP = 0x00017E28            # the select states' per-frame call before their maps (the dispatcher's, below 12)
DRAW_OBJECTS = 0x00017ACC           # g0: an object list (0x226518)
def _f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


MAP_X, MAP_Y, MAP_Z = 5.145, -0.389, 9.0
"""The two maps' places (x -MAP_X and MAP_X): Circuit Select's positions 0 and 2 (x -5 and 5, y -1, z 9:
select.map_records) moved to the areas' centres (measured in MAME: 31.1 pixels a unit at z 9, y up)."""
MAP_PLACES = tuple((_f32(x), _f32(MAP_Y), _f32(MAP_Z)) for x in (-MAP_X, MAP_X))
MAP_ANGLES, MAP_SCALE = (1 << 14, 1 << 15), _f32(0.8)       # 0x22653c's g4, g5 (its g6 0.9: smaller here, to frame)
RECORD = 20                         # a directory record: char list, palette list, panel map, band maps normal, reverse
MODES = 3


class Layout(NamedTuple):
    """The screen's graphics in data ROM (i960 addresses): the common char list, palette list and the banner's tile
    map; the directory of the panel variants (entry * MODES + mode: RECORD bytes each); SELECT_PANEL_OF_COURSE."""
    chars: int
    palettes: int
    banner: int
    footer: int
    captions: tuple[int, int]       # NORMAL's and REVERSE's maps
    caption_palettes: int           # u32[2]: both sides' palette list for choice 0 (NORMAL yellow) and 1
    sides: int                      # u32[8]: each entry's arrow variant (SIDE_RECORD bytes: its NORMAL clockwise or not)
    directory: int
    panel_of_course: int
    objects: int                    # u32[8]: each entry's map object list (the select's map records)
    end: int


_PLACEHOLDER = Layout(*(DATA_AT,) * 4, (DATA_AT, DATA_AT), *(DATA_AT,) * 6)


def _palette_list(first_char: int, words: tuple[int, ...], blocks: int) -> bytes:
    return TL.encode_palette_list([TL.PaletteUpload(TL.PALETTE_RAM[0] + 32 * (first_char >> 7), tuple(words) * blocks)])


def graphics(rs: RomSet) -> tuple[Layout, bytes]:
    """The screen's graphics data for `rs`, laid out from DATA_AT: the banner (dirgfx.banner_pens: its char block,
    char list, palette list and tile map), then per entry and game mode the panel's (dirgfx.panel_art: char block,
    char list, palette list, the panel's map and its two band maps) and the directory naming them."""
    out = bytearray()

    def put(blob: bytes) -> int:
        at = DATA_AT + len(out)
        out.extend(blob)
        out.extend(b"\0" * (-len(out) % 4))
        return at

    banner = G.tiled(G.banner_pens(rs), G.CHAR_FIRST)
    if G.CHAR_FIRST + len(banner.chars) > G.PANEL_FIRST:
        raise DirectionError(f"the banner's chars ({len(banner.chars)}) run into the panel's at {G.PANEL_FIRST:#x}")
    try:
        variants = [[G.side_art(rs, side, cw != bool(side)) for side in (0, 1)] for cw in (True, False)]
    except G.DirgfxError as e:
        raise DirectionError(str(e)) from e
    block = put(TL.encode_char_block(TL.encode_chars(banner.chars)))
    chars_at = put(TL.encode_char_list([(block, TL.CHAR_RAM[0] + TL.CHAR_SIZE * G.CHAR_FIRST)]))
    palettes_at = put(_palette_list(G.CHAR_FIRST, G.banner_palette(rs), (len(banner.chars) + G.BLOCK - 1) // G.BLOCK))
    banner_at = put(TL.encode_tile_map(TL.TileMap(0, True, banner.rows, banner.cols, banner.tiles)))
    rows = FOOTER_ROWS[1] - FOOTER_ROWS[0]
    footer_at = put(TL.encode_tile_map(TL.TileMap(0, True, rows, TL.TILE_LAYER, (HOLE,) * (rows * TL.TILE_LAYER))))
    def tmap(a: G.Art) -> int:
        return put(TL.encode_tile_map(TL.TileMap(-0x8000 if a.tiles[0] & G.PRIORITY else 0, True, a.rows, a.cols,
                                                 tuple(t - (0x8000 if t & G.PRIORITY else 0) for t in a.tiles))))

    cap_maps = tuple(tmap(variants[0][side].caption) for side in (0, 1))
    if any(variants[1][side].caption.tiles != variants[0][side].caption.tiles for side in (0, 1)):
        raise DirectionError("a caption's chars differ between the arrow variants")
    lists = [put(TL.encode_palette_list([TL.PaletteUpload(TL.PALETTE_RAM[0] + 32 * (f >> 7),
                                                          G.side_palette(k, k == choice))
                                         for k, f in enumerate(G.CAPTION_FIRST)])) for choice in (0, 1)]
    cap_table = put(struct.pack("<2I", *lists))
    side_records = []
    for v in variants:
        cblocks = [put(TL.encode_char_block(TL.encode_chars(sd.chars))) for sd in v]
        cl = put(TL.encode_char_list([(b, TL.CHAR_RAM[0] + TL.CHAR_SIZE * f) for b, f in zip(cblocks,
                                                                                             G.CAPTION_FIRST)]))
        areas = [tmap(v[side].framed if side == choice else v[side].plain) for choice in (0, 1) for side in (0, 1)]
        side_records.append(put(struct.pack("<5I", cl, *areas)))
    sides = put(struct.pack(f"<{len(S.ENTRIES)}I",
                            *(side_records[0 if G.course_clockwise(rs, e.course) else 1] for e in S.ENTRIES)))
    letters = G.reverse_letters(rs)
    records = []
    band_rows = G.BAND_TILE_ROWS[1] - G.BAND_TILE_ROWS[0]
    for entry in range(len(S.ENTRIES)):
        for mode in range(MODES):
            art = G.panel_art(rs, entry, mode, letters)
            blk = put(TL.encode_char_block(TL.encode_chars(art.chars)))
            cl = put(TL.encode_char_list([(blk, TL.CHAR_RAM[0] + TL.CHAR_SIZE * G.PANEL_FIRST)]))
            pl = put(_palette_list(G.PANEL_FIRST, art.palette, 2))
            pm = put(TL.encode_tile_map(TL.TileMap(0, True, G.PANEL_ROWS, G.PANEL_COLS, art.panel)))
            nb = put(TL.encode_tile_map(TL.TileMap(0, True, band_rows, G.PANEL_COLS, art.normal_band)))
            rb = put(TL.encode_tile_map(TL.TileMap(0, True, band_rows, G.PANEL_COLS, art.reverse_band)))
            records.append(struct.pack("<5I", cl, pl, pm, nb, rb))
    directory = put(b"".join(records))
    shown = {}
    for r in S.map_records(rs):
        shown.setdefault(r.first + r.position, r.objects)
    objects = put(struct.pack(f"<{len(S.ENTRIES)}I", *(shown[e] for e in range(len(S.ENTRIES)))))
    end = DATA_AT + len(out)
    if end > DATA_AT + DATA_SPAN:
        raise DirectionError(f"the screen's graphics ({end - DATA_AT} bytes) do not fit DATA_SPAN")
    return Layout(chars_at, palettes_at, banner_at, footer_at, cap_maps, cap_table, sides, directory,
                  C.id_table(rs, "SELECT_PANEL_OF_COURSE"), objects, end), bytes(out)


def code(lay: Layout = _PLACEHOLDER) -> i960.Block:
    """The direction code at CODE_AT for the graphics at `lay`: the wrapper and the three states (each called by the
    dispatcher: a frame of its own), then the two tables (TABLE_AT, LINKED_TABLE_AT). Its length is the same for every
    layout (every address a two-word MEMB operand)."""
    b = i960.Block(CODE_AT)
    g = "g13"

    b.label("dir7")                                     # the stock wipe, then the screen for a course id below 8
    b.ctrl("call", STOCK_STATE7)                        # (both tables: a cabinet racing alone, linked or not)
    b.mem("ldob", "r3", S._STATE, g)
    b.cobr("cmpobne", MISSION_SETUP, "r3", "dir7_ret")
    b.mem("ldob", "r3", S.COURSE_ID)
    b.cobr("cmpoble", REVERSE_OFFSET, "r3", "dir7_ret")  # 8 <= id: the test row's reverse ids skip it
    b.mem("ldob", "r3", LINKED)
    b.cobr("cmpobe", 0, "r3", "dir7_on")
    b.mem("ldob", "r3", VOTES)
    b.cobr("cmpobl", 1, "r3", "dir7_ret")               # linked with another cabinet's vote: no screen
    b.label("dir7_on")
    b.reg("mov", STATE_SETUP, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.label("dir7_ret")
    b.ctrl("ret")

    b.label("dir17")                                    # set-up: the art, NORMAL, not armed, the timer
    b.mem("ld", "r3", TIMER)                            # Mission Select's own timer, for state 19
    b.mem("st", "r3", SAVED_TIMER)
    b.mem("lda", "g0", lay.chars)
    b.ctrl("call", UPLOAD_CHARS)
    b.mem("lda", "g0", lay.palettes)
    b.ctrl("call", UPLOAD_PALETTES)
    b.mem("lda", "g0", lay.banner)
    b.mem("lda", "g1", BANNER_DST)
    b.ctrl("call", BLIT)
    b.mem("lda", "g0", COURSE_BOX)
    b.mem("lda", "g1", BOX_DST)
    b.ctrl("call", BLIT)
    b.mem("lda", "g0", lay.footer)
    b.mem("lda", "g1", TL.TILE_RAM[0] + 2 * FOOTER_ROWS[0] * TL.TILE_LAYER)
    b.ctrl("call", BLIT)
    for k in (0, 1):
        b.mem("lda", "g0", lay.captions[k])
        b.mem("lda", "g1", CAPTION_DST[k])
        b.ctrl("call", BLIT)
    b.mem("ldob", "r3", S.COURSE_ID)                    # the entry's panel in this game mode
    b.mem("ld", "r4", lay.panel_of_course, None, "r3", 4)
    b.reg("mulo", MODES, "r4", "r4")
    b.mem("ldos", "r5", GAME_MODE)
    b.reg("addo", "r5", "r4", "r4")
    b.mem("lda", "r5", RECORD)
    b.reg("mulo", "r5", "r4", "r4")
    b.mem("lda", "r5", lay.directory)
    b.reg("addo", "r5", "r4", "r4")
    b.mem("st", "r4", DIR_PTR)
    b.mem("ldob", "r3", S.COURSE_ID)
    b.mem("ld", "r5", lay.panel_of_course, None, "r3", 4)
    b.mem("ld", "r6", lay.sides, None, "r5", 4)
    b.mem("st", "r6", DIR_SIDES)
    b.mem("ld", "r5", lay.objects, None, "r5", 4)
    b.mem("st", "r5", DIR_OBJECTS)
    b.mem("ld", "g0", None, "r6")                       # the arrow variant's chars (both sides)
    b.ctrl("call", UPLOAD_CHARS)
    b.mem("ld", "r4", DIR_PTR)
    b.mem("ld", "g0", 0, "r4")
    b.ctrl("call", UPLOAD_CHARS)
    b.mem("ld", "r4", DIR_PTR)
    b.mem("ld", "g0", 4, "r4")
    b.ctrl("call", UPLOAD_PALETTES)
    b.mem("ld", "r4", DIR_PTR)
    b.mem("ld", "g0", 8, "r4")
    b.mem("lda", "g1", PANEL_DST)
    b.ctrl("call", BLIT)
    b.reg("mov", 0, "r3")
    b.mem("stob", "r3", SHOWN)                          # the panel's map shows NORMAL's band
    b.mem("stob", "r3", CHOICE)
    b.mem("stob", "r3", ARMED)
    b.ctrl("call", "dir_side")                          # NORMAL framed, its caption yellow
    b.mem("lda", "r3", SCREEN_FRAMES)
    b.mem("st", "r3", TIMER)
    b.reg("mov", STATE_RUN, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.ctrl("ret")

    b.label("dir_maps")                                 # the course map at both places (0x226438's commands)
    b.ctrl("call", FRAME_SETUP)
    for place in MAP_PLACES:
        b.mem("lda", "g14", 0x505)
        b.mem("st", "g14", 0x50, "g11")
        b.reg("mov", 0, "g14")
        b.mem("lda", "g14", 0x1010)
        b.mem("st", "g14", 0x100, "g11")
        b.reg("mov", 0, "g14")
        for reg_, word in zip(("g1", "g2", "g3"), place):
            b.mem("lda", reg_, word)
        b.reg("shlo", 14, 1, "g4")
        b.reg("shlo", 15, 1, "g5")
        b.mem("lda", "g6", MAP_SCALE)
        b.mem("lda", "g14", 0x1212)
        b.mem("st", "g14", 0x120, "g11")
        for reg_ in ("g1", "g2", "g3"):
            b.mem("st", reg_, None, "g11", "g12", 1)
        b.reg("mov", 0, "g14")
        b.mem("lda", "g14", 0x1414)
        b.mem("st", "g14", 0x140, "g11")
        b.mem("st", "g4", None, "g11", "g12", 1)
        b.reg("mov", 0, "g14")
        b.mem("lda", "g14", 0x1515)
        b.mem("st", "g14", 0x150, "g11")
        b.mem("st", "g5", None, "g11", "g12", 1)
        b.reg("mov", 0, "g14")
        b.mem("lda", "g14", 0x1313)
        b.mem("st", "g14", 0x130, "g11")
        for _ in range(3):
            b.mem("st", "g6", None, "g11", "g12", 1)
        b.reg("mov", 0, "g14")
        b.mem("ld", "g0", DIR_OBJECTS)
        b.ctrl("call", DRAW_OBJECTS)
        b.mem("lda", "g14", 0x606)
        b.mem("st", "g14", 0x60, "g11")
        b.reg("mov", 0, "g14")
    b.ctrl("ret")

    b.label("dir_side")                                 # the chosen side: framed, its caption yellow
    b.mem("ldob", "r6", CHOICE)
    b.mem("ld", "g0", lay.caption_palettes, None, "r6", 4)
    b.ctrl("call", UPLOAD_PALETTES)
    for side in (0, 1):
        b.mem("ldob", "r6", CHOICE)
        b.mem("ld", "r7", DIR_SIDES)
        b.mem("ld", "g0", 4 + 4 * side, "r7", "r6", 8)  # the area map [choice][side]
        b.mem("lda", "g1", AREA_DST[side])
        b.ctrl("call", BLIT)
    b.ctrl("ret")

    b.label("dir18")                                    # the maps, the choice, the gas, the timer
    b.ctrl("call", "dir_maps")
    b.ctrl("call", DRAW_TIMER)
    b.mem("ld", "r3", WHEEL_A)
    b.mem("ld", "r4", WHEEL_B)
    b.reg("addi", "r3", "r4", "r3")
    b.reg("shri", 1, "r3", "r3")
    b.mem("ldob", "r5", ARMED)
    b.cobr("cmpobne", 0, "r5", "dir18_armed")
    b.cobr("cmpibl", ARM_AT, "r3", "dir18_gas")         # ARM_AT < w: not yet at centre or left
    b.reg("mov", 1, "r5")
    b.mem("stob", "r5", ARMED)
    b.label("dir18_armed")
    b.reg("mov", 0, "r6")
    b.cobr("cmpibge", CHOOSE_AT, "r3", "dir18_set")     # w <= CHOOSE_AT: NORMAL
    b.reg("mov", 1, "r6")
    b.label("dir18_set")
    b.mem("stob", "r6", CHOICE)
    b.mem("ldob", "r7", SHOWN)
    b.cobr("cmpobe", "r6", "r7", "dir18_gas")           # the band already shows it
    b.mem("stob", "r6", SHOWN)
    b.mem("ld", "r4", DIR_PTR)
    b.mem("ld", "g0", 12, "r4", "r6", 4)                # the band map: normal (+12) or REVERSE (+16)
    b.mem("lda", "g1", BAND_DST)
    b.ctrl("call", BLIT)
    b.ctrl("call", "dir_side")
    b.label("dir18_gas")
    b.mem("ld", "r3", TIMER)
    b.mem("lda", "r4", GAS_FROM)
    b.reg("cmpi", "r3", "r4")
    b.ctrl("bg", "dir18_tick")                          # the timer above GAS_FROM: the gas not yet
    b.mem("ld", "r3", GAS)
    b.cobr("cmpobne", 0, "r3", "dir18_confirm")
    b.label("dir18_tick")
    b.mem("ld", "r3", TIMER)
    b.reg("subo", 1, "r3", "r3")
    b.mem("st", "r3", TIMER)
    b.cobr("cmpobl", 0, "r3", "dir18_ret")              # 0 < timer: on
    b.label("dir18_confirm")
    b.mem("lda", "g0", CONFIRM_SOUND)
    b.ctrl("call", SOUND)
    b.mem("lda", "r3", HOLD_FRAMES)
    b.mem("stos", "r3", HOLD, g)
    b.reg("mov", STATE_HOLD, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.label("dir18_ret")
    b.ctrl("ret")

    b.label("dir19")                                    # the hold (the maps still drawn), then the id and Mission Select
    b.ctrl("call", "dir_maps")
    b.mem("ldos", "r3", HOLD, g)
    b.reg("subo", 1, "r3", "r3")
    b.mem("stos", "r3", HOLD, g)
    b.cobr("cmpobl", 0, "r3", "dir19_ret")
    b.mem("ldob", "r3", CHOICE)
    b.cobr("cmpobe", 0, "r3", "dir19_mission")
    b.mem("ldob", "r3", S.COURSE_ID)
    b.reg("addo", REVERSE_OFFSET, "r3", "r3")
    b.mem("stob", "r3", S.COURSE_ID)
    b.ctrl("call", C.WIDE_CODE)                         # widen_ids' update: the id's shadows (and view_ids' copy)
    b.label("dir19_mission")                            # Mission Select's layers as state 7 leaves them
    b.ctrl("call", MISSION_LAYERS)
    b.mem("ld", "r3", SAVED_TIMER)
    b.mem("st", "r3", TIMER)                            # Mission Select's timer, as state 7 left it
    b.reg("mov", MISSION_SETUP, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.label("dir19_ret")
    b.ctrl("ret")

    b.label("table")
    b.code += bytes(4 * STATES)                         # placeholders: _tables fills both from the set's own
    b.label("table_linked")
    return b


def _tables(rs: RomSet, blk: i960.Block) -> bytes:
    """The single and the linked table, each its stock 17 handlers with state 7 the wrapper, then states 17-19."""
    out = b""
    for at in (STOCK_TABLE, LINKED_TABLE):
        stock = list(struct.unpack(f"<{STOCK_STATES}I", rs.read_i960(at, 4 * STOCK_STATES)))
        stock[7] = blk.labels["dir7"]
        out += struct.pack(f"<{STATES}I", *stock, *(blk.labels[f"dir{k}"] for k in (STATE_SETUP, STATE_RUN,
                                                                                     STATE_HOLD)))
    return out


_BLOCK = code()
TABLE_AT = _BLOCK.labels["table"]
LINKED_TABLE_AT = _BLOCK.labels["table_linked"]


def patched(rs: RomSet) -> bool:
    """True when the dispatcher reads the direction table."""
    if not isinstance(rs, RomSet):
        raise DirectionError(f"{type(rs).__name__} is not a RomSet")
    return struct.unpack("<I", rs.read_i960(DISPATCH_WORD, 4))[0] == TABLE_AT


def patch_direction(rs: RomSet) -> RomSet:
    """`rs` (with select.patch_select and course.widen_ids) with the direction screen: the code and the new table at
    CODE_AT (free bytes, 0xff), the dispatcher's word at DISPATCH_WORD repointed. DirectionError for a set without the
    select patch or widen_ids, a set already patched, a dispatcher word that is not the stock table's, or code space
    that is not free."""
    if not isinstance(rs, RomSet):
        raise DirectionError(f"{type(rs).__name__} is not a RomSet")
    if not S.patched(rs):
        raise DirectionError("the direction screen needs the select patch (select.patch_select) first")
    if not C.widened(rs):
        raise DirectionError("the direction screen needs widen_ids (its shadows' update) first")
    if patched(rs):
        raise DirectionError("the set already has the direction screen")
    for word, table in ((DISPATCH_WORD, STOCK_TABLE), (LINKED_WORD, LINKED_TABLE)):
        if struct.unpack("<I", rs.read_i960(word, 4))[0] != table:
            raise DirectionError(f"the dispatcher's word at {word:#x} is not the stock table's ({table:#x})")
    lay, data = graphics(rs)
    if rs.read_i960(DATA_AT, len(data)) != b"\xff" * len(data):
        raise DirectionError(f"the data bytes at {DATA_AT:#x} (+{len(data)}) are not free")
    blk = code(lay)
    body = blk.bytes()[:TABLE_AT - CODE_AT] + _tables(rs, blk)
    if not CODE_AT + len(body) <= S.CODE_FREE[1]:
        raise DirectionError(f"the direction code ({len(body)} bytes) does not fit the select's program run")
    if rs.read_i960(CODE_AT, len(body)) != b"\xff" * len(body):
        raise DirectionError(f"the program bytes at {CODE_AT:#x} (+{len(body)}) are not free")
    return rs.with_patches({C.rom_word_at(DATA_AT): data, C.rom_word_at(CODE_AT): body,
                            C.rom_word_at(DISPATCH_WORD): struct.pack("<I", TABLE_AT),
                            C.rom_word_at(LINKED_WORD): struct.pack("<I", LINKED_TABLE_AT)})
