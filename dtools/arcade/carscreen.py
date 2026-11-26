"""The Car Select screen (v0.4 stage 2; spec docs/superpowers/specs/2026-11-26-dreamcast-cars-design.md, plan
docs/superpowers/plans/2026-11-26-cars-stage2-screen.md): after Course Direction, the five cars in a row (the
arcade's Hornet, the Dreamcast's Grasshopper, Falcon, Lightning and Unicorn), before Mission Select.

Built on the Course Direction screen (dirscreen.py), the same way. Its two handler tables are copied once more with
three states: state 19 (the direction screen's hold) is wrapped (`car19`: when it hands over to Mission Select, state
20 instead), 20 sets the screen up, 21 runs it, 22 holds after the choice and hands over as state 19 does
(dirscreen.MISSION_LAYERS, the saved timer, state 8). State 7's wrapper is wrapped too (`car7`): it clears the choice
before every game's direction screen, so a game always starts from the Hornet. The screen therefore shows exactly
when Course Direction does.

The cars are drawn as objects, as the direction screen draws the course map: each its body list and its four wheels
on their hubs (the Hornet: the arcade's own, course.CAR_BODIES' entry 2 and the tables at course.CAR_WHEEL_TABLES[0]
and CAR_HUB_TABLE; a Dreamcast car: its row of the cars' table, course.CAR_TABLE_ADDR), side by side at CAR_X, a
little from above; the chosen one turns. The frame's polygon cap is raised for the screen (five cars are about 7,000
polygons); the race's culling writes it again every frame.

The choice: w as Mission Select reads the wheel ((0x501050 + 0x50105c) >> 1); the Hornet while w <= BANDS[0] (centre
or left: the idle wheel), then one car per band to the right. The gas counts once the timer is at most
dirscreen.GAS_FROM; the timer's end takes what is chosen. The choice is the byte course.CAR_CHOICE_RAM, which the
car choice's hooks read (course.car_choice with default 0).

CarScreenError (a DecodeError) for a set the patch does not fit."""

from __future__ import annotations

import struct
from typing import NamedTuple

import numpy as np

from dtools.arcade import cargfx as CG
from dtools.arcade import course as C
from dtools.arcade import dirgfx as G
from dtools.arcade import dirscreen as D
from dtools.arcade import i960
from dtools.arcade import select as S
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet


class CarScreenError(DecodeError):
    pass


STATE_SETUP, STATE_RUN, STATE_HOLD = 20, 21, 22
STATES = STATE_HOLD + 1
CODE_AT = 0x0003F800                # romset.RESERVED: program ROM's free tail, before the car choice's hooks
CODE_SIZE = 0x700
DATA_AT = C.DATA_BASE + 0x01FF0000  # romset.RESERVED: the custom data run, past the direction screen's graphics
DATA_SPAN = 0x4000
CARS = 5
CHOICE = C.CAR_CHOICE_RAM           # byte: the chosen car
SHOWN = CHOICE + 1                  # byte: the choice the names show (0xff: none yet)
ANGLE = CHOICE + 2                  # u16: the chosen car's turn
BANDS = (20, 47, 74, 101)           # w past BANDS[k - 1]: car k (measured: w is 0 to paddle 133, then 1.7 a paddle
#                                     unit, 127 from paddle 208 on: five bands of about 27)
SPIN = 0x180                        # the chosen car's turn a frame (0x10000 a full turn)
REST_ANGLE = 0x6000                 # the others' heading: three-quarter, nose to the left
TILT = 0x0A00                       # every car seen a little from above
CLEAR_LAYER0 = 0x0001BB2C           # Mission Select's layer-0 clear (0x225ae4)
STEP_BOX, STEP_BOX_DST = 0x02835454, 0x010012CA    # Mission Select's right box, STEP TO START (0x226ae0)
HORNET_BODY = 0x02848B10            # course.CAR_BODIES' entry 2: the Hornet's body list (read from the set at patch)
HORNET_PARTS = (C.CAR_PARTS_LIST, 0x028487C8, 0x028487DC)
"""The lists the race draws after the Hornet's body (0x7d7c-0x7d9c): the common one, part A of selector 2
(0x0281e39c's entry 2) and the first cabinet's number (0x0281e3b4's entry 0)."""
DRAW_PARTS = 0x00017CF8             # the draw routine those use
BANNER_ROW = 1
NAME_ROW = 31
SCREEN_COLS = 62                    # visible tile columns (496 pixels)


def _f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


CAR_SPACING, CAR_Y, CAR_Z, CAR_SCALE = 3.18, -0.6, 9.0, 0.62
"""The cars' places: x = (k - 2) * CAR_SPACING at y, z (31.1 pixels a unit at z 9, dirscreen.MAP_Z: 99 pixels
apart), each scaled so that its length is about 84 pixels."""
CAR_X = tuple(_f32((k - 2) * CAR_SPACING) for k in range(CARS))


def _cell(row: int, col: int) -> int:
    return TL.TILE_RAM[0] + 2 * (row * TL.TILE_LAYER + col)


BAR_DST = _cell(37, 5)               # the bars: the bottom left, beside STEP TO START
NAME_DST = tuple(_cell(NAME_ROW, min(SCREEN_COLS - CG.NAME_COLS, max(0, round((248 + 31.1 * (k - 2) * CAR_SPACING) / 8)
                                                                      - CG.NAME_COLS // 2))) for k in range(CARS))


class Layout(NamedTuple):
    """The screen's graphics in data ROM (i960 addresses): the char list and the palette list (the banner and the
    names' two blocks), the banner's map and where it goes, the names' maps (u32[CARS][2]: plain, chosen), their
    places (u32[CARS]), the cars' x (f32[CARS]) and the end."""
    chars: int
    palettes: int
    banner: int
    banner_dst: int
    names: int
    name_dst: int
    bars: int                       # u32[CARS]: each car's bars' map
    car_x: int
    end: int


_PLACEHOLDER = Layout(*(DATA_AT,) * 9)


def graphics(rs: RomSet) -> tuple[Layout, bytes]:
    """The screen's graphics data for `rs`, laid out from DATA_AT."""
    out = bytearray()

    def put(blob: bytes) -> int:
        at = DATA_AT + len(out)
        out.extend(blob)
        out.extend(b"\0" * (-len(out) % 4))
        return at

    try:
        banner = G.tiled(CG.banner_pens(rs), CG.CHAR_FIRST)
        names = G.tiled(np.concatenate([CG.name_pens(rs, k) for k in range(CARS)], 0), CG.NAME_FIRST[0])
    except DecodeError as e:
        raise CarScreenError(str(e)) from e
    drive = [struct.unpack("<f", rs.read_i960(C.CAR_HANDLING_ADDR + 32 * k + 16, 4))[0] for k in range(CARS)]
    if not all(1.0 < d < 30.0 for d in drive):
        raise CarScreenError(f"the cars' handling rows are not filed (first-gear drives {drive!r:.60})")
    bar_pens = [CG.bars_pens(rs, (CG.segments(CG.TOP_SPEED[k], *CG.SPEED_RANGE), CG.segments(drive[k], *CG.ACCEL_RANGE)))
                for k in range(CARS)]
    bars = G.tiled(np.concatenate(bar_pens, 0), CG.NAME_FIRST[1] + len(names.chars))
    if CG.CHAR_FIRST + len(banner.chars) > CG.NAME_FIRST[0] or len(names.chars) + len(bars.chars) > G.BLOCK:
        raise CarScreenError(f"the banner's {len(banner.chars)} or the names' {len(names.chars)} chars do not fit "
                             f"their blocks")
    bblock = put(TL.encode_char_block(TL.encode_chars(banner.chars)))
    nblock = put(TL.encode_char_block(TL.encode_chars(np.concatenate([names.chars, bars.chars], 0))))
    chars_at = put(TL.encode_char_list([(bblock, TL.CHAR_RAM[0] + TL.CHAR_SIZE * CG.CHAR_FIRST)] +
                                       [(nblock, TL.CHAR_RAM[0] + TL.CHAR_SIZE * f) for f in CG.NAME_FIRST]))
    bpal = tuple(G.banner_palette(rs)) * ((len(banner.chars) + G.BLOCK - 1) // G.BLOCK)
    palettes_at = put(TL.encode_palette_list(
        [TL.PaletteUpload(TL.PALETTE_RAM[0] + 32 * (CG.CHAR_FIRST >> 7), bpal)] +
        [TL.PaletteUpload(TL.PALETTE_RAM[0] + 32 * (f >> 7), CG.name_palette(bool(k)))
         for k, f in enumerate(CG.NAME_FIRST)]))
    banner_at = put(TL.encode_tile_map(TL.TileMap(0, True, banner.rows, banner.cols, banner.tiles)))
    banner_dst = _cell(BANNER_ROW, (SCREEN_COLS - banner.cols) // 2)
    maps = []
    for k in range(CARS):
        row = names.tiles[k * CG.NAME_COLS:(k + 1) * CG.NAME_COLS]
        for chosen in (0, 1):
            maps.append(put(TL.encode_tile_map(TL.TileMap(-0x8000, True, 1, CG.NAME_COLS, tuple(
                t + (CG.NAME_FIRST[1] - CG.NAME_FIRST[0]) * chosen for t in row)))))
    names_at = put(struct.pack(f"<{2 * CARS}I", *maps))
    per = len(bars.tiles) // CARS
    bars_at = put(struct.pack(f"<{CARS}I", *(put(TL.encode_tile_map(TL.TileMap(
        -0x8000, True, bars.rows // CARS, bars.cols, bars.tiles[k * per:(k + 1) * per]))) for k in range(CARS))))
    dst_at = put(struct.pack(f"<{CARS}I", *NAME_DST))
    x_at = put(struct.pack(f"<{CARS}I", *CAR_X))
    end = DATA_AT + len(out)
    if end > DATA_AT + DATA_SPAN:
        raise CarScreenError(f"the screen's graphics ({end - DATA_AT} bytes) do not fit DATA_SPAN")
    return Layout(chars_at, palettes_at, banner_at, banner_dst, names_at, dst_at, bars_at, x_at, end), bytes(out)


def _geo(b: i960.Block, op: int, *regs: str) -> None:
    """One geometry command as the game's own code writes it: the opcode word through g14, then its operands."""
    b.mem("lda", "g14", op | op << 8)
    b.mem("st", "g14", op << 4, "g11")
    for r in regs:
        b.mem("st", r, None, "g11", "g12", 1)
    b.reg("mov", 0, "g14")


PUSH, POP, IDENTITY, TRANSLATE, SCALE, ROT_X, ROT_Y = 0x05, 0x06, 0x10, 0x12, 0x13, 0x14, 0x15


def code(lay: Layout = _PLACEHOLDER, cap: int = C.CAP_MAX - 2000) -> i960.Block:
    """The screen's code at CODE_AT for the graphics at `lay`, then the two tables' room. Its length is the same for
    every layout."""
    b = i960.Block(CODE_AT)
    g = "g13"

    b.label("car7")                                     # every game: the Hornet, then the direction screen's wrapper
    b.reg("mov", 0, "r3")
    b.mem("stob", "r3", CHOICE)
    b.ctrl("b", D._BLOCK.labels["dir7"])

    b.label("car19")                                    # the direction screen's hold; its hand-over comes here
    b.ctrl("call", D._BLOCK.labels["dir19"])
    b.mem("ldob", "r3", S._STATE, g)
    b.cobr("cmpobne", D.MISSION_SETUP, "r3", "car19_ret")
    b.reg("mov", STATE_SETUP, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.label("car19_ret")
    b.ctrl("ret")

    b.label("car20")                                    # set-up: the layers, the art, the cap, the timer
    b.ctrl("call", CLEAR_LAYER0)
    b.mem("lda", "g0", lay.chars)
    b.ctrl("call", D.UPLOAD_CHARS)
    b.mem("lda", "g0", lay.palettes)
    b.ctrl("call", D.UPLOAD_PALETTES)
    b.mem("lda", "g0", lay.banner)
    b.mem("lda", "g1", lay.banner_dst)
    b.ctrl("call", D.BLIT)
    b.mem("lda", "g0", STEP_BOX)
    b.mem("lda", "g1", STEP_BOX_DST)
    b.ctrl("call", D.BLIT)
    b.mem("lda", "r3", cap)
    b.mem("st", "r3", C.POLYGON_CAP_RAM)
    b.reg("mov", 0, "r3")
    b.mem("stob", "r3", CHOICE)
    b.mem("stos", "r3", ANGLE)
    b.reg("subo", 1, 0, "r3")
    b.mem("stob", "r3", SHOWN)                          # 0xff: the first frame draws the names
    b.mem("lda", "r3", D.SCREEN_FRAMES)
    b.mem("st", "r3", D.TIMER)
    b.reg("mov", STATE_RUN, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.ctrl("ret")

    b.label("car_draw")                                 # the five cars, each its body and four wheels
    b.ctrl("call", D.FRAME_SETUP)
    b.reg("mov", 0, "r8")
    b.label("draw_car")
    b.cobr("cmpobne", 0, "r8", "draw_dc")
    b.mem("lda", "r9", HORNET_BODY)
    b.mem("lda", "r10", C.CAR_WHEEL_TABLES[0])
    b.mem("lda", "r11", C.CAR_HUB_TABLE)
    b.ctrl("b", "draw_have")
    b.label("draw_dc")
    b.reg("shlo", 4, "r8", "r3")
    b.mem("ld", "r9", C.CAR_TABLE_ADDR, "r3")
    b.mem("ld", "r10", C.CAR_TABLE_ADDR + 8, "r3")
    b.mem("ld", "r11", C.CAR_TABLE_ADDR + 12, "r3")
    b.label("draw_have")
    _geo(b, PUSH)
    _geo(b, IDENTITY)
    b.mem("ld", "g1", lay.car_x, None, "r8", 4)
    b.mem("lda", "g2", _f32(CAR_Y))
    b.mem("lda", "g3", _f32(CAR_Z))
    _geo(b, TRANSLATE, "g1", "g2", "g3")
    b.mem("lda", "g4", TILT)
    _geo(b, ROT_X, "g4")
    b.mem("lda", "g5", REST_ANGLE)
    b.mem("ldob", "r3", CHOICE)
    b.cobr("cmpobne", "r3", "r8", "draw_angle")
    b.mem("ldos", "g5", ANGLE)
    b.label("draw_angle")
    _geo(b, ROT_Y, "g5")
    b.mem("lda", "g6", _f32(CAR_SCALE))
    _geo(b, SCALE, "g6", "g6", "g6")
    b.reg("mov", "r9", "g0")
    b.ctrl("call", D.DRAW_OBJECTS)
    b.cobr("cmpobne", 0, "r8", "draw_wheels")
    for part in HORNET_PARTS:                           # the Hornet's roof, windows and number: its own part lists
        b.mem("lda", "g0", part)
        b.ctrl("call", DRAW_PARTS)
    b.label("draw_wheels")
    b.reg("mov", 0, "r13")
    b.label("draw_wheel")
    _geo(b, PUSH)
    b.reg("mulo", 12, "r13", "r3")
    b.reg("addo", "r11", "r3", "r3")
    b.mem("ld", "g1", 0, "r3")
    b.mem("ld", "g2", 4, "r3")
    b.mem("ld", "g3", 8, "r3")
    _geo(b, TRANSLATE, "g1", "g2", "g3")
    b.mem("ld", "g0", None, "r10", "r13", 4)
    b.ctrl("call", D.DRAW_OBJECTS)
    _geo(b, POP)
    b.reg("addo", 1, "r13", "r13")
    b.cobr("cmpobg", 4, "r13", "draw_wheel")
    _geo(b, POP)
    b.reg("addo", 1, "r8", "r8")
    b.cobr("cmpobg", CARS, "r8", "draw_car")
    b.ctrl("ret")

    b.label("car_names")                                # every name, the chosen one's from the yellow block
    b.reg("mov", 0, "r8")
    b.label("names_next")
    b.reg("shlo", 1, "r8", "r3")
    b.mem("ldob", "r4", CHOICE)
    b.cobr("cmpobne", "r4", "r8", "names_plain")
    b.reg("addo", 1, "r3", "r3")
    b.label("names_plain")
    b.mem("ld", "g0", lay.names, None, "r3", 4)
    b.mem("ld", "g1", lay.name_dst, None, "r8", 4)
    b.ctrl("call", D.BLIT)
    b.reg("addo", 1, "r8", "r8")
    b.cobr("cmpobg", CARS, "r8", "names_next")
    b.mem("ldob", "r4", CHOICE)                         # the chosen car's bars
    b.mem("ld", "g0", lay.bars, None, "r4", 4)
    b.mem("lda", "g1", BAR_DST)
    b.ctrl("call", D.BLIT)
    b.ctrl("ret")

    b.label("car21")                                    # the cars, the wheel's band, the gas, the timer
    b.ctrl("call", "car_draw")
    b.ctrl("call", D.DRAW_TIMER)
    b.mem("ldos", "r3", ANGLE)
    b.mem("lda", "r4", SPIN)
    b.reg("addo", "r4", "r3", "r3")
    b.mem("stos", "r3", ANGLE)
    b.mem("ld", "r3", D.WHEEL_A)
    b.mem("ld", "r4", D.WHEEL_B)
    b.reg("addi", "r3", "r4", "r3")
    b.reg("shri", 1, "r3", "r3")
    b.reg("mov", 0, "r6")
    for k, edge in enumerate(BANDS):
        b.mem("lda", "r5", edge)
        b.cobr("cmpibge", "r5", "r3", "car21_set")      # w <= edge: the band below
        b.reg("mov", k + 1, "r6")
    b.label("car21_set")
    b.mem("stob", "r6", CHOICE)
    b.mem("ldob", "r7", SHOWN)
    b.cobr("cmpobe", "r6", "r7", "car21_gas")
    b.mem("stob", "r6", SHOWN)
    b.ctrl("call", "car_names")
    b.label("car21_gas")
    b.mem("ld", "r3", D.TIMER)
    b.mem("lda", "r4", D.GAS_FROM)
    b.reg("cmpi", "r3", "r4")
    b.ctrl("bg", "car21_tick")
    b.mem("ld", "r3", D.GAS)
    b.cobr("cmpobne", 0, "r3", "car21_confirm")
    b.label("car21_tick")
    b.mem("ld", "r3", D.TIMER)
    b.reg("subo", 1, "r3", "r3")
    b.mem("st", "r3", D.TIMER)
    b.cobr("cmpobl", 0, "r3", "car21_ret")
    b.label("car21_confirm")
    b.mem("lda", "g0", D.CONFIRM_SOUND)
    b.ctrl("call", D.SOUND)
    b.mem("lda", "r3", D.HOLD_FRAMES)
    b.mem("stos", "r3", D.HOLD, g)
    b.reg("mov", STATE_HOLD, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.label("car21_ret")
    b.ctrl("ret")

    b.label("car22")                                    # the hold (the cars still drawn), then Mission Select
    b.mem("ldos", "r3", D.HOLD, g)
    b.reg("subo", 1, "r3", "r3")
    b.mem("stos", "r3", D.HOLD, g)
    b.cobr("cmpobe", 0, "r3", "car22_over")
    b.ctrl("call", "car_draw")
    b.ctrl("ret")
    b.label("car22_over")
    b.ctrl("call", D.FRAME_SETUP)                       # an empty list: the cars do not stay over Mission Select
    b.ctrl("call", D.MISSION_LAYERS)
    b.mem("ld", "r3", D.SAVED_TIMER)
    b.mem("st", "r3", D.TIMER)
    b.reg("mov", D.MISSION_SETUP, "r3")
    b.mem("stob", "r3", S._STATE, g)
    b.ctrl("ret")

    b.label("table")
    b.code += bytes(4 * STATES)
    b.label("table_linked")
    b.code += bytes(4 * STATES)
    return b


_BLOCK = code()
TABLE_AT = _BLOCK.labels["table"]
LINKED_TABLE_AT = _BLOCK.labels["table_linked"]


def _tables(rs: RomSet, blk: i960.Block) -> bytes:
    """The single and the linked table: the direction screen's twenty handlers with states 7 and 19 wrapped, then
    states 20-22."""
    out = b""
    for at in (D.TABLE_AT, D.LINKED_TABLE_AT):
        t = list(struct.unpack(f"<{D.STATES}I", rs.read_i960(at, 4 * D.STATES)))
        if t[7] != D._BLOCK.labels["dir7"] or t[D.STATE_HOLD] != D._BLOCK.labels["dir19"]:
            raise CarScreenError(f"the table at {at:#x} is not the direction screen's")
        t[7], t[D.STATE_HOLD] = blk.labels["car7"], blk.labels["car19"]
        out += struct.pack(f"<{STATES}I", *t, *(blk.labels[f"car{k}"] for k in (STATE_SETUP, STATE_RUN, STATE_HOLD)))
    return out


def patched(rs: RomSet) -> bool:
    """True when the dispatcher reads the Car Select table."""
    if not isinstance(rs, RomSet):
        raise CarScreenError(f"{type(rs).__name__} is not a RomSet")
    return struct.unpack("<I", rs.read_i960(D.DISPATCH_WORD, 4))[0] == TABLE_AT


def patch_cars(rs: RomSet, cap: int = C.CAP_MAX - 2000) -> RomSet:
    """`rs` (with the Course Direction screen and the car choice reading its byte: course.car_choice with default 0)
    with the Car Select screen: its graphics at DATA_AT, its code and tables at CODE_AT, the two dispatchers' words
    repointed. CarScreenError for a set without the direction screen or the car choice, a car choice with a forced
    car, a set already patched, a Hornet body list that is not the arcade's, or space that is not free."""
    if not isinstance(rs, RomSet):
        raise CarScreenError(f"{type(rs).__name__} is not a RomSet")
    if not D.patched(rs):
        raise CarScreenError("the Car Select screen needs the direction screen (dirscreen.patch_direction) first")
    if patched(rs):
        raise CarScreenError("the set already has the Car Select screen")
    if not C.car_chosen(rs):
        raise CarScreenError("the Car Select screen needs the car choice (course.car_choice) first")
    block0 = C.car_code(0, cap)[0].bytes()
    if rs.read_i960(C.CAR_CODE, len(block0)) != block0:
        raise CarScreenError("the car choice does not read the choice byte (course.car_choice with default 0 and "
                             "this cap)")
    if struct.unpack("<I", rs.read_i960(C.CAR_BODIES + 8, 4))[0] != HORNET_BODY:
        raise CarScreenError(f"the Hornet's body list is not {HORNET_BODY:#x}")
    lay, data = graphics(rs)
    if rs.read_i960(DATA_AT, len(data)) != b"\xff" * len(data):
        raise CarScreenError(f"the data bytes at {DATA_AT:#x} (+{len(data)}) are not free")
    blk = code(lay, cap)
    body = blk.bytes()[:TABLE_AT - CODE_AT] + _tables(rs, blk)
    if len(body) > CODE_SIZE:
        raise CarScreenError(f"the screen's code ({len(body)} bytes) does not fit CODE_SIZE {CODE_SIZE}")
    if rs.read_i960(CODE_AT, CODE_SIZE) != b"\xff" * CODE_SIZE:
        raise CarScreenError(f"the program bytes at {CODE_AT:#x} (+{CODE_SIZE:#x}) are not free")
    return rs.with_patches({C.rom_word_at(DATA_AT): data, C.rom_word_at(CODE_AT): body,
                            C.rom_word_at(D.DISPATCH_WORD): struct.pack("<I", TABLE_AT),
                            C.rom_word_at(D.LINKED_WORD): struct.pack("<I", LINKED_TABLE_AT)})
