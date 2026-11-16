"""Decode and encode the arcade Daytona USA per-course tables (Sega Model 2, i960).

Layouts: docs/formats/model2-course-tables.md; the runs that found them: docs/spike/2026-11-arcade-course-tables.md.
Per-course tables are indexed by the game's course id, the byte at work RAM 0x00501460: 0 Beginner, 1 Expert,
2 Advanced (not the Circuit Select panel order). Entries are returned in id order.

Scenery placement (`scenery`, `actors`): the course's visible geometry is a 16 x 16 grid of 128-unit cells, one object
list per cell (data ROM, through GRID_TABLES), plus actors spawned from per-cell actor lists (program window, through
ACTOR_TABLES) whose handlers draw their own objects. Both draw only when their cell is in the frame's visible set.

Road line and AI lines (`road_line`, `ai_lines`): the record table is the road line, one cross-section of LANES lane
points per segment (a closed loop, about 8 units apart); the line table holds AI_LINE_COUNT AI lines with one point
per segment. A car's segment (car +0x12c) indexes both. The road surface the cars stand on is not here: it is a
TGP-side table the i960 queries through TGP function 0x36 (milestone M1b).

Checkpoints, start grid, pit (`checkpoints`, `start_grid`, `pit`): per-id arrays outside the course record. A checkpoint
is a road segment; passing it adds a word of the time table (indexed by pass number) to the timer at 0x5010d0. The
grid is the spawn table (one record per car, shared by the courses) laid out from a per-course base segment. The pit
lane is the step-table polygons flagged STEP_PIT.

Select screen, backdrop, rankings (`select_assets`, `backdrop`, `rankings_layout`): the Circuit Select panels are
10 x 18 tile maps (TileMap) blitted into tile layer 0 from per-mode record tables in the program window, their chars
uploaded from data ROM by upload sets (UploadSet); the race horizon is a 256-column panorama of eight segments
streamed into tile layer 2, reached through CourseEntry.upload_list; the rankings are three 24-entry tables in backup
SRAM (sixteen on a ranked set, grow_rankings), described by a layout only.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

Point = tuple[float, float, float]

from dtools.arcade import i960, polyrom
from dtools.arcade.romset import RomSet, RomSetError
from dtools.arcade.tablebase import (CourseError, DATA_BASE, DATA_SIZE, DATA_WINDOW, PROGRAM_WINDOW, _S16, _U16,
                                     _data_off, _data_ptr_ok, _f32, _int, _read, _seq, _u32, data_size, is_int)
from dtools.arcade.tiles import (CHAR_MASK, CHAR_SIZE, TILE_LAYER, backdrop_segment, char_list, tile_map, upload_set,
                                 BackdropSegment, TileMap, UploadSet)


COURSE_POINTERS = 0x00004770
"""Program ROM (i960 address = program offset): u32[4], one pointer per course id to its course record. The loader
at 0x46e8 does `ld 0x4770[id*4]`, keeps the pointer at work RAM 0x5fe5e4 and copies record words 0-5 to
0x501400-0x501414. A fourth pointer (id 3) points at a stub record; course_table does not return it."""

COURSE_TABLE = 0x02600020
"""The course table, in the DATA ROM address space (i960 address; RomSet.data offset 0x600020): COURSE_ENTRY_SIZE
bytes per course id, eight little-endian u32 words. Words 0-5 are data-ROM pointers to the course's blobs (the
CourseEntry fields), words 6-7 are not read by any code found so far (kept as CourseEntry.raw). The 32 bytes before
it (0x02600000) are eight upload-list pointers for the non-course screens."""

COURSE_ENTRY_SIZE = 0x20
COURSE_COUNT = 3
COURSE_NAMES = ("Beginner", "Expert", "Advanced")       # by course id

# CourseEntry pointer fields in record-word order (words 0-5)
_POINTERS = ("upload_list", "step_table", "block_ref_a", "block_ref_b", "line_table", "record_table")
_RAW_WORDS = 2


@dataclass(frozen=True)
class CourseEntry:
    """One course record. Pointer fields are RomSet.data offsets (i960 address - 0x02000000). Field names are
    provisional, after what reads them (see docs/formats/model2-course-tables.md)."""
    course: int             # the game's course id (0 Beginner, 1 Expert, 2 Advanced)
    upload_list: int        # word 0: list of upload lists, run while the course loads (0x1de40, 0x1dfd0)
    step_table: int         # word 1: n, steps, then per road polygon (u16 segment, u16 flags) (0x4534 -> 0x501430)
    block_ref_a: int        # word 2: -> pointer to the 128 x 128 u16 ground-query grid, 32 KB (0x4668 -> 0x501428)
    block_ref_b: int        # word 3: -> pointer to one attribute byte per ground polygon (0x467c -> 0x50142c)
    line_table: int         # word 4: the AI lines: eight (pointer, count) pairs (0xe8a0 picks one, via 0x50144c)
    record_table: int       # word 5: the road line: n, 8, then n + 1 records of 164 bytes (0x4690 -> 0x501448)
    raw: tuple[int, ...]    # words 6-7, unknown (equal to *block_ref_a on the real set)

    @property
    def name(self) -> str:
        return course_name(self.course)

    def blobs(self) -> dict[str, int]:
        """The data-ROM offsets of the course's blobs, by field name."""
        return {n: getattr(self, n) for n in _POINTERS}


def course_name(course: int) -> str:
    """COURSE_NAMES[course] for ids 0-2, else "course <id>"."""
    return COURSE_NAMES[course] if is_int(course) and 0 <= course < len(COURSE_NAMES) else f"course {course}"


def _decode(rs: RomSet, course: int) -> CourseEntry:
    at = record_address(course)
    words = [_u32(rs, at + 4 * k) for k in range(COURSE_ENTRY_SIZE // 4)]
    offsets, end = [], DATA_BASE + data_size(rs)
    for k, (name, w) in enumerate(zip(_POINTERS, words)):
        if not DATA_BASE <= w < end:
            raise CourseError(f"course {course} ({course_name(course)}): {name} at {at + 4 * k:#x} is {w:#x}, "
                              f"outside data ROM {DATA_BASE:#x}-{end - 1:#x}")
        offsets.append(w - DATA_BASE)
    return CourseEntry(course, *offsets, raw=tuple(words[len(_POINTERS):]))


def course_table(rs: RomSet) -> list[CourseEntry]:
    """The course records, in course-id order: ids 0-2 on the original set, 0-15 on a grown one (id_count). The
    record of id `i` is at record_address(i) and COURSE_POINTERS[i] (id_table) must point there. CourseError (naming
    the address) when the game's pointer to a record is not where this decoder reads it, or when a blob pointer is
    outside data ROM."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    out, pointers = [], id_table(rs, "COURSE_POINTERS")
    for course in range(id_count(rs)):
        ptr_at = pointers + 4 * course
        want = record_address(course)
        if (ptr := _u32(rs, ptr_at)) != want:
            raise CourseError(f"course pointer at {ptr_at:#x} (id {course}) is {ptr:#x}, expected {want:#x}: "
                              f"the course table is not where expected")
        out.append(_decode(rs, course))
    return out


def encode_course_entry(e: CourseEntry) -> bytes:
    """The COURSE_ENTRY_SIZE bytes of a record. CourseError when a pointer field is not a data-ROM offset or `raw`
    is not two u32 words."""
    words = []
    for name in _POINTERS:
        off = getattr(e, name)
        if not is_int(off) or not 0 <= off < DATA_WINDOW:
            raise CourseError(f"{e.name}: {name} {off!r} is not a data-ROM offset (0-{DATA_WINDOW - 1:#x})")
        words.append(DATA_BASE + int(off))
    raw = e.raw
    if not isinstance(raw, (tuple, list)) or len(raw) != _RAW_WORDS or \
            not all(is_int(w) and 0 <= w <= 0xFFFFFFFF for w in raw):
        raise CourseError(f"{e.name}: raw must be {_RAW_WORDS} u32 words, got {raw!r}")
    return struct.pack("<8I", *words, *raw)


# --- scenery placement ---------------------------------------------------------------------------------------------

GRID_TABLES = 0x02805298
"""Data ROM: u32[4], per course id the address of its cell grid (the load code at 0x4534 stores it to 0x501420)."""
GRID_CELLS = 256
GRID_DIM = 16                   # cells per row; cell = row << 4 | col, row from z, col from x
CELL_UNITS = 128.0              # cell edge in world units
GRID_ORIGIN = -1024.0           # world x / z of cell 0's low corner
ACTOR_TABLES = 0x00233BFC
"""Program window: u32[4], per course id the address of 256 actor-list pointers (0x4534 stores it to 0x501444;
the spawner at 0x22002c walks them). A list is a u32 count, then `count` ACTOR_SIZE-byte records."""
ACTOR_SIZE = 24
ACTOR_POOL = 65
"""Actors the spawner at 0x22002c creates at most: the count at data 0x02802f74 (65 on the real set; Expert, the
largest course, has 63). golden.check_course refuses a Course with more."""
OBJECT_LIBRARY = (0x0283FBB4, 0x0284F4EC)
"""The bounds the draw routines 0x17acc and 0x17cf8 check an object-list pointer against (i960 addresses, end
exclusive). A list is OBJECT_ENTRY_SIZE-byte entries (oba, tpa, tha, budget) ended by an oba of 0."""
LIBRARY_BOUND_AT = (0x17AD0, 0x17CFC)
"""Program ROM offsets of the lower bound's word in the two draw routines: the displacement of `lda 0x283fbb4,r3`
at 0x17acc and at 0x17cf8. Each routine then loads the upper bound (`lda 0x284f4ec,r4`, its word 8 bytes on:
LIBRARY_END_AT) and returns at once when `cmpobl g0,r3` (the list below the lower bound) or `cmpobge g0,r4` (at or
past the upper). They are the only readers of the bounds (docs/spike/2026-11-convert.md, "Object library")."""
LIBRARY_END_AT = (0x17AD8, 0x17D04)
_LDA_R3, _LDA_R4 = 0x8C183000, 0x8C203000      # i960 `lda <abs32>,r3` / `lda <abs32>,r4`: the word before each bound
PATCHED_LIBRARY = (0x0265D120, 0x0284F4EC)
"""The bounds with LIBRARY_BOUND_PATCH: the lower bound moved down to the data ROM's free run (tgp.DATA_FREE's
start, i960 0x0265d120), so lists written there are drawn. Every list pointer of the original set lies inside
OBJECT_LIBRARY, so the originals draw as before (the MAME proof: docs/spike/2026-11-convert.md, "Object library")."""
LIBRARY_BOUND_PATCH = {at: (OBJECT_LIBRARY[0], PATCHED_LIBRARY[0]) for at in LIBRARY_BOUND_AT}
"""The object-library bound patch, program ROM offset -> (original word, patched word): the lower bound's word of
both draw routines (LIBRARY_BOUND_AT), 0x0283fbb4 -> 0x0265d120. Two program words. `patch_library_bound` applies
it; the converter's real scenery (build, scenery="model") writes its object lists into DATA_FREE and needs it."""
OBJECT_ENTRY_SIZE = 16
OBC = 5000
"""The object count the draw routines send with every object: the word at 0x5010f4, set to 0x1388 at 0x12f8. It is
also the frame's polygon budget the entries' `budget` words are summed against (0x5010e8). Not stored per object."""

STATIC_PROP = 0x220118          # handler: draws its obj list under V * T(position) * Ry(angle) (update 0x220138)
SWITCH = 0x220A54               # handler: draws one of three lists, chosen by race state, under V (update 0x220a74)
SWITCH_LISTS = (0x220ABC, 0x220AC8, 0x220AD4)   # the `lda <list>,g0` instructions holding the switch's three lists
VIEW_BILLBOARD = 0x221E40       # handler: obj is u32[64] of lists, one per view angle; draws one under V (0x221e60)
BILLBOARD_VARIANTS = 64
BILLBOARD_REF = 0x00235BB4
"""Program window: f32[3], the point the view-angle billboard's update (0x221e60) measures the camera's angle from
(`lda 0x235bb4,r8`: x at +0, z at +8, y not read): the variant is ((-a) >> 5) & 63 for a = TGP function 0x0f's
angle of (camera - point) in the XZ plane, 0x10000 a turn (atan2(dz, dx), traced: docs/spike/2026-11-mirror.md,
"Billboards"). (210, 0, -380) on the real set, just before Beginner's billboard array (0x235bc0); one point for
every VIEW_BILLBOARD actor (Beginner's is the only one)."""
_LDA_G0 = 0x8C803000            # i960 `lda <abs32>,g0`: the instruction word; the list address is the next word


@dataclass(frozen=True)
class Actor:
    """One actor record of a course's actor lists (copied by the spawner at 0x22002c into a work-RAM actor)."""
    slot: int               # index of the list in the course's 256 actor-list pointers
    addr: int               # i960 address of the 24-byte record (program window)
    position: Point         # +0: x, y, z floats (to actor +0x1c / +0x20 / +0x24)
    angle: int              # +12: s16 rotation about y (to +0x2a; 0x10000 = one turn)
    ident: int              # +14: u16, stored as a byte to +4 (a per-handler index)
    handler: int            # +16: the actor's init routine (program window), stored to +0xc
    obj: int                # +20: handler data (to +0x14): an object list, a list array, or other


@dataclass(frozen=True)
class Placement:
    """One object the game draws for the course, with what places and culls it."""
    oba: int                # object address (always a polygon-ROM object here; see check_rom_object)
    obc: int                # the object count sent with it: OBC on every object (not stored)
    tpa: int                # texture point address
    tha: int                # texture header address
    raw: tuple[int, ...]    # the entry's undecoded word: (budget,), word 3
    cell: int               # the grid cell whose visibility gates the draw (row << 4 | col)
    entry: int              # data-ROM offset of the 16-byte object entry
    in_world: bool          # True: vertices are world coordinates, drawn under the view matrix V; False: drawn
    #                         under V * T(actor.position) * Ry(actor.angle)
    actor: Actor | None = None   # None for a grid-cell object, else the actor that draws it

    @property
    def budget(self) -> int:
        """Entry word 3, added to the frame's polygon count (0x5010e8); an object is skipped once it exceeds OBC."""
        return self.raw[0]


def check_rom_object(oba: int, rs: RomSet | None = None) -> int:
    """The polygon-ROM word index of object address `oba`. CourseError for the polygon-RAM selectors (bit 24 set: RAM 1;
    bit 23 clear: RAM 0), an index past the polygon ROM (0x400000 words, the original's 16 MB; with `rs`, its
    RomSet.poly_mask() + 1 when larger: 0x800000 on a custom set; the hardware would wrap it), bits 25-31 set, a
    value that is not a u32, or `rs` neither None nor a RomSet."""
    if rs is not None and not isinstance(rs, RomSet):
        raise CourseError(f"check_rom_object: {type(rs).__name__} is not a RomSet")
    return _rom_object(oba, _POLY_WORDS if rs is None else max(_POLY_WORDS, rs.poly_mask() + 1))  # a synthetic set's
    #                                                                                                short image


def _rom_object(oba, words: int) -> int:
    if not is_int(oba) or not 0 <= oba <= 0xFFFFFFFF:
        raise CourseError(f"oba {oba!r} is not a u32")
    oba = int(oba)
    if oba & 0xFE000000:
        raise CourseError(f"oba {oba:#x} has bits 25-31 set; object addresses are 25 bits")
    if oba & 0x01000000:
        raise CourseError(f"oba {oba:#x} selects polygon RAM 1 (bit 24), not the polygon ROM")
    if not oba & 0x00800000:
        raise CourseError(f"oba {oba:#x} selects polygon RAM 0 (bit 23 clear), not the polygon ROM")
    index = oba & 0x7FFFFF
    if index >= words:
        raise CourseError(f"oba {oba:#x}: polygon-ROM word index {index:#x} is past the {4 * words >> 20} MB ROM "
                          f"(>= {words:#x})")
    return index


_POLY_WORDS = 0x400000          # the original set's polygon ROM in words (RomSet.poly_mask() + 1)
POLY_WINDOW = 0x800000
"""The polygon ROM's decoding limit in words (the 23-bit word index under `oba` bit 0x800000: a custom set's 32 MB,
docs/spike/2026-11-sixteen.md, "Room"). The encoders, which have no set, bound an object address by it
(`rom_object_index`), as the data encoders bound an address by DATA_WINDOW; RomSet.with_patch refuses a write past a
stock image."""


def rom_object_index(oba: int) -> int:
    """The polygon-ROM word index of object address `oba`, bounded by POLY_WINDOW (no set: the encoders'
    check). CourseError as check_rom_object."""
    return _rom_object(oba, POLY_WINDOW)


def cell_of(x: float, z: float) -> int:
    """The grid cell of world point (x, z), as 0x172a0 computes it: cvtri (round to nearest), + 1024, >> 7, & 15 on
    each axis (positions outside -1024..1023 wrap)."""
    col = (round(x) + 1024) >> 7 & (GRID_DIM - 1)
    row = (round(z) + 1024) >> 7 & (GRID_DIM - 1)
    return row << 4 | col


def cell_box(cell: int) -> tuple[float, float, float, float]:
    """(x0, z0, x1, z1): the world area of `cell` (half-open)."""
    if not is_int(cell) or not 0 <= cell < GRID_CELLS:
        raise CourseError(f"cell {cell!r} is not 0-{GRID_CELLS - 1}")
    x0 = GRID_ORIGIN + CELL_UNITS * (cell & 15)
    z0 = GRID_ORIGIN + CELL_UNITS * (cell >> 4)
    return x0, z0, x0 + CELL_UNITS, z0 + CELL_UNITS


def library_bounds(rs: RomSet) -> tuple[int, int]:
    """(lower, upper) object-list bounds the set's draw routines check (end exclusive): OBJECT_LIBRARY on the
    original set, PATCHED_LIBRARY with LIBRARY_BOUND_PATCH. Read from the two `lda` pairs at LIBRARY_BOUND_AT and
    LIBRARY_END_AT. CourseError for `rs` not a RomSet, a word before a bound that is not the `lda` instruction, or the
    two routines checking different bounds."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if len(rs.program) < max(LIBRARY_END_AT) + 4:
        raise CourseError(f"the program image ({len(rs.program):#x} bytes) does not hold the draw routines")
    got = set()
    for at in LIBRARY_BOUND_AT:             # each lower bound's lda, then the upper bound's (LIBRARY_END_AT: at + 8)
        ins_lo, lo, ins_hi, hi = struct.unpack_from("<4I", rs.program, at - 4)
        if (ins_lo, ins_hi) != (_LDA_R3, _LDA_R4):
            raise CourseError(f"program {at - 4:#x}: {ins_lo:#010x} {ins_hi:#010x} are not the draw routine's "
                              f"`lda <lower>,r3` / `lda <upper>,r4`")
        got.add((lo, hi))
    if len(got) != 1:
        raise CourseError(f"the draw routines 0x17acc and 0x17cf8 check different object-list bounds: "
                          f"{sorted(got)}")
    return got.pop()


def patch_library_bound(rs: RomSet) -> RomSet:
    """`rs` with LIBRARY_BOUND_PATCH applied (its two program words written), or `rs` itself when it is applied
    already. CourseError for `rs` not a RomSet or a bound word that is neither the original nor the patched one."""
    if library_bounds(rs) == PATCHED_LIBRARY:
        return rs
    patches = {}
    for at, (old, new) in LIBRARY_BOUND_PATCH.items():
        word = struct.unpack_from("<I", rs.program, at)[0]
        if word != old:
            raise CourseError(f"program {at:#x}: the object library's lower bound is {word:#x}, not the original "
                              f"{old:#x}: not patched")
        patches[("program", at)] = struct.pack("<I", new)
    return rs.with_patches(patches)


def _object_list(rs: RomSet, ptr: int, what: str, cell: int, in_world: bool, actor: Actor | None) -> list[Placement]:
    lo, hi = library_bounds(rs)
    if not lo <= ptr < hi:
        raise CourseError(f"{what} is {ptr:#x}, outside the object library {lo:#x}-{hi - 1:#x}")
    out = []
    at = ptr
    while True:
        if at + 4 > hi:
            raise CourseError(f"{what}: the object list at {ptr:#x} runs past the library end {hi:#x}")
        oba = _u32(rs, at)
        if oba == 0:
            return out
        if at + OBJECT_ENTRY_SIZE > hi:
            raise CourseError(f"{what}: the object list at {ptr:#x} runs past the library end {hi:#x}")
        try:
            check_rom_object(oba, rs)
        except CourseError as e:
            raise CourseError(f"{what}: entry at {at:#x}: {e}") from None
        tpa, tha, budget = (_u32(rs, at + 4 * k) for k in (1, 2, 3))
        out.append(Placement(oba, OBC, tpa, tha, (budget,), cell, at - DATA_BASE, in_world, actor))
        at += OBJECT_ENTRY_SIZE


def _per_course(rs: RomSet, table: int, entry: CourseEntry, lo: int, hi: int, what: str) -> int:
    at = table + 4 * entry.course
    ptr = _u32(rs, at)
    if not lo <= ptr < hi:
        raise CourseError(f"{entry.name}: {what} pointer at {at:#x} is {ptr:#x}, outside {lo:#x}-{hi - 1:#x}")
    return ptr


def actors(rs: RomSet, entry: CourseEntry) -> list[Actor]:
    """The course's actor records, in spawn order (slot by slot, then list order). CourseError naming the address
    when the table or a list pointer is outside the program window or a list's count is implausible."""
    table = _per_course(rs, id_table(rs, "ACTOR_TABLES"), entry, _WINDOW[0], _WINDOW[1] - 4 * GRID_CELLS,
                        "actor table")
    out = []
    for slot in range(GRID_CELLS):
        at = table + 4 * slot
        lst = _u32(rs, at)
        if lst == 0:
            continue
        if not _WINDOW[0] <= lst < _WINDOW[1] - 4:
            raise CourseError(f"{entry.name}: actor list {slot} at {at:#x} is {lst:#x}, outside the program window")
        n = _u32(rs, lst)
        if lst + 4 + ACTOR_SIZE * n > _WINDOW[1]:
            raise CourseError(f"{entry.name}: actor list at {lst:#x} counts {n} records, past the program window")
        for k in range(n):
            a = lst + 4 + ACTOR_SIZE * k
            x, y, z, angle, ident, handler, obj = struct.unpack("<3fhH2I", rs.read_i960(a, ACTOR_SIZE))
            out.append(Actor(slot, a, (x, y, z), angle, ident, handler, obj))
    return out


MODELLED_HANDLERS = (STATIC_PROP, SWITCH, VIEW_BILLBOARD)
UNMODELLED_LOG_ENTRY = ("docs/spike/2026-11-arcade-course-tables.md, "
                        "\"Scenery placement / Not found / not modelled\"")


def unmodelled_actors(rs: RomSet, entry: CourseEntry) -> list[Actor]:
    """The actors whose handler `scenery` does not model (not in MODELLED_HANDLERS), in spawn order: exactly the
    actors whose objects a `scenery(..., allow_unmodelled=True)` result lacks."""
    return [a for a in actors(rs, entry) if a.handler not in MODELLED_HANDLERS]


def scenery(rs: RomSet, entry: CourseEntry, allow_unmodelled: bool = False) -> list[Placement]:
    """Every object the game places for the course: the cell grid's objects in cell order (`actor` None, world
    coordinates), then the objects of the actors whose handlers are modelled (STATIC_PROP, SWITCH: all three lists,
    VIEW_BILLBOARD: all 64 variants), in actor order. CourseError naming the address when a table is not where
    expected.

    When the course has actors of other handlers (`unmodelled_actors`; every real course does), their objects are
    not decoded: NotImplementedError naming the discovery-log entry (UNMODELLED_LOG_ENTRY), the count and the
    handler addresses, unless `allow_unmodelled` is True, in which case the modelled placements are returned and
    the gap is the caller's to handle (`unmodelled_actors` lists it)."""
    grid, cells = cell_grid(rs, entry)
    out = []
    for cell, ptr in enumerate(cells):
        out += _object_list(rs, ptr, f"{entry.name}: cell {cell} (pointer at {grid + 4 * cell:#x})", cell, True, None)
    for a in actors(rs, entry):
        cell = cell_of(a.position[0], a.position[2])
        what = f"{entry.name}: actor at {a.addr:#x} (handler {a.handler:#x})"
        if a.handler == STATIC_PROP:
            out += _object_list(rs, a.obj, f"{what} obj", cell, False, a)
        elif a.handler == SWITCH:
            for ins, ptr in zip(SWITCH_LISTS, switch_lists(rs, what)):
                out += _object_list(rs, ptr, f"{what} list at {ins + 4:#x}", cell, True, a)
        elif a.handler == VIEW_BILLBOARD:
            for k, ptr in enumerate(billboard_lists(rs, a.obj)):
                out += _object_list(rs, ptr, f"{what} variant {k} (pointer at {a.obj + 4 * k:#x})", cell, True, a)
    if not allow_unmodelled and (skipped := unmodelled_actors(rs, entry)):
        handlers = ", ".join(f"{h:#x}" for h in sorted({a.handler for a in skipped}))
        raise NotImplementedError(
            f"{entry.name}: {len(skipped)} actor(s) with handlers not modelled ({handlers}); their objects are not "
            f"decoded, see {UNMODELLED_LOG_ENTRY}. Pass allow_unmodelled=True for the modelled placements only "
            f"(unmodelled_actors lists the skipped actors).")
    return out


def _u32_field(value, name: str, what: str) -> int:
    if not is_int(value) or not 0 <= value <= 0xFFFFFFFF:
        raise CourseError(f"{what}: {name} {value!r} is not a u32")
    return int(value)


def encode_placement(p: Placement) -> bytes:
    """The 16-byte object entry (oba, tpa, tha, budget). CourseError for an oba that is not a polygon-ROM object
    (check_rom_object), an obc other than OBC (the game sends OBC; it is not stored), a field past 32 bits or a `raw`
    that is not one u32."""
    if not isinstance(p, Placement):
        raise CourseError(f"placement {p!r:.60} is not a Placement")
    what = f"placement {p.oba!r}"
    rom_object_index(p.oba)
    oba = int(p.oba)
    if p.obc != OBC:
        raise CourseError(f"{what}: obc {p.obc!r} cannot be stored; the game sends {OBC} with every object")
    raw = tuple(p.raw) if isinstance(p.raw, (tuple, list)) else (p.raw,)
    if len(raw) != 1:
        raise CourseError(f"{what}: raw must be one u32 word (the budget), got {raw!r}")
    words = [oba, _u32_field(p.tpa, "tpa", what), _u32_field(p.tha, "tha", what), _u32_field(raw[0], "raw", what)]
    return struct.pack("<4I", *words)


def encode_actor(a: Actor) -> bytes:
    """The 24-byte actor record. CourseError for a position that is not three single-precision floats, an angle outside
    s16, an ident outside u16, or a handler / obj past 32 bits."""
    what = f"actor at {a.addr!r}"
    packed = _f32(a.position, 3, "position", what)
    return packed + struct.pack("<hH2I", _int(a.angle, "angle (s16)", *_S16, what),
                                _int(a.ident, "ident (u16)", *_U16, what), _u32_field(a.handler, "handler", what),
                                _u32_field(a.obj, "obj", what))


# The structures around the entries (golden re-emit, docs/formats/model2-course-tables.md "Scenery placement"): the
# cell grid (u32[256] object-list pointers), an object list (16-byte entries, then a 4-byte oba of 0: lists are 20
# bytes apart on the real set), the actor table (u32[256] actor-list pointers, 0 = none), an actor list (u32 count,
# then 24-byte records), a billboard's u32[64] list array, and the switch's three `lda <list>,g0` instructions.

_WINDOW = PROGRAM_WINDOW


def _library_list(ptr, what: str, bounds: tuple[int, int] = OBJECT_LIBRARY) -> int:
    lo, hi = bounds
    if not is_int(ptr) or not lo <= ptr < hi:
        raise CourseError(f"{what}: {ptr!r} is outside the object library {lo:#x}-{hi - 1:#x}")
    return int(ptr)


def cell_grid(rs: RomSet, entry: CourseEntry) -> tuple[int, tuple[int, ...]]:
    """(GRID_TABLES[id], the GRID_CELLS object-list pointers of the course's grid). CourseError naming the address
    for a grid outside data ROM."""
    grid = _per_course(rs, id_table(rs, "GRID_TABLES"), entry, DATA_BASE, DATA_BASE + data_size(rs) - 4 * GRID_CELLS,
                       "cell grid")
    return grid, tuple(_u32(rs, grid + 4 * c) for c in range(GRID_CELLS))


def encode_cell_grid(lists) -> bytes:
    """A cell grid: GRID_CELLS u32 object-list pointers. CourseError for another count or a pointer outside
    PATCHED_LIBRARY (the draw 0x17cf8 skips such a list on any set). A pointer below OBJECT_LIBRARY is drawn only on
    a set with LIBRARY_BOUND_PATCH; decoding refuses it on a set without (`library_bounds`)."""
    lists = _seq(lists, None, "lists", "cell grid")
    if len(lists) != GRID_CELLS:
        raise CourseError(f"a cell grid holds {GRID_CELLS} list pointers, got {len(lists)}")
    return struct.pack(f"<{GRID_CELLS}I", *(_library_list(p, f"cell {c}", PATCHED_LIBRARY)
                                            for c, p in enumerate(lists)))


def object_list(rs: RomSet, ptr: int, what: str = "object list") -> list[Placement]:
    """The entries of the object list at `ptr` (inside the set's `library_bounds`) as placements that are not placed
    (cell -1, in_world True, no actor): for re-emitting lists reached outside the grid (spawn records, map objects).
    CourseError as `scenery`."""
    return _object_list(rs, ptr, f"{what} at {ptr:#x}", -1, True, None)


def encode_object_list(placements) -> bytes:
    """An object list: the entries (encode_placement), then the 4-byte oba of 0 that ends it."""
    placements = _seq(placements, None, "placements", "object list")
    return b"".join(encode_placement(p) for p in placements) + b"\0\0\0\0"


def actor_table(rs: RomSet, entry: CourseEntry) -> tuple[int, tuple[int, ...]]:
    """(ACTOR_TABLES[id], the GRID_CELLS actor-list pointers of the course's actor table, 0 = no list)."""
    table = _per_course(rs, id_table(rs, "ACTOR_TABLES"), entry, _WINDOW[0], _WINDOW[1] - 4 * GRID_CELLS,
                        "actor table")
    return table, tuple(_u32(rs, table + 4 * s) for s in range(GRID_CELLS))


def encode_actor_table(lists) -> bytes:
    """An actor table: GRID_CELLS u32 actor-list pointers, 0 for none. CourseError for another count or a pointer that
    is not 0 nor a list (its count word at least) inside the program window."""
    lists = _seq(lists, None, "lists", "actor table")
    if len(lists) != GRID_CELLS:
        raise CourseError(f"an actor table holds {GRID_CELLS} list pointers, got {len(lists)}")
    for s, p in enumerate(lists):
        if not is_int(p) or p and not _WINDOW[0] <= p < _WINDOW[1] - 4:
            raise CourseError(f"actor table slot {s}: {p!r} is neither 0 nor a list in the program window")
    return struct.pack(f"<{GRID_CELLS}I", *lists)


def encode_actor_list(records) -> bytes:
    """An actor list: u32 count, then the records (encode_actor)."""
    records = _seq(records, None, "records", "actor list")
    return struct.pack("<I", len(records)) + b"".join(encode_actor(a) for a in records)


def billboard_lists(rs: RomSet, addr: int) -> tuple[int, ...]:
    """A VIEW_BILLBOARD actor's `obj`: BILLBOARD_VARIANTS u32 object-list pointers, one per view angle."""
    return tuple(_u32(rs, addr + 4 * k) for k in range(BILLBOARD_VARIANTS))


def encode_billboard_lists(lists) -> bytes:
    """CourseError for a count other than BILLBOARD_VARIANTS or a pointer outside PATCHED_LIBRARY (the draw 0x17acc
    skips such a list on any set; a pointer below OBJECT_LIBRARY, a mirror's own list in DATA_FREE, is drawn only on a
    set with LIBRARY_BOUND_PATCH, and decoding refuses it on a set without: `library_bounds`)."""
    lists = _seq(lists, None, "lists", "billboard")
    if len(lists) != BILLBOARD_VARIANTS:
        raise CourseError(f"a billboard holds {BILLBOARD_VARIANTS} lists, got {len(lists)}")
    return struct.pack(f"<{len(lists)}I", *(_library_list(p, f"billboard variant {k}", PATCHED_LIBRARY)
                                            for k, p in enumerate(lists)))


def billboard_ref(rs: RomSet, course: int = 0) -> tuple[float, float, float]:
    """BILLBOARD_REF of course id `course`: (x, y, z). One point every id reads, at BILLBOARD_REF, unless the set is
    widened (widen_ids: a point per id, ID_ARRAYS["BILLBOARD_REF"].new). CourseError for an id outside 0..15."""
    return struct.unpack("<3f", _read(rs, _wide_entry(rs, "BILLBOARD_REF", course), 12, "billboard reference"))


def encode_billboard_ref(p) -> bytes:
    """BILLBOARD_REF. CourseError for a point that is not three finite single-precision floats."""
    return _f32(p, 3, "point", "billboard reference")


def switch_lists(rs: RomSet, what: str = "switch") -> tuple[int, ...]:
    """The SWITCH handler's three object lists, the literals of the `lda <list>,g0` instructions at SWITCH_LISTS
    (shared by every course: they are program code). CourseError when an instruction is not that `lda`."""
    out = []
    for ins in SWITCH_LISTS:
        if (word := _u32(rs, ins)) != _LDA_G0:
            raise CourseError(f"{what}: the instruction at {ins:#x} is {word:#x}, not `lda <list>,g0` "
                              f"({_LDA_G0:#x}): not the expected program")
        out.append(_u32(rs, ins + 4))
    return tuple(out)


def encode_switch_lists(lists) -> dict[int, bytes]:
    """The three `lda <list>,g0` instructions (opcode word and literal), by address. CourseError for a count other
    than three or a list outside OBJECT_LIBRARY."""
    lists = _seq(lists, None, "lists", "switch")
    if len(lists) != len(SWITCH_LISTS):
        raise CourseError(f"the switch has {len(SWITCH_LISTS)} lists, got {len(lists)}")
    return {ins: struct.pack("<2I", _LDA_G0, _library_list(p, f"switch list at {ins + 4:#x}"))
            for ins, p in zip(SWITCH_LISTS, lists)}


# --- culling and camera scripts (shared: read on every course as they are) --------------------------------------------
# Layouts: docs/formats/model2-course-tables.md, "Culling (per frame)" and "Camera scripts"; the runs:
# docs/spike/2026-11-convert.md, "Confirmed with scenery" (Phase 7: read on every race frame of Desert City's lap).

VIEW_TESTS = 0x00017908         # program: u32[4], per heading quadrant ((heading + 0x2000) >> 14 & 3) a point table
VIEW_TEST_SIZE = 0xA0           # bytes between the tables (data ROM 0x0286f7e0, 0x0286f880, 0x0286f920, 0x0286f9c0)
VIEW_TEST_POINTS = (0x00, 0x0C, 0x18, 0x24, 0x30, 0x3C, 0x50, 0x5C, 0x68, 0x74, 0x80, 0x8C)
"""Offsets of the 12 points `0x1770c` reads in a table (two groups of six; the 8 bytes at 0x48 and 0x98 are unread)."""
BLOCK = 5                       # the culling block: BLOCK x BLOCK cells round the reference cell
BLOCK_CELLS = BLOCK * BLOCK
VISIT_ORDER = 0x00017594        # program: u8[25], block indices in the order the cells are visited (centre first)
VISIT_ORDER_ADVANCED = 0x000175AD   # program: u8[25], Advanced's order at its cells 72 and 88 (0x17418)
VISIT_OFFSETS = 0x000175C6      # program: s8[25], per block index i the cell offset 16 * dr + dc, (dr, dc) =
#                                 (i // 5 - 2, i % 5 - 2)


class ViewPoint(NamedTuple):
    """One test point of a view-test table: `dx`, `dz` (f32) from the reference cell's low corner (`round(x, z) &
    ~0x7f`), and the block cells it opens (`mask`, bit i = block index i) when the point lies in the 90-degree wedge
    ahead (0x17918, TGP functions 0x1d / 0x1e, inferred)."""
    dx: float
    dz: float
    mask: int


class Culling(NamedTuple):
    """The view test's four quadrant tables (`view_tests`, quadrant order), the visit orders and the block offsets."""
    view_tests: tuple[tuple[ViewPoint, ...], ...]
    order: tuple[int, ...]
    order_advanced: tuple[int, ...]
    offsets: tuple[int, ...]


def culling(rs: RomSet) -> Culling:
    """The culling tables `0x172e8` / `0x1770c` read (shared by all courses; no course id indexes them). CourseError
    for `rs` not a RomSet, a table pointer outside data ROM, a point offset that is not finite or a mask past the
    block's 25 cells, a visit order that is not a permutation of the block, or offsets other than 16 * dr + dc."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"culling: {type(rs).__name__} is not a RomSet")
    tables = []
    for q in range(4):
        ptr = _data_ptr_ok(_u32(rs, VIEW_TESTS + 4 * q), f"view-test table {q}", VIEW_TEST_SIZE, rs)
        raw = _read(rs, ptr, VIEW_TEST_SIZE, f"view-test table {q}")
        points = []
        for off in VIEW_TEST_POINTS:
            p = ViewPoint(*struct.unpack_from("<2fI", raw, off))
            if not (math.isfinite(p.dx) and math.isfinite(p.dz)) or p.mask >> BLOCK_CELLS:
                raise CourseError(f"view-test table {q} at {ptr:#x}: point {off:#x} {p} is not a finite offset with "
                                  f"a {BLOCK_CELLS}-cell mask")
            points.append(p)
        tables.append(tuple(points))
    orders = []
    for at in (VISIT_ORDER, VISIT_ORDER_ADVANCED):
        order = tuple(_read(rs, at, BLOCK_CELLS, "visit order"))
        if sorted(order) != list(range(BLOCK_CELLS)):
            raise CourseError(f"visit order at {at:#x} {order} is not a permutation of the {BLOCK_CELLS} block cells")
        orders.append(order)
    offsets = struct.unpack(f"<{BLOCK_CELLS}b", _read(rs, VISIT_OFFSETS, BLOCK_CELLS, "block offsets"))
    if offsets != tuple(16 * (i // BLOCK - 2) + i % BLOCK - 2 for i in range(BLOCK_CELLS)):
        raise CourseError(f"block offsets at {VISIT_OFFSETS:#x} {offsets} are not 16 * dr + dc")
    return Culling(tuple(tables), orders[0], orders[1], offsets)


CAMERA_SCRIPTS = 0x00006C98     # program: u32[4], per course id the camera script 0x63d8 plays (0: none, id 3)
CAMERA_SCRIPTS_SLOT0 = 0x00006CA8   # program: u32[4], slot 0's variants, index 8 - the byte 0x540027 when it is past 1
KEYFRAME_SIZE = 0x40
CAMERA_JUMP = -1                # a keyframe time of -1: the script continues at the word at +8
CAMERA_ABSOLUTE = -1            # a keyframe car of -1: the point is a world position; otherwise the car's frame
_MAX_KEYFRAMES = 256


class Keyframe(NamedTuple):
    """One 64-byte keyframe of a camera script (0x65b4 plays them, 0x6a10 places the points): `addr`, `time` (ticks
    from the script's start, one per frame), `skip` (word +4: skipped when its bit 0x501752 is set), `position_car` /
    `target_car` (car index, or CAMERA_ABSOLUTE) and `position` / `target` (f32 x 3, in the car's frame: T(car
    +0x1c) R(car +0x28) when the car is not CAMERA_ABSOLUTE), `roll`, `fov` (f32 at +0x28, +0x2c, named after their
    use), `rest` (the four words at +0x30: a value to 0x501740, a byte to 0x501746, a hook address, padding)."""
    addr: int
    time: int
    skip: int
    position_car: int
    position: Point
    target_car: int
    target: Point
    roll: float
    fov: float
    rest: tuple[int, int, int, int]


class CameraScript(NamedTuple):
    """A camera script walked from `start`: its keyframes in play order (jumps followed, not listed) and `loop`, the
    address the last jump returns to (the script repeats from there)."""
    start: int
    keyframes: tuple[Keyframe, ...]
    loop: int


def camera_script(rs: RomSet, addr: int) -> CameraScript:
    """The camera script at data-ROM address `addr` (a CAMERA_SCRIPTS or CAMERA_SCRIPTS_SLOT0 entry), walked until a
    jump returns to a keyframe already walked. CourseError for `rs` not a RomSet, an address that is not a 4-aligned
    data-ROM address (bool refused), a jump outside data ROM, a keyframe point that is not finite, or more than 256
    keyframes without a loop."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"camera script: {type(rs).__name__} is not a RomSet")
    start = _data_ptr_ok(addr, "camera script", KEYFRAME_SIZE, rs)
    if start % 4:
        raise CourseError(f"camera script at {start:#x} is not 4-aligned")
    keys, seen, at = [], set(), start
    while at not in seen:
        if len(seen) >= _MAX_KEYFRAMES:
            raise CourseError(f"camera script at {start:#x}: no loop within {_MAX_KEYFRAMES} keyframes")
        seen.add(at)
        raw = _read(rs, _data_ptr_ok(at, f"camera script at {start:#x}: keyframe", KEYFRAME_SIZE, rs), KEYFRAME_SIZE,
                    "keyframe")
        time, skip, car = struct.unpack_from("<iIi", raw, 0)
        if time == CAMERA_JUMP:
            jump, at = at, car & 0xFFFFFFFF
            if at % 4:
                raise CourseError(f"camera script at {start:#x}: the jump at {jump:#x} goes to {at:#x}, not 4-aligned")
            continue
        position, target = struct.unpack_from("<3f", raw, 0x0C), struct.unpack_from("<3f", raw, 0x1C)
        roll, fov = struct.unpack_from("<2f", raw, 0x28)
        if not all(math.isfinite(v) for v in (*position, *target, roll, fov)):
            raise CourseError(f"camera script at {start:#x}: keyframe at {at:#x} has a point that is not finite")
        keys.append(Keyframe(at, time, skip, car, position, struct.unpack_from("<i", raw, 0x18)[0], target, roll, fov,
                             struct.unpack_from("<4I", raw, 0x30)))
        at += KEYFRAME_SIZE
    return CameraScript(start, tuple(keys), at)


# --- road line and AI lines ------------------------------------------------------------------------------------------
# Layouts: docs/formats/model2-course-tables.md, "Road line" and "AI lines". Both are indexed by the car's segment
# (car +0x12c): 0x140f0 reads record_table[segment], 0x1117c reads line[segment] (offset segment * 28 at car +0x138).

LANES = 8                       # lane slots per road point (the record table's second header word)
LANE_SIZE = 20
ROAD_POINT_SIZE = 4 + LANES * LANE_SIZE      # 164
AI_LINE_COUNT = 8               # (pointer, count) pairs in a line table; a car's line is picked by its spawn record
AI_POINT_SIZE = 28
AI_LINE_WRAP = 8                # each list is followed by its first AI_LINE_WRAP points again (look-ahead past the end)
LANE_UNITS = 3.0                # world units between lanes (0x14018 scales the lateral offset by -85.0 = -256 / 3.01)
_MAX_SEGMENTS = 0x10000         # segment indices are u16 (car +0x12c)


class Lane(NamedTuple):
    """One lane point of a road point (20 bytes)."""
    position: Point         # +0: x, y, z f32 (0x140f0 interpolates two lanes into car +0x1c / +0x20 / +0x24)
    pitch: int              # +12: s16 (to car +0x28)
    heading: int            # +14: s16; `heading << 2` is the car's yaw (car +0x2a, 0x10000 per turn); the direction
    #                         of travel is (-sin, cos) of that angle in (x, z)
    roll: int               # +16: s16 (to car +0x2c)
    pad: int                # +18: u16, 0 on the real set, no reader found


@dataclass(frozen=True)
class RoadPoint:
    """One record of the road line (record_table): a cross-section of the road at one segment, LANES lane points
    LANE_UNITS apart; only lanes lane_lo..lane_hi are used (the others are zero on the real set). The next point is
    the next record; the last one is followed by the first (a closed loop)."""
    lane_lo: int            # +0: u8, the first used lane (0x140f0 clamps the car's lane to lane_lo..lane_hi)
    lane_hi: int            # +1: u8, the last used lane
    curve: int              # +2: s16, signed bend (0 on straights; 0x15848 tests its sign); unit not known
    lanes: tuple[Lane, ...]  # +4: LANES lane points

    @property
    def used_lanes(self) -> tuple[Lane, ...]:
        return self.lanes[self.lane_lo:self.lane_hi + 1]

    @property
    def centre(self) -> Point:
        """Midpoint of the outer used lanes (the lanes are evenly spaced, so also their mean)."""
        a, b = self.lanes[self.lane_lo].position, self.lanes[self.lane_hi].position
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)

    @property
    def width(self) -> float:
        """Distance from lane_lo to lane_hi: the span the AI's lateral coordinate covers, not the kerb-to-kerb road
        width (the road surface itself is a TGP-side table, see the discovery log)."""
        return math.dist(self.lanes[self.lane_lo].position, self.lanes[self.lane_hi].position)

    @property
    def heading(self) -> int:
        return self.lanes[self.lane_lo].heading


class AiPoint(NamedTuple):
    """One point of an AI line (28 bytes); read by 0x1117c / 0x1127c for the car's current segment."""
    position: Point         # +0: x, y, z f32 (to the car's target, g1 / g5 / g2 at 0x112d8)
    speed: float            # +12: f32, multiplies the car's speed setting (+0x23c) into +0x44 (0x114a0); inferred
    pitch: int              # +16: s16 (to car +0x28)
    heading: int            # +18: s16, as Lane.heading (shifted left 2 at 0x112f4)
    roll: int               # +20: s16 (to car +0x2c)
    steer: int              # +22: s16, scaled by the car's speed and added to +0x48 at 0x11648; inferred
    lane: int               # +24: s16, the lateral position on the road line in lanes, 8.8 fixed (to car +0x142)
    step: int               # +26: u16, the step-table index of the point (to car +0x30 at 0x111f8)


def _road_n(rs: RomSet, entry: CourseEntry) -> int:
    """n from the record-table header (n + 1 road points), with the header checked."""
    at = DATA_BASE + _data_off(DATA_BASE + entry.record_table, 8, f"{entry.name}: record table", rs)
    n, lanes = _u32(rs, at), _u32(rs, at + 4)
    if lanes != LANES:
        raise CourseError(f"{entry.name}: record table header at {at + 4:#x} says {lanes} lanes per record, "
                          f"expected {LANES}")
    if n + 1 >= _MAX_SEGMENTS:
        raise CourseError(f"{entry.name}: record table at {at:#x}: n {n:#x} is past the u16 segment index")
    _data_off(at + 8, ROAD_POINT_SIZE * (n + 1), f"{entry.name}: record table ({n + 1} records)", rs)
    return n


def _decode_road_point(b: bytes, what: str) -> RoadPoint:
    lo, hi, curve = struct.unpack_from("<BBh", b, 0)
    if not lo <= hi < LANES:
        raise CourseError(f"{what}: lanes {lo}-{hi} are not a lane range inside 0-{LANES - 1}")
    lanes = []
    for k in range(LANES):
        x, y, z, pitch, heading, roll, pad = struct.unpack_from("<3f3hH", b, 4 + LANE_SIZE * k)
        lanes.append(Lane((x, y, z), pitch, heading, roll, pad))
    return RoadPoint(lo, hi, curve, tuple(lanes))


def road_line(rs: RomSet, entry: CourseEntry) -> list[RoadPoint]:
    """The course's road line: the n + 1 records of record_table, a closed loop of cross-sections (one per segment,
    about 8 units apart). CourseError naming the address for a header other than (n, LANES), records running past
    data ROM, or a lane range outside 0..LANES-1."""
    n = _road_n(rs, entry)
    out = []
    for i in range(n + 1):
        at = DATA_BASE + entry.record_table + 8 + ROAD_POINT_SIZE * i
        out.append(_decode_road_point(rs.read_i960(at, ROAD_POINT_SIZE), f"{entry.name}: record {i} at {at:#x}"))
    return out


_LOG = "docs/spike/2026-11-arcade-course-tables.md"

SEGMENT_FLAGS = 0x028058BC
"""Data ROM: u32[4], per course id the address of its per-segment flags (stored to 0x501450 by the course load): one
u16 per road segment (n + 1), back to back on the real set (Beginner 0x028058cc, Expert 0x02805a54, Advanced
0x02806212; the id-3 stub repeats Beginner's pointer). 0x10f7c and 0x13c7c read entry `segment` (`ldos`, the car's
segment from its step) and OR it with the step's flags into car +0x13c, so the bits share the step flags' space
(STEP_*). docs/spike/2026-11-convert.md, "Per-segment flags"."""
SEGMENT_TUNNEL = 0x100          # car +0x13c bit 8: in a tunnel (inferred from its readers): 0x7a08 switches a sound
#                                 effect on entering it (on Advanced) and off on leaving it, 0x7b88 lights the car from
#                                 the course's tunnel light zones (window 0x230742 / 0x23074a) instead of the default
#                                 (0x232db8).
#                                 Expert's segments 208-218; Advanced carries it in its step flags (0x101) instead
SEGMENT_AI_FREE = 0x400         # car +0x13c bit 10: 0x1129c skips 0x13ccc, the AI car's lane reaction to the player
#                                 (blocking and making room; inferred). Expert's segments 190-240


def segment_flags(rs: RomSet, entry: CourseEntry) -> tuple[int, ...]:
    """The course's per-segment flags: one u16 per road segment (n + 1, the road line's count) at SEGMENT_FLAGS[id].
    Read with `ldos` but used as bits (SEGMENT_TUNNEL, SEGMENT_AI_FREE; ORed with the step flags into car +0x13c),
    so returned as u16. CourseError for an entry that is not a CourseEntry of a course id the set holds (id_count),
    a record table `road_line` refuses, or a pointer or table outside data ROM; for `rs` not a RomSet."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not isinstance(entry, CourseEntry):
        raise CourseError(f"{entry!r:.60} is not a CourseEntry")
    _int(entry.course, "course", 0, id_count(rs) - 1, "per-segment flags")
    count = _road_n(rs, entry) + 1
    at = _data_ptr(rs, id_table(rs, "SEGMENT_FLAGS") + 4 * entry.course, 2 * count, f"{entry.name}: per-segment flags")
    return struct.unpack(f"<{count}H", _read(rs, at, 2 * count, f"{entry.name}: per-segment flags"))


def encode_segment_flags(flags) -> bytes:
    """A per-segment flag table: one u16 per road segment (1 to 0xffff values: segments are u16, car +0x12c).
    CourseError for a value that is not an integer inside 0..0xffff (bool refused) or a count outside 1..0xffff."""
    flags = _seq(flags, None, "flags", "per-segment flags")
    if not 1 <= len(flags) < _MAX_SEGMENTS:
        raise CourseError(f"per-segment flags: {len(flags)} values; a road line has 1..{_MAX_SEGMENTS - 1} segments")
    return struct.pack(f"<{len(flags)}H", *(_int(f, f"flags of segment {k}", *_U16, "per-segment flags")
                                           for k, f in enumerate(flags)))


def centreline(points: list[RoadPoint]) -> list[Point]:
    """The road line's centre, one point per segment (RoadPoint.centre); a closed loop."""
    return [p.centre for p in points]


def encode_road_point(p: RoadPoint) -> bytes:
    """The ROAD_POINT_SIZE bytes of a road point. CourseError for a lane range that is not lo <= hi inside
    0..LANES-1, a curve or lane angle outside s16, a pad outside u16, a position that is not three finite
    single-precision floats, or a lane count other than LANES."""
    what = "road point"
    lo = _int(p.lane_lo, "lane_lo", 0, LANES - 1, what)
    hi = _int(p.lane_hi, "lane_hi", lo, LANES - 1, what)
    out = struct.pack("<BBh", lo, hi, _int(p.curve, "curve", *_S16, what))
    for k, lane in enumerate(_seq(p.lanes, LANES, "lanes", what)):
        w = f"{what} lane {k}"
        out += _f32(lane.position, 3, "position", w)
        out += struct.pack("<3hH", _int(lane.pitch, "pitch", *_S16, w),
                           _int(lane.heading, "heading", *_S16, w), _int(lane.roll, "roll", *_S16, w),
                           _int(lane.pad, "pad", *_U16, w))
    return out


def encode_road_line(points: list[RoadPoint]) -> bytes:
    """A whole record table: u32 n (= len(points) - 1), u32 LANES, then the points. CourseError for no points or a
    point that does not fit (encode_road_point)."""
    points = _seq(points, None, "points", "road line")
    if not points:
        raise CourseError("a road line needs at least one point")
    if len(points) >= _MAX_SEGMENTS:
        raise CourseError(f"{len(points)} road points: past the u16 segment index")
    return struct.pack("<2I", len(points) - 1, LANES) + b"".join(encode_road_point(p) for p in points)


def _decode_ai_point(b: bytes, off: int) -> AiPoint:
    x, y, z, speed, pitch, heading, roll, steer, lane, step = struct.unpack_from("<4f5hH", b, off)
    return AiPoint((x, y, z), speed, pitch, heading, roll, steer, lane, step)


def ai_lines(rs: RomSet, entry: CourseEntry) -> list[list[AiPoint]]:
    """The course's AI_LINE_COUNT AI lines (line_table), each n + 1 points (the same count as the road line: a car's
    segment indexes both), in pair order. CourseError naming the address for a pair whose count is not n + 1, a list
    outside data ROM, or a list whose AI_LINE_WRAP points after the end differ from its first ones."""
    n = _road_n(rs, entry)
    table = DATA_BASE + _data_off(DATA_BASE + entry.line_table, 8 * AI_LINE_COUNT, f"{entry.name}: line table", rs)
    out = []
    for k in range(AI_LINE_COUNT):
        at = table + 8 * k
        ptr, count = _u32(rs, at), _u32(rs, at + 4)
        if count != n + 1:
            raise CourseError(f"{entry.name}: line {k} count at {at + 4:#x} is {count}, expected {n + 1} "
                              f"(the road line's n + 1)")
        size = AI_POINT_SIZE * (count + AI_LINE_WRAP)
        off = _data_off(ptr, size, f"{entry.name}: line {k} (pointer at {at:#x})", rs)
        b = rs.data[off:off + size]
        body, wrap = b[:AI_POINT_SIZE * count], b[AI_POINT_SIZE * count:]
        if wrap != body[:len(wrap)]:
            raise CourseError(f"{entry.name}: line {k} at {ptr:#x}: the {AI_LINE_WRAP} wrap points after the end "
                              f"(at {ptr + AI_POINT_SIZE * count:#x}) differ from the first ones")
        out.append([_decode_ai_point(body, AI_POINT_SIZE * i) for i in range(count)])
    return out


def ai_path(points: list[AiPoint]) -> list[Point]:
    """An AI line's positions; a closed loop."""
    return [p.position for p in points]


def encode_ai_point(p: AiPoint) -> bytes:
    """The AI_POINT_SIZE bytes of an AI point. CourseError for a position / speed that is not finite single
    precision, an angle, steer or lane outside s16, or a step outside u16."""
    what = "AI point"
    return (_f32(p.position, 3, "position", what) + _f32([p.speed], 1, "speed", what)
            + struct.pack("<5hH", *(_int(getattr(p, f), f, *_S16, what)
                                    for f in ("pitch", "heading", "roll", "steer", "lane")),
                          _int(p.step, "step", *_U16, what)))


def encode_ai_line(points: list[AiPoint]) -> bytes:
    """One AI line as stored: the points, then the first AI_LINE_WRAP points again. CourseError for fewer than
    AI_LINE_WRAP points or 0x10000 and more (a segment is a u16)."""
    points = _seq(points, None, "points", "AI line")
    if len(points) < AI_LINE_WRAP:
        raise CourseError(f"an AI line needs at least {AI_LINE_WRAP} points (the wrap repeats them), got {len(points)}")
    if len(points) >= _MAX_SEGMENTS:
        raise CourseError(f"{len(points)} AI points: past the u16 segment index (one point per segment)")
    body = b"".join(encode_ai_point(p) for p in points)
    return body + body[:AI_POINT_SIZE * AI_LINE_WRAP]


def encode_line_table(offset: int, lines: list[list[AiPoint]]) -> bytes:
    """A whole line table for data-ROM `offset`, laid out as on the real set: AI_LINE_COUNT (pointer, count) pairs,
    then the lines one after another (encode_ai_line). CourseError for a line count other than AI_LINE_COUNT, lines of
    different lengths (a car's segment indexes the road line and every AI line alike) or a table that would not fit
    in data ROM."""
    lines = [_seq(l, None, "line", "line table") for l in _seq(lines, None, "lines", "line table")]
    if len(lines) != AI_LINE_COUNT:
        raise CourseError(f"a line table holds {AI_LINE_COUNT} lines, got {len(lines)}")
    if len({len(l) for l in lines}) != 1:
        raise CourseError(f"the AI lines must have the same number of points, got {[len(l) for l in lines]}")
    blobs = [encode_ai_line(l) for l in lines]
    size = 8 * AI_LINE_COUNT + sum(map(len, blobs))
    if not is_int(offset) or not 0 <= offset or offset + size > DATA_WINDOW:
        raise CourseError(f"a line table of {size} bytes at {offset!r} does not fit in data ROM")
    pairs, at = b"", DATA_BASE + int(offset) + 8 * AI_LINE_COUNT
    for l, blob in zip(lines, blobs):
        pairs += struct.pack("<2I", at, len(l))
        at += len(blob)
    return pairs + b"".join(blobs)


# --- checkpoints, time limit, start grid, pit ------------------------------------------------------------------------
# Layouts: docs/formats/model2-course-tables.md, "Checkpoints", "Start grid" and "Pit". Unlike the blobs above these
# are not reached through the course record: the course load (0x4534-0x4640) and the car set-up (0xe898) index
# per-id arrays with the course id.

CHECKPOINT_TABLES = 0x02823C10
"""Data ROM: u32[4], per course id the address of its checkpoint table (stored to 0x50143c by 0x4628): u32 count, then
count (u16 segment, u16 raw) entries. 0x1395c tests the player's segment against each entry when it changes."""

TIME_TABLES = 0x02805228
"""Data ROM: u32[MODES] per game mode (0x5fe5e0) -> u32[4] per course id -> u32[DIFFICULTIES] per difficulty
(0x5fe59c) -> the time table, u32 words in ticks: the initial time (0x3ad8 -> the timer at 0x5010d0), then one
extension per checkpoint pass (0x13b08). Stored to 0x501440 by 0x4604."""

LAP_COUNTS = 0x02805234
"""Data ROM: u32[MODES] per game mode -> u32[4] per course id: the race's laps (stored to the byte at 0x5010ec)."""

CAR_COUNTS = 0x02805240
"""Data ROM: u32[2] per cabinet setting (0x4570-0x45d0 from 0x5fe588 and 0x540027; 0 in the MAME runs) -> u32[4] per
course id: the cars in the race, player included (to 0x501080). course.py reads entry 0."""

ROLLING_STARTS = 0x028050E8
"""Data ROM: s16[4] per course id, stored to 0x501850 (0x3a8c); 0 = rolling start (Beginner), -1 = standing start."""

SPAWN_TABLES = (0x0281E3D4, 0x0281EC14)
"""Data ROM: two tables of SPAWN_RECORDS spawn records, one per car index; 0xeb00 takes the first when work RAM
0x5fe588 is 0 and the second otherwise (MAME's default settings: the second). They differ only in the object words."""

SPAWN_RECORD_SIZE = 48
SPAWN_RECORDS = 44
GRID_BASES = 0x00230E48
"""Program window: s16[4] per course id, the segment the grid starts from: a car's segment is
(spawn start + base) mod n (0xe8d4-0xe8f4)."""

GRID_LANES = 0x00231B30
"""Program window: u16[2], the two grid lanes (8.8 lane coordinate, car +0x142): 0x180 and 0x580 on the real set."""

TICKS_PER_SECOND = 64           # the timer at 0x5010d0 counts down one per frame; 0x3b04 shows ticks >> 6 (inferred)
MODES = 3                       # entries of TIME_TABLES and LAP_COUNTS (0x5fe5e0)
DIFFICULTIES = 4                # entries per course of a TIME_TABLES mode (0x5fe59c)
DEFAULT_MODE, DEFAULT_DIFFICULTY = 0, 1     # the values in every MAME run of this phase
MAX_LAPS = 127                  # laps go to the byte 0x5010ec (stib at 0x460c); 0xdfa4 reads it signed (ldib) against
#                                 the car's signed lap byte +0x70 (-1 before the first line, 0xe90c), so 1..127
MAX_CHECKPOINTS = 30            # 0xcb44 keeps all checkpoints and the lap line as bits of one u32, (1 << (count + 1)) - 1

STEP_ROAD = 0x01                # step flags (the high u16 of a step-table word, to car +0x13c): on the road line
STEP_PIT = 0x02                 # a pit polygon (lane and box)
STEP_PIT_LANE = 0x40            # the pit lane (with STEP_PIT)
STEP_PIT_BOX = 0x80             # the pit box: 0xdf9c starts the pit stop (0xe288) on it
PIT_LANE_FLAGS = STEP_PIT | STEP_PIT_LANE
PIT_BOX_FLAGS = STEP_PIT | STEP_PIT_BOX
_PIT_BITS = STEP_PIT | STEP_PIT_LANE | STEP_PIT_BOX


class Checkpoint(NamedTuple):
    """A point the player must pass for a time extension. The lap line (segment 0, the wrap from segment n to 0) is
    last; it is not stored in the checkpoint table and counts only when every checkpoint of the lap was passed."""
    index: int              # position in the checkpoint table (0..count-1); count for the lap line
    segment: int            # the road segment (car +0x12c) that triggers it (0x13970)
    raw: int                # the entry's high u16 (0 on the real set; no reader found); 0 for the lap line
    lap_line: bool
    position: Point         # the road line's centre at that segment
    extensions: tuple[int, ...]  # ticks added when it is passed, one per lap (time table word lap*(count+1)+index+1)
    #   The lap line's last value (the final lap) is a phantom: the race ends at the final lap line and never adds it
    #   (0x13898 does not clear the bit set on the final lap). It is the time table's word laps * (count + 1), which
    #   time_table reads and the golden re-emit keeps, so every checkpoint has one value per lap; a converter may write
    #   any value there.
    #   index, position, lap_line and extensions are views: of the table's row order, the road line and the time
    #   table (checkpoint_views); golden.check_course refuses a Course where they differ from their sources.


class GridSlot(NamedTuple):
    """One car of the start grid: its spawn record (car index = spawn record = car ident, car +4; 0 is the player)
    and where the car set-up (0xe898) puts it."""
    car: int
    objects: tuple[int, ...]  # +0x00: 8 u32, object-list pointers (car models; 0xec64 copies five to car +0x74-+0x88)
    line: int               # +0x20: u32, the AI line (0xe8a0: pair `line` of the line table, to car +0x134)
    start: int              # +0x24: u32, segments ahead of the grid base (to car +0x12e / +0x132 as a u16)
    speed: float            # +0x28: f32, the car's speed setting (scaled into car +0x23c and +0x38 / +0x44)
    kind: int               # +0x2c: u32, read as a byte (to car +0x23a); 0-2 on the real set
    segment: int            # (start + GRID_BASES[id]) mod n (car +0x12c)
    lane: int               # GRID_LANES[...] by the car's parity (car +0x142, 8.8 lanes)
    position: Point         # lane_position(road[segment], lane); the game then sets y from the ground query
    #   segment, lane and position are views of the stored record, GRID_BASES, GRID_LANES, ROLLING_STARTS and the road
    #   line (grid_slot); golden.check_course refuses a Course where they differ from their sources.


class Step(NamedTuple):
    """One word of the step table: a ground polygon of kind road (see "The ground under the road")."""
    segment: int            # u16, to car +0x12c (0x13c60)
    flags: int              # u16, ORed with the per-segment flags into car +0x13c (STEP_* bits)


class Pit(NamedTuple):
    """The pit lane: the step-table polygons flagged STEP_PIT. Entry and exit are the first and last segment of their
    run in the direction of travel (where the lane leaves and rejoins the road); the box is the one STEP_PIT_BOX
    polygon. Its geometry is on the TGP side (the ground polygons), like the rest of the road surface."""
    entry_segment: int
    exit_segment: int
    box_segment: int
    lane_polygons: tuple[int, ...]   # step indices (= ground polygon indices) with PIT_LANE_FLAGS
    box_polygon: int                 # the step index with PIT_BOX_FLAGS


def _check_id_array(rs: RomSet, array: int, index: int, count: int, what: str) -> int:
    if not is_int(index) or not 0 <= index < count:
        raise CourseError(f"{what} {index!r} is outside 0..{count - 1}")
    return _u32(rs, array + 4 * index)


def _data_ptr(rs: RomSet, at: int, size: int, what: str) -> int:
    """The i960 address held by the word at `at`, checked to point at `size` bytes of data ROM."""
    ptr = _u32(rs, at)
    _data_off(ptr, size, f"{what} (pointer at {at:#x})", rs)
    return ptr


def laps(rs: RomSet, entry: CourseEntry, mode: int = DEFAULT_MODE) -> int:
    """The race's laps for the course in game mode `mode` (LAP_COUNTS). CourseError for a mode outside 0..MODES-1, a
    pointer outside data ROM or a count outside 1..MAX_LAPS."""
    _check_id_array(rs, LAP_COUNTS, mode, MODES, "mode")
    arr = _data_ptr(rs, LAP_COUNTS + 4 * mode, 16, f"{entry.name}: lap counts")
    n = _u32(rs, arr + 4 * entry.course)
    if not 1 <= n <= MAX_LAPS:
        raise CourseError(f"{entry.name}: laps {n} at {arr + 4 * entry.course:#x} is outside 1..{MAX_LAPS}")
    return n


def _checkpoint_entries(rs: RomSet, entry: CourseEntry) -> list[tuple[int, int]]:
    at = id_table(rs, "CHECKPOINT_TABLES") + 4 * entry.course
    table = _data_ptr(rs, at, 4, f"{entry.name}: checkpoint table")
    count = _u32(rs, table)
    if count > MAX_CHECKPOINTS:
        raise CourseError(f"{entry.name}: checkpoint table at {table:#x} holds {count} checkpoints, more than "
                          f"{MAX_CHECKPOINTS} (the u32 bit set at 0x5014b4)")
    _data_off(table, 4 + 4 * count, f"{entry.name}: checkpoint table ({count} entries)", rs)
    n = _road_n(rs, entry)
    out = []
    for j in range(count):
        seg, raw = struct.unpack("<2H", rs.read_i960(table + 4 + 4 * j, 4))
        if seg > n:
            raise CourseError(f"{entry.name}: checkpoint {j} at {table + 4 + 4 * j:#x}: segment {seg} is past the "
                              f"road line (n {n})")
        out.append((seg, raw))
    return out


def time_corrections(rs: RomSet, entry: CourseEntry, difficulty: int = DEFAULT_DIFFICULTY):
    """The two corrections to a time extension (0x13b14-0x13b6c): minus data 0x0280587c[id][difficulty] when
    0x501a80 is 1, plus 0x0280583c[id][difficulty] when the player car's +0x0 bit 28 is set. Read, not traced:
    NotImplementedError (golden.EXCLUDED, "time corrections")."""
    raise NotImplementedError(f"{entry.name}: the time corrections (data 0x0280587c and 0x0280583c by id and "
                              f"difficulty) are read, not traced; not decoded. See {_LOG}, \"Checkpoints, start grid, "
                              f"pit / Not found / not decoded\"")


def time_table(rs: RomSet, entry: CourseEntry, mode: int = DEFAULT_MODE,
               difficulty: int = DEFAULT_DIFFICULTY) -> list[int]:
    """The words of the course's time table the race reads (TIME_TABLES): the initial time, then one extension per
    checkpoint pass, 1 + laps * (checkpoints + 1) words, in ticks (TICKS_PER_SECOND); the last of them is the final lap
    line's, which the race never adds (the phantom word, see Checkpoint). The stored table may be longer.
    CourseError for a mode or difficulty out of range or a pointer outside data ROM."""
    _check_id_array(rs, TIME_TABLES, mode, MODES, "mode")
    if not is_int(difficulty) or not 0 <= difficulty < DIFFICULTIES:
        raise CourseError(f"difficulty {difficulty!r} is outside 0..{DIFFICULTIES - 1}")
    by_id = _data_ptr(rs, TIME_TABLES + 4 * mode, 16, f"{entry.name}: time tables")
    by_diff = _data_ptr(rs, by_id + 4 * entry.course, 4 * DIFFICULTIES, f"{entry.name}: time tables")
    count = len(_checkpoint_entries(rs, entry))
    size = 1 + laps(rs, entry, mode) * (count + 1)
    table = _data_ptr(rs, by_diff + 4 * difficulty, 4 * size, f"{entry.name}: time table")
    return list(struct.unpack(f"<{size}I", rs.read_i960(table, 4 * size)))


def checkpoints(rs: RomSet, entry: CourseEntry, mode: int = DEFAULT_MODE,
                difficulty: int = DEFAULT_DIFFICULTY) -> list[Checkpoint]:
    """The course's checkpoints in table order, then the lap line, each with its time extension per lap (mode and
    difficulty as in time_table). CourseError naming the address for a pointer outside data ROM, a count past
    MAX_CHECKPOINTS or a segment past the road line."""
    return checkpoint_views(_checkpoint_entries(rs, entry), road_line(rs, entry),
                            time_table(rs, entry, mode, difficulty), laps(rs, entry, mode))


def checkpoint_views(rows, road: list[RoadPoint], words, n_laps: int) -> list[Checkpoint]:
    """The Checkpoints of a checkpoint table's (segment, raw) `rows`, then the lap line (segment 0, raw 0): `position`
    from the road line's centre, `extensions` from the time table `words` (one mode's and difficulty's: word
    lap * (count + 1) + j + 1 for checkpoint j on lap `lap`; the lap line's last one is the phantom word, see
    Checkpoint) for `n_laps` laps. What `checkpoints` returns and `golden.check_course` compares a Course against.
    The words must hold 1 + n_laps * (len(rows) + 1) words and every segment must be on the road line."""
    cl = centreline(road)
    count = len(rows)
    return [Checkpoint(j, seg, raw, j == count, cl[seg],
                       tuple(words[lap * (count + 1) + j + 1] for lap in range(n_laps)))
            for j, (seg, raw) in enumerate(list(rows) + [(0, 0)])]


def encode_checkpoint_table(cps: list[Checkpoint]) -> bytes:
    """A checkpoint table: u32 count, then (u16 segment, u16 raw) per checkpoint that is not the lap line (the lap line
    is implicit). CourseError for more than MAX_CHECKPOINTS or a segment / raw outside u16."""
    cps = _seq(cps, None, "checkpoints", "checkpoint table")
    if not all(isinstance(c, Checkpoint) for c in cps):
        raise CourseError(f"checkpoint table: {cps!r:.60} holds something that is not a Checkpoint")
    rows = [c for c in cps if not c.lap_line]
    if len(rows) > MAX_CHECKPOINTS:
        raise CourseError(f"{len(rows)} checkpoints: at most {MAX_CHECKPOINTS} fit the game's bit set")
    out = struct.pack("<I", len(rows))
    for c in rows:
        what = f"checkpoint {c.index!r}"
        out += struct.pack("<2H", _int(c.segment, "segment", *_U16, what), _int(c.raw, "raw", *_U16, what))
    return out


def encode_time_table(words: list[int]) -> bytes:
    """A time table: u32 words (initial time, then the extensions), in ticks. CourseError for no words or a word that
    is not a u32."""
    words = _seq(words, None, "words", "time table")
    if not words:
        raise CourseError("a time table needs at least the initial time")
    return b"".join(struct.pack("<I", _int(w, f"word {k}", 0, 0xFFFFFFFF, "time table"))
                    for k, w in enumerate(words))


def car_count(rs: RomSet, entry: CourseEntry) -> int:
    """The cars in the race, player included (CAR_COUNTS entry 0). CourseError outside 1..SPAWN_RECORDS: 44 is the car
    pool (the boot sets 0x501080 to 44 at 0x12e0, the ident loop 0x3d1c reads 44-byte rows at 0x3d4c, and the spawn
    tables hold 44 records)."""
    arr = _data_ptr(rs, CAR_COUNTS, 16, f"{entry.name}: car counts")
    n = _u32(rs, arr + 4 * entry.course)
    if not 1 <= n <= SPAWN_RECORDS:
        raise CourseError(f"{entry.name}: car count {n} at {arr + 4 * entry.course:#x} is outside 1..{SPAWN_RECORDS}")
    return n


def grid_base(rs: RomSet, entry: CourseEntry) -> int:
    """GRID_BASES[id]: the segment the grid is laid out from."""
    at = id_table(rs, "GRID_BASES") + 2 * entry.course
    base = struct.unpack("<h", _read(rs, at, 2, f"{entry.name}: grid base"))[0]
    if base < 0:
        raise CourseError(f"{entry.name}: grid base {base} at {at:#x} is negative")
    return base


def lane_position(p: RoadPoint, lane: int) -> Point:
    """The point at lateral coordinate `lane` (8.8 lanes) of a road point, as 0x140f0 places a car: lanes `lane >> 8`
    and the next, each clamped to lane_lo..lane_hi, blended by data 0x02871c80[lane & 0xff] (= frac / 255 as f32)."""
    k = lane >> 8
    lo, hi = p.lane_lo, p.lane_hi
    a = p.lanes[min(max(k, lo), hi)].position
    b = p.lanes[min(max(k + 1, lo), hi)].position
    w = struct.unpack("<f", struct.pack("<f", (lane & 0xFF) / 255))[0]
    return tuple(x + (y - x) * w for x, y in zip(a, b))


def start_grid(rs: RomSet, entry: CourseEntry, table: int = 1) -> list[GridSlot]:
    """The course's start grid: one slot per car (car_count), from spawn table `table` (SPAWN_TABLES index; 1 in the
    MAME runs). A car's segment is (start + grid_base) mod n; its lane is GRID_LANES[(car & 1) ^ 1] (0xe980), or, on a
    rolling start, GRID_LANES[car & 1] for the player (0xd840) and for every car at a segment >= n >> 1 (0xdb0c).
    CourseError for a table index out of range, a car count outside 1..SPAWN_RECORDS, an AI line past 7 or an
    object pointer outside OBJECT_LIBRARY."""
    if not is_int(table) or not 0 <= table < len(SPAWN_TABLES):
        raise CourseError(f"spawn table {table!r} is outside 0..{len(SPAWN_TABLES) - 1}")
    road = road_line(rs, entry)
    n = len(road) - 1
    count, base = car_count(rs, entry), grid_base(rs, entry)
    lanes = grid_lanes(rs, entry.course)
    rolling = rolling_start(rs, entry)
    out = []
    for car in range(count):
        at = SPAWN_TABLES[table] + SPAWN_RECORD_SIZE * car
        w = struct.unpack("<10IfI", rs.read_i960(at, SPAWN_RECORD_SIZE))
        objects, line, start, speed, kind = w[:8], w[8], w[9], w[10], w[11]
        if line >= AI_LINE_COUNT:
            raise CourseError(f"{entry.name}: spawn record {car} at {at:#x}: line {line} is past {AI_LINE_COUNT - 1}")
        for k, o in enumerate(objects):
            _check_library_list(o, f"{entry.name}: spawn record {car} at {at:#x} object {k}")
        out.append(grid_slot(car, objects, line, start, speed, kind, road, base, lanes, rolling))
    return out


def grid_slot(car: int, objects, line: int, start: int, speed: float, kind: int, road: list[RoadPoint], base: int,
              lanes: tuple[int, int], rolling: bool) -> GridSlot:
    """The GridSlot of spawn record `car` (objects, line, start, speed, kind as stored) on a course with road line
    `road`, grid base `base`, GRID_LANES `lanes` and start type `rolling`: segment, lane and position are views of
    them (see start_grid). What `start_grid` returns and `golden.check_course` compares a Course against."""
    n = len(road) - 1
    segment = (start + base) % n
    side = (car & 1) ^ 1
    if rolling and (car == 0 or segment >= n >> 1):
        side = car & 1
    return GridSlot(car, tuple(objects), line, start, speed, kind, segment, lanes[side],
                    lane_position(road[segment], lanes[side]))


def _wide_entry(rs: RomSet, name: str, course: int) -> int:
    """id_entry of WIDE_ARRAYS array `name` where `rs`'s readers read it (id_table)."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    return id_entry({name: id_table(rs, name)}, name, course)


def grid_lanes(rs: RomSet, course: int = 0) -> tuple[int, int]:
    """GRID_LANES of course id `course`: the two grid lanes (u16, 8.8 lanes; 0x180 and 0x580 on the real set). One
    pair every id reads, at GRID_LANES, unless the set is widened (widen_ids: a pair per id,
    ID_ARRAYS["GRID_LANES"].new). CourseError for an id outside 0..15."""
    return struct.unpack("<2H", _read(rs, _wide_entry(rs, "GRID_LANES", course), 4, "grid lanes"))


def encode_grid_lanes(lanes) -> bytes:
    """GRID_LANES. CourseError for a count other than two or a lane outside u16."""
    lanes = _seq(lanes, 2, "lanes (2)", "grid lanes")
    return struct.pack("<2H", *(_int(v, "lane", *_U16, "grid lanes") for v in lanes))


def encode_spawn_record(s: GridSlot) -> bytes:
    """The SPAWN_RECORD_SIZE bytes of a spawn record (segment, lane and position are derived, not stored). CourseError
    for objects that are not 8 object-list pointers into OBJECT_LIBRARY, a line outside 0..7, a start outside u16 (the game keeps it in a u16), a
    speed that is not finite single precision, or a kind outside u8 (the game reads one byte)."""
    what = f"spawn record {s.car!r}"
    words = [_check_library_list(_int(o, "object", 0, 0xFFFFFFFF, what), f"{what} object {k}")
             for k, o in enumerate(_seq(s.objects, 8, "objects (8 words)", what))]
    return (struct.pack("<10I", *words, _int(s.line, "line", 0, AI_LINE_COUNT - 1, what),
                        _int(s.start, "start", *_U16, what))
            + _f32([s.speed], 1, "speed", what) + struct.pack("<I", _int(s.kind, "kind", 0, 0xFF, what)))


def _check_library_list(ptr: int, what: str) -> int:
    """A spawn object word must be an object list of OBJECT_LIBRARY (one 16-byte entry and the terminator on the real
    set): the list draw 0x17cf8 skips any list pointer outside 0x0283fbb4-0x0284f4eb."""
    lo, hi = OBJECT_LIBRARY
    if not lo <= ptr < hi:
        raise CourseError(f"{what}: {ptr:#x} is outside the object library {lo:#x}-{hi - 1:#x}")
    return ptr


def rolling_start(rs: RomSet, entry: CourseEntry) -> bool:
    """ROLLING_STARTS[id]: True for a rolling start (0), False for a standing start (-1). The game stores it to
    0x501850, the rolling start's countdown state (0x3a94; 0xc334 steps it 0 -> 1 -> 2 -> 3 at segments n-29, n-19,
    n-9), so other values are refused with CourseError."""
    at = id_table(rs, "ROLLING_STARTS") + 2 * entry.course
    v = struct.unpack("<h", _read(rs, at, 2, f"{entry.name}: rolling start"))[0]
    if v not in (0, -1):
        raise CourseError(f"{entry.name}: rolling-start word at {at:#x} is {v}, expected 0 (rolling) or -1 (standing)")
    return v == 0


def encode_rolling_start(rolling: bool) -> bytes:
    """A ROLLING_STARTS entry: s16 0 for a rolling start, -1 for a standing one. CourseError for a non-bool."""
    if not isinstance(rolling, bool):
        raise CourseError(f"rolling start {rolling!r} is not a bool")
    return struct.pack("<h", 0 if rolling else -1)


def encode_laps(n: int) -> bytes:
    """A LAP_COUNTS entry (u32). CourseError outside 1..MAX_LAPS."""
    return struct.pack("<I", _int(n, "laps", 1, MAX_LAPS, "lap count"))


def encode_car_count(n: int) -> bytes:
    """A CAR_COUNTS entry (u32). CourseError outside 1..SPAWN_RECORDS (the car pool, see car_count)."""
    return struct.pack("<I", _int(n, "car count", 1, SPAWN_RECORDS, "car count"))


def encode_table_pointer(addr: int) -> bytes:
    """A per-id CHECKPOINT_TABLES entry or a per-difficulty TIME_TABLES entry: the u32 i960 address of a table in data
    ROM. CourseError for an address outside data ROM."""
    _int(addr, "table pointer", 0, 0xFFFFFFFF, "pointer")
    _data_off(addr, 4, "table pointer")
    return struct.pack("<I", addr)


def race_value_addresses(rs: RomSet, entry: CourseEntry, mode: int = DEFAULT_MODE,
                         difficulty: int = DEFAULT_DIFFICULTY) -> dict[str, int]:
    """The i960 addresses of the course's per-course race values, for writing them in place: `laps` (u32),
    `car_count` (u32), `rolling_start` (s16), `grid_base` (s16), `checkpoint_pointer` / `time_pointer` (the u32 words
    that hold the tables' addresses) and `checkpoint_table` / `time_table` (where the tables are). The pointer arrays
    above them (per mode, per id) are fixed. CourseError as laps / time_table for a bad mode, difficulty or pointer."""
    laps(rs, entry, mode)
    time_table(rs, entry, mode, difficulty)
    by_id = _u32(rs, TIME_TABLES + 4 * mode)
    time_pointer = _u32(rs, by_id + 4 * entry.course) + 4 * difficulty
    checkpoint_pointer = id_table(rs, "CHECKPOINT_TABLES") + 4 * entry.course
    return {"laps": _u32(rs, LAP_COUNTS + 4 * mode) + 4 * entry.course,
            "car_count": _u32(rs, CAR_COUNTS) + 4 * entry.course,
            "rolling_start": id_table(rs, "ROLLING_STARTS") + 2 * entry.course,
            "grid_base": id_table(rs, "GRID_BASES") + 2 * entry.course,
            "checkpoint_pointer": checkpoint_pointer, "checkpoint_table": _u32(rs, checkpoint_pointer),
            "time_pointer": time_pointer, "time_table": _u32(rs, time_pointer)}


def encode_grid_base(base: int) -> bytes:
    """A GRID_BASES entry (s16). CourseError outside 0..0x7fff (the game adds it to a start and takes it mod n)."""
    return struct.pack("<h", _int(base, "grid base", 0, 0x7FFF, "grid"))


def step_table(rs: RomSet, entry: CourseEntry) -> list[Step]:
    """The step table's words (CourseEntry.step_table: u32 n, u32 steps, then steps (u16 segment, u16 flags)), one
    per road polygon. CourseError when it runs past data ROM or a segment is past n."""
    at = DATA_BASE + _data_off(DATA_BASE + entry.step_table, 8, f"{entry.name}: step table", rs)
    n, steps = _u32(rs, at), _u32(rs, at + 4)
    _data_off(at + 8, 4 * steps, f"{entry.name}: step table ({steps} steps)", rs)
    out = []
    for i in range(steps):
        seg, flags = struct.unpack("<2H", rs.read_i960(at + 8 + 4 * i, 4))
        if seg > n:
            raise CourseError(f"{entry.name}: step {i} at {at + 8 + 4 * i:#x}: segment {seg} is past n {n}")
        out.append(Step(seg, flags))
    return out


def encode_step_table(n: int, steps: list[Step]) -> bytes:
    """A step table: u32 n, u32 len(steps), then the steps. CourseError for an n outside the u16 segment range, a
    segment past n or flags outside u16."""
    n = _int(n, "n", 0, _MAX_SEGMENTS - 2, "step table")
    steps = _seq(steps, None, "steps", "step table")
    out = struct.pack("<2I", n, len(steps))
    for i, s in enumerate(steps):
        what = f"step {i}"
        out += struct.pack("<2H", _int(s.segment, "segment", 0, n, what), _int(s.flags, "flags", *_U16, what))
    return out


def _pit_run(segments: set[int], m: int, what: str) -> tuple[int, int]:
    """The first and last segment of `segments` as one run on the closed loop 0..m-1 (m = n + 1)."""
    starts = [s for s in sorted(segments) if (s - 1) % m not in segments]
    ends = [s for s in sorted(segments) if (s + 1) % m not in segments]
    if len(starts) != 1 or len(ends) != 1:
        raise CourseError(f"{what}: the pit polygons' segments {sorted(segments)} are not one run of the loop")
    return starts[0], ends[0]


def _pit_of(steps: list[Step], n: int, what: str) -> Pit | None:
    for i, s in enumerate(steps):
        if not is_int(s.segment) or not 0 <= s.segment <= n:
            raise CourseError(f"{what}: step {i} is on segment {s.segment!r}, outside the road line's 0..{n}")
    lane = tuple(i for i, s in enumerate(steps) if s.flags & STEP_PIT and not s.flags & STEP_PIT_BOX)
    box = [i for i, s in enumerate(steps) if s.flags & STEP_PIT_BOX]
    if not lane and not box:
        return None
    if len(box) != 1:
        raise CourseError(f"{what}: {len(box)} pit-box polygons {box}, expected one")
    entry_seg, exit_seg = _pit_run({steps[i].segment for i in lane + (box[0],)}, n + 1, what)
    return Pit(entry_seg, exit_seg, steps[box[0]].segment, lane, box[0])


def pit(rs: RomSet, entry: CourseEntry) -> Pit | None:
    """The course's pit lane from the step-table flags, or None when no polygon is flagged STEP_PIT. CourseError when
    there is not exactly one pit-box polygon, the pit polygons' segments are not one run of the loop, a step is on a
    segment outside the road line, or the step table's n differs from the road line's."""
    n = _u32(rs, DATA_BASE + entry.step_table)
    if n != _road_n(rs, entry):
        raise CourseError(f"{entry.name}: the step table's n {n} differs from the road line's {_road_n(rs, entry)}")
    return _pit_of(step_table(rs, entry), n, f"{entry.name}: step table")


def encode_pit(n: int, steps: list[Step], p: Pit) -> list[Step]:
    """`steps` (of a step table with road line `n`, the loop 0..n) with the pit written into the flags: the pit bits
    (STEP_PIT, STEP_PIT_LANE, STEP_PIT_BOX) of p.lane_polygons set to PIT_LANE_FLAGS and of p.box_polygon to
    PIT_BOX_FLAGS; every other bit of the word (Advanced's 0x100) is kept.
    CourseError for a polygon index outside `steps`, a polygon used twice or flagged STEP_ROAD, another polygon with
    any pit bit (a lone STEP_PIT_LANE too), a step on a segment past `n`, or a pit whose entry, exit or box segment
    differs from what its polygons give."""
    what = "pit"
    steps = _seq(steps, None, "steps", what)
    polys = list(_seq(p.lane_polygons, None, "lane polygons", what)) + [p.box_polygon]
    for i in polys:
        if not is_int(i) or not 0 <= i < len(steps):
            raise CourseError(f"{what}: polygon {i!r} is outside the {len(steps)} steps")
        if steps[i].flags & STEP_ROAD:
            raise CourseError(f"{what}: road polygon {i} (flags {steps[i].flags:#x}) cannot be a pit polygon")
    if len(set(polys)) != len(polys):
        raise CourseError(f"{what}: a polygon is listed twice in {polys}")
    stale = [i for i, s in enumerate(steps) if s.flags & _PIT_BITS and i not in polys]
    if stale:
        raise CourseError(f"{what}: polygons {stale} are already pit polygons and not in this pit")
    out = list(steps)
    for i in p.lane_polygons:
        out[i] = Step(out[i].segment, out[i].flags & ~_PIT_BITS | PIT_LANE_FLAGS)
    out[p.box_polygon] = Step(out[p.box_polygon].segment, out[p.box_polygon].flags & ~_PIT_BITS | PIT_BOX_FLAGS)
    _int(n, "n", 0, _MAX_SEGMENTS - 2, what)
    got = _pit_of(out, n, what)
    if got.box_segment != p.box_segment:
        raise CourseError(f"{what}: the box polygon is on segment {got.box_segment}, not {p.box_segment}")
    if (got.entry_segment, got.exit_segment) != (p.entry_segment, p.exit_segment):
        raise CourseError(f"{what}: the polygons run from entry {got.entry_segment} to exit {got.exit_segment}, not "
                          f"{p.entry_segment} to {p.exit_segment}")
    return out


# The pit stalls (v0.1.2): the per-id array CAR_TABLES_230E50. The trigger 0xdf9c starts the stop (handler 0xe288)
# when the player's car arrives on the box polygon; the stop 0xe384 steers the car (0xe744) to waypoint car +0xcc of
# its stall (id's entry -> u32[8] -> by car ident +0x4 & 7), sets its speed from the waypoint's third word, steps to the
# next waypoint once the one aimed at lies more than a quarter turn from the box polygon's direction (+0x1c6), and
# ends at the seventh; the stop itself (the car standing, the race timer held by 0x501a8c) counts 0x501a8c down from
# 640. A single cabinet is ident 0. docs/follow-ups/v0.1.2.md, "Pit stops".

PIT_STALLS = 8
"""Stalls per course id: one per car ident (car +0x4 & 7), the cabinets of a link."""
PIT_WAYPOINTS = 7
"""Waypoints per stall: the stop ends when car +0xcc reaches it (0xe3b4)."""
PIT_WAYPOINT_SIZE = 16
"""x, z, speed f32, then a word that is 0 on the original set (not read by 0xe744)."""
PIT_STALLS_SIZE = 4 * PIT_STALLS + PIT_STALLS * PIT_WAYPOINTS * PIT_WAYPOINT_SIZE
"""encode_pit_stalls' bytes: the eight stall pointers, then the stalls."""


class PitWaypoint(NamedTuple):
    """One waypoint of a pit stall: where the car steers (x, z, world units) and the speed it is set to (the third
    word, times a constant at 0x286f68c, to car +0x44)."""
    x: float
    z: float
    speed: float


def pit_stalls_at(rs: RomSet, course: int) -> int:
    """The address course id `course`'s CAR_TABLES_230E50 entry names: its eight stall pointers. CourseError for an id
    outside 0..15 or a word that cannot be read."""
    course = _int(course, "course id", 0, COURSE_IDS - 1, "pit stalls")
    return _u32(rs, id_table(rs, "CAR_TABLES_230E50") + 4 * course)


def pit_stalls(rs: RomSet, course: int) -> tuple[tuple[PitWaypoint, ...], ...]:
    """Course id `course`'s pit stalls: PIT_STALLS stalls (car ident 0-7) of PIT_WAYPOINTS waypoints, through its
    CAR_TABLES_230E50 entry (id_table: the grown table on a grown set). CourseError for an id outside 0..15, a word
    or record that cannot be read, a value that is not finite or a fourth word that is not 0."""
    at = pit_stalls_at(rs, course)
    what = f"course id {course}: pit stalls at {at:#x}"
    out = []
    for k in range(PIT_STALLS):
        p = _u32(rs, at + 4 * k)
        pts = []
        for j in range(PIT_WAYPOINTS):
            x, z, v, pad = struct.unpack("<3fI", _read(rs, p + PIT_WAYPOINT_SIZE * j, PIT_WAYPOINT_SIZE,
                                                       f"{what}: stall {k}"))
            if not all(math.isfinite(f) for f in (x, z, v)) or pad:
                raise CourseError(f"{what}: stall {k} waypoint {j} at {p + PIT_WAYPOINT_SIZE * j:#x} is "
                                  f"{(x, z, v, pad)}: not three finite floats and a 0 word")
            pts.append(PitWaypoint(x, z, v))
        out.append(tuple(pts))
    return tuple(out)


def encode_pit_stalls(stalls, addr: int) -> bytes:
    """The bytes of `stalls` (PIT_STALLS stalls of PIT_WAYPOINTS (x, z, speed)) for i960 address `addr`: the eight
    stall pointers, then the stalls back to back (x, z, speed f32, 0); PIT_STALLS_SIZE bytes, which a course id's
    CAR_TABLES_230E50 entry names by `addr`. CourseError for a wrong count, a value that is not finite or not exactly
    single precision, or an address that is not 4-aligned u32 room for the block."""
    what = "pit stalls"
    if not is_int(addr) or addr % 4 or not 0 <= addr <= 0xFFFFFFFF - PIT_STALLS_SIZE:
        raise CourseError(f"{what}: address {addr!r} is not 4-aligned room for {PIT_STALLS_SIZE} bytes")
    stalls = _seq(stalls, None, "stalls", what)
    if len(stalls) != PIT_STALLS:
        raise CourseError(f"{what}: {len(stalls)} stalls, expected {PIT_STALLS} stalls")
    body = bytearray()
    for k, st in enumerate(stalls):
        st = _seq(st, None, f"stall {k}", what)
        if len(st) != PIT_WAYPOINTS:
            raise CourseError(f"{what}: stall {k} has {len(st)} waypoints, expected {PIT_WAYPOINTS} waypoints")
        for j, w in enumerate(st):
            b = _f32(w, 3, f"stall {k} waypoint {j}", what)
            if struct.unpack("<3f", b) != tuple(float(v) for v in w):
                raise CourseError(f"{what}: stall {k} waypoint {j} {tuple(w)} is not exactly single precision")
            body += b + bytes(4)
    return struct.pack(f"<{PIT_STALLS}I", *(addr + 4 * PIT_STALLS + PIT_WAYPOINTS * PIT_WAYPOINT_SIZE * k
                                             for k in range(PIT_STALLS))) + bytes(body)


# --- Select screen, backdrop, rankings ---------------------------------------------------------------------------
#
# The tile-layer primitives they are built from (tile maps, chars, char and palette lists, upload sets, backdrop
# segments) are in dtools/arcade/tiles.py.

SELECT_TABLES = (0x0023862C, 0x002386E0, 0x00238790)
"""Program window: one table of Circuit Select panel records per game mode (0x2263b0: mode 0x5fe5e0 = 0, 1, other).
Each holds four SELECT_RECORD_SIZE records; 0x2269b4 draws the first SELECT_PANELS (the fourth is course id 3, the
stub)."""
SELECT_RECORD_SIZE = 44
SELECT_PANELS = 3
SELECT_VARIANTS = 3             # (normal, blink) map pairs per record, picked by the half-word 0x5fe59a
SELECT_PANEL_OF_COURSE = 0x00238F0C   # u32[4]: course id -> panel (0, 2, 1, 3)
PANEL_COURSES = (0, 2, 1)
"""Circuit Select panel order -> course id: panel 0 Beginner (id 0), 1 Advanced (id 2), 2 Expert (id 1). The panel is
what autoplay.lua's DAYTONA_COURSE picks; the mapping is its own inverse, so PANEL_COURSES[id] is also the panel of
course id `id` (the game's SELECT_PANEL_OF_COURSE array, ids 0-2)."""
SELECT_MAP_OBJECTS = 0x0023889C       # 32 bytes per panel: x, y, z (f32), the course-map object list, 4 more words
SELECT_SCREEN = (0x02600004, 0x02700004)
"""Data-ROM words holding the upload sets the Circuit Select load applies, in order (0x2d5c via 0x5cc4, then 0x5d68)."""
BOOT_SCREEN = (0x02600000, 0x02700000)  # the boot's upload sets (0x5cb8, 0x5d38); 0x5d38 first applies BOOT_CHAR_LIST
BOOT_CHAR_LIST = 0x0023A270     # program window: a char list the boot applies before 0x02700000
SELECT_PANEL_COUNT_AT = 0x002263E0     # `mov 3,g1`: the panel draw's count; select.patch_select makes it 16
_SELECT_PANEL_COUNT_WORD = 0x5C881E03
WHEEL_LO_AT = 0x00225F90        # `cmpibge lo,r3,..`: panel 0 when the wheel value is <= lo
WHEEL_HI_AT = 0x00225F98        # `shlo s,n,r5`: hi = n << s; panel 1 when <= hi, else panel 2
_WHEEL_LO_WORD, _WHEEL_LO_MASK = 0x3B00E014, 0x1F << 19
_WHEEL_HI_WORD, _WHEEL_HI_MASK = 0x592EDE02, (0x1F << 14) | 0x1F
_S32_MAX = 0x7FFFFFFF
PANEL_ROWS, PANEL_COLS = 10, 18
NAME_ROWS = range(1, 6)         # inside the frame: the course name
LABEL_ROWS = range(6, 9)        # inside the frame: the length and laps label
BACKDROP_LAYER = 2              # 0x23011c = 0x01004000 (layer 2); 0x230120 = 0x01006000 (layer 3, filled only)


# backdrop


class Backdrop(NamedTuple):
    course: int
    uploads: UploadSet          # the course's upload set (CourseEntry.upload_list): chars, palette, segments
    segments: tuple[BackdropSegment, ...]

    @property
    def top_tile(self) -> int:
        """The tile the load fills layer 2 above the band (and layer 3 below row 20) with: segment 0, row 0, col 0."""
        return self.segments[0].tiles[0]

    @property
    def bottom_tile(self) -> int:
        """The fill below the band: segment 0, last row, col 0 (0x1dd08: (rows - 1) * cols)."""
        s = self.segments[0]
        return s.tiles[(s.rows - 1) * s.cols]

    def panorama(self) -> np.ndarray:
        """The 256-column panorama the streamer scrolls through: (rows, 8 * 32) u16 tiles."""
        return np.concatenate([np.array(s.tiles, np.uint16).reshape(s.rows, s.cols) for s in self.segments], 1)

    def chars(self, blocks) -> dict[int, np.ndarray]:
        """The char RAM the upload set fills: char index -> (8, 8) pixels, read from `blocks` (block address ->
        (n, 8, 8) pixels, `golden.Course.char_blocks`), a later upload over an earlier one as 0x1dd70 copies them.
        CourseError for a block `blocks` lacks or whose char count is not the upload's."""
        if not isinstance(blocks, dict):
            raise CourseError(f"char blocks {blocks!r:.60} is not a dict (block address -> pixels)")
        out: dict[int, np.ndarray] = {}
        for u in self.uploads.chars:
            px = blocks.get(u.src)
            if not isinstance(px, np.ndarray) or px.ndim != 3 or px.shape[0] != u.count or px.shape[1:] != (8, 8):
                raise CourseError(f"backdrop of course {self.course!r}: char block {u.src!r} is missing or does not "
                                  f"hold the upload's {u.count!r} chars of 8 x 8")
            for k in range(u.count):
                out[u.first_char + k] = px[k]
        return out


def flip_chars(pixels) -> np.ndarray:
    """Chars flipped left to right (pixel x -> 7 - x): (8, 8) or (n, 8, 8) pixels, a new array. System 24 tiles
    have no flip bit (MAME segaic24.cpp: a tile is a char index and a colour), so a mirrored tile layer needs its
    flipped chars as chars of their own (`mirror.mirror_backdrop`). CourseError for another shape."""
    a = np.asarray(pixels)
    if a.shape[-2:] != (8, 8) or a.ndim not in (2, 3):
        raise CourseError(f"chars of shape {a.shape}: expected (8, 8) or (n, 8, 8)")
    return np.ascontiguousarray(a[..., ::-1])


def backdrop(rs: RomSet, entry: CourseEntry) -> Backdrop:
    """The race horizon backdrop of `entry`: its upload set (chars to 0x10e0000+, palette 0x60+) and the eight
    segments of the panorama streamed into tile layer 2 rows 6.. by 0x1dfd0. CourseError for a set without segments
    or segments of different heights."""
    addr = DATA_BASE + entry.upload_list
    if _u32(rs, addr + 8) == 0:
        raise CourseError(f"{entry.name}: the upload set at {addr:#x} has no backdrop segments")
    u = upload_set(rs, addr, segments=True)
    segs = tuple(backdrop_segment(rs, s) for s in u.segments)
    if len({s.rows for s in segs}) != 1:
        raise CourseError(f"{entry.name}: backdrop segments of different heights {[s.rows for s in segs]}")
    return Backdrop(entry.course, u, segs)


# select screen

class WheelZones(NamedTuple):
    """Circuit Select's wheel zones (0x225f74): the wheel value w = (s32 0x501050 + s32 0x50105c) >> 1 gives panel 0
    when w <= lo, 1 when w <= hi, else 2."""
    lo: int
    hi: int

    def panel(self, w: int) -> int:
        return 0 if w <= self.lo else 1 if w <= self.hi else 2


def wheel_zones(rs: RomSet) -> WheelZones:
    """The two thresholds, from the literals of the instructions at WHEEL_LO_AT and WHEEL_HI_AT. CourseError when
    either instruction is not the expected one (other opcode or registers)."""
    lo_w, hi_w = _u32(rs, WHEEL_LO_AT), _u32(rs, WHEEL_HI_AT)
    if lo_w & ~_WHEEL_LO_MASK != _WHEEL_LO_WORD & ~_WHEEL_LO_MASK:
        raise CourseError(f"wheel zones: the word at {WHEEL_LO_AT:#x} is {lo_w:#010x}, not `cmpibge lo,r3,0x225fa4`")
    if hi_w & ~_WHEEL_HI_MASK != _WHEEL_HI_WORD & ~_WHEEL_HI_MASK:
        raise CourseError(f"wheel zones: the word at {WHEEL_HI_AT:#x} is {hi_w:#010x}, not `shlo s,n,r5`")
    hi = (((hi_w >> 14) & 0x1F) << (hi_w & 0x1F)) & 0xFFFFFFFF      # `shlo` keeps 32 bits
    if hi > _S32_MAX:
        hi -= 1 << 32                                                # `cmpibge r5,r3` at 0x225f9c compares signed
    return WheelZones((lo_w >> 19) & 0x1F, hi)


def encode_wheel_zones(z: WheelZones) -> dict[int, bytes]:
    """The two instruction words, by address. CourseError for lo outside 0..31 (a 5-bit literal) or above hi, a hi
    past 0x7fffffff (`shlo` keeps 32 bits and the compare at 0x225f9c is signed, so it would wrap), or a hi that is
    not n << s with n, s in 0..31."""
    _int(z.lo, "lo", 0, 31, "wheel zones (5-bit literal)")
    if not is_int(z.hi) or z.hi < z.lo:
        raise CourseError(f"wheel zones: lo {z.lo} is above hi {z.hi!r}")
    if z.hi > _S32_MAX:
        raise CourseError(f"wheel zones: hi {z.hi:#x} is past {_S32_MAX:#x} (a signed 32-bit register)")
    for s in range(32):
        if z.hi >> s <= 31 and (z.hi >> s) << s == z.hi:
            n = z.hi >> s
            break
    else:
        raise CourseError(f"wheel zones: hi {z.hi} is not n << s with n, s in 0..31")
    lo_w = (_WHEEL_LO_WORD & ~_WHEEL_LO_MASK) | z.lo << 19
    hi_w = (_WHEEL_HI_WORD & ~_WHEEL_HI_MASK) | n << 14 | s
    return {WHEEL_LO_AT: struct.pack("<I", lo_w), WHEEL_HI_AT: struct.pack("<I", hi_w)}


class SelectPanel(NamedTuple):
    """One Circuit Select record: words 0-1 the cursor target (eased into the 3D window centre 0x50280c by 0x226244,
    where the red frame object 0x2843818 is drawn), word 2 the course id (0x226c74 reads its low byte), words 3-4 the
    panel's tile position in layer 0, words 5-10 the (normal, blink) tile maps of variants 0, 1, 2."""
    panel: int
    course: int
    cursor: tuple[int, int]
    tile: tuple[int, int]
    maps: tuple[int, ...]


def decode_select_panel(b: bytes, panel: int) -> SelectPanel:
    w = struct.unpack("<11I", b)
    return SelectPanel(panel, w[2], (w[0], w[1]), (w[3], w[4]), tuple(w[5:]))


def encode_select_panel(p: SelectPanel) -> bytes:
    """CourseError for a course id outside u8, a cursor outside 0..0xfff (packed as 12 bits), a tile position where
    a 10 x 18 panel does not fit the 64 x 64 layer, or maps that are not 6 data-ROM addresses."""
    what = f"select panel {p.panel}"
    course = _int(p.course, "course", 0, 0xFF, what)
    cursor = tuple(_int(v, "cursor", 0, 0xFFF, what) for v in _seq(p.cursor, 2, "cursor (x, y)", what))
    tx, ty = _seq(p.tile, 2, "tile (x, y)", what)
    tile = (_int(tx, "tile x", 0, TILE_LAYER - PANEL_COLS, what), _int(ty, "tile y", 0, TILE_LAYER - PANEL_ROWS, what))
    maps = tuple(_data_ptr_ok(m, f"{what}: map", 12)
                 for m in _seq(p.maps, 2 * SELECT_VARIANTS, f"maps ({2 * SELECT_VARIANTS})", what))
    return struct.pack("<11I", *cursor, course, *tile, *maps)


class MapObject(NamedTuple):
    """The 3D course map drawn on a panel (0x22653c, through 0x17acc): position, an object list in the object
    library, and words 4-7 (word 4 equals the list on the real set; no reader found)."""
    panel: int
    position: tuple[float, float, float]
    objects: int
    raw: tuple[int, int, int, int]


def encode_map_object(o: MapObject) -> bytes:
    """CourseError for a position that is not three finite floats, a list outside the object library or raw words
    outside u32."""
    what = f"map object {o.panel}"
    objects = _library_list(o.objects, f"{what}: list")
    raw = tuple(_int(v, "raw", 0, 0xFFFFFFFF, what) for v in _seq(o.raw, 4, "raw (4 words)", what))
    return _f32(o.position, 3, "position", what) + struct.pack("<5I", objects, *raw)


def _check_panel_of_course(pocs, panels, where) -> None:
    if len(pocs) != COURSE_COUNT:
        raise CourseError(f"panel of course: {len(pocs)} entries, expected {COURSE_COUNT}")
    for c, p in enumerate(pocs):
        if not is_int(p) or not 0 <= p < len(panels) or panels[p].course != c:
            shows = panels[p].course if is_int(p) and 0 <= p < len(panels) else "nothing"
            raise CourseError(f"panel of course {c}{where(c)} is {p!r}, but panel {p!r} shows {shows}")


def encode_panel_of_course(pocs, panels) -> bytes:
    """SELECT_PANEL_OF_COURSE for ids 0-2 (u32 each; the game's array has a fourth entry for id 3): the panel of
    each course id. `panels` are the records in panel order (one mode's SelectPanels). CourseError for another count,
    a panel outside the records, or an array that does not invert the records' course ids."""
    pocs = _seq(pocs, None, "panel of course")
    _check_panel_of_course(pocs, _seq(panels, None, "panels", "panel of course"), lambda c: "")
    return struct.pack(f"<{len(pocs)}I", *pocs)


class CourseSelect(NamedTuple):
    course: int
    panel: int
    tile: tuple[int, int]
    maps: tuple[tuple[tuple[TileMap, TileMap], ...], ...]   # [mode][variant] -> (normal, blink)
    name: tuple[int, ...]       # chars in NAME_ROWS inside the frame, over all maps
    label: tuple[int, ...]      # chars in LABEL_ROWS inside the frame
    map_object: MapObject


class SelectAssets(NamedTuple):
    panels: tuple[tuple[SelectPanel, ...], ...]   # per mode, panel order
    courses: tuple[CourseSelect, ...]             # id order
    wheel: WheelZones
    panel_of_course: tuple[int, ...]
    uploads: tuple[UploadSet, ...]                # boot sets, then the select sets, in the order applied
    char_source: dict[int, int]                   # char -> i960 address of its 32 bytes on Circuit Select


def _region_chars(m: TileMap, rows: range) -> set[int]:
    return {m.tile(r, c) & CHAR_MASK for r in rows for c in range(1, PANEL_COLS - 1)}


def select_assets(rs: RomSet) -> SelectAssets:
    """The Circuit Select screen: the panel records per mode, each course's panel maps, name and label chars, the
    course map object, the wheel zones, and where each char comes from (BOOT_CHAR_LIST, the BOOT_SCREEN and
    SELECT_SCREEN sets, in that order; the last upload of a char wins). `courses` holds every id of the set
    (id_count; ids 3-15 of a grown set show the panel SELECT_PANEL_OF_COURSE gives them, Beginner's; on a set with
    the sixteen-entry select, select.patch_select, that array names entries, and ids 3-15 are given Beginner's
    panel here: select.select_row decodes the row), `panel_of_course` ids 0-2's. CourseError for a record's course
    id not in 0..2, a panel-of-course array that does not invert the records (ids 0-2) or names no panel (3-15; an
    entry 0-15 on a patched set), a map that is not 10 x 18, or a panel char that no upload provides."""
    panels = []
    for mode, table in enumerate(SELECT_TABLES):
        row = []
        for i in range(SELECT_PANELS):
            at = table + SELECT_RECORD_SIZE * i
            p = decode_select_panel(_read(rs, at, SELECT_RECORD_SIZE, "select panel"), i)
            if p.course >= COURSE_COUNT:
                raise CourseError(f"select panel {i} (mode {mode}): the course id at {at + 8:#x} is {p.course}, "
                                  f"outside 0..{COURSE_COUNT - 1}")
            for m in p.maps:
                _data_ptr_ok(m, f"select panel {i} at {at:#x}: map", 12, rs)
            row.append(p)
        panels.append(tuple(row))
    if len({p.course for p in panels[0]}) != SELECT_PANELS or \
            any([p.course for p in t] != [p.course for p in panels[0]] for t in panels):
        raise CourseError(f"select panels: course ids {[[p.course for p in t] for t in panels]} differ by mode or repeat")
    poc_at = id_table(rs, "SELECT_PANEL_OF_COURSE")
    every = tuple(_u32(rs, poc_at + 4 * c) for c in range(id_count(rs)))
    pocs = every[:COURSE_COUNT]
    _check_panel_of_course(pocs, panels[0], lambda c: f" at {poc_at + 4 * c:#x}")
    row = _u32(rs, SELECT_PANEL_COUNT_AT) != _SELECT_PANEL_COUNT_WORD
    for c in range(COURSE_COUNT, len(every)):
        if row and 0 <= every[c] < COURSE_IDS:
            every = every[:c] + (0,) + every[c + 1:]    # an entry of the sixteen-entry row: no original panel
        elif not 0 <= every[c] < SELECT_PANELS:
            raise CourseError(f"panel of course {c} at {poc_at + 4 * c:#x} is {every[c]}, not a panel "
                              f"0..{SELECT_PANELS - 1}")
    uploads = []
    source: dict[int, int] = {}

    def apply(chars):
        for c in chars:
            for k in range(c.count):
                source[c.first_char + k] = c.src + 4 + CHAR_SIZE * k

    apply(char_list(rs, BOOT_CHAR_LIST))
    for slot in BOOT_SCREEN + SELECT_SCREEN:
        u = upload_set(rs, _u32(rs, slot), segments=False)
        uploads.append(u)
        apply(u.chars)
    courses = []
    for c in range(len(every)):
        pi = every[c]
        maps = []
        for mode in range(len(SELECT_TABLES)):
            p = panels[mode][pi]
            pairs = []
            for v in range(SELECT_VARIANTS):
                pair = tuple(tile_map(rs, p.maps[2 * v + k]) for k in range(2))
                for k, m in enumerate(pair):
                    if (m.rows, m.cols) != (PANEL_ROWS, PANEL_COLS):
                        raise CourseError(f"select map at {p.maps[2 * v + k]:#x} is {m.rows} x {m.cols}, "
                                          f"not {PANEL_ROWS} x {PANEL_COLS}")
                pairs.append(pair)
            maps.append(tuple(pairs))
        flat = [m for per_mode in maps for pair in per_mode for m in pair]
        name = tuple(sorted(set().union(*(_region_chars(m, NAME_ROWS) for m in flat))))
        label = tuple(sorted(set().union(*(_region_chars(m, LABEL_ROWS) for m in flat))))
        for ch in sorted({t & CHAR_MASK for m in flat for t in m.tiles()}):
            if ch not in source:
                raise CourseError(f"{course_name(c)}: panel char {ch:#x} is not uploaded by any boot or select set")
        at = SELECT_MAP_OBJECTS + 32 * pi
        w = _read(rs, at, 32, "map object")
        words = struct.unpack_from("<5I", w, 12)
        obj = MapObject(pi, struct.unpack_from("<3f", w), words[0], words[1:])
        courses.append(CourseSelect(c, pi, panels[0][pi].tile, tuple(maps), name, label, obj))
    return SelectAssets(tuple(panels), tuple(courses), wheel_zones(rs), pocs, tuple(uploads), source)


# HUD map

HUD_MAP_LISTS = 0x028052DC      # data: u32[4] per course id -> the outline's object list (0x12ee0 -> HUD struct +0x14)
HUD_MAP_PLACES = 0x028052FC     # data: HUD_MAP_PLACE_SIZE bytes per id: translation x, y, z (f32), angle (s16),
HUD_MAP_PLACE_SIZE = 16         #   pad (u16) (0x12ef0 -> HUD struct +0x3c..+0x48; the angle is read with ldis)
HUD_MAP_ZOOMS = 0x0280534C      # data: f32[HUD_MAP_ZOOM_LEVELS] per id: the scale, by zoom level (HUD struct +0x4a)
HUD_MAP_ZOOM_LEVELS = 4
HUD_MAP_MODE = 1                # the race's geometry mode (golden.GEOMETRY_MODE): the outline decodes in it


@dataclass(frozen=True)
class HudMap:
    """The race HUD's course map (docs/formats/model2-course-tables.md, "HUD map"; found in
    docs/spike/2026-11-mirror.md, "HUD map"). Every frame 0x1308c sends the TGP: translate `position` (view
    coordinates), scale by the zoom (eased towards `zooms[level]`, level 1 from the set-up 0x12e58), a quarter turn
    (function 0x14, 0x4000) and `angle` (function 0x15), then draws the outline list. The outline is a polygon object
    in world coordinates (x, z; y 0): the cars' markers are drawn in the same frame at each car's world position
    (+0x1c, +0x24) and yaw (+0x58), so nothing but the outline and the cars places them."""
    course: int
    outline_list: int                       # HUD_MAP_LISTS[id]: i960 address of the outline's object list
    entries: tuple[Placement, ...]          # the list's entries (one on the real set), not placed (cell -1)
    objects: dict[int, polyrom.Object]      # polygon-ROM word index -> the object an entry names
    position: Point                         # view-space translation (f32)
    angle: int                              # s16, 0x10000 per turn (TGP function 0x15)
    pad: int                                # u16 after the angle (0 on the real set; no reader)
    zooms: tuple[float, ...]                # HUD_MAP_ZOOM_LEVELS scales, world units -> view units


def hud_map(rs: RomSet, entry: CourseEntry) -> HudMap:
    """The course's HUD map: HUD_MAP_LISTS[id], its object list and objects (geometry mode HUD_MAP_MODE), the place
    record HUD_MAP_PLACES[id] and the zooms HUD_MAP_ZOOMS[id]. CourseError for a list outside the object library
    (PATCHED_LIBRARY here; `object_list` then holds it to the set's `library_bounds`) or malformed (as
    `object_list`); PolyError for an object that runs past the polygon ROM."""
    if not isinstance(entry, CourseEntry):
        raise CourseError(f"{entry!r:.60} is not a CourseEntry")
    _int(entry.course, "course", 0, id_count(rs) - 1, "HUD map")
    lst = _per_course(rs, id_table(rs, "HUD_MAP_LISTS"), entry, *PATCHED_LIBRARY, "HUD map list")
    entries = tuple(object_list(rs, lst, f"{entry.name}: HUD map"))
    objects = {}
    for p in entries:
        index = check_rom_object(p.oba, rs)
        objects.setdefault(index, polyrom.decode_object(rs.polygons, index, p.obc, HUD_MAP_MODE))
    place = _read(rs, id_table(rs, "HUD_MAP_PLACES") + HUD_MAP_PLACE_SIZE * entry.course, HUD_MAP_PLACE_SIZE,
                  "HUD map place")
    x, y, z, angle, pad = struct.unpack("<3fhH", place)
    zooms = struct.unpack(f"<{HUD_MAP_ZOOM_LEVELS}f",
                          _read(rs, id_table(rs, "HUD_MAP_ZOOMS") + 4 * HUD_MAP_ZOOM_LEVELS * entry.course,
                                4 * HUD_MAP_ZOOM_LEVELS, "HUD map zooms"))
    return HudMap(entry.course, lst, entries, objects, (x, y, z), angle, pad, zooms)


def encode_hud_map(m: HudMap, arrays: dict[str, int] | None = None) -> list[tuple[int, bytes]]:
    """(i960 address, bytes) of the HUD map's tables, in order: the list pointer HUD_MAP_LISTS[id], the outline list
    (entries and terminator), the place record HUD_MAP_PLACES[id], the zooms HUD_MAP_ZOOMS[id]. `arrays` gives the
    three arrays' addresses (id_tables of the set; golden.Course.id_arrays), the original ones when None. The outline
    objects are polygon-ROM data, not on the i960 bus: `polyrom.encode_object` writes them (golden.tables, "HUD map
    outline"). CourseError for a course id outside 0..COURSE_COUNT-1 (0..COURSE_IDS-1 with grown `arrays`), a list
    outside the object library (PATCHED_LIBRARY: drawn on a set with LIBRARY_BOUND_PATCH), no entry, an entry whose
    object `objects` lacks or an object no entry names, a position that is not three finite f32, an angle outside
    s16, a pad outside u16, zooms that are not HUD_MAP_ZOOM_LEVELS finite f32."""
    if not isinstance(m, HudMap):
        raise CourseError(f"{m!r:.60} is not a HudMap")
    course = _int(m.course, "course", 0, ids_of(arrays) - 1, "HUD map")
    what = f"HUD map of course {course}"
    lst = _library_list(m.outline_list, f"{what}: list", PATCHED_LIBRARY)   # DATA_FREE: a new id's own list
    entries = _seq(m.entries, None, "entries", what)
    if not entries:
        raise CourseError(f"{what}: the list has no entry (the outline would not be drawn)")
    blob = encode_object_list(entries)
    if not isinstance(m.objects, dict) or not all(is_int(k) for k in m.objects):
        raise CourseError(f"{what}: objects {m.objects!r:.60} is not a dict of polygon-ROM word indices")
    named = {rom_object_index(p.oba) for p in entries}
    for index in sorted(named | set(m.objects)):
        if index not in m.objects:
            raise CourseError(f"{what}: object {index:#x} named by the list is not in objects")
        if index not in named:
            raise CourseError(f"{what}: object {index!r} is named by no entry of the list")
        if not isinstance(m.objects[index], polyrom.Object):
            raise CourseError(f"{what}: object {index:#x} {m.objects[index]!r:.40} is not a polyrom.Object")
    place = (_f32(m.position, 3, "position", what)
             + struct.pack("<hH", _int(m.angle, "angle", *_S16, what), _int(m.pad, "pad", *_U16, what)))
    lists, places, zooms = (array_at(arrays, n) for n in ("HUD_MAP_LISTS", "HUD_MAP_PLACES", "HUD_MAP_ZOOMS"))
    return [(lists + 4 * course, struct.pack("<I", lst)), (lst, blob),
            (places + HUD_MAP_PLACE_SIZE * course, place),
            (zooms + 4 * HUD_MAP_ZOOM_LEVELS * course, _f32(m.zooms, HUD_MAP_ZOOM_LEVELS, "zooms", what))]


# rankings (backup SRAM)

SRAM = (0x01D00000, 0x01D04000)
RANKING_TABLES = 0x002390E4     # program window: u32[4] per course id -> its ranking table (0x227594 -> 0x50138c)
RANKING_LAST = 0x002390F4       # u32[4] per id -> the SRAM word that points at the newest entry (-> 0x5013b4)
RANKING_DEFAULTS = 0x002390CC   # (defaults, table) pairs per id, read by the boot-time reset 0x2270d0
RANKING_ENTRIES = 24
RANKING_ENTRY_SIZE = 24


class RankingsLayout(NamedTuple):
    base: int
    size: int
    signature: tuple[int, int]              # (address, length): "S-RAM CHECK OK (4.0)" checked at boot (0x5a98)
    signature_source: int                   # program address of the expected bytes
    tables: tuple[int, ...]                 # per course id
    last_entry: tuple[int, ...]             # per course id
    entries: int
    entry_size: int
    fields: tuple[tuple[str, int, int], ...]   # (name, offset, size) in an entry
    used: tuple[tuple[str, int, int], ...]     # other areas the game keeps in SRAM, (name, lo, hi exclusive)
    free: tuple[int, int]

    def entry(self, course: int, rank: int) -> int:
        _int(course, "course", 0, len(self.tables) - 1, "rankings")
        _int(rank, "rank", 0, self.entries - 1, "rankings (0-based)")
        return self.tables[course] + self.entry_size * rank

    def field(self, course: int, rank: int, name: str) -> int:
        for n, off, _ in self.fields:
            if n == name:
                return self.entry(course, rank) + off
        raise CourseError(f"rankings: no field {name!r}")


def rankings_layout(rs: RomSet | None = None) -> RankingsLayout:
    """Where the per-course race rankings live in backup SRAM 0x01d00000-0x01d03fff (a layout, no data). Each table
    is RANKING_ENTRIES entries sorted by `time` (0x227508 bubbles a new entry up from the last slot, which 0x227438
    overwrites only when the new time is lower); `rank` is the slot's fixed number 1-24 and is not moved. Found by a
    traced write at the end of a 1-lap Beginner race (docs/spike/2026-11-arcade-course-tables.md, "Rankings").

    Without `rs`, or for a set that is not ranked: the stock layout (three tables). For a ranked set (grow_rankings):
    its sixteen tables and newest-entry words as RANKING_TABLES and RANKING_LAST name them; `used` the stock areas
    but the old tables and words (no longer read), with RANKING_PAD and the new ones; `free` the rest after them.
    CourseError for `rs` neither None nor a RomSet, or a ranked set's entry outside the new area."""
    stock = RankingsLayout(
        base=SRAM[0], size=SRAM[1] - SRAM[0],
        signature=(0x01D00200, 24), signature_source=0x00005B78,
        tables=(0x01D00360, 0x01D005A0, 0x01D007E0),
        last_entry=(0x01D00A20, 0x01D00A24, 0x01D00A2C),
        entries=RANKING_ENTRIES, entry_size=RANKING_ENTRY_SIZE,
        fields=(("rank", 0x00, 4),          # u32 1..24, the slot's number
                ("time", 0x04, 4),          # u32 race time 0x5010cc (frames), the sort key
                ("view", 0x08, 4),          # u32 the most used view 0-3 (largest of 0x501700-0x50170c)
                ("flags", 0x0C, 4),         # u32 bit 0 = 0x501a80, bit 1 = 0x501a90 (not interpreted)
                ("shown_time", 0x10, 4),    # u32 the time the ranking screen shows (0x2285d0); = time when written
                ("name", 0x14, 4)),         # 3 ASCII bytes (low 24 bits); 0x202020 until the name entry (0x228490)
        used=(("system: settings, a copy, counters", 0x01D00000, 0x01D00200),
              ("signature", 0x01D00200, 0x01D00218),
              ("view presets: 16 x 20 bytes", 0x01D00220, 0x01D00360),
              ("newest-entry pointers", 0x01D00A20, 0x01D00A30),
              ("settings 0x2270d0", 0x01D00B60, 0x01D00BA4),
              ("settings 0x2271fc", 0x01D00BE0, 0x01D00C28)),
        free=(0x01D00C28, SRAM[1]))
    if rs is None:
        return stock
    if not ranked(rs):              # CourseError for a non-RomSet
        return stock
    out = []
    for name, lo, hi, n in (("RANKING_TABLES", RANKING_SRAM, RANKING_LAST_SRAM, RANKING_TABLE_SIZE),
                            ("RANKING_LAST", RANKING_LAST_SRAM, RANKING_LAST_SRAM + 4 * COURSE_IDS, 4)):
        words = struct.unpack(f"<{COURSE_IDS}I", _read(rs, id_table(rs, name), 4 * COURSE_IDS, name))
        for k, a in enumerate(words):
            if not lo <= a or a + n > hi:
                raise CourseError(f"{name}[{k}] of a ranked set is {a:#x}, outside {lo:#x}..{hi:#x}")
        out.append(words)
    end = RANKING_LAST_SRAM + 4 * COURSE_IDS
    used = tuple(u for u in stock.used if u[0] != "newest-entry pointers")
    used += (("time-attack records' padding word (0x230b9c)", RANKING_PAD, RANKING_PAD + 4),
             ("ranking tables, 16 x 576 bytes", RANKING_SRAM, RANKING_LAST_SRAM),
             ("newest-entry words, 16 x 4 bytes", RANKING_LAST_SRAM, end))
    return stock._replace(tables=out[0], last_entry=out[1], used=used, free=(end, SRAM[1]))


def ranking_pointers(rs: RomSet) -> tuple[int, ...]:
    """The per-id table addresses at RANKING_TABLES (id_table; ids 0-2). CourseError for one whose table does not fit
    SRAM."""
    out, table = [], id_table(rs, "RANKING_TABLES")
    for c in range(COURSE_COUNT):
        a = _u32(rs, table + 4 * c)
        if not SRAM[0] <= a or a + RANKING_ENTRIES * RANKING_ENTRY_SIZE > SRAM[1]:
            raise CourseError(f"ranking table of course {c} at {table + 4 * c:#x} is {a:#x}, outside SRAM")
        out.append(a)
    return tuple(out)


def encode_ranking_pointers(tables) -> bytes:
    """u32 per course id (ids 0-2; the game's array has a fourth entry for id 3). CourseError for another count, an
    address that is not 4-aligned or a table that does not fit SRAM."""
    tables = _seq(tables, None, "tables", "ranking pointers")
    if len(tables) != COURSE_COUNT:
        raise CourseError(f"ranking pointers: {len(tables)} tables, expected {COURSE_COUNT}")
    for a in tables:
        if not is_int(a) or a % 4 or not SRAM[0] <= a or a + RANKING_ENTRIES * RANKING_ENTRY_SIZE > SRAM[1]:
            raise CourseError(f"ranking pointers: table {a!r} is not a 4-aligned table inside SRAM")
    return struct.pack(f"<{len(tables)}I", *tables)


# --- course ids 0-15 (M4a) -----------------------------------------------------------------------------------------
# Every array the game indexes by the course id (the byte 0x501460) holds four entries on the original set (id 3 is a
# stub). grow_ids moves each to a sixteen-entry table in the custom set's new data area (romset.CUSTOM_AREAS, i960
# 0x03000000+; reserved for it, romset.RESERVED) and repoints its readers. The readers were found by disassembly
# (every instruction that loads 0x501460 or 0x501464, followed to the table it indexes) and by tracing reads over the
# old arrays: the discovery log docs/spike/2026-11-sixteen.md, "Ids"; the layouts in
# docs/formats/model2-course-tables.md, "Course ids 0-15".

COURSE_IDS = 16
"""Course ids on a grown set (grow_ids): 0 Beginner, 1 Expert, 2 Advanced as before; 3 Desert City (Phase 8, Task 5);
4-7 reserved for M5's courses; 8-15 the mirrors (M4b). Until an entry is filled, ids 3-15 hold Beginner's values."""
ID_HOME = 0x03000000
"""i960 address of the grown tables: the first byte of the custom data area (romset.CUSTOM_AREAS["data"], data offset
0x1000000). Up to SRAM_SINK it is grow_ids' home, reserved (romset.RESERVED) whatever its bytes: the unused record
slots 0-2 stay 0xff and are not free. The original set has no ROM there, so a read the game makes there is the proof
the patched readers run (the positive control, tests/test_sixteen_mame.py)."""
ID_RECORDS = ID_HOME
"""The grown course records: COURSE_ENTRY_SIZE bytes per id, record `i` at ID_RECORDS + 0x20 * i for ids 3-15 (the
first three slots stay unused: ids 0-2 keep their records at COURSE_TABLE, which Beginner's line table follows at
0x026000a0, so there is no room there for more)."""


class IdArray(NamedTuple):
    """One per-id array: `old` (i960 address of entry 0 on the original set), `new` (its sixteen-entry home on a grown
    set), `stride` (bytes per id), `sites` (i960 addresses of the words that hold the array's address: a reader's
    32-bit displacement, i.e. the instruction address + 4, or a pointer word of the level above), `keep` (entries
    grow_ids copies; the rest get entry 0's bytes, Beginner's), `entries` (the original array's entries: 4, id 3
    the stub's, or 3 where the fourth slot is another table's first word), `note` (what reads it)."""
    old: int
    new: int
    stride: int
    sites: tuple[int, ...]
    keep: int
    entries: int
    note: str

    @property
    def old_span(self) -> tuple[int, int]:
        """(i960 address, bytes) of the original array: `entries` entries of `stride` bytes."""
        return self.old, self.stride * self.entries


_ID_SPECS = (
    # name, old address, stride, sites (displacement / pointer words), entries kept, readers and use
    ("COURSE_POINTERS", COURSE_POINTERS, 4, (0x46F4,), 3, "0x46f0 ld [id*4]: the course record (ids 3-15: ID_RECORDS)"),
    ("COURSE_SHEETS", 0x000015A0, 4, (0x1448,), 3, "0x1444 ld [id*4]: the texel block 0x146c uploads (texture.py)"),
    ("ATTRACT_NEXT", 0x00002160, 1, (0x2120,), 3, "0x211c ldob [id]: the next course id (then id rem 4, 0x2138)"),
    ("OUTPUT_CODES", 0x000048A4, 1, (0x4794,), 3, "0x4790 ldob [id]: or-ed with 0x48a8[0x501464] | 0x70 into the "
                                                  "0x502900 queue (not traced)"),
    ("CAMERA_SCRIPTS", CAMERA_SCRIPTS, 4, (0x63E8,), 3, "0x63e4 lda, 0x6418 ld [id*4]: the pre-race camera script"),
    ("SHIFTS_8410", 0x00008410, 1, (0x83FC,), 3, "0x83f8 ldob [id]: a shift count, result to 0x5014f7"),
    ("TABLES_DF44", 0x0000DF44, 4, (0xDEE8,), 3, "0xdee4 ld [id*4]: a per-id u8 threshold table (0xdeec)"),
    ("PROGRESS_MASKS", 0x000175E0, 4, (0x176A0,), 3, "0x1769c ld [id*4]: the progress-mask records (0x17688)"),
    ("ACTOR_TABLES", ACTOR_TABLES, 4, (0x46B8,), 3, "0x46b4 ld [id*4] -> 0x501444"),
    ("GRID_BASES", GRID_BASES, 2, (0xE8E0,), 3, "0xe8dc ldis [id*2]: the grid's base segment"),
    ("CAR_TABLES_230E50", 0x00230E50, 4, (0xE758,), 3, "0xe754 ld [id*4]: the pit stalls, u32[8] per car ident "
                                                         "(pit_stalls)"),
    ("OFFSETS_230B18", 0x00230B18, 4, (0x63BC,), 3, "0x63b8 ld [id*4]: an index offset into 0x230b28"),
    ("TUNNEL_BYTES", 0x00230742, 1, (0x6D0C,), 3, "0x6d08 ldob [id] -> car +0x85"),
    ("TUNNEL_STRIDE", 0x00230746, 4, (0x6D18,), 1, "0x6d14 ld: the stride of TUNNEL_LIGHTS (20), not per id; moved "
                                                  "with the records it strides (entry 0 is the one read)"),
    ("TUNNEL_LIGHTS", 0x0023074A, 20, (0x6D24,), 3, "0x6d20 lda (id * the word at 0x230746, 20): five words to "
                                                    "+0x70-+0x80"),
    ("WORDS_230CCC", 0x00230CCC, 4, (0xB668, 0xBADC), 3, "0xb664 ld [id*4] -> 0x501ac4; 0xbad8 ld [g9*4] (g9 = the "
                                                       "id, from 0xb818: time attack)"),
    ("TABLES_231960", 0x00231960, 4, (0xCED0,), 3, "0xcecc ld [id*4]: a per-id table (id 3: not a pointer)"),
    ("OFFSETS_231B34", 0x00231B34, 4, (0xD420,), 3, "0xd41c ld [id*4]: an index offset into 0x2880004"),
    ("FLOATS_231C44", 0x00231C44, 4, (0xD47C,), 3, "0xd478 ld [(id & 3)*4] -> +0x20 (mask widened)"),
    ("FLOATS_231C54", 0x00231C54, 4, (0xD4A4,), 3, "0xd4a0 ld [(id & 3)*4] (mask widened)"),
    ("FLAGS_23329C", 0x0023329C, 4, (0x195D4, 0x19708), 3, "0x195d0, 0x19704 ld [(id & 3)*4] (masks widened)"),
    ("TABLES_2332AC", 0x002332AC, 4, (0x19880,), 3, "0x1987c ld [id*4]"),
    ("TABLES_236F58", 0x00236F58, 4, (0x1C4F4,), 3, "0x1c4f0 ld [id*4]: per car ident & 7, 16 bytes"),
    ("TABLES_2370E8", 0x002370E8, 4, (0x1C534,), 3, "0x1c530 ld [id*4]: per car ident, 32 bytes"),
    ("TABLES_232CB0", 0x00232CB0, 4, (0x16BEC,), 3, "0x16be8 ld [id*4] (r4 = the id from 0x16b94)"),
    ("TABLES_236B40", 0x00236B40, 4, (0x2201C8,), 3, "0x2201c4 ld [id*4]: per actor ident, 16 bytes"),
    ("SELECT_238EFC", 0x00238EFC, 4, (0x225B58,), 3, "0x225b54 ld [(id & 3)*4] (mask widened): select screen"),
    ("SELECT_UPLOADS", 0x0023899C, 4, (0x225EA8,), 3, "0x225ea4 ld [id*4] -> 0x5fe5e4 (data 0x02600008-)"),
    ("SELECT_SOUNDS", 0x00238EE4, 4, (0x225F68,), 3, "0x225f64 ld [id*4]: a sound call (0x1a064)"),
    ("SELECT_PANEL_OF_COURSE", SELECT_PANEL_OF_COURSE, 4, (0x226460, 0x22657C), 3, "0x22645c, 0x226578 ld [id*4]"),
    ("SELECT_2389AC", 0x002389AC, 4, (0x226BFC,), 3, "0x226bf8 ld [id*4]: an upload (0x19304)"),
    ("SELECT_PALETTES", 0x0023895C, 16, (0x226C2C,), 3, "0x226c28 lda [id*16]: the select palette animation"),
    ("RANKING_SCREEN_2273E8", 0x002273E8, 8, (0x22725C,), 3, "0x227258 lda [id*8]: two words (tile position)"),
    ("RANKING_SCREEN_2273D8", 0x002273D8, 4, (0x227284,), 3, "0x227280 ld [id*4]: an upload (0x19304)"),
    ("RANKING_TABLES", RANKING_TABLES, 4, (0x22759C,), 3, "0x227598 ld [(id & 3)*4] -> 0x50138c (mask widened)"),
    ("RANKING_LAST", RANKING_LAST, 4, (0x2275AC,), 3, "0x2275a8 ld [(id & 3)*4] -> 0x5013b4"),
    ("TIME_ATTACK_230A90", 0x00230A90, 4, (0x00230A88,), 3, "0xbf8c ld (0x230a88[gear])[id*4]: SRAM words"),
    ("TIME_ATTACK_230AA0", 0x00230AA0, 4, (0x00230A8C,), 3, "0xbf8c ld (0x230a88[gear])[id*4]: SRAM words"),
    ("TIME_ATTACK_230AB8", 0x00230AB8, 8, (0x00230A70, 0x00230AB0), 3, "0xb524, 0xc0b4 lda (per gear)[id*8]: SRAM "
                                                                       "word pairs"),
    ("TIME_ATTACK_230AD8", 0x00230AD8, 8, (0x00230A78, 0x00230AB4), 3, "0xb524, 0xc0b4 lda (per gear)[id*8]: SRAM "
                                                                       "word pairs"),
    ("ROLLING_STARTS", ROLLING_STARTS, 2, (0x3A90,), 3, "0x3a8c ldis [id*2] -> 0x501850"),
    ("MUSIC", 0x028050F0, 4, (0x3BE4,), 4, "0x3be0 ld [music*4]: the music id 0x501a98 is the course id unless a view "
                                           "button picks 0-3, so all four entries are kept"),
    ("BEST_LAPS_2805100", 0x02805100, 4, (0x4260,), 3, "0x425c ld [id*4]: an SRAM word (written)"),
    ("BEST_LAPS_2805110", 0x02805110, 4, (0x4274,), 3, "0x4270 ld [id*4]: an SRAM word (written)"),
    ("TIME_TABLES[0]", 0x0281F458, 4, (TIME_TABLES,), 3, "0x45ec ld (TIME_TABLES[mode])[id*4]"),
    ("TIME_TABLES[1]", 0x0281F468, 4, (TIME_TABLES + 4,), 3, "0x45ec ld (TIME_TABLES[mode])[id*4]"),
    ("TIME_TABLES[2]", 0x0281F478, 4, (TIME_TABLES + 8,), 3, "0x45ec ld (TIME_TABLES[mode])[id*4]"),
    ("LAP_COUNTS[0]", 0x02805248, 4, (LAP_COUNTS,), 3, "0x45f0 ld (LAP_COUNTS[mode])[id*4]"),
    ("LAP_COUNTS[1]", 0x02805258, 4, (LAP_COUNTS + 4,), 3, "0x45f0 ld (LAP_COUNTS[mode])[id*4]"),
    ("LAP_COUNTS[2]", 0x02805268, 4, (LAP_COUNTS + 8,), 3, "0x45f0 ld (LAP_COUNTS[mode])[id*4]"),
    ("CAR_COUNTS[0]", 0x02805278, 4, (CAR_COUNTS,), 3, "0x45f4 ld (CAR_COUNTS[setting])[id*4]"),
    ("CAR_COUNTS[1]", 0x02805288, 4, (CAR_COUNTS + 4,), 3, "0x45f4 ld (CAR_COUNTS[setting])[id*4]"),
    ("CHECKPOINT_TABLES", CHECKPOINT_TABLES, 4, (0x462C, 0x463C), 3, "0x4628, 0x4638 ld [id*4] -> 0x50143c"),
    ("GRID_TABLES", GRID_TABLES, 4, (0x464C,), 3, "0x4648 ld [id*4] -> 0x501420"),
    ("SEGMENT_FLAGS", SEGMENT_FLAGS, 4, (0x46C8,), 3, "0x46c4 ld [id*4] -> 0x501450"),
    ("HUD_MAP_LISTS", HUD_MAP_LISTS, 4, (0x12EE4,), 3, "0x12ee0 ld [id*4]"),
    ("HUD_MAP_PLACES", HUD_MAP_PLACES, HUD_MAP_PLACE_SIZE, (0x12EF4,), 3, "0x12ef0 lda [(id << 2)*4]"),
    ("HUD_MAP_ZOOMS", HUD_MAP_ZOOMS, 4 * HUD_MAP_ZOOM_LEVELS, (0x12ED0, 0x130D4), 3,
     "0x12ecc, 0x130d0 ld [((id << 2) + level)*4]"),
    ("TIME_CORRECTIONS_SUB", 0x0280587C, 4, (0x13B2C,), 3, "0x13b28 ld [id*4]: per difficulty, subtracted"),
    ("TIME_CORRECTIONS_ADD", 0x0280583C, 4, (0x13B5C,), 3, "0x13b58 ld [id*4]: per difficulty, added"),
    ("WORDS_280539C", 0x0280539C, 4, (0x13BB0,), 3, "0x13bac ld [id*4]: times the laps (0x13bbc)"),
    ("TABLES_28053AC", 0x028053AC, 4, (0x13BE0,), 3, "0x13bdc ld [id*4]: sound calls per lap"),
    ("TABLES_280538C", 0x0280538C, 4, (0x13C00,), 3, "0x13bfc ld [id*4]: half-words per lap -> 0x501a02"),
    ("TABLES_281399C", 0x0281399C, 4, (0x226DAC,), 3, "0x226da8 ld [id*4] -> 0x501858"),
    ("TABLES_281397C", 0x0281397C, 4, (0x02813978,), 3, "0x226e00 ld (0x2813974[setting])[id*4] -> 0x501858"),
    ("TABLES_281398C", 0x0281398C, 4, (0x02813974,), 3, "0x226e00 ld (0x2813974[setting])[id*4] -> 0x501858"),
)
_THREE_ENTRIES = frozenset({"TABLES_231960"})     # 0x23196c, its fourth slot, is Beginner's table (entry 0's target)

WIDE_HOME = 0x0001F800
"""Program ROM (i960 address = program offset): widen_ids' per-id tables and code, WIDE_SIZE bytes near the end of the
program's 0xff run after the last code (reserved: romset.RESERVED; the select's code keeps the run's first part,
RANK_HOME its last 0x200 bytes)."""
WIDE_SIZE = 0x600                       # then RANK_HOME (grow_rankings)
WIDE_LANES = WIDE_HOME                  # GRID_LANES per id: u16[2] per id (64 bytes)
WIDE_BILLBOARD = WIDE_HOME + 0x40       # BILLBOARD_REF per id: f32[3] per id (192 bytes)
TUNES = WIDE_HOME + 0x100               # u8 per id: the music id the race plays (16 bytes)
VISIT_IDS = WIDE_HOME + 0x110           # u8 per id: whose culling-walk test the id takes (16 bytes)
WIDE_CODE = WIDE_HOME + 0x200           # widen_ids' code: the shadows' update and the writers' trampolines

WIDE_RAM = 0x00501B40
"""Work RAM (romset.RESERVED_RAM; touched by nothing else in a whole session: traced,
docs/spike/2026-11-mirrors-live.md, "Literal ids", trace sessions 1-3, and docs/spike/2026-11-sixteen.md): the shadows
of the course id's per-id values that the widened sites read (WIDE_RAM_SIZE bytes, cleared by the boot)."""
SH_PATH = WIDE_RAM                      # u8: id & 7, what the and7 sites test
SH_TUNE = WIDE_RAM + 1                  # u8: TUNES[id], the music id's source
SH_VISIT = WIDE_RAM + 2                 # u8: VISIT_IDS[id], what the culling walk tests (2: Advanced's order)
SH_LANES = WIDE_RAM + 4                 # u16[2]: GRID_LANES' entry of the id
SH_REF = WIDE_RAM + 8                   # f32[3]: BILLBOARD_REF's entry of the id
WIDE_RAM_SIZE = 0x14

_WIDE_SPECS = (
    ("GRID_LANES", GRID_LANES, WIDE_LANES, 4, (0xD844, 0xDB10, 0xE984),
     "0xd840, 0xdb0c, 0xe980 ldos [lane*2] (lane = car & 1, or its complement): car +0x142 / +0x144; widened: "
     "the per-id table's [(2 * id + lane)*2]"),
    ("BILLBOARD_REF", BILLBOARD_REF, WIDE_BILLBOARD, 12, (0x221E8C,),
     "0x221e88 lda: the view-angle billboard's reference point (x at +0, z at +8); widened: + id*12"),
)
WIDE_ARRAYS = frozenset(name for name, *_ in _WIDE_SPECS)
"""The ID_ARRAYS entries widen_ids makes per id (shared on the original and on a grown set: one entry every id reads;
grow_ids leaves them where they are). Their `new` tables are in WIDE_HOME; their `sites` are the displacement words
of the readers' instructions, which widen_ids points at the id's shadow in work RAM (SH_LANES, SH_REF), filled from
`new` whenever the id is written. id_table gives `new` on a widened set."""


def _id_layout() -> dict[str, IdArray]:
    out, at = {}, ID_RECORDS + COURSE_IDS * COURSE_ENTRY_SIZE
    for name, old, stride, sites, keep, note in _ID_SPECS:
        entries = 3 if name in _THREE_ENTRIES else 1 if name == "TUNNEL_STRIDE" else 4
        out[name] = IdArray(old, at, stride, sites, keep, entries, note)
        at += -(-COURSE_IDS * stride // 16) * 16
    for name, old, new, stride, sites, note in _WIDE_SPECS:
        out[name] = IdArray(old, new, stride, sites, 1, 1, note)
    return out


ID_ARRAYS: dict[str, IdArray] = _id_layout()
"""The per-id arrays, by name: original address, new address, stride, the readers' words that hold the array's
address (`sites`) and what reads them. grow_ids relocates GROWN_ARRAYS (to ID_HOME on); widen_ids makes WIDE_ARRAYS
per id (to WIDE_HOME). Every reader of a grown array the disassembly found loads the array's address whole (MEMB,
32-bit displacement), so any address works and each table goes to the data area. The program window has 11,624 free
bytes; none is needed. Not here: arrays read for ids 0-2 only (the ranking reset loop 0x2270d0 over RANKING_DEFAULTS
and the time bases 0x23909c, `mov 3,r15`), tables reached through a grown array's entries (0x230b28, 0x2880004:
index offsets), SRAM (the counters 0x1d00164[id] at 0x22902c), the TGP slot table (sixteen slots already: grow_ids
sets slot 3 to Beginner's) and the Expert / Advanced time-attack records 0x230b9c (0xb864, (id - 1) * 8 + gear * 16,
ids 1-2 only: ID_PATCHES guards it). The discovery log has each one's evidence."""
GROWN_ARRAYS: dict[str, IdArray] = {n: a for n, a in ID_ARRAYS.items() if n not in WIDE_ARRAYS}
"""The ID_ARRAYS grow_ids relocates: every one but WIDE_ARRAYS."""

OLD_ARRAY_READERS = {
    0x153C: ("COURSE_SHEETS", 0x15A0, 4, "0x1534 ld 0x1578[r10*4]: the texel upload's mip run-length table (0x1578, "
                                         "ten words) runs on into COURSE_SHEETS' entry 0 (r10 = 10) and reads it as "
                                         "a run length (0x2200000: the run never ends inside a row); not by id, so "
                                         "it keeps reading the original word, which grow_ids leaves as it is")}
"""Reads of an original array's bytes that are not per-id reads, by trace ip (the address after the instruction):
ip -> (array, i960 address, bytes, why). tests/test_sixteen_mame.py leaves exactly those bytes out of its taps and
allows no read of the rest of the original arrays."""

ID_PATCHES: dict[int, tuple[int, int, str]] = {
    0xD474: (0x5818C883, 0x5818C88F, "and 3,r3,r3 -> and 15,r3,r3 (FLOATS_231C44)"),
    0xD49C: (0x5818C883, 0x5818C88F, "and 3,r3,r3 -> and 15,r3,r3 (FLOATS_231C54)"),
    0x195CC: (0x5818C883, 0x5818C88F, "and 3,r3,r3 -> and 15,r3,r3 (FLAGS_23329C)"),
    0x19700: (0x5818C883, 0x5818C88F, "and 3,r3,r3 -> and 15,r3,r3 (FLAGS_23329C)"),
    0x225B50: (0x58210883, 0x5821088F, "and 3,r4,r4 -> and 15,r4,r4 (SELECT_238EFC)"),
    0x227594: (0x58840883, 0x5884088F, "and 3,g0,g0 -> and 15,g0,g0 (RANKING_TABLES, RANKING_LAST)"),
    0xB820: (0x58CE4883, 0x34166028, "and 3,g9,g9 -> cmpobl 2,g9,0xb848: the time-attack records 0x230b9c are for "
                                     "ids 1-2 only (0xb864 indexes them by (id - 1) * 8); ids 3-15 skip like id 0"),
    0x32BC: (0x5CC81E03, 0x5CC81E04, "mov 3,g9 -> mov 4,g9: the course vote (0x32ac) takes the largest of four "
                                     "slots, ids 0-3, not three"),
    0x32D0: (0x5990D199, 0x59911199, "subi g9,3,g2 -> subi g9,4,g2: the vote's slot index, 4 - g9"),
}
"""Program words grow_ids rewrites besides the sites, i960 address -> (original word, new word, what): the `and 3`
masks before reads of grown arrays become `and 15` (an id 4-15 would otherwise read entry id & 3), the guard of the
time-attack records (0xb818: `and 3`, then `cmpi g9,0; be` skips id 0) also skips ids 3 and up, and the course vote
counts id 3. The vote (0x32ac, after Circuit Select and the link wait) stores to 0x501460 the id whose half-word at
0x501474 + 2 * id is largest (high byte: the cabinets' picks, counted by 0x3504-0x3524 from the link packets; low
byte: the priority set at 0x3064-0x3080, Beginner 2, Advanced 1, Expert 0), over ids 0-2 only, so a selected id 3
became 0 again. The nine bytes 0x3048 clears (a count byte and four half-words) already hold id 3's slot (low byte
0), so the loop now takes four; ids 0-2 vote as before (id 3's slot is 0 unless picked, and a tie keeps the earlier
slot). Ids 4-15 still do not survive the vote: their slots would lie on 0x501480 and 0x501488, which other code uses,
so sixteen slots need free work RAM (Task 4); until then 0x3524 would write a picked id 4-15's vote there. Ids 0-2
take the same paths as before (id & 3 == id, 2 < id is false for them, and id 3's vote slot stays 0)."""

_TGP_SLOT_WORDS = (0x10, 0x20)          # tgp.SLOT_LIST, tgp.SLOT_POOL: copro_data words of slot 0's two addresses
_TGP_STUB_SLOT = (0x18000, 0x465A0)     # slot 3's two words on the original set (tgp.STUB_BLOCKS)
_STUB_ID = 3


def record_address(course: int) -> int:
    """The i960 address of course id `course`'s record: COURSE_TABLE + 0x20 * id for ids 0-2, ID_RECORDS + 0x20 * id
    for ids 3-15 (grown sets; the original's id 3 is a stub at 0x02600080, which course_table does not return).
    CourseError for an id outside 0..COURSE_IDS-1 (bool refused)."""
    course = _int(course, "course id", 0, COURSE_IDS - 1, "course record")
    return (COURSE_TABLE if course < COURSE_COUNT else ID_RECORDS) + COURSE_ENTRY_SIZE * course


def id_table(rs: RomSet, name: str) -> int:
    """The i960 address the game reads per-id array `name` (ID_ARRAYS) from in `rs`: its `new` address when its first
    site holds it (a grown set, grow_ids), else its original one (the original set, and synthetic sets without the
    reader code). CourseError for an unknown name or `rs` not a RomSet."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not isinstance(name, str) or name not in ID_ARRAYS:
        raise CourseError(f"{name!r} is not a per-id array (course.ID_ARRAYS)")
    a = ID_ARRAYS[name]
    try:
        word = struct.unpack("<I", rs.read_i960(a.sites[0], 4))[0]
    except RomSetError:                     # a synthetic set without that word
        return a.old
    if name in WIDE_ARRAYS:                 # the site names the id's shadow in work RAM (widen_ids)
        return a.new if word == _WIDE_SHADOW[name] else a.old
    return a.new if word == a.new else a.old


def id_tables(rs: RomSet) -> dict[str, int]:
    """id_table of every ID_ARRAYS name: the per-id arrays' addresses in `rs` (golden.Course.id_arrays)."""
    return {n: id_table(rs, n) for n in ID_ARRAYS}


def grown(rs: RomSet) -> bool:
    """True when `rs` is a grown set: its COURSE_POINTERS reader (0x46f0) reads the sixteen-entry table."""
    return id_table(rs, "COURSE_POINTERS") == ID_ARRAYS["COURSE_POINTERS"].new


def id_count(rs: RomSet) -> int:
    """The course ids `rs` holds: COURSE_IDS on a grown set, COURSE_COUNT (3: the stub id 3 is not a course) else."""
    return COURSE_IDS if grown(rs) else COURSE_COUNT


def id_arrays_grown(arrays) -> bool:
    """True when `arrays` (name -> address, golden.Course.id_arrays) names every GROWN_ARRAYS entry at its new address
    (WIDE_ARRAYS may be at either: a grown set is widened or not)."""
    return isinstance(arrays, dict) and all(arrays.get(n) == a.new for n, a in GROWN_ARRAYS.items())


def id_entry(arrays: dict[str, int] | None, name: str, course: int) -> int:
    """The i960 address of course id `course`'s entry of per-id array `name` at `arrays` (array_at): the array's
    address + stride * id, except a WIDE_ARRAYS array at its original address (a set not widened), which is one entry
    every id reads. CourseError as array_at, and for a course id outside 0..COURSE_IDS-1 (bool refused)."""
    base = array_at(arrays, name)
    course = _int(course, "course id", 0, COURSE_IDS - 1, f"{name} entry")
    a = ID_ARRAYS[name]
    return base if name in WIDE_ARRAYS and base == a.old else base + a.stride * course


_ROM_SPANS = ((0x00000000, 0x00040000), PROGRAM_WINDOW, (DATA_BASE, DATA_BASE + DATA_WINDOW))


def array_at(arrays: dict[str, int] | None, name: str) -> int:
    """Array `name`'s address in `arrays` (name -> address, id_tables), its original address when `arrays` is None
    or lacks it. CourseError for a name that is not a str naming an ID_ARRAYS entry, `arrays` neither None nor a
    dict, or an address that is not an integer (bool refused) with the array's sixteen entries inside an i960 ROM
    range (program, its window, the data window)."""
    if not isinstance(name, str) or name not in ID_ARRAYS:
        raise CourseError(f"{name!r:.60} is not a per-id array (course.ID_ARRAYS)")
    if arrays is not None and not isinstance(arrays, dict):
        raise CourseError(f"per-id arrays {arrays!r:.60} is not a dict (name -> address)")
    v = None if arrays is None else arrays.get(name)
    if v is None:
        return ID_ARRAYS[name].old
    size = COURSE_IDS * ID_ARRAYS[name].stride
    if not is_int(v) or not any(lo <= v and v + size <= hi for lo, hi in _ROM_SPANS):
        raise CourseError(f"{name}: address {v!r:.60} is not an i960 ROM address with room for {COURSE_IDS} entries")
    return int(v)


def ids_of(arrays: dict[str, int] | None) -> int:
    """The course ids per-id arrays at `arrays` hold: COURSE_IDS when every one is at its new address (a grown set),
    COURSE_COUNT else."""
    return COURSE_IDS if id_arrays_grown(arrays) else COURSE_COUNT


def _rom_word_at(addr: int) -> tuple[str, int]:
    """(RomSet region, offset) of the i960 ROM word at `addr`: program, its window, or data."""
    if 0 <= addr <= PROGRAM_WINDOW[0] - 4 and addr < 0x40000:
        return "program", addr
    if PROGRAM_WINDOW[0] <= addr <= PROGRAM_WINDOW[1] - 4:
        return "program", addr - PROGRAM_WINDOW[0] + 0x20000
    if DATA_BASE <= addr:
        return "data", addr - DATA_BASE
    raise CourseError(f"{addr:#x} is not an i960 ROM address")


def rom_word_at(addr: int) -> tuple[str, int]:
    """(RomSet region, image offset) of i960 ROM address `addr`: program, its window, or data. CourseError for an
    address that is not an int (bool refused) or not ROM."""
    if not is_int(addr):
        raise CourseError(f"address {addr!r:.60} is not an int")
    return _rom_word_at(int(addr))


def grow_ids(rs: RomSet) -> RomSet:
    """The set with course ids 0-15 (COURSE_IDS): every GROWN_ARRAYS array copied to its sixteen-entry table at `new`
    (entries 0..keep-1 as they are, the others entry 0's bytes: Beginner's, the safe default until an entry is
    filled), every site repointed, the ID_PATCHES words rewritten, the records of ids 3-15 written at ID_RECORDS as
    copies of Beginner's (COURSE_POINTERS[3..15] point there), and TGP slot 3 given slot 0's two words (slots 4-15
    already repeat slot 0's). Ids 0-2 keep every table where it was; only the arrays that index them move, so they
    race as before (tests/test_sixteen_mame.py). A stock set is made custom first (RomSet.as_custom).
    CourseError for `rs` not a RomSet, a set already grown, a site that does not hold its array's original address,
    an ID_PATCHES word that is not the original one, a TGP slot 3 that no longer holds the stub's words, or a new home
    that is not free (0xff)."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if grown(rs):
        raise CourseError("the set is already grown (its COURSE_POINTERS reader reads the sixteen-entry table)")
    rs = rs.as_custom()
    patches: dict[tuple[str, int], bytes] = {}

    def put(addr: int, blob: bytes) -> None:
        patches[_rom_word_at(addr)] = blob

    def free(addr: int, n: int, what: str) -> None:
        off = addr - DATA_BASE
        if rs.data[off:off + n] != b"\xff" * n:
            raise CourseError(f"{what}: the new home {addr:#x} (+{n}) is not free (0xff)")

    for name, a in GROWN_ARRAYS.items():
        for s in a.sites:
            if (w := _u32(rs, s)) != a.old:
                raise CourseError(f"{name}: the word at {s:#x} is {w:#x}, not the array's address {a.old:#x}")
        kept = _read(rs, a.old, a.stride * a.keep, name)
        if name == "COURSE_POINTERS":
            blob = kept + b"".join(struct.pack("<I", record_address(i)) for i in range(a.keep, COURSE_IDS))
        else:
            blob = kept + kept[:a.stride] * (COURSE_IDS - a.keep)
        free(a.new, len(blob), name)
        put(a.new, blob)
        for s in a.sites:
            put(s, struct.pack("<I", a.new))
    for addr, (old, new, what) in ID_PATCHES.items():
        if (w := _u32(rs, addr)) != old:
            raise CourseError(f"ID_PATCHES {addr:#x} ({what}): the word is {w:#x}, not {old:#x}")
        put(addr, struct.pack("<I", new))
    beginner = _read(rs, record_address(0), COURSE_ENTRY_SIZE, "Beginner's record")
    for i in range(COURSE_COUNT, COURSE_IDS):
        free(record_address(i), COURSE_ENTRY_SIZE, f"course record {i}")
        put(record_address(i), beginner)
    for w, stub in zip(_TGP_SLOT_WORDS, _TGP_STUB_SLOT):
        at = 4 * (w + _STUB_ID)
        if (got := struct.unpack_from("<I", rs.copro_data, at)[0]) != stub:
            raise CourseError(f"TGP slot {_STUB_ID}: copro_data word {w + _STUB_ID:#x} is {got:#x}, not the stub's "
                              f"{stub:#x} (the slot was moved: grow_ids gives the stub's slot Beginner's words)")
        patches[("copro_data", at)] = rs.copro_data[4 * w:4 * w + 4]
    return rs.with_patches(patches)


# --- the new ids' guards (M4a, Task 5) ------------------------------------------------------------------------------
# Course ids 3-15 race through code written for ids 0-2. Two things follow (docs/spike/2026-11-sixteen.md, "Desert City
# at id 3"): the code that tests the id against literals (decided one by one; two `cmpobe 3` skips are lifted so id 3
# races like the others), and the backup SRAM the game keeps per id (sized for ids 0-2, so ids 3-15 would write
# Beginner's words or the next array's): no new id may write another id's SRAM.

ID3_LITERALS: dict[int, tuple[int, int, str]] = {
    0x102E4: (0x3218E08C, 0x32F8E08C, "cmpobe 3,r3,0x10370 -> cmpobe 31,r3: the player car's heading assist "
                                      "(0x102e8-0x1036c, 0x1f4(g13)) runs for id 3 too"),
    0x10B54: (0x3218E3D8, 0x32F8E3D8, "cmpobe 3,r3,0x10f2c -> cmpobe 31,r3: the player car's event timers "
                                      "(0x10b58-0x10f28, 0x269-0x26d(g13), 0x19c, 0x2a0) run for id 3 too"),
}
"""The literal-id comparisons a course at id 3 needs changed, i960 address -> (original word, new word, what). On the
original set id 3 was a stub and two routines of the player car's physics return early for it (`cmpobe 3`); the
literal becomes 31, an id no course has (ids are 0-15), so every id takes the path ids 0-2 take. The other literal
comparisons (0x3eac, 0x63ec, 0x7370, 0x7ad4, 0xc2a4, 0xc2d0, 0xc3e0, 0x15d90, 0x17420, 0x19ae4-0x19af8, 0x224cf0) are
left: id 3 takes Expert's and Advanced's path there, which is the standing start's and the general one (the log has
each decision)."""

COUNTER_GUARD: dict[int, tuple[tuple[int, ...], tuple[int, ...], str]] = {
    0x229020: ((0x8C983000, 0xFFFF, 0x58ACC096), (0x3415A074, 0x5CA81616, 0x5CA81616),
               "lda 0xffff,g3; and g6,g3,g5 -> cmpobl 2,g6,0x229094; mov g6,g5; mov g6,g5"),
    0x229040: ((0x8C983000, 0xFFFF, 0x58A4C096), (0x3415A054, 0x5CA01616, 0x5CA01616),
               "lda 0xffff,g3; and g6,g3,g4 -> cmpobl 2,g6,0x229094; mov g6,g4; mov g6,g4"),
}
"""The bookkeeping counters' guard, i960 address -> (original words, new words, what). The routine 0x228fb0 (called at
the race's end, 0x5680) adds to four u32[3] arrays in backup SRAM indexed by the id (0x1d00164[id] at 0x229038,
0x1d00140 / 0x1d0014c / 0x1d00158[id] at 0x229070-0x229090): an id 3-15 would write the next array's entries (id 3:
0x1d0014c[0], Beginner's) and on to 0x1d001a3. Both blocks now start `cmpobl 2,g6,0x229094` (id > 2: on to the
routine's tail 0x229230, the checksum and the backup copy); `ldos` already zero-extends the id, so `and 0xffff` was a
no-op, and g3 (0xffff) is overwritten by the tail's `bal` (0x2292d8 `mov g14,g3`) on every path. Ids 0-2 count as
before; ids 3-15 are not counted (per-entry statistics are M4b's)."""

SRAM_SINK = 0x03001580
"""i960 address (custom data ROM, right after grow_ids' tables) of the read-only stand-in for the per-id SRAM words of
ids 3-15 (SRAM_SINK_ARRAYS): a ranking table, a newest-entry word and the four time-attack / best-lap words, as a
freshly reset SRAM holds them for Beginner (0x2270d0 with setting 0). ROM: the game's writes there are dropped, so a
new id writes no SRAM at all, and reads what a fresh Beginner reads. SRAM_SINK_SIZE bytes; no 4-aligned word of it
is 0xffffffff but the newest-entry word and the ranking entries' view words (fresh SRAM: 0xffffffff); it is
registered in romset.RESERVED, so tgp.write_free and the converter's free-space search never touch it. On a ranked
set (grow_rankings) no entry names its ranking table or newest-entry word any more: the sink then stands in for the
time-attack and best-lap words only."""
SINK_LAST, SINK_TIME_AUTO, SINK_BEST_AUTO, SINK_TIME_MANUAL, SINK_BEST_MANUAL = 0x240, 0x244, 0x248, 0x24C, 0x250
SRAM_SINK_SIZE = 0x260
SRAM_SINK_ARRAYS = {"RANKING_TABLES": (0,), "RANKING_LAST": (SINK_LAST,),
                    "TIME_ATTACK_230A90": (SINK_TIME_AUTO,), "TIME_ATTACK_230AA0": (SINK_TIME_MANUAL,),
                    "TIME_ATTACK_230AB8": (SINK_BEST_AUTO, SINK_TIME_AUTO),
                    "TIME_ATTACK_230AD8": (SINK_BEST_MANUAL, SINK_TIME_MANUAL),
                    "BEST_LAPS_2805100": (SINK_BEST_AUTO,), "BEST_LAPS_2805110": (SINK_BEST_MANUAL,)}
"""The per-id arrays whose entries name SRAM words (on a grown set ids 3-15 hold Beginner's: 0x1d00360, 0x1d00a20,
0x1d00b60-0x1d00b9c), name -> the sink offsets entry 3-15 holds (one per word of the entry). Beginner's entries show
the pattern: 0x230ab8[id] = (BEST_LAPS_2805100[id], TIME_ATTACK_230A90[id]), 0x230ad8 the same for the manual
transmission."""
_FRESH_RANK_TIME = 0x140         # 0x22713c: (rank - 1) * (5 << 6) + the time base
_FRESH_ATTACK = 0x500            # 0x2271c0: 5 << 8, the time-attack words of a fresh SRAM


def fresh_ranking(rs: RomSet, course: int) -> bytes:
    """The ranking table (RANKING_ENTRIES x RANKING_ENTRY_SIZE bytes) the reset 0x2270d0 writes for course id 0-2 with
    setting 0, as a fresh SRAM holds it: per entry the rank byte, the time (rank - 1) * 0x140 + the time base
    0x23909c[course] (0x22713c), the view word left 0xffffffff (fresh SRAM is 0xff, nvram DEFAULT_ALL_1), the flags,
    the shown time and the name of RANKING_DEFAULTS[course]. On a ranked set (grow_rankings) course ids 0-15, from
    RANK_DEFAULTS[course] and RANK_BASES[course]. CourseError for `rs` not a RomSet, a course outside 0-2 (0-15 on a
    ranked set), or a time outside u32 (the hardware would wrap it: not a table the reset meant)."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if ranked(rs):
        course = _int(course, "course id", 0, COURSE_IDS - 1, "ranking defaults")
        defaults, base = _u32(rs, RANK_DEFAULTS + 8 * course), _u32(rs, RANK_BASES + 4 * course)
    else:
        course = _int(course, "course id", 0, COURSE_COUNT - 1, "ranking defaults")
        defaults, base = _u32(rs, RANKING_DEFAULTS + 8 * course), _u32(rs, RANKING_TIME_BASES + 4 * course)
    out = bytearray()
    for k in range(RANKING_ENTRIES):
        rank, name, flags = struct.unpack("<B3xII", _read(rs, defaults + 12 * k, 12, "ranking defaults"))
        time = (rank - 1) * _FRESH_RANK_TIME + base
        if not 0 <= time <= 0xFFFFFFFF:
            raise CourseError(f"ranking defaults of course {course}, entry {k}: rank {rank} and time base {base:#x} "
                              f"give the time {time:#x}, outside u32")
        out += struct.pack("<6I", rank, time, 0xFFFFFFFF, flags, time, name)
    return bytes(out)


def fresh_settings(rs: RomSet) -> bytes:
    """The SRAM words 0x1d00b60-0x1d00ba3 as the reset 0x227168 leaves them with setting 0: per transmission (auto at
    0x1d00b60, manual at 0x1d00b80) the time-attack words 5 << 8, 15 << 8, 15 << 9, one word left 0xffffffff, the
    best laps 0x23906c[0..2], one word left 0xffffffff; then the setting (0) at 0x1d00ba0. CourseError for `rs` not a
    RomSet."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    best = struct.unpack("<3I", _read(rs, 0x0023906C, 12, "best laps"))
    half = struct.pack("<8I", _FRESH_ATTACK, 0xF00, 0x1E00, 0xFFFFFFFF, *best, 0xFFFFFFFF)
    return half + half + struct.pack("<I", 0)


def sram_sink(rs: RomSet) -> bytes:
    """The SRAM_SINK bytes of `rs`: Beginner's fresh ranking table (`fresh_ranking(rs, 0)`), the newest-entry word
    0xffffffff (fresh SRAM), the time-attack and best-lap words of each transmission as `fresh_settings` gives them for
    Beginner (5 << 8 and 0x23906c[0]), padded with zeros. CourseError as `fresh_ranking`."""
    table = fresh_ranking(rs, 0)
    best = _u32(rs, 0x0023906C)
    out = table + struct.pack("<5I", 0xFFFFFFFF, _FRESH_ATTACK, best, _FRESH_ATTACK, best)
    return out + bytes(SRAM_SINK_SIZE - len(out))


def guarded(rs: RomSet) -> bool:
    """True when `rs` has the new ids' guards (`guard_new_ids`): its counter guard's first word is written."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    at = next(iter(COUNTER_GUARD))
    return len(rs.program) >= 0x40000 and _u32(rs, at) == COUNTER_GUARD[at][1][0]


def guard_new_ids(rs: RomSet) -> RomSet:
    """The grown set `rs` (grow_ids) with the new ids' guards: ID3_LITERALS and COUNTER_GUARD written, the sink's bytes
    (`sram_sink`) at SRAM_SINK, and entries 3-15 of every SRAM_SINK_ARRAYS array naming the sink. Ids 0-2 take the
    paths they took (the literals and the guard test only ids 3 and up) and keep their SRAM; no id 3-15 writes SRAM.
    The SRAM layout is unchanged, so the signature at 0x5b78 stays and old SRAM is read as before. CourseError for
    `rs` not a RomSet, a set that is not grown or is guarded already, a word that is not the original one, an array
    entry 3-15 that is not Beginner's, or a sink area that is not free (0xff)."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not grown(rs):
        raise CourseError("the new ids' guards need a grown set (course.grow_ids)")
    if guarded(rs):
        raise CourseError("the set has the new ids' guards already")
    patches: dict[tuple[str, int], bytes] = {}
    for addr, (old, new, what) in ID3_LITERALS.items():
        if (w := _u32(rs, addr)) != old:
            raise CourseError(f"ID3_LITERALS {addr:#x} ({what}): the word is {w:#x}, not {old:#x}")
        patches[_rom_word_at(addr)] = struct.pack("<I", new)
    for addr, (old, new, what) in COUNTER_GUARD.items():
        if (w := struct.unpack("<3I", rs.read_i960(addr, 12))) != old:
            raise CourseError(f"COUNTER_GUARD {addr:#x} ({what}): the words are {[hex(x) for x in w]}, not the "
                              f"original ones")
        patches[_rom_word_at(addr)] = struct.pack("<3I", *new)
    off = SRAM_SINK - DATA_BASE
    if rs.data[off:off + SRAM_SINK_SIZE] != b"\xff" * SRAM_SINK_SIZE:
        raise CourseError(f"the SRAM sink {SRAM_SINK:#x} (+{SRAM_SINK_SIZE:#x}) is not free (0xff)")
    patches[("data", off)] = sram_sink(rs)
    for name, words in SRAM_SINK_ARRAYS.items():
        a = ID_ARRAYS[name]
        beginner = _read(rs, a.new, a.stride, name)
        blob = struct.pack(f"<{len(words)}I", *(SRAM_SINK + w for w in words))
        for i in range(COURSE_COUNT, COURSE_IDS):
            if _read(rs, a.new + a.stride * i, a.stride, name) != beginner:
                raise CourseError(f"{name}[{i}] is not Beginner's entry: the new id's SRAM words were moved already")
            patches[_rom_word_at(a.new + a.stride * i)] = blob
    return rs.with_patches(patches)


# --- a mirror tests as its original (M4b, Task 2) -------------------------------------------------------------------
# A mirror's id is its original's + 8 (ids 8-15 mirror 0-7). The program tests the id against 0, 1 and 2 in a dozen
# places, and reads two shared tables a mirror needs its own of. The race is cycle-exact work (MAME's i960 charges
# cycles by opcode): a trampoline at a site run every frame, three cycles, moves the AI within a minute
# (docs/spike/2026-11-mirrors-live.md, "Literal ids"). So no site gains an instruction: each keeps its own and reads a
# shadow in work RAM (WIDE_RAM) instead of the id or the shared table, and the shadows are refreshed from the id and
# the per-id tables in WIDE_HOME where the program writes the id (and after the boot's clear). Ids 0-7 take the paths
# they took and read the values they read; the layouts: docs/formats/model2-course-tables.md, "A mirror tests as its
# original".

ID_BYTE = 0x00501460                     # work RAM: the course id


class LiteralSite(NamedTuple):
    """A literal test of the course id: `load` (i960 address of the 8-byte load of the id, MEMB with a displacement:
    widen_ids points the displacement at a shadow, the instruction stays), `reg` (its register), `insn` (the test,
    unidasm's text on the guarded set), `tests` (what it guards) and `rule`: "and7" (the load reads SH_PATH, id & 7,
    so ids 8-15 test as id - 8) or "table" (it reads a per-id table's shadow: SH_TUNE from TUNES, SH_VISIT from
    VISIT_IDS)."""
    load: int
    reg: str
    insn: str
    tests: str
    rule: str


LITERAL_ID_SITES: dict[int, LiteralSite] = {
    0x3EB4: LiteralSite(0x3EAC, "r3", "cmpibe  0,r3,0x3ee8", "state 20: Beginner goes to state 21 at once; the others "
                        "wait 0x5010a8 (320 frames) for the standing start's countdown and lights", "and7"),
    0x7370: LiteralSite(0x7368, "r8", "cmpobe  0,r8,0x741c", "the chase camera's pitch correction on a slope, skipped "
                        "for Beginner (flat)", "and7"),
    0x7AD4: LiteralSite(0x7ACC, "r3", "cmpobne 2,r3,0x7af4", "a sound call with Advanced's parameters (0x7ad8-0x7af0)",
                        "and7"),
    0xC2A4: LiteralSite(0xC29C, "r3", "cmpi    r3,0", "the rolling start's countdown steps (0xc334 at segments "
                        "n - 29 / 19 / 9), Beginner only", "and7"),
    0xC2D0: LiteralSite(0xC2C8, "r3", "cmpi    r3,0", "the rolling start's steps (0xc2e4, 0xc334), Beginner only",
                        "and7"),
    0x15D90: LiteralSite(0x15D88, "r3", "cmpi    r3,0", "the lap after which an AI car's slipstream message "
                         "(0x501898 = 4) can fire: Beginner 3, the others 1", "and7"),
    0x17420: LiteralSite(0x17418, "r3", "cmpobne 2,r3,0x1743c", "Advanced's cell visit order (0x175ad) at its cells "
                         "72 and 88 (g0): VISIT_IDS[id] (2: Advanced's), so a mirror can drop it", "table"),
    0x19AE4: LiteralSite(0x19ADC, "r4", "cmpi    r4,0", "a positional sound's segment range: Beginner 0x77-0x79",
                         "and7"),
    0x19AEC: LiteralSite(0x19ADC, "r4", "cmpi    r4,1", "a positional sound's segment range: Expert 0xcf-0xdb",
                         "and7"),
    0x19AF4: LiteralSite(0x19ADC, "r4", "cmpi    r4,2", "a positional sound's segment ranges: Advanced 0xab-0xeb, "
                         "0x190-0x19b; the other ids return", "and7"),
    0x224CF0: LiteralSite(0x224CE8, "r3", "cmpibne 0,r3,0x224cf8", "the rolling-start pace task (0x501248) enabled "
                          "at once for Beginner, 192 frames after the green light for the others", "and7"),
    0x3BEC: LiteralSite(0x3B9C, "r3", "cmpi    r3,2", "the music id (0x501a98, the id unless a view button picks "
                        "0-3) 2, Advanced's tune, adds the sound 0xb70700: the music id's source is TUNES[id]",
                        "table"),
}
"""The literal tests of the course id widen_ids patches, by the test's i960 address. The sweep (every load of the id
or of the music id 0x501a98 followed through its register, its copies and the calls it is passed to, until
overwritten) finds these and LITERAL_ID_KEPT's, no more, on the guarded set and on the widened one, its shadows
seeded too (tests/test_course.py, test_literal_sites_cover)."""

LITERAL_ID_KEPT: dict[int, tuple[str, str]] = {
    0x63F4: ("cmpo    r3,0", "Beginner's camera script with a linked cabinet (0x6ca8): linked mode only, left on the "
                             "generic path (decision 20)"),
    0xC3E0: ("cmpobe  0,r3,0xc3e4", "a branch to the next instruction: no path depends on it"),
    0x102E4: ("cmpobe  31,r3,0x10370", "ID3_LITERALS: 31 is no course id, every id runs the heading assist"),
    0x10B54: ("cmpobe  31,r3,0x10f2c", "ID3_LITERALS: 31 is no course id, every id runs the event timers"),
    0xB820: ("cmpobl  2,g9,0xb848", "ID_PATCHES: the time-attack section records 0x230b9c are Expert's and Advanced's "
                                    "SRAM words (0x1d00bec-0x1d00c00); ids 3-15 skip them as Beginner does, so a "
                                    "mirror writes none of its original's SRAM (Review Focus 1)"),
    0xB824: ("cmpi    g9,0", "Beginner has no time-attack section records (with 0xb820)"),
    0x229020: ("cmpobl  2,g6,0x229094", "COUNTER_GUARD: the bookkeeping counters in SRAM are ids 0-2's"),
    0x229040: ("cmpobl  2,g6,0x229094", "COUNTER_GUARD: the bookkeeping counters in SRAM are ids 0-2's"),
}
"""The literal tests of the course id widen_ids leaves, by address: (unidasm's text on the guarded set, why)."""

_LOAD_OPS = {0x3EAC: "ldib", 0x7368: "ldob", 0x7ACC: "ldob", 0xC29C: "ldib", 0xC2C8: "ldib", 0x15D88: "ldob",
             0x19ADC: "ldib", 0x224CE8: "ldib", 0x17418: "ldob", 0x3B9C: "ldob", 0x33D8: "ldob"}
TUNE_SITES = (0x33D8, 0x3B9C)
"""The two loads of the id that set the music id 0x501a98 (0x340c, 0x3bd0: the id unless a view button picks 0-3),
which the race reads three tables by (MUSIC 0x28050f0 at 0x3be0, 0x280eeb8 at 0x223da8, 0x2810d58 at 0x224644, four
words each) and tests at 0x3bec: widen_ids makes them load SH_TUNE (TUNES[id])."""
TUNE_DEFAULTS = (0, 1, 2, 3, 0, 0, 0, 0) * 2
"""TUNES as widen_ids writes it: ids 0-3 their own music id as before (Desert City at id 3 plays the fourth tune, as
in Phase 8), ids 4-7 Beginner's (on a grown set they read the music tables past their four words, 0x280eeb8[4..7]:
no tune), each mirror 8-15 its original's (decision 12)."""
VISIT_DEFAULTS = tuple(k & 7 for k in range(16))
"""VISIT_IDS as widen_ids writes it: each id tests as id & 7 (Advanced's order at cells 72 and 88 for ids 2 and 10).
The cells are the original's: a mirror's own (72 -> 184, 88 -> 168: rows flip) cannot be named in place, so a
mirror that should not take Advanced's order at the original's cells writes another value (VISIT_NONE)."""
PROGRESS_MASK_SIZE = 36
"""A progress-mask record (PROGRESS_MASKS[id], walked by 0x17688): u16 first and last segment, a u16 column mask per
scenery row (16 rows)."""
PROGRESS_MASK_END = 9999
"""The last segment of the record that ends a progress-mask table (0x17688 walks on while the player's segment is
past a record's last segment)."""
VISIT_NONE = 0xFF
"""A VISIT_IDS value that is never 2: the id takes the general visit order at every cell (Advanced's mirror, id 10:
mirror.VISIT_OF_MIRROR)."""
LANE_READERS = {0xD840: "r4", 0xDB0C: "r3", 0xE980: "r3"}
"""GRID_LANES' readers, `ldos 0x231b30[x*2],x` (x the lane, car & 1 or its complement), by address: x."""
BILLBOARD_READER = (0x221E88, "r8")      # `lda 0x235bb4,r8`, then ld 0x0(r8), ld 0x8(r8)
ID_WRITERS = {0x17A4: (4, "after the boot's clear of work RAM (0x1794): id 0, the shadows 0"),
              0x213C: (8, "the attract's next course (stib r4; 0x2124 stores first, on the same straight run)"),
              0x32DC: (8, "the course vote (stob g2): the race's id after the select"),
              0x5720: (8, "linked mode: the link packets' course through 0x501468 (stob r3)"),
              0x1B084: (8, "linked mode: a link packet's course byte (stob r3)")}
"""The instructions after which widen_ids refreshes the shadows (WIDE_RAM): every program store to the id but the
select's (traced: no widened site reads the id between the select's writes and the vote's, frames 617-1143), by
address: (bytes replaced, what). The instruction is moved into a trampoline, which then calls the update."""


def _original_words() -> dict[int, bytes]:
    out = {s.load: i960.mem(_LOAD_OPS[s.load], s.reg, ID_BYTE) for s in LITERAL_ID_SITES.values()}
    for at in TUNE_SITES:
        out[at] = i960.mem("ldob", "r3", ID_BYTE)
    for at, x in LANE_READERS.items():
        out[at] = i960.mem("ldos", x, GRID_LANES, None, x, 2)
    out[BILLBOARD_READER[0]] = i960.mem("lda", BILLBOARD_READER[1], BILLBOARD_REF)
    return out


WIDE_ORIGINALS: dict[int, bytes] = _original_words()
"""i960 address -> the 8 bytes of each widened site on the original set (a load of the id, a GRID_LANES reader, the
BILLBOARD_REF lda). widen_ids changes only the second word, the displacement (WIDE_SHADOWS)."""
WRITER_ORIGINALS: dict[int, bytes] = {
    0x17A4: i960.reg("mov", 0, "g0"), 0x213C: i960.mem("stib", "r4", ID_BYTE), 0x32DC: i960.mem("stob", "g2", ID_BYTE),
    0x5720: i960.mem("stob", "r3", ID_BYTE), 0x1B084: i960.mem("stob", "r3", ID_BYTE)}
"""i960 address -> the original bytes of each ID_WRITERS instruction."""


def _shadows() -> dict[int, int]:
    out = {s.load: SH_PATH for s in LITERAL_ID_SITES.values() if s.rule == "and7"}
    out.update({at: SH_TUNE for at in TUNE_SITES})
    out[0x17418] = SH_VISIT
    out.update({at: SH_LANES for at in LANE_READERS})
    out[BILLBOARD_READER[0]] = SH_REF
    return out


WIDE_SHADOWS: dict[int, int] = _shadows()
"""i960 address of each widened site -> the shadow its displacement names on a widened set."""
_WIDE_SHADOW = {"GRID_LANES": SH_LANES, "BILLBOARD_REF": SH_REF}


def wide_code() -> i960.Block:
    """The code at WIDE_CODE: `update` (called: from the id at 0x501460, masked to 0-15, writes SH_PATH = id & 7,
    SH_TUNE = TUNES[id], SH_VISIT = VISIT_IDS[id], SH_LANES and SH_REF from the per-id tables; locals only), then per
    ID_WRITERS instruction a trampoline `writer_<address>`: the instruction, `call update`, a branch back."""
    b = i960.Block(WIDE_CODE)
    b.label("update")
    b.mem("ldob", "r3", ID_BYTE)
    b.reg("and", 15, "r3", "r3")
    b.reg("and", 7, "r3", "r4")
    b.mem("stob", "r4", SH_PATH)
    b.mem("ldob", "r4", TUNES, "r3")
    b.mem("stob", "r4", SH_TUNE)
    b.mem("ldob", "r4", VISIT_IDS, "r3")
    b.mem("stob", "r4", SH_VISIT)
    b.mem("ld", "r4", WIDE_LANES, None, "r3", 4)
    b.mem("st", "r4", SH_LANES)
    b.reg("mulo", 12, "r3", "r5")
    for k in range(3):
        b.mem("ld", "r4", WIDE_BILLBOARD + 4 * k, "r5")
        b.mem("st", "r4", SH_REF + 4 * k)
    b.ctrl("ret")
    for at, (n, _) in sorted(ID_WRITERS.items()):
        b.label(f"writer_{at:x}")
        b.code += WRITER_ORIGINALS[at]
        b.ctrl("call", "update")
        b.ctrl("b", at + n)
    return b


def wide_patches() -> dict[int, bytes]:
    """i960 address -> the bytes widen_ids writes there: each site's displacement word (its WIDE_SHADOWS shadow; the
    instruction word stays), each writer's `b` to its trampoline (a 4-byte instruction is the branch alone; an 8-byte
    one is the branch and its old displacement word, no longer reached)."""
    b = wide_code()
    out = {at + 4: struct.pack("<I", sh) for at, sh in WIDE_SHADOWS.items()}
    for at, (n, _) in ID_WRITERS.items():
        out[at] = i960.ctrl("b", b.labels[f"writer_{at:x}"] - at)
    return out


_WIDE_PATCHES = wide_patches()
_WIDE_CODE_BYTES = wide_code().bytes()
_VISIT_TAIL = bytes.fromhex("1ce01035035e1a590c001c3203de1a590c001c350030308cad750100")
"""0x17420-0x1743b on the original set: cmpobne 2,r3,0x1743c; shlo 3,9,r3; cmpobe r3,g0,0x17434; shlo 3,11,r3;
cmpobne r3,g0,0x1743c; lda 0x175ad,r6 (what VISIT_IDS feeds)."""


def visit_ids(rs: RomSet) -> tuple[int, ...]:
    """VISIT_IDS of a widened set: per course id the value the culling walk (0x17418) tests against 2 (2: Advanced's
    visit order 0x175ad at cells 72 and 88). CourseError for a set that is not widened."""
    if not widened(rs):
        raise CourseError("VISIT_IDS is a widened set's (course.widen_ids)")
    return tuple(rs.read_i960(VISIT_IDS, COURSE_IDS))


def tunes(rs: RomSet) -> tuple[int, ...]:
    """TUNES of a widened set: the music id per course id (0-15). CourseError for a set that is not widened."""
    if not widened(rs):
        raise CourseError("TUNES is a widened set's (course.widen_ids)")
    return tuple(rs.read_i960(TUNES, COURSE_IDS))


def widened(rs: RomSet) -> bool:
    """True when `rs` carries widen_ids: the vote's store (0x32dc) branches to its trampoline. CourseError for `rs`
    not a RomSet."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    return len(rs.program) >= 0x40000 and rs.read_i960(0x32DC, 4) == _WIDE_PATCHES[0x32DC]


def widen_ids(rs: RomSet) -> RomSet:
    """The grown set `rs` (grow_ids) on which a mirror tests as its original: every LITERAL_ID_SITES load, the music
    id's two loads (TUNE_SITES), GRID_LANES' three readers and BILLBOARD_REF's reader read their shadow in WIDE_RAM
    (the instruction unchanged, its displacement repointed: not a cycle more); WIDE_HOME holds the per-id tables
    (GRID_LANES and BILLBOARD_REF, every entry the shared value; TUNES TUNE_DEFAULTS; VISIT_IDS VISIT_DEFAULTS) and the
    code that refreshes the shadows after every ID_WRITERS instruction. Ids 0-7 take the paths they took and read the
    values they read. `rs` itself when it is widened already (idempotent). In the playtest chain: grow_ids,
    guard_new_ids, widen_ids, then select.patch_select. CourseError for `rs` not a RomSet, a set that is not grown, a
    site or writer that does not hold its original instruction (WIDE_ORIGINALS, WRITER_ORIGINALS), a changed culling
    test (0x17420) or a WIDE_HOME that is not free (0xff)."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not grown(rs):
        raise CourseError("widen_ids needs a grown set (course.grow_ids)")
    if widened(rs):
        return rs
    patches: dict[tuple[str, int], bytes] = {}
    for at, want in list(WIDE_ORIGINALS.items()) + list(WRITER_ORIGINALS.items()):
        if (have := rs.read_i960(at, len(want))) != want:
            raise CourseError(f"widen_ids {at:#x}: the instruction is {have.hex()}, not the original {want.hex()}")
    for at, blob in _WIDE_PATCHES.items():
        patches[_rom_word_at(at)] = blob
    if rs.read_i960(0x17420, 28) != _VISIT_TAIL:
        raise CourseError("widen_ids 0x17420: the culling walk's Advanced test is not the original one")
    if WIDE_CODE + len(_WIDE_CODE_BYTES) > WIDE_HOME + WIDE_SIZE:
        raise CourseError(f"the code ({len(_WIDE_CODE_BYTES)} bytes) does not fit WIDE_HOME")
    if (home := rs.read_i960(WIDE_HOME, WIDE_SIZE)) != b"\xff" * WIDE_SIZE:
        k = next(i for i, v in enumerate(home) if v != 0xFF)
        raise CourseError(f"WIDE_HOME {WIDE_HOME:#x} (+{WIDE_SIZE:#x}) is not free (0xff): a byte at "
                          f"{WIDE_HOME + k:#x}")
    lanes, ref = _read(rs, GRID_LANES, 4, "grid lanes"), _read(rs, BILLBOARD_REF, 12, "billboard reference")
    patches[_rom_word_at(WIDE_LANES)] = lanes * COURSE_IDS
    patches[_rom_word_at(WIDE_BILLBOARD)] = ref * COURSE_IDS
    patches[_rom_word_at(TUNES)] = bytes(TUNE_DEFAULTS)
    patches[_rom_word_at(VISIT_IDS)] = bytes(VISIT_DEFAULTS)
    patches[_rom_word_at(WIDE_CODE)] = _WIDE_CODE_BYTES
    return rs.with_patches(patches)


# --- the view per course id (Phase 11 Task 3) ----------------------------------------------------------------------
# docs/spike/2026-11-courses-4-7.md, "Rin Rin Rink", "Per-course view, largest fit". The view test (0x1770c) reads its
# quadrant's point table through `ld 0x17908[r8*4],r8` (r8 the heading quadrant): one table set every course shares.
# view_ids gives each id its own four table pointers without an instruction more in the race: the load keeps its
# opcode and reads a work-RAM shadow (VIEW_RAM) instead of 0x17908 (MAME's i960 charges cycles by opcode, as the
# widened sites), and widen_ids' `update` (run after every program store of the id: the boot, the attract's rotation,
# the vote, linked mode; never in a race frame) ends in a branch to code that copies the id's four pointers there.

VIEW_READER = 0x00017744                # `ld 0x17908[r8*4],r8`: the view test's table load (8 bytes, MEMB)
VIEW_HOME = WIDE_HOME + 0x400           # program: VIEW_PTRS then the code, in widen_ids' reserved space (free on a
#                                         widened set: widen_ids' tables and code end at WIDE_CODE + 0xcc)
VIEW_PTRS = VIEW_HOME                   # u32[4] per id: the id's four view-test table pointers (256 bytes)
VIEW_CODE = VIEW_HOME + 0x100           # the copy (`view`)
VIEW_SIZE = 0x180
VIEW_RAM = WIDE_RAM + WIDE_RAM_SIZE     # work RAM u32[4] (0x501b54-0x501b63): the course id's view-test pointers
VIEW_RAM_SIZE = 0x10


def _update_ret() -> int:
    """The i960 address of `update`'s `ret` in wide_code (the instruction before the first trampoline)."""
    b = wide_code()
    return min(v for k, v in b.labels.items() if k.startswith("writer_")) - 4


UPDATE_RET = _update_ret()


def view_code() -> i960.Block:
    """The code at VIEW_CODE: `view` (branched to from `update`'s end, so it returns to update's caller): from the id
    at 0x501460 (masked to 0-15) copy its four VIEW_PTRS words to VIEW_RAM; locals only."""
    b = i960.Block(VIEW_CODE)
    b.label("view")
    b.mem("ldob", "r3", ID_BYTE)
    b.reg("and", 15, "r3", "r3")
    for k in range(4):
        b.mem("ld", "r4", VIEW_PTRS + 4 * k, None, "r3", 16)
        b.mem("st", "r4", VIEW_RAM + 4 * k)
    b.ctrl("ret")
    return b


VIEW_ORIGINAL = i960.mem("ld", "r8", VIEW_TESTS, None, "r8", 4)
"""The 8 bytes of VIEW_READER on the original and on a widened set."""


def viewed(rs: RomSet) -> bool:
    """True when `rs` carries view_ids (VIEW_READER reads VIEW_RAM). CourseError for `rs` not a RomSet."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    return rs.read_i960(VIEW_READER, 8) == VIEW_ORIGINAL[:4] + struct.pack("<I", VIEW_RAM)


def view_tables(rs: RomSet, k: int) -> tuple[int, int, int, int]:
    """The four view-test table pointers course id `k` (0-15) reads: its VIEW_PTRS entry on a set with view_ids, else
    the shared VIEW_TESTS. CourseError for `rs` not a RomSet or an id outside 0..15."""
    if not is_int(k) or not 0 <= k < COURSE_IDS:
        raise CourseError(f"view tables: course id {k!r} is not 0..{COURSE_IDS - 1}")
    at = VIEW_PTRS + 16 * int(k) if viewed(rs) else VIEW_TESTS
    return struct.unpack("<4I", _read(rs, at, 16, "view-test pointers"))


def view_ids(rs: RomSet, views: dict) -> RomSet:
    """The widened set `rs` with a view per course id: `views` maps an id (0-15) to its four view-test table pointers
    (i960 addresses of tables laid out as VIEW_TESTS' own, VIEW_TEST_SIZE bytes each, written by the caller); every
    other id gets the shared four (VIEW_TESTS'), so its race reads the tables it read. Writes VIEW_PTRS and the code
    (VIEW_CODE), `update`'s `ret` (UPDATE_RET) becomes a branch to it, VIEW_READER's displacement names VIEW_RAM.
    CourseError for `rs` not a widened RomSet, one that already carries view_ids, a VIEW_READER or `update` end that
    does not hold its original instruction, a VIEW_HOME that is not free (0xff), or `views` that is not a dict of ids
    0-15 to four 4-aligned u32 pointers into ROM."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not widened(rs):
        raise CourseError("view_ids needs a widened set (course.widen_ids): its `update` copies the views")
    if viewed(rs):
        raise CourseError("view_ids: the set carries view_ids already")
    if rs.read_i960(VIEW_READER, 8) != VIEW_ORIGINAL:
        raise CourseError(f"view_ids {VIEW_READER:#x}: the view test's table load is "
                          f"{rs.read_i960(VIEW_READER, 8).hex()}, not the original {VIEW_ORIGINAL.hex()}")
    if rs.read_i960(UPDATE_RET, 4) != i960.ctrl("ret"):
        raise CourseError(f"view_ids {UPDATE_RET:#x}: widen_ids' update does not end in ret")
    if not isinstance(views, dict) or not all(
            is_int(k) and 0 <= k < COURSE_IDS and isinstance(v, (tuple, list)) and len(v) == 4 and all(
                is_int(a) and 0 <= a <= 0xFFFFFFFF and a % 4 == 0 for a in v) for k, v in views.items()):
        raise CourseError(f"views {views!r:.80} is not a dict of course ids 0-15 to four 4-aligned pointers")
    for v in views.values():
        for a in v:
            try:
                _rom_word_at(int(a))
            except CourseError as e:
                raise CourseError(f"views: table pointer {int(a):#x} is not ROM: {e}") from None
            _read(rs, int(a), VIEW_TEST_SIZE, f"views: the table at {int(a):#x}")      # refuses one past the ROM
    code = view_code().bytes()
    if (home := rs.read_i960(VIEW_HOME, VIEW_SIZE)) != b"\xff" * VIEW_SIZE:
        k = next(i for i, v in enumerate(home) if v != 0xFF)
        raise CourseError(f"VIEW_HOME {VIEW_HOME:#x} (+{VIEW_SIZE:#x}) is not free (0xff): a byte at {VIEW_HOME + k:#x}")
    stock = struct.unpack("<4I", _read(rs, VIEW_TESTS, 16, "view-test pointers"))
    ptrs = b"".join(struct.pack("<4I", *map(int, views.get(k, stock))) for k in range(COURSE_IDS))
    return rs.with_patches({_rom_word_at(VIEW_PTRS): ptrs, _rom_word_at(VIEW_CODE): code,
                            _rom_word_at(UPDATE_RET): i960.ctrl("b", VIEW_CODE - UPDATE_RET),
                            _rom_word_at(VIEW_READER + 4): struct.pack("<I", VIEW_RAM)})


# --- rankings per entry (M4b, Task 6) -------------------------------------------------------------------------------
# Spec decision 19: every course id keeps its own best-time ranking table in backup SRAM; a new entry never writes an
# original's table and none is shared. The game reaches a table only through two per-id arrays (RANKING_TABLES and
# RANKING_LAST, grown to sixteen entries by grow_ids; the race's end 0x227408 and the ranking screen read them) and
# writes it fresh only in the reset 0x2270d0 (the boot on a signature mismatch 0x5b08, a changed difficulty setting
# 0x12dc, the test menu's backup clear 0x22edcc), which loops over ids 0-2 through RANKING_DEFAULTS and the time bases
# RANKING_TIME_BASES. grow_rankings moves all sixteen tables to the SRAM area no code uses, gives the reset sixteen
# (defaults, table) pairs and time bases in RANK_HOME and changes the signature, so SRAM written by an older set (with
# the stock layout) fails the boot's check and is reset rather than read with the new layout. Same instructions, no
# code added: the race loop runs no instruction of it (docs/spike/2026-11-mirrors-live.md, "Rankings").

RANKING_TABLE_SIZE = RANKING_ENTRIES * RANKING_ENTRY_SIZE      # 576
RANKING_PAD = 0x01D00C28
"""The SRAM word one past the second settings area (0x1d00be0-0x1d00c27, the time-attack section records): the padding
entries of the section-record pointers 0x230b9c name it (sections 6-7 of Expert, 3-7 of Advanced; 0xb888 loads the
pointer, 0xb89c stores a lower time there). Not reached (the section counts WORDS_230CCC are 5 and 2, the real
entries' last index), but named, so the rankings start past it."""
RANKING_SRAM = 0x01D00C30
"""Backup SRAM address of course id 0's ranking table on a ranked set (grow_rankings): id k's at RANKING_SRAM +
RANKING_TABLE_SIZE * k, ids 0-15, in the area nothing else uses (0x1d00c28-0x1d03fff; RANKING_PAD left out)."""
RANKING_LAST_SRAM = RANKING_SRAM + COURSE_IDS * RANKING_TABLE_SIZE    # 0x1d03030: the newest-entry words, u32 per id
RANKING_TIME_BASES = 0x0023909C
"""u32[MODES][4] (program window): the time of a fresh table's first entry, by mode (0x5fe5e0) and id 0-2 (the fourth
entry Beginner's); 0x22713c reads [mode * 4 + id]."""
RANK_HOME = 0x0001FE00
"""Program ROM (i960 address = program offset): grow_rankings' sixteen-id copies of RANKING_DEFAULTS (RANK_DEFAULTS)
and of the time bases (RANK_BASES), RANK_SIZE bytes reserved (romset.RESERVED) after widen_ids' space."""
RANK_SIZE = 0x200
RANK_DEFAULTS = RANK_HOME               # (defaults list, table) u32 pairs per id: 128 bytes
RANK_BASES = RANK_HOME + 0x80           # u32[MODES][16]: [mode * 16 + id], 192 bytes
"""The time bases per mode and id. 0x22713c does not mask the mode (0x5fe5e0, 0-2): a mode of 3 or more would read
RANK_HOME's 0xff tail (a time base of 0xffffffff), where the stock set read other garbage (RANKING_DEFAULTS' words)."""
RANK_SOURCES = (0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0)
"""Whose stock defaults (names, flags, time base) a fresh table of each id takes: ids 0-2 their own, the mirrors of the
originals (8-10) their original's (the same course and laps, so the stock times fit), the others Beginner's (Desert
City at 3 and its mirror at 11 too: M6 sets their real times)."""
SIGNATURE_AT = 0x00005B78               # the 24 bytes the boot (0x5a98) compares with SRAM 0x1d00200
SIGNATURE_STOCK = b"S-RAM CHECK OK (4.0)" + bytes(4)
SIGNATURE_RANKED = b"S-RAM CHECK OK (4.0/16)" + bytes(1)
"""The signature of a ranked set: a stock or older SRAM ("(4.0)") fails the boot's compare, which then writes this one
and runs the reset (0x5b08), so its sixteen tables start fresh and the old layout is never read as the new one."""
RANKING_PATCHES: dict[int, tuple[int, int, str]] = {
    0x2270D4: (0x5C781E03, 0x5C781E10, "mov 3,r15 -> mov 16,r15: the reset's loop counts sixteen ids"),
    0x2270DC: (RANKING_DEFAULTS, RANK_DEFAULTS, "lda 0x2390cc,r14 -> lda RANK_DEFAULTS,r14 (the displacement word)"),
    0x227100: (0x5988D10F, 0x598C110F, "subo r15,3,g1 -> subo r15,16,g1: the id passed to 0x22713c"),
    0x227148: (0x59294E02, 0x59294E04, "shlo 2,r5,r5 -> shlo 4,r5,r5: 0x22713c's index mode * 16 + id"),
    0x227154: (RANKING_TIME_BASES, RANK_BASES, "ld 0x23909c[r5*4],r4 -> ld RANK_BASES[r5*4],r4 (the displacement "
                                               "word)"),
}
"""The reset's words grow_rankings rewrites, i960 address -> (original word, new word, what). 0x22713c is called only
by the reset (0x227104); the instructions stay, so the reset is as long per id as before (sixteen ids instead of
three: at boot, never in a race)."""


def ranked(rs: RomSet) -> bool:
    """True when `rs` has a ranking table per course id (grow_rankings): its signature is SIGNATURE_RANKED.
    CourseError for `rs` not a RomSet."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    return len(rs.program) >= 0x40000 and rs.read_i960(SIGNATURE_AT, len(SIGNATURE_RANKED)) == SIGNATURE_RANKED


def grow_rankings(rs: RomSet) -> RomSet:
    """The guarded set `rs` (grow_ids, guard_new_ids) with a ranking table of its own for every course id: entry k of
    RANKING_TABLES names RANKING_SRAM + RANKING_TABLE_SIZE * k and of RANKING_LAST the word RANKING_LAST_SRAM + 4 * k
    (ids 0-2 move too, so every table starts in SRAM no older set wrote); the reset 0x2270d0 loops over the sixteen ids
    (RANKING_PATCHES) through RANK_DEFAULTS ((RANKING_DEFAULTS' defaults list of RANK_SOURCES[k], the table) per id)
    and RANK_BASES (the stock time bases of RANK_SOURCES[k], per mode); the signature becomes SIGNATURE_RANKED, so an
    older SRAM is reset at boot. The guard's sink keeps the time-attack and best-lap words of ids 3-15 and the play
    counters stay ids 0-2's (COUNTER_GUARD). In the chain: after the mirrors (mirror.mirrored_entry), before
    select.patch_select (they commute). CourseError for `rs` not a RomSet, a set that is not grown, has no guards or
    is ranked already, a RANKING_PATCHES word or the signature that is not the original one, an entry of the two
    arrays that is not as the guard left it (ids 0-2 the stock tables and words, 3-15 the sink), or a RANK_HOME that
    is not free (0xff)."""
    if not isinstance(rs, RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    if not grown(rs):
        raise CourseError("grow_rankings needs a grown set (course.grow_ids)")
    if not guarded(rs):
        raise CourseError("grow_rankings needs the new ids' guards (course.guard_new_ids)")
    if ranked(rs):
        raise CourseError("the set is ranked already (course.grow_rankings)")
    patches: dict[tuple[str, int], bytes] = {}
    for at, (old, new, what) in RANKING_PATCHES.items():
        if (w := _u32(rs, at)) != old:
            raise CourseError(f"RANKING_PATCHES {at:#x} ({what}): the word is {w:#x}, not {old:#x}")
        patches[_rom_word_at(at)] = struct.pack("<I", new)
    if (sig := rs.read_i960(SIGNATURE_AT, len(SIGNATURE_STOCK))) != SIGNATURE_STOCK:
        raise CourseError(f"the signature at {SIGNATURE_AT:#x} is {sig!r}, not the stock {SIGNATURE_STOCK!r}")
    patches[_rom_word_at(SIGNATURE_AT)] = SIGNATURE_RANKED
    if (home := rs.read_i960(RANK_HOME, RANK_SIZE)) != b"\xff" * RANK_SIZE:
        k = next(i for i, v in enumerate(home) if v != 0xFF)
        raise CourseError(f"RANK_HOME {RANK_HOME:#x} (+{RANK_SIZE:#x}) is not free (0xff): a byte at "
                          f"{RANK_HOME + k:#x}")
    for name, sink in (("RANKING_TABLES", SRAM_SINK), ("RANKING_LAST", SRAM_SINK + SINK_LAST)):
        a = ID_ARRAYS[name]
        want = struct.unpack("<3I", _read(rs, a.old, 12, name)) + (sink,) * (COURSE_IDS - COURSE_COUNT)
        have = struct.unpack(f"<{COURSE_IDS}I", _read(rs, a.new, 4 * COURSE_IDS, name))
        if have != want:
            k = next(i for i in range(COURSE_IDS) if have[i] != want[i])
            raise CourseError(f"{name}[{k}] is {have[k]:#x}, not {want[k]:#x} (the stock entry, or the guard's sink)")
    lists = [_u32(rs, RANKING_DEFAULTS + 8 * c) for c in range(COURSE_COUNT)]
    bases = struct.unpack(f"<{4 * MODES}I", _read(rs, RANKING_TIME_BASES, 16 * MODES, "ranking time bases"))
    tables = [RANKING_SRAM + RANKING_TABLE_SIZE * k for k in range(COURSE_IDS)]
    patches[_rom_word_at(ID_ARRAYS["RANKING_TABLES"].new)] = struct.pack(f"<{COURSE_IDS}I", *tables)
    patches[_rom_word_at(ID_ARRAYS["RANKING_LAST"].new)] = struct.pack(
        f"<{COURSE_IDS}I", *(RANKING_LAST_SRAM + 4 * k for k in range(COURSE_IDS)))
    patches[_rom_word_at(RANK_DEFAULTS)] = b"".join(struct.pack("<2I", lists[RANK_SOURCES[k]], tables[k])
                                                    for k in range(COURSE_IDS))
    patches[_rom_word_at(RANK_BASES)] = b"".join(struct.pack("<I", bases[4 * m + RANK_SOURCES[k]])
                                                 for m in range(MODES) for k in range(COURSE_IDS))
    return rs.with_patches(patches)
