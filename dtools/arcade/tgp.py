"""The Model 2 TGP (MB86234, MAME's mb86233 core): the program the i960 uploads to it, its function dispatch, and
the ground it answers the i960's height queries (function 0x36) from.

Addresses on the TGP side are 32-bit words; copro_data words are little-endian u32 at byte 4 * word of
`RomSet.copro_data` (its first 2 MB copy; the second is identical). See docs/spike/2026-11-tgp-ground.md ("The
program", "Layout", "Course selection") and docs/formats/model2-tgp-ground.md.
"""

from __future__ import annotations

import numbers
import struct
from dataclasses import dataclass, replace

import numpy as np

from dtools.arcade.course import (COURSE_COUNT, COURSE_ENTRY_SIZE, COURSE_NAMES, COURSE_TABLE, course_name,
                                  course_table, encode_table_pointer, id_count)
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import (COPRO_SIZE, DATA_COPY_BASE, DATA_COPY_COUNT, DATA_COPY_END, DATA_COPY_LEN,
                                  RomSet, RomSetError, reserved)
from dtools.arcade.tablebase import DATA_BASE, DATA_SIZE, DATA_WINDOW, _f32, _int, _seq, is_int
from dtools.arcade.texture import custom_sheet_blocks

# (RomSet region, byte offset, word count) of the TGP program, as found: the i960 routine at 0xfa0 sets bit 31 of
# copro control 0x00980000, copies the words at i960 0x02800020-0x02801fbf (data ROM 0x800020) to the FIFO
# 0x00884000 and clears bit 31. It runs once, at boot; no later upload on any course.
PROGRAM: tuple[str, int, int] = ("data", 0x800020, 2024)

# The main loop (word 0x4c) reads a FIFO word and jumps to DISPATCH + fn, fn = the word's bits 23-30 (the function
# number MAME's function port puts there); each table word is "brif alw #handler".
DISPATCH = 0x58

# Function number -> handler word address (the dispatch table's target).
FUNCTIONS: dict[int, int] = {
    0x36: 0x457,         # ground query: y, grid value, x, z (sent by the i960 at 0x11c28)
}


class TgpError(DecodeError):
    pass


# ---- the ground: docs/formats/model2-tgp-ground.md ----
#
# copro_data words 0x10 + slot / 0x20 + slot hold the slot's list block / polygon pool word address; routine 0x7cb
# copies them (+ 0x800000, the copro_data bank) into $0x69 / $0x6a, at init with slot 0 and from function 0x0d,
# whose argument the i960 sends at course load: the course id (0x4534 ldib 0x501460 -> r10, st r10 at 0x46dc). So
# slot = course id. Function 0x36 reads the list at $0x69 + grid value (count, then polygon indices) and each
# polygon's first CORNER_WORDS words at $0x6a + POLY_WORDS * index.
GRID_CELLS = 128 * 128          # the i960's grid: u16 per 16-unit cell from -1024 (cell = row * 128 + col)
SLOT_LIST = 0x10                # word of slot 0's list block address (16 slots)
SLOT_POOL = 0x20                # word of slot 0's polygon pool address
SLOTS = 16
HEADER_WORDS = 0x30             # identity matrix (0x00-0x0f) and the slot table (0x10-0x2f)
POLY_WORDS = 16                 # the pool's stride: corners (12 words), centre (3), direction (1)
CORNER_WORDS = 12               # what function 0x36 copies per polygon (rep #0xc at 0x48e)
WINDOW = 0x10000                # words per bank window: the TGP reads (bank & 0xff0000) | offset16, and a list or
                                # a polygon read on past a window's end wraps to its start
HALF_WORDS = COPRO_SIZE // 8    # 0x80000 words: one 2 MB copy of copro_data
CUSTOM_WORDS = (COPRO_SIZE // 4, 0x800000)
"""A custom set's new copro_data area in words (romset.CUSTOM_FILES "dx-copro.bin": byte 0x400000 on, one copy), up to
the TGP's 23-bit word address under bank bit 0x800000 (docs/spike/2026-11-sixteen.md, "Room"). A ground may lie
there (a new course id's, Phase 8 Task 5) as well as in the first 2 MB copy."""


def _area_end(at: int) -> int | None:
    """The end (exclusive, words) of the copro_data area word `at` lies in: the first copy (HALF_WORDS) or a custom
    set's new area (CUSTOM_WORDS[1]); None elsewhere (the second copy, the stock region's zero half)."""
    if 0 <= at < HALF_WORDS:
        return HALF_WORDS
    if CUSTOM_WORDS[0] <= at < CUSTOM_WORDS[1]:
        return CUSTOM_WORDS[1]
    return None
ATTR_PAD = 0xCD                 # the attribute block's padding to 4 bytes on the real set
FREE_WORD = 0xFFFFFFFF          # copro_data's free run
COORD_LIMIT = 1024.0            # the grid spans -1024 <= x, z < 1024
MIN_AREA = 1e-6                 # |signed XZ area| below this is zero
_U16_MAX = 0xFFFF

Point = tuple[float, float, float]


@dataclass(frozen=True)
class GroundPoly:
    """One polygon of a course's pool (POLY_WORDS words). The originals are convex quads (a repeated corner makes a
    triangle) with negative signed XZ area (sum of x_k * z_k+1 - x_k+1 * z_k): the TGP's edge test is one-sided."""
    corners: tuple[Point, ...]      # words 0-11: four (x, y, z) f32; function 0x36 reads only these
    raw: tuple[int, ...] = ()       # the 16 words as read; () for a polygon not read from a ROM; not encoded
    centre: Point | None = None     # words 12-14: (x, y, z) f32, the corners' mean on most originals (functions
                                    # 0x2c and 0x33 read it by index); None encodes the corners' mean
    direction: int = 0              # word 15: a sign-extended s16 angle, 1/65536 turn (function 0x30 returns it,
                                    # 0x2a / 0x2b pass it to the sin/cos unit)


@dataclass(frozen=True)
class Ground:
    """A course's ground: the i960's grid and attribute block (data ROM) and the TGP's lists and polygons
    (copro_data), with where each lives."""
    course: int                         # course id (0 Beginner, 1 Expert, 2 Advanced)
    grid: tuple[int, ...]               # GRID_CELLS u16: 0 off the ground, else the word offset of a list
    attributes: bytes                   # one byte per polygon (low nibble: surface kind, 1 = road)
    polys: tuple[GroundPoly, ...]
    lists: dict[int, tuple[int, ...]]   # grid value (word offset in the list block) -> polygon indices
    base: int                           # the slot: function 0x0d's argument, the course id on the real set
    raw: tuple[int, ...]                # the list block's words as read (word 0 = 0, the off-road list); not encoded
    grid_at: int                        # data-ROM offset of the grid (*block_ref_a)
    attributes_at: int                  # data-ROM offset of the attribute block (*block_ref_b)
    list_base: int                      # copro_data word of the list block (slot table word SLOT_LIST + base)
    pool_base: int                      # copro_data word of the polygon pool (slot table word SLOT_POOL + base)


def _hex(v) -> str:
    return f"{int(v):#x}" if is_int(v) else repr(v)


def _cell_name(cell: int) -> str:
    return f"cell {cell} (row {cell >> 7}, col {cell & 127})"


def _crosses(start: int, n: int) -> bool:
    """True when n words from `start` cross a WINDOW boundary (n >= 1)."""
    return start // WINDOW != (start + n - 1) // WINDOW


def _copro_words(rs: RomSet, start: int, n: int, what: str) -> tuple[int, ...]:
    area = _area_end(start)
    end = min(len(rs.copro_data) // 4, area or 0)
    if start < 0 or n < 0 or start + n > end:
        top = min(len(rs.copro_data) // 4, CUSTOM_WORDS[1])
        raise TgpError(f"{what}: words {start:#x}..{start + n:#x} are outside copro_data (its first copy, "
                       f"{HALF_WORDS:#x} words, or a custom set's area {CUSTOM_WORDS[0]:#x}-{top:#x})")
    return struct.unpack_from(f"<{n}I", rs.copro_data, 4 * start)


STUB_BLOCKS = (0x18000, 0x465A0)
"""The id-3 stub's list block and polygon pool on the original set (slot 3's two words). course.grow_ids gives slot 3
Beginner's words but leaves the stub's blocks in place, so they still end Advanced's list block and pool: _bound
counts them as block starts whether or not a slot names them (on the original set slot 3 does, so nothing
changes)."""


def _bound(table: tuple[int, ...], start: int) -> int:
    """The end of the block at `start`: the next higher slot-table address or STUB_BLOCKS address (another block) in
    its area, else the area's end (the copy's, or a custom set's new area's)."""
    end = _area_end(start) or HALF_WORDS
    return min([a for a in tuple(table) + STUB_BLOCKS if start < a < end] + [end])


def _data_ptr(rs: RomSet, ref: int, size: int, what: str) -> int:
    """The data-ROM offset the pointer word at data offset `ref` holds, with `size` bytes inside data ROM."""
    if not 0 <= ref <= len(rs.data) - 4:
        raise TgpError(f"{what}: the pointer word at data {ref:#x} is outside data ROM")
    ptr = struct.unpack_from("<I", rs.data, ref)[0]
    off = ptr - DATA_BASE
    if not 0 <= off <= len(rs.data) - size:
        raise TgpError(f"{what}: the pointer at data {ref:#x} is {ptr:#x}; {size} bytes there are outside data ROM")
    return off


def decode_ground(rs: RomSet, course: int) -> Ground:
    """The ground of `course` (id 0-2; 0-15 on a grown set, course.grow_ids), as the game reaches it: the grid and
    attribute block through the course record's block_ref_a / block_ref_b, the list block and pool through slot
    `course` of the slot table (TgpError for `rs` not a RomSet or an id the set does not hold). The polygon
    count is the highest index any list names, plus one (every original polygon is in a list); TgpError when the pool
    holds more: a record from the count up to the pool's bound (the next slot-table address, else the copy's end)
    that is not free (FREE_WORD in all its words), so a pool may end early before free records. TgpError naming the
    cell and its grid value for a list that starts or runs past the course's list block (to the next slot-table
    address or the copy's end: it would be another course's bytes), that crosses a 64K-word window (the TGP would
    wrap), or that names a polygon past the course's pool (same bound); and for an off-road list (word 0) that is
    not empty, a direction word that is not a sign-extended s16, or a block outside its region."""
    if not isinstance(rs, RomSet):
        raise TgpError(f"{type(rs).__name__} is not a RomSet")
    n = id_count(rs)
    if not is_int(course) or not 0 <= course < n:
        raise TgpError(f"course {course!r} is not a course id 0..{n - 1}")
    course = int(course)
    name = course_name(course)
    entry = course_table(rs)[course]
    grid_at = _data_ptr(rs, entry.block_ref_a, 2 * GRID_CELLS, f"{name}: ground grid")
    attributes_at = _data_ptr(rs, entry.block_ref_b, 0, f"{name}: ground attributes")
    grid = struct.unpack_from(f"<{GRID_CELLS}H", rs.data, grid_at)
    slot = course
    table = _copro_words(rs, SLOT_LIST, 2 * SLOTS, f"{name}: slot table")
    list_base, pool_base = table[slot], table[SLOTS + slot]
    for what, at in (("list block", list_base), ("polygon pool", pool_base)):
        if _area_end(at) is None or at >= len(rs.copro_data) // 4:
            raise TgpError(f"{name}: slot {slot}'s {what} address {at:#x} is past copro_data's first copy and not in "
                           f"the set's custom area")
    list_end, pool_end = _bound(table, list_base), _bound(table, pool_base)
    capacity = (pool_end - pool_base) // POLY_WORDS
    block = f"the course's list block (words {list_base:#x}-{list_end - 1:#x})"
    off_road = _copro_words(rs, list_base, 1, f"{name}: list block")[0]
    if off_road:
        raise TgpError(f"{name}: the off-road list (grid value 0) at word {list_base:#x} counts {off_road}: "
                       f"it must be empty")
    lists: dict[int, tuple[int, ...]] = {}
    end = list_base + 1
    for cell, v in enumerate(grid):
        if v == 0 or v in lists:
            continue
        where = f"{name}: {_cell_name(cell)}: grid value {v:#x}"
        at = list_base + v
        if at >= list_end:
            raise TgpError(f"{where}: the list at word {at:#x} is past {block}")
        n = _copro_words(rs, at, 1, where)[0]
        if at + 1 + n > list_end:
            raise TgpError(f"{where}: the list at word {at:#x} counts {n} entries, running past {block}")
        if _crosses(at, 1 + n):
            raise TgpError(f"{where}: the list at word {at:#x} ({1 + n} words) crosses a 64K-word bank window: the "
                           f"TGP would wrap to the window's start")
        idx = _copro_words(rs, at + 1, n, where)
        for k, i in enumerate(idx):
            if i >= capacity:
                raise TgpError(f"{where}: list entry {k} (word {at + 1 + k:#x}) is polygon {i}, past the course's "
                               f"pool ({capacity} polygons at words {pool_base:#x}-{pool_end - 1:#x})")
        lists[v] = idx
        end = max(end, at + 1 + n)
    count = max((i + 1 for v in lists.values() for i in v), default=0)
    # The count is inferred from the lists; a polygon past it that no list names would be dropped unseen. Every
    # record from the count up to the pool's bound (the next higher slot-table address, else the copy's end) must be
    # free (FREE_WORD x POLY_WORDS): on the originals there is none (count == capacity) for the three courses; a pool
    # whose following block moved away and was erased (relocate_ground) reaches to the next block over free records,
    # and the last pool is followed by the free run. Words after the last whole record hold no polygon.
    limit = min(pool_end, len(rs.copro_data) // 4)
    tail_at = pool_base + POLY_WORDS * count
    tail_end = pool_base + POLY_WORDS * max(0, (limit - pool_base) // POLY_WORDS)
    if tail_end > tail_at:
        tail = np.frombuffer(rs.copro_data, "<u4", tail_end - tail_at, 4 * tail_at)
        used = np.flatnonzero(tail != FREE_WORD)
        if used.size:
            k = count + int(used[0]) // POLY_WORDS
            bound = (f"the next slot-table address {pool_end:#x}" if pool_end != _area_end(pool_base)
                     else f"the area's end {pool_end:#x}")
            raise TgpError(f"{name}: the pool at word {pool_base:#x} holds {capacity} polygons up to {bound} and the "
                           f"lists name {count}, but record {k} (word {pool_base + POLY_WORDS * k:#x}) is not free "
                           f"({FREE_WORD:#x}): polygon {k} is in no list")
    polys = []
    for i in range(count):
        at = pool_base + POLY_WORDS * i
        if _crosses(at, POLY_WORDS):
            raise TgpError(f"{name}: polygon {i} at word {at:#x} crosses a 64K-word bank window")
        words = _copro_words(rs, at, POLY_WORDS, f"{name}: polygon {i}")
        f = struct.unpack("<15f", struct.pack("<15I", *words[:15]))
        d = words[15] - (1 << 32) if words[15] & 0x80000000 else words[15]
        if not -0x8000 <= d <= 0x7FFF:
            raise TgpError(f"{name}: polygon {i} at word {at:#x}: direction word {words[15]:#x} is not a "
                           f"sign-extended s16")
        polys.append(GroundPoly(tuple(f[3 * k:3 * k + 3] for k in range(4)), words, f[12:15], d))
    if attributes_at + count > len(rs.data):
        raise TgpError(f"{name}: the attribute block at data {attributes_at:#x} ({count} bytes) runs past data ROM")
    return Ground(course, grid, rs.data[attributes_at:attributes_at + count], tuple(polys), lists, slot,
                  _copro_words(rs, list_base, end - list_base, f"{name}: list block"), grid_at, attributes_at,
                  list_base, pool_base)


def _encode_poly(p, i: int) -> bytes:
    """The POLY_WORDS words of polygon `i`; TgpError for one the TGP cannot answer for (see encode_ground)."""
    what = f"polygon {i}"
    if not isinstance(p, GroundPoly):
        raise TgpError(f"{what}: {p!r} is not a GroundPoly")
    corners = _seq(p.corners, 4, "corners", what, TgpError)
    blob = b"".join(_f32(c, 3, f"corner {k}", what, TgpError) for k, c in enumerate(corners))
    v = struct.unpack("<12f", blob)
    xz = [(v[3 * k], v[3 * k + 2]) for k in range(4)]
    for k, (x, z) in enumerate(xz):
        if not (-COORD_LIMIT <= x < COORD_LIMIT and -COORD_LIMIT <= z < COORD_LIMIT):
            raise TgpError(f"{what}: corner {k} (x {x}, z {z}) is outside the grid (-1024 <= x, z < 1024)")
    area = sum(xz[k][0] * xz[(k + 1) % 4][1] - xz[(k + 1) % 4][0] * xz[k][1] for k in range(4)) / 2
    if abs(area) < MIN_AREA:
        raise TgpError(f"{what}: zero XZ area ({area}): no point is inside it")
    if area > 0:
        raise TgpError(f"{what}: XZ area {area} is positive: the TGP's one-sided edge test needs the originals' "
                       f"winding (negative area) and would never hit it")
    for k in range(4):
        (ax, az), (bx, bz), (cx, cz) = xz[k], xz[(k + 1) % 4], xz[(k + 2) % 4]
        if (bx - ax) * (cz - bz) - (bz - az) * (cx - bx) > MIN_AREA:
            raise TgpError(f"{what}: not convex at corner {(k + 1) % 4}: the TGP's edge test would miss the notch")
    if p.centre is None:
        centre = struct.pack("<3f", *(sum(v[3 * k + j] for k in range(4)) / 4 for j in range(3)))
    else:
        centre = _f32(p.centre, 3, "centre", what, TgpError)
    direction = _int(p.direction, "direction", -0x8000, 0x7FFF, what, TgpError)
    return blob + centre + struct.pack("<i", direction)


def ground_tables(g: Ground) -> list[tuple[str, int, bytes, str]]:
    """(region, byte offset, bytes, name) of every block `g` occupies, and the one place that judges a ground's own
    validity (encode_ground, ground_spans, verify_lossless_ground and golden's rules call it); TgpError as
    encode_ground, one message per violation."""
    if not isinstance(g, Ground):
        raise TgpError(f"{g!r} is not a Ground")
    what = f"ground of course {g.course!r}"
    slot = _int(g.base, "slot (base)", 0, SLOTS - 1, what, TgpError)
    grid = _seq(g.grid, GRID_CELLS, "grid", what, TgpError)
    polys = _seq(g.polys, None, "polys", what, TgpError)
    n = len(polys)
    if not isinstance(g.attributes, (bytes, bytearray)) or len(g.attributes) != n:
        raise TgpError(f"{what}: attributes {g.attributes!r:.60} are not {n} bytes (one per polygon)")
    if not isinstance(g.lists, dict):
        raise TgpError(f"{what}: lists {g.lists!r:.60} is not a dict (grid value -> polygon indices)")
    bases = {}
    for field in ("list_base", "pool_base"):
        at = getattr(g, field)
        if not is_int(at) or _area_end(at) is None:
            raise TgpError(f"{what}: {field} {_hex(at)} is outside copro_data's first copy (0..{HALF_WORDS - 1:#x}) "
                           f"and a custom set's area ({CUSTOM_WORDS[0]:#x}..{CUSTOM_WORDS[1] - 1:#x})")
        if at < HEADER_WORDS:
            raise TgpError(f"{what}: {field} {at:#x} is inside the identity matrix and slot table (words 0-0x2f)")
        bases[field] = int(at)
    list_base, pool_base = bases["list_base"], bases["pool_base"]
    lists: dict[int, tuple[int, ...]] = {}
    for key in g.lists:
        if not is_int(key) or not 0 <= key <= _U16_MAX:
            raise TgpError(f"{what}: list key {_hex(key)} is not a grid value (u16)")
        idx = tuple(_int(i, f"list {int(key):#x} entry {k} polygon", 0, n - 1, what, TgpError)
                    for k, i in enumerate(_seq(g.lists[key], None, f"list {int(key):#x}", what, TgpError)))
        if key == 0:
            if idx:
                raise TgpError(f"{what}: the off-road list (grid value 0) must be empty: {idx!r:.60}")
            continue
        lists[int(key)] = idx
    block = [0]
    for key in sorted(lists):
        idx = lists[key]
        if key < len(block):
            raise TgpError(f"{what}: list {key:#x} would overlap the list before it, which ends at word "
                           f"{len(block):#x} of the block")
        if _crosses(list_base + key, 1 + len(idx)):
            raise TgpError(f"{what}: list {key:#x} at word {list_base + key:#x} ({1 + len(idx)} words) crosses a "
                           f"64K-word bank window: the TGP would wrap to the window's start")
        block += [0] * (key - len(block)) + [len(idx), *idx]
    for cell, v in enumerate(grid):
        if not is_int(v) or not 0 <= v <= _U16_MAX:
            raise TgpError(f"{what}: {_cell_name(cell)}: grid value {_hex(v)} is not a u16")
        if v and int(v) not in lists:
            raise TgpError(f"{what}: {_cell_name(cell)}: grid value {int(v):#x} has no list")
    reached = {int(v) for v in grid}
    for key in sorted(lists):
        if key not in reached:
            raise TgpError(f"{what}: list {key:#x} is reached by no grid cell: the decoder would not find it")
    named = {i for idx in lists.values() for i in idx}
    for i in range(n):
        if i not in named:
            raise TgpError(f"{what}: polygon {i} is in no list: the decoder would not find it")
    pool = b"".join(_encode_poly(p, i) for i, p in enumerate(polys))
    spans = (("list block", list_base, len(block)), ("polygon pool", pool_base, POLY_WORDS * n))
    for label, at, size in spans:
        if at + size > _area_end(at):
            raise TgpError(f"{what}: the {label} (words {at:#x}+{size:#x}) runs past its copro_data area (the first "
                           f"copy, the half that ends at {HALF_WORDS:#x}; or a custom set's, to {CUSTOM_WORDS[1]:#x})")
    for i in range(n):
        if _crosses(pool_base + POLY_WORDS * i, POLY_WORDS):
            raise TgpError(f"{what}: polygon {i} at word {pool_base + POLY_WORDS * i:#x} crosses a 64K-word bank "
                           f"window")
    if n and list_base < pool_base + POLY_WORDS * n and pool_base < list_base + len(block):
        raise TgpError(f"{what}: the list block (words {list_base:#x}+{len(block):#x}) and the polygon pool "
                       f"(words {pool_base:#x}+{POLY_WORDS * n:#x}) overlap")
    attrs = bytes(g.attributes) + bytes([ATTR_PAD]) * (-n % 4)
    data = []
    for field, size in (("grid_at", 2 * GRID_CELLS), ("attributes_at", len(attrs))):
        at = getattr(g, field)
        if not is_int(at) or not 0 <= at <= DATA_WINDOW - size:
            raise TgpError(f"{what}: {field} {_hex(at)} ({size} bytes) is outside data ROM (the 32 MB window)")
        data.append(int(at))
    grid_at, attributes_at = data
    if grid_at < attributes_at + len(attrs) and attributes_at < grid_at + 2 * GRID_CELLS:
        raise TgpError(f"{what}: the grid (data {grid_at:#x}) and the attribute block (data {attributes_at:#x}) "
                       f"overlap")
    return [("data", grid_at, struct.pack(f"<{GRID_CELLS}H", *(int(v) for v in grid)), "ground grid"),
            ("data", attributes_at, attrs, "ground attributes"),
            ("copro_data", 4 * (SLOT_LIST + slot), struct.pack("<I", list_base), "ground slot: list block address"),
            ("copro_data", 4 * (SLOT_POOL + slot), struct.pack("<I", pool_base), "ground slot: polygon pool address"),
            ("copro_data", 4 * list_base, struct.pack(f"<{len(block)}I", *block), "ground lists"),
            ("copro_data", 4 * pool_base, pool, "ground polygons")]


def encode_ground(g: Ground) -> dict[tuple[str, int], bytes]:
    """{(region, byte offset): bytes} for `g`: the grid and the attribute block (padded to 4 bytes with ATTR_PAD) in
    data ROM; the slot's two words, the list block (word 0 = 0, each list at its grid value: count, indices; gaps
    0) and the polygon pool in copro_data (first copy; `RomSet.with_patch` writes both). `raw` fields are not
    encoded. TgpError for a polygon the TGP cannot answer for (corners not four finite f32 triples, a corner outside
    -1024 <= x, z < 1024, zero or positive XZ area, not convex; a centre that is not three finite f32 or a direction
    outside s16), a grid value that is not a u16 or has no list (naming the cell and the value), a list key past
    u16, an index past the polygon count, a polygon no list names, a list no grid cell reaches (the decoder could
    not find either), lists that overlap, a non-empty off-road list, a list or polygon crossing
    a 64K-word window, blocks past their region or overlapping, a slot past 15 or attributes that are not one byte
    per polygon. The game reads slot = course id (function 0x0d's argument), so the ground of course c is found only
    in slot c: `base` must be the course id (golden.check_course refuses another; encode_ground writes any slot)."""
    return {(region, off): blob for region, off, blob, _ in ground_tables(g)}


def ground_spans(rs: RomSet, course: int) -> list[tuple[str, int, int, str]]:
    """(region, byte offset, length, name) of the bytes `course`'s ground occupies, as `encode_ground` writes them.
    No span is shared between courses: each has its own slot words, list block, pool, grid and attribute block."""
    return [(region, off, len(blob), name) for region, off, blob, name in ground_tables(decode_ground(rs, course))]


def verify_lossless_ground(rs: RomSet, course: int) -> list[str]:
    """One line per block where `encode_ground(decode_ground(rs, course))` differs from the ROM ([] when lossless)."""
    out = []
    for region, off, blob, name in ground_tables(decode_ground(rs, course)):
        rom = getattr(rs, region)[off:off + len(blob)]
        if rom != blob:
            k = next((k for k, (a, b) in enumerate(zip(rom, blob)) if a != b), min(len(rom), len(blob)))
            got = f"{rom[k]:#04x}" if k < len(rom) else "past the region"
            out.append(f"{course_name(course)}: {name} ({region} {off:#x}, {len(blob)} bytes): first difference at "
                       f"{off + k:#x}: ROM {got}, encoded {blob[k]:#04x}")
    return out


# ---- relocation: a course's ground copied into free space, every pointer rewritten ----

# (byte offset, length) of the free runs (0xff): copro_data in each 2 MB copy, from the word after slot 3's last
# polygon (0x4a250) to the copy's end (the plan's 0x12893e starts 2 bytes inside that polygon: Task 3); data ROM to
# 0x700000.
COPRO_FREE = (0x128940, 882368)
DATA_FREE = (0x65D120, 667360)
_FREE = 0xFF
_RECORD_RAW = 0x18          # course record words 6-7 (CourseEntry.raw): no reader found; equal to *block_ref_a


def write_free(rs: RomSet, region: str, offset: int, blob: bytes) -> RomSet:
    """A copy of `rs` with `blob` at byte `offset` of `region` ("copro_data": its first 2 MB copy, written into
    both, or on a custom set its new area from COPRO_SIZE, one copy; "data": data ROM, a span in 0x800000-0xffffff
    written into all eight 1 MB copies, one from DATA_COPY_END on a custom set written once). TgpError when the
    target is not free (a byte that is not 0xff, in either copy), is reserved (romset.RESERVED), lies in a new id's
    course sheet block (texture.custom_sheet_blocks) or, in any copy, in any other span a table or block owns
    (golden.owned_spans: every course's tables, a ground pool's 0xffffffff words among them; owned whatever its
    bytes, as the allocators treat it), the offset is not 4-byte aligned, the span is past the region (or before it;
    copro_data's second 2 MB copy is not a target), or a data span crosses 0x800000, two 1 MB copies or their end; for
    an argument of the wrong type; and when a course of the set does not decode (its owned space is then unknown).

    The owned space is read through golden.owned (cached per set) and carried to the result (golden.carry_owned):
    the bytes written were owned by nothing, so the result owns what `rs` owns until a pointer names them (written
    after, not through write_free). A chain of writes from one set (relocate_ground, the converter) decodes it
    once, and never decodes a half-written set (a pool bounded by its area's end before the slot table names the
    new one)."""
    if not isinstance(rs, RomSet):
        raise TgpError(f"{rs!r:.60} is not a RomSet")
    if region not in ("copro_data", "data"):
        raise TgpError(f"region {region!r} is not one the ground is written to (copro_data, data)")
    if not isinstance(blob, (bytes, bytearray)):
        raise TgpError(f"{region}: {blob!r:.60} is not bytes")
    if not is_int(offset):
        raise TgpError(f"{region}: offset {offset!r} is not a byte offset (int)")
    offset, n = int(offset), len(blob)
    image = getattr(rs, region)
    new_area = region == "copro_data" and len(image) > COPRO_SIZE and offset >= COPRO_SIZE
    size = min(len(image), COPRO_SIZE // 2) if region == "copro_data" and not new_area else len(image)
    if offset % 4:
        raise TgpError(f"{region} {offset:#x}: unaligned (the target must be 4-byte aligned)")
    if offset < 0 or offset + n > size:
        raise TgpError(f"{region} {offset:#x}+{n:#x}: past the region ({size:#x} bytes"
                       f"{', the first copy' if region == 'copro_data' and not new_area else ''})")
    if region == "data" and offset + n > DATA_COPY_BASE and offset < DATA_COPY_END and n and (
            offset < DATA_COPY_BASE or offset + n > DATA_COPY_END or (offset - DATA_COPY_BASE) // DATA_COPY_LEN !=
            (offset + n - 1 - DATA_COPY_BASE) // DATA_COPY_LEN):
        raise TgpError(f"data {offset:#x}+{n:#x} crosses 0x800000, spans two of the 1 MB copies above it or their "
                       f"end at {DATA_COPY_END:#x}")
    if owner := reserved(region, offset, n):
        raise TgpError(f"{region} {offset:#x}+{n:#x} is reserved for {owner} (romset.RESERVED)")
    if region == "data" and n:
        try:
            blocks = custom_sheet_blocks(rs)
        except DecodeError as e:
            raise TgpError(f"data {offset:#x}+{n:#x}: {e}") from e
        for _, lo, size, owner in blocks:
            if offset < lo + size and lo < offset + n:
                raise TgpError(f"data {offset:#x}+{n:#x} is in {owner} ({lo:#x}+{size:#x}): a new id's texel "
                               f"block, owned whatever its bytes")
    copies = [offset]
    if region == "copro_data" and len(image) >= COPRO_SIZE and not new_area:
        copies.append(offset + COPRO_SIZE // 2)
    for at in copies:
        target = image[at:at + n]
        if target != bytes([_FREE]) * n:
            k = next(k for k, b in enumerate(target) if b != _FREE)
            raise TgpError(f"{region} {at + k:#x} is not free (byte {target[k]:#04x}, not {_FREE:#04x}): the "
                           f"target {offset:#x}+{n:#x} must be free")
    if region == "data" and DATA_COPY_BASE <= offset < DATA_COPY_END:
        rel = (offset - DATA_COPY_BASE) % DATA_COPY_LEN
        copies = [DATA_COPY_BASE + k * DATA_COPY_LEN + rel for k in range(DATA_COPY_COUNT + 1)]
    from dtools.arcade import golden        # golden imports this module
    for at in copies if n else ():
        try:
            owner = golden.owner(rs, region, at, n)
        except DecodeError as e:
            raise TgpError(f"{region} {offset:#x}+{n:#x}: the set's owned space (golden.owned_spans): {e}") from e
        if owner:
            raise TgpError(f"{region} {offset:#x}+{n:#x} is owned by {owner} (golden.owned_spans"
                           f"{f', at {at:#x}' if at != offset else ''}), whatever its bytes")
    try:
        out = rs.with_patch(region, offset, bytes(blob))
    except RomSetError as e:
        raise TgpError(f"{region} {offset:#x}+{n:#x}: {e}") from e
    golden.carry_owned(rs, out)
    return out


def list_block_words(lists: dict[int, tuple[int, ...]]) -> int:
    """The list block's length in words as encode_ground lays it out: word 0, then each list at its grid value."""
    return max([1] + [k + 1 + len(v) for k, v in lists.items() if k])


def window_safe(lists: dict[int, tuple[int, ...]], n: int, list_base: int, block: int) -> bool:
    """True when, with the list block at word `list_base` and `n` polygons right after its `block` words, no list and
    no polygon crosses a bank window (a pool off a 16-word boundary crosses one exactly when its span does)."""
    keys = np.array([k for k in lists if k], np.int64)
    if keys.size:
        lens = np.array([len(lists[k]) for k in lists if k], np.int64)
        at = list_base + keys
        if np.any(at // WINDOW != (at + lens) // WINDOW):
            return False
    pool = list_base + block
    return not n or pool % POLY_WORDS == 0 or not _crosses(pool, POLY_WORDS * n)


def placed_ground(g: Ground, list_base: int, data_at: int) -> Ground:
    """`g` placed as relocate_ground places a ground: the list block at copro_data word `list_base` and the pool
    right after it (list_block_words), the grid at data byte `data_at` and the attribute block right after it. The
    slot and the grid values (word offsets in the list block) are relative and stay. Not checked here
    (ground_tables judges the layout: a bank-window crossing, a block past its region)."""
    return replace(g, list_base=list_base, pool_base=list_base + list_block_words(g.lists), grid_at=data_at,
                   attributes_at=data_at + 2 * GRID_CELLS)


def repoint_record_raw(raw: tuple[int, ...], old: int, new: int) -> tuple[int, ...]:
    """A course record's words 6-7 (CourseEntry.raw) with every word that holds `old` (the grid pointer, as on the
    real set) set to `new`; the others kept. No reader of these words is known (_RECORD_RAW)."""
    return tuple(new if w == old else w for w in raw)


def relocate_ground(rs: RomSet, course: int, copro_at: int | None = None, data_at: int | None = None, *,
                    erase_original: bool = False) -> RomSet:
    """A copy of `rs` with `course`'s ground moved: the list block and then the pool (right after it) to copro_data
    byte `copro_at`, the grid and then the attribute block to data byte `data_at` (default DATA_FREE's start), each
    written only into free bytes (`write_free`). The default `copro_at` is the first word from COPRO_FREE's start at
    which no list and no polygon crosses a bank window: COPRO_FREE's start for Beginner, 6 / 14 words on for Expert
    / Advanced (so their pools start on a 16-word boundary). That default assumes COPRO_FREE's start is still free:
    a second relocation (another course after the first) passes its own `copro_at`, which is used as given, not
    aligned or searched. Rewritten: the course record's two ground pointer words
    (at block_ref_a / block_ref_b) and its words 6-7 where they hold the grid pointer (they do on the real set), and
    the slot-table words of slot `course` and of every other slot that holds the same two addresses (on the real
    set slots 4-15 repeat slot 0's). The slot (= course id, function 0x0d's argument) and the grid values (word
    offsets in the list block) are relative and stay. With `erase_original` the old list block, pool, grid and
    attribute block are filled with 0xff (the slot words of the course's id are rewritten either way).

    This is more than golden writes for a moved ground: golden.tables writes only slot `base`'s two words and keeps
    the record's words 6-7 as entry.raw holds them. No reader of either is known (words 6-7: none found; slots 4-15:
    function 0x0d is sent the course id 0-2), so the two agree on everything the game is known to read (only this
    function's output was run in MAME, for Beginner).

    A list or a polygon must not cross a 64K-word bank window (the TGP reads (bank & 0xff0000) | offset16 and walks
    on with a 16-bit index); the blocks may, since the program re-banks for every list and every polygon ($0x69 +
    grid value, $0x6a + 16 * index: Task 3), as Expert's original pool does. The result is decoded again and must
    equal the original ground in every field but the four addresses, and every other course that decoded before
    must decode the same: so a layout the decoder rejects raises here. Moving Expert or Advanced needs
    `erase_original`: the slot table no longer bounds the pool before the old one (Beginner's before Expert's,
    Expert's before Advanced's), which then reaches to the next slot-table address over the old polygons, neither
    listed nor free (decode_ground's pool-end rule); erased, they are free records and the neighbour decodes as
    before. Beginner moves either way (no pool ends at its old blocks). TgpError for all of it: the course, a target
    that is not free, unaligned or past its region, a bank-window crossing, the check (a neighbour broken by the old
    blocks says that erase_original frees them)."""
    if not isinstance(rs, RomSet):
        raise TgpError(f"{rs!r:.60} is not a RomSet")
    if not isinstance(erase_original, bool):
        raise TgpError(f"erase_original {erase_original!r} is not a bool")
    g = decode_ground(rs, course)
    course, name = g.course, course_name(g.course)
    block = list_block_words(g.lists)
    if copro_at is None:
        lb = COPRO_FREE[0] // 4
        while lb < COPRO_FREE[0] // 4 + WINDOW and not window_safe(g.lists, len(g.polys), lb, block):
            lb += 1
        copro_at = 4 * lb
    data_at = DATA_FREE[0] if data_at is None else data_at
    sizes = {"copro_at": ("copro_data", COPRO_SIZE // 2, 4 * (block + POLY_WORDS * len(g.polys))),
             "data_at": ("data", DATA_SIZE, 2 * GRID_CELLS + len(g.attributes) + (-len(g.attributes) % 4))}
    for label, at in (("copro_at", copro_at), ("data_at", data_at)):
        region, size, need = sizes[label]
        if not is_int(at):
            raise TgpError(f"{name}: {label} {at!r} is not a byte offset (int)")
        if at % 4:
            raise TgpError(f"{name}: {label} {at:#x} is unaligned (4-byte words)")
        if at < 0 or at + need > size:
            raise TgpError(f"{name}: {label} {at:#x}: {need:#x} bytes there run past {region} ({size:#x} bytes)")
    new = placed_ground(g, int(copro_at) // 4, int(data_at))
    layout = ground_tables(new)         # TgpError for a list or polygon across a bank window, before any write
    others = {}
    for c in range(COURSE_COUNT):
        if c != course:
            try:
                others[c] = decode_ground(rs, c)
            except TgpError:
                pass                    # a course that does not decode before is not checked after
    out = rs
    for region, off, blob, label in layout:
        if not label.startswith("ground slot"):
            out = write_free(out, region, off, blob)
    entry = course_table(rs)[course]
    grid_ptr, attr_ptr = DATA_BASE + new.grid_at, DATA_BASE + new.attributes_at
    out = out.with_patch("data", entry.block_ref_a, encode_table_pointer(grid_ptr))
    out = out.with_patch("data", entry.block_ref_b, encode_table_pointer(attr_ptr))
    record = COURSE_TABLE - DATA_BASE + course * COURSE_ENTRY_SIZE
    raw = repoint_record_raw(entry.raw, DATA_BASE + g.grid_at, grid_ptr)
    for k, (word, moved) in enumerate(zip(entry.raw, raw)):
        if moved != word:
            out = out.with_patch("data", record + _RECORD_RAW + 4 * k, encode_table_pointer(moved))
    table = _copro_words(rs, SLOT_LIST, 2 * SLOTS, f"{name}: slot table")
    for slot in range(SLOTS):
        if slot == g.base or (table[slot], table[SLOTS + slot]) == (g.list_base, g.pool_base):
            out = out.with_patch("copro_data", 4 * (SLOT_LIST + slot), struct.pack("<I", new.list_base))
            out = out.with_patch("copro_data", 4 * (SLOT_POOL + slot), struct.pack("<I", new.pool_base))
    if erase_original:
        for region, off, blob, label in ground_tables(g):
            if not label.startswith("ground slot"):
                out = out.with_patch(region, off, bytes([_FREE]) * len(blob))
    try:
        got = decode_ground(out, course)
    except TgpError as e:
        raise TgpError(f"{name}: the relocated layout does not decode: {e}") from e
    for field in ("course", "grid", "attributes", "lists", "base", "raw", "polys", "grid_at", "attributes_at",
                  "list_base", "pool_base"):
        if getattr(got, field) != getattr(new, field):
            raise TgpError(f"{name}: the relocated ground decodes with a different {field}")
    for c, before in others.items():
        try:
            after = decode_ground(out, c)
        except TgpError as e:
            hint = ("" if erase_original else f"; without erase_original {name}'s old blocks stay, now inside "
                    f"{COURSE_NAMES[c]}'s bound: erase_original=True frees them")
            raise TgpError(f"{name}: relocating it breaks {COURSE_NAMES[c]}: {e}{hint}") from e
        if after != before:
            raise TgpError(f"{name}: relocating it changes {COURSE_NAMES[c]}'s decoded ground")
    return out


def program(rs: RomSet) -> tuple[int, ...]:
    """The TGP program as the i960 uploads it: `PROGRAM`'s words, little-endian, word 0 at TGP program address 0.
    The game uploads one program, once (at boot); it is the one running during the race. TgpError when the region
    is too short to hold it."""
    region, offset, count = PROGRAM
    image = getattr(rs, region)
    if offset + 4 * count > len(image):
        raise TgpError(f"the TGP program ({count} words at {region} {offset:#x}) runs past {region} "
                       f"({len(image):#x} bytes)")
    return struct.unpack_from(f"<{count}I", image, offset)


# --- The query model: function 0x36 (docs/formats/model2-tgp-ground.md section 7) ---

_F = np.float32


def _finite(value, what: str):
    """`value` as a finite f32 (what the i960 sends), else TgpError."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Real, np.integer, np.floating)):
        raise TgpError(f"{what} must be a finite number, got {value!r}")
    with np.errstate(all="ignore"):
        out = _F(value)
    if not np.isfinite(out):
        raise TgpError(f"{what} must be a finite f32, got {value!r}")
    return out


def cell(x: float, z: float) -> int:
    """The grid index the i960 computes for a point (routine 0x11bec): per axis round to nearest (cvtri, ties to
    even), + 1024, >> 4, & 0x7f; then `row * 128 + col` with the row from z and the column from x. A point outside
    -1024 <= x, z < 1024 wraps (the & 0x7f), it is not clamped and does not raise. TgpError for a non-finite
    argument."""
    col = ((round(float(_finite(x, "x"))) + 1024) >> 4) & 0x7F
    row = ((round(float(_finite(z, "z"))) + 1024) >> 4) & 0x7F
    return row * 128 + col


def surface(g: Ground, poly: int) -> int:
    """The surface kind of polygon `poly`: its attribute byte's low nibble (1 = road). TgpError when `poly` is not an
    index into the course's polygons, `g` is not a Ground or its attributes are not bytes."""
    if not isinstance(g, Ground):
        raise TgpError(f"{g!r:.60} is not a Ground")
    if not isinstance(g.attributes, (bytes, bytearray)):
        raise TgpError(f"ground of course {g.course!r}: attributes {g.attributes!r:.60} are not bytes")
    if not is_int(poly) or not 0 <= poly < len(g.attributes):
        raise TgpError(f"polygon {poly!r} is not an index into the {len(g.attributes)} attribute bytes")
    return g.attributes[poly] & 15


INV_TABLE_FILES = ("opr-14742a.45", "opr-14743a.46")    # the TGP tables ROMs (RomSet.extra), 16-bit halves of one u32
INV_TABLE_BASE = 0x8000                                   # the 1/x table's first word in that 0x10000-word region
NO_HEIGHT = -0.1                                          # TGP word 0x206 (init 0x38-0x3c): ny == 0 answers this


def inv_table(rs: RomSet) -> tuple[int, ...]:
    """The TGP's 1/x lookup (MAME `copro_tgp_tables`, words INV_TABLE_BASE.. used) from the set's two tables ROMs
    (the first is each word's low 16 bits, the second its high 16 bits). Pass it to `query` for the TGP's own
    reciprocal; TgpError when the set lacks the ROMs."""
    try:
        lo, hi = (rs.extra[n] for n in INV_TABLE_FILES)
    except (KeyError, AttributeError) as e:
        raise TgpError(f"the set has no {', '.join(INV_TABLE_FILES)} (the 1/x table)") from e
    if len(lo) != 0x20000 or len(hi) != 0x20000:
        raise TgpError(f"the tables ROMs are {len(lo):#x} and {len(hi):#x} bytes, not 0x20000")
    a = np.frombuffer(lo, "<u2").astype(np.uint32)
    b = np.frombuffer(hi, "<u2").astype(np.uint32)
    return tuple(int(v) for v in (b << 16) | a)


def _bits(v) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _from_bits(b: int):
    return _F(struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0])


def _recip_fn(inv):
    """The reciprocal step of routines 0x684 / 0x72b (code `mov a,$0x28 (e)` ... `fml`): with the divisor a stored to
    the 1/x unit, m = a with its exponent set to 0x7f, T0 / T1 the unit's two words, R = T1 - m * |T0|; the
    quotient is numerator * R. `inv` is `inv_table(rs)`; None gives plain f32 `1 / a`, which differs by a few f32
    ulps (so near-equal choices can come out the other way)."""
    if inv is None:
        return lambda v: _F(1) / v

    def recip(v):
        base = _bits(v)
        index = ((base >> 9) & 0x3FFE) | INV_TABLE_BASE
        words = []
        for k in (0, 1):
            w = inv[index | k]
            exp = ((w >> 23) + (0x7F - ((base >> 23) & 0xFF))) & 0xFF
            words.append((w & 0x007FFFFF) | (exp << 23))
        words[1] |= base & 0x80000000
        m = _from_bits((base & 0x807FFFFF) | (0x7F << 23))
        return _from_bits(words[1]) - m * _from_bits(words[0])
    return recip


def _hit(corners, x, z) -> bool:
    """Routine 0x499-0x4a5: d >= 0 on all four edges (a -0.0 passes: the TGP's sign flag needs a non-zero
    magnitude), d = z X_k - x Z_k - z X_k+1 + x Z_k+1 - Z_k+1 X_k + X_k+1 Z_k evaluated left to right with every
    product and sum rounded to f32."""
    for k in range(4):
        xa, za = _F(corners[k][0]), _F(corners[k][2])
        xb, zb = _F(corners[(k + 1) % 4][0]), _F(corners[(k + 1) % 4][2])
        d = z * xa - x * za
        d = d - z * xb
        d = d + x * zb
        d = d - zb * xa
        d = d + xb * za
        if _bits(d) & 0x80000000 and _bits(d) & 0x7FFFFFFF:      # SGD: negative and not zero
            return False
    return True


def _plane(a, b, c, x, z, recip):
    """Routine 0x72b-0x796: the height at (x, z) of the plane through corners a, b, c: n = (b - a) x (c - a),
    r = (((nx x + nz z) - nx Xa) - ny Ya) - nz Za, height = -1 * (r * 1/ny). ny == 0 (the plane has no XZ extent:
    a triangle's 0-2-3 when corners 2 and 3 coincide) answers NO_HEIGHT, the constant at TGP word 0x206."""
    ax, ay, az = (_F(v) for v in a)
    bx, by, bz = (_F(v) for v in b)
    cx, cy, cz = (_F(v) for v in c)
    nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
    ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
    nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if ny == 0:
        return _F(NO_HEIGHT)
    r = nx * x + nz * z
    r = r - nx * ax
    r = r - ny * ay
    r = r - nz * az
    return _F(-1) * (r * recip(ny))


def _height(corners, x, z, recip):
    """Routine 0x684: express (x, z) - corner 2 as alpha (corner 0 - corner 2) + beta (corner 1 - corner 2). When
    alpha and beta are both non-negative (the sign flag of their bitwise OR plus 0.0: set only for a negative,
    non-zero value) the point is on corner 1's side of the 0-2 diagonal and the plane is 0-1-2, else 2-3-0
    (corners 0, 2, 3). The solve branches on Z0 == Z2: with q = Z0 - Z2 non-zero beta = (q s - t p) /
    (q u - v p), else beta = t / v; then alpha = (s - u beta) / p, or when p = X0 - X2 is zero
    (t u - s v) / (q u). (u, v) = corner 1 - corner 2, (s, t) = point - corner 2, each division a numerator times
    `recip`."""
    (x0, _, z0), (x1, _, z1), (x2, _, z2) = ((_F(v) for v in corners[i]) for i in range(3))
    u, v = x1 - x2, z1 - z2
    p, q = x0 - x2, z0 - z2
    s, t = x - x2, z - z2
    if q != 0:
        beta = (q * s - t * p) * recip(q * u - v * p)
    else:
        beta = t * recip(v)
    if p != 0:
        alpha = (s - u * beta) * recip(p)
    else:
        alpha = (t * u - s * v) * recip(q * u)
    side = _from_bits(_bits(alpha) | _bits(beta)) + _F(0)
    if not (_bits(side) & 0x80000000 and _bits(side) & 0x7FFFFFFF):
        return _plane(corners[0], corners[1], corners[2], x, z, recip)
    return _plane(corners[0], corners[2], corners[3], x, z, recip)


def _cell_polys(g: Ground, c: int) -> list[tuple[int, tuple[Point, ...]]]:
    """(index, corners) of the polygons cell `c`'s list names, in list order ([] off the ground, grid value 0), each
    corner three f32 as Python floats; TgpError for a Ground the TGP could not answer from: `g` not a Ground, a grid
    that is not GRID_CELLS values, a grid value that is not a u16 or names no list, a list that is not a sequence of
    polygon indices, an index past the polygons, a polygon that is not a GroundPoly or whose corners are not four
    finite f32 triples. Only what the query reads is checked (the cell's list and its polygons), not the layout."""
    if not isinstance(g, Ground):
        raise TgpError(f"{g!r:.60} is not a Ground")
    what = f"ground of course {g.course!r}"
    grid = g.grid
    if (not isinstance(grid, (tuple, list, np.ndarray)) or (isinstance(grid, np.ndarray) and grid.ndim != 1)
            or len(grid) != GRID_CELLS):
        shown = (f"{len(grid)} values" if isinstance(grid, (tuple, list)) or (isinstance(grid, np.ndarray)
                                                                             and grid.ndim == 1) else f"{grid!r:.60}")
        raise TgpError(f"{what}: grid: {shown}, not {GRID_CELLS} (128 x 128 u16)")
    value = grid[c]
    if not is_int(value) or not 0 <= value <= _U16_MAX:
        raise TgpError(f"{what}: {_cell_name(c)}: grid value {value!r:.60} is not a u16")
    if not value:
        return []
    if not isinstance(g.lists, dict):
        raise TgpError(f"{what}: lists {g.lists!r:.60} is not a dict (grid value -> polygon indices)")
    if int(value) not in g.lists:
        raise TgpError(f"{what}: {_cell_name(c)}: grid value {int(value):#x} has no list")
    idx = _seq(g.lists[int(value)], None, f"list {int(value):#x}", what, TgpError)
    polys = g.polys
    if not isinstance(polys, (tuple, list)):
        raise TgpError(f"{what}: polys {polys!r:.60} is not a sequence of GroundPoly")
    out = []
    for k, i in enumerate(idx):
        i = _int(i, f"list {int(value):#x} entry {k} polygon", 0, len(polys) - 1, what, TgpError)
        p = polys[i]
        if not isinstance(p, GroundPoly):
            raise TgpError(f"{what}: polygon {i}: {p!r:.60} is not a GroundPoly")
        corners = _seq(p.corners, 4, "corners", f"{what}: polygon {i}", TgpError)
        v = struct.unpack("<12f", b"".join(_f32(c3, 3, f"corner {j}", f"{what}: polygon {i}", TgpError)
                                           for j, c3 in enumerate(corners)))
        out.append((i, tuple(v[3 * j:3 * j + 3] for j in range(4))))
    return out


def query(g: Ground, x: float, y: float, z: float, inv=None) -> tuple[float, int] | None:
    """What the TGP answers function 0x36 with for a car at (x, y, z): (height, polygon index), or None when no
    listed polygon of the point's cell contains it (the i960 then uses -0.1). The cell is `cell(x, z)`, so a point
    outside the grid wraps as the game's does. Polygons are tested in list order; among the hits the one with the
    smallest | |height| - y | wins and a later hit replaces an equal one. The height is an f32 as a Python float
    (-0.0 on flat ground at y 0). `inv` is `inv_table(rs)`: with it the divisions are the TGP's own (the result
    reproduces its f32 bits, to the best of the listing), without it plain f32 divisions. TgpError for a
    non-finite x, y or z, and for a Ground the TGP could not answer from (`_cell_polys`: the cell's list and its
    polygons are checked)."""
    fx, fy, fz = _finite(x, "x"), _finite(y, "y"), _finite(z, "z")
    listed = _cell_polys(g, cell(fx, fz))
    recip = _recip_fn(inv)
    best = None                                         # (| |height| - y |, height, index)
    with np.errstate(all="ignore"):
        for i, corners in listed:
            if not _hit(corners, fx, fz):
                continue
            h = _height(corners, fx, fz, recip)
            miss = abs(fy - abs(h))
            if best is None or miss <= best[0]:
                best = (miss, h, i)
    return None if best is None else (float(best[1]), best[2])
