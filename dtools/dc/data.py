"""The Dreamcast course data, `CS_DATAn.BIN` / `CSDATn{v}.BIN`: a lossless decoder and encoder.

Layout (docs/formats/dreamcast-disc.md, "`CS_DATAn.BIN` / `CSDATnv.BIN`"; docs/spike/2026-11-dc-decode.md, "Course
data"): eight little-endian u32 section offsets H0-H7 (H0's word is 0: its section starts at 0x20, first), then the
sections in an order that varies by course (`CourseData.order`), each at the end of the one before it rounded up to
its alignment (`ALIGN`: 2 for the u16 sections H2 and H7, 4 for the rest; the pad is zero), the file ending with the
last. The sections are the arcade's course tables (docs/formats/model2-course-tables.md, model2-tgp-ground.md):
courses 0-2 hold the arcade Beginner, Expert and Advanced tables value for value, with z negated.

| section | content | arcade table |
|---|---|---|
| H0 | 8 AI lines, each `count` points of 28 bytes then its first 8 points again (the wrap) | the line table (`AiPoint`) |
| H1 | 3 u32: segments (road points - 1), steps, the file offset of the steps (inside H3) | the record table's n, steps' n |
| H2 | the grid: 128 x 128 u16 (row z, column x, 16-unit squares from -1024), the u16 offset of a list in H7 | `*block_ref_a` |
| H3 | one attribute byte per cell (1 = road), a zero pad to 2 bytes, then the steps (u16 segment, u16 flags) | `*block_ref_b`, the step table |
| H4 | the line directory: 8 x (u32 offset, u32 count) | the line table's pairs |
| H5 | the road line: per segment a u8 lane_lo, u8 lane_hi, s16 curve and 8 lanes of 20 bytes | the record table (`RoadPoint`) |
| H6 | the cells, 64 bytes: 4 corners and a centre (x, y, z f32) and a direction word; the first `steps` are the road | the TGP ground polygons |
| H7 | the cell lists: u16 count, then that many u16 cell indices; list 0 is the empty one (grid value 0) | the TGP list block |

Against the arcade (courses 0-2, measured): positions are (x, y, -z); an AI point's speed differs by under 6e-6, its
pitch is negated, its heading is the arcade's + 0x1fff (mod 0x10000), roll, steer, lane and step are the arcade's;
a lane's pitch is negated, its heading + 0x1fff, its roll kept; a cell's direction word is the arcade's s16 + 0x8000
as a u16. The grid and lists are rebuilt for the negated z (they do not equal the arcade's).

Every malformed input is a `DcError` naming the file and the offset; the encoder refuses what the format cannot hold
and everything the decoder refuses, so whatever decodes re-encodes byte for byte."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from dtools.arcade.course import AI_LINE_WRAP as LINE_WRAP, AI_POINT_SIZE as LINE_POINT_SIZE, LANE_SIZE, LANES, \
    ROAD_POINT_SIZE
from dtools.arcade.tablebase import _f32, _int, _seq, is_number
from dtools.dc.errors import DcError

HEADER_SIZE = 0x20
SECTIONS = 8
ALIGN = (4, 4, 2, 4, 4, 4, 4, 2)        # by section H0-H7
SECTION_NAMES = ("H0 (AI lines)", "H1 (counts)", "H2 (grid)", "H3 (attributes and steps)", "H4 (line directory)",
                 "H5 (road line)", "H6 (cells)", "H7 (cell lists)")
LINE_COUNT = 8                          # LINE_WRAP, LINE_POINT_SIZE, LANES, LANE_SIZE, ROAD_POINT_SIZE: the arcade's
CELL_SIZE = 64
STEP_SIZE = 4
COUNTS_SIZE = 12
DIRECTORY_SIZE = 8 * LINE_COUNT
GRID_SIZE = 128
GRID_CELLS = GRID_SIZE * GRID_SIZE
GRID_BYTES = 2 * GRID_CELLS
GRID_ORIGIN = -1024.0                   # world x and z of square (0, 0)'s corner
GRID_SQUARE = 16.0                      # world units per square

_LINE_POINT = struct.Struct("<4f5hH")
_LANE = struct.Struct("<3f3hH")
_ROAD_HEAD = struct.Struct("<BBh")
_CELL = struct.Struct("<15fI")


@dataclass(frozen=True)
class LinePoint:
    """One point of an AI line (H0, 28 bytes): the arcade's `AiPoint` (course.AiPoint). speed multiplies the car's
    speed setting; pitch, heading, roll and steer are s16 angles (heading = the arcade's + 0x1fff); lane is the
    lateral position on the road line in lanes, 8.8 fixed (s16); step the index of the point's step (u16)."""
    x: float
    y: float
    z: float
    speed: float
    pitch: int
    heading: int
    roll: int
    steer: int
    lane: int
    step: int


@dataclass(frozen=True)
class Lane:
    """One lane point of a road point (20 bytes): x, y, z f32, pitch, heading, roll s16 and a u16 pad (0 on the disc).
    Unused slots (outside lane_lo..lane_hi) are (0, 0, -0.0) with heading 0x1fff."""
    x: float
    y: float
    z: float
    pitch: int
    heading: int
    roll: int
    pad: int


@dataclass(frozen=True)
class RoadPoint:
    """One segment of the road line (H5, 164 bytes): lanes lane_lo..lane_hi of the LANES slots are used; curve is the
    arcade's signed bend (s16)."""
    lane_lo: int
    lane_hi: int
    curve: int
    lanes: tuple[Lane, ...]


@dataclass(frozen=True)
class Cell:
    """One cell (H6, 64 bytes): a ground polygon, the arcade's TGP polygon. corners: four (x, y, z); centre: (x, y,
    z), the corners' mean on most cells; direction: the u32 last word (the low 16 bits an angle, 0x10000 a turn; the
    high half 0 on the disc), kept whole."""
    corners: tuple[tuple[float, float, float], ...]
    centre: tuple[float, float, float]
    direction: int


@dataclass(frozen=True)
class Step:
    """One step (H3, 4 bytes): the road cell of the same index lies on road segment `segment`; flags are the arcade's
    STEP_* bits (0x01 road, 0x42 pit lane, 0x82 pit box, 0x100 course 2's segments 174-232)."""
    segment: int
    flags: int


@dataclass(frozen=True)
class CourseData:
    """One course data file. `order`: the sections H0-H7 in file order (H0 first). `lines`: the 8 AI lines in
    directory order (H4 entry k names line k); `line_slots[k]`: where line k is stored in H0 (0 = first; the M files
    store them in another order). `road`: the road line (H5; segments = len(road) - 1). `attributes`: one byte per
    cell; `steps`: one per road cell (H3). `cells` (H6): the first len(steps) are the road. `grid` (H2): GRID_CELLS
    indices into `lists` (H7), row-major, row z; `lists[0]` is what an off-ground square (grid value 0) reaches.
    H1 is derived: (len(road) - 1, len(steps), the steps' offset)."""
    order: tuple[int, ...]
    lines: tuple[tuple[LinePoint, ...], ...]
    line_slots: tuple[int, ...]
    road: tuple[RoadPoint, ...]
    attributes: bytes
    steps: tuple[Step, ...]
    cells: tuple[Cell, ...]
    grid: tuple[int, ...]
    lists: tuple[tuple[int, ...], ...]


def _align(pos: int, a: int) -> int:
    return pos + (-pos % a)


def road_cells(d: CourseData) -> tuple[Cell, ...]:
    """The road's cells: the first len(d.steps) (H1's second word), in step order. DcError for a `d` that is not a
    CourseData, whose cells or steps are not sequences, or with more steps than cells."""
    if not isinstance(d, CourseData):
        raise DcError(f"{type(d).__name__} is not a CourseData")
    cells = _seq(d.cells, None, "cells", "course data", DcError)
    steps = _seq(d.steps, None, "steps", "course data", DcError)
    if len(steps) > len(cells):
        raise DcError(f"course data: {len(steps)} steps, more than the {len(cells)} cells (a step is a road cell)")
    return cells[:len(steps)]


def _short(v) -> str:
    r = repr(v)
    return r if len(r) <= 40 else r[:37] + "..."


def grid_square(x: float, z: float) -> int:
    """The grid index (row * 128 + column) of world (x, z); wraps outside -1024..1024 like the arcade's. DcError for
    a coordinate that is not a finite number (bool refused)."""
    if not (is_number(x) and is_number(z)):      # is_number refuses an int past the float range (10**400)
        raise DcError(f"grid square: ({_short(x)}, {_short(z)}) is not two finite numbers")
    fx, fz = float(x), float(z)
    c = int(math.floor((fx - GRID_ORIGIN) / GRID_SQUARE)) % GRID_SIZE
    r = int(math.floor((fz - GRID_ORIGIN) / GRID_SQUARE)) % GRID_SIZE
    return r * GRID_SIZE + c


# ---- decode ----

def decode_data(blob: bytes, name: str = "") -> CourseData:
    """The course data in `blob`. DcError (naming `name` and the offset) for a file that is truncated, whose offsets,
    counts or indices point past their section or table, whose pads are not zero, or whose floats are not finite."""
    if not isinstance(name, str):
        raise DcError(f"course data: the name {type(name).__name__} is not a str")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise DcError(f"{name or 'course data'}: {type(blob).__name__} is not bytes")
    b = bytes(blob)
    who = name or "course data"
    size = len(b)
    isfinite = math.isfinite

    if size < HEADER_SIZE:
        raise DcError(f"{who}: the header at offset 0x0 needs {HEADER_SIZE:#x} bytes, the file is {size:#x}")
    hdr = struct.unpack_from("<8I", b, 0)
    if hdr[0] != 0:
        raise DcError(f"{who}: H0's word at offset 0x0 is {hdr[0]:#x}, expected 0 (the AI lines start at 0x20)")
    start = (HEADER_SIZE,) + hdr[1:]
    for i in range(1, SECTIONS):
        if not HEADER_SIZE < start[i] <= size:
            raise DcError(f"{who}: {SECTION_NAMES[i]} at offset {start[i]:#x} (header word {i}) is outside the file "
                          f"({HEADER_SIZE:#x}..{size:#x})")
        if start[i] % ALIGN[i]:
            raise DcError(f"{who}: {SECTION_NAMES[i]} at offset {start[i]:#x} is not {ALIGN[i]}-aligned")
    order = tuple(sorted(range(SECTIONS), key=lambda i: start[i]))
    end = {}
    for j, i in enumerate(order):
        end[i] = start[order[j + 1]] if j + 1 < SECTIONS else size
        if j + 1 < SECTIONS and end[i] == start[i]:
            raise DcError(f"{who}: {SECTION_NAMES[i]} and {SECTION_NAMES[order[j + 1]]} both start at offset "
                          f"{start[i]:#x}")

    def need(i: int, off: int, n: int, what: str) -> None:
        if off + n > end[i]:
            raise DcError(f"{who}: {what} at offset {off:#x} needs {n} bytes, {SECTION_NAMES[i]} ends at {end[i]:#x}")

    def finite(vals: tuple, off: int, what: str) -> None:
        if not all(isfinite(v) for v in vals):
            raise DcError(f"{who}: {what} at offset {off:#x} is not finite: {vals!r}")

    used = {}                                   # section -> its content's size (the rest to the next is the pad)

    # H1: counts
    need(1, start[1], COUNTS_SIZE, "H1 (counts)")
    nseg, nsteps, steps_at = struct.unpack_from("<3I", b, start[1])
    used[1] = COUNTS_SIZE

    # H6: cells, as many whole cells as the section holds
    ncell, rem = divmod(end[6] - start[6], CELL_SIZE)
    if rem:
        raise DcError(f"{who}: a truncated cell at offset {start[6] + CELL_SIZE * ncell:#x}: {rem} bytes before "
                      f"{end[6]:#x}, a cell is {CELL_SIZE}")
    cells = []
    for k in range(ncell):
        off = start[6] + CELL_SIZE * k
        v = _CELL.unpack_from(b, off)
        finite(v[:15], off, f"cell {k}")
        cells.append(Cell((v[0:3], v[3:6], v[6:9], v[9:12]), v[12:15], v[15]))
    used[6] = CELL_SIZE * ncell

    # H3: attributes, pad, steps
    want = start[3] + ncell + ncell % 2
    if steps_at != want:
        raise DcError(f"{who}: H1 word 2 at offset {start[1] + 8:#x} puts the steps at {steps_at:#x}; the {ncell} "
                      f"attribute bytes (cells) at {start[3]:#x} put them at {want:#x}")
    need(3, start[3], ncell + ncell % 2, f"the attributes of {ncell} cells")
    attributes = b[start[3]:start[3] + ncell]
    if ncell % 2 and b[start[3] + ncell]:
        raise DcError(f"{who}: the attribute pad at offset {start[3] + ncell:#x} is not zero")
    if nsteps > ncell:
        raise DcError(f"{who}: H1 word 1 at offset {start[1] + 4:#x}: {nsteps} steps, more than the {ncell} cells")
    need(3, steps_at, STEP_SIZE * nsteps, f"{nsteps} steps")
    nroad = nseg + 1
    steps = []
    for k in range(nsteps):
        seg, flags = struct.unpack_from("<HH", b, steps_at + STEP_SIZE * k)
        if seg >= nroad:
            raise DcError(f"{who}: step {k} at offset {steps_at + STEP_SIZE * k:#x}: segment {seg} is past the "
                          f"{nroad} road points")
        steps.append(Step(seg, flags))
    used[3] = steps_at - start[3] + STEP_SIZE * nsteps

    # H5: the road line, segments + 1 points
    need(5, start[5], ROAD_POINT_SIZE * nroad, f"the road line of {nroad} points (H1 word 0 at {start[1]:#x})")
    road = []
    for k in range(nroad):
        off = start[5] + ROAD_POINT_SIZE * k
        lo, hi, curve = _ROAD_HEAD.unpack_from(b, off)
        if not lo <= hi < LANES:
            raise DcError(f"{who}: road point {k} at offset {off:#x}: lanes {lo}..{hi} are not inside 0..{LANES - 1}")
        lanes = []
        for j in range(LANES):
            lo_off = off + 4 + LANE_SIZE * j
            v = _LANE.unpack_from(b, lo_off)
            finite(v[:3], lo_off, f"road point {k} lane {j}")
            lanes.append(Lane(*v))
        road.append(RoadPoint(lo, hi, curve, tuple(lanes)))
    used[5] = ROAD_POINT_SIZE * nroad

    # H4 and H0: the directory, the lines stored back to back from 0x20, each with its wrap
    need(4, start[4], DIRECTORY_SIZE, "H4 (line directory)")
    directory = [struct.unpack_from("<2I", b, start[4] + 8 * k) for k in range(LINE_COUNT)]
    by_offset = sorted(range(LINE_COUNT), key=lambda k: directory[k][0])
    pos = HEADER_SIZE
    slots = [0] * LINE_COUNT
    for slot, k in enumerate(by_offset):
        off, count = directory[k]
        entry = start[4] + 8 * k
        if off != pos:
            raise DcError(f"{who}: line {k} (directory entry at offset {entry:#x}) starts at {off:#x}; the stored "
                          f"lines put the next one at {pos:#x}")
        if count < LINE_WRAP:
            raise DcError(f"{who}: line {k} (directory entry at offset {entry:#x}) has {count} points, at least "
                          f"{LINE_WRAP} are needed (the wrap repeats them)")
        if off + LINE_POINT_SIZE * (count + LINE_WRAP) > end[0]:
            raise DcError(f"{who}: line {k} at offset {off:#x}: {count} points and the wrap run past "
                          f"{SECTION_NAMES[0]} (ends {end[0]:#x}; directory entry at offset {entry:#x})")
        slots[k] = slot
        pos = off + LINE_POINT_SIZE * (count + LINE_WRAP)
    used[0] = pos - HEADER_SIZE
    used[4] = DIRECTORY_SIZE
    lines = []
    for k in range(LINE_COUNT):
        off, count = directory[k]
        wrap = off + LINE_POINT_SIZE * count
        if b[wrap:wrap + LINE_POINT_SIZE * LINE_WRAP] != b[off:off + LINE_POINT_SIZE * LINE_WRAP]:
            raise DcError(f"{who}: line {k}: the wrap at offset {wrap:#x} is not the line's first {LINE_WRAP} points")
        pts = []
        for i in range(count):
            at = off + LINE_POINT_SIZE * i
            v = _LINE_POINT.unpack_from(b, at)
            finite(v[:4], at, f"line {k} point {i}")
            if v[9] >= nsteps:
                raise DcError(f"{who}: line {k} point {i} at offset {at:#x}: step {v[9]} is past the {nsteps} steps")
            pts.append(LinePoint(*v))
        lines.append(tuple(pts))

    # H7: the lists, to the end of the section
    pos = start[7]
    list_at = {}
    lists = []
    while pos < end[7]:
        if pos + 2 > end[7]:
            raise DcError(f"{who}: a list at offset {pos:#x} runs past {SECTION_NAMES[7]} (ends {end[7]:#x})")
        (count,) = struct.unpack_from("<H", b, pos)
        if pos + 2 + 2 * count > end[7]:
            raise DcError(f"{who}: list {len(lists)} at offset {pos:#x}: {count} entries run past "
                          f"{SECTION_NAMES[7]} (ends {end[7]:#x})")
        entries = struct.unpack_from(f"<{count}H", b, pos + 2)
        for e in entries:
            if e >= ncell:
                raise DcError(f"{who}: list {len(lists)} at offset {pos:#x}: cell {e} is past the {ncell} cells")
        list_at[(pos - start[7]) // 2] = len(lists)
        lists.append(entries)
        pos += 2 + 2 * count
    used[7] = pos - start[7]

    # H2: the grid
    need(2, start[2], GRID_BYTES, f"{SECTION_NAMES[2]} ({GRID_SIZE} x {GRID_SIZE} u16)")
    grid = []
    for k, g in enumerate(struct.unpack_from(f"<{GRID_CELLS}H", b, start[2])):
        idx = list_at.get(g)
        if idx is None:
            raise DcError(f"{who}: grid square (row {k // GRID_SIZE}, column {k % GRID_SIZE}) at offset "
                          f"{start[2] + 2 * k:#x}: value {g} is not the start of a list in {SECTION_NAMES[7]}")
        grid.append(idx)
    used[2] = GRID_BYTES

    # the pads: each section's end rounded up to the next one's alignment, zero; nothing after the last
    for j, i in enumerate(order):
        stop = start[i] + used[i]
        if j + 1 == SECTIONS:
            if stop != size:
                raise DcError(f"{who}: {size - stop} bytes at offset {stop:#x} after {SECTION_NAMES[i]}, before the "
                              f"end of the file at {size:#x} (a truncated or extra record)")
            continue
        nxt = order[j + 1]
        if _align(stop, ALIGN[nxt]) != start[nxt]:
            raise DcError(f"{who}: {SECTION_NAMES[i]} ends at offset {stop:#x}; {SECTION_NAMES[nxt]} starts at "
                          f"{start[nxt]:#x}, not at the next {ALIGN[nxt]}-byte boundary (a truncated record, or a "
                          f"count that disagrees with the section)")
        if any(b[stop:start[nxt]]):
            raise DcError(f"{who}: the pad at offset {stop:#x} before {SECTION_NAMES[nxt]} is not zero")

    return CourseData(order, tuple(lines), tuple(slots), tuple(road), attributes, tuple(steps), tuple(cells),
                      tuple(grid), tuple(lists))


# ---- encode ----

def _u(v, bits: int, name: str, what: str) -> int:
    return _int(v, name, 0, (1 << bits) - 1, what, DcError)


def _s16(v, name: str, what: str) -> int:
    return _int(v, name, -0x8000, 0x7FFF, what, DcError)


def _line(points, k: int, nsteps: int) -> bytes:
    what = f"course data: line {k}"
    points = _seq(points, None, "points", what, DcError)
    if len(points) < LINE_WRAP:
        raise DcError(f"{what}: {len(points)} points, at least {LINE_WRAP} are needed (the wrap repeats them)")
    out = bytearray()
    for i, p in enumerate(points):
        pw = f"course data: line {k} point {i}"
        if not isinstance(p, LinePoint):
            raise DcError(f"{pw}: {type(p).__name__} is not a LinePoint")
        step = _u(p.step, 16, "step", pw)
        if step >= nsteps:
            raise DcError(f"{pw}: step {step} is past the {nsteps} steps")
        out += _f32((p.x, p.y, p.z, p.speed), 4, "x, y, z, speed", pw, DcError)
        out += struct.pack("<5hH", _s16(p.pitch, "pitch", pw), _s16(p.heading, "heading", pw),
                           _s16(p.roll, "roll", pw), _s16(p.steer, "steer", pw), _s16(p.lane, "lane", pw), step)
    return bytes(out + out[:LINE_POINT_SIZE * LINE_WRAP])


def _road_point(r, k: int) -> bytes:
    what = f"course data: road point {k}"
    if not isinstance(r, RoadPoint):
        raise DcError(f"{what}: {type(r).__name__} is not a RoadPoint")
    lo, hi = _u(r.lane_lo, 8, "lane_lo", what), _u(r.lane_hi, 8, "lane_hi", what)
    if not lo <= hi < LANES:
        raise DcError(f"{what}: lanes {lo}..{hi} are not inside 0..{LANES - 1}")
    out = bytearray(_ROAD_HEAD.pack(lo, hi, _s16(r.curve, "curve", what)))
    for j, ln in enumerate(_seq(r.lanes, LANES, "lanes", what, DcError)):
        lw = f"{what} lane {j}"
        if not isinstance(ln, Lane):
            raise DcError(f"{lw}: {type(ln).__name__} is not a Lane")
        out += _f32((ln.x, ln.y, ln.z), 3, "x, y, z", lw, DcError)
        out += struct.pack("<3hH", _s16(ln.pitch, "pitch", lw), _s16(ln.heading, "heading", lw),
                           _s16(ln.roll, "roll", lw), _u(ln.pad, 16, "pad", lw))
    return bytes(out)


def _cell(c, k: int) -> bytes:
    what = f"course data: cell {k}"
    if not isinstance(c, Cell):
        raise DcError(f"{what}: {type(c).__name__} is not a Cell")
    corners = _seq(c.corners, 4, "corners", what, DcError)
    out = b"".join(_f32(p, 3, f"corner {j}", what, DcError) for j, p in enumerate(corners))
    return out + _f32(c.centre, 3, "centre", what, DcError) + struct.pack("<I", _u(c.direction, 32, "direction", what))


def encode_data(d: CourseData) -> bytes:
    """The file bytes of `d`. DcError for anything the format cannot hold or that disagrees: a non-finite or
    out-of-single-precision float, an integer outside its field, a step, cell, list or lane index past its table, a
    count that disagrees (attributes and cells, steps and cells, 8 lines of at least 8 points), a section order or
    line slots that are not a permutation (H0 first)."""
    if not isinstance(d, CourseData):
        raise DcError(f"{type(d).__name__} is not a CourseData")
    who = "course data"
    order = tuple(_u(i, 8, "section", f"{who} order") for i in _seq(d.order, SECTIONS, "order", who, DcError))
    if sorted(order) != list(range(SECTIONS)) or order[0] != 0:
        raise DcError(f"{who}: order {order!r} is not a permutation of the sections 0-7 with H0 first")
    lines = _seq(d.lines, None, "lines", who, DcError)
    if len(lines) != LINE_COUNT:
        raise DcError(f"{who}: {len(lines)} lines, the directory holds {LINE_COUNT} lines")
    slots = tuple(_u(s, 8, "slot", f"{who} line_slots")
                  for s in _seq(d.line_slots, LINE_COUNT, "line_slots", who, DcError))
    if sorted(slots) != list(range(LINE_COUNT)):
        raise DcError(f"{who}: line_slots {slots!r} is not a permutation of 0-{LINE_COUNT - 1}")
    road = _seq(d.road, None, "road", who, DcError)
    if not road:
        raise DcError(f"{who}: the road line needs at least one road point")
    cells = _seq(d.cells, None, "cells", who, DcError)
    ncell = len(cells)
    if not isinstance(d.attributes, (bytes, bytearray)):
        raise DcError(f"{who}: attributes {type(d.attributes).__name__} are not bytes")
    if len(d.attributes) != ncell:
        raise DcError(f"{who}: {len(d.attributes)} attributes for {ncell} cells")
    steps = _seq(d.steps, None, "steps", who, DcError)
    if len(steps) > ncell:
        raise DcError(f"{who}: {len(steps)} steps, more than the {ncell} cells (a step is a road cell)")

    # sections' bytes (H0 at 0x20; H1 and H4 need the offsets, filled in below)
    nsteps = len(steps)
    stored = [b""] * LINE_COUNT
    for k, pts in enumerate(lines):
        stored[slots[k]] = _line(pts, k, nsteps)
    line_at = []
    pos = HEADER_SIZE
    for blob in stored:
        line_at.append(pos)
        pos += len(blob)
    body = {0: b"".join(stored)}
    body[5] = b"".join(_road_point(r, k) for k, r in enumerate(road))
    body[6] = b"".join(_cell(c, k) for k, c in enumerate(cells))
    sb = bytearray()
    for k, s in enumerate(steps):
        if not isinstance(s, Step):
            raise DcError(f"{who}: step {k}: {type(s).__name__} is not a Step")
        seg = _u(s.segment, 16, "segment", f"{who}: step {k}")
        if seg >= len(road):
            raise DcError(f"{who}: step {k}: segment {seg} is past the {len(road)} road points")
        sb += struct.pack("<HH", seg, _u(s.flags, 16, "flags", f"{who}: step {k}"))
    body[3] = bytes(d.attributes) + bytes(ncell % 2) + bytes(sb)
    lb = bytearray()
    list_at = []
    for k, entries in enumerate(_seq(d.lists, None, "lists", who, DcError)):
        entries = _seq(entries, None, f"list {k}", who, DcError)
        if len(entries) > 0xFFFF:
            raise DcError(f"{who}: list {k}: {len(entries)} entries do not fit a u16 count")
        list_at.append(len(lb) // 2)
        lb += struct.pack("<H", len(entries))
        for e in entries:
            e = _u(e, 16, "entry", f"{who}: list {k}")
            if e >= ncell:
                raise DcError(f"{who}: list {k}: cell {e} is past the {ncell} cells")
            lb += struct.pack("<H", e)
    body[7] = bytes(lb)
    gb = bytearray()
    for k, g in enumerate(_seq(d.grid, GRID_CELLS, "grid", who, DcError)):
        gw = f"{who}: grid square {k}"
        g = _int(g, "list index", 0, 0xFFFFFFFF, gw, DcError)
        if g >= len(list_at):
            raise DcError(f"{gw}: list {g} is past the {len(list_at)} lists")
        if list_at[g] > 0xFFFF:
            raise DcError(f"{gw}: list {g} starts at u16 {list_at[g]:#x} of H7, past what a u16 grid value reaches")
        gb += struct.pack("<H", list_at[g])
    body[2] = bytes(gb)
    sizes = {0: len(body[0]), 1: COUNTS_SIZE, 2: GRID_BYTES, 3: len(body[3]), 4: DIRECTORY_SIZE, 5: len(body[5]),
             6: len(body[6]), 7: len(body[7])}

    # layout
    offs = {}
    pos = HEADER_SIZE
    for i in order:
        pos = _align(pos, ALIGN[i])
        offs[i] = pos
        pos += sizes[i]
    body[1] = struct.pack("<3I", len(road) - 1, nsteps, offs[3] + ncell + ncell % 2)
    body[4] = b"".join(struct.pack("<2I", line_at[slots[k]], len(lines[k])) for k in range(LINE_COUNT))
    out = bytearray(struct.pack("<8I", 0, *(offs[i] for i in range(1, SECTIONS))))
    for i in order:
        out += bytes(offs[i] - len(out))
        out += body[i]
    return bytes(out)


# ---- the disc ----

def data_names() -> list[str]:
    """The 32 course data file names, course by course, variants in VARIANTS order."""
    from dtools.dc.disc import COURSE_COUNT, VARIANTS, course_files
    return [course_files(n, v, "data") for n in range(COURSE_COUNT) for v in VARIANTS]


def verify_lossless_data(disc) -> list[str]:
    """One line per course data file that `encode_data(decode_data(file))` does not reproduce byte for byte, or that
    does not decode ([] when all 32 are lossless)."""
    out = []
    for name in data_names():
        blob = disc.read(name)
        try:
            back = encode_data(decode_data(blob, name))
        except DcError as exc:
            out.append(f"{name}: {exc}")
            continue
        if back != blob:
            k = next((k for k, (a, b) in enumerate(zip(blob, back)) if a != b), min(len(blob), len(back)))
            out.append(f"{name}: first difference at offset {k:#x} (file {len(blob):#x} bytes, encoded {len(back):#x})")
    return out
