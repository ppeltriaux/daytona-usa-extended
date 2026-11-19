"""The arcade's race tables from a Dreamcast course: the road line, the AI lines, the step table, the ground
attributes (dtools.dc.data.CourseData) and the checkpoint rows (1ST_READ.BIN, dtools.dc.exe.checkpoints).

The transform (Phase 5, exact on n 0-2 against the arcade's ids 0-2; docs/spike/2026-11-convert.md, "Race tables"):

- a position (x, y, z) becomes (x, y, -z): scale 1, no rotation or offset;
- a pitch is negated; a heading becomes the Dreamcast's - 0x1fff; both wrap to s16 (on n 3 the AI headings fit
  0x2000 by +0.92 units on average, 0.02 degrees: the converter still writes - 0x1fff on every course);
- the AI speed, roll, steer, lane and step, the road point's lane_lo, lane_hi and curve, a lane's roll and pad, the
  steps (segment, flags) and the attribute bytes are copied;
- the AI lines are the plain file's in directory order, which is the arcade's line order (the M / MR files' other
  storage order, `CourseData.line_slots`, is not used);
- step_n (the course record's word 1) is the road line's n, len(road) - 1, as `golden.check_course` wants;
- the checkpoint rows are the plain list's segments in race order, raw 0; the lap line is not a row;
- the per-segment flags (course.segment_flags) have no Dreamcast source: n + 1 zeros, the documented default
  (docs/spike/2026-11-convert.md, "Per-segment flags": Beginner's and Advanced's are all 0; Expert's 51 non-zero
  segments are not found on the disc).

Every table function validates the race tables of `d` together (`_check`): the five tables it gives agree with one
another (a line per road point, a step for every AI point, a segment on the road line, one attribute per cell with the
road's first) and fit the arcade's fields. ConvertError otherwise; a DcError from the Dreamcast readers passes
through."""

from __future__ import annotations

import dataclasses

from dtools.arcade import course as C
from dtools.arcade.tablebase import _f32, _int, _seq, is_int
from dtools.convert import ConvertError
from dtools.dc.data import CourseData, Lane, LinePoint, RoadPoint, Step
from dtools.dc.disc import COURSE_COUNT
from dtools.dc.exe import checkpoints as _dc_checkpoints

HEADING_OFFSET = 0x1FFF          # arcade heading = Dreamcast heading - HEADING_OFFSET (s16)
ROAD_KIND = 1                    # attribute & 15 of a road polygon (tgp; the step table's polygons)
_S16 = (-0x8000, 0x7FFF)
_U16 = (0, 0xFFFF)
_ORIGIN = (0.0, 0.0, 0.0)        # a checkpoint row's position placeholder (course.checkpoint_views recomputes it)


def _s16(v: int) -> int:
    return (v + 0x8000) % 0x10000 - 0x8000


def _angle(v, name: str, what: str) -> int:
    return _int(v, name, *_S16, what, ConvertError)


def _xyz(x, y, z, what: str) -> tuple[float, float, float]:
    """(x, y, -z) as Python floats; ConvertError unless three finite numbers that fit single precision."""
    _f32((x, y, z), 3, "x, y, z", what, ConvertError)
    return float(x), float(y), -float(z)


# ---- validation ----

def _check(d) -> None:
    """ConvertError unless `d` is a CourseData whose race tables the arcade can hold, consistent with each other:
    1..0xffff road points (a u16 segment index), each lane_lo <= lane_hi < LANES with LANES Lanes; 8 AI lines of
    len(road) points; 1..len(cells) steps, each a Step on the road line; every AI point's step < len(steps); one
    attribute byte per cell, the first len(steps) of kind road (& 15 == 1) and no other; every value inside its
    arcade field (s16 angles, curve, steer and lane; u16 step, flags and pad; finite single-precision floats)."""
    if not isinstance(d, CourseData):
        raise ConvertError(f"{type(d).__name__} is not a CourseData")
    who = "course data"
    road = _seq(d.road, None, "road", who, ConvertError)
    if not road:
        raise ConvertError(f"{who}: the road line has no road points")
    if len(road) >= 0x10000:
        raise ConvertError(f"{who}: {len(road)} road points, past the arcade's u16 segment index")
    for k, r in enumerate(road):
        what = f"{who}: road point {k}"
        if not isinstance(r, RoadPoint):
            raise ConvertError(f"{what}: {type(r).__name__} is not a RoadPoint")
        lo = _int(r.lane_lo, "lane_lo", 0, C.LANES - 1, what, ConvertError)
        hi = _int(r.lane_hi, "lane_hi", 0, C.LANES - 1, what, ConvertError)
        if lo > hi:
            raise ConvertError(f"{what}: lanes {lo}..{hi} are not lo <= hi")
        _int(r.curve, "curve", *_S16, what, ConvertError)
        for j, ln in enumerate(_seq(r.lanes, C.LANES, "lanes", what, ConvertError)):
            lw = f"{what} lane {j}"
            if not isinstance(ln, Lane):
                raise ConvertError(f"{lw}: {type(ln).__name__} is not a Lane")
            _xyz(ln.x, ln.y, ln.z, lw)
            for f in ("pitch", "heading", "roll"):
                _angle(getattr(ln, f), f, lw)
            _int(ln.pad, "pad", *_U16, lw, ConvertError)

    cells = _seq(d.cells, None, "cells", who, ConvertError)
    steps = _seq(d.steps, None, "steps", who, ConvertError)
    if not steps:
        raise ConvertError(f"{who}: no steps (the AI points index the step table)")
    if len(steps) > len(cells):
        raise ConvertError(f"{who}: {len(steps)} steps, more than the {len(cells)} cells (a step is a road cell)")
    for k, s in enumerate(steps):
        what = f"{who}: step {k}"
        if not isinstance(s, Step):
            raise ConvertError(f"{what}: {type(s).__name__} is not a Step")
        _int(s.segment, "segment", 0, len(road) - 1, f"{what} (on {len(road)} road points)", ConvertError)
        _int(s.flags, "flags", *_U16, what, ConvertError)

    lines = _seq(d.lines, None, "lines", who, ConvertError)
    if len(lines) != C.AI_LINE_COUNT:
        raise ConvertError(f"{who}: {len(lines)} lines, the arcade's line table holds {C.AI_LINE_COUNT}")
    for k, line in enumerate(lines):
        points = _seq(line, None, f"line {k}", who, ConvertError)
        if len(points) != len(road):
            raise ConvertError(f"{who}: line {k} has {len(points)} points, the road line {len(road)} road points "
                               f"(a car's segment indexes both)")
        for i, p in enumerate(points):
            what = f"{who}: line {k} point {i}"
            if not isinstance(p, LinePoint):
                raise ConvertError(f"{what}: {type(p).__name__} is not a LinePoint")
            _xyz(p.x, p.y, p.z, what)
            _f32((p.speed,), 1, "speed", what, ConvertError)
            for f in ("pitch", "heading", "roll", "steer", "lane"):
                _angle(getattr(p, f), f, what)
            _int(p.step, "step", 0, len(steps) - 1, f"{what} (of {len(steps)} steps)", ConvertError)

    attrs = d.attributes
    if not isinstance(attrs, (bytes, bytearray)):
        raise ConvertError(f"{who}: attributes {type(attrs).__name__} are not bytes")
    if len(attrs) != len(cells):
        raise ConvertError(f"{who}: {len(attrs)} attributes for {len(cells)} cells (one per ground polygon)")
    for k, a in enumerate(attrs):
        if (a & 15 == ROAD_KIND) != (k < len(steps)):
            raise ConvertError(f"{who}: attribute {k} is {a:#x}: the polygons of kind road (& 15 == {ROAD_KIND}) "
                               f"must be the first {len(steps)}, one per step")


# ---- the tables ----

def _empty(ln) -> bool:
    """An unused lane slot: position (0, 0, 0) (the disc's (0, 0, -0.0))."""
    return (ln.x, ln.y, ln.z) == (0.0, 0.0, 0.0)


def lane_ranges_fixed(d: CourseData) -> dict[int, tuple[int, int]]:
    """{road point index: (lane_lo, lane_hi)} for every road point whose used range names an empty slot: its range
    mirrored, (LANES - 1 - lane_hi, LANES - 1 - lane_lo), which must be exactly its non-empty slots. 1R's road
    points hold the plain lanes mirrored (lane j the plain lane LANES - 1 - j) but keep the plain range where it is
    not symmetric; every other file gives {}. ConvertError for a `d` that `_check` refuses, and for a range naming
    an empty slot whose mirror is not exactly the non-empty slots."""
    _check(d)
    out = {}
    for k, r in enumerate(d.road):
        lo, hi = int(r.lane_lo), int(r.lane_hi)
        if not any(_empty(r.lanes[j]) for j in range(lo, hi + 1)):
            continue
        mlo, mhi = C.LANES - 1 - hi, C.LANES - 1 - lo
        if any(_empty(ln) == (mlo <= j <= mhi) for j, ln in enumerate(r.lanes)):
            raise ConvertError(f"course data: road point {k}: lanes {lo}..{hi} name an empty slot, and the mirrored "
                               f"range {mlo}..{mhi} is not exactly its non-empty slots")
        out[k] = (mlo, mhi)
    return out


def with_lane_ranges_fixed(d: CourseData) -> CourseData:
    """`d` with lane_ranges_fixed's ranges in its road points (nothing else changed); `d` itself when there is none.
    ConvertError as lane_ranges_fixed."""
    fixed = lane_ranges_fixed(d)
    if not fixed:
        return d
    road = tuple(dataclasses.replace(r, lane_lo=fixed[k][0], lane_hi=fixed[k][1]) if k in fixed else r
                 for k, r in enumerate(d.road))
    return dataclasses.replace(d, road=road)


def road_line(d: CourseData) -> tuple[C.RoadPoint, ...]:
    """The arcade's road line (record table) of `d`: per road point lane_lo, lane_hi and curve copied, each lane's
    position (x, y, -z), pitch negated, heading - 0x1fff (s16), roll and pad copied. ConvertError for a `d` that
    `_check` refuses."""
    _check(d)
    return tuple(C.RoadPoint(int(r.lane_lo), int(r.lane_hi), int(r.curve),
                             tuple(C.Lane(_xyz(ln.x, ln.y, ln.z, "lane"), _s16(-int(ln.pitch)),
                                          _s16(int(ln.heading) - HEADING_OFFSET), int(ln.roll), int(ln.pad))
                                   for ln in r.lanes))
                 for r in d.road)


def ai_lines(d: CourseData) -> tuple[tuple[C.AiPoint, ...], ...]:
    """The arcade's eight AI lines of `d`, in the arcade's line order (the plain file's directory order): each
    point's position (x, y, -z), pitch negated, heading - 0x1fff (s16), speed, roll, steer, lane and step copied.
    ConvertError for a `d` that `_check` refuses."""
    _check(d)
    return tuple(tuple(C.AiPoint(_xyz(p.x, p.y, p.z, "AI point"), float(p.speed), _s16(-int(p.pitch)),
                                 _s16(int(p.heading) - HEADING_OFFSET), int(p.roll), int(p.steer), int(p.lane),
                                 int(p.step))
                       for p in line)
                 for line in d.lines)


def steps(d: CourseData) -> tuple[int, tuple[C.Step, ...]]:
    """(step_n, steps): the course record's word 1, the road line's n (len(road) - 1), and the step table, each
    step's segment and flags copied. ConvertError for a `d` that `_check` refuses."""
    _check(d)
    return len(d.road) - 1, tuple(C.Step(int(s.segment), int(s.flags)) for s in d.steps)


def segment_flags(d: CourseData) -> tuple[int, ...]:
    """The per-segment flags (course.segment_flags): one u16 per road point, all 0. The Dreamcast data holds no
    per-segment flags (no section of CS_DATAn.BIN; the arcade's values are in no file of the disc), so this is the
    documented default, exact for Beginner and Advanced (all 0 on the arcade) and missing Expert's SEGMENT_AI_FREE
    (segments 190-240) and SEGMENT_TUNNEL (208-218): docs/spike/2026-11-convert.md, "Per-segment flags" (Not
    found). A course's tunnel bit, where the Dreamcast keeps one, is in its step flags (Advanced's 0x101), which
    `steps` copies. ConvertError for a `d` that `_check` refuses."""
    _check(d)
    return (0,) * len(d.road)


def attributes(d: CourseData) -> bytes:
    """The ground's attribute bytes, one per polygon (the cells' order; the road's polygons first, one per step),
    copied. ConvertError for a `d` that `_check` refuses."""
    _check(d)
    return bytes(d.attributes)


def checkpoint_rows(exe, n: int, v: str = "") -> tuple[C.Checkpoint, ...]:
    """The rows the arcade's checkpoint table writes for Dreamcast course `n` (0-7), variant `v` (default the plain
    course; "R" its reverse, M9), from `exe` (the bytes of 1ST_READ.BIN; dtools.dc.exe.checkpoints): Checkpoint(index
    j, segment, raw 0, not the lap line), in race order. `position` (0, 0, 0) and `extensions` () are placeholders: they are views of the road line and the
    time table, which course.checkpoint_views recomputes (and golden.check_course compares); the lap line is added
    there too. () for a course without checkpoints (n 0 and 5). ConvertError for an n that is not an integer 0-7, a
    segment past the arcade's u16 or more than course.MAX_CHECKPOINTS rows; DcError for an `exe` the reader refuses."""
    if not is_int(n) or not 0 <= n < COURSE_COUNT:
        raise ConvertError(f"checkpoint rows: course {n!r} is not a Dreamcast course 0..{COURSE_COUNT - 1}")
    segs = _dc_checkpoints(exe, int(n), v)
    if len(segs) > C.MAX_CHECKPOINTS:
        raise ConvertError(f"checkpoint rows: course {int(n)} has {len(segs)} checkpoints, more than the arcade's "
                           f"{C.MAX_CHECKPOINTS} (its u32 bit set)")
    return tuple(C.Checkpoint(j, _int(s, f"checkpoint {j}: segment", *_U16, f"checkpoint rows: course {int(n)}",
                                      ConvertError), 0, False, _ORIGIN, ())
                 for j, s in enumerate(segs))
