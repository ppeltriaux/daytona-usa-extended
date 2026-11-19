"""Tests for the race tables converted from the Dreamcast course data (dtools/convert/tables.py).

The known answer: the Dreamcast's courses n 0-2 are the arcade's Beginner, Expert and Advanced, so their converted
tables must equal the arcade's own field for field (Review Focus 2: a sign or field-order slip passes every round
trip, not this)."""

import dataclasses
import math
import random
import struct

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade.errors import DecodeError
from dtools.convert import ConvertError, tables
from dtools.dc.data import CourseData, Lane, LinePoint, Step
from dtools.dc.errors import DcError
from dtools.dc.exe import CheckpointTable
from tests.test_dc_data import make_data
from tests.test_dc_exe import ENTRIES, LISTS, _exe

FUNCS = (tables.road_line, tables.ai_lines, tables.steps, tables.attributes)


# ---- synthetic ----

def test_convert_error_is_a_decode_error():
    assert issubclass(ConvertError, DecodeError) and not issubclass(ConvertError, DcError)


def test_tables_synthetic():
    """The transform on make_data's values: (x, y, z) -> (x, y, -z), pitch negated, heading - 0x1fff (s16), the rest
    copied; the steps, step_n (= road points - 1) and the attributes as they are."""
    d = make_data()
    road = tables.road_line(d)
    assert len(road) == 9 and all(isinstance(p, C.RoadPoint) for p in road)
    p = road[2]
    assert (p.lane_lo, p.lane_hi, p.curve) == (1, 6, 0)
    assert p.lanes[3] == C.Lane((2.0 + 9.0, 0.0, 2.0), 0, 0x0AA9 - 0x1FFF, 0, 0)
    assert road[1].curve == 0x1111
    lines = tables.ai_lines(d)
    assert len(lines) == 8 and all(len(line) == 9 for line in lines)
    assert lines[5][4] == C.AiPoint((4.0, 2.5, 4.0), 1.25, 27, 0x0AAA - 0x1FFF, 4, -2, 502 + 5, 1)
    assert all(type(v) is float for v in lines[5][4].position) and type(lines[5][4].speed) is float
    assert tables.steps(d) == (8, (C.Step(0, 1), C.Step(1, 0x42), C.Step(8, 0x82)))
    assert tables.attributes(d) == bytes([1, 1, 1, 5, 5])
    # the wrap of the s16 angles: -(-0x8000) and -0x8000 - 0x1fff wrap round a turn
    pt = dataclasses.replace(d.lines[0][0], pitch=-0x8000, heading=-0x8000)
    ln = dataclasses.replace(d.road[0].lanes[1], pitch=-0x8000, heading=-0x7000)
    rp = dataclasses.replace(d.road[0], lanes=(d.road[0].lanes[0], ln) + d.road[0].lanes[2:])
    d2 = dataclasses.replace(d, lines=((pt,) + d.lines[0][1:],) + d.lines[1:], road=(rp,) + d.road[1:])
    a = tables.ai_lines(d2)[0][0]
    assert (a.pitch, a.heading) == (-0x8000, 0x6001)
    assert (tables.road_line(d2)[0].lanes[1].pitch, tables.road_line(d2)[0].lanes[1].heading) == (-0x8000, 0x7001)
    # numpy values become Python ones; the arcade's encoders take the result
    pn = dataclasses.replace(d.lines[0][0], x=np.float32(1.5), pitch=np.int16(3), step=np.uint16(2))
    a = tables.ai_lines(dataclasses.replace(d, lines=((pn,) + d.lines[0][1:],) + d.lines[1:]))[0][0]
    assert type(a.position[0]) is float and type(a.pitch) is int and (a.pitch, a.step) == (-3, 2)
    C.encode_road_line(list(road))
    C.encode_line_table(0, [list(line) for line in lines])


def test_tables_refuse():
    """A CourseData the arcade's tables cannot hold is refused with ConvertError, by every table function (each
    validates the race tables together, so the tables it returns agree with one another)."""
    d = make_data()
    line0 = d.lines[0]
    lane = d.road[0].lanes[0]

    def road_with(**kw):
        return (dataclasses.replace(d.road[0], **kw),) + d.road[1:]

    def lanes_with(**kw):
        return road_with(lanes=(dataclasses.replace(lane, **kw),) + d.road[0].lanes[1:])

    def line_with(**kw):
        return ((dataclasses.replace(line0[0], **kw),) + line0[1:],) + d.lines[1:]

    cases = [
        (dict(lines=d.lines[:7]), "7 lines"),
        (dict(lines=d.lines + d.lines[:1]), "9 lines"),
        (dict(lines=None), "lines"),
        (dict(lines=(line0[:8],) + d.lines[1:]), "line 0.*8 points.*9 road points"),
        (dict(lines=(line0 + (line0[0],),) + d.lines[1:]), "line 0"),
        (dict(lines=(line0[:3] + (None,) + line0[4:],) + d.lines[1:]), "line 0 point 3.*LinePoint"),
        (dict(steps=d.steps[:2] + (Step(9, 0x82),)), "step 2.*9 road points.*segment 9"),
        (dict(steps=d.steps[:2] + (Step(-1, 0x82),)), "step 2.*segment"),
        (dict(steps=d.steps[:2] + (Step(8, 0x10000),)), "step 2.*flags"),
        (dict(steps=d.steps[:2] + ((8, 0x82),)), "step 2.*Step"),
        (dict(steps=()), "steps"),
        (dict(steps=d.steps * 2), "6 steps.*5 cells"),
        (dict(road=lanes_with(x=float("nan"))), "road point 0 lane 0.*finite"),
        (dict(road=lanes_with(z=float("inf"))), "road point 0 lane 0.*finite"),
        (dict(road=lanes_with(y=1e39)), "road point 0 lane 0.*single precision"),
        (dict(road=lanes_with(heading=0x8000)), "road point 0 lane 0.*heading"),
        (dict(road=lanes_with(pitch=True)), "road point 0 lane 0.*pitch"),
        (dict(road=lanes_with(pad=-1)), "road point 0 lane 0.*pad"),
        (dict(road=road_with(lane_lo=7, lane_hi=6)), "road point 0.*lanes 7..6"),
        (dict(road=road_with(lane_hi=8)), "road point 0.*lane"),
        (dict(road=road_with(curve=0x8000)), "road point 0.*curve"),
        (dict(road=road_with(lanes=d.road[0].lanes[:7])), "road point 0.*lanes"),
        (dict(road=road_with(lanes=(None,) + d.road[0].lanes[1:])), "road point 0 lane 0.*Lane"),
        (dict(road=(None,) + d.road[1:]), "road point 0.*RoadPoint"),
        (dict(road=()), "road"),
        (dict(lines=line_with(x=float("nan"))), "line 0 point 0.*finite"),
        (dict(lines=line_with(speed=float("inf"))), "line 0 point 0.*finite"),
        (dict(lines=line_with(heading=-0x8001)), "line 0 point 0.*heading"),
        (dict(lines=line_with(lane=1.0)), "line 0 point 0.*lane"),
        (dict(lines=line_with(steer=np.bool_(True))), "line 0 point 0.*steer"),
        (dict(lines=line_with(step=3)), "line 0 point 0.*3 steps.*step 3"),
        (dict(attributes=d.attributes[:4]), "4 attributes.*5 cells"),
        (dict(attributes=[1, 1, 1, 5, 5]), "attributes"),
        (dict(attributes=bytes([1, 1, 5, 5, 5])), "attribute.*2.*road"),
        (dict(attributes=bytes([1, 1, 1, 0x11, 5])), "attribute.*3.*road"),
        (dict(cells=None), "cells"),
    ]
    for kw, match in cases:
        bad = dataclasses.replace(d, **kw)
        for f in FUNCS:
            with pytest.raises(ConvertError, match=match):
                f(bad)
    # a road line past the arcade's u16 segment index (checked before the points are read)
    many = dataclasses.replace(d, road=d.road[:1] * 0x10000)
    for f in FUNCS:
        with pytest.raises(ConvertError, match="u16"):
            f(many)
    for bad in (None, "CS_DATA3.BIN", b"", d.lines, dataclasses.astuple(d)):
        for f in FUNCS:
            with pytest.raises(ConvertError, match="CourseData"):
                f(bad)


def test_tables_never_raise_other_errors():
    """Fuzz: a make_data with fields swapped for junk is converted or refused with ConvertError, never another
    error; whatever converts, the arcade's encoders take."""
    d = make_data()
    rng = random.Random(11)
    junk = (None, True, np.bool_(False), -1, 0x10000, 0x8000, -0x8001, 10**400, float("nan"), float("inf"), 1e39,
            "x", 1.5, (), (1,), np.int64(7), np.float32(2.0))
    converted = 0
    for _ in range(3000):
        lines = [list(line) for line in d.lines]
        lanes = [list(r.lanes) for r in d.road]
        heads = [{} for _ in d.road]
        steps, top = list(d.steps), {}
        for _ in range(rng.randint(1, 3)):
            where = rng.random()
            v = rng.choice(junk)
            if where < 0.35:
                k, i = rng.randrange(8), rng.randrange(9)
                f = rng.choice([f.name for f in dataclasses.fields(LinePoint)])
                lines[k][i] = dataclasses.replace(lines[k][i], **{f: v})
            elif where < 0.6:
                i, j = rng.randrange(9), rng.randrange(8)
                f = rng.choice([f.name for f in dataclasses.fields(Lane)])
                lanes[i][j] = dataclasses.replace(lanes[i][j], **{f: v})
            elif where < 0.75:
                i = rng.randrange(9)
                f = rng.choice(("lane_lo", "lane_hi", "curve", "lanes"))
                heads[i][f] = v
            elif where < 0.9:
                i = rng.randrange(3)
                steps[i] = dataclasses.replace(steps[i], **{rng.choice(("segment", "flags")): v})
            else:
                top[rng.choice(("lines", "road", "steps", "attributes", "cells"))] = v
        road_pts = tuple(dataclasses.replace(r, **{"lanes": tuple(ls), **h}) for r, ls, h in zip(d.road, lanes, heads))
        bad = dataclasses.replace(d, **{"lines": tuple(map(tuple, lines)), "road": road_pts, "steps": tuple(steps),
                                        **top})
        got = []
        for f in FUNCS:
            try:
                got.append(f(bad))
            except ConvertError:
                got.append(None)
        assert len({g is None for g in got}) == 1           # one validation: all convert or all refuse
        if got[0] is None:
            continue
        converted += 1
        C.encode_road_line(list(got[0]))
        C.encode_line_table(0, [list(line) for line in got[1]])
    assert converted > 100                  # some junk is legal (a numpy int, 1.5 as a coordinate)


def test_checkpoint_rows_synthetic():
    """The rows the arcade's checkpoint table writes: index, segment, raw 0, not the lap line; position and
    extensions are placeholders that course.checkpoint_views recomputes."""
    blob = _exe(CheckpointTable(LISTS, ENTRIES))
    assert tables.checkpoint_rows(blob, 0) == ()
    rows = tables.checkpoint_rows(blob, 1)
    assert rows == tuple(C.Checkpoint(j, s, 0, False, (0.0, 0.0, 0.0), ()) for j, s in enumerate(LISTS[1]))
    assert [type(r.segment) for r in rows] == [int] * 5
    assert C.encode_checkpoint_table(list(rows)) == struct.pack("<6I", 5, *LISTS[1])
    assert tables.checkpoint_rows(memoryview(blob), np.int64(3)) == tables.checkpoint_rows(blob, 3)


def test_checkpoint_rows_refuse():
    blob = _exe(CheckpointTable(LISTS, ENTRIES))
    for n in (-1, 8, True, np.bool_(False), 3.0, "3", None):
        with pytest.raises(ConvertError, match="course"):
            tables.checkpoint_rows(blob, n)
    # a segment past the arcade's u16; more checkpoints than the arcade's bit set holds
    wide = _exe(CheckpointTable(((0x10000, 5),) + LISTS[1:], ENTRIES))
    with pytest.raises(ConvertError, match="segment 65536"):
        tables.checkpoint_rows(wide, 0)
    many = _exe(CheckpointTable((tuple(range(1, 32)),) + LISTS[1:], ENTRIES))
    with pytest.raises(ConvertError, match="31 checkpoints"):
        tables.checkpoint_rows(many, 0)
    assert len(tables.checkpoint_rows(_exe(CheckpointTable((tuple(range(1, 31)),) + LISTS[1:], ENTRIES)), 0)) == 30
    # a file the Dreamcast reader refuses: its DcError passes through (a DecodeError too)
    for bad in (b"", "1ST_READ.BIN", None):
        with pytest.raises(DcError):
            tables.checkpoint_rows(bad, 1)


# ---- the disc against the arcade ----

@pytest.fixture(scope="module")
def disc(request):
    from dtools import config
    from dtools.dc.disc import DcDisc
    p = config.dc_path()
    if p is None or not p.is_file():
        pytest.skip("set DAYTONA_DC to the Dreamcast disc .cue or .7z")
    d = DcDisc.open(p)
    request.addfinalizer(d.close)
    return d


def _data(disc, n: int) -> CourseData:
    from dtools.dc.data import decode_data
    from dtools.dc.disc import course_files
    name = course_files(n, kind="data")
    return decode_data(disc.read(name), name)


def _arcade(romset_zip, ids):
    from dtools.arcade import golden
    from dtools.arcade.romset import RomSet
    rs = RomSet.from_zip(romset_zip)
    return [golden.decode_course(rs, i) for i in ids]


def _same_tables(d: CourseData, c) -> int:
    """Review Focus 2: the converted race tables equal the arcade course `c`'s field for field (the AI speed within
    6e-6, as Phase 5 measured; everything else exactly, the road line byte for byte). Returns the AI coordinates whose
    f32 bytes differ (a zero of the other sign: equal as values)."""
    road = tables.road_line(d)
    assert road == tuple(c.road)
    assert C.encode_road_line(list(road)) == C.encode_road_line(list(c.road))       # the zeros' signs too
    lines = tables.ai_lines(d)                          # the plain file's directory order is the arcade's
    assert len(lines) == len(c.ai_lines) == 8
    worst, signs = 0.0, 0
    for k, (got, want) in enumerate(zip(lines, c.ai_lines, strict=True)):
        assert len(got) == len(want) == len(road), k
        for i, (p, a) in enumerate(zip(got, want, strict=True)):
            assert p._replace(speed=0.0) == a._replace(speed=0.0), (k, i)
            signs += sum(struct.pack("<f", u) != struct.pack("<f", v) for u, v in zip(p.position, a.position))
            worst = max(worst, abs(p.speed - a.speed))
    assert worst < 6e-6, worst                          # measured: at most 5.73e-6
    step_n, steps = tables.steps(d)
    assert (step_n, steps) == (c.step_n, tuple(c.steps))
    assert tables.attributes(d) == bytes(c.ground_data.attributes)
    return signs


def test_tables_beginner(disc, romset_zip):
    """Dreamcast n 0 (Three Seven Speedway) converts to the arcade's Beginner (id 0)."""
    (c,) = _arcade(romset_zip, [0])
    d = _data(disc, 0)
    assert _same_tables(d, c) == 0                      # the AI lines byte for byte too (the speed aside)
    assert (len(c.road), len(c.steps)) == (196, 503)


def test_tables_expert_advanced(disc, romset_zip):
    """n 1 converts to Expert (id 1), n 2 to Advanced (id 2). Measured: one AI coordinate's zero has the other sign
    (n 2 line 3 point 445, y: -0.0 on the disc, 0.0 in the arcade), equal as a value."""
    for n, c, signs in zip((1, 2), _arcade(romset_zip, (1, 2)), (0, 1)):
        assert _same_tables(_data(disc, n), c) == signs, n


def test_checkpoint_rows(disc, romset_zip):
    """n 0-2's rows are the arcade's checkpoint rows (segment, raw, order); with the road line and the slot's time
    table, course.checkpoint_views gives the arcade's checkpoints whole; n 3's are segments 136 and 263."""
    from dtools.arcade import golden
    from dtools.dc.exe import EXE_NAME
    blob = disc.read(EXE_NAME)
    for n, c in enumerate(_arcade(romset_zip, (0, 1, 2))):
        rows = tables.checkpoint_rows(blob, n)
        arcade_rows = [cp for cp in c.checkpoints if not cp.lap_line]
        assert [(r.index, r.segment, r.raw, r.lap_line) for r in rows] == \
            [(r.index, r.segment, r.raw, r.lap_line) for r in arcade_rows]
        key = (C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)
        views = C.checkpoint_views([(r.segment, r.raw) for r in rows], list(tables.road_line(_data(disc, n))),
                                   c.time_tables[key][2], c.laps[C.DEFAULT_MODE][1])
        assert tuple(views) == c.checkpoints
        # the converted race tables together pass the arcade's cross-table rules in the slot
        step_n, steps = tables.steps(_data(disc, n))
        golden.check_course(dataclasses.replace(c, road=tables.road_line(_data(disc, n)),
                                                ai_lines=tables.ai_lines(_data(disc, n)), step_n=step_n,
                                                steps=steps, checkpoints=tuple(views)))
    assert [r.segment for r in tables.checkpoint_rows(blob, 3)] == [136, 263]
    assert tables.checkpoint_rows(blob, 0) == ()


def test_tables_desert_city(disc):
    """n 3 (Desert City): 465 road points, 527 steps; eight AI lines, each a closed loop of 465 points; every
    heading an s16, every lane inside 0x000-0x700; the steps on the road line; the first 527 polygons the road."""
    d = _data(disc, 3)
    road = tables.road_line(d)
    assert len(road) == 465
    step_n, steps = tables.steps(d)
    assert (step_n, len(steps)) == (464, 527)
    assert all(0 <= s.segment <= step_n for s in steps)
    lines = tables.ai_lines(d)
    assert len(lines) == 8
    for k, line in enumerate(lines):
        assert len(line) == 465
        xyz = np.array([p.position for p in line])
        gaps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        close, med = float(np.linalg.norm(xyz[0] - xyz[-1])), float(np.median(gaps))
        assert 0.25 * med < close < 2.0 * med, (k, close, med)
        assert all(-0x8000 <= p.heading <= 0x7FFF and -0x8000 <= p.pitch <= 0x7FFF for p in line)
        assert all(0x000 <= p.lane <= 0x700 for p in line), k
        assert all(0 <= p.step < 527 for p in line)
    for p in road:
        assert all(-0x8000 <= ln.heading <= 0x7FFF for ln in p.lanes)
    attrs = tables.attributes(d)
    assert len(attrs) == len(d.cells) and all(a & 15 == 1 for a in attrs[:527])
    assert not any(a & 15 == 1 for a in attrs[527:])
    # the arcade's encoders take every table
    C.encode_road_line(list(road))
    C.encode_line_table(0, [list(line) for line in lines])
    # the AI lines run the same way as the road line: the converted z is the Dreamcast's negated
    assert lines[0][0].position[2] == -d.lines[0][0].z
    assert math.isclose(road[0].lanes[road[0].lane_lo].position[2], -d.road[0].lanes[d.road[0].lane_lo].z)


def _heading_residuals(road, rule: str) -> np.ndarray:
    """Each used lane's heading less a direction in the arcade's convention (0x4000 a turn; travel (x, z) along
    (-sin, cos)), wrapped to half a turn: rule "lane", the direction to the same lane of the next road point (closed
    loop); rule "cross", the cross-section lane_lo -> lane_hi turned back a quarter (one value per road point)."""
    turn, out = 0x4000, []
    for k, p in enumerate(road):
        if rule == "lane":
            q = road[(k + 1) % len(road)]
            pairs = [(p.lanes[j], p.lanes[j].position, q.lanes[j].position) for j in range(p.lane_lo, p.lane_hi + 1)
                     if q.lane_lo <= j <= q.lane_hi]
        else:
            pairs = [(p.lanes[p.lane_lo], p.lanes[p.lane_lo].position, p.lanes[p.lane_hi].position)]
        for ln, a, b in pairs:
            geo = math.atan2(-(b[0] - a[0]), b[2] - a[2]) / (2.0 * math.pi) * turn
            geo -= turn / 4 if rule == "cross" else 0
            out.append((ln.heading - geo + turn / 2) % turn - turn / 2)
    return np.array(out)


def test_lane_headings_are_the_cross_section_normal(disc, romset_zip):
    """Discovery (Phase 5 residual: n 3's lane headings are up to 527 units off their lane's direction): the arcade's
    own are further off (ids 0-2: up to 225, 1336, 935 units). A road point's lanes share one heading, the normal
    of its cross-section (lane_lo -> lane_hi turned back a quarter) within about one unit, on ids 0-2 and on the
    converted n 3 alike (its mean +1 unit higher: the 0x2000 offset of the AI headings again). Copying is right."""
    for i, c in enumerate(_arcade(romset_zip, (0, 1, 2))):
        assert all(len({ln.heading for ln in p.used_lanes}) == 1 for p in c.road)
        assert float(np.abs(_heading_residuals(c.road, "lane")).max()) > (200, 1300, 900)[i]
        r = _heading_residuals(c.road, "cross")
        assert float(np.abs(r).max()) < 1.2 and -0.4 < float(r.mean()) < -0.2, i     # measured 0.9, 1.1, 0.8
    road = tables.road_line(_data(disc, 3))
    assert all(len({ln.heading for ln in p.used_lanes}) == 1 for p in road)
    assert 520 < float(np.abs(_heading_residuals(road, "lane")).max()) < 530             # measured 527.0
    r = _heading_residuals(road, "cross")
    assert float(np.abs(r).max()) < 1.7 and 0.6 < float(r.mean()) < 0.9                  # measured 1.6, +0.75


# ---- the R files' lane ranges (reverse stage 2, step 2) ----

def _rdata(disc, n: int) -> CourseData:
    from dtools.dc.data import decode_data
    from dtools.dc.disc import course_files
    name = course_files(n, "R", kind="data")
    return decode_data(disc.read(name), name)


# Road points of 1R whose used range names an empty slot (the lanes mirrored, the range not); measured 17 Nov 2026.
FIXED_RANGES = {1: 38}


def _centres(d: CourseData) -> list[tuple[float, float, float]]:
    out = []
    for r in d.road:
        used = r.lanes[r.lane_lo:r.lane_hi + 1]
        out.append(tuple(sum(getattr(ln, a) for ln in used) / len(used) for a in "xyz"))
    return out


def test_lane_ranges_fixed_on_1r(disc):
    """1R's road points hold the plain lanes mirrored (lane j the plain lane LANES - 1 - j) but some keep the plain
    lane_lo..lane_hi, naming an empty slot: lane_ranges_fixed names each with its mirrored range, exactly the
    non-empty slots; none on the other fifteen files. with_lane_ranges_fixed applies them and nothing else."""
    for n in range(8):
        for d, v in ((_data(disc, n), ""), (_rdata(disc, n), "R")):
            fixed = tables.lane_ranges_fixed(d)
            if (n, v) != (1, "R"):
                assert fixed == {} and tables.with_lane_ranges_fixed(d) is d, (n, v)
                continue
            assert len(fixed) == FIXED_RANGES[1]
            out = tables.with_lane_ranges_fixed(d)
            for k, (r, s) in enumerate(zip(d.road, out.road, strict=True)):
                assert s.lanes == r.lanes and s.curve == r.curve
                if k in fixed:
                    assert (s.lane_lo, s.lane_hi) == fixed[k] == (C.LANES - 1 - r.lane_hi, C.LANES - 1 - r.lane_lo)
                else:
                    assert (s.lane_lo, s.lane_hi) == (r.lane_lo, r.lane_hi)
            assert dataclasses.replace(out, road=d.road) == d


def test_every_road_line_is_continuous(disc):
    """All sixteen road lines, the lane ranges fixed: no step between consecutive road centres (the lap line too)
    passes 20 units."""
    for n in range(8):
        for d in (_data(disc, n), _rdata(disc, n)):
            c = _centres(tables.with_lane_ranges_fixed(d))
            worst = max(math.dist(c[k - 1], c[k]) for k in range(len(c)))
            assert worst < 20, (n, worst)


def test_lane_range_repair_refuses_a_non_mirror():
    """A used range naming an empty slot whose mirror is not exactly the non-empty slots is refused, never guessed;
    a range with no empty slot is left alone."""
    d = make_data()
    empty = Lane(0.0, 0.0, -0.0, 0, 0x1FFF, 0, 0)
    r = d.road[2]
    lanes = (empty,) + r.lanes[1:]                                  # slot 0 empty, range 0..6: mirror 1..7 exact
    good = dataclasses.replace(d, road=d.road[:2] + (dataclasses.replace(r, lane_lo=0, lane_hi=6, lanes=lanes),)
                               + d.road[3:])
    assert tables.lane_ranges_fixed(good) == {2: (1, 7)}
    lanes = (empty,) + r.lanes[1:6] + (empty, r.lanes[7])           # slots 0 and 6 empty: no mirror is exact
    bad = dataclasses.replace(d, road=d.road[:2] + (dataclasses.replace(r, lane_lo=0, lane_hi=6, lanes=lanes),)
                              + d.road[3:])
    with pytest.raises(ConvertError, match=r"road point 2: lanes 0\.\.6 name an empty slot"):
        tables.lane_ranges_fixed(bad)
    assert tables.lane_ranges_fixed(d) == {}
