"""Tests for the Dreamcast course data decoder and encoder (dtools/dc/data.py)."""

import math
import random
import re
import struct
from dataclasses import replace

import numpy as np
import pytest

from dtools.dc import data
from dtools.dc.data import Cell, CourseData, Lane, LinePoint, RoadPoint, Step
from dtools.dc.errors import DcError

ORDER_C0 = (0, 4, 5, 3, 1, 6, 7, 2)      # course 0's section order (and courses 2-7)
ORDER_C1 = (0, 4, 5, 3, 1, 7, 2, 6)      # course 1's: the cells last, after a 2-byte pad


def P(i, k=0):
    return LinePoint(float(i), 0.5 * k, -float(i), 1.25, -27, 0x0AAA, 4, -2, 502 + k, i % 3)


def L(i, j):
    return Lane(float(i) + 3.0 * j, 0.0, -float(i), 0, 0x0AA9, 0, 0)


def make_data(order=ORDER_C0, ncell=5, npoints=9, **kw) -> CourseData:
    """npoints road points and AI line points (8 lines, line k shifted by k), ncell cells (the first 3 the road),
    two non-empty lists and the grid pointing at them."""
    lines = tuple(tuple(P(i, k) for i in range(npoints)) for k in range(8))
    road = tuple(RoadPoint(1, 6, 0x1111 * (i % 2), tuple(L(i, j) for j in range(8))) for i in range(npoints))
    cells = tuple(Cell(((0.0, 0.0, float(c)), (1.0, 0.0, float(c)), (1.0, 0.0, c + 1.0), (0.0, 0.0, c + 1.0)),
                       (0.5, 0.0, c + 0.5), 0x2AAB) for c in range(ncell))
    steps = (Step(0, 1), Step(1, 0x42), Step(npoints - 1, 0x82))
    attributes = bytes([1, 1, 1] + [5] * (ncell - 3))
    lists = ((), (0, 1), (2, ncell - 1))
    grid = [0] * data.GRID_CELLS
    grid[64 * 128 + 64] = 1
    grid[64 * 128 + 65] = 2
    grid[65 * 128 + 64] = 2
    d = CourseData(order, lines, tuple(range(8)), road, attributes, steps, cells, tuple(grid), lists)
    return replace(d, **kw)


def test_data_round_trip_synthetic():
    for order, ncell in ((ORDER_C0, 5), (ORDER_C0, 6), (ORDER_C1, 5), (ORDER_C1, 6)):
        d = make_data(order, ncell)
        blob = data.encode_data(d)
        back = data.decode_data(blob, "SYN_DATA.BIN")
        assert back == d
        assert data.encode_data(back) == blob
        assert data.road_cells(back) == d.cells[:3]
        h = struct.unpack_from("<8I", blob, 0)
        assert h[0] == 0                                    # H0's word is 0; the lines start at 0x20
        # H1: segments (road points - 1), steps, the file offset of the steps (after the attributes, even)
        assert struct.unpack_from("<3I", blob, h[1]) == (8, 3, h[3] + ncell + ncell % 2)
        assert all(h[i] % data.ALIGN[i] == 0 for i in range(1, 8))
        # every line is stored with its first 8 points again after its end
        off, count = struct.unpack_from("<2I", blob, h[4])
        assert (off, count) == (0x20, 9)
        assert blob[off + 28 * count:off + 28 * count + 224] == blob[off:off + 224]
    # course 1's order: the 2-byte pad after the grid (H2) before the cells (H6), which must be 4-aligned
    blob = data.encode_data(make_data(ORDER_C1))
    h = struct.unpack_from("<8I", blob, 0)
    assert h[6] == h[2] + 32768 + 2 and blob[h[2] + 32768:h[6]] == b"\0\0"
    # the line directory may name the stored lines in another order (the M files): kept as `line_slots`
    d = make_data(line_slots=(7, 6, 5, 4, 3, 2, 1, 0))
    blob = data.encode_data(d)
    back = data.decode_data(blob)
    assert back == d and data.encode_data(back) == blob
    assert struct.unpack_from("<I", blob, struct.unpack_from("<I", blob, 16)[0])[0] == 0x20 + 7 * 28 * 17


def test_data_accepts_numpy_numbers():
    d = make_data()
    p = d.lines[0][0]
    p2 = replace(p, x=np.float32(0.0), pitch=np.int16(-27), step=np.uint16(0))
    d2 = replace(d, lines=((p2,) + d.lines[0][1:],) + d.lines[1:], grid=np.array(d.grid, dtype=np.uint16))
    assert data.encode_data(d2) == data.encode_data(d)


def _refuses(d, match):
    with pytest.raises(DcError, match=match):
        data.encode_data(d)


def test_data_refuses_encode():
    d = make_data()
    # floats the format cannot hold
    for bad in (float("nan"), float("inf"), 1e39, True, "1"):
        _refuses(replace(d, lines=((replace(d.lines[0][0], x=bad),) + d.lines[0][1:],) + d.lines[1:]), "line 0")
    c = d.cells[0]
    _refuses(replace(d, cells=(replace(c, centre=(0.0, float("nan"), 0.0)),) + d.cells[1:]), "cell 0")
    # an integer past the float range is refused, not an OverflowError (tablebase.is_number)
    _refuses(replace(d, lines=((replace(d.lines[0][0], z=-10**400),) + d.lines[0][1:],) + d.lines[1:]), "line 0")
    _refuses(replace(d, cells=(replace(c, centre=(0.0, 10**400, 0.0)),) + d.cells[1:]), "cell 0")
    _refuses(replace(d, cells=(replace(c, corners=c.corners[:3]),) + d.cells[1:]), "cell 0")
    # integers out of their field
    _refuses(replace(d, lines=((replace(d.lines[0][0], pitch=0x8000),) + d.lines[0][1:],) + d.lines[1:]), "pitch")
    _refuses(replace(d, steps=(Step(0, 0x10000),) + d.steps[1:]), "flags")
    _refuses(replace(d, cells=(replace(c, direction=-1),) + d.cells[1:]), "direction")
    # indices past their table: a list entry past the cells, a grid value past the lists, a lane past the 8 slots,
    # a step's segment past the road, a line point's step past the steps
    _refuses(replace(d, lists=d.lists[:2] + ((2, 5),)), "past the 5 cells")
    grid = list(d.grid)
    grid[3] = 3
    _refuses(replace(d, grid=tuple(grid)), "past the 3 lists")
    r = d.road[0]
    _refuses(replace(d, road=(replace(r, lane_hi=8),) + d.road[1:]), "lane")
    _refuses(replace(d, road=(replace(r, lane_lo=7, lane_hi=6),) + d.road[1:]), "lane")
    _refuses(replace(d, steps=(Step(9, 1),) + d.steps[1:]), "segment 9")
    _refuses(replace(d, lines=((replace(d.lines[0][0], step=3),) + d.lines[0][1:],) + d.lines[1:]), "step 3")
    # counts that disagree
    _refuses(replace(d, attributes=d.attributes[:4]), "attributes")
    _refuses(replace(d, steps=d.steps + (Step(0, 1),) * 3), "steps")
    _refuses(replace(d, lines=d.lines[:7]), "8 lines")
    _refuses(replace(d, lines=(d.lines[0][:7],) + d.lines[1:]), "at least 8")
    _refuses(replace(d, line_slots=(0, 0, 1, 2, 3, 4, 5, 6)), "permutation")
    _refuses(replace(d, order=(4, 0, 5, 3, 1, 6, 7, 2)), "order")
    _refuses(replace(d, order=(0, 4, 5, 3, 1, 6, 7, 7)), "order")
    _refuses(replace(d, grid=d.grid[:-1]), "16384")
    _refuses(replace(d, road=()), "road point")
    # types
    _refuses(replace(d, cells=(None,) + d.cells[1:]), "not a Cell")
    _refuses(replace(d, road=("x",) + d.road[1:]), "not a RoadPoint")
    _refuses(replace(d, steps=((0, 1),) + d.steps[1:]), "not a Step")
    _refuses(replace(d, attributes=[1, 1, 1, 5, 5]), "bytes")
    with pytest.raises(DcError):
        data.encode_data(None)


def test_data_refuses():
    """Review Focus 1: a truncated file, or an offset, count or index past the end, is a DcError naming the file and
    the offset."""
    d = make_data()
    blob = data.encode_data(d)
    h = list(struct.unpack_from("<8I", blob, 0))
    name = "BAD_DATA.BIN"

    def patched(off, value, fmt="<I", b=blob):
        b = bytearray(b)
        struct.pack_into(fmt, b, off, value)
        return bytes(b)

    def refused(b, match):
        with pytest.raises(DcError, match=match) as e:
            data.decode_data(b, name)
        msg = str(e.value)
        assert name in msg
        assert re.search(r"offset 0x[0-9a-f]+", msg), msg          # Review Focus 1: the offset is named
        return msg

    # a truncated header
    refused(blob[:0x1c], "header")
    # a section offset past the end of the file, or before the header
    for i in range(1, 8):
        msg = refused(patched(4 * i, len(blob) + 4), f"H{i}")
        assert f"{len(blob) + 4:#x}" in msg
    refused(patched(4 * 6, 0x10), "H6")
    # H0's word is not 0
    refused(patched(0, 0x20), "H0")
    # a line count past its section: line 7's count grows into the next line's (and past H0)
    refused(patched(h[4] + 8 * 7 + 4, 10_000), "line 7")
    refused(patched(h[4] + 8 * 3, 0x24), "line")                 # an offset that is not where a line starts
    # a truncated cell: the file cut inside the cells (course 1's order puts them last)
    b1 = data.encode_data(make_data(ORDER_C1))
    refused(b1[:-10], "cell")
    # cut inside the grid (the last section of course 0's order): a short grid
    refused(blob[:-2], "H2|grid")
    # the steps offset (H1 word 2) that disagrees with the attributes' size
    refused(patched(h[1] + 8, h[1] + 8 + 2), "steps")
    # a step count past the section
    refused(patched(h[1] + 4, 10_000), "step")
    # the segment count disagreeing with the road section's size
    refused(patched(h[1], 3), "road")
    # a grid value that is not the start of a list, or past the lists
    refused(patched(h[2] + 2 * (64 * 128 + 64), 2, "<H"), "grid")
    refused(patched(h[2] + 2 * (64 * 128 + 64), 0x7000, "<H"), "grid")
    # a list entry past the cells; a list count past the section
    refused(patched(h[7] + 2 * 2, 99, "<H"), "past the 5 cells")
    refused(patched(h[7] + 2 * 1, 900, "<H"), "list")
    # a lane slot past the 8
    refused(patched(h[5], 0x0901, "<H"), "lane")
    # a non-zero pad byte (the attribute pad with an odd cell count, the 2-byte pad of course 1's order)
    refused(patched(h[3] + 5, 7, "<B"), "pad")
    h1 = struct.unpack_from("<8I", b1, 0)
    refused(patched(h1[6] - 2, 1, "<H", b1), "pad")
    # a wrap point that is not the line's first points again
    refused(patched(0x20 + 28 * 9, 0x3F800000), "wrap")
    # a NaN coordinate
    refused(patched(h[6], 0x7FC00000), "not finite")
    # extra bytes after the last section
    refused(blob + b"\0\0\0\0", "end")
    with pytest.raises(DcError):
        data.decode_data("text")
    for bad_name in (np.array(["a", "b"]), None, 3):
        with pytest.raises(DcError, match="name"):
            data.decode_data(blob, bad_name)


def test_data_decode_never_raises_other_errors():
    for order in (ORDER_C0, ORDER_C1):
        blob = data.encode_data(make_data(order))
        h = struct.unpack_from("<8I", blob, 0)
        rng = random.Random(5)
        hot = [0, 4, 8, 12, 16, 20, 24, 28, h[1], h[1] + 4, h[1] + 8, h[4], h[4] + 4, h[4] + 60, h[3], h[5], h[7],
               h[7] + 2, h[2] + 2 * (64 * 128 + 64)]
        for i in range(2500):
            b = bytearray(blob)
            for _ in range(rng.randint(1, 3)):
                at = rng.choice(hot) + rng.randrange(4) if rng.random() < 0.5 else rng.randrange(len(b))
                b[min(at, len(b) - 1)] = rng.randrange(256)
            if i % 10 == 0:
                b = b[:rng.randrange(len(b))]
            try:
                d = data.decode_data(bytes(b), "FUZZ_DATA.BIN")
            except DcError:
                continue
            data.road_cells(d)
            assert data.encode_data(d) == bytes(b)


def test_grid_square():
    # 16-unit squares from -1024, row from z, column from x; outside -1024..1024 wraps
    assert data.grid_square(-1024.0, -1024.0) == 0
    assert data.grid_square(1023.9, 1023.9) == data.GRID_CELLS - 1
    assert data.grid_square(0.0, 0.0) == 64 * 128 + 64
    assert data.grid_square(-1008.0, -1024.0) == 1 and data.grid_square(-1024.0, -1008.0) == 128
    assert data.grid_square(1024.0, 0.0) == 64 * 128
    assert data.grid_square(np.float32(0.0), np.int64(0)) == 64 * 128 + 64
    for bad in (float("nan"), float("inf"), -float("inf"), "a", None, True, np.bool_(False), (1.0,), 10**400,
                -10**400):
        with pytest.raises(DcError, match="grid square"):
            data.grid_square(bad, 0.0)
        with pytest.raises(DcError, match="grid square"):
            data.grid_square(0.0, bad)


def test_road_cells_refuses():
    d = make_data()
    assert data.road_cells(d) == d.cells[:3]
    for kw, match in ((dict(cells=None), "cells"), (dict(steps=None), "steps"), (dict(cells=5), "cells"),
                      (dict(steps=d.steps * 3), "more than")):
        with pytest.raises(DcError, match=match):
            data.road_cells(replace(d, **kw))
    with pytest.raises(DcError, match="CourseData"):
        data.road_cells(None)


def test_data_names():
    names = data.data_names()
    assert len(names) == 32 == len(set(names))
    assert names[0] == "CS_DATA0.BIN" and names[1] == "CSDAT0R.BIN" and names[-1] == "CSDAT7MR.BIN"


# ---- the disc ----

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


def test_data_lossless_all(disc):
    """Review Focus 4: all 32 files, each variant with its own offsets and sizes (course 5's R and MR files are 80
    bytes longer: one more cell, four more list words)."""
    assert data.verify_lossless_data(disc) == []
    sizes = {name: len(disc.read(name)) for name in data.data_names()}
    assert sizes["CSDAT5R.BIN"] - sizes["CS_DATA5.BIN"] == 80
    d5, d5r = (data.decode_data(disc.read(n), n) for n in ("CS_DATA5.BIN", "CSDAT5R.BIN"))
    assert (len(d5.cells), len(d5r.cells)) == (874, 875)
    assert sum(map(len, d5r.lists)) + len(d5r.lists) - sum(map(len, d5.lists)) - len(d5.lists) == 8


def test_data_counts_real(disc):
    for n, points, road in ((0, 196, 503), (1, 991, 1064), (2, 487, 818)):
        name = f"CS_DATA{n}.BIN"
        d = data.decode_data(disc.read(name), name)
        assert [len(line) for line in d.lines] == [points] * 8
        assert len(d.road) == points
        assert len(d.steps) == road == len(data.road_cells(d))
        assert set(d.attributes[:road]) == {1} and 1 not in set(d.attributes[road:])
    d0 = data.decode_data(disc.read("CS_DATA0.BIN"))
    assert len(d0.cells) == 1184                                      # 681 beyond the road
    # the pit (steps flagged 0x42) and its box (0x82): course 0's box is cell 152
    assert [k for k, s in enumerate(d0.steps) if s.flags == 0x82] == [152]
    # the grid: every cell is listed in the square of its own centre (H2 row z, column x, 16 units from -1024)
    for k, c in enumerate(d0.cells):
        assert k in d0.lists[d0.grid[data.grid_square(c.centre[0], c.centre[2])]]
    # the M file's directory names the stored lines in reverse
    dm = data.decode_data(disc.read("CSDAT0M.BIN"))
    assert dm.line_slots == (7, 6, 5, 4, 3, 2, 1, 0)


# Course n 0-2 against the arcade's ids 0-2: (steps, cells whose corners or centre differ, cells whose direction
# differs), all off the road (measured; tests/test_convert_ground.py pins the converted ground to the same counts).
OFF_ROAD_DIFFERENCES = {0: (503, 2, 8), 1: (1064, 43, 0), 2: (818, 3, 0)}


def test_data_matches_arcade(disc, romset_zip):
    """Courses 0-2 are the arcade's Beginner, Expert and Advanced value for value, with z negated, at scale 1
    (docs/formats/dreamcast-disc.md, "Field rules against the arcade"): every road cell; the off-road cells that
    differ are a pinned count. The AI points, road lanes, steps and attributes are compared through the converter,
    which holds their rules (dtools/convert/tables.py; tests/test_convert_tables.py, "the disc against the arcade")."""
    from dtools.arcade import golden
    from dtools.arcade.romset import RomSet
    from dtools.convert import tables
    rs = RomSet.from_zip(romset_zip)
    for n, (count, moved, turned) in OFF_ROAD_DIFFERENCES.items():
        c = golden.decode_course(rs, n)
        d = data.decode_data(disc.read(f"CS_DATA{n}.BIN"))
        assert len(data.road_cells(d)) == len(c.steps) == count
        # H0, H3, H5 by the converter's rules (the AI speed within 6e-6: test_convert_tables)
        assert tables.road_line(d) == tuple(c.road)
        assert tables.steps(d) == (c.step_n, tuple(c.steps))
        assert tables.attributes(d) == bytes(c.ground_data.attributes)
        assert [[p._replace(speed=0.0) for p in line] for line in tables.ai_lines(d)] == \
            [[a._replace(speed=0.0) for a in line] for line in c.ai_lines]
        # H6: the road cells exactly; the off-road cells that moved, pinned
        polys = c.ground_data.polys
        assert len(d.cells) == len(polys)

        def same_place(cell, poly):
            neg = tuple((x, y, -z) for x, y, z in poly.corners)
            return cell.corners == neg and cell.centre == (poly.centre[0], poly.centre[1], -poly.centre[2])

        def same_turn(cell, poly):
            return cell.direction == (poly.direction + 0x8000) & 0xFFFF

        for k in range(count):
            assert same_place(d.cells[k], polys[k]) and same_turn(d.cells[k], polys[k]), (n, k)
        assert sum(not same_place(x, y) for x, y in zip(d.cells, polys, strict=True)) == moved
        assert sum(not same_turn(x, y) for x, y in zip(d.cells, polys, strict=True)) == turned


def test_lines_closed_loops(disc):
    """Every line of every plain course is a closed loop: last to first about one step."""
    for n in range(8):
        name = f"CS_DATA{n}.BIN"
        d = data.decode_data(disc.read(name), name)
        for k, line in enumerate(d.lines):
            xyz = np.array([(p.x, p.y, p.z) for p in line])
            steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            close = float(np.linalg.norm(xyz[0] - xyz[-1]))
            med = float(np.median(steps))
            assert 0.25 * med < close < 2.0 * med, (name, k, close, med)
        centre = np.array([[np.mean([ln.x for ln in r.lanes[r.lane_lo:r.lane_hi + 1]]),
                            np.mean([ln.z for ln in r.lanes[r.lane_lo:r.lane_hi + 1]])] for r in d.road])
        gap = math.dist(centre[0], centre[-1])
        assert gap < 2.0 * float(np.median(np.linalg.norm(np.diff(centre, axis=0), axis=1))), (name, gap)
