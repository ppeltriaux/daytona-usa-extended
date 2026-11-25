"""v0.3: the per-course polygon cap, the whole block and the far view (course.far_view, build.far_viewed_set;
docs/follow-ups/v0.3.md). The MAME side: tests/test_far_view_mame.py."""
import struct

import pytest

from dtools.arcade import course as C
from dtools.arcade import i960, romset
from dtools.arcade.tablebase import DATA_BASE
from dtools.convert import ConvertError, build
from tests.test_convert_build import _romset


def _wide():
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return rs, C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))


def test_far_table():
    """A far table is the cap (u32), (row, column) signed bytes, FAR_END, zero padding to FAR_TABLE_SIZE; far_table
    refuses a cap outside OBC..CAP_MAX, the reference cell, a pair twice, more than FAR_CELLS, offsets past -8..8."""
    t = C.far_table(6000, ((3, -1), (-4, 3), (1, 0)))
    assert t == struct.pack("<I", 6000) + bytes([3, 0xFF, 0xFC, 3, 1, 0, C.FAR_END, 0]) + bytes(C.FAR_TABLE_SIZE - 12)
    assert C.far_table(C.OBC, ()) == struct.pack("<I", 5000) + bytes([C.FAR_END]) + bytes(C.FAR_TABLE_SIZE - 5)
    assert C.FAR_CELLS == 62 and C.FAR_TABLE_SIZE % 4 == 0 and C.FAR_TABLE_SIZE >= 4 + 2 * C.FAR_CELLS + 2
    most = tuple((r, c) for r in range(-3, 4) for c in range(-4, 5) if (r, c) != (0, 0))
    assert len(most) == C.FAR_CELLS and len(C.far_table(C.CAP_MAX, most)) == C.FAR_TABLE_SIZE
    for cap, bad in ((6000, ((0, 0),)), (6000, ((3, 0), (3, 0))), (6000, most + ((5, 5),)), (6000, ((9, 0),)),
                     (6000, [(3, True)]), (6000, "x"), (6000, ((3,),)), (4999, ()), (C.CAP_MAX + 1, ()), (True, ()),
                     (6000.0, ())):
        with pytest.raises(C.CourseError):
            C.far_table(cap, bad)


def test_far_shape():
    """build.FAR_SHAPE: rows 3-4 ahead across -3..3 and the columns +-3 beside rows 0-2, 20 cells past the block;
    with BLOCK_OFFSETS (the block's 24 cells but the reference cell) 44, 45 with the reference cell (the visible
    list holds 63); far_offsets turns a shape per quadrant as view_tables turns a view."""
    assert (build.FAR_DEPTH, build.FAR_HALF) == (4, 3) and len(build.FAR_SHAPE) == 20
    assert set(build.FAR_SHAPE) == {(d, c) for d in (3, 4) for c in range(-3, 4)} | {(d, c) for d in range(3)
                                                                                    for c in (-3, 3)}
    assert set(build.BLOCK_OFFSETS) == {(r, c) for r in range(-2, 3) for c in range(-2, 3)} - {(0, 0)}
    assert len(build.BLOCK_OFFSETS) == 24 and 1 + 24 + len(build.FAR_SHAPE) <= 63 and 24 + 20 <= C.FAR_CELLS
    one = ((4, 1),)
    assert [build.far_offsets(one, q) for q in range(4)] == [((4, 1),), ((1, -4),), ((-4, 1),), ((1, 4),)]
    assert build.FAR_IDS == (3, 4, 5, 6, 7, 11, 12, 13, 14, 15)
    assert C.OBC < build.POLYGON_CAP <= C.CAP_MAX < 32768


def test_far_code():
    """FAR_CODE is program ROM's last 256 bytes, reserved (romset.RESERVED); the hook fits, starts with the store
    the branch replaces and ends in ret; FAR_END is a cobr literal no offset can equal."""
    code = C.far_code().bytes()
    assert (C.FAR_CODE, C.FAR_SIZE) == (0x3FF00, 0x100) and C.FAR_CODE + C.FAR_SIZE == romset.PROGRAM_SIZE
    assert any((lo, n) == (C.FAR_CODE, C.FAR_SIZE) for lo, n, _ in romset.RESERVED["program"])
    assert len(code) <= C.FAR_SIZE
    assert code[:8] == C.far_original()[:8] and code[-4:] == i960.ctrl("ret") == C.far_original()[8:]
    assert 0 <= C.FAR_END <= 31 and C.FAR_END > 8
    assert i960.mem("st", "r3", C.POLYGON_CAP_RAM) in code


def test_far_viewed_set():
    """build.far_viewed_set on a widened set: every id reads its own four view-test tables, the bytes viewed_set
    gives it (the stock tables, or REVERSE_VIEW's cut: the game's own view test is v0.2's), each followed by its
    far table: a far id's names POLYGON_CAP, the block's 24 cells, then FAR_SHAPE turned per quadrant; any other id's
    OBC and no cell. The culling's end branches to FAR_CODE; the boot's cap word is untouched; nothing else changes.
    The steps refuse a second application, a set without view_ids or far tables, and a bad shape, ids or views."""
    rs, w = _wide()
    assert not C.far_viewed(w) and all(C.polygon_cap(w, k) == C.OBC == 5000 for k in range(C.COURSE_IDS))
    v = build.far_viewed_set(w)
    assert C.viewed(v) and C.far_viewed(v)
    stock = b"".join(w.read_i960(p, C.VIEW_TEST_SIZE) for p in C.view_tables(w, 0))
    seen = set()
    for k in range(C.COURSE_IDS):
        ptrs = C.view_tables(v, k)
        assert len(seen | set(ptrs)) == len(seen) + 4, k
        seen |= set(ptrs)
        assert [b - a for a, b in zip(ptrs, ptrs[1:])] == [C.VIEW_TEST_SIZE + C.FAR_TABLE_SIZE] * 3
        tables = b"".join(v.read_i960(p, C.VIEW_TEST_SIZE) for p in ptrs)
        assert tables == (build.view_tables(w, build.REVERSE_VIEW[k]) if k in build.REVERSE_VIEW else stock), k
        far = [C.read_far_table(v, p) for p in ptrs]
        if k in build.FAR_IDS:
            assert far == [(build.POLYGON_CAP, build.BLOCK_OFFSETS + build.far_offsets(build.FAR_SHAPE, q))
                           for q in range(4)], k
            assert C.polygon_cap(v, k) == build.POLYGON_CAP == 28000
        else:
            assert far == [(C.OBC, ())] * 4 and C.polygon_cap(v, k) == C.OBC, k
    code = C.far_code().bytes()
    assert v.read_i960(C.FAR_CODE, len(code)) == code
    assert v.read_i960(C.FAR_SITE, 4) == i960.ctrl("b", C.FAR_CODE - C.FAR_SITE)
    assert v.read_i960(C.FAR_SITE + 4, 8) == w.read_i960(C.FAR_SITE + 4, 8)
    assert v.read_i960(0x12F8, 8) == rs.read_i960(0x12F8, 8) and struct.unpack("<I", v.read_i960(0x12FC, 4))[0] == C.OBC
    plain = C.view_ids(w, {})                                     # the program bytes view_ids itself writes
    diff = {i for i in range(len(v.program)) if v.program[i] != plain.program[i]}
    own = (set(range(C.FAR_CODE, C.FAR_CODE + C.FAR_SIZE)) | set(range(C.FAR_SITE, C.FAR_SITE + 4))
           | set(range(C.VIEW_PTRS, C.VIEW_PTRS + 16 * C.COURSE_IDS)))
    assert diff and diff <= own
    lo = build.ID_DATA_FREE[0]
    size = C.COURSE_IDS * 4 * (C.VIEW_TEST_SIZE + C.FAR_TABLE_SIZE)
    assert min(seen) - DATA_BASE >= lo and v.data[:lo] == w.data[:lo] and v.data[lo + size:] == w.data[lo + size:]
    assert v.copro_data == w.copro_data and v.polygons == w.polygons and v.textures == w.textures
    with pytest.raises(C.CourseError):
        C.far_view(v)
    with pytest.raises(C.CourseError):
        C.far_view(w)                                             # no view_ids
    with pytest.raises(C.CourseError):
        C.far_view(build.viewed_set(w))                           # view_ids, the stock tables: no far tables
    for kw in ({"shape": ((3, 1),)}, {"shape": ((2, 1), (2, -1))}, {"far_ids": (16,)}, {"views": {16: build.VIEW_12}}):
        with pytest.raises(ConvertError):
            build.far_viewed_set(w, **kw)
    with pytest.raises(C.CourseError):
        build.far_viewed_set(w, cap=C.CAP_MAX + 1)
