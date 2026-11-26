"""Tests for the converted course and set (dtools/convert/build.py): the Dreamcast course's race tables and ground in
Beginner's slot, written into the arcade set's free runs.

Review Focus 1: every byte the converted set changes is inside one of the slot's own tables that the conversion
rewrites in place (the record's pointer words, the checkpoint and ground pointer words, the slot words, the grid
base), or was free (0xff) in DATA_FREE / COPRO_FREE. Review Focus 4: converted_course replaces every race table
together, golden.check_course passes on it and the set decodes back to it; a Course with a race table left as the
slot's is refused by `encode` with ConvertError naming it. The known answer: n 0 converted into slot 0 is Beginner."""

import collections
import dataclasses
import functools
import math
import random
import statistics
import struct

import numpy as np
import pytest

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade import polyrom as P
from dtools.arcade import texture as T
from dtools.arcade import tgp
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import COPRO_SIZE, DATA_COPY_BASE, DATA_COPY_LEN, RomSet
from dtools.arcade.tablebase import DATA_BASE
from dtools.convert import ConvertError, build, model, tables
from dtools.dc import pol
from dtools.dc.data import Step
from dtools.dc.errors import DcError
from dtools.dc.exe import EXE_NAME, CheckpointTable
from tests.test_convert_tables import _data, disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_dc_data import OFF_ROAD_DIFFERENCES, make_data
from tests.test_dc_exe import ENTRIES, LISTS, _exe
from tests.test_mirror import _differ

DATA_LO, DATA_HI = tgp.DATA_FREE[0], tgp.DATA_FREE[0] + tgp.DATA_FREE[1]
COPRO_LO, COPRO_HI = tgp.COPRO_FREE[0], tgp.COPRO_FREE[0] + tgp.COPRO_FREE[1]
HALF = COPRO_SIZE // 2
PROG_LO, PROG_HI = build.PROGRAM_FREE[0], build.PROGRAM_FREE[0] + build.PROGRAM_FREE[1]
POLY_LO, POLY_HI = build.POLYGON_FREE[0], build.POLYGON_FREE[0] + build.POLYGON_FREE[1]
TEX_LO, TEX_HI = build.TEXTURES_FREE[0], build.TEXTURES_FREE[0] + build.TEXTURES_FREE[1]

# The tables the conversion rewrites in place (golden.coverage names): every other byte it changes was free.
IN_PLACE = frozenset({"course record", "checkpoint pointer", "grid base", "ground grid pointer",
                      "ground attribute pointer", "ground slot: list block address",
                      "ground slot: polygon pool address", "segment flags pointer", "rolling start"})
RACE = ("step_n", "steps", "pit", "ground", "ground_data", "road", "ai_lines", "checkpoint_table", "checkpoints",
        "time_tables", "laps", "car_count", "rolling_start", "grid_base", "grid_lanes", "spawn", "segment_flags")


@functools.lru_cache(maxsize=1)
def _romset():
    d = config.rom_dir()
    if d is None or not (d / "daytona.zip").is_file():
        return None
    return RomSet.from_zip(d / "daytona.zip")


@functools.lru_cache(maxsize=1)
def _beginner():
    return G.decode_course(_romset(), 0)


@pytest.fixture
def rs():
    got = _romset()
    if got is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    return got


def _syn():
    """make_data's nine road points and five cells, its steps without a pit (make_data's lane and box are not one
    run of its short loop)."""
    return dataclasses.replace(make_data(), steps=(Step(0, 1), Step(1, 1), Step(8, 1)))


LAYOUT_FIELDS = ("data_at", "copro_at", "program_at", "polygons_at")
SYN_EXE = _exe(CheckpointTable(((3, 6),) + LISTS[1:], ENTRIES))     # n 0: checkpoints at segments 3 and 6


# ---- no data ----

def test_grid_bases_and_runs():
    assert build.GRID_BASES == (100, 984, 480, 458, 418, 123, 493, 493)
    assert build.PROGRAM_FREE == (0x3CE98, 12648)
    assert build.POLYGON_FREE == (0xE3C364, 1850524)
    assert build.Layout(1, 2, 3) == build.Layout(data_at=1, copro_at=2, program_at=3,
                                                 polygons_at=build.POLYGON_FREE[0])
    assert build.Layout(1, 2) == build.Layout(1, 2, build.EMPTY_AT)      # the shared actor table's (Phase 10 Task 1)


# ---- synthetic course data, the real set ----

def test_converted_synthetic(rs):
    """The tables in DATA_FREE from its first byte in order (grid, attribute block, step table, road line, line
    table, checkpoint table), the ground's list block and pool in COPRO_FREE (window-safe from its first word); the
    record's pointer words, words 6-7 (the grid pointer), the ground view and the checkpoint rows repointed; laps,
    time tables (their words read on for the new checkpoint count), car count, rolling start, lanes and spawn
    records the slot's, the grid's views recomputed; check_course passes; `encode` writes a set that decodes to it."""
    b = _beginner()
    c = build.converted_course(rs, _syn(), SYN_EXE, 0)
    e, g = c.entry, c.ground_data
    assert (g.grid_at, g.attributes_at) == (DATA_LO, DATA_LO + 0x8000)
    assert e.step_table == DATA_LO + 0x8000 + 8                         # 5 attribute bytes padded to 8
    assert e.record_table == e.step_table + 8 + 4 * 3
    assert e.line_table == e.record_table + 8 + C.ROAD_POINT_SIZE * 9
    assert c.checkpoint_table == DATA_BASE + e.line_table + 8 * 8 + 8 * C.AI_POINT_SIZE * (9 + C.AI_LINE_WRAP)
    assert c.segment_flags == (c.checkpoint_table + 4 + 4 * 2, (0,) * 9)     # after the 2 checkpoint rows
    assert (g.list_base, g.pool_base) == (COPRO_LO // 4, COPRO_LO // 4 + len(g.raw))
    assert (g.course, g.base) == (0, 0)
    assert e.raw == (DATA_BASE + DATA_LO,) * 2 and b.entry.raw == (DATA_BASE + b.ground_data.grid_at,) * 2
    assert (e.course, e.upload_list, e.block_ref_a, e.block_ref_b) == \
        (0, b.entry.upload_list, b.entry.block_ref_a, b.entry.block_ref_b)
    assert c.ground == (DATA_BASE + DATA_LO, DATA_BASE + DATA_LO + 0x8000)
    assert (c.step_n, c.steps) == (8, (C.Step(0, 1), C.Step(1, 1), C.Step(8, 1))) and c.pit is None
    assert c.grid_base == (b.grid_base[0], 100)
    assert [(cp.segment, cp.lap_line) for cp in c.checkpoints] == [(3, False), (6, False), (0, True)]
    for f in ("laps", "car_count", "rolling_start", "grid_lanes", "objects", "cells", "actor_lists", "hud_map"):
        assert getattr(c, f) == getattr(b, f), f
    for key, (ptr, table, words) in c.time_tables.items():
        n = 1 + c.laps[key[0]][1] * 3
        assert (ptr, table) == b.time_tables[key][:2] and len(words) == n
        assert words == struct.unpack(f"<{n}I", rs.read_i960(table, 4 * n))     # the slot's table, read on
    for got, want in zip(c.spawn, b.spawn, strict=True):
        assert [s[:6] for s in got] == [s[:6] for s in want]             # the stored fields: the slot's
        assert all(s.segment == (s.start + 100) % 8 for s in got)
    G.check_course(c)
    out = build.encode(rs, c)
    d = G.decode_course(out, 0)
    assert [f for f in RACE + ("entry",) if getattr(d, f) != getattr(c, f)] == []
    assert G.mismatches(out, c) == []
    # a layout: the first byte used in each run
    lay = build.Layout(DATA_LO + 0x100, COPRO_LO + 0x40, build.PROGRAM_FREE[0])
    c2 = build.converted_course(rs, _syn(), SYN_EXE, 0, layout=lay)
    assert (c2.ground_data.grid_at, c2.ground_data.list_base) == (DATA_LO + 0x100, (COPRO_LO + 0x40) // 4)
    assert c2.entry.step_table == DATA_LO + 0x100 + 0x8008


def test_converted_honours_free_bytes(rs):
    """The default layout takes the first bytes of each run that are free in the set it is given (a set whose runs
    were partly used, e.g. by the mirror's char blocks); a layout over a byte that is not free is refused, and so is
    a Course whose tables land on one."""
    busy = tgp.write_free(rs, "data", DATA_LO + 0x40, b"\0" * 4)
    busy = tgp.write_free(busy, "copro_data", COPRO_LO + 0x40, b"\0" * 4)
    c = build.converted_course(busy, _syn(), SYN_EXE, 0)
    assert c.ground_data.grid_at == DATA_LO + 0x44
    assert c.ground_data.list_base == (COPRO_LO + 0x44) // 4
    build.encode(busy, c)
    first = build.Layout(DATA_LO, COPRO_LO + 0x44, build.PROGRAM_FREE[0])
    with pytest.raises(ConvertError, match="not free"):
        build.converted_course(busy, _syn(), SYN_EXE, 0, layout=first)
    with pytest.raises(ConvertError, match="not free"):
        build.converted_course(busy, _syn(), SYN_EXE, 0, layout=first._replace(data_at=DATA_LO + 0x44,
                                                                                copro_at=COPRO_LO))
    with pytest.raises(ConvertError, match="not free"):             # laid out on rs: its grid lands on busy's word
        build.encode(busy, build.converted_course(rs, _syn(), SYN_EXE, 0))


def test_converted_refuses(rs):
    """n outside 0-7, a slot other than 0, scenery other than "keep" and "empty", a layout that is not a
    Layout, not aligned, outside its run or too short for the tables: ConvertError; an exe the Dreamcast reader
    refuses: its DcError. And `encode` (Review Focus 4) refuses a Course with a race table left as the slot's:
    unset (encode_course would fill it from the set), at the slot's own address, or disagreeing with the others."""
    syn, b = _syn(), _beginner()
    for n in (-1, 8, True, np.bool_(False), 3.0, "3", None):
        with pytest.raises(ConvertError, match="course"):
            build.converted_course(rs, syn, SYN_EXE, n)
    for slot in (1, 2, -1, True, False, None, "0", 0.0):
        with pytest.raises(ConvertError, match="slot"):
            build.converted_course(rs, syn, SYN_EXE, 0, slot=slot)
    for scenery in ("", None, 0, "KEEP", "EMPTY", ("empty",)):
        with pytest.raises(ConvertError, match="scenery"):
            build.converted_course(rs, syn, SYN_EXE, 0, scenery=scenery)
    p0 = build.PROGRAM_FREE[0]
    for lay in ("x", (DATA_LO, COPRO_LO, p0), build.Layout(True, COPRO_LO, p0),
                build.Layout(DATA_LO, 1.0 * COPRO_LO, p0), build.Layout(DATA_LO, COPRO_LO, None), build.Layout(DATA_LO + 2, COPRO_LO, p0),
                build.Layout(DATA_LO, COPRO_LO + 2, p0), build.Layout(0x600000, COPRO_LO, p0),
                build.Layout(DATA_LO, 0x40, p0), build.Layout(DATA_LO, COPRO_LO, 0x3000),
                build.Layout(DATA_HI - 0x100, COPRO_LO, p0), build.Layout(DATA_LO, COPRO_HI - 0x40, p0),
                build.Layout(DATA_LO, COPRO_LO, build.PROGRAM_FREE[0] + build.PROGRAM_FREE[1] + 4),
                build.Layout(DATA_LO, COPRO_LO, p0, 0x100), build.Layout(DATA_LO, COPRO_LO, p0, POLY_LO + 2),
                build.Layout(DATA_LO, COPRO_LO, p0, None), build.Layout(DATA_LO, COPRO_LO, p0, POLY_HI + 4)):
        with pytest.raises(ConvertError, match="layout|Layout"):
            build.converted_course(rs, syn, SYN_EXE, 0, layout=lay)
    for bad in (None, "rs", b"\xff" * 16):
        with pytest.raises(ConvertError, match="RomSet"):
            build.converted_course(bad, syn, SYN_EXE, 0)
    with pytest.raises(ConvertError, match="CourseData"):
        build.converted_course(rs, "data", SYN_EXE, 0)
    for bad in (b"", None, "1ST_READ.BIN"):
        with pytest.raises(DcError):
            build.converted_course(rs, syn, bad, 0)

    c = build.converted_course(rs, syn, SYN_EXE, 0)
    build.encode(rs, c)
    cases = {
        "ai_lines": dataclasses.replace(c, ai_lines=()),                                  # unset: the slot's would
        "road": dataclasses.replace(c, road=()),                                          # be filled in
        "step_n": dataclasses.replace(c, step_n=None),
        "ground_data": dataclasses.replace(c, ground_data=None),
        "checkpoint_table": dataclasses.replace(c, checkpoint_table=None),
        "line table": dataclasses.replace(c, entry=dataclasses.replace(c.entry, line_table=b.entry.line_table)),
        "road line": dataclasses.replace(c, entry=dataclasses.replace(c.entry, record_table=b.entry.record_table)),
        "step table": dataclasses.replace(c, entry=dataclasses.replace(c.entry, step_table=b.entry.step_table)),
        "checkpoint table  ": dataclasses.replace(c, checkpoint_table=b.checkpoint_table),
        "ground": dataclasses.replace(c, ground_data=b.ground_data, ground=b.ground,
                                      entry=dataclasses.replace(c.entry, raw=b.entry.raw)),
        "AI lines": dataclasses.replace(c, ai_lines=b.ai_lines),                          # the slot's tables
        "time table ": dataclasses.replace(c, time_tables=b.time_tables),
        "checkpoint": dataclasses.replace(c, checkpoints=b.checkpoints),
        "spawn": dataclasses.replace(c, spawn=b.spawn),
        "slot": dataclasses.replace(c, course=1),
        "car_count": dataclasses.replace(c, car_count=(c.car_count[0], 20)),         # not a replaced field
        "spawn record": dataclasses.replace(c, spawn=tuple(tuple(x._replace(speed=x.speed + 1.0) for x in t)
                                                           for t in c.spawn)),       # not rewritten in place
        "time table": dataclasses.replace(c, time_tables={k: (p, t, (w[0] + 1,) + w[1:])
                                                          for k, (p, t, w) in c.time_tables.items()}),
    }
    for name, bad in cases.items():
        with pytest.raises(ConvertError, match=name.strip()):
            build.encode(rs, bad)
    for x, y in ((None, c), (rs, None), (rs, b"course")):
        with pytest.raises(ConvertError):
            build.encode(x, y)


def test_encode_fixes_the_slot(rs):
    """Review Focus 1 by address, not by table name: `encode` takes the record but for the moved tables' pointers,
    words 6-7 only as repoint_record_raw changes them, grid_base and the time tables at the slot's addresses, and
    every field outside RACE_FIELDS as the slot's. (Moving block_ref_a onto Expert's step table was accepted when the
    guard looked only at table names: the ground pointer word was written over Expert's table.)"""
    b, ex = _beginner(), G.decode_course(rs, 1)
    c = build.converted_course(rs, _syn(), SYN_EXE, 0)
    entry = c.entry
    cases = [("entry.block_ref_a", dict(entry=dataclasses.replace(entry, block_ref_a=ex.entry.step_table))),
             ("entry.block_ref_b", dict(entry=dataclasses.replace(entry, block_ref_b=ex.entry.step_table))),
             ("entry.upload_list", dict(entry=dataclasses.replace(entry, upload_list=ex.entry.upload_list))),
             ("entry.raw", dict(entry=dataclasses.replace(entry, raw=(1, 2)))),
             ("entry.raw", dict(entry=dataclasses.replace(entry, raw=b.entry.raw))),
             ("entry", dict(entry=dataclasses.replace(entry, course=np.int64(0)))),
             ("entry", dict(entry=dataclasses.replace(entry, course="0"))),
             ("grid_base", dict(grid_base=(ex.grid_base[0], 100))),
             ("rolling_start", dict(rolling_start=(ex.rolling_start[0], True))),
             ("rolling_start", dict(rolling_start=(c.rolling_start[0], 0))),
             ("time_tables", dict(time_tables={**c.time_tables, (0, 1): (c.time_tables[(0, 1)][0],
                                                                         ex.time_tables[(0, 1)][1],
                                                                         c.time_tables[(0, 1)][2])})),
             ("laps", dict(laps={**c.laps, 0: (c.laps[0][0], 3)})),
             ("billboards", dict(billboards=None)),
             ("objects", dict(objects={**c.objects, next(iter(c.objects)): None})),
             ("record_pointer", dict(record_pointer=np.int64(c.record_pointer))),
             ("step_n", dict(step_n=np.int64(c.step_n))),                       # a replaced field: plain values
             ("ground_data", dict(ground_data=dataclasses.replace(c.ground_data, raw=np.int64(3))))]
    for match, kw in cases:
        with pytest.raises(ConvertError, match=match):
            build.encode(rs, dataclasses.replace(c, **kw))
    for bad in (("step_n", 1), "steps", (), ["course"], ("entry",), None):
        with pytest.raises(ConvertError, match="fields"):
            build.encode(rs, c, bad)


def _junk(rng, value):
    """A junk value for a field holding `value`: a wrong type, a numpy value, an empty or truncated container, or
    the value with one element replaced."""
    pool = [None, True, -1, 0, 2 ** 70, 0.5, float("nan"), "", b"", (), [], {}, object(), np.int64(3),
            np.zeros(3), np.float32(1.0)]
    if isinstance(value, tuple) and value and rng.random() < 0.4:
        k = rng.randrange(len(value))
        return value[:k] + (rng.choice(pool),) + value[k + 1:]
    if isinstance(value, dict) and value and rng.random() < 0.4:
        k = rng.choice(list(value))
        return {**value, k: rng.choice(pool)}
    return rng.choice(pool)


def test_encode_never_raises_other_errors(rs):
    """Junk in any one field of the Course, its CourseEntry or its tgp.Ground raises a DecodeError (ConvertError),
    never a raw TypeError, ValueError, KeyError, IndexError, OverflowError or AttributeError."""
    rng = random.Random(3)
    c = build.converted_course(rs, _syn(), SYN_EXE, 0)
    targets = ([("course", f.name) for f in dataclasses.fields(G.Course)]
               + [("entry", f.name) for f in dataclasses.fields(C.CourseEntry)]
               + [("ground", f.name) for f in dataclasses.fields(tgp.Ground)])
    refused = 0
    for where, f in targets:
        if where == "course":
            bad = dataclasses.replace(c, **{f: _junk(rng, getattr(c, f))})
        elif where == "entry":
            bad = dataclasses.replace(c, entry=dataclasses.replace(c.entry, **{f: _junk(rng, getattr(c.entry, f))}))
        else:
            g = dataclasses.replace(c.ground_data, **{f: _junk(rng, getattr(c.ground_data, f))})
            bad = dataclasses.replace(c, ground_data=g)
        try:
            build.encode(rs, bad)
        except DecodeError:
            refused += 1
    assert refused >= len(targets) - 3, refused         # junk that happens to be legal is rare


def test_converted_never_raises_other_errors(rs):
    """Junk in any one argument (or a Layout field) raises a DecodeError (ConvertError, or a DcError from the exe
    reader), never a raw TypeError, ValueError, KeyError, IndexError, OverflowError or AttributeError."""
    rng = random.Random(6)
    junk = [None, True, False, -1, 0, 1, 2, 7, 8, 2 ** 40, -2 ** 70, 0.5, float("nan"), float("inf"), "", "keep",
            "empty", b"", b"\xff" * 8, (), [], {}, object(), np.int64(0), np.int64(-5), np.float32(1.0),
            np.bool_(True), DATA_LO, COPRO_LO, DATA_LO + 1, build.Layout(DATA_LO, COPRO_LO, 0)]
    syn = _syn()
    base = dict(rs=rs, d=syn, exe=SYN_EXE, n=0, slot=0, scenery="keep", layout=None)
    lay = build.Layout(DATA_LO, COPRO_LO, build.PROGRAM_FREE[0])
    converted = 0
    for _ in range(120):
        kw = dict(base)
        which = rng.choice(list(kw) + ["data_at", "copro_at", "program_at"])
        if which in LAYOUT_FIELDS:
            kw["layout"] = lay._replace(**{which: rng.choice(junk)})
        else:
            kw[which] = rng.choice(junk)
        try:
            build.converted_course(kw.pop("rs"), kw.pop("d"), kw.pop("exe"), kw.pop("n"), **kw)
            converted += 1
        except DecodeError:
            pass
    assert converted > 2                    # measured 4: legal junk (np.int64(0) for n or slot, 0 for n, a free
    #                                         layout field); most junk is refused before the tables are converted


# ---- the disc, the real set ----

@pytest.fixture(scope="module")
def desert(disc):
    """n 3 (Desert City) converted into Beginner's slot: (rs, course, set)."""
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    c = build.converted_course(rs, _data(disc, 3), disc.read(EXE_NAME), 3)
    return rs, c, build.converted_set(rs, disc, 3)


def test_grid_bases_from_exe(disc):
    """GRID_BASES is 1ST_READ.BIN's u16[8] at file offset 0x17e814 (repeated after it for the variants)."""
    exe = disc.read(EXE_NAME)
    assert struct.unpack_from("<8H", exe, build.GRID_BASES_AT) == build.GRID_BASES
    assert struct.unpack_from("<8H", exe, build.GRID_BASES_AT + 16) == build.GRID_BASES


def test_converted_round_trip(desert):
    """Review Focus 4: the converted set's slot 0 decodes to converted_course(...) on every field (every table field,
    the views and the record), check_course passes on both, and every table re-emits byte for byte. n 3's tables:
    465 road points, 527 steps, eight lines, checkpoints 136 and 263, grid base 458, 3,073 polygons; the layout's
    bytes per run are pinned (docs/spike/2026-11-convert.md, "Layout")."""
    rs, c, out = desert
    d = G.decode_course(out, 0)
    assert _differ(d, c) == []
    assert d.entry == c.entry and d.pit == c.pit and d.ground == c.ground
    G.check_course(c)
    G.check_course(d)
    assert G.mismatches(out, c) == []
    assert (len(c.road), c.step_n, len(c.steps), len(c.ai_lines)) == (465, 464, 527, 8)
    assert [cp.segment for cp in c.checkpoints] == [136, 263, 0]
    assert c.grid_base[1] == 458 and len(c.ground_data.polys) == 3073
    assert c.pit is not None and c.pit.box_polygon == 526
    # the layout: from each run's first byte, the tables back to back
    e, g = c.entry, c.ground_data
    spans = sorted([(g.grid_at, 0x8000), (g.attributes_at, 3076), (e.step_table, 8 + 4 * 527),
                    (e.record_table, 8 + 164 * 465), (e.line_table, 64 + 8 * 28 * 473),
                    (c.checkpoint_table - DATA_BASE, 12), (c.segment_flags[0] - DATA_BASE, 2 * 465 + 2)])
    assert spans[0][0] == DATA_LO and all(a + n == b for (a, n), (b, _) in zip(spans, spans[1:]))
    data_used = spans[-1][0] + spans[-1][1] - DATA_LO
    copro_used = 4 * (g.pool_base + 16 * 3073) - COPRO_LO
    # the list block 13 words on from COPRO_FREE's start: the pool then starts on a 16-word boundary, so no polygon
    # crosses the bank window at word 0x50000 (tgp.window_safe), as relocate_ground places Expert and Advanced
    assert (g.list_base, g.pool_base) == (COPRO_LO // 4 + 13, COPRO_LO // 4 + 13 + 8739)
    assert g.pool_base % 16 == 0 and g.pool_base < 0x50000 < g.pool_base + 16 * 3073
    assert (data_used, copro_used) == (220256 + 932, 52 + 231628)
    # mode 0 / difficulty 1 (the MAME runs'): 1 + 8 laps x 3 words, inside Beginner's own stored table
    ptr, table, words = c.time_tables[(C.DEFAULT_MODE, C.DEFAULT_DIFFICULTY)]
    assert len(words) == 25 and table + 4 * 25 <= min(t for _, t, _ in _beginner().time_tables.values() if t > table)


def _changed_outside(rs, out, names, spans=()) -> dict[str, np.ndarray]:
    """Review Focus 1: the byte offsets, per region, where `out` differs from `rs`; asserts each lies in a span of
    golden.coverage(rs, 0) whose table is one of `names`, in one of `spans` ((region, offset, length)), or in a free
    run (DATA_FREE, COPRO_FREE in both copro_data halves, the program and polygon runs) where it was 0xff. Data spans
    at 0x800000-0xffffff count in all eight 1 MB copies, copro_data spans in both halves."""
    allowed: dict[str, list] = {}
    for region, off, n, name in G.coverage(rs, 0):
        if name in names:
            allowed.setdefault(region, []).append((off, n))
    for region, off, n in spans:
        allowed.setdefault(region, []).append((off, n))
    allowed.setdefault("copro_data", []).extend([(off + HALF, n) for off, n in allowed.get("copro_data", [])])
    for off, n in list(allowed.get("data", [])):        # 0x800000-0xffffff: one 1 MB pair loaded eight times
        if off >= DATA_COPY_BASE:
            allowed["data"] += [(DATA_COPY_BASE + (off - DATA_COPY_BASE) % DATA_COPY_LEN + DATA_COPY_LEN * k, n)
                                for k in range(8)]
    free = {"data": [(DATA_LO, DATA_HI - DATA_LO)],
            "copro_data": [(COPRO_LO, COPRO_HI - COPRO_LO), (HALF + COPRO_LO, COPRO_HI - COPRO_LO)],
            "program": [(PROG_LO, PROG_HI - PROG_LO)], "polygons": [(POLY_LO, POLY_HI - POLY_LO)],
            "textures": [(TEX_LO, TEX_HI - TEX_LO)]}
    changed = {}
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a = np.frombuffer(getattr(rs, region), np.uint8)
        b = np.frombuffer(getattr(out, region), np.uint8)
        assert a.shape == b.shape
        idx = np.flatnonzero(a != b)
        ok = np.zeros(idx.shape, bool)
        for off, n in allowed.get(region, []):
            ok |= (idx >= off) & (idx < off + n)
        for off, n in free.get(region, []):
            ok |= (idx >= off) & (idx < off + n) & (a[idx] == 0xFF)
        assert ok.all(), (region, [hex(int(k)) for k in idx[~ok][:8]])
        changed[region] = idx
    return changed


def test_converted_only_free_bytes(desert):
    """Review Focus 1: every byte that differs between rs and the converted set lies in a span of
    golden.coverage(rs, 0) the conversion rewrites in place (IN_PLACE), or in DATA_FREE / COPRO_FREE (both copro_data
    halves) where it was 0xff. In the record only words 1, 4, 5 (step table, line table, road line) and 6-7 (the
    grid pointer) change."""
    rs, c, out = desert
    changed = _changed_outside(rs, out, IN_PLACE)
    record = C.COURSE_TABLE - DATA_BASE
    words = {int(k - record) // 4 for k in changed["data"] if record <= k < record + 32}
    assert words == {1, 4, 5, 6, 7}
    assert not len(changed["polygons"]) and not len(changed["textures"])
    assert {int(k) for k in changed["program"]} <= {C.GRID_BASES - 0x220000 + 0x20000 + k for k in (0, 1)}


def test_converted_leaves_other_courses(desert):
    """Expert and Advanced re-emit identically from the converted set (golden.mismatches empty), their grounds decode
    the same; the shared GRID_LANES, BILLBOARD_REF, both spawn tables (all 44 records) and the car models are
    byte-identical."""
    rs, c, out = desert
    for k in (1, 2):
        other = G.decode_course(rs, k)
        assert G.mismatches(out, other) == [], k
        assert tgp.decode_ground(out, k) == other.ground_data
    assert out.read_i960(C.GRID_LANES, 4) == rs.read_i960(C.GRID_LANES, 4)
    assert out.read_i960(C.BILLBOARD_REF, 12) == rs.read_i960(C.BILLBOARD_REF, 12)
    for t in C.SPAWN_TABLES:
        n = C.SPAWN_RECORD_SIZE * C.SPAWN_RECORDS
        assert out.read_i960(t, n) == rs.read_i960(t, n)
    b = _beginner()
    for key in b.car_objects:
        span = slice(4 * key, 4 * (key + b.objects[key].word_count))
        assert out.polygons[span] == rs.polygons[span], key


def test_converted_beginner_from_dc(disc, rs):
    """The known answer: n 0 converted into slot 0 decodes to Beginner's race tables: road line, steps, pit,
    checkpoints, time tables, laps, car count, start, grid base (GRID_BASES[0] = Beginner's 100) and spawn views
    exactly; the AI lines but for the speed (within 6e-6, as Phase 5 measured); the ground's grid, lists, list block
    and attributes exactly, its polygons but for the pinned off-road cells (OFF_ROAD_DIFFERENCES). Only the tables'
    addresses differ (moved into the free runs)."""
    out = build.converted_set(rs, disc, 0)
    got, want = G.decode_course(out, 0), _beginner()
    for f in ("step_n", "steps", "pit", "road", "checkpoints", "time_tables", "laps", "car_count", "rolling_start",
              "grid_base", "grid_lanes", "spawn"):
        assert getattr(got, f) == getattr(want, f), f
    assert got.segment_flags[1] == want.segment_flags[1]            # the values; the table moved (below)
    for line, ref in zip(got.ai_lines, want.ai_lines, strict=True):
        assert [p._replace(speed=0.0) for p in line] == [p._replace(speed=0.0) for p in ref]
        assert max(abs(p.speed - q.speed) for p, q in zip(line, ref)) < 6e-6
    g, w = got.ground_data, want.ground_data
    assert (g.grid, g.lists, g.raw, g.attributes) == (w.grid, w.lists, w.raw, w.attributes)
    road, moved, turned = OFF_ROAD_DIFFERENCES[0]
    pairs = list(zip(g.polys, w.polys, strict=True))
    assert sum((p.corners, p.centre) != (q.corners, q.centre) for p, q in pairs) == moved
    assert sum(p.direction != q.direction for p, q in pairs) == turned
    assert min(k for k, (p, q) in enumerate(pairs) if p != q) >= road
    assert (g.grid_at, g.list_base, got.entry.step_table) == (DATA_LO, COPRO_LO // 4, DATA_LO + 0x8000 + 1184)
    assert dataclasses.replace(got.entry, step_table=0, line_table=0, record_table=0, raw=()) == \
        dataclasses.replace(want.entry, step_table=0, line_table=0, record_table=0, raw=())
    assert G.mismatches(out, G.decode_course(rs, 1)) == [] and G.mismatches(out, G.decode_course(rs, 2)) == []
    # Review Focus 4 where only the address tells: n 0's tables are Beginner's, so a Course that keeps Beginner's own
    # line table (its address and lines) agrees with every other table and writes the set's bytes; still refused
    c = build.converted_course(rs, _data(disc, 0), disc.read(EXE_NAME), 0)
    for f, kw in (("line table", dict(ai_lines=want.ai_lines)), ("road line", {}), ("step table", {})):
        field = {"line table": "line_table", "road line": "record_table", "step table": "step_table"}[f]
        left = dataclasses.replace(c, entry=dataclasses.replace(c.entry, **{field: getattr(want.entry, field)}), **kw)
        G.check_course(left)
        with pytest.raises(ConvertError, match=f"the {f} at data\\+{getattr(want.entry, field):#x}.*not inside"):
            build.encode(rs, left)
    left = dataclasses.replace(c, segment_flags=want.segment_flags)     # Beginner's own table: its bytes, its place
    G.check_course(left)
    with pytest.raises(ConvertError, match=f"the segment flags at data\\+{want.segment_flags[0] - DATA_BASE:#x}.*not "
                                           f"inside"):
        build.encode(rs, left)


def test_converted_set_refuses(disc, rs):
    for n in (-1, 8, True, None, "3"):
        with pytest.raises(ConvertError, match="course"):
            build.converted_set(rs, disc, n)
    for bad in (None, "disc", 3):
        with pytest.raises(ConvertError, match="DcDisc"):
            build.converted_set(rs, bad, 3)
    with pytest.raises(ConvertError, match="slot"):
        build.converted_set(rs, disc, 3, slot=1)
    with pytest.raises(ConvertError, match="RomSet"):
        build.converted_set(None, disc, 3)


def test_converted_segment_flags(disc, rs, desert):
    """The per-segment flags (docs/spike/2026-11-convert.md, "Per-segment flags"). The Dreamcast data holds none (no
    section of CS_DATAn.BIN, and the arcade's values are not in 1ST_READ.BIN nor any other file of the disc), so the
    rule is the documented default: a converted course gets n + 1 zeros (`tables.segment_flags`). Where the answer is
    known: n 0 gives Beginner's flags and n 2 Advanced's exactly; n 1 differs from Expert's on exactly its 51 flagged
    segments 190-240 (SEGMENT_AI_FREE, with SEGMENT_TUNNEL on 208-218): the gap the rule leaves (Not found). n 3
    gives 465 zeros (n + 1), written into DATA_FREE after the checkpoint table, SEGMENT_FLAGS[0] repointed to them
    (Beginner's 196-value table and the id-3 stub's pointer to it stay); neither its flags nor its steps set
    SEGMENT_TUNNEL, so the tunnel's light zones (window 0x230742 / 0x23074a, read for the slot's id) stay dormant."""
    exe = disc.read(EXE_NAME)
    for n in range(3):
        want = G.decode_course(rs, n).segment_flags[1]
        got = tables.segment_flags(_data(disc, n))
        assert len(got) == len(want)
        assert [s for s, (a, b) in enumerate(zip(got, want)) if a != b] == ([] if n != 1 else list(range(190, 241)))
    c0 = build.converted_course(rs, _data(disc, 0), exe, 0)
    assert c0.segment_flags[1] == _beginner().segment_flags[1] == (0,) * 196
    _, c, out = desert
    at, flags = c.segment_flags
    assert flags == (0,) * 465 == tables.segment_flags(_data(disc, 3))
    assert at == c.checkpoint_table + 12 and DATA_LO <= at - DATA_BASE < DATA_HI
    assert struct.unpack("<I", out.read_i960(C.SEGMENT_FLAGS, 4))[0] == at
    assert C.segment_flags(out, G.decode_course(out, 0).entry) == flags
    assert out.read_i960(C.SEGMENT_FLAGS + 4, 12) == rs.read_i960(C.SEGMENT_FLAGS + 4, 12)    # Expert, Advanced, stub
    assert out.read_i960(_beginner().segment_flags[0], 2 * 196) == rs.read_i960(_beginner().segment_flags[0], 2 * 196)
    assert not any(f & C.SEGMENT_TUNNEL for f in flags) and not any(s.flags & C.SEGMENT_TUNNEL for s in c.steps)
    with pytest.raises(ConvertError, match="CourseData"):
        tables.segment_flags(None)


def test_standing_start(disc, rs, desert):
    """The start (docs/spike/2026-11-convert.md, "Desert City in MAME"): ROLLING_STARTS[n], a rolling start for n 0
    only (Beginner's own), a standing start for every other course. Stored as Expert and Advanced store theirs:
    rolling_start (ROLLING_STARTS[slot], False), the s16 -1 (course.encode_rolling_start). n 0 converted keeps
    Beginner's (ROLLING_STARTS[0], True) and its byte; n 3 converted is standing: the word at ROLLING_STARTS[0] is -1
    in the set, Expert's, Advanced's and the stub's words unchanged, and the grid's views are a standing start's (every
    car on GRID_LANES[(car & 1) ^ 1]: the player on 0x580; a rolling start puts it on 0x180). With Beginner's rolling
    start n 3's grid (base 458) sits past the countdown's steps (segments 435, 445, 455) and the race fires early."""
    assert build.ROLLING_STARTS == (True,) + (False,) * 7
    for k in (1, 2):                                            # the convention: Expert's and Advanced's
        assert G.decode_course(rs, k).rolling_start == (C.ROLLING_STARTS + 2 * k, False)
        assert rs.read_i960(C.ROLLING_STARTS + 2 * k, 2) == C.encode_rolling_start(False) == struct.pack("<h", -1)
    c0 = build.converted_course(rs, _data(disc, 0), disc.read(EXE_NAME), 0)
    assert c0.rolling_start == _beginner().rolling_start == (C.ROLLING_STARTS, True)
    assert c0.spawn == _beginner().spawn
    out0 = build.converted_set(rs, disc, 0)                     # the encoded word: Beginner's 0, rolling
    assert out0.read_i960(C.ROLLING_STARTS, 8) == rs.read_i960(C.ROLLING_STARTS, 8)
    assert out0.read_i960(C.ROLLING_STARTS, 2) == struct.pack("<h", 0) == C.encode_rolling_start(True)
    assert G.decode_course(out0, 0).rolling_start == (C.ROLLING_STARTS, True)
    _, c, out = desert
    assert c.rolling_start == (C.ROLLING_STARTS, False)
    assert out.read_i960(C.ROLLING_STARTS, 2) == struct.pack("<h", -1)
    assert rs.read_i960(C.ROLLING_STARTS, 2) == struct.pack("<h", 0)
    assert out.read_i960(C.ROLLING_STARTS + 2, 6) == rs.read_i960(C.ROLLING_STARTS + 2, 6)
    got = G.decode_course(out, 0)
    assert got.rolling_start == c.rolling_start and got.spawn == c.spawn
    lanes = c.grid_lanes
    assert all(s.lane == lanes[(s.car & 1) ^ 1] for slots in c.spawn for s in slots)
    assert c.spawn[0][0].segment == 458 and c.spawn[0][0].lane == lanes[1] == 0x580
    rolling = dataclasses.replace(c, rolling_start=(C.ROLLING_STARTS, True))       # the views follow the start
    with pytest.raises(ConvertError, match="spawn|grid"):
        build.encode(rs, rolling)


# ---- the empty scenery and the HUD map (Task 4) ----

SCENERY = build.RACE_FIELDS + build.SCENERY_FIELDS
SCENERY_IN_PLACE = frozenset({"cell grid", "actor table pointer", "HUD map"})


def _quad_widths(o: P.Object) -> list[float]:
    """The width of each drawn quad of an outline strip: its area over the mean length of its two long sides
    (P0(n-1)-P0(n) and P1(n-1)-P1(n); corners (P1(n-1), P0(n-1), P0(n), P1(n)), in x and z)."""
    out = []
    for q in P.polygons(o):
        if not q.drawn or len(q.points) != 4:
            continue
        a, b, c, d = ((p[0], p[2]) for p in q.points)
        area = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip((a, b, c, d), (b, c, d, a)))) / 2
        out.append(area / ((math.dist(b, c) + math.dist(a, d)) / 2))
    return out


def _check_outline(o: P.Object, road) -> None:
    """`o` is the outline of `road`: one edge (P0, P1) per kept road point (`build.outline_points`: all of a road of at
    most OUTLINE_QUADS points, else OUTLINE_QUADS, point 0 first) and the first again (a closed strip), each at y 0,
    its midpoint on that road point's centre (within 1e-3 in x and z: f32 rounding), OUTLINE_WIDTH long and square to
    the centre line there; every skipped road centre within OUTLINE_WIDTH / 2 of its chord (inside the stroke); every
    record a drawn link-2 quad with OUTLINE_ATTR and the normal (0, -1, 0); it encodes and decodes back (geometry
    mode 1, as the HUD map is read)."""
    cl = C.centreline(list(road))
    kept, deviation = build.outline_points([(x, z) for x, _, z in cl])
    assert len(kept) == min(len(road), build.OUTLINE_QUADS) and kept[0] == 0 and kept == sorted(set(kept))
    assert deviation <= build.OUTLINE_WIDTH / 2
    edges = [(o.seed0, o.seed1)] + [(r.p0, r.p1) for r in o.records]
    assert len(edges) == len(kept) + 1 and edges[-1] == edges[0]
    for k, (p0, p1) in zip(kept, edges[:-1]):
        c, before, after = cl[k], cl[k - 1], cl[(k + 1) % len(cl)]
        assert p0[1] == p1[1] == 0.0
        assert abs((p0[0] + p1[0]) / 2 - c[0]) < 1e-3 and abs((p0[2] + p1[2]) / 2 - c[2]) < 1e-3, k
        assert abs(math.dist((p0[0], p0[2]), (p1[0], p1[2])) - build.OUTLINE_WIDTH) < 1e-3, k
        tx, tz = after[0] - before[0], after[2] - before[2]
        assert abs(tx * (p1[0] - p0[0]) + tz * (p1[2] - p0[2])) < 1e-3 * math.hypot(tx, tz), k
    for a, b in zip(kept, kept[1:] + kept[:1]):                 # the skipped centres lie inside the stroke
        for j in range(1, (b - a) % len(cl)):
            assert build._chord_distance(cl[(a + j) % len(cl)][::2], cl[a][::2], cl[b][::2]) <= deviation + 1e-9
    assert all(r.attr == build.OUTLINE_ATTR and r.normal == (0.0, -1.0, 0.0) and r.p1 is not None
               for r in o.records)
    assert all(q.drawn for q in P.polygons(o)) and len(P.polygons(o)) == len(kept)
    words = P.encode_object(o)
    assert len(words) == o.word_count == 6 + 10 * len(kept) + 4
    assert P.decode_object(struct.pack(f"<{len(words)}I", *words), 0, C.OBC, C.HUD_MAP_MODE) == o


def _view_box(hud: C.HudMap, road, level: int = 1) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1): the road line's centre in view coordinates under the HUD map's place and zoom `level`:
    view = position + zoom * (x cos a - z sin a, x sin a + z cos a), a = angle (0x10000 a turn). At Beginner's
    angle 0x4000 that is the traced view x = -zoom z + tx, view y = zoom x + ty (docs/spike/2026-11-mirror.md)."""
    a = 2 * math.pi * hud.angle / 0x10000
    s, (tx, ty, _) = hud.zooms[level], hud.position
    pts = [(tx + s * (x * math.cos(a) - z * math.sin(a)), ty + s * (x * math.sin(a) + z * math.cos(a)))
           for x, _, z in C.centreline(list(road))]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def test_hud_outline(rs):
    """The outline's stroke and attribute are Beginner's, measured on Beginner's outline first: its drawn quads are
    a median 21.9 units wide (OUTLINE_WIDTH 22), its link-2 quads carry OUTLINE_ATTR with a texture-header step of
    1 (the outline's strip leaves it 0: every polygon reads the list's one header, untextured colour 1). The outline
    of a road line follows its centre (`_check_outline`): the synthetic one and Beginner's own. The frame rule
    applied to Beginner's own road gives Beginner's own place record and zooms."""
    b = _beginner()
    beginner = next(iter(b.hud_map.objects.values()))
    assert abs(statistics.median(_quad_widths(beginner)) - build.OUTLINE_WIDTH) < 0.5
    assert {r.attr for r in beginner.records if r.link == 2 and r.is_quad} == {build.OUTLINE_ATTR | 1 << 12}
    assert {r.normal for r in beginner.records if r.link} == {(0.0, -1.0, 0.0)}
    assert beginner.end == build.OUTLINE_END
    for road in (tables.road_line(_syn()), b.road):
        o = build.hud_outline(road)
        _check_outline(o, road)
        assert abs(statistics.median(_quad_widths(o)) - build.OUTLINE_WIDTH) < 0.5
    assert build.hud_frame(b.hud_map, b.road, b.road) == (b.hud_map.position, b.hud_map.angle, b.hud_map.zooms)
    # the frame of a road along z (Advanced's long axis): a quarter turn less than Beginner's, as Advanced's angle 0
    turned = [dataclasses.replace(p, lanes=tuple(ln._replace(position=(ln.position[2], ln.position[1],
                                                                        ln.position[0])) for ln in p.lanes))
              for p in b.road]
    position, angle, zooms = build.hud_frame(b.hud_map, b.road, turned)
    assert angle == 0 and zooms == b.hud_map.zooms
    got = _view_box(C.HudMap(0, 0, (), {}, position, angle, 0, zooms), turned)
    want = _view_box(b.hud_map, b.road)
    assert all(abs(g - w) < 1e-11 for g, w in zip(got, want))


def test_hud_outline_refuses(rs):
    """An empty road line, too short a loop, a point that is not a course.RoadPoint, a non-finite or unpackable
    position, used lanes out of range, two neighbours of a point at the same place: ConvertError (never a raw
    error). hud_frame: a reference that is not a HudMap or whose place or zooms are not finite numbers, a road it
    cannot frame (all at one point)."""
    road = list(tables.road_line(_syn()))
    p = road[3]

    def at(k, pos):
        lanes = list(p.lanes)
        lanes[k] = lanes[k]._replace(position=pos)
        return road[:3] + [dataclasses.replace(p, lanes=tuple(lanes))] + road[4:]

    lo = p.lane_lo
    bad = [(), [], None, "road", road[:2], road[:1], road + [None], road + [(1.0, 2.0, 3.0)],
           at(lo, (float("nan"), 0.0, 0.0)), at(lo, (float("inf"), 0.0, 0.0)), at(lo, (1e39, 0.0, 0.0)),
           at(lo, (1.0, 2.0)), at(lo, None), at(lo, ("1", 0.0, 0.0)), at(lo, (True, 0.0, 0.0)),
           road[:3] + [dataclasses.replace(p, lane_lo=9)] + road[4:],
           road[:3] + [dataclasses.replace(p, lane_hi=True)] + road[4:],
           road[:3] + [dataclasses.replace(p, lanes=p.lanes[:3])] + road[4:],
           road[:3] + [dataclasses.replace(p, lanes=None)] + road[4:],
           road[:3] + [dataclasses.replace(p, lanes=(None,) * 8)] + road[4:],
           [road[0], road[1], road[0], road[1]]]                  # point 1's neighbours coincide
    b = _beginner()
    for k, r in enumerate(bad):
        with pytest.raises(ConvertError, match="road|point|lane|outline"):
            build.hud_outline(r)
        if k < len(bad) - 1:                                    # the last one has a centre extent to frame
            with pytest.raises(ConvertError, match="HUD map"):
                build.hud_frame(b.hud_map, b.road, r)
    for ref in (None, "hud", dataclasses.replace(b.hud_map, position=(float("nan"), 0.0, 0.0)),
                dataclasses.replace(b.hud_map, position=None), dataclasses.replace(b.hud_map, zooms=(1.0,)),
                dataclasses.replace(b.hud_map, zooms=(0.0,) * 4), dataclasses.replace(b.hud_map, angle=1.5),
                dataclasses.replace(b.hud_map, angle=True), dataclasses.replace(b.hud_map, angle=0x10000)):
        with pytest.raises(ConvertError, match="HUD"):
            build.hud_frame(ref, b.road, road)
    flat = [dataclasses.replace(q, lanes=tuple(ln._replace(position=(1.0, 0.0, 1.0)) for ln in q.lanes))
            for q in road]
    with pytest.raises(ConvertError):
        build.hud_frame(b.hud_map, b.road, flat)
    with pytest.raises(ConvertError):
        build.hud_frame(b.hud_map, flat, road)


def test_outline_points():
    """The merge (Phase 7 Task 8 fix: the HUD outline's budget word counts against the frame cap, so an outline has at
    most OUTLINE_QUADS quads): a loop of at most `quads` points keeps all; a longer one keeps point 0, the farthest
    point and then the points farthest from their chords, so a square of 400 points keeps its four corners first and
    deviates 0; ConvertError for a loop under 3 points, a point that is not two finite numbers, a bad `quads`."""
    square = [(float(i), 0.0) for i in range(100)] + [(100.0, float(i)) for i in range(100)] + \
        [(100.0 - i, 100.0) for i in range(100)] + [(0.0, 100.0 - i) for i in range(100)]
    assert build.outline_points(square[::25], 64) == (list(range(16)), 0.0)
    kept, deviation = build.outline_points(square, 4)
    assert kept == [0, 100, 200, 300] and deviation < 1e-9
    kept, deviation = build.outline_points(square, 64)
    assert len(kept) == 64 and {0, 100, 200, 300} <= set(kept) and deviation < 1e-9       # f32-free: chords
    circle = [(100 * math.cos(2 * math.pi * k / 500), 100 * math.sin(2 * math.pi * k / 500)) for k in range(500)]
    kept, deviation = build.outline_points(circle, 64)
    assert len(kept) == 64 and 0 < deviation < 100 * (1 - math.cos(math.pi / 32)) + 1e-9   # under one 64-gon sag
    for bad, quads in ((square[:2], 64), ([(0.0, 0.0), (1.0, 1.0), (1.0,)], 64), ([(0.0, 0.0), (1.0, 1.0), None], 64),
                       ([(0.0, 0.0), (1.0, float("nan")), (2.0, 0.0)], 64), ("abc", 64), (square, True),
                       (square, 2), (square, 4.0)):
        with pytest.raises(ConvertError, match="HUD outline"):
            build.outline_points(bad, quads)


def test_empty_list_choice(rs):
    """The empty object list (Review Focus 5, the ruling): the object library holds no list that decodes to no
    placement (walked as back-to-back lists, every one of the 3,190 has an entry), so EMPTY_LIST is the shortest list
    of Beginner's that nothing else names: every Beginner-only list has one entry; the lowest of the 256 that only
    Beginner's grid names (cell 0's, a never-drawn link-0 stub in the far corner). Expert, Advanced, the id-3 stub's
    grid, the spawn records, the select screen's map objects and the HUD maps do not name it."""
    lo, hi = C.OBJECT_LIBRARY
    at, lists = lo, 0
    while at < hi:
        lists += 1
        n = 0
        while struct.unpack("<I", rs.read_i960(at, 4))[0]:
            at += C.OBJECT_ENTRY_SIZE
            n += 1
        at += 4
        assert n, f"an empty list at {at - 4:#x}"
    assert (at, lists) == (hi, 3190)
    courses = [G.decode_course(rs, k) for k in range(3)]
    b = courses[0]
    assert b.cells.count(build.EMPTY_LIST) == 1 and b.cells[0] == build.EMPTY_LIST
    assert len(b.object_lists[build.EMPTY_LIST]) == 1
    others = set()
    for k, c in enumerate(courses):
        others |= {c.map_object.objects, c.hud_map.outline_list}
        others |= {o for slots in c.spawn for s in slots for o in s.objects}
        if k:
            others |= set(c.object_lists)
    stub = struct.unpack("<I", rs.read_i960(C.GRID_TABLES + 12, 4))[0]
    others |= set(struct.unpack(f"<{C.GRID_CELLS}I", rs.read_i960(stub, 4 * C.GRID_CELLS)))
    assert build.EMPTY_LIST not in others
    named = set(b.cells) - others
    assert build.EMPTY_LIST == min(named) and len(named) == 256
    handlers = ({a.obj for acts in b.actor_lists.values() for a in acts if a.handler == C.STATIC_PROP}
                | {p for lists in b.billboards.values() for p in lists} | set(b.switch_lists))
    assert handlers and named.isdisjoint(handlers)            # only the grid names them, not Beginner's actors
    beginner_only = set(b.object_lists) - others
    assert min(len(b.object_lists[k]) for k in beginner_only) == 1


@pytest.fixture(scope="module")
def desert_empty(disc):
    """n 3 converted into Beginner's slot with the empty scenery: (rs, course, set)."""
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    c = build.converted_course(rs, _data(disc, 3), disc.read(EXE_NAME), 3, scenery="empty")
    return rs, c, build.converted_set(rs, disc, 3, scenery="empty")


def test_empty_scenery(desert_empty):
    """Review Focus 5: in the converted set GRID_TABLES[0] is still Beginner's grid, rewritten in place, its 256 cells
    naming EMPTY_LIST, which decodes to no placement; ACTOR_TABLES[0] names the shared empty table (build.EMPTY_ACTORS,
    the program run's first bytes, written by the conversion) whose 256 slots are 0; the HUD map's outline is the new
    course's (hud_outline of its road, in the polygon run, named by Beginner's list) with the place and zooms of
    hud_frame. Expert's and Advanced's cells and actors are unchanged and re-emit identically (golden.mismatches);
    Beginner's actor lists, billboard array and the switch's instructions stay in place unchanged (nothing spawns them).
    The set decodes to the converted Course on every field but objects, headers, points and texels, which keep the
    slot's (Beginner's scenery stays in the ROM, unchanged): the decoded ones are theirs restricted to what is reached,
    but for the HUD outline's points, which the new (longer) outline reads further on from the slot's outline tpa, over
    unchanged words."""
    rs, c, out = desert_empty
    b = _beginner()
    d = G.decode_course(out, 0)
    assert struct.unpack("<I", out.read_i960(C.GRID_TABLES, 4))[0] == b.grid == d.grid
    assert d.cells == (build.EMPTY_LIST,) * C.GRID_CELLS
    assert C.object_list(out, build.EMPTY_LIST) == [] and d.object_lists[build.EMPTY_LIST] == ()
    table = struct.unpack("<I", out.read_i960(C.ACTOR_TABLES, 4))[0]
    assert table == d.actor_table == build.EMPTY_ACTORS and table - 0x200000 == PROG_LO
    assert struct.unpack("<256I", out.read_i960(table, 1024)) == (0,) * 256 == d.actor_slots
    assert (d.actor_lists, d.unmodelled, d.billboards, d.billboard_ref, d.switch_lists) == ({}, (), {}, None, ())
    assert C.actors(out, d.entry) == []
    # the HUD map
    h = d.hud_map
    assert h.outline_list == b.hud_map.outline_list and len(h.entries) == 1
    (index, outline), = h.objects.items()
    assert outline == build.hud_outline(c.road) and h.entries[0].oba == 0x800000 | index
    assert POLY_LO <= 4 * index and 4 * (index + outline.word_count) <= POLY_HI
    assert h.entries[0].raw == (len(outline.records),) == (build.OUTLINE_QUADS,)      # 465 road points merged
    assert (h.entries[0].tpa, h.entries[0].tha) == (b.hud_map.entries[0].tpa, b.hud_map.entries[0].tha)
    assert (h.position, h.angle, h.zooms) == build.hud_frame(b.hud_map, b.road, c.road)
    _check_outline(outline, c.road)
    # the round trip
    assert _differ(d, c, ("objects", "headers", "points", "texels")) == []
    assert not {"objects", "headers", "points", "texels"} & set(_differ(c, b))     # the slot's
    hud_at = h.entries[0].tpa & T.TEX_ADDR_MASK
    assert set(d.points) <= set(b.points) and all(d.points[k] == b.points[k] for k in d.points if k != hud_at)
    assert d.points[hud_at] == T.texture_points(out, h.entries[0].tpa, outline) == \
        T.texture_points(rs, h.entries[0].tpa, outline)
    assert all(d.objects[k] == b.objects[k] for k in d.objects) and set(d.objects) == set(b.car_objects)
    assert set(d.headers) <= set(b.headers)
    G.check_course(d)
    assert G.mismatches(out, c) == []
    # Expert and Advanced, the shared tables, Beginner's handler data
    for k in (1, 2):
        other, now = G.decode_course(rs, k), G.decode_course(out, k)
        assert G.mismatches(out, other) == [], k
        assert (now.cells, now.object_lists, now.actor_slots, now.actor_lists) == \
            (other.cells, other.object_lists, other.actor_slots, other.actor_lists), k
    assert out.read_i960(C.GRID_LANES, 4) == rs.read_i960(C.GRID_LANES, 4)
    assert out.read_i960(C.BILLBOARD_REF, 12) == rs.read_i960(C.BILLBOARD_REF, 12)
    for t in C.SPAWN_TABLES:
        n = C.SPAWN_RECORD_SIZE * C.SPAWN_RECORDS
        assert out.read_i960(t, n) == rs.read_i960(t, n)
    for a, records in b.actor_lists.items():
        n = 4 + C.ACTOR_SIZE * len(records)
        assert out.read_i960(a, n) == rs.read_i960(a, n)
    for a in b.billboards:
        assert out.read_i960(a, 4 * C.BILLBOARD_VARIANTS) == rs.read_i960(a, 4 * C.BILLBOARD_VARIANTS)
    assert C.switch_lists(out) == b.switch_lists
    assert out.read_i960(b.actor_table, 1024) == rs.read_i960(b.actor_table, 1024)


def test_empty_scenery_only_free_bytes(desert_empty):
    """Review Focus 1 with the scenery: every changed byte is in a table rewritten in place (the race's, the cell
    grid, ACTOR_TABLES[0], the HUD map's list entry, place record and zooms), in EMPTY_LIST's first word (its
    terminator now), or was 0xff in a free run. The bytes used: the shared actor table's 1,024 (EMPTY_ACTORS, written
    by the first course that names it) in the program run, the outline's 2,600 (OUTLINE_QUADS quads merged from 465
    road points, 650 words) in the polygon run, each from the run's first byte (Phase 6 to Phase 7 Task 8: 465 quads,
    18,640 bytes)."""
    rs, c, out = desert_empty
    changed = _changed_outside(rs, out, IN_PLACE | SCENERY_IN_PLACE,
                               spans=[("data", build.EMPTY_LIST - DATA_BASE, 4)])
    assert out.program[PROG_LO:PROG_LO + 1024] == bytes(1024)
    assert {int(k) for k in changed["program"] if k >= PROG_LO} == set(range(PROG_LO, PROG_LO + 1024))
    (index, outline), = c.hud_map.objects.items()
    assert 4 * index == POLY_LO and outline.word_count == 650 and len(outline.records) == build.OUTLINE_QUADS
    assert int(changed["polygons"].min()) >= POLY_LO and int(changed["polygons"].max()) < POLY_LO + 2600
    assert not len(changed["textures"])
    assert c.actor_table == 0x220000 + PROG_LO - 0x20000


def test_frame_of_the_new_course(desert_empty):
    """The place record and zooms frame n 3 as Beginner's frame Beginner (the rule measured on Beginner: hud_frame
    gives Beginner's own record for Beginner's road): at zoom level 1 (the race's) the road line's centre fills the
    box Beginner's fills, centred where Beginner's is, its long axis vertical as Beginner's (n 3 runs along z: angle
    0, a quarter turn less than Beginner's 0x4000, as Advanced's); every zoom level scaled by one factor, the 0 of
    level 3 kept, the depth and pad Beginner's."""
    rs, c, out = desert_empty
    b, h = _beginner(), c.hud_map
    assert h.angle == 0 and (h.position[2], h.pad) == (b.hud_map.position[2], b.hud_map.pad)
    x0, y0, x1, y1 = _view_box(h, c.road)
    bx0, by0, bx1, by1 = _view_box(b.hud_map, b.road)
    assert abs((x0 + x1) - (bx0 + bx1)) < 2e-10 and abs((y0 + y1) - (by0 + by1)) < 2e-10
    w, ht, bw, bh = x1 - x0, y1 - y0, bx1 - bx0, by1 - by0
    assert w <= bw * (1 + 1e-6) and ht <= bh * (1 + 1e-6) and abs(max(w / bw, ht / bh) - 1) < 1e-6
    assert ht > w and bh > bw
    k = h.zooms[1] / b.hud_map.zooms[1]
    assert all(abs(z - k * zb) <= 1e-7 * zb for z, zb in zip(h.zooms, b.hud_map.zooms)) and h.zooms[3] == 0.0
    assert 0.49 < k < 0.50


def test_encode_scenery_fields(rs):
    """`encode` with the scenery fields (an extended field set): the empty scenery is written with
    RACE_FIELDS + SCENERY_FIELDS and refused with RACE_FIELDS alone (its cells differ from the slot's); the slot's
    own scenery is refused with them (Review Focus 5: no cell may name a list but EMPTY_LIST, no actor slot a list).
    Each scenery table is pinned to its place: the grid and the HUD map's list are the slot's, rewritten in place,
    the actor table the shared EMPTY_ACTORS (refused where its span holds other bytes), the outline inside its free
    run, only EMPTY_LIST changes among the object lists; the scenery fields are replaced together."""
    b = _beginner()
    c = build.converted_course(rs, _syn(), SYN_EXE, 0, scenery="empty")
    keep = build.converted_course(rs, _syn(), SYN_EXE, 0)
    out = build.encode(rs, c, SCENERY)
    assert G.decode_course(out, 0).cells == (build.EMPTY_LIST,) * 256
    assert build.encode(rs, c, list(SCENERY)).program == out.program
    with pytest.raises(ConvertError, match="cells differs"):
        build.encode(rs, c)
    with pytest.raises(ConvertError, match="cells"):
        build.encode(rs, keep, SCENERY)
    for part in (build.RACE_FIELDS + ("cells",), build.RACE_FIELDS + build.SCENERY_FIELDS[1:]):
        with pytest.raises(ConvertError, match="together"):
            build.encode(rs, c, part)
    h = c.hud_map
    (index, outline), = h.objects.items()
    entry = h.entries[0]
    moved = POLY_LO // 4 + 0x10000
    other = next(k for k in b.object_lists if k != build.EMPTY_LIST and k in c.object_lists)
    cases = [
        ("cells", dict(cells=(build.EMPTY_LIST,) * 255 + (b.cells[1],))),
        ("cells", dict(cells=list(c.cells))),
        # a slot naming a list: its actor lists are then unset (golden.unset_fields), refused first
        ("unset fields actor_lists", dict(actor_slots=(0,) * 255 + (next(s for s in b.actor_slots if s),))),
        ("unset fields actor_lists", dict(actor_slots=(False,) * 256)),
        ("actor_slots", dict(actor_slots=(0,) * 255 + (next(s for s in b.actor_slots if s),),
                             actor_lists=b.actor_lists, unmodelled=b.unmodelled)),
        ("actor_slots", dict(actor_slots=[0] * 256)),
        ("actor_lists", dict(actor_lists=b.actor_lists)),
        ("billboards", dict(billboards=b.billboards)),
        ("billboard_ref", dict(billboard_ref=b.billboard_ref)),
        ("switch_lists", dict(switch_lists=b.switch_lists)),
        ("unmodelled", dict(unmodelled=b.unmodelled)),
        ("actor_table", dict(actor_table=b.actor_table)),
        ("actor_table", dict(actor_table=0x220000 + PROG_HI - 0x20000 - 512)),
        ("actor_table", dict(actor_table=c.actor_table + 2)),
        ("object_lists", dict(object_lists={**c.object_lists, other: ()})),
        ("object_lists", dict(object_lists={**c.object_lists, build.EMPTY_LIST: b.object_lists[build.EMPTY_LIST]})),
        ("object_lists", dict(object_lists={**c.object_lists, b.cells[1]: ()})),
        ("object_lists", dict(object_lists={k: v for k, v in c.object_lists.items() if k != build.EMPTY_LIST})),
        ("HUD map", dict(hud_map=dataclasses.replace(h, outline_list=b.cells[1]))),
        ("HUD map", dict(hud_map=dataclasses.replace(h, entries=h.entries * 2))),
        ("HUD map", dict(hud_map=dataclasses.replace(h, entries=(dataclasses.replace(entry, tha=0),)))),
        ("HUD map", dict(hud_map=dataclasses.replace(h, course=1))),
        ("HUD map", dict(hud_map=dataclasses.replace(
            h, entries=(dataclasses.replace(entry, oba=0x800000 | 0x1000),), objects={0x1000: outline}))),
        ("HUD map", dict(hud_map=dataclasses.replace(
            h, entries=(dataclasses.replace(entry, oba=0x800000 | (POLY_HI // 4 - 8)),),
            objects={POLY_HI // 4 - 8: outline}))),
        ("HUD map", dict(hud_map=b.hud_map)),
        ("HUD map.*hud_outline", dict(hud_map=dataclasses.replace(h, objects={index: build.hud_outline(b.road)}))),
        ("HUD map.*hud_frame", dict(hud_map=dataclasses.replace(h, zooms=b.hud_map.zooms))),
        ("HUD map.*hud_frame", dict(hud_map=dataclasses.replace(h, angle=0))),
        ("HUD map.*hud_frame", dict(hud_map=dataclasses.replace(h, position=b.hud_map.position))),
    ]
    for match, kw in cases:
        with pytest.raises(ConvertError, match=match):
            build.encode(rs, dataclasses.replace(c, **kw), SCENERY)
    # a moved outline inside the run is accepted where it is free
    c2 = dataclasses.replace(c, hud_map=dataclasses.replace(
        h, entries=(dataclasses.replace(entry, oba=0x800000 | moved),), objects={moved: outline}))
    out2 = build.encode(rs, c2, SCENERY)
    assert G.decode_course(out2, 0).hud_map == c2.hud_map
    # a byte of the run that is not free is refused
    busy = rs.with_patch("polygons", POLY_LO + 8, b"\0" * 4)
    with pytest.raises(ConvertError, match="not free"):
        build.encode(busy, c, SCENERY)
    busy = rs.with_patch("program", PROG_LO + 8, b"\0" * 4)
    with pytest.raises(ConvertError, match="EMPTY_ACTORS"):
        build.encode(busy, c, SCENERY)
    # the actor table does not move: it is the shared empty one (Phase 10 Task 1), refused where it is not free
    c3 = build.converted_course(busy, _syn(), SYN_EXE, 0, scenery="empty")
    assert c3.actor_table == build.EMPTY_ACTORS == 0x220000 + PROG_LO - 0x20000
    with pytest.raises(ConvertError, match="EMPTY_ACTORS"):
        build.encode(busy, c3, SCENERY)


def test_encode_scenery_never_raises_other_errors(rs):
    """The extended-fields leak (Task 3's re-review): junk in any scenery field, in any field of its HUD map, its
    list entry, its outline object or the outline's first record raises a DecodeError (ConvertError), never a raw
    TypeError, ValueError, KeyError, IndexError, OverflowError or AttributeError: `encode` validates the scenery
    fields itself before golden runs."""
    rng = random.Random(4)
    c = build.converted_course(rs, _syn(), SYN_EXE, 0, scenery="empty")
    h = c.hud_map
    (index, outline), = h.objects.items()
    entry, rec = h.entries[0], outline.records[0]

    def with_hud(**kw):
        return dataclasses.replace(c, hud_map=dataclasses.replace(h, **kw))

    def with_outline(o):
        return with_hud(objects={index: o})

    targets = []
    for f in build.SCENERY_FIELDS:
        targets += [lambda f=f: dataclasses.replace(c, **{f: _junk(rng, getattr(c, f))})] * 3
    for f in dataclasses.fields(C.HudMap):
        targets += [lambda f=f: with_hud(**{f.name: _junk(rng, getattr(h, f.name))})] * 3
    for f in dataclasses.fields(C.Placement):
        targets += [lambda f=f: with_hud(entries=(dataclasses.replace(entry, **{f.name: _junk(
            rng, getattr(entry, f.name))}),))] * 2
    for f in dataclasses.fields(P.Object):
        targets += [lambda f=f: with_outline(dataclasses.replace(outline, **{f.name: _junk(
            rng, getattr(outline, f.name))}))] * 2
    for f in dataclasses.fields(P.Record):
        targets += [lambda f=f: with_outline(dataclasses.replace(outline, records=[dataclasses.replace(
            rec, **{f.name: _junk(rng, getattr(rec, f.name))})] + outline.records[1:]))] * 2
    targets += [lambda: with_outline(dataclasses.replace(outline, records=[None] + outline.records[1:])),
                lambda: with_hud(objects={index: None}), lambda: with_hud(objects={"x": outline}),
                lambda: with_hud(objects={index + 0.0: outline}), lambda: with_hud(entries=[entry]),
                lambda: dataclasses.replace(c, object_lists={**c.object_lists, "x": ()}),
                lambda: dataclasses.replace(c, object_lists={**c.object_lists, 1.5: ()}),
                lambda: dataclasses.replace(c, object_lists={**c.object_lists, build.EMPTY_LIST + 0.0: ()})]
    refused = 0
    for make in targets:
        try:
            build.encode(rs, make(), SCENERY)
        except DecodeError:
            refused += 1
    assert refused >= len(targets) - 8, (refused, len(targets))   # measured 6 legal: junk equal to the value, or in
    #                                                                 a field equality skips (Record.skipped, Object.end)


# ---- the model scenery: real scenery through `encode`, the object lists in DATA_FREE (Phase 7, Task 1) ----

MODEL = build.RACE_FIELDS + build.MODEL_FIELDS
MODEL_IN_PLACE = frozenset({"cell grid", "actor table pointer", "HUD map"})     # + texels in the course block
QUAD_ATTR = 0x010E0A01      # Beginner's double-sided link-2 quad (z max, texture parameter 3), header step 0
MAGENTA = 59                # palette entry 0x1000 + 59: (31, 0, 31), the polygon palette's purest magenta
QUAD_SEGMENT = 40           # Desert City's main straight (x -332 from segment 450 to 40, along +z): the quad's place
QUAD_SIZE = 10.0
QUAD_LIFT = 2.0             # its bottom edge above the road (+y is up: the view matrix, docs/spike "Object library")
# Where the synthetic scenery's tables go: the end of each run, clear of the race tables and the HUD outline.
LIST_AT = DATA_BASE + DATA_HI - 0x40                # the quad's list (20 bytes); the empty list (4 bytes) after it
EMPTY_AT = LIST_AT + 0x20
OBJECT_AT = POLY_HI // 4 - 0x40                     # the quad's 20 words
HEADER_AT = TEX_HI // 2 - 0x20                      # its header (4 words), then its texture points (8 words)
POINTS_AT = HEADER_AT + 8
TEX_AT = (0, 0)                                     # the textured quad's 32 x 32 texture in Beginner's course sheet


def _quad_object(x: float, y: float, z: float, size: float = QUAD_SIZE) -> P.Object:
    """An upright size x size quad facing -z (towards a car driving +z), its bottom edge at height y, centred on x at
    depth z: two seed points and one QUAD_ATTR record, the end record OUTLINE_END; single-precision values. The
    rasterizer's corners are (P1(n-1), P0(n-1), P0(n), P1(n)) = (seed 1, seed 0, the record's p0, p1): seed 0 the
    bottom left, seed 1 the bottom right, p0 the top left, p1 the top right go round the quad (p0 and p1 the other way
    draw a bow-tie: the first MAME run, docs/spike/2026-11-convert.md, "Object library")."""
    def f32(*v):
        return struct.unpack("<3f", struct.pack("<3f", *v))

    h = size / 2
    return P.Object(f32(x - h, y, z), f32(x + h, y, z),
                    [P.Record(QUAD_ATTR, (0.0, 0.0, 1.0), f32(x - h, y + size, z), f32(x + h, y + size, z))],
                    6 + 10 + 4, build.OUTLINE_END)


def _quad_header(textured: bool) -> T.TexHeader:
    """The quad's header: MAGENTA, luma table 0; textured: 32 x 32 at TEX_AT in sheet 0 (bit 11 set, as every
    textured course header); else flat colour."""
    return T.TexHeader(tha=HEADER_AT, step=0, width=32, height=32, tex_x=TEX_AT[0] if textured else 0,
                       tex_y=TEX_AT[1] if textured else 0, sheet=0, textured=textured, translucent=False,
                       checker=False, wrap_x=False, wrap_y=False, mirror_x=False, mirror_y=False, utex=False,
                       utex_min_lod=0, utex_x=0, utex_y=0, luma_base=0, colour=MAGENTA,
                       raw=(0, int(textured), 0))


def quad_scenery(x: float, y: float, z: float, *, textured: bool = False, size: float = QUAD_SIZE) -> build.Scenery:
    """A Scenery of one quad (`_quad_object`) in its cell (course.cell_of(x, z)), every other cell naming one empty
    list; textured: a 32 x 32 texture of seeded random indices (mips decimated) over the whole quad."""
    obj = _quad_object(x, y, z, size)
    cell = C.cell_of(x, z)
    lists = {LIST_AT: build.object_list_at(LIST_AT, [(0x800000 | OBJECT_AT, POINTS_AT, HEADER_AT, 1)]),
             EMPTY_AT: ()}
    header = _quad_header(textured)
    texels = ()
    if textured:
        idx = np.random.default_rng(7).integers(0, 16, (32, 32), dtype=np.uint8)
        texels = (G.Texels(header, idx, tuple(idx[::1 << lv, ::1 << lv].copy() for lv in T.mip_levels(header))),)
    points = (((256, 0), (256, 256), (0, 256), (0, 0)),)    # its record's (v, u) per corner, 1/8 texel: the texture
    return build.Scenery({OBJECT_AT: obj}, lists, tuple(LIST_AT if k == cell else EMPTY_AT for k in range(256)),
                         {HEADER_AT: header}, {POINTS_AT: points}, texels)


def desert_quad(road, *, textured: bool = False) -> build.Scenery:
    """The quad on Desert City's main straight: centred on road point QUAD_SEGMENT's centre, QUAD_LIFT above it."""
    x, y, z = road[QUAD_SEGMENT].centre
    return quad_scenery(x, y + QUAD_LIFT, z, textured=textured)


def test_list_home_and_bound():
    """Step 1's ruling (docs/spike/2026-11-convert.md, "Object library"): the lists live in DATA_FREE, the draw
    routines' lower bound patched down to DATA_FREE's start (two program words); the model scenery replaces the
    scenery fields and the objects, headers and texels; the textures run."""
    assert build.LIST_HOME == "data"
    assert C.LIBRARY_BOUND_AT == (0x17AD0, 0x17CFC) and C.LIBRARY_END_AT == (0x17AD8, 0x17D04)
    assert C.PATCHED_LIBRARY == (DATA_BASE + tgp.DATA_FREE[0], C.OBJECT_LIBRARY[1]) == (0x0265D120, 0x0284F4EC)
    assert C.LIBRARY_BOUND_PATCH == {0x17AD0: (0x0283FBB4, 0x0265D120), 0x17CFC: (0x0283FBB4, 0x0265D120)}
    assert build.MODEL_FIELDS == build.SCENERY_FIELDS + ("objects", "headers", "points", "texels")
    assert build.SCENERY == ("keep", "empty", "model")
    assert build.TEXTURES_FREE == (0xB52D48, 709304) and TEX_HI == 0xC00000


def test_library_bound_patch(rs):
    """The bound checks (disassembly, the discovery log): 0x17acc and 0x17cf8 each load the bounds with `lda` and
    return when the list pointer g0 is below the lower or at or past the upper. The patch writes the two lower-bound
    words only (6 bytes differ: the top byte stays 0x02); `library_bounds` reads OBJECT_LIBRARY from the original set
    and PATCHED_LIBRARY from the patched one; the three courses decode and re-emit the same on it; patching twice is
    a no-op. Refused: a set whose two routines disagree, a bound that is neither, not a RomSet."""
    from dtools.arcade import disasm
    for at in (0x17ACC, 0x17CF8):
        text = [ln.text for ln in disasm.listing(rs, at, 5)]
        assert text[:2] == ["lda     0x283fbb4,r3", "lda     0x284f4ec,r4"]
        assert text[2].startswith("cmpobl  g0,r3,") and text[3].startswith("cmpobge g0,r4,")
    assert C.library_bounds(rs) == C.OBJECT_LIBRARY
    p = C.patch_library_bound(rs)
    assert C.library_bounds(p) == C.PATCHED_LIBRARY and C.patch_library_bound(p) is p
    assert [ln.text for ln in disasm.listing(p, 0x17ACC, 1)] == ["lda     0x265d120,r3"]
    a, b = np.frombuffer(rs.program, np.uint8), np.frombuffer(p.program, np.uint8)
    assert np.flatnonzero(a != b).tolist() == [0x17AD0, 0x17AD1, 0x17AD2, 0x17CFC, 0x17CFD, 0x17CFE]
    for region in ("data", "polygons", "textures", "copro_data"):
        assert getattr(p, region) == getattr(rs, region)
    for k in range(3):
        assert G.mismatches(p, G.decode_course(p, k)) == [] and G.mismatches(p, G.decode_course(rs, k)) == []
    one = rs.with_patch("program", 0x17AD0, struct.pack("<I", C.PATCHED_LIBRARY[0]))
    with pytest.raises(DecodeError, match="different"):
        C.library_bounds(one)
    with pytest.raises(DecodeError, match="different"):
        C.patch_library_bound(one)
    odd = rs.with_patches({("program", at): struct.pack("<I", 0x02000000) for at in C.LIBRARY_BOUND_AT})
    with pytest.raises(DecodeError, match="not the original"):
        C.patch_library_bound(odd)
    with pytest.raises(DecodeError, match="lda"):
        C.library_bounds(rs.with_patch("program", 0x17ACC, b"\0" * 4))
    for bad in (None, b"rs"):
        with pytest.raises(DecodeError, match="RomSet"):
            C.library_bounds(bad)
    # a list in DATA_FREE: drawn (decoded) only on the patched set; the cell grid encoder admits it
    assert C.encode_cell_grid([LIST_AT] * C.GRID_CELLS) == struct.pack("<I", LIST_AT) * C.GRID_CELLS
    with pytest.raises(DecodeError, match="outside"):
        C.encode_cell_grid([C.PATCHED_LIBRARY[0] - 4] * C.GRID_CELLS)
    with pytest.raises(DecodeError, match="outside the object library"):
        C.object_list(rs, LIST_AT)
    assert C.object_list(p.with_patch("data", LIST_AT - DATA_BASE, bytes(4)), LIST_AT) == []


@pytest.fixture(scope="module")
def desert_model(disc):
    """n 3 converted into Beginner's slot with the model scenery of one textured quad on the main straight:
    (rs, course, set, scenery)."""
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    d, exe = _data(disc, 3), disc.read(EXE_NAME)
    sd = desert_quad(tables.road_line(d), textured=True)
    c = build.converted_course(rs, d, exe, 3, scenery="model", scenery_data=sd)
    return rs, c, build.converted_set(rs, disc, 3, scenery="model", scenery_data=sd), sd


def test_model_scenery_round_trip(desert_model):
    """A synthetic Scenery of one textured quad in one cell (Desert City's main straight), converted_set with
    scenery="model": the set decodes to the converted Course on every field (check_course passes, every table
    re-emits); its cells name the quad's list in DATA_FREE (its cell) and the empty list (the 255 others); the
    Course's objects are those the set reaches: the quad and the car models (Beginner's scenery is not reached), its
    headers the quad's (tha and step as decoded) and the cars' and the HUD outline's, its texels the quad's texture
    and the cars' (sheet 1); the texture points read back; the bound is patched; the actor table and HUD map are the
    empty scenery's."""
    rs, c, out, sd = desert_model
    b = _beginner()
    d = G.decode_course(out, 0)
    assert _differ(d, c) == []
    assert d.entry == c.entry and d.pit == c.pit and d.ground == c.ground
    G.check_course(c)
    G.check_course(d)
    assert G.mismatches(out, c) == []
    assert C.library_bounds(out) == C.PATCHED_LIBRARY and C.library_bounds(rs) == C.OBJECT_LIBRARY
    cell = C.cell_of(c.road[QUAD_SEGMENT].centre[0], c.road[QUAD_SEGMENT].centre[2])
    assert d.cells == sd.cells and d.cells[cell] == LIST_AT and set(d.cells) == {LIST_AT, EMPTY_AT}
    assert d.cells.count(LIST_AT) == 1
    assert d.object_lists[LIST_AT] == sd.lists[LIST_AT] and d.object_lists[EMPTY_AT] == ()
    assert set(d.object_lists) == {LIST_AT, EMPTY_AT} | build._slot_lists(b)
    assert set(d.objects) == set(b.car_objects) | {OBJECT_AT} and d.objects[OBJECT_AT] == sd.objects[OBJECT_AT]
    assert all(d.objects[k] == b.objects[k] for k in b.car_objects)
    assert d.headers[HEADER_AT] == sd.headers[HEADER_AT]                     # tha and step as the walk sends them
    quad = [t for t in d.texels if t.header.sheet == 0 and t.header.textured]
    assert len(quad) == 1 and quad[0].header == sd.headers[HEADER_AT]
    assert np.array_equal(quad[0].indices, sd.texels[0].indices)
    assert all(np.array_equal(m, n) for m, n in zip(quad[0].mips, sd.texels[0].mips, strict=True))
    assert {t.header.sheet for t in d.texels if t is not quad[0]} == {1}       # the car models' (sheet 1)
    assert out.textures[2 * POINTS_AT:2 * POINTS_AT + 16] == struct.pack("<8H", 256, 0, 256, 256, 0, 256, 0, 0)
    assert d.points[POINTS_AT] == sd.points[POINTS_AT] == c.points[POINTS_AT]
    assert set(d.points) == {POINTS_AT} | {p.tpa & T.TEX_ADDR_MASK for k in build._slot_lists(b) if k in d.object_lists
                                           for p in d.object_lists[k] if C.check_rom_object(p.oba) in d.objects} | \
        {p.tpa & T.TEX_ADDR_MASK for p in d.hud_map.entries}
    assert (d.actor_slots, d.actor_lists, d.billboards, d.switch_lists) == ((0,) * 256, {}, {}, ())
    assert d.hud_map.objects == {k: build.hud_outline(c.road) for k in d.hud_map.objects}
    # the object list as the game walks it: one entry (oba, tpa, tha, budget 1), then the terminator
    assert out.read_i960(LIST_AT, 20) == struct.pack("<5I", 0x800000 | OBJECT_AT, POINTS_AT, HEADER_AT, 1, 0)
    assert P.polygons(P.decode_object(out.polygons, OBJECT_AT, C.OBC, G.GEOMETRY_MODE)) == \
        P.polygons(sd.objects[OBJECT_AT])


def test_model_scenery_only_free_bytes(desert_model):
    """Review Focus 4 with the model scenery: every changed byte is in a table rewritten in place (the race's, the
    cell grid, ACTOR_TABLES[0], the HUD map's list entry, place record and zooms), in the bound patch's two words, in
    Beginner's 1 MB course sheet block (the quad's texels and mips), or was 0xff in a free run (DATA_FREE: the race
    tables and the two lists; the program run: the actor table; the polygon run: the outline and the quad; the
    textures run: the quad's header and points). The program changes only there; the common sheet block (sheet 1) and
    Expert's and Advanced's blocks are byte-identical, and Expert and Advanced re-emit identically; Beginner's
    EMPTY_LIST is not rewritten (the empty list is the scenery's own)."""
    rs, c, out, sd = desert_model
    block = c.sheet_sources[0]
    assert block == 0x200000
    changed = _changed_outside(rs, out, IN_PLACE | MODEL_IN_PLACE,
                               spans=[("program", at, 4) for at in C.LIBRARY_BOUND_AT] + [("data", block, 0x100000)])
    prog = {int(k) for k in changed["program"] if k < PROG_LO}
    bound = {0x17AD0, 0x17AD1, 0x17AD2, 0x17CFC, 0x17CFD, 0x17CFE}
    assert bound <= prog <= bound | {C.GRID_BASES - 0x200000 + k for k in (0, 1)} | \
        {C.ACTOR_TABLES - 0x200000 + k for k in range(4)}
    tex = {int(k) for k in changed["textures"]}
    assert tex <= set(range(2 * HEADER_AT, 2 * HEADER_AT + 8)) | set(range(2 * POINTS_AT, 2 * POINTS_AT + 16))
    assert out.textures[2 * HEADER_AT:2 * HEADER_AT + 8] == T.encode_tex_header(sd.headers[HEADER_AT])
    poly = changed["polygons"]
    (index, outline), = c.hud_map.objects.items()
    assert 4 * index == POLY_LO
    assert {int(k) for k in poly} <= set(range(POLY_LO, POLY_LO + 4 * outline.word_count)) | \
        set(range(4 * OBJECT_AT, 4 * OBJECT_AT + 80))
    texels = {int(k) for k in changed["data"] if block <= k < block + 0x100000}
    assert texels and out.data[block:block + 0x100000] != rs.data[block:block + 0x100000]
    for other in (T.sheet_source(rs, 1), T.sheet_source(rs, 0, 1), T.sheet_source(rs, 0, 2)):
        assert out.data[other:other + 0x100000] == rs.data[other:other + 0x100000], hex(other)
    assert out.read_i960(build.EMPTY_LIST, 20) == rs.read_i960(build.EMPTY_LIST, 20)
    assert out.read_i960(C.GRID_TABLES, 4) == rs.read_i960(C.GRID_TABLES, 4)          # the grid is rewritten in place
    lists = {int(k) for k in changed["data"] if DATA_LO <= k < DATA_HI and k >= LIST_AT - DATA_BASE}
    assert lists <= set(range(LIST_AT - DATA_BASE, EMPTY_AT - DATA_BASE + 4))
    for k in (1, 2):
        assert G.mismatches(out, G.decode_course(rs, k)) == [], k


def test_model_scenery_synthetic(rs):
    """The model scenery on the synthetic course (no disc): the flat-colour quad and the textured one write and
    decode back; the race tables and the HUD outline are laid out around the scenery's tables (a scenery list at
    DATA_FREE's first byte moves the race tables past it); the bound patch is applied only with the model scenery."""
    syn = _syn()
    for textured in (False, True):
        sd = quad_scenery(0.0, 2.0, 0.0, textured=textured)
        c = build.converted_course(rs, syn, SYN_EXE, 0, scenery="model", scenery_data=sd)
        out = build.encode(rs, c, MODEL)
        d = G.decode_course(out, 0)
        assert _differ(d, c) == [] and d.objects[OBJECT_AT] == sd.objects[OBJECT_AT]
        assert C.library_bounds(out) == C.PATCHED_LIBRARY
        assert len([t for t in d.texels if t.header.sheet == 0]) == int(textured)
    for scenery in ("keep", "empty"):
        out = build.encode(rs, build.converted_course(rs, syn, SYN_EXE, 0, scenery=scenery),
                           build.RACE_FIELDS + (build.SCENERY_FIELDS if scenery == "empty" else ()))
        assert C.library_bounds(out) == C.OBJECT_LIBRARY
    first = DATA_BASE + DATA_LO
    sd = quad_scenery(0.0, 2.0, 0.0)
    moved = build.Scenery(sd.objects, {first: build.object_list_at(first, [(0x800000 | OBJECT_AT, POINTS_AT,
                                                                            HEADER_AT, 1)]), EMPTY_AT: ()},
                          tuple(first if k == sd.cells.index(LIST_AT) else EMPTY_AT for k in range(256)),
                          sd.headers, sd.points, sd.texels)
    c = build.converted_course(rs, syn, SYN_EXE, 0, scenery="model", scenery_data=moved)
    assert c.ground_data.grid_at == DATA_LO + 20                        # past the 20-byte list
    assert G.decode_course(build.encode(rs, c, MODEL), 0).object_lists[first] == \
        moved.lists[first]
    with pytest.raises(ConvertError, match="not free"):                 # a layout over the scenery's list
        build.converted_course(rs, syn, SYN_EXE, 0, scenery="model", scenery_data=moved,
                               layout=build.Layout(DATA_LO, COPRO_LO, PROG_LO))


def test_model_scenery_refuses(rs):
    """Each scenery table pinned to its place (ConvertError naming it): a list outside the list home (in the object
    library, past DATA_FREE's end), an object outside POLYGON_FREE, a header outside the textures run (a word the
    slot's headers do not hold) or a textured one in sheet 1, texture points outside the textures run, of the wrong
    length, unreached or over the header, texels missing or rewritten in sheet 1, a placement not as decode_course
    reads it back or reading texture RAM, objects unreached or without normals (geometry mode 1), cells naming a
    list the scenery lacks or one of the slot's spawn lists; `scenery_data` only with scenery="model"; `encode`'s
    model fields together (the texture points among them: of the reaching objects' lengths, the slot's
    unchanged)."""
    syn, b = _syn(), _beginner()
    sd = quad_scenery(0.0, 2.0, 0.0, textured=True)
    ok = build.converted_course(rs, syn, SYN_EXE, 0, scenery="model", scenery_data=sd)
    build.encode(rs, ok, MODEL)
    entry = (0x800000 | OBJECT_AT, POINTS_AT, HEADER_AT, 1)
    cell = sd.cells.index(LIST_AT)

    def with_(**kw):
        return dataclasses.replace(sd, **kw)

    def at(addr, e=entry):
        return with_(lists={addr: build.object_list_at(addr, [e]), EMPTY_AT: ()},
                     cells=tuple(addr if k == cell else EMPTY_AT for k in range(256)))

    def entries(**kw):
        p = dataclasses.replace(sd.lists[LIST_AT][0], **kw)
        return with_(lists={**sd.lists, LIST_AT: (p,)})

    free_key = next(k for k in range(TEX_LO // 2 - 0x100000, TEX_LO // 2, 4) if k not in b.headers)
    quad, header, texel = sd.objects[OBJECT_AT], sd.headers[HEADER_AT], sd.texels[0]
    spawn = next(iter(build._slot_lists(b)))
    cases = [
        ("list home", lambda: at(b.cells[1])),
        ("list home", lambda: at(DATA_BASE + DATA_HI - 16)),
        ("list home", lambda: at(DATA_BASE + DATA_LO - 0x20)),
        ("POLYGON_FREE", lambda: with_(objects={0x1000: quad}, lists={**sd.lists, LIST_AT: build.object_list_at(
            LIST_AT, [(0x801000, POINTS_AT, HEADER_AT, 1)])})),
        ("POLYGON_FREE", lambda: with_(objects={POLY_HI // 4 - 10: quad}, lists={**sd.lists, LIST_AT: build.object_list_at(
            LIST_AT, [(0x800000 | (POLY_HI // 4 - 10), POINTS_AT, HEADER_AT, 1)])})),
        ("textures run", lambda: with_(headers={free_key: header}, lists={**sd.lists, LIST_AT: build.object_list_at(
            LIST_AT, [(0x800000 | OBJECT_AT, POINTS_AT, free_key, 1)])})),
        ("sheet|no texels", lambda: with_(headers={HEADER_AT: dataclasses.replace(header, sheet=1)})),
        ("points", lambda: with_(points={POINTS_AT: (((0, 0),) * 3,)})),                 # a quad given 3 points
        ("points", lambda: with_(points={POINTS_AT: (((0, 0),) * 4,) * 2})),             # 2 polygons, 1 record
        ("points", lambda: with_(points={POINTS_AT: (((0, 0x10000),) * 4,)})),
        ("points", lambda: with_(points={POINTS_AT: [((0, 0),) * 4]})),                  # a list: not as decoded
        ("points", lambda: with_(points={POINTS_AT: (((np.uint16(0), 0),) * 4,)})),
        ("points", lambda: with_(points={**sd.points, POINTS_AT + 64: (((0, 0),) * 4,)})),
        ("no texture points", lambda: with_(points={})),
        ("points", lambda: with_(points={HEADER_AT: sd.points[POINTS_AT]}, lists={**sd.lists, LIST_AT:
            build.object_list_at(LIST_AT, [(0x800000 | OBJECT_AT, HEADER_AT, HEADER_AT, 1)])})),
        ("points", lambda: with_(points={TEX_LO // 2 - 2: sd.points[POINTS_AT]}, lists={**sd.lists, LIST_AT:
            build.object_list_at(LIST_AT, [(0x800000 | OBJECT_AT, TEX_LO // 2 - 2, HEADER_AT, 1)])})),
        ("no texels", lambda: with_(texels=())),
        ("neither", lambda: with_(headers={})),
        ("texture RAM", lambda: entries(tpa=POINTS_AT | 0x800000)),
        ("decode_course reads", lambda: entries(cell=cell)),
        ("decode_course reads", lambda: entries(entry=0)),
        ("decode_course reads", lambda: entries(in_world=False)),
        ("budget|u32", lambda: entries(raw=(1, 2))),
        ("not one of the scenery's objects", lambda: entries(oba=0x800000 | (OBJECT_AT - 0x100))),
        ("not reached", lambda: with_(objects={**sd.objects, OBJECT_AT - 0x40: quad})),
        ("normal-less", lambda: with_(objects={OBJECT_AT: dataclasses.replace(quad, records=[
            dataclasses.replace(quad.records[0], normal=None, skipped=(0, 0, 0))], end=(0,), word_count=17)})),
        ("cells", lambda: with_(cells=sd.cells[:255])),
        ("cells", lambda: with_(cells=sd.cells[:255] + (LIST_AT + 0x100,))),
        ("not reached", lambda: with_(lists={**sd.lists, LIST_AT - 0x20: ()})),
        ("Scenery", lambda: None),
        ("overlap in DATA_FREE", lambda: with_(lists={LIST_AT: build.object_list_at(LIST_AT, [entry]),
                                                      LIST_AT + 8: build.object_list_at(LIST_AT + 8, [entry]),
                                                      EMPTY_AT: ()},
                                               cells=tuple(LIST_AT if k == cell else LIST_AT + 8 if k == cell + 1
                                                           else EMPTY_AT for k in range(256)))),
        ("overlap in the polygon run", lambda: with_(
            objects={OBJECT_AT: quad, OBJECT_AT + 4: quad}, points={POINTS_AT: sd.points[POINTS_AT]},
            lists={LIST_AT - 0x10: (), LIST_AT: build.object_list_at(LIST_AT, [entry, (
                0x800000 | (OBJECT_AT + 4), POINTS_AT, HEADER_AT, 1)])},                  # 36 bytes: the empty list
            cells=tuple(LIST_AT if k == cell else LIST_AT - 0x10 for k in range(256)))),   # moved before it
    ]
    for match, make in cases:
        with pytest.raises(ConvertError, match=match):
            got = make()
            build.converted_course(rs, syn, SYN_EXE, 0, scenery="model",
                                   scenery_data=got if got is not None else "scenery")
    with pytest.raises(ConvertError, match="scenery_data"):
        build.converted_course(rs, syn, SYN_EXE, 0, scenery="model")
    for scenery in ("keep", "empty"):
        with pytest.raises(ConvertError, match="scenery_data"):
            build.converted_course(rs, syn, SYN_EXE, 0, scenery=scenery, scenery_data=sd)
    # encode: the model fields together, the points among them; texels written only in the slot's course block
    with pytest.raises(ConvertError, match="together"):
        build.encode(rs, ok, build.RACE_FIELDS + build.SCENERY_FIELDS + ("objects", "headers", "texels"))
    with pytest.raises(ConvertError, match="together"):
        build.encode(rs, ok, build.RACE_FIELDS + build.SCENERY_FIELDS + ("objects",))
    for bad in ({}, {**ok.points, POINTS_AT: (((0, 0),) * 3,)}, {**ok.points, POINTS_AT + 64: ok.points[POINTS_AT]},
                {k: v for k, v in ok.points.items() if k != POINTS_AT}):
        with pytest.raises(ConvertError, match="points"):
            build.encode(rs, dataclasses.replace(ok, points=bad), MODEL)
    car_at = next(k for k in ok.points if k != POINTS_AT)              # a slot's points (a car model's): not rewritten
    with pytest.raises(ConvertError, match="texture points.*differs from the set's"):
        build.encode(rs, dataclasses.replace(ok, points={**ok.points, car_at: tuple(
            tuple((v ^ 8, u) for v, u in q) for q in ok.points[car_at])}), MODEL)
    empty = build.converted_course(rs, syn, SYN_EXE, 0, scenery="empty")
    with pytest.raises(ConvertError, match="points"):                   # the empty scenery keeps the slot's points
        build.encode(rs, dataclasses.replace(empty, points=ok.points), SCENERY)
    car = next(t for t in ok.texels if t.header.sheet == 1)
    rewritten = tuple(t._replace(indices=(t.indices + 1) % 16) if t is car else t for t in ok.texels)
    with pytest.raises(ConvertError, match="sheet 0"):
        build.encode(rs, dataclasses.replace(ok, texels=rewritten), MODEL)
    with pytest.raises(ConvertError, match="cells"):
        build.encode(rs, dataclasses.replace(ok, cells=(EMPTY_AT,) * 256), MODEL)
    with pytest.raises(ConvertError, match="cells"):                       # a spawn list (the car models) as a cell's
        build.encode(rs, dataclasses.replace(ok, cells=ok.cells[:255] + (spawn,)), MODEL)
    with pytest.raises(ConvertError, match="headers"):
        build.encode(rs, dataclasses.replace(ok, headers={**ok.headers, HEADER_AT: dataclasses.replace(
            header, step=1)}), MODEL)
    with pytest.raises(ConvertError, match="texels"):
        build.encode(rs, dataclasses.replace(ok, texels=ok.texels[::-1]), MODEL)
    (outline_at, _), = ok.hud_map.objects.items()                     # the quad over the HUD outline
    over = dataclasses.replace(ok, objects={**{k: v for k, v in ok.objects.items() if k != OBJECT_AT},
                                            outline_at: quad},
                               object_lists={**ok.object_lists, LIST_AT: build.object_list_at(
                                   LIST_AT, [(0x800000 | outline_at, POINTS_AT, HEADER_AT, 1)])})
    with pytest.raises(ConvertError, match="HUD outline.*overlap"):
        build.encode(rs, over, MODEL)
    busy = rs.with_patch("textures", 2 * POINTS_AT, b"\0\0")
    with pytest.raises(ConvertError, match="not free"):
        build.encode(busy, build.converted_course(busy, syn, SYN_EXE, 0, scenery="model", scenery_data=sd), MODEL)


def test_model_scenery_never_raises_other_errors(rs):
    """Junk in any field of the Scenery, of its object, record, placement, header or texels, or in any MODEL field
    of the converted Course raises a DecodeError (ConvertError), never a raw TypeError, ValueError, KeyError,
    IndexError, OverflowError or AttributeError."""
    rng = random.Random(5)
    syn = _syn()
    sd = quad_scenery(0.0, 2.0, 0.0, textured=True)
    ok = build.converted_course(rs, syn, SYN_EXE, 0, scenery="model", scenery_data=sd)
    quad, header, texel, place = sd.objects[OBJECT_AT], sd.headers[HEADER_AT], sd.texels[0], sd.lists[LIST_AT][0]

    def course(make):
        return lambda: build.converted_course(rs, syn, SYN_EXE, 0, scenery="model", scenery_data=make())

    def enc(**kw):
        return lambda: build.encode(rs, dataclasses.replace(ok, **kw), MODEL)

    targets = []
    for f in dataclasses.fields(build.Scenery):
        targets += [course(lambda f=f: dataclasses.replace(sd, **{f.name: _junk(rng, getattr(sd, f.name))}))] * 3
    for f in dataclasses.fields(P.Object):
        targets += [course(lambda f=f: dataclasses.replace(sd, objects={OBJECT_AT: dataclasses.replace(
            quad, **{f.name: _junk(rng, getattr(quad, f.name))})}))] * 2
    for f in dataclasses.fields(P.Record):
        targets += [course(lambda f=f: dataclasses.replace(sd, objects={OBJECT_AT: dataclasses.replace(
            quad, records=[dataclasses.replace(quad.records[0], **{f.name: _junk(
                rng, getattr(quad.records[0], f.name))})])}))] * 2
    for f in dataclasses.fields(C.Placement):
        targets += [course(lambda f=f: dataclasses.replace(sd, lists={**sd.lists, LIST_AT: (dataclasses.replace(
            place, **{f.name: _junk(rng, getattr(place, f.name))}),)}))] * 2
    for f in dataclasses.fields(T.TexHeader):
        targets += [course(lambda f=f: dataclasses.replace(sd, headers={HEADER_AT: dataclasses.replace(
            header, **{f.name: _junk(rng, getattr(header, f.name))})}))] * 2
    for f in G.Texels._fields:
        targets += [course(lambda f=f: dataclasses.replace(sd, texels=(texel._replace(**{f: _junk(
            rng, getattr(texel, f))}),)))] * 2
    targets += [course(lambda: dataclasses.replace(sd, points={POINTS_AT: _junk(rng, sd.points[POINTS_AT])}))] * 4
    for f in build.MODEL_FIELDS:
        targets += [enc(**{f: _junk(rng, getattr(ok, f))})] * 3
    targets += [enc(points={**ok.points, POINTS_AT: _junk(rng, ok.points[POINTS_AT])}) for _ in range(4)]
    # numpy arrays in every point field, deterministically (P.encode_object takes them; equality then is an array)
    rec = quad.records[0]
    arrays = ([dataclasses.replace(quad, **{f: np.array(getattr(quad, f))}) for f in ("seed0", "seed1")]
              + [dataclasses.replace(quad, records=[dataclasses.replace(rec, **{f: np.array(getattr(rec, f))})])
                 for f in ("p0", "p1", "normal")])
    for o in arrays:
        with pytest.raises(ConvertError, match="objects"):
            build.converted_course(rs, syn, SYN_EXE, 0, scenery="model",
                                   scenery_data=dataclasses.replace(sd, objects={OBJECT_AT: o}))
        with pytest.raises(ConvertError, match="objects"):
            build.encode(rs, dataclasses.replace(ok, objects={**ok.objects, OBJECT_AT: o}), MODEL)
    refused = 0
    for make in targets:
        try:
            make()
        except DecodeError:
            refused += 1
    assert refused >= len(targets) - 8, (refused, len(targets))     # measured 146 of 151: junk that is legal (equal
    #                                                                 to the value, or in a field equality skips)


# ---- Desert City's scenery, assembled (Phase 7, Task 5) ----

OPAQUE_TSP = 0x2008216D     # the synthetic meshes' TSP word: src one, dst zero, ignore texture alpha (bit 19)
BLEND_TSP = 0x9400216D      # src alpha, dst one minus src alpha, texture alpha used
FLIP_U, FLIP_V, CLAMP_U, CLAMP_V = 1 << 18, 1 << 17, 1 << 16, 1 << 15


def _dc_square(x: float, z: float, u: tuple[float, float], v: tuple[float, float]):
    """A flat double-sided (cull mode 1) 2 x 2 square strip at height 0, Dreamcast frame: two triangles, one quad,
    its corners' (u, v) the box u x v."""
    from tests.test_convert_model import V
    return pol.Strip(0x11, tuple(V(x + (k & 1) * 2, 0.0, z + (k // 2) * 2, u=u[k & 1], v=v[k // 2]) for k in range(4)))


INTENSITY_1, INTENSITY_2 = 0x8000002C, 0x8000003C    # parameter word 0: colour mode 2 (own face colour), 3 (reused)


def _dc_mesh(strip, texture: int, tsp: int = OPAQUE_TSP, colour=(1.0, 1.0, 1.0), pcw: int = INTENSITY_1):
    return pol.Mesh((pcw, 0x83000000, tsp, 0xCC000000), (0.0, 0.0, 0.0, 1.0), texture, 0,
                    (1.0, 1.0) + tuple(colour) + (0.0, 1.0, 1.0, 1.0), (strip,))


MODE_MESHES = ("repeat", "mirror u", "mirror v", "clamp v", "clamp u and mirror u", "translucent", "opaque copy",
               "flat red", "shadow (dropped)", "faint (dropped)", "tinted blue", "intensity 2 after blue")


def _modes_model():
    """A synthetic Dreamcast course: four textures (64 x 64 RGB565; 64 x 64 ARGB1555, half its texels transparent; two
    32 x 32 ARGB4444 decals: a shadow, one colour (0x111) at random alpha, and a faint one, random colours below 50 %
    alpha) and one square per MODE_MESHES, each its own mesh; every textured one modulates (OPAQUE_TSP's shading 1)
    by a white face colour but "tinted blue" (0, 0, 1), and "intensity 2 after blue" stores white but reuses the blue
    (colour mode 3). Returns (model, its _TEX.BIN blob)."""
    from tests.test_convert_model import make_model
    rng = np.random.default_rng(3)
    opaque = rng.integers(0, 0x10000, 64 * 64, dtype=np.uint16)
    alpha = rng.integers(0, 0x10000, 64 * 64, dtype=np.uint16)              # bit 15: the ARGB1555 alpha, random
    shadow = (rng.integers(0, 16, 32 * 32, dtype=np.uint16) << 12) | 0x111
    faint = (rng.integers(0, 8, 32 * 32, dtype=np.uint16) << 12) | rng.integers(0, 0x1000, 32 * 32, dtype=np.uint16)
    blob = opaque.tobytes() + alpha.tobytes() + shadow.tobytes() + faint.tobytes()
    texs = (pol.TexEntry(64, 64, 0x101, 0, 0), pol.TexEntry(64, 64, 0x100, 8192, 0),
            pol.TexEntry(32, 32, 0x102, 16384, 0), pol.TexEntry(32, 32, 0x102, 18432, 0))
    meshes = [_dc_mesh(_dc_square(0, 10, (0, 2), (0, 1)), 0),
              _dc_mesh(_dc_square(4, 10, (0, 2), (0, 1)), 0, OPAQUE_TSP | FLIP_U),
              _dc_mesh(_dc_square(8, 10, (0, 1), (0, 2)), 0, OPAQUE_TSP | FLIP_V),
              _dc_mesh(_dc_square(12, 10, (0, 1), (-0.5, 1.5)), 0, OPAQUE_TSP | CLAMP_V),
              _dc_mesh(_dc_square(16, 10, (-0.25, 1.25), (0, 1)), 0, OPAQUE_TSP | CLAMP_U | FLIP_U),
              _dc_mesh(_dc_square(20, 10, (0, 1), (0, 1)), 1, BLEND_TSP),
              _dc_mesh(_dc_square(24, 10, (0, 1), (0, 1)), 1, OPAQUE_TSP),
              _dc_mesh(_dc_square(28, 10, (0, 0), (0, 0)), -1, colour=(1.0, 0.0, 0.0)),
              _dc_mesh(_dc_square(32, 10, (0, 1), (0, 1)), 2, BLEND_TSP),
              _dc_mesh(_dc_square(36, 10, (0, 1), (0, 1)), 3, BLEND_TSP),
              _dc_mesh(_dc_square(40, 10, (0, 1), (0, 1)), 0, colour=(0.0, 0.0, 1.0)),
              _dc_mesh(_dc_square(44, 10, (0, 1), (0, 1)), 0, pcw=INTENSITY_2)]
    return make_model(meshes, texs), blob


def _box(lo, hi) -> list:
    """The 12 triangles (Dreamcast frame) of the axis-aligned box lo..hi, as vertices for a triangle list."""
    from tests.test_convert_model import V
    c = [V(*(hi[i] if k >> i & 1 else lo[i] for i in range(3))) for k in range(8)]
    out = []
    for a, b, d, e in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        out += [c[a], c[b], c[d], c[a], c[d], c[e]]
    return out


def _volumes_model():
    """A synthetic course for the Task 6 drops: object 0 a sky dome (radius 500, every vertex on it, textured); object
    1 as large but with a vertex at its centre; object 2 a small sphere-like solid (radius 1), all three kind 3;
    object 3, kind 1 (CS4's 19:0's, so only a black mesh is a volume there; v0.1.1), its meshes: (0) a black closed
    box, (1) a box stored red but in colour mode 3 after the black one (so black), (2) a red closed box, (3) a black box
    less one face (open), (4) a black box drawn with a texture, (5) two black boxes meeting along one edge (four
    triangles on it). Returns (model, its _TEX.BIN blob: one opaque 32 x 32 RGB565 texture)."""
    from tests.test_convert_model import V
    lst = pol.LIST_FLAG | 2
    octa = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]

    def solid(r, centre=False):
        ps = [V(*(r * x for x in o)) for o in octa]
        tri = [ps[i] for f in faces for i in f]
        if centre:
            tri += [V(0, 0, 0), ps[0], ps[2]]
        return pol.Strip(lst, tuple(tri))

    def obj(meshes, r, kind=3):
        return pol.Object(kind, (0.0, 0.0, 0.0, float(r)), tuple(meshes),
                          sum(len(st.entries) for me in meshes for st in me.strips))

    black, red = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    box = _box((10, 0, 10), (12, 3, 14))
    meshes = [_dc_mesh(pol.Strip(lst, tuple(box)), -1, colour=black),
              _dc_mesh(pol.Strip(lst, tuple(_box((20, 0, 10), (22, 3, 14)))), -1, colour=red, pcw=INTENSITY_2),
              _dc_mesh(pol.Strip(lst, tuple(_box((30, 0, 10), (32, 3, 14)))), -1, colour=red),
              _dc_mesh(pol.Strip(lst, tuple(_box((40, 0, 10), (42, 3, 14))[:-6])), -1, colour=black),
              _dc_mesh(pol.Strip(lst, tuple(_box((50, 0, 10), (52, 3, 14)))), 0, colour=black),
              _dc_mesh(pol.Strip(lst, tuple(_box((60, 0, 10), (62, 3, 14)) + _box((62, 3, 10), (64, 5, 14)))), -1,
                       colour=black)]
    objects = (obj([_dc_mesh(solid(500.0), 0)], 500), obj([_dc_mesh(solid(500.0, True), 0)], 500),
               obj([_dc_mesh(solid(1.0), 0)], 1), obj(meshes, 40, kind=1))
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    m = pol.fix_header(pol.Model(hdr, objects, (pol.TexEntry(32, 32, 0x101, 0, 0),)))
    return m, np.random.default_rng(5).integers(0, 0x10000, 32 * 32, dtype=np.uint16).tobytes()


def test_sky_domes_and_shadow_volumes():
    """Task 6 (docs/spike/2026-11-convert.md, "Desert City seen"): a sky dome is an object past SKY_RADIUS whose every
    vertex lies on its bounding sphere (within SKY_SHELL); a shadow volume an untextured mesh, black (its face colour,
    colour mode 3 inheriting) or of a kind-3 object (v0.1.1: test_shadow_volumes_of_kind_3), whose triangles close
    (every edge an even number of them, two boxes meeting along an edge included); kept_model drops them with the
    decals, keeping every other mesh and the object indices. Object 3 is kind 1: its red closed box stays."""
    m, blob = _volumes_model()
    assert build.sky_domes(m) == (0,)
    assert build.shadow_volumes(m) == ((3, 0), (3, 1), (3, 5))
    kept = build.kept_model(m, blob)
    assert len(kept.objects) == 4 and kept.objects[0].meshes == ()
    assert kept.objects[1] == m.objects[1]
    assert build.origin_props(m) == (2,) and kept.objects[2].meshes == ()      # Task 7: a small solid at the origin
    assert kept.objects[3].meshes == tuple(m.objects[3].meshes[k] for k in (2, 3, 4))
    small = dataclasses.replace(m, objects=m.objects[1:])
    assert build.sky_domes(small) == () and build.shadow_volumes(small) == ((2, 0), (2, 1), (2, 5))
    for f in (build.sky_domes, build.shadow_volumes):
        for bad in (None, "m", dataclasses.replace(m, objects=None)):
            with pytest.raises(ConvertError):
                f(bad)
    for sphere in (None, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, math.nan), (0, 0, 0, "500"), (0, 0, 0, True)):
        bad = dataclasses.replace(m, objects=(dataclasses.replace(m.objects[0], sphere=sphere),) + m.objects[1:])
        with pytest.raises(ConvertError, match="sphere"):
            build.sky_domes(bad)
    with pytest.raises(ConvertError):
        build.kept_model(m, None)


def test_shadow_volumes_of_kind_3():
    """v0.1.1 (14 Nov, out/debug/np-start/report.md): National Park Speedway (CS4) gives each of its shadow volumes a
    per-volume ID colour (a grey ramp 0.05..0.9, some pinkish), Rin Rin Rink (CS6) one (0, 0.02, 0), not black; drawn
    by the arcade they were flat grey and black walls. A shadow volume is an untextured closed mesh that is black or
    belongs to an object of kind 3 (the kind of the Dreamcast's volumes and its sky domes). Object 0, a kind-3 textured
    sky dome, is none; object 1 (kind 3): (0) a grey (0.6) closed box and (1) a near-black (0, 0.02, 0) one are
    volumes, (2) a pink box drawn with a texture and (3) a grey box less one face (open) are not; object 2 (kind 1,
    CS4's 19:0's): (0) a black closed box is one (the black rule kept), (1) a grey closed box is not."""
    from tests.test_convert_model import V
    lst = pol.LIST_FLAG | 2
    octa = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    dome = pol.Strip(lst, tuple(V(*(500.0 * x for x in octa[i])) for f in faces for i in f))

    def obj(kind, meshes, r):
        return pol.Object(kind, (0.0, 0.0, 0.0, float(r)), tuple(meshes),
                          sum(len(st.entries) for me in meshes for st in me.strips))

    def box(x0, n=12):
        return pol.Strip(lst, tuple(_box((x0, 0, 10), (x0 + 2, 3, 14))[:3 * n]))

    grey, green, pink, black = (0.6, 0.6, 0.6), (0.0, 0.02, 0.0), (0.65, 0.35, 0.65), (0.0, 0.0, 0.0)
    objects = (obj(3, [_dc_mesh(dome, 0)], 500),
               obj(3, [_dc_mesh(box(10), -1, colour=grey), _dc_mesh(box(20), -1, colour=green),
                       _dc_mesh(box(30), 0, colour=pink), _dc_mesh(box(40, 10), -1, colour=grey)], 40),
               obj(1, [_dc_mesh(box(50), -1, colour=black), _dc_mesh(box(60), -1, colour=grey)], 70))
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    m = pol.fix_header(pol.Model(hdr, objects, (pol.TexEntry(32, 32, 0x101, 0, 0),)))
    assert build.sky_domes(m) == (0,)
    assert build.shadow_volumes(m) == ((1, 0), (1, 1), (2, 0))
    blob = np.random.default_rng(5).integers(0, 0x10000, 32 * 32, dtype=np.uint16).tobytes()
    kept = build.kept_model(m, blob)
    assert kept.objects[0].meshes == ()
    assert kept.objects[1].meshes == m.objects[1].meshes[2:] and kept.objects[2].meshes == m.objects[2].meshes[1:]


def _props_model():
    """A synthetic course for the run-time props (Task 7): object 0 a small box at the world origin (a prop); object
    1 a box as small but 100 units out; object 2 a large box about the origin (60 units each way, not small); object 3
    a prop at the origin reaching 31 units out; object 4 no mesh; object 5 one box at the origin and one 200 out."""
    lst = pol.LIST_FLAG | 2

    def obj(*boxes):
        meshes = tuple(_dc_mesh(pol.Strip(lst, tuple(_box(lo, hi))), -1) for lo, hi in boxes)
        return pol.Object(1, (0.0, 0.0, 0.0, 10.0), meshes, sum(len(me.strips[0].entries) for me in meshes))

    objects = (obj(((-2, 0, -2), (2, 5, 3))), obj(((100, 0, 0), (102, 5, 2))), obj(((-60, 0, -60), (60, 9, 60))),
               obj(((-31, 0, -1), (1, 2, 1))), pol.Object(1, (0.0, 0.0, 0.0, 1.0), (), 0),
               obj(((-2, 0, -2), (2, 5, 3)), ((200, 0, 0), (202, 5, 2))))
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    return pol.fix_header(pol.Model(hdr, objects, (pol.TexEntry(32, 32, 0x101, 0, 0),)))


def test_origin_props():
    """Task 7 ruling 1 (docs/spike/2026-11-convert.md, "Thinning"): the run-time props the Dreamcast models at the
    world origin (an object whose every vertex lies within PROP_RADIUS of it in x and z) are dropped by kept_model;
    an object as small elsewhere, a large one about the origin, a meshless one and one reaching away are kept."""
    m = _props_model()
    assert build.PROP_RADIUS == 32.0
    assert build.origin_props(m) == (0, 3)
    blob = np.zeros(32 * 32, dtype=np.uint16).tobytes()
    kept = build.kept_model(m, blob)
    assert [len(ob.meshes) for ob in kept.objects] == [0, 1, 1, 0, 0, 2]
    assert kept.objects[1:3] + kept.objects[4:] == m.objects[1:3] + m.objects[4:]
    for bad in (None, "m", dataclasses.replace(m, objects=None),
                dataclasses.replace(m, objects=(dataclasses.replace(m.objects[0], meshes=None),))):
        with pytest.raises(ConvertError):
            build.origin_props(bad)


def test_origin_props_road_guard():
    """Phase 10 Task 3, the road guard (docs/spike/2026-11-courses-4-7.md, "National Park Speedway"): with the course
    it is drawn on, an object at the origin with a vertex within ROAD_NEAR of the road line (x, y and z, the exact
    point-to-segment distance road_triangles uses) is road-side scenery, kept by kept_model; one as near in x and z
    but under the road is still a prop. `_road_course`: lanes 1-6 at x 2.8-16.8 (y 0) along -z from the origin."""
    m = _props_model()
    assert build.origin_props(m, None) == build.origin_props(m) == (0, 3)
    assert build.origin_props(m, _road_course()) == ()            # 0's corner x 2 is 0.8 from lane 1, 3's x 1 1.8
    assert build.origin_props(m, _road_course(40.0)) == (0, 3)    # the road 40 out: both props
    blob = np.zeros(32 * 32, dtype=np.uint16).tobytes()
    assert [len(ob.meshes) for ob in build.kept_model(m, blob, _road_course()).objects] == [1, 1, 1, 1, 0, 2]
    lst = pol.LIST_FLAG | 2
    buried = dataclasses.replace(m.objects[0], meshes=(_dc_mesh(pol.Strip(lst, tuple(_box((-2, -20, -2), (2, -9, 3)))),
                                                                -1),))
    under = dataclasses.replace(m, objects=(buried,) + m.objects[1:])
    assert build.origin_props(under, _road_course()) == (0,)      # 9 below the lanes, as near in x and z: a prop
    for bad in ("d", 3, dataclasses.replace(_road_course(), road=None)):
        with pytest.raises(ConvertError):
            build.origin_props(m, bad)


def _road_course(x0: float = 0.0):
    """make_data with a straight road line along -z (9 points 10 units apart, lanes 1-6 at x x0 + 2.8 j, y 0) and two
    road cells: one under the lanes at x0 = 0 (x -3..20, z 0..-80), one a pit apron away from them (x 40..50, z
    -20..-30, y 0); a third cell is not the road's."""
    from dtools.dc.data import Cell, Lane, RoadPoint
    road = tuple(RoadPoint(1, 6, 0, tuple(Lane(x0 + 2.8 * j, 0.0, -10.0 * i, 0, 0x1FFF, 0, 0) for j in range(8)))
                 for i in range(9))
    cells = (Cell(((-3.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 0.0, -80.0), (-3.0, 0.0, -80.0)), (8.5, 0.0, -40.0), 0),
             Cell(((40.0, 0.0, -20.0), (50.0, 0.0, -20.0), (50.0, 0.0, -30.0), (40.0, 0.0, -30.0)), (45.0, 0.0, -25.0),
                  0),
             Cell(((90.0, 0.0, -20.0), (99.0, 0.0, -20.0), (99.0, 0.0, -30.0), (90.0, 0.0, -30.0)), (95.0, 0.0, -25.0),
                  0))
    return dataclasses.replace(make_data(), road=road, cells=cells, steps=(Step(0, 1), Step(3, 0x42)),
                               attributes=bytes([1, 1, 5]))


def test_road_triangles():
    """build.road_triangles (Task 7): the triangles thinning never removes: any corner within ROAD_NEAR of the road
    line (the used lanes' polylines, the last point joined to the first), or lying on a road cell (its centroid
    inside the cell in x and z, within ROAD_RISE above or below it). Indices into pol.triangles(m)."""
    from tests.test_convert_model import V, make_model, mesh
    lst = pol.LIST_FLAG | 2
    tris = [
        (V(8, 0, -5), V(9, 0, -5), V(8, 0, -6)),            # 0 on the road, between the lanes
        (V(22, 0, -40), V(23, 0, -40), V(22, 0, -41)),      # 1 5.2 off the outer lane (x 16.8)
        (V(24.5, 0, -40), V(26, 0, -40), V(25, 0, -41)),    # 2 7.7 off
        (V(25.5, 0, -40), V(27, 0, -40), V(26, 0, -41)),    # 3 8.7 off: not near
        (V(44, 0.5, -24), V(45, 0.5, -24), V(44, 0.5, -25)),    # 4 on the pit cell (0.5 above it), 23 off the line
        (V(44, 5, -24), V(45, 5, -24), V(44, 5, -25)),      # 5 5 above it: not on it
        (V(60, 0, -24), V(61, 0, -24), V(60, 0, -25)),      # 6 beside it: neither
        (V(9, 7, -45), V(9, 30, -45), V(10, 30, -45)),      # 7 a pole: its foot 7 above the lane (x 8.4)
        (V(5, 12, -45), V(9, 30, -45), V(10, 30, -45)),     # 8 a sign over the road, 12 up: not near
        (V(-4, 0, 3), V(-3, 0, 3), V(-4, 0, 2)),            # 9 by the lap line, 3 past the last point's lanes
        (V(94, 0.2, -24), V(95, 0.2, -24), V(94, 0.2, -25)),    # 10 on a cell that is not the road's (no step)
    ]
    m = make_model([mesh([pol.Strip(lst, tuple(v for t in tris for v in t))])])
    d = _road_course()
    assert (build.ROAD_NEAR, build.ROAD_RISE) == (8.0, 1.0)
    assert build.road_triangles(m, d) == frozenset({0, 1, 2, 4, 7, 9})
    for bad in ((None, d), ("m", d), (m, None), (m, "d"), (m, dataclasses.replace(d, road=None)),
                (m, dataclasses.replace(d, road=())), (m, dataclasses.replace(d, cells=None))):
        with pytest.raises(ConvertError):
            build.road_triangles(*bad)


def test_scenery_model():
    """build.scenery_model (Task 7): kept_model, then model.thin to `keep` with road_triangles protected; no keep is
    no thinning; build.mesh_colours gives a rebuilt mesh its kept mesh's face colour. Refusals."""
    from tests.test_convert_model import terrain
    hdr = (0x200, 0x20001117, 0x213342, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    m = pol.fix_header(pol.Model(hdr, (terrain(16, 40.0), terrain(8, 80.0, texture=1)),
                                 (pol.TexEntry(64, 64, 0x101, 0, 0), pol.TexEntry(32, 32, 0x101, 8192, 0))))
    blob = np.zeros(64 * 64 + 32 * 32, dtype=np.uint16).tobytes()
    d = _road_course(40.0)
    kept = build.kept_model(m, blob)
    assert build.scenery_model(m, blob, d) == kept and build.scenery_model(m, blob, d, {}) == kept
    road = build.road_triangles(kept, d)
    assert len(road) == ROAD_FIELD               # the road (x 42.8-56.8, z 0..-80) passes the first field's edge
    drawn = build.scenery_model(m, blob, d, {0: 0.5})
    assert drawn == model.thin(kept, lambda oi: 0.5 if oi == 0 else 1.0, protected=road)
    assert drawn.objects[0] != kept.objects[0] and drawn.objects[1] == kept.objects[1]
    face = build.mesh_colours(m, kept, drawn)
    assert face[id(drawn.objects[0].meshes[0])] == face[id(m.objects[0].meshes[0])] == (1.0, 1.0, 1.0)
    for keep in ([0.5], {0.0: 0.5}, {True: 0.5}, {2: 0.5}, {-1: 0.5}, {0: 0.0}, {0: True}, {0: "0.5"}):
        with pytest.raises(ConvertError):
            build.scenery_model(m, blob, d, keep)
    for bad in ((None, blob, d), (m, None, d), (m, blob, None), (m, blob, "d")):
        with pytest.raises(ConvertError):
            build.scenery_model(*bad, {0: 0.5})
    for args in ((None, kept, drawn), (m, None, drawn), (m, kept, "drawn"), (m, drawn, drawn),
                 (m, dataclasses.replace(kept, objects=kept.objects[:1]), drawn),
                 (m, kept, dataclasses.replace(drawn, objects=(drawn.objects[0], pol.Object(1, (0.0,) * 4, (), 0))))):
        with pytest.raises(ConvertError):
            build.mesh_colours(*args)


ROAD_FIELD = 240
"""Measured: the first field's triangles with a corner within ROAD_NEAR of the road line (its rows from z 0 to 8: the
line ends at z 0, by the field's near edge)."""


def _walked(sd: build.Scenery):
    """(cell, placement, record, header, points) of every record of the scenery's cell placements, in cell order,
    each record's header the one the rasterizer's walk reads (tha, then step * 4 words on per record)."""
    out = []
    for cell, k in enumerate(sd.cells):
        for p in sd.lists[k]:
            obj = sd.objects[C.check_rom_object(p.oba)]
            tha, pts = p.tha, sd.points[p.tpa & T.TEX_ADDR_MASK]
            assert len(pts) == len(obj.records)
            for r, q in zip(obj.records, pts):
                out.append((cell, p, r, sd.headers[tha & T.TEX_ADDR_MASK], q))
                tha += 4 * build._step(r.attr)
    return out


def _inside(h: T.TexHeader, q) -> bool:
    """Review Focus 3 on one record: its points inside its texture, or the header's wrap / mirror bit for the axis."""
    us, vs = [u for _, u in q], [v for v, _ in q]
    return (min(us) >= 0 and min(vs) >= 0 and (max(us) <= 8 * h.width or h.wrap_x or h.mirror_x)
            and (max(vs) <= 8 * h.height or h.wrap_y or h.mirror_y))


def _faces(m, drawn=None, blob=None) -> dict:
    """id(mesh) -> its face colour (r, g, b 0-1): colours[2:5], or for colour mode 3 (parameter word 0 bits 4-5)
    the last colour-mode-2 mesh's before it in file order. With `drawn` (the scenery model, thinned) and `blob`, a
    mesh model.thin rebuilt takes the colour of the kept mesh (build.kept_model(m, blob)) at its object and index."""
    out, last = {}, None
    for ob in m.objects:
        for me in ob.meshes:
            own = tuple(min(max(float(c), 0.0), 1.0) for c in me.colours[2:5])
            mode = (me.params[0] >> 4) & 3
            last = own if mode == 2 else last
            out[id(me)] = last if mode == 3 and last is not None else own
    if drawn is not None:
        for a, b in zip(build.kept_model(m, blob).objects, drawn.objects, strict=True):
            for ma, mb in zip(a.meshes, b.meshes, strict=True):
                out.setdefault(id(mb), out[id(ma)])
    return out


def _expected_headers(m, kept, sd, rs, blob):
    """Per drawn record of `sd` (in `_walked` order), (record's mesh, the header that mesh's mode implies, the flat
    entry): the converted texture's header with mirror_x / mirror_y from the mesh's flip bits (unless clamped),
    translucent where the texture is and the mesh uses its alpha (bit 19 clear), its palette entry the tint's
    (`textures.tint_entry`; the face colour where the shading, TSP bits 6-7, is 1 or 3); the flat colour's for an
    untextured mesh."""
    from dtools.convert import model, textures
    base, indices = textures.course_textures(m, blob, rs)
    faces = _faces(m, kept, blob)
    tables = T.colour_tables(rs)
    pieces = model.convert_model(kept)
    where = model.triangle_meshes(kept)
    meshes = {(oi, mi): me for oi, ob in enumerate(kept.objects) for mi, me in enumerate(ob.meshes)}
    out = []
    for cell in sorted(pieces):
        for pc in pieces[cell]:
            for r, src in zip(pc.obj.records, pc.sources):
                if not src:
                    continue
                me = meshes[where[src[0]]]
                tsp = me.params[2]
                if me.texture < 0:
                    out.append((me, None, textures.flat_entry(tuple(255 * c for c in faces[id(me)]), tables)))
                    continue
                h = base[me.texture]
                tint = faces[id(me)] if (tsp >> 6) & 3 in (1, 3) else (1.0, 1.0, 1.0)
                out.append((me, dataclasses.replace(
                    h, mirror_x=bool(tsp & FLIP_U and not tsp & CLAMP_U),
                    mirror_y=bool(tsp & FLIP_V and not tsp & CLAMP_V),
                    translucent=h.translucent and not tsp & (1 << 19),
                    colour=textures.tint_entry(indices[me.texture], h, tint, tables)), None))
    return out


def _check_modes(walked, expected) -> None:
    drawn = [w for w in walked if w[2].attr != model.MOVE_ATTR]
    assert len(drawn) == len(expected)
    for (cell, p, r, h, q), (me, want, flat) in zip(drawn, expected):
        if want is None:                    # untextured: flat, the nearest entry, the arcade's untextured header
            assert not h.textured and h.colour == flat and not h.translucent
            assert (h.width, h.height, h.tex_x, h.tex_y, h.luma_base, h.raw) == (32, 32, 0, 0, 0, (0, 0, 0))
            assert all(c == (0, 0) for c in q)
            continue
        got = dataclasses.replace(h, tha=want.tha, step=want.step, wrap_x=False, wrap_y=False)
        assert got == want, (cell, me.texture, hex(me.params[2]))
        assert _inside(h, q), (cell, me.texture, q)
        tsp = me.params[2]
        if tsp & CLAMP_U:
            assert not h.wrap_x and max(u for _, u in q) <= 8 * h.width
        if tsp & CLAMP_V:
            assert not h.wrap_y and max(v for v, _ in q) <= 8 * h.height


def test_decals_near_one_colour():
    """v0.1.2 (out/debug/lake/report.md): Mermaid Lake's and Circuit Pixie's shadow texture 67 holds 994 texels
    (17, 17, 17) and six one ARGB4444 step off ((17, 17, 34), (34, 34, 34) ...), up to 53 % alpha; kept, it was drawn
    solid black under the start gantry, the bridge and the checkpoint arch. A blended translucent texture is a decal
    when its modal visible colour holds at least DECAL_SHARE of the visible texels and none is more than DECAL_STEP
    from it per channel: _modes_model's shadow with six visible texels one step off is still dropped; with 2 % of
    them off, or one texel two steps off, it is kept (a picture, not a shadow)."""
    m, blob = _modes_model()
    shadow = np.frombuffer(blob[16384:18432], np.uint16).copy()
    visible = np.flatnonzero(shadow >> 12)

    def with_texels(idx, colour):
        s = shadow.copy()
        s[idx] = (s[idx] & 0xF000) | colour
        return blob[:16384] + s.tobytes() + blob[18432:]

    assert build.DECAL_STEP == 17 and build.DECAL_SHARE == 0.99
    assert build.decals(m, with_texels(visible[:6], 0x112)) == (2, 3)
    assert build.decals(m, with_texels(visible[:6], 0x221)) == (2, 3)
    assert build.decals(m, with_texels(visible[:len(visible) // 50 + 1], 0x112)) == (3,)
    assert build.decals(m, with_texels(visible[:1], 0x113)) == (3,)


def test_model_scenery_modes(rs):
    """Task 5's rulings on a synthetic course: each Dreamcast (texture, mode) its own arcade header, read by its own
    polygons' walk (ruling 1: mirror_x / mirror_y for the flip modes, clamp no wrap bit with its points inside); a
    translucent texture's opaque users an opaque copy of its header (ruling 2), the dropped texture's mesh not drawn;
    an untextured mesh flat in the palette entry nearest its colour (ruling 3); every list entry's budget its
    object's record count (ruling 5); the scenery through converted_course and encode decodes back. `decals` names
    the two blended textures whose picture is in their alpha alone (the shadow, one colour; the faint one, nothing at
    50 %), not the translucent texture with an opaque user; once that user blends too it is still not one (its
    visible texels differ)."""
    from dtools.convert import model
    m, blob = _modes_model()
    assert build.decals(m, blob) == (2, 3)
    ob = m.objects[0]
    blended = dataclasses.replace(m, objects=(dataclasses.replace(ob, meshes=ob.meshes[:6] + (dataclasses.replace(
        ob.meshes[6], params=ob.meshes[6].params[:2] + (BLEND_TSP,) + ob.meshes[6].params[3:]),) + ob.meshes[7:]),))
    assert build.decals(blended, blob) == (2, 3)
    shadow_opaque = dataclasses.replace(m, objects=(dataclasses.replace(ob, meshes=ob.meshes[:8] + (dataclasses.replace(
        ob.meshes[8], params=ob.meshes[8].params[:2] + (OPAQUE_TSP,) + ob.meshes[8].params[3:]),) + ob.meshes[9:]),))
    assert build.decals(shadow_opaque, blob) == (3,)               # a mesh draws it opaque: kept
    solid = blob[:16384] + (np.full(32 * 32, 0xF111, np.uint16)).tobytes() + blob[18432:]
    assert build.decals(m, solid) == (3,)                          # one colour but wholly opaque: drawn as it is
    for bad_m, bad_blob, match in ((None, blob, "Model"), ("m", blob, "Model"), (m, None, "bytes"),
                                   (m, "blob", "bytes")):
        with pytest.raises(ConvertError, match=match):
            build.decals(bad_m, bad_blob)
    with pytest.raises(DecodeError):
        build.decals(m, blob[:100])
    syn = _syn()
    sd = build.model_scenery(rs, syn, m, blob)
    kept = model.drop_textures(m, (2, 3))
    walked = _walked(sd)
    _check_modes(walked, _expected_headers(m, kept, sd, rs, blob))
    drawn = [w for w in walked if w[2].attr != model.MOVE_ATTR]
    assert len(drawn) == len(MODE_MESHES) - 2                    # one quad per square, the dropped ones not drawn
    variants = {T.encode_tex_header(h) for _, _, _, h, _ in drawn}
    assert len(variants) == len(MODE_MESHES) - 4                 # the two clamped squares' words agree (no mirror, no
    #                     wrap), and the two blue ones' (one reuses the other's face colour): the rest differ
    textured = [h for _, _, _, h, _ in drawn if h.textured]
    assert [h.mirror_x for h in textured] == [False, True] + [False] * 7
    assert [h.mirror_y for h in textured] == [False, False, True] + [False] * 6
    assert [h.wrap_x for h in textured] == [True] + [False] * 8
    assert [h.translucent for h in textured] == [False] * 5 + [True] + [False] * 3
    white, blue, reused = textured[0], textured[7], textured[8]
    assert blue.colour != white.colour and reused.colour == blue.colour      # the tint, the reused face colour
    assert {h.colour for h in textured[:5]} == {white.colour}                # white: the texture's own entry
    for k in sd.cells:
        for p in sd.lists[k]:
            assert p.raw == (len(sd.objects[C.check_rom_object(p.oba)].records),)
    # every header and points inside the textures run, the objects in the polygon run, the lists in the list home
    assert all(TEX_LO <= 2 * k and 2 * k + 8 <= TEX_HI for k in sd.headers)
    assert all(POLY_LO <= 4 * k and 4 * (k + o.word_count) <= POLY_HI for k, o in sd.objects.items())
    assert all(DATA_BASE + DATA_LO <= k and k + 16 * len(v) + 4 <= DATA_BASE + DATA_HI for k, v in sd.lists.items())
    c = build.converted_course(rs, syn, SYN_EXE, 0, scenery="model", scenery_data=sd)
    d = G.decode_course(build.encode(rs, c, MODEL), 0)
    assert _differ(d, c) == []
    assert {k: d.headers[k] for k in sd.headers} == {k: dataclasses.replace(
        h, tha=c.headers[k].tha, step=c.headers[k].step) for k, h in sd.headers.items()}


def _density(points, uvs, size) -> float:
    """Texels per world unit of a polygon (fan areas): sqrt(texel area / world area)."""
    p = np.array(points, float)
    t = np.array(uvs, float) * size
    world = sum(np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])) for k in range(1, len(p) - 1)) / 2
    texel = sum(abs((t[k][0] - t[0][0]) * (t[k + 1][1] - t[0][1]) - (t[k][1] - t[0][1]) * (t[k + 1][0] - t[0][0]))
                for k in range(1, len(t) - 1)) / 2
    return math.sqrt(texel / world)


def test_model_scenery_coefficients():
    """Ruling 4: a record's distance coefficient (attr bits 27-31) assumes the Dreamcast texture's size; where a
    converted texture's size differs it is computed again on the converted size, floored once:
    floor(3 log2(texel density)) clamped to 0-31 (fix round 1: not the floored value shifted); a side halved lowers it
    by about 3/2, both by 3; the same size leaves the objects as they are (Desert City: every texture kept its
    size)."""
    from tests.test_convert_model import synthetic
    m = synthetic()                                     # texture 0 64 x 64 (the ribbon), texture 1 32 x 32 (the quads)
    pieces = model.convert_model(m)
    sizes = [(t.width, t.height) for t in m.textures]

    def head(w, h):
        return T.TexHeader(tha=0, step=0, width=w, height=h, tex_x=0, tex_y=0, sheet=0, textured=True,
                           translucent=False, checker=False, wrap_x=False, wrap_y=False, mirror_x=False,
                           mirror_y=False, utex=False, utex_min_lod=1, utex_x=0, utex_y=0, luma_base=0, colour=0,
                           raw=(0, 1, 0))

    same, moved = build._fit_coefficients(pieces, sizes, {0: head(64, 64), 1: head(32, 32)})
    assert same is pieces and moved == 0
    for new in (((32, 32), (32, 32)), ((128, 128), (64, 64)), ((64, 32), (32, 32)), ((256, 32), (32, 64))):
        got, moved = build._fit_coefficients(pieces, sizes, {0: head(*new[0]), 1: head(*new[1])})
        n = 0
        for cell, ps in pieces.items():
            for a, b in zip(ps, got[cell]):
                assert a.textures == b.textures and a.uvs == b.uvs and a.sources == b.sources
                for ra, rb, poly, t, uv in zip(a.obj.records, b.obj.records, P.polygons(a.obj), a.textures, a.uvs):
                    assert ra.attr & 0x07FFFFFF == rb.attr & 0x07FFFFFF and ra.normal == rb.normal
                    if t < 0 or new[t] == sizes[t]:
                        assert rb.coef == ra.coef
                        continue
                    assert ra.coef == model.lod_coef(_density(poly.points, uv, sizes[t]))
                    assert rb.coef == model.lod_coef(_density(poly.points, uv, new[t])), (t, ra.coef, rb.coef)
                    n += rb.coef != ra.coef
        assert moved == n > 0
    for bad in (None, {0: None}, {0: head(64, 64)}):
        with pytest.raises(ConvertError):
            build._fit_coefficients(pieces, sizes, bad)


def test_model_scenery_refuses_input(rs, disc):
    """model_scenery's arguments: a RomSet, a CourseData, a pol.Model, the _TEX.BIN bytes (ConvertError; a DcError
    from the texture decoder passes through, a DecodeError)."""
    m, blob = _modes_model()
    syn = _syn()
    for args, match in (((None, syn, m, blob), "RomSet"), ((rs, None, m, blob), "CourseData"),
                        ((rs, "d", m, blob), "CourseData"), ((rs, syn, None, blob), "Model"),
                        ((rs, syn, dataclasses.astuple(m), blob), "Model"), ((rs, syn, m, None), "bytes"),
                        ((rs, syn, m, "blob"), "bytes"), ((rs, syn, m, True), "bytes")):
        with pytest.raises(ConvertError, match=match):
            build.model_scenery(*args)
    with pytest.raises(DecodeError):
        build.model_scenery(rs, syn, m, blob[:100])
    with pytest.raises(ConvertError, match="scenery_data"):
        build.converted_set(rs, disc, 3, scenery="empty", scenery_data=build.model_scenery(rs, syn, m, blob))


@pytest.fixture(scope="module")
def desert_city(disc):
    """n 3 (Desert City) with its own scenery in Beginner's slot, built from the original set: (rs, model, the model
    the scenery converts (build.scenery_model: kept, then thinned to build.THIN[3]), texture blob, scenery, course,
    set). The set is converted_set(rs, disc, 3, scenery="model"), which builds the scenery from the disc itself."""
    from dtools.convert import model
    from dtools.dc.disc import course_files
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    d, exe = _data(disc, 3), disc.read(EXE_NAME)
    mn, tn = course_files(3, kind="model"), course_files(3, kind="textures")
    m, blob = pol.decode_pol(disc.read(mn), mn), disc.read(tn)
    sd = build.model_scenery(rs, d, m, blob, build.THIN[3])
    c = build.converted_course(rs, d, exe, 3, scenery="model", scenery_data=sd)
    out = build.converted_set(rs, disc, 3, scenery="model")
    return rs, m, build.scenery_model(m, blob, d, build.THIN[3]), blob, sd, c, out


def test_desert_city_thinned(desert_city, disc):  # noqa: F811
    """Task 7 on Desert City (docs/spike/2026-11-convert.md, "Thinning"): the scenery model is the kept model thinned
    to build.THIN[3]; every road triangle of the kept model (build.road_triangles: the road surface and every triangle
    within 8 units of the road line) is in it with its corners, order and winding; per thinned object at most
    round(keep * n) of its n non-degenerate triangles are left and at most one fewer; every other object is the kept
    model's own; the counts the log gives."""
    from tests.test_convert_model import _tri_key
    rs, m, drawn, blob, sd, c, out = desert_city
    d = _data(disc, 3)
    kept = build.kept_model(m, blob)
    road = build.road_triangles(kept, d)
    tris = list(pol.triangles(kept))
    have = collections.Counter((t[3], _tri_key(t)) for t in pol.triangles(drawn))
    want = collections.Counter((tris[i][3], _tri_key(tris[i])) for i in road)
    assert not want - have

    def solid(mm) -> collections.Counter:
        return collections.Counter(t[3] for t in pol.triangles(mm)
                                   if not model.degenerate([model._pos(v) for v in t[:3]]))

    before, after = solid(kept), solid(drawn)
    for oi, ob in enumerate(kept.objects):
        if oi in build.THIN[3]:
            target = round(build.THIN[3][oi] * before[oi])
            assert target - 1 <= after[oi] <= target, (oi, before[oi], after[oi], target)
        else:
            assert drawn.objects[oi] == ob
    counts = (len(road), sum(before.values()), sum(after.values()))
    print(f"\nroad triangles, kept and thinned triangles: {counts}; per object "
          f"{ {oi: (before[oi], after[oi]) for oi in build.THIN[3]} }")
    assert counts == DESERT_THINNED
    # the shipped objects come from convert_model(drawn): every non-degenerate triangle drawn exactly once
    from tests.test_convert_model import check_pieces
    owned = check_pieces(drawn, model.convert_model(drawn))["owned"]
    assert owned == DESERT_THINNED[2]


DESERT_THINNED = (5397, 16992, 15224)
"""Measured: the kept model's road triangles (road surface or within ROAD_NEAR of the road line), its non-degenerate
triangles, and those left after THIN[3]."""


def test_desert_city_scenery_round_trip(desert_city):
    """Desert City's scenery (CS3_POL.BIN / CS3_TEX.BIN) in Beginner's slot: the set decodes to the converted Course
    on every field, check_course passes on both, every table re-emits; its cells, lists and objects are the
    scenery's (with the car models); every drawn polygon reads its own mode's header (rulings 1-3, the same check as
    the synthetic test, on all of them); every texture point inside its texture or under a wrap / mirror bit (Review
    Focus 3, as the set reads them back); the counts the discovery log gives ("Scenery assembled")."""
    rs, m, kept, blob, sd, c, out = desert_city
    b = _beginner()
    d = G.decode_course(out, 0)
    assert _differ(d, c) == []
    G.check_course(c)
    G.check_course(d)
    assert G.mismatches(out, c) == []
    assert C.library_bounds(out) == C.PATCHED_LIBRARY
    assert d.cells == sd.cells and set(d.objects) == set(sd.objects) | set(b.car_objects)
    assert all(d.objects[k] == o for k, o in sd.objects.items())
    assert {k: d.object_lists[k] for k in sd.lists} == sd.lists
    assert set(sd.headers) <= set(d.headers) and set(sd.points) <= set(d.points)    # every one reached and decoded
    walked = _walked(dataclasses.replace(sd, headers={k: d.headers[k] for k in sd.headers},
                                         points={k: d.points[k] for k in sd.points}))
    _check_modes(walked, _expected_headers(m, kept, sd, rs, blob))
    variants = {}
    for _, _, r, h, _ in walked:
        if r.attr != model.MOVE_ATTR:
            variants.setdefault(T.encode_tex_header(dataclasses.replace(h, wrap_x=False, wrap_y=False)), h)
    geometry = {}
    for h in variants.values():
        if h.textured:
            geometry.setdefault((h.tex_x, h.tex_y), set()).add(h.translucent)
    counts = dict(cells=sum(1 for k in set(sd.cells) if sd.lists[k]), objects=len(sd.objects),
                  headers=len(sd.headers), points=len(sd.points),
                  texels=len([t for t in d.texels if t.header.sheet == 0]), variants=len(variants),
                  flat=sum(not h.textured for h in variants.values()),
                  mirrored=sum(h.mirror_x or h.mirror_y for h in variants.values()),
                  translucent=sum(h.translucent for h in variants.values()),
                  opaque_copies=sum(v == {False, True} for v in geometry.values()),
                  textures=len(geometry))
    print(f"\nDesert City scenery: {counts}")
    assert counts == DESERT_SCENERY


DESERT_SCENERY = dict(cells=69, objects=69, headers=12061, points=69, texels=56, variants=101, flat=0, mirrored=12,
                      translucent=12, opaque_copies=1, textures=56)
"""Measured: occupied cells, objects, headers, points runs, sheet-0 textures, distinct header variants (wrap bits
aside; fix round 1: a tinted use its own), the flat ones, the mirrored ones, the translucent ones, the textures drawn
both translucent and opaque (ruling 2's copies: 57; 51 was a prop's), the textures drawn. Task 7 (the props dropped,
the scenery thinned to THIN[3]); Task 6's: 70 cells and objects, 15,415 headers, 78 textures, 174 variants, 14 flat
(every untextured mesh was a prop's), 21 mirrored, 20 translucent, 2 copies."""


def _record_tints(m, kept, blob) -> list[tuple[int, tuple]]:
    """(Dreamcast texture, tint) per drawn record of the scenery model's conversion (`kept`, thinned), in `_walked`
    order: the tint the face colour where the mesh's shading modulates (TSP bits 6-7 1 or 3), else (1, 1, 1)."""
    faces, pieces = _faces(m, kept, blob), model.convert_model(kept)
    where = model.triangle_meshes(kept)
    meshes = {(oi, mi): me for oi, ob in enumerate(kept.objects) for mi, me in enumerate(ob.meshes)}
    out = []
    for cell in sorted(pieces):
        for pc in pieces[cell]:
            for t, src in zip(pc.textures, pc.sources):
                if src:
                    me = meshes[where[src[0]]]
                    out.append((t, faces[id(me)] if (me.params[2] >> 6) & 3 in (1, 3) else (1.0, 1.0, 1.0)))
    return out


def _decoded_mean(rs, h) -> np.ndarray:
    """The mean colour of a header's texture as the set draws it (texture.decode_texels at full light, opaque
    texels)."""
    rgba = T.decode_texels(rs, h, 0).reshape(-1, 4).astype(float)
    return rgba[rgba[:, 3] > 0, :3].mean(0)


def test_desert_city_tints(desert_city):
    """Fix round 1 ruling: a textured mesh whose shading modulates its texture by its face colour reads a header of
    its own tint (one header variant per texture, mode, translucency and tint), as the set decodes it: texture 42's
    red-tinted (1, 0.6, 0.6) and grey (0.5) polygons read different palette entries (Task 7: texture 0's blue and red
    uses were the props', dropped); a darkened use of textures 35, 38 and 42 (grey 0.2-0.5) decodes darker than their
    untinted use; an untinted use reads the texture's own entry. The tinted polygons per texture are pinned
    (DESERT_TINTED of the textured drawn polygons of the thinned scenery)."""
    from dtools.convert import textures
    rs, m, kept, blob, sd, c, out = desert_city
    d = G.decode_course(out, 0)
    base, _ = textures.course_textures(m, blob, rs)
    drawn = [w for w in _walked(dataclasses.replace(sd, headers={k: d.headers[k] for k in sd.headers}))
             if w[2].attr != model.MOVE_ATTR]
    tints = _record_tints(m, kept, blob)
    assert len(drawn) == len(tints)
    seen: dict[tuple, set] = {}
    per: dict[int, int] = {}
    for (_, _, _, h, _), (t, tint) in zip(drawn, tints):
        if t < 0:
            continue
        seen.setdefault((t, tuple(round(x, 3) for x in tint)), set()).add(h.colour)     # single precision: 0.2 ~
        if tint != (1.0, 1.0, 1.0):
            per[t] = per.get(t, 0) + 1
        else:
            assert h.colour == base[t].colour and h.luma_base == base[t].luma_base
    assert all(len(v) == 1 for v in seen.values())             # one entry per (texture, tint)
    entry = {k: next(iter(v)) for k, v in seen.items()}
    print(f"\ntinted {sum(per.values())} of {len(tints) - sum(t < 0 for t, _ in tints)}, {len(per)} textures: "
          f"{dict(sorted(per.items()))}")
    assert (sum(per.values()), len(per), len(tints) - sum(t < 0 for t, _ in tints)) == DESERT_TINTED
    assert dict(sorted(per.items())) == DESERT_TINTS
    assert entry[(42, (1.0, 0.6, 0.6))] != entry[(42, (0.5, 0.5, 0.5))]
    for t, dark in ((35, 0.2), (38, 0.5), (42, 0.5)):
        h = base[t]
        light = _decoded_mean(out, h)
        darker = _decoded_mean(out, dataclasses.replace(h, colour=entry[(t, (dark,) * 3)]))
        assert darker.sum() < light.sum(), (t, darker, light)


DESERT_TINTS = {28: 64, 32: 25, 33: 112, 35: 223, 37: 107, 38: 130, 42: 225, 45: 20, 47: 1, 52: 4, 53: 47, 54: 47,
                59: 18, 60: 5, 61: 12, 64: 9, 76: 1, 77: 3, 78: 88, 82: 182}
DESERT_TINTED = (1323, 20, 12061)
"""Measured: Desert City's tinted textured drawn polygons per Dreamcast texture (shading modulate, face colour not
white), and (tinted, textures, textured drawn polygons) of the thinned scenery (Task 7; Task 6: 2,391 of 15,251 over 29
textures, before the props were dropped and the scenery thinned)."""


def test_desert_city_only_free_bytes(desert_city):
    """Review Focus 4 on Desert City: every changed byte is in a table rewritten in place (the race's, the cell grid,
    ACTOR_TABLES[0], the HUD map's list entry, place record and zooms), in the bound patch's two words, in Beginner's
    1 MB course sheet block (the course's texels and mips), or was 0xff in a free run: DATA_FREE (the race tables and
    the scenery's lists), COPRO_FREE (the ground), the program run (the actor table), the polygon run (the outline and
    the scenery's objects), the textures run (its headers and points). The program changes only there; the common
    block (sheet 1) and Expert's and Advanced's blocks are byte-identical."""
    rs, m, kept, blob, sd, c, out = desert_city
    block = c.sheet_sources[0]
    assert block == 0x200000
    changed = _changed_outside(rs, out, IN_PLACE | MODEL_IN_PLACE,
                               spans=[("program", at, 4) for at in C.LIBRARY_BOUND_AT] + [("data", block, 0x100000)])
    prog = {int(k) for k in changed["program"] if k < PROG_LO}
    bound = {0x17AD0, 0x17AD1, 0x17AD2, 0x17CFC, 0x17CFD, 0x17CFE}
    assert bound <= prog <= bound | {C.GRID_BASES - 0x200000 + k for k in (0, 1)} | \
        {C.ACTOR_TABLES - 0x200000 + k for k in range(4)}
    spans = [(2 * k, 8) for k in sd.headers] + [(2 * k, 4 * sum(len(q) for q in v)) for k, v in sd.points.items()]
    mine = np.zeros(len(rs.textures), bool)
    for off, n in spans:
        assert TEX_LO <= off and off + n <= TEX_HI
        mine[off:off + n] = True
    assert mine[changed["textures"]].all()
    (index, outline), = c.hud_map.objects.items()
    poly = np.zeros(len(rs.polygons), bool)
    poly[4 * index:4 * (index + outline.word_count)] = True
    for k, o in sd.objects.items():
        poly[4 * k:4 * (k + o.word_count)] = True
    assert poly[changed["polygons"]].all()
    for other in (T.sheet_source(rs, 1), T.sheet_source(rs, 0, 1), T.sheet_source(rs, 0, 2)):
        assert out.data[other:other + 0x100000] == rs.data[other:other + 0x100000], hex(other)
    assert out.data[block:block + 0x100000] != rs.data[block:block + 0x100000]
    lists = np.zeros(len(rs.data), bool)
    for k, v in sd.lists.items():
        lists[k - DATA_BASE:k - DATA_BASE + 16 * len(v) + 4] = True
    data = np.array([int(k) for k in changed["data"] if DATA_LO <= k < DATA_HI])
    first = min(sd.lists) - DATA_BASE
    assert (lists[data[data >= first]]).all()                  # past the race tables: the scenery's lists only
    assert out.read_i960(C.GRID_TABLES, 4) == rs.read_i960(C.GRID_TABLES, 4)


def test_desert_city_leaves_other_courses(desert_city):
    """Expert and Advanced re-emit identically from the Desert City set (golden.mismatches empty) and their grounds
    decode the same; sheet 1's block, Expert's and Advanced's course blocks, the car models (objects, their headers
    and texture points), GRID_LANES, BILLBOARD_REF and both spawn tables are byte-identical."""
    rs, m, kept, blob, sd, c, out = desert_city
    for k in (1, 2):
        other = G.decode_course(rs, k)
        assert G.mismatches(out, other) == [], k
        assert tgp.decode_ground(out, k) == other.ground_data
    for other in (T.sheet_source(rs, 1), T.sheet_source(rs, 0, 1), T.sheet_source(rs, 0, 2)):
        assert out.data[other:other + 0x100000] == rs.data[other:other + 0x100000], hex(other)
    assert out.read_i960(C.GRID_LANES, 4) == rs.read_i960(C.GRID_LANES, 4)
    assert out.read_i960(C.BILLBOARD_REF, 12) == rs.read_i960(C.BILLBOARD_REF, 12)
    for t in C.SPAWN_TABLES:
        n = C.SPAWN_RECORD_SIZE * C.SPAWN_RECORDS
        assert out.read_i960(t, n) == rs.read_i960(t, n)
    b = _beginner()
    for key in b.car_objects:
        span = slice(4 * key, 4 * (key + b.objects[key].word_count))
        assert out.polygons[span] == rs.polygons[span], key
    d = G.decode_course(out, 0)
    cars = {p.tpa & T.TEX_ADDR_MASK for k in build._slot_lists(b) for p in b.object_lists[k]}
    for k in cars & set(b.points):
        assert d.points[k] == b.points[k]
    for k, h in b.headers.items():
        if h.sheet == 1 and h.textured and k in d.headers:
            assert T.encode_tex_header(d.headers[k]) == T.encode_tex_header(h)


def test_desert_city_culling_and_camera(desert_city):
    """Phase 7 Task 8 (docs/spike/2026-11-convert.md, "Confirmed with scenery"): what scenery wakes up is shared and
    unchanged. The converted set keeps the culling tables (`course.culling`: the view test's four quadrant tables, the
    visit orders, the block offsets) and the camera scripts (CAMERA_SCRIPTS, CAMERA_SCRIPTS_SLOT0, and the scripts
    slot 0 plays) byte for byte; slot 0's scripts are car-relative (`test_camera_scripts_real`), so the fly-by frames
    the player's car on Desert City.

    The reach (by the block; the road part measured on one lap): the view test opens only the 5 x 5 block round the
    reference cell, and on the autoplay lap the reference cell was always a cell of the road line
    (test_desert_city_scenery_reads), so there a scenery cell more than two cells from every road cell was never
    drawn. On Desert City only cells 0x08 and 0x09 (row 0, 3 + 7 budget words of 14,223) lie that far; a car or
    camera a cell off the road could reach them. UNREACHED pins them."""
    rs, m, drawn, blob, sd, c, out = desert_city
    assert C.culling(out) == C.culling(rs)
    starts = set(struct.unpack("<4I", rs.read_i960(C.CAMERA_SCRIPTS_SLOT0, 16))) | {
        struct.unpack("<I", rs.read_i960(C.CAMERA_SCRIPTS, 4))[0]}
    for table in (C.CAMERA_SCRIPTS, C.CAMERA_SCRIPTS_SLOT0):
        assert out.read_i960(table, 16) == rs.read_i960(table, 16)
    for start in starts:
        script = C.camera_script(rs, start)
        assert C.camera_script(out, start) == script
        lo, hi = script.start, max(k.addr for k in script.keyframes) + 2 * C.KEYFRAME_SIZE     # the jump record too
        assert out.read_i960(lo, hi - lo) == rs.read_i960(lo, hi - lo)
    d = G.decode_course(out, 0)
    budget = {cell: sum(p.raw[0] for p in d.object_lists[k]) for cell, k in enumerate(d.cells)}
    road = {C.cell_of(p.centre[0], p.centre[2]) for p in c.road}
    half = C.BLOCK // 2
    near = range(-half, half + 1)
    reach = {(row << 4) | col for r in road for dr in near for dc in near        # the block, off-grid cells dropped
             if 0 <= (row := (r >> 4) + dr) < C.GRID_DIM and 0 <= (col := (r & 15) + dc) < C.GRID_DIM}
    unreached = {cell: n for cell, n in budget.items() if n and cell not in reach}
    print(f"\nroad cells {len(road)}; occupied {sum(1 for n in budget.values() if n)}; beyond the 5 x 5 reach "
          f"{ {hex(k): v for k, v in unreached.items()} }")
    assert len(road) == ROAD_CELLS and unreached == UNREACHED


ROAD_CELLS = 31                 # Desert City's road line passes 31 cells (rows 3-12, columns 5-10): the trace's 31
#                                 reference cells exactly (test_desert_city_scenery_reads)
UNREACHED = {0x08: 3, 0x09: 7}  # occupied cells more than two cells from every road cell (not drawn on the
#                                 autoplay lap): cell -> budget words


def test_desert_city_budget(desert_city):
    """Ruling 5: every cell placement's budget word (entry word 3) is its object's record count, so per cell the
    budget words sum to the cell's records (the frame cap 5,000, 0x5010f4: Task 6 sees where it bites; nothing is
    thinned here); the scenery's lists, objects, headers and points lie inside their free runs beside the race
    tables and the HUD outline, the room left in each pinned (docs/spike/2026-11-convert.md, "Scenery assembled")."""
    rs, m, kept, blob, sd, c, out = desert_city
    d = G.decode_course(out, 0)
    per_cell = {}
    for cell, k in enumerate(d.cells):
        for p in d.object_lists[k]:
            records = len(d.objects[C.check_rom_object(p.oba)].records)
            assert p.raw == (records,)
            per_cell[cell] = per_cell.get(cell, 0) + records
    assert sum(per_cell.values()) == sum(len(o.records) for o in sd.objects.values())
    # the runs: what the race tables, the outline and the scenery take, and the room left in each
    list_lo = min(sd.lists) - DATA_BASE
    list_bytes = sum(16 * len(v) + 4 for v in sd.lists.values())
    assert list_lo + list_bytes == DATA_HI
    race_end = max(c.segment_flags[0] - DATA_BASE + 2 * len(c.road) + 2, c.checkpoint_table - DATA_BASE + 12)
    obj_lo = 4 * min(sd.objects)
    obj_bytes = sum(4 * o.word_count for o in sd.objects.values())
    assert obj_lo + obj_bytes == POLY_HI
    (index, outline), = c.hud_map.objects.items()
    outline_end = 4 * (index + outline.word_count)
    tex_words = [2 * k for k in sd.headers] + [2 * k for k in sd.points]
    tex_end = max([2 * k + 8 for k in sd.headers] + [2 * k + 4 * sum(map(len, v)) for k, v in sd.points.items()])
    assert min(tex_words) == TEX_LO
    room = dict(cells=dict(sorted(per_cell.items())), max_cell=max(per_cell.values()),
                total=sum(per_cell.values()), lists=list_bytes, data_left=list_lo - race_end,
                objects=obj_bytes, polygons_left=obj_lo - outline_end,
                textures=tex_end - TEX_LO, textures_left=TEX_HI - tex_end)
    print(f"\nDesert City budget and room: { {k: v for k, v in room.items() if k != 'cells'} }")
    print(f"per cell: {room['cells']}")
    assert {k: v for k, v in room.items() if k != "cells"} == DESERT_ROOM
    assert room["data_left"] > 0 and room["polygons_left"] > 0 and room["textures_left"] > 0


DESERT_ROOM = dict(max_cell=868, total=14223, lists=1384, data_left=444788, objects=571680, polygons_left=1276244,
                   textures=288464, textures_left=420840)
"""Measured: the largest cell's budget, the total, the list bytes and DATA_FREE left between the race tables and
the lists, the object bytes and POLYGON_FREE left between the HUD outline and the objects, the headers and points
bytes and TEXTURES_FREE left after them. Task 7 (props dropped, thinned); Task 6's: largest 1,212 (cell 0x78), total
18,042. Task 8 fix (the HUD outline merged to 64 quads, 2,600 bytes against 18,640): polygons_left 1,260,204 to
1,276,244."""


def test_long_time(desert_empty):
    """build.long_time (the playtest set's start time, moved here from a test module in Phase 7 Task 9): the first
    word of every time table of the Course becomes `ticks` (default LONG_TIME) and nothing else changes; a RomSet,
    Course or tick count of the wrong kind is a ConvertError."""
    rs, c, out = desert_empty
    long = build.long_time(out, c)
    assert build.LONG_TIME == 12000 and long.data != out.data
    for _, t, _ in c.time_tables.values():
        assert struct.unpack("<I", long.read_i960(t, 4))[0] == build.LONG_TIME
    assert (long.program, long.polygons, long.textures) == (out.program, out.polygons, out.textures)   # data ROM only
    assert len(np.flatnonzero(np.frombuffer(long.data, np.uint8) != np.frombuffer(out.data, np.uint8))) <= 16 * len(
        c.time_tables)                  # four bytes a word, a few copies of the data ROM's window
    assert struct.unpack("<I", build.long_time(out, c, 777).read_i960(next(iter(c.time_tables.values()))[1], 4))[0] == 777
    for bad in ((None, c), (out, None), (out, c, 0), (out, c, True), (out, c, 1 << 32), (out, c, 1.5)):
        with pytest.raises(ConvertError):
            build.long_time(*bad)



def test_race_times_refuses_overlapping_tables(desert_empty):
    """build.race_times on a course converted into Beginner's slot (no `slot=`): Sega's per-difficulty tables overlap,
    so distinct words per (mode, difficulty) cannot be written in place: ConvertError (the release set's courses have
    their own disjoint tables: tests/test_convert_build_courses.py test_race_times_writes_in_place)."""
    _, _, out = desert_empty
    c = G.decode_course(out, 0)
    tables = {key: tuple(1000 + 10 * key[0] + key[1] + i for i in range(len(w))) for key, (_, _, w) in c.time_tables.items()}
    with pytest.raises(ConvertError, match="overlaps"):
        build.race_times(out, c, tables)

def test_mirrored_converted_set(desert_city):
    """docs/follow-ups/phase-07-m3b.md, "What M4 starts from": the mirror of a converted course works offline. mirrored_set of
    converted_set(..., scenery="model") decodes, re-emits (verify_lossless empty for course 0), keeps the patched
    library bounds, and Expert's and Advanced's tables differ from the ROM only by GRID_LANES (Phase 4's swap)."""
    from dtools.arcade import mirror
    rs, m, kept, blob, sd, c, out = desert_city
    mir = mirror.mirrored_set(out, 0)
    G.decode_course(mir, 0)
    assert G.verify_lossless(mir, 0) == []
    assert C.library_bounds(mir) == C.PATCHED_LIBRARY
    for k in (1, 2):
        lines = G.mismatches(mir, G.decode_course(rs, k))
        assert len(lines) == 1 and "grid lanes at program+0x31b30" in lines[0], lines


# ---- Desert City at its own id (Phase 8, Task 5): docs/spike/2026-11-sixteen.md, "Desert City at id 3" ----

def _guarded(rs: RomSet) -> RomSet:
    """The grown set with the new ids' guards: what a new id is converted into, before the select (the guards first:
    patch_select refuses a live new id on an unguarded set; Desert City converted first: since Phase 9's final review
    patch_select refuses a live id 3 that holds no course of its own). The MAME test test_desert_city_id3 builds
    patch_select(converted_set(_guarded(rs), ...), M4A_ENTRIES)."""
    return C.guard_new_ids(C.grow_ids(rs))


def _sixteen(rs: RomSet) -> RomSet:
    """`_guarded` with the sixteen-entry select of Phase 8's row, id 3 not converted (test_select.unconverted_select):
    Phase 8's order, the select before the conversion, which `desert_id3` keeps on purpose: it proves the converter
    places a new id around the select's code, data and map objects (test_final_fix1_owned_bytes_are_busy). The same
    bytes as the chain's order (test_desert_city_before_select)."""
    from tests.test_select import unconverted_select
    return unconverted_select(_guarded(rs))


@pytest.fixture(scope="module")
def desert_id3(disc):
    """n 3 (Desert City) with its own scenery at course id 3 of the grown set (`_sixteen`): (set before, scenery,
    course, set after). The set is converted_set(base, disc, 3, slot=3, scenery="model")."""
    from dtools.dc.disc import course_files
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    base = _sixteen(rs)
    d, exe = _data(disc, 3), disc.read(EXE_NAME)
    mn, tn = course_files(3, kind="model"), course_files(3, kind="textures")
    sd = build.model_scenery(base, d, pol.decode_pol(disc.read(mn), mn), disc.read(tn), build.THIN[3], slot=3)
    c = build.converted_course(base, d, exe, 3, slot=3, scenery="model", scenery_data=sd)
    out = build.converted_set(base, disc, 3, slot=3, scenery="model")
    return base, sd, c, out


def _runs_mask(image_len: int, runs) -> np.ndarray:
    mask = np.zeros(image_len, bool)
    for lo, n in runs:
        mask[lo:lo + n] = True
    return mask


def test_converted_id3_round_trip(desert_id3):
    """Desert City as course id 3 (Review Focus 4 and 5):

    - the set decodes id 3 to the converted Course on every field, check_course passes, every table re-emits
      (verify_lossless empty) and encode(base, c) is the set converted_set builds;
    - it has its own of every table it changes: the race tables, the cell grid, the ground pointer words and the time
      words and tables in ID_DATA_FREE (the custom data run), its 1 MB course sheet block there (1 MB-aligned; the
      slot's block, Beginner's, copied into it and its texels written over it), its ground in the custom copro_data
      area, its objects and HUD outline in the custom polygon run, its headers and points in the custom textures run,
      its lists and its HUD outline's own list in the list home DATA_FREE, its actor table in the program run; and the
      per-id words that name them are its own entries (COURSE_SHEETS[3], GRID_TABLES[3], the HUD map's, the TIME_TABLES
      levels', ...);
    - ids 0-2 are byte-identical to the grown and patched set's: each decodes and re-emits to the same Course, every
      byte their tables cover (golden.coverage) is unchanged, Beginner's, Expert's and Advanced's course blocks and the
      common block too;
    - the free-bytes rule on the larger regions: every byte that differs from the set before is an id-3 per-id word,
      record 3, TGP slot 3, the bound patch or a byte that was 0xff in a free run (NEW_ID_RUNS and DATA_FREE); the
      stock POLYGON_FREE, TEXTURES_FREE and COPRO_FREE are untouched (a new id lives in the custom areas)."""
    base, sd, c, out = desert_id3
    d = G.decode_course(out, 3)
    assert _differ(d, c) == []
    G.check_course(c)
    G.check_course(d, out)
    assert G.verify_lossless(out, 3) == [] and G.mismatches(out, c) == []
    assert build.encode(base, c, build.RACE_FIELDS + build.MODEL_FIELDS + build.ID_FIELDS).split_files() == \
        out.split_files()
    assert C.library_bounds(out) == C.PATCHED_LIBRARY
    # its own tables, in its runs
    runs = build.NEW_ID_RUNS
    lo, n = runs["data"]

    def inside(off, size, run=(lo, n)):
        return run[0] <= off and off + size <= run[0] + run[1]

    e = c.entry
    for f in ("step_table", "record_table", "line_table", "block_ref_a", "block_ref_b"):
        assert inside(getattr(e, f), 4), f
    assert inside(c.grid - DATA_BASE, 4 * C.GRID_CELLS) and inside(c.checkpoint_table - DATA_BASE, 4)
    assert inside(c.segment_flags[0] - DATA_BASE, 2) and inside(c.ground_data.grid_at, 2 * tgp.GRID_CELLS)
    for (m, k), (ptr, table, words) in c.time_tables.items():
        assert inside(ptr - DATA_BASE, 4) and inside(table - DATA_BASE, 4 * len(words))
        assert C.array_at(d.id_arrays, f"TIME_TABLES[{m}]") >= C.ID_HOME
    block = c.sheet_sources[0]
    assert block % 0x100000 == 0 and inside(block, 0x100000) and c.sheet_sources[1] == T.sheet_source(base, 1)
    assert T.sheet_source(out, 0, 3) == block and T.sheet_source(out, 0, 0) == T.sheet_source(base, 0, 0)
    beginner = T.sheet_source(base, 0, 0)
    texels = np.zeros(0x100000, bool)
    for tx in c.texels:
        if tx.header.sheet == 0:
            for off, blob, _ in G.texel_writes(tx, block):
                texels[off - block:off - block + len(blob)] = True
    a = np.frombuffer(out.data, np.uint8, 0x100000, block)
    b = np.frombuffer(base.data, np.uint8, 0x100000, beginner)
    assert (a[~texels] == b[~texels]).all() and texels.any()
    g = c.ground_data
    assert 4 * g.list_base >= runs["copro_data"][0] and 4 * g.pool_base >= runs["copro_data"][0]
    assert all(inside(4 * k, 4 * o.word_count, runs["polygons"]) for k, o in sd.objects.items())
    assert all(inside(4 * k, 4 * o.word_count, runs["polygons"]) for k, o in c.hud_map.objects.items())
    assert all(inside(2 * k, 8, runs["textures"]) for k in sd.headers)
    assert all(inside(2 * k, 4 * sum(map(len, v)), runs["textures"]) for k, v in sd.points.items())
    home = tgp.DATA_FREE
    assert all(inside(k - DATA_BASE, 16 * len(v) + 4, home) for k, v in sd.lists.items())
    assert inside(c.hud_map.outline_list - DATA_BASE, 20, home)
    assert c.hud_map.outline_list != G.decode_course(base, 0).hud_map.outline_list
    assert PROG_LO <= c.actor_table - 0x200000 < PROG_HI
    assert c.rolling_start[1] is False and c.grid_base[1] == build.GRID_BASES[3]
    # ids 0-2 byte-identical
    for k in range(3):
        ref = G.decode_course(base, k)
        assert _differ(G.decode_course(out, k), ref) == [], k
        assert G.mismatches(out, ref) == [], k
        for region, off, size, _ in G.coverage(base, k):
            assert getattr(out, region)[off:off + size] == getattr(base, region)[off:off + size], (k, region, off)
        blk = T.sheet_source(base, 0, k)
        assert out.data[blk:blk + 0x100000] == base.data[blk:blk + 0x100000], k
    common = T.sheet_source(base, 1)
    assert out.data[common:common + 0x100000] == base.data[common:common + 0x100000]
    # Review Focus 5: the free-bytes rule
    own = set()
    for name in ("COURSE_POINTERS", "COURSE_SHEETS", "GRID_TABLES", "ACTOR_TABLES", "CHECKPOINT_TABLES",
                 "SEGMENT_FLAGS", "HUD_MAP_LISTS", "HUD_MAP_PLACES", "HUD_MAP_ZOOMS", "GRID_BASES", "ROLLING_STARTS",
                 "TIME_TABLES[0]", "TIME_TABLES[1]", "TIME_TABLES[2]"):
        a_ = C.ID_ARRAYS[name]
        own |= set(range(a_.new - DATA_BASE + 3 * a_.stride, a_.new - DATA_BASE + 4 * a_.stride))
    rec = C.record_address(3) - DATA_BASE
    own |= set(range(rec, rec + C.COURSE_ENTRY_SIZE))
    allowed = {"data": own, "program": {at + k for at in C.LIBRARY_BOUND_AT for k in range(4)},
               "copro_data": {4 * (w + 3) + k + h for w in (tgp.SLOT_LIST, tgp.SLOT_POOL) for k in range(4)
                              for h in (0, HALF)}}
    free = {"data": [runs["data"], home], "copro_data": [runs["copro_data"]], "program": [runs["program"]],
            "polygons": [runs["polygons"]], "textures": [runs["textures"]]}
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a = np.frombuffer(getattr(base, region), np.uint8)
        b = np.frombuffer(getattr(out, region), np.uint8)
        idx = np.flatnonzero(a != b)
        ok = np.isin(idx, np.fromiter(allowed.get(region, ()), np.int64)) | (
            _runs_mask(len(a), free[region])[idx] & (a[idx] == 0xFF))
        assert ok.all(), (region, [hex(int(k)) for k in idx[~ok][:8]])
    for region, (lo_, n_) in (("polygons", build.POLYGON_FREE), ("textures", build.TEXTURES_FREE),
                              ("copro_data", tgp.COPRO_FREE)):
        assert getattr(out, region)[lo_:lo_ + n_] == getattr(base, region)[lo_:lo_ + n_], region
    used = {r: int((np.frombuffer(getattr(out, r), np.uint8)[_runs_mask(len(getattr(out, r)), [runs[r]])] != 0xFF).sum()
                   - (np.frombuffer(getattr(base, r), np.uint8)[_runs_mask(len(getattr(base, r)), [runs[r]])] != 0xFF)
                   .sum()) for r in runs}
    used["lists"] = int((np.frombuffer(out.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum()
                        - (np.frombuffer(base.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum())
    print(f"\nid 3's bytes per run (non-0xff bytes added): {used}")


def test_converted_id3_refuses(desert_id3, disc):  # noqa: F811
    """A new id is refused where it has no tables of its own: on the original set, a grown set without the guards,
    with the empty scenery; slots other than 0 and CONVERTED_IDS (bool, 8, 2); Layout places a slot-0 conversion does
    not have, or a sheet block that is not 1 MB-aligned in ID_DATA_FREE; encode with ID_FIELDS but not the model
    scenery, the model scenery without ID_FIELDS, or a converted Course whose own sheet block is moved off its run;
    every one a ConvertError."""
    base, sd, c, out = desert_id3
    rs = _romset()
    d, exe = _data(disc, 3), disc.read(EXE_NAME)
    for s, match in ((rs, "course id 3 of its own"), (C.grow_ids(rs), "course id 3 of its own")):
        with pytest.raises(ConvertError, match=match):
            build.converted_course(s, d, exe, 3, slot=3)
        with pytest.raises(ConvertError, match=match):
            build.converted_set(s, disc, 3, slot=3)
    with pytest.raises(ConvertError, match="empty"):
        build.converted_course(base, d, exe, 3, slot=3, scenery="empty")
    for slot in (True, 16, 2, 1.0, None):                # 4, 5: converted ids (Phase 10 Task 3, Phase 11 Task 4);
        with pytest.raises(ConvertError, match="slot"):  # 8-15: reverse ids (M9), refused below for a plain course
            build.converted_course(base, d, exe, 3, slot=slot)
        with pytest.raises(ConvertError, match="slot"):
            build.model_scenery(base, d, pol.Model((), (), ()), b"", slot=slot)
    with pytest.raises(ConvertError, match="slot 8: a reverse id"):
        build.converted_course(base, d, exe, 3, slot=8)
    lay = build.Layout(DATA_LO, COPRO_LO, build.PROGRAM_FREE[0], sheet_at=0x1100000)
    with pytest.raises(ConvertError, match="sheet_at"):
        build.converted_course(rs, d, exe, 3, layout=lay)
    runs = build.NEW_ID_RUNS
    ok = build.Layout(runs["data"][0], runs["copro_data"][0], build.PROGRAM_FREE[0], runs["polygons"][0])
    with pytest.raises(ConvertError, match="list_at"):
        build.converted_course(base, d, exe, 3, slot=3, layout=ok._replace(list_at=DATA_LO))   # keep: no HUD list
    for bad in (0x1100004, 0x800000, True):
        with pytest.raises(ConvertError, match="sheet_at"):
            build.converted_course(base, d, exe, 3, slot=3, scenery="model", scenery_data=sd,
                                   layout=ok._replace(sheet_at=bad, list_at=None, polygons_at=runs["polygons"][1]))
    fields = build.RACE_FIELDS + build.MODEL_FIELDS
    with pytest.raises(ConvertError, match="grid, sheet_sources"):
        build.encode(base, c, fields)
    with pytest.raises(ConvertError, match="ID_FIELDS|grid, sheet_sources"):
        build.encode(base, c, build.RACE_FIELDS + build.ID_FIELDS)
    moved = dataclasses.replace(c, sheet_sources=(0x200000,) + c.sheet_sources[1:])
    with pytest.raises(ConvertError, match="sheet_sources"):
        build.encode(base, moved, fields + build.ID_FIELDS)


def _painted(rs: RomSet, spans) -> dict[str, np.ndarray]:
    """Per region of `rs`, True on every byte of the (region, offset, length, ...) spans."""
    out = {r: np.zeros(len(getattr(rs, r)), bool) for r in ("program", "data", "copro_data", "polygons", "textures")}
    for r, o, n, *_ in spans:
        out[r][o:o + n] = True
    return out


def test_write_free_refuses_owned_other_regions(desert_id3):
    """Phase 9 Task 1: build._write_free's program / polygons / textures branch refuses a byte a table owns whatever
    its value (golden.owned_spans): a 0xffffffff word inside a polygon object passes the 0xff check and is refused,
    naming its owner; a free word of the polygons run still takes the write."""
    out = desert_id3[3]
    owned = [(o, n, w) for r, o, n, w in G.owned_spans(out) if r == "polygons"]
    word = next(((a, w) for o, n, w in owned for a in range((o + 3) & ~3, o + n - 3, 4)
                 if out.polygons[a:a + 4] == b"\xff" * 4), None)
    assert word is not None
    with pytest.raises(ConvertError, match=f"owned by {word[1]}"):
        build._write_free(out, "polygons", word[0], b"\0" * 4)
    at = build._first_fit(out, "polygons", build.NEW_ID_RUNS["polygons"], 4, "probe")
    assert build._write_free(out, "polygons", at, b"\1" * 4).polygons[at:at + 4] == b"\1" * 4


def test_final_fix1_owned_bytes_are_busy(desert_id3):
    """Final review fix 1: a byte a table or block owns is busy to every allocator whatever its value, 0xff included
    (golden.owned_spans). On the converted id-3 set that is Desert City's 1 MB texel block (thousands of 0xff runs),
    grow_ids' home (its unused record slots 0-2 are 0xff), the select's code and data, the SRAM sink and every table
    of ids 0-3 (golden.coverage: id 3's ground pool holds 0xffffffff words): the converter's free-space search
    (`_usable` over every run it searches and the custom areas whole) finds none of them free, a small allocation is
    never placed inside them, tgp.write_free refuses the texel block and grow_ids' home, and mirror.mirrored_set's
    search skips id 3's object lists in DATA_FREE. On the set before the conversion the owned bytes of the custom
    areas are the static reservations (romset.RESERVED) and, since Phase 9 Task 5, the select's mirrored map objects
    (select.map_spans: the custom polygon area's last bytes, which a first fit reaches last), which is why id 3's
    layout is unchanged."""
    from dtools.arcade import mirror as M
    from dtools.arcade import romset
    base, sd, c, out = desert_id3
    block, home = c.sheet_sources[0], C.ID_HOME - DATA_BASE
    owned = G.owned_spans(out)
    names = {(r, o, n): w for r, o, n, w in owned}
    assert ("data", block, T.SHEET_BLOCK) in names and "COURSE_SHEETS[3]" in names[("data", block, T.SHEET_BLOCK)]
    assert ("data", home, C.SRAM_SINK - C.ID_HOME) in names
    assert (np.frombuffer(out.data, np.uint8, T.SHEET_BLOCK, block) == 0xFF).sum() > 100_000
    assert out.data[home:home + 3 * C.COURSE_ENTRY_SIZE] == b"\xff" * (3 * C.COURSE_ENTRY_SIZE)
    paint = _painted(out, owned)
    for k in range(4):
        cov = _painted(out, G.coverage(out, k))
        assert all(not (cov[r] & ~paint[r]).any() for r in cov), k
    runs = {r: [build.NEW_ID_RUNS[r]] + ([build._RUNS[r]] if r in build._RUNS else [])
            + list(romset.CUSTOM_AREAS.get(r, ())) for r in build.NEW_ID_RUNS}
    for region, rr in runs.items():
        for lo, n in rr:
            usable = build._usable(out, region, (), (lo, n))
            assert not (usable & paint[region][lo:lo + n]).any(), (region, hex(lo))
    ff = paint["copro_data"] & (np.frombuffer(out.copro_data, np.uint8) == 0xFF)
    assert ff[romset.COPRO_SIZE:].any()             # id 3's ground holds 0xff bytes, owned
    # a small allocation: never inside the block or grow_ids' home
    for run, need in (((block, T.SHEET_BLOCK), 16), ((home, C.SRAM_SINK - C.ID_HOME), 4)):
        with pytest.raises(ConvertError, match="no .* free bytes"):
            build._first_fit(out, "data", run, need, "probe")
    at = build._first_fit(out, "data", build.ID_DATA_FREE, 16, "probe")
    assert not paint["data"][at:at + 16].any()
    texel = np.frombuffer(out.data, np.uint8, T.SHEET_BLOCK, block) == 0xFF
    k = next(k for k in range(0, T.SHEET_BLOCK - 16, 4) if texel[k:k + 16].all())
    with pytest.raises(tgp.TgpError, match="COURSE_SHEETS"):
        tgp.write_free(out, "data", block + k, b"\0" * 16)
    with pytest.raises(tgp.TgpError, match="reserved.*grow_ids"):
        tgp.write_free(out, "data", home, b"\0" * 4)
    # the mirror's search
    lists = _painted(out, [s for s in G.coverage(out, 3) if s[3] == "object list"])["data"]
    lo, n = tgp.DATA_FREE
    assert lists[lo:lo + n].any()
    b0 = G.decode_course(out, 0)
    others = [C.backdrop(out, e) for e in C.course_table(out) if e.course != 0]
    assert not (M._char_run(out, b0, others) & lists[lo:lo + n]).any()
    # before: only the static reservations and the select's map objects (at the polygon area's end) in the custom areas
    from dtools.arcade import select as S
    maps = [(r, o, n) for r, o, n, _ in S.map_spans(base)]
    assert maps and max(o + n for r, o, n in maps if r == "polygons") == sum(build.NEW_ID_RUNS["polygons"])
    static = _painted(base, [(r, o, n) for r, spans in romset.RESERVED.items() for o, n, _ in spans] + maps)
    before = _painted(base, G.owned_spans(base))
    for r, areas in romset.CUSTOM_AREAS.items():
        for lo, n in areas:
            assert (before[r][lo:lo + n] == static[r][lo:lo + n]).all(), (r, hex(lo))


# ---- Phase 9 Task 4: Desert City mirrored (id 11): docs/spike/2026-11-mirrors-live.md, "Desert City mirrored" ----

@pytest.fixture(scope="module")
def desert_mirror(disc):  # noqa: F811
    """Phase 9 Task 4's chain, step by step (the order: test_desert_city_before_select): (Phase 8's order on the widened
    chain: grow_ids, guard_new_ids, widen_ids, the select with M4A_ENTRIES (id 3's check lifted, since Phase 9's final
    review: test_select.unconverted_select), Desert City converted at id 3; the widened
    set with Desert City converted at id 3 and no select; that with the mirrors of 0, 1 and 2 at ids 8-10
    (mirror.mirrored_entry); that with the mirror of Desert City at id 11; that with the select of Phase 9's row,
    select.M4B_ENTRIES, entries 8-11 live: the set holds no id 4)."""
    from dtools.arcade import mirror as M
    from dtools.arcade import select as S
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    widened = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    from tests.test_select import unconverted_select
    phase8 = build.converted_set(unconverted_select(widened), disc, 3, slot=3, scenery="model")
    c3 = build.converted_set(widened, disc, 3, slot=3, scenery="model")
    before = c3
    for o in (0, 1, 2):
        before = M.mirrored_entry(before, o)
    after = M.mirrored_entry(before, 3)
    return phase8, c3, before, after, S.patch_select(after, S.M4B_ENTRIES)


def test_desert_city_before_select(desert_mirror):
    """The chain's order: entry 11 is live in M4B_ENTRIES and id 11 is not exempt from patch_select's check (select.
    FILLED_FIRST), so Desert City's mirror is written before the select, and its original, Desert City, before that:
    grow, guard, widen, convert id 3, mirror 0-3, select. Desert City converted before the select is Phase 8's order's
    set byte for byte (with Phase 8's row, M4A_ENTRIES: every region equal), so ids 0-3 of the full chain decode as
    Phase 8's order's and every byte their tables cover (golden.coverage) is the same; on the set without the
    mirror of Desert City, M4B_ENTRIES is refused for entry 11 alone. Since Phase 9's final review (item 3) id 3 is not
    exempt either (FILLED_FIRST 3): Phase 8's order itself is refused for entry 3 (built here with that check lifted),
    so every chain converts Desert City before the select."""
    from dtools.arcade import select as S
    from dtools.arcade.romset import REGIONS
    phase8, c3, before, after, full = desert_mirror
    swapped = S.patch_select(c3, S.M4A_ENTRIES)
    assert [r for r in REGIONS if getattr(swapped, r) != getattr(phase8, r)] == []
    for i in range(4):
        assert _differ(G.decode_course(full, i), G.decode_course(phase8, i)) == [], i
        for region, off, n, _ in G.coverage(phase8, i):
            assert getattr(full, region)[off:off + n] == getattr(phase8, region)[off:off + n], (i, region, off)
    with pytest.raises(S.SelectError, match=r"entries 11 \(course id 11\) are live"):
        S.patch_select(before, S.M4B_ENTRIES)
    widened = C.widen_ids(C.guard_new_ids(C.grow_ids(_romset())))
    with pytest.raises(S.SelectError, match=r"entries 3 \(course id 3\) are live"):
        S.patch_select(widened, S.M4A_ENTRIES)
    row = S.select_row(full)
    assert row.live[11] and row.panel_of_course[11] == 11 and row.records[0][11].course == 11


def test_mirrored_desert_city_round_trip(desert_mirror):
    """mirror.mirrored_entry(rs, 3) on the set with Desert City converted at id 3 (and the originals' mirrors at
    8-10): course id 11 decodes to mirror_course(decode_course(rs, 3)) with every table resolved through its pointers
    (test_mirror._content) and re-emits losslessly; Desert City's race behaviour is its own (the standing start, grid
    base 458, its checkpoints, time tables and laps: the same words, shared), its tune and VISIT_IDS
    mirror.VISIT_OF_MIRROR[3]; its texel block (COURSE_SHEETS[3], 1 MB in the custom data area) is shared, not copied:
    COURSE_SHEETS[11] names the same block, the block's bytes are unchanged and the call adds far less than a block
    to the data ROM; every per-id entry of id 11 is Desert City's but the stated ones (mirror.OWN_ARRAYS,
    MIRRORED_ARRAYS, UNSHARED_ARRAYS), and ACTOR_TABLES is Desert City's too (no actor: the shared EMPTY_ACTORS); ids
    0-3 and 8-10 decode as before and every byte of their tables is unchanged; every byte the call changes lies in a
    free run where the input held 0xff, in id 11's own words or the bound patch."""
    from dtools.arcade import mirror as M
    from tests.test_mirror import _content, _eq, _free
    _, _, rs, out, _ = desert_mirror
    k = 3 + M.MIRROR_ID_OFFSET
    orig, got = G.decode_course(rs, 3), G.decode_course(out, k)
    a, b = _content(got), _content(M.mirror_course(orig))
    assert [f for f in a if not _eq(a[f], b[f])] == []
    assert G.verify_lossless(out, k) == []
    assert got.course == got.entry.course == k and got.record_pointer == C.record_address(k)
    # Desert City's own behaviour, shared
    assert got.rolling_start[1] is False and got.grid_base[1] == orig.grid_base[1] == build.GRID_BASES[3]
    assert [cp.segment for cp in got.checkpoints] == [cp.segment for cp in orig.checkpoints] == [136, 263, 0]
    assert got.time_tables == orig.time_tables and got.car_count[1] == orig.car_count[1]   # the same words
    assert {m: v[1] for m, v in got.laps.items()} == {m: v[1] for m, v in orig.laps.items()}
    assert C.tunes(out)[k] == C.tunes(out)[3] and C.visit_ids(out)[k] == M.VISIT_OF_MIRROR[3]
    # the texel block: shared
    block = orig.sheet_sources[0]
    assert got.sheet_sources == orig.sheet_sources and build.ID_DATA_FREE[0] <= block
    assert T.sheet_source(out, 0, k) == block
    assert out.data[block:block + T.SHEET_BLOCK] == rs.data[block:block + T.SHEET_BLOCK]
    added = int((np.frombuffer(out.data, np.uint8) != np.frombuffer(rs.data, np.uint8)).sum())
    print(f"\nthe mirror of Desert City adds {added} bytes to the data ROM")
    assert added < T.SHEET_BLOCK // 2
    # the per-id arrays
    arrays = C.id_tables(out)
    differ = {n for n, x in C.ID_ARRAYS.items()
              if out.read_i960(C.id_entry(arrays, n, k), x.stride) != out.read_i960(C.id_entry(arrays, n, 3), x.stride)}
    assert differ <= M.OWN_ARRAYS | M.MIRRORED_ARRAYS | M.UNSHARED_ARRAYS, differ
    assert {"GRID_TABLES", "HUD_MAP_LISTS", "GRID_LANES", "CAMERA_SCRIPTS"} <= differ
    assert "ACTOR_TABLES" not in differ             # no actor: the shared empty table, Desert City's (Phase 10 Task 1)
    assert "COURSE_SHEETS" not in differ and "BILLBOARD_REF" not in differ and "PROGRESS_MASKS" not in differ
    # ids 0-3 and 8-10 unchanged
    for i in (0, 1, 2, 3, 8, 9, 10):
        assert _differ(G.decode_course(out, i), G.decode_course(rs, i)) == [], i
        for region, off, n, _ in G.coverage(rs, i):
            assert getattr(out, region)[off:off + n] == getattr(rs, region)[off:off + n], (i, region, off)
    own = M._own_words(rs, k)
    own += [(r, lo + COPRO_SIZE // 2, n) for r, lo, n in own if r == "copro_data"]   # its mirrored 2 MB half
    bound = [("program", at, 4) for at in C.LIBRARY_BOUND_AT]
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        x, y = (np.frombuffer(getattr(s, region), np.uint8) for s in (rs, out))
        changed = np.flatnonzero(x != y)
        inside = np.zeros(len(changed), bool)
        for lo, n in _free(rs, region):
            inside |= (changed >= lo) & (changed < lo + n) & (x[changed] == 0xFF)
        for r, lo, n in own + bound:
            if r == region:
                inside |= (changed >= lo) & (changed < lo + n)
        assert inside.all(), (region, [hex(v) for v in changed[~inside][:4]])


def test_mirror_check_desert_city(disc, desert_mirror):  # noqa: F811
    """build.mirror_check(disc, 3): the arcade's mirror of the converted Desert City (the mirror's rules on the
    converter's tables: what id 11 of the set holds) against the Dreamcast's own mirror, CSDAT3M.BIN converted the
    same way, its z moved by -1 (the Dreamcast mirrors in z = 0, the arcade in z = -0.5).

    The bounds (build.MIRROR_TOLERANCES) are set before the run: road 0.004, centre 0.35, heading 2 and lane 134 are
    Phase 5's documented numbers (a priori); point (2^-14, one f32 unit of the arcade plane's rounding) is derived,
    and that the ground corners are exact as sets is this task's measurement. Within them: every used road slot
    (2,790), every AI point (3,720; line k against M line 7 - k, Phase 5's pairing) and every ground corner (3,073
    polygons, as sets: the corner orders differ), every AI lane; steps, attributes and the checkpoints equal; the grid
    equal on every square but the two whose polygons have a corner at z = -64.5 exactly. Past them (findings,
    measured: `outside`): one ground centre, polygon 241's, 0.3548 against Phase 5's 0.35 (Phase 5's figure is
    rounded); the headings, 996 of 2,790 road slots and 1,603 of 3,720 AI points 3 or 4 off (Phase 5's "within 2"
    is the Dreamcast's own rule; the arcade's 0x2000 - h, one unit off it in either frame, adds the rest). What differs
    is named: the fields Phase 5 found recomputed on courses 3-7 and the two grid squares."""
    from dtools.arcade import mirror as M
    r = build.mirror_check(disc, 3)
    print(f"\nmirror_check(disc, 3): {r}")
    tol = build.MIRROR_TOLERANCES
    assert tol == {"road": 0.004, "point": 2.0 ** -14, "centre": 0.35, "heading": 2, "lane": 134}
    a_priori = {k for k, v in build.MIRROR_TOLERANCE_SOURCES.items() if v == "Phase 5"}
    assert a_priori == {"road", "centre", "heading", "lane"} and set(build.MIRROR_TOLERANCE_SOURCES) == set(tol)
    assert build.MIRROR_TOLERANCE_SOURCES["point"].startswith("derived")
    assert r["plane"] == -1.0
    # within the bounds
    assert r["road"]["count"] == r["road"]["within"] == 2790 and r["road"]["max"] <= tol["road"]
    assert r["road"]["lane_ranges"] == (465, 465)
    assert r["ai_lines"]["order"] == tuple(range(7, -1, -1))
    assert r["ai_lines"]["count"] == r["ai_lines"]["within"] == 3720 and r["ai_lines"]["max"] <= tol["point"]
    assert r["ai_lines"]["steps"] == r["ai_lines"]["lanes_within"] == 3720 and r["ai_lines"]["lanes_max"] == 132
    g = r["ground"]
    assert g["count"] == g["corners_within"] == 3073 and g["corners_max"] <= tol["point"]
    assert r["grid"]["count"] == 16384 and r["grid"]["equal"] == 16382 and r["grid"]["differ"] == ((59, 77), (60, 77))
    assert r["steps"] == (527, 527) and r["attributes"] == (3073, 3073)
    assert r["checkpoints"] == {"arcade": (136, 263), "dreamcast": (136, 263)}
    # past the bounds: the findings (measured)
    assert set(r["outside"]) == {"road headings", "AI headings", "ground centres"}
    assert g["centres_within"] == 3072 and g["centres_outside"] == (241,) and 0.35 < g["centres_max"] < 0.355
    assert (r["road"]["headings_within"], r["road"]["headings_max"]) == (1794, 4)
    assert (r["ai_lines"]["headings_within"], r["ai_lines"]["headings_max"]) == (2117, 4)
    assert set(r["differs"]) == {"road positions", "road headings", "AI line order", "AI headings", "AI lanes",
                                 "ground directions", "ground centres", "grid squares"}
    # the arcade side is the mirror the set holds at id 11
    road, lines, g = build._mirror_tables(_data(disc, 3))
    got = G.decode_course(desert_mirror[3], 3 + M.MIRROR_ID_OFFSET)
    assert got.road == road and got.ai_lines == lines
    assert (got.ground_data.polys, got.ground_data.grid, got.ground_data.lists) == (g.polys, g.grid, g.lists)
    for bad in (True, 8, -1, "3", 3.0):
        with pytest.raises(ConvertError):
            build.mirror_check(disc, bad)
    with pytest.raises(ConvertError):
        build.mirror_check("disc", 3)


# ---- Phase 10 Task 1: program room, shared empty actor tables (docs/spike/2026-11-courses-4-7.md, "Program room") ----

def _actor_tables(rs: RomSet, ids) -> dict[int, int]:
    """ACTOR_TABLES[k] of the grown set `rs` for each id in `ids`."""
    arrays = C.id_tables(rs)
    return {k: struct.unpack("<I", rs.read_i960(C.id_entry(arrays, "ACTOR_TABLES", k), 4))[0] for k in ids}


def test_empty_actor_table_shared(desert_mirror):
    """Review Focus 3: an actor table with no actors is shared, not copied, and only such a table. On the chain
    (grow, guard, widen, Desert City converted at id 3, the mirrors of 0-3 at 8-11, the select) Desert City and its
    mirror name build.EMPTY_ACTORS, the one shared table (256 zero words at the program run's first bytes, program
    window 0x0023ce98): its span is reserved (romset.RESERVED, so owned by golden.owned_spans whatever its bytes, and
    no allocator or free-space writer takes it), and both ids' tables are owned there. The mirrors of the originals,
    whose tables name actor lists, keep tables of their own, distinct and not the shared one; on every filled id the
    table is EMPTY_ACTORS exactly when it names no list. The mirror of Desert City adds no byte to the program run
    (the table is shared), and two ids naming one table leave golden.coverage, owned_spans and verify_lossless sound.
    A set whose EMPTY_ACTORS span holds other bytes is refused (test_empty_actor_table_written_once:
    build.empty_actors, which the converter and the mirror both write it with), and the free-space writer refuses the
    span."""
    from dtools.arcade import mirror as M
    from dtools.arcade import romset
    phase8, c3, before, after, full = desert_mirror
    at = build.EMPTY_AT
    assert build.EMPTY_ACTORS == 0x0023CE98 and at == build.PROGRAM_FREE[0] == build.EMPTY_ACTORS - 0x200000
    assert (at, 4 * C.GRID_CELLS) in [(o, n) for o, n, _ in romset.RESERVED["program"]]
    assert "EMPTY_ACTORS" in romset.reserved("program", at, 4)
    ids = (0, 1, 2, 3, 8, 9, 10, 11)
    tables = _actor_tables(full, ids)
    assert tables[3] == tables[11] == build.EMPTY_ACTORS
    assert full.read_i960(build.EMPTY_ACTORS, 4 * C.GRID_CELLS) == bytes(4 * C.GRID_CELLS)
    for k in ids:
        c = G.decode_course(full, k)
        assert c.actor_table == tables[k] and (tables[k] == build.EMPTY_ACTORS) == (not any(c.actor_slots)), k
        assert G.verify_lossless(full, k) == [], k
    own = [tables[k] for k in (0, 1, 2, 8, 9, 10)]
    assert len(set(own)) == len(own) and build.EMPTY_ACTORS not in own
    owners = {w for r, o, n, w in G.owned_spans(full) if (r, o, n) == ("program", at, 4 * C.GRID_CELLS)}
    assert {"actor table of course 3", "actor table of course 11"} <= owners and len(owners) == 3
    # the mirror of Desert City: no byte of the program run (its widened words are in WIDE_HOME), the table named again
    assert after.program[PROG_LO:PROG_HI] == before.program[PROG_LO:PROG_HI]
    assert _actor_tables(after, (11,))[11] == _actor_tables(before, (3,))[3]
    # the span is not free space
    with pytest.raises(ConvertError, match="reserved"):
        build._write_free(C.widen_ids(C.guard_new_ids(C.grow_ids(_romset()))), "program", at, b"\0" * 4)


def test_empty_actor_table_written_once(rs):
    """build.empty_actors: the shared table written into its reserved span when it is free (0xff), the set unchanged
    when it holds the table already; ConvertError for other bytes there or an argument that is not a RomSet. A slot-0
    conversion with the empty scenery names it too (the same bytes as before the rule: its table was the run's first
    1,024 bytes), `encode` writes it or finds it written and refuses the span holding other bytes, and the converter's
    layout takes no other actor table (Layout.program_at must be EMPTY_AT)."""
    at, n = build.EMPTY_AT, 4 * C.GRID_CELLS
    assert rs.program[at:at + n] == b"\xff" * n
    out = build.empty_actors(rs)
    assert out.program[at:at + n] == bytes(n)
    assert build.empty_actors(out) is out
    for bad in (rs.with_patch("program", at + n - 4, b"\0\0\0\1"), rs.with_patch("program", at, b"\0" * 4)):
        with pytest.raises(ConvertError, match="EMPTY_ACTORS"):
            build.empty_actors(bad)
    for bad in (None, "rs", b"\xff"):
        with pytest.raises(ConvertError, match="RomSet"):
            build.empty_actors(bad)
    c = build.converted_course(rs, _syn(), SYN_EXE, 0, scenery="empty")
    assert c.actor_table == build.EMPTY_ACTORS
    fields = build.RACE_FIELDS + build.SCENERY_FIELDS
    assert build.encode(rs, c, fields).program[at:at + n] == bytes(n)
    assert build.encode(out, c, fields).program[at:at + n] == bytes(n)
    with pytest.raises(ConvertError, match="EMPTY_ACTORS"):
        build.encode(rs.with_patch("program", at + 8, b"\1\0\0\0"), c, fields)
    for p in (at + 4, at + n, build.PROGRAM_FREE[0] + build.PROGRAM_FREE[1] - n):
        with pytest.raises(ConvertError, match="program_at"):
            build.converted_course(rs, _syn(), SYN_EXE, 0, scenery="empty", layout=build.Layout(DATA_LO, COPRO_LO, p))


def test_program_room(desert_mirror):
    """build.program_room: the free bytes of PROGRAM_FREE (0xff and owned by nothing). 8,808 on the stock and the
    grown set (12,648 less the shared table's reserved 1,024, since v0.3 the far view's hook's 256 and since v0.4
    the car choice's 768 and the Car Select screen's 1,792); Desert City's conversion and its mirror take none of
    it (both share EMPTY_ACTORS); the mirrors of the originals take their own tables and lists (1,896 / 2,588 /
    1,828); the final chain (grow_rankings before the select, as test_mirrors_mame.final_set) leaves 2,496 (5,312 before v0.3's
    256 for the far view and v0.4's 2,560 for the cars), where Phase 9 left 4,288. M5's eight entries (courses 4-7 converted, their mirrors 12-15) have no actors, so each shares
    the table (per course: test_every_conversion_shares_the_empty_table; a mirror writing it itself:
    test_mirror_writes_the_empty_table): they need none of the run, and the 2,496 bytes stay for a later course with
    actors."""
    from dtools.arcade import mirror as M
    from dtools.arcade import select as S
    phase8, c3, before, after, full = desert_mirror
    stock = _romset()
    widened = C.widen_ids(C.guard_new_ids(C.grow_ids(stock)))
    assert build.program_room(stock) == build.program_room(widened) == 12648 - 1024 - 256 - 768 - 1792 == 8808
    assert build.program_room(c3) == build.program_room(widened)
    rooms = [build.program_room(c3)]
    s = c3
    for o in (0, 1, 2):
        s = M.mirrored_entry(s, o)
        rooms.append(build.program_room(s))
    assert [a - b for a, b in zip(rooms, rooms[1:])] == [1896, 2588, 1828]
    assert build.program_room(after) == build.program_room(before) == rooms[-1] == 2496
    final = S.patch_select(C.grow_rankings(after), S.M4B_ENTRIES)
    print(f"\nprogram room: stock {build.program_room(stock)}, final chain {build.program_room(final)} (Phase 9: 4288)")
    assert build.program_room(final) == 2496
    for bad in (None, "rs", 3):
        with pytest.raises(ConvertError, match="RomSet"):
            build.program_room(bad)


def test_every_conversion_shares_the_empty_table(disc):  # noqa: F811
    """test_program_room's "M5 needs 0 bytes", per course: every Dreamcast course n 0-7 converts (n 4-7 since Phase 10
    Task 2's ground repair) and names build.EMPTY_ACTORS (slot 0, the empty scenery; `_check_actors_hud` refuses any
    other table, so the model scenery and a new id cannot name another). A course build.LIFT names converts lifted (n 7:
    converted_course refuses it unlifted, Task 4 fix round 1), one build.SHIFT names shifted (n 6, Phase 11 Task 3)."""
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    exe = disc.read(EXE_NAME)
    for n in range(8):
        d = build.lifted_data(_data(disc, n), build.LIFT[n]) if n in build.LIFT else _data(disc, n)
        d = build.shifted_data(d, build.SHIFT[n]) if n in build.SHIFT else d
        c = build.converted_course(rs, d, exe, n, scenery="empty")
        assert c.actor_table == build.EMPTY_ACTORS and not any(c.actor_slots) and c.actor_lists == {}, n


def test_mirror_writes_the_empty_table(rs):
    """mirror.mirrored_entry writes the shared table itself when its original has no actor and no course named it yet
    (the span still 0xff): the widened chain with Beginner's actor table swapped for an all-zero one of its own (a
    synthetic actor-less original, its 1,024 zero words past the shared span), then the mirror of Beginner at id 8.
    Id 8 names EMPTY_ACTORS, the span now holds the 256 zero words, nothing else of PROGRAM_FREE changes (no table or
    list of the mirror's own: the `shared` skip of mirrored_entry's byte check), and id 8 re-emits losslessly."""
    from dtools.arcade import mirror as M
    at, n = build.EMPTY_AT, 4 * C.GRID_CELLS
    w = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    own = at + n                                        # Beginner's synthetic table, right past the shared span
    s = w.with_patch("program", own, bytes(n))
    for k in [0] + list(range(C.COURSE_COUNT, C.COURSE_IDS)):     # Beginner and its copies at ids 3-15 (unfilled)
        region, off = C.rom_word_at(C.id_entry(C.id_tables(w), "ACTOR_TABLES", k))
        s = s.with_patch(region, off, struct.pack("<I", own + 0x200000))
    assert not any(G.decode_course(s, 0).actor_slots) and s.program[at:at + n] == b"\xff" * n
    out = M.mirrored_entry(s, 0)
    assert _actor_tables(out, (8,))[8] == build.EMPTY_ACTORS
    assert out.program[at:at + n] == bytes(n)
    assert out.program[own + n:PROG_HI] == s.program[own + n:PROG_HI]
    assert build.program_room(out) == build.program_room(s)
    assert G.verify_lossless(out, 8) == []


# ---- the reverse variant (M9 stage 1) ----

def test_reverse_variant_refusals(disc, rs):
    """converted_set's variant: "" or "R"; "R" fills only the reverse id n + 8 (REVERSE_IDS), "" never a reverse id;
    the arcade's three (n 0-2) are refused in reverse (their scenery is the arcade's: stage 2). Refused before any
    file is read."""
    g = C.guard_new_ids(C.grow_ids(rs))
    assert build.REVERSE_IDS == tuple(range(8, 16))
    for kw in ({"slot": 11, "variant": ""}, {"slot": 11}, {"slot": 3, "variant": "R"}, {"slot": 12, "variant": "R"},
               {"slot": 0, "variant": "R"}, {"slot": 11, "variant": "M"}, {"slot": 11, "variant": None}):
        with pytest.raises(ConvertError, match="variant|slot"):
            build.converted_set(g, disc, 3, scenery="model", **kw)
    with pytest.raises(ConvertError, match="arcade|stage 2"):
        build.converted_set(g, disc, 0, slot=8, variant="R", scenery="model")
    with pytest.raises(ConvertError, match="slot"):
        build.converted_set(rs, disc, 3, slot=11, variant="R")         # not grown: no id 11 of its own


def test_reverse_variant_reads_the_r_files(disc, rs, monkeypatch):
    """converted_set(..., 3, slot=11, variant="R", scenery="model") reads CSDAT3R.BIN, the plain CS_DATA3.BIN (the
    ground's lender, handed to converted_course as `plain`: M9 stage 2, step 3), 1ST_READ.BIN, CS3R_POL.BIN and
    CS3R_TEX.BIN, never the plain model or textures, and hands converted_course the variant (the conversion itself
    stubbed after the reads)."""
    g = C.guard_new_ids(C.grow_ids(rs))
    read, seen = [], {}
    real = disc.read
    monkeypatch.setattr(disc, "read", lambda name: read.append(name) or real(name))
    budget = {}
    monkeypatch.setattr(build, "model_scenery", lambda rs_, d, m, t, thin, **k: budget.update(k, thin=thin) or "scenery")
    monkeypatch.setattr(build, "converted_course", lambda *a, **k: seen.update(k) or "course")
    monkeypatch.setattr(build, "encode", lambda s, c, fields: s)
    monkeypatch.setattr(build, "ROAD_COPY", {3: (16.0, 10.0)})        # v0.2: a course's road copy, both directions
    build.converted_set(g, disc, 3, slot=11, variant="R", scenery="model", road_copy=True)
    assert read == ["CSDAT3R.BIN", "CS_DATA3.BIN", "1ST_READ.BIN", "CS3R_POL.BIN", "CS3R_TEX.BIN"]
    assert budget["road_copy"] == (16.0, 10.0)
    budget.clear()
    build.converted_set(g, disc, 3, slot=11, variant="R", scenery="model")       # the default: no copy (the old pins)
    assert budget["road_copy"] is None
    assert seen["variant"] == "R" and seen["slot"] == 11 and seen["plain"] is not None
    # the reverse model's own budget settings (REVERSE_*), never the plain model's per-object THIN / NAMED_PROPS
    assert budget["thin"] == build.REVERSE_THIN.get(3) and budget["thin"] != build.THIN[3]
    assert budget["quad"] == build.REVERSE_QUAD.get(3) and budget["remesh"] == build.REVERSE_REMESH.get(3)
    assert budget["props"] == build.REVERSE_NAMED_PROPS.get(3, ()) and budget["cell_cut"] == build.CELL_CUT.get(3)
