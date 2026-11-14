import functools
import math
import struct
import zlib

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import tgp
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import COPRO_SIZE, DATA_SIZE, PROGRAM_SIZE, RomSet
from dtools.arcade.tablebase import DATA_BASE

# The upload as logged (docs/spike/2026-11-tgp-ground.md, "The program"): the i960 loop at 0xfa0 writes 2,024 words
# to the FIFO at frame 26 of a fresh boot; the CRC32 is of those words as captured by the write tap (little-endian).
UPLOAD_WORDS = 2024
UPLOAD_CRC32 = 0xD6D611DD
BRIF_ALW = 0xBF600000                    # "brif alw #adr": the dispatch table's entries


def _rs(data: bytes = b"") -> RomSet:
    return RomSet(program=bytes(PROGRAM_SIZE), data=data, copro_data=b"", polygons=b"", textures=b"", extra={})


def test_tgp_error_is_a_decode_error():
    assert issubclass(tgp.TgpError, DecodeError)


def test_program_short_region_raises():
    region, offset, count = tgp.PROGRAM
    assert region == "data"
    with pytest.raises(tgp.TgpError, match="program"):
        tgp.program(_rs(bytes(offset + 4 * count - 4)))


def test_program_real(romset_zip):
    rs = RomSet.from_zip(romset_zip)
    words = tgp.program(rs)
    assert len(words) == UPLOAD_WORDS
    assert zlib.crc32(struct.pack(f"<{len(words)}I", *words)) == UPLOAD_CRC32
    region, offset, count = tgp.PROGRAM
    assert words == struct.unpack_from(f"<{count}I", getattr(rs, region), offset)
    assert 0 <= tgp.FUNCTIONS[0x36] < len(words)
    for fn, handler in tgp.FUNCTIONS.items():          # each handler is the dispatch table's entry for its number
        assert words[tgp.DISPATCH + fn] == BRIF_ALW | handler


# ---- the ground (function 0x36's data): docs/formats/model2-tgp-ground.md ----

GRID_AT, ATTR_AT = 0x100000, 0x108000           # synthetic data-ROM offsets of the grid and the attribute block
ENTRY_AT = 0x200000                             # synthetic course records' pointer words (data offsets)


def _cell(row: int, col: int) -> int:
    return row * 128 + col


def _sq(x: float, z: float, s: float = 8.0, y: float = 1.0, d: int = 0) -> "tgp.GroundPoly":
    """A square in the originals' winding (negative signed XZ area): (x, z), (x, z + s), (x + s, z + s), (x + s, z)."""
    corners = ((x, y, z), (x, y, z + s), (x + s, y, z + s), (x + s, y, z))
    return tgp.GroundPoly(corners, centre=(x + s / 2, y, z + s / 2), direction=d)


def _ground(**kw) -> "tgp.Ground":
    grid = [0] * tgp.GRID_CELLS
    grid[_cell(64, 64)] = 1                     # list at word 1: polygons 0, 1
    grid[_cell(64, 65)] = 4                     # list at word 4: polygons 2, 3, 4
    polys = (_sq(0, 0), _sq(8, 0, d=-5), _sq(16, 0, y=2.5, d=0x4000), _sq(16, 8, d=-0x8000), _sq(24, 0))
    g = dict(course=0, grid=tuple(grid), attributes=bytes([1, 1, 2, 5, 8]), polys=polys,
             lists={1: (0, 1), 4: (2, 3, 4)}, base=0, raw=(), grid_at=GRID_AT, attributes_at=ATTR_AT,
             list_base=0x30, pool_base=0x1000)
    g.update(kw)
    return tgp.Ground(**g)


def _rom(g: "tgp.Ground", patches=()) -> RomSet:
    """A blank set with a course table whose three records point at `g`'s grid and attribute block, and `g`
    encoded into it; then `patches` ((region, offset, bytes)) on top."""
    program, data = bytearray(PROGRAM_SIZE), bytearray(DATA_SIZE)
    for i in range(C.COURSE_COUNT):
        struct.pack_into("<I", program, C.COURSE_POINTERS + 4 * i, C.COURSE_TABLE + i * C.COURSE_ENTRY_SIZE)
        words = [DATA_BASE + ENTRY_AT + 0x100 * i + 0x10 * k for k in range(6)] + [0, 0]
        struct.pack_into("<8I", data, C.COURSE_TABLE - DATA_BASE + i * C.COURSE_ENTRY_SIZE, *words)
        struct.pack_into("<I", data, words[2] - DATA_BASE, DATA_BASE + g.grid_at)        # *block_ref_a
        struct.pack_into("<I", data, words[3] - DATA_BASE, DATA_BASE + g.attributes_at)  # *block_ref_b
    rs = RomSet(program=bytes(program), data=bytes(data), copro_data=b"\xff" * COPRO_SIZE, polygons=b"",
                textures=b"", extra={})                        # copro_data all free (0xff), as past the real pools
    for (region, off), blob in tgp.encode_ground(g).items():
        rs = rs.with_patch(region, off, blob)
    for region, off, blob in patches:
        rs = rs.with_patch(region, off, blob)
    return rs


def _same(a: "tgp.Ground", b: "tgp.Ground") -> None:
    """Equal in every decoded field (raw, the words as read, aside)."""
    for f in ("course", "grid", "attributes", "lists", "base", "grid_at", "attributes_at", "list_base", "pool_base"):
        assert getattr(a, f) == getattr(b, f), f
    assert len(a.polys) == len(b.polys)
    for p, q in zip(a.polys, b.polys):
        assert (p.corners, p.centre, p.direction) == (q.corners, q.centre, q.direction)


def test_ground_round_trip_synthetic():
    g = _ground()
    rs = _rom(g)
    d = tgp.decode_ground(rs, 0)
    _same(d, g)
    assert d.raw == (0, 2, 0, 1, 3, 2, 3, 4)                       # the list block as stored, word 0 = off road
    assert d.polys[1].raw[15] == 0xFFFFFFFB                         # direction -5, sign-extended
    assert struct.unpack("<f", struct.pack("<I", d.polys[2].raw[1]))[0] == 2.5
    assert tgp.encode_ground(d) == tgp.encode_ground(g)
    enc = tgp.encode_ground(g)
    assert enc[("data", ATTR_AT)] == bytes([1, 1, 2, 5, 8, 0xCD, 0xCD, 0xCD])   # padded to 4 like the originals
    assert enc[("copro_data", 4 * (tgp.SLOT_LIST + 0))] == struct.pack("<I", 0x30)
    assert enc[("copro_data", 4 * (tgp.SLOT_POOL + 0))] == struct.pack("<I", 0x1000)
    assert len(enc[("copro_data", 4 * 0x1000)]) == 4 * tgp.POLY_WORDS * 5
    assert len(enc[("data", GRID_AT)]) == 2 * tgp.GRID_CELLS
    assert tgp.verify_lossless_ground(rs, 0) == []
    assert sorted((r, o) for r, o, _, _ in tgp.ground_spans(rs, 0)) == sorted(enc)
    assert {(r, o): n for r, o, n, _ in tgp.ground_spans(rs, 0)} == {k: len(v) for k, v in enc.items()}
    # a centre left out is the corners' mean
    p = tgp.GroundPoly(g.polys[0].corners)
    pool = tgp.encode_ground(_ground(polys=(p,) + g.polys[1:]))[("copro_data", 4 * 0x1000)]
    words = struct.unpack_from("<16f", pool)
    assert words[12:15] == (4.0, 1.0, 4.0) and words[15] == 0.0


def test_verify_lossless_reports_difference():
    g = _ground()
    rs = _rom(g, [("data", ATTR_AT + 5, b"\x00")])                 # a padding byte the encoder writes as 0xcd
    problems = tgp.verify_lossless_ground(rs, 0)
    assert len(problems) == 1 and "attributes" in problems[0]


def test_decode_course_out_of_range_raises():
    rs = _rom(_ground())
    for course in (-1, 3, True):
        with pytest.raises(tgp.TgpError, match="course"):
            tgp.decode_ground(rs, course)


def test_grid_value_past_data_raises():
    g = _ground()
    # slot 1's list block starts at word 0x40: course 0's lists end there (0x30 + 0x10 words)
    bound = ("copro_data", 4 * (tgp.SLOT_LIST + 1), struct.pack("<I", 0x40))
    cell = _cell(70, 3)
    rs = _rom(g, [bound, ("data", GRID_AT + 2 * cell, struct.pack("<H", 0x10))])
    with pytest.raises(tgp.TgpError, match=rf"cell {cell}\b.*0x10\b"):
        tgp.decode_ground(rs, 0)
    # a count that runs the list past the bound
    rs = _rom(g, [bound, ("data", GRID_AT + 2 * cell, struct.pack("<H", 0xC)),
                  ("copro_data", 4 * (0x30 + 0xC), struct.pack("<I", 5))])
    with pytest.raises(tgp.TgpError, match=rf"cell {cell}\b.*0xc\b"):
        tgp.decode_ground(rs, 0)
    # a list that crosses a 64K-word window (list 4 at words 0xfffd-0x10000): the TGP's 16-bit offset would wrap.
    # Built by hand: the encoder refuses it.
    rs = _rom(g, [("copro_data", 4 * tgp.SLOT_LIST, struct.pack("<I", 0xFFF9)),
                  ("copro_data", 4 * 0xFFF9, struct.pack("<8I", 0, 2, 0, 1, 3, 2, 3, 4))])
    with pytest.raises(tgp.TgpError, match=rf"cell {_cell(64, 65)}\b.*0x4\b.*window"):
        tgp.decode_ground(rs, 0)


def test_poly_index_past_count_raises():
    g = _ground()
    # slot 1's pool starts 4 polygons after course 0's: index 4 lies in another course's data
    rs = _rom(g, [("copro_data", 4 * (tgp.SLOT_POOL + 1), struct.pack("<I", 0x1000 + 4 * tgp.POLY_WORDS))])
    with pytest.raises(tgp.TgpError, match=rf"cell {_cell(64, 65)}\b.*0x4\b.*polygon 4\b"):
        tgp.decode_ground(rs, 0)


def test_unreferenced_polygon_raises():
    """Five polygons in the ROM (count 1 mod 4, so the attribute padding cannot tell), list 4 cut to (2, 3): the
    lists name four. Decoding four would drop polygon 4 unseen; the decoder refuses, whether the pool is bounded by a
    following slot address or is the last (bounded by the copy's end): every record past the fourth up to the bound
    must be free."""
    g = _ground()
    cut = ("copro_data", 4 * (0x30 + 4), struct.pack("<I", 2))
    with pytest.raises(tgp.TgpError, match="polygon 4 is in no list"):
        tgp.decode_ground(_rom(g, [cut]), 0)
    bound = ("copro_data", 4 * (tgp.SLOT_POOL + 1), struct.pack("<I", 0x1000 + 5 * tgp.POLY_WORDS))
    with pytest.raises(tgp.TgpError, match="holds 5 polygons.*polygon 4 is in no list"):
        tgp.decode_ground(_rom(g, [cut, bound]), 0)
    assert len(tgp.decode_ground(_rom(g, [bound]), 0).polys) == 5        # count == capacity: fine
    _refused("polygon 4 is in no list", lists={1: (0, 1), 4: (2, 3)})


def test_pool_free_records_up_to_the_bound():
    """A pool may hold fewer polygons than fit before the next slot-table address when every record past the last
    listed one is free (FREE_WORD x POLY_WORDS): the layout a pool has once the block after it moved away and was
    erased. A record there that is neither listed nor free is refused, wherever it lies, for every pool: the last
    one's bound is the copy's end."""
    g = _ground()
    bound = ("copro_data", 4 * (tgp.SLOT_POOL + 1), struct.pack("<I", 0x1000 + 8 * tgp.POLY_WORDS))
    _same(tgp.decode_ground(_rom(g, [bound]), 0), g)                      # records 5-7 free
    used = ("copro_data", 4 * (0x1000 + 6 * tgp.POLY_WORDS + 3), b"\x00" * 4)
    with pytest.raises(tgp.TgpError, match="holds 8 polygons.*record 6 .*not free.*polygon 6 is in no list"):
        tgp.decode_ground(_rom(g, [bound, used]), 0)
    far = ("copro_data", 4 * (0x1000 + 1000 * tgp.POLY_WORDS), b"\x00" * 4)
    with pytest.raises(tgp.TgpError, match="record 1000 .*not free.*polygon 1000 is in no list"):
        tgp.decode_ground(_rom(g, [far]), 0)                              # the last pool: to the copy's end
    odd = _ground(pool_base=0x1004)                                      # 12 words after the last whole record
    tail = ("copro_data", 4 * (tgp.HALF_WORDS - 4), b"\x00" * 4)          # no polygon fits there: not checked
    _same(tgp.decode_ground(_rom(odd, [tail]), 0), odd)


def test_orphan_list_refused():
    """A list no grid cell reaches is invisible to the decoder: the encoder refuses it."""
    _refused("list 0x9 is reached by no grid cell", lists={1: (0, 1), 4: (2, 3, 4), 9: (0,)})


def _refused(match: str, **kw) -> None:
    with pytest.raises(tgp.TgpError, match=match):
        tgp.encode_ground(_ground(**kw))


def test_encode_refuses_bad_poly():
    g = _ground()

    def with_poly(p):
        return {"polys": (p,) + g.polys[1:]}
    c = list(g.polys[0].corners)
    _refused("polygon 0", **with_poly(tgp.GroundPoly(((math.nan, 1.0, 0.0),) + tuple(c[1:]))))
    _refused("polygon 0", **with_poly(tgp.GroundPoly(((0.0, math.inf, 0.0),) + tuple(c[1:]))))
    _refused("polygon 0", **with_poly(tgp.GroundPoly(tuple(c), centre=(math.nan, 0.0, 0.0))))
    _refused("area", **with_poly(tgp.GroundPoly(((0, 0, 0), (0, 1, 8), (0, 2, 16), (0, 3, 4)))))   # a wall
    _refused("1024", **with_poly(_sq(1016, 0)))                       # corners at x = 1024
    _refused("1024", **with_poly(_sq(0, -1032)))                      # z = -1032
    _refused("winding", **with_poly(tgp.GroundPoly(tuple(reversed(c)))))   # the TGP's one-sided test never hits it
    _refused("convex", **with_poly(tgp.GroundPoly(((0, 0, 0), (0, 0, 8), (2, 0, 2), (8, 0, 0)))))
    _refused("corners", **with_poly(tgp.GroundPoly(tuple(c[:3]))))
    _refused("corners", **with_poly(tgp.GroundPoly(None)))
    tgp.encode_ground(_ground(**with_poly(_sq(-1024, 1008))))         # -1024 is inside the grid, 1023.x too


def test_encode_refuses_field_overflow():
    g = _ground()
    grid = list(g.grid)
    grid[5] = 0x10000
    _refused(r"cell 5\b.*0x10000", grid=tuple(grid))                  # the grid is u16
    grid[5] = 7
    _refused(r"cell 5\b.*0x7\b", grid=tuple(grid))                    # a value with no list
    grid[5] = True
    _refused(r"cell 5\b", grid=tuple(grid))
    _refused("grid", grid=g.grid[:-1])
    _refused("polygon 5", lists={1: (0, 1), 4: (2, 3, 5)})            # past the polygon count
    _refused("polygon", lists={1: (0, -1), 4: (2, 3, 4)})
    _refused("0x10000", lists={1: (0, 1), 4: (2, 3, 4), 0x10000: (0,)})
    _refused("overlap", lists={1: (0, 1), 3: (2, 3, 4)})              # list 1 runs over word 3
    _refused("off-road", lists={0: (1,), 1: (0, 1), 4: (2, 3, 4)})
    _refused("direction", polys=g.polys[:4] + (_sq(24, 0, d=0x8000),))
    _refused("direction", polys=g.polys[:4] + (_sq(24, 0, d=-0x8001),))
    _refused("attributes", attributes=bytes(4))
    _refused("slot", base=16)
    _refused("slot", base=True)
    _refused("window", list_base=0xFFF9)                              # list 4 would wrap inside the TGP's bank
    _refused("window", pool_base=0xFFF8)
    _refused("half", pool_base=tgp.HALF_WORDS - 4 * tgp.POLY_WORDS)   # 5 polygons do not fit before the copy
    _refused("overlap", pool_base=0x34)                               # over the list block
    _refused("slot table", list_base=0x20)


@functools.lru_cache(maxsize=1)
def _real(path) -> RomSet:
    return RomSet.from_zip(path)


# Checkpoints (global constraints and docs/formats/model2-course-tables.md): per course id 0 Beginner, 1 Expert,
# 2 Advanced. The attribute block is the polygon count padded to 4 bytes with 0xcd (1 and 2 bytes on Expert and
# Advanced): its extent on the real set (to the next step table) is 1,184 / 7,240 / 2,988 bytes.
POLYS = {0: 1184, 1: 7239, 2: 2986}
ATTR_BLOCK = {0: 1184, 1: 7240, 2: 2988}
ROAD = {0: 503, 1: 1064, 2: 818}
CELLS = {0: 626, 1: 2315, 2: 1288}


def test_counts_real(romset_zip):
    rs = _real(romset_zip)
    for course in range(3):
        g = tgp.decode_ground(rs, course)
        entry = C.course_table(rs)[course]
        assert g.base == course                                     # slot = course id (i960 0x46dc, function 0x0d)
        assert len(g.polys) == POLYS[course]
        table = struct.unpack_from(f"<{2 * tgp.SLOTS}I", rs.copro_data, 4 * tgp.SLOT_LIST)
        nxt = min(a for a in table if a > g.pool_base)               # each course's pool ends at the next block
        assert len(g.polys) == (nxt - g.pool_base) // tgp.POLY_WORDS and (nxt - g.pool_base) % tgp.POLY_WORDS == 0
        assert len(g.attributes) == POLYS[course]
        assert len(tgp.encode_ground(g)[("data", g.attributes_at)]) == ATTR_BLOCK[course]
        assert sum(1 for v in g.grid if v) == CELLS[course] == len(g.lists)
        assert len(C.step_table(rs, entry)) == ROAD[course]
        assert g.grid_at == struct.unpack_from("<I", rs.data, entry.block_ref_a)[0] - DATA_BASE
        assert g.attributes_at == struct.unpack_from("<I", rs.data, entry.block_ref_b)[0] - DATA_BASE
        used = {i for v in g.lists.values() for i in v}
        assert used == set(range(len(g.polys)))                     # every polygon is in some list


def _seg_dist(x: float, z: float, a, b) -> float:
    dx, dz = b[0] - a[0], b[2] - a[2]
    n = dx * dx + dz * dz
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[2]) * dz) / n))
    return math.hypot(x - a[0] - t * dx, z - a[2] - t * dz)


def test_road_polys_on_road(romset_zip):
    """The first `steps` polygons are the road (attribute kind 1, and no other polygon is); those not flagged pit
    have their centre within 15 units in XZ of the road line's centreline, the pit-flagged ones (STEP_PIT: the pit
    lane and box, 13-125 units off the road line) within 130."""
    rs = _real(romset_zip)
    for course in range(3):
        entry = C.course_table(rs)[course]
        g = tgp.decode_ground(rs, course)
        steps = C.step_table(rs, entry)
        assert [i for i, a in enumerate(g.attributes) if a & 15 == 1] == list(range(len(steps)))
        line = C.centreline(C.road_line(rs, entry))
        segs = list(zip(line, line[1:] + line[:1]))
        road = [i for i, s in enumerate(steps) if not s.flags & C.STEP_PIT]
        assert len(road) > 0.8 * len(steps)
        for i, s in enumerate(steps):
            x, _, z = g.polys[i].centre
            bound = 130.0 if s.flags & C.STEP_PIT else 15.0
            assert min(_seg_dist(x, z, a, b) for a, b in segs) <= bound, (course, i)


def test_verify_lossless_ground(romset_zip):
    rs = _real(romset_zip)
    for course in range(3):
        assert tgp.verify_lossless_ground(rs, course) == []


# Ground data the three courses share: none. Each course has its own two slot-table words (0x10 + slot,
# 0x20 + slot), list block, polygon pool, grid and attribute block. The rest of copro_data's header (the identity
# matrix at words 0x00-0x0f, slots 3-15) and slot 3's list block and pool belong to no course.
SHARED: set[tuple[str, int, int]] = set()


def test_ground_spans_disjoint(romset_zip):
    rs = _real(romset_zip)
    spans = [(course, s) for course in range(3) for s in tgp.ground_spans(rs, course)]
    for course in range(3):
        g = tgp.decode_ground(rs, course)
        mine = {(r, o): n for c, (r, o, n, _) in spans if c == course}
        assert mine == {k: len(v) for k, v in tgp.encode_ground(g).items()}
    for i, (c1, (r1, o1, n1, name1)) in enumerate(spans):
        for c2, (r2, o2, n2, name2) in spans[i + 1:]:
            if r1 == r2 and o1 < o2 + n2 and o2 < o1 + n1:
                assert {(r1, o1, n1), (r2, o2, n2)} <= SHARED, (c1, name1, hex(o1), c2, name2, hex(o2))


# --- The query model (function 0x36, docs/formats/model2-tgp-ground.md section 7) ---

def _flat(*polys, grid_value=1):
    """A ground whose only listed cell is (64, 64) (x, z in 0..16), listing `polys` in order."""
    grid = [0] * tgp.GRID_CELLS
    grid[_cell(64, 64)] = grid_value
    return _ground(grid=tuple(grid), polys=tuple(polys), attributes=bytes([1] * len(polys)),
                   lists={grid_value: tuple(range(len(polys)))})


def test_cell_wraps():
    assert tgp.cell(1024.0, 0.0) == tgp.cell(-1024.0, 0.0)
    assert tgp.cell(-1024.0, -1024.0) == 0
    assert tgp.cell(1023.4, 1023.4) == 16383
    assert tgp.cell(0.0, 0.0) == 64 * 128 + 64
    assert tgp.cell(16.0, 0.0) == 64 * 128 + 65                      # x is the column, z the row (0x11bec)
    assert tgp.cell(0.0, 16.0) == 65 * 128 + 64
    assert tgp.cell(7.5, 0.0) == tgp.cell(8.0, 0.0) == 64 * 128 + 64     # round to even: 7.5 -> 8, 8.5 -> 8
    assert tgp.cell(8.5, 0.0) == 64 * 128 + 64
    assert tgp.cell(1e6, -1e6) == tgp.cell(1e6 % 2048, -1e6 % 2048)       # far outside the grid: the wrap, no error


def test_cell_refuses_non_finite():
    for bad in (math.nan, math.inf, -math.inf, "1", None):
        with pytest.raises(tgp.TgpError):
            tgp.cell(bad, 0.0)
        with pytest.raises(tgp.TgpError):
            tgp.cell(0.0, bad)


def test_query_synthetic():
    g = _flat(_sq(0, 0, 8.0, y=1.0))
    h, poly = tgp.query(g, 4.0, 1.0, 4.0)
    assert (h, poly) == (1.0, 0)
    assert tgp.query(g, 4.0, 1.0, 9.0) is None                       # outside the square, same cell
    assert tgp.query(g, 300.0, 1.0, 300.0) is None                   # a cell with no list
    assert tgp.query(g, -3.0, 1.0, 4.0) is None
    for x, z in ((0.0, 0.0), (8.0, 8.0), (0.0, 8.0), (8.0, 0.0), (0.0, 4.0), (4.0, 8.0)):
        assert tgp.query(g, x, 1.0, z) == (1.0, 0), (x, z)           # edges and corners are hits (d >= 0)
    assert tgp.query(g, 8.001, 1.0, 4.0) is None


def test_query_wraps_outside_the_grid():
    g = _flat(_sq(0, 0, 8.0, y=1.0))
    assert tgp.query(g, 4.0 + 2048.0, 1.0, 4.0) is None             # same cell as (4, 4), polygon is elsewhere in XZ
    assert tgp.query(g, 4.0 + 2048.0, 1.0, 4.0 - 4096.0) is None    # no raise, no clamp
    assert tgp.cell(1e9, -1e9) == tgp.cell(1e9 % 2048, -1e9 % 2048)    # f32 rounding of 1e9 first, as the i960 sends


def test_query_refuses_bad_input():
    g = _flat(_sq(0, 0))
    for args in ((math.nan, 0.0, 0.0), (0.0, math.inf, 0.0), (0.0, 0.0, -math.inf), ("a", 0.0, 0.0)):
        with pytest.raises(tgp.TgpError):
            tgp.query(g, *args)


def test_query_sloped_plane_both_triangles_and_both_branches():
    """Coplanar diamonds (clockwise like the originals): y = 3 + 0.5 x - 0.25 z. The first has Z0 == Z2 (routine
    0x684's other branch), the second X0 == X2 (its other division), the third neither."""
    def plane(x, z):
        return 3.0 + 0.5 * x - 0.25 * z
    shapes = {"z0==z2": ((-4, 0), (0, 4), (4, 0), (0, -4)),
              "x0==x2": ((0, -4), (-4, 0), (0, 4), (4, 0)),
              "general": ((-3, -1), (-2, 5), (5, 3), (4, -4))}
    for name, pts in shapes.items():
        pts = tuple((x + 8, z + 8) for x, z in pts)                 # inside cell (64, 64): 0 <= x, z < 16
        poly = tgp.GroundPoly(tuple((x, plane(x, z), z) for x, z in pts), centre=(0, 0, 0))
        g = _flat(poly)
        hits = 0
        for ix in range(-40, 41):
            for iz in range(-40, 41):
                x, z = 8 + ix / 8.0, 8 + iz / 8.0
                got = tgp.query(g, x + 0.0, plane(x, z), z + 0.0)
                inside = all((pts[(k + 1) % 4][0] - pts[k][0]) * (z - pts[k][1])
                             - (pts[(k + 1) % 4][1] - pts[k][1]) * (x - pts[k][0]) <= 0 for k in range(4))
                if inside:
                    assert got is not None and got[1] == 0, (name, x, z)
                    assert got[0] == pytest.approx(plane(x, z), abs=2e-5), (name, x, z)
                    hits += 1
        assert hits > 500, name


def test_query_picks_nearest_height_and_a_later_tie_wins():
    low, high, same = _sq(0, 0, 8.0, y=1.0), _sq(0, 0, 8.0, y=5.0), _sq(0, 0, 8.0, y=1.0)
    g = _flat(low, high)
    assert tgp.query(g, 4.0, 1.2, 4.0) == (1.0, 0)
    assert tgp.query(g, 4.0, 4.0, 4.0) == (5.0, 1)
    assert tgp.query(g, 4.0, 3.0, 4.0) == (5.0, 1)                  # |1-3| == |5-3|: the later polygon replaces
    assert tgp.query(_flat(low, same), 4.0, 9.0, 4.0) == (1.0, 1)
    below = _sq(0, 0, 8.0, y=-5.0)
    assert tgp.query(_flat(low, below), 4.0, 0.0, 4.0) == (1.0, 0)
    assert tgp.query(_flat(below, low), 4.0, 0.0, 4.0)[1] == 1       # the choice compares |height|: 1 vs 5 against y 0


def test_query_other_winding_never_hit_and_zero_keeps_its_sign():
    p = _sq(0, 0, 8.0)
    flipped = tgp.GroundPoly(tuple(reversed(p.corners)), centre=p.centre)
    assert tgp.query(_flat(flipped), 4.0, 1.0, 4.0) is None
    h, _ = tgp.query(_flat(_sq(0, 0, 8.0, y=0.0)), 4.0, 0.0, 4.0)
    assert h == 0.0 and math.copysign(1.0, h) == -1.0                # the logged "-0.0000"


def test_surface():
    g = _ground()
    assert [tgp.surface(g, i) for i in range(5)] == [1, 1, 2, 5, 8]
    assert tgp.surface(_ground(attributes=bytes([0xcd, 1, 2, 5, 8])), 0) == 13
    for bad in (5, -1, 99, 1.0, None, True):
        with pytest.raises(tgp.TgpError):
            tgp.surface(g, bad)


# --- The query model against the game (MAME, 45 s per course) ---

RACE_FROM = 1330            # the course is loaded by frame 1328 (before it the TGP answers from Beginner's slot 0)
RACING_FROM = 2400          # the race starts (tests/test_mame_runner.py, test_original_beginner_ground)
LOG_STEP = 5e-5             # the probe logs x, y, z with %.4f: the true value is within this of the logged one
MIN_AGREE = 0.99            # per course (the brief)


@pytest.fixture(scope="module")
def drives(tmp_path_factory):
    """Per course id, (Ground, 1/x table, logged ground calls, positions) of a 45 s autoplay run (DAYTONA_COURSE is
    the panel, `course.PANEL_COURSES`). The probe logs floats with %.4f."""
    from dtools import config
    from dtools.mame import runner
    d = config.rom_dir()
    if d is None or not (d / "daytona.zip").is_file():
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    romset_zip = d / "daytona.zip"
    rs = _real(romset_zip)
    out = {}
    for course in range(3):
        panel = C.PANEL_COURSES[course]
        d = tmp_path_factory.mktemp(f"drive{course}")
        runner.run_mame(runner.LUA_DIR / "autoplay.lua", 45, d / "run", romset_zip.parent,
                        env={"DAYTONA_COURSE": str(panel), "DAYTONA_GROUND_LOG": str(d / "ground.txt")})
        samples, positions = runner.parse_ground(d / "ground.txt")
        runner.require_calls(samples)
        out[course] = (tgp.decode_ground(rs, course), tgp.inv_table(rs),
                       [s for s in samples if s.frame >= RACE_FROM], positions)
    return out


def _agrees(got, s) -> bool:
    want = None if s.poly is None else (s.height, s.poly)
    return (got is None) == (want is None) and (got is None or (got[1] == want[1] and abs(got[0] - want[0]) <= 0.01))


def _rounding_reproduces(g, inv, s) -> bool:
    """True when some (x, z) within the log's rounding (+-LOG_STEP, a 15 x 15 grid), y as logged, gives the logged
    answer: the call's outcome rests on bits the %.4f log discarded (a point within about 0.003 of an edge two
    polygons share, or two stacked polygons whose heights differ by 1-2 f32 ulps). y is not varied: its rounding
    moves each candidate's | |height| - y | by up to LOG_STEP, the same way for candidates on one side of y but in
    opposite directions for two on either side of it, so it could flip a choice between two such hits whose
    distances differ by under 2 * LOG_STEP; x and z alone reproduce every measured miss."""
    for k in range(15):
        for j in range(15):
            dx, dz = (k - 7) * LOG_STEP / 7, (j - 7) * LOG_STEP / 7
            if _agrees(tgp.query(g, s.x + dx, s.y, s.z + dz, inv), s):
                return True
    return False


@pytest.mark.mame
def test_query_reproduces_probe(drives):
    """Every logged call of each course from frame 1330: `cell` finds the grid value the i960 sent, and `query`
    (with the TGP's own 1/x table) gives the logged polygon and a height within 0.01 on at least 99 % of the calls
    of each course. Measured 2026-11-07: Beginner 100 % (9,487), Advanced 100 % (11,535), Expert 99.03 % (11,232;
    109 differ). Each of those 109 is reproduced by some x, z within the log's +-5e-5 rounding, so each is
    consistent with the model under the log's rounding; that is asserted for every miss (`_rounding_reproduces`)."""
    for course, (g, inv, samples, _) in drives.items():
        agree, misses = 0, []
        for s in samples:
            if s.id:
                assert g.grid[tgp.cell(s.x, s.z)] == s.id, (course, s)
            if _agrees(tgp.query(g, s.x, s.y, s.z, inv), s):
                agree += 1
            else:
                misses.append(s)
        assert len(samples) > 9000
        assert agree >= MIN_AGREE * len(samples), (course, len(misses), len(samples), misses[:5])
        unexplained = [s for s in misses if not _rounding_reproduces(g, inv, s)]
        assert not unexplained, (course, len(unexplained), unexplained[:5])


@pytest.mark.mame
def test_road_surface(drives):
    """Every logged player call on Beginner during racing is on the road (kind 1) or one of the kinds the road
    border and pit lane use (2, 5, 8, 10, 13)."""
    from dtools.mame import runner
    g, _, samples, positions = drives[0]
    player = [s for s in runner.player_samples(samples, positions) if s.frame >= RACING_FROM]
    assert len(player) > 100
    assert {tgp.surface(g, s.poly) for s in player} <= {1, 2, 5, 8, 10, 13}
    assert sum(tgp.surface(g, s.poly) == 1 for s in player) > 0.9 * len(player)


def test_query_triangle_edge_answers_the_no_height_constant(romset_zip):
    """An original triangle (corners 2 and 3 equal) has ny == 0 for its 0-2-3 plane; a point on its 0-2 edge on
    the far side of the 0-1 line from the plane test's view takes that plane and the TGP answers its constant at
    word 0x206, -0.1 (init 0x38-0x3c), not the polygon's ~82 height. Advanced polygon 2956."""
    rs = _real(romset_zip)
    g = tgp.decode_ground(rs, 2)
    inv = tgp.inv_table(rs)
    assert g.polys[2956].corners[2] == g.polys[2956].corners[3]
    x, z = 308.5551452636719, -491.7867431640625
    for table in (inv, None):
        h, poly = tgp.query(g, x, -0.1, z, table)
        assert poly == 2956 and h == float(tgp._F(-0.1))
        assert tgp.query(g, 308.5525, 82.0, -491.783, table) == (pytest.approx(82.25, abs=1e-4), 2956)


def test_query_triangle_with_corners_2_and_3_equal_pins_the_no_height_constant():
    """Synthetic: the originals' triangle shape (corner 3 = corner 2). Points along the 0-2 edge where f32 rounding
    gives alpha or beta a sign select plane 0-2-3, which has no XZ extent (ny == 0): the TGP answers the constant
    at its word 0x206, -0.1 (init 0x38-0x3c), where the polygon's own height is 5."""
    tri = tgp.GroundPoly(((0.1, 5.0, 0.3), (0.7, 5.0, 8.1), (8.3, 5.0, 7.9), (8.3, 5.0, 7.9)), centre=(3.0, 5.0, 5.0))
    g = _flat(tri)
    answers = {tgp.query(g, 0.1 + i / 64 * 8.2 + e, -0.1, 0.3 + i / 64 * 7.6 + e)
               for i in range(1, 64) for e in (0.0, 1e-5, -1e-5, 1e-4)}
    assert (float(tgp._F(-0.1)), 0) in answers and (5.0, 0) in answers


def test_inv_table_refuses_a_set_without_the_tables_roms():
    with pytest.raises(tgp.TgpError):
        tgp.inv_table(_rs())
    bad = RomSet(program=b"", data=b"", copro_data=b"", polygons=b"", textures=b"",
                 extra={n: b"\0" * 8 for n in tgp.INV_TABLE_FILES})
    with pytest.raises(tgp.TgpError):
        tgp.inv_table(bad)


def test_inv_table_real(romset_zip):
    t = tgp.inv_table(_real(romset_zip))
    assert len(t) == 0x10000 and t[tgp.INV_TABLE_BASE] == 0xBF7FF800 and t[tgp.INV_TABLE_BASE + 1] == 0x3FFFFC00
    r = tgp._recip_fn(t)
    for v in (1.0, 3.0, -7.5, 0.1, 1234.5678, -0.003):
        assert float(r(tgp._F(v))) == pytest.approx(1 / v, rel=1e-5)


# --- Relocation (Task 5): the ground copied into free space, every pointer rewritten ---

SYN_FREE = 0x300000                     # synthetic data ROM: a free (0xff) run for the grid and attribute block


def _syn() -> RomSet:
    """The synthetic set of `_ground()` with 0x9000 free data bytes at SYN_FREE (copro_data is free past it)."""
    return _rom(_ground(), [("data", SYN_FREE, b"\xff" * 0x9000)])


def _addresses_aside(a: "tgp.Ground", b: "tgp.Ground") -> None:
    """Equal in every field but the four addresses: the polygons word for word (raw), the list block as read."""
    for f in ("course", "grid", "attributes", "lists", "base", "raw", "polys"):
        assert getattr(a, f) == getattr(b, f), f


def test_free_spans():
    assert tgp.COPRO_FREE == (0x128940, 882368)          # Task 3: the plan's 0x12893e is inside slot 3's last polygon
    assert tgp.DATA_FREE == (0x65D120, 667360)
    assert tgp.COPRO_FREE[0] + tgp.COPRO_FREE[1] == COPRO_SIZE // 2 and tgp.COPRO_FREE[0] % 4 == 0


@pytest.fixture
def syn_owned(monkeypatch):
    """The synthetic sets (`_rom`) hold a ground and the course records' ground pointers only: golden.owned_spans
    cannot decode their courses, so tgp.write_free (which refuses every owned byte) would refuse every write. Here
    their owned space is RESERVED's and each ground that decodes (tgp.ground_spans), all those sets own."""
    from dtools.arcade import golden as G
    from dtools.arcade import romset

    def owned(rs):
        out = [(r, o, n, w) for r, spans in romset.RESERVED.items() for o, n, w in spans]
        for k in range(C.COURSE_COUNT):
            try:
                out += tgp.ground_spans(rs, k)
            except DecodeError:
                pass
        return sorted(out)
    monkeypatch.setattr(G, "owned_spans", owned)
    yield
    G._OWNED[:] = [None, {}, []]                # no synthetic answer outlives the patch


@pytest.mark.usefixtures("syn_owned")
def test_write_free_guards():
    rs = _syn()
    with pytest.raises(tgp.TgpError, match="not free"):                         # the synthetic list block
        tgp.write_free(rs, "copro_data", 4 * 0x30, b"\x00" * 4)
    with pytest.raises(tgp.TgpError, match="not free"):                         # free, then one byte that is not
        tgp.write_free(_rom(_ground(), [("data", SYN_FREE, b"\xff" * 7)]), "data", SYN_FREE, b"\x00" * 8)
    with pytest.raises(tgp.TgpError, match="unaligned"):
        tgp.write_free(rs, "copro_data", 4 * 0x4000 + 2, b"\x00" * 4)
    with pytest.raises(tgp.TgpError, match="past"):                             # past copro_data's first copy
        tgp.write_free(rs, "copro_data", COPRO_SIZE // 2 - 4, b"\x00" * 8)
    with pytest.raises(tgp.TgpError, match="past"):
        tgp.write_free(rs, "data", DATA_SIZE - 4, b"\x00" * 8)
    with pytest.raises(tgp.TgpError, match="past"):
        tgp.write_free(rs, "data", -4, b"\x00" * 4)
    for bad in (("polygons", 0, b"\x00"), ("copro_data", "0", b"\x00"), ("copro_data", True, b"\x00"),
                ("copro_data", 4 * 0x4000, "abcd"), ("copro_data", 4 * 0x4000, None)):
        with pytest.raises(tgp.TgpError):
            tgp.write_free(rs, *bad)
    with pytest.raises(tgp.TgpError):                                           # rs is not a RomSet
        tgp.write_free(None, "data", SYN_FREE, b"\x00" * 4)
    free = RomSet(program=b"", data=b"\xff" * DATA_SIZE, copro_data=b"", polygons=b"", textures=b"", extra={})
    with pytest.raises(tgp.TgpError, match="copies|0x800000"):                 # across 0x800000 / two 1 MB copies
        tgp.write_free(free, "data", 0x7FFFFC, b"\x00" * 8)
    with pytest.raises(tgp.TgpError, match="copies|0x800000"):
        tgp.write_free(free, "data", 0x8FFFFC, b"\x00" * 8)
    out = tgp.write_free(rs, "copro_data", 4 * 0x4000, b"\x01\x02\x03\x04")
    half = COPRO_SIZE // 2
    assert out.copro_data[4 * 0x4000:4 * 0x4000 + 4] == out.copro_data[half + 4 * 0x4000:half + 4 * 0x4000 + 4] \
        == b"\x01\x02\x03\x04"
    assert rs.copro_data[4 * 0x4000:4 * 0x4000 + 4] == b"\xff" * 4                # a copy: rs is unchanged
    assert tgp.write_free(rs, "data", SYN_FREE, b"\x05" * 4).data[SYN_FREE:SYN_FREE + 5] == b"\x05" * 4 + b"\xff"


def _ff_words(rs: RomSet, spans) -> list[tuple[int, int]]:
    """(offset, length) of the 4-aligned all-0xff words inside the (region, offset, length, name) spans."""
    out = []
    for region, o, n, _ in spans:
        img = getattr(rs, region)
        out += [(a, 4) for a in range((o + 3) & ~3, o + n - 3, 4) if img[a:a + 4] == b"\xff" * 4]
    return out


def test_write_free_refuses_owned():
    """Phase 9 Task 1 (the Phase 8 residual "write_free guards less than the allocators"): every byte a table or
    block owns is busy to tgp.write_free, whatever its value (golden.owned_spans, cached per set: golden.owned). On
    the set with Desert City at id 3: a write over id 3's race tables, over its ground's 0xffffffff words (copro_data's
    custom area, free by value) and over its objects (polygons: no write_free region; their object lists in
    DATA_FREE) is refused; the owner is named where the bytes are 0xff. A write into a free run (build.ID_DATA_FREE's,
    copro_data's custom area) still works, and a second write into the result (the converter's and relocate_ground's
    chains, written before any pointer names them) asks the cache, not a decode of the half-written set."""
    from tests.test_mirror import _desert3
    from dtools.arcade import golden as G
    from dtools.arcade import romset
    from dtools.convert import build
    got = _desert3()
    if got is None:
        pytest.skip("set DAYTONA_ROMDIR and DAYTONA_DC")
    _, out = got
    cov = G.coverage(out, 3)
    race = [s for s in cov if s[3] in ("step table", "road line", "line table", "checkpoint table", "time table",
                                       "segment flags", "cell grid", "ground grid")]
    assert {s[3] for s in race} >= {"step table", "road line", "line table", "time table"}
    lists = [s for s in cov if s[3] == "object list" and tgp.DATA_FREE[0] <= s[1] < sum(tgp.DATA_FREE)]
    assert len(lists) >= 50                                                     # its own, in the list home
    for region, o, n, name in race + lists:
        assert G.owner(out, region, o, n) == f"{name} of course 3", (name, hex(o))
    # the race tables' 4-aligned 0xff bytes: free by value, owned (a one-byte write passed the old guard)
    ff_bytes = [(region, a, name) for region, o, n, name in race
                for a in range((o + 3) & ~3, o + n, 4) if getattr(out, region)[a] == 0xFF]
    assert {name for _, _, name in ff_bytes} >= {"step table", "road line", "line table", "ground grid"}
    for region, a, name in ff_bytes:
        with pytest.raises(tgp.TgpError, match=f"{name} of course 3"):
            tgp.write_free(out, region, a, b"\0")
    pool = [s for s in cov if s[3] == "ground polygons"]
    ff = _ff_words(out, pool)
    assert pool and pool[0][1] >= romset.COPRO_SIZE and ff                   # free by value, in the custom area
    for o, n in ff[:8] + ff[-8:]:
        with pytest.raises(tgp.TgpError, match="ground polygons of course 3"):
            tgp.write_free(out, "copro_data", o, b"\0" * n)
    # the owned 0xff bytes in the custom data area: the texel block's and grow_ids' home (the RESERVED message)
    block = G.decode_course(out, 3).sheet_sources[0]
    tex = _ff_words(out, [("data", block, 0x100000, "texels")])
    with pytest.raises(tgp.TgpError, match="COURSE_SHEETS"):
        tgp.write_free(out, "data", tex[0][0], b"\0" * 4)
    # free runs still take a write, and a chain of writes from one set reads the cache
    at = build._first_fit(out, "data", build.ID_DATA_FREE, 16, "probe")
    cat = build._first_fit(out, "copro_data", romset.CUSTOM_FREE["copro_data"][0], 16, "probe")
    rows = G.owned(out)
    w = tgp.write_free(out, "data", at, b"\1" * 8)
    assert w.data[at:at + 8] == b"\1" * 8 and w.data[at + 8:at + 16] == b"\xff" * 8
    w2 = tgp.write_free(w, "copro_data", cat, b"\2" * 8)
    assert w2.copro_data[cat:cat + 8] == b"\2" * 8 and w2.data == w.data
    assert G.owned(w2) is rows                       # carried along the chain: no second decode
    with pytest.raises(tgp.TgpError, match="ground polygons of course 3"):
        tgp.write_free(w2, "copro_data", ff[0][0], b"\0" * 4)
    assert G.owner(w2, "copro_data", cat, 8) is None and G.owner(w2, "copro_data", ff[0][0], 4).endswith("course 3")
    for bad in ((None, "data", 0, 4), (w2, 1, 0, 4), (w2, "data", True, 4), (w2, "data", 0, 4.0)):
        with pytest.raises(DecodeError):
            G.owner(*bad)
    with pytest.raises(DecodeError):
        G.owned("set")
    with pytest.raises(DecodeError):
        G.carry_owned(w2, None)


@pytest.mark.usefixtures("syn_owned")
def test_relocate_round_trip_synthetic():
    rs = _syn()
    at = 4 * 0x40000
    out = tgp.relocate_ground(rs, 0, at, SYN_FREE)
    d = tgp.decode_ground(out, 0)
    _addresses_aside(d, tgp.decode_ground(rs, 0))
    assert (d.list_base, d.pool_base, d.grid_at, d.attributes_at) == (0x40000, 0x40008, SYN_FREE,
                                                                       SYN_FREE + 2 * tgp.GRID_CELLS)
    assert tgp.verify_lossless_ground(out, 0) == []
    e = C.course_table(out)[0]
    assert struct.unpack_from("<I", out.data, e.block_ref_a)[0] == DATA_BASE + SYN_FREE
    assert struct.unpack_from("<I", out.data, e.block_ref_b)[0] == DATA_BASE + SYN_FREE + 2 * tgp.GRID_CELLS
    gone = tgp.relocate_ground(rs, 0, at, SYN_FREE, erase_original=True)
    _addresses_aside(tgp.decode_ground(gone, 0), d)
    assert gone.copro_data[4 * 0x30:4 * 0x38] == b"\xff" * 32                       # the old list block
    assert gone.copro_data[4 * 0x1000:4 * (0x1000 + 5 * tgp.POLY_WORDS)] == b"\xff" * (20 * tgp.POLY_WORDS)
    assert gone.data[GRID_AT:GRID_AT + 2 * tgp.GRID_CELLS] == b"\xff" * (2 * tgp.GRID_CELLS)
    assert gone.data[ATTR_AT:ATTR_AT + 8] == b"\xff" * 8


def test_placement_helpers():
    """The placement relocate_ground and the converter (dtools/convert/build.py) share: the list block's length,
    the ground placed (list block then pool right after it; grid then attribute block right after it), the bank
    window rule, and the record's words 6-7 repointed where they hold the grid pointer."""
    g = _ground()
    block = tgp.list_block_words(g.lists)
    assert block == 8                                   # word 0, list 1 (count + 2), list 4 (count + 3)
    p = tgp.placed_ground(g, 0x40000, SYN_FREE)
    assert (p.list_base, p.pool_base, p.grid_at, p.attributes_at) == (0x40000, 0x40000 + block, SYN_FREE,
                                                                       SYN_FREE + 2 * tgp.GRID_CELLS)
    _addresses_aside(p, g)
    n = len(g.polys)
    assert tgp.window_safe(g.lists, n, 0x40000, block)
    assert not tgp.window_safe(g.lists, n, 0x50000 - 6, block)          # list 4 at 0x4fffe..0x50001
    assert not tgp.window_safe(g.lists, n, 0x50000 - 16, block)         # polygon 0 at 0x4fff8 crosses
    assert tgp.window_safe(g.lists, n, 0x50000 - 8 - 2 * tgp.POLY_WORDS, block)     # pool polygon-aligned
    assert tgp.repoint_record_raw((5, 7), 5, 9) == (9, 7)
    assert tgp.repoint_record_raw((5, 5), 5, 9) == (9, 9) and tgp.repoint_record_raw((1, 2), 5, 9) == (1, 2)


@pytest.mark.usefixtures("syn_owned")
def test_bank_window_crossing_raises():
    """The TGP reads (bank & 0xff0000) | offset16 and walks on with a 16-bit index: a list or a polygon past a
    64K-word boundary would wrap. The program re-banks for each list ($0x69 + grid value) and each polygon ($0x6a +
    16 * index; Task 3), so a block may span a boundary as long as no list or polygon does."""
    rs = _syn()
    with pytest.raises(tgp.TgpError, match="bank window"):          # list 4 at words 0x4fffe..0x50001
        tgp.relocate_ground(rs, 0, 4 * (0x50000 - 6), SYN_FREE)
    with pytest.raises(tgp.TgpError, match="bank window"):          # block 0x4fff0..0x4fff7, polygon 0 at 0x4fff8
        tgp.relocate_ground(rs, 0, 4 * (0x50000 - 16), SYN_FREE)
    out = tgp.relocate_ground(rs, 0, 4 * (0x50000 - 4), SYN_FREE)   # block 0x4fffc..0x50003: list 4 at 0x50000
    d = tgp.decode_ground(out, 0)
    assert d.list_base // tgp.WINDOW != (d.list_base + len(d.raw) - 1) // tgp.WINDOW
    _addresses_aside(d, tgp.decode_ground(rs, 0))
    out = tgp.relocate_ground(rs, 0, 4 * (0x50000 - 8 - 2 * tgp.POLY_WORDS), SYN_FREE)   # pool across, polygon-aligned
    d = tgp.decode_ground(out, 0)
    assert d.pool_base // tgp.WINDOW != (d.pool_base + 5 * tgp.POLY_WORDS - 1) // tgp.WINDOW
    _addresses_aside(d, tgp.decode_ground(rs, 0))


@pytest.mark.usefixtures("syn_owned")
def test_relocate_refuses():
    rs = _syn()
    for course in (-1, 3, None):
        with pytest.raises(tgp.TgpError, match="course"):
            tgp.relocate_ground(rs, course, 4 * 0x40000, SYN_FREE)
    with pytest.raises(tgp.TgpError, match="unaligned"):
        tgp.relocate_ground(rs, 0, 4 * 0x40000 + 2, SYN_FREE)
    with pytest.raises(tgp.TgpError, match="unaligned"):
        tgp.relocate_ground(rs, 0, 4 * 0x40000, SYN_FREE + 1)
    with pytest.raises(tgp.TgpError, match="past"):                 # the pool would run past the first copy
        tgp.relocate_ground(rs, 0, COPRO_SIZE // 2 - 4 * 16, SYN_FREE)
    with pytest.raises(tgp.TgpError, match="past"):
        tgp.relocate_ground(rs, 0, -4, SYN_FREE)
    with pytest.raises(tgp.TgpError, match="past"):
        tgp.relocate_ground(rs, 0, 4 * 0x40000, DATA_SIZE - 0x100)
    with pytest.raises(tgp.TgpError, match="not free"):             # over the original pool
        tgp.relocate_ground(rs, 0, 4 * 0x1000, SYN_FREE)
    with pytest.raises(tgp.TgpError, match="not free"):             # the synthetic data ROM is zeros elsewhere
        tgp.relocate_ground(rs, 0, 4 * 0x40000)
    for bad in ("0", 1.5, True):
        with pytest.raises(tgp.TgpError):
            tgp.relocate_ground(rs, 0, bad, SYN_FREE)
        with pytest.raises(tgp.TgpError):
            tgp.relocate_ground(rs, 0, 4 * 0x40000, bad)
    with pytest.raises(tgp.TgpError, match="erase_original"):
        tgp.relocate_ground(rs, 0, 4 * 0x40000, SYN_FREE, erase_original="yes")


@pytest.mark.usefixtures("syn_owned")
def test_relocate_checks_its_output_decodes():
    """encode_ground cannot see the decoder's pool-end rule (the records after the last pool must be free): a target
    whose pool is followed by a used word decodes to a TgpError, raised by relocate_ground itself. (Slot 1's pool
    address bounds the original pool, so the used word is past its bound.)"""
    used = ("copro_data", 4 * (0x40008 + 5 * tgp.POLY_WORDS), b"\x00" * 4)   # the word right after the new pool
    bound = ("copro_data", 4 * (tgp.SLOT_POOL + 1), struct.pack("<I", 0x1000 + 5 * tgp.POLY_WORDS))
    rs = _rom(_ground(), [("data", SYN_FREE, b"\xff" * 0x9000), used, bound])
    with pytest.raises(tgp.TgpError, match="does not decode.*not free"):
        tgp.relocate_ground(rs, 0, 4 * 0x40000, SYN_FREE)


def _changed_outside(a: bytes, b: bytes, spans) -> list[int]:
    """The first few byte offsets where `a` and `b` differ outside the (start, end) `spans`."""
    diff = np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8)
    for lo, hi in spans:
        diff[lo:hi] = False
    return np.flatnonzero(diff)[:8].tolist()


def _check_relocated(rs: RomSet, course: int, moved: RomSet, gone: RomSet, lb: int) -> None:
    """`moved` / `gone` (erase_original) are `rs` with `course`'s ground relocated by relocate_ground's defaults, the
    list block at word `lb` and the pool right after it (polygon-aligned on the real set: no polygon crosses a bank
    window), the grid and attribute block at DATA_FREE's start. Each decodes to the original in every field but the
    addresses; only the new blocks and the pointers change (the course record's block_ref_a / block_ref_b words and
    words 6-7, which hold the grid pointer too; the slot-table words of slot `course`, and for Beginner of slots
    4-15, which repeat slot 0's). With erase_original the old blocks are 0xff; the other courses decode unchanged
    and losslessly in both builds."""
    g = tgp.decode_ground(rs, course)
    pb = lb + len(g.raw)
    ga, aa = tgp.DATA_FREE[0], tgp.DATA_FREE[0] + 2 * tgp.GRID_CELLS
    half = COPRO_SIZE // 2
    pool_bytes = 4 * tgp.POLY_WORDS * len(g.polys)
    new_copro = [(4 * lb, 4 * pb + pool_bytes)]
    slots = [course, *range(4, 16)] if course == 0 else [course]
    slot_words = [(4 * (w + s), 4 * (w + s) + 4) for w in (tgp.SLOT_LIST, tgp.SLOT_POOL) for s in slots]
    e = C.course_table(rs)[course]
    rec = C.COURSE_TABLE - DATA_BASE + C.COURSE_ENTRY_SIZE * course
    attr_block = len(g.attributes) + -len(g.attributes) % 4         # with its 0xcd padding (Expert 1, Advanced 2)
    new_data = [(ga, aa + attr_block), (e.block_ref_a, e.block_ref_a + 4), (e.block_ref_b, e.block_ref_b + 4),
                (rec + 0x18, rec + 0x20)]
    old_copro = [(4 * g.list_base, 4 * (g.list_base + len(g.raw))), (4 * g.pool_base, 4 * g.pool_base + pool_bytes)]
    old_data = [(g.grid_at, g.grid_at + 2 * tgp.GRID_CELLS), (g.attributes_at, g.attributes_at + attr_block)]
    table0 = struct.unpack_from(f"<{2 * tgp.SLOTS}I", rs.copro_data, 4 * tgp.SLOT_LIST)
    for out, erased in ((moved, False), (gone, True)):
        if out is None:
            continue
        d = tgp.decode_ground(out, course)
        _addresses_aside(d, g)
        assert (d.list_base, d.pool_base, d.grid_at, d.attributes_at) == (lb, pb, ga, aa)
        assert all(not tgp._crosses(pb + tgp.POLY_WORDS * i, tgp.POLY_WORDS) for i in range(len(d.polys)))
        assert tgp.verify_lossless_ground(out, course) == []
        for other in {0, 1, 2} - {course}:
            assert tgp.verify_lossless_ground(out, other) == []
            _same(tgp.decode_ground(out, other), tgp.decode_ground(rs, other))
        table = struct.unpack_from(f"<{2 * tgp.SLOTS}I", out.copro_data, 4 * tgp.SLOT_LIST)
        assert [table[s] for s in slots] == [lb] * len(slots)
        assert [table[16 + s] for s in slots] == [pb] * len(slots)
        assert [table[s] for s in range(32) if s % 16 not in slots] == [table0[s] for s in range(32)
                                                                        if s % 16 not in slots]
        assert C.course_table(out)[course].raw == (DATA_BASE + ga,) * 2
        assert out.copro_data[:half] == out.copro_data[half:]
        assert out.program == rs.program and out.polygons == rs.polygons and out.textures == rs.textures
        assert _changed_outside(rs.copro_data[:half], out.copro_data[:half],
                                new_copro + slot_words + (old_copro if erased else [])) == []
        assert _changed_outside(rs.data[:0x800000], out.data[:0x800000],
                                new_data + (old_data if erased else [])) == []
        assert out.data[0x800000:] == rs.data[0x800000:]
        for lo, hi in old_copro:
            assert (out.copro_data[lo:hi] == b"\xff" * (hi - lo)) == erased
        for lo, hi in old_data:
            assert (out.data[lo:hi] == b"\xff" * (hi - lo)) == erased


def test_relocate_round_trip_offline(romset_zip):
    """Beginner relocated to the free spans (list block and pool at copro_data byte 0x128940, grid and attribute
    block at data 0x65d120), with and without erase_original (`_check_relocated`)."""
    rs = _real(romset_zip)
    g = tgp.decode_ground(rs, 0)
    _check_relocated(rs, 0, tgp.relocate_ground(rs, 0), tgp.relocate_ground(rs, 0, erase_original=True),
                     tgp.COPRO_FREE[0] // 4)
    with pytest.raises(tgp.TgpError, match="bank window"):          # Beginner's pool across 0x50000, mid-polygon
        tgp.relocate_ground(rs, 0, 4 * (0x50000 - len(g.raw) - 8))


# relocate_ground's default list-block word for Expert and Advanced: the first word from COPRO_FREE's start (0x4a250)
# at which no list and no polygon crosses a bank window. From 0x4a250 both pools would cross 0x50000 mid-polygon
# (Expert's polygon 119, Advanced's 855); 6 / 14 words on, the pool starts on a 16-word boundary.
RELOC_LIST_BASE = {1: 0x4A256, 2: 0x4A25E}


@pytest.mark.parametrize("course", [1, 2])
def test_relocate_expert_advanced_offline(romset_zip, course):
    """Expert and Advanced move with erase_original (`_check_relocated`): the pool before the moved one's old pool
    (Beginner's before Expert's, Expert's before Advanced's) then reaches to the next slot-table address over the
    erased records, which the decoder accepts as free. Without erase_original the old pool stays inside that
    neighbour's bound, neither listed nor free: relocate_ground refuses, saying why."""
    rs = _real(romset_zip)
    lb = RELOC_LIST_BASE[course]
    gone = tgp.relocate_ground(rs, course, erase_original=True)
    _check_relocated(rs, course, None, gone, lb)
    neighbour = {1: "Beginner", 2: "Expert"}[course]
    with pytest.raises(tgp.TgpError, match=rf"breaks {neighbour}.*not free.*erase_original"):
        tgp.relocate_ground(rs, course)
    # the default is the first window-safe word: the start itself is refused, for the polygon across 0x50000
    with pytest.raises(tgp.TgpError, match="bank window"):
        tgp.relocate_ground(rs, course, tgp.COPRO_FREE[0], erase_original=True)


def test_relocate_all_three_offline(romset_zip):
    """All three grounds moved into the free spans one after another, originals erased (Advanced by the defaults,
    then Expert and Beginner after it at window-safe words): each course decodes to itself, losslessly, and
    golden.decode_course reads every one."""
    from dtools.arcade import golden
    rs = _real(romset_zip)
    expert_at = tgp.DATA_FREE[0] + 2 * tgp.GRID_CELLS + 2988                # after Advanced's grid and attributes
    out = tgp.relocate_ground(rs, 2, erase_original=True)
    out = tgp.relocate_ground(out, 1, 4 * 0x58536, expert_at, erase_original=True)  # after Advanced's pool, 0x58530
    out = tgp.relocate_ground(out, 0, 4 * 0x79FE0, expert_at + 2 * tgp.GRID_CELLS + 7240, erase_original=True)
    for c, lb in ((0, 0x79FE0), (1, 0x58536), (2, 0x4A25E)):
        d = tgp.decode_ground(out, c)
        assert d.list_base == lb
        _addresses_aside(d, tgp.decode_ground(rs, c))
        assert tgp.verify_lossless_ground(out, c) == []
        assert golden.decode_course(out, c).ground_data == d


RELOC_SECONDS = 45


@pytest.mark.mame
def test_relocated_beginner_drives_identically(tmp_path, romset_zip):
    """Three 45 s autoplay runs on Beginner (DAYTONA_COURSE 0), each on a fresh workdir: the original set; Beginner's
    ground relocated (relocate_ground defaults); relocated with the original erased. The ground probe's full `G`
    sequences and the render-rate logs of the two relocated runs equal the original's call for call and frame for
    frame. MAME names the changed files under WRONG CHECKSUMS: the run loaded the patched set."""
    from dtools.mame import build, runner
    rs = _real(romset_zip)
    builds = {"original": None, "relocated": tgp.relocate_ground(rs, 0),
              "erased": tgp.relocate_ground(rs, 0, erase_original=True)}
    logs = {}
    for name, built in builds.items():
        d = tmp_path / name
        out = build.run_set(built, d, RELOC_SECONDS, romset_zip, {"DAYTONA_COURSE": "0"}, logs=("ground", "rate"))
        samples, positions = runner.parse_ground(out["ground"])
        runner.require_calls(samples)
        logs[name] = (samples, runner.parse_rate(out["rate"]))
    samples, rate = logs["original"]
    assert len(samples) > 9000 and max(s.frame for s in samples) > RACING_FROM      # the race was reached
    assert len(rate) > 2500
    for name in ("relocated", "erased"):
        got, got_rate = logs[name]
        assert len(got) == len(samples), name
        assert got == samples, (name, next(i for i, (a, b) in enumerate(zip(got, samples)) if a != b))
        assert got_rate == rate, name


def test_query_speed(romset_zip):
    """20,000 queries on Beginner's decoded ground with the TGP's own divisions take under a second: the grid is
    not converted to an array on every call (residual of the Phase 3 final review)."""
    import time
    rs = _real(romset_zip)
    g = tgp.decode_ground(rs, 0)
    inv = tgp.inv_table(rs)
    t = time.perf_counter()
    for k in range(20000):
        tgp.query(g, 100.0 + 0.01 * k, 0.0, -300.0 + 0.007 * k, inv)
    assert time.perf_counter() - t < 1.0


def test_ground_tables_is_public():
    g = _ground()
    assert tgp.ground_tables(g) and {(r, o): b for r, o, b, _ in tgp.ground_tables(g)} == tgp.encode_ground(g)
