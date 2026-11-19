"""Offline tests of M9 (reverse courses; spec docs/superpowers/specs/2026-11-16-reverse-courses-design.md). Stage 2,
step 2 (docs/superpowers/plans/2026-11-17-reverse-stage2-originals.md): the arcade's three in reverse at ids 8-10,
the arcade's own scenery shared, the Dreamcast's R race tables. Need DAYTONA_ROMDIR and DAYTONA_DC."""

import pytest

from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade.romset import RomSet
from dtools.convert import ConvertError, build, ground, tables
from tests.test_convert_tables import _rdata, disc  # noqa: F401  (disc: the module-scoped disc fixture)

SHARED = ("cells", "object_lists", "objects", "headers", "points", "actor_table", "actor_slots", "actor_lists",
          "billboards", "billboard_ref", "switch_lists", "unmodelled", "grid", "sheet_sources")
"""The scenery fields id n + 8 shares with id n (by pointer: mirror._alias): equal on decode."""

_SETS: dict = {}


def _sets(romset_zip, disc):  # noqa: F811
    """(the grown, guarded, widened set; it with the arcade's three in reverse at ids 8-10)."""
    key = str(romset_zip)
    if key not in _SETS:
        base = C.widen_ids(C.guard_new_ids(C.grow_ids(RomSet.from_zip(romset_zip))))
        out = base
        for n in (0, 1, 2):
            out = build.converted_set(out, disc, n, slot=n + 8, variant="R", scenery="keep")
        _SETS[key] = (base, out)
    return _SETS[key]


def test_originals_in_reverse_share_the_scenery(romset_zip, disc):  # noqa: F811
    """Ids 8-10: the arcade's three in reverse decode with id n's scenery (SHARED equal; the object lists less the
    select panel's course-map list, which goes by panel, stage 3's) and the culling walk of id n
    (VISIT_IDS[n + 8] = n), and the R file's race tables: its road line (the lane ranges fixed), its steps, its
    ground (convert_ground's polygons) and the R checkpoints."""
    from dtools.dc.exe import EXE_NAME
    _, out = _sets(romset_zip, disc)
    exe = disc.read(EXE_NAME)
    for n in (0, 1, 2):
        o, c = G.decode_course(out, n), G.decode_course(out, n + 8)
        for f in SHARED:
            a, b = getattr(o, f), getattr(c, f)
            if f == "object_lists":         # less the select panel's course-map list (SELECT_MAP_OBJECTS: by panel)
                a = {k: v for k, v in a.items() if k != o.map_object.objects}
                b = {k: v for k, v in b.items() if k != c.map_object.objects}
            assert build._same(a, b), (n, f)
        assert out.read_i960(C.VISIT_IDS + n + 8, 1)[0] == n
        d = tables.with_lane_ranges_fixed(_rdata(disc, n))
        assert tuple(c.road) == tables.road_line(d), n
        assert tuple(c.steps) == tables.steps(d)[1], n
        assert [p.corners for p in c.ground_data.polys] == [p.corners for p in ground.convert_ground(d, n + 8).polys]
        rows = [r.segment for r in tables.checkpoint_rows(exe, n, "R")]
        assert [ck.segment for ck in c.checkpoints if not ck.lap_line] == rows, n          # the lap line a view


def test_originals_in_reverse_leave_ids_0_to_7(romset_zip, disc):  # noqa: F811
    """Ids 0-7 decode as on the set before (every field), so the alias and the R tables wrote nothing of theirs."""
    base, out = _sets(romset_zip, disc)
    for k in range(8):
        a, b = G.decode_course(base, k), G.decode_course(out, k)
        diff = [f for f in a.__dataclass_fields__ if not build._same(getattr(a, f), getattr(b, f))]
        assert diff == [], (k, diff)


def test_variant_refusals_for_the_originals(romset_zip, disc):  # noqa: F811
    """n 0-2 in reverse take the arcade's scenery only ("keep"): "model" and "empty" refused; the slot must be
    n + 8."""
    base, _ = _sets(romset_zip, disc)
    for scenery in ("model", "empty"):
        with pytest.raises(ConvertError, match=r"course 0 in reverse keeps the arcade's scenery"):
            build.converted_set(base, disc, 0, slot=8, variant="R", scenery=scenery)
    with pytest.raises(ConvertError, match=r"slot 9: course 0 in reverse fills its reverse id 8"):
        build.converted_set(base, disc, 0, slot=9, variant="R")


# ---- the segment-keyed tables (Task 3) ----

POINT_OFFSET = {0: 192, 1: 987, 2: 483}
"""R road point r is plain road point (POINT_OFFSET[n] - r) mod the point count, its lane j the plain lane 7 - j,
exactly (measured 17 Nov 2026)."""


def _plain_and_reverse(disc, n):  # noqa: F811
    from tests.test_convert_tables import _data
    return (tables.with_lane_ranges_fixed(_data(disc, n)), tables.with_lane_ranges_fixed(_rdata(disc, n)))


def test_point_offset(disc):  # noqa: F811
    """reverse.point_offset finds POINT_OFFSET on n 0-2 (every used lane of every R point the plain point's mirrored
    lane within 1e-3) and refuses a road it does not fit; plain_segment is the map of segments it gives."""
    import dataclasses
    from dtools.convert import reverse
    for n in (0, 1, 2):
        p, r = _plain_and_reverse(disc, n)
        assert reverse.point_offset(p, r) == POINT_OFFSET[n]
        L = len(p.road)
        assert reverse.plain_segment(POINT_OFFSET[n], L, 0) == (POINT_OFFSET[n] - 1) % L
    p, r = _plain_and_reverse(disc, 0)
    with pytest.raises(ConvertError, match=r"no plain road point matches reverse road point 0"):
        reverse.point_offset(p, p)                                  # the plain lanes are not their own mirror
    with pytest.raises(ConvertError, match=r"reverse road point 1: .* not plain road point 191"):
        reverse.point_offset(p, dataclasses.replace(r, road=r.road[:1] + r.road[2:] + r.road[1:2]))


def test_reversed_segment_flags(romset_zip, disc):  # noqa: F811
    """Id 9's segment flags are Expert's on the reverse road's segments: SEGMENT_AI_FREE (0x400) where Expert's
    segments 190-240 run, SEGMENT_TUNNEL (0x100) where 208-218 do; ids 8 and 10 all zero (Beginner's and
    Advanced's); id 1's unchanged."""
    from dtools.convert import reverse
    base, out = _sets(romset_zip, disc)
    flags = G.decode_course(out, 9).segment_flags[1]
    L = len(flags)
    m = {s: reverse.plain_segment(POINT_OFFSET[1], L, s) for s in range(L)}
    assert {s for s in range(L) if flags[s] & 0x400} == {s for s in range(L) if 190 <= m[s] <= 240}
    assert {s for s in range(L) if flags[s] & 0x100} == {s for s in range(L) if 208 <= m[s] <= 218}
    assert not any(G.decode_course(out, 8).segment_flags[1]) and not any(G.decode_course(out, 10).segment_flags[1])
    assert G.decode_course(out, 1).segment_flags == G.decode_course(base, 1).segment_flags


def _masks(rs, k):
    import struct
    from dtools.arcade import mirror
    at = struct.unpack("<I", rs.read_i960(C.id_entry(C.id_tables(rs), "PROGRESS_MASKS", k), 4))[0]
    return mirror.mask_records(rs, at)


def test_reversed_progress_masks(romset_zip, disc):  # noqa: F811
    """Id 10's progress masks are Advanced's on the reverse road: each record's segments [a, b] -> [M(b), M(a)]
    (M = plain_segment, its own inverse), its row masks kept, the records in ascending order; ids 8 and 9 none (as
    Beginner and Expert); id 2's unchanged."""
    import struct
    from dtools.convert import reverse
    base, out = _sets(romset_zip, disc)
    plain = _masks(out, 2)
    assert plain == _masks(base, 2) and len(plain) == 4
    L = len(G.decode_course(out, 10).road)
    M = lambda s: reverse.plain_segment(POINT_OFFSET[2], L, s)  # noqa: E731
    want = sorted((M(struct.unpack_from("<2H", r)[1]), M(struct.unpack_from("<2H", r)[0]), r[4:]) for r in plain)
    got = [(*struct.unpack_from("<2H", r), r[4:]) for r in _masks(out, 10)]
    assert got == want
    assert _masks(out, 8) == [] and _masks(out, 9) == []


def _tunnel(rs, k):
    """(K, starts, ends, range, mask, {lit zone: position}) of id k's tunnel light record."""
    import struct
    arrays = C.id_tables(rs)
    K = rs.read_i960(C.id_entry(arrays, "TUNNEL_BYTES", k), 1)[0]
    starts, ends, rng, mask_at, pos = struct.unpack("<5I", rs.read_i960(C.id_entry(arrays, "TUNNEL_LIGHTS", k), 20))
    mask = struct.unpack("<I", rs.read_i960(mask_at, 4))[0]
    return (K, struct.unpack(f"<{K + 1}H", rs.read_i960(starts, 2 * K + 2)),
            struct.unpack(f"<{K + 1}H", rs.read_i960(ends, 2 * K + 2)), struct.unpack("<2H", rs.read_i960(rng, 4)),
            mask, {z: rs.read_i960(pos + 12 * z, 12) for z in range(K + 1) if mask >> z & 1})


def test_reversed_tunnel_lights(romset_zip, disc):  # noqa: F811
    """Id 10's tunnel zones are Advanced's on the reverse road (reverse.reversed_tunnel's rule): zone z' = K - z, the
    boundary thresholds swapped and mapped (starts'[j] = M(ends[K - j]), ends'[i] = M(starts[K - i])), each lit
    zone's position with it, the walk's unreachable mask bit K + 1 dropped; the lit range [lo, hi) -> [M(hi - 1),
    M(lo) + 1). Id 9: Expert's range so (no zones). Id 8 keeps the shared record (no tunnel bit on the reverse road);
    ids 1 and 2 unchanged."""
    from dtools.convert import reverse
    base, out = _sets(romset_zip, disc)
    for n, k in ((2, 10), (1, 9)):
        K, starts, ends, (lo, hi), mask, pos = _tunnel(out, n)
        assert _tunnel(base, n) == (K, starts, ends, (lo, hi), mask, pos)
        L = len(G.decode_course(out, k).road)
        M = lambda s: reverse.plain_segment(POINT_OFFSET[n], L, s)  # noqa: E731,B023
        K2, s2, e2, (lo2, hi2), mask2, pos2 = _tunnel(out, k)
        assert K2 == K and (lo2, hi2) == (M(hi - 1), M(lo) + 1), (k, lo, hi, lo2, hi2)
        assert s2[1:] == tuple(M(ends[K - j]) for j in range(1, K + 1))
        assert e2[:K] == tuple(M(starts[K - i]) for i in range(K))
        assert pos2 == {K - z: p for z, p in pos.items() if z <= K} and mask2 == sum(1 << (K - z) for z in pos if z <= K)
        assert list(s2[1:]) == sorted(s2[1:]) and list(e2[:K]) == sorted(e2[:K])
    assert _tunnel(out, 8) == _tunnel(out, 0)


# ---- step 3: the converted courses in reverse (ids 12-15) ----

def test_4r_builds_with_the_plain_cell(romset_zip, disc):  # noqa: F811
    """National Park Speedway in reverse at id 12 (model scenery): converted_set passes the plain course's data to
    the ground, so 4R's off-road cell 1765 takes plain 4's (ground.convert_ground(..., plain=)); id 12's polygon 1765 is
    that conversion's."""
    from tests.test_convert_tables import _data
    base, _ = _sets(romset_zip, disc)
    out = build.converted_set(base, disc, 4, slot=12, variant="R", scenery="model")
    want = ground.convert_ground(tables.with_lane_ranges_fixed(_rdata(disc, 4)), 12,
                                 plain=tables.with_lane_ranges_fixed(_data(disc, 4))).polys[1765]
    assert G.decode_course(out, 12).ground_data.polys[1765].corners == want.corners
