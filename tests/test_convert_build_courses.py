"""Tests for the converted courses of Phase 10 (dtools/convert/build.py): National Park Speedway (id 4), Mermaid Lake
(id 7), the re-meshed cache, the mirrors of 4 and 7 (ids 12 and 15) and the final-set checks. Split out of
tests/test_convert_build.py (Phase 11 Task 1: the pre-commit hook refuses files over 256 KB); the helpers and fixtures
are that file's."""

import dataclasses
import hashlib
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade import polyrom as P
from dtools.arcade import texture as T
from dtools.arcade import tgp
from dtools.arcade.romset import COPRO_SIZE
from dtools.arcade.tablebase import DATA_BASE
from dtools.convert import ConvertError, build, model
from dtools.dc import pol
from dtools.dc.exe import EXE_NAME
from tests.test_convert_build import HALF, _romset, _runs_mask
from tests.test_convert_tables import _data, disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_mirror import _differ


# ---- Phase 10 Task 3: National Park Speedway (n 4) at id 4 (discovery log docs/spike/2026-11-courses-4-7.md) ----

PARK_DECALS = (27, 43, 69, 70, 71)
PARK_VOLUMES = ((7, 0), (8, 0), (9, 0)) + tuple((13, k) for k in range(15)) + ((19, 0), (27, 0), (30, 0), (33, 0),
                (38, 0), (41, 0)) + tuple((45, k) for k in range(5)) + tuple((46, k) for k in range(4)) + ((47, 0),)
PARK_PROPS = (2, 4, 5, 6, 7, 8, 9, 52, 53)
"""CS4's drops, measured: the decal textures, the shadow volumes (19:0 among them, ruled one), the run-time props
(the road guard keeps none of them)."""
PARK_WHEEL, PARK_WHEEL_NEAR = 3, 12
"""CS4 object 3, the big wheel's frame (build.NAMED_PROPS[4]), and how many of its 198 vertices lie within ROAD_NEAR of
the road line."""
PARK_FLAT, PARK_BELOW = {52: 4.09, 53: 2.67}, {52: 15.59, 53: 10.16}
"""CS4 objects 52 and 53: the distance to the road line in x and z only (the follow-up's premise, "3.8-4.4 units"),
and how far every vertex lies below the ground (the course's CS_DATA cells) under it, at least (measured, to 0.01)."""


def _ground_heights(d, x: float, z: float) -> list[float]:
    """The heights of `d`'s ground cells (each quad split 0-1-2 and 0-2-3) at (x, z)."""
    out = []
    for c in d.cells:
        q = np.array(c.corners, dtype=float)
        if not (q[:, 0].min() <= x <= q[:, 0].max() and q[:, 2].min() <= z <= q[:, 2].max()):
            continue
        for t in ((0, 1, 2), (0, 2, 3)):
            a, b, e = q[list(t)]
            mat = np.array([[b[0] - a[0], e[0] - a[0]], [b[2] - a[2], e[2] - a[2]]])
            if abs(np.linalg.det(mat)) < 1e-12:
                continue
            u, v = np.linalg.solve(mat, [x - a[0], z - a[2]])
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                out.append(a[1] + u * (b[1] - a[1]) + v * (e[1] - a[1]))
    return out


def test_cs4_drop_rules(disc):  # noqa: F811
    """CS4 (National Park Speedway) as the scenery draws it: the decals, the two sky domes, 34 shadow volumes
    (v0.1.1: 23 of them in per-volume ID colours, not black, found by their kind) and nine run-time props are dropped (kept_model with the course, so past the road guard).

    - 19:0 (kind 1, 263 x 15 x 253 units) is a shadow volume and dropped (the ruling: discovery log, the renders with
      and without it in out/phase10/snaps/task3): untextured, black and closed like the ten black ones, with their PowerVR
      parameter words exactly (0x8000002d, 0x83400000, 0x20080140, 0x4000000; v0.1.1: the 23 in ID colours take
      0x8000002d or 0x800000ad first) and their face colour; it lies along
      the road (50 road points, segments 76-125, inside its box), where drawn as polygons it stands as a black wall
      over the road's left wall and verge; its kind word (1, the others 3) is the only difference.
    - the road guard keeps none of CS4's props: objects 52 (a frame, texture 35) and 53 (a tree, texture 42) lie
      PARK_FLAT from the road line in x and z only; in x, y and z they are 16.80 and 12.63 units from it, past
      ROAD_NEAR, every vertex at least PARK_BELOW under the ground there (modelled at y 0-15, the road and ground at
      y 23-27): props placed at run time, not road-side scenery.
    - object 3 (fix round 1 of Task 4, the ruling on I1) is a run-time prop the origin rule misses, named in
      NAMED_PROPS[4]: a big wheel's frame (PARK_WHEEL: 100 x 100 x 14 units, textures 46 and 54) modelled about its
      hub at the origin (sphere centre at the origin and radius 50, to 0.05), so it reaches 50 units out; converted as
      scenery it stands across the road near segment 394 (PARK_WHEEL_NEAR of its vertices within ROAD_NEAR of the road
      line; one-sided strips, culling mode 2: in MAME drawn, but its faces turn away from the lap's views). kept_model
      drops it with NAMED_PROPS[4] and keeps it without."""
    from dtools.dc.disc import course_files
    d = _data(disc, 4)
    mn, tn = course_files(4, kind="model"), course_files(4, kind="textures")
    m, blob = pol.decode_pol(disc.read(mn), mn), disc.read(tn)
    assert build.decals(m, blob) == PARK_DECALS and build.sky_domes(m) == (0, 1)
    assert build.shadow_volumes(m) == PARK_VOLUMES
    assert build.origin_props(m) == build.origin_props(m, d) == PARK_PROPS
    # 19:0
    vol = m.objects[19]
    assert vol.kind == 1 and len(vol.meshes) == 1 and vol.meshes[0].texture == pol.NO_TEXTURE
    assert {m.objects[o].meshes[k].params for o, k in PARK_VOLUMES} == {(p, 0x83400000, 0x20080140, 0x4000000)
                                                                        for p in (0x8000002D, 0x800000AD)}
    assert {m.objects[o].kind for o, k in PARK_VOLUMES if o != 19} == {3}
    ps = np.array([(e.x, e.y, e.z) for st in vol.meshes[0].strips for e in st.entries if isinstance(e, pol.Vertex)])
    lo, hi = ps.min(axis=0), ps.max(axis=0)
    assert np.allclose(hi - lo, (263.3, 15.5, 253.6), atol=0.05)
    a, _ = build._road_line(d)
    centres = [np.mean([(ln.x, ln.y, ln.z) for ln in rp.lanes[rp.lane_lo:rp.lane_hi + 1]], axis=0) for rp in d.road]
    inside = [i for i, c in enumerate(centres) if lo[0] <= c[0] <= hi[0] and lo[2] <= c[2] <= hi[2]]
    assert (inside[0], inside[-1], len(inside)) == (76, 125, 50)
    # 52 and 53 against the road line
    b = build._road_line(d)[1]
    for oi in (52, 53):
        pts = np.array([(e.x, e.y, e.z) for me in m.objects[oi].meshes for st in me.strips for e in st.entries
                        if isinstance(e, pol.Vertex)])
        assert not build._near_line(pts, a, b).any(), oi
        assert build._near_line(pts * (1, 0, 1) + (0, 24.5, 0), a, b).any(), oi     # raised to the road: near
        flat = min(np.linalg.norm(pts[:, [0, 2]] - (p + np.clip((pts - p)[:, [0, 2]] @ (q - p)[[0, 2]]
                                                                / ((q - p)[[0, 2]] @ (q - p)[[0, 2]]), 0, 1)[:, None]
                                                    * (q - p))[:, [0, 2]], axis=1).min() for p, q in zip(a, b))
        assert round(flat, 2) == PARK_FLAT[oi], (oi, flat)
        below = min(min(_ground_heights(d, x, z)) - y for x, y, z in pts)
        assert math.floor(below * 100) / 100 == PARK_BELOW[oi], (oi, below)
    # 3, the wheel's frame
    assert build.NAMED_PROPS[4] == (PARK_WHEEL,) and PARK_WHEEL not in PARK_PROPS
    wheel = m.objects[PARK_WHEEL]
    pts = np.array([(e.x, e.y, e.z) for me in wheel.meshes for st in me.strips for e in st.entries
                    if isinstance(e, pol.Vertex)])
    assert {me.texture for me in wheel.meshes} == {46, 54} and len(pts) == 198
    assert np.allclose(pts.max(axis=0) - pts.min(axis=0), (100.0, 100.0, 14.0), atol=0.5)
    assert np.allclose(wheel.sphere, (0.0, 0.0, 0.0, 50.0), atol=0.05)
    assert np.hypot(pts[:, 0], pts[:, 2]).max() > build.PROP_RADIUS
    assert int(build._near_line(pts, a, b).sum()) == PARK_WHEEL_NEAR
    kept = build.kept_model(m, blob, d, props=build.NAMED_PROPS[4])
    assert [len(kept.objects[o].meshes) for o in (19, 52, 53, PARK_WHEEL)] == [0, 0, 0, 0]
    plain = build.kept_model(m, blob, d)
    assert plain.objects[PARK_WHEEL] == wheel
    assert [ob for k, ob in enumerate(kept.objects) if k != PARK_WHEEL] == \
        [ob for k, ob in enumerate(plain.objects) if k != PARK_WHEEL]
    assert sum(len(ob.meshes) for ob in kept.objects) < sum(len(ob.meshes) for ob in m.objects)


@pytest.fixture(scope="module")
def park_id4(disc):  # noqa: F811
    """n 4 (National Park Speedway) with its own scenery (build.REMESH[4], QUAD[4], NAMED_PROPS[4]) at course id 4 of
    the chain after Desert City (grow_ids, guard_new_ids, widen_ids, Desert City converted at id 3): (set before,
    scenery, course, set after). The set is converted_set(base, disc, 4, slot=4, scenery="model")."""
    from dtools.dc.disc import course_files
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    base = build.converted_set(C.widen_ids(C.guard_new_ids(C.grow_ids(rs))), disc, 3, slot=3, scenery="model")
    d, exe = _data(disc, 4), disc.read(EXE_NAME)
    mn, tn = course_files(4, kind="model"), course_files(4, kind="textures")
    sd = build.model_scenery(base, d, pol.decode_pol(disc.read(mn), mn), disc.read(tn), build.THIN.get(4), slot=4,
                             quad=build.QUAD.get(4), remesh=build.REMESH.get(4), props=build.NAMED_PROPS[4])
    c = build.converted_course(base, d, exe, 4, slot=4, scenery="model", scenery_data=sd)
    out = build.converted_set(base, disc, 4, slot=4, scenery="model")
    return base, sd, c, out


def test_race_times_writes_in_place(park_id4, monkeypatch):
    """build.race_times (v0.1.2): every time table of the Course is rewritten word for word in place (same addresses,
    same lengths), the decoded course reads the new words and passes golden.check_course, and only those words change;
    a missing or extra key, a wrong length, a bad word, a table shared with different words, or a course with no
    RACE_TIMES entry is a ConvertError."""
    _, _, _, out = park_id4
    c = G.decode_course(out, 4)
    tables = {key: tuple(1000 + 10 * key[0] + key[1] + i for i in range(len(w))) for key, (_, _, w) in c.time_tables.items()}
    new = build.race_times(out, c, tables)
    back = G.decode_course(new, 4)
    assert {k: v[2] for k, v in back.time_tables.items()} == tables
    assert {k: v[:2] for k, v in back.time_tables.items()} == {k: v[:2] for k, v in c.time_tables.items()}
    G.check_course(back, new)
    assert (new.program, new.polygons, new.textures) == (out.program, out.polygons, out.textures)
    changed = np.flatnonzero(np.frombuffer(new.data, np.uint8) != np.frombuffer(out.data, np.uint8))
    spans = {t - DATA_BASE + i for _, t, w in c.time_tables.values() for i in range(4 * len(w))}
    assert changed.size and set(changed.tolist()) <= spans
    key = next(iter(tables))
    for bad in ({k: v for k, v in tables.items() if k != key}, {**tables, (9, 9): (1,)},
                {**tables, key: tables[key][:-1]}, {**tables, key: (0,) + tables[key][1:]},
                {**tables, key: (True,) + tables[key][1:]}, {**tables, key: (1 << 32,) + tables[key][1:]}):
        with pytest.raises(ConvertError):
            build.race_times(out, c, bad)
    frozen = G.decode_course(build.race_times(out, c), 4)          # the default: RACE_TIMES[4], expanded per table
    assert {k: v[2] for k, v in frozen.time_tables.items()} == {
        k: build.race_table(*build.RACE_TIMES[4][k], len(v[2])) for k, v in c.time_tables.items()}
    assert build.race_table(5, (1, 2), 5) == (5, 1, 2, 1, 2)
    for n in (4, 2, 1):
        with pytest.raises(ConvertError):
            build.race_table(5, (1, 2), n)
    monkeypatch.setattr(build, "RACE_TIMES", {})
    with pytest.raises(ConvertError):
        build.race_times(out, c)                    # no RACE_TIMES entry for the course
    shared = dataclasses.replace(c, time_tables={k: (p, next(iter(c.time_tables.values()))[1], w)
                                                 for k, (p, _, w) in c.time_tables.items()})
    with pytest.raises(ConvertError):
        build.race_times(out, shared, tables)
    for bad in ((None, c, tables), (out, None, tables)):
        with pytest.raises(ConvertError):
            build.race_times(*bad)


def test_converted_id4_round_trip(park_id4):
    """National Park Speedway as course id 4, after Desert City on the chain (Review Focus 4 and 5 as id 3's):

    - the set decodes id 4 to the converted Course on every field, check_course passes, every table re-emits
      (verify_lossless empty) and encode(base, c) is the set converted_set builds; a standing start (ROLLING_STARTS[4]),
      grid base GRID_BASES[4], its 425 segments and three checkpoints;
    - its own of every table it changes, in NEW_ID_RUNS (each past Desert City's), its lists and HUD outline's list in
      the list home DATA_FREE, its own 1 MB course sheet block (not Desert City's), the shared EMPTY_ACTORS;
    - ids 0-3 byte-identical to the input set's: each decodes and re-emits to the same Course, every byte their tables
      cover (golden.coverage) is unchanged, their course blocks and the common block too;
    - the free-bytes rule: every byte that differs from the set before is an id-4 per-id word, record 4, TGP slot 4
      or a byte that was 0xff in a free run (NEW_ID_RUNS and DATA_FREE)."""
    base, sd, c, out = park_id4
    d = G.decode_course(out, 4)
    assert _differ(d, c) == []
    G.check_course(c)
    G.check_course(d, out)
    assert G.verify_lossless(out, 4) == [] and G.mismatches(out, c) == []
    assert build.encode(base, c, build.RACE_FIELDS + build.MODEL_FIELDS + build.ID_FIELDS).split_files() == \
        out.split_files()
    assert c.rolling_start[1] is False and c.grid_base[1] == build.GRID_BASES[4] == 418
    assert len(c.road) - 1 == 425 and [cp.segment for cp in c.checkpoints] == [125, 257, 0]
    assert c.actor_table == build.EMPTY_ACTORS and build.CONVERTED_IDS[:2] == (3, 4)
    runs, home = build.NEW_ID_RUNS, tgp.DATA_FREE
    c3 = G.decode_course(base, 3)

    def inside(off, size, run=runs["data"]):
        return run[0] <= off and off + size <= run[0] + run[1]

    e = c.entry
    for f in ("step_table", "record_table", "line_table", "block_ref_a", "block_ref_b"):
        assert inside(getattr(e, f), 4) and getattr(e, f) > getattr(c3.entry, f), f
    assert inside(c.grid - DATA_BASE, 4 * C.GRID_CELLS) and c.grid != c3.grid
    block = c.sheet_sources[0]
    assert block % 0x100000 == 0 and inside(block, 0x100000) and block != c3.sheet_sources[0]
    assert T.sheet_source(out, 0, 4) == block and T.sheet_source(out, 0, 3) == c3.sheet_sources[0]
    assert 4 * c.ground_data.list_base > 4 * c3.ground_data.list_base >= runs["copro_data"][0]
    assert all(inside(4 * k, 4 * o.word_count, runs["polygons"]) for k, o in sd.objects.items())
    assert min(sd.objects) > max(c3.objects)
    assert all(inside(2 * k, 8, runs["textures"]) for k in sd.headers)
    assert all(inside(k - DATA_BASE, 16 * len(v) + 4, home) for k, v in sd.lists.items())
    assert inside(c.hud_map.outline_list - DATA_BASE, 20, home) and c.hud_map.outline_list != c3.hud_map.outline_list
    # ids 0-3 byte-identical
    for k in range(4):
        ref = G.decode_course(base, k)
        assert _differ(G.decode_course(out, k), ref) == [], k
        assert G.mismatches(out, ref) == [], k
        for region, off, size, _ in G.coverage(base, k):
            assert getattr(out, region)[off:off + size] == getattr(base, region)[off:off + size], (k, region, off)
        blk = T.sheet_source(base, 0, k)
        assert out.data[blk:blk + 0x100000] == base.data[blk:blk + 0x100000], k
    common = T.sheet_source(base, 1)
    assert out.data[common:common + 0x100000] == base.data[common:common + 0x100000]
    # the free-bytes rule
    own = set()
    for name in ("COURSE_POINTERS", "COURSE_SHEETS", "GRID_TABLES", "ACTOR_TABLES", "CHECKPOINT_TABLES",
                 "SEGMENT_FLAGS", "HUD_MAP_LISTS", "HUD_MAP_PLACES", "HUD_MAP_ZOOMS", "GRID_BASES", "ROLLING_STARTS",
                 "TIME_TABLES[0]", "TIME_TABLES[1]", "TIME_TABLES[2]"):
        a_ = C.ID_ARRAYS[name]
        own |= set(range(a_.new - DATA_BASE + 4 * a_.stride, a_.new - DATA_BASE + 5 * a_.stride))
    rec = C.record_address(4) - DATA_BASE
    own |= set(range(rec, rec + C.COURSE_ENTRY_SIZE))
    allowed = {"data": own, "copro_data": {4 * (w + 4) + k + h for w in (tgp.SLOT_LIST, tgp.SLOT_POOL)
                                           for k in range(4) for h in (0, HALF)}}
    free = {"data": [runs["data"], home], "copro_data": [runs["copro_data"]], "program": [],
            "polygons": [runs["polygons"]], "textures": [runs["textures"]]}
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a = np.frombuffer(getattr(base, region), np.uint8)
        b = np.frombuffer(getattr(out, region), np.uint8)
        idx = np.flatnonzero(a != b)
        ok = np.isin(idx, np.fromiter(allowed.get(region, ()), np.int64)) | (
            _runs_mask(len(a), free[region])[idx] & (a[idx] == 0xFF))
        assert ok.all(), (region, [hex(int(k)) for k in idx[~ok][:8]])
    used = {r: int((np.frombuffer(getattr(out, r), np.uint8)[_runs_mask(len(getattr(out, r)), [runs[r]])] != 0xFF).sum()
                   - (np.frombuffer(getattr(base, r), np.uint8)[_runs_mask(len(getattr(base, r)), [runs[r]])] != 0xFF)
                   .sum()) for r in runs}
    used["lists"] = int((np.frombuffer(out.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum()
                        - (np.frombuffer(base.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum())
    print(f"\nid 4's bytes per run (non-0xff bytes added): {used}")


M4B_SHA1 = {"program": "c45108f1b915bd220b700e7e98b169c6901960be", "data": "7c64c309ec3c603c933031c6d1db6d64d010f001",
            "copro_data": "4300f0d41c609a4e16c4894d8dc8b84291e0b581",
            "polygons": "0a031942ac4da0de905e4f2fb61c67098577e59b",
            "textures": "6dfd10c3de7a40cb42eea479f246995d81fe4dfc"}
"""Phase 9's final set (M4b), the sha1 of each region: tests/test_mirrors_mame.py's final_set built by the code at
278b42a, before Task 3 (the Task 3 review's independent build); m4b_set must equal it."""


FINAL_SHA1 = {"program": "5e8524361d3369a27cc88adcddeafe7c936a643d",
              "data": "11d72024a985ba1d690f15ebec2061aa44caea4b",
              "copro_data": "49473ef2c22b66e022d67c9e71326e29e338624c",
              "polygons": "f095aae1e96fe4528f6d9f5566d5d481219160ef",
              "textures": "d8535a4977e588d3ab9dedd9e5a361d027701d46"}
"""Task 3's final set (tests/test_mirrors_mame.py `task3_set`: Desert City at 3, National Park Speedway at 4, the
mirrors of 0-3, the rankings, the select with TASK3_ENTRIES), the sha1 of each region, built independently of the code
under test: 0f0a36c's code (git archive, its own empty re-mesh cache) with the one deliberate change of Task 4's fix
round 1 (the ruling on I1): CS4's object 3, the big wheel's frame, dropped with the props while n 4 converts
(origin_props wrapped, which is what NAMED_PROPS[4] does). The same harness without the change gives FINAL_SHA1_T3,
0f0a36c's own pin, byte for byte. What the change moves (docs/spike/2026-11-courses-4-7.md, "Fix round 1"): program
and copro_data are unchanged; in data, id 4's object lists, HUD map words and course-sheet texels and the object lists
and HUD map words of the mirrors 8-11 (2,363 bytes); in polygons, id 4's objects and HUD outline and the mirrors'
(placed after id 4's, so every address moves; 2,300,567 bytes); in textures, id 4's headers and points (132,863
bytes). The re-mesh's output is the same object for object but 3 (REMESH4_SHA1). Phase 13 Task 2 round 2 repinned it:
model._merge's quad tests made exact (5b970ee; four National Park Speedway pairs on the shear bound, three of which the
float lstsq had rejected on this Mac, now merge), built by 5b970ee's code from a git archive (fresh venv, Python 3.15.0,
numpy 2.5.4; out/phase13/r2/pins.txt); linux/amd64 builds the same bytes. v0.1.1 repinned it (data, polygons, textures; program and copro_data unchanged): build.shadow_volumes
also takes an untextured closed mesh of a kind-3 object (CS4's 23 volumes in per-volume ID colours), built by the code
under test; the release pin, test_release.py RELEASE_SHA1, matches an independent build of v0.1's code with only that
rule patched in (out/debug/np-start/fixset.py)."""
FINAL_SHA1_T3 = {"program": "5e8524361d3369a27cc88adcddeafe7c936a643d",
                 "data": "c03545426e2990359b095f5766b45b238f96ea3a",
                 "copro_data": "49473ef2c22b66e022d67c9e71326e29e338624c",
                 "polygons": "24be10ac3556fa97e9a55e687f648e6eb3237409",
                 "textures": "4395cd15b6a4969570b6497d4521f6bbe0b48df7"}
"""Task 3's final set as 5833095's and 0f0a36c's code built it (before Task 4's fix round 1), the set Task 3's gate
measured: kept as the record FINAL_SHA1's harness reproduces without the deliberate change."""


def test_final_set_with_id4(disc):  # noqa: F811
    """The final chain with National Park Speedway (tests/test_mirrors_mame.py `final_set`: id 4 converted after
    Desert City, before the mirrors) against Phase 9's (`m4b_set`): ids 0-3 decode to the same Course and every byte
    their tables cover is the same; the mirrors 8-11 draw and race the same (test_mirror._content: every table resolved
    through its pointers) though their tables now lie past id 4's; id 4 holds its own course (golden.own_tables) and
    entry 4 is live, entry 12 locked; the select's char upload is Phase 9's (127 chars, four runs). m4b_set's bytes
    are M4B_SHA1 (built by the code before Task 3), the set with id 4's FINAL_SHA1 (built by 0f0a36c's code with
    Task 4 fix round 1's one deliberate change, CS4's object 3 dropped) and the program room stays 5,312
    (test_program_room's final chain). Since Phase 10 Task 4 the final set holds Mermaid Lake too, so the set with id
    4 measured here is Task 3's chain (tests/test_mirrors_mame.py `task3_set`, its row TASK3_ENTRIES)."""
    from dtools.arcade import select as S
    from tests.test_mirror import _content, _eq
    from tests.test_mirrors_mame import TASK3_ENTRIES, m4b_set, task3_set
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    new, old = task3_set(rs, disc), m4b_set(rs, disc)
    # an independent reference (fix round 1, M4): m4b_set is Phase 9's final set byte for byte, as built by the code
    # at 278b42a (git archive; the review's sha1 per region, out/phase10/review-t3-analysis.txt)
    assert {r: hashlib.sha1(getattr(old, r)).hexdigest() for r in M4B_SHA1} == M4B_SHA1
    assert build.program_room(new) == build.program_room(old) == 5056     # id 4 takes no program bytes
    assert {r: hashlib.sha1(getattr(new, r)).hexdigest() for r in FINAL_SHA1} == FINAL_SHA1    # the reference's
    select = ("char_blocks", "screen_uploads")         # the select's upload set, read from the set (its row moved)
    for k in range(4):
        assert _differ(G.decode_course(new, k), G.decode_course(old, k), skip=select) == [], k
        for region, off, n, name in G.coverage(old, k):
            if not name.startswith(("char block", "screen upload", "upload")):
                assert getattr(new, region)[off:off + n] == getattr(old, region)[off:off + n], (k, region, off, name)
    moved = 0
    for k in (8, 9, 10, 11):
        a, b = G.decode_course(new, k), G.decode_course(old, k)
        assert _eq(_content(a), _content(b)), k
        assert G.verify_lossless(new, k) == [], k
        moved += a.grid != b.grid
    assert moved == 4                                           # their tables moved past id 4's
    assert G.own_tables(new, 4) and not G.own_tables(old, 4)
    row = S.select_row(new)
    assert row.live[4] and not row.live[12] and not S.select_row(old).live[4]
    assert S.graphics(new, TASK3_ENTRIES).uploaded() == S.graphics(old, S.M4B_ENTRIES).uploaded() == 127


def test_remeshed_cache(tmp_path, monkeypatch):
    """build.remeshed (fix round 2): model.remesh kept on disk by a key of its inputs and code. A miss computes and
    stores the result; a hit reads it back without computing (model.remesh replaced by one that fails); another eps, a
    changed model, or changed code (model.py's bytes) is a miss; a damaged file is a miss and is rewritten; bad inputs
    are refused before the cache is read. No test needs a warm cache: test_remesh_cs4_bound computes id 4's re-mesh
    itself."""
    from tests.test_convert_model import _arc_road
    m = _arc_road(200.0)
    want = model.remesh(m, 0.1)
    assert build.remeshed(m, 0.1, tmp_path) == want                      # miss
    files = sorted(tmp_path.glob("*.pkl"))
    assert len(files) == 1 and files[0].stem == build._remesh_key(m, 0.1)
    real = model.remesh

    def fail(*args):
        raise AssertionError("computed on a hit")
    monkeypatch.setattr(build.model, "remesh", fail)
    assert build.remeshed(m, 0.1, tmp_path) == want                      # hit
    with pytest.raises(AssertionError):
        build.remeshed(m, 0.05, tmp_path)                                # another eps: a miss
    with pytest.raises(AssertionError):
        build.remeshed(_arc_road(100.0), 0.1, tmp_path)                  # another model: a miss
    copy = tmp_path / "model_copy.py"
    copy.write_bytes(Path(model.__file__).read_bytes() + b"\n")
    monkeypatch.setattr(build.model, "__file__", str(copy))
    assert build._remesh_key(m, 0.1) != files[0].stem                   # changed code: another key
    monkeypatch.undo()
    assert model.remesh is real
    files[0].write_bytes(b"not a pickle")
    assert build.remeshed(m, 0.1, tmp_path) == want                      # damaged: recomputed, rewritten
    assert pickle.loads(files[0].read_bytes())[0] == files[0].stem
    for bad in ((None, 0.1), (m, 0), (m, math.nan), (m, True)):
        with pytest.raises(ConvertError):
            build.remeshed(*bad, cache=tmp_path)


def test_remeshed_cache_leaves_no_tmp(tmp_path, monkeypatch):
    """build.remeshed, final review of Phase 10 (Minor 4): a write that fails (here os.replace) removes its
    `<key>.<pid>.tmp` and re-raises; no file is left in the cache folder, and the next call stores the result."""
    from tests.test_convert_model import _arc_road
    m = _arc_road(200.0)
    real = os.replace

    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(build.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        build.remeshed(m, 0.1, tmp_path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(build.os, "replace", real)
    want = build.remeshed(m, 0.1, tmp_path)
    assert [f.suffix for f in tmp_path.iterdir()] == [".pkl"] and want == model.remesh(m, 0.1)


# ---- Phase 10 Task 4: Mermaid Lake (n 7) at id 7 (discovery log docs/spike/2026-11-courses-4-7.md, "Mermaid Lake") --

LAKE_DECALS, LAKE_DOMES = (67, 112), (0, 1)
LAKE_VOLUMES = ((132, 0), (132, 1), (132, 2), (132, 3), (132, 4), (134, 0), (134, 1), (137, 0), (140, 0), (142, 0),
                (145, 0), (162, 0))
LAKE_FLAGS = (tuple(range(3, 67)), tuple(range(67, 131)))
"""CS7 objects 3-66 and 67-130: two waving flags, 64 frames each (an animation the course plays at run time): each
frame one mesh of 48 vertices, the first series textured 0 (the Stars and Stripes under a SEGA banner), the second 1
(the same flag under Captain Future's), every frame within x -8.0..8.5, y 25.8..35.0, z -7.6..0.4 of the origin."""
LAKE_PROPS = LAKE_FLAGS[0] + LAKE_FLAGS[1] + (131, 132)
"""CS7's run-time props by the origin rule: the 128 flag frames, the figure 131 (a marshal 4.9 units tall holding a
SEGA card: textures 11-13) and 132, its shadow volume (also one of LAKE_VOLUMES). The road guard keeps none: the
road passes 31 units from the origin, and every prop lies at least LAKE_PROP_ROAD from the road line in x, y and z."""
LAKE_PROP_ROAD = 29.0
LAKE_AIRSHIP = 2
"""CS7 object 2: the Dreamcast airship (the blue DREAMCAST / Deep Space Planet banner, textures 2-6: hull panels,
the gondola's windows, the banner, the propeller disc, a gradient), modelled about the origin (its bounding sphere's
centre x 0, z -16.5), 40 x 45 x 139 units: it reaches 86 units from the origin, so the origin rule (every vertex
within PROP_RADIUS) does not find it; its keel lies 24.1 units below the ground at the origin. A run-time prop:
build.NAMED_PROPS[7]."""


def test_cs7_origin_props(disc):  # noqa: F811
    """CS7 (Mermaid Lake) as the scenery draws it (kept_model with the course, so past the road guard, and
    NAMED_PROPS[7]): the decal textures (v0.1.2 adds 67, the shadow nearly one colour), the two sky domes, twelve shadow volumes (v0.1.1:
    132:1-4, kind 3, in a prop already dropped) and the run-time props are dropped.

    - the origin rule finds 130 objects (LAKE_PROPS), and they are props: two flag animations of 64 frames each
      (LAKE_FLAGS: the same mesh shape frame to frame, moved a little, all at the same place: a single object would
      not be modelled 64 times over), a marshal and its shadow; the guard keeps none (kept: nothing; dropped: all
      130), every vertex at least LAKE_PROP_ROAD from the road line;
    - object 2, the airship (LAKE_AIRSHIP), is a run-time prop the origin rule misses: modelled about the origin
      (sphere centre x 0), 24 units into the ground there, reaching 86 units out. NAMED_PROPS names it for n 7 only;
      kept_model drops it with `props` and keeps it without (CS3 names none; CS4 names its wheel's frame,
      test_cs4_drop_rules)."""
    from dtools.dc.disc import course_files
    d = _data(disc, 7)
    mn, tn = course_files(7, kind="model"), course_files(7, kind="textures")
    m, blob = pol.decode_pol(disc.read(mn), mn), disc.read(tn)
    assert build.decals(m, blob) == LAKE_DECALS and build.sky_domes(m) == LAKE_DOMES
    assert build.shadow_volumes(m) == LAKE_VOLUMES
    props = build.origin_props(m, d)
    assert props == build.origin_props(m) == LAKE_PROPS and len(props) == 130
    kept_by_guard = sorted(set(build.origin_props(m)) - set(props))
    assert kept_by_guard == []                                  # kept: none; dropped: all 130
    a, b = build._road_line(d)

    def points(oi):
        return np.array([(e.x, e.y, e.z) for me in m.objects[oi].meshes for st in me.strips for e in st.entries
                         if isinstance(e, pol.Vertex)])

    def road_distance(ps):
        return min(np.linalg.norm(ps - (p + np.clip((ps - p) @ (q - p) / ((q - p) @ (q - p)), 0, 1)[:, None]
                                        * (q - p)), axis=1).min() for p, q in zip(a, b))
    assert min(road_distance(points(oi)) for oi in props) >= LAKE_PROP_ROAD
    for series, texture in zip(LAKE_FLAGS, (0, 1)):
        assert len(series) == 64
        for oi in series:
            ob, ps = m.objects[oi], points(oi)
            assert ob.kind == 1 and len(ob.meshes) == 1 and ob.meshes[0].texture == texture and len(ps) == 48, oi
            assert (ps.min(axis=0) >= (-8.1, 25.7, -7.7)).all() and (ps.max(axis=0) <= (8.6, 35.1, 0.5)).all(), oi
        frames = [points(oi) for oi in series]
        assert all(not np.array_equal(f, g) for f, g in zip(frames, frames[1:]))         # each frame moves
    assert {me.texture for me in m.objects[131].meshes} == {11, 12, 13} and (132, 0) in LAKE_VOLUMES
    # the airship
    assert {k: v for k, v in build.NAMED_PROPS.items() if k not in (5, 6)} == {4: (3,), 7: (LAKE_AIRSHIP,)}
    ship, ps = m.objects[LAKE_AIRSHIP], points(LAKE_AIRSHIP)
    assert abs(ship.sphere[0]) < 1e-5 and round(ship.sphere[2], 1) == -16.5 and LAKE_AIRSHIP not in props
    assert np.allclose(ps.max(axis=0) - ps.min(axis=0), (40.2, 44.6, 138.7), atol=0.1)
    assert np.hypot(ps[:, 0], ps[:, 2]).max() > build.PROP_RADIUS
    ground = min(_ground_heights(d, 0.0, 0.0))
    assert round(ground - ps[:, 1].min(), 1) == 24.1
    assert {me.texture for me in ship.meshes} >= {2, 3, 4, 5, 6}
    kept, plain = build.kept_model(m, blob, d, props=build.NAMED_PROPS[7]), build.kept_model(m, blob, d)
    assert len(kept.objects[LAKE_AIRSHIP].meshes) == 0 and plain.objects[LAKE_AIRSHIP] == ship
    assert all(len(kept.objects[o].meshes) == 0 for o in props)
    assert [ob for k, ob in enumerate(kept.objects) if k != LAKE_AIRSHIP] == \
        [ob for k, ob in enumerate(plain.objects) if k != LAKE_AIRSHIP]
    for bad in ((999,), (-1,), ("2",), (True,), 2):
        with pytest.raises(ConvertError):
            build.kept_model(m, blob, d, props=bad)


LAKE_ROAD_TRIANGLES = 4217
"""build.road_triangles of CS7's kept model (never thinned), plain or lifted. v0.1.2: 4,498 before; the 281 gone are
the shadow decal 67's triangles on the road (build.decals' nearly-one-colour rule)."""
LIFT_HEIGHT_TOL = 0.005
"""Task 2's tolerance for a converted ground's height (docs/spike/2026-11-courses-4-7.md, "The tolerance, stated
before measuring"): the lifted ground's answer less LIFT against the plain ground's (measured 8.4e-5 by the review)."""
LAKE_LOW = -40.5
"""CS7's lowest ground corner (the CS_DATA cells, to 0.1): the lake's tunnel under the road (segments 316-345 run
down to -36.1 and back, passing under segment 81 at segment 328)."""


def test_cs7_lift(disc):  # noqa: F811
    """build.LIFT[7] (discovery log, "Mermaid Lake", "The tunnel: the TGP's choice and the lift"): the TGP chooses,
    among the polygons a height query hits, the one with the smallest | |height| - y | (docs/formats/
    model2-tgp-ground.md, "Choice"): for ground below 0 that is the one nearest 0, not the one nearest the car. On
    CS7 the road dives to -36 under another part of the course, so there the arcade's ground answers the upper
    surface: on every lane point of segments 322-335 (82 points) and 112 AI points. Lifted by 41 units, every corner
    of the ground lies above 0, where | |height| - y | is the plain distance: every lane and AI point gets its own
    surface. The cells the ground repair changes are the same (REPAIRED[7]).

    lifted_data / lifted_model move every y (the used lanes, the AI points, the cells' corners and centres; the
    model's vertices and bounding spheres) by `dy`, as f32, and nothing else; past single precision they refuse
    without a numpy warning.

    Fix round 1 (M2): on the lifted course the lifted ground answers each lane point the plain ground answers with its
    own polygon with the same polygon, its height the plain one + 41 within Task 2's tolerance (LIFT_HEIGHT_TOL); the
    drop rules (decals, domes, shadow volumes, origin props past the guard, kept_model's meshes, road_triangles) give
    the same result lifted as plain."""
    from dtools.convert import ground
    from dtools.dc.disc import course_files
    d, mn = _data(disc, 7), course_files(7, kind="model")
    m = pol.decode_pol(disc.read(mn), mn)
    assert build.LIFT == {7: 41.0}
    low = min(c[1] for cell in d.cells for c in cell.corners)
    assert round(low, 1) == LAKE_LOW and low + build.LIFT[7] > 0
    up = build.lifted_data(d, build.LIFT[7])
    f32 = lambda v: float(np.float32(v))                                                     # noqa: E731
    for r0, r1 in zip(d.road, up.road):
        for j, (l0, l1) in enumerate(zip(r0.lanes, r1.lanes)):
            want = dataclasses.replace(l0, y=f32(l0.y + 41.0)) if r0.lane_lo <= j <= r0.lane_hi else l0
            assert l1 == want
        assert (r1.lane_lo, r1.lane_hi, r1.curve) == (r0.lane_lo, r0.lane_hi, r0.curve)
    assert all(q == dataclasses.replace(p_, y=f32(p_.y + 41.0)) for a_, b_ in zip(d.lines, up.lines)
               for p_, q in zip(a_, b_))
    assert all(c1.corners == tuple((x, f32(y + 41.0), z) for x, y, z in c0.corners) and c1.centre[1] ==
               f32(c0.centre[1] + 41.0) and c1.centre[::2] == c0.centre[::2] and c1.direction == c0.direction
               for c0, c1 in zip(d.cells, up.cells))
    assert (up.grid, up.lists, up.steps, up.attributes, up.order, up.line_slots) == \
        (d.grid, d.lists, d.steps, d.attributes, d.order, d.line_slots)
    assert build.lifted_data(d, 0.0) == d
    mu = build.lifted_model(m, build.LIFT[7])
    for o0, o1 in zip(m.objects, mu.objects):
        assert o1.sphere == (o0.sphere[0], f32(o0.sphere[1] + 41.0)) + tuple(o0.sphere[2:])
        for e0, e1 in zip((e for me in o0.meshes for st in me.strips for e in st.entries),
                          (e for me in o1.meshes for st in me.strips for e in st.entries)):
            assert e1 == (dataclasses.replace(e0, y=f32(e0.y + 41.0)) if isinstance(e0, pol.Vertex) else e0)
    assert (mu.header, mu.textures) == (m.header, m.textures) and build.lifted_model(m, 0.0) == m
    # final review, Minor 3: the lift is marked on the model (not a field) and carried to the Scenery
    assert build._lift_of(m) == 0.0 and build._lift_of(mu) == 41.0 and build._lift_of(build.lifted_model(mu, 1.0)) == 42.0
    for bad in ((None, 1.0), (d, math.inf), (d, True), (d, "1"), (d, 1e39)):
        with pytest.raises(ConvertError):
            build.lifted_data(*bad)
    for bad in ((None, 1.0), (m, math.nan), (m, True)):
        with pytest.raises(ConvertError):
            build.lifted_model(*bad)
    assert ground.repairs(up) == ground.repairs(d) == ground.REPAIRED[7]

    def off(dd):
        g = ground.convert_ground(dd, 7)
        lanes = [(i, j) for i, rp in enumerate(dd.road) for j in range(rp.lane_lo, rp.lane_hi + 1)
                 if (lambda r, ln: r is None or abs(r[0] - ln.y) > 2)(tgp.query(g, rp.lanes[j].x, rp.lanes[j].y + 0.5,
                                                                              -rp.lanes[j].z), rp.lanes[j])]
        ai = sum((lambda r: r is None or abs(r[0] - q.y) > 2)(tgp.query(g, q.x, q.y + 0.5, -q.z))
                 for line in dd.lines for q in line)
        return lanes, ai
    lanes, ai = off(d)
    assert (len(lanes), ai, min(lanes)[0], max(lanes)[0]) == (82, 112, 322, 335)
    assert off(up) == ([], 0)
    # M2 (fix round 1): the lifted ground answers the plain one + LIFT within Task 2's tolerance (LIFT_HEIGHT_TOL, the
    # ground repair's 0.005) at every used lane point the plain ground answers with its own polygon
    g0, g1 = ground.convert_ground(d, 7), ground.convert_ground(up, 7)
    worst, n_points = 0.0, 0
    for i, (r0, r1) in enumerate(zip(d.road, up.road)):
        for j in range(r0.lane_lo, r0.lane_hi + 1):
            if (i, j) in lanes:
                continue
            l0, l1 = r0.lanes[j], r1.lanes[j]
            h0, h1 = tgp.query(g0, l0.x, l0.y + 0.5, -l0.z), tgp.query(g1, l1.x, l1.y + 0.5, -l1.z)
            assert h1[1] == h0[1], (i, j)                               # the same polygon
            worst, n_points = max(worst, abs(h1[0] - build.LIFT[7] - h0[0])), n_points + 1
    print(f"\nlifted ground against plain + {build.LIFT[7]}: {n_points} lane points, worst {worst:.2e}")
    assert n_points == 2918 and worst <= LIFT_HEIGHT_TOL
    # M2: the drop rules give the same objects on the lifted course (the shipped path runs them there)
    tn = course_files(7, kind="textures")
    blob = disc.read(tn)
    assert build.decals(mu, blob) == build.decals(m, blob) == LAKE_DECALS
    assert build.sky_domes(mu) == build.sky_domes(m) == LAKE_DOMES
    assert build.shadow_volumes(mu) == build.shadow_volumes(m) == LAKE_VOLUMES
    assert build.origin_props(mu, up) == build.origin_props(m, d) == LAKE_PROPS
    k0, k1 = (build.kept_model(mm, blob, dd, props=build.NAMED_PROPS[7]) for mm, dd in ((m, d), (mu, up)))
    assert [len(ob.meshes) for ob in k1.objects] == [len(ob.meshes) for ob in k0.objects]
    assert len(build.road_triangles(k1, up)) == len(build.road_triangles(k0, d)) == LAKE_ROAD_TRIANGLES
    # M6: a lift past single precision is refused without a numpy overflow warning first
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConvertError, match="single precision"):
            build.lifted_data(d, 1e39)
        with pytest.raises(ConvertError, match="single precision"):
            build.lifted_model(m, 1e39)


@pytest.fixture(scope="module")
def lake_id7(disc):  # noqa: F811
    """n 7 (Mermaid Lake) with its own scenery at course id 7 of the chain after National Park Speedway (grow_ids,
    guard_new_ids, widen_ids, Desert City at id 3, National Park Speedway at id 4), lifted by build.LIFT[7] and
    without build.NAMED_PROPS[7]: (set before, scenery, course, set after). The set is converted_set(base, disc, 7,
    slot=7, scenery="model")."""
    from dtools.dc.disc import course_files
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    base = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    for n in (3, 4):
        base = build.converted_set(base, disc, n, slot=n, scenery="model")
    lift = build.LIFT[7]
    d, exe = build.lifted_data(_data(disc, 7), lift), disc.read(EXE_NAME)
    mn, tn = course_files(7, kind="model"), course_files(7, kind="textures")
    sd = build.model_scenery(base, d, build.lifted_model(pol.decode_pol(disc.read(mn), mn), lift), disc.read(tn),
                             build.THIN.get(7), slot=7, quad=build.QUAD.get(7), remesh=build.REMESH.get(7),
                             props=build.NAMED_PROPS[7], cell_cut=build.CELL_CUT.get(7))
    c = build.converted_course(base, d, exe, 7, slot=7, scenery="model", scenery_data=sd)
    out = build.converted_set(base, disc, 7, slot=7, scenery="model")
    return base, sd, c, out


def test_converted_id7_round_trip(lake_id7, disc):  # noqa: F811
    """Mermaid Lake as course id 7, after National Park Speedway on the chain (as id 4's test):

    - the set decodes id 7 to the converted Course on every field, check_course passes, every table re-emits and
      encode(base, c) is the set converted_set builds; a standing start (ROLLING_STARTS[7]), grid base GRID_BASES[7],
      its 499 segments and three checkpoints; its road the Dreamcast's lifted by LIFT[7] ((x, y + 41, -z));
    - its own of every table it changes, in NEW_ID_RUNS (each past National Park Speedway's), its lists and HUD
      outline's list in the list home DATA_FREE, its own 1 MB course sheet block, the shared EMPTY_ACTORS; no
      reduction (QUAD, REMESH, THIN name no 7: the gate's lap is within the target unreduced);
    - ids 0-4 byte-identical to the input set's (decode, coverage, course blocks, the common block);
    - the free-bytes rule: every byte that differs from the set before is an id-7 per-id word, record 7, TGP slot 7
      or a byte that was 0xff in a free run (NEW_ID_RUNS and DATA_FREE)."""
    base, sd, c, out = lake_id7
    # M4 (fix round 1): converted_course refuses n 7 unlifted (build.LIFT: the tunnel's ground would answer the roof)
    with pytest.raises(ConvertError, match="LIFT"):
        build.converted_course(base, _data(disc, 7), disc.read(EXE_NAME), 7, slot=7, scenery="model",
                               scenery_data=sd)
    # final review, Minor 3: the lifted data with a scenery made from an unlifted model is refused too (Scenery.lift)
    assert sd.lift == build.LIFT[7]
    with pytest.raises(ConvertError, match="raised by 0.0"):
        build.converted_course(base, build.lifted_data(_data(disc, 7), build.LIFT[7]), disc.read(EXE_NAME), 7,
                               slot=7, scenery="model", scenery_data=dataclasses.replace(sd, lift=0.0))
    d = G.decode_course(out, 7)
    assert _differ(d, c) == []
    G.check_course(c)
    G.check_course(d, out)
    assert G.verify_lossless(out, 7) == [] and G.mismatches(out, c) == []
    assert build.encode(base, c, build.RACE_FIELDS + build.MODEL_FIELDS + build.ID_FIELDS).split_files() == \
        out.split_files()
    assert c.rolling_start[1] is False and c.grid_base[1] == build.GRID_BASES[7] == 493
    assert len(c.road) - 1 == 499 and [cp.segment for cp in c.checkpoints] == [120, 363, 0]
    assert c.actor_table == build.EMPTY_ACTORS and build.CONVERTED_IDS[:3] == (3, 4, 7)
    assert build.QUAD[7] == (10.0, 0.125) and 7 not in build.REMESH and 7 not in build.THIN   # v0.1.2: QUAD for CELL_CUT
    dc = _data(disc, 7)
    for r0, r1 in zip(dc.road, c.road):
        for j in range(r0.lane_lo, r0.lane_hi + 1):
            ln = r0.lanes[j]
            assert r1.lanes[j].position == (ln.x, float(np.float32(ln.y + build.LIFT[7])), -ln.z)
    runs, home = build.NEW_ID_RUNS, tgp.DATA_FREE
    c4 = G.decode_course(base, 4)

    def inside(off, size, run=runs["data"]):
        return run[0] <= off and off + size <= run[0] + run[1]

    e = c.entry
    for f in ("step_table", "record_table", "line_table", "block_ref_a", "block_ref_b"):
        assert inside(getattr(e, f), 4) and getattr(e, f) > getattr(c4.entry, f), f
    assert inside(c.grid - DATA_BASE, 4 * C.GRID_CELLS) and c.grid != c4.grid
    block = c.sheet_sources[0]
    assert block % 0x100000 == 0 and inside(block, 0x100000) and block not in (c4.sheet_sources[0],
                                                                             G.decode_course(base, 3).sheet_sources[0])
    assert T.sheet_source(out, 0, 7) == block
    assert 4 * c.ground_data.list_base > 4 * c4.ground_data.list_base >= runs["copro_data"][0]
    assert all(inside(4 * k, 4 * o.word_count, runs["polygons"]) for k, o in sd.objects.items())
    assert min(sd.objects) > max(c4.objects)
    assert all(inside(2 * k, 8, runs["textures"]) for k in sd.headers)
    assert all(inside(k - DATA_BASE, 16 * len(v) + 4, home) for k, v in sd.lists.items())
    assert inside(c.hud_map.outline_list - DATA_BASE, 20, home) and c.hud_map.outline_list != c4.hud_map.outline_list
    # ids 0-4 byte-identical
    for k in range(5):
        ref = G.decode_course(base, k)
        assert _differ(G.decode_course(out, k), ref) == [], k
        assert G.mismatches(out, ref) == [], k
        for region, off, size, _ in G.coverage(base, k):
            assert getattr(out, region)[off:off + size] == getattr(base, region)[off:off + size], (k, region, off)
        blk = T.sheet_source(base, 0, k)
        assert out.data[blk:blk + 0x100000] == base.data[blk:blk + 0x100000], k
    common = T.sheet_source(base, 1)
    assert out.data[common:common + 0x100000] == base.data[common:common + 0x100000]
    # the free-bytes rule
    own = set()
    for name in ("COURSE_POINTERS", "COURSE_SHEETS", "GRID_TABLES", "ACTOR_TABLES", "CHECKPOINT_TABLES",
                 "SEGMENT_FLAGS", "HUD_MAP_LISTS", "HUD_MAP_PLACES", "HUD_MAP_ZOOMS", "GRID_BASES", "ROLLING_STARTS",
                 "TIME_TABLES[0]", "TIME_TABLES[1]", "TIME_TABLES[2]"):
        a_ = C.ID_ARRAYS[name]
        own |= set(range(a_.new - DATA_BASE + 7 * a_.stride, a_.new - DATA_BASE + 8 * a_.stride))
    rec = C.record_address(7) - DATA_BASE
    own |= set(range(rec, rec + C.COURSE_ENTRY_SIZE))
    allowed = {"data": own, "copro_data": {4 * (w + 7) + k + h for w in (tgp.SLOT_LIST, tgp.SLOT_POOL)
                                           for k in range(4) for h in (0, HALF)}}
    free = {"data": [runs["data"], home], "copro_data": [runs["copro_data"]], "program": [],
            "polygons": [runs["polygons"]], "textures": [runs["textures"]]}
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a = np.frombuffer(getattr(base, region), np.uint8)
        b = np.frombuffer(getattr(out, region), np.uint8)
        idx = np.flatnonzero(a != b)
        ok = np.isin(idx, np.fromiter(allowed.get(region, ()), np.int64)) | (
            _runs_mask(len(a), free[region])[idx] & (a[idx] == 0xFF))
        assert ok.all(), (region, [hex(int(k)) for k in idx[~ok][:8]])
    used = {r: int((np.frombuffer(getattr(out, r), np.uint8)[_runs_mask(len(getattr(out, r)), [runs[r]])] != 0xFF).sum()
                   - (np.frombuffer(getattr(base, r), np.uint8)[_runs_mask(len(getattr(base, r)), [runs[r]])] != 0xFF)
                   .sum()) for r in runs}
    used["lists"] = int((np.frombuffer(out.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum()
                        - (np.frombuffer(base.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum())
    print(f"\nid 7's bytes per run (non-0xff bytes added): {used}")


def test_final_set_with_id7(disc):  # noqa: F811
    """The final chain with Mermaid Lake (tests/test_mirrors_mame.py `task4_set`: id 7 converted after National Park
    Speedway, before the mirrors of 4 and 7, the select with the frozen TASK4_ENTRIES, so a new live row of Phase 11
    cannot look like a byte regression; final review of Phase 10, Minor 2) against Task 3's chain (`task3_set`: its bytes are FINAL_SHA1, the independent
    reference, built by 0f0a36c's code with fix round 1's one deliberate change, NAMED_PROPS[4]): ids 0-4 decode to the
    same Course and every byte their tables cover is the same; the mirrors 8-11 draw and race the same
    (test_mirror._content) though their tables now lie past id 7's; id 7 holds its own course and entry 7 is live,
    entry 15 locked (12 too: the mirrors of 4 and 7 are Task 5's); the select's char upload is 127 chars in four
    runs; the program room stays 5,312 (id 7 shares EMPTY_ACTORS)."""
    from dtools.arcade import select as S
    from tests.test_mirror import _content, _eq
    from tests.test_mirrors_mame import TASK3_ENTRIES, TASK4_ENTRIES, task3_set, task4_set
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    new, old = task4_set(rs, disc), task3_set(rs, disc)
    assert {r: hashlib.sha1(getattr(old, r)).hexdigest() for r in FINAL_SHA1} == FINAL_SHA1      # the reference's
    assert build.program_room(new) == build.program_room(old) == 5056
    select = ("char_blocks", "screen_uploads")
    for k in range(5):
        assert _differ(G.decode_course(new, k), G.decode_course(old, k), skip=select) == [], k
        for region, off, n, name in G.coverage(old, k):
            if not name.startswith(("char block", "screen upload", "upload")):
                assert getattr(new, region)[off:off + n] == getattr(old, region)[off:off + n], (k, region, off, name)
    moved = 0
    for k in (8, 9, 10, 11):
        a, b = G.decode_course(new, k), G.decode_course(old, k)
        assert _eq(_content(a), _content(b)), k
        assert G.verify_lossless(new, k) == [], k
        moved += a.grid != b.grid
    assert moved == 4                                           # their tables moved past id 7's
    assert G.own_tables(new, 7) and not G.own_tables(old, 7)
    row = S.select_row(new)
    assert row.live[7] and not row.live[15] and not row.live[12] and not S.select_row(old).live[7]
    assert S.graphics(new, TASK4_ENTRIES).uploaded() == S.graphics(old, TASK3_ENTRIES).uploaded() == 127
    assert len(S.graphics(new, TASK4_ENTRIES).runs) == 4


# ---- Phase 10 Task 5: the mirrors of National Park Speedway (id 12) and Mermaid Lake (id 15) on the final set ----

FINAL_SHA1_T4 = {"program": "4f376c906defb9e64a006a6527fe6aaa6560132e",
                 "data": "e3bdbf50414cd010df34fd475c1df7227f3fd7d8",
                 "copro_data": "34229d4fbcbfb2efe041417bef8b7abb66466214",
                 "polygons": "18194128c4910eefa0adb7904edf62ea56bbf615",
                 "textures": "7ada6ea7c82d015053d900f294b0b7a178bc2f84"}
"""Task 4's final set (tests/test_mirrors_mame.py `task4_set`: Desert City at 3, National Park Speedway at 4, Mermaid
Lake at 7, the mirrors of 0-3, the rankings, the select with TASK4_ENTRIES), the sha1 of each region, built
independently of the code under test: c7a4855's code (git archive into a scratch folder, its own empty re-mesh cache,
343 s cold) building its own tests/test_mirrors_mame.py `final_set`. Phase 13 Task 2 round 2 repinned it: model._merge's
quad tests made exact (5b970ee; four National Park Speedway pairs on the shear bound, three of which the float lstsq had
rejected on this Mac, now merge), built by 5b970ee's code from a git archive (fresh venv, Python 3.15.0, numpy 2.5.4;
out/phase13/r2/pins.txt); linux/amd64 builds the same bytes. v0.1.1 repinned it (data, polygons, textures; program and copro_data unchanged): build.shadow_volumes
also takes an untextured closed mesh of a kind-3 object (CS4's 23 volumes in per-volume ID colours), built by the code
under test; the release pin, test_release.py RELEASE_SHA1, matches an independent build of v0.1's code with only that
rule patched in (out/debug/np-start/fixset.py). v0.1.2 repinned it (data, polygons, textures; program and copro_data unchanged): build.decals also drops a
texture nearly one colour (DECAL_SHARE, DECAL_STEP: Mermaid Lake's and Circuit Pixie's shadow 67), built by the code
under test; the release pin, test_release.py RELEASE_SHA1, matches an independent build of v0.1.1's code with only that
rule patched in (out/debug/lake/fixset.py). v0.1.2's lake fix repinned it: build.CELL_CUT cuts Mermaid Lake's long scenery polygons on the cell grid (its start straight was not drawn during the start's fly-by) and build.QUAD[7] merges its quads as on 4-6, so the lap stays in budget; only CS7's scenery changes (program too: the mirrors' tables placed after it move; copro_data unchanged)."""


def test_shipped_row_is_eight(disc):  # noqa: F811
    """Phase 11 Task 2 (replaces Phase 10's test_task5_changes_no_entry; M5A_ENTRIES == TASK4_ENTRIES now lives in
    tests/test_select.py test_frozen_rows_unchanged): the shipped row (select.M5B_ENTRIES) is the eight courses, mirror
    mode dropped (Pascal, 2026-11-11; reverse at M9): the originals, Desert City, National Park Speedway, Circuit Pixie
    (live since Task 4), Rin Rin Rink (since Task 3) and Mermaid Lake, all eight live, the last three each with an
    empty band (the plan's labels ruling: no label on the Dreamcast's select). On the shipped set
    (tests/test_courses_mame.py `m5b_set`: no mirror built) its graphics upload within UPLOAD_MAX / UPLOAD_RUNS_MAX,
    the select reads a row of 8, ids 8-15 hold no course of their own, and only ids 5 and 6 read view-test tables of
    their own (Task 4: build.VIEW, the stock tables cut to VIEW_12), every other id the stock ones."""
    from dtools.arcade import select as S
    from tests.test_courses_mame import m5b_set
    E = S.Entry
    assert S.M5B_ENTRIES == (
        E(0, "BEGINNER", "", False, True, "BEGINNER"),
        E(2, "ADVANCED", "", False, True, "ADVANCED"),
        E(1, "EXPERT", "", False, True, "EXPERT"),
        E(3, "DESERT CITY", "", False, True, "CIRCUIT EDITION"),
        E(4, "NATIONAL PARK SPEEDWAY", "", False, True, "CIRCUIT EDITION"),
        E(5, "CIRCUIT PIXIE", "", False, True),
        E(6, "RIN RIN RINK", "", False, True),
        E(7, "MERMAID LAKE", "", False, True))
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    full = m5b_set(rs, disc)
    g = S.graphics(full, S.M5B_ENTRIES)
    print(f"\nthe row of 8 uploads {g.uploaded()} chars in {len(g.runs)} runs")
    assert g.uploaded() <= S.UPLOAD_MAX and len(g.runs) <= S.UPLOAD_RUNS_MAX
    row = S.select_row(full)
    assert row.live == (True,) * 8 and [r.course for r in row.records[0]] == [0, 2, 1, 3, 4, 5, 6, 7]
    assert [k for k in range(3, C.COURSE_IDS) if G.own_tables(full, k)] == [3, 4, 5, 6, 7]
    stock = C.view_tables(full, 0)                  # the shared view-test tables (Beginner reads them)
    assert [k for k in range(C.COURSE_IDS) if C.view_tables(full, k) != stock] == [5, 6]
    assert all(b"".join(full.read_i960(p, C.VIEW_TEST_SIZE) for p in C.view_tables(full, k)) ==
               build.view_tables(full, build.VIEW[k]) for k in (5, 6))


def test_final_set_with_mirrors_12_15(disc):  # noqa: F811
    """The final chain with the mirrors of 4 and 7 (tests/test_mirrors_mame.py `task5_set`: mirrored_entry of 4 and 7
    after the mirrors of 0-3, before the rankings and the select; the select with TASK4_ENTRIES, frozen, so the check
    does not follow the live row: `final_set`'s bytes, tests/test_select.py test_frozen_rows_unchanged) against Task 4's
    final set (`task4_set`: its bytes are FINAL_SHA1_T4, the independent reference, c7a4855's code building its own
    final set): ids 0-4 and 7 and the mirrors 8-11 keep their content, every byte their tables cover the same (but the
    select's upload set) and their decoded Courses equal; ids 12 and 15 hold their own courses (golden.own_tables), each
    the mirror of its original (test_mirror._content); entries 12 and 15 stay locked (Pascal's 2026-11-11 ruling), the
    select's char upload unchanged (127 chars in four runs); the program room stays 5,312 (both mirrors share
    EMPTY_ACTORS); no texture byte differs (the mirrors share their originals' headers and points), and every byte that
    differs from Task 4's set lies in a free run where Task 4's set held 0xff (the mirrors' new tables), in ids 12 and
    15's own words (record, per-id entries, TGP slot words, TUNES / VISIT_IDS) or in the bound patch."""
    from dtools.arcade import mirror as M
    from dtools.arcade import select as S
    from tests.test_mirror import _content, _eq
    from tests.test_mirrors_mame import TASK4_ENTRIES, task4_set, task5_set
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    new, old = task5_set(rs, disc), task4_set(rs, disc)
    assert {r: hashlib.sha1(getattr(old, r)).hexdigest() for r in FINAL_SHA1_T4} == FINAL_SHA1_T4   # the reference's
    assert build.program_room(new) == build.program_room(old) == 5056
    select = ("char_blocks", "screen_uploads")
    for k in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11):
        assert _differ(G.decode_course(new, k), G.decode_course(old, k), skip=select) == [], k
        for region, off, n, name in G.coverage(old, k):
            if not name.startswith(("char block", "screen upload", "upload")):
                assert getattr(new, region)[off:off + n] == getattr(old, region)[off:off + n], (k, region, off, name)
        assert G.verify_lossless(new, k) == [], k
    for o in (4, 7):
        k = o + M.MIRROR_ID_OFFSET
        assert G.own_tables(new, k) and not G.own_tables(old, k), k
        a, b = _content(G.decode_course(new, k)), _content(M.mirror_course(G.decode_course(new, o)))
        assert [f for f in a if not _eq(a[f], b[f])] == [], k
        assert G.verify_lossless(new, k) == [], k
    row = S.select_row(new)
    assert [k for k in range(16) if row.live[k]] == [k for k, e in enumerate(TASK4_ENTRIES) if e.live] == \
        [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    g, g_old = S.graphics(new, TASK4_ENTRIES), S.graphics(old, TASK4_ENTRIES)
    assert g.uploaded() == g_old.uploaded() == 127 and len(g.runs) == 4 and g.runs == g_old.runs
    assert _eq(g.chars, g_old.chars) and _eq(g.palettes, g_old.palettes)
    # where the two sets differ
    differ = {r: int((np.frombuffer(getattr(new, r), np.uint8) != np.frombuffer(getattr(old, r), np.uint8)).sum())
              for r in ("program", "data", "copro_data", "polygons", "textures")}
    print(f"\nbytes that differ from Task 4's final set: {differ}")
    assert differ["textures"] == 0                              # the mirrors share their originals' headers and points
    from tests.test_mirror import _free
    own = [w for k in (12, 15) for w in M._own_words(old, k)]
    own += [(r, lo + COPRO_SIZE // 2, n) for r, lo, n in own if r == "copro_data"]   # the mirrored 2 MB half
    own += [("program", at, 4) for at in C.LIBRARY_BOUND_AT]
    for region in ("program", "data", "copro_data", "polygons"):
        x, y = (np.frombuffer(getattr(s_, region), np.uint8) for s_ in (old, new))
        changed = np.flatnonzero(x != y)
        inside = np.zeros(len(changed), bool)
        for lo, n in _free(old, region):
            inside |= (changed >= lo) & (changed < lo + n) & (x[changed] == 0xFF)
        for r, lo, n in own:
            if r == region:
                inside |= (changed >= lo) & (changed < lo + n)
        assert inside.all(), (region, [hex(v) for v in changed[~inside][:8]])


# ---- Phase 11 Task 3: Rin Rin Rink (n 6) at id 6 (discovery log docs/spike/2026-11-courses-4-7.md, "Rin Rin Rink") --

RINK_OUTSIDE = {158: 12, 159: 4}
"""CS6's objects with vertices past the world (model.WORLD, x 1023) and how many: to x 1073.1 and 1063.3. Every other
vertex, the road, the AI lines and the ground lie inside, and every z."""
RINK_X = {"model": (-928.9, 1073.1), "road": (-651.7, 636.8), "ai": (-645.2, 624.8), "cells": (-735.9, 657.9)}
"""CS6's x extents (to 0.1) as the disc stores them: the model's vertices, the road's used lanes, the AI points, the
ground cells' corners. Any shift in [-95.1, -49.1] fits them all inside the world; build.SHIFT[6] is -64.0."""
RINK_SHIFTED_X = (-992.9, 1009.1)
"""The shifted model's x extent (to 0.1)."""
RINK_NEAR_ORIGIN = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 51, 52, 83, 84, 85, 118)
"""CS6's objects with every vertex within build.PROP_RADIUS of the Dreamcast's origin in x and z: the origin rule's
candidates. The road passes the origin, so the road guard keeps every one (none is dropped as a prop)."""
RINK_NUDGED = ({643}, {644, 1133})
"""The ground cells the converter's "ulps" repair (ground.NUDGE_ULPS) nudges on the plain data only, and on the
shifted data only: rounding x + SHIFT[6] to single precision moves a near-collinear corner by part of an ulp, so the
sub-ulp notch is there on one and not the other. All three off the road; the other repairs (ground.REPAIRED[6]) are
the same."""
SHIFT_ULP = 2.0 ** -14
"""One f32 ulp at 512-1024 units, the largest |x| of the shifted course: the bound on a value the conversion computes
from the shifted positions (a road centre, an outline edge) against the plain one's plus SHIFT[6]."""


def _model_points(m) -> dict[int, np.ndarray]:
    """Per object of `m` with a vertex: its vertices (x, y, z) as stored."""
    out = {}
    for oi, ob in enumerate(m.objects):
        ps = [(e.x, e.y, e.z) for me in ob.meshes for st in me.strips for e in st.entries if isinstance(e, pol.Vertex)]
        if ps:
            out[oi] = np.array(ps, dtype=float)
    return out


def _cut(m, objects):
    """`m` with the meshes of `objects` removed (the plain model's drop rules then run inside the world)."""
    return dataclasses.replace(m, objects=tuple(dataclasses.replace(ob, meshes=()) if k in objects else ob
                                                for k, ob in enumerate(m.objects)))


def test_cs6_inside_world_shifted(disc):  # noqa: F811
    """build.SHIFT[6] (discovery log, "Rin Rin Rink"): CS6's model reaches x 1073.1, past the world (model.WORLD,
    -1024..1023, where the scenery cell rule and the TGP grid do not wrap): objects 158 and 159 (RINK_OUTSIDE) of 169;
    the road, the AI lines, the ground and every z lie inside. Unshifted the scenery refuses the model, naming the two
    objects and SHIFT[6]. Shifted by -64.0 every model vertex, used lane, AI point, cell corner and centre lies inside.

    shifted_data / shifted_model move every x (the used lanes, the AI points, the cells' corners and centres; the
    model's vertices and bounding spheres) by `dx`, as f32, and nothing else; past single precision they refuse
    without a numpy warning; each marks what it made (`_shift_of`, not a field), and the lift keeps the mark.

    The drop rules give the same objects on the shifted course as on the plain one (its two outside objects cut so
    the rules can run): decals, domes, shadow volumes and the origin rule, which measures from the Dreamcast's origin,
    moved by the shift (a rebuilt model, without the mark, would look at the arcade's origin and miss every one of
    RINK_NEAR_ORIGIN); with the road guard none of them is a prop, plain or shifted."""
    import warnings
    from dtools.dc.disc import course_files
    d, mn, tn = _data(disc, 6), course_files(6, kind="model"), course_files(6, kind="textures")
    m, blob = pol.decode_pol(disc.read(mn), mn), disc.read(tn)
    assert build.SHIFT == {6: -64.0} and 6 in build.CONVERTED_IDS and 6 not in build.LIFT
    dx, (lo, hi) = build.SHIFT[6], model.WORLD
    f32 = lambda v: float(np.float32(v))                                                     # noqa: E731

    def inside(x, z) -> bool:
        return lo <= x <= hi and lo <= -z <= hi

    def extents(dd) -> dict:
        lanes = [ln for rp in dd.road for j, ln in enumerate(rp.lanes) if rp.lane_lo <= j <= rp.lane_hi]
        return {"road": [(ln.x, ln.z) for ln in lanes], "ai": [(q.x, q.z) for line in dd.lines for q in line],
                "cells": [(c[0], c[2]) for cell in dd.cells for c in cell.corners] +
                         [(cell.centre[0], cell.centre[2]) for cell in dd.cells]}
    # unshifted: two objects past the world, nothing else
    pts = _model_points(m)
    every = np.concatenate(list(pts.values()))
    assert (round(every[:, 0].min(), 1), round(every[:, 0].max(), 1)) == RINK_X["model"]
    assert {oi: int((~((p[:, 0] >= lo) & (p[:, 0] <= hi) & (-p[:, 2] >= lo) & (-p[:, 2] <= hi))).sum())
            for oi, p in pts.items() if not all(inside(x, z) for x, _, z in p)} == RINK_OUTSIDE
    for k, v in extents(d).items():
        xs = [x for x, _ in v[:len(v) if k != "cells" else 4 * len(d.cells)]]
        assert (round(min(xs), 1), round(max(xs), 1)) == RINK_X[k], k
        assert all(inside(x, z) for x, z in v), k
    with pytest.raises(ConvertError, match=r"objects 158, 159 .*SHIFT\[6\]"):
        build.scenery_model(m, blob, d)
    # shifted: every position inside
    ms, ds = build.shifted_model(m, dx), build.shifted_data(d, dx)
    every = np.concatenate(list(_model_points(ms).values()))
    assert (round(every[:, 0].min(), 1), round(every[:, 0].max(), 1)) == RINK_SHIFTED_X
    assert all(inside(x, z) for x, _, z in every)
    model.triangle_meshes(ms)                       # the converter's own world check passes
    assert all(inside(x, z) for v in extents(ds).values() for x, z in v)
    # every x moved by dx as f32, nothing else
    for r0, r1 in zip(d.road, ds.road):
        for j, (l0, l1) in enumerate(zip(r0.lanes, r1.lanes)):
            assert l1 == (dataclasses.replace(l0, x=f32(l0.x + dx)) if r0.lane_lo <= j <= r0.lane_hi else l0)
        assert (r1.lane_lo, r1.lane_hi, r1.curve) == (r0.lane_lo, r0.lane_hi, r0.curve)
    assert all(q == dataclasses.replace(p_, x=f32(p_.x + dx)) for a_, b_ in zip(d.lines, ds.lines)
               for p_, q in zip(a_, b_))
    assert all(c1.corners == tuple((f32(x + dx), y, z) for x, y, z in c0.corners) and c1.centre ==
               (f32(c0.centre[0] + dx),) + tuple(c0.centre[1:]) and c1.direction == c0.direction
               for c0, c1 in zip(d.cells, ds.cells))
    assert (ds.grid, ds.lists, ds.steps, ds.attributes, ds.order, ds.line_slots) == \
        (d.grid, d.lists, d.steps, d.attributes, d.order, d.line_slots)
    for o0, o1 in zip(m.objects, ms.objects):
        assert o1.sphere == (f32(o0.sphere[0] + dx),) + tuple(o0.sphere[1:])
        for e0, e1 in zip((e for me in o0.meshes for st in me.strips for e in st.entries),
                          (e for me in o1.meshes for st in me.strips for e in st.entries)):
            assert e1 == (dataclasses.replace(e0, x=f32(e0.x + dx)) if isinstance(e0, pol.Vertex) else e0)
    assert (ms.header, ms.textures) == (m.header, m.textures)
    assert build.shifted_data(d, 0.0) == d and build.shifted_model(m, 0.0) == m
    # the marks: not fields (eq and repr unchanged, so the re-mesh cache key is too), kept by the lift
    assert build._shift_of(m) == build._shift_of(d) == 0.0 and build._shift_of(ms) == build._shift_of(ds) == dx
    assert build._shift_of(build.shifted_model(ms, 1.0)) == build._shift_of(build.shifted_data(ds, 1.0)) == dx + 1
    up_m, up_d = build.lifted_model(ms, 1.0), build.lifted_data(ds, 1.0)
    assert build._shift_of(up_m) == build._shift_of(up_d) == dx and build._lift_of(up_m) == 1.0
    assert build._lift_of(build.shifted_model(up_m, 0.0)) == 1.0
    assert repr(build.shifted_model(m, 0.0)) == repr(m)
    for bad in ((None, 1.0), (d, math.inf), (d, True), (d, "1"), (d, 1e39)):
        with pytest.raises(ConvertError):
            build.shifted_data(*bad)
    for bad in ((None, 1.0), (m, math.nan), (m, True), (m, "1"), (m, 1e39)):
        with pytest.raises(ConvertError):
            build.shifted_model(*bad)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConvertError, match="single precision"):
            build.shifted_data(d, 1e39)
        with pytest.raises(ConvertError, match="single precision"):
            build.shifted_model(m, 1e39)
    # the drop rules: the same objects shifted as plain; the origin rule measures from the Dreamcast's origin
    mc = _cut(m, RINK_OUTSIDE)
    assert build.decals(ms, blob) == build.decals(mc, blob) and build.sky_domes(ms) == build.sky_domes(mc) == (0, 1)
    assert build.shadow_volumes(ms) == build.shadow_volumes(mc)
    assert build.origin_props(ms) == build.origin_props(mc) == RINK_NEAR_ORIGIN
    assert build.origin_props(dataclasses.replace(ms)) == ()        # without the mark: the arcade's origin, all missed
    assert build.origin_props(ms, ds) == build.origin_props(mc, d) == ()
    k0, k1 = build.kept_model(mc, blob, d), build.kept_model(_cut(ms, RINK_OUTSIDE), blob, ds)
    assert [len(ob.meshes) for ob in k1.objects] == [len(ob.meshes) for ob in k0.objects]
    assert build.road_triangles(k1, ds) == build.road_triangles(k0, d)


def test_shift_moves_every_position(disc):  # noqa: F811
    """Review Focus 4 of Phase 11: every positional value of Rin Rin Rink's conversion is the unshifted data's
    conversion moved by SHIFT[6] in x, single precision (the ground and the road convert unshifted; the model does
    not), or is position-free:

    - the road line (the record table) and the AI lines: every used lane's and AI point's x is the plain one's + dx,
      rounded to f32, exactly; every other field the plain one's;
    - the ground: the attributes the same; the grid and lists the plain ones four squares (64 units) to the left; every
      polygon's corners and centre the plain one's + dx exactly but the cells the "ulps" repair nudges on one side only
      (RINK_NUDGED: within ground.NUDGE_ULPS ulps a coordinate, off the road); every used lane and AI point answers the
      same polygon at the same height within LIFT_HEIGHT_TOL (Task 2's tolerance);
    - the checkpoints: segments in 1ST_READ.BIN (one, 263, then the lap line), position-free; the views the game
      computes from the road line move with it (within SHIFT_ULP);
    - the start grid: the spawn records and the grid base (GRID_BASES[6], a segment) are position-free; each car's
      segment and lane are the same, its position (a view of the road line) moved by dx within SHIFT_ULP;
    - the HUD map: the outline keeps the same road points, its edges move by dx (x and z within SHIFT_ULP); the
      frame (hud_frame) keeps the angle and the zooms and moves the place record so that every outline point and road
      centre lands where the plain one's lands (within a millionth of the map's width): the player's dot is the car's
      world position under the same matrix (docs/formats/model2-course-tables.md, "HUD map"), so it lands with them;
    - the fly-by camera: id 6 plays Beginner's script, every keyframe in car 0's frame (none CAMERA_ABSOLUTE):
      position-free;
    - the tunnel light zones (world positions, per id): read only while a car's SEGMENT_TUNNEL bit is set, which no
      step of CS6 and no segment flag sets: never read on id 6."""
    import struct
    from dtools.convert import ground, tables
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    base = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    b = G.decode_course(base, 6)
    d, dx = _data(disc, 6), build.SHIFT[6]
    ds = build.shifted_data(d, dx)
    f32 = lambda v: float(np.float32(v))                                                     # noqa: E731
    # the road line and the AI lines: exact
    r0, r1 = tables.road_line(d), tables.road_line(ds)
    for p0, p1 in zip(r0, r1):
        assert (p1.lane_lo, p1.lane_hi, p1.curve) == (p0.lane_lo, p0.lane_hi, p0.curve)
        for j, (l0, l1) in enumerate(zip(p0.lanes, p1.lanes)):
            used = p0.lane_lo <= j <= p0.lane_hi
            assert l1 == (l0._replace(position=(f32(l0.position[0] + dx),) + l0.position[1:]) if used else l0)
    for a_, b_ in zip(tables.ai_lines(d), tables.ai_lines(ds)):
        assert [q._replace(position=(f32(q.position[0] + dx),) + q.position[1:]) for q in a_] == list(b_)
    # the ground
    g0, g1 = ground.convert_ground(d, 6), ground.convert_ground(ds, 6)
    assert g1.attributes == g0.attributes and len(g1.polys) == len(g0.polys)
    for sq in range(tgp.GRID_CELLS):
        row, col = divmod(sq, 128)
        a = g0.lists.get(g0.grid[sq]) if g0.grid[sq] else None
        assert a is None or col >= 4, sq
        if col >= 4:
            assert a == (g1.lists.get(g1.grid[sq - 4]) if g1.grid[sq - 4] else None), sq
    r_plain, r_shift = ground.repairs(d), ground.repairs(ds)
    assert r_plain == ground.REPAIRED[6]
    only = ({k for k in r_plain if k not in r_shift}, {k for k in r_shift if k not in r_plain})
    assert only == RINK_NUDGED and all(r_shift[k] == v for k, v in r_plain.items() if k in r_shift)
    moved = set()
    for k, (p0, p1) in enumerate(zip(g0.polys, g1.polys)):
        assert p1.direction == p0.direction and p1.centre == (f32(p0.centre[0] + dx),) + tuple(p0.centre[1:]), k
        want = tuple((f32(x + dx), y, z) for x, y, z in p0.corners)
        if p1.corners != want:
            moved.add(k)
            for c1, c0 in zip(p1.corners, want):
                for v1, v0 in zip(c1, c0):
                    assert abs(v1 - v0) <= ground.NUDGE_ULPS * abs(float(np.spacing(np.float32(v0)))), (k, c1, c0)
    assert moved == RINK_NUDGED[0] | RINK_NUDGED[1] and min(moved) >= len(d.steps)        # off the road
    worst, points = 0.0, 0
    pairs = [(rp.lanes[j], rq.lanes[j]) for rp, rq in zip(d.road, ds.road) for j in range(rp.lane_lo, rp.lane_hi + 1)]
    pairs += [(q, qs) for line, lines in zip(d.lines, ds.lines) for q, qs in zip(line, lines)]
    for p_, q in pairs:
        h0, h1 = tgp.query(g0, p_.x, p_.y + 0.5, -p_.z), tgp.query(g1, q.x, q.y + 0.5, -q.z)
        assert h0 is not None and h1 is not None and h1[1] == h0[1] and abs(h1[0] - q.y) <= 2, (p_, h0, h1)
        worst, points = max(worst, abs(h1[0] - h0[0])), points + 1
    print(f"\nshifted ground against plain: {points} lane and AI points, the same polygons, worst {worst:.2e}")
    assert points == 7000 and worst <= LIFT_HEIGHT_TOL
    # the checkpoints
    exe = disc.read(EXE_NAME)
    rows = tables.checkpoint_rows(exe, 6)
    assert [(r.segment, r.raw) for r in rows] == [(263, 0)]
    words = list(range(1, 64))
    v0, v1 = (C.checkpoint_views([(r.segment, r.raw) for r in rows], list(r), words, 3) for r in (r0, r1))
    assert [(v.segment, v.raw, v.lap_line) for v in v1] == [(v.segment, v.raw, v.lap_line) for v in v0]
    assert C.encode_checkpoint_table(v1) == C.encode_checkpoint_table(v0)
    assert all(abs(q.position[0] - p_.position[0] - dx) <= SHIFT_ULP and q.position[1:] == p_.position[1:]
               for p_, q in zip(v0, v1))
    # the start grid
    assert build.GRID_BASES[6] == 493 and build.ROLLING_STARTS[6] is False
    for slots in b.spawn:
        for s in slots:
            g_0, g_1 = (C.grid_slot(s.car, s.objects, s.line, s.start, s.speed, s.kind, list(r), 493,
                                    tuple(b.grid_lanes), False) for r in (r0, r1))
            assert g_1[:-1] == g_0[:-1] and g_1.position[1:] == g_0.position[1:]
            assert abs(g_1.position[0] - g_0.position[0] - dx) <= SHIFT_ULP
    # the HUD map
    o0, o1 = build.hud_outline(r0), build.hud_outline(r1)
    c0, c1 = build.road_centres(r0, "plain"), build.road_centres(r1, "shifted")
    assert build.outline_points(c0)[0] == build.outline_points(c1)[0]
    assert [(r.attr, r.normal) for r in o1.records] == [(r.attr, r.normal) for r in o0.records]
    e0, e1 = ([o.seed0, o.seed1] + [p for r in o.records for p in (r.p0, r.p1)] for o in (o0, o1))
    for p_, q in zip(e0, e1):
        assert abs(q[0] - p_[0] - dx) <= SHIFT_ULP and q[1] == p_[1] and abs(q[2] - p_[2]) <= SHIFT_ULP, (p_, q)
    (pos0, ang0, zoom0), (pos1, ang1, zoom1) = (build.hud_frame(b.hud_map, b.road, r) for r in (r0, r1))
    assert (ang1, zoom1, pos1[2]) == (ang0, zoom0, pos0[2])
    cos, sin = build._turn(ang0)
    k = zoom0[build.HUD_LEVEL]

    def screen(x, z, pos):
        return k * (x * cos - z * sin) + pos[0], k * (x * sin + z * cos) + pos[1]
    width = max(build._box(c0, ang0)[:2]) * k
    gap = max(max(abs(u - v) for u, v in zip(screen(p_[0], p_[2], pos0), screen(q[0], q[2], pos1)))
              for p_, q in zip(e0, e1))
    gap = max(gap, max(max(abs(u - v) for u, v in zip(screen(*p_, pos0), screen(*q, pos1))) for p_, q in zip(c0, c1)))
    print(f"the HUD map: outline and road centres on screen within {gap:.2e} of the plain ({gap / width:.1e} of the "
          f"map's width)")
    assert gap <= 1e-6 * width
    # the fly-by camera and the tunnel light zones
    arrays = C.id_tables(base)
    script = C.camera_script(base, struct.unpack("<I", base.read_i960(C.id_entry(arrays, "CAMERA_SCRIPTS", 6), 4))[0])
    assert script.keyframes and all(kf.position_car != C.CAMERA_ABSOLUTE and kf.target_car != C.CAMERA_ABSOLUTE
                                    for kf in script.keyframes)
    assert not any(s.flags & C.SEGMENT_TUNNEL for s in d.steps) and set(tables.segment_flags(ds)) == {0}


RINK_WIDE = {(124, 3): (111, 1), (158, 19): (169, 2), (159, 16): (169, 1), (160, 26): (169, 2)}
"""CS6's meshes with polygons whose texture points pass the u16 range on their converted texture, (mesh) -> (texture,
records): long thin strips (stripes and markings, 0.5-1 unit across and 130-712 units long) repeating texture 169
(32 x 32) 300 times or texture 111 (64 x 64) 140 times; model.convert_model cuts them (Phase 11 Task 3, the
controller's ruling: no rescaling, no dropping)."""


def test_cs6_wide_texture_points(disc):  # noqa: F811
    """The cut on CS6 (model.convert_model's `points`, the converted headers' sizes as model_scenery passes them; the
    shifted kept model): without it the six records of RINK_WIDE do not fit a texture point (textures.polygon_points
    refuses them); with it every record of every piece fits; the polygons of those meshes are cut (more records,
    their areas adding up to the same, every one of their triangles still drawn), and every other drawn polygon is
    the same as without the cut."""
    from dtools.convert import textures
    from dtools.dc.disc import course_files
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    dx = build.SHIFT[6]
    d, mn, tn = build.shifted_data(_data(disc, 6), dx), course_files(6, kind="model"), course_files(6, kind="textures")
    m, blob = build.shifted_model(pol.decode_pol(disc.read(mn), mn), dx), disc.read(tn)
    kept = build.kept_model(m, blob, d)
    where = model.triangle_meshes(kept)
    base, _ = textures.course_textures(m, blob, rs)
    sizes = {t: (h.width, h.height) for t, h in base.items()}

    def drawn(pieces):
        out = []
        for ps in pieces.values():
            for pc in ps:
                for poly, t, uv, src in zip(P.polygons(pc.obj), pc.textures, pc.uvs, pc.sources):
                    if src:
                        out.append((where[src[0]], t, uv, src, tuple(map(tuple, poly.points))))
        return out

    def fits(t, uv) -> bool:
        try:
            textures.polygon_points(uv, *sizes[t])
            return True
        except ConvertError:
            return False
    plain, cut = drawn(model.convert_model(kept)), drawn(model.convert_model(kept, points=sizes))
    wide = {}
    for key, t, uv, _, _ in plain:
        if t >= 0 and not fits(t, uv):
            wide[key] = (t, wide.get(key, (t, 0))[1] + 1)
    assert wide == RINK_WIDE
    assert all(t < 0 or fits(t, uv) for _, t, uv, _, _ in cut)
    named = set(RINK_WIDE)

    def same(rows):                                 # each polygon from its least corner (chains rotate them)
        out = []
        for k, t, uv, src, pts in rows:
            if k not in named:
                j = min(range(len(pts)), key=lambda i: (pts[i], uv[i]))
                out.append((k, t, src, pts[j:] + pts[:j], uv[j:] + uv[:j]))
        return sorted(out)
    assert same(cut) == same(plain)
    for key in named:
        a, b = [r for r in plain if r[0] == key], [r for r in cut if r[0] == key]
        area = sum(float(np.linalg.norm(model._newell(r[4]))) / 2 for r in a)
        assert len(b) > len(a) and math.isclose(sum(float(np.linalg.norm(model._newell(r[4]))) / 2 for r in b),
                                                area, rel_tol=1e-6), key
        assert {i for r in b for i in r[3]} == {i for r in a for i in r[3]}, key
    print(f"\nCS6's wide meshes: {sum(r[0] in named for r in plain)} records -> {sum(r[0] in named for r in cut)}")


RINK_BANNER = tuple(range(19, 51))
"""CS6's animated advertising banner: 32 objects of 70 vertices at one place (texture 117, the sponsor sheet), the
frames of one animation (Pascal's decision, step 1: the banner as one still frame, object 19)."""
RINK_COPIES = (2, 3, 4)
"""Three copies of one prop modelled about the Dreamcast's origin (193 vertices each, identical positions and u, v)."""


def test_cs6_named_props(disc):  # noqa: F811
    """build.NAMED_PROPS[6] (Phase 11 Task 3, Pascal's decision 2026-11-12, step 1): the banner's frames after the first
    (20-50: every frame 70 vertices, texture 117, the same u, v as frame 19, the positions within 4.6 units of it, all
    kind 1) and the three copies of the origin prop (2-4: identical positions and u, v, every vertex within
    PROP_RADIUS of the origin, kept by the road guard since the road passes the origin) are run-time props:
    kept_model drops them and keeps frame 19 and everything else of the shifted model."""
    from dtools.dc.disc import course_files
    d, mn, tn = _data(disc, 6), course_files(6, kind="model"), course_files(6, kind="textures")
    m, blob = pol.decode_pol(disc.read(mn), mn), disc.read(tn)
    assert build.NAMED_PROPS[6] == RINK_COPIES + RINK_BANNER[1:]

    def ents(o):
        return [e for me in m.objects[o].meshes for st in me.strips for e in st.entries if isinstance(e, pol.Vertex)]
    first = ents(19)
    for o in RINK_BANNER:
        es = ents(o)
        assert len(es) == 70 and {me.texture for me in m.objects[o].meshes} == {117} and m.objects[o].kind == 1, o
        assert [(e.u, e.v) for e in es] == [(e.u, e.v) for e in first], o
        assert max(abs(a - b) for e, f in zip(es, first) for a, b in zip((e.x, e.y, e.z), (f.x, f.y, f.z))) < 4.6, o
    assert all([(e.x, e.y, e.z, e.u, e.v) for e in ents(o)] == [(e.x, e.y, e.z, e.u, e.v) for e in ents(2)]
               for o in RINK_COPIES)
    assert all(math.hypot(e.x, e.z) <= build.PROP_RADIUS for e in ents(2))
    dx = build.SHIFT[6]
    ms, ds = build.shifted_model(m, dx), build.shifted_data(d, dx)
    assert build.origin_props(ms) == RINK_NEAR_ORIGIN and build.origin_props(ms, ds) == ()     # the road guard
    k0, k1 = build.kept_model(ms, blob, ds), build.kept_model(ms, blob, ds, props=build.NAMED_PROPS[6])
    gone = set(build.NAMED_PROPS[6])
    assert all(len(k1.objects[o].meshes) == 0 for o in gone) and len(k1.objects[19].meshes) == 3
    assert [ob for k, ob in enumerate(k1.objects) if k not in gone] == [ob for k, ob in enumerate(k0.objects)
                                                                       if k not in gone]


@pytest.fixture(scope="module")
def rink_id6(disc):  # noqa: F811
    """n 6 (Rin Rin Rink) with its own scenery at course id 6 of the shipped chain after Mermaid Lake (grow_ids,
    guard_new_ids, widen_ids, Desert City at id 3, National Park Speedway at id 4, Mermaid Lake at id 7), shifted by
    build.SHIFT[6]: (set before, scenery, course, set after). The set is converted_set(base, disc, 6, slot=6,
    scenery="model")."""
    from dtools.dc.disc import course_files
    from tests.test_mirrors_mame import _converted
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    base = _converted(rs, disc, (3, 4, 7))
    dx = build.SHIFT[6]
    d, exe = build.shifted_data(_data(disc, 6), dx), disc.read(EXE_NAME)
    mn, tn = course_files(6, kind="model"), course_files(6, kind="textures")
    sd = build.model_scenery(base, d, build.shifted_model(pol.decode_pol(disc.read(mn), mn), dx), disc.read(tn),
                             build.THIN.get(6), slot=6, quad=build.QUAD.get(6), remesh=build.REMESH.get(6),
                             props=build.NAMED_PROPS.get(6, ()))
    c = build.converted_course(base, d, exe, 6, slot=6, scenery="model", scenery_data=sd)
    out = build.converted_set(base, disc, 6, slot=6, scenery="model")
    return base, sd, c, out


def test_converted_id6_round_trip(rink_id6, disc):  # noqa: F811
    """Rin Rin Rink as course id 6, after Mermaid Lake on the shipped chain (as id 7's test):

    - converted_course refuses n 6 unshifted (its data not shifted by SHIFT[6], or its scenery made from a model not
      shifted by it: Scenery.shift), naming SHIFT[6];
    - the set decodes id 6 to the converted Course on every field, check_course passes, every table re-emits and
      encode(base, c) is the set converted_set builds (tgp.encode_ground accepts its shifted ground); a standing start
      (ROLLING_STARTS[6]), grid base GRID_BASES[6], its 499 segments and one checkpoint (263) before the lap line; its
      road the Dreamcast's shifted by SHIFT[6] ((x - 64, y, -z), f32); every used lane and AI point answers its own
      ground polygon (within 2 units of its height);
    - its own of every table it changes, in NEW_ID_RUNS (each past Mermaid Lake's), its lists and HUD outline's list
      in the list home DATA_FREE, its own 1 MB course sheet block, the shared EMPTY_ACTORS; the scenery's shift
      SHIFT[6], its lift 0;
    - ids 0-4 and 7 byte-identical to the input set's (decode, coverage, course blocks, the common block);
    - the free-bytes rule: every byte that differs from the set before is an id-6 per-id word, record 6, TGP slot 6
      or a byte that was 0xff in a free run (NEW_ID_RUNS and DATA_FREE)."""
    base, sd, c, out = rink_id6
    dx = build.SHIFT[6]
    with pytest.raises(ConvertError, match=r"SHIFT\[6\]"):
        build.converted_course(base, _data(disc, 6), disc.read(EXE_NAME), 6, slot=6, scenery="model", scenery_data=sd)
    assert sd.shift == dx and sd.lift == 0.0
    with pytest.raises(ConvertError, match=r"shifted by 0\.0.*SHIFT\[6\]"):
        build.converted_course(base, build.shifted_data(_data(disc, 6), dx), disc.read(EXE_NAME), 6, slot=6,
                               scenery="model", scenery_data=dataclasses.replace(sd, shift=0.0))
    with pytest.raises(ConvertError, match="SHIFT"):        # a course SHIFT does not name is not shifted
        build.converted_course(base, build.shifted_data(_data(disc, 3), dx), disc.read(EXE_NAME), 3, slot=3)
    d = G.decode_course(out, 6)
    assert _differ(d, c) == []
    G.check_course(c)
    G.check_course(d, out)
    assert G.verify_lossless(out, 6) == [] and G.mismatches(out, c) == []
    assert build.encode(base, c, build.RACE_FIELDS + build.MODEL_FIELDS + build.ID_FIELDS).split_files() == \
        out.split_files()
    assert c.rolling_start[1] is False and c.grid_base[1] == build.GRID_BASES[6] == 493
    assert len(c.road) - 1 == 499 and [cp.segment for cp in c.checkpoints] == [263, 0]
    assert c.actor_table == build.EMPTY_ACTORS and build.CONVERTED_IDS[:4] == (3, 4, 7, 6)
    dc = _data(disc, 6)
    for r0, r1 in zip(dc.road, c.road):
        for j in range(r0.lane_lo, r0.lane_hi + 1):
            ln = r0.lanes[j]
            assert r1.lanes[j].position == (float(np.float32(ln.x + dx)), ln.y, -ln.z)
    points = [ln.position for rp in c.road for ln in rp.lanes[rp.lane_lo:rp.lane_hi + 1]]
    points += [q.position for line in c.ai_lines for q in line]
    for x, y, z in points:
        h = tgp.query(c.ground_data, x, y + 0.5, z)
        assert h is not None and abs(h[0] - y) <= 2, (x, y, z, h)
    runs, home = build.NEW_ID_RUNS, tgp.DATA_FREE
    c7 = G.decode_course(base, 7)

    def inside(off, size, run=runs["data"]):
        return run[0] <= off and off + size <= run[0] + run[1]

    e = c.entry
    for f in ("step_table", "record_table", "line_table", "block_ref_a", "block_ref_b"):
        assert inside(getattr(e, f), 4) and getattr(e, f) > getattr(c7.entry, f), f
    assert inside(c.grid - DATA_BASE, 4 * C.GRID_CELLS) and c.grid != c7.grid
    block = c.sheet_sources[0]
    assert block % 0x100000 == 0 and inside(block, 0x100000)
    assert block not in {G.decode_course(base, k).sheet_sources[0] for k in (3, 4, 7)}
    assert T.sheet_source(out, 0, 6) == block
    assert 4 * c.ground_data.list_base > 4 * c7.ground_data.list_base >= runs["copro_data"][0]
    assert all(inside(4 * k, 4 * o.word_count, runs["polygons"]) for k, o in sd.objects.items())
    assert min(sd.objects) > max(c7.objects)
    assert all(inside(2 * k, 8, runs["textures"]) for k in sd.headers)
    assert all(inside(k - DATA_BASE, 16 * len(v) + 4, home) for k, v in sd.lists.items())
    assert inside(c.hud_map.outline_list - DATA_BASE, 20, home) and c.hud_map.outline_list != c7.hud_map.outline_list
    # ids 0-4 and 7 byte-identical
    for k in (0, 1, 2, 3, 4, 7):
        ref = G.decode_course(base, k)
        assert _differ(G.decode_course(out, k), ref) == [], k
        assert G.mismatches(out, ref) == [], k
        for region, off, size, _ in G.coverage(base, k):
            assert getattr(out, region)[off:off + size] == getattr(base, region)[off:off + size], (k, region, off)
        blk = T.sheet_source(base, 0, k)
        assert out.data[blk:blk + 0x100000] == base.data[blk:blk + 0x100000], k
    common = T.sheet_source(base, 1)
    assert out.data[common:common + 0x100000] == base.data[common:common + 0x100000]
    # the free-bytes rule
    own = set()
    for name in ("COURSE_POINTERS", "COURSE_SHEETS", "GRID_TABLES", "ACTOR_TABLES", "CHECKPOINT_TABLES",
                 "SEGMENT_FLAGS", "HUD_MAP_LISTS", "HUD_MAP_PLACES", "HUD_MAP_ZOOMS", "GRID_BASES", "ROLLING_STARTS",
                 "TIME_TABLES[0]", "TIME_TABLES[1]", "TIME_TABLES[2]"):
        a_ = C.ID_ARRAYS[name]
        own |= set(range(a_.new - DATA_BASE + 6 * a_.stride, a_.new - DATA_BASE + 7 * a_.stride))
    rec = C.record_address(6) - DATA_BASE
    own |= set(range(rec, rec + C.COURSE_ENTRY_SIZE))
    allowed = {"data": own, "copro_data": {4 * (w + 6) + k + h for w in (tgp.SLOT_LIST, tgp.SLOT_POOL)
                                           for k in range(4) for h in (0, HALF)}}
    free = {"data": [runs["data"], home], "copro_data": [runs["copro_data"]], "program": [],
            "polygons": [runs["polygons"]], "textures": [runs["textures"]]}
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a = np.frombuffer(getattr(base, region), np.uint8)
        b = np.frombuffer(getattr(out, region), np.uint8)
        idx = np.flatnonzero(a != b)
        ok = np.isin(idx, np.fromiter(allowed.get(region, ()), np.int64)) | (
            _runs_mask(len(a), free[region])[idx] & (a[idx] == 0xFF))
        assert ok.all(), (region, [hex(int(k)) for k in idx[~ok][:8]])
    used = {r: int((np.frombuffer(getattr(out, r), np.uint8)[_runs_mask(len(getattr(out, r)), [runs[r]])] != 0xFF).sum()
                   - (np.frombuffer(getattr(base, r), np.uint8)[_runs_mask(len(getattr(base, r)), [runs[r]])] != 0xFF)
                   .sum()) for r in runs}
    used["lists"] = int((np.frombuffer(out.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum()
                        - (np.frombuffer(base.data, np.uint8)[home[0]:sum(home)] != 0xFF).sum())
    print(f"\nid 6's bytes per run (non-0xff bytes added): {used}; scenery objects {len(sd.objects)}, lists "
          f"{len(sd.lists)}")


def test_view_per_course():
    """Rin Rin Rink's and Circuit Pixie's view (Phase 11 Task 3, Pascal's decision "Per-course view, largest fit";
    Task 4, Pascal's ruling: Circuit Pixie the same shape; build.VIEW, view_tables, viewed_set; course.view_ids):

    - view_tables cuts every point's mask of the shared view-test tables to the cells the view names, rotated per
      heading quadrant, and keeps the offsets and the unread words; it refuses a shape that is not left-right
      symmetric (fix round 1, M6); the whole half-block (15 cells) gives the shared
      tables byte for byte; VIEW[6] and VIEW[5] are one definition (build.VIEW_12), the 12 cells of the largest
      fitting shape (fix round 1);
    - viewed_set on a widened set: ids 5 and 6 each read their own four tables (written in ID_DATA_FREE, where the
      bytes were free; the same bytes, the same shape), every other id the shared four (course.view_tables); the shared tables, the visit orders and offsets are
      untouched (course.culling the same); the view test's load names VIEW_RAM with its opcode word unchanged, widen_ids'
      `update` ends in a branch to the copy; no other byte changes; with no course named the set is returned as it is;
    - course.view_ids refuses a set that is not widened, a second application, and views that are not ids 0-15 to four
      4-aligned ROM pointers."""
    import struct
    from dtools.arcade import i960, romset
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    w = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    half = frozenset((d, c) for d in range(3) for c in range(-2, 3))
    stock = C.view_tables(w, 0)
    assert all(C.view_tables(w, k) == stock for k in range(C.COURSE_IDS)) and not C.viewed(w)
    assert build.view_tables(w, half) == b"".join(w.read_i960(p, C.VIEW_TEST_SIZE) for p in stock)
    assert build.VIEW_12 == frozenset({(0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1),
                                       (1, 2), (2, -2), (2, 2)})
    assert build.VIEW == {6: build.VIEW_12, 5: build.VIEW_12} and build.VIEW[5] is build.VIEW[6]
    blob = build.view_tables(w, build.VIEW[6])
    turn = (lambda d, c: (d, c), lambda d, c: (c, -d), lambda d, c: (-d, c), lambda d, c: (c, d))
    for q in range(4):
        keep = {turn[q](d, c) for d, c in build.VIEW[6]}
        mask = sum(1 << i for i in range(25) if (i // 5 - 2, i % 5 - 2) in keep)
        old, new = w.read_i960(stock[q], C.VIEW_TEST_SIZE), blob[C.VIEW_TEST_SIZE * q:C.VIEW_TEST_SIZE * (q + 1)]
        for off in range(0, C.VIEW_TEST_SIZE, 4):
            a, b = struct.unpack_from("<I", old, off)[0], struct.unpack_from("<I", new, off)[0]
            assert b == (a & mask if off - 8 in C.VIEW_TEST_POINTS else a), (q, off)
    for bad in ({(3, 0)}, {(0, 3)}, [(0, 0, 0)], "x"):
        with pytest.raises(ConvertError):
            build.view_tables(w, bad)
    with pytest.raises(ConvertError, match="symmetric"):            # fix round 1, M6: one-sided shapes flip sides
        build.view_tables(w, build.VIEW[6] | {(2, 1)})
    assert build.viewed_set(w, {}) is w
    v = build.viewed_set(w)
    assert C.viewed(v) and [C.view_tables(v, k) == stock for k in range(C.COURSE_IDS)] == [k not in (5, 6)
                                                                                            for k in range(16)]
    own, own5 = C.view_tables(v, 6), C.view_tables(v, 5)          # in id order: 5's tables, then 6's
    assert own5 == tuple(own5[0] + C.VIEW_TEST_SIZE * q for q in range(4))
    assert own[0] == own5[0] + 4 * C.VIEW_TEST_SIZE
    assert own == tuple(own[0] + C.VIEW_TEST_SIZE * q for q in range(4))
    assert b"".join(v.read_i960(p, C.VIEW_TEST_SIZE) for p in own) == blob
    assert b"".join(v.read_i960(p, C.VIEW_TEST_SIZE) for p in own5) == blob
    assert C.culling(v) == C.culling(w)
    assert v.read_i960(C.VIEW_READER, 8) == C.VIEW_ORIGINAL[:4] + struct.pack("<I", C.VIEW_RAM)
    assert v.read_i960(C.UPDATE_RET, 4) == i960.ctrl("b", C.VIEW_CODE - C.UPDATE_RET)
    assert v.read_i960(C.VIEW_CODE, len(C.view_code().bytes())) == C.view_code().bytes()
    assert (C.VIEW_RAM, C.VIEW_RAM_SIZE) in [(a, n) for a, n, _ in romset.RESERVED_RAM]
    allowed = set(range(C.VIEW_HOME, C.VIEW_HOME + C.VIEW_SIZE)) | set(range(C.UPDATE_RET, C.UPDATE_RET + 4)) | set(
        range(C.VIEW_READER + 4, C.VIEW_READER + 8))
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a, b = np.frombuffer(getattr(w, region), np.uint8), np.frombuffer(getattr(v, region), np.uint8)
        idx = np.flatnonzero(a != b)
        if region == "program":
            assert set(map(int, idx)) <= allowed, [hex(int(k)) for k in idx[:8]]
        elif region == "data":
            lo = own5[0] - DATA_BASE
            assert idx.min() >= lo and idx.max() < lo + 8 * C.VIEW_TEST_SIZE and (a[idx] == 0xFF).all()
        else:
            assert not len(idx), region
    with pytest.raises(C.CourseError, match="widened"):
        C.view_ids(C.guard_new_ids(C.grow_ids(rs)), {6: stock})
    with pytest.raises(C.CourseError, match="already"):
        C.view_ids(v, {6: stock})
    for bad in ({16: stock}, {6: stock[:3]}, {6: (1, 2, 3, 4)}, {True: stock}, {6: (0x7FFFFFFC,) * 4},
                {6: (0x00500000,) * 4}):
        with pytest.raises(C.CourseError):
            C.view_ids(w, bad)


# ---- Phase 11 Task 4: Circuit Pixie (n 5) at id 5 (discovery log, "Circuit Pixie") ----

@pytest.fixture(scope="module")
def pixie_id5(disc):  # noqa: F811
    """n 5 (Circuit Pixie) with its own scenery at course id 5 of the shipped chain after Rin Rin Rink (grow_ids,
    guard_new_ids, widen_ids, Desert City at id 3, National Park Speedway at id 4, Mermaid Lake at id 7, Rin Rin Rink at
    id 6): (set before, scenery, course, set after). The set is converted_set(base, disc, 5, slot=5,
    scenery="model")."""
    from dtools.dc.disc import course_files
    from tests.test_mirrors_mame import _converted
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    base = _converted(rs, disc, (3, 4, 7, 6))
    d, exe = _data(disc, 5), disc.read(EXE_NAME)
    mn, tn = course_files(5, kind="model"), course_files(5, kind="textures")
    sd = build.model_scenery(base, d, pol.decode_pol(disc.read(mn), mn), disc.read(tn), build.THIN.get(5), slot=5,
                             quad=build.QUAD.get(5), remesh=build.REMESH.get(5), props=build.NAMED_PROPS.get(5, ()))
    c = build.converted_course(base, d, exe, 5, slot=5, scenery="model", scenery_data=sd)
    out = build.converted_set(base, disc, 5, slot=5, scenery="model")
    return base, sd, c, out


def test_converted_id5_round_trip(pixie_id5):
    """Circuit Pixie as course id 5, after Rin Rin Rink on the shipped chain (as id 6's test; neither lifted nor
    shifted: its model, road and ground lie inside the world as stored):

    - the set decodes id 5 to the converted Course on every field, check_course passes, every table re-emits and
      encode(base, c) is the set converted_set builds; a standing start (ROLLING_STARTS[5]), grid base GRID_BASES[5],
      its 129 segments and no checkpoint before the lap line; the scenery's shift and lift 0;
    - its own of every table it changes, in NEW_ID_RUNS (each past Rin Rin Rink's), its lists and HUD outline's list
      in the list home DATA_FREE, its own 1 MB course sheet block, the shared EMPTY_ACTORS;
    - ids 0-4, 6 and 7 byte-identical to the input set's (decode, coverage, course blocks, the common block);
    - the free-bytes rule: every byte that differs from the set before is an id-5 per-id word, record 5, TGP slot 5
      or a byte that was 0xff in a free run (NEW_ID_RUNS and DATA_FREE)."""
    base, sd, c, out = pixie_id5
    assert 5 not in build.LIFT and 5 not in build.SHIFT and sd.shift == sd.lift == 0.0
    d = G.decode_course(out, 5)
    assert _differ(d, c) == []
    G.check_course(c)
    G.check_course(d, out)
    assert G.verify_lossless(out, 5) == [] and G.mismatches(out, c) == []
    assert build.encode(base, c, build.RACE_FIELDS + build.MODEL_FIELDS + build.ID_FIELDS).split_files() == \
        out.split_files()
    assert c.rolling_start[1] is False and c.grid_base[1] == build.GRID_BASES[5] == 123
    assert len(c.road) - 1 == 129 and [cp.segment for cp in c.checkpoints] == [0]
    assert c.actor_table == build.EMPTY_ACTORS and build.CONVERTED_IDS == (3, 4, 7, 6, 5)
    runs, home = build.NEW_ID_RUNS, tgp.DATA_FREE
    c6 = G.decode_course(base, 6)

    def inside(off, size, run=runs["data"]):
        return run[0] <= off and off + size <= run[0] + run[1]

    e = c.entry
    for f in ("step_table", "record_table", "line_table", "block_ref_a", "block_ref_b"):
        assert inside(getattr(e, f), 4) and getattr(e, f) > getattr(c6.entry, f), f
    assert inside(c.grid - DATA_BASE, 4 * C.GRID_CELLS) and c.grid != c6.grid
    block = c.sheet_sources[0]
    assert block % 0x100000 == 0 and inside(block, 0x100000)
    assert block not in {G.decode_course(base, k).sheet_sources[0] for k in (3, 4, 6, 7)}
    assert T.sheet_source(out, 0, 5) == block
    assert 4 * c.ground_data.list_base > 4 * c6.ground_data.list_base >= runs["copro_data"][0]
    assert all(inside(4 * k, 4 * o.word_count, runs["polygons"]) for k, o in sd.objects.items())
    assert min(sd.objects) > max(c6.objects)
    assert all(inside(2 * k, 8, runs["textures"]) for k in sd.headers)
    assert all(inside(k - DATA_BASE, 16 * len(v) + 4, home) for k, v in sd.lists.items())
    assert inside(c.hud_map.outline_list - DATA_BASE, 20, home) and c.hud_map.outline_list != c6.hud_map.outline_list
    # ids 0-4, 6 and 7 byte-identical
    for k in (0, 1, 2, 3, 4, 6, 7):
        ref = G.decode_course(base, k)
        assert _differ(G.decode_course(out, k), ref) == [], k
        assert G.mismatches(out, ref) == [], k
        for region, off, size, _ in G.coverage(base, k):
            assert getattr(out, region)[off:off + size] == getattr(base, region)[off:off + size], (k, region, off)
        blk = T.sheet_source(base, 0, k)
        assert out.data[blk:blk + 0x100000] == base.data[blk:blk + 0x100000], k
    common = T.sheet_source(base, 1)
    assert out.data[common:common + 0x100000] == base.data[common:common + 0x100000]
    # the free-bytes rule
    own = set()
    for name in ("COURSE_POINTERS", "COURSE_SHEETS", "GRID_TABLES", "ACTOR_TABLES", "CHECKPOINT_TABLES",
                 "SEGMENT_FLAGS", "HUD_MAP_LISTS", "HUD_MAP_PLACES", "HUD_MAP_ZOOMS", "GRID_BASES", "ROLLING_STARTS",
                 "TIME_TABLES[0]", "TIME_TABLES[1]", "TIME_TABLES[2]"):
        a_ = C.ID_ARRAYS[name]
        own |= set(range(a_.new - DATA_BASE + 5 * a_.stride, a_.new - DATA_BASE + 6 * a_.stride))
    rec = C.record_address(5) - DATA_BASE
    own |= set(range(rec, rec + C.COURSE_ENTRY_SIZE))
    allowed = {"data": own, "copro_data": {4 * (w + 5) + k + h for w in (tgp.SLOT_LIST, tgp.SLOT_POOL)
                                           for k in range(4) for h in (0, HALF)}}
    free = {"data": [runs["data"], home], "copro_data": [runs["copro_data"]], "program": [],
            "polygons": [runs["polygons"]], "textures": [runs["textures"]]}
    for region in ("program", "data", "copro_data", "polygons", "textures"):
        a = np.frombuffer(getattr(base, region), np.uint8)
        b = np.frombuffer(getattr(out, region), np.uint8)
        idx = np.flatnonzero(a != b)
        ok = np.isin(idx, np.fromiter(allowed.get(region, ()), np.int64)) | (
            _runs_mask(len(a), free[region])[idx] & (a[idx] == 0xFF))
        assert ok.all(), (region, [hex(int(k)) for k in idx[~ok][:8]])
    print(f"\nid 5: scenery objects {len(sd.objects)}, lists {len(sd.lists)}")


PIXIE_FLAG = tuple(range(34, 98))
"""CS5's waving flag (the US flag and SEGA sheet, texture 0): 64 objects of 48 vertices at one place (x 33.9..50.4,
y 25.8..35.0, z 193.3..201.3 as the disc stores them), the frames of one animation, the same shape as Mermaid Lake's
flags (LAKE_FLAGS) but placed in the course instead of at the origin, so the origin rule does not find them."""
PIXIE_BANNER = tuple(range(2, 34))
"""CS5's waving sponsor banner (texture 23, the sponsor sheet): 32 objects of 84 vertices in two meshes at one place
(x -81.2..-14.3, y 13.0..19.1, z -147.6..-141.5), the frames of one animation."""


def test_cs5_named_props(disc):  # noqa: F811
    """build.NAMED_PROPS[5] (Phase 11 Task 4, discovery log "Circuit Pixie"): the frames after the first of CS5's two
    animations are run-time props the origin rule does not find (every frame far from the origin), all kind 1:

    - the flag (35-97): every frame 48 vertices, one mesh of texture 0, the same strips (u, v) as frame 34, its
      vertices within 3.4 units of it, all 64 frames different;
    - the sponsor banner (3-33): 84 vertices, two meshes of texture 23, the same six strips as frame 2 (the same u, v,
      within 4.6 units; frames 5-8 and 21 group them five and one between the two meshes, the others four and two),
      31 of the 32 frames different (29 and 30 alike).

    kept_model drops them and keeps frames 34 and 2 and everything else of the model. The Dreamcast shows one frame at
    a time; the static scenery drew all 96 at once (unreduced, their cells 0x68 and 0x97 were the course's heaviest:
    3,824 and 2,006 records)."""
    from dtools.dc.disc import course_files
    d, mn, tn = _data(disc, 5), course_files(5, kind="model"), course_files(5, kind="textures")
    m, blob = pol.decode_pol(disc.read(mn), mn), disc.read(tn)
    assert build.NAMED_PROPS[5] == PIXIE_BANNER[1:] + PIXIE_FLAG[1:]

    def strips(o):                  # {(u, v) of a strip's vertices: their positions}, every mesh of the object
        out = {}
        for me in m.objects[o].meshes:
            for st in me.strips:
                vs = [e for e in st.entries if isinstance(e, pol.Vertex)]
                out[tuple((e.u, e.v) for e in vs)] = np.array([(e.x, e.y, e.z) for e in vs])
        return out
    for frames, verts, meshes, texture, near, distinct in ((PIXIE_FLAG, 48, 1, 0, 3.4, 64),
                                                           (PIXIE_BANNER, 84, 2, 23, 4.6, 31)):
        first = strips(frames[0])
        pts = [np.concatenate([strips(o)[k] for k in first]) for o in frames]
        for o, p in zip(frames, pts):
            ob = m.objects[o]
            assert set(strips(o)) == set(first) and len(p) == verts, o              # the same strips, the same u, v
            assert len(ob.meshes) == meshes and ob.kind == 1 and {me.texture for me in ob.meshes} == {texture}, o
            assert np.abs(p - pts[0]).max() < near and np.hypot(p[:, 0], p[:, 2]).min() > 100, o
        assert len({p.tobytes() for p in pts}) == distinct          # the frames move (the banner's 29 and 30 alike)
    assert not set(build.origin_props(m, d)) & set(PIXIE_FLAG + PIXIE_BANNER)
    k0, k1 = build.kept_model(m, blob, d), build.kept_model(m, blob, d, props=build.NAMED_PROPS[5])
    gone = set(build.NAMED_PROPS[5])
    assert all(len(k1.objects[o].meshes) == 0 for o in gone)
    assert k1.objects[34] == m.objects[34] and k1.objects[2] == m.objects[2]
    assert [ob for k, ob in enumerate(k1.objects) if k not in gone] == [ob for k, ob in enumerate(k0.objects)
                                                                       if k not in gone]


def test_cs5_levers(disc):  # noqa: F811
    """Review Focus 5 on Circuit Pixie (Phase 11 Task 4): each lever its scenery converts with lies within the bound
    the discovery log stated before measuring ("Circuit Pixie", the levers' bounds), and no object class is dropped:

    - QUAD[5] is (10, 0.125); every quad of the converted model (a record drawing two triangles) folds by at most 10
      degrees between its triangles' normals (by the signed dot, as model._merge refuses: a back-folded pair reads
      near 180) and its fourth corner's (u, v) lies within 1/8 of its texture's smaller side (in texels, the converted
      size) of the affine prediction from the other three;
    - REMESH[5] is 0.1: the re-meshed model lies within 0.1 world units of the kept one both ways, mesh by mesh, and
      opens no crack (tests/test_convert_model.py's two_way and cracks);
    - THIN[5] is not set: thinning does not earn its place (the log: every object thinned to 0.25 at once removes 120
      of the 5,511 records and the start, where the target fails, still draws 4,289);
    - every object and every texture the kept model draws (kept_model with NAMED_PROPS[5]) is still drawn after the
      levers: the levers merge and re-mesh triangles, they drop no scenery."""
    from dtools.convert import textures
    from dtools.dc.disc import course_files
    from tests.test_convert_model import cracks, two_way
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    assert build.QUAD[5] == (10.0, 0.125) and build.REMESH[5] == 0.1 and 5 not in build.THIN
    d, mn, tn = _data(disc, 5), course_files(5, kind="model"), course_files(5, kind="textures")
    m, blob = pol.decode_pol(disc.read(mn), mn), disc.read(tn)
    kept, work = build._scenery_models(m, blob, d, None, build.REMESH[5], build.NAMED_PROPS[5])
    there, back = two_way(kept, work, 0.5)
    assert there <= build.REMESH[5] and back <= build.REMESH[5], (there, back)
    assert cracks(kept, work) == 0
    headers, _ = textures.course_textures(m, blob, rs)
    sizes = {t: (h.width, h.height) for t, h in headers.items()}
    pieces = model.convert_model(work, build.QUAD[5], sizes)
    tris = list(pol.triangles(work))

    def xyz(v):
        return np.array((v.x, v.y, v.z))
    fold = shear = 0.0
    quads = 0
    for ps in pieces.values():
        for pc in ps:
            for tex, src in zip(pc.textures, pc.sources):
                if len(src) != 2:
                    continue
                quads += 1
                a, b = (tris[k][:3] for k in src)
                na, nb = (np.cross(xyz(t[1]) - xyz(t[0]), xyz(t[2]) - xyz(t[0])) for t in (a, b))
                fold = max(fold, math.degrees(math.acos(max(-1.0, min(1.0, (na @ nb) / np.linalg.norm(na) /
                                                                      np.linalg.norm(nb))))))   # signed: model._merge's
                if tex < 0:
                    continue
                at = {tuple(xyz(v)): v for v in a}
                p, q = (at[tuple(xyz(v))] for v in b if tuple(xyz(v)) in at)     # the shared edge, a's (u, v)
                x = next(v for v in a if v is not p and v is not q)
                far = next(v for v in b if tuple(xyz(v)) not in at)
                s_, t_ = np.linalg.lstsq(np.array([xyz(p) - xyz(x), xyz(q) - xyz(x)]).T, xyz(far) - xyz(x),
                                         rcond=None)[0]
                w, h = sizes[tex]
                ux, up, uq, uf = (np.array((v.u * w, v.v * h)) for v in (x, p, q, far))
                shear = max(shear, float(np.linalg.norm(ux + s_ * (up - ux) + t_ * (uq - ux) - uf)) / min(w, h))
    print(f"\nCS5 levers: re-mesh {there:.4f} / {back:.4f}; {quads} quads, fold at most {fold:.3f} degrees, shear "
          f"at most {shear:.4f} of the texture's side")
    assert quads and fold <= build.QUAD[5][0] + 1e-6 and shear <= build.QUAD[5][1] + 1e-9
    live = [(t[3], i[2]) for t, i in zip(pol.triangles(kept), model._walk(kept))
            if not model.degenerate([model._pos(v) for v in t[:3]])]          # (object, texture) of what kept draws
    drawn = {k for ps in pieces.values() for pc in ps for src in pc.sources for k in src}
    info = model._walk(work)
    assert {tris[k][3] for k in drawn} == {o for o, _ in live}
    assert {info[k][2] for k in drawn} == {t for _, t in live}


# ---- Phase 11 Task 5: the shipped set pinned ----

M5B_SHA1 = {"program": "c13354b25ce3d067864b232241618eceeee72b7e",
            "data": "a8029a2520351879944c1c545d05319ecf8f55b8",
            "copro_data": "06e8923fa71bf8f5935b89c746058e3910f2396f",
            "polygons": "87b912f2b4fd7fa15103bfaad96375bd59c38f1b",
            "textures": "428a5540d09d18f2d7b6cd5e22d436a322881cbc"}
"""Phase 11's shipped set (M5b; tests/test_courses_mame.py `m5b_set`: grow, guard, widen, ids 3, 4, 7, 6 and 5
converted, the rankings, the select with the row of eight, then `build.viewed_set`), the sha1 of each region, built
independently of the code under test: fd5703f's code (Task 4's last commit; git archive into a scratch folder, its own
empty re-mesh cache via DAYTONA_OUT, the re-meshes of 4, 5 and 6 cold, 393 s) building its own `m5b_set`
(out/phase11/t5/m5b_ref_sha1.txt). The playtest set (`--m5b`) differs from it only in the start words
(tests/test_playtest.py test_m5b_set_bytes). Phase 13 Task 2 round 2 repinned it: model._merge's quad tests made exact
(5b970ee; four National Park Speedway pairs on the shear bound, three of which the float lstsq had rejected on this Mac,
now merge), built by 5b970ee's code from a git archive (fresh venv, Python 3.15.0, numpy 2.5.4;
out/phase13/r2/pins.txt); linux/amd64 builds the same bytes. v0.1.1 repinned it (data, polygons, textures; program and copro_data unchanged): build.shadow_volumes
also takes an untextured closed mesh of a kind-3 object (CS4's 23 and CS6's 5 volumes not black), built by the code
under test; the release pin, test_release.py RELEASE_SHA1, matches an independent build of v0.1's code with only that
rule patched in (out/debug/np-start/fixset.py). v0.1.2 repinned it (data, polygons, textures; program and copro_data unchanged): build.decals also drops a
texture nearly one colour (DECAL_SHARE, DECAL_STEP: Mermaid Lake's and Circuit Pixie's shadow 67), built by the code
under test; the release pin, test_release.py RELEASE_SHA1, matches an independent build of v0.1.1's code with only that
rule patched in (out/debug/lake/fixset.py). v0.1.2's lake fix repinned it: build.CELL_CUT cuts Mermaid Lake's long scenery polygons on the cell grid (its start straight was not drawn during the start's fly-by) and build.QUAD[7] merges its quads as on 4-6, so the lap stays in budget; only CS7's scenery changes (data, polygons, textures; program and copro_data unchanged)."""


def test_m5b_set_pinned(disc):  # noqa: F811
    """The shipped set (`m5b_set`) is M5B_SHA1 region for region: the bytes Phase 11 ships (the courses, the row of
    eight, the two views) cannot move without this pin moving."""
    from tests.test_courses_mame import m5b_set
    rs = _romset()
    if rs is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    s = m5b_set(rs, disc)
    assert {r: hashlib.sha1(getattr(s, r)).hexdigest() for r in M5B_SHA1} == M5B_SHA1


# ---- M9 stage 1: Desert City in reverse at id 11 ----

def test_reverse_desert_city_id11(disc, pixie_id5):  # noqa: F811
    """Desert City in reverse at id 11 on the shipped chain (the five converted: pixie_id5's set), with its own R
    scenery (converted_set(..., 3, slot=11, variant="R", scenery="model")): id 11 decodes and golden.check_course
    passes; its road is CSDAT3R's (as many segments), its checkpoints are dc.exe.checkpoints(exe, 3, "R"), every step
    and checkpoint segment lies on the reverse road, its grid base is dc.exe.grid_base(exe, 3, "R"), its pit run is
    one run of the reverse road; ids 0-10 and 12-15 decode as before the step."""
    from dtools.dc import exe as X
    from dtools.dc.data import decode_data
    from dtools.dc.disc import course_files
    before = pixie_id5[3]
    after = build.converted_set(before, disc, 3, slot=11, variant="R", scenery="model")
    blob = disc.read(X.EXE_NAME)
    name = course_files(3, "R", "data")
    d = decode_data(disc.read(name), name)
    c11 = G.decode_course(after, 11)
    G.check_course(c11, after)
    assert len(c11.road) == len(d.road) and len(c11.road) != 0
    assert [cp.segment for cp in c11.checkpoints if not cp.lap_line] == list(X.checkpoints(blob, 3, "R"))
    assert X.checkpoints(blob, 3, "R") != X.checkpoints(blob, 3)
    assert max(s.segment for s in c11.steps) < len(c11.road)
    assert all(cp.segment < len(c11.road) for cp in c11.checkpoints)
    assert c11.grid_base[1] == X.grid_base(blob, 3, "R")
    assert c11.pit is not None and c11.pit.box_segment < len(c11.road)
    from tests.test_golden import _differ
    for k in [k for k in range(16) if k != 11]:
        assert not _differ(G.decode_course(after, k), G.decode_course(before, k)), k
