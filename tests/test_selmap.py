"""dtools/arcade/selmap.py (Phase 12 Tasks 6 and 7, M7): the converted courses' maps below Circuit Select's panels,
flat polygons drawn from the set's own road line in the originals' map style over a painting of the course's own
scenery seen from above. Offline: the strips of a hand-made painting; on the player's set (the grown set's ids 3-7 are
Beginner's copies); the converted courses' paintings on the release set (tests/test_release.py's set); in MAME:
tests/test_release_mame.py."""

from pathlib import Path

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import polyrom as P
from dtools.arcade import select as S
from dtools.arcade import selmap as M
from dtools.arcade import texture as T
from dtools.arcade.romset import RomSet
from dtools.convert import build
from dtools.convert import textures as TX
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)


def _grown(romset_zip) -> RomSet:
    return C.guard_new_ids(C.grow_ids(RomSet.from_zip(Path(romset_zip))))


def _texel(x: float, z: float) -> tuple[float, float]:
    """(u, v) of quad point (x, z) on the originals' map texture: their texture points put (2, -2) at texel (0, 0)
    and (-2, 2) at (128, 128), u along -x, v along z (texture.texture_points: (v, u) per vertex, read as the
    rasterizer reads them; the converted maps' fit is checked against the textures through it)."""
    return (M.QUAD - x) * M.TEXELS / (2 * M.QUAD), (z + M.QUAD) * M.TEXELS / (2 * M.QUAD)


def _area(poly) -> float:
    """The signed area of a polygon's corners in the quad's (x, z) plane."""
    xz = [(x, z) for x, _, z in poly.points]
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(xz, xz[1:] + xz[:1])) / 2


def _header_index(obj) -> list[int]:
    """Per record of `obj` the index (from 0) of the header it is drawn with: the attribute's header step summed over
    the records before it (texture.object_headers' walk)."""
    out, at = [], 0
    for r in obj.records:
        out.append(at)
        at += (r.attr >> 12) & 0x1F
    return out


def test_selmap_pictures(romset_zip):
    """A course map (`course_map`): deterministic (two sets of the same bytes give the same object and headers), read
    from the set alone (its road line, its scenery); one flat (untextured) header per part (`parts`: the ground, each
    painted colour, the frame, the edge, the road, the start stripe) in the palette entries the game loads and never
    rewrites (textures.flat_entry of the painting's colours, ROAD_RGB, EDGE_RGB); every polygon after the ground at the
    ground's depth, drawn with its part's header, wound as the ground quad (single-sided: all face up), within
    PAINT_MAX_DRAWN (`drawn_polygons`); its road strip follows the road's kept points (each strip edge centred on its
    point, ROAD_TEXELS wide, the edge strip EDGE_TEXELS wider either side); the start stripe across the road at road
    point 0, ROAD_TEXELS wide and START_TEXELS long; the ground the whole quad. `fit` turns a road as the originals'
    maps do: Beginner's and Advanced's road centres land on their own maps' road texels (the dark grey of the base
    layer, index 1) through the texture points, and the longer axis spans FIT_TEXELS."""
    grown = _grown(romset_zip)
    again = _grown(romset_zip)
    tables = T.colour_tables(grown)
    texel = 2 * M.QUAD / M.TEXELS
    road_entry, edge_entry = TX.flat_entry(M.ROAD_RGB, tables), TX.flat_entry(M.EDGE_RGB, tables)
    for course in (0, 1, 2, 3):
        obj, heads = M.course_map(grown, course)
        assert (obj, heads) == M.course_map(again, course)
        assert P.encode_object(obj) == P.encode_object(M.course_map(again, course)[0])
        colours, pieces = M.map_paint(grown, course)
        entries = M.map_colours(grown, course)
        names = M.parts(pieces)
        assert len(heads) == 8 * len(names) == 8 * len(entries) and len(colours) == len(pieces) + 1
        assert names[0] == "ground" and names[-4:] == ("frame", "edge", "road", "start")
        assert entries == tuple(TX.flat_entry(c, tables) for c in colours) + (road_entry, edge_entry) * 2
        for k, c in enumerate(entries):
            h = T.TexHeader(colour=c, **build.FLAT_HEADER)
            assert heads[8 * k:8 * k + 8] == T.encode_tex_header(h) and not h.textured
            assert 0 <= c < TX.PALETTE_COLOURS and c not in TX.REWRITTEN
        # the polygons: ground, the painted strips, frame, edge strip, road strip, stripe; one depth (the ground's)
        polys = P.polygons(obj)
        drawn = [p for p in polys if p.drawn]
        pts = M.fit(C.road_line(grown, C.course_table(grown)[course]))
        strips = [st for _, sts in pieces for st in sts]
        assert len(drawn) == M.drawn_polygons(len(pts), pieces) <= M.PAINT_MAX_DRAWN
        assert len(drawn) == 1 + sum(len(st) - 1 for st in strips) + 4 + 2 * len(pts) + 1
        assert len(polys) == len(drawn) + len(strips) + 4
        assert [(r.attr >> 10) & 3 for r in obj.records] == [M.Z_MIN] + [M.Z_OLD] * (len(obj.records) - 1)
        assert all(_area(p) < 0 for p in drawn)                 # the ground quad's winding: every polygon faces up
        part = _header_index(obj)
        assert part[-1] == len(names) - 1 and sorted(set(part)) == list(range(len(names)))
        assert [part[k] for k, p in enumerate(polys) if p.drawn][:1] == [0]
        at = 1
        for i, (_, sts) in enumerate(pieces, 1):                 # each colour's strips: a move, then its quads
            for st in sts:
                recs = obj.records[at:at + len(st)]
                assert not recs[0].link and all(r.link for r in recs[1:]) and part[at + 1] == i
                assert [(r.p0[::2], r.p1[::2]) for r in recs] == [tuple(e) for e in st]
                at += len(st)
        assert sorted({p.points for p in drawn[:1]}) == [((-2.0, 0.0, -2.0), (-2.0, 0.0, 2.0), (2.0, 0.0, 2.0),
                                                          (2.0, 0.0, -2.0))]
        frame_at = at
        assert {part[k] for k in range(frame_at + 1, frame_at + 5)} == {len(names) - 4}
        edge = obj.records[frame_at + 6:frame_at + 6 + len(pts)]
        road = obj.records[frame_at + 7 + len(pts):frame_at + 7 + 2 * len(pts)]
        assert {part[frame_at + 6]} == {len(names) - 3} and {part[frame_at + 7 + len(pts)]} == {len(names) - 2}
        for recs, width in ((road, M.ROAD_TEXELS), (edge, M.ROAD_TEXELS + 2 * M.EDGE_TEXELS)):
            ends = recs[-1:] + recs[:-1]                     # the strip's edges in point order (the last is point 0)
            for (x, z), r in zip(pts, ends):
                a, b = np.array(r.p0), np.array(r.p1)
                assert np.allclose((a + b) / 2, (x, 0, z), atol=1e-5)
                assert abs(np.linalg.norm(a - b) - width * texel) < 1e-5
        move, stripe = obj.records[-2:]
        assert not move.link and stripe.link and part[-1] == len(names) - 1
        c0, c1 = (np.array(move.p0) + move.p1) / 2, (np.array(stripe.p0) + stripe.p1) / 2
        assert np.allclose((c0 + c1) / 2, (pts[0][0], 0, pts[0][1]), atol=1e-5)
        assert abs(np.linalg.norm(c1 - c0) - M.START_TEXELS * texel) < 1e-5
        assert abs(np.linalg.norm(np.array(stripe.p0) - stripe.p1) - M.ROAD_TEXELS * texel) < 1e-5
        span = np.ptp(np.array(pts), 0)
        assert abs(span.max() - M.FIT_TEXELS * texel) < 1e-4 and np.abs(np.array(pts)).max() < M.QUAD
    # the originals' turn, measured on their maps: their road centres on their road texels
    for panel in (0, 1):
        course = C.PANEL_COURSES[panel]
        lst = S._stock_map_lists(grown)[panel]
        q = C.object_list(grown, lst)[0]
        o = P.decode_object(grown.polygons, C.check_rom_object(q.oba, grown), q.obc, S.MAP_MODE)
        assert T.texture_points(grown, q.tpa, o)[0] == ((1024, 0), (1024, 1024), (0, 1024), (0, 0))
        base = T.decode_indices(grown, T.object_headers(grown, q.tha, o)[0])
        road = np.zeros_like(base, bool)
        road[2:-2, 2:-2] = base[2:-2, 2:-2] == 1           # the dark grey, less the texture's border
        pts = M.fit(C.road_line(grown, C.course_table(grown)[course]))
        hits = [road[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2].any() for u, v in (_texel(x, z) for x, z in pts)]
        assert all(hits), (panel, np.mean(hits))           # each within a texel of the road (96 % / 90 % on it)
    # refusals
    for bad in (None, [], [(0.0, 0.0)] * 2, "road", ["ab", "cd", "ef"], [(0.0, 0.0), (1.0, 1.0), (True, 0.0)],
                [(0.0, 0.0), (1.0, 1.0), (2.5, 0.0)], [(0.0, 0.0), (1.0, float("nan")), (1.0, 0.0)]):
        with pytest.raises(S.SelectError):
            M.map_object(bad)
    with pytest.raises(S.SelectError):
        M.map_object([(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)])          # a point whose neighbours meet
    loop = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    edge = ((1.0, 0.0), (0.0, 0.0))
    for bad in ("x", [[]], [[[edge]]], [[[edge, ((2.5, 0.0), (0.0, 0.0))]]], [[[edge, (edge[0],)]]],
                [[[edge, ((True, 0.0), (0.0, 0.0))]]], None):
        with pytest.raises(S.SelectError):
            M.map_object(loop, bad)
    for bad in (16, -1, True, None):
        for f in (M.course_map, M.map_colours, M.map_paint, M.paint, M.scenery_picture):
            with pytest.raises(S.SelectError):
                f(grown, bad)
    for cells in (3, 0, 512, True, 64.0):
        with pytest.raises(S.SelectError):
            M.paint(grown, 3, cells)
    with pytest.raises(S.SelectError):
        M.fit([])


def _painted(ground: int, pieces, cells: int) -> np.ndarray:
    """The labels `pieces` (paint_pieces) draw over the `ground` label, sampled at each cell's centre in drawing order:
    a cell takes a strip quad's label when its centre's row lies between the quad's two edges and its column between
    their ends interpolated there (ends inclusive)."""
    out = np.full((cells, cells), ground)
    yy, uu = np.mgrid[0:cells, 0:cells] + 0.5
    for label, strips in pieces:
        for st in strips:
            cs = [((z + M.QUAD) * cells / (2 * M.QUAD), (M.QUAD - b[0]) * cells / (2 * M.QUAD),
                   (M.QUAD - a[0]) * cells / (2 * M.QUAD)) for a, b in st for z in (a[1],)]
            for (y0, a0, b0), (y1, a1, b1) in zip(cs, cs[1:]):
                t = (yy - y0) / (y1 - y0)
                inside = (t >= 0) & (t <= 1) & (uu >= a0 + (a1 - a0) * t) & (uu <= b0 + (b1 - b0) * t)
                out[inside] = label
    return out


def test_paint_pieces_shapes():
    """`paint_pieces` on a hand-made painting (no game data): a ring (label 1, a hole and a notch in it), a square of
    label 2 inside the ring's hole and a diagonal band of label 3 crossing both, over the ground (label 0); with no
    tolerance, drawing the strips over the ground in order gives every cell its label but the road's (any label may
    fill those); each strip's edges run down the rows, the right end first, every quad of the map object wound as the
    ground quad; a label's strips may run over the cells of the labels drawn after it (they draw over them), never
    the ground's."""
    n = 16
    lab = np.zeros((n, n), int)
    lab[2:14, 2:14] = 1
    lab[5:11, 5:11] = 0
    lab[2:5, 7:9] = 0                                        # a notch: the ring's top runs split in two
    lab[6:10, 6:10] = 2
    for r in range(n):
        lab[r, max(0, r - 1):r + 1] = 3
    road = np.zeros((n, n), bool)
    road[0, :] = True
    colours = ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3))
    painting = M.Painting(n, lab, colours, (0, 1, 2, 3), road)
    pieces = M.paint_pieces(painting, 0.0)
    assert [c for c, _ in pieces] == [1, 2, 3]
    got = _painted(0, pieces, n)
    assert (got == lab)[~road].all()
    for _, strips in pieces:
        for st in strips:
            rows = [a[1] for a, _ in st]
            assert rows == sorted(rows) and all(a[0] <= b[0] for a, b in st)   # quad x falls as the column grows
    pts = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    obj = M.map_object(pts, tuple(st for _, st in pieces))
    assert all(_area(p) < 0 for p in P.polygons(obj) if p.drawn)
    first = _painted(0, pieces[:1], n)                      # label 1's strips run over 2's and 3's cells
    assert (first[lab == 2] == 1).any() and (first[lab == 3] == 1).any() and ((first == 0) | (lab != 0) | road).all()
    # with tolerance the staircase's steps merge into slanted sides
    assert sum(len(st) for _, sts in M.paint_pieces(painting) for st in sts) < sum(len(st) for _, sts in pieces
                                                                                  for st in sts)
    assert M.drawn_polygons(4, pieces) == 1 + sum(len(st) - 1 for _, sts in pieces for st in sts) + 4 + 8 + 1


PAINTED_FAMILIES = {3: {"sand", "brown"}, 4: {"olive", "white"}, 5: {"olive", "grey_dark", "white"},
                    6: {"grass", "olive", "sand", "white"}, 7: {"grass", "olive", "water", "grey"}}
"""Families each converted course's map draws, at least (measured on the release set: Desert City's sand and brown
canyon, National Park Speedway's forest, Circuit Pixie's infield and stands, Rin Rin Rink's valley and river, Mermaid
Lake's lake, woods and town)."""


def test_selmap_painted(romset_zip, disc):  # noqa: F811
    """Task 7 (Pascal: "can we get some backgrounds on the tracks?"): each converted course's painting (`paint`,
    `map_paint`) on the release set: deterministic, read from the set alone (its scenery and road, decoded from the
    disc at build time); its colours only FAMILIES' shades and OUTSIDE's (the reduced palette from Sega's maps), each
    drawn with the palette entry nearest it the game never rewrites; the ground and at least two more colours, the
    families of PAINTED_FAMILIES among them, and every label covering 2 % or more of the painting outside the road
    drawn; the strips drawn in order over the ground give at least 97 % of the cells outside the road their painting's
    label (the rest: tolerance at slanted sides); within PAINT_MAX_DRAWN drawn polygons, at the finest level that fits
    (PAINT_LEVELS); the road strips over all of it (test_selmap_pictures)."""
    from tests.test_release import _release
    rel = _release(romset_zip, disc)
    tables = T.colour_tables(rel)
    names = list(M.FAMILIES)
    palette = {c for f in M.FAMILIES.values() for c in f} | set(M.OUTSIDE.values())
    for e in S.ENTRIES[3:]:
        k = e.course
        colours, pieces = M.map_paint(rel, k)
        assert (colours, pieces) == M.map_paint(rel, k)
        level = next(n for n in M.PAINT_LEVELS
                     if M.drawn_polygons(M.MAP_POINTS, M.paint_pieces(M.paint(rel, k, n))) <= M.PAINT_MAX_DRAWN)
        painting = M.paint(rel, k, level)
        again = M.paint(rel, k, level)
        assert np.array_equal(painting.labels, again.labels) and np.array_equal(painting.road, again.road)
        assert (painting.colours, painting.order) == (again.colours, again.order)
        assert M.paint_pieces(painting) == pieces
        assert set(colours) <= palette and len(colours) >= 3
        assert colours[0] == painting.colours[painting.order[0]] and colours[0] in M.OUTSIDE.values()
        assert M.map_colours(rel, k)[:len(colours)] == tuple(TX.flat_entry(c, tables) for c in colours)
        assert not set(M.map_colours(rel, k)) & set(TX.REWRITTEN)
        drawn = {names[c // 3] for c, _ in pieces if c < 3 * len(names)}
        assert PAINTED_FAMILIES[k] <= drawn, (k, drawn)
        seen = painting.labels[~painting.road]
        share = np.bincount(seen, minlength=len(painting.colours)) / seen.size
        assert {int(c) for c in np.flatnonzero(share >= 0.02)} <= {painting.order[0]} | {c for c, _ in pieces}, k
        got = _painted(painting.order[0], pieces, painting.cells)
        agree = (got == painting.labels)[~painting.road].mean()
        obj, _ = M.course_map(rel, k)
        polys = sum(1 for p in P.polygons(obj) if p.drawn)
        print(f"\ncourse {k} {e.name}: {level} cells, {len(colours)} colours, {sum(len(s) for _, s in pieces)} "
              f"strips, {polys} drawn polygons ({len(obj.records)} records), {agree:.1%} of the cells as painted")
        assert agree >= 0.97, (k, agree)
        assert polys == M.drawn_polygons(len(M.fit(C.road_line(rel, C.course_table(rel)[k]))), pieces)
        assert polys <= M.PAINT_MAX_DRAWN


def test_map_paint_fallback_and_refusal(romset_zip, monkeypatch):
    """`map_paint`'s level pick on the grown set's course 0, which draws 574 / 321 / 185 polygons at PAINT_LEVELS' 128 /
    64 / 32 cells: a cap of 574 takes 128, of 321 the 64-cell painting (`paint(..., 64)`), of 185 the 32-cell one, and
    one below 185 is a SelectError naming the cap and the 32 cells."""
    grown = _grown(romset_zip)
    points = len(M.fit(M._road(grown, 0)))
    drawn = [M.drawn_polygons(points, M.paint_pieces(M.paint(grown, 0, n))) for n in M.PAINT_LEVELS]
    assert drawn == [574, 321, 185]
    for cap, cells in ((574, 128), (573, 64), (321, 64), (320, 32), (185, 32)):
        monkeypatch.setattr(M, "PAINT_MAX_DRAWN", cap)
        painting = M.paint(grown, 0, cells)
        colours, pieces = M.map_paint(grown, 0)
        assert pieces == M.paint_pieces(painting), cap
        assert colours == (painting.colours[painting.order[0]],) + tuple(painting.colours[c] for c, _ in pieces)
    monkeypatch.setattr(M, "PAINT_MAX_DRAWN", 184)
    with pytest.raises(S.SelectError, match=r"course 0's map: more than 184 drawn polygons at 32 cells"):
        M.map_paint(grown, 0)
