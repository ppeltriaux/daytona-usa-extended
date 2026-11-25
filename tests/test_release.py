"""dtools/release.py (Phase 12, M7): the release set's chain (`release_set`). Offline: the chain's steps replaced by
recorders (no game data); with the data, the release set against the M5b playtest set (tools/desert_city_playtest.py
`m5b_set`): the same chain but for the select's row and the title, so every byte that differs is one patch_select or
title.extended_title writes."""

from pathlib import Path

import numpy as np
import pytest

from dtools import release as R
from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade import select as S
from dtools.arcade import tiles as TL
from dtools.arcade import title as TT
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import REGIONS, RomSet
from dtools.arcade.tablebase import DATA_BASE
from dtools.convert import build
from dtools.dc.disc import DcDisc
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_playtest import _record, _tool


def _inputs() -> tuple[RomSet, DcDisc]:
    """A RomSet and a DcDisc holding no game data: the chain's type checks pass, the recorders do the rest."""
    return RomSet(b"", b"", b"", b"", b"", {}), object.__new__(DcDisc)


def _record_long_times(monkeypatch, calls: list) -> None:
    monkeypatch.setattr(G, "decode_course", lambda s, k: calls.append(("decode", (s, k), ())) or f"c{k}")
    monkeypatch.setattr(build, "long_time", lambda s, c: calls.append(("long_time", (s, c), ())) or f"{s}+{c}")
    monkeypatch.setattr(build, "race_times", lambda s, c: calls.append(("race_times", (s, c), ())) or f"{s}+t{c}")
    monkeypatch.setattr(TT, "extended_title", lambda s, *a, **k: calls.append(
        ("extended_title", (s, *a), tuple(sorted(k.items())))) or f"{s}+title")
    monkeypatch.setattr(build, "pit_stalls", lambda s, d, n: calls.append(("pit_stalls", (s, d, n), ())) or f"{s}+p{n}")


def test_release_chain(monkeypatch):
    """`release_set` runs grow_ids, guard_new_ids, widen_ids, converts 3, 4, 7, 6 and 5 at their own ids
    (`converted_set(..., n, slot=n, scenery="model")`), grow_rankings, the select with the release row
    (select.ENTRIES itself), `build.race_times` on each of 3-7 (RELEASE_TIME_IDS, v0.1.2), no long time by default,
    `build.pit_stalls(..., disc, n)` on each of 3-7 (RELEASE_PIT_IDS, v0.1.2), the title (`title.extended_title` on the
    last pit step's set) and `build.viewed_set` last. `long_ids` adds `build.long_time` on those ids after the race
    times; `times=False` (test-only) drops the race times and `pits=False` (test-only) the pit stalls, so
    `release_set(rs, dc, RELEASE_LONG_IDS, times=False, pits=False)` is v0.1.1's chain: the M5b playtest set's steps
    (tools/desert_city_playtest.py `m5b_set`) on the same inputs, the select's row and the title aside. `title=False`
    (test-only) drops only the title; a long id outside 3-7 (an original's times are the arcade's), `title`, `times` or
    `pits` not a bool, and inputs that are not a RomSet and a DcDisc are ReleaseError (a DecodeError) before any step
    runs. These are with `direction=False` (test-only: the chain before M9 stage 3); test_release_chain_direction
    has the default."""
    rs, dc = _inputs()
    calls = _record(monkeypatch)
    _record_long_times(monkeypatch, calls)
    R.release_set(rs, dc, direction=False)
    names = [c[0] for c in calls]
    head = ["grow_ids", "guard_new_ids", "widen_ids"] + ["converted_set"] * 5 + ["grow_rankings", "patch_select"]
    pits = ["pit_stalls"] * 5
    assert names == head + ["decode", "race_times"] * 5 + pits + ["extended_title", "viewed_set"]
    assert R.RELEASE_CONVERTED == (3, 4, 7, 6, 5)
    assert R.RELEASE_TIME_IDS == R.RELEASE_LONG_IDS == R.RELEASE_PIT_IDS == (3, 4, 5, 6, 7)
    assert [c[1][1:] for c in calls if c[0] == "pit_stalls"] == [(dc, n) for n in (3, 4, 5, 6, 7)]
    conv = [c for c in calls if c[0] == "converted_set"]
    assert [c[1][1:] for c in conv] == [(dc, n) for n in (3, 4, 7, 6, 5)]
    assert [c[2] for c in conv] == [(("scenery", "model"), ("slot", n)) for n in (3, 4, 7, 6, 5)]
    assert calls[0][1] == (rs,)
    sel = calls[names.index("patch_select")]
    assert sel[1][0] == "grow_rankings#9" and sel[1][1] is S.ENTRIES and sel[2] == (("faceted_maps", False),)
    assert [c[1][1] for c in calls if c[0] == "decode"] == [3, 4, 5, 6, 7]
    timed = "patch_select#10+tc3+tc4+tc5+tc6+tc7"
    pitted = "+p3+p4+p5+p6+p7"
    assert calls[-2] == ("extended_title", (timed + pitted,), ())
    assert calls[-1] == ("viewed_set", (timed + pitted + "+title",), ())
    # long_ids after the race times
    calls[:] = []
    R.release_set(rs, dc, (7, 3), direction=False)
    assert [c[0] for c in calls][len(head):-2] == ["decode", "race_times"] * 5 + ["decode", "long_time"] * 2 + pits
    assert [c[1][1] for c in calls if c[0] == "decode"][5:] == [7, 3]
    assert calls[-2] == ("extended_title", (timed + "+c7+c3" + pitted,), ())
    # times=False, pits=False with RELEASE_LONG_IDS: v0.1.1's chain, the M5b playtest chain step for step but the row
    # and title
    calls[:] = []
    R.release_set(rs, dc, R.RELEASE_LONG_IDS, times=False, pits=False, direction=False)
    full = list(calls)
    assert [c[0] for c in full] == head + ["decode", "long_time"] * 5 + ["extended_title", "viewed_set"]
    assert full[-2] == ("extended_title", ("patch_select#10+c3+c4+c5+c6+c7",), ())
    calls[:] = []
    _tool().m5b_set(rs, dc)
    m5b = [c if c[0] != "patch_select" else (c[0], (c[1][0], S.ENTRIES), (("faceted_maps", False),)) for c in calls]
    assert calls[names.index("patch_select")][1][1] is S.M5B_ENTRIES and m5b[:-1] == full[:-2]
    assert m5b[-1] == ("viewed_set", ("patch_select#10+c3+c4+c5+c6+c7",), ())
    # times=False, long_ids=(): the gates' set, no time step at all (the pit stalls kept)
    calls[:] = []
    R.release_set(rs, dc, (), times=False, direction=False)
    assert [c[0] for c in calls] == head + pits + ["extended_title", "viewed_set"]
    assert calls[-2:] == [("extended_title", ("patch_select#10" + pitted,), ()),
                          ("viewed_set", ("patch_select#10" + pitted + "+title",), ())]
    calls[:] = []
    R.release_set(rs, dc, (), title=False, times=False, pits=False, direction=False)
    assert [c[0] for c in calls] == head + ["viewed_set"] and calls[-1] == ("viewed_set", ("patch_select#10",), ())
    # refusals, before any step
    for bad in ((rs, dc, (2,)), (rs, dc, (8,)), (rs, dc, (True,)), (rs, dc, (3.0,)), (rs, dc, 3), (rs, dc, "3"),
                (rs, dc, None), ("stock", dc), (None, dc), (rs, "disc"), (rs, None)):
        calls[:] = []
        with pytest.raises(R.ReleaseError):
            R.release_set(*bad)
        assert calls == [], bad
    for kw in ("title", "times", "pits", "direction"):
        for bad in (1, None, "no"):
            with pytest.raises(R.ReleaseError):
                R.release_set(rs, dc, (), **{kw: bad})
            assert calls == [], (kw, bad)
    with pytest.raises(TypeError):
        R.release_set(rs, dc, (), False)         # title and times are keyword-only
    assert issubclass(R.ReleaseError, DecodeError)

def test_release_chain_direction(monkeypatch):
    """M9 stage 3: by default `release_set` also converts the eight courses in reverse after the five (the arcade's
    three with their own scenery, `converted_set(..., n, slot=n + 8, variant="R", scenery="keep")`, then 3-7's
    with "model"), reads the reverse ids' views (`build.viewed_set(..., build.REVERSE_VIEW)`) and adds the Course
    Direction screen (`dirscreen.patch_direction`) last, and (M9 stage 4) after the five pit steps the reverses' own
    stalls (`build.pit_stalls(..., disc, n, "R")` for n 0-7: ids 8-15) and after the five race-time steps the
    reverses' own times (`build.race_times` on ids 8-15), and after the reverse stalls the pit stop's wrapped angle
    differences (`course.patch_pit_heading`: trigger, steering, passed test; the reverses' boxes at -32768); every other step as with `direction=False`."""
    from dtools.arcade import dirscreen as D
    rs, dc = _inputs()
    calls = _record(monkeypatch)
    _record_long_times(monkeypatch, calls)
    monkeypatch.setattr(build, "pit_stalls", lambda s, d, n, v="": calls.append(
        ("pit_stalls", (s, d, n) + ((v,) if v else ()), ())) or f"{s}+p{n}{v}")
    monkeypatch.setattr(D, "patch_direction", lambda s: calls.append(("patch_direction", (s,), ())) or f"{s}+dir")
    monkeypatch.setattr(C, "patch_pit_heading", lambda s: calls.append(("patch_pit_heading", (s,), ())) or f"{s}+h")
    monkeypatch.setattr(build, "far_viewed_set", lambda s: calls.append(("far_viewed_set", (s,), ()))
                        or f"viewed_set#{len(calls)}")
    # v0.3: the far view by default; far_view=False is v0.2's chain (thinned models, viewed_set with REVERSE_VIEW)
    R.release_set(rs, dc, far_view=False)
    v02, calls[:] = list(calls), []
    assert all("thin" not in dict(c[2]) and "depth" not in dict(c[2]) for c in v02 if c[0] == "converted_set")
    assert v02[-2][0] == "viewed_set" and v02[-2][1][1] is build.REVERSE_VIEW and v02[-1][0] == "patch_direction"
    assert "far_viewed_set" not in [c[0] for c in v02]
    for bad in ({"far_view": 1}, {"far_view": True, "direction": False}):
        with pytest.raises(R.ReleaseError):
            R.release_set(rs, dc, **bad)
    assert not calls
    R.release_set(rs, dc, direction=False)
    assert "patch_pit_heading" not in [c[0] for c in calls]
    plain, calls[:] = list(calls), []
    R.release_set(rs, dc)
    names = [c[0] for c in calls]
    assert names[:8] == [c[0] for c in plain[:8]] and names[8:16] == ["converted_set"] * 8
    assert all("road_copy" not in dict(c[2]) for c in plain if c[0] == "converted_set")   # v0.1.2's chain: no copy
    # v0.2's select maps, faceted (selmap.facets), with the screen only: without it, v0.1.2's painted maps
    assert [dict(c[2]) for c in plain if c[0] == "patch_select"] == [{"faceted_maps": False}]
    assert [dict(c[2]) for c in calls if c[0] == "patch_select"] == [{"faceted_maps": True}]
    rev = calls[8:16]
    assert [c[1][1:] for c in rev] == [(dc, n) for n in range(8)]
    assert [dict(c[2]) for c in rev] == [{"slot": n + 8, "variant": "R", "scenery": "keep" if n < 3 else "model",
                                          "road_copy": True, "thin": False, "depth": True} for n in range(8)]
    # v0.2: the road copied across cell edges (build.ROAD_COPY) on the converted courses too, with the screen only
    assert [dict(c[2]) for c in calls[3:8]] == [{"slot": n, "scenery": "model", "road_copy": True, "thin": False,
                                                 "depth": True}
                                                for n in (3, 4, 7, 6, 5)]
    assert [c[0] for c in calls] == [c[0] for c in v02[:-2]] + ["far_viewed_set", "patch_direction"]
    after = [c[0] for c in plain[8:]]
    times_end = 16 + after.index("decode") + 2 * 5          # the five converted courses' decode / race_times
    assert names[times_end:times_end + 16] == ["decode", "race_times"] * 8
    assert [c[1][1] for c in calls if c[0] == "decode"] == list(range(3, 16))
    recorded = len(calls)
    del names[times_end:times_end + 16], calls[times_end:times_end + 16]
    last_pit = 16 + len(after) - 1 - after[::-1].index("pit_stalls")
    assert names[16:last_pit + 1] == after[:last_pit - 15] and names[last_pit + 1:last_pit + 9] == ["pit_stalls"] * 8
    assert names[last_pit + 9] == "patch_pit_heading"
    assert calls[last_pit + 9][1] == (f"{calls[last_pit + 8][1][0]}+p7R",)
    del names[last_pit + 9], calls[last_pit + 9]
    assert names[last_pit + 9:-1] == after[last_pit - 15:-1] + ["far_viewed_set"] and names[-1] == "patch_direction"
    rpits = calls[last_pit + 1:last_pit + 9]
    assert [c[1][1:] for c in rpits] == [(dc, n, "R") for n in range(8)]
    view = calls[-2]
    assert view[0] == "far_viewed_set" and len(view[1]) == 1
    assert calls[-1] == ("patch_direction", (f"viewed_set#{recorded - 1}",), ())


def _select_bytes(rs: RomSet, entries) -> set[tuple[str, int]]:
    """(region, offset) of every byte patch_select writes for `entries` on `rs` (patched): its data from DATA_AT to
    the end of its char blocks (`layout_of`), its code at CODE_AT, its site words (the record, course, voice, count,
    hook and vote sites, the upload word and SELECT_PANEL_OF_COURSE's sixteen words) and its maps' lists, objects and
    headers (`map_spans`: the own panels' maps, Task 6)."""
    lay = S.layout_of(rs)
    g = S.graphics(rs, entries)
    out: set[tuple[str, int]] = set()

    def span(addr: int, n: int) -> None:
        region, off = C.rom_word_at(addr)
        out.update((region, off + i) for i in range(n))
    span(S.DATA_AT, lay.char_block + 4 * len(g.runs) + 32 * g.uploaded() - S.DATA_AT)
    span(S.CODE_AT, len(S.code(lay, S.WINDOW_NAME in g.windows).bytes()))
    sites = [*S.TABLE_SITES, S.COURSE_OF_PANEL_SITE[0], S.VOICE_SITE[0], S.PANEL_COUNT_SITE[0], *S.HOOKS,
             *S.VOTE_SITES, *S.VOTE_WORDS, S.SELECT_UPLOAD_WORD]
    for a in sites:
        span(a, 4)
    span(C.id_table(rs, "SELECT_PANEL_OF_COURSE"), 4 * S.ENTRY_COUNT)
    for region, off, n, _ in S.map_spans(rs):
        out.update((region, off + i) for i in range(n))
    return out


_RELEASE: dict = {}


def _release(romset_zip, disc) -> RomSet:  # noqa: F811
    """release_set of the player's set and disc, built once per session (the tests read it, none writes it)."""
    if str(romset_zip) not in _RELEASE:
        _RELEASE[str(romset_zip)] = R.release_set(RomSet.from_zip(romset_zip), disc)
    return _RELEASE[str(romset_zip)]


RELEASE_SHA1 = {"program": "27370a5e3cac58fb149f4d5af6e75a8be7c76436",
                "data": "89b8cb80b968dbc49b00a3a65bc28245e67a1a95",
                "copro_data": "b63256bc55ad101e4415c88fe3b0b5ff0fe119f1",
                "polygons": "595529c184d4dbb11f3e44742a9aa11b9d190e12",
                "textures": "457d32ed84fe168662803c62270fb5482f37f0d8"}
RELEASE_SHA1_V02 = {"program": "31df04b7f701d7f5d830fc5f9d2d44af0ba1dffe",
                    "data": "b34898b10d94beaffe8c474420dbbdb778e619f5",
                    "copro_data": "b63256bc55ad101e4415c88fe3b0b5ff0fe119f1",
                    "polygons": "e34bb2e6bda6e444251620fe11874db52e4e9219",
                    "textures": "4d414e14324145eaf0599770cb9d1fee784b0c91"}
"""`release_set(..., far_view=False)`: v0.2's release set byte for byte (RELEASE_SHA1's pin before the far view).
v0.3 (20 Nov 2026, in development): RELEASE_SHA1 moved with the far view (build.far_viewed_set in place of
viewed_set: every id's own view-test tables, v0.2's, each with its far table, and the hook, which also sets the
id's polygon cap: 28,000 on the converted courses and their reverses) and the converted models left unthinned
(converted_set's thin=False), docs/follow-ups/v0.3.md."""
"""The release set (`release_set`: the M5b chain, the named row, the start words of 3-7, the title, the views; Task 7's
painted maps on the select), the sha1 of each region, built independently of the code under test: ed570c2 (Task 5's
first commit) from a git archive into a scratch folder, a fresh venv on the repository's Python 3.15.0 with `pip install
-e .` only (numpy 2.5.4, no dev extras), DAYTONA_OUT an empty folder, its own `dtools build` (cold: the disc's
extraction and the three re-meshes, 407 s; out/phase12/t5/release_ref_sha1.txt). The in-repository build of the same
commit wrote the same 41 members byte for byte. Phase 13 Task 2 round 2 repinned it: model._merge's quad tests made
exact (5b970ee; four National Park Speedway pairs on the shear bound, three of which the float lstsq had rejected on
this Mac, now merge), built by 5b970ee's code from a git archive (tools/clean_install.sh: fresh venv, Python 3.15.0,
numpy 2.5.4; out/phase13/r2/pins.txt, its `dtools build` 370 s cold); the Docker linux/amd64 clean install wrote the
same 41 members (docs/spike/2026-11-page.md, "The clean install"). v0.1.1 repinned it (data, polygons, textures;
program and copro_data unchanged): build.shadow_volumes also takes an untextured closed mesh of a kind-3 object (CS4's
23 and CS6's 5 volumes not black; out/debug/np-start/report.md); out/debug/np-start/fixset.py, v0.1's code with only
that rule patched in, built these bytes before the change was written. v0.1.2 repinned it again (data, polygons,
textures): build.decals also drops a texture nearly one colour (Mermaid Lake's and Circuit Pixie's shadow 67, drawn
solid black under gantries and bridges); out/debug/lake/fixset.py, v0.1.1's code with only that rule patched in,
built these bytes before the change was written. v0.1.2's lake fix repinned it: build.CELL_CUT cuts Mermaid Lake's long scenery polygons on the cell grid (its start straight was not drawn during the start's fly-by) and build.QUAD[7] merges its quads as on 4-6, so the lap stays in budget; only CS7's scenery changes (data, polygons, textures; program and copro_data unchanged).
v0.1.2's pit stops repinned it (data only): build.pit_stalls gives ids 3-7 the Dreamcast's pit stalls (4,640 bytes at
the end of ID_DATA_FREE and the five CAR_TABLES_230E50 entries; out/pit/set-fix of the pit worktree, 4,655 bytes
differ from the set before, all there). M9 stage 3 repinned every region: the eight courses in reverse at ids 8-15,
the reverse views and the Course Direction screen (release_set's `direction`, the default); measured on the
repository's build (18 Nov 2026); the independent clean-install build is due before the public release (v0.2).
M9 stage 4 repinned it (program, data): the reverses' own time tables and pit stalls (ids 8-15) and the pit stop's
angle differences wrapped (course.patch_pit_heading: 132 program bytes at three sites). v0.2's clean
installs proved it (18 Nov 2026, tools/clean_install.sh on tools/export_public.sh of ccd2f5a, out/v02/clean): native
macOS arm64 (Python 3.15.0, numpy 2.5.5; offline suite 1066 passed, 461 skipped; cold build 1,264 s) and Docker
linux/amd64 python:3.12-slim (Python 3.12.16, numpy 2.5.5, bsdtar; 1070 passed, 457 skipped; 3,188 s, emulated) wrote
the same 41 members, byte for byte (the region sha1s before the faceted maps). v0.2's faceted maps repinned it (data,
polygons, textures: selmap.facets draws the five converted courses' Circuit Select maps; repository build, 18 Nov
2026), and the clean installs proved it again (tools/export_public.sh of d47df08, out/v02/clean2): native macOS arm64
(Python 3.15.0, numpy 2.5.5; 1066 passed, 462 skipped; cold build 1,330 s) and Docker linux/amd64 (Python 3.12.16,
numpy 2.5.5; 1070 passed, 458 skipped; 2,625 s) wrote the same 41 members, byte for byte (the region sha1s then).
The road holes fix repinned it (data, polygons, textures): build.ROAD_COPY copies the converted courses' road across
cell edges (converted_set's road_copy, with the screen only); the faceted maps came with the screen only too, so the
chain without it rebuilds v0.1.2 again (test_release_plain_is_v012). The clean installs proved it (tools/
export_public.sh of 463ad82, out/v02/clean3): native macOS arm64 (Python 3.15.0, numpy 2.5.5; 1067 passed, 463
skipped; cold build 1,893 s) and Docker linux/amd64 (Python 3.12.16, numpy 2.5.5; 1071 passed, 459 skipped; 4,221 s)
wrote the same 41 members, byte for byte, the region sha1s these."""

PLAIN_SHA1 = {"program": "1f6a3b04d75b03489a2f5c1278cbd299af917ef7",
              "data": "d15ab850188262f41dfe4387de99654c2416fea4",
              "copro_data": "06e8923fa71bf8f5935b89c746058e3910f2396f",
              "polygons": "3a1fbd3a1652b2ffbc87b1392d1f408f27ab5525",
              "textures": "7f563df006f2691e91845764f281e7deda45b176"}
"""`release_set(..., direction=False)`: v0.1.2's release set byte for byte (RELEASE_SHA1's pin before M9 stage 3)."""


def _release_plain(romset_zip, disc) -> RomSet:  # noqa: F811
    """release_set(..., direction=False) of the player's set and disc (v0.1.2's chain), built once per session."""
    key = ("plain", str(romset_zip))
    if key not in _RELEASE:
        _RELEASE[key] = R.release_set(RomSet.from_zip(romset_zip), disc, direction=False)
    return _RELEASE[key]


def test_release_plain_is_v012(romset_zip, disc):  # noqa: F811
    """`direction=False` builds v0.1.2's release set, PLAIN_SHA1 region for region: M9 stage 3 adds its steps and
    moves no byte of the chain before them."""
    import hashlib
    rel = _release_plain(romset_zip, disc)
    assert {r: hashlib.sha1(getattr(rel, r)).hexdigest() for r in PLAIN_SHA1} == PLAIN_SHA1


def test_release_without_far_view_is_v02(romset_zip, disc):  # noqa: F811
    """`far_view=False` builds v0.2's release set, RELEASE_SHA1_V02 region for region: the far view adds its steps
    (and leaves the thinning out) and moves no byte of the chain before them."""
    import hashlib
    rel = R.release_set(RomSet.from_zip(romset_zip), disc, far_view=False)
    assert {r: hashlib.sha1(getattr(rel, r)).hexdigest() for r in RELEASE_SHA1_V02} == RELEASE_SHA1_V02


def test_release_set_pinned(romset_zip, disc):  # noqa: F811
    """The release set is RELEASE_SHA1 region for region: the bytes Phase 12 ships (the eight courses, the names, the
    laps and painted maps, the race times and pit stalls of 3-7 (v0.1.2), the title, the views) cannot move without
    this pin moving."""
    import hashlib
    rel = _release(romset_zip, disc)
    assert {r: hashlib.sha1(getattr(rel, r)).hexdigest() for r in RELEASE_SHA1} == RELEASE_SHA1


def _m5b(romset_zip, disc) -> RomSet:  # noqa: F811
    """The M5b playtest set (tools/desert_city_playtest.py `m5b_set`): the release chain with Phase 11's row, built
    once per session."""
    key = ("m5b", str(romset_zip))
    if key not in _RELEASE:
        _RELEASE[key] = _tool().m5b_set(RomSet.from_zip(romset_zip), disc)
    return _RELEASE[key]


def test_release_differs_only_in_names_and_title(romset_zip, disc):  # noqa: F811
    """The release set (`release_set`: the names on the select, the race times of 3-7, the title) against the
    M5b playtest set (tools/desert_city_playtest.py `m5b_set`, the same chain with Phase 11's row and no title): every
    byte that differs lies in what patch_select writes on one or the other (its data, `layout_of`; its code at CODE_AT;
    its sites) or in the title's spans (`title.title_spans`: TITLE_RUN's start, the two repointed words), and every
    course id reads the same view-test tables (`course.view_tables`). Since Task 6 the bytes include the own panels'
    maps in the polygon and textures regions. The title on the release set: `titled`, its spans owned
    (golden.owned_spans) and clear of the select's, TITLE_RUN 0xff on the M5b set and owned by nothing there, its logo
    `extended_pixels(PICKED)` of the M5b set's (the select does not touch the logo) pixel for pixel, and of every upload
    set the release set names only the title's copy writes chars of title.ROOM or palettes 0x85-0xf5
    (tests/test_title.py `_hole_uploads`)."""
    from tests.test_title import _hole_uploads
    rel = _release_plain(romset_zip, disc)            # M9 stage 3's steps aside (test_release_plain_is_v012)
    ref = _m5b(romset_zip, disc)
    spans = TT.title_spans(rel)
    title = {(r, o + i) for r, o, n, _ in spans for i in range(n)}
    select = _select_bytes(rel, S.ENTRIES) | _select_bytes(ref, S.M5B_ENTRIES)
    assert TT.titled(rel) and not TT.titled(ref) and len(spans) == 3 and not title & select
    owned = G.owned_spans(rel)
    assert all(sp in owned for sp in spans)
    assert ref.data[TT.TITLE_RUN[0]:sum(TT.TITLE_RUN)] == b"\xff" * TT.TITLE_RUN[1]
    assert G.owner(ref, "data", *TT.TITLE_RUN) is None
    lg = TT.logo(ref)
    want = TT.palette_rgb(TT.palette_ram(lg))[TT.extended_pixels(ref, TT.PICKED)]
    assert np.array_equal(TT.render(rel, TT.logo(rel)), want)
    chars, pals = _hole_uploads(rel)
    upload = TT._u32(rel, TT.BOOT_WORD)
    assert [a for a, _ in chars] == [upload] and [a for a, _ in pals] == [upload]
    times = {("data", o) for o in _time_spans(rel, R.RELEASE_TIME_IDS)}     # v0.1.2: race times, M5b: long clock
    pits = {("data", o) for o in _pit_spans(rel, R.RELEASE_PIT_IDS)}        # v0.1.2: the pit stalls of 3-7
    assert not pits & (select | title | times)
    allowed = select | title | times | pits
    changed = {}
    for region in REGIONS:
        a, b = getattr(rel, region), getattr(ref, region)
        assert len(a) == len(b), region
        d = np.flatnonzero(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8))
        changed[region] = d.size
        stray = [hex(int(i)) for i in d if (region, int(i)) not in allowed]
        assert not stray, (region, len(stray), stray[:8])
    print(f"\nbytes differing from the M5b playtest set: {changed}; the title's spans {spans}")
    assert all(changed[r] for r in ("program", "data", "polygons", "textures"))
    assert not any(changed[r] for r in REGIONS if r not in ("program", "data", "polygons", "textures"))
    assert all(C.view_tables(rel, k) == C.view_tables(ref, k) for k in range(C.COURSE_IDS))
    assert rel.extra == ref.extra and rel.custom and ref.custom
    assert S.select_row(rel).live == (True,) * 8


def _release_short(romset_zip, disc) -> RomSet:  # noqa: F811
    """release_set(..., long_ids=(), times=False) of the player's set and disc (with direction=False, the set the
    gates race: test_release_mame.py `test_release_gates`; the converted courses keep Beginner's words), built once
    per session."""
    key = ("short", str(romset_zip))
    if key not in _RELEASE:
        _RELEASE[key] = R.release_set(RomSet.from_zip(romset_zip), disc, long_ids=(), times=False)
    return _RELEASE[key]


def _time_spans(s: RomSet, ids) -> set[int]:
    """Data offsets of every byte of the time tables of `ids` on `s`."""
    return {t - DATA_BASE + i for k in ids for _, t, w in G.decode_course(s, k).time_tables.values()
            for i in range(4 * len(w))}


def test_release_race_times(romset_zip, disc):  # noqa: F811
    """v0.1.2: the release set against `release_set(..., long_ids=(), times=False)` (the set the gates race): the
    bytes that differ all lie in the time tables of ids 3-7 (RELEASE_TIME_IDS) and, M9 stage 4, 8-15
    (REVERSE_TIME_IDS), no other region differs; on the
    release set each of those courses' tables reads build.RACE_TIMES word for word at the same addresses and with the
    same lengths (1 + laps x (checkpoints + 1) words per mode), golden.check_course passes on it, and the originals'
    (ids 0-2) tables equal the stock set's and share no byte with them. The title's spans are the same on both."""
    rel, short = _release(romset_zip, disc), _release_short(romset_zip, disc)
    stock = RomSet.from_zip(romset_zip)
    spans = _time_spans(short, (R.RELEASE_TIME_IDS + R.REVERSE_TIME_IDS))
    for region in REGIONS:
        a, b = getattr(short, region), getattr(rel, region)
        assert len(a) == len(b), region
        if region != "data":
            assert a == b, region
    diff = set(np.flatnonzero(np.frombuffer(short.data, np.uint8) != np.frombuffer(rel.data, np.uint8)).tolist())
    assert diff and diff <= spans
    assert sorted(build.RACE_TIMES) == list(R.RELEASE_TIME_IDS + R.REVERSE_TIME_IDS)
    for k in (R.RELEASE_TIME_IDS + R.REVERSE_TIME_IDS):
        c, before = G.decode_course(rel, k), G.decode_course(short, k)
        assert {key: v[:2] for key, v in c.time_tables.items()} == {key: v[:2] for key, v in before.time_tables.items()}
        assert {key: v[2] for key, v in c.time_tables.items()} == {
            key: build.race_table(*build.RACE_TIMES[k][key], len(v[2])) for key, v in before.time_tables.items()}, k
        rows = [cp for cp in c.checkpoints if not cp.lap_line]
        for (m, d), (_, _, words) in c.time_tables.items():
            assert len(words) == len(before.time_tables[(m, d)][2]) == 1 + c.laps[m][1] * (len(rows) + 1), (k, m, d)
        G.check_course(c, rel)
    for k in range(3):
        assert ({key: v[1:] for key, v in G.decode_course(rel, k).time_tables.items()} ==
                {key: v[1:] for key, v in G.decode_course(stock, k).time_tables.items()}), k
    assert not _time_spans(rel, range(3)) & spans
    assert TT.title_spans(rel) == TT.title_spans(short) != []


def _pit_spans(s: RomSet, ids) -> set[int]:
    """Data offsets of every byte build.pit_stalls writes for `ids` on `s`: each id's CAR_TABLES_230E50 entry and
    the stalls it names (course.PIT_STALLS_SIZE bytes from course.pit_stalls_at)."""
    out = set()
    for k in ids:
        entry = C.id_table(s, "CAR_TABLES_230E50") + 4 * k - DATA_BASE
        at = C.pit_stalls_at(s, k) - DATA_BASE
        out |= set(range(entry, entry + 4)) | set(range(at, at + C.PIT_STALLS_SIZE))
    return out


def test_release_pit_stalls(romset_zip, disc):  # noqa: F811
    """v0.1.2: ids 3-7 (RELEASE_PIT_IDS) read the Dreamcast's pit stalls of their course (dc.exe.pit_stalls, the
    plain group: 8 stalls of 7 waypoints x, z, speed; x moved by build.SHIFT as the course, Rin Rin Rink's by -64),
    each in its own block at the end of ID_DATA_FREE (build.last_fit), padded with 0; the originals (ids 0-2) keep the stock set's. Each course's stalls lie on its pit
    lane and run with it: each stall's first five waypoints (to the one before the stop; the last two are the stop and
    the lane's end, never reached) are within PIT_LANE_REACH of a pit polygon's centre, and the segments of those
    polygons, counted from the pit entry, never step back (the R group's step back: the check's control). Before v0.1.2 every id 3-15 read Beginner's stalls: the stop
    (0xe384) skipped all seven waypoints in 7 frames (spike/pit_drive.py, docs/follow-ups/v0.1.2.md)."""
    from dtools.arcade import tgp
    from dtools.dc import exe as X
    rel, stock = _release(romset_zip, disc), RomSet.from_zip(romset_zip)
    blob = disc.read(X.EXE_NAME)
    assert all(C.pit_stalls(rel, k) == C.pit_stalls(stock, k) for k in range(3))
    blocks = []
    for k in R.RELEASE_PIT_IDS:
        got = C.pit_stalls(rel, k)
        dx = build.SHIFT.get(k, 0.0)                    # Rin Rin Rink's x moved by -64, as the course
        assert got == tuple(tuple(C.PitWaypoint(float(np.float32(w.x + dx)), w.z, w.speed) for w in st)
                            for st in X.pit_stalls(blob, k)), k
        assert got != C.pit_stalls(stock, 0)
        at = C.pit_stalls_at(rel, k)
        assert build.ID_DATA_FREE[0] <= at - DATA_BASE <= sum(build.ID_DATA_FREE) - C.PIT_STALLS_SIZE
        blocks.append(at)
        c = G.decode_course(rel, k)
        g = c.ground_data or tgp.decode_ground(rel, k)
        n = len(c.road)
        idx = c.pit.lane_polygons + (c.pit.box_polygon,)
        cen = np.array([[(g.polys[i].centre or np.mean(g.polys[i].corners, axis=0))[a] for a in (0, 2)] for i in idx])
        seg = np.array([c.steps[i].segment for i in idx])
        for v, forward in (("", True), ("R", False)):
            for st in (got if v == "" else X.pit_stalls(blob, k, "R")):
                w = np.array([(p[0] + (dx if v else 0.0), p[1]) for p in st[:5]])
                d = np.linalg.norm(w[:, None, :] - cen[None, :, :], axis=2)
                assert not forward or d.min(axis=1).max() <= PIT_LANE_REACH, (k, v, d.min(axis=1))
                run = [int((s - c.pit.entry_segment) % n) for s in seg[d.argmin(axis=1)]]
                assert (run == sorted(run)) == forward, (k, v, run)
    assert all(abs(a - b) >= C.PIT_STALLS_SIZE for a in blocks for b in blocks if a != b) and len(set(blocks)) == 5


PIT_LANE_REACH = 25.0
"""World units: how far a pit stall's waypoint may lie from the nearest pit polygon's centre (the lane polygons are
up to about 40 units long); measured once the test passed: see docs/follow-ups/v0.1.2.md, "Pit stops"."""


RACE_TIMES_MEASURED = Path(__file__).resolve().parent / "data" / "race_times_measured.json"
"""The AI field's sector times on courses 0-7 (frame counts only; spike/race_times.py, docs/spike/2026-11-race-times.md)."""


def race_times_inputs(stock: RomSet, short: RomSet):
    """racetimes.derive's inputs besides the measurements: the originals' time tables (ids 0-2, from the stock set),
    the laps per mode and the checkpoint count (lap line not counted) of ids 0-7 (from the release set)."""
    originals = {k: {key: v[2] for key, v in G.decode_course(stock, k).time_tables.items()} for k in range(3)}
    laps, cps = {}, {}
    for k in range(16):                                 # M9 stage 4: the reverses too
        c = G.decode_course(short, k)
        laps[k] = {m: v[1] for m, v in c.laps.items()}
        cps[k] = sum(not cp.lap_line for cp in c.checkpoints)
    return originals, laps, cps


def test_race_times_frozen(romset_zip, disc):  # noqa: F811
    """build.RACE_TIMES equals racetimes.derive on the recorded measurements (RACE_TIMES_MEASURED), the stock set's
    original tables and the release set's laps and checkpoints: the frozen numbers cannot drift from their
    derivation."""
    import json
    from dtools.convert import racetimes
    measured = json.loads(RACE_TIMES_MEASURED.read_text())
    originals, laps, cps = race_times_inputs(RomSet.from_zip(romset_zip), _release_short(romset_zip, disc))
    derived = racetimes.derive(measured, originals, laps, cps, racetimes.CONVERTED + racetimes.REVERSE)
    assert sorted(derived) == sorted(build.RACE_TIMES) == list(range(3, 16))
    for k, tabs in derived.items():
        assert sorted(tabs) == sorted(build.RACE_TIMES[k]), k
        for key, words in tabs.items():
            assert build.race_table(*build.RACE_TIMES[k][key], len(words)) == words, (k, key)

RELEASE_SET_UPLOAD = (31, 24, 29, 126)
"""The release set's number (210 chars), not the bare grown set's 188 (RELEASE_UPLOAD in tests/test_select.py): on the
release set the converted courses' bands are MEDIUM and SHORT, on the grown set all SHORT, so more digit chars are cut
there. The release set's select upload per run (graphics(release set, ENTRIES)): the originals' three schemes, then
NEW_SCHEME with the own panels' names and their six bands (MEDIUM and SHORT x 8 / 20 / 80), 210 chars in all, inside
UPLOAD_MAX 288 and the four runs of UPLOAD_RUNS_MAX (NEW_SCHEME's block holds 128)."""
RELEASE_LENGTHS = {3: 1, 4: 1, 5: 0, 6: 1, 7: 1}
"""The length panel of each converted course (select.length_panel: the original whose lap is nearest in length): Desert
City 3,460 world units, National Park Speedway 2,986, Rin Rin Rink 2,990 and Mermaid Lake 3,845 nearest Dinosaur
Canyon's 3,820 (MEDIUM), Circuit Pixie 1,644 nearest Three Seven Speedway's 1,555 (SHORT); Sea-Side Street Galaxy's
is 7,688 (LONG)."""


def test_release_bands_and_maps(romset_zip, disc):  # noqa: F811
    """Task 6 on the release set: entries 3-7 draw their own panels; per game mode each band reads `<LENGTH> x <N>
    LAPS` with LENGTH the nearest original's (RELEASE_LENGTHS) and N the set's own laps for that course and mode
    (course.laps: 8 / 20 / 80, the slot's until M6 tunes them), the upload RELEASE_SET_UPLOAD; each converted course's
    map record names a list of one object, selmap.course_map's of the set (its road, its scenery's painting: Task 7).
    The maps' spans (select.map_spans: lists, objects, headers) are owned on the release set (golden.owned_spans) and
    were free on the input: on the M5b playtest set (the same chain but the row) every byte is 0xff and owned by
    nothing."""
    from dtools.arcade import polyrom as P
    from dtools.arcade import selmap as M
    from tests.test_select import _pens, _read_band
    rel = _release(romset_zip, disc)
    g = S.graphics(rel, S.ENTRIES)
    art = S.panel_art(rel)
    table = C.course_table(rel)
    print(f"\nthe release set's upload: {g.uploaded()} chars in {[n for _, n in g.runs]}; bands {g.bands[3:]}")
    assert tuple(n for _, n in g.runs) == RELEASE_SET_UPLOAD and g.uploaded() <= S.UPLOAD_MAX
    assert {e.course: S.length_panel(rel, e.course) for e in S.ENTRIES[3:]} == RELEASE_LENGTHS
    row = S.select_row(rel)
    for k, e in enumerate(S.ENTRIES[3:], 3):
        for mode in range(C.MODES):
            laps = C.laps(rel, table[e.course], mode)
            tiles = TL.tile_map(rel, row.records[mode][k].maps[0]).tiles()
            assert _read_band(art, _pens(g, tiles, range(6, 9))) == (RELEASE_LENGTHS[e.course], laps), (k, mode)
            assert tiles == g.panels[k][mode][0]
    parts = {}
    for r in S.map_records(rel):
        e = S.ENTRIES[r.first + r.position]
        if e.own_panel:
            (q,) = C.object_list(rel, r.objects)
            obj = P.decode_object(rel.polygons, C.check_rom_object(q.oba, rel), q.obc, S.MAP_MODE)
            want, heads = M.course_map(rel, e.course)
            assert obj == want and rel.textures[2 * q.tha:2 * q.tha + len(heads)] == heads
            parts[e.course] = len(heads) // 8
    spans = S.map_spans(rel)
    assert len(spans) == 5 * 2 + sum(parts.values()) and sorted({r for r, *_ in spans}) == ["data", "polygons",
                                                                                              "textures"]
    owned = G.owned_spans(rel)
    assert all(sp in owned for sp in spans)
    ref = _m5b(romset_zip, disc)
    taken = G.owned(ref)
    for region, off, n, owner in spans:
        assert getattr(ref, region)[off:off + n] == b"\xff" * n, (region, hex(off), owner)
        rows = taken.get(region, np.zeros((0, 2), np.int64))
        assert not ((rows[:, 0] < off + n) & (rows[:, 1] > off)).any(), (region, hex(off), owner)


def test_release_reverse_pit_stalls(romset_zip, disc):  # noqa: F811
    """M9 stage 4: ids 8-15 read the Dreamcast's R stalls of course n (dc.exe.pit_stalls(..., n, "R"), x moved by
    build.SHIFT[n] as the course: none for the arcade's three, whose plain stalls are the arcade's own), each in its
    own block at the end of ID_DATA_FREE; on id n + 8's pit lane they run forward (the segments of the nearest pit
    polygons, counted from the reverse's pit entry, never step back) and lie within PIT_LANE_REACH of a pit polygon;
    ids 0-7 keep their stalls (the plain release's)."""
    from dtools.arcade import tgp
    from dtools.dc import exe as X
    rel = _release(romset_zip, disc)
    plain = _release_plain(romset_zip, disc)
    blob = disc.read(X.EXE_NAME)
    assert all(C.pit_stalls(rel, k) == C.pit_stalls(plain, k) for k in range(8))
    blocks = []
    for n in range(8):
        k = n + 8
        got = C.pit_stalls(rel, k)
        dx = build.SHIFT.get(n, 0.0)
        assert got == tuple(tuple(C.PitWaypoint(float(np.float32(w.x + dx)), w.z, w.speed) for w in st)
                            for st in X.pit_stalls(blob, n, "R")), k
        at = C.pit_stalls_at(rel, k)
        assert build.ID_DATA_FREE[0] <= at - DATA_BASE <= sum(build.ID_DATA_FREE) - C.PIT_STALLS_SIZE
        blocks.append(at)
        c = G.decode_course(rel, k)
        g = c.ground_data or tgp.decode_ground(rel, k)
        size = len(c.road)
        idx = c.pit.lane_polygons + (c.pit.box_polygon,)
        cen = np.array([[(g.polys[i].centre or np.mean(g.polys[i].corners, axis=0))[a] for a in (0, 2)] for i in idx])
        seg = np.array([c.steps[i].segment for i in idx])
        for st in got:
            w = np.array([(p.x, p.z) for p in st[:5]])
            d = np.linalg.norm(w[:, None, :] - cen[None, :, :], axis=2)
            assert d.min(axis=1).max() <= PIT_LANE_REACH, (k, d.min(axis=1))
            run = [int((s_ - c.pit.entry_segment) % size) for s_ in seg[d.argmin(axis=1)]]
            assert run == sorted(run), (k, run)
    assert len(set(blocks)) == 8 and all(abs(a - b) >= C.PIT_STALLS_SIZE for a in blocks for b in blocks if a != b)
