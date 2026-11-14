"""dtools/release.py (Phase 12, M7): the release set's chain (`release_set`). Offline: the chain's steps replaced by
recorders (no game data); with the data, the release set against the M5b playtest set (tools/desert_city_playtest.py
`m5b_set`): the same chain but for the select's row and the title, so every byte that differs is one patch_select or
title.extended_title writes."""

import struct

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
    monkeypatch.setattr(TT, "extended_title", lambda s, *a, **k: calls.append(
        ("extended_title", (s, *a), tuple(sorted(k.items())))) or f"{s}+title")


def test_release_chain(monkeypatch):
    """`release_set` runs grow_ids, guard_new_ids, widen_ids, converts 3, 4, 7, 6 and 5 at their own ids
    (`converted_set(..., n, slot=n, scenery="model")`), grow_rankings, the select with the release row
    (select.ENTRIES itself), `build.long_time` on each of 3-7 (RELEASE_LONG_IDS), the title (`title.extended_title` on
    the last long_time's set, its variant the default, title.PICKED) and `build.viewed_set` last: the M5b playtest
    set's steps (tools/desert_city_playtest.py `m5b_set`) on the same inputs, the select's row and the title aside.
    `long_ids=()` drops only the long_time calls, `title=False` (test-only) only the title; a long id outside 3-7 (an original's times are the arcade's) and
    inputs that are not a RomSet and a DcDisc are ReleaseError (a DecodeError) before any step runs."""
    rs, dc = _inputs()
    calls = _record(monkeypatch)
    _record_long_times(monkeypatch, calls)
    R.release_set(rs, dc)
    names = [c[0] for c in calls]
    assert names == (["grow_ids", "guard_new_ids", "widen_ids"] + ["converted_set"] * 5 + ["grow_rankings",
                     "patch_select"] + ["decode", "long_time"] * 5 + ["extended_title", "viewed_set"])
    assert R.RELEASE_CONVERTED == (3, 4, 7, 6, 5) and R.RELEASE_LONG_IDS == (3, 4, 5, 6, 7)
    conv = [c for c in calls if c[0] == "converted_set"]
    assert [c[1][1:] for c in conv] == [(dc, n) for n in (3, 4, 7, 6, 5)]
    assert [c[2] for c in conv] == [(("scenery", "model"), ("slot", n)) for n in (3, 4, 7, 6, 5)]
    assert calls[0][1] == (rs,)
    sel = calls[names.index("patch_select")]
    assert sel[1][0] == "grow_rankings#9" and sel[1][1] is S.ENTRIES and sel[2] == ()
    assert [c[1][1] for c in calls if c[0] == "decode"] == [3, 4, 5, 6, 7]
    assert calls[-2] == ("extended_title", ("patch_select#10+c3+c4+c5+c6+c7",), ())
    assert calls[-1] == ("viewed_set", ("patch_select#10+c3+c4+c5+c6+c7+title",), ())
    # the M5b playtest chain, step for step, but for the row and the title
    full, calls[:] = list(calls), []
    _tool().m5b_set(rs, dc)
    m5b = [c if c[0] != "patch_select" else (c[0], (c[1][0], S.ENTRIES), c[2]) for c in calls]
    assert calls[names.index("patch_select")][1][1] is S.M5B_ENTRIES and m5b[:-1] == full[:-2]
    assert m5b[-1] == ("viewed_set", ("patch_select#10+c3+c4+c5+c6+c7",), ())
    # long_ids=(): the same steps without the start words
    calls[:] = []
    R.release_set(rs, dc, ())
    assert [c for c in full if c[0] not in ("decode", "long_time")][:-2] == calls[:-2]
    assert calls[-2:] == [("extended_title", ("patch_select#10",), ()), ("viewed_set", ("patch_select#10+title",), ())]
    calls[:] = []
    R.release_set(rs, dc, (), title=False)
    assert [c for c in full if c[0] not in ("decode", "long_time", "extended_title")][:-1] == calls[:-1]
    assert calls[-1] == ("viewed_set", ("patch_select#10",), ())
    calls[:] = []
    R.release_set(rs, dc, (7, 3))
    assert [c[1][1] for c in calls if c[0] == "decode"] == [7, 3]
    # refusals, before any step
    for bad in ((rs, dc, (2,)), (rs, dc, (8,)), (rs, dc, (True,)), (rs, dc, (3.0,)), (rs, dc, 3), (rs, dc, "3"),
                (rs, dc, None), ("stock", dc), (None, dc), (rs, "disc"), (rs, None)):
        calls[:] = []
        with pytest.raises(R.ReleaseError):
            R.release_set(*bad)
        assert calls == [], bad
    for bad in (1, None, "no"):
        with pytest.raises(R.ReleaseError):
            R.release_set(rs, dc, (), title=bad)
        assert calls == [], bad
    with pytest.raises(TypeError):
        R.release_set(rs, dc, (), False)         # title is keyword-only
    assert issubclass(R.ReleaseError, DecodeError)


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


RELEASE_SHA1 = {"program": "1f6a3b04d75b03489a2f5c1278cbd299af917ef7",
                "data": "c28afcee5ec4e513b913e47e4a4551ac47bf993a",
                "copro_data": "06e8923fa71bf8f5935b89c746058e3910f2396f",
                "polygons": "c80e616e717766a66d565f7b653d7f4af205d513",
                "textures": "428ceda679d7bd41e9767d0720764b937ef84488"}
"""The release set (`release_set`: the M5b chain, the named row, the start words of 3-7, the title, the views; Task 7's
painted maps on the select), the sha1 of each region, built independently of the code under test: ed570c2 (Task 5's
first commit) from a git archive into a scratch folder, a fresh venv on the repository's Python 3.15.0 with `pip install
-e .` only (numpy 2.5.4, no dev extras), DAYTONA_OUT an empty folder, its own `dtools build` (cold: the disc's
extraction and the three re-meshes, 407 s; out/phase12/t5/release_ref_sha1.txt). The in-repository build of the same
commit wrote the same 41 members byte for byte. Phase 13 Task 2 round 2 repinned it: model._merge's quad tests made
exact (5b970ee; four National Park Speedway pairs on the shear bound, three of which the float lstsq had rejected on
this Mac, now merge), built by 5b970ee's code from a git archive (tools/clean_install.sh: fresh venv, Python 3.15.0,
numpy 2.5.4; out/phase13/r2/pins.txt, its `dtools build` 370 s cold); the Docker linux/amd64 clean install wrote the
same 41 members (docs/spike/2026-11-page.md, "The clean install")."""


def test_release_set_pinned(romset_zip, disc):  # noqa: F811
    """The release set is RELEASE_SHA1 region for region: the bytes Phase 12 ships (the eight courses, the names, the
    laps and painted maps, the long clock on 3-7, the title, the views) cannot move without this pin moving."""
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
    """The release set (`release_set`: the names on the select, the start words of 3-7 raised, the title) against the
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
    rel = _release(romset_zip, disc)
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
    allowed = select | title
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
    """release_set(..., long_ids=()) of the player's set and disc (the set the gates race: test_release_mame.py
    `test_release_gates`), built once per session."""
    key = ("short", str(romset_zip))
    if key not in _RELEASE:
        _RELEASE[key] = R.release_set(RomSet.from_zip(romset_zip), disc, long_ids=())
    return _RELEASE[key]


def test_release_long_ids_only_start_words(romset_zip, disc):  # noqa: F811
    """`release_set(...)` against `release_set(..., long_ids=())` (the set the gates race): the bytes that differ are
    exactly those of the first word of each time table of ids 3-7 (RELEASE_LONG_IDS) that build.LONG_TIME changes, each
    such word reads LONG_TIME on the release set and not on the other, and no time table of the originals (ids 0-2)
    shares one (Phase 11's test_m5b_set_bytes pattern). The title's spans are the same on both."""
    rel, short = _release(romset_zip, disc), _release_short(romset_zip, disc)

    def words(s: RomSet, k: int) -> set[int]:
        return {t - DATA_BASE for _, t, _ in G.decode_course(s, k).time_tables.values()}
    raised = set().union(*(words(short, k) for k in R.RELEASE_LONG_IDS))
    new = struct.pack("<I", build.LONG_TIME)
    expected = {w + i for w in raised for i in range(4) if short.data[w + i] != new[i]}
    assert expected
    for region in REGIONS:
        a, b = getattr(short, region), getattr(rel, region)
        assert len(a) == len(b), region
        if region != "data":
            assert a == b, region
    diff = set(np.flatnonzero(np.frombuffer(short.data, np.uint8) != np.frombuffer(rel.data, np.uint8)).tolist())
    assert diff == expected
    assert all(rel.data[w:w + 4] == new != short.data[w:w + 4] for w in raised)
    assert all(not raised & words(short, k) for k in range(C.COURSE_COUNT))
    assert TT.title_spans(rel) == TT.title_spans(short) != []


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
