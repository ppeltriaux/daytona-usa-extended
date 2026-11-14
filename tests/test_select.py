"""The sixteen-entry Circuit Select (dtools/arcade/select.py) and its i960 encoder (dtools/arcade/i960.py), offline.
The MAME tests are in tests/test_sixteen_mame.py."""

import dataclasses
import re
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from dtools.arcade import course as C
from dtools.arcade import disasm
from dtools.arcade import i960
from dtools.arcade import select as S
from dtools.arcade import tiles as TL
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet
from dtools.config import REPO_ROOT


# --- the encoder ---------------------------------------------------------------------------------------------------

ROM_WORDS = [  # (encoding, the program window's bytes at that address, as unidasm prints them)
    (lambda: i960.mem("ldob", "r3", 0x501460), "0030188060145000"),                       # 0x225e9c
    (lambda: i960.mem("ld", "g0", 0x23899C, None, "r3", 4), "033980909c892300"),          # 0x225ea4
    (lambda: i960.mem("ld", "r3", 0x34, "g13"), "34601f90"),                              # 0x2262cc
    (lambda: i960.mem("lda", "g14", 0x505), "0505f08c"),                                  # 0x2262f4
    (lambda: i960.mem("ld", "g0", 0x14, "r12", "r11", 8), "8b3d839014000000"),           # 0x2269d4
    (lambda: i960.mem("lda", "g0", 0x23862C, "r3"), "00f4808c2c862300"),                  # 0x22564c
    (lambda: i960.mem("stos", "r4", 0x501468), "0030208a68145000"),                       # 0x225fa4
    (lambda: i960.mem("stob", "r4", 0x5E, "g13"), "5e602782"),                            # 0x225fac
    (lambda: i960.mem("ldob", "g0", 0x238634, "g0"), "0034848034862300"),                 # 0x226c7c
    (lambda: i960.mem("ld", "r5", None, "r10"), "00902a90"),                              # 0x22624c
    (lambda: i960.reg("mov", 0, "r3"), "001e185c"),
    (lambda: i960.reg("mov", "r4", "g0"), "0416805c"),
    (lambda: i960.reg("addo", 1, "r4", "r4"), "01082159"),
    (lambda: i960.reg("shlo", 2, 27, "r5"), "02de2e59"),
    (lambda: i960.reg("subi", "g9", 3, "g2"), "99d19059"),
    (lambda: i960.reg("mulo", "r3", "r4", "r3"), "83001970"),
    (lambda: i960.reg("cmpi", "r6", "r4"), "8620015a"),
    (lambda: i960.reg("cmpo", "r3", 4), "0330015a"),
    (lambda: i960.reg("and", 7, "r3", "r3"), "87c81858"),
    (lambda: i960.cobr("cmpibge", 0, "r3", 20), "14e0003b"),                               # 0x225f90
    (lambda: i960.cobr("cmpobl", 0, "r3", 0x225ED8 - 0x225E8C), "4ce00034"),
    (lambda: i960.ctrl("call", 0x226C74 - 0x225FB4), "c00c0009"),
    (lambda: i960.ctrl("bl", 0x226440 - 0x226534), "0cffff14"),
    (lambda: i960.ctrl("ret"), "0000000a"),
]


@pytest.mark.parametrize("make,want", ROM_WORDS)
def test_i960_encodings_match_the_rom(make, want):
    assert make().hex() == want


def test_i960_block_and_refusals():
    b = i960.Block(0x1000)
    b.label("top")
    b.reg("addo", 1, "r4", "r4")
    b.cobr("cmpobg", 15, "r4", "top")
    b.ctrl("b", "end")
    b.ctrl("call", 0x2000)
    b.label("end")
    b.ctrl("ret")
    if disasm.unidasm().exists():   # the disassembly check needs MAME's unidasm (tools/build-mame.sh); a clean install lacks it
        text = [ln.text for ln in disasm.listing_bytes(b.bytes(), 0x1000)]
        assert text == ["addo    1,r4,r4", "cmpobg  15,r4,0x1000", "b       0x00001010", "call    0x00002000", "ret"]
    for bad in (lambda: i960.reg("mov", 32, "r3"), lambda: i960.reg("mov", True, "r3"),
                lambda: i960.reg("mov", 0, "r16"), lambda: i960.reg("addo", 1, "r3"),
                lambda: i960.reg("nope", 1, "r3"), lambda: i960.cobr("cmpobe", 0, "r3", 0x1000),
                lambda: i960.cobr("cmpobe", 0, "r3", 2), lambda: i960.ctrl("b", 0x800000),
                lambda: i960.ctrl("b", 6), lambda: i960.mem("ld", "r3"), lambda: i960.mem("ld", "r3", 4, scale=3),
                lambda: i960.mem("ld", "r3", True), lambda: i960.Block(2)):
        with pytest.raises(i960.I960Error):
            bad()
    b = i960.Block(0)
    b.ctrl("b", "nowhere")
    with pytest.raises(i960.I960Error):
        b.bytes()
    b.label("x")
    with pytest.raises(i960.I960Error):
        b.label("x")
    assert issubclass(i960.I960Error, DecodeError)


# --- entries, wheel, names -----------------------------------------------------------------------------------------

def test_entries_are_the_ruled_row():
    """Phase 11's row (Phase 11 Task 2; M5B_ENTRIES since Phase 12 Task 1 froze it): the eight courses, no mirror
    (mirror mode dropped, Pascal 2026-11-11); the frozen rows of sixteen (M5A_ENTRIES, M4B_ENTRIES, M4A_ENTRIES) as
    Phases 8-10 left them. The release row, ENTRIES, is test_release_row's."""
    courses = ["BEGINNER", "ADVANCED", "EXPERT", "DESERT CITY", "NATIONAL PARK SPEEDWAY", "CIRCUIT PIXIE",
               "RIN RIN RINK", "MERMAID LAKE"]
    assert [e.name for e in S.M5B_ENTRIES] == courses and len(S.M5B_ENTRIES) in S.ROW_LENGTHS
    assert [e.course for e in S.M5B_ENTRIES] == [0, 2, 1, 3, 4, 5, 6, 7]
    assert [e.mirror for e in S.M5B_ENTRIES] == [False] * 8
    # Desert City live since Phase 8 Task 5, National Park Speedway since Phase 10 Task 3 (build.converted_set at id
    # 4), Mermaid Lake since Phase 10 Task 4 (at id 7), Rin Rin Rink since Phase 11 Task 3 (at id 6) and Circuit Pixie
    # since Phase 11 Task 4 (at id 5), both with their bands empty: all eight live
    assert [e.live for e in S.M5B_ENTRIES] == [True] * 8
    soon = tuple(S.M5A_ENTRIES[k]._replace(label="", live=True) for k in (5, 6))
    assert S.M5B_ENTRIES == S.M5A_ENTRIES[:5] + soon + S.M5A_ENTRIES[7:8]    # the row of 16's first half, mirrors off
    names = [e.name for e in S.M5A_ENTRIES]
    assert names == courses + courses
    assert [e.course for e in S.M5A_ENTRIES[8:]] == [e.course + 8 for e in S.M5A_ENTRIES[:8]]
    assert [e.mirror for e in S.M5A_ENTRIES] == [False] * 8 + [True] * 8
    assert [e.live for e in S.M5A_ENTRIES] == [True] * 5 + [False] * 2 + [True] * 5 + [False] * 4
    # Phase 9's row is frozen and checked by value, independent of this row (test_pinned_rows_frozen)
    old4 = S.Entry(4, "NATIONAL PARK SPEEDWAY", "COMING SOON", False, False)
    old7 = S.Entry(7, "MERMAID LAKE", "COMING SOON", False, False)
    # Phase 10's entries 12 and 15 stayed locked though Phase 10 Task 5 wrote their mirrors (Pascal, 2026-11-11)
    assert S.M5A_ENTRIES[12] == old4._replace(course=12, mirror=True, label="", difficulty="CIRCUIT EDITION")
    assert S.M5A_ENTRIES[15] == old7._replace(course=15, mirror=True)
    assert [e.live for e in S.M4B_ENTRIES] == [True] * 4 + [False] * 4 + [True] * 4 + [False] * 4
    assert S.check_entries(S.M4B_ENTRIES)
    # Phase 8's row (the M4a playtest set, which has no mirror): Phase 9's entries, every mirror locked
    assert S.M4A_ENTRIES == tuple(e._replace(live=e.live and not e.mirror) for e in S.M4B_ENTRIES)
    assert [e.live for e in S.M4A_ENTRIES] == [True] * 4 + [False] * 12 and S.check_entries(S.M4A_ENTRIES)
    assert [k for k, e in enumerate(S.M5B_ENTRIES) if S.is_original(e)] == [0, 1, 2]
    assert [k for k, e in enumerate(S.M5A_ENTRIES) if S.is_original(e)] == [0, 1, 2]
    assert [S.ORIGINAL_PANEL[e.course] for e in S.M5B_ENTRIES[:3]] == [0, 1, 2]
    assert S.check_entries(S.M5B_ENTRIES) == S.M5B_ENTRIES and S.check_entries(S.M5A_ENTRIES) == S.M5A_ENTRIES


def _names_in_assignment(path: Path, target: str) -> set[str]:
    """Every name the module-level assignment to `target` in `path` reads."""
    import ast
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == target for t in targets):
                return {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(node.value) if isinstance(n, ast.Attribute)}
    raise AssertionError(f"no module-level assignment to {target} in {path}")


def test_pinned_rows_frozen():
    """Phase 10 Task 5 (the Task 4 review's M5): the rows the pinned sets are built with are literals, so a change to
    the live row (ENTRIES) cannot move them: Phase 9's row (M4B_ENTRIES, M4B_SHA1's set), Phase 8's (M4A_ENTRIES,
    derived from Phase 9's alone), Task 3's (tests/test_mirrors_mame.py TASK3_ENTRIES, FINAL_SHA1's set) and Phase
    10's (M5A_ENTRIES, the M5a playtest set and the tests' final set; Phase 11 Task 1) and Phase 11's (M5B_ENTRIES, the
    shipped set M5B_SHA1 pins and the M5b playtest set; Phase 12 Task 1). Their
    definitions read no ENTRIES, and each equals the row its pinned set was built with."""
    E = S.Entry
    names = ["BEGINNER", "ADVANCED", "EXPERT", "DESERT CITY", "NATIONAL PARK SPEEDWAY", "CIRCUIT PIXIE",
             "RIN RIN RINK", "MERMAID LAKE"]
    ids = [0, 2, 1, 3, 4, 5, 6, 7]
    diff = ["BEGINNER", "ADVANCED", "EXPERT", "CIRCUIT EDITION"]
    soon = [E(k, n, "COMING SOON", False, False) for k, n in zip(ids[4:], names[4:])]
    lead = tuple(E(k, n, "", False, True, d) for k, n, d in zip(ids[:4], names[:4], diff))
    mirrors = tuple(E(k + 8, n, "", True, True, d) for k, n, d in zip(ids[:4], names[:4], diff))
    m4b = lead + tuple(soon) + mirrors + tuple(e._replace(course=e.course + 8, mirror=True) for e in soon)
    assert S.M4B_ENTRIES == m4b
    assert S.M4A_ENTRIES == tuple(e._replace(live=e.live and not e.mirror) for e in m4b)
    from tests.test_mirrors_mame import TASK3_ENTRIES
    park = E(4, names[4], "", False, True, "CIRCUIT EDITION")
    task3 = lead + (park, soon[1], soon[2], soon[3]) + mirrors + (
        park._replace(course=12, mirror=True, live=False), soon[1]._replace(course=13, mirror=True),
        soon[2]._replace(course=14, mirror=True), soon[3]._replace(course=15, mirror=True))
    assert TASK3_ENTRIES == task3
    # Phase 10's row (M5a, Phase 11 Task 1): ENTRIES as main ef7ad08 had it, a literal of its own
    lake = E(7, names[7], "", False, True)
    m5a = lead + (park, soon[1], soon[2], lake) + mirrors + (
        park._replace(course=12, mirror=True, live=False), soon[1]._replace(course=13, mirror=True),
        soon[2]._replace(course=14, mirror=True), E(15, names[7], "COMING SOON", True, False))
    assert S.M5A_ENTRIES == m5a and S.check_entries(S.M5A_ENTRIES)
    assert [e.live for e in S.M5A_ENTRIES] == [True] * 5 + [False] * 2 + [True] + [True] * 4 + [False] * 4
    # Phase 11's row (M5b, Phase 12 Task 1): ENTRIES as main 1684e10 had it, a literal of its own
    m5b = (lead[0], lead[1], lead[2], lead[3], park, E(5, names[5], "", False, True), E(6, names[6], "", False, True),
           lake)
    assert S.M5B_ENTRIES == m5b and S.check_entries(S.M5B_ENTRIES)
    assert all(e.live and not e.mirror for e in S.M5B_ENTRIES) and len(S.M5B_ENTRIES) == 8
    select_py = REPO_ROOT / "dtools" / "arcade" / "select.py"
    for target in ("M4B_ENTRIES", "M4A_ENTRIES", "M5A_ENTRIES", "M5B_ENTRIES"):
        assert "ENTRIES" not in _names_in_assignment(select_py, target), target
    assert "ENTRIES" not in _names_in_assignment(REPO_ROOT / "tests" / "test_mirrors_mame.py", "TASK3_ENTRIES")


def test_wheel_bands():
    seen = [S.entry_of_wheel(w) for w in range(-127, 128)]
    assert seen == sorted(seen) and set(seen) == set(range(16))
    assert min(seen.count(k) for k in range(16)) >= 15
    assert [S.wheel_value(p) for p in (0x50, 0x80, 0x20, 0xE0)] == [-73, 0, -127, 127]   # the measured -73 at 0x50
    for k in range(16):
        assert S.entry_of_wheel(S.wheel_value(S.paddle_for_entry(k))) == k
        assert S.slot_of(k) == (0 if k == 0 else 2 if k == 15 else 1)
        assert S.first_of(k) + S.slot_of(k) == k
    lua = (REPO_ROOT / "dtools/mame/lua/autoplay.lua").read_text()
    table = re.search(r"local ENTRY_WHEEL = \{\[0\] = ([\d, ]+)\}", lua).group(1)
    assert [int(v) for v in table.split(",")] == [S.paddle_for_entry(k) for k in range(16)]
    lead = dict(map(int, kv) for kv in re.findall(r"\[(\d+)\] = (\d+)", re.search(r"local LEAD, LEAD_IN = \{([^}]*)\}",
                                                                              lua).group(1)))
    inside = [k for k in range(16) if abs(S.wheel_value(S.paddle_for_entry(k))) <= S.IDLE_BAND]
    assert sorted(lead) == inside == [7, 8]                   # the entries whose own paddle never leaves the band
    for k, via in lead.items():                               # led in from a neighbour outside it, then followed
        assert abs(k - via) == 1 and abs(S.wheel_value(S.paddle_for_entry(via))) > S.IDLE_BAND
        assert S.follow([S.wheel_value(S.paddle_for_entry(via)), S.wheel_value(S.paddle_for_entry(k))]) == [via, k]
    for bad in (-1, 256, True, 1.0):
        with pytest.raises(S.SelectError):
            S.wheel_value(bad)
    with pytest.raises(S.SelectError):
        S.first_of(16)


def test_names_and_graphics(romset_zip):
    assert S.name_lines("NATIONAL PARK SPEEDWAY") == ("NATIONAL PARK", "SPEEDWAY")
    assert S.name_lines("RIN RIN RINK") == ("RIN RIN RINK",) and S.name_lines("SEA-SIDE STREET GALAXY") == (
        "SEA-SIDE", "STREET GALAXY")                       # the final polish's names fit (docs/follow-ups.md)
    for bad in ("", "  ", "Desert", "A" * 15, "AAAA " * 8, 3, None, "DESERT\tCITY"):
        with pytest.raises(S.SelectError):
            S.name_lines(bad)
    grown, _ = _sets(str(romset_zip))
    g = S.graphics(grown, S.M5A_ENTRIES)                                  # a row of 16 (the row of 8: Task 2's tests)
    assert g.chars[0].tolist() == np.zeros((8, 8)).tolist()               # the hole
    for b in range(0, len(g.chars), 128):                                 # each char once in its block
        used = [c.tobytes() for c in g.chars[b:b + 128] if c.any()]
        assert len(set(used)) == len(used)
    assert S.CHAR_HOLE + len(g.chars) <= S.CHAR_END
    assert g.name_maps[:3] == (None, None, None)
    for k, (e, m) in enumerate(zip(S.M5A_ENTRIES, g.name_maps)):
        if m is None:
            continue
        rows = range(1, 9) if g.windows[k] == S.WINDOW_ALL else range(6, 9)
        inner = [m[r * 18 + c] for r in rows for c in range(1, 17)]
        frame = [t for i, t in enumerate(m) if i // 18 in (0, 9) or i % 18 in (0, 17)]
        assert set(frame) == {S.LAYER1_BLANK}
        assert all(t & S.FRONT and S.CHAR_HOLE <= t & 0x3FFF < S.CHAR_END for t in inner)
        assert all(bool(t & S.GREY_BIT) == (not e.live) for t in inner), k
    # mirrors draw their original's name: the same name chars, a different band
    for o, m in ((3, 11), (4, 12)):
        assert [t & ~S.GREY_BIT for t in g.name_maps[o][:6 * 18]] == [t & ~S.GREY_BIT for t in g.name_maps[m][:6 * 18]]
        assert g.name_maps[o][7 * 18:9 * 18] != g.name_maps[m][7 * 18:9 * 18]
    for flag in (True, False):          # the row of 8: entry 5 live, then locked (its own panels' interiors greyed)
        row = S.ENTRIES[:5] + (S.ENTRIES[5]._replace(live=flag),) + S.ENTRIES[6:]
        own = S.graphics(grown, row).panels[5]
        inner = [t for pair in own for tiles in pair for i, t in enumerate(tiles) if 1 <= i // 18 <= 8 and
                 1 <= i % 18 <= 16]
        assert len(inner) == 3 * 2 * 8 * 16 and all(bool(t & S.GREY_BIT) != flag for t in inner)
    for bad in ("x", None):
        with pytest.raises(S.SelectError):
            S.graphics(bad, S.ENTRIES)
    with pytest.raises(S.SelectError):
        S.panel_pixels(S.panel_art(grown), S.M5B_ENTRIES[0])    # an original with its difficulty keeps its art


def test_check_entries_refuses():
    e = list(S.M5A_ENTRIES)                         # a row of 16 (the row of 8: test_row_of_eight_refusals)
    for bad in (e[:15], e + [e[0]], tuple(e[:15]) + (e[0],), "x", None,
                [e[0]._replace(live=False)] + e[1:],                       # entry 0 is the timer's fallback
                e[:3] + [e[3]._replace(course=True)] + e[4:], e[:3] + [e[3]._replace(course=16)] + e[4:],
                e[:3] + [e[3]._replace(live=1)] + e[4:], e[:3] + [e[3]._replace(mirror=None)] + e[4:],
                e[:3] + [e[3]._replace(name="desert")] + e[4:], e[:3] + [tuple(e[3])] + e[4:],
                e[:3] + [e[3]._replace(label="x")] + e[4:], e[:3] + [e[3]._replace(difficulty=None)] + e[4:]):
        with pytest.raises(S.SelectError):
            S.check_entries(bad)


# --- the patch on the real set -------------------------------------------------------------------------------------

def unconverted_select(rs: RomSet, entries=S.M4A_ENTRIES) -> RomSet:
    """patch_select(rs, entries) (Phase 8's row by default) on a set whose id 3 holds no course of its own (Desert City
    not converted: Beginner's copy). Since Phase 9's final review (item 3) patch_select refuses a live id 3 without its
    course (select.FILLED_FIRST 3); this lifts that check for id 3 alone (ids 4-15 are still checked), so the tests'
    reference sets without Desert City, which race the originals only and never entry 3, keep the bytes they had
    (they predate the check). Not a chain step: a set that races entry 3 converts it first (build.converted_set,
    then patch_select)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "FILLED_FIRST", 4)
        return S.patch_select(rs, entries)


@lru_cache(maxsize=1)
def _sets(path: str):
    """(the grown set with the new ids' guards, it with the select patched with Phase 8's row, M4A_ENTRIES: Desert
    City (id 3) is live, so patch_select needs the guards first (final review fix 6); the mirrors are locked, the set
    has none; id 3 is not converted, so the select goes through `unconverted_select`)."""
    grown = C.guard_new_ids(C.grow_ids(RomSet.from_zip(Path(path))))
    return grown, unconverted_select(grown)


def _changed(a: bytes, b: bytes) -> list[tuple[int, int]]:
    """(start, end) runs where `a` and `b` differ."""
    d = np.flatnonzero(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8))
    runs: list[tuple[int, int]] = []
    for i in d.tolist():
        if runs and i <= runs[-1][1] + 8:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    return runs


def test_select_tables(romset_zip):
    """Sixteen records per mode in row order, 0x238f0c (grown) their inverse, every site naming the new tables, the
    code and data where they belong, the vote in VOTE_RAM, the mirrored maps' lists and objects (map_spans) and the
    object library's lower bound (course.LIBRARY_BOUND_PATCH, which their lists need), and nothing else changed."""
    grown, rs = _sets(str(romset_zip))
    row = S.select_row(rs)
    lay = S.layout_of(rs)
    old = [[C.decode_select_panel(grown.read_i960(t + 44 * p, 44), p) for p in range(3)] for t in C.SELECT_TABLES]
    assert row.tables == lay.tables and lay.tables[0] == S.DATA_AT + S.DIRECTORY_SIZE
    for mode in range(3):
        recs = row.records[mode]
        assert [r.course for r in recs] == [e.course for e in S.M4A_ENTRIES]     # _sets: Phase 8's row
        for k, r in enumerate(recs):
            s = S.slot_of(k)
            assert r.cursor == old[0][s].cursor and r.tile == old[0][s].tile
            w = S.window_of(S.M4A_ENTRIES[k])
            assert r.maps == (lay.hole_maps * 3 if w == S.WINDOW_ALL else
                              old[mode][S.scheme_of(S.M4A_ENTRIES[k])].maps)
    assert row.panel_of_course == tuple(S.M4A_ENTRIES.index(next(e for e in S.M4A_ENTRIES if e.course == c))
                                        for c in range(16))
    assert all(row.records[0][row.panel_of_course[c]].course == c for c in range(16))   # the inverse
    assert row.live == tuple(e.live for e in S.M4A_ENTRIES)       # _sets: Phase 8's row
    assert row.voices[:3] == (0xAE215E, 0xAE215F, 0xAE2160) and set(row.voices[3:]) == {S.VOICE_DEFAULT}
    assert row.bands == S.BANDS and row.vote == S.VOTE_RAM
    # the sites, as unidasm reads them
    want = {0x225448: f"lda     {lay.tables[0]:#x},r13", 0x22564C: f"lda     {lay.tables[0]:#x}(r3),g0",
            0x2263B0: f"lda     {lay.tables[0]:#x},g0", 0x2263C8: f"lda     {lay.tables[1]:#x},g0",
            0x2263D8: f"lda     {lay.tables[2]:#x},g0", 0x2263E0: "mov     16,g1",
            0x226C7C: f"ldob    {lay.tables[0] + 8:#x}(g0),g0", 0x2261C8: f"ld      {lay.voices:#x}[r3*4],g0",
            0x3048: "lda     0x501b03,r4", 0x3050: "lda     0x21,r5", 0x3070: "stob    r3,0x501b04",
            0x3078: "stob    r4,0x501b08", 0x3080: "stob    r5,0x501b06", 0x32B4: "lda     0x501b04,r11",
            0x32BC: "mov     16,g9", 0x32D0: "subi    g9,16,g2", 0x3470: "lda     0x501b03,r11"}
    for a in (0x2256C4, 0x2256F4, 0x22572C, 0x225760, 0x22579C):
        want[a] = want[0x22564C]
    blk = S.code(lay)
    for a, (_, label, _) in S.HOOKS.items():
        op = "call" if label in S.CALLED else "b"
        want[a] = f"{op:8s}{blk.labels[label]:#010x}"
    for a, text in want.items():
        assert disasm.listing(rs, a, 1)[0].text == text, hex(a)
    # the code: every instruction decodes, every branch out lands on the instruction after its site
    lines = disasm.listing_bytes(blk.bytes(), S.CODE_AT)
    assert sum(len(ln.bytes) for ln in lines) == len(blk.bytes()) and not any("?" in ln.text for ln in lines)
    outs = {int(m.group(1), 16) for ln in lines if (m := re.match(r"(?:b|be|bne|call)\s+0x([0-9a-f]+)$", ln.text))}
    assert outs - set(range(S.CODE_AT, S.CODE_AT + len(blk.bytes()))) == {
        0x225FC0, 0x22600C, 0x226478, 0x22646C, 0x22654C, 0x2266F4, 0x226584, 0x226C74, 0x2261C0, S.BLIT,
        0x2263B0, 0x2259C8, *S.IDLE_BACK.values()}
    # nothing else changed; code and data only where they were free
    prog = _changed(grown.program, rs.program)
    sites = {a - 0x200000 if a >= 0x220000 else a for a in want} | {a + 4 - 0x200000 for a in S.TABLE_SITES} | \
        set(C.LIBRARY_BOUND_AT)
    assert C.library_bounds(rs) == C.PATCHED_LIBRARY
    for lo, hi in prog:
        if S.CODE_AT <= lo < S.CODE_FREE[1]:
            assert hi <= S.CODE_AT + len(blk.bytes())
            assert set(grown.program[lo:hi]) == {0xFF}
        else:
            assert any(s <= lo < s + 8 for s in sites), hex(lo)
    poc = C.id_table(rs, "SELECT_PANEL_OF_COURSE") - C.DATA_BASE
    upload = S.SELECT_UPLOAD_WORD - C.DATA_BASE
    spans = S.map_spans(rs)

    def in_spans(region: str, lo: int, hi: int) -> bool:
        return all(any(r == region and o <= x < o + n for r, o, n, _ in spans) for x in range(lo, hi))
    for lo, hi in _changed(grown.data, rs.data):
        if lo >= S.DATA_AT - C.DATA_BASE:
            assert hi <= S.DATA_AT - C.DATA_BASE + S.DATA_SPAN and set(grown.data[lo:hi]) == {0xFF}
        else:
            for x in range(lo, hi):
                if grown.data[x] != rs.data[x]:
                    assert poc <= x < poc + 64 or upload <= x < upload + 4 or (
                        in_spans("data", x, x + 1) and grown.data[x] == 0xFF), hex(x)
    for lo, hi in _changed(grown.polygons, rs.polygons):
        assert in_spans("polygons", lo, hi) and set(grown.polygons[lo:hi]) == {0xFF}, hex(lo)
    for region in ("textures", "copro_data"):
        assert getattr(grown, region) == getattr(rs, region)
    # the upload set: the old one's chars and palettes, then the name chars and their palettes
    old_up = TL.upload_set(grown, struct.unpack("<I", grown.read_i960(S.SELECT_UPLOAD_WORD, 4))[0], False)
    new_up = TL.upload_set(rs, struct.unpack("<I", rs.read_i960(S.SELECT_UPLOAD_WORD, 4))[0], False)
    g = S.graphics(grown, S.M4A_ENTRIES)                    # _sets: Phase 8's row
    n = len(g.runs)
    assert new_up.chars[:-n] == old_up.chars and new_up.palette[:-2] == old_up.palette
    assert [(u.first_char, u.count) for u in new_up.chars[-n:]] == list(g.runs) and g.runs[0][0] == S.CHAR_HOLE
    for u in new_up.chars[-n:]:
        at = u.first_char - S.CHAR_HOLE
        assert TL.decode_chars(u.data(rs)).tolist() == g.chars[at:at + u.count].tolist()
    assert new_up.palette[-2].words == sum(g.palettes, ()) and new_up.palette[-1].words == g.grey * S.BLOCKS
    # the name maps and the placeholder panel
    maps = S.tile_maps(rs, row)
    for k in range(16):
        assert maps[k][1].tiles() == (g.name_maps[k] or (S.LAYER1_BLANK,) * 180)
    hole = maps[3][0]
    beginner = TL.tile_map(grown, old[0][0].maps[0])
    for r in range(10):
        for c in range(18):
            edge = r in (0, 9) or c in (0, 17)
            assert hole.tile(r, c) == (beginner.tile(r, c) if edge else S.FRONT | S.CHAR_HOLE)
    # the 3D map records: the position's place, the entry's object list (the mirrors of 0-2 their own, the
    # generated entries none: test_map_records)
    for first in range(14):
        for s in range(3):
            rec = rs.read_i960(lay.maps + 32 * (3 * first + s), 32)
            assert rec[:12] == grown.read_i960(C.SELECT_MAP_OBJECTS + 32 * s, 12)
            k = first + s
            lst = struct.unpack_from("<I", rec, 12)[0]
            assert lst == (struct.unpack("<I", grown.read_i960(C.SELECT_MAP_OBJECTS + 32 * k + 12, 4))[0]
                           if k < 3 else lst if S.window_of(S.M4A_ENTRIES[k]) == S.WINDOW_LABEL else 0)
    # the window mask bits of each position cover its interior columns
    assert [struct.unpack("<4H", S.slot_mask(p.tile[0])) for p in old[0]] == [
        (0x0FFF, 0xF000, 0, 0), (0, 0x01FF, 0xFE00, 0), (0, 0, 0x003F, 0xFFC0)]


def test_select_assets_and_golden_on_a_patched_set(romset_zip):
    """course.select_assets (golden's select decode) reads a patched set: ids 0-2 as on the grown set, ids 3-15 on
    Beginner's panel; golden decodes id 3."""
    from dtools.arcade import golden
    grown, rs = _sets(str(romset_zip))
    a, b = C.select_assets(grown), C.select_assets(rs)
    assert a.courses[:3] == b.courses[:3] and a.panels == b.panels
    assert [c.panel for c in b.courses[3:]] == [0] * 13
    assert golden.decode_course(rs, 3).course == 3


def test_patch_select_refuses(romset_zip):
    grown, rs = _sets(str(romset_zip))
    stock = RomSet.from_zip(romset_zip)
    for bad_rs in ("x", None, stock, rs):
        with pytest.raises(S.SelectError):
            S.patch_select(bad_rs)
    with pytest.raises(S.SelectError):
        S.patch_select(grown, S.M5A_ENTRIES[:15])
    moved = grown.with_patch("program", 0x2263E0 - 0x200000, i960.reg("mov", 2, "g1"))
    with pytest.raises(S.SelectError, match="mov 3,g1"):
        S.patch_select(moved, S.M4A_ENTRIES)
    taken = grown.with_patch("program", S.CODE_AT + 0x40, b"\0")
    with pytest.raises(S.SelectError, match="not free"):
        unconverted_select(taken)
    taken = grown.with_patch("data", S.DATA_AT - C.DATA_BASE + 0x400, b"\0")
    with pytest.raises(S.SelectError, match="not free"):
        unconverted_select(taken)
    with pytest.raises(S.SelectError):
        S.select_row(grown)
    for bad_rs in ("x", 3):
        with pytest.raises(S.SelectError):
            S.patched(bad_rs)
    assert S.patched(rs) and not S.patched(grown)


def test_helpers_refuse_junk(romset_zip):
    grown, rs = _sets(str(romset_zip))
    for bad in (lambda: S.slot_mask(True), lambda: S.slot_mask(-1), lambda: S.slot_mask(47), lambda: S.code("x"),
                lambda: S.tile_maps(rs, "x"), lambda: S.is_original(("x",)), lambda: S.layout_of(grown),
                lambda: S.layout_of("x"), lambda: S.paddle_for_entry(16), lambda: S.entry_of_wheel(1.5),
                lambda: S.select_row("x")):
        with pytest.raises(S.SelectError):
            bad()
    assert S.slot_mask(46) and S.layout_of(rs).tables == S.select_row(rs).tables


def test_fix1_junk_refused():
    """Fix round 1: every public entry point refuses junk with a DecodeError subclass (bool never an int)."""
    e = S.ENTRIES[3]
    for bad in (lambda: S.is_original(e._replace(course=[1])), lambda: S.is_original(e._replace(course=True)),
                lambda: S.is_original(e._replace(mirror=None)), lambda: S.is_original(e._replace(mirror=1)),
                lambda: S.first_of(1, "x"), lambda: S.first_of(0, True), lambda: S.first_of(1, 2),
                lambda: S.first_of(0, 17)):
        with pytest.raises(S.SelectError):
            bad()
    assert S.first_of(2, 3) == 0 and S.first_of(9, 10) == 7
    for bad in (lambda: i960.reg(["mov"], 0, "r3"), lambda: i960.cobr({}, 0, "r3", 4),
                lambda: i960.mem([1], "r3", 4), lambda: i960.ctrl(["b"], 4),
                lambda: i960.mem("ld", "r3", 4, scale=True),
                lambda: i960.cobr("cmpobe", 0, "r3", False), lambda: i960.ctrl("b"), lambda: i960.ctrl("call", True),
                lambda: i960.ctrl("ret", 4)):
        with pytest.raises(i960.I960Error):
            bad()
    b = i960.Block(0)
    for bad in (lambda: b.label([1]), lambda: b.cobr("cmpobe", 0, "r3", None), lambda: b.ctrl("call", [1]),
                lambda: b.ctrl("b"), lambda: b.ctrl("nope", "x"), lambda: b.cobr("nope", 0, "r3", "x"),
                lambda: b.ctrl("b", True)):
        with pytest.raises(i960.I960Error):
            bad()
    assert b.bytes() == b""
    for bad in (True, "x", None):
        with pytest.raises(DecodeError):
            C.rom_word_at(bad)
    assert C.rom_word_at(0x2263E0) == ("program", 0x263E0)


def test_fix1_tile_maps_short_row(romset_zip):
    _, rs = _sets(str(romset_zip))
    row = S.select_row(rs)
    for bad in (row._replace(names=row.names[:3]), row._replace(records=row.records[:2]),
                row._replace(records=(row.records[0][:4],) * 3)):
        with pytest.raises(S.SelectError):
            S.tile_maps(rs, bad)


def test_fix1_entries_0_2_are_the_panels():
    e = list(S.M5A_ENTRIES)
    for bad in ([e[1], e[0]] + e[2:], e[:2] + [e[2]._replace(mirror=True)] + e[3:],
                [e[0]] + [e[3], e[2], e[1]] + e[4:], e[:2] + [e[10]] + e[3:10] + [e[2]] + e[11:]):
        with pytest.raises(S.SelectError, match="panel"):
            S.check_entries(bad)
    assert S.check_entries(e[:3] + e[3:][::-1]) == tuple(e[:3] + e[3:][::-1])


def test_fix1_reserved_space(romset_zip):
    """The select's code, data and RAM are registered (romset.RESERVED, RESERVED_RAM), left out of CUSTOM_FREE, and
    the free-space writers refuse them."""
    from dtools.arcade import romset, tgp
    from dtools.convert import build
    code = romset.RESERVED["program"][0]
    data = romset.RESERVED["data"][0]
    assert (code[0], code[0] + code[1]) == S.CODE_FREE and code[0] + code[1] == C.WIDE_HOME     # widen_ids' after it
    assert (C.DATA_BASE + data[0], data[1]) == (S.DATA_AT, S.DATA_SPAN)
    assert romset.RESERVED_RAM[0][:2] == (S.SELECT_RAM[0], S.SELECT_RAM[1] - S.SELECT_RAM[0])
    assert S.SELECT_RAM[0] <= S.VOTE_RAM and S.FIRST_RAM < S.SELECT_RAM[1]
    _, rs = _sets(str(romset_zip))
    lay = S.layout_of(rs)
    g = S.graphics(rs, S.M4A_ENTRIES)
    used_end = lay.char_block + 4 * len(g.runs) + 32 * g.uploaded()
    assert S.DATA_AT <= min(lay.tables) and used_end <= S.DATA_AT + S.DATA_SPAN       # padding included
    assert S.CODE_AT + len(S.code(lay).bytes()) <= S.CODE_FREE[1]
    for lo, n in romset.CUSTOM_FREE["data"]:
        assert lo + n <= data[0] or data[0] + data[1] <= lo
    assert sum(n for _, n in romset.CUSTOM_FREE["data"]) == 0x1000000 - sum(n for _, n, _ in romset.RESERVED["data"])
    assert romset.CUSTOM_FREE["data"][0][0] == 0x10017E0 and romset.CUSTOM_AREAS["data"] == ((0x1000000, 0x1000000),)
    assert romset.reserved("data", data[0] + 4, 4) and romset.reserved("data", data[0] - 4, 4) is None
    assert romset.reserved("program", 0x1F000, 4) and romset.reserved("polygons", 0, 4) is None
    for bad in ((1, 0, 4), ("data", True, 4), ("data", 0, "4")):
        with pytest.raises(romset.RomSetError):
            romset.reserved(*bad)
    custom = RomSet.from_zip(romset_zip).as_custom()
    with pytest.raises(tgp.TgpError, match="reserved"):
        tgp.write_free(custom, "data", data[0] + data[1] - 8, b"\0" * 8)
    assert tgp.write_free(custom, "data", data[0] + data[1], b"\1" * 8).data[data[0] + data[1]] == 1
    with pytest.raises(DecodeError, match="reserved"):
        build._write_free(custom, "program", 0x1E800, b"\0" * 4)


def test_final_fix6_live_new_id_needs_the_guards(romset_zip):
    """Final review fix 6: an entry with a course id 3-15 that is live races that id, so patch_select refuses a grown
    set without the new ids' guards (course.guard_new_ids; unguarded, id 3's play counters would hit Beginner's):
    ENTRIES (Desert City live) and a row with only entry 15 past the originals live are refused; a row whose live
    entries are ids 0-2 patches the unguarded set, and the guarded set takes ENTRIES."""
    grown = C.grow_ids(RomSet.from_zip(romset_zip))
    assert not C.guarded(grown)
    locked3 = S.M4A_ENTRIES[:3] + (S.M4A_ENTRIES[3]._replace(live=False),) + S.M4A_ENTRIES[4:]   # ids 0-2 only
    live15 = locked3[:15] + (locked3[15]._replace(live=True),)
    for entries in (S.ENTRIES, live15):
        with pytest.raises(S.SelectError, match="guard_new_ids"):
            S.patch_select(grown, entries)
    assert S.patched(S.patch_select(grown, locked3))
    guarded, rs = _sets(str(romset_zip))
    assert C.guarded(guarded) and S.patched(rs) and S.select_row(rs).live[3]


def test_live_mirrors_need_their_courses(romset_zip):
    """Phase 9 Task 3 fix 1: ENTRIES' live mirror entries (8-11, ids 8, 10, 9, 11: 11 since Task 4) race their ids,
    so patch_select refuses a set that holds no course at a live id 4-15 (golden.own_tables: Beginner's copy, whose
    rolling flag would race Expert's and Advanced's standing paths), naming the entries; Phase 8's row (M4A_ENTRIES,
    the mirrors locked) patches it, and a row with only id 8 live is refused for id 8 alone. Id 3 is lifted here
    (`unconverted_select`): since Phase 9's final review it is checked too (test_unconverted_id3_refused)."""
    guarded, _ = _sets(str(romset_zip))
    with pytest.raises(S.SelectError, match=r"entries 8 \(course id 8\), 9 \(course id 10\), 10 \(course id 9\), "
                                            r"11 \(course id 11\) are live"):
        unconverted_select(guarded, S.M4B_ENTRIES)
    with pytest.raises(S.SelectError, match=r"entries 4 \(course id 4\), 7 \(course id 7\), 8 \(course id 8\)"):
        unconverted_select(guarded, S.M5A_ENTRIES)              # ids 4 and 7 live since Phase 10 Tasks 3 and 4
    with pytest.raises(S.SelectError, match=r"entries 4 \(course id 4\), 5 \(course id 5\), 6 \(course id 6\), "
                                            r"7 \(course id 7\) are live"):
        unconverted_select(guarded, S.ENTRIES)                  # the row of 8: no mirror; 6 and 5 live since Phase 11
                                                                # Tasks 3 and 4
    only8 = S.M4A_ENTRIES[:8] + (S.M4A_ENTRIES[8]._replace(live=True),) + S.M4A_ENTRIES[9:]
    with pytest.raises(S.SelectError, match=r"entries 8 \(course id 8\) are live"):
        unconverted_select(guarded, only8)
    assert S.patched(unconverted_select(guarded))
    assert S._unfilled(guarded, S.M5A_ENTRIES)[1:] == [(4, 4), (7, 7), (8, 8), (9, 10), (10, 9), (11, 11)]
    assert S._unfilled(guarded, S.ENTRIES) == [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]


def test_unconverted_id3_refused(romset_zip):
    """Phase 9 final review item 3: entry 3 (Desert City, id 3) is live in ENTRIES and M4A_ENTRIES, so patch_select
    checks id 3 too (FILLED_FIRST 3): on a set where id 3 was never converted (golden.own_tables: Beginner's copy, the
    rolling word on the standing path, Phase 6's crash) Phase 8's row is refused for entry 3 alone, naming it, and
    ENTRIES for entry 3 and the mirrors; with entry 3 locked the set patches. Desert City is converted before the
    select (build.converted_set, then patch_select: test_convert_build.test_desert_city_before_select)."""
    guarded, _ = _sets(str(romset_zip))
    assert S.FILLED_FIRST == 3
    with pytest.raises(S.SelectError, match=r"entries 3 \(course id 3\) are live, but the set has no course of its "
                                            r"own"):
        S.patch_select(guarded, S.M4A_ENTRIES)
    with pytest.raises(S.SelectError, match=r"entries 3 \(course id 3\), 4 \(course id 4\), 7 \(course id 7\), "
                                            r"8 \(course id 8\)"):
        S.patch_select(guarded, S.M5A_ENTRIES)
    with pytest.raises(S.SelectError, match=r"entries 3 \(course id 3\), 4 \(course id 4\), 5 \(course id 5\), "
                                            r"6 \(course id 6\), 7 \(course id 7\) are live"):
        S.patch_select(guarded)                                 # ENTRIES, the row of 8 (6 and 5 live since Phase 11)
    assert S._unfilled(guarded, S.M5A_ENTRIES) == [(3, 3), (4, 4), (7, 7), (8, 8), (9, 10), (10, 9), (11, 11)]
    locked3 = S.M4A_ENTRIES[:3] + (S.M4A_ENTRIES[3]._replace(live=False),) + S.M4A_ENTRIES[4:]
    assert S._unfilled(guarded, locked3) == [] and S.patched(S.patch_select(guarded, locked3))


def test_final_fix5_dead_select_fields_refused(romset_zip):
    """Final review fix 5: on a select.patched set the original Circuit Select's tables that golden decodes
    (select_panels: the records from 0x23862c; wheel: the zone instructions; map_object: the records from 0x23889c)
    are read by nothing (the sixteen-entry select draws from its own tables), and ids 3-15 decode as Beginner's panel.
    check_course and encode_course with the set refuse an edit of any of them (CourseError naming the field and
    select.patched); the unedited Course encodes, and on the unpatched set the same edits are taken."""
    from dtools.arcade import golden as G
    grown, rs = _sets(str(romset_zip))
    for k in (0, 3):
        c = G.decode_course(rs, k)
        G.check_course(c, rs)
        G.encode_course(c, rs)
        (addr, rec), *rest = c.select_panels
        x, y, z = c.map_object.position
        w = c.wheel
        edits = {"select_panels": (((addr, rec._replace(cursor=(rec.cursor[0] + 1, rec.cursor[1]))),) + tuple(rest)),
                 "wheel": C.WheelZones(w.lo - 1 if w.lo else w.lo + 1, w.hi),
                 "map_object": c.map_object._replace(position=(x + 1.0, y, z))}
        for f, v in edits.items():
            bad = dataclasses.replace(c, **{f: v})
            for fn in (G.check_course, G.encode_course):
                with pytest.raises(DecodeError, match=f"{f}.*select.patched"):
                    fn(bad, rs)
    c = G.decode_course(grown, 0)
    for f in ("wheel", "map_object"):
        v = C.WheelZones(c.wheel.lo - 1 if c.wheel.lo else c.wheel.lo + 1, c.wheel.hi) if f == "wheel" else \
            c.map_object._replace(position=(c.map_object.position[0] + 1.0,) + c.map_object.position[1:])
        G.encode_course(dataclasses.replace(c, **{f: v}), grown)


# --- Phase 9 Task 5: generated panels, the MIRROR marking, the 3D maps, the idle wheel ----------------------------

def _tiles_used(maps) -> set[int]:
    return {t for m in maps if m is not None for t in m if t != S.LAYER1_BLANK}


def test_panels_fit(romset_zip, monkeypatch):
    """Every non-original entry's window map (layer 1) draws chars CHAR_FIRST-0x3fff in colours 0x76-0x7f (0xf6-0xff
    greyed, tile bit 14), none of which a boot or select upload of the stock screen covers, and the extended upload
    set writes exactly those chars and palettes: the sixteen panels fit the free ranges and stay out of the originals'
    (chars 0x3480-0x39c5, colours 0x69-0x73). The windows: entries 0-2 none (their own art), the mirrors of the
    originals (8-10) the label rows only (MIRROR drawn over the original's art), every other entry the whole interior
    (a generated panel). Every entry's chars sit in a block whose palette is its scheme's (the original panel's for a
    mirror of 0-2, NEW_SCHEME's for the rest), live or locked, so the chars and palettes are the same for both rows;
    a locked entry's tiles are all greyed."""
    grown, rs = _sets(str(romset_zip))
    stock = C.select_assets(grown)
    taken = set(stock.char_source)
    stock_pal = {(r.dst - TL.PALETTE_RAM[0]) // 2 + k for u in stock.uploads for r in u.palette
                 for k in range(len(r.words))}
    for entries in (S.M5A_ENTRIES, S.M4B_ENTRIES, S.M4A_ENTRIES):
        g = S.graphics(grown, entries)
        assert g.windows == tuple(S.window_of(e) for e in entries)
        assert g.windows[:3] == (S.WINDOW_NONE,) * 3 and g.windows[8:11] == (S.WINDOW_LABEL,) * 3
        assert set(g.windows[3:8] + g.windows[11:]) == {S.WINDOW_ALL}
        used = _tiles_used(g.name_maps)
        chars = {t & 0x3FFF for t in used}
        colours = {(t >> 7) & 0xFF for t in used}
        assert chars and min(chars) >= S.CHAR_FIRST and max(chars) < S.CHAR_END
        assert colours <= set(range(0x76, 0x80)) | set(range(0xF6, 0x100))
        assert not chars & taken and not set(range(S.CHAR_HOLE, S.CHAR_HOLE + len(g.chars))) & taken
        for k, (e, m) in enumerate(zip(entries, g.name_maps)):
            if g.windows[k] == S.WINDOW_NONE:
                assert m is None
                continue
            rows = range(1, 9) if g.windows[k] == S.WINDOW_ALL else range(6, 9)
            inner = [m[r * 18 + c] for r in rows for c in range(1, 17)]
            rest = [t for i, t in enumerate(m) if not (i // 18 in rows and 1 <= i % 18 <= 16)]
            assert set(rest) == {S.LAYER1_BLANK}, k
            assert all(t & S.FRONT for t in inner) and all(bool(t & S.GREY_BIT) == (not e.live) for t in inner), k
            want = g.schemes[S.scheme_of(e)]
            drawn = [t & 0x3FFF for t in inner if t & 0x3FFF != S.CHAR_HOLE]
            assert all(g.palettes[(ch - S.CHAR_HOLE) >> 7] == want for ch in drawn), k
    # the live flags change tiles only (ENTRIES with its mirrors locked; M4A_ENTRIES is Phase 9's row so locked, whose
    # entries 4 and 12 still show COMING SOON)
    locked = tuple(e._replace(live=e.live and not e.mirror) for e in S.M5A_ENTRIES)
    a, b = S.graphics(grown, S.M5A_ENTRIES), S.graphics(grown, locked)
    assert a.chars.tolist() == b.chars.tolist() and a.palettes == b.palettes
    assert [[t & ~S.GREY_BIT for t in m] if m else m for m in a.name_maps] == \
        [[t & ~S.GREY_BIT for t in m] if m else m for m in b.name_maps]
    # the patched set uploads them: one char block from CHAR_HOLE, the ten palettes and their grey copies
    g = S.graphics(grown, S.M4A_ENTRIES)
    up = TL.upload_set(rs, struct.unpack("<I", rs.read_i960(S.SELECT_UPLOAD_WORD, 4))[0], False)
    for u in up.chars[-len(g.runs):]:
        assert S.CHAR_HOLE <= u.first_char and u.first_char + u.count <= S.CHAR_END and u.first_char % 128 == 0
        at = u.first_char - S.CHAR_HOLE
        assert TL.decode_chars(u.data(rs)).tolist() == g.chars[at:at + u.count].tolist()
    # every drawn char is uploaded, and no more than UPLOAD_MAX (the select's set-up keeps its frame: "Panels")
    drawn = {t & 0x3FFF for t in _tiles_used(g.name_maps)}
    assert all(any(f <= ch < f + n for f, n in g.runs) for ch in drawn | {S.CHAR_HOLE})
    assert g.uploaded() <= S.UPLOAD_MAX and S.UPLOAD_MAX == 288
    assert len(g.runs) <= S.UPLOAD_RUNS_MAX == 4
    # Phase 10 Task 3: entry 4 live with its label (and 12's) uploads as Phase 9's row did: 127 chars in four runs;
    # Task 4: entry 7 live with an empty band too (its band draws no char; 15 keeps COMING SOON)
    assert S.graphics(grown, S.M5A_ENTRIES).runs == S.graphics(grown, S.M4B_ENTRIES).runs == g.runs
    assert g.uploaded() == 127
    monkeypatch.setattr(S, "UPLOAD_MAX", g.uploaded() - 1)
    with pytest.raises(S.SelectError, match="UPLOAD_MAX"):
        S.graphics(grown, S.M4A_ENTRIES)
    monkeypatch.setattr(S, "UPLOAD_MAX", 288)
    monkeypatch.setattr(S, "UPLOAD_RUNS_MAX", len(g.runs) - 1)
    with pytest.raises(S.SelectError, match="UPLOAD_RUNS_MAX"):
        S.graphics(grown, S.M4A_ENTRIES)
    normal, grey = up.palette[-2:]
    assert (normal.dst, grey.dst) == (TL.PALETTE_RAM[0] + 2 * 16 * S.PALETTE_NORMAL,
                                      TL.PALETTE_RAM[0] + 2 * 16 * S.PALETTE_GREY)
    assert normal.words == sum(g.palettes, ()) and grey.words == g.grey * 10
    ours = set(range(16 * S.PALETTE_NORMAL, 16 * 0x80)) | set(range(16 * S.PALETTE_GREY, 16 * 0x100))
    assert not ours & stock_pal
    # the schemes are the original panels' palettes, read from the set (no game data in the repo)
    for p in range(3):
        m = TL.tile_map(grown, C.decode_select_panel(grown.read_i960(C.SELECT_TABLES[0] + 44 * p, 44), p).maps[0])
        colour = {(t >> 7) & 0xFF for t in m.tiles()}
        assert len(colour) == 1
        (c,) = colour
        assert g.schemes[p] == S.palette_of(grown, c)
    assert g.schemes[S.NEW_SCHEME] != g.schemes[0] and len(g.palettes) == 10


def test_panel_pixels(romset_zip):
    """The generated panel is the original panels' style: Beginner's interior (box, label band) with its lettering
    erased, the name in rows 1-5 and the band's line in rows 6-8 in the arcade's 8 x 8 font (chars 0x20-0x5f of the
    boot upload). The window map's chars draw exactly panel_pixels. Mirror twins share their name rows; the band
    carries the difficulty, a mirror's under it MIRROR in the accent pen; the mirrors of 0-2 the label rows of their
    original's art with MIRROR over the band."""
    grown, _ = _sets(str(romset_zip))
    art = S.panel_art(grown)
    assert set(art.font) == {chr(c) for c in range(0x20, 0x60)} and art.font["A"].shape == (8, 8)
    g = S.graphics(grown, S.M5A_ENTRIES)
    by_char = {S.CHAR_HOLE + i: c for i, c in enumerate(g.chars)}
    for k, e in enumerate(S.M5A_ENTRIES):
        if g.windows[k] == S.WINDOW_NONE:
            continue
        rows = range(1, 9) if g.windows[k] == S.WINDOW_ALL else range(6, 9)
        img = np.concatenate([np.concatenate([by_char[g.name_maps[k][r * 18 + c] & 0x3FFF] for c in range(1, 17)], 1)
                              for r in rows], 0)
        assert img.tolist() == S.panel_pixels(art, e).tolist(), k
    a3, a11, a4 = (S.panel_pixels(art, S.M5A_ENTRIES[k]) for k in (3, 11, 4))
    assert a3.shape == (64, 128) and a3[:40].tolist() == a11[:40].tolist() and a3[40:].tolist() != a11[40:].tolist()
    assert (a11[8 * S.BAND_ROWS[0]:] == S.ACCENT).any()
    assert not (a3 == S.ACCENT).any() and not (a4 == S.ACCENT).any()
    b8 = S.panel_pixels(art, S.M5A_ENTRIES[8])
    assert b8.shape == (24, 128) and (b8 == S.ACCENT).any()
    # the band holds the label lines; too long a difficulty is refused
    assert S.band_lines(S.M5A_ENTRIES[3]) == (("CIRCUIT EDITION", 2, S.TEXT),)
    assert S.band_lines(S.M5A_ENTRIES[11]) == (("CIRCUIT EDITION", 1, S.TEXT), ("MIRROR", 1, S.ACCENT))
    assert S.band_lines(S.M5A_ENTRIES[8]) == (("MIRROR", 2, S.ACCENT),)
    assert S.band_lines(S.M5A_ENTRIES[4]) == (("CIRCUIT EDITION", 2, S.TEXT),)      # Phase 10 Task 3
    assert S.band_lines(S.M5A_ENTRIES[12]) == (("CIRCUIT EDITION", 1, S.TEXT), ("MIRROR", 1, S.ACCENT))
    assert S.band_lines(S.M5A_ENTRIES[5]) == (("COMING SOON", 2, S.TEXT),)
    assert S.band_lines(S.M5A_ENTRIES[7]) == ()                                     # Phase 10 Task 4: no origin line
    assert S.band_lines(S.M5A_ENTRIES[15]) == (("COMING SOON", 1, S.TEXT), ("MIRROR", 1, S.ACCENT))
    for bad in ("A" * (S.LABEL_LINE + 1), "lower", 3):
        with pytest.raises(S.SelectError):
            S.check_entries(S.M5A_ENTRIES[:3] + (S.M5A_ENTRIES[3]._replace(difficulty=bad),) + S.M5A_ENTRIES[4:11]
                            + (S.M5A_ENTRIES[11]._replace(difficulty=bad),) + S.M5A_ENTRIES[12:])


def test_difficulty_labels():
    """Decision 11: the originals' own labels for the originals and their mirrors, the Dreamcast's for Desert City
    (its origin line, discovery log "Panels") and, since Phase 10 Task 3, National Park Speedway (its origin line on
    the Dreamcast's course select is FROM CIRCUIT EDITION too: docs/spike/2026-11-dc-decode.md, "Course names"); a
    mirror takes its original's (check_entries refuses another), so entry 12, still locked, carries it too. Mermaid
    Lake (Phase 10 Task 4) has no origin line on the Dreamcast's select, so neither difficulty nor label: its band is
    empty; its mirror (entry 15, locked) keeps COMING SOON as its line."""
    d = {e.course: e.difficulty for e in S.M5A_ENTRIES}
    assert (d[0], d[2], d[1]) == ("BEGINNER", "ADVANCED", "EXPERT") and d[3] == d[4] == "CIRCUIT EDITION"
    assert d[5] == d[6] == d[7] == ""
    assert all(d[c + 8] == d[c] for c in range(8))
    e = list(S.M5A_ENTRIES)
    with pytest.raises(S.SelectError, match="difficulty"):
        S.check_entries(e[:8] + [e[8]._replace(difficulty="EXPERT")] + e[9:])
    assert [x.label for x in S.M5A_ENTRIES] == [""] * 5 + ["COMING SOON"] * 2 + [""] * 6 + ["COMING SOON"] * 3
    # the row of 8 (Phase 11 Task 2): the same labels for its eight courses, but Rin Rin Rink's and Circuit Pixie's
    # bands empty since they went live (Phase 11 Tasks 3 and 4, the plan's labels ruling)
    assert [(x.difficulty, x.label) for x in S.M5B_ENTRIES] == [(x.difficulty, "" if k in (5, 6) else x.label)
                                                            for k, x in enumerate(S.M5A_ENTRIES[:8])]


def test_idle_entry(romset_zip):
    """The idle-wheel ruling (plan, "Ruling: the idle wheel"): the select's state 0 (single and linked, 0x22563c and
    0x2256b4) starts the cursor on IDLE_ENTRY (0) and clears FOLLOW_RAM; the wheel keeps the cursor there until the
    wheel value leaves the centre band (the band of wheel value 0, CENTRE_ENTRY's), then follows it; the timeout on a
    locked entry races IDLE_ENTRY (no walk to the left). `follow` is the rule's model."""
    _, rs = _sets(str(romset_zip))
    lay = S.layout_of(rs)
    blk = S.code(lay)
    assert S.IDLE_ENTRY == 0 and S.CENTRE_ENTRY == S.entry_of_wheel(0) == 7 and S.check_entries(S.ENTRIES)[0].live
    assert S.SELECT_RAM[0] <= S.FOLLOW_RAM < S.SELECT_RAM[1] and S.FOLLOW_RAM not in (S.FIRST_RAM, *range(
        S.VOTE_RAM, S.VOTE_RAM + 1 + 2 * S.ENTRY_COUNT))

    def text(at: int, n: int) -> list[str]:
        return [" ".join(ln.text.split()) for ln in disasm.listing(rs, at, n)]
    for site, back in ((0x22563C, 0x225644), (0x2256B4, 0x2256BC)):
        (hook,) = text(site, 1)
        assert hook.startswith("b ")
        at = int(hook.split()[1], 16)
        assert text(at, 5) == ["mov 0,r3", f"stob r3,{S.FOLLOW_RAM:#x}", f"stos r3,{S.PANEL:#x}", "stob r3,0x5e(g13)",
                               f"b {back:#010x}"]
    wheel = text(blk.labels["wheel"], 7)
    left = f"{blk.labels['wheel_left']:#x}"
    assert wheel[0] == f"ldob {S.FOLLOW_RAM:#x},r5" and wheel[1] == f"cmpobne 0,r5,{blk.labels['wheel_follow']:#x}"
    # the two band edges: w > IDLE_BAND, and w + IDLE_BAND < 0, leave the band; inside it, IDLE_ENTRY
    assert wheel[2:6] == [f"cmpibl {S.IDLE_BAND},r3,{left}", f"addi {S.IDLE_BAND},r3,r6", f"cmpibg 0,r6,{left}",
                          f"mov {S.IDLE_ENTRY},r4"]
    assert text(blk.labels["wheel_left"], 2) == ["mov 1,r5", f"stob r5,{S.FOLLOW_RAM:#x}"] and S.IDLE_BAND == 16
    t0 = blk.labels["timeout"]
    timeout = disasm.listing(rs, t0, 16)
    end = next(i for i, ln in enumerate(timeout) if ln.text == "ret")
    body = [" ".join(ln.text.split()) for ln in timeout[:end + 1]]
    assert body[0] == f"ldis {S.PANEL:#x},r4" and body[1] == f"ldob {lay.live:#x}[r4],r6"
    assert body[2].startswith("cmpobne 0,r6,") and body[3] == f"mov {S.IDLE_ENTRY},r4"
    targets = [int(t.split(",")[-1], 16) for t in body if t.startswith("cmp")]
    assert all(t > t0 for t in targets) and all(not ln.startswith("b ") for ln in body)     # no walk back
    # the model
    assert S.follow([0] * 30) == [0] * 30
    assert S.follow([0, -16, 16, 5, -5]) == [0] * 5 and S.follow([0, 0, -17, 0, 0]) == [0, 0, 6, 7, 7]
    assert S.follow([0, 17, 1, 0]) == [0, 9, 8, 7] and S.follow([-127, 0]) == [0, 7] and S.follow([]) == []
    assert [S.wheel_value(p) for p in (113, 114, 120, 128, 136, 142, 143)] == [-17, -15, -5, 0, 5, 15, 17]
    assert S.follow([S.wheel_value(S.paddle_for_entry(k)) for k in range(16)]) == list(range(16))
    with pytest.raises(S.SelectError):
        S.follow([0, True])


def test_map_records(romset_zip):
    """The 3D maps (discovery log, "Panels"): every map object is a textured quad (-2..2 in x and z, y 0); a mirror of
    an original shows its original's quad mirrored (mirror.mirror_object, local: z -> -z, the texture flipped), in a
    list and objects of its own written to the last free bytes of the list home (tgp.DATA_FREE: lists past the
    object library's upper bound are not drawn) and of the custom polygon area, owned (golden.owned_spans through
    map_spans); every other non-original entry keeps an empty record (no map: Desert City's from its road is Not
    found in the time box). map_records reads the 42 records back."""
    from dtools.arcade import golden as G
    from dtools.arcade import mirror as M
    from dtools.arcade import polyrom as P
    from dtools.convert import build
    grown, rs = _sets(str(romset_zip))
    recs = S.map_records(rs)
    assert len(recs) == 3 * 14
    assert [(r.first, r.position) for r in recs] == [(f, s) for f in range(14) for s in range(3)]
    stock_lists = [struct.unpack("<I", grown.read_i960(C.SELECT_MAP_OBJECTS + 32 * p + 12, 4))[0] for p in range(3)]
    seen = {}
    for r in recs:
        e = S.M4A_ENTRIES[r.first + r.position]
        assert r.place == struct.unpack("<3f", grown.read_i960(C.SELECT_MAP_OBJECTS + 32 * r.position, 12))
        if S.is_original(e):
            assert r.objects == stock_lists[S.ORIGINAL_PANEL[e.course]]
        elif e.mirror and e.course - 8 in S.ORIGINAL_PANEL:
            seen.setdefault(e.course, r.objects)
            assert r.objects == seen[e.course] and r.objects not in stock_lists and r.raw[0] == r.objects
            assert C.library_bounds(rs)[0] <= r.objects < C.library_bounds(rs)[1]
            o = C.object_list(grown, stock_lists[S.ORIGINAL_PANEL[e.course - 8]])
            m = C.object_list(rs, r.objects)
            assert [(p.tpa, p.tha, p.raw) for p in m] == [(p.tpa, p.tha, p.raw) for p in o]
            for p, q in zip(o, m):
                a = P.decode_object(grown.polygons, C.check_rom_object(p.oba, grown), p.obc, S.MAP_MODE)
                b = P.decode_object(rs.polygons, C.check_rom_object(q.oba, rs), q.obc, S.MAP_MODE)
                assert b == M.mirror_object(a, world=False)
        else:
            assert r.objects == 0
    assert len(seen) == 3
    spans = S.map_spans(rs)
    assert {r for r, *_ in spans} == {"data", "polygons"} and len(spans) == 6
    owned = G.owned_spans(rs)
    for s in spans:
        assert s in owned
    lo, n = build.NEW_ID_RUNS["polygons"]
    assert all(lo <= o and o + k <= lo + n for r, o, k, _ in spans if r == "polygons")
    from dtools.arcade import tgp
    dlo, dn = tgp.DATA_FREE
    assert all(dlo <= o and o + k <= dlo + dn for r, o, k, _ in spans if r == "data")
    # last fit: the run's last bytes, so a converter's first fit (before or after the select) never meets them
    assert max(o + k for r, o, k, _ in spans if r == "data") == dlo + dn
    assert max(o + k for r, o, k, _ in spans if r == "polygons") == lo + n
    for bad in ("x", grown):
        with pytest.raises(S.SelectError):
            S.map_records(bad)


# --- Phase 11 Task 2: the row of eight (mirror mode dropped; reverse at M9) -----------------------------------------

FROZEN_ROW_SHA1 = {"M5A": "707d5af5cdca7ddd6bed86c5cfdfc982e754a6c5",
                   "TASK4": "707d5af5cdca7ddd6bed86c5cfdfc982e754a6c5",
                   "M4B": "21f034a503ef45360e2c601b4d6050bc0500f550", "M4A": "cdc8c6970364a2ae21a0e0797cdb212c868b10d0"}
"""sha1 of the program, data and polygon regions of patch_select(grown and guarded stock set, row) for each frozen row
of sixteen (`_unfilled` lifted: the rows' live ids 3-15 need courses the bare set does not hold; the check is not
what is pinned), computed with the code at c273118 (patch_select as at main ef7ad08) before Task 2 changed it."""


def test_row_of_eight_wheel():
    """Review Focus 2 (Phase 11): for a row of 8 the wheel's bands are 32 wide (bands_of(8)); every paddle 0-255 maps
    monotonically onto entries 0..7, each entry has paddles and paddle_for_entry(k, 8) picks k; the positions show
    entries first_of(k, 8)..+2 (k in the middle but at the ends). The row of 16 is unchanged (its bands are BANDS, its
    paddles autoplay.lua's ENTRY_WHEEL). autoplay.lua's ENTRY_WHEEL_8 and LEAD_8 (DAYTONA_ROW 8) are the model's."""
    assert S.ROW_LENGTHS == (8, 16)
    assert S.bands_of(16) == S.BANDS and S.bands_of(8) == (-96, -64, -32, 0, 32, 64, 96)
    seen = [S.entry_of_wheel(S.wheel_value(p), 8) for p in range(256)]
    assert seen == sorted(seen) and set(seen) == set(range(8))
    assert min(seen.count(k) for k in range(8)) >= 19
    assert [S.entry_of_wheel(w, 8) for w in range(-127, 128)] == [sum(w > t for t in S.bands_of(8))
                                                                  for w in range(-127, 128)]
    for k in range(8):
        p = S.paddle_for_entry(k, 8)
        assert S.entry_of_wheel(S.wheel_value(p), 8) == k
        assert S.slot_of(k, 8) == (0 if k == 0 else 2 if k == 7 else 1) and S.first_of(k, 8) + S.slot_of(k, 8) == k
    assert S.entry_of_wheel(S.wheel_value(255), 8) == 7 and S.entry_of_wheel(S.wheel_value(0), 8) == 0
    # the row of 16 as before
    assert [S.entry_of_wheel(w) for w in range(-127, 128)] == [S.entry_of_wheel(w, 16) for w in range(-127, 128)] == \
        [sum(w > t for t in S.BANDS) for w in range(-127, 128)]
    assert [S.paddle_for_entry(k) for k in range(16)] == [S.paddle_for_entry(k, 16) for k in range(16)] == [
        29, 62, 71, 81, 90, 100, 109, 124, 138, 148, 157, 167, 176, 186, 195, 228]
    assert [S.slot_of(k) for k in range(16)] == [S.slot_of(k, 16) for k in range(16)]
    # the idle wheel on the row of 8: the centred wheel races IDLE_ENTRY until the wheel leaves the band
    assert S.follow([0] * 10, 8) == [S.IDLE_ENTRY] * 10 and S.follow([0, -17, 0], 8) == [0, 3, 3]
    assert S.follow([S.wheel_value(S.paddle_for_entry(k, 8)) for k in range(8)], 8) == list(range(8))
    # autoplay.lua's row of 8
    lua = (REPO_ROOT / "dtools/mame/lua/autoplay.lua").read_text()
    table = re.search(r"local ENTRY_WHEEL_8 = \{\[0\] = ([\d, ]+)\}", lua).group(1)
    assert [int(v) for v in table.split(",")] == [S.paddle_for_entry(k, 8) for k in range(8)]
    lead = dict(map(int, kv) for kv in re.findall(r"\[(\d+)\] = (\d+)",
                                                  re.search(r"local LEAD_8 = \{([^}]*)\}", lua).group(1)))
    inside = [k for k in range(8) if abs(S.wheel_value(S.paddle_for_entry(k, 8))) <= S.IDLE_BAND]
    assert sorted(lead) == inside == [3]                    # entry 4's paddle, 143, gives 17
    for k, via in lead.items():
        assert abs(k - via) == 1 and abs(S.wheel_value(S.paddle_for_entry(via, 8))) > S.IDLE_BAND
        assert S.follow([S.wheel_value(S.paddle_for_entry(via, 8)), S.wheel_value(S.paddle_for_entry(k, 8))], 8) == \
            [via, k]
    for bad in (lambda: S.bands_of(10), lambda: S.bands_of(True), lambda: S.bands_of("8"),
                lambda: S.entry_of_wheel(0, 9), lambda: S.entry_of_wheel(0, True), lambda: S.paddle_for_entry(8, 8),
                lambda: S.paddle_for_entry(0, 12), lambda: S.slot_of(8, 8), lambda: S.follow([0], 4)):
        with pytest.raises(S.SelectError):
            bad()


def test_row_of_eight_refusals(romset_zip):
    """A row of 8: course ids a permutation of 0..7, no mirror entry; check_entries and patch_select refuse a row of 8
    with an id 8 or more, a mirror entry or a repeated id, and a row whose length is not in ROW_LENGTHS."""
    e = list(S.ENTRIES)
    assert len(e) == 8 and S.check_entries(e) == S.ENTRIES
    m = list(S.M5A_ENTRIES)
    bad_rows = (e[:7] + [e[7]._replace(course=8)],                    # an id past the row
                e[:7] + [e[7]._replace(course=15)],
                e[:7] + [e[7]._replace(mirror=True)],                  # a mirror entry
                e[:3] + [m[11]] + e[4:],                               # a mirror entry (and id 11)
                e[:7] + [e[6]],                                        # a repeated id (7 missing)
                e[:7], e + [e[7]._replace(course=8)], m[:15], m + [m[0]], [], e[:3])     # lengths not 8 or 16
    for bad in bad_rows:
        with pytest.raises(S.SelectError):
            S.check_entries(bad)
    grown, _ = _sets(str(romset_zip))
    for bad in bad_rows:
        with pytest.raises(S.SelectError):
            S.patch_select(grown, bad)
    with pytest.raises(S.SelectError, match="mirror"):
        S.check_entries(e[:7] + [e[7]._replace(mirror=True)])
    with pytest.raises(S.SelectError, match="8 or 16|ROW_LENGTHS"):
        S.check_entries(e[:7])


def test_frozen_rows_unchanged(romset_zip, monkeypatch):
    """patch_select with the frozen rows of sixteen (M5A_ENTRIES, tests/test_mirrors_mame.py TASK4_ENTRIES,
    M4B_ENTRIES, M4A_ENTRIES) writes the bytes it wrote at c273118 (FROZEN_ROW_SHA1: patch_select as at main ef7ad08),
    so the sets built with them keep their pins (tests/test_convert_build_courses.py FINAL_SHA1, FINAL_SHA1_T4,
    M4B_SHA1, unchanged). M5A_ENTRIES is TASK4_ENTRIES (Phase 10 Task 5 changed no entry: `final_set` is the byte
    check's `task5_set`)."""
    import hashlib
    from tests.test_mirrors_mame import TASK4_ENTRIES
    assert S.M5A_ENTRIES == TASK4_ENTRIES
    grown, _ = _sets(str(romset_zip))
    monkeypatch.setattr(S, "_unfilled", lambda rs, entries: [])
    for name, row in (("M5A", S.M5A_ENTRIES), ("TASK4", TASK4_ENTRIES), ("M4B", S.M4B_ENTRIES),
                      ("M4A", S.M4A_ENTRIES)):
        rs = S.patch_select(grown, row)
        h = hashlib.sha1()
        for r in ("program", "data", "polygons"):
            h.update(getattr(rs, r))
        assert h.hexdigest() == FROZEN_ROW_SHA1[name], name
        assert len(S.select_row(rs).live) == 16 and S.layout_of(rs).tables[1] - S.layout_of(rs).tables[0] == 704


def test_row_of_eight_patch(romset_zip):
    """patch_select with a row of 8 (M5B_ENTRIES' entries, 3, 4 and 7 locked: the bare grown set holds no course there):
    eight records per mode in row order, SELECT_PANEL_OF_COURSE still sixteen words (indexed by course id): ids 0-7
    the records' inverse, ids 8-15 (no entry) the entry of id - 8; live, voices and window maps per entry; the bands
    bands_of(8); 0x2263e0 `mov 8,g1` (the draw's row length: first = clamp(E - 1, 0, 5)); the vote's three words
    sixteen slots as before (indexed by course id); the code the same instructions as for a row of 16 with the band
    count 7 (`cmpobg 7,r4`) and the data's addresses; 18 map records (first 0-5), no mirrored map, so no map
    objects, no list and no library-bound patch from the select; the upload within UPLOAD_MAX / UPLOAD_RUNS_MAX; every
    byte that changed is a site or was free (0xff)."""
    grown, rs16 = _sets(str(romset_zip))
    row8 = tuple(e._replace(live=False) if e.course >= 3 else e for e in S.M5B_ENTRIES)
    rs = S.patch_select(grown, row8)
    row, lay = S.select_row(rs), S.layout_of(rs)
    assert len(row.live) == 8 and row.live == tuple(e.live for e in row8)
    assert lay.tables[1] - lay.tables[0] == -(-8 * C.SELECT_RECORD_SIZE // 16) * 16
    old = [[C.decode_select_panel(grown.read_i960(t + 44 * p, 44), p) for p in range(3)] for t in C.SELECT_TABLES]
    for mode in range(3):
        recs = row.records[mode]
        assert [r.course for r in recs] == [e.course for e in S.M5B_ENTRIES] == [0, 2, 1, 3, 4, 5, 6, 7]
        for k, r in enumerate(recs):
            s = S.slot_of(k, 8)
            assert r.cursor == old[0][s].cursor and r.tile == old[0][s].tile
            w = S.window_of(S.M5B_ENTRIES[k])
            assert r.maps == (lay.hole_maps * 3 if w == S.WINDOW_ALL else old[mode][S.scheme_of(S.M5B_ENTRIES[k])].maps)
    entry_of = {e.course: k for k, e in enumerate(S.M5B_ENTRIES)}
    assert row.panel_of_course == tuple(entry_of[c % 8] for c in range(16))
    assert row.voices[:3] == (0xAE215E, 0xAE215F, 0xAE2160) and set(row.voices[3:]) == {S.VOICE_DEFAULT}
    assert row.bands == S.bands_of(8) and row.vote == S.VOTE_RAM
    g = S.graphics(grown, row8)
    assert g.windows == (S.WINDOW_NONE,) * 3 + (S.WINDOW_ALL,) * 5 and len(g.name_maps) == 8
    assert g.uploaded() <= S.UPLOAD_MAX and len(g.runs) <= S.UPLOAD_RUNS_MAX
    maps = S.tile_maps(rs, row)
    assert sorted(maps) == list(range(8))
    for k in range(8):
        assert maps[k][1].tiles() == (g.name_maps[k] or (S.LAYER1_BLANK,) * 180)
    want = {0x2263E0: "mov     8,g1", 0x3050: "lda     0x21,r5", 0x32BC: "mov     16,g9", 0x32D0: "subi    g9,16,g2",
            0x3048: "lda     0x501b03,r4", 0x32B4: "lda     0x501b04,r11",
            0x226C7C: f"ldob    {lay.tables[0] + 8:#x}(g0),g0", 0x2261C8: f"ld      {lay.voices:#x}[r3*4],g0"}
    for a, text in want.items():
        assert disasm.listing(rs, a, 1)[0].text == text, hex(a)
    # the code: the same instructions as the row of sixteen's, the band count and the data's addresses aside
    a, b = S.code(S.layout_of(rs16)), S.code(lay)
    ta = [ln.text.split()[0] for ln in disasm.listing_bytes(a.bytes(), S.CODE_AT)]
    tb = [ln.text.split()[0] for ln in disasm.listing_bytes(b.bytes(), S.CODE_AT)]
    assert ta == tb and len(a.bytes()) == len(b.bytes()) and a.labels == b.labels
    loop = [" ".join(ln.text.split()) for ln in disasm.listing_bytes(b.bytes(), S.CODE_AT)
            if ln.text.startswith("cmpobg") and ln.addr < b.labels["wheel_done"]]
    assert loop == [f"cmpobg 7,r4,{b.labels['wheel_loop']:#x}"]
    recs = S.map_records(rs)
    assert [(r.first, r.position) for r in recs] == [(f, s) for f in range(6) for s in range(3)]
    stock_lists = [struct.unpack("<I", grown.read_i960(C.SELECT_MAP_OBJECTS + 32 * p + 12, 4))[0] for p in range(3)]
    for r in recs:
        e = S.M5B_ENTRIES[r.first + r.position]
        assert r.objects == (stock_lists[S.ORIGINAL_PANEL[e.course]] if S.is_original(e) else 0)
    assert S.map_spans(rs) == [] and C.library_bounds(rs) == C.library_bounds(grown)
    assert rs.polygons == grown.polygons and rs.textures == grown.textures and rs.copro_data == grown.copro_data
    sites = {a - 0x200000 if a >= 0x220000 else a for a in want} | {a + 4 - 0x200000 for a in S.TABLE_SITES} | \
        {a - 0x200000 for a in S.TABLE_SITES} | {a - 0x200000 if a >= 0x220000 else a for a in S.HOOKS} | \
        {a for a in S.VOTE_SITES} | {0x3050, 0x32BC, 0x32D0}
    blk = S.code(lay)
    for lo, hi in _changed(grown.program, rs.program):
        if S.CODE_AT <= lo < S.CODE_FREE[1]:
            assert hi <= S.CODE_AT + len(blk.bytes()) and set(grown.program[lo:hi]) == {0xFF}
        else:
            assert any(s <= lo < s + 8 for s in sites), hex(lo)
    poc = C.id_table(rs, "SELECT_PANEL_OF_COURSE") - C.DATA_BASE
    upload = S.SELECT_UPLOAD_WORD - C.DATA_BASE
    for lo, hi in _changed(grown.data, rs.data):
        if lo >= S.DATA_AT - C.DATA_BASE:
            assert hi <= S.DATA_AT - C.DATA_BASE + S.DATA_SPAN and set(grown.data[lo:hi]) == {0xFF}
        else:
            assert all(poc <= x < poc + 64 or upload <= x < upload + 4 for x in range(lo, hi)
                       if grown.data[x] != rs.data[x]), hex(lo)


# --- Phase 12 Task 2: the courses' names on the select (the release row) --------------------------------------------

RELEASE_UPLOAD = 188
"""This is the bare grown set's number (not the release set's, which is 210: RELEASE_SET_UPLOAD in tests/test_release.py;
the two differ because the grown set's own bands are all SHORT, so fewer digit chars are cut). The chars
graphics(grown, ENTRIES) uploads, in four runs of 31, 24, 29 and 104 (the originals' three schemes, then
NEW_SCHEME) on the bare grown set, whose ids 3-7 are Beginner's copies (all SHORT, 8 / 20 / 80 laps: Sega's own
band); Task 2's 129 had the bands empty. On the release set (MEDIUM and SHORT bands, MEDIUM's word moved) 210 in 31,
24, 29 and 126: tests/test_release.py RELEASE_SET_UPLOAD. Inside UPLOAD_MAX and UPLOAD_RUNS_MAX (docs/spike/
2026-11-release.md, "Names on the select", "Maps and laps")."""


def _name_cells(e: S.Entry) -> np.ndarray:
    """(40, 128) True on the char cells name_lines(e.name) is drawn into on rows 1-5 (`_text`'s placement: each line
    centred in NAME_CELLS, rounded left, two char rows tall, from NAME_ROWS)."""
    lines = S.name_lines(e.name)
    cells = np.zeros((40, 128), bool)
    for ln, r in zip(lines, S.NAME_ROWS[len(lines) - 1]):
        left = S.NAME_CELLS[0] + (S.NAME_LINE - len(ln)) // 2
        cells[8 * r:8 * r + 16, 8 * left:8 * (left + len(ln))] = True
    return cells


def test_original_name_window(romset_zip):
    """WINDOW_NAME (Phase 12 Task 2): an original with no difficulty shows its course's name on layer 1 over rows 1-5
    of its own art (the name box; the band, rows 6-8, stays Sega's, since its lap count changes with the mode); an
    original with a difficulty keeps WINDOW_NONE, so every frozen row's originals do, and their sets keep their bytes.
    The window's pixels are the erased box (PanelArt.interior[:40]) with the name placed as on a generated panel, in
    the original's scheme."""
    from tests.test_mirrors_mame import TASK3_ENTRIES, TASK4_ENTRIES
    assert S.WINDOW_NAME == 3 and len({S.WINDOW_NONE, S.WINDOW_LABEL, S.WINDOW_ALL, S.WINDOW_NAME}) == 4
    for row in (S.M4A_ENTRIES, S.M4B_ENTRIES, S.M5A_ENTRIES, S.M5B_ENTRIES, TASK3_ENTRIES, TASK4_ENTRIES):
        assert [S.window_of(e) for e in row[:3]] == [S.WINDOW_NONE] * 3
        assert S.WINDOW_NAME not in [S.window_of(e) for e in row]
    assert [S.window_of(e) for e in S.ENTRIES] == [S.WINDOW_NAME] * 3 + [S.WINDOW_NONE] * 5     # 3-7: own panels
    assert S.window_of(S.ENTRIES[0]._replace(difficulty="BEGINNER")) == S.WINDOW_NONE
    assert S.window_of(S.M5A_ENTRIES[8]) == S.WINDOW_LABEL and S.window_of(S.M5A_ENTRIES[3]) == S.WINDOW_ALL
    assert [S.scheme_of(e) for e in S.ENTRIES[:3]] == [0, 1, 2]
    assert [S.name_lines(e.name) for e in S.ENTRIES[:3]] == [("THREE SEVEN", "SPEEDWAY"), ("DINOSAUR", "CANYON"),
                                                             ("SEA-SIDE", "STREET GALAXY")]
    grown, _ = _sets(str(romset_zip))
    art = S.panel_art(grown)
    box = art.interior[:S.LABEL_TOP]
    for e in S.ENTRIES[:3]:
        img = S.panel_pixels(art, e)
        cells = _name_cells(e)
        assert img.shape == (40, 128)
        assert img[~cells].tolist() == box[~cells].tolist()
        assert (img[cells] == S.TEXT).any() and (img[cells] == S.SHADOW).any()
        generated = S.panel_pixels(art, S.Entry(3, e.name, "", False, True))       # the same name, generated
        assert img.tolist() == generated[:S.LABEL_TOP].tolist()
        with pytest.raises(S.SelectError, match="band"):
            S.band_lines(e)                                         # the band is the original's art
    with pytest.raises(S.SelectError):
        S.band_lines(S.M5B_ENTRIES[0])


def test_release_row(romset_zip):
    """The release row (select.ENTRIES, Phase 12 Task 2): every course under its own name, in Phase 11's order and
    ids, all live, no mirror, no difficulty or label line anywhere (CIRCUIT EDITION included: the plan's call). Its
    generated chars: four upload runs (the three originals' schemes and NEW_SCHEME), within UPLOAD_MAX; entries 3-7
    draw their own panels (Task 6: own_panel, the band their length and laps; test_converted_bands)."""
    E = S.Entry
    assert S.ENTRIES == (E(0, "THREE SEVEN SPEEDWAY", "", False, True), E(2, "DINOSAUR CANYON", "", False, True),
                         E(1, "SEA-SIDE STREET GALAXY", "", False, True),
                         E(3, "DESERT CITY", "", False, True, own_panel=True),
                         E(4, "NATIONAL PARK SPEEDWAY", "", False, True, own_panel=True),
                         E(5, "CIRCUIT PIXIE", "", False, True, own_panel=True),
                         E(6, "RIN RIN RINK", "", False, True, own_panel=True),
                         E(7, "MERMAID LAKE", "", False, True, own_panel=True))
    assert [e.own_panel for e in S.ENTRIES] == [False] * 3 + [True] * 5
    assert all(e.difficulty == "" and e.label == "" for e in S.ENTRIES)
    assert [e.course for e in S.ENTRIES] == [e.course for e in S.M5B_ENTRIES]
    assert [e.name for e in S.ENTRIES[3:]] == [e.name for e in S.M5B_ENTRIES[3:]]
    assert S.check_entries(S.ENTRIES) == S.ENTRIES
    grown, _ = _sets(str(romset_zip))
    g = S.graphics(grown, S.ENTRIES)
    print(f"\nthe release row's upload: {g.uploaded()} chars in {len(g.runs)} runs {[(hex(f), n) for f, n in g.runs]}")
    assert len(g.runs) == 4 and g.uploaded() <= S.UPLOAD_MAX
    assert g.uploaded() == RELEASE_UPLOAD
    assert sorted(g.palettes.index(g.schemes[s]) for s in (0, 1, 2, S.NEW_SCHEME)) == [0, 1, 2, 3]
    assert g.name_maps[3:] == (None,) * 5 and all(pm is not None for pm in g.panels[3:])


def _tiles(rs: RomSet, addr: int, rows) -> tuple[int, ...]:
    m = TL.tile_map(rs, addr)
    return tuple(m.tile(r, c) for r in rows for c in range(1, C.PANEL_COLS - 1))


def test_name_window_keeps_bands(romset_zip, monkeypatch):
    """Review Focus 5 (Phase 12): on patch_select(grown, ENTRIES) each mode table gives entries 0-2 their original
    panel's six maps for that mode (the band's lap count per mode and its blink map stay Sega's); their window maps
    hold tiles in rows 1-5 only, every other tile LAYER1_BLANK. The measurement behind it, on the set's own maps: rows
    1-5 (the box with the difficulty word) change with the variant (the language) and the blink only, rows 6-8 (the
    band) with the mode and the blink only, the frame with the blink only. The draw gains the WINDOW_NAME branch
    (`code(lay, True)`: one instruction); select_row and tile_maps decode the row."""
    grown, _ = _sets(str(romset_zip))
    old = [[C.decode_select_panel(grown.read_i960(t + 44 * p, 44), p) for p in range(3)] for t in C.SELECT_TABLES]
    for p in range(3):
        for kind in (0, 1):
            for v in range(C.SELECT_VARIANTS):
                per_mode = [old[mode][p].maps[2 * v + kind] for mode in range(3)]
                assert len({_tiles(grown, m, range(1, 6)) for m in per_mode}) == 1, (p, kind, v)
                assert len({_tiles(grown, m, range(6, 9)) for m in per_mode}) == 3, (p, kind, v)
            for mode in range(3):
                v0, v1 = old[mode][p].maps[kind], old[mode][p].maps[2 + kind]
                assert _tiles(grown, v0, range(6, 9)) == _tiles(grown, v1, range(6, 9))
                assert _tiles(grown, v0, range(1, 6)) != _tiles(grown, v1, range(1, 6))
    monkeypatch.setattr(S, "_unfilled", lambda rs, entries: [])       # the bare set holds no course at 3-7
    rs = S.patch_select(grown, S.ENTRIES)
    row, lay = S.select_row(rs), S.layout_of(rs)
    for mode in range(3):
        for k in range(3):
            assert row.records[mode][k].maps == old[mode][S.ORIGINAL_PANEL[S.ENTRIES[k].course]].maps, (mode, k)
            assert row.records[mode][k].course == S.ENTRIES[k].course
        assert not any(set(row.records[mode][k].maps) & set(lay.hole_maps) for k in range(3, 8))     # own panels
    assert bytes(rs.read_i960(lay.generated, 8)) == bytes([S.WINDOW_NAME] * 3 + [S.WINDOW_NONE] * 5)
    maps = S.tile_maps(rs, row)
    g = S.graphics(grown, S.ENTRIES)
    for k in range(3):
        tiles = maps[k][1].tiles()
        assert tiles == g.name_maps[k]
        inner = [tiles[r * 18 + c] for r in range(1, 6) for c in range(1, 17)]
        rest = [t for i, t in enumerate(tiles) if not (1 <= i // 18 <= 5 and 1 <= i % 18 <= 16)]
        assert set(rest) == {S.LAYER1_BLANK} and all(t & S.FRONT and not t & S.GREY_BIT for t in inner), k
        assert maps[k][0].tiles() == TL.tile_map(grown, old[0][k].maps[0]).tiles()
    # the code: the M5b row's instructions and one more, the branch that masks an original's name rows only
    plain, named = S.code(lay), S.code(lay, True)
    assert rs.read_i960(S.CODE_AT, len(named.bytes())) == named.bytes()
    assert len(named.bytes()) == len(plain.bytes()) + 4 and S.code(lay, False).bytes() == plain.bytes()
    for bad in (1, 0, None, "yes"):
        with pytest.raises(S.SelectError, match="name_window"):
            S.code(lay, bad)
    text = [" ".join(ln.text.split()) for ln in disasm.listing_bytes(named.bytes(), S.CODE_AT)]
    plain_text = [ln.text.split()[0] for ln in disasm.listing_bytes(plain.bytes(), S.CODE_AT)]
    branch = text.index(f"cmpobe {S.WINDOW_NAME},r5,{named.labels['draw_name_rows']:#x}")
    assert [t.split()[0] for t in text[:branch] + text[branch + 1:]] == plain_text
    assert text[branch + 1:branch + 3] == ["or g2,r14,r14", "or g3,r15,r15"]       # skipped: the label rows' words
    assert disasm.listing_bytes(named.bytes(), S.CODE_AT)[branch + 4].addr == named.labels["draw_name_rows"]
    assert text[branch + 4:branch + 6] == ["or g2,r9,r9", "or g3,r10,r10"]          # taken: the name rows' words
    # a row without WINDOW_NAME writes the code it wrote (M5b's row: test_row_of_eight_patch, the frozen pins)
    m5b = S.patch_select(grown, S.M5B_ENTRIES)
    body = S.code(S.layout_of(m5b)).bytes()
    assert S.select_row(m5b) and m5b.read_i960(S.CODE_AT, len(body) + 4) == body + b"\xff" * 4


# --- Phase 12 Task 6: the Dreamcast courses' bands and maps (own panels) --------------------------------------------

def _read_band(art: S.PanelArt, strip: np.ndarray) -> tuple[int, int]:
    """(length panel, laps) a band strip (rows 6-8 pens) reads as: its word Sega's word of that length (pixel for
    pixel, the band's rows), then a times sign and LAPS that are Sega's, and its digits Sega's glyphs (each digit run
    one glyph of a band with as many digits)."""
    band = strip[S.BAND_TEXT_TOP:]
    runs = S._runs(np.isin(band, (S.TEXT, S.ACCENT)).any(0))
    digits = [r for r in runs if (band[:, r[0]:r[1]] == S.ACCENT).any()]
    pre = [r for r in runs if r[1] <= digits[0][0]]
    post = [r for r in runs if r[0] >= digits[-1][1]]
    assert len(pre) >= 2 and post and len(pre) + len(digits) + len(post) == len(runs)

    def cols(img, lo, hi):
        return img[S.BAND_TEXT_TOP:, lo:hi].tobytes(), hi - lo
    word = cols(strip, pre[0][0], pre[-2][1])
    length = {p for (p, _), b in art.bands.items() if cols(b.pens, *b.word) == word}
    assert len(length) == 1, "the word is no one length's"
    assert cols(strip, *pre[-1]) in {cols(b.pens, *b.times) for b in art.bands.values()}
    assert cols(strip, post[0][0], post[-1][1]) in {cols(b.pens, *b.laps_word) for b in art.bands.values()}
    glyphs = {(len(b.digits), cols(b.pens, *d)): str(b.laps)[i] for b in art.bands.values()
              for i, d in enumerate(b.digits)}
    return length.pop(), int("".join(glyphs[(len(digits), cols(strip, *d))] for d in digits))


def _pens(g: S.Graphics, tiles, rows) -> np.ndarray:
    """The pens of `tiles` (a 10 x 18 map) at panel rows `rows`, columns 1-16, through the generated chars."""
    return np.concatenate([np.concatenate([g.chars[(tiles[r * 18 + c] & TL.CHAR_MASK) - S.CHAR_HOLE]
                                           for c in range(1, 17)], 1) for r in rows])


def test_converted_bands(romset_zip, monkeypatch):
    """Task 6 (Pascal: "the number of laps"): an own panel's band is `<LENGTH> x <N> LAPS` in Sega's band style, its
    word, times sign, digits and LAPS Sega's own glyphs. Sega's nine bands come back exactly (band_pixels with no
    shift), each read as its panel and laps; composed bands of every length read as what they were asked for (whole
    digit groups Sega draws, or digits from glyphs: 12, 41, 84, 21; a digit Sega's bands lack refused), the length's
    word moved left by `band_shift` where the digits need the room (MEDIUM x 20: 2 px, so MEDIUM x 8 / 20 / 80 share
    the word's chars). On the release row (grown set: ids 3-7 Beginner's copies) each own panel's maps, per game mode,
    hold its name in the box and the band of (length_panel, course.laps in that mode); the blink map the placeholder's
    blink frame round the same interior. The row's upload stays within the caps; the frozen row M5B_ENTRIES has no own
    panel."""
    grown, _ = _sets(str(romset_zip))
    art = S.panel_art(grown)
    assert sorted(art.bands) == [(p, m) for p in range(3) for m in range(3)]
    assert [art.bands[(p, m)].laps for p in range(3) for m in range(3)] == [8, 20, 80, 4, 10, 40, 2, 5, 20]
    for (p, m), b in art.bands.items():
        assert S.band_pixels(art, p, b.laps).tolist() == b.pens.tolist(), (p, m)
        assert _read_band(art, b.pens) == (p, b.laps)
    assert S.BAND_WORD_GAP == min(b.times[0] - b.word[1] for b in art.bands.values())
    for p in S.LENGTH_PANELS:
        for laps in (8, 20, 80, 4, 10, 40, 2, 5, 12, 41, 84, 21):
            img = S.band_pixels(art, p, laps, S.band_shift(art, p, [laps]))
            assert _read_band(art, img) == (p, laps), (p, laps)
            assert img[:S.BAND_TEXT_TOP].tolist() == art.interior[S.LABEL_TOP:S.BAND[0]].tolist()
    assert S.band_shift(art, 0, [8, 20, 80]) == 0 and S.band_shift(art, 1, [8, 20, 80]) == 2
    assert S.band_shift(art, 2, [8, 20, 80]) == 0
    for bad in ((1, 3), (0, 6), (0, 1), (0, 99), (0, 100), (0, 0), (0, True), (3, 8), (-1, 8), (None, 8), (True, 8)):
        with pytest.raises(S.SelectError):
            S.band_pixels(art, *bad)
    with pytest.raises(S.SelectError, match="does not fit"):
        S.band_pixels(art, 1, 20)                               # MEDIUM x 20 needs its word moved
    for bad in (-1, 128, 1.0, None):
        with pytest.raises(S.SelectError):
            S.band_pixels(art, 0, 8, bad)
    with pytest.raises(S.SelectError):
        S.band_shift(art, 0, [])
    with pytest.raises(S.SelectError):
        S.band_pixels("art", 0, 8)
    # the lengths: each original's lap nearest its own
    assert [S.length_panel(grown, C.PANEL_COURSES[p]) for p in range(3)] == [0, 1, 2]
    assert S.road_length(grown, 0) < S.road_length(grown, 2) < S.road_length(grown, 1)
    for bad in (16, -1, True, "0"):
        with pytest.raises(S.SelectError):
            S.road_length(grown, bad)
    # the release row on the grown set
    g = S.graphics(grown, S.ENTRIES)
    table = C.course_table(grown)
    a = C.select_assets(grown)
    frames = [TL.tile_map(grown, a.panels[0][0].maps[i]).tiles() for i in (0, 1)]
    inner = {r * 18 + c for r in range(1, 9) for c in range(1, 17)}
    print(f"\nthe release row's upload (grown set): {g.uploaded()} chars in {[n for _, n in g.runs]}")
    assert g.uploaded() <= S.UPLOAD_MAX and len(g.runs) <= S.UPLOAD_RUNS_MAX
    for k, e in enumerate(S.ENTRIES):
        if not e.own_panel:
            assert g.panels[k] is None and g.bands[k] is None
            continue
        assert len(g.panels[k]) == C.MODES and g.windows[k] == S.WINDOW_NONE and g.name_maps[k] is None
        for mode, (normal, blink) in enumerate(g.panels[k]):
            length, laps, shift = g.bands[k][mode]
            assert (length, laps) == (S.length_panel(grown, e.course), C.laps(grown, table[e.course], mode))
            assert _read_band(art, _pens(g, normal, range(6, 9))) == (length, laps), (k, mode)
            assert _pens(g, normal, range(1, 9)).tolist() == S.own_pixels(art, e, length, laps, shift).tolist()
            assert _pens(g, normal, range(1, 6)).tolist() == \
                S.panel_pixels(art, e._replace(own_panel=False))[:S.LABEL_TOP].tolist()
            for i in range(180):
                assert (normal[i], blink[i]) == ((normal[i],) * 2 if i in inner else (frames[0][i], frames[1][i]))
            assert all(normal[i] & S.FRONT and not normal[i] & S.GREY_BIT for i in inner)
    # own_panel's refusals and the frozen rows
    for bad in (S.ENTRIES[0]._replace(own_panel=True), S.ENTRIES[3]._replace(label="X"),
                S.ENTRIES[3]._replace(difficulty="CIRCUIT EDITION"), S.ENTRIES[3]._replace(own_panel=1)):
        with pytest.raises(S.SelectError):
            S.check_entries(S.ENTRIES[:3] + (bad,) + S.ENTRIES[4:] if bad.course == 3 else (bad,) + S.ENTRIES[1:])
    with pytest.raises(S.SelectError):
        S.check_entries(S.M5A_ENTRIES[:8] + (S.M5A_ENTRIES[8]._replace(own_panel=True),) + S.M5A_ENTRIES[9:])
    with pytest.raises(S.SelectError, match="own_panel"):
        S.panel_pixels(art, S.ENTRIES[3])
    with pytest.raises(S.SelectError, match="own_panel"):
        S.own_pixels(art, S.M5B_ENTRIES[3], 0, 8)
    assert not any(e.own_panel for row in (S.M5B_ENTRIES, S.M5A_ENTRIES, S.M4B_ENTRIES) for e in row)
    assert S.graphics(grown, S.M5B_ENTRIES).panels == (None,) * 8


def test_select_maps(romset_zip, monkeypatch):
    """Task 6 (Pascal: "the track representation below their respective names"): patch_select(grown, ENTRIES) gives
    every MAPS record of an own panel's entry the list of its course's map (one placement: panel 0's map placement
    with selmap.course_map's object and headers swapped in; one list per course whatever the position), the
    originals' records Sega's; the objects, headers and lists at the last free bytes of the custom polygon and
    textures areas and of the list home, owned (map_spans, golden.owned_spans), and the object library's bound
    patched (course.LIBRARY_BOUND_PATCH, which a list in the list home needs)."""
    from dtools.arcade import golden as G
    from dtools.arcade import polyrom as P
    from dtools.arcade import selmap as M
    from dtools.arcade import texture as T
    from dtools.arcade import tgp
    from dtools.convert import build
    grown, _ = _sets(str(romset_zip))
    monkeypatch.setattr(S, "_unfilled", lambda rs, entries: [])
    rs = S.patch_select(grown, S.ENTRIES)
    stock = [struct.unpack("<5I", grown.read_i960(C.SELECT_MAP_OBJECTS + 32 * p + 12, 20)) for p in range(3)]
    place0 = C.object_list(grown, stock[0][0])[0]
    lists, parts = {}, {}
    for r in S.map_records(rs):
        e = S.ENTRIES[r.first + r.position]
        assert r.place == struct.unpack("<3f", grown.read_i960(C.SELECT_MAP_OBJECTS + 32 * r.position, 12))
        if S.is_original(e):
            assert (r.objects, *r.raw) == stock[S.ORIGINAL_PANEL[e.course]]
            continue
        assert e.own_panel and lists.setdefault(e.course, r.objects) == r.objects
        assert r.raw == (r.objects, 0, 0, 0) == (r.objects, *stock[0][2:])
        assert C.library_bounds(rs)[0] <= r.objects < C.library_bounds(rs)[1]
        (q,) = C.object_list(rs, r.objects)
        assert (q.obc, q.tpa, q.raw) == (place0.obc, place0.tpa, place0.raw)
        obj = P.decode_object(rs.polygons, C.check_rom_object(q.oba, rs), q.obc, S.MAP_MODE)
        want, heads = M.course_map(grown, e.course)
        assert obj == want
        assert rs.textures[2 * q.tha:2 * q.tha + len(heads)] == heads
        parts[e.course] = len(heads) // 8
        assert [h.colour for h in T.object_headers(rs, q.tha, obj)][:2] == list(M.map_colours(grown, e.course)[:2])
    assert sorted(lists) == [3, 4, 5, 6, 7]
    assert C.patch_library_bound(rs).program == rs.program
    spans = S.map_spans(rs)
    assert sorted({r for r, *_ in spans}) == ["data", "polygons", "textures"]
    assert len(spans) == 5 * 2 + sum(parts.values())                 # a list and an object a map, a header a part
    owned = G.owned_spans(rs)
    assert all(s in owned for s in spans)
    for region, (lo, n) in (("polygons", build.NEW_ID_RUNS["polygons"]), ("textures", build.NEW_ID_RUNS["textures"]),
                            ("data", tgp.DATA_FREE)):
        mine = [(o, k) for r, o, k, _ in spans if r == region]
        assert all(lo <= o and o + k <= lo + n for o, k in mine) and max(o + k for o, k in mine) == lo + n, region
