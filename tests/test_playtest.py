"""tools/desert_city_playtest.py: the M4b set (`--mirrors`) is Phase 9's final set (tests/test_mirrors_mame.py
`m4b_set`; `final_set` is Phase 10's since Task 3, with National Park Speedway at id 4) with Desert City's and its
mirror's start time raised (the controller's ruling, Phase 9 Task 7); the M5a set (`--m5a`, Phase 10 Task 6) is
Phase 10's final set (`final_set`: ids 3, 4 and 7 converted, mirrors 8-12 and 15, the frozen row select.M5A_ENTRIES)
with the start time raised on the converted courses and their mirrors; the M5b set (`--m5b`, Phase 11 Task 5) is the
shipped set (tests/test_courses_mame.py `m5b_set`: ids 3-7 converted, no mirror, the row of eight, the per-course views
last) with the start time raised on ids 3-7 before the views. Offline: the chain's steps are replaced by
recorders (no game data), and each set's bytes are compared with its reference's (game data: the differing bytes are
exactly the start words `build.long_time` writes)."""

import importlib.util
import struct
from pathlib import Path

import numpy as np

from dtools.arcade import course as C
from dtools.arcade import golden as G
from dtools.arcade import mirror as M
from dtools.arcade import select as S
from dtools.arcade.romset import REGIONS, RomSet
from dtools.arcade.tablebase import DATA_BASE
from dtools.convert import build
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)
from tests.test_courses_mame import m5b_set
from tests.test_mirrors_mame import final_set, m4b_set

TOOL = Path(__file__).resolve().parent.parent / "tools" / "desert_city_playtest.py"


def _tool():
    spec = importlib.util.spec_from_file_location("desert_city_playtest", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(monkeypatch) -> list:
    """Replace each step of the chain with a recorder: it logs (step, its arguments) and returns a token naming the
    call, so a step's input shows which step ran before it."""
    calls = []

    def recorder(name):
        def step(*args, **kwargs):
            calls.append((name, args, tuple(sorted(kwargs.items()))))
            return f"{name}#{len(calls)}"
        return step

    for mod, name in ((C, "grow_ids"), (C, "guard_new_ids"), (C, "widen_ids"), (build, "converted_set"),
                      (M, "mirrored_entry"), (C, "grow_rankings"), (S, "patch_select"), (build, "viewed_set")):
        monkeypatch.setattr(mod, name, recorder(name))
    return calls


def test_mirrors_set_is_m4b_set(monkeypatch):
    """`mirror_set` runs the steps of tests/test_mirrors_mame.py `m4b_set` (Phase 9's final set), in its order, on the
    same inputs (the select with Phase 9's row, select.M4B_ENTRIES), and the M4b command keeps the set's NVRAM out of
    MAME's shared directory."""
    tool = _tool()
    calls = _record(monkeypatch)
    m4b_set("stock", "disc")
    reference, calls[:] = list(calls), []
    tool.mirror_set("stock", "disc")
    names = [c[0] for c in calls]
    assert names == ["grow_ids", "guard_new_ids", "widen_ids", "converted_set", "mirrored_entry", "mirrored_entry",
                     "mirrored_entry", "mirrored_entry", "grow_rankings", "patch_select"]
    assert [c[1][1] for c in calls if c[0] == "mirrored_entry"] == [0, 1, 2, 3]
    assert calls[-1][1][1:] == (S.M4B_ENTRIES,)
    assert calls == reference
    cmd = tool.MIRRORS_COMMAND.format(path="/x/out/play/mirrors-m4b")
    assert cmd.startswith("build/mame/model2 daytona ") and '-nvram_directory "/x/out/play/mirrors-m4b/nvram"' in cmd


def test_mirrors_long_times_steps(monkeypatch):
    """`long_times` raises the start time on Desert City's (id 3) and its mirror's (id 11) tables only."""
    tool = _tool()
    calls = []
    monkeypatch.setattr(G, "decode_course", lambda s, k: calls.append(("decode", s, k)) or f"course{k}")
    monkeypatch.setattr(build, "long_time", lambda s, c: calls.append(("long_time", s, c)) or f"{s}+{c}")
    assert tool.long_times("set") == "set+course3+course11"
    assert tool.LONG_TIME_IDS == (3, 11)
    assert calls == [("decode", "set", 3), ("long_time", "set", "course3"), ("decode", "set+course3", 11),
                     ("long_time", "set+course3", "course11")]


def test_mirrors_set_bytes(romset_zip, disc):  # noqa: F811
    """The playtest set (`long_times` of Phase 9's final set) equals tests/test_mirrors_mame.py `m4b_set` image for
    image but for the start words: the bytes that differ are exactly those of the first word of each time table of ids 3
    and 11 (the same tables: the mirror shares Desert City's) that LONG_TIME changes, and each such word reads
    LONG_TIME. The originals' and their mirrors' time tables are untouched."""
    tool = _tool()
    ref = m4b_set(RomSet.from_zip(romset_zip), disc)
    out = tool.long_times(ref)
    words = set()
    for k in (3, 11):
        words |= {t - DATA_BASE for _, t, _ in G.decode_course(ref, k).time_tables.values()}
    assert words == {t - DATA_BASE for _, t, _ in G.decode_course(ref, 3).time_tables.values()}
    new = struct.pack("<I", build.LONG_TIME)
    expected = {w + i for w in words for i in range(4) if ref.data[w + i] != new[i]}
    assert expected
    for r in REGIONS:
        a, b = getattr(ref, r), getattr(out, r)
        assert len(a) == len(b), r
        if r != "data":
            assert a == b, r
    diff = set(np.flatnonzero(np.frombuffer(ref.data, np.uint8) != np.frombuffer(out.data, np.uint8)).tolist())
    assert diff == expected
    assert all(out.data[w:w + 4] == new for w in words)
    for k in (0, 1, 2, 8, 9, 10):
        assert not words & {t - DATA_BASE for _, t, _ in G.decode_course(ref, k).time_tables.values()}, k


def test_m5a_set_is_final_set(monkeypatch):
    """`m5a_set` runs the steps of tests/test_mirrors_mame.py `final_set` (Phase 10's final set), in its order, on the
    same inputs: Desert City, National Park Speedway and Mermaid Lake converted at their own ids, the mirrors of 0-4
    and 7, the rankings, the select with Phase 10's frozen row (select.M5A_ENTRIES: 0-4, 7 and 8-11 live; 5, 6 and
    12-15 locked, so mirrors 12 and 15 are on the set but not reachable from the select). The M5a command keeps the
    set's NVRAM in its own directory."""
    tool = _tool()
    calls = _record(monkeypatch)
    final_set("stock", "disc")
    reference, calls[:] = list(calls), []
    tool.m5a_set("stock", "disc")
    names = [c[0] for c in calls]
    assert names == (["grow_ids", "guard_new_ids", "widen_ids"] + ["converted_set"] * 3 + ["mirrored_entry"] * 6
                     + ["grow_rankings", "patch_select"])
    assert [c[1][2] for c in calls if c[0] == "converted_set"] == [3, 4, 7]
    assert [c[1][1] for c in calls if c[0] == "mirrored_entry"] == [0, 1, 2, 3, 4, 7]
    assert calls[-1][1][1:] == (S.M5A_ENTRIES,) and calls[-1][2] == ()
    assert {e.course for e in S.M5A_ENTRIES if e.live} == {0, 1, 2, 3, 4, 7, 8, 9, 10, 11}
    assert calls == reference
    cmd = tool.M5A_COMMAND.format(path="/x/out/play/m5a")
    assert cmd.startswith("build/mame/model2 daytona ") and '-nvram_directory "/x/out/play/m5a/nvram"' in cmd


def test_m5a_row_frozen(monkeypatch):
    """Phase 11 Task 1: `--m5a` builds the select with Phase 10's frozen row (select.M5A_ENTRIES), never the live
    ENTRIES: with ENTRIES replaced by another row, `m5a_set` still passes M5A_ENTRIES to patch_select, as the tests'
    `final_set` does, so Phase 11's new row moves no M5a byte."""
    tool = _tool()
    monkeypatch.setattr(S, "ENTRIES", S.M4B_ENTRIES)
    calls = _record(monkeypatch)
    tool.m5a_set("stock", "disc")
    assert calls[-1][0] == "patch_select" and calls[-1][1][1:] == (S.M5A_ENTRIES,) and calls[-1][2] == ()
    assert calls[-1][1][1] is not S.ENTRIES
    final_set("stock", "disc")
    assert calls[-1][1][1:] == (S.M5A_ENTRIES,)


def test_m5b_row_frozen(monkeypatch):
    """Phase 12 Task 1: `--m5b` builds the select with Phase 11's frozen row (select.M5B_ENTRIES), never the live
    ENTRIES (Task 2 changes it to named entries): the row passed to patch_select is M5B_ENTRIES itself (identity), and
    with ENTRIES replaced by another row `m5b_set` still passes it, as the tests' `m5b_set` does."""
    tool = _tool()
    monkeypatch.setattr(S, "ENTRIES", S.M4B_ENTRIES)
    calls = _record(monkeypatch)
    tool.m5b_set("stock", "disc", ())
    sel = [c for c in calls if c[0] == "patch_select"]
    assert len(sel) == 1 and sel[0][1][1] is S.M5B_ENTRIES and sel[0][1][1] is not S.ENTRIES and sel[0][2] == ()
    calls[:] = []
    m5b_set("stock", "disc")
    sel = [c for c in calls if c[0] == "patch_select"]
    assert len(sel) == 1 and sel[0][1][1] is S.M5B_ENTRIES


def test_m5a_long_times_steps(monkeypatch):
    """The M5a set raises the start time on the converted courses (3, 4, 7) and their mirrors (11, 12, 15) only."""
    tool = _tool()
    calls = []
    monkeypatch.setattr(G, "decode_course", lambda s, k: calls.append(("decode", s, k)) or f"c{k}")
    monkeypatch.setattr(build, "long_time", lambda s, c: calls.append(("long_time", s, c)) or f"{s}+{c}")
    assert tool.M5A_LONG_TIME_IDS == (3, 4, 7, 11, 12, 15)
    assert tool.long_times("set", tool.M5A_LONG_TIME_IDS) == "set+c3+c4+c7+c11+c12+c15"
    assert [c[2] for c in calls if c[0] == "decode"] == [3, 4, 7, 11, 12, 15]


def test_m5a_set_bytes(romset_zip, disc):  # noqa: F811
    """The M5a playtest set (`long_times(..., M5A_LONG_TIME_IDS)` of Phase 10's final set) equals tests/
    test_mirrors_mame.py `final_set` image for image but for the start words: the bytes that differ are exactly those
    of the first word of each time table of ids 3, 4 and 7 (their mirrors 11, 12 and 15 share them) that LONG_TIME
    changes, and each such word reads LONG_TIME. The originals' and their mirrors' time tables are untouched."""
    tool = _tool()
    ref = final_set(RomSet.from_zip(romset_zip), disc)
    out = tool.long_times(ref, tool.M5A_LONG_TIME_IDS)

    def words(k):
        return {t - DATA_BASE for _, t, _ in G.decode_course(ref, k).time_tables.values()}

    raised = set().union(*(words(k) for k in tool.M5A_LONG_TIME_IDS))
    assert raised == words(3) | words(4) | words(7)
    for o in (3, 4, 7):
        assert words(o + 8) == words(o), o
    new = struct.pack("<I", build.LONG_TIME)
    expected = {w + i for w in raised for i in range(4) if ref.data[w + i] != new[i]}
    assert expected
    for r in REGIONS:
        a, b = getattr(ref, r), getattr(out, r)
        assert len(a) == len(b), r
        if r != "data":
            assert a == b, r
    diff = set(np.flatnonzero(np.frombuffer(ref.data, np.uint8) != np.frombuffer(out.data, np.uint8)).tolist())
    assert diff == expected
    assert all(out.data[w:w + 4] == new for w in raised)
    for k in (0, 1, 2, 8, 9, 10):
        assert not raised & words(k), k


def test_m5b_set_is_m5b_set(monkeypatch):
    """`m5b_set` with no start word raised (`long_ids=()`) runs the steps of tests/test_courses_mame.py `m5b_set` (the
    shipped set), in its order, on the same inputs: ids 3, 4, 7, 6 and 5 converted at their own ids, no mirror, the
    rankings, the select with the row of eight (select.M5B_ENTRIES), then `build.viewed_set`. By default the start words of
    ids 3-7 are raised (`long_times`) after the select and before `viewed_set`, which stays the last step (its tables
    are not in golden.owned_spans). The M5b command keeps the set's NVRAM in its own directory, out/play/m5b/nvram."""
    tool = _tool()
    calls = _record(monkeypatch)
    m5b_set("stock", "disc")
    reference, calls[:] = list(calls), []
    tool.m5b_set("stock", "disc", ())
    names = [c[0] for c in calls]
    assert names == (["grow_ids", "guard_new_ids", "widen_ids"] + ["converted_set"] * 5
                     + ["grow_rankings", "patch_select", "viewed_set"])
    assert [c[1][2] for c in calls if c[0] == "converted_set"] == [3, 4, 7, 6, 5] == list(tool.M5B_CONVERTED)
    assert calls[-2][1][1:] == (S.M5B_ENTRIES,) and calls[-2][2] == ()
    assert calls[-1][1] == ("patch_select#10",) and calls[-1][2] == ()
    assert calls == reference
    calls[:] = []
    monkeypatch.setattr(G, "decode_course", lambda s, k: calls.append(("decode", (s, k), ())) or f"c{k}")
    monkeypatch.setattr(build, "long_time", lambda s, c: calls.append(("long_time", (s, c), ())) or f"{s}+{c}")
    tool.m5b_set("stock", "disc")
    assert tool.M5B_LONG_TIME_IDS == (3, 4, 5, 6, 7)
    assert [c[1][1] for c in calls if c[0] == "decode"] == [3, 4, 5, 6, 7]
    assert [c[0] for c in calls][10:] == ["decode", "long_time"] * 5 + ["viewed_set"]
    assert calls[-1][1] == ("patch_select#10+c3+c4+c5+c6+c7",)
    cmd = tool.M5B_COMMAND.format(path="/x/out/play/m5b")
    assert cmd.startswith("build/mame/model2 daytona ") and '-nvram_directory "/x/out/play/m5b/nvram"' in cmd


def test_m5b_set_bytes(romset_zip, disc):  # noqa: F811
    """The M5b playtest set (`m5b_set`: the start words of ids 3-7 raised, then `build.viewed_set`) equals tests/
    test_courses_mame.py `m5b_set` image for image but for the start words: the bytes that differ are exactly those of
    the first word of each time table of ids 3-7 that LONG_TIME changes, and each such word reads LONG_TIME; the
    originals' time tables are untouched. Raising the start words before `viewed_set` leaves the views' bytes where
    they are: the set equals `long_times` applied after `viewed_set` (the reference with its start words raised),
    region for region, and ids 5 and 6 read the same view-test tables as on the reference."""
    tool = _tool()
    ref = m5b_set(RomSet.from_zip(romset_zip), disc)
    out = tool.m5b_set(RomSet.from_zip(romset_zip), disc)

    def words(k):
        return {t - DATA_BASE for _, t, _ in G.decode_course(ref, k).time_tables.values()}

    raised = set().union(*(words(k) for k in tool.M5B_LONG_TIME_IDS))
    new = struct.pack("<I", build.LONG_TIME)
    expected = {w + i for w in raised for i in range(4) if ref.data[w + i] != new[i]}
    assert expected
    for r in REGIONS:
        a, b = getattr(ref, r), getattr(out, r)
        assert len(a) == len(b), r
        if r != "data":
            assert a == b, r
    diff = set(np.flatnonzero(np.frombuffer(ref.data, np.uint8) != np.frombuffer(out.data, np.uint8)).tolist())
    assert diff == expected
    assert all(out.data[w:w + 4] == new for w in raised)
    for k in (0, 1, 2):
        assert not raised & words(k), k
    after = tool.long_times(ref, tool.M5B_LONG_TIME_IDS)          # the other order: the views first, then the words
    assert all(getattr(after, r) == getattr(out, r) for r in REGIONS)
    assert all(C.view_tables(out, k) == C.view_tables(ref, k) for k in range(C.COURSE_IDS))
