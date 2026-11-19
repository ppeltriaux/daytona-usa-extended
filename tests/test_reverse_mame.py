"""MAME tests of M9 (reverse courses; spec docs/superpowers/specs/2026-11-16-reverse-courses-design.md, discovery log
docs/spike/2026-11-reverse.md). Stage 1: Desert City in reverse at id 11, reached from a test-only select row.

Marked `mame` and `custom_mame` (the set is a custom one)."""

import pytest

from dtools import config
from dtools.arcade import course as C
from dtools.arcade import select as S
from dtools.arcade.romset import RomSet
from tests.test_convert_tables import disc  # noqa: F401  (disc: the module-scoped disc fixture)

REVERSE_NAMES = {8: "THREE SEVEN REVERSE", 9: "SEA-SIDE REVERSE", 10: "DINOSAUR REVERSE", 11: "DESERT CITY REVERSE",
                 12: "NATIONAL PARK REVERSE", 13: "CIRCUIT PIXIE REVERSE", 14: "RIN RIN RINK REVERSE",
                 15: "MERMAID LAKE REVERSE"}
REVERSE_TEST_ROW = tuple(e._replace(own_panel=False) for e in S.ENTRIES) + tuple(
    S.Entry(k, REVERSE_NAMES.get(k, "REVERSE"), "", False, k in REVERSE_NAMES) for k in range(8, 16))
"""The release row without its own panels (Phase 12 Task 6's painted maps), followed by the reverse ids 8-15 (a row of
16: select.check_entries wants a permutation of 0..n - 1), of which the arcade's three in reverse (entries 8-10,
ids 8-10) and the converted courses' (entries 11-15, ids 11-15) are live: the test hook (the player's choice is stage 3's NORMAL / REVERSE screen). With the own panels the row of
16 uploads 214 generated chars in 5 runs, past select.UPLOAD_RUNS_MAX (4)."""
REV_SNAP_DIR = config.out_dir() / "m9" / "stage1"

REV11_GO = 1651
REV11_REPEAT = frozenset({REV11_GO + 1})
REV11_SECONDS = 185
REV11_LAP_END = 5315
REV11_LONG_LAPS = 2
REV11_TIME_OVER = 6051
REV11_RISES = 1
REV11_MAX = (3825, 1771, 4651)
CALLS_REV11_LONG_TIME = {0: 32315, 1: 3481, 2: 1849, 3: 2057, 4: 1718, 5: 2787, 6: 5109, 7: 5713, 8: 2227, 9: 2809, 10: 2738, 11: 688, 12: 279, 13: 1894, 14: 188, 15: 234, 16: 198, 17: 198, 18: 1994, 19: 1549, 20: 163, 21: 4233, 22: 486, 23: 224, 24: 513, 25: 1302, 26: 439, 27: 403, 28: 231, 29: 141, 30: 746, 31: 202, 32: 284, 33: 291, 34: 211, 35: 208, 36: 370, 37: 268, 38: 160, 39: 398}
CALLS_REV11_TO_TIME_OVER = {0: 22244, 1: 3361, 2: 2204, 3: 704, 4: 51, 5: 14, 8: 33, 9: 12, 32: 7, 36: 357, 37: 33, 38: 42, 39: 18}
"""Desert City in reverse (id 11) on reverse_set, entry 11 of REVERSE_TEST_ROW: GO, the lists the start repeats, the
long-time run's length and the player's lap end (the autopilot), the AI cars' laps, the time-over and the clock's rises
before it (in the session the player rolls on to the first reverse checkpoint, segment 201, at frame 4656: the clock
rises at 4657 by the placeholder tables' extension, 1200, so the start time runs out at 6051), the lap's drawn maximum
(drawn, frame, the count's maximum) and the ground calls per car (long-time run, session to the time-over). Measured
16 Nov."""

_SETS: dict = {}


def reverse_set(rs: RomSet, disc) -> RomSet:  # noqa: F811
    """The release chain's conversions with the reverses after them, without times, pits or title: grow_ids,
    guard_new_ids, widen_ids, the five converted at their ids (release.RELEASE_CONVERTED), the arcade's three in
    reverse (converted_set(..., n, slot=n + 8, variant="R", scenery="keep"), stage 2), the five converted in reverse
    (converted_set(..., n, slot=n + 8, variant="R", scenery="model"), n 3-7), grow_rankings, the select with REVERSE_TEST_ROW, the views (build.REVERSE_VIEW)."""
    from dtools.convert import build
    from dtools.release import RELEASE_CONVERTED
    out = C.widen_ids(C.guard_new_ids(C.grow_ids(rs)))
    for n in RELEASE_CONVERTED:
        out = build.converted_set(out, disc, n, slot=n, scenery="model")
    for n in (0, 1, 2):                                             # stage 2: the arcade's three in reverse
        out = build.converted_set(out, disc, n, slot=n + 8, variant="R", scenery="keep")
    for n in range(3, 8):                                           # Desert City's (stage 1), then step 3's
        out = build.converted_set(out, disc, n, slot=n + 8, variant="R", scenery="model")
    out = S.patch_select(C.grow_rankings(out), REVERSE_TEST_ROW)
    return build.viewed_set(out, build.REVERSE_VIEW)


def _reverse_set(romset_zip, disc) -> RomSet:  # noqa: F811
    key = str(romset_zip)
    if key not in _SETS:
        _SETS[key] = reverse_set(RomSet.from_zip(romset_zip), disc)
    return _SETS[key]


def reverse_gate():
    """Desert City reverse's gate (tests/test_courses_mame.py `_gate`), snapshots to REV_SNAP_DIR."""
    from tests.test_courses_mame import Gate
    return Gate(course=11, entry=11, name="desert-city-reverse", label="Desert City reverse", dest=str(REV_SNAP_DIR),
                seconds=REV11_SECONDS, go=REV11_GO, repeat=REV11_REPEAT, lap_end=REV11_LAP_END,
                long_laps=REV11_LONG_LAPS, maximum=REV11_MAX, calls_long_time=CALLS_REV11_LONG_TIME,
                calls_to_time_over=CALLS_REV11_TO_TIME_OVER, time_over=REV11_TIME_OVER, rises=REV11_RISES,
                snaps={"select": 750, "start": REV11_GO + 30, "mid-lap": REV11_GO + 2400}, autopilot=1)


@pytest.mark.mame
@pytest.mark.custom_mame
def test_desert_city_reverse_id11(tmp_path, romset_zip, disc):  # noqa: F811
    """Desert City in reverse at id 11, entered from the select (REVERSE_TEST_ROW entry 11), custom MAME:
    test_national_park_id4's checks (`_gate`): the long-time run (the autopilot drives the player's lap, every
    segment of the reverse road; the AI field laps; every ground call a polygon; the lap's drawn maximum at most
    BAR - THIN_MARGIN, no cap hit, no visible cell left out, none drawn outside; frozen REV11_MAX; SRAM fresh after),
    then the time-over session and Beginner after it, against Beginner then Beginner."""
    from tests.test_courses_mame import _gate
    full = _reverse_set(romset_zip, disc)
    assert S.select_row(full).records[0][11].course == 11
    _gate(tmp_path, romset_zip, full, reverse_gate())


# ---- stage 2, step 2: the arcade's three in reverse (ids 8-10) ----

REV_SNAP_DIR2 = config.out_dir() / "m9" / "stage2"
ORIGINALS = {8: ("three-seven-reverse", "Three Seven Speedway reverse"),
             9: ("sea-side-reverse", "Sea-Side Street Galaxy reverse"),
             10: ("dinosaur-canyon-reverse", "Dinosaur Canyon reverse")}
"""Id (= entry of REVERSE_TEST_ROW) -> (snapshot stem, label)."""
REV_ORIGINAL = {
    8: dict(seconds=185, go=1885, repeat=frozenset(), lap_end=3094, long_laps=(5, 6, 7), maximum=(3625, 1890, 4568),
            calls_long_time={0: 33472, 1: 1545, 2: 4859, 3: 1747, 4: 2924, 5: 1483, 6: 2859, 7: 2559, 8: 1944, 9: 2201,
                             10: 1628, 11: 2423, 12: 603, 13: 1581, 14: 1128, 15: 495, 16: 974, 17: 719, 18: 1325,
                             19: 1561, 20: 1001, 21: 1473, 22: 888, 23: 403, 24: 1239, 25: 764, 26: 437, 27: 1491,
                             28: 956, 29: 903, 30: 1101, 31: 938, 32: 1363, 33: 1825, 34: 1057, 35: 390, 36: 996,
                             37: 1554, 38: 515, 39: 321},
            calls_to_time_over={0: 20899, 1: 1374, 2: 861, 3: 1327, 4: 1370, 5: 2075, 6: 1089, 7: 1070, 8: 585,
                                9: 342, 10: 1389, 11: 1025, 12: 88, 13: 139, 14: 556, 15: 509, 16: 1375, 17: 163,
                                18: 1330, 19: 1089, 20: 478, 21: 56, 22: 795, 23: 207, 24: 1266, 25: 1089, 26: 256,
                                27: 326, 28: 817, 29: 646, 30: 349, 31: 365, 32: 143, 33: 444, 34: 159, 35: 92,
                                36: 861, 37: 444, 38: 114, 39: 33},
            time_over=7126, rises=2, left_out=frozenset()),
    9: dict(seconds=240, go=1652, repeat=frozenset({1653}), lap_end=10295, long_laps=1, maximum=(4102, 5477, 4734),
            calls_long_time={0: 57039, 1: 4866, 2: 7004, 3: 2771, 4: 10619, 5: 579, 6: 39, 7: 274, 8: 3417, 9: 3833,
                             18: 4711},
            calls_to_time_over={0: 17838, 1: 1246, 2: 202, 3: 455, 4: 628, 5: 1348, 6: 27, 7: 408, 8: 52, 9: 657,
                                11: 4, 21: 8, 22: 30, 24: 10, 25: 13, 27: 11, 29: 12},
            time_over=5491, rises=0, left_out=frozenset()),
    10: dict(seconds=185, go=1651, repeat=frozenset({1652}), lap_end=5335, long_laps=2, maximum=(4059, 2899, 4639),
             calls_long_time={0: 40733, 1: 4026, 2: 3325, 3: 5780, 4: 3679, 5: 1774, 6: 1137, 7: 1740, 8: 2110,
                              9: 1842, 10: 243, 11: 351, 12: 233, 13: 190, 14: 252, 15: 288, 16: 264, 17: 280,
                              18: 1492, 19: 73},
             calls_to_time_over={0: 19215, 1: 2152, 2: 963, 3: 401, 4: 743, 5: 139, 8: 171},
             time_over=5490, rises=0, left_out=frozenset()),
}
"""The frozen numbers of each reverse id's gate (`_gate`, budget "bar"), measured 17 Nov 2026 on reverse_set (ids 9 and
10 with build.REVERSE_VIEW's VIEW_9): GO (id 8's rolling start ticks at 1885, ids 9 and 10's standing starts at 1652
and 1651, a list repeated the frame after), the player's lap end (the autopilot), the AI cars' laps (Three Seven
Speedway's short oval: 5 to 7), the lap's drawn maximum (drawn, frame, the count's maximum: all under BAR, no cap hit),
the ground calls per car of the long-time run and of the session to the time-over, the time-over (the placeholder
times: 3200 ticks on id 8, whose clock rises at its two checkpoints, 3840 on ids 9 and 10) and its rises."""


def original_reverse_gate(k: int):
    """Reverse id k's gate (tests/test_courses_mame.py `_gate`, budget "bar": BAR and no cap hit), snapshots to
    REV_SNAP_DIR2."""
    from tests.test_courses_mame import Gate
    v = REV_ORIGINAL[k]
    name, label = ORIGINALS[k]
    return Gate(course=k, entry=k, name=name, label=label, dest=str(REV_SNAP_DIR2), seconds=v["seconds"], go=v["go"],
                repeat=v["repeat"], lap_end=v["lap_end"], long_laps=v["long_laps"], maximum=v["maximum"],
                calls_long_time=v["calls_long_time"], calls_to_time_over=v["calls_to_time_over"],
                time_over=v["time_over"], rises=v["rises"],
                snaps={"select": 750, "start": v["go"] + 30, "mid-lap": v["go"] + 2400}, autopilot=1, budget="bar",
                left_out=v["left_out"])


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("k", sorted(ORIGINALS))
def test_original_in_reverse(tmp_path, romset_zip, disc, k):  # noqa: F811
    """The arcade's three in reverse at ids 8-10, entered from the select (REVERSE_TEST_ROW entries 8-10), custom
    MAME: `_gate`'s checks with the arcade's scenery (budget "bar"): the autopilot laps the whole reverse road, the AI
    field laps, every ground call a polygon (step 1's polygons with corner 1 on the 0-2 diagonal), at most BAR drawn
    polygons and no cap hit on every frame of the lap, no scenery drawn outside the visible cells and none left out
    but at the race's load (left_out), the frozen numbers (REV_ORIGINAL), SRAM fresh after; then the time-over
    session and Beginner after it."""
    from tests.test_courses_mame import _gate
    full = _reverse_set(romset_zip, disc)
    assert S.select_row(full).records[0][k].course == k
    _gate(tmp_path, romset_zip, full, original_reverse_gate(k))


# ---- stage 2, step 3: the four other converted courses in reverse (ids 12-15) ----

CONVERTED = {12: ("national-park-reverse", "National Park Speedway reverse"),
             13: ("circuit-pixie-reverse", "Circuit Pixie reverse"),
             14: ("rin-rin-rink-reverse", "Rin Rin Rink reverse"),
             15: ("mermaid-lake-reverse", "Mermaid Lake reverse")}
REV_CONVERTED = {
    12: dict(seconds=185, go=1651, repeat=frozenset({1652}), lap_end=4894, long_laps=(2, 3),
             maximum=(3811, 1829, 4406),
             calls_long_time={0: 36130, 1: 3525, 2: 2910, 3: 2381, 4: 3545, 5: 964, 6: 3472, 7: 970, 8: 1764, 9: 2718,
                              10: 3615, 11: 153, 12: 3392, 13: 419, 14: 774, 15: 433, 16: 1639, 17: 304, 18: 2242,
                              19: 1383, 20: 1295, 21: 520, 22: 121, 23: 627, 24: 379, 25: 517, 26: 960, 27: 618,
                              28: 391, 29: 515, 30: 338, 31: 1092, 32: 103, 33: 631, 34: 62, 35: 625, 36: 75, 37: 918,
                              38: 283, 39: 214},
             calls_to_time_over={0: 21072, 1: 3770, 2: 768, 3: 1214, 4: 87, 8: 49, 9: 375, 29: 48, 31: 8, 33: 735,
                                 37: 670, 39: 786},
             time_over=6051, rises=1),
    13: dict(seconds=185, go=1651, repeat=frozenset({1652}), lap_end=4349, long_laps=(4, 5, 6),
             maximum=(3749, 1818, 4528),
             calls_long_time={0: 22093, 1: 1897, 2: 1506, 3: 483, 4: 1439, 5: 2316, 6: 1299, 7: 1087, 8: 1149,
                              9: 1318, 10: 2438, 11: 1239, 12: 2761, 13: 404, 14: 1008, 15: 1297, 16: 1719, 17: 2448,
                              18: 1147, 19: 1589, 20: 1875, 21: 2230, 22: 1487, 23: 1724, 24: 2744, 25: 3320,
                              26: 1925, 27: 1454, 28: 1554, 29: 659, 30: 602, 31: 766, 32: 1933, 33: 3025, 34: 1290,
                              35: 2100, 36: 241, 37: 1937, 38: 5000, 39: 2677},
             calls_to_time_over={0: 13485, 1: 511, 2: 791, 3: 350, 4: 80, 5: 1428, 6: 178, 7: 187, 10: 351, 11: 1310,
                                 12: 1044, 13: 421, 14: 411, 15: 1128, 16: 670, 17: 1133, 18: 881, 19: 337, 20: 1273,
                                 21: 402, 22: 704, 23: 421, 24: 794, 25: 1259, 26: 1117, 27: 397, 28: 625, 29: 288,
                                 30: 229, 31: 160, 32: 572, 33: 1114, 34: 1073, 35: 54, 36: 213, 37: 97, 38: 3442,
                                 39: 1808},
             time_over=6051, rises=1),
    14: dict(seconds=185, go=1651, repeat=frozenset({1652}), lap_end=4645, long_laps=(2, 3),
             maximum=(3660, 2587, 4564),
             calls_long_time={0: 35580, 1: 3019, 2: 1123, 3: 2221, 4: 1137, 5: 5198, 6: 3499, 7: 3159, 8: 1214,
                              9: 1924, 10: 777, 11: 1297, 12: 964, 13: 887, 14: 266, 15: 386, 16: 692, 17: 534,
                              18: 2114, 19: 3389, 20: 2093, 21: 421, 22: 451, 23: 719, 24: 158, 25: 340, 26: 139,
                              27: 465, 28: 367, 29: 737, 30: 247, 31: 294, 32: 589, 33: 283, 34: 221, 35: 481,
                              36: 344, 37: 604, 38: 404, 39: 217},
             calls_to_time_over={0: 22022, 1: 1957, 2: 1855, 3: 1716, 4: 425, 5: 97, 6: 23, 8: 1276, 9: 70, 39: 100},
             time_over=6051, rises=1),
    15: dict(seconds=185, go=1651, repeat=frozenset({1652}), lap_end=5030, long_laps=(2, 3),
             maximum=(3894, 1715, 4579),
             calls_long_time={0: 37393, 1: 4520, 2: 2631, 3: 4194, 4: 2486, 5: 4144, 6: 2631, 7: 1298, 8: 2041,
                              9: 1967, 10: 1562, 11: 306, 12: 396, 13: 574, 14: 314, 15: 160, 16: 213, 17: 120,
                              18: 1763, 19: 1122, 20: 103, 21: 115, 22: 122, 23: 135, 24: 227, 25: 154, 26: 82,
                              27: 190, 28: 182, 29: 133, 30: 130, 31: 279, 32: 270, 33: 160, 34: 158, 35: 379,
                              36: 150, 37: 864, 38: 318, 39: 1039},
             calls_to_time_over={0: 21588, 1: 2499, 2: 964, 3: 1386, 4: 428, 5: 132, 6: 142, 8: 89, 9: 343, 35: 185,
                                 36: 32, 37: 266, 38: 52, 39: 325},
             time_over=6051, rises=1),
}
"""The frozen numbers of each converted reverse's gate (`_gate`, budget "thin"), measured 17 Nov 2026 on reverse_set
(build.REVERSE_NAMED_PROPS, REVERSE_THIN, REVERSE_REMESH and REVERSE_VIEW: VIEW_13 on id 13, VIEW_12 on id 14): GO
1651 (the standing start, a list repeated the frame after), the player's lap end (the autopilot), the AI cars' laps,
the lap's drawn maximum (drawn, frame, the count's maximum: all within BAR - THIN_MARGIN and OBC - THIN_MARGIN), the
ground calls per car of the long-time run and of the session to the time-over, the time-over (the grown ids' default
tables: 3200 ticks, the clock rising once at the first reverse checkpoint by 1200, as on Desert City's reverse)."""


def converted_reverse_gate(k: int):
    """Converted reverse id k's gate (`_gate`, budget "thin": BAR - THIN_MARGIN, count OBC - THIN_MARGIN), snapshots
    to REV_SNAP_DIR2."""
    from tests.test_courses_mame import Gate
    v = REV_CONVERTED[k]
    name, label = CONVERTED[k]
    return Gate(course=k, entry=k, name=name, label=label, dest=str(REV_SNAP_DIR2), seconds=v["seconds"], go=v["go"],
                repeat=v["repeat"], lap_end=v["lap_end"], long_laps=v["long_laps"], maximum=v["maximum"],
                calls_long_time=v["calls_long_time"], calls_to_time_over=v["calls_to_time_over"],
                time_over=v["time_over"], rises=v["rises"],
                snaps={"select": 750, "start": v["go"] + 30, "mid-lap": v["go"] + 2400}, autopilot=1)


@pytest.mark.mame
@pytest.mark.custom_mame
@pytest.mark.parametrize("k", sorted(CONVERTED))
def test_converted_in_reverse(tmp_path, romset_zip, disc, k):  # noqa: F811
    """The four other converted courses in reverse at ids 12-15, from the select (REVERSE_TEST_ROW entries 12-15),
    custom MAME: test_desert_city_reverse_id11's checks (`_gate`, the converted courses' budget), the frozen numbers
    REV_CONVERTED."""
    from tests.test_courses_mame import _gate
    full = _reverse_set(romset_zip, disc)
    assert S.select_row(full).records[0][k].course == k
    _gate(tmp_path, romset_zip, full, converted_reverse_gate(k))

