"""Race times, Task 2: the tables derived from synthetic measurements (no ROM, no MAME)."""
import importlib.util
import math
import statistics
from pathlib import Path

import pytest

from dtools.convert import racetimes as R
from dtools.mame.runner import Progress

SPIKE = Path(__file__).resolve().parents[1] / "spike" / "race_times.py"
"""The measuring spike; not in the public export (spike/ is not exported), where its one test skips."""
rt = None
if SPIKE.is_file():
    _spec = importlib.util.spec_from_file_location("race_times", SPIKE)
    rt = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rt)

CP = {0: 0, 1: 2, 2: 1, 3: 2, 4: 2, 5: 0, 6: 1, 7: 2}
LAPS = {i: {0: 3, 1: 2, 2: 2} for i in range(8)}
SECT = {0: [1000], 1: [500, 300, 700], 2: [400, 900], 3: [600, 400, 500], 4: [300, 300, 600], 5: [800],
        6: [700, 500], 7: [250, 450, 550]}
# difficulty ratios per (mode, d) and per original: mean over the three originals is DR[(mode, d)]
DR = {(m, d): base + 0.1 * m for m, base in ((0, 1), (1, 1), (2, 1)) for d in (0, 2, 3)}
DR = {(m, d): {0: 0.5, 2: 1.5, 3: 2.0}[d] + 0.1 * m for m in range(3) for d in (0, 2, 3)}
SPREAD = {0: -0.1, 1: 0.0, 2: 0.1}          # per original, averages out to zero


def paid(i, j, reading):
    """Independent of the module: the sector pass j of course i pays for."""
    k = CP[i]
    return SECT[i][(j + 1) % (k + 1)] if reading == "after" else SECT[i][j]


def meas():
    return {str(i): {"start": s[0] + 100 * i, "sectors": list(s), "lap": sum(s)} for i, s in SECT.items()}


def originals(reading="after", start_f=3.0, ext_f=2.0, flat_start=5.0, flat_ext=1.5, mult=None):
    """Tables of ids 0-2 built from the factors; `mult[i]` multiplies the extension of pass j (non-uniform slack)."""
    m = meas()
    o = {}
    for i in (0, 1, 2):
        k = CP[i]
        o[i] = {}
        for mode in range(3):
            if mode == 0:
                base = [start_f * m[str(i)]["start"]]
                for _ in range(LAPS[i][0]):
                    base += [ext_f * (mult or {}).get(i, [1] * (k + 1))[j] * paid(i, j, reading) for j in range(k + 1)]
            else:
                base = [flat_start * m[str(i)]["lap"]] + [flat_ext * statistics.fmean(SECT[i])] * (LAPS[i][mode] * (k + 1))
            base = [round(w) for w in base]
            for d in range(4):
                if d == 1:
                    o[i][(mode, d)] = tuple(base)
                else:
                    r = DR[(mode, d)] + SPREAD[i]
                    # the phantom (last word) is garbage on purpose: the ratios must leave it out
                    o[i][(mode, d)] = tuple(round(w * r) for w in base[:-1]) + (base[-1] * 7 + 1,)
    return o


def test_factors_recover_known_slack():
    f = R.factors(meas(), originals(), LAPS, CP)
    assert f["mode0"]["start"] == pytest.approx(3.0, rel=1e-3)
    assert f["mode0"]["ext"] == pytest.approx(2.0, rel=1e-3)
    assert f["flat"][1]["start"] == pytest.approx(5.0, rel=1e-3)
    assert f["flat"][2]["ext"] == pytest.approx(1.5, rel=1e-3)
    for (mode, d), r in DR.items():
        assert f["ratios"][(mode, d)]["ratio"] == pytest.approx(r, rel=2e-3)      # phantom excluded, per (mode, d)
        for i in (0, 1, 2):
            assert f["ratios"][(mode, d)]["per_original"][i] == pytest.approx(r + SPREAD[i], rel=2e-3)
    assert f["ratios"][(2, 1)]["ratio"] == 1.0
    assert set(f["mode0"]["per_original"]) == {0, 1, 2}


def test_factor_two_times_sector_times():
    f = R.factors(meas(), originals(start_f=2.0, ext_f=2.0), LAPS, CP)
    assert f["mode0"]["start"] == pytest.approx(2.0, rel=1e-3)
    assert f["mode0"]["ext"] == pytest.approx(2.0, rel=1e-3)


def test_extension_factor_is_a_ratio_of_totals():
    # id 2 (checkpoint + lap line): slack 1x on pass 0, 3x on pass 1
    o = originals(mult={2: [1.0, 3.0]})
    f = R.factors(meas(), o, LAPS, CP)
    reading = f["reading"]
    words = o[2][(0, 1)][1:-1]
    passes = [(i - 1) % 2 for i in range(1, len(o[2][(0, 1)]) - 1)]
    want_totals = sum(words) / sum(paid(2, j, reading) for j in passes)
    want_mean_of_ratios = statistics.fmean(w / paid(2, j, reading) for w, j in zip(words, passes))
    assert f["mode0"]["per_original"][2]["ext"] == pytest.approx(want_totals)
    assert want_totals != pytest.approx(want_mean_of_ratios, rel=1e-3)


@pytest.mark.parametrize("reading", ["after", "before"])
def test_flatter_reading_is_chosen(reading):
    o = originals(reading)
    # check the table really pays for the sector the test says (independent of the module), by hand
    k = CP[1]
    assert o[1][(0, 1)][1:1 + k + 1] == tuple(round(2.0 * paid(1, j, reading)) for j in range(k + 1))
    f = R.factors(meas(), o, LAPS, CP)
    assert f["reading"] == reading
    other = "before" if reading == "after" else "after"
    assert f["readings"][reading]["cv"] < f["readings"][other]["cv"]
    assert f["readings"][reading]["cv"] == pytest.approx(0, abs=1e-3)


def test_before_reading_by_hand_tables():
    # literal numbers: the word at a pass is 2 x the sector BEFORE it (id 1: 500, 300, 700; id 2: 400, 900)
    o = originals()
    o[1][(0, 1)] = (o[1][(0, 1)][0],) + (1000, 600, 1400) * 4
    o[2][(0, 1)] = (o[2][(0, 1)][0],) + (800, 1800) * 2 + (800, 1800)[:0] + (800, 1800)
    o[2][(0, 1)] = o[2][(0, 1)][:7]
    for i in (1, 2):                                # keep the difficulty tables the same length as difficulty 1
        for dd in (0, 2, 3):
            o[i][(0, dd)] = tuple(w * 1 for w in o[i][(0, 1)])
    f = R.factors(meas(), o, LAPS, CP)
    assert f["reading"] == "before"
    assert f["readings"]["before"]["cv"] == pytest.approx(0, abs=1e-9)
    assert f["readings"]["after"]["cv"] > 0.1
    assert f["mode0"]["per_original"][1]["ext"] == pytest.approx(2.0)


def race(i, mode, m=None):
    """Independent of the module: the field's race time on course i (GO to the first pass, the rest of lap 1, then
    laps - 1 laps)."""
    v = (m or meas())[str(i)]
    return v["start"] + sum(v["sectors"][1:]) + (LAPS[i][mode] - 1) * v["lap"]


def ratio(words, i, mode, m=None):
    return (words[0] + sum(words[1:-1])) / race(i, mode, m)


def test_derived_tables_follow_factors():
    o = originals()
    d = R.derive(meas(), o, LAPS, CP)
    m = meas()
    want = {mode: statistics.fmean(ratio(o[i][(mode, 1)], i, mode) for i in (0, 1, 2)) for mode in range(3)}
    assert sorted(d) == [3, 4, 5, 6, 7]
    for cid in d:
        s, k = SECT[cid], CP[cid]
        assert sorted(d[cid]) == [(mo, di) for mo in range(3) for di in range(4)]
        t = d[cid][(0, 1)]
        for lap in range(LAPS[cid][0]):
            for j in range(k + 1):
                assert t[1 + lap * (k + 1) + j] == round(2.0 * s[(j + 1) % (k + 1)])
        # flat modes: ONE identical word on every pass
        for mode in (1, 2):
            w = d[cid][(mode, 1)]
            assert set(w[1:]) == {round(1.5 * statistics.fmean(s))}
            assert len(w) == 1 + LAPS[cid][mode] * (k + 1)
        # the start: the whole race as generous as the originals' mean in that mode (unless the floor binds)
        for mode in range(3):
            w = d[cid][(mode, 1)]
            r = want[mode] if cid != 5 else ratio(o[0][(mode, 1)], 0, mode)
            assert w[0] == max(round(r * race(cid, mode) - sum(w[1:-1])), math.ceil(2.0 * m[str(cid)]["start"]))


def test_race_ratio_matches_the_originals():
    """The point of the start rule: every converted course's allowance over its field's race time equals the
    originals' mean ratio of that mode (to a tick), though the courses' laps and starts differ."""
    o = originals()
    d = R.derive(meas(), o, LAPS, CP)
    f = R.factors(meas(), o, LAPS, CP)
    for mode in range(3):
        per = [ratio(o[i][(mode, 1)], i, mode) for i in (0, 1, 2)]
        assert f["race"][mode]["ratio"] == pytest.approx(statistics.fmean(per))
        for cid in (3, 4, 6, 7):
            w = d[cid][(mode, 1)]
            if w[0] > math.ceil(2.0 * meas()[str(cid)]["start"]):
                assert abs(ratio(w, cid, mode) * race(cid, mode) - f["race"][mode]["ratio"] * race(cid, mode)) <= 0.5


def test_start_floor_binds():
    """A start the race rule would put below twice the field's time to the first pass is raised to it."""
    o = originals(start_f=0.1, ext_f=0.5, flat_start=0.1, flat_ext=0.5)     # stingy originals: race ratios ~0.5
    d = R.derive(meas(), o, LAPS, CP)
    for cid in (3, 4, 5, 6, 7):
        floor = math.ceil(R.START_FLOOR * meas()[str(cid)]["start"])
        w = d[cid][(0, 1)]
        assert round(R.factors(meas(), o, LAPS, CP)["course"][cid]["race"][0] * race(cid, 0) - sum(w[1:-1])) < floor
        assert w[0] == floor


def test_circuit_pixie_takes_three_seven_factors_in_every_mode():
    o = originals()
    # make id 1 differ so the three-course mean differs from id 0's own factors, in every mode
    o[1] = {key: tuple(w * 2 for w in ws) for key, ws in o[1].items()}
    m = meas()
    f = R.factors(m, o, LAPS, CP)
    assert f["mode0"]["start"] != pytest.approx(f["mode0"]["per_original"][0]["start"])
    d = R.derive(m, o, LAPS, CP)
    p0 = f["mode0"]["per_original"][0]
    assert d[5][(0, 1)][1] == round(p0["ext"] * SECT[5][0])
    for mode in range(3):
        r0 = ratio(o[0][(mode, 1)], 0, mode)
        assert f["race"][mode]["ratio"] != pytest.approx(r0)
        assert f["course"][5]["race"][mode] == pytest.approx(r0)
        assert f["course"][4]["race"][mode] == f["race"][mode]["ratio"]          # others: the mean
        w = d[5][(mode, 1)]
        assert w[0] == max(round(r0 * race(5, mode) - sum(w[1:-1])), math.ceil(2.0 * m["5"]["start"]))
    for mode in (1, 2):
        q = f["flat"][mode]["per_original"][0]
        assert f["flat"][mode]["ext"] != pytest.approx(q["ext"])
        assert set(d[5][(mode, 1)][1:]) == {round(q["ext"] * SECT[5][0])}
    assert f["course"][4]["ext"] == f["mode0"]["ext"]


def test_lengths_and_phantom():
    d = R.derive(meas(), originals(), LAPS, CP)
    for cid, tabs in d.items():
        for (mode, _), words in tabs.items():
            assert len(words) == 1 + LAPS[cid][mode] * (CP[cid] + 1)
        t = d[cid][(0, 1)]
        k = CP[cid]
        assert t[-1] == t[-1 - (k + 1)]                # the phantom equals the same pass a lap earlier


def test_difficulty_ratios_applied_per_mode():
    d = R.derive(meas(), originals(), LAPS, CP)
    f = R.factors(meas(), originals(), LAPS, CP)
    for cid in d:
        for mode in range(3):
            base = d[cid][(mode, 1)]
            for dd in (0, 2, 3):
                r = f["ratios"][(mode, dd)]["ratio"]
                assert len(d[cid][(mode, dd)]) == len(base)
                assert d[cid][(mode, dd)] == tuple(round(w * r) for w in base)
            assert d[cid][(mode, 3)][1] > d[cid][(mode, 0)][1]


def test_unequal_difficulty_lengths_rejected():
    o = originals()
    o[0][(0, 3)] = o[0][(0, 3)][:-1]
    with pytest.raises(ValueError):
        R.factors(meas(), o, LAPS, CP)


def test_word_guard():
    o = originals(ext_f=0.0)                        # zero extension: words of 0
    with pytest.raises(ValueError, match="course 3 mode 0 difficulty 0"):
        R.derive(meas(), originals(ext_f=0.0, start_f=3.0), LAPS, CP)
    big = R.derive(meas(), originals(), LAPS, CP)
    assert all(0 < w < 2 ** 32 for t in big.values() for ws in t.values() for w in ws)
    assert o


def test_accepts_int_keys_and_rejects_incomplete():
    m = {int(k): v for k, v in meas().items()}
    assert R.derive(m, originals(), LAPS, CP)[3][(0, 1)]
    m[3]["sectors"] = [600, None, 500]
    with pytest.raises(ValueError):
        R.derive(m, originals(), LAPS, CP)


# ---- through the spike's own cutting: summarise(sector_times(...)), one no-checkpoint and two-checkpoint courses

N = 100
GO = 10
CPSEG = [30, 70]


def crossings(sect, start, laps=4):
    k = len(sect) - 1
    cps = CPSEG[:k]
    ev, t = [], GO
    if k == 0:
        ev += [(t + start // 2, 50), (t + start - 1, 99), (t + start, 0)]
        t += start
    else:
        t += start
        ev.append((t, cps[0]))
        for j in range(1, k):
            t += sect[j]
            ev.append((t, cps[j]))
        t += sect[k]
        ev += [(t - sect[k] // 2, 65), (t - 1, 99), (t, 0)]
    for _ in range(1, laps):
        t += sect[0]
        if k == 0:
            ev += [(t - sect[0] // 2, 50), (t - 1, 99), (t, 0)]
            continue
        ev.append((t, cps[0]))
        for j in range(1, k):
            t += sect[j]
            ev.append((t, cps[j]))
        t += sect[k]
        ev += [(t - sect[k] // 2, 65), (t - 1, 99), (t, 0)]
    return ev, t


def rows_for(car, ev, last):
    rows, seg, todo = [], 0, list(ev)
    for f in range(GO - 1, last + 1):
        while todo and todo[0][0] <= f:
            seg = todo.pop(0)[1]
        rows.append(Progress(f, car, seg, 1))
    return rows


def measure_through_spike(i):
    sect = SECT[i]
    start = sect[0] + 100 * i
    ev, end = crossings(sect, start)
    rows = rows_for(1, ev, end + 5) + rows_for(2, [(f + 10, s) for f, s in ev], end + 15)
    cps = CPSEG[:CP[i]]
    return rt.summarise(rt.sector_times(rows, cps, N, GO, 3), cps)


@pytest.mark.skipif(rt is None, reason="spike/race_times.py is not in this tree (the public export)")
def test_derive_on_measurements_from_the_spike_cutting():
    m = {str(i): measure_through_spike(i) for i in range(8)}
    for i in (0, 5, 1, 7):                                  # lap line only; two checkpoints
        assert m[str(i)]["start"] == SECT[i][0] + 100 * i + 5, i        # median of the two cars (second +10)
        assert m[str(i)]["sectors"] == SECT[i], i
        assert m[str(i)]["lap"] == sum(SECT[i]), i
    d = R.derive(m, originals(), LAPS, CP)
    assert d[5][(0, 1)][0] > 0 and len(d[5][(0, 1)]) == 1 + LAPS[5][0]
    assert len(d[7][(0, 1)]) == 1 + LAPS[7][0] * 3
    same = meas()
    for i in same:
        same[i]["start"] += 5
    assert d == R.derive(same, originals(), LAPS, CP)


def test_reverse_ids_derived():
    """M9 stage 4: `derive(..., ids=CONVERTED + REVERSE)` gives ids 8-15 too, by the same rule; a course with the lap
    line only (Three Seven Speedway reverse, 8; Circuit Pixie reverse, 13) takes id 0's factors as Circuit Pixie does
    (LAP_LINE_SOURCE), the others the originals' means; ids 3-7 as without them."""
    cp = {**CP, 8: 0, 9: 2, 10: 1, 11: 2, 12: 2, 13: 0, 14: 1, 15: 2}
    laps = {i: {0: 3, 1: 2, 2: 2} for i in range(16)}
    m = meas()
    for i in range(8, 16):
        sect = [900] if cp[i] == 0 else [400 + 50 * j for j in range(cp[i] + 1)]
        m[str(i)] = {"start": 1200, "sectors": sect, "lap": sum(sect)}
    o = originals()
    ids = R.CONVERTED + R.REVERSE
    assert R.REVERSE == tuple(range(8, 16))
    f = R.factors(m, o, laps, cp, ids)
    assert f["course"][8] == f["course"][13] == f["course"][5]
    assert all(f["course"][i] == f["course"][3] for i in (9, 10, 11, 12, 14, 15))
    d = R.derive(m, o, laps, cp, ids)
    assert sorted(d) == list(ids)
    plain = R.derive(meas(), o, LAPS, CP)
    assert all(d[i] == plain[i] for i in R.CONVERTED)
    for i in R.REVERSE:
        assert len(d[i][(0, 1)]) == 1 + laps[i][0] * (cp[i] + 1)
