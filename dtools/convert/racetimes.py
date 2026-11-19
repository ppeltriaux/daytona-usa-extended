"""Race times for the converted courses (ids 3-7): the time tables derived from the AI field's measured sector times
and the arcade's own tables (docs/superpowers/specs/2026-11-15-race-times-design.md, 7.1-7.4, 7.7). Pure: plain dicts
in, plain dicts out, no ROM and no MAME; the measurement is spike/race_times.py (its JSON is the `measured` input).

Inputs (ids may be int or str keys in `measured`, as the JSON has them):
- measured[id]: the spike's summary: "start" (frames GO -> first pass), "sectors" (median frames in lap order: lap
  line -> checkpoint 0, checkpoint i-1 -> i, last checkpoint -> lap line; len checkpoints + 1), "lap" (frames).
- originals[id][(mode, difficulty)]: the time-table words of ids 0-2 (`golden.Course.time_tables[key][2]`).
- laps[id][mode], checkpoints[id]: the count of checkpoints (the lap line not included), for every id 0-7.

Pass j (0..k) of a lap is checkpoint j, the lap line being pass k; word 1 + lap * (k + 1) + j is paid at that pass; the
last word (final lap line) is a phantom the race never adds. The sector a pass "pays for" is, reading "after", the one
after the pass (sectors[(j + 1) % (k + 1)]), reading "before", the one before it (sectors[j]); `factors` records both
readings' spread on the originals and the flatter one is used.

The start (v0.1.2 Task 2 round 2, the builder's call under spec 7.2, measured): the extensions keep the originals'
per-sector slack, the start is set so the whole race is as generous as the originals' in that mode. The race ratio of a
table is (start + every word the race adds) / the field's race time (start sector + the rest of lap 1 + laps - 1 full
laps); Sega's three courses hold 0.88-0.97 in every mode from 2 laps to 80, while a start of factor x start sector
alone gave the 8-lap converted courses 0.76-0.80 (and Circuit Pixie 1.06): docs/spike/2026-11-race-times.md. The start
never drops below START_FLOOR x the start sector."""
import math
import statistics

ORIGINALS = (0, 1, 2)
CONVERTED = (3, 4, 5, 6, 7)
PIXIE = 5                                   # Circuit Pixie: lap line only, takes Three Seven Speedway's mode-0 factors
PIXIE_SOURCE = 0
REVERSE = tuple(range(8, 16))               # M9 stage 4: the reverses (id n + 8), derived by the same rule
LAP_LINE_SOURCE = {PIXIE: PIXIE_SOURCE, 8: PIXIE_SOURCE, 13: PIXIE_SOURCE}
"""The courses with the lap line only and the original whose factors they take in every mode: Circuit Pixie, and
(M9) Three Seven Speedway reverse and Circuit Pixie reverse, as Circuit Pixie: Three Seven Speedway's."""
MODES = (0, 1, 2)
DIFFICULTIES = (0, 1, 2, 3)
READINGS = ("after", "before")
START_FLOOR = 2.0                           # a start never below twice the field's GO -> first pass time


def _norm(measured):
    return {int(k): v for k, v in measured.items()}


def paid_sector(sectors, j, reading):
    """The measured sector the pass j of a lap pays for (len(sectors) = k + 1)."""
    return sectors[(j + 1) % len(sectors)] if reading == "after" else sectors[j]


def _passes(words, k):
    """[(pass index j, word)] of the non-phantom words, all laps."""
    return [(((i - 1) % (k + 1)), w) for i, w in enumerate(words) if 1 <= i < len(words) - 1]


def race_time(m, laps):
    """The field's race time in frames: GO to the first pass, the rest of lap 1, then laps - 1 full laps."""
    return m["start"] + sum(m["sectors"][1:]) + (laps - 1) * m["lap"]


def allowance(words):
    """What a table gives over a race: the start plus every word the race adds (the phantom last word left out)."""
    return words[0] + sum(words[1:-1])


def _cv(values):
    m = statistics.fmean(values)
    return statistics.pstdev(values) / m if m else float("inf")


def _check(measured, cid, k):
    m = measured[cid]
    if m.get("start") is None or m.get("lap") is None or None in m["sectors"] or len(m["sectors"]) != k + 1:
        raise ValueError(f"course {cid}: measurement incomplete for {k} checkpoints: {m.get('start')=}, "
                         f"{m.get('lap')=}, sectors {m.get('sectors')}")


def factors(measured, originals, laps, checkpoints, ids=CONVERTED):
    """The slack of the originals, everything recorded:
    {"reading": "after"|"before",
     "readings": {reading: {"cv": mean over ids 1, 2 of the coefficient of variation of word / paid sector over the
                  passes, "per_id": {id: cv}}},
     "mode0": {"start": mean start word / reference start sector, "ext": mean of per-original (sum of the non-phantom
               words / sum of the reference time paid for), "per_original": {id: {"start", "ext"}}},
     "flat": {mode (1, 2): {"start": mean start / reference lap time, "ext": as mode0's ext, "per_original": {...}}},
     "ratios": {(mode, d): {"ratio": mean over originals of the mean over words of word_d / word_1, "per_original"}},
     "race": {mode: {"ratio": mean over originals of allowance / race_time at difficulty 1, "per_original"}},
     "course": {id of `ids` (3-7 by default; REVERSE too, M9): {"start", "ext", "flat": {1, 2: {"start", "ext"}},
     "race": {mode: ratio}}} (the LAP_LINE_SOURCE courses = id 0's in every mode)}
    Flat modes (1, 2): start word / lap time, and one identical word on every pass over the mean sector; ratios cover
    the words the race adds (the phantom last word excluded)"""
    measured = _norm(measured)
    for i in ORIGINALS:
        _check(measured, i, checkpoints[i])

    def spread(reading):
        per = {}
        for i in (1, 2):
            k = checkpoints[i]
            vals = [w / paid_sector(measured[i]["sectors"], j, reading) for j, w in _passes(originals[i][(0, 1)], k)]
            per[i] = _cv(vals)
        return {"cv": statistics.fmean(per.values()), "per_id": per}
    readings = {r: spread(r) for r in READINGS}
    reading = min(READINGS, key=lambda r: (readings[r]["cv"], READINGS.index(r)))   # a tie keeps "after"

    def ext_ratio(i, mode):
        k = checkpoints[i]
        ps = _passes(originals[i][(mode, 1)], k)
        return sum(w for _, w in ps) / sum(paid_sector(measured[i]["sectors"], j, reading) for j, _ in ps)

    mode0 = {i: {"start": originals[i][(0, 1)][0] / measured[i]["start"], "ext": ext_ratio(i, 0)} for i in ORIGINALS}
    out = {"reading": reading, "readings": readings,
           "mode0": {"start": statistics.fmean(v["start"] for v in mode0.values()),
                     "ext": statistics.fmean(v["ext"] for v in mode0.values()), "per_original": mode0}}
    def flat_ext(i, mode):
        # flat tables: one word on every pass; ratio of totals = mean non-phantom word / mean reference sector
        ws = originals[i][(mode, 1)][1:-1]
        return statistics.fmean(ws) / statistics.fmean(measured[i]["sectors"])

    flat = {}
    for m in (1, 2):
        per = {i: {"start": originals[i][(m, 1)][0] / measured[i]["lap"], "ext": flat_ext(i, m)} for i in ORIGINALS}
        flat[m] = {"start": statistics.fmean(v["start"] for v in per.values()),
                   "ext": statistics.fmean(v["ext"] for v in per.values()), "per_original": per}
    out["flat"] = flat
    ratios = {}
    for m in MODES:
        for d in DIFFICULTIES:
            per = {}
            for i in ORIGINALS:
                base, tab = originals[i][(m, 1)], originals[i][(m, d)]
                if len(tab) != len(base):
                    raise ValueError(f"original {i} mode {m}: difficulty {d} table has {len(tab)} words, "
                                     f"difficulty 1 has {len(base)}")
                per[i] = statistics.fmean(w / b for w, b in zip(tab[:-1], base[:-1]) if b)   # the phantom is out
            ratios[(m, d)] = {"ratio": 1.0 if d == 1 else statistics.fmean(per.values()), "per_original": per}
    out["ratios"] = ratios
    race = {}
    for m in MODES:
        per = {i: allowance(originals[i][(m, 1)]) / race_time(measured[i], laps[i][m]) for i in ORIGINALS}
        race[m] = {"ratio": statistics.fmean(per.values()), "per_original": per}
    out["race"] = race
    def course(c):
        if c in LAP_LINE_SOURCE:            # the lap line only: its source original's factors in every mode
            src = LAP_LINE_SOURCE[c]
            return {"start": mode0[src]["start"], "ext": mode0[src]["ext"],
                    "flat": {m: dict(flat[m]["per_original"][src]) for m in (1, 2)},
                    "race": {m: race[m]["per_original"][src] for m in MODES}}
        return {"start": out["mode0"]["start"], "ext": out["mode0"]["ext"],
                "flat": {m: {"start": flat[m]["start"], "ext": flat[m]["ext"]} for m in (1, 2)},
                "race": {m: race[m]["ratio"] for m in MODES}}
    out["course"] = {c: course(c) for c in ids}
    return out


def derive(measured, originals, laps, checkpoints, ids=CONVERTED):
    """{id of `ids` (3-7 by default; REVERSE too, M9 stage 4): {(mode, difficulty): words}} for modes 0-2, difficulties 0-3, each table 1 + laps * (k + 1) words.
    Difficulty 1: mode 0's pass word = round(extension factor * the reference sector it pays for), the same every lap;
    modes 1, 2: every pass word = round(flat extension factor * the mean reference sector). The start, every mode:
    round(the course's race ratio * race_time - the words the race adds), at least ceil(START_FLOOR * reference start
    sector). Difficulty d: round(difficulty-1 word * the originals' mean d/1 ratio of that mode). The phantom last
    word is the one that pass would have."""
    measured = _norm(measured)
    f = factors(measured, originals, laps, checkpoints, ids)
    out = {}
    for cid in ids:
        k = checkpoints[cid]
        _check(measured, cid, k)
        m = measured[cid]
        tables = {}
        for mode in MODES:
            if mode == 0:
                ef = f["course"][cid]["ext"]
                ext = [round(ef * paid_sector(m["sectors"], j, f["reading"])) for j in range(k + 1)]
            else:                           # flat: start plus one identical word on every pass
                ef = f["course"][cid]["flat"][mode]["ext"]
                ext = [round(ef * statistics.fmean(m["sectors"]))] * (k + 1)
            passes = tuple(ext[j] for _ in range(laps[cid][mode]) for j in range(k + 1))
            start = round(f["course"][cid]["race"][mode] * race_time(m, laps[cid][mode]) - sum(passes[:-1]))
            base = (max(start, math.ceil(START_FLOOR * m["start"])),) + passes
            for d in DIFFICULTIES:
                r = f["ratios"][(mode, d)]["ratio"]
                words = base if d == 1 else tuple(round(w * r) for w in base)
                for idx, w in enumerate(words):
                    if not isinstance(w, int) or not 0 < w < 2 ** 32:
                        raise ValueError(f"course {cid} mode {mode} difficulty {d}: word {idx} is {w!r}, "
                                         f"not an int in 1..2**32-1")
                tables[(mode, d)] = words
        out[cid] = tables
    return out
