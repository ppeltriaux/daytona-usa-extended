"""v0.4 in MAME: the Car Select screen, the car raced and its handling, on the release set with the cars
(release_set(car=0)'s bytes: the release set, then build.car_set, course.car_choice and carscreen.patch_cars)."""
import pytest

from dtools.arcade import carscreen as K
from dtools.arcade import course as C
from dtools.arcade import geo
from dtools.arcade import select as S
from dtools.convert import build
from dtools.release import RELEASE_CARS
from tests.test_convert_tables import disc  # noqa: F401
from tests.test_release_mame import _release

_SETS: dict = {}
PADDLE = {0: 128, 1: 160, 2: 172, 3: 190, 4: 250}
"""The wheel's paddle value held on the screen for each car (w 0, 46, 66, 97, 127: inside the bands of K.BANDS)."""
TOP_SPEED = {0: 325, 1: 305, 2: 316, 3: 335, 4: 349}
"""Top speed, km/h, on Three Seven Speedway, automatic, the autopilot's 75 s (measured 26 Nov 2026; the Hornet's is
v0.3's)."""
HORNET_OBJECTS = {0xB456D1, 0x8D7B13, 0x8D7C9A, 0xB3CBC9, 0xB3CCBA}
"""The arcade Hornet's body, its two wheels and its two part objects (docs/spike/2026-11-cars.md)."""
SECONDS = 75


def _cars(romset_zip, disc):  # noqa: F811
    key = str(romset_zip)
    if key not in _SETS:
        rel = _release(romset_zip, disc)
        _SETS[key] = (rel, K.patch_cars(C.car_choice(build.car_set(rel, disc, RELEASE_CARS, field=True), default=0,
                                                     cap=build.POLYGON_CAP), build.POLYGON_CAP))
    return _SETS[key]


def _race(rs, tmp_path, romset_zip, car, logs=("car",), seconds=SECONDS, extra=None):
    from dtools.arcade import golden as G
    from dtools.mame import build as mame_build
    from dtools.mame.road import road_file
    from tests.test_custom_mame import _custom
    env = {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": "0", "DAYTONA_SNAPS": "0", "DAYTONA_AUTOPILOT_LAPS": "99",
           "DAYTONA_DIRECTION": "normal", "DAYTONA_CAR": "step",
           "DAYTONA_WHEEL_SCRIPT": f"1:128,985:{PADDLE[car]},1045:128",
           "DAYTONA_AUTOPILOT": str(road_file(G.decode_course(rs, 0), tmp_path / "road.txt")), **(extra or {})}
    return mame_build.run_set(rs, tmp_path, seconds, romset_zip, env, logs=logs, mame=_custom())


def _arcade_field(rs) -> set:
    """The objects of the arcade's own opponents: every spawn record's detail lists but the player's."""
    e = C.course_table(rs)[0]
    return {p.oba for table in (0, 1) for n, s in enumerate(C.start_grid(rs, e, table)) if n
            for lst in s.objects[2:6] for p in C.object_list(rs, lst, "car list")}


def _rows(path):
    return [tuple(int(v) for v in ln.split()[1:]) for ln in open(path) if ln.startswith("K ")]


@pytest.mark.mame
@pytest.mark.parametrize("car", range(5))
def test_car_select_races(tmp_path, romset_zip, disc, car):  # noqa: F811
    """With the paddle held for car `car` on the Car Select screen: the select runs Course Direction (17-19), then
    Car Select (20, 21, 22), then Mission Select (9 on); the choice byte is `car` when the screen ends and through the
    race; the race's player is that car (a race frame's list holds its body and wheel objects and, for a Dreamcast
    car, none of the Hornet's; for the Hornet, the Hornet's); the field is the Dreamcast's opponents with a Dreamcast car and
    the arcade's own with the Hornet; its top speed is TOP_SPEED's within 2 km/h."""
    _, rs = _cars(romset_zip, disc)
    out = _race(rs, tmp_path, romset_zip, car, logs=("car", "geo"), extra={"DAYTONA_GEO_FROM": "2400",
                                                                             "DAYTONA_GEO_EVERY": "100000"})
    rows = _rows(out["car"])
    states = [r[1] for r in rows]
    order = [s for k, s in enumerate(states) if k == 0 or states[k - 1] != s]
    first = order.index(17)
    assert order[first:first + 7] == [17, 18, 19, 20, 21, 22, 9], order[:20]   # Mission Select's set-up (8) runs in
    #                                                                            the hand-over's frame, as after 19
    end = max(k for k, s in enumerate(states) if s == K.STATE_HOLD)
    assert rows[end][2] == car and {r[2] for r in rows[end:]} == {car}
    race = [r for r in rows if r[0] >= 1900]
    assert len(race) > 2000 and abs(max(r[4] for r in race) - TOP_SPEED[car]) <= 2, max(r[4] for r in race)
    (_, read_start, _, words), = list(geo.parse_log(out["geo"]))[:1]
    sent = {c.args["oba"] for c in geo.walk(words, read_start // 4) if c.kind in (0x01, 0x11)}
    field = {p.oba for n in range(1, build.FIELD_ROWS) for lst in build.field_lists(rs, n)
             for p in C.object_list(rs, lst, "field list")}
    if car == 0:
        assert {0xB456D1, 0x8D7B13, 0x8D7C9A} <= sent and not sent & field       # the arcade's own field
    else:
        assert len(sent & field) >= 3 and not sent & _arcade_field(rs)           # the Dreamcast's opponents only
        e = build.car_entry(rs, car)
        own = {p.oba for lst in (e.body, e.wheels[0], e.wheels[1]) for p in C.object_list(rs, lst, "car list")}
        assert own <= sent and not sent & HORNET_OBJECTS


@pytest.mark.mame
@pytest.mark.parametrize("entry", (0, 3))
def test_hornet_lists_as_release(tmp_path, romset_zip, disc, entry):  # noqa: F811
    """The car choice's hooks with choice 0 do what their sites did: the release set with the cars filed and the
    hooks (no screen, so the race's frames are the release set's) sends the release set's geometry lists, word for
    word, on Three Seven Speedway and Desert City (17 race frames each)."""
    from dtools.arcade import golden as G
    from dtools.mame import build as mame_build
    from dtools.mame.road import road_file
    from tests.test_custom_mame import _custom
    rel, _ = _cars(romset_zip, disc)
    hooked = C.car_choice(build.car_set(rel, disc, RELEASE_CARS, field=True), default=0, cap=build.POLYGON_CAP)
    lists = []
    for name, rs in (("release", rel), ("hooked", hooked)):
        w = tmp_path / name
        w.mkdir()
        env = {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": str(entry), "DAYTONA_SNAPS": "0", "DAYTONA_AUTOPILOT_LAPS": "99",
               "DAYTONA_DIRECTION": "normal", "DAYTONA_GEO_FROM": "1500", "DAYTONA_GEO_EVERY": "150",
               "DAYTONA_AUTOPILOT": str(road_file(G.decode_course(rs, S.ENTRIES[entry].course), w / "road.txt"))}
        p = mame_build.run_set(rs, w, 70, romset_zip, env, logs=("geo",), mame=_custom())
        lists.append([(f, [(c.kind, sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in c.args.items()))
                           for c in geo.walk(words, rs_ // 4)]) for f, rs_, _, words in geo.parse_log(p["geo"])])
    assert len(lists[0]) >= 15 and lists[0] == lists[1]


@pytest.mark.mame
def test_race_after_car_screen_as_release(tmp_path, romset_zip, disc):  # noqa: F811
    """The screen moves the race and changes nothing else: with the wheel idle (the Hornet), the set with the cars
    raced through the screen (DAYTONA_CAR=step) sends, on the same frames, the geometry lists of the release set
    raced with its Mission Select step as late (DAYTONA_CAR=late), over 60 race frames of Three Seven Speedway (as
    tests/test_dirscreen_mame.py test_race_after_screen_as_stock for Course Direction: the field depends on the
    race's start frame, so only an exact shift compares)."""
    rel, cars = _cars(romset_zip, disc)
    lists = {}
    for name, rs, mode in (("release", rel, "late"), ("cars", cars, "step")):
        w = tmp_path / name
        w.mkdir()
        out = _race(rs, w, romset_zip, 0, logs=("geo",), seconds=42,
                    extra={"DAYTONA_GEO_FROM": "2250", "DAYTONA_GEO_EVERY": "1", "DAYTONA_WHEEL_SCRIPT": "1:128",
                           "DAYTONA_CAR": mode})
        lists[name] = {f: [(c.kind, sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in c.args.items()))
                           for c in geo.walk(words, rs_ // 4)] for f, rs_, _, words in geo.parse_log(out["geo"])}
    a, b = lists["release"], lists["cars"]
    common = sorted(set(a) & set(b))
    assert len(common) >= 60
    diff = [f for f in common if a[f] != b[f]]
    assert not diff, (len(diff), diff[:3])


@pytest.mark.mame
def test_car_survives_a_crash(tmp_path, romset_zip, disc):  # noqa: F811
    """A Dreamcast car stays itself after a hit (27 Nov 2026: the first captures showed the Hornet from the first
    crash on: the damage routine stores a dented Hornet body, and two routines re-install the player's lists).
    Desert City with the Falcon, the autopilot (it hits the wall before frame 3200): at frames 3200 and 4000 the
    list holds the Falcon's body and no body of the Hornet's (whole or dented: the library's 314-polygon entries)."""
    from dtools.arcade import golden as G
    from dtools.arcade.polyrom import decode_object, polygons
    from dtools.mame import build as mame_build
    from dtools.mame.road import road_file
    from tests.test_custom_mame import _custom
    _, rs = _cars(romset_zip, disc)
    env = {"DAYTONA_ROW": "8", "DAYTONA_ENTRY": "3", "DAYTONA_SNAPS": "0", "DAYTONA_AUTOPILOT_LAPS": "99",
           "DAYTONA_DIRECTION": "normal", "DAYTONA_CAR": "step", "DAYTONA_CAR_WHEEL": "172",
           "DAYTONA_GEO_FROM": "3200", "DAYTONA_GEO_EVERY": "800",
           "DAYTONA_AUTOPILOT": str(road_file(G.decode_course(rs, 3), tmp_path / "road.txt"))}
    out = mame_build.run_set(rs, tmp_path, 72, romset_zip, env, logs=("geo",), mame=_custom())
    body = {p.oba for p in C.object_list(rs, build.car_entry(rs, 2).body, "car list")}
    frames = list(geo.parse_log(out["geo"]))
    assert [f[0] for f in frames] == [3200, 4000]
    for f, read_start, _, words in frames:
        sent = [c.args["oba"] for c in geo.walk(words, read_start // 4) if c.kind in (0x01, 0x11)]
        assert body <= set(sent), f
        for oba in set(sent) - body:
            if 0xB40000 <= oba < 0xB60000:              # the Hornet's bodies' part of the polygon ROM
                n = sum(1 for q in polygons(decode_object(rs.polygons, oba & rs.poly_mask(), 0xFFFFF, 0)) if q.drawn)
                assert n < 300, (f, hex(oba), n)
