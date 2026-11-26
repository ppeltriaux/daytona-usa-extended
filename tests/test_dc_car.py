"""The Dreamcast's car sets (v0.4): file names, the parts of a level-0 set, the wheel hubs (dtools/dc/car.py;
docs/formats/dreamcast-disc.md, "Car sets")."""
import struct

import pytest

from dtools.dc import car, pol
from dtools.dc.errors import DcError
from tests.test_convert_tables import disc  # noqa: F401


def _object(kind=1):
    return pol.Object(kind, (0.0, 0.0, 0.0, 1.0), (), 0)


def _set(n=car.SET_OBJECTS, kinds=None):
    kinds = kinds or car.SET_KINDS
    return pol.Model((0,) * pol.HEADER_WORDS, tuple(_object(kinds[i] if i < len(kinds) else 1) for i in range(n)), ())


def test_car_files():
    assert car.CARS == ("H", "G", "F", "L", "U") and car.NAMES[1] == "Grasshopper"
    assert car.car_files(1, 0) == ("CARG0_POL.BIN", "CARG0_TEX.BIN")
    assert car.car_files(4, 2) == ("CARU2_POL.BIN", "CARU2_TEX.BIN")
    for bad in ((5, 0), (-1, 0), (1, 3), (True, 0), ("G", 0), (1, None)):
        with pytest.raises(DcError, match="car"):
            car.car_files(*bad)


def test_car_parts_on_synthetic_set():
    p = car.car_parts(_set())
    assert p.body == 0 and p.damaged == tuple(range(1, 9))
    assert p.lower == 9 and p.inner == 17 and p.trim == 18 and p.glass == (27, 28)
    assert p.hulls == (29, 30, 31) and p.numbers == tuple(range(32, 52)) and p.loose == tuple(range(52, 58))
    assert p.wheels_left == tuple(range(58, 63)) and p.wheels_right == tuple(range(63, 68))
    assert p.cockpit == (68, 69, 74, 78) and p.bonnets == tuple(range(69, 74))
    every = [p.body, *p.damaged, p.lower, p.inner, p.trim, *p.glass, *p.hulls, *p.numbers, *p.loose, *p.wheels_left,
             *p.wheels_right]
    assert len(set(every)) == len(every)
    with pytest.raises(DcError, match="79 objects"):
        car.car_parts(_set(79), "CARX0_POL.BIN")
    with pytest.raises(DcError, match="CARX0_POL.BIN.*object 29"):
        car.car_parts(_set(kinds=car.SET_KINDS[:29] + (1,) + car.SET_KINDS[30:]), "CARX0_POL.BIN")
    with pytest.raises(DcError, match="Model"):
        car.car_parts("CARG0")


def test_wheel_positions_synthetic():
    exe = bytearray(car.HUBS_AT + 10 * 48)
    hubs = [(0.75, 0.343, 1.333), (-0.75, 0.343, 1.333), (0.75, 0.343, -1.34), (-0.75, 0.343, -1.34)]
    struct.pack_into("<12f", exe, car.HUBS_AT + 48, *(v for h in hubs for v in h))
    got = car.wheel_positions(bytes(exe), 1)
    assert [tuple(round(v, 3) for v in h) for h in got] == hubs
    with pytest.raises(DcError, match="car"):
        car.wheel_positions(bytes(exe), 5)
    with pytest.raises(DcError, match="short"):
        car.wheel_positions(bytes(exe[:car.HUBS_AT + 50]), 1)
    with pytest.raises(DcError, match="hub"):
        car.wheel_positions(bytes(exe), 2)                    # all zero: not four hubs


def test_handling_on_disc(disc):  # noqa: F811
    """dc.car.handling: each car's four gear ratios, its engine torque factor and its two grip factors; the Hornet's
    ratios are the arcade's own (data ROM 0x86f780) and its factors 1.0."""
    exe = disc.read("1ST_READ.BIN")
    h = car.handling(exe, 0)
    assert [round(g, 3) for g in h.gears] == [9.216, 5.633, 3.9, 3.071] and h.torque == 1.0 and h.grip == (1.0, 1.0)
    got = {c: car.handling(exe, c) for c in range(1, 5)}
    assert [round(got[c].gears[0], 3) for c in range(1, 5)] == [9.962, 9.501, 8.922, 8.457]
    assert [round(got[c].torque, 3) for c in range(1, 5)] == [0.9, 0.95, 1.015, 1.26]
    assert [tuple(round(v, 3) for v in got[c].grip) for c in range(1, 5)] == [(1.005, 1.011), (1.002, 1.006),
                                                                              (1.015, 1.001), (1.023, 1.006)]
    for bad in (5, -1, True):
        with pytest.raises(DcError, match="car"):
            car.handling(exe, bad)
    with pytest.raises(DcError, match="short"):
        car.handling(exe[:car.GEARS_AT + 8], 1)
    with pytest.raises(DcError, match="1.0"):
        car.handling(bytes(len(exe)), 1)


def test_car_parts_on_disc(disc):  # noqa: F811
    from dataclasses import replace
    for c in range(5):
        name = car.car_files(c, 0)[0]
        m = pol.decode_pol(disc.read(name), name)
        p = car.car_parts(m, name)
        tris = lambda i: sum(1 for _ in pol.triangles(replace(m, objects=(m.objects[i],))))  # noqa: E731
        assert 500 <= tris(p.body) <= 700, name
        assert all(tris(w) == 160 for w in p.wheels_left + p.wheels_right), name


def test_wheel_positions_on_disc(disc):  # noqa: F811
    exe = disc.read("1ST_READ.BIN")
    assert [round(v, 3) for v in car.wheel_positions(exe, 0)[0]] == [0.81, 0.343, 1.41]
    for c in range(5):
        fl, fr, rl, rr = car.wheel_positions(exe, c)
        assert fl[0] == -fr[0] > 0.6 and rl[0] == -rr[0] > 0.6 and fl[2] > 1.2 and rl[2] < -1.2


def test_enemy_parts(disc):  # noqa: F811
    """The opponents' sets: four files of 44 objects, ten cars each (five liveries, two number panels); a car's near
    level is its body, the shared floor and cabin and its panel, its mid level the lighter body and that panel, its
    far level the lightest body; the two numbers of a livery differ in the panels only."""
    from dataclasses import replace
    assert car.enemy_files(3) == ("EC3_POL.BIN", "EC3_TEX.BIN") and car.ENEMY_FILES * car.ENEMY_CARS == 40
    for k in range(car.ENEMY_FILES):
        name = car.enemy_files(k)[0]
        m = pol.decode_pol(disc.read(name), name)
        tris = lambda i: sum(1 for _ in pol.triangles(replace(m, objects=(m.objects[i],))))  # noqa: E731
        for c in range(car.ENEMY_CARS):
            p = car.enemy_parts(m, c, name)
            assert p.near[1:3] == (2, 5) and 500 < tris(p.near[0]) > tris(p.mid[0]) > tris(p.far[0]) > 80, (name, c)
            assert 20 < tris(p.near[3]) < 80 and 20 < tris(p.mid[1]) < 80
        a, b = car.enemy_parts(m, 2, name), car.enemy_parts(m, 7, name)
        assert (a.near[0], a.mid[0], a.far) == (b.near[0], b.mid[0], b.far) and a.near[3] != b.near[3]
        with pytest.raises(DcError, match="car"):
            car.enemy_parts(m, 10, name)
    for bad in (4, -1, True):
        with pytest.raises(DcError, match="set"):
            car.enemy_files(bad)
    with pytest.raises(DcError, match="44"):
        car.enemy_parts(_set(), 0)
