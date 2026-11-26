"""The Dreamcast's car sets, `CAR{c}{level}_POL.BIN` / `_TEX.BIN` (v0.4): which files, which object of a set is
which part, and where the wheels sit.

A set is a course model file (`dtools.dc.pol` decodes it as it is; its textures add the layout 0x0d,
`tex.LAYOUTS`). The five stock cars are `CARS`, in the executable's order (its file table at `0x154908`). Level 0 is
the full-detail set (80 objects), level 2 a lighter one (54 objects); what level 1 is for is not known. The roles of
a level-0 set's objects were read from a rendered sheet of each (`spike/v04/car_sheet.py`) and hold by index on the
five cars (docs/formats/dreamcast-disc.md, "Car sets"). The wheels are modelled round the origin; their hubs are in
`1ST_READ.BIN` (`HUBS_AT`, twelve floats a car)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import NamedTuple

from dtools.dc import pol
from dtools.dc.errors import DcError

CARS = ("H", "G", "F", "L", "U")
NAMES = ("Hornet", "Grasshopper", "Falcon", "Lightning", "Unicorn")
LEVELS = 3
SET_OBJECTS = 80
SET_KINDS = (5,) * 9 + (1,) * 20 + (3,) * 3 + (1,) * 36 + (5,) * 6 + (1,) * 6
"""The Hornet's object kinds (`pol.Object.kind`). `car_parts` checks the ones every car shares: `CHECKED`."""
CHECKED = tuple(range(9)) + (29, 30, 31)
HUBS_AT = 0x124AC0
"""`1ST_READ.BIN`: per car (the ten of the file table) four hubs of x, y, z f32: front left, front right, rear left,
rear right (x positive on the left as stored; y up; z forward)."""
HUB_BYTES = 48
GEARS_AT = 0x11A510
"""`1ST_READ.BIN`: per car four gear ratios, f32 (the Hornet's are the arcade's own, data ROM `0x86f780`); read by
the engine routine at `0x192d8` as torque curve x ratio x TORQUE's entry."""
TORQUE_AT = 0x1337F4
"""Per car an engine torque factor, f32 (the Hornet 1.0)."""
GRIP_AT = (0x1337A4, 0x1337CC)
"""Per car a factor on two grip limits (car +0x1a4 and +0x1a8 in `0x184e8`: read as front and rear), f32, the
Hornet 1.0; a per-setting table follows each (five values a car, the middle one 0)."""
CAR_ROWS = 10


class Handling(NamedTuple):
    gears: tuple[float, float, float, float]
    torque: float
    grip: tuple[float, float]


def handling(exe: bytes, car: int) -> Handling:
    """Car 0-4's driving values from `1ST_READ.BIN`. DcError for a car outside 0-4, an executable too short, a
    Hornet whose factors are not 1.0, or values that are not gear ratios falling from first to fourth."""
    _car(car, "handling")
    if not isinstance(exe, (bytes, bytearray)) or len(exe) < max(GEARS_AT + 16 * CAR_ROWS, TORQUE_AT + 4 * CAR_ROWS):
        raise DcError("handling: the executable is too short for the cars' tables")
    if struct.unpack_from("<3f", exe, TORQUE_AT)[0] != 1.0 or any(struct.unpack_from("<f", exe, a)[0] != 1.0 for a in GRIP_AT):
        raise DcError("handling: the Hornet's factors are not 1.0: not the cars' tables")
    gears = struct.unpack_from("<4f", exe, GEARS_AT + 16 * car)
    if not all(20.0 > a > b > 1.0 for a, b in zip(gears, gears[1:])):
        raise DcError(f"handling: car {car}'s row {tuple(round(g, 3) for g in gears)} is not four gear ratios")
    return Handling(gears, struct.unpack_from("<f", exe, TORQUE_AT + 4 * car)[0],
                    tuple(struct.unpack_from("<f", exe, a + 4 * car)[0] for a in GRIP_AT))


@dataclass(frozen=True)
class CarParts:
    """Object indices of a level-0 set. `body` is the undamaged shell and `damaged` its eight dented versions;
    `lower` the floor and bumpers under it (objects 10-16 its damaged versions, not listed), `inner` the inside of
    the shell, `trim` the decal layer over the body (19-26 the damaged versions'), `glass` the windscreen and the
    cabin with its cage, `hulls` three untextured boxes, `numbers` twenty roof-and-door panels (two sets of ten
    colours), `loose` the bumpers and skirts as separate pieces, `wheels_left` / `wheels_right` five wheels each (the
    first the still one, the others more blurred), `cockpit` what the driver's seat view draws (the windscreen frame,
    the undamaged bonnet, the dashboard, the cabin) and `bonnets` the bonnet's five versions."""
    body: int
    damaged: tuple[int, ...]
    lower: int
    inner: int
    trim: int
    glass: tuple[int, ...]
    hulls: tuple[int, ...]
    numbers: tuple[int, ...]
    loose: tuple[int, ...]
    wheels_left: tuple[int, ...]
    wheels_right: tuple[int, ...]
    cockpit: tuple[int, ...]
    bonnets: tuple[int, ...]


_PARTS = CarParts(body=0, damaged=tuple(range(1, 9)), lower=9, inner=17, trim=18, glass=(27, 28), hulls=(29, 30, 31),
                  numbers=tuple(range(32, 52)), loose=tuple(range(52, 58)), wheels_left=tuple(range(58, 63)),
                  wheels_right=tuple(range(63, 68)), cockpit=(68, 69, 74, 78), bonnets=tuple(range(69, 74)))


def _car(car, what: str) -> int:
    if not isinstance(car, int) or isinstance(car, bool) or not 0 <= car < len(CARS):
        raise DcError(f"{what}: car {car!r} is not 0..{len(CARS) - 1} ({', '.join(NAMES)})")
    return car


def car_files(car: int, level: int) -> tuple[str, str]:
    """The model and texture file of car 0-4, level 0-2. DcError for anything else."""
    _car(car, "car_files")
    if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level < LEVELS:
        raise DcError(f"car_files: car level {level!r} is not 0..{LEVELS - 1}")
    stem = f"CAR{CARS[car]}{level}"
    return f"{stem}_POL.BIN", f"{stem}_TEX.BIN"


def car_parts(m: pol.Model, name: str = "car set") -> CarParts:
    """The parts of a level-0 set. DcError naming `name` when `m` is not a Model of SET_OBJECTS objects whose
    bodies and hulls have the kinds every car's have."""
    if not isinstance(m, pol.Model):
        raise DcError(f"{name}: {type(m).__name__} is not a Model")
    if len(m.objects) != SET_OBJECTS:
        raise DcError(f"{name}: {len(m.objects)} objects, a level-0 car set has {SET_OBJECTS}")
    for i in CHECKED:
        if m.objects[i].kind != SET_KINDS[i]:
            raise DcError(f"{name}: object {i} has kind {m.objects[i].kind}, a car set's has {SET_KINDS[i]}")
    return _PARTS


def wheel_positions(exe: bytes, car: int) -> tuple[tuple[float, float, float], ...]:
    """The four hubs of car 0-4 from `1ST_READ.BIN`: front left, front right, rear left, rear right. DcError for a
    car outside 0-4, an executable too short, or a row that is not four hubs (left and right mirrored, the fronts
    ahead of the rears)."""
    _car(car, "wheel_positions")
    at = HUBS_AT + HUB_BYTES * car
    if not isinstance(exe, (bytes, bytearray)) or len(exe) < at + HUB_BYTES:
        raise DcError(f"wheel_positions: the executable is too short for car {car}'s hubs at {at:#x}")
    v = struct.unpack_from("<12f", exe, at)
    fl, fr, rl, rr = (v[0:3], v[3:6], v[6:9], v[9:12])
    if not (fl[0] == -fr[0] > 0 and rl[0] == -rr[0] > 0 and fl[2] > rl[2] and fl[1] == fr[1] and rl[1] == rr[1]):
        raise DcError(f"wheel_positions: the row at {at:#x} is not four hubs: {tuple(round(x, 3) for x in v)}")
    return fl, fr, rl, rr


# ---- the opponents' sets, `EC{k}_POL.BIN` / `_TEX.BIN` (v0.4) ----

ENEMY_FILES = 4
ENEMY_OBJECTS = 44
ENEMY_LIVERIES = 5
ENEMY_CARS = 2 * ENEMY_LIVERIES
"""An opponents' set holds one body shape in ENEMY_LIVERIES liveries, each with two number panels: ENEMY_CARS cars a
file, forty on the disc. Read from each object's triangles and textures (`spike/v04/dc_car.py EC0`); how the
Dreamcast spreads them over its field was not traced."""


class EnemyParts(NamedTuple):
    """Object indices of one opponent car: `near` its body, the floor, the cabin and its number panel; `mid` its
    lighter body and that body's panel; `far` its lightest body."""
    near: tuple[int, ...]
    mid: tuple[int, ...]
    far: tuple[int, ...]


def enemy_files(k: int) -> tuple[str, str]:
    """The model and texture file of opponents' set 0-3. DcError for anything else."""
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < ENEMY_FILES:
        raise DcError(f"enemy_files: set {k!r} is not 0..{ENEMY_FILES - 1}")
    return f"EC{k}_POL.BIN", f"EC{k}_TEX.BIN"


def enemy_parts(m: pol.Model, car: int, name: str = "opponents' set") -> EnemyParts:
    """Car 0-9 of an opponents' set: livery `car % 5`, number panel `car // 5`. Livery 0's objects are 0 (body), 6
    (lighter), 8 (lightest), its panels 4 and 7; livery i's start at 9 + 6 (i - 1): body, a dented body, panel,
    lighter body, its panel, lightest body. Objects 2 and 5 are the floor and the cabin every livery shares; 33-42 the
    second number panels (near, mid a livery); 43 a hull. DcError naming `name` for a model that is not such a set, or
    a car outside 0-9."""
    if not isinstance(m, pol.Model):
        raise DcError(f"{name}: {type(m).__name__} is not a Model")
    if len(m.objects) != ENEMY_OBJECTS or m.objects[ENEMY_OBJECTS - 1].kind != 3:
        raise DcError(f"{name}: {len(m.objects)} objects (the last a hull, kind 3): an opponents' set has "
                      f"{ENEMY_OBJECTS}")
    if not isinstance(car, int) or isinstance(car, bool) or not 0 <= car < ENEMY_CARS:
        raise DcError(f"{name}: car {car!r} is not 0..{ENEMY_CARS - 1}")
    i, j = car % ENEMY_LIVERIES, car // ENEMY_LIVERIES
    if i == 0:
        body, mid, far, panels = 0, 6, 8, (4, 7)
    else:
        b = 9 + 6 * (i - 1)
        body, mid, far, panels = b, b + 3, b + 5, (b + 2, b + 4)
    if j:
        panels = (33 + 2 * i, 34 + 2 * i)
    return EnemyParts((body, 2, 5, panels[0]), (mid, panels[1]), (far,))

