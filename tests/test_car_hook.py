"""The car choice's hooks (v0.4; course.car_choice, docs/spike/2026-11-cars.md "The draw routine")."""
import hashlib
import struct

import pytest

from dtools.arcade import course as C, i960, romset as R
from dtools.arcade.tablebase import CourseError
from dtools.convert import build
from tests.test_convert_build import _romset
from tests.test_convert_tables import disc  # noqa: F401

CAR_CODE_SHA1 = {0: "76995e6cc753af523035c86e4cd7d52268458988", 1: "5c0f7d54b93d6c7589b91213a8eb0c24e90cfc40"}
"""course.car_code(default, build.POLYGON_CAP)'s bytes: the choice byte read (0), the Grasshopper forced (1)."""


def _far_viewed(rs):
    """A custom set that carries the far view's code and branch (not its tables: car_choice reads neither)."""
    return rs.as_custom().with_patches({C._rom_word_at(C.FAR_CODE): C.far_code().bytes(),
                                        C._rom_word_at(C.FAR_SITE): i960.ctrl("b", C.FAR_CODE - C.FAR_SITE)})


def test_car_constants():
    assert C.CAR_TABLE_ADDR == build.DATA_BASE + build.CAR_TABLE and C.CAR_CHOICES == build.CAR_COUNT
    assert any((lo, n) == (C.CAR_CODE, C.CAR_SIZE) for lo, n, _ in R.RESERVED["program"])
    assert any((lo, n) == (C.CAR_CHOICE_RAM, 4) for lo, n, _ in R.RESERVED_RAM)
    assert C.CAR_CODE + C.CAR_SIZE <= C.FAR_CODE and set(C.CAR_INSTALLS) <= set(C.car_sites()) and "damage" in C.car_sites()
    assert C.CAR_SCRATCH_RAM == C.CAR_GEARS_RAM + 0x20
    assert C.CAR_HANDLING_ADDR == build.DATA_BASE + build.CAR_HANDLING and C.GEAR_RATIOS == build.DATA_BASE + build.GEAR_RATIOS_AT
    assert any((lo, n) == (build.CAR_HANDLING, 0x100) for lo, n, _ in R.RESERVED["data"])
    assert any((lo, n) == (C.CAR_GEARS_RAM, 0x24) for lo, n, _ in R.RESERVED_RAM) and C.CAR_DRIVE_RAM == C.CAR_GEARS_RAM + 16
    assert 32 * C.CAR_CHOICES <= 0x100 and C.CAR_HANDLING_WORDS * 4 == 32
    assert C.FIELD_TABLE_ADDR == build.DATA_BASE + build.FIELD_TABLE and build.FIELD_ROWS == C.SPAWN_RECORDS
    assert any((lo, n) == (build.FIELD_TABLE, 0x300) for lo, n, _ in R.RESERVED["data"]) and 16 * build.FIELD_ROWS <= 0x300


def test_car_code():
    """car_code: the hooks fit CAR_SIZE; with a forced car (1-4) no instruction reads CAR_CHOICE_RAM (the attract hook
    still clears it); the bytes are
    pinned. Refused: a default outside 0-4, a cap outside OBC..CAP_MAX."""
    for default in (0, 1):
        block, at = C.car_code(default, build.POLYGON_CAP)
        code = block.bytes()
        assert len(code) <= C.CAR_SIZE and set(at) == set(C.car_sites()) | {"cap"}
        assert all(C.CAR_CODE <= a < C.CAR_CODE + len(code) for a in at.values())
        reads = code.count(i960.mem("ldob", "r4", C.CAR_CHOICE_RAM)[4:])
        assert (reads == 1) if default else (reads >= 9)     # forced: only the attract hook's store names the byte
        assert hashlib.sha1(code).hexdigest() == CAR_CODE_SHA1[default]
    for bad in (-1, 5, True, "1", None):
        with pytest.raises(CourseError, match="default"):
            C.car_code(bad)
    for bad in (C.OBC - 1, C.CAP_MAX + 1, True):
        with pytest.raises(CourseError, match="cap"):
            C.car_code(0, bad)


def test_car_choice():
    """car_choice on a set with the far view: CAR_CODE written, every site (the far view's cap store among them) a
    branch to its hook with its second word kept, nothing else changed; refused without the far view, twice, on a
    site that does not hold its instruction, and where CAR_CODE is not free."""
    stock = _romset()
    if stock is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    for name, (addr, own) in C.car_sites().items():
        assert stock.read_i960(addr, 8) == own, name
    with pytest.raises(CourseError, match="far view"):
        C.car_choice(stock.as_custom())
    rs = _far_viewed(stock)
    assert not C.car_chosen(rs)
    out = C.car_choice(rs, 1, build.POLYGON_CAP)
    assert C.car_chosen(out)
    block, at = C.car_code(1, build.POLYGON_CAP)
    assert out.read_i960(C.CAR_CODE, len(block.bytes())) == block.bytes()
    sites = dict(C.car_sites(), cap=(C._far_cap_site(), i960.mem("st", "r3", C.POLYGON_CAP_RAM)))
    changed = {k for k, (a, b) in enumerate(zip(rs.program, out.program)) if a != b}
    expect = set(range(C.CAR_CODE, C.CAR_CODE + len(block.bytes())))
    for name, (addr, own) in sites.items():
        assert out.read_i960(addr, 4) == i960.ctrl("b", at[name] - addr) and out.read_i960(addr + 4, 4) == own[4:], name
        off = C._rom_word_at(addr)[1]               # a window site (setup2) is that much into the program image
        expect |= set(range(off, off + 4))
    for addrs, ram in ((C.GEAR_SITES, C.CAR_GEARS_RAM), (C.DRIVE_SITES, C.CAR_DRIVE_RAM)):
        for addr in addrs:                              # the gear ratios' loads read the RAM copies
            assert stock.read_i960(addr + 4, 4) == struct.pack("<I", C.GEAR_RATIOS)
            assert out.read_i960(addr, 4) == stock.read_i960(addr, 4) and out.read_i960(addr + 4, 4) == struct.pack("<I", ram)
            expect |= set(range(addr + 4, addr + 8))
    assert changed <= expect and out.data == rs.data and out.polygons == rs.polygons
    with pytest.raises(CourseError, match="already"):
        C.car_choice(out)
    with pytest.raises(CourseError, match="parts"):
        C.car_choice(rs.with_patches({C._rom_word_at(C.car_sites()["parts"][0]): b"\0" * 4}))
    with pytest.raises(CourseError, match="not free"):
        C.car_choice(rs.with_patches({C._rom_word_at(C.CAR_CODE + 8): b"\0"}))


def test_release_car_refusals(disc):  # noqa: F811
    """release_set's `car`: None or 0-4 (bool refused), and only with the far view; refused before any step runs."""
    from dtools.release import ReleaseError, release_set
    stock = _romset()
    if stock is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    for bad in (5, -1, True, "1"):
        with pytest.raises(ReleaseError, match="car"):
            release_set(stock, disc, car=bad)
    with pytest.raises(ReleaseError, match="far_view"):
        release_set(stock, disc, far_view=False, car=1)
