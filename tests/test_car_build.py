"""The Dreamcast's cars filed in the set (v0.4; build.car_set, docs/spike/2026-11-cars.md)."""
import struct

import numpy as np
import pytest

from dtools.arcade import course as C, polyrom as P, romset as R, texture as T
from dtools.convert import ConvertError, build, car as ccar, model
from tests.test_convert_build import _romset
from tests.test_convert_tables import disc  # noqa: F401


def test_car_table_reserved():
    """The car table is a reserved span at the custom data area's end, clear of every free run."""
    assert (build.CAR_TABLE, build.CAR_TABLE_SIZE) == (0x1FFFE00, 0x200)
    assert any(lo == build.CAR_TABLE and n == build.CAR_TABLE_SIZE for lo, n, _ in R.RESERVED["data"])
    assert all(not (lo < build.CAR_TABLE + build.CAR_TABLE_SIZE and build.CAR_TABLE < lo + n)
               for lo, n in R.CUSTOM_FREE["data"])
    assert build.CAR_WHEELS + 16 * 5 <= build.CAR_HUBS and build.CAR_HUBS + 48 * 5 <= build.CAR_TABLE_SIZE


def test_car_set(disc):  # noqa: F811
    """build.car_set files car 1: its texels in the textures region's second custom area, its objects in the custom
    polygon area, per piece its headers and points, four lists in the list home, and its row of the car table (body
    list, cockpit list, the wheel-list table, the hubs). Read back: each list names objects that decode to the
    converted ones; each drawn record's header is textured, its words 2-3 the address of its baked texture's words;
    nothing outside the custom areas, the list home and the table changes. Car 0 has no row; a car twice is refused."""
    stock = _romset()
    if stock is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    rs = C.patch_library_bound(stock.as_custom())
    out = build.car_set(rs, disc, (1,))
    conv = ccar.car_from_disc(disc, 1)
    e = build.car_entry(out, 1)
    assert e.wheels[0] == e.wheels[2] and e.wheels[1] == e.wheels[3] and e.wheels[0] != e.wheels[1]
    assert [tuple(round(v, 4) for v in h) for h in e.hubs] == [tuple(round(v, 4) for v in h) for h in conv.hubs]
    for lst, pieces in ((e.body, conv.body), (e.cockpit, conv.cockpit), (e.wheels[0], conv.wheel_left),
                        (e.wheels[1], conv.wheel_right)):
        pl = C.object_list(out, lst, "car list")
        assert len(pl) == len(pieces)
        for p, pc in zip(pl, pieces):
            obj = P.decode_object(out.polygons, p.oba & out.poly_mask(), len(pc.obj.records))
            assert obj.records == pc.obj.records
            assert T.texture_points(out, p.tpa, obj) == pc.points
            tha = p.tha
            for r, t, chk in zip(pc.obj.records, pc.textures, pc.checker):
                if r.attr == model.MOVE_ATTR:
                    continue
                h0, h1, h2, h3 = struct.unpack_from("<4H", out.textures, 2 * tha)
                tex = conv.textures[t]
                assert h0 >> 14 & 1 and (h0 >> 15 & 1) == chk and not h0 >> 13 & 1
                assert (32 << (h0 & 7), 32 << (h0 >> 3 & 7)) == (tex.shape[1], tex.shape[0])
                base = h3 << 16 | h2
                lo, n = R.CUSTOM_AREAS["textures"][1]
                assert lo <= 2 * base < lo + n
                assert out.textures[2 * base:2 * base + 2 * tex.size] == tex.astype("<u2").tobytes()
                tha += 4
    # the handling rows: row 0 the arcade's own ratios twice, row 1 the Grasshopper's and those times its torque
    own = struct.unpack_from("<4f", rs.data, build.GEAR_RATIOS_AT)
    assert struct.unpack_from("<8f", out.data, build.CAR_HANDLING) == own + own
    row = struct.unpack_from("<8f", out.data, build.CAR_HANDLING + 32)
    assert [round(v, 3) for v in row[:4]] == [9.962, 6.089, 4.216, 3.319]
    assert all(abs(d - 0.9 * g) < 1e-5 for g, d in zip(row[:4], row[4:]))
    with pytest.raises(ConvertError, match="car 0"):
        build.car_entry(out, 0)
    with pytest.raises(ConvertError, match="car 2"):
        build.car_entry(out, 2)
    with pytest.raises(ConvertError, match="already"):
        build.car_set(out, disc, (1,))
    # only the custom areas, the list home and the table change
    for region in ("program", "copro_data"):
        assert getattr(out, region) == getattr(rs, region)
    a, b = np.frombuffer(rs.data, np.uint8), np.frombuffer(out.data, np.uint8)
    changed = np.nonzero(a != b)[0]
    from dtools.arcade import tgp
    home = range(tgp.DATA_FREE[0], tgp.DATA_FREE[0] + tgp.DATA_FREE[1])
    assert all(int(k) in home or build.CAR_HANDLING <= k < build.CAR_TABLE + build.CAR_TABLE_SIZE for k in changed)
    for region in ("polygons", "textures"):
        a, b = np.frombuffer(getattr(rs, region), np.uint8), np.frombuffer(getattr(out, region), np.uint8)
        changed = np.nonzero(a != b)[0]
        assert len(changed) and all(any(lo <= k < lo + n for lo, n in R.CUSTOM_AREAS[region]) for k in changed[[0, -1]])
    print(f"\ncar 1: {build.car_room(rs, out)}")


def test_car_set_refusals(disc):  # noqa: F811
    stock = _romset()
    if stock is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    with pytest.raises(ConvertError, match="custom"):
        build.car_set(stock, disc, (1,))
    rs = C.patch_library_bound(stock.as_custom())
    for bad in ((0,), (5,), (1, 1), "1", ()):
        with pytest.raises(ConvertError, match="cars"):
            build.car_set(rs, disc, bad)


def test_car_set_field(disc):  # noqa: F811
    """car_set's `field`: the Dreamcast's forty opponents filed after the cars. Every car index 1-43 has a row of
    four lists (near, mid, far, far) that read back as objects of falling polygon counts; car n drives car
    (n // 4) % 10 of set n % 4, so indices 1 and 41 share their lists and 1, 2, 5 do not; the cars' own rows are as
    without the field; nothing outside the custom areas, the list home and the reserved tables changes."""
    stock = _romset()
    if stock is None:
        pytest.skip("set DAYTONA_ROMDIR to a directory containing daytona.zip")
    rs = C.patch_library_bound(stock.as_custom())
    plain, out = build.car_set(rs, disc, (1,)), build.car_set(rs, disc, (1,), field=True)
    assert build.car_entry(out, 1) == build.car_entry(plain, 1)
    with pytest.raises(ConvertError, match="field"):
        build.field_lists(plain, 1)

    def polys(lst):
        return sum(1 for p in C.object_list(out, lst, "field list")
                   for r in P.decode_object(out.polygons, p.oba & out.poly_mask(), p.raw[0]).records
                   if r.attr != model.MOVE_ATTR)

    for n in (1, 2, 5, 23, 43):
        near, mid, far, far2 = build.field_lists(out, n)
        assert far == far2 and 600 < polys(near) > polys(mid) > polys(far) > 60, n
        assert len(C.object_list(out, near, "x")) >= 4 and len(C.object_list(out, mid, "x")) >= 2
    assert build.field_lists(out, 1) == build.field_lists(out, 41)
    assert len({build.field_lists(out, n) for n in (1, 2, 5)}) == 3
    for bad in (0, 44, True):
        with pytest.raises(ConvertError, match="car"):
            build.field_lists(out, bad)
    a, b = np.frombuffer(rs.data, np.uint8), np.frombuffer(out.data, np.uint8)
    from dtools.arcade import tgp
    home = (tgp.DATA_FREE[0], tgp.DATA_FREE[0] + tgp.DATA_FREE[1])
    assert all(home[0] <= k < home[1] or build.FIELD_TABLE <= k < build.CAR_TABLE + build.CAR_TABLE_SIZE
               for k in np.nonzero(a != b)[0])
    assert out.program == rs.program and out.copro_data == rs.copro_data
