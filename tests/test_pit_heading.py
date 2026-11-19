"""Offline tests of the pit stop's heading patch (M9 stage 4; course.patch_pit_heading): the three angle differences
of the pit stop wrapped to 16 bits (the trigger 0xdf9c's window, the stall steering's turn and its waypoint-passed
test in 0xe744), so a box polygon whose direction word lies near +-0x8000 (Dinosaur Canyon and Desert City in
reverse: -32768) starts the stop and steers it to the stall whichever side of the wrap the angles lie. Need
DAYTONA_ROMDIR."""

import pytest

from dtools.arcade import course as C
from dtools.arcade.romset import RomSet


def _window(yaw: int, ref: int, patched: bool) -> bool:
    """The trigger's check 0xdfc8-0xdff8 as the code computes it: ldis/ldis (sign-extended), subi, + 0x3fff, then
    (patched) the low 16 bits; in 0x2800..0x47ff."""
    v = yaw - ref + 0x3FFF
    return 0x2800 <= (v & 0xFFFF if patched else v) <= 0x47FF


def _passed(angle: int, ref: int, patched: bool) -> bool:
    """0xe7fc-0xe824: the waypoint at `angle` from the car counts as passed when angle - box + 0x3fff (patched: its
    low 16 bits) is outside 0..0x7fff."""
    v = angle - ref + 0x3FFF
    v = v & 0xFFFF if patched else v
    return not 0 <= v <= 0x7FFF


def _turn(angle: int, yaw: int, patched: bool) -> int:
    """0xe79c-0xe7c0: the turn this frame, angle - yaw (patched: sign-extended from its low 16 bits) held to
    -0x80..0x80."""
    d = angle - yaw
    if patched:
        d = (d + 0x8000) % 0x10000 - 0x8000
    return max(-0x80, min(0x80, d))


def test_pit_heading_window():
    """The stock check fails a car heading due south on a box at -32768 when its yaw is +32522 (246 past the wrap:
    Desert City in reverse, test_pit_stop_driven[11]) and passes it at -32522; the patched one passes both. Off the
    wrap (every original box: -16383, 21587, ...) the two agree on every yaw."""
    assert not _window(32522, -32768, False) and _window(-32522, -32768, False)
    assert _window(32522, -32768, True) and _window(-32522, -32768, True)
    for ref in (-16383, -4, 16351, 21587, -26625, 30719):
        assert all(_window(y, ref, False) == _window(y, ref, True) for y in range(-32768, 32768, 7)), ref


def test_pit_stall_angles():
    """On a box at -32768, a waypoint dead ahead at +32700 counts as passed in stock (Desert City in reverse: the
    stop ran through its seven waypoints in 238 frames, the car never standing) and not once patched; a waypoint at
    +32700 with the car's yaw at -32700 turns the car 0x80 the long way in stock and the short way (-0x80) patched.
    The patched passed test is the true one on every box (the waypoint more than a quarter turn from the box's
    direction, the difference wrapped); the stock one agrees with it while the half-turn window does not straddle
    the wrap (a box within 0x4000 of 0: -16383, -4, 16351), not on National Park Speedway's 21587, where it called a
    waypoint about 61-90 degrees to one side passed (-30000: 77 degrees). The turn agrees where neither angle
    straddles the wrap."""
    assert _passed(32700, -32768, False) and not _passed(32700, -32768, True)
    assert not _passed(-32700, -32768, True) and _passed(0, -32768, True)
    assert _turn(32700, -32700, False) == 0x80 and _turn(32700, -32700, True) == -0x80
    for ref in (-32768, -16383, -4, 16351, 21587, 30000):
        assert all(_passed(a, ref, True) == (not -0x3FFF <= (a - ref + 0x8000) % 0x10000 - 0x8000 <= 0x4000)
                   for a in range(-32768, 32768, 7)), ref
    for ref in (-16383, -4, 16351):
        assert all(_passed(a, ref, False) == _passed(a, ref, True) for a in range(-32768, 32768, 7)), ref
    assert _passed(-30000, 21587, False) and not _passed(-30000, 21587, True)
    for yaw in (-16383, 0, 16351):
        assert all(_turn(a, yaw, False) == _turn(a, yaw, True) for a in range(yaw - 16000, yaw + 16000, 7)), yaw


def test_patch_pit_heading(romset_zip):
    """The patch rewrites each of PIT_HEADING_SITES (the trigger's window 0xdfd0-0xdffb, the steering 0xe798-0xe7c3,
    the passed test 0xe7fc-0xe827) from its stock bytes to its patched ones of the same length, and nothing else;
    applied twice it is the same set; unidasm reads every patched word (no `?`), each with its shlo 16 / shro 16 or
    shri 16 (MAME's i960 has no `extract`: "Unhandled 65.1")."""
    from dtools.arcade.disasm import listing
    stock = RomSet.from_zip(romset_zip)
    out = C.patch_pit_heading(stock)
    allowed = set()
    for at, (old, new) in C.PIT_HEADING_SITES.items():
        assert len(old) == len(new), hex(at)
        assert bytes(stock.program[at:at + len(old)]) == old, hex(at)
        assert bytes(out.program[at:at + len(new)]) == new, hex(at)
        allowed |= set(range(at, at + len(new)))
        text = [ln.text.replace(" ", "") for ln in listing(out, at, len(new) // 4) if ln.addr < at + len(new)]
        assert not [t for t in text if t.startswith("?")], (hex(at), text)
        assert "shlo16,r3,r3" in text or "shlo16,r6,r6" in text, (hex(at), text)
        assert "shro16,r3,r3" in text or "shri16,r6,r6" in text, (hex(at), text)
    diff = {i for i in range(len(stock.program)) if stock.program[i] != out.program[i]}
    assert diff and diff <= allowed, sorted(diff - allowed)[:8]
    assert set(C.PIT_HEADING_SITES) == {0xDFD0, 0xE798, 0xE7FC}
    assert C.patch_pit_heading(out) is out


def test_patch_pit_heading_refused(romset_zip):
    """Refused (CourseError) on a set whose words at a site are neither the stock nor the patched ones, and on a set
    with some sites patched and others not."""
    stock = RomSet.from_zip(romset_zip)
    for at in C.PIT_HEADING_SITES:
        other = stock.with_patches({("program", at): b"\0" * 4})
        with pytest.raises(C.CourseError, match="pit"):
            C.patch_pit_heading(other)
    at, (_, new) = next(iter(C.PIT_HEADING_SITES.items()))
    with pytest.raises(C.CourseError, match="pit"):
        C.patch_pit_heading(stock.with_patches({("program", at): new}))
