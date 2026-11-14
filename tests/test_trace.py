import pytest

from dtools.arcade.romset import CRCS, RomSet
from dtools.arcade.trace import Event, TraceError, by_ip, by_page, parse_trace, resolve, summary

SAMPLE = """R 1500 00001234 02123400 5
W 1501 00005678 00900100 3
V 1501 00011c4c 0050da9d 2a
D 1520 ram /tmp/x.ram.1520.bin
END 2 8
"""


def test_parse_trace_round_trip(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(SAMPLE)
    ev = parse_trace(p)
    assert ev == [Event("R", 1500, 0x1234, 0x2123400, 5, None),
                  Event("W", 1501, 0x5678, 0x900100, 3, None),
                  Event("V", 1501, 0x11c4c, 0x50da9d, 1, 0x2a)]   # D lines are bookkeeping, not events
    assert by_ip(ev)[0x1234][0x2123400] == 5
    assert by_page(ev, "W")[0x900100][0x5678] == 3
    assert "events" in summary(ev)


def test_missing_end_line_raises(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("R 1 0 100 1\n")
    with pytest.raises(TraceError, match="END"):
        parse_trace(p)


def test_malformed_line_names_line_number(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("R 1 0 100 1\nR 2 zz\nEND 1 1\n")
    with pytest.raises(TraceError, match="line 2"):
        parse_trace(p)


def test_old_v_line_without_ip_is_rejected(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("V 1501 0050da9c 0000002a\nEND 1 1\n")
    with pytest.raises(TraceError, match="line 1"):
        parse_trace(p)


def test_resolve_regions():
    assert resolve(0x02123456) == ("data", 0x123456)
    assert resolve(0x03001234) == ("data", 0x1001234)        # the custom set's second 16 MB (course.ID_HOME)
    assert resolve(0x00221000) == ("program", 0x21000)
    assert resolve(0x0050da9c) == ("workram", 0xda9c)
    assert resolve(0x00200010) == ("ram2", 0x10)
    assert resolve(0x00900004) == ("buffer", 4)
    assert resolve(0x01000008) == ("tile", 8)
    assert resolve(0x01080008) == ("char", 8)
    assert resolve(0x01800008) == ("palette", 8)
    assert resolve(0x01d00008) == ("backup", 8)
    assert resolve(0x00700000) is None
    assert resolve(0x00040000) is None


def test_resolve_agrees_with_read_i960():
    import random
    from dtools.arcade.romset import DATA_PAIRS, PROGRAM_PAIR
    rng = random.Random(1)
    files = {n: bytes([i + 1] * 8) for i, n in enumerate(CRCS)}
    for n in PROGRAM_PAIR:
        files[n] = rng.randbytes(0x20000)
    for n in DATA_PAIRS[0][0]:
        files[n] = rng.randbytes(0x100000)
    rs = RomSet.from_files(files, verify=False)
    for addr in (0x100, 0x221000, 0x23fffc, 0x02123456):
        region, off = resolve(addr)
        blob = rs.data if region == "data" else rs.program
        got = rs.read_i960(addr, 4)
        assert blob[off:off + 4] == got and got != bytes(4)


def test_empty_window_gives_zero_events(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("END 0 0\n")
    ev = parse_trace(p)
    assert ev == []
    assert "0 events" in summary(ev)


@pytest.mark.mame
def test_trace_real_run(tmp_path, romset_zip):
    """Course 0, 30 s emulated; reads of data ROM, writes of buffer RAM, frames 1500-1530, dump at 1520.
    Run time: about 9 s wall (measured 2026-11-07, MAME 0.289)."""
    from dtools.mame.runner import LUA_DIR, run_mame
    log = tmp_path / "trace.txt"
    run_mame(LUA_DIR / "autoplay.lua", 30, tmp_path / "run", romset_zip.parent, env={
        "DAYTONA_COURSE": "0", "DAYTONA_TRACE_LOG": str(log), "DAYTONA_TRACE_FROM": "1500",
        "DAYTONA_TRACE_TO": "1530", "DAYTONA_TRACE_READ": "2000000-2ffffff",
        "DAYTONA_TRACE_WRITE": "900000-91ffff,500000-5fffff!", "DAYTONA_TRACE_DUMP": "1520"})
    ev = parse_trace(log)
    assert any(e.kind == "R" and (r := resolve(e.addr)) and r[0] == "data" for e in ev)
    assert any(e.kind == "W" and (r := resolve(e.addr)) and r[0] == "buffer" for e in ev)
    assert (tmp_path / "trace.txt.ram.1520.bin").stat().st_size == 0x100000
    assert (tmp_path / "trace.txt.ram2.1520.bin").stat().st_size == 0x20000
    ips = by_ip(ev)
    assert ips
    assert all((r := resolve(ip)) and r[0] == "program" for ip in ips)
    assert all(1500 <= e.frame < 1530 for e in ev)
    # V lines: the writer's ip, the byte actually written and the masked value (work RAM sees byte, half and word stores)
    vs = [e for e in ev if e.kind == "V"]
    assert vs and all((r := resolve(e.ip)) and r[0] == "program" for e in vs)
    assert all(0x500000 <= e.addr <= 0x5fffff and 0 <= e.value < 1 << 32 for e in vs)
    assert any(e.addr & 3 for e in vs)                     # sub-word stores land on their own byte address
    widths = {len(line.split()[4]) for line in log.read_text().splitlines() if line.startswith("V ")}
    assert widths <= {2, 4, 8} and len(widths) > 1
