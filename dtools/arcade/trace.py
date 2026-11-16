"""Parse and classify the logs written by dtools/mame/lua/memtrace.lua.

Line formats (hex fields without 0x):
  R <frame> <ip> <page> <count>   reads aggregated per frame per (ip, addr & ~0xff); `page` is the page base address
  W <frame> <ip> <page> <count>   writes, same aggregation
  V <frame> <ip> <addr> <value>   one write with the value written (ranges given with a `!` suffix)
  D <frame> <kind> <path>         a RAM dump file was written
  END <frames_traced> <events>    last line; its absence means the run was cut off

`ip` is MAME's i960 `state["ip"]` read inside the tap callback. It is the address of the instruction AFTER the
accessing one (checked 2026-11-07 against unidasm: `st r3,0x91fff0` at 0x11c44, 8 bytes, logs ip 0x11c4c; `st
r3,0x900000(r10)` at 0x1a410 logs 0x1a418), so the accessing instruction is the one that ends at `ip`.
Addresses are i960 bus addresses; MAME mirrors are not resolved (`resolve` gives None for them).
V events carry count 1 and an exact `addr`; R/W events carry the page base. A V line records what the CPU stored,
not the raw tap arguments: MAME passes the tap a 32-bit-aligned address, the whole data word and a byte-lane mask,
and memtrace.lua turns that into the lowest byte address written and the masked data shifted down to it, printed
with two hex digits per byte (a byte store `V 5 1234 50da9d 2a`, a half-word `... 002a`, a word `... 0000002a`).
`Event.value` is that integer; the store width is only in the digit count of the line.
D lines are not returned as events.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


class TraceError(Exception):
    pass


@dataclass(frozen=True)
class Event:
    kind: str            # "R", "W" or "V"
    frame: int
    ip: int
    addr: int            # page base for R/W, exact address for V
    count: int
    value: int | None


# (lo, hi inclusive, name, offset subtracted). MAME mirrors are ignored: a mirrored address resolves to None.
_REGIONS = (
    (0x00000000, 0x0003ffff, "program", 0),
    (0x00200000, 0x0021ffff, "ram2", 0x200000),
    (0x00220000, 0x0023ffff, "program", 0x200000),     # program ROM window: offset 0x20000+, as RomSet.read_i960
    (0x00500000, 0x005fffff, "workram", 0x500000),
    (0x00900000, 0x0091ffff, "buffer", 0x900000),
    (0x01000000, 0x0100ffff, "tile", 0x1000000),
    (0x01080000, 0x010fffff, "char", 0x1080000),
    (0x01800000, 0x01803fff, "palette", 0x1800000),
    (0x01d00000, 0x01d03fff, "backup", 0x1d00000),
    (0x02000000, 0x03ffffff, "data", 0x2000000),        # the custom set fills the whole window (RomSet.as_custom)
)


def resolve(addr: int) -> tuple[str, int] | None:
    """(region, offset) for an i960 address, None outside the known regions (mirrors included)."""
    for lo, hi, name, sub in _REGIONS:
        if lo <= addr <= hi:
            return name, addr - sub
    return None


def parse_trace(path: Path) -> list[Event]:
    events: list[Event] = []
    end = None
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            p = line.split()
            if not p:
                continue
            try:
                if end is not None:
                    raise ValueError("data after END")
                k = p[0]
                if k in ("R", "W") and len(p) == 5:
                    events.append(Event(k, int(p[1]), int(p[2], 16), int(p[3], 16), int(p[4]), None))
                elif k == "V" and len(p) == 5:
                    events.append(Event("V", int(p[1]), int(p[2], 16), int(p[3], 16), 1, int(p[4], 16)))
                elif k == "D" and len(p) == 4:
                    int(p[1])
                elif k == "END" and len(p) == 3:
                    end = (int(p[1]), int(p[2]))
                else:
                    raise ValueError("unknown line")
            except ValueError as e:
                raise TraceError(f"{path}: line {n}: malformed ({e}): {line.strip()!r}") from None
    if end is None:
        raise TraceError(f"{path}: no END line (the run was cut off or the log is incomplete)")
    return events


def by_ip(events: list[Event], kind: str = "R") -> dict[int, Counter]:
    out: dict[int, Counter] = defaultdict(Counter)
    for e in events:
        if e.kind == kind:
            out[e.ip][e.addr] += e.count
    return dict(out)


def by_page(events: list[Event], kind: str = "R") -> dict[int, Counter]:
    out: dict[int, Counter] = defaultdict(Counter)
    for e in events:
        if e.kind == kind:
            out[e.addr][e.ip] += e.count
    return dict(out)


def summary(events: list[Event]) -> str:
    if not events:
        return "0 events"
    frames = [e.frame for e in events]
    parts = [f"{len(events)} events, frames {min(frames)}-{max(frames)}"]
    for k in ("R", "W", "V"):
        sel = [e for e in events if e.kind == k]
        if sel:
            parts.append(f"{k}: {sum(e.count for e in sel)} accesses, {len({e.ip for e in sel})} ips, "
                         f"{len({e.addr for e in sel})} addrs")
    return "; ".join(parts)
