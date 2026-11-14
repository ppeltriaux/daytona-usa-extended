"""Walk a Model 2 geometry display list the way MAME's geo_parse does, and count the polygons it draws.

Ports MAME 0.289 src/mame/sega/model2_v.cpp (BSD-3-Clause, the MAME team): geo_parse, geo_process_command and the
geo_* command functions (the words each one reads), and src/mame/sega/model2.cpp for the buffer and the addresses.
See docs/formats/model2-geometry-list.md.

`words` is the buffer RAM image (0x8000 little-endian 32-bit words at i960 0x00900000); offsets are word indices.
`emit` is the inverse of `walk`: the words of a command list, jumps left out (see its docstring for what it keeps).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from dtools.arcade.errors import DecodeError
from dtools.arcade.polyrom import decode_object, polygons
from dtools.arcade.romset import RomSet

BUFFER_WORDS = 0x20000 // 4        # geo_parse stops when the read pointer reaches the end of buffer RAM
COMMAND_CAP = 0x8000               # ... or after this many words read as commands (jumps included)
JUMP = 0x80000000
END = -1
POLY_RAM_WORDS = 0x8000            # polygon RAM 0 and 1, each
POLY_ROM_MASK = 0x3FFFFF           # polygon_rom_mask for the 16 MB polygon ROM (a set's own: RomSet.poly_mask())
NO_COUNT = 0xFFFFF                 # obc == 0 rolls over to this


class GeoError(DecodeError):
    pass


@dataclass(frozen=True)
class Cmd:
    kind: int                      # (opcode >> 23) & 0x1f, or END (-1) when the walk stopped early
    offset: int                    # word index of the opcode
    args: dict
    # The opcode word as read (its data bits 0-22 and 28-30 are not part of `kind`); None for a command built by hand,
    # which `emit` writes as kind << 23. Not part of equality.
    opcode: int | None = field(default=None, compare=False)


class _PastEnd(Exception):
    pass


def _f(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def walk(words: Sequence[int], start: int) -> list[Cmd]:
    """The commands of the list starting at word `start` (the read start byte offset // 4), jumps followed.

    Ends with the end command (0x0f/0x1f), or with Cmd(END, offset, {"stopped_on": kind | None, "reason": ...}):
    "0x8000 commands" (MAME's cap, reached only by a looping list), "buffer end" (the pointer reached the end of
    the buffer between commands) or "past buffer end" (a command's payload runs past the buffer, which MAME would
    read out of bounds; `stopped_on` is that command and `offset` its opcode)."""
    size = min(len(words), BUFFER_WORDS)
    cmds: list[Cmd] = []
    pos = (start * 4 & 0x1FFFF) // 4
    count = 0

    def take(n: int) -> list[int]:
        nonlocal pos
        if n < 0 or pos + n > size:
            raise _PastEnd
        out = list(words[pos:pos + n])
        pos += n
        return out

    def one() -> int:
        return take(1)[0]

    while True:
        if pos >= size:
            cmds.append(Cmd(END, pos, {"stopped_on": None, "reason": "buffer end"}))
            return cmds
        if count >= COMMAND_CAP:
            cmds.append(Cmd(END, pos, {"stopped_on": None, "reason": "0x8000 commands"}))
            return cmds
        count += 1
        offset = pos
        opcode = one()
        if opcode & JUMP:
            pos = (opcode & 0x1FFFF) // 4
            continue
        kind = (opcode >> 23) & 0x1F
        try:
            args = _COMMANDS[kind](take, one)
        except _PastEnd:
            cmds.append(Cmd(END, offset, {"stopped_on": kind, "reason": "past buffer end"}))
            return cmds
        cmds.append(Cmd(kind, offset, args, opcode))
        if kind in (0x0F, 0x1F):
            return cmds


def _nop(take, one) -> dict:
    return {}


def _object(take, one) -> dict:                                    # geo_object_data
    tpa, tha, oba, obc = take(4)
    return {"tpa": tpa, "tha": tha, "oba": oba, "obc": obc}


def _direct(take, one) -> dict:                                    # geo_direct_data
    tpa, tha = take(2)
    p0, p1 = tuple(take(3)), tuple(take(3))
    records = []
    while (attr := one()) & 3:
        luma, distance = take(2)
        points = [tuple(take(3))]
        if attr & 1:                                               # a quad carries a second point
            points.append(tuple(take(3)))
        records.append({"attr": attr, "luma": luma, "distance": distance, "points": points})
    # "end": the attribute word that ended the records (attr & 3 == 0), as read
    return {"tpa": tpa, "tha": tha, "p0": p0, "p1": p1, "records": records, "end": attr}


def _window(take, one) -> dict:                                    # geo_window_data: six 0XXX0YYY coordinates
    raw = take(6)                                                  # "words": as read (bits outside the 12-bit fields)
    return {"coords": [((w >> 16) & 0xFFF, w & 0xFFF) for w in raw], "words": raw}


def _texture(take, one) -> dict:                                   # geo_texture_data / geo_log_data
    address, count = take(2)
    return {"address": address, "count": count, "data": take(count)}


def _polygon_data(take, one) -> dict:                              # geo_polygon_data: an upload to polygon RAM
    address, count = take(2)
    return {"address": address, "ram": 1 if address & 0x01000000 else 0, "index": address & 0x7FFF,
            "count": count, "data": take(count)}


def _texture_parameters(take, one) -> dict:                        # geo_texture_parameters
    index, count = take(2)
    data = take(2 * count)
    return {"index": index >> 2, "index_low": index & 3, "count": count,
            "params": [(data[2 * i], _f(data[2 * i + 1])) for i in range(count)]}


def _mode(take, one) -> dict:
    return {"mode": one()}


def _zsort(take, one) -> dict:
    return {"zsort": one()}


def _focus(take, one) -> dict:
    return {"focus": tuple(_f(w) for w in take(2))}


def _light(take, one) -> dict:
    return {"light": tuple(_f(w) for w in take(3))}


def _matrix(take, one) -> dict:
    return {"mat": [_f(w) for w in take(12)]}


def _translate(take, one) -> dict:
    return {"translate": tuple(_f(w) for w in take(3))}


def _data_mem_push(take, one) -> dict:                             # geo_data_mem_push: the push loop is disabled
    address, count = take(2)
    return {"address": address, "count": count}


def _test(take, one) -> dict:                                      # geo_test: FIFO pattern, then checksum blocks
    fifo = take(32)
    blocks = [tuple(b) for b in (take(3) for _ in range(one()))]
    # "fifo": the 32 words as read; "block_high": each block address word's bits above the 23-bit address
    return {"fifo_ok": fifo == [1 << i for i in range(32)], "fifo": fifo,
            "blocks": [(a & 0x7FFFFF, n, s) for a, n, s in blocks], "block_high": [a & ~0x7FFFFF for a, _, _ in blocks]}


def _dummy(take, one) -> dict:
    return {"value": one()}


def _lod(take, one) -> dict:
    return {"lod": _f(one())}


def _code_upload(take, one) -> dict:                               # geo_code_upload: three words per opcode
    flags, count = take(2)
    return {"flags": flags, "count": count, "data": take(3 * count)}


def _code_jump(take, one) -> dict:
    return {"address": one()}


_COMMANDS = {
    0x00: _nop, 0x01: _object, 0x02: _direct, 0x03: _window, 0x04: _texture, 0x05: _polygon_data,
    0x06: _texture_parameters, 0x07: _mode, 0x08: _zsort, 0x09: _focus, 0x0A: _light, 0x0B: _matrix,
    0x0C: _translate, 0x0D: _data_mem_push, 0x0E: _test, 0x0F: _nop,
    0x10: _dummy, 0x11: _object, 0x12: _direct, 0x13: _window, 0x14: _texture, 0x15: _polygon_data,
    0x16: _lod, 0x17: _mode, 0x18: _zsort, 0x19: _focus, 0x1A: _light, 0x1B: _matrix,
    0x1C: _translate, 0x1D: _code_upload, 0x1E: _code_jump, 0x1F: _nop,
}


# --- emit: the inverse of walk -------------------------------------------------------------------------------------

def _w(v, name: str, bits: int = 32) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < 1 << bits:
        raise GeoError(f"{name} {v!r} is not a u{bits}")
    return v


def _fw(v, name: str) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except (struct.error, OverflowError, TypeError) as e:
        raise GeoError(f"{name} {v!r} is not a single-precision float ({e})") from None


def _floats(vs, n: int, name: str) -> list[int]:
    if not isinstance(vs, (tuple, list)) or len(vs) != n:
        raise GeoError(f"{name} {vs!r} is not {n} floats")
    return [_fw(v, name) for v in vs]


def _ints(vs, n: int, name: str) -> list[int]:
    if not isinstance(vs, (tuple, list)) or len(vs) != n:
        raise GeoError(f"{name} {vs!r} is not {n} words")
    return [_w(v, name) for v in vs]


def _counted(a: dict, head: list[int], what: str) -> list[int]:
    data = a["data"]
    if not isinstance(data, (tuple, list)) or _w(a["count"], f"{what} count") != len(data):
        raise GeoError(f"{what}: count {a['count']!r} is not the {len(data) if isinstance(data, (tuple, list)) else data!r}"
                       f" data words")
    return head + [len(data)] + [_w(v, f"{what} data") for v in data]


def _e_object(a):
    return [_w(a[k], k) for k in ("tpa", "tha", "oba", "obc")]


def _e_direct(a):
    out = [_w(a["tpa"], "tpa"), _w(a["tha"], "tha")] + _ints(a["p0"], 3, "p0") + _ints(a["p1"], 3, "p1")
    for r in a["records"]:
        attr = _w(r["attr"], "record attr")
        if not attr & 3:
            raise GeoError(f"record attr {attr:#x} has attr & 3 == 0, which ends the records")
        out += [attr, _w(r["luma"], "luma"), _w(r["distance"], "distance")]
        pts = r["points"]
        if not isinstance(pts, (tuple, list)) or len(pts) != (2 if attr & 1 else 1):
            raise GeoError(f"record attr {attr:#x}: {pts!r} is not {2 if attr & 1 else 1} point(s)")
        for pt in pts:
            out += _ints(pt, 3, "point")
    end = _w(a.get("end", 0), "end attr")
    if end & 3:
        raise GeoError(f"end attr {end:#x} has attr & 3 != 0, which would not end the records")
    return out + [end]


def _e_window(a):
    coords, raw = a["coords"], a.get("words", [0] * 6)
    if not isinstance(coords, (tuple, list)) or len(coords) != 6:
        raise GeoError(f"window coords {coords!r} are not six (x, y) pairs")
    out = []
    for (x, y), r in zip(coords, _ints(raw, 6, "window words")):
        out.append((r & ~0x0FFF0FFF) | _w(x, "window coords x", 12) << 16 | _w(y, "window coords y", 12))
    return out


def _e_texture(a):
    return _counted(a, [_w(a["address"], "address")], "texture data")


def _e_polygon_data(a):
    return _counted(a, [_w(a["address"], "address")], "polygon data")


def _e_texture_parameters(a):
    params = a["params"]
    if not isinstance(params, (tuple, list)) or _w(a["count"], "count") != len(params):
        raise GeoError(f"texture parameters: count {a['count']!r} is not the {len(params)} parameters")
    out = [_w(a["index"], "index", 30) << 2 | _w(a.get("index_low", 0), "index_low", 2), len(params)]
    for word, coef in params:
        out += [_w(word, "parameter"), _fw(coef, "parameter coefficient")]
    return out


def _e_test(a):
    blocks = a["blocks"]
    fifo = a.get("fifo", [1 << i for i in range(32)])
    high = a.get("block_high", [0] * len(blocks))
    out = _ints(fifo, 32, "fifo") + [len(blocks)]
    if len(high) != len(blocks):
        raise GeoError(f"geo test: {len(high)} block_high words for {len(blocks)} blocks")
    for (addr, n, chk), h in zip(blocks, high):
        if _w(h, "block_high") & 0x7FFFFF:
            raise GeoError(f"block_high {h:#x} overlaps the 23-bit address")
        out += [h | _w(addr, "block address", 23), _w(n, "block count"), _w(chk, "block checksum")]
    return out


def _e_code_upload(a):
    count = _w(a["count"], "count")
    data = a.get("data", [0] * (3 * count))
    if len(data) != 3 * count:
        raise GeoError(f"code upload: {len(data)} data words for count {count} (three per opcode)")
    return [_w(a["flags"], "flags"), count] + [_w(v, "code word") for v in data]


_EMIT = {
    0x00: lambda a: [], 0x01: _e_object, 0x02: _e_direct, 0x03: _e_window, 0x04: _e_texture,
    0x05: _e_polygon_data, 0x06: _e_texture_parameters, 0x07: lambda a: [_w(a["mode"], "mode")],
    0x08: lambda a: [_w(a["zsort"], "zsort")], 0x09: lambda a: _floats(a["focus"], 2, "focus"),
    0x0A: lambda a: _floats(a["light"], 3, "light"), 0x0B: lambda a: _floats(a["mat"], 12, "mat"),
    0x0C: lambda a: _floats(a["translate"], 3, "translate"),
    0x0D: lambda a: [_w(a["address"], "address"), _w(a["count"], "count")], 0x0E: _e_test, 0x0F: lambda a: [],
    0x10: lambda a: [_w(a["value"], "value")], 0x11: _e_object, 0x12: _e_direct, 0x13: _e_window,
    0x14: _e_texture, 0x15: _e_polygon_data, 0x16: lambda a: [_fw(a["lod"], "lod")],
    0x17: lambda a: [_w(a["mode"], "mode")], 0x18: lambda a: [_w(a["zsort"], "zsort")],
    0x19: lambda a: _floats(a["focus"], 2, "focus"), 0x1A: lambda a: _floats(a["light"], 3, "light"),
    0x1B: lambda a: _floats(a["mat"], 12, "mat"), 0x1C: lambda a: _floats(a["translate"], 3, "translate"),
    0x1D: _e_code_upload, 0x1E: lambda a: [_w(a["address"], "address")], 0x1F: lambda a: [],
}


def emit(cmds: Sequence[Cmd]) -> list[int]:
    """The 32-bit words of `cmds` as one list, in order, with no jumps: the inverse of `walk` (for a walked list,
    exactly the words walk read, the jump words left out). Each command is its opcode word (`Cmd.opcode` as read, else
    `kind << 23`) then its payload from `args`. The words walk's fields do not show are taken from the companion
    keys walk adds, when present: window "words" (bits outside the 12-bit fields), texture parameters "index_low"
    (index bits 0-1), geo test "fifo" and "block_high", code upload "data", direct data "end" (the end attribute);
    missing, they are written as 0 (the FIFO pattern for "fifo"). Floats are written as single precision. GeoError
    for a stopped walk's END entry (not a whole list), a kind outside 0x00-0x1f, an opcode whose bits 23-27 are not
    the kind or that is a jump, a word that is not a u32 (or past its field: window coordinates 12 bits, a texture
    parameter index 30 bits, a test-block address 23 bits), a count that does not match its data, a float that does
    not fit single precision, a direct-data record attribute that would end the records."""
    out: list[int] = []
    for c in cmds:
        if c.kind == END:
            raise GeoError(f"the walk stopped at {c.offset:#x} ({c.args.get('reason')}): not a whole list")
        if not isinstance(c.kind, int) or c.kind not in _EMIT:
            raise GeoError(f"command kind {c.kind!r} is outside 0x00-0x1f")
        opcode = c.kind << 23 if c.opcode is None else _w(c.opcode, "opcode")
        if opcode & JUMP or (opcode >> 23) & 0x1F != c.kind:
            raise GeoError(f"opcode {opcode:#x} is not command {c.kind:#x} (bits 23-27, bit 31 clear)")
        try:
            out += [opcode] + _EMIT[c.kind](c.args)
        except KeyError as e:
            raise GeoError(f"command {c.kind:#x} at {c.offset:#x}: args lack {e}") from None
    return out


@dataclass
class PolyRam:
    """The geometry engine state that outlives a list: polygon RAM 0 and 1 (0x8000 words each, laid out one after
    the other as in MAME's geo_state, so an upload or object that runs past RAM 0 continues into RAM 1) and the
    geometry mode (command 0x07/0x17). Fill it by passing every logged list through `count_polygons` in order;
    uploads in frames that were not logged are missed."""
    words: bytearray = field(default_factory=lambda: bytearray(2 * POLY_RAM_WORDS * 4))
    mode: int = 0

    def upload(self, ram: int, index: int, data: Sequence[int]) -> None:
        base = ram * POLY_RAM_WORDS + index
        n = max(0, min(len(data), 2 * POLY_RAM_WORDS - base))      # past RAM 1's end MAME writes outside the RAMs
        struct.pack_into(f"<{n}I", self.words, base * 4, *data[:n])


def count_polygons(cmds: list[Cmd], rs: RomSet, poly_ram: PolyRam) -> int:
    """Drawn polygons in one list: object records whose link bits ((attr >> 8) & 3) are not 0 (the hardware culls
    link 0), decoded in the current mode from polygon ROM or `poly_ram`, plus direct-data records under the same rule.
    Applies the list's uploads and mode writes to `poly_ram` in order. A malformed object raises PolyError."""
    total = 0
    for c in cmds:
        if c.kind in (0x05, 0x15):
            poly_ram.upload(c.args["ram"], c.args["index"], c.args["data"])
        elif c.kind in (0x07, 0x17):
            poly_ram.mode = c.args["mode"]
        elif c.kind in (0x02, 0x12):
            total += sum(1 for r in c.args["records"] if (r["attr"] >> 8) & 3)
        elif c.kind in (0x01, 0x11):
            oba, obc = c.args["oba"], c.args["obc"] or NO_COUNT
            if oba & 0x01000000:
                blob, index = poly_ram.words, POLY_RAM_WORDS + (oba & 0x7FFF)
            elif oba & 0x00800000:
                blob, index = rs.polygons, oba & rs.poly_mask()
            else:
                blob, index = poly_ram.words, oba & 0x7FFF
            obj = decode_object(blob, index, obc, poly_ram.mode & 3)
            total += sum(1 for p in polygons(obj) if p.drawn)
    return total


def parse_log(path: Path) -> Iterator[tuple[int, int, int, list[int]]]:
    """(frame, read_start, videoctl, words) per `G` line of geo_log.lua. `read_start` is the byte offset the i960
    last wrote to 0x00803008 (walk the list with `walk(words, read_start // 4)`); `words` are the 0x8000 buffer
    RAM words."""
    with open(path) as f:
        for line in f:
            if not line.startswith("G "):
                continue
            _, frame, read_start, videoctl, hexdata = line.split()
            data = bytes.fromhex(hexdata)
            yield int(frame), int(read_start, 16), int(videoctl, 16), list(struct.unpack(f">{len(data) // 4}I", data))
