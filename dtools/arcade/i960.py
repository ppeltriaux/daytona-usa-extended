"""A small i960 (Model 2 main CPU) instruction encoder for code patches: the REG, COBR, CTRL and MEM formats, the
instructions the select patch (dtools/arcade/select.py) needs, labels and branch fixups. Every block it builds is
checked against MAME's unidasm in the tests (tests/test_select.py), and its encodings against words already in the
program ROM.

Formats (i960 KB/SA, little-endian words):

- REG:  opcode[11:4] in bits 31-24, src/dst 23-19, src2 18-14, m3 m2 m1 13-11, opcode[3:0] 10-7, s2 s1 6-5, src1
  4-0; m1 / m2 set: src1 / src2 is a literal 0-31 (an unused src2 or dst has its m bit set, as in the ROM).
  `subi a,b,d` is d = b - a; `cmpi a,b` compares a with b.
- COBR: opcode 31-24, src1 23-19 (a literal when m1), src2 18-14, m1 13, a signed byte displacement in 12-2
  (+-4 KB): `cmpobl a,b,t` branches when a < b (unsigned).
- CTRL: opcode 31-24, a signed byte displacement in 23-2 (+-8 MB): b, call, ret, bal and the conditional branches.
- MEMA: opcode 31-24, src/dst 23-19, abase 18-14, bit 13 (abase + offset, else offset alone), bit 12 = 0, offset
  11-0. MEMB: opcode 31-24, src/dst 23-19, abase 18-14, mode 13-10, scale 9-7, index 4-0, then a 32-bit
  displacement word for the modes that take one.

Errors are I960Error (a DecodeError)."""

from __future__ import annotations

import struct

from dtools.arcade.errors import DecodeError


class I960Error(DecodeError):
    pass


_REGS = {**{f"r{i}": i for i in range(16)}, **{f"g{i}": 16 + i for i in range(16)},
         "pfp": 0, "sp": 1, "rip": 2, "fp": 31}

REG_OPS = {"addo": 0x590, "addi": 0x591, "subo": 0x592, "subi": 0x593, "shro": 0x598, "shri": 0x59B,
           "shlo": 0x59C, "cmpo": 0x5A0, "cmpi": 0x5A1, "and": 0x581, "or": 0x587, "xor": 0x586, "mov": 0x5CC,
           "mulo": 0x701}
COBR_OPS = {"bbc": 0x30, "bbs": 0x37, "cmpobg": 0x31, "cmpobe": 0x32, "cmpobge": 0x33, "cmpobl": 0x34, "cmpobne": 0x35, "cmpoble": 0x36,
            "cmpibg": 0x39, "cmpibe": 0x3A, "cmpibge": 0x3B, "cmpibl": 0x3C, "cmpibne": 0x3D, "cmpible": 0x3E}
CTRL_OPS = {"b": 0x08, "call": 0x09, "ret": 0x0A, "bal": 0x0B, "bg": 0x11, "be": 0x12, "bge": 0x13, "bl": 0x14,
            "bne": 0x15, "ble": 0x16}
MEM_OPS = {"ldob": 0x80, "stob": 0x82, "ldos": 0x88, "stos": 0x8A, "lda": 0x8C, "ld": 0x90, "st": 0x92,
           "ldib": 0xC0, "stib": 0xC2, "ldis": 0xC8, "stis": 0xCA}
_SCALES = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}


def _reg(r) -> int:
    if not isinstance(r, str) or r not in _REGS:
        raise I960Error(f"{r!r} is not an i960 register (r0-r15, g0-g15)")
    return _REGS[r]


def _op(op, table: dict, kind: str) -> str:
    if not isinstance(op, str) or op not in table:
        raise I960Error(f"unknown {kind} instruction {op!r:.60}")
    return op


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _operand(v) -> tuple[int, int]:
    """(field, literal flag) of a register name or a literal 0-31 (bool refused)."""
    if isinstance(v, str):
        return _reg(v), 0
    if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 31:
        return v, 1
    raise I960Error(f"{v!r} is neither a register nor a literal 0-31")


def _u32(v, what: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not -0x80000000 <= v <= 0xFFFFFFFF:
        raise I960Error(f"{what} {v!r} is not a 32-bit integer")
    return v & 0xFFFFFFFF


def reg(op: str, *operands) -> bytes:
    """A REG instruction: `mov src1,dst`, `cmpo/cmpi src1,src2`, else `op src1,src2,dst`."""
    _op(op, REG_OPS, "REG")
    want = 2 if op in ("mov", "cmpo", "cmpi") else 3
    if len(operands) != want:
        raise I960Error(f"{op} takes {want} operands, not {len(operands)}")
    if op == "mov":
        src1, src2, dst = operands[0], None, operands[1]
    elif want == 2:
        src1, src2, dst = operands[0], operands[1], None
    else:
        src1, src2, dst = operands
    code = REG_OPS[op]
    s1, m1 = _operand(src1)
    s2, m2 = (0, 1) if src2 is None else _operand(src2)       # an unused src2 (mov) is a literal 0, as the ROM's
    d, m3 = (0, 1) if dst is None else (_reg(dst), 0)          # and an unused dst (cmpo / cmpi) sets m3
    w = (code >> 4) << 24 | d << 19 | s2 << 14 | m3 << 13 | m2 << 12 | m1 << 11 | (code & 0xF) << 7 | s1
    return struct.pack("<I", w)


def cobr(op: str, src1, src2: str, disp: int) -> bytes:
    """A compare-and-branch to `disp` bytes from this instruction (a multiple of 4 within +-4 KB)."""
    _op(op, COBR_OPS, "COBR")
    s1, m1 = _operand(src1)
    s2 = _reg(src2)
    if not _int(disp) or disp % 4 or not -0x1000 <= disp < 0x1000:
        raise I960Error(f"{op}: displacement {disp!r} is not a multiple of 4 within -4096..4095")
    w = COBR_OPS[op] << 24 | s1 << 19 | s2 << 14 | m1 << 13 | (disp & 0x1FFC)
    return struct.pack("<I", w)


def ctrl(op: str, disp: int | None = None) -> bytes:
    """A branch or call to `disp` bytes from this instruction (a multiple of 4 within +-8 MB), or `ret` (no disp)."""
    _op(op, CTRL_OPS, "CTRL")
    if op == "ret":
        if disp is not None:
            raise I960Error(f"ret takes no displacement, not {disp!r:.60}")
        return struct.pack("<I", CTRL_OPS[op] << 24)
    if not _int(disp) or disp % 4 or not -0x800000 <= disp < 0x800000:
        raise I960Error(f"{op}: displacement {disp!r} is not a multiple of 4 within +-8 MB")
    return struct.pack("<I", CTRL_OPS[op] << 24 | (disp & 0xFFFFFC))


def mem(op: str, reg_: str, disp: int | None = None, base: str | None = None, index: str | None = None,
        scale: int = 1) -> bytes:
    """A load, store or lda: `op reg_, disp(base)[index*scale]` with any of disp / base / index left out. MEMA
    (4 bytes) when there is no index and disp fits 0-4095, else MEMB (8 bytes with a displacement)."""
    _op(op, MEM_OPS, "MEM")
    d = _reg(reg_)
    b = 0 if base is None else _reg(base)
    if not _int(scale) or scale not in _SCALES:
        raise I960Error(f"scale {scale!r} is not 1, 2, 4, 8 or 16")
    if disp is None and base is None:
        raise I960Error(f"{op}: no address (neither a displacement nor a base)")
    head = MEM_OPS[op] << 24 | d << 19 | b << 14
    if index is None and (disp is None or (isinstance(disp, int) and not isinstance(disp, bool)
                                           and 0 <= disp <= 0xFFF)):
        if disp is None:                                             # (abase): MEMB mode 0100
            return struct.pack("<I", head | 0b0100 << 10)
        return struct.pack("<I", head | (1 << 13 if base is not None else 0) | disp)
    if index is None:
        mode = 0b1101 if base is not None else 0b1100
        x = 0
    else:
        x = _reg(index)
        if disp is None:
            return struct.pack("<I", head | 0b0111 << 10 | _SCALES[scale] << 7 | x)
        mode = 0b1111 if base is not None else 0b1110
    return struct.pack("<II", head | mode << 10 | _SCALES[scale] << 7 | x, _u32(disp, f"{op} displacement"))


class Block:
    """Code assembled at i960 address `at`: append instructions, place labels, branch to labels or absolute
    addresses; `bytes()` resolves the fixups. I960Error for an unknown or twice-placed label or a target out of
    range."""

    def __init__(self, at: int):
        if not isinstance(at, int) or isinstance(at, bool) or at % 4 or not 0 <= at <= 0xFFFFFFFF:
            raise I960Error(f"block address {at!r} is not a 4-aligned 32-bit address")
        self.at = at
        self.code = bytearray()
        self.labels: dict[str, int] = {}
        self.fixups: list[tuple[int, str, str, tuple]] = []

    @property
    def here(self) -> int:
        return self.at + len(self.code)

    def label(self, name: str) -> None:
        if not isinstance(name, str):
            raise I960Error(f"label {name!r:.60} is not a str")
        if name in self.labels:
            raise I960Error(f"label {name!r} placed twice")
        self.labels[name] = self.here

    def reg(self, op: str, *args) -> None:
        self.code += reg(op, *args)

    def mem(self, op: str, reg_: str, disp=None, base=None, index=None, scale: int = 1) -> None:
        self.code += mem(op, reg_, disp, base, index, scale)

    def ctrl(self, op: str, target: int | str | None = None) -> None:
        """`target` an absolute address or a label (None for ret only)."""
        _op(op, CTRL_OPS, "CTRL")
        if target is None or op == "ret":
            self.code += ctrl(op) if target is None else ctrl(op, target)
        elif isinstance(target, str):
            self.fixups.append((len(self.code), "ctrl", op, (target,)))
            self.code += b"\0" * 4
        elif _int(target):
            self.code += ctrl(op, target - self.here)
        else:
            raise I960Error(f"{op}: target {target!r:.60} is neither an address nor a label")

    def cobr(self, op: str, src1, src2: str, target: int | str) -> None:
        _op(op, COBR_OPS, "COBR")
        if isinstance(target, str):
            self.fixups.append((len(self.code), "cobr", op, (src1, src2, target)))
            self.code += b"\0" * 4
        elif _int(target):
            self.code += cobr(op, src1, src2, target - self.here)
        else:
            raise I960Error(f"{op}: target {target!r:.60} is neither an address nor a label")

    def bytes(self) -> bytes:
        out = bytearray(self.code)
        for pos, kind, op, args in self.fixups:
            name = args[-1]
            if name not in self.labels:
                raise I960Error(f"label {name!r} is never placed")
            disp = self.labels[name] - (self.at + pos)
            out[pos:pos + 4] = ctrl(op, disp) if kind == "ctrl" else cobr(op, args[0], args[1], disp)
        return bytes(out)
