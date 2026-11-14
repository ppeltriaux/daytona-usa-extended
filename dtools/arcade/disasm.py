"""i960 (Model 2), MB86233 (the Model 2 TGP) and SH-4 (Dreamcast) disassembly via MAME's unidasm, plus a cheap
cross-reference finder.

unidasm prints `<addr>: <word> [<word>]  <text>`: the address is unpadded hex, each word is the little-endian
value of its bytes (8 hex digits for the i960 and the MB86233, 4 for the SH-4), then two or more spaces and the
instruction. The MB86233 is word-addressed (`-basepc` and the printed address count 32-bit words, one instruction
per word), so its programs go through `listing_words`; the byte-addressed ones through `listing_bytes`.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dtools.config import REPO_ROOT
from dtools.arcade.romset import RomSet


def unidasm() -> Path:
    """The unidasm binary, resolved at each call: DAYTONA_UNIDASM, else ../virtua-racing/build/mame/unidasm, else
    build/mame/unidasm."""
    if env := os.environ.get("DAYTONA_UNIDASM"):
        return Path(env)
    sibling = REPO_ROOT.parent / "virtua-racing" / "build" / "mame" / "unidasm"
    return sibling if sibling.exists() else REPO_ROOT / "build" / "mame" / "unidasm"


# i960 base address of each region, and the RomSet image and offset it shows (as RomSet.read_i960 maps them)
REGIONS = {"program": 0x00000000, "window": 0x00220000, "data": 0x02000000}
_SOURCE = {"program": ("program", 0), "window": ("program", 0x20000), "data": ("data", 0)}
ARCHES = ("i960", "sh4", "mb86233")
WORD_ARCHES = ("mb86233",)                               # addresses count 32-bit words, one instruction per word
_MAX_INSN = 8                                            # longest i960 instruction (opcode + 32-bit displacement)


class DisasmError(Exception):
    pass


@dataclass(frozen=True)
class Line:
    addr: int
    bytes: bytes
    text: str


_LINE = re.compile(r"^([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{4,8} ?)+?)\s{2,}(.*)$")
_HEX = re.compile(r"(?<![0-9a-zA-Z_])(?:#|0x)?([0-9A-Fa-f]{5,8})h?(?![0-9a-zA-Z_])")


def _parse_line(raw: str) -> Line | None:
    m = _LINE.match(raw.rstrip())
    if not m:
        return None
    data = b"".join(int(w, 16).to_bytes(len(w) // 2, "little") for w in m.group(2).split())
    return Line(int(m.group(1), 16), data, m.group(3).strip())


def _blob(rs: RomSet, name: str) -> bytes:
    image, offset = _SOURCE[name]
    return getattr(rs, image)[offset:]


def _region_for(rs: RomSet, addr: int) -> tuple[str, int, bytes]:
    for name, base in REGIONS.items():
        blob = _blob(rs, name)
        if base <= addr < base + len(blob):
            return name, base, blob
    raise DisasmError(f"address {addr:#x} is not in program or data ROM (nor the program window at 0x00220000)")


def _unidasm(blob: bytes, base: int, arch: str, nbytes: int | None = None, suffix: str = ".bin") -> list[Line]:
    if arch not in ARCHES:
        raise DisasmError(f"unsupported architecture {arch!r}; use one of {ARCHES}")
    tool = unidasm()
    if not tool.exists():
        raise DisasmError(f"unidasm not found at {tool}; run tools/build-mame.sh or set DAYTONA_UNIDASM")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(blob)
        path = tmp.name
    cmd = [str(tool), "-arch", arch, "-basepc", f"0x{base:x}"]
    if nbytes is not None:
        cmd += ["-count", str(nbytes)]                   # -count is bytes
    try:
        proc = subprocess.run(cmd + [path], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise DisasmError(f"unidasm failed: {e.stderr[-300:]}") from e
    finally:
        Path(path).unlink(missing_ok=True)
    return [ln for raw in proc.stdout.splitlines() if (ln := _parse_line(raw))]


def listing(rs: RomSet, addr: int, count: int = 64) -> list[Line]:
    """`count` i960 instructions starting at i960 address `addr` (in the program ROM, its window at 0x00220000, or
    the data ROM)."""
    name, base, blob = _region_for(rs, addr)
    n = min(count * _MAX_INSN, base + len(blob) - addr)
    return _unidasm(rs.read_i960(addr, n), addr, "i960", n, f".{name}.bin")[:count]


def listing_bytes(blob: bytes, base: int, arch: str = "i960") -> list[Line]:
    """Every instruction of `blob` as placed at address `base` (`arch` is "i960" or "sh4", little-endian)."""
    if arch in WORD_ARCHES:
        raise DisasmError(f"{arch} is word-addressed; use listing_words")
    return [ln for ln in _unidasm(blob, base, arch) if ln.addr < base + len(blob)]


def listing_words(words: Sequence[int], base: int = 0, arch: str = "mb86233") -> list[Line]:
    """One `Line` per 32-bit word of `words` (stored little-endian) as placed at word address `base`; `Line.addr`
    is a word address. `arch` must be word-addressed ("mb86233")."""
    if arch not in ARCHES:
        raise DisasmError(f"unsupported architecture {arch!r}; use one of {ARCHES}")
    if arch not in WORD_ARCHES:
        raise DisasmError(f"{arch} is not word-addressed (one of {WORD_ARCHES}); use listing_bytes")
    for i, w in enumerate(words):
        if not isinstance(w, int) or not 0 <= w <= 0xFFFFFFFF:
            raise DisasmError(f"word {i} is {w!r}, not a 32-bit unsigned integer")
    blob = b"".join(w.to_bytes(4, "little") for w in words)
    lines = [ln for ln in _unidasm(blob, base, arch, suffix=".tgp.bin") if ln.addr < base + len(words)]
    if [ln.addr for ln in lines] != list(range(base, base + len(words))):
        raise DisasmError(f"unidasm gave {len(lines)} lines for {len(words)} {arch} words at {base:#x}")
    return lines


def immediates(lines: list[Line], lo: int = 0x02000000, hi: int = 0x02FFFFFF) -> set[int]:
    out: set[int] = set()
    for ln in lines:
        for m in _HEX.finditer(ln.text):
            v = int(m.group(1), 16)
            if lo <= v <= hi:
                out.add(v)
    return out


def find_refs(rs: RomSet, value: int, regions: tuple[str, ...] = ("program", "data"), width: int = 4) -> list[int]:
    """i960 addresses where `value` appears little-endian with the given byte width."""
    needle = value.to_bytes(width, "little")
    hits: list[int] = []
    for name in regions:
        base, blob = REGIONS[name], _blob(rs, name)
        i = blob.find(needle)
        while i != -1:
            hits.append(base + i)
            i = blob.find(needle, i + 1)
    return hits
