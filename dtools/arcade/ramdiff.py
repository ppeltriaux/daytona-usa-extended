"""Load and compare the RAM dumps written by dtools/mame/lua/memtrace.lua (DAYTONA_TRACE_DUMP).

`<log>.ram.<frame>.bin` is work RAM 0x00500000-0x005fffff (1 MB), `<log>.ram2.<frame>.bin` is extra RAM
0x00200000-0x0021ffff (128 KB), both raw i960 (little-endian) bytes. Addresses returned here are i960 bus addresses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class RamDiffError(Exception):
    pass


# dump kind -> (base address, size in bytes)
KINDS = {"ram": (0x00500000, 0x100000), "ram2": (0x00200000, 0x20000)}
_NAME = re.compile(r"\.(ram2?)\.(\d+)\.bin$")
_WIDTHS = {1: "<u1", 2: "<u2", 4: "<u4"}


@dataclass(frozen=True)
class Dump:
    frame: int
    base: int
    data: bytes


@dataclass(frozen=True)
class Diff:
    addr: int
    value_a: int
    value_b: int


def load_dump(path: Path) -> Dump:
    """A dump file, its frame and base taken from the `.ram.<frame>.bin` / `.ram2.<frame>.bin` suffix."""
    path = Path(path)
    m = _NAME.search(path.name)
    if m is None:
        raise RamDiffError(f"{path}: name does not end in .ram.<frame>.bin or .ram2.<frame>.bin")
    base, size = KINDS[m.group(1)]
    data = path.read_bytes()
    if len(data) != size:
        raise RamDiffError(f"{path}: size {len(data):#x}, a {m.group(1)} dump is {size:#x} bytes")
    return Dump(int(m.group(2)), base, data)


def _check_pair(a: Dump, b: Dump, allow_frame_mismatch: bool) -> None:
    if len(a.data) != len(b.data):
        raise RamDiffError(f"size mismatch: {len(a.data):#x} (frame {a.frame}) vs {len(b.data):#x} (frame {b.frame})")
    if a.base != b.base:
        raise RamDiffError(f"base mismatch: {a.base:#010x} vs {b.base:#010x}")
    if a.frame != b.frame and not allow_frame_mismatch:
        raise RamDiffError(f"frame mismatch: {a.frame} vs {b.frame} (pass allow_frame_mismatch=True to compare)")


def diff_dumps(a: Dump, b: Dump, width: int = 1, allow_frame_mismatch: bool = False) -> list[Diff]:
    """The `width`-byte little-endian units (1, 2 or 4, at addresses aligned to `width`) that differ between two
    dumps of the same region and, unless allowed, the same frame. RamDiffError names a size, base or frame
    mismatch instead of comparing misaligned bytes."""
    if width not in _WIDTHS:
        raise RamDiffError(f"width must be 1, 2 or 4, not {width}")
    _check_pair(a, b, allow_frame_mismatch)
    if len(a.data) % width:
        raise RamDiffError(f"size {len(a.data):#x} is not a multiple of width {width}")
    va = np.frombuffer(a.data, dtype=_WIDTHS[width])
    vb = np.frombuffer(b.data, dtype=_WIDTHS[width])
    idx = np.flatnonzero(va != vb)
    return [Diff(a.base + int(i) * width, int(va[i]), int(vb[i])) for i in idx]


def find_smooth_vectors(dumps: list[Dump], stride: int = 4) -> list[tuple[int, list[float]]]:
    """Addresses (every `stride` bytes) whose 32-bit float is finite in every dump and changes on every step
    between consecutive dumps by less than 10 units: candidate positions of something moving. Dumps are taken in
    the order given and must share base and size."""
    if len(dumps) < 2:
        raise RamDiffError("find_smooth_vectors needs at least two dumps")
    for d in dumps[1:]:
        _check_pair(dumps[0], d, allow_frame_mismatch=True)
    n = len(dumps[0].data)
    offs = np.arange(0, n - 3, stride)
    # (dumps, addresses) of float32 read at each offset (unaligned strides allowed, so gather bytes per offset)
    with np.errstate(invalid="ignore"):                    # signalling NaNs in raw RAM warn on the cast
        vals = np.stack([np.frombuffer(d.data, dtype=np.uint8)[offs[:, None] + np.arange(4)].copy().view("<f4")[:, 0]
                         for d in dumps]).astype(np.float64)
        ok = np.isfinite(vals).all(axis=0)
        step = np.abs(np.diff(vals, axis=0))
        ok &= ((step > 0) & (step < 10)).all(axis=0)
    return [(dumps[0].base + int(offs[i]), [float(v) for v in vals[:, i]]) for i in np.flatnonzero(ok)]
