"""The error class, the number rule and the bounds-checked ROM readers shared by the arcade Daytona USA modules
(course.py, tiles.py, golden.py; texture.py and polyrom.py use the number rule with their own error classes).

Addresses are i960 bus addresses; data ROM 0x02000000-0x02ffffff (0x03ffffff on a custom set) is RomSet.data at
offset addr - DATA_BASE.

The number rule (`_int`, `_f32`): an integer field takes Python and numpy integers, never a bool; a float field takes
Python and numpy numbers (integers included, bool not) that are finite and fit single precision. Shapes (`_seq`):
a fixed-length field takes a tuple, list or 1-D array of that length."""

from __future__ import annotations

import math
import numbers
import struct
from typing import TYPE_CHECKING

import numpy as np

from dtools.arcade.errors import DecodeError

# The one home of the address constants (romset, course, texture and tgp use these objects). Defined before the
# romset import: romset imports them from here, and tablebase needs RomSet only at call time.
DATA_BASE = 0x02000000          # data ROM: i960 address = RomSet.data offset + DATA_BASE
DATA_SIZE = 0x01000000          # the original set's data image (16 MB)
DATA_WINDOW = 0x02000000        # the i960's data-ROM window 0x02000000-0x03ffffff: a custom set's whole image (32 MB)
PROGRAM_WINDOW = (0x00220000, 0x00240000)       # program ROM seen again from offset 0x20000

from dtools.arcade import romset as _romset      # noqa: E402  (after the constants: romset imports them)

if TYPE_CHECKING:
    from dtools.arcade.romset import RomSet


class CourseError(DecodeError):
    pass


def _u32(rs: RomSet, addr: int) -> int:
    try:
        return struct.unpack("<I", rs.read_i960(addr, 4))[0]
    except (_romset.RomSetError, struct.error) as e:
        raise CourseError(f"cannot read the word at {addr:#x}: {e}") from None


def data_size(rs: RomSet | None) -> int:
    """The data image's size: len(rs.data) (DATA_SIZE on an original set, DATA_WINDOW on a custom one); DATA_WINDOW,
    the most the i960 window addresses, without a set (an encoder checks an address against what any set can hold;
    RomSet.with_patch refuses a write past the image). CourseError for `rs` neither None nor a RomSet."""
    if rs is None:
        return DATA_WINDOW
    if not isinstance(rs, _romset.RomSet):
        raise CourseError(f"{type(rs).__name__} is not a RomSet")
    return len(rs.data)


def _data_off(ptr: int, size: int, what: str, rs: RomSet | None = None) -> int:
    """The data-ROM offset of i960 address `ptr`, checking that `size` bytes fit `rs`'s data image (`data_size`)."""
    end = DATA_BASE + data_size(rs)
    if not DATA_BASE <= ptr or ptr + size > end:
        raise CourseError(f"{what} at {ptr:#x} ({size} bytes) is outside data ROM {DATA_BASE:#x}-{end - 1:#x}")
    return ptr - DATA_BASE


def is_int(value) -> bool:
    """True for a Python or numpy integer; False for bool (and numpy bool), floats and everything else."""
    return isinstance(value, (numbers.Integral, np.integer)) and not isinstance(value, (bool, np.bool_))


def is_number(value) -> bool:
    """True for a finite Python or numpy integer or float; False for bool, NaN, infinity, an integer past the float
    range (10**400: math.isfinite would raise OverflowError) and everything else."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Real, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _int(value, name: str, lo: int, hi: int, what: str | None = None, error: type[Exception] = CourseError) -> int:
    """`value` as a Python int, checked to be an integer (`is_int`) inside lo..hi; else `error` naming the field."""
    if not is_int(value) or not lo <= value <= hi:
        raise error(f"{what + ': ' if what else ''}{name} {value!r} is outside {lo}..{hi}")
    return int(value)



def _seq(value, n: int | None, name: str, what: str | None = None, error: type[Exception] = CourseError) -> tuple:
    """`value` as a tuple, checked to be a tuple, list or 1-D array (of `n` items when `n` is given)."""
    if isinstance(value, np.ndarray) and value.ndim == 1:
        value = tuple(value.tolist() if value.dtype == object else value)
    prefix = what + ": " if what else ""
    if not isinstance(value, (tuple, list)):
        raise error(f"{prefix}{name} {value!r} is not a sequence (tuple, list or 1-D array)")
    if n is not None and len(value) != n:
        shown = repr(tuple(value)) if len(value) <= 8 else f"{len(value)} items"
        raise error(f"{prefix}{name}: {shown}, expected {n} items")
    return tuple(value)


def _f32(values, n: int, name: str, what: str | None = None, error: type[Exception] = CourseError) -> bytes:
    """The `n` little-endian f32 of `values`: `n` numbers (`is_number`: finite, not bool) that fit single precision,
    else `error` naming the field."""
    prefix = what + ": " if what else ""
    values = _seq(values, n, name, what, error)
    if not all(is_number(v) for v in values):
        raise error(f"{prefix}{name} {values!r} is not {n} finite numbers")
    try:
        return struct.pack(f"<{n}f", *(float(v) for v in values))
    except (struct.error, OverflowError) as e:
        raise error(f"{prefix}{name} {values!r} does not fit single precision ({e})") from None


_S16, _U16 = (-0x8000, 0x7FFF), (0, 0xFFFF)


def _read(rs: RomSet, addr: int, n: int, what: str) -> bytes:
    try:
        return rs.read_i960(addr, n)
    except _romset.RomSetError as e:
        raise CourseError(f"{what}: {e}") from None


def _data_ptr_ok(ptr, what: str, size: int = 1, rs: RomSet | None = None) -> int:
    if not is_int(ptr):
        raise CourseError(f"{what} {ptr!r} is not an address")
    _data_off(int(ptr), size, what, rs)
    return int(ptr)
