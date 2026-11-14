"""The Dreamcast background screens `LG_BGn.BIN`, `CHP_BGn.BIN` and `NAME_BGnn.BIN` (n the course, 0-7; 24 files):
a lossless decoder and encoder.

Each file is one 640 x 480 RGB565 picture, 614,400 bytes, no header: eight PowerVR textures one after the other,
each twiddled like the course textures (`tex.twiddle_index`: bit 2i of the index is y bit i, bit 2i+1 is x bit i; a
rectangle is squares of its short side in sequence) and each stored bottom row first. The pieces (`PIECES`) cut the
picture into bands of 256, 128, 64 and 32 rows from the top, each band a 512-wide piece then a 128-wide one
(docs/formats/dreamcast-disc.md, "Background screens"; docs/spike/2026-11-dc-decode.md, "Screens").

`Screen.data` holds the file's bytes as stored; `texels` and `to_rgba` give the picture top row first (unlike
`tex.to_rgba`, which keeps a texture's stored order: here the whole picture was matched pixel for pixel against
MAME's frame buffer). Every malformed input is a `DcError` naming the file and the offset, or the field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dtools.arcade.tablebase import _int, is_int
from dtools.dc.disc import COURSE_COUNT, course_files
from dtools.dc.errors import DcError
from dtools.dc.tex import texels_to_rgba, twiddle_index

KINDS = ("bg_lg", "bg_chp", "bg_name")      # dtools.dc.disc.course_files kinds: LG_BGn, CHP_BGn, NAME_BGnn
WIDTH, HEIGHT = 640, 480
PIXEL = "RGB565"
LAYOUT = "twiddled_bands"
SIZE = WIDTH * HEIGHT * 2                   # 614,400 bytes


def _pieces() -> tuple[tuple[int, int, int, int, int], ...]:
    """(y, x, width, height, first texel) of each piece in stored order."""
    out, y, start = [], 0, 0
    for h in (256, 128, 64, 32):
        x = 0
        for w in (512, 128):
            out.append((y, x, w, h, start))
            x += w
            start += w * h
        y += h
    assert y == HEIGHT and start == WIDTH * HEIGHT
    return tuple(out)


PIECES = _pieces()


def _stored_order() -> np.ndarray:
    """For each stored texel, its flat index (row * 640 + column) in the picture, top row first."""
    order = np.empty(WIDTH * HEIGHT, dtype=np.int64)
    for y, x, w, h, start in PIECES:
        idx = twiddle_index(w, h)[::-1]                     # piece rows stored bottom first
        rows, cols = np.mgrid[y:y + h, x:x + w]
        order[start + idx] = rows * WIDTH + cols
    return order


_ORDER = _stored_order()


@dataclass(frozen=True)
class Screen:
    """One background screen: its kind (`KINDS`) and course n, the picture's size, layout and pixel format, and the
    file's bytes as stored."""
    kind: str
    n: int
    width: int
    height: int
    layout: str
    pixel: str
    data: bytes


def _name(kind, n) -> tuple[str, int, str]:
    """(kind, n as int, the file name), else DcError naming the bad field."""
    if not isinstance(kind, str) or kind not in KINDS:
        raise DcError(f"screen kind {kind!r} is not one of {KINDS}")
    n = _int(n, "course", 0, COURSE_COUNT - 1, f"screen {kind}", DcError)
    return kind, n, course_files(n, kind=kind)


def screen_names() -> list[tuple[str, int, str]]:
    """The 24 (kind, n, file name): for each course, LG_BGn, CHP_BGn, NAME_BGnn."""
    return [(kind, n, course_files(n, kind=kind)) for n in range(COURSE_COUNT) for kind in KINDS]


def decode_screen(blob, kind: str, n: int) -> Screen:
    """The screen `kind` of course n from its file's bytes. DcError for a kind outside `KINDS`, an n outside 0-7, a
    blob that is not bytes, or a file that is not 614,400 bytes (naming the file and the offset where it ends or
    runs past)."""
    kind, n, name = _name(kind, n)
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise DcError(f"{name}: {type(blob).__name__} is not bytes")
    b = bytes(blob)
    if len(b) < SIZE:
        raise DcError(f"{name}: the file ends at offset {len(b):#x}; a {WIDTH} x {HEIGHT} {PIXEL} screen is "
                      f"{SIZE:#x} bytes")
    if len(b) > SIZE:
        raise DcError(f"{name}: {len(b) - SIZE} bytes past the screen's end at offset {SIZE:#x}")
    return Screen(kind, n, WIDTH, HEIGHT, LAYOUT, PIXEL, b)


def _checked(s) -> tuple[str, bytes]:
    """(file name, data) of a screen the format can hold, else DcError naming the field."""
    if not isinstance(s, Screen):
        raise DcError(f"screen: {type(s).__name__} is not a Screen")
    _, _, name = _name(s.kind, s.n)
    if not (is_int(s.width) and is_int(s.height)) or (s.width, s.height) != (WIDTH, HEIGHT):
        raise DcError(f"{name}: size {s.width!r} x {s.height!r} is not {WIDTH} x {HEIGHT}")
    if s.layout != LAYOUT or s.pixel != PIXEL:
        raise DcError(f"{name}: layout {s.layout!r}, pixel {s.pixel!r}: a screen is {LAYOUT!r} {PIXEL!r}")
    if not isinstance(s.data, (bytes, bytearray)) or len(s.data) != SIZE:
        what = f"{len(s.data)} bytes" if isinstance(s.data, (bytes, bytearray)) else type(s.data).__name__
        raise DcError(f"{name}: data is {what}, a screen is {SIZE:#x} bytes")
    return name, bytes(s.data)


def encode_screen(s: Screen) -> bytes:
    """The file's bytes of a screen (checked: kind, n, size, layout, pixel format, 614,400 bytes of data)."""
    return _checked(s)[1]


def texels(s: Screen) -> np.ndarray:
    """The (480, 640) uint16 RGB565 texels of a screen, top row first."""
    _, data = _checked(s)
    out = np.empty(WIDTH * HEIGHT, dtype=np.uint16)
    out[_ORDER] = np.frombuffer(data, "<u2")
    return out.reshape(HEIGHT, WIDTH)


def to_rgba(s: Screen) -> np.ndarray:
    """The (480, 640, 4) uint8 RGBA of a screen, top row first (alpha 255)."""
    return texels_to_rgba(texels(s), s.pixel)


def screen_from_texels(pic, kind: str, n: int) -> Screen:
    """The screen `kind` of course n holding a (480, 640) picture of RGB565 texels, top row first (integers
    0-0xFFFF, not bool). DcError for any other picture, kind or n."""
    kind, n, name = _name(kind, n)
    try:
        a = np.asarray(pic)
    except (ValueError, TypeError) as exc:
        raise DcError(f"{name}: {type(pic).__name__} is not a picture of texels ({exc})") from None
    if a.shape != (HEIGHT, WIDTH) or a.dtype.kind not in "ui":
        raise DcError(f"{name}: {type(pic).__name__} of shape {a.shape} and dtype {a.dtype} is not a "
                      f"({HEIGHT}, {WIDTH}) array of integer texels")
    if int(a.min()) < 0 or int(a.max()) > 0xFFFF:
        raise DcError(f"{name}: texel values {int(a.min())}..{int(a.max())} are not 16-bit (0-0xFFFF)")
    stream = a.reshape(-1)[_ORDER].astype("<u2")
    return Screen(kind, n, WIDTH, HEIGHT, LAYOUT, PIXEL, stream.tobytes())


def verify_lossless_screens(disc) -> list[str]:
    """One line per screen file that does not come back byte for byte through `decode_screen` / `encode_screen` and
    through its picture (`texels` / `screen_from_texels`), or that does not decode ([] when all 24 are lossless)."""
    out = []
    for kind, n, name in screen_names():
        blob = disc.read(name)
        try:
            s = decode_screen(blob, kind, n)
            backs = (encode_screen(s), encode_screen(screen_from_texels(texels(s), kind, n)))
        except DcError as exc:
            out.append(f"{name}: {exc}")
            continue
        for how, back in zip(("encode", "picture"), backs):
            if back != blob:
                k = next((k for k, (a, c) in enumerate(zip(blob, back)) if a != c), min(len(blob), len(back)))
                out.append(f"{name}: {how}: first difference at offset {k:#x}")
    return out
