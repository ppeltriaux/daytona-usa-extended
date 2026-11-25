"""Load the MAME `daytona` (Model 2) ROM set into contiguous memory images.

Layout follows MAME src/mame/sega/model2.cpp (ROM_START( daytona ) and the MODEL2_CPU_BOARD macro), BSD-3-Clause,
copyright the MAME team. Every ROM_LOAD32_WORD pair is interleaved here so callers see flat little-endian byte
strings addressed the way the i960 and the geometry engine see them.

A **custom** set (`RomSet.custom`, `as_custom`) is the stock set plus five files the custom MAME build loads
(tools/mame-daytona-rom-space.patch): the regions grown as far as the game's address decoding reaches
(REGION_SIZES_CUSTOM, docs/spike/2026-11-sixteen.md "Room"), the new areas (CUSTOM_FREE) 0xff until written. The 36
stock files are unchanged in it; stock MAME ignores the five.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass, replace
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from dtools.arcade.errors import DecodeError
from dtools.arcade.tablebase import DATA_BASE, DATA_SIZE, PROGRAM_WINDOW

PROGRAM_PAIR = ("epr-16722a.12", "epr-16723a.13")                    # i960 program, 256 KB
DATA_PAIRS = ((("mpr-16528.10", "mpr-16529.11"), 0), (("mpr-16808.8", "mpr-16809.9"), 0x400000),
              (("epr-16724a.6", "epr-16725a.7"), 0x800000))          # (pair, image offset); last one is 1 MB, copied
COPRO_PAIR = ("mpr-16537.ic28", "mpr-16536.ic29")                    # 4 MB
POLY_PAIRS = (("mpr-16523.ic16", "mpr-16518.ic20"), ("mpr-16524.ic17", "mpr-16519.ic21"),
              ("mpr-16525.ic18", "mpr-16520.ic22"), ("mpr-16772.ic19", "mpr-16771.ic23"))   # 4 MB each, 16 MB
TEX_PAIRS = ((("mpr-16522.25", "mpr-16521.24"), 0), (("mpr-16770.27", "mpr-16769.26"), 0x800000))

OTHER_FILES = ("epr-16726.bin", "epr-16720.7", "epr-16721.8", "mpr-16491.32", "mpr-16492.33", "mpr-16493.4",
               "mpr-16494.5", "epr-16488a.ic12", "opr-14742a.45", "opr-14743a.46", "opr-14744.58", "opr-14745.59",
               "opr-14746.62", "opr-14747.63")

# CRC32 per file, from MAME 0.289 model2.cpp (daytona + MODEL2_CPU_BOARD).
CRCS: dict[str, int] = {
    "epr-16722a.12": 0x48b94318, "epr-16723a.13": 0x8af8b32d,
    "mpr-16528.10": 0x9ce591f6, "mpr-16529.11": 0xf7095eaf,
    "mpr-16808.8": 0x44f1f5a0, "mpr-16809.9": 0x37a2dd12,
    "epr-16724a.6": 0x469f10fd, "epr-16725a.7": 0xba0df8db,
    "mpr-16537.ic28": 0x36b7c35a, "mpr-16536.ic29": 0x6d6afed9,
    "mpr-16523.ic16": 0x2f484d42, "mpr-16518.ic20": 0xdf683bf7,
    "mpr-16524.ic17": 0x34658bd7, "mpr-16519.ic21": 0xfacd1c81,
    "mpr-16525.ic18": 0xfb517521, "mpr-16520.ic22": 0xd66bd9bd,
    "mpr-16772.ic19": 0x770ed912, "mpr-16771.ic23": 0xa2205124,
    "mpr-16522.25": 0x55d39a57, "mpr-16521.24": 0xaf1934fb,
    "mpr-16770.27": 0xf9fa7bfb, "mpr-16769.26": 0xe57429e9,
    "epr-16726.bin": 0xc179b8c7, "epr-16720.7": 0x8e73cffd,
    "epr-16721.8": 0x1bb3b7b7, "mpr-16491.32": 0x89920903,
    "mpr-16492.33": 0x459e701b, "mpr-16493.4": 0x9990db15,
    "mpr-16494.5": 0x600e1d6c, "epr-16488a.ic12": 0x546c5d1a,
    "opr-14742a.45": 0x90c6b117, "opr-14743a.46": 0xae7f446b,
    "opr-14744.58": 0x730ea9e0, "opr-14745.59": 0x4c934d96,
    "opr-14746.62": 0x2a266cbd, "opr-14747.63": 0xa4ad5e19,
}

PROGRAM_SIZE = 256 * 1024
COPRO_SIZE = 4 << 20
POLY_SIZE = 16 << 20
TEX_SIZE = 16 << 20
DATA_COPY_BASE, DATA_COPY_LEN, DATA_COPY_COUNT = 0x800000, 0x100000, 7   # ROM_COPY to 0x900000 .. 0xf00000
DATA_COPY_END = DATA_COPY_BASE + (DATA_COPY_COUNT + 1) * DATA_COPY_LEN    # 0x1000000: no copies past it

REGIONS = ("program", "data", "copro_data", "polygons", "textures")
REGION_SIZES = {"program": PROGRAM_SIZE, "data": DATA_SIZE, "copro_data": COPRO_SIZE, "polygons": POLY_SIZE,
                "textures": TEX_SIZE}
"""The stock images' sizes (copro_data: the 4 MB the ROM pair fills; MAME's region is 8 MB, the rest zero)."""
REGION_SIZES_CUSTOM = {"program": PROGRAM_SIZE, "data": 0x2000000, "copro_data": 0x2000000, "polygons": 0x2000000,
                       "textures": TEX_SIZE}
"""The custom build's images, each the most the game's address decoding reaches (docs/spike/2026-11-sixteen.md,
"Room"): data the i960's whole data ROM window 0x02000000-0x03ffffff; copro_data the TGP's 23-bit word address under
bank bit 0x800000 (MAME masks it by the region: 0x7fffff words); polygons the geometrizer's 23-bit word index under
`oba` bit 0x800000 (mask 0x7fffff); textures unchanged, already the 23-bit word address under bit 0x800000 (texture
RAM), its two 4 MB zero gaps filled instead."""
CUSTOM_FILES = {"dx-data.bin": ("data", 0x1000000, 0x1000000),
                "dx-copro.bin": ("copro_data", 0x400000, 0x1C00000),
                "dx-poly.bin": ("polygons", 0x1000000, 0x1000000),
                "dx-tex1.bin": ("textures", 0x400000, 0x400000),
                "dx-tex2.bin": ("textures", 0xC00000, 0x400000)}
"""name -> (region, image offset, length) of the custom build's extra files: each a plain `ROM_LOAD` (no interleave)
of exactly those image bytes, declared NO_DUMP (loaded when present, never checked; absent, the area stays zero)."""
CUSTOM_AREAS = {r: tuple((lo, n) for reg, lo, n in CUSTOM_FILES.values() if reg == r)
                for r in ("data", "copro_data", "polygons", "textures")}
"""region -> ((image offset, length), ...): the custom set's new areas (its CUSTOM_FILES), 0xff in `as_custom`."""
RESERVED = {
    "program": ((0x1E780, 0x1080, "the sixteen-entry select's code (select.CODE_FREE)"),
                (0x1F800, 0x600, "widen_ids' per-id tables and code (course.WIDE_HOME, WIDE_SIZE)"),
                (0x1FE00, 0x200, "grow_rankings' sixteen-id ranking defaults and time bases (course.RANK_HOME, "
                                 "RANK_SIZE)"),
                (0x3CE98, 0x400, "the shared empty actor table (build.EMPTY_ACTORS: 256 zero words at "
                                 "build.EMPTY_AT, every course without actors names it)"),
                (0x3FF00, 0x100, "the far view's hook (course.FAR_CODE, FAR_SIZE)")),
    "data": ((0x1F00000, 0x20000, "the sixteen-entry select's data (select.DATA_AT, DATA_SPAN)"),
             (0x1001580, 0x260, "the new ids' read-only SRAM stand-in (course.SRAM_SINK, SRAM_SINK_SIZE)"),
             (0x1000000, 0x1580, "grow_ids' sixteen-entry tables and records (course.ID_HOME up to SRAM_SINK)")),
}
"""region -> ((image offset, length, owner), ...): space a patch owns whatever its bytes (grow_ids' home has 0xff
holes: the unused record slots 0-2). Left out of CUSTOM_FREE; tgp.write_free and the converter's program writes
refuse it (`reserved`), and the allocators treat it as busy (golden.owned_spans, which adds the space the set's own
tables and blocks take). The literals are course's and select's addresses (romset cannot import them): the tests tie
them together."""
RESERVED_RAM = ((0x00501B00, 0x40, "select.SELECT_RAM: the course vote's sixteen slots and the select's first"),
                (0x00501B40, 0x14, "course.WIDE_RAM: widen_ids' shadows of the course id's per-id values"),
                (0x00501B54, 0x10, "course.VIEW_RAM: view_ids' shadow of the course id's view-test pointers"))
"""i960 work RAM a patch owns: (address, length, owner)."""


def _minus(runs, holes):
    out = []
    for lo, n in runs:
        pieces = [(lo, lo + n)]
        for h, hn, _ in holes:
            pieces = [p for a, z in pieces for p in ((a, min(z, h)), (max(a, h + hn), z)) if p[0] < p[1]]
        out += [(a, z - a) for a, z in pieces]
    return tuple(out)


CUSTOM_FREE = {r: _minus(runs, RESERVED.get(r, ())) for r, runs in CUSTOM_AREAS.items()}
"""region -> ((image offset, length), ...): the new free runs of a custom set (CUSTOM_AREAS less RESERVED), 0xff in
`as_custom`. No copies: the data ROM's 1 MB copies end at DATA_COPY_END and copro_data's mirrored halves at
COPRO_SIZE."""


def reserved(region: str, offset: int, n: int) -> str | None:
    """The owner of RESERVED space that `region` bytes offset..offset+n-1 overlap, else None. RomSetError for a region
    that is not a str or an offset / length that is not an int (bool refused)."""
    if not isinstance(region, str):
        raise RomSetError(f"region {region!r:.60} is not a str")
    for v, name in ((offset, "offset"), (n, "length")):
        if not isinstance(v, int) or isinstance(v, bool):
            raise RomSetError(f"{name} {v!r:.60} is not an int")
    for lo, size, owner in RESERVED.get(region, ()):
        if offset < lo + size and lo < offset + n:
            return owner
    return None
_FREE = 0xFF

I960_PROGRAM = (0x00000000, 0x00040000)
I960_WINDOW = PROGRAM_WINDOW                 # program ROM seen again from offset 0x20000 (tablebase)
I960_DATA = (DATA_BASE, DATA_BASE + DATA_SIZE)   # the data image's first 16 MB


class RomSetError(DecodeError):
    pass


def _words32(image: np.ndarray, base: int, pair: tuple[str, str], files: dict[str, bytes]) -> None:
    """ROM a supplies bytes 0-1 and ROM b bytes 2-3 of each LE32 word (ROM_LOAD32_WORD at 0 and 2)."""
    a, b = files[pair[0]], files[pair[1]]
    if len(a) != len(b):
        raise RomSetError(f"{pair[1]} is {len(b)} bytes, {pair[0]} is {len(a)}")
    span = 2 * len(a)
    if base + span > len(image):
        raise RomSetError(f"{pair[0]}/{pair[1]} ({len(a)} bytes each) do not fit at {base:#x}")
    view = image[base:base + span]
    aa, bb = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    view[0::4], view[1::4] = aa[0::2], aa[1::2]
    view[2::4], view[3::4] = bb[0::2], bb[1::2]


def _crc32(blob: bytes) -> int:
    import zlib
    return zlib.crc32(blob)


def _apply(buf: bytearray, region: str, offset: int, blob: bytes) -> None:
    """Write `blob` at `offset` of `buf` (the `region` image) and into every copy of a copied region; the one
    implementation behind `with_patch` and `with_patches`. The bounds are checked by the caller. A custom image's new
    areas (from DATA_COPY_END, from COPRO_SIZE) have no copies; a patch across either boundary is refused."""
    end = offset + len(blob)
    if region == "data" and offset < DATA_COPY_END < end:
        raise RomSetError(f"data patch {offset:#x}..{end:#x} straddles {DATA_COPY_END:#x}, where the copied 1 MB "
                          f"blocks end")
    if region == "copro_data" and offset < COPRO_SIZE < end:
        raise RomSetError(f"copro_data patch {offset:#x}..{end:#x} straddles {COPRO_SIZE:#x}, where the two "
                          f"identical halves end")
    buf[offset:end] = blob
    if region == "data" and DATA_COPY_BASE < end <= DATA_COPY_END:
        if offset < DATA_COPY_BASE:
            raise RomSetError("data patch straddles 0x800000, where the copied 1 MB block starts")
        if (offset - DATA_COPY_BASE) // DATA_COPY_LEN != (end - 1 - DATA_COPY_BASE) // DATA_COPY_LEN:
            raise RomSetError("data patch spans two 1 MB copies")
        rel = (offset - DATA_COPY_BASE) % DATA_COPY_LEN
        for k in range(DATA_COPY_COUNT + 1):
            lo = DATA_COPY_BASE + k * DATA_COPY_LEN + rel
            buf[lo:lo + len(blob)] = blob
    elif region == "copro_data" and end <= COPRO_SIZE:
        half = COPRO_SIZE // 2
        if offset < half < end:
            raise RomSetError("copro_data patch spans the two identical halves")
        rel = offset % half
        for lo in (rel, half + rel):
            buf[lo:lo + len(blob)] = blob


def _canonical(region: str, offset: int) -> int:
    """The offset of the bytes `offset` aliases in a copied region: a data offset in 0x800000-0xffffff is the same
    place of the 1 MB block at 0x800000 (`_apply` writes every copy), a copro_data offset below 4 MB the same place of
    the first 2 MB half; every other offset (a custom image's new areas among them) is itself."""
    if region == "data" and DATA_COPY_BASE <= offset < DATA_COPY_END:
        return DATA_COPY_BASE + (offset - DATA_COPY_BASE) % DATA_COPY_LEN
    if region == "copro_data" and 0 <= offset < COPRO_SIZE:
        return offset % (COPRO_SIZE // 2)
    return offset


@dataclass(frozen=True)
class RomSet:
    """The five images and the untouched files (`extra`). `custom`: the images are the custom build's
    (REGION_SIZES_CUSTOM, the five CUSTOM_FILES in them), else the stock ones (REGION_SIZES)."""
    program: bytes
    data: bytes
    copro_data: bytes
    polygons: bytes
    textures: bytes
    extra: dict[str, bytes]
    custom: bool = False

    def __post_init__(self):
        if type(self.custom) is not bool:
            raise RomSetError(f"custom must be a bool, not {self.custom!r:.40}")

    @classmethod
    def from_files(cls, files: dict[str, bytes], verify: bool = True) -> "RomSet":
        """The set of the 36 stock files (CRCS; checked when `verify`), custom when `files` holds any of the five
        CUSTOM_FILES: then all five, each exactly its length (not checked against a CRC: their content is ours).
        RomSetError naming every missing or bad file."""
        if not isinstance(files, Mapping):
            raise RomSetError(f"files must be a mapping of name to bytes, not {type(files).__name__}")
        problems = [f"{n}: missing" for n in CRCS if n not in files]
        custom = any(n in files for n in CUSTOM_FILES)
        for n, (_, _, size) in CUSTOM_FILES.items() if custom else ():
            if n not in files:
                problems.append(f"{n}: missing (a custom set needs all of {', '.join(CUSTOM_FILES)})")
            elif not isinstance(files[n], (bytes, bytearray)):
                problems.append(f"{n}: not bytes")
            elif len(files[n]) != size:
                problems.append(f"{n}: {len(files[n])} bytes, expected {size}")
        if verify:
            for n, want in CRCS.items():
                if n in files and (got := _crc32(files[n])) != want:
                    problems.append(f"{n}: crc {got:08x}, expected {want:08x}")
        if problems:
            raise RomSetError("not a good MAME 0.289 `daytona` set: " + "; ".join(problems))

        def image(size: int) -> np.ndarray:
            return np.zeros(size, np.uint8)

        try:
            program = image(PROGRAM_SIZE)
            _words32(program, 0, PROGRAM_PAIR, files)
            data = image(DATA_SIZE)
            for pair, off in DATA_PAIRS:
                _words32(data, off, pair, files)
            block = data[DATA_COPY_BASE:DATA_COPY_BASE + DATA_COPY_LEN].copy()
            for k in range(1, DATA_COPY_COUNT + 1):
                data[DATA_COPY_BASE + k * DATA_COPY_LEN:DATA_COPY_BASE + (k + 1) * DATA_COPY_LEN] = block
            copro = image(COPRO_SIZE)
            _words32(copro, 0, COPRO_PAIR, files)
            polys = image(POLY_SIZE)
            for i, pair in enumerate(POLY_PAIRS):
                _words32(polys, i * 0x400000, pair, files)
            tex = image(TEX_SIZE)
            for pair, off in TEX_PAIRS:
                _words32(tex, off, pair, files)
        except RomSetError as e:
            raise RomSetError(f"bad ROM sizes: {e}") from None
        rs = cls(program=program.tobytes(), data=data.tobytes(), copro_data=copro.tobytes(),
                 polygons=polys.tobytes(), textures=tex.tobytes(), extra={n: files[n] for n in OTHER_FILES})
        return rs._grown({n: bytes(files[n]) for n in CUSTOM_FILES}) if custom else rs

    def _grown(self, blobs: dict[str, bytes]) -> "RomSet":
        """The custom set with each CUSTOM_FILES area holding `blobs[name]` (stock images grown to
        REGION_SIZES_CUSTOM)."""
        bufs = {}
        for r in REGIONS:
            buf = bytearray(REGION_SIZES_CUSTOM[r])
            stock = getattr(self, r)
            buf[:len(stock)] = stock
            bufs[r] = buf
        for name, (r, lo, n) in CUSTOM_FILES.items():
            bufs[r][lo:lo + n] = blobs[name]
        return replace(self, custom=True, **{r: bytes(b) for r, b in bufs.items()})

    def as_custom(self) -> "RomSet":
        """The custom set of this one: every stock byte kept, the regions grown to REGION_SIZES_CUSTOM and each new
        area (CUSTOM_FREE, the textures' two zero-filled 4 MB gaps included) 0xff. A custom set is returned as is.
        RomSetError for a stock set whose images are not REGION_SIZES."""
        if self.custom:
            return self
        for r, n in REGION_SIZES.items():
            if len(getattr(self, r)) != n:
                raise RomSetError(f"{r} is {len(getattr(self, r))} bytes, expected {n}")
        return self._grown({name: bytes([_FREE]) * n for name, (_, _, n) in CUSTOM_FILES.items()})

    @classmethod
    def from_zip(cls, path: Path, verify: bool = True) -> "RomSet":
        """Load `daytona.zip`, stock or custom (the five CUSTOM_FILES at its root). Merged sets keep clone files in
        subdirectories; only root-level members count."""
        with zipfile.ZipFile(path) as z:
            roots = {i.filename for i in z.infolist() if "/" not in i.filename}
            files = {n: z.read(n) for n in (*CRCS, *CUSTOM_FILES) if n in roots}
        return cls.from_files(files, verify=verify)

    def split_files(self) -> dict[str, bytes]:
        """Inverse of `from_files`: every file named in `CRCS` (36), re-split from the five images, and on a custom
        set the five CUSTOM_FILES (their image bytes as they are).

        Regions whose image is a copy are collapsed: the data image's 1 MB pair at 0x800000 is loaded seven more
        times (0x900000..0xf00000) and `copro_data` has identical halves; the file bytes come from copy 0 / the
        first half, and `RomSetError` is raised when a copy differs (a patch there must go into every copy, which
        `with_patch` does). For an unmodified set every file's CRC equals `CRCS`. RomSetError for an image whose
        size is not REGION_SIZES (REGION_SIZES_CUSTOM on a custom set)."""
        sizes = REGION_SIZES_CUSTOM if self.custom else REGION_SIZES
        for name in REGIONS:
            img, size = getattr(self, name), sizes[name]
            if len(img) != size:
                raise RomSetError(f"{name} is {len(img)} bytes, expected {size}")
        missing = [n for n in OTHER_FILES if n not in self.extra]
        if missing:
            raise RomSetError("extra is missing " + ", ".join(missing))
        block = self.data[DATA_COPY_BASE:DATA_COPY_BASE + DATA_COPY_LEN]
        for k in range(1, DATA_COPY_COUNT + 1):
            lo = DATA_COPY_BASE + k * DATA_COPY_LEN
            if self.data[lo:lo + DATA_COPY_LEN] != block:
                raise RomSetError(f"data copy {k} at {lo:#x} differs from the 1 MB block at {DATA_COPY_BASE:#x}")
        if self.copro_data[:COPRO_SIZE // 2] != self.copro_data[COPRO_SIZE // 2:COPRO_SIZE]:
            raise RomSetError("copro_data halves differ (the ROM pair holds the same 2 MB twice)")
        out: dict[str, bytes] = {}

        def lanes(image: bytes, base: int, span: int, pair: tuple[str, str]) -> None:
            v = np.frombuffer(image, np.uint8)[base:base + span]
            a, b = np.empty(span // 2, np.uint8), np.empty(span // 2, np.uint8)
            a[0::2], a[1::2], b[0::2], b[1::2] = v[0::4], v[1::4], v[2::4], v[3::4]
            out[pair[0]], out[pair[1]] = a.tobytes(), b.tobytes()

        lanes(self.program, 0, PROGRAM_SIZE, PROGRAM_PAIR)
        for (pair, off), span in zip(DATA_PAIRS, (0x400000, 0x400000, DATA_COPY_LEN)):
            lanes(self.data, off, span, pair)
        lanes(self.copro_data, 0, COPRO_SIZE, COPRO_PAIR)
        for i, pair in enumerate(POLY_PAIRS):
            lanes(self.polygons, i * 0x400000, 0x400000, pair)
        for pair, off in TEX_PAIRS:
            lanes(self.textures, off, 0x400000, pair)
        out.update({n: self.extra[n] for n in OTHER_FILES})
        if self.custom:
            for name, (r, lo, n) in CUSTOM_FILES.items():
                out[name] = getattr(self, r)[lo:lo + n]
        return out

    def write_zip(self, path: Path) -> Path:
        """Write a `daytona.zip` of `split_files()` plus the `extra` members (all of them, byte for byte); a custom
        set's five files included."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        files = self.split_files()
        files.update(self.extra)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, blob in files.items():
                z.writestr(name, blob)
        return path

    def with_patch(self, region: str, offset: int, blob: bytes) -> "RomSet":
        """A copy with `blob` written at `offset` of `region` (program, data, copro_data, polygons, textures).
        `RomSetError` when the patch runs past the region. Copied regions are patched in every copy: a data patch
        in 0x800000..0xffffff is written into all eight 1 MB blocks (the pair loaded seven more times) and a
        copro_data patch into both 2 MB halves, so `split_files` still sees identical copies; the patch must then
        not span the copy boundary of the region (refused)."""
        return self.with_patches({(region, offset): blob})

    def with_patches(self, entries: Mapping[tuple[str, int], bytes]) -> "RomSet":
        """Equal to folding `with_patch` over `entries` in key order, with one copy per region. Two entries of a region
        may overlap where their bytes agree (golden.tables emits nested texel blocks, e.g. Expert's data 0x440200 + 128
        and 0x440220 + 32); `RomSetError` where they differ, the copies of a copied region counting as the same bytes
        (a data entry in 0x800000-0xffffff overlaps one at the same place of another 1 MB copy, a copro_data entry one
        at the same place of the other 2 MB half), and on everything `with_patch` refuses."""
        bufs: dict[str, bytearray] = {}
        reach: dict[str, tuple[int, int]] = {}      # region -> (canonical end of the furthest entry, its offset)
        for (region, offset), blob in sorted(entries.items(), key=lambda kv: (kv[0][0], _canonical(*kv[0]), kv[0][1])):
            if region not in REGIONS:
                raise RomSetError(f"unknown region {region!r}")
            end = offset + len(blob)
            size = len(getattr(self, region))
            if offset < 0 or end > size:
                raise RomSetError(f"patch {offset:#x}..{end:#x} is outside {region} ({size:#x} bytes)")
            buf = bufs.get(region)
            if buf is None:
                buf = bufs[region] = bytearray(getattr(self, region))
            at = _canonical(region, offset)
            if len(blob) and region in reach and at < reach[region][0]:
                # every byte from `at` to the reach is written (by the furthest-reaching entry, which starts at or
                # before `at`), every byte past it is not: the overlap is exactly this span
                n = min(reach[region][0], at + len(blob)) - at
                if buf[at:at + n] != blob[:n]:
                    k = next(k for k in range(n) if buf[at + k] != blob[k])
                    raise RomSetError(f"patches at {reach[region][1]:#x} and {offset:#x} of {region} overlap and "
                                      f"differ at {offset + k:#x} ({buf[at + k]:#04x} against {blob[k]:#04x})")
            _apply(buf, region, offset, blob)
            if len(blob) and (region not in reach or at + len(blob) > reach[region][0]):
                reach[region] = (at + len(blob), offset)
        return replace(self, **{r: bytes(b) for r, b in bufs.items()})

    def read_i960(self, addr: int, n: int) -> bytes:
        """`n` bytes at i960 address `addr`: the program ROM, its window, the data ROM (its first 16 MB,
        0x02000000-0x02ffffff; on a custom set the whole 32 MB window to 0x03ffffff). RomSetError elsewhere."""
        data = (DATA_BASE, DATA_BASE + len(self.data))
        for (lo, hi), region, base in ((I960_PROGRAM, self.program, 0), (I960_WINDOW, self.program, 0x20000),
                                       (data, self.data, 0)):
            if lo <= addr and addr + n <= hi:
                off = base + addr - lo
                return region[off:off + n]
        raise RomSetError(f"i960 address {addr:#x} (+{n}) is not in a ROM region")

    def poly_mask(self) -> int:
        """The polygon ROM word-index mask, MAME's `polygon_rom_mask` (region bytes / 4 - 1): 0x3fffff stock,
        0x7fffff custom."""
        return len(self.polygons) // 4 - 1

    def poly_word(self, index: int) -> int:
        return struct.unpack_from("<I", self.polygons, (index & self.poly_mask()) * 4)[0]

    def poly_float(self, index: int) -> float:
        return struct.unpack_from("<f", self.polygons, (index & self.poly_mask()) * 4)[0]

    def dump_regions(self, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, blob in (("program.bin", self.program), ("data.bin", self.data),
                           ("copro_data.bin", self.copro_data), ("polygons.bin", self.polygons),
                           ("textures.bin", self.textures)):
            p = out_dir / name
            p.write_bytes(blob)
            written.append(p)
        return written
