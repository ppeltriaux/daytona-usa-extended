"""Minimal ISO9660 reader for a raw 2352-byte-per-sector MODE1 data track.

Only what the Dreamcast Daytona USA disc needs: the primary volume descriptor,
directory records (recursed), file extents. No Joliet, no interleaving.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dtools.dc.errors import DcError

RAW_SECTOR = 2352
SECTOR = 2048
SYNC_HEADER = 16
PVD_SECTOR = 16


class IsoError(DcError):
    pass


@dataclass(frozen=True)
class Entry:
    path: str
    lba: int
    size: int


class Iso2352:
    def __init__(self, path: Path, lba_base: int = 0):
        """lba_base: the LBA held by the first sector of the file (45000 for a GD-ROM high-density track)."""
        self.path = Path(path)
        self.lba_base = lba_base
        self._f = open(self.path, "rb")
        try:
            pvd = self.sector(lba_base + PVD_SECTOR)
            if pvd[1:6] != b"CD001":
                raise IsoError(f"{self.path}: no ISO9660 volume descriptor at LBA {lba_base + PVD_SECTOR}")
            root = pvd[156:190]
            self._root_lba = struct.unpack_from("<I", root, 2)[0]
            self._root_size = struct.unpack_from("<I", root, 10)[0]
        except BaseException:
            self._f.close()
            raise
        self._entries: list[Entry] | None = None

    def close(self) -> None:
        """Close the image file (harmless a second time)."""
        self._f.close()

    def __enter__(self) -> Iso2352:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def sector(self, lba: int) -> bytes:
        """The 2048 user bytes of the frame holding this LBA (file index lba - lba_base)."""
        n = lba - self.lba_base
        if n < 0:
            raise IsoError(f"{self.path}: LBA {lba} is before the image base {self.lba_base}")
        self._f.seek(n * RAW_SECTOR + SYNC_HEADER)
        data = self._f.read(SECTOR)
        if len(data) != SECTOR:
            raise IsoError(f"{self.path}: LBA {lba} past end of image")
        return data

    def _read_extent(self, lba: int, size: int) -> bytes:
        count = (size + SECTOR - 1) // SECTOR
        return b"".join(self.sector(lba + i) for i in range(count))[:size]

    def _records(self, lba: int, size: int) -> Iterator[tuple[str, int, int, int]]:
        data = self._read_extent(lba, size)
        off = 0
        while off < len(data):
            length = data[off]
            if length == 0:                       # padding: skip to the next sector
                off = (off // SECTOR + 1) * SECTOR
                continue
            rec = data[off:off + length]
            extent = struct.unpack_from("<I", rec, 2)[0]
            fsize = struct.unpack_from("<I", rec, 10)[0]
            flags = rec[25]
            name_len = rec[32]
            name = rec[33:33 + name_len].decode("latin-1")
            yield name, extent, fsize, flags
            off += length

    def _walk(self, lba: int, size: int, prefix: str) -> Iterator[Entry]:
        for name, extent, fsize, flags in self._records(lba, size):
            if name in ("\x00", "\x01"):
                continue
            if flags & 2:
                yield from self._walk(extent, fsize, f"{prefix}{name}/")
            else:
                yield Entry(prefix + name.split(";")[0], extent, fsize)

    def entries(self) -> list[Entry]:
        if self._entries is None:
            self._entries = list(self._walk(self._root_lba, self._root_size, ""))
        return self._entries

    def read(self, path: str) -> bytes:
        for e in self.entries():
            if e.path == path:
                return self._read_extent(e.lba, e.size)
        raise IsoError(f"{self.path}: no file {path!r}")

    def extract_all(self, out: Path, exclude: tuple[str, ...] = ("ADX", "DP")) -> int:
        """Write every entry under `out`, except those whose first path component starts with one of `exclude`
        (by default the `ADX*` audio files and the `DP*` browser directories). Returns the number written."""
        out = Path(out)
        written = 0
        for e in self.entries():
            if e.path.split("/", 1)[0].startswith(exclude):
                continue
            target = out / e.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._read_extent(e.lba, e.size))
            written += 1
        return written
