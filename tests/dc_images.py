"""Synthetic Dreamcast / ISO9660 images shared by the dtools.dc tests (no game data)."""

import struct

SECTOR = 2048
RAW = 2352


def _record(name: str, extent: int, size: int, flags: int = 0) -> bytes:
    ident = name.encode("ascii") if name not in ("\x00", "\x01") else name.encode("latin-1")
    length = 33 + len(ident)
    if length % 2:
        length += 1
    rec = bytearray(length)
    rec[0] = length
    rec[2:6] = struct.pack("<I", extent)
    rec[6:10] = struct.pack(">I", extent)
    rec[10:14] = struct.pack("<I", size)
    rec[14:18] = struct.pack(">I", size)
    rec[25] = flags
    rec[32] = len(ident)
    rec[33:33 + len(ident)] = ident
    return bytes(rec)


def _wrap(sectors: list[bytes]) -> bytes:
    out = bytearray()
    for i, data in enumerate(sectors):
        assert len(data) == SECTOR
        out += b"\x00" + b"\xff" * 10 + b"\x00" + bytes([0, 2, i & 0xFF, 1])  # 12-byte sync + 4-byte header
        out += data
        out += b"\x00" * 288
    return bytes(out)


def _pack_dir(records: list[bytes]) -> bytes:
    out = bytearray()
    for rec in records:
        if (len(out) % SECTOR) + len(rec) > SECTOR:        # records never span a sector: pad to the next
            out += b"\x00" * (SECTOR - len(out) % SECTOR)
        out += rec
    out += b"\x00" * (-len(out) % SECTOR)
    return bytes(out)


def make_image(files: dict[str, bytes], base: int = 0, ip: bytes | None = None) -> bytes:
    """ISO9660 image: PVD at index 16, root dir at 18, then each subdirectory, then the files.

    A name with '/' lives in a subdirectory ('DP1/X.BIN'). Extents are written as base + index, as on a GD-ROM
    high-density track. ip: optional 2048 bytes for sector 0 (the IP header).
    """
    dirs: dict[str, list[tuple[str, bool]]] = {"": []}      # directory path -> [(name, is_dir)]
    for path in files:
        parts = path.split("/")
        for i in range(len(parts) - 1):
            parent, sub = "/".join(parts[:i]), "/".join(parts[:i + 1])
            if sub not in dirs:
                dirs[sub] = []
                dirs[parent].append((parts[i], True))
        dirs["/".join(parts[:-1])].append((parts[-1], False))

    def build(extents: dict[str, int], sizes: dict[str, int]) -> dict[str, bytes]:
        built = {}
        for d, entries in dirs.items():
            parent = d.rsplit("/", 1)[0] if "/" in d else ""
            records = [_record("\x00", extents.get(d, 0), sizes.get(d, SECTOR), 2),
                       _record("\x01", extents.get(parent, 0), sizes.get(parent, SECTOR), 2)]
            for name, is_dir in entries:
                key = f"{d}/{name}" if d else name
                if is_dir:
                    records.append(_record(name, extents.get(key, 0), sizes.get(key, SECTOR), 2))
                else:
                    records.append(_record(f"{name};1", extents.get(key, 0), len(files[key])))
            built[d] = _pack_dir(records)
        return built

    dir_sizes = {d: len(b) for d, b in build({}, {}).items()}
    extents: dict[str, int] = {}
    lba = 18
    for d in dirs:
        extents[d] = base + lba
        lba += dir_sizes[d] // SECTOR
    for path, data in files.items():
        extents[path] = base + lba
        lba += max(1, (len(data) + SECTOR - 1) // SECTOR)
    built = build(extents, dir_sizes)

    pvd = bytearray(SECTOR)
    pvd[0] = 1
    pvd[1:6] = b"CD001"
    root = _record("\x00", extents[""], dir_sizes[""], 2)
    pvd[156:156 + len(root)] = root

    sectors = [ip or bytes(SECTOR)] + [bytes(SECTOR)] * 15 + [bytes(pvd), bytes(SECTOR)]  # 0..15 blank, 16 PVD, 17 end
    for d in dirs:
        sectors += [built[d][i:i + SECTOR] for i in range(0, len(built[d]), SECTOR)]
    for data in files.values():
        padded = data + b"\x00" * (-len(data) % SECTOR) or bytes(SECTOR)
        sectors += [padded[i:i + SECTOR] for i in range(0, len(padded), SECTOR)]
    return _wrap(sectors)
