"""The Dreamcast in MAME's `dc` driver: the BIOS set, a disc image MAME accepts, and the run.

MAME's `dc` (Dreamcast, USA, NTSC; driver status "preliminary") boots BIOS 101d from `mpr-21931.ic501` and reads the
flash from `dcus_ntsc.bin`. The user's dump is `dc_boot.bin` and `dc_flash.bin` (ROMS/dc_bios.zip); the boot ROM must
be 101d (CRC32 0x89f2b1a1). The flash is a user dump and differs from MAME's (0x4136c25b): MAME prints "WRONG
CHECKSUMS" and "the machine might not run correctly", then runs (measured 2026-11-08, MAME 0.289).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from dtools.dc import disc as dcdisc
from dtools.dc.errors import DcError
from dtools.mame import runner

BOOT_CRC = 0x89F2B1A1                       # mpr-21931.ic501, BIOS 101d
BIOS_FILES = {"dc_boot.bin": "mpr-21931.ic501", "dc_flash.bin": "dcus_ntsc.bin"}   # the dump's name -> MAME's


def _read_bios(src: Path, name: str) -> bytes:
    if src.is_dir():
        try:
            return (src / name).read_bytes()
        except OSError as exc:
            raise DcError(f"{src}: no readable {name}: {exc}") from exc
    try:
        with zipfile.ZipFile(src) as z:
            return z.read(name)
    except KeyError as exc:
        raise DcError(f"{src}: the zip holds no {name}") from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise DcError(f"{src}: not a zip or a directory holding {name}: {exc}") from exc


def bios_set(src: Path, out_dir: Path) -> Path:
    """Write `out_dir/dc.zip`, MAME's `dc` set, from `src` (a zip or a directory holding `dc_boot.bin` and
    `dc_flash.bin`), the files under MAME's names. DcError when a file is missing or the boot ROM is not 101d (its
    CRC32 named); the flash's CRC is not checked (a user dump). A filesystem failure (out_dir not creatable, dc.zip
    not writable) is a DcError naming the path, and leaves no `dc.zip.partial`."""
    src, out_dir = Path(src), Path(out_dir)
    data = {name: _read_bios(src, name) for name in BIOS_FILES}
    crc = zlib.crc32(data["dc_boot.bin"])
    if crc != BOOT_CRC:
        raise DcError(f"{src}: dc_boot.bin has CRC32 {crc:08x}, not {BOOT_CRC:08x} (BIOS 101d, MAME's "
                      f"{BIOS_FILES['dc_boot.bin']})")
    out = out_dir / "dc.zip"
    tmp = out.with_name(out.name + ".partial")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for name, mame_name in BIOS_FILES.items():
                z.writestr(mame_name, data[name])
        os.replace(tmp, out)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise DcError(f"{out}: cannot write the BIOS set: {exc}") from exc
    return out


def _extract_member(archive: Path, name: str, dest: Path) -> None:
    """Extract the archive member whose file name is `name` to `dest/name` (through a `.partial` directory, so an
    interrupted extraction leaves no short file under the name). DcError when the archive holds no such member, or
    more than one, or tar fails, or the file cannot be put in place (the path named)."""
    listing = dcdisc._tar("-tf", str(archive))
    if listing.returncode != 0:
        raise DcError(f"{archive}: cannot list archive: {listing.stderr.strip()}")
    members = [m for m in listing.stdout.splitlines() if PurePosixPath(m).name == name]
    if len(members) != 1:
        raise DcError(f"{archive}: {len(members)} members named {name!r}, expected one")
    partial = dest / (name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        partial.mkdir(parents=True)
        done = dcdisc._tar("-xf", str(archive), "-C", str(partial), dcdisc._literal(members[0]))
        if done.returncode != 0:
            raise DcError(f"{archive}: extraction of {members[0]!r} failed: {done.stderr.strip()}")
        os.replace(partial / members[0], dest / name)
    except OSError as exc:
        raise DcError(f"{dest / name}: cannot put {members[0]!r} from {archive} in place: {exc}") from exc
    finally:
        shutil.rmtree(partial, ignore_errors=True)


def disc_image(disc: dcdisc.DcDisc, out_dir: Path) -> Path:
    """A cue sheet MAME's `dc` accepts for `-cdrom` (MAME 0.289 reads the Redump GD-ROM cue with its
    SINGLE-DENSITY / HIGH-DENSITY AREA remarks as it is): `out_dir/<the disc's cue name>`, a copy of the cue, beside a
    symbolic link to each track file found beside the disc's cue. MAME resolves a cue's FILE names in the cue's own
    directory, hence the links rather than paths. A track missing there (a disc opened from the `.7z` has only the
    cue and track 3) is extracted from `disc.archive` into `out_dir` once (tracks 1 and 2 are 2.3 MB together);
    DcError naming it when there is no archive to take it from, or when a FILE name has a directory part (the cue
    is read from a dump: such a name would reach outside `out_dir`). `out_dir` may be the cue's own directory. A
    filesystem failure (out_dir not creatable, a link or the cue copy not writable) is a DcError naming the path."""
    cue = Path(disc.cue)
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DcError(f"{out_dir}: cannot make the image directory: {exc}") from exc
    for _, (name, _mode) in sorted(dcdisc._cue_tracks(cue).items()):
        if PurePosixPath(name).name != name or Path(name).name != name or name in (".", ".."):
            raise DcError(f"{cue}: track file {name!r} has a directory part; expected a name beside the cue")
        target, here = cue.parent / name, out_dir / name
        if target.is_file():
            if (here.is_symlink() or here.exists()) and here.resolve() == target.resolve():
                continue
            try:
                if here.is_symlink() or here.exists():
                    here.unlink()
                here.symlink_to(target.resolve())
            except OSError as exc:
                raise DcError(f"{here}: cannot link it to {target}: {exc}") from exc
        elif here.is_file() and not here.is_symlink():
            continue
        elif disc.archive is not None:
            _extract_member(Path(disc.archive), name, out_dir)
        else:
            raise DcError(f"{cue}: track file {name!r} is not beside the cue (and no archive to extract it from)")
    image = out_dir / cue.name
    try:
        if not (image.exists() and image.samefile(cue)):
            shutil.copyfile(cue, image)
    except OSError as exc:
        raise DcError(f"{image}: cannot copy the cue {cue} there: {exc}") from exc
    return image


def run_dc(script: Path, seconds: int, workdir: Path, rom_dir: Path, image: Path, env: dict[str, str],
           *, mame: str = "mame", wait: float = runner.MAME_WAIT) -> subprocess.CompletedProcess:
    """`runner.run_mame` on the `dc` system with `-cdrom image`: the same wait for another MAME, the same MameError
    on a non-zero exit, a missing "Average speed" or a `[LUA ERROR]`. Snapshots land in `workdir/snap/dc/`. The run
    starts with fresh NVRAM in `workdir` unless one is left there: the BIOS then asks for the date first. The
    script, workdir, ROM directory and image are made absolute (MAME runs in `workdir`)."""
    return runner.run_mame(Path(script).resolve(), seconds, Path(workdir).resolve(), Path(rom_dir).resolve(),
                           env=env, mame=mame, system="dc", wait=wait,
                           extra_args=["-cdrom", str(Path(image).resolve())])
