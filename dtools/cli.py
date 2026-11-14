"""The `dtools` command (Phase 12, M7): check the player's own inputs, build Daytona USA Extended.

    dtools check --roms DIR --disc PATH --out DIR
    dtools build --roms DIR --disc PATH --out DIR

`--roms` is the directory holding MAME 0.289's `daytona.zip` (and `model1io.zip`), default DAYTONA_ROMDIR; `--disc`
the US Dreamcast disc, the Redump `.7z` or its `.cue` with the track `.bin` files beside it, default DAYTONA_DC;
`--out` (required) the output directory: `daytona.zip`, `model1io.zip`, `build-report.txt`, and the work files under
`<out>/work` (the .7z's extraction, the re-mesh cache: DAYTONA_OUT points there while the command runs). The inputs
are never written. `check` creates `--out` (a .7z is extracted under `<out>/work` then, for the build to reuse); a
check that fails removes an `--out` it created. `build` runs the check first and refuses until it passes
(dtools/release.py `check_inputs`, `release_set`, `write_release`). Exit 0 done, 1 a failed check or build (one line,
never a traceback), 2 a usage error.

Where the line is drawn: `check_inputs` turns what a bad input raises into problem lines (a damaged disc image's
IndexError or struct.error included, release._UNREADABLE); what still reaches `main` from it is a DecodeError or an
OSError (one line, exit 1) or a defect in this code, which keeps its traceback. `build`'s conversion step folds any
exception into one line (the brief's contract for the build)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from dtools import config
from dtools import release
from dtools.arcade.errors import DecodeError
from dtools.arcade.romset import RomSet
from dtools.dc.disc import DcDisc


@contextmanager
def _work_env(work: Path):
    """DAYTONA_OUT set to `work` (config.out_dir(): the re-mesh cache) while the command runs, restored after (main may
    run inside another process, the tests)."""
    old = os.environ.get("DAYTONA_OUT")
    os.environ["DAYTONA_OUT"] = str(work)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("DAYTONA_OUT", None)
        else:
            os.environ["DAYTONA_OUT"] = old


def _one_line(e: BaseException) -> str:
    return " ".join(str(e).split()) or type(e).__name__


def _build(roms: Path, disc: Path, out: Path, checks: list[str]) -> int:
    start = time.monotonic()
    try:
        rs = RomSet.from_zip(roms / release.ROM_ZIP)
        with DcDisc.open(disc, out=out / release.WORK) as dc:
            built = release.release_set(rs, dc)
        path = release.write_release(built, out, roms, checks=checks)
    except Exception as e:      # noqa: BLE001  one line, never a traceback (DecodeError, OSError, anything else)
        print(f"build failed: {_one_line(e)}", file=sys.stderr)
        return 1
    print(f"built {path} in {time.monotonic() - start:.0f} s; report: {out / release.REPORT}")
    print("run it from the repository root, with the custom MAME:")
    print(f"  {release.mame_command(out)}")
    print(f"note: {release.CUSTOM_NOTE}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dtools", description="Daytona USA Extended: the Dreamcast's five courses added "
                                "to the Model 2 arcade game, built from your own MAME 0.289 `daytona` set and US "
                                "Dreamcast disc (runs on the custom MAME build, tools/build-mame.sh).")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in (("check", "verify the ROM set, the disc and the output directory (creates --out; a .7z is "
                         "extracted under <out>/work for the build; a failed check removes an --out it created)"),
                        ("build", "check, then build the release set into --out")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--roms", type=Path, help="directory with daytona.zip (MAME 0.289) and model1io.zip "
                       "(default: $DAYTONA_ROMDIR)")
        s.add_argument("--disc", type=Path, help="the US Dreamcast disc: the Redump .7z, or its .cue with the track "
                       ".bin files beside it (default: $DAYTONA_DC)")
        s.add_argument("--out", type=Path, required=True, help="output directory (daytona.zip, model1io.zip, "
                       "build-report.txt; work files under <out>/work); never the ROM directory")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    roms, disc = args.roms or config.rom_dir(), args.disc or config.dc_path()
    missing = [n for n, v in (("--roms (or DAYTONA_ROMDIR)", roms), ("--disc (or DAYTONA_DC)", disc)) if v is None]
    if missing:
        parser.error("missing " + ", ".join(missing))
    # absolute: MAME runs the printed command with -rompath, from the repository root
    roms, disc, out = (Path(p).expanduser().absolute() for p in (roms, disc, args.out))
    out = out.resolve() if out.exists() else out
    with _work_env(out / release.WORK):
        try:
            report = release.check_inputs(roms, disc, out)
        except (DecodeError, OSError) as e:     # the inputs' errors; anything else is a defect and keeps its traceback
            print(f"check failed: {_one_line(e)}", file=sys.stderr)
            return 1
        print("\n".join(report.lines))
        if not report.ok:
            print("not ready: fix the problems above")
            return 1
        if args.command == "check":
            print("ok: ready to build")
            return 0
        return _build(roms, disc, out, report.lines)


if __name__ == "__main__":
    raise SystemExit(main())
