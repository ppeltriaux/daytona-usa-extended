"""The converter: Dreamcast course data (dtools/dc) into the arcade's course tables (dtools/arcade).

Modules: `tables` (the road line, AI lines, steps, attributes and checkpoint rows), `ground` (the TGP ground), `model`
(the course model as polygon objects per scenery cell), `textures` (the course textures as 4-bit texels with one
hue each, packed into the course sheet) and `build` (the converted golden.Course in an arcade slot,
written into the set's free runs).
The transform, measured exact on the Dreamcast's courses n 0-2 against the arcade's Beginner, Expert and Advanced
(Phase 5), is in their docstrings and docs/spike/2026-11-convert.md ("Method").

Every public function validates its inputs and raises `ConvertError` (a `DecodeError`); a `DcError` from the
Dreamcast readers it calls passes through (a `DecodeError` too), so a caller catches `DecodeError` once."""

from dtools.arcade.errors import DecodeError


class ConvertError(DecodeError):
    """A Dreamcast value the arcade's tables cannot hold, or an input that is not what a converter takes."""
