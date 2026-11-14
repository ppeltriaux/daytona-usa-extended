"""The error base of the Dreamcast side (dtools/dc): `DcError`, a `DecodeError`.

`disc.DiscError` and `iso.IsoError` derive from it, as every later Dreamcast decoder's and encoder's error does, so
the converter catches `DecodeError` once for the arcade and the Dreamcast alike. This module imports only the arcade
error base."""

from dtools.arcade.errors import DecodeError


class DcError(DecodeError):
    """A Dreamcast disc, file or value that the Dreamcast layouts do not allow: refused by a reader or a decoder."""
