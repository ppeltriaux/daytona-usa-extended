"""The error base of the arcade decoders and encoders (dtools/arcade).

Every module's own error derives from `DecodeError`: `tablebase.CourseError` (course.py, tiles.py, golden.py),
`texture.TextureError`, `polyrom.PolyError`, `geo.GeoError` and `romset.RomSetError`. A caller that feeds the
encoders converted data catches `DecodeError` once. This module imports nothing, so every module can import it
(romset.py, at the bottom of the dependency order, included); `tablebase` re-exports it."""


class DecodeError(Exception):
    """A table, value or image that the arcade layouts do not allow: refused by a decoder or an encoder."""
