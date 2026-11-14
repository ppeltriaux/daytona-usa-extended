"""Tests for the 2352-byte MODE1 ISO9660 reader using a synthetic image."""

from pathlib import Path

import pytest

from dtools.dc.iso import Iso2352, IsoError
from tests.dc_images import RAW, SECTOR, make_image


@pytest.fixture
def image(tmp_path) -> Path:
    p = tmp_path / "track01.bin"
    p.write_bytes(make_image({"A.BIN": b"x" * 5000, "B.TXT": b"hi"}))
    return p


def test_entries_strip_version_and_skip_dirs(image):
    iso = Iso2352(image)
    assert {e.path for e in iso.entries()} == {"A.BIN", "B.TXT"}


def test_read_spans_multiple_sectors(image):
    iso = Iso2352(image)
    assert iso.read("A.BIN") == b"x" * 5000
    assert iso.read("B.TXT") == b"hi"


def test_directory_padding_skips_to_next_sector(tmp_path):
    files = {f"F{i:03d}.DAT": bytes([i]) for i in range(64)}   # 64 records > one 2048-byte sector
    p = tmp_path / "big.bin"
    p.write_bytes(make_image(files))
    iso = Iso2352(p)
    assert len(iso.entries()) == 64
    assert iso.read("F063.DAT") == bytes([63])


def test_extract_all_excludes_pattern(tmp_path, image):
    out = tmp_path / "out"
    n = Iso2352(image).extract_all(out, exclude=("B.",))
    assert n == 1
    assert (out / "A.BIN").read_bytes() == b"x" * 5000
    assert not (out / "B.TXT").exists()


def test_extract_all_excludes_by_prefix_of_the_first_component(tmp_path):
    p = tmp_path / "t.bin"
    p.write_bytes(make_image({"ADXB_000.ADX": b"a", "DPLOGO.BIN": b"d", "2_DP.BIN": b"2", "SG_DPLDR.BIN": b"s"}))
    out = tmp_path / "out"
    assert Iso2352(p).extract_all(out) == 2                      # default exclude ("ADX", "DP")
    assert sorted(f.name for f in out.iterdir()) == ["2_DP.BIN", "SG_DPLDR.BIN"]


def test_iso_context_manager_closes(image):
    with Iso2352(image) as iso:
        assert iso.read("B.TXT") == b"hi"
        f = iso._f
        assert not f.closed
    assert f.closed
    iso.close()                                                  # closing twice is harmless


def test_iso_closes_its_file_when_the_volume_is_refused(tmp_path):
    p = tmp_path / "blank.bin"
    p.write_bytes(bytes(RAW * 20))
    opened = []
    real_open = open

    def tracking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        opened.append(f)
        return f

    import builtins
    mp = pytest.MonkeyPatch()
    mp.setattr(builtins, "open", tracking_open)
    try:
        with pytest.raises(IsoError, match="no ISO9660"):
            Iso2352(p)
    finally:
        mp.undo()
    assert opened and all(f.closed for f in opened)


def test_subdirectory_entries_read_and_extract(tmp_path):
    p = tmp_path / "sub.bin"
    files = {"TOP.BIN": b"t", "DP1/INNER.BIN": b"i" * 3000, "DP1/DEEP/X.BIN": b"x", "OTHER/Y.BIN": b"y"}
    p.write_bytes(make_image(files, base=45000))
    with Iso2352(p, lba_base=45000) as iso:
        assert {e.path for e in iso.entries()} == set(files)
        assert iso.read("DP1/INNER.BIN") == b"i" * 3000
        assert iso.read("DP1/DEEP/X.BIN") == b"x"
        out = tmp_path / "out"
        assert iso.extract_all(out, exclude=("DP",)) == 2
    assert (out / "OTHER" / "Y.BIN").read_bytes() == b"y"
    assert not (out / "DP1").exists()


def test_lba_base_offsets_extents(tmp_path):
    p = tmp_path / "gd.bin"
    p.write_bytes(make_image({"A.BIN": b"x" * 5000, "B.TXT": b"hi"}, base=45000))
    iso = Iso2352(p, lba_base=45000)
    assert min(e.lba for e in iso.entries()) >= 45018
    assert iso.read("A.BIN") == b"x" * 5000
    assert iso.read("B.TXT") == b"hi"
    assert iso.sector(45016)[1:6] == b"CD001"


def test_lba_below_base_is_an_error(tmp_path):
    p = tmp_path / "gd.bin"
    p.write_bytes(make_image({"A.BIN": b"x"}, base=45000))
    with pytest.raises(IsoError, match="before the image base"):
        Iso2352(p, lba_base=45000).sector(100)


def test_real_disc_listing(dc_path):
    from dtools.dc.disc import DcDisc

    track3 = DcDisc.open(dc_path).iso
    sizes = {e.path: e.size for e in track3.entries()}
    assert sizes["1ST_READ.BIN"] == 2010720
    assert sizes["CS0_POL.BIN"] == 474248
    assert sizes["CS1_POL.BIN"] == 2334752
    assert sizes["CS7_POL.BIN"] == 1288312
    assert sizes["CS_DATA0.BIN"] == 200030
    assert sizes["CS_DATA7.BIN"] == 435990
    assert sizes["GAME_MAP.BIN"] == 77824
    assert sizes["CSE_POL.BIN"] == 205120
    # 580 files in the root; the DP* browser directories add 164 more
    assert sum(1 for e in track3.entries() if "/" not in e.path) == 580
    assert len(track3.entries()) == 744
