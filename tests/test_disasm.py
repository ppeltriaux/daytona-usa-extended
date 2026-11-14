import struct
import pytest

from dtools.arcade import disasm
from dtools.arcade.disasm import DisasmError, Line, find_refs, immediates, listing, listing_bytes, listing_words
from dtools.arcade.romset import PROGRAM_SIZE, DATA_SIZE, RomSet

needs_unidasm = pytest.mark.skipif(not disasm.unidasm().exists(),
                                   reason=f"unidasm not found at {disasm.unidasm()}; set DAYTONA_UNIDASM")


def _rs(program: bytes = b"", data: bytes = b"") -> RomSet:
    return RomSet(program=program.ljust(PROGRAM_SIZE, b"\0"), data=data.ljust(DATA_SIZE, b"\0"),
                  copro_data=b"", polygons=b"", textures=b"", extra={})


def test_line_parses_real_i960_output():
    ln = disasm._parse_line("0: 90283000 02001234  ld      0x2001234,r5")
    assert ln == Line(0, bytes.fromhex("0030289034120002"), "ld      0x2001234,r5")
    ln = disasm._parse_line("c: 08000040  b       0x0000004c")
    assert ln == Line(0xC, bytes.fromhex("40000008"), "b       0x0000004c")


def test_line_parses_sh4_output():
    assert disasm._parse_line("2: 8e00  invalid $8E00") == Line(2, b"\x00\x8e", "invalid $8E00")


def test_find_refs_locates_little_endian_value_in_both_images():
    prog = bytes(16) + struct.pack("<I", 0x02123456) + bytes(8) + struct.pack("<I", 0x02123456)
    rs = _rs(prog, bytes(4) + struct.pack("<I", 0x02123456))
    assert find_refs(rs, 0x02123456, regions=("program",)) == [0x10, 0x1C]
    assert find_refs(rs, 0x02123456) == [0x10, 0x1C, 0x02000004]
    assert find_refs(rs, 0x123456, regions=("program",), width=3) == [0x10, 0x1C]


def test_immediates_extracts_data_rom_addresses():
    lines = [Line(0, b"", "ld      0x2001234,r5"), Line(8, b"", "lda     0x02ffff00,g1"),
             Line(12, b"", "b       0x0000004c"), Line(16, b"", "ld      0x03000000,r4"),
             Line(20, b"", "mov     0,r4")]
    assert immediates(lines) == {0x02001234, 0x02FFFF00}


def test_listing_requires_unidasm(monkeypatch, tmp_path):
    monkeypatch.setenv("DAYTONA_UNIDASM", str(tmp_path / "missing"))   # set after import: resolved at call time
    with pytest.raises(DisasmError, match="unidasm not found at .*missing"):
        listing(_rs(), 0, 2)


def test_unidasm_path_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DAYTONA_UNIDASM", str(tmp_path / "u1"))
    assert disasm.unidasm() == tmp_path / "u1"
    monkeypatch.setenv("DAYTONA_UNIDASM", str(tmp_path / "u2"))
    assert disasm.unidasm() == tmp_path / "u2"


def test_program_window_region_is_served_as_read_i960_does():
    rs = _rs(program=bytes(0x20000) + b"WIN!")
    name, base, blob = disasm._region_for(rs, 0x00220000)
    assert (name, base) == ("window", 0x00220000)
    assert blob[:4] == rs.read_i960(0x00220000, 4) == b"WIN!"
    assert len(blob) == 0x20000
    assert find_refs(rs, int.from_bytes(b"WIN!", "little"), regions=("window",)) == [0x00220000]


def test_listing_rejects_unmapped_address():
    with pytest.raises(DisasmError, match="not in program or data"):
        listing(_rs(), 0x00100000, 2)


def test_listing_bytes_rejects_unknown_arch():
    with pytest.raises(DisasmError, match="architecture"):
        listing_bytes(bytes(4), 0, arch="v60")


def test_unknown_arch_still_rejected():
    assert "mb86233" in disasm.ARCHES
    with pytest.raises(DisasmError, match="architecture"):
        listing_words([0] * 4, arch="v60")
    with pytest.raises(DisasmError, match="word-addressed"):
        listing_words([0] * 4, arch="i960")          # i960 addresses are bytes: listing_bytes
    with pytest.raises(DisasmError, match="listing_words"):
        listing_bytes(bytes(4), 0, arch="mb86233")   # MB86233 addresses are words


def test_listing_words_rejects_non_u32():
    with pytest.raises(DisasmError, match="word 1"):
        listing_words([0, 1 << 32])


@needs_unidasm
def test_listing_bytes_i960_and_sh4():
    word = (0x90 << 24) | (5 << 19) | (0xC << 10)
    blob = struct.pack("<III", word, 0x02001234, 0x08000040)
    lines = listing_bytes(blob, 0)
    assert [(l.addr, len(l.bytes)) for l in lines] == [(0, 8), (8, 4)]
    assert lines[0].text.startswith("ld") and immediates(lines) == {0x02001234}
    sh = listing_bytes(struct.pack("<HH", 0x6500, 0x1234), 0x8C010000, arch="sh4")
    assert sh[0].addr == 0x8C010000 and sh[0].bytes == b"\x00\x65" and "MOV.B" in sh[0].text


@needs_unidasm
def test_listing_words_synthetic():
    words = [0xBF600010, 0x3C02105C, 0x7C000002, 0x1C1F2621, 0x39000058, 0xBF624019, 0xBF6A0000, 0x00000000]
    lines = listing_words(words, base=0x40)
    assert [l.addr for l in lines] == list(range(0x40, 0x48))
    assert [l.bytes for l in lines] == [struct.pack("<I", w) for w in words]
    assert lines[0].text == "brif alw #0x10" and lines[5].text == "brul alw d"


@needs_unidasm
def test_listing_serves_data_region():
    rs = _rs(data=struct.pack("<II", 0x08000040, 0x08000040))
    lines = listing(rs, 0x02000000, 2)
    assert [l.addr for l in lines] == [0x02000000, 0x02000004]


@needs_unidasm
def test_listing_serves_program_window():
    rs = _rs(program=bytes(0x20000) + struct.pack("<II", 0x08000040, 0x08000040))
    lines = listing(rs, 0x00220000, 2)
    assert [l.addr for l in lines] == [0x00220000, 0x00220004]
    assert all(l.text.startswith("b") for l in lines)


@needs_unidasm
def test_reset_code_disassembles(romset_zip):
    # The first 0x20000 bytes of the program ROM are the boot tables (data, printed as "?"); code starts at 0x20000.
    lines = listing(RomSet.from_zip(romset_zip), 0x20000, 32)
    assert len(lines) == 32
    assert not [l for l in lines if "unknown" in l.text or "dw " in l.text]
    assert {l.text.split()[0] for l in lines} & {"b", "bal", "ld", "lda", "mov"}
