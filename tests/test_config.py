from dtools import config


def test_rom_dir_none_when_unset(monkeypatch):
    monkeypatch.delenv("DAYTONA_ROMDIR", raising=False)
    assert config.rom_dir() is None


def test_rom_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DAYTONA_ROMDIR", str(tmp_path))
    assert config.rom_dir() == tmp_path


def test_dc_path_from_env(monkeypatch, tmp_path):
    cue = tmp_path / "Daytona USA (USA).cue"
    monkeypatch.setenv("DAYTONA_DC", str(cue))
    assert config.dc_path() == cue


def test_out_dir_is_created(tmp_path, monkeypatch):
    monkeypatch.setenv("DAYTONA_OUT", str(tmp_path / "o"))
    assert config.out_dir().is_dir()
    assert config.out_dir() == tmp_path / "o"


def _fake_build(root, patch_text="the patch\n", stamp=True, exe=True):
    """A fake build tree at `root`: tools/<patch>, build/mame/model2 and its stamp naming the patch's SHA-256."""
    import hashlib
    (root / "tools").mkdir(parents=True, exist_ok=True)
    patch = root / "tools" / "mame-daytona-rom-space.patch"
    patch.write_text(patch_text)
    (root / "build" / "mame").mkdir(parents=True, exist_ok=True)
    binary = root / "build" / "mame" / "model2"
    binary.write_text("#!/bin/sh\n")
    if exe:
        binary.chmod(0o755)
    if stamp:
        (root / "build" / "mame" / ".daytona-rom-space").write_text(
            hashlib.sha256(patch.read_bytes()).hexdigest() + "\n")
    return patch, binary


def test_custom_mame_found_when_built_with_the_current_patch(monkeypatch, tmp_path):
    monkeypatch.delenv("DAYTONA_CUSTOM_MAME", raising=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    patch, binary = _fake_build(tmp_path)
    assert config.custom_mame() == binary
    patch.write_text("a changed patch\n")                   # built from another patch: stale, not found
    assert config.custom_mame() is None


def test_custom_mame_none_without_stamp_or_binary(monkeypatch, tmp_path):
    monkeypatch.delenv("DAYTONA_CUSTOM_MAME", raising=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    assert config.custom_mame() is None
    _fake_build(tmp_path, stamp=False)
    assert config.custom_mame() is None
    _fake_build(tmp_path / "x", exe=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "x")
    assert config.custom_mame() is None


def test_custom_mame_from_env(monkeypatch, tmp_path):
    exe = tmp_path / "mymame"
    monkeypatch.setenv("DAYTONA_CUSTOM_MAME", str(exe))
    assert config.custom_mame() is None                     # not there
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert config.custom_mame() == exe
