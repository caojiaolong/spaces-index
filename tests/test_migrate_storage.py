import pytest

from scripts import migrate_storage as migration
from scripts.mirror_content import MirrorError


def test_migration_is_lossless_idempotent_and_keeps_originals(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    source, destination = tmp_path / "content/articles", tmp_path / "data/articles"
    (source / "9119").mkdir(parents=True)
    raw = '正文 $x^2$\r\n'.encode("utf8")
    (source / "9119/article.md").write_bytes(raw)
    (source / ".extract.lock").write_bytes(b" ")
    assert migration.migrate(source, destination) == 1
    assert (destination / "9119/article.md").read_bytes() == raw
    assert (source / "9119/article.md").read_bytes() == raw
    assert not (destination / ".extract.lock").exists()
    assert migration.migrate(source, destination) == 0


def test_migration_preflights_all_conflicts_before_copying(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    source, destination = tmp_path / "content", tmp_path / "data"
    source.mkdir()
    destination.mkdir()
    (source / "a.md").write_text("original", encoding="utf8")
    (source / "z.md").write_text("original", encoding="utf8")
    (destination / "z.md").write_text("newer", encoding="utf8")
    with pytest.raises(MirrorError, match="Conflicting"):
        migration.migrate(source, destination)
    assert not (destination / "a.md").exists()
    assert (destination / "z.md").read_text(encoding="utf8") == "newer"


@pytest.mark.parametrize("source,destination", [("data", "data"), ("data", "data/articles"), ("data/articles", "data"), ("../outside", "data")])
def test_migration_rejects_overlapping_or_external_paths(tmp_path, monkeypatch, source, destination):
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    with pytest.raises(MirrorError, match="non-overlapping"):
        migration.migrate(tmp_path / source, tmp_path / destination)
