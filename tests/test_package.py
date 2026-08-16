from pathlib import Path

from apar import __version__
from apar.config import Settings


def test_settings_are_root_relative(tmp_path: Path) -> None:
    settings = Settings.from_root(tmp_path)
    assert __version__ == "0.1.0"
    assert settings.database_path == tmp_path / ".apar" / "state.db"
    assert settings.artifact_root == tmp_path / ".apar" / "artifacts"
