from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    database_path: Path
    artifact_root: Path

    @classmethod
    def from_root(cls, root: Path) -> "Settings":
        resolved = root.resolve()
        state_root = resolved / ".apar"
        return cls(resolved, state_root / "state.db", state_root / "artifacts")
