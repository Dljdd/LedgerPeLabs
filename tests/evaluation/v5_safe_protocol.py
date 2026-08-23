"""Test-only Sentinel v5 protocol copy that never executes locked seed 2404."""

from pathlib import Path

from apar.evaluation.v5_protocol import (
    V5DevelopmentProtocol,
    load_v5_development_protocol,
)


def load_safe_v5_test_protocol(root: Path) -> V5DevelopmentProtocol:
    locked = load_v5_development_protocol(
        root / "config/defense/defense-v5-development.json"
    )
    if locked.seeds.development_test != 2404:
        raise AssertionError("locked development-test seed changed unexpectedly")
    return locked.model_copy(
        update={
            "seeds": locked.seeds.model_copy(
                update={"development_test": 404}
            )
        }
    )
