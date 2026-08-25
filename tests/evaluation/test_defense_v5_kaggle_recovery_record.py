"""Terminal evidence checks for the consumed local Sentinel v5 attempt."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from apar.evaluation.v5_kaggle_protocol import load_v5_kaggle_protocol

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/defense/defense-v5-kaggle-recovery.json"

_BOUND_PATHS = (
    "docs/experiments/defense-v5-locked-development-attempt.json",
    "docs/experiments/defense-v5-locked-development-abort.json",
    "docs/experiments/defense-v5-development-result.json",
    "config/defense/defense-v5-safe-core-freeze.json",
)


def _copy_bound_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for relative in _BOUND_PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return root


def test_recovery_record_binds_the_consumed_attempt_and_unchanged_evidence() -> None:
    """The successor protocol must preserve the failed attempt as terminal evidence."""
    protocol = load_v5_kaggle_protocol(CONFIG, root=ROOT)
    recovery = protocol.recovery

    assert recovery.attempt_receipt_raw_sha256 == (
        "c9093272309605293f6377699df1810485901e0e3c5dfa9f81226ddea31151e8"
    )
    assert recovery.attempt_receipt_self_sha256 == (
        "2cd207fdef0b808a8623843152195495d25d40b5d7903c5e71fd936611a09b93"
    )
    assert recovery.abort_record_sha256 == (
        "dc0743f1fe93356ea1e06af188d7a0e08cf46f0fcea02a674dbc1b2ec63d94d8"
    )
    assert recovery.historical_result_sha256 == (
        "af326f3a0fcbbe12c9b8623fc7d82a1ba6d0f327ec9a80f462cacd4bea1dd185"
    )
    assert recovery.historical_safe_core_sha256 == (
        "784a762fd90a65219a233e87df35290ac87c8fe8e4b9024de46564568f633719"
    )
    assert recovery.consumed_attempt_safe_core_sha256 == (
        "8cd3bba2cda47aa5b0d0a85fed4476eeeff787f3f0d2fcec973cc3e30a7b0435"
    )
    assert recovery.retry_permitted is False


@pytest.mark.parametrize(
    ("relative", "mutation"),
    [
        (_BOUND_PATHS[0], b"{}\n"),
        (_BOUND_PATHS[1], b"{}\n"),
        (_BOUND_PATHS[2], b"changed historical result\n"),
        (_BOUND_PATHS[3], b"{}\n"),
    ],
)
def test_recovery_loader_rejects_bound_file_mutation(
    tmp_path: Path,
    relative: str,
    mutation: bytes,
) -> None:
    """Mutating any prior evidence byte must invalidate the successor protocol."""
    root = _copy_bound_root(tmp_path)
    (root / relative).write_bytes(mutation)
    with pytest.raises(ValueError):
        load_v5_kaggle_protocol(CONFIG, root=root)


def test_recovery_loader_rejects_missing_abort_binding(tmp_path: Path) -> None:
    """A successor run cannot erase the terminal explanation of the consumed attempt."""
    document = json.loads(CONFIG.read_bytes())
    del document["recovery"]["abort_record_path"]
    path = tmp_path / "missing-abort.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_v5_kaggle_protocol(path, root=ROOT)


def test_recovery_loader_rejects_retry_permission_mutation(tmp_path: Path) -> None:
    """No config edit can turn the crashed local attempt into a retryable attempt."""
    document = json.loads(CONFIG.read_bytes())
    document["recovery"]["retry_permitted"] = True
    path = tmp_path / "retryable.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_v5_kaggle_protocol(path, root=ROOT)
