"""Task 6 round-three Phase D approval, protocol, and tree regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import scripts.run_task6_holdout as holdout_runner

ROOT = Path(__file__).resolve().parents[2]
V32_CANCELLATION = ROOT / "docs/experiments/task6-v3.2-cancellation.json"
V32_RESULT = ROOT / "docs/experiments/task6-v3.2-holdout-result.json"
V33_PREREGISTRATION = ROOT / "docs/experiments/task6-v3.3-holdout-preregistration.json"
V33_RESULT = ROOT / "docs/experiments/task6-v3.3-holdout-result.json"
V33_REJECTION = ROOT / "docs/experiments/task6-v3.3-postexecution-rejection.json"
V34_PREREGISTRATION = ROOT / "docs/experiments/task6-v3.4-holdout-preregistration.json"
V34_RESULT = ROOT / "docs/experiments/task6-v3.4-holdout-result.json"
SOURCE_COMMIT = "4ed9f6acabcdaeaa2e6c4a58ed150ffb2f87b7f6"
V32_FREEZE_COMMIT = "239617e70563c9af3566b821cdaeb82df48cf1c7"


def _digest(document: object) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _entry(*, mode: str = "100644", content: str = "b") -> dict[str, str]:
    return {
        "git_mode": mode,
        "object_type": "blob",
        "git_object_id": "a" * 40,
        "content_sha256": content * 64,
    }


def _manifest(entries: dict[str, dict[str, str]]) -> dict[str, object]:
    return {"entries": entries, "digest": _digest(entries)}


def test_v32_is_canonically_cancelled_without_its_result() -> None:
    assert V32_CANCELLATION.exists()
    cancellation = json.loads(V32_CANCELLATION.read_text(encoding="utf-8"))

    assert cancellation["status"] == "cancelled_before_execution"
    assert cancellation["cancelled_preregistration_commit"] == V32_FREEZE_COMMIT
    assert cancellation["cancelled_source_commit"] == SOURCE_COMMIT
    assert cancellation["confirmatory_execution_invoked"] is False
    assert cancellation["seeds_used"] is False
    assert cancellation["result_created"] is False
    assert cancellation["replacement_preregistration_path"] == (
        "docs/experiments/task6-v3.3-holdout-preregistration.json"
    )
    assert not V32_RESULT.exists()
    assert V33_RESULT.exists()
    assert V33_REJECTION.exists()
    assert not V34_RESULT.exists()


def test_execute_parser_requires_both_external_approval_values() -> None:
    parser = getattr(holdout_runner, "_parse_args", None)
    assert callable(parser)

    with pytest.raises(SystemExit):
        parser(["--execute-confirmatory"])
    with pytest.raises(SystemExit):
        parser(
            [
                "--execute-confirmatory",
                "--approved-freeze-commit",
                "a" * 40,
            ]
        )
    with pytest.raises(SystemExit):
        parser(
            [
                "--execute-confirmatory",
                "--approved-prereg-sha256",
                "b" * 64,
            ]
        )


def test_verify_parser_rejects_only_one_approval_value() -> None:
    parser = getattr(holdout_runner, "_parse_args", None)
    assert callable(parser)

    with pytest.raises(SystemExit):
        parser(["--verify-only", "--approved-freeze-commit", "a" * 40])
    with pytest.raises(SystemExit):
        parser(["--verify-only", "--approved-prereg-sha256", "b" * 64])


def test_external_approval_binds_exact_head_and_preregistration_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = getattr(holdout_runner, "_validate_external_approval", None)
    assert callable(validate)
    preregistration = tmp_path / "prereg.json"
    preregistration.write_bytes(b'{"frozen":true}\n')
    expected_sha = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    approved_commit = "c" * 40
    monkeypatch.setattr(holdout_runner, "_head_commit", lambda: approved_commit)

    assert validate(
        approved_freeze_commit=approved_commit,
        approved_prereg_sha256=expected_sha,
        preregistration_path=preregistration,
    ) == {
        "approved_freeze_commit": approved_commit,
        "approved_prereg_sha256": expected_sha,
    }

    with pytest.raises(RuntimeError, match="approved freeze commit|HEAD"):
        validate(
            approved_freeze_commit="d" * 40,
            approved_prereg_sha256=expected_sha,
            preregistration_path=preregistration,
        )
    with pytest.raises(RuntimeError, match="preregistration.*SHA|approved SHA"):
        validate(
            approved_freeze_commit=approved_commit,
            approved_prereg_sha256="e" * 64,
            preregistration_path=preregistration,
        )


def test_clean_descendant_cannot_satisfy_approved_freeze_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = getattr(holdout_runner, "_validate_external_approval", None)
    assert callable(validate)
    preregistration = tmp_path / "prereg.json"
    preregistration.write_bytes(b"{}\n")
    preregistration_sha = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    monkeypatch.setattr(holdout_runner, "_head_commit", lambda: "2" * 40)

    with pytest.raises(RuntimeError, match="exact approved freeze commit|HEAD"):
        validate(
            approved_freeze_commit="1" * 40,
            approved_prereg_sha256=preregistration_sha,
            preregistration_path=preregistration,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("seeds",), [503]),
        (("budgets", "proposal"), 23),
        (("budgets", "query"), 23),
        (("budgets", "logical_time"), 23),
        (("budgets", "wall_time_ms"), 1),
        (("maximum_confirmatory_attempts",), 2),
        (("metrics", "app_scam_mule", "primary_outcome"), "valid_yield"),
        (("metrics", "card_testing_cnp", "minimum_delta"), "0.09"),
        (("negative_control", "expected_observed_delta"), "0.01"),
        (("network", "allowed_calls"), 1),
        (("uncertainty", "post_hoc_gate"), True),
        (("stopping_rule",), "run again"),
    ),
)
def test_protocol_tampering_rejects_before_runtime(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    expected = getattr(holdout_runner, "_expected_protocol", None)
    validate = getattr(holdout_runner, "_validate_protocol", None)
    assert callable(expected) and callable(validate)
    protocol = copy.deepcopy(expected())
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(RuntimeError, match="protocol"):
        validate(protocol)


def test_protocol_rejects_extra_field_and_non_exact_json_scalar() -> None:
    expected = getattr(holdout_runner, "_expected_protocol", None)
    validate = getattr(holdout_runner, "_validate_protocol", None)
    assert callable(expected) and callable(validate)
    extra = copy.deepcopy(expected())
    extra["undeclared"] = True
    subclassed = copy.deepcopy(expected())

    class MutableString(str):
        pass

    subclassed["stopping_rule"] = MutableString(subclassed["stopping_rule"])

    with pytest.raises((RuntimeError, TypeError), match="protocol|exact"):
        validate(extra)
    with pytest.raises((RuntimeError, TypeError), match="protocol|exact"):
        validate(subclassed)


def test_behavior_manifest_records_mode_type_object_and_content_hash() -> None:
    document = holdout_runner._source_freeze_document(holdout_runner._head_commit())
    entries = document["behavior_manifest"]["entries"]

    assert entries
    assert all(
        set(entry)
        == {"git_mode", "object_type", "git_object_id", "content_sha256"}
        for entry in entries.values()
    )
    assert all(entry["git_mode"] in {"100644", "100755"} for entry in entries.values())
    assert all(entry["object_type"] == "blob" for entry in entries.values())


@pytest.mark.parametrize(
    "changed_entries",
    (
        {
            "scripts/run_task6_holdout.py": _entry(),
            "scripts/unapproved.py": _entry(content="c"),
        },
        {
            "scripts/run_task6_holdout.py": _entry(),
            "sitecustomize.py": _entry(content="c"),
        },
        {"scripts/run_task6_holdout.py": _entry(mode="100755")},
        {},
    ),
)
def test_behavior_manifest_rejects_added_deleted_or_mode_changed_paths(
    changed_entries: dict[str, dict[str, str]],
) -> None:
    validate = getattr(holdout_runner, "_validate_matching_manifests", None)
    assert callable(validate)
    source = _manifest({"scripts/run_task6_holdout.py": _entry()})

    with pytest.raises(RuntimeError, match="manifest|path|mode"):
        validate(source, _manifest(changed_entries))


def test_behavior_manifest_rejects_git_symlink_even_with_same_content() -> None:
    validate = getattr(holdout_runner, "_validate_matching_manifests", None)
    assert callable(validate)
    normal = _manifest({"scripts/run_task6_holdout.py": _entry()})
    symlink = _manifest(
        {"scripts/run_task6_holdout.py": _entry(mode="120000")}
    )

    with pytest.raises(RuntimeError, match="symlink|mode|regular"):
        validate(normal, symlink)


def test_filesystem_symlink_rejects_even_when_path_and_bytes_appear_valid(
    tmp_path: Path,
) -> None:
    validate = getattr(holdout_runner, "_validate_filesystem_behavior_paths", None)
    assert callable(validate)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    target = tmp_path / "runner-target.py"
    target.write_text("pass\n", encoding="utf-8")
    link = scripts / "run_task6_holdout.py"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink|regular"):
        validate(
            {"scripts/run_task6_holdout.py": _entry()},
            root=tmp_path,
        )


@pytest.mark.parametrize("module_name", ("sitecustomize", "usercustomize"))
def test_unapproved_loaded_python_customization_module_rejects(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = getattr(holdout_runner, "_validate_python_customization_modules", None)
    assert callable(validate)
    module = ModuleType(module_name)
    module.__file__ = str(tmp_path / f"{module_name}.py")
    monkeypatch.setitem(sys.modules, module_name, module)

    with pytest.raises(RuntimeError, match="sitecustomize|usercustomize|customization"):
        validate({}, root=tmp_path / "repository")


def test_verify_only_handles_v34_freeze_without_search() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_task6_holdout.py"), "--verify-only"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    if V34_PREREGISTRATION.exists():
        assert "awaiting external approval" in completed.stdout
        assert "--approved-freeze-commit" in completed.stdout
        assert "--approved-prereg-sha256" in completed.stdout
    else:
        assert "source stage" in completed.stdout
    assert "no holdout trial executed" in completed.stdout
    assert not V34_RESULT.exists()


def test_execute_refuses_when_v34_preregistration_is_absent() -> None:
    if V34_PREREGISTRATION.exists():
        pytest.skip("source-stage missing-preregistration regression")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_task6_holdout.py"),
            "--execute-confirmatory",
            "--approved-freeze-commit",
            "a" * 40,
            "--approved-prereg-sha256",
            "b" * 64,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "preregistration" in completed.stderr
    assert not V34_RESULT.exists()


def test_v34_preregistration_is_absent_until_separate_freeze_commit() -> None:
    if not V34_PREREGISTRATION.exists():
        assert not V34_RESULT.exists()
        return
    artifact = json.loads(V34_PREREGISTRATION.read_text(encoding="utf-8"))
    validate = getattr(holdout_runner, "_validate_preregistration_schema", None)
    assert callable(validate)

    validate(artifact)
    assert artifact["status"] == "final_v3_4_frozen_before_evidence_replication"
    assert artifact["protocol"] == holdout_runner._expected_protocol()
    assert artifact["source_freeze"]["source_commit"] != holdout_runner._head_commit()
    assert not V34_RESULT.exists()
