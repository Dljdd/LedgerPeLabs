from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_g3.py"
PASS_LINE = (
    "G3 PASS: causal features, rules/GBDT/hybrid, matched budgets, "
    "frozen hidden evaluation, and judge scorecards"
)


def _load_verifier() -> ModuleType:
    assert SCRIPT.is_file(), "scripts/verify_g3.py must exist"
    spec = importlib.util.spec_from_file_location("verify_g3_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_manifest_uses_current_interpreter_and_exact_preregistered_order() -> None:
    verifier = _load_verifier()

    expected = (
        ("G0", [sys.executable, "scripts/verify_g0.py"]),
        ("G1_G2", [sys.executable, "scripts/verify_g1_g2.py"]),
        ("FEATURES", [sys.executable, "-m", "pytest", "tests/features", "-q"]),
        ("DEFENSE", [sys.executable, "-m", "pytest", "tests/defense", "-q"]),
        ("CASES", [sys.executable, "-m", "pytest", "tests/cases", "-q"]),
        ("EVALUATION", [sys.executable, "-m", "pytest", "tests/evaluation", "-q"]),
        (
            "G3",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_g3_defense.py",
                "-q",
            ],
        ),
    )
    assert expected == verifier.CHECKS
    assert all("verify_g3.py" not in argument for _, argv in verifier.CHECKS for argument in argv)


def test_main_runs_every_check_once_then_prints_the_exact_final_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()
    observed: list[tuple[str, list[str]]] = []

    def succeed(label: str, argv: list[str]) -> int:
        observed.append((label, argv))
        return 0

    monkeypatch.setattr(verifier, "_run_check", succeed)

    assert verifier.main() == 0
    captured = capsys.readouterr()
    assert observed == list(verifier.CHECKS)
    assert captured.out == f"{PASS_LINE}\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.replace(
            b'"campaigns_per_family":50', b'"campaigns_per_family":49'
        ),
        lambda payload: payload.replace(b'"families":', b'"families": '),
        lambda payload: payload + b"\n",
    ),
)
def test_profile_mutation_aborts_before_g0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: Callable[[bytes], bytes],
) -> None:
    verifier = _load_verifier()
    original = verifier.COMPETITION_PROFILE.read_bytes()
    profile = tmp_path / "competition-profile.json"
    profile.write_bytes(mutation(original))
    monkeypatch.setattr(verifier, "COMPETITION_PROFILE", profile)
    invoked = False

    def forbidden(_label: str, _argv: list[str]) -> int:
        nonlocal invoked
        invoked = True
        return 0

    monkeypatch.setattr(verifier, "_run_check", forbidden)
    assert verifier.main() == verifier.EXECUTION_ERROR_EXIT
    captured = capsys.readouterr()
    assert not invoked
    assert captured.out == ""
    assert captured.err == "G3 ERROR [PROFILE]: competition profile attestation failed\n"


def test_main_stops_at_first_failure_and_never_prints_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()
    observed: list[str] = []

    def fail_second(label: str, argv: list[str]) -> int:
        del argv
        observed.append(label)
        return 23 if label == "G1_G2" else 0

    monkeypatch.setattr(verifier, "_run_check", fail_second)

    assert verifier.main() == 23
    captured = capsys.readouterr()
    assert observed == ["G0", "G1_G2"]
    assert PASS_LINE not in captured.out
    assert PASS_LINE not in captured.err


def test_main_fails_closed_on_malformed_check_execution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()

    def malformed(label: str, argv: list[str]) -> NoReturn:
        del label, argv
        raise TypeError("untrusted details must not be echoed")

    monkeypatch.setattr(verifier, "_run_check", malformed)

    assert verifier.main() == verifier.EXECUTION_ERROR_EXIT
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "G3 ERROR [G0]: verifier execution failed (TypeError)\n"
    assert "untrusted details" not in captured.err
    assert PASS_LINE not in captured.err


def test_main_does_not_accept_boolean_as_a_valid_child_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()

    def malformed_status(label: str, argv: list[str]) -> bool:
        del label, argv
        return True

    monkeypatch.setattr(verifier, "_run_check", malformed_status)

    assert verifier.main() == verifier.EXECUTION_ERROR_EXIT
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "G3 ERROR [G0]: verifier execution returned a malformed status\n"
    assert PASS_LINE not in captured.err


def test_check_preserves_child_streams_and_nonzero_exit_status(
    capfd: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()
    argv = [
        sys.executable,
        "-c",
        "import sys; print('child-out'); print('child-err', file=sys.stderr); sys.exit(19)",
    ]

    assert verifier._run_check("TEST", argv) == 19
    captured = capfd.readouterr()
    assert captured.out == "child-out\n"
    assert captured.err == "child-err\nG3 ERROR [TEST]: exited with status 19\n"
    assert PASS_LINE not in captured.out
    assert PASS_LINE not in captured.err


def test_check_times_out_and_does_not_publish_pass(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()
    monkeypatch.setattr(verifier, "CHECK_TIMEOUT_SECONDS", 0.05)

    assert verifier._run_check(
        "SLOW",
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(5)"],
    ) == verifier.TIMEOUT_EXIT
    captured = capfd.readouterr()
    assert captured.out == "started\n"
    assert captured.err == "G3 ERROR [SLOW]: timed out after 0.05 seconds\n"
    assert PASS_LINE not in captured.out
    assert PASS_LINE not in captured.err


def test_check_rejects_output_beyond_the_capture_bound(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()
    monkeypatch.setattr(verifier, "MAX_CAPTURE_BYTES", 32)

    assert verifier._run_check(
        "NOISY",
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096); sys.stdout.flush()"],
    ) == verifier.OUTPUT_LIMIT_EXIT
    captured = capfd.readouterr()
    assert captured.out == "x" * 32
    assert captured.err == "G3 ERROR [NOISY]: output exceeded 32 bytes\n"
    assert PASS_LINE not in captured.out
    assert PASS_LINE not in captured.err


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal exit semantics")
def test_check_reports_child_signal_as_a_conventional_shell_status(
    capfd: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()
    argv = [
        sys.executable,
        "-c",
        f"import os, signal; os.kill(os.getpid(), {signal.SIGTERM})",
    ]

    assert verifier._run_check("SIGNALLED", argv) == 128 + signal.SIGTERM
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"G3 ERROR [SIGNALLED]: terminated by signal {signal.SIGTERM}\n"
    )
    assert PASS_LINE not in captured.err


def test_check_fails_closed_when_the_child_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verifier = _load_verifier()

    def refuse_spawn(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise OSError("path details must not be echoed")

    monkeypatch.setattr(subprocess, "Popen", refuse_spawn)

    assert verifier._run_check("BROKEN", [sys.executable, "-c", "pass"]) == (
        verifier.EXECUTION_ERROR_EXIT
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "G3 ERROR [BROKEN]: could not start check (OSError)\n"
    assert "path details" not in captured.err
    assert PASS_LINE not in captured.err


def test_child_environment_removes_python_and_pytest_injection_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier()
    monkeypatch.setenv("PYTHONPATH", "/attacker")
    monkeypatch.setenv("PYTHONHOME", "/attacker")
    monkeypatch.setenv("PYTHONSTARTUP", "/attacker/startup.py")
    monkeypatch.setenv("PYTHONINSPECT", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=tests")

    environment = verifier._child_environment()

    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "PYTHONSTARTUP" not in environment
    assert "PYTHONINSPECT" not in environment
    assert "PYTEST_ADDOPTS" not in environment
