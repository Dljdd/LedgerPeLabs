from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleStage,
    load_v5_kaggle_protocol,
)
from scripts.build_defense_v5_kaggle_notebooks import (
    V5GeneratedNotebook,
    build_v5_kaggle_notebooks,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_kaggle_protocol(
    ROOT / "config/defense/defense-v5-kaggle-recovery.json",
    root=ROOT,
)
FROZEN_OWNER = "dylanmoraes"
FROZEN_SOURCE = "apar-sentinel-v5-source3"
FROZEN_WHEELS = "apar-sentinel-v5-wheelhouse-py312-linux-x86-64"
FROZEN_SAFE = "apar-sentinel-v5-safe-evidence"


def _generate(output: Path) -> tuple[V5GeneratedNotebook, ...]:
    return build_v5_kaggle_notebooks(
        root=ROOT,
        output_dir=output,
        owner_slug=FROZEN_OWNER,
        source_dataset_slug=FROZEN_SOURCE,
        wheelhouse_dataset_slug=FROZEN_WHEELS,
        safe_evidence_dataset_slug=FROZEN_SAFE,
    )


def _load_notebook(path: Path) -> dict[str, object]:
    document = json.loads(path.read_bytes())
    assert isinstance(document, dict)
    return document


def _cell_sources(document: dict[str, object]) -> tuple[str, ...]:
    cells = document["cells"]
    assert isinstance(cells, list)
    sources: list[str] = []
    for cell in cells:
        assert isinstance(cell, dict)
        assert cell["cell_type"] == "code"
        source = cell["source"]
        assert isinstance(source, str)
        sources.append(source)
    return tuple(sources)


def test_generated_notebooks_are_private_cpu_and_network_disabled(
    tmp_path: Path,
) -> None:
    generated = _generate(tmp_path / "notebooks")
    assert tuple(item.stage for item in generated) == tuple(V5KaggleStage)
    for item in generated:
        metadata = json.loads(item.metadata_path.read_bytes())
        assert metadata["is_private"] is True
        assert metadata["enable_internet"] is False
        assert metadata["enable_gpu"] is False
        assert metadata["enable_tpu"] is False
        assert metadata["machine_shape"] == ""
        assert metadata["accelerator"] == "none"
        assert metadata["id"].startswith(f"{FROZEN_OWNER}/")


def test_notebooks_have_only_closed_bootstrap_install_and_invoke_cells(
    tmp_path: Path,
) -> None:
    generated = _generate(tmp_path / "notebooks")
    expected_datasets = [
        f"{FROZEN_OWNER}/{FROZEN_SOURCE}",
        f"{FROZEN_OWNER}/{FROZEN_WHEELS}",
        f"{FROZEN_OWNER}/{FROZEN_SAFE}",
    ]
    forbidden = re.compile(
        r"(?:kaggle(?:hub|\.api|_secrets)|api[_-]?token|password|secret|"
        r"--(?:seed|profile|stage|resume|retry)|is_fraud|interpret|plot|pandas)",
        re.IGNORECASE,
    )
    for index, item in enumerate(generated):
        notebook = _load_notebook(item.notebook_path)
        sources = _cell_sources(notebook)
        assert len(sources) == 3
        assert "pip" in sources[1]
        assert "--no-index" in sources[1]
        assert "--find-links" in sources[1]
        assert "--force-reinstall" in sources[1]
        assert sources[2].count("scripts/run_defense_v5_kaggle_stage.py") == 1
        assert "--root" in sources[2]
        assert "--input-root" in sources[2]
        assert "--output-root" in sources[2]
        assert "--safe-evidence" in sources[2]
        assert "--execution-manifest" in sources[2]
        assert "--approved-commit" in sources[2]
        assert "--authorize-successor" not in sources[2]
        assert "--force" not in sources[2]
        assert 'SAFE_MANIFEST.get("execution_mode")' in sources[2]
        assert "kaggle_capacity_validation" in sources[2]
        assert "kaggle_locked_successor" in sources[2]
        assert forbidden.search("\n".join(sources)) is None

        metadata = json.loads(item.metadata_path.read_bytes())
        assert metadata["dataset_sources"] == expected_datasets
        expected_predecessors = [] if index == 0 else [generated[index - 1].kernel_id]
        assert metadata["kernel_sources"] == expected_predecessors
        assert metadata["code_file"] == item.notebook_path.name
        assert metadata["apar_notebook_sha256"] == item.notebook_sha256
        assert hashlib.sha256(item.notebook_path.read_bytes()).hexdigest() == (
            item.notebook_sha256
        )


def test_notebooks_use_owner_namespaced_kaggle_mounts(tmp_path: Path) -> None:
    """Catch regressions to Kaggle's retired flat /kaggle/input mount layout."""

    generated = _generate(tmp_path / "notebooks")
    dataset_root = f"/kaggle/input/datasets/{FROZEN_OWNER}"
    notebook_root = f"/kaggle/input/notebooks/{FROZEN_OWNER}"
    for index, item in enumerate(generated):
        notebook = _load_notebook(item.notebook_path)
        sources = _cell_sources(notebook)
        bootstrap_namespace: dict[str, object] = {"Path": Path}
        bootstrap_tree = ast.parse(sources[0])
        bootstrap_assignments = [
            node
            for node in bootstrap_tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in {"DATASET_INPUT_ROOT", "SOURCE_INPUT", "WHEELHOUSE_INPUT", "SAFE_INPUT"}
        ]
        exec(
            compile(ast.Module(bootstrap_assignments, type_ignores=[]), "<mounts>", "exec"),
            bootstrap_namespace,
        )
        assert bootstrap_namespace["SOURCE_INPUT"] == Path(
            f"{dataset_root}/{FROZEN_SOURCE}"
        )
        assert bootstrap_namespace["WHEELHOUSE_INPUT"] == Path(
            f"{dataset_root}/{FROZEN_WHEELS}"
        )
        assert bootstrap_namespace["SAFE_INPUT"] == Path(
            f"{dataset_root}/{FROZEN_SAFE}"
        )
        if index == 0:
            assert "PREDECESSOR_CHAIN" not in sources[2]
        else:
            predecessor = generated[index - 1].kernel_id.split("/", maxsplit=1)[1]
            invoke_namespace: dict[str, object] = {"Path": Path}
            invoke_tree = ast.parse(sources[2])
            predecessor_assignment = next(
                node
                for node in invoke_tree.body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "PREDECESSOR_CHAIN"
            )
            exec(
                compile(
                    ast.Module([predecessor_assignment], type_ignores=[]),
                    "<predecessor>",
                    "exec",
                ),
                invoke_namespace,
            )
            assert invoke_namespace["PREDECESSOR_CHAIN"] == Path(
                f"{notebook_root}/{predecessor}/apar-v5-chain"
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_slug", "DylanMoraes"),
        ("owner_slug", "https://kaggle.com/dylanmoraes"),
        ("source_dataset_slug", "source dataset"),
        ("wheelhouse_dataset_slug", "other-wheelhouse"),
        ("safe_evidence_dataset_slug", "safe/evidence"),
    ],
)
def test_generator_rejects_nonfrozen_or_malformed_identifiers(
    tmp_path: Path, field: str, value: str
) -> None:
    arguments = {
        "root": ROOT,
        "output_dir": tmp_path,
        "owner_slug": FROZEN_OWNER,
        "source_dataset_slug": FROZEN_SOURCE,
        "wheelhouse_dataset_slug": FROZEN_WHEELS,
        "safe_evidence_dataset_slug": FROZEN_SAFE,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        build_v5_kaggle_notebooks(**arguments)  # type: ignore[arg-type]


def test_direct_and_module_generations_are_byte_identical_to_committed_files(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct"
    module = tmp_path / "module"
    commands = (
        [sys.executable, str(ROOT / "scripts/build_defense_v5_kaggle_notebooks.py")],
        [sys.executable, "-m", "scripts.build_defense_v5_kaggle_notebooks"],
    )
    for command, output in zip(commands, (direct, module), strict=True):
        completed = subprocess.run(
            [*command, "--root", str(ROOT), "--output", str(output)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == ""
        assert completed.stderr == ""
    committed = ROOT / "kaggle/defense_v5"
    expected_names = sorted(path.name for path in committed.iterdir())
    assert expected_names
    for directory in (direct, module):
        assert sorted(path.name for path in directory.iterdir()) == expected_names
        for name in expected_names:
            assert (directory / name).read_bytes() == (committed / name).read_bytes()


def test_generator_refuses_existing_output_or_unknown_cli_surface(
    tmp_path: Path,
) -> None:
    output = tmp_path / "notebooks"
    _generate(output)
    with pytest.raises(FileExistsError):
        _generate(output)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_defense_v5_kaggle_notebooks",
            "--root",
            str(ROOT),
            "--output",
            str(tmp_path / "other"),
            "--seed",
            "404",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
