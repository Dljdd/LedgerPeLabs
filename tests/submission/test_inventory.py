from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.submission.inventory import validate_dependency_inventory
from scripts.submission.model import ReleaseError


def _write_inventory(root: Path, *, component_version: str = "1.2.3") -> None:
    (root / "requirements.txt").write_text("example-package==1.2.3\n")
    (root / "NOTICE.md").write_text("Project license status: unspecified.\n")
    (root / "sbom.json").write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "components": [
                    {
                        "licenses": [{"license": {"id": "MIT"}}],
                        "name": "example-package",
                        "purl": f"pkg:pypi/example-package@{component_version}",
                        "type": "library",
                        "version": component_version,
                    }
                ],
                "metadata": {
                    "properties": [
                        {"name": "apar:project-license-status", "value": "unspecified"},
                        {"name": "apar:web-dependencies-shipped", "value": "false"},
                    ]
                },
                "specVersion": "1.6",
                "version": 1,
            },
            sort_keys=True,
        )
    )


def test_inventory_matches_every_pinned_runtime_dependency(tmp_path: Path) -> None:
    """A dependency absent from the SBOM would make the shipped inventory incomplete."""
    _write_inventory(tmp_path)

    validate_dependency_inventory(
        requirements_path=tmp_path / "requirements.txt",
        sbom_path=tmp_path / "sbom.json",
        notice_path=tmp_path / "NOTICE.md",
        web_status="pending",
    )


def test_inventory_rejects_version_or_web_scope_drift(tmp_path: Path) -> None:
    """A stale SBOM or false web scope would misdescribe the clean-room environment."""
    _write_inventory(tmp_path, component_version="9.9.9")
    with pytest.raises(ReleaseError, match="SBOM components differ"):
        validate_dependency_inventory(
            requirements_path=tmp_path / "requirements.txt",
            sbom_path=tmp_path / "sbom.json",
            notice_path=tmp_path / "NOTICE.md",
            web_status="pending",
        )
    _write_inventory(tmp_path)
    sbom = json.loads((tmp_path / "sbom.json").read_text())
    sbom["metadata"]["properties"][1]["value"] = "true"
    (tmp_path / "sbom.json").write_text(json.dumps(sbom))
    with pytest.raises(ReleaseError, match="web dependency scope differs"):
        validate_dependency_inventory(
            requirements_path=tmp_path / "requirements.txt",
            sbom_path=tmp_path / "sbom.json",
            notice_path=tmp_path / "NOTICE.md",
            web_status="pending",
        )
