"""Cross-check judge-facing claims against the committed machine evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from scripts.submission.model import ReleaseError


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"claim source is unreadable: {path}") from error
    if not isinstance(document, dict):
        raise ReleaseError(f"claim source is not an object: {path}")
    return cast(dict[str, Any], document)


def verify_submission_claims(root: Path) -> dict[str, object]:
    """Require the selected arm, metrics, hashes, and evidence caveats in the pack."""
    repo = root.resolve()
    spec = _load(repo / "demo/sentinel-v5/spec.json")
    manifest = _load(repo / "demo/sentinel-v5/manifest.json")
    recovered = _load(
        repo / "evidence/sentinel-v5-recovered-metrics/verified-report.json"
    )
    if spec.get("arm") != "ensemble_with_graph":
        raise ReleaseError("portable selected arm is not ensemble_with_graph")
    if manifest.get("authoritative") is not False or manifest.get(
        "accepted_capacity_evidence"
    ) is not False:
        raise ReleaseError("portable evidence authority flags are unsafe")
    if recovered.get("authoritative") is not False or recovered.get(
        "first_missing_official_stage"
    ) != "70_metrics":
        raise ReleaseError("recovered evidence boundary differs")
    arms = {
        row["arm"]: row["aggregate"]
        for row in cast(list[dict[str, Any]], recovered["arms"])
    }
    graph = cast(dict[str, float], arms["ensemble_with_graph"])
    expected = {
        "99.867%": graph["recall"] * 100,
        "95.876%": graph["precision"] * 100,
        "97.831%": graph["f1"] * 100,
        "0.0037%": graph["false_decline_rate"] * 100,
        "0.572%": graph["challenge_rate"] * 100,
        "3.544 ms": graph["p95_latency_ms"],
    }
    model_card = (repo / "docs/submission/MODEL_CARD.md").read_text(encoding="utf-8")
    evaluation = (
        repo / "docs/submission/EVALUATION_AND_LIMITATIONS.md"
    ).read_text(encoding="utf-8")
    journey = (
        repo / "docs/submission/RESEARCH_AND_EXPERIMENT_JOURNEY.md"
    ).read_text(encoding="utf-8")
    corpus = "\n".join((model_card, evaluation, journey))
    for rendered, value in expected.items():
        decimals = 4 if rendered == "0.0037%" else 3
        observed = f"{value:.{decimals}f}"
        if observed not in rendered:
            raise ReleaseError(f"internal claim formatter differs for {rendered}")
        if rendered not in corpus:
            raise ReleaseError(f"judge-facing metric claim is absent: {rendered}")
    required_phrases = (
        "non-authoritative",
        "official chain remains incomplete at Stage 70",
        "not the champion",
        "Synthetic only",
        manifest["manifest_sha256"],
    )
    for phrase in required_phrases:
        if phrase not in corpus:
            raise ReleaseError(f"judge-facing evidence boundary is absent: {phrase}")
    return {
        "arm": spec["arm"],
        "bundle_manifest_sha256": manifest["manifest_sha256"],
        "checked_metric_count": len(expected),
        "official_chain_complete": False,
        "verified": True,
    }
