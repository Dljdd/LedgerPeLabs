#!/usr/bin/env python3
"""Build the deterministic, evidence-bound presentation document for the APAR console."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from apar.compiler.compiler import compile_scenario
from apar.generators.campaigns import (
    APP_SCAM_MULE_MOTIF,
    CampaignParams,
    _CampaignEvaluator,
)
from apar.generators.population import PopulationEntity, PopulationGenerator
from apar.registry.models import ThreatCard

_RECOVERED_QUALIFIER = "Recovered diagnostic evidence — non-authoritative"
_CAMPAIGN_ID = "00000000-0000-4000-8000-000000000901"


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], document)


def _verify_self_hash(document: dict[str, Any], field: str, *, label: str) -> None:
    claimed = document.get(field)
    if not isinstance(claimed, str):
        raise ValueError(f"{label} self-hash is absent")
    unsigned = {key: value for key, value in document.items() if key != field}
    if _sha256(_canonical(unsigned)) != claimed:
        raise ValueError(f"{label} self-hash differs")


def _verify_portable(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = root / "demo" / "sentinel-v5"
    manifest = _load_object(bundle / "manifest.json", label="portable manifest")
    _verify_self_hash(manifest, "manifest_sha256", label="portable manifest")
    if (
        manifest.get("authoritative") is not False
        or manifest.get("accepted_capacity_evidence") is not False
        or manifest.get("demo_only") is not True
    ):
        raise ValueError("portable evidence flags differ")
    for relative, digest in cast(dict[str, str], manifest["bundle_files"]).items():
        path = bundle / relative
        if not path.is_file() or _sha256(path.read_bytes()) != digest:
            raise ValueError(f"portable file differs: {relative}")
    spec = _load_object(bundle / "spec.json", label="portable spec")
    scenarios = _load_object(bundle / "scenarios.json", label="portable scenarios")
    if spec.get("arm") != "ensemble_with_graph" or scenarios.get("arm") != spec.get("arm"):
        raise ValueError("portable arm must be ensemble_with_graph")
    return manifest, spec, scenarios


def _scenario_context(root: Path, threat_document: dict[str, Any]) -> dict[str, Any]:
    card = ThreatCard.model_validate(threat_document)
    config = card.default_config
    if config is None or config.seed != 260816:
        raise ValueError("approved APP scenario seed differs")
    bundle = compile_scenario(card, config)
    population = PopulationGenerator(seed=config.seed).generate(bundle)
    params = CampaignParams(
        campaign_id=_CAMPAIGN_ID,
        seed=config.seed,
        payment_count=10,
        target_illicit_rate=Decimal("0.70"),
        class_rate_tolerance=Decimal("0.05"),
        target_value_total=Decimal("500.00"),
        value_tolerance=Decimal("0.01"),
        min_amount=Decimal("10.00"),
        max_amount=Decimal("90.00"),
        currency="USD",
        duration_hours=12,
        query_budget=config.query_budget,
        min_delay_seconds=1,
        max_delay_seconds=300,
        expected_motif=APP_SCAM_MULE_MOTIF,
    )
    commands, evidence = _CampaignEvaluator(seed=config.seed).generate(
        "app_scam_mule", population, params
    )
    entity_by_id: dict[str, PopulationEntity] = {
        entity.entity_id: entity for entity in population.entities
    }
    used_entities: set[str] = set()
    edges: list[dict[str, object]] = []
    cumulative = Decimal("0.00")
    for index, command in enumerate(commands):
        if command.name != "a2a.initiate":
            continue
        payload = command.payload
        actor_id = cast(str, payload["actor_id"])
        counterparty_id = cast(str, payload["counterparty_id"])
        actor = entity_by_id[actor_id]
        counterparty = entity_by_id[counterparty_id]
        amount = cast(Decimal, payload["amount"])
        cumulative += amount
        used_entities.update((actor_id, counterparty_id))
        if actor.role == "mule" and counterparty.role == "mule":
            stage = "mule_layering"
        elif actor.role == "mule":
            stage = "cash_out"
        else:
            stage = "victim_transfer"
        edges.append(
            {
                "amount": str(amount),
                "cumulative_attempted_value": str(cumulative),
                "currency": cast(str, payload["currency"]),
                "event_time": evidence.schedule[index].isoformat().replace("+00:00", "Z"),
                "payment_id": cast(str, payload["payment_id"]),
                "source": actor_id,
                "source_account": cast(str, payload["payer_account"]),
                "stage": stage,
                "target": counterparty_id,
                "target_account": cast(str, payload["payee_account"]),
            }
        )
    role_order = {"victim": 0, "consumer": 0, "mule": 1, "attacker": 2}
    grouped: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for entity_id in sorted(used_entities):
        role = entity_by_id[entity_id].role
        grouped.setdefault(role_order.get(role, 2), []).append(entity_id)
    nodes: list[dict[str, object]] = []
    for layer in sorted(grouped):
        identifiers = grouped[layer]
        for row, entity_id in enumerate(identifiers):
            entity = entity_by_id[entity_id]
            nodes.append(
                {
                    "account_id": entity.account_id,
                    "country": entity.attributes["country"],
                    "id": entity.entity_id,
                    "illicit": entity.illicit,
                    "label": f"{entity.role.replace('_', ' ').title()} {row + 1}",
                    "role": entity.role,
                    "x": 90 + layer * 250,
                    "y": 72 + row * 94,
                }
            )
    if not edges or len(nodes) < 3:
        raise ValueError("generated APP graph is incomplete")
    return {
        "campaign_id": evidence.campaign_id,
        "case_grouping": {
            "basis": "generated_campaign_id",
            "case_id": f"case:{evidence.graph_digest}",
            "estimated_analyst_minutes": {"status": "evidence_pending"},
            "event_count": len(edges),
        },
        "family": evidence.family,
        "graph": {
            "edges": edges,
            "graph_sha256": evidence.graph_digest,
            "nodes": nodes,
        },
        "ledger_conserved": evidence.ledger_conserved,
        "motif_signature": evidence.motif_signature,
        "payment_count": evidence.payment_count,
        "schedule_sha256": evidence.schedule_digest,
        "seed": config.seed,
        "settled_value": str(evidence.settled_value),
        "synthetic": True,
        "value_total": str(evidence.value_total),
    }


def _threat_projection(threat: dict[str, Any]) -> dict[str, Any]:
    config = cast(dict[str, Any], threat["default_config"])
    return {
        "attacker_objective": threat["attacker_objective"],
        "channels": threat["channels"],
        "confidence": threat["confidence"],
        "default_config": {
            "attacker_mode": config["attacker_mode"],
            "campaign_stages": config["campaign_stages"],
            "duration_hours": config["duration_hours"],
            "economics": config["economics"],
            "event_ordering": cast(dict[str, Any], config["replay"])["event_ordering"],
            "export_level": config["export_level"],
            "feedback": config["feedback"],
            "query_budget": config["query_budget"],
            "seed": config["seed"],
            "viewpoint": config["viewpoint"],
        },
        "evidence": threat["evidence"],
        "family": threat["family"],
        "genai_capability": threat["genai_capability"],
        "implementation_status": threat["implementation_status"],
        "rails": threat["rails"],
        "safety_class": threat["safety_class"],
        "status": threat["status"],
        "threat_id": threat["threat_id"],
        "title": threat["title"],
    }


def _portable_projection(
    manifest: dict[str, Any], spec: dict[str, Any], scenarios: dict[str, Any]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for raw in cast(list[dict[str, Any]], scenarios["scenarios"]):
        records.append(
            {
                "accepted_checkpoint_evidence": raw["accepted_checkpoint_evidence"],
                "event_id": raw["event_id"],
                "model_input": {"features": raw["features"]},
                "post_event_truth": raw["presentation_ground_truth"],
            }
        )
    return {
        "accepted_capacity_evidence": manifest["accepted_capacity_evidence"],
        "arm": spec["arm"],
        "arm_spec_sha256": spec["arm_spec_sha256"],
        "authoritative": manifest["authoritative"],
        "bundle_manifest_sha256": manifest["manifest_sha256"],
        "demo_only": manifest["demo_only"],
        "feature_names": spec["feature_names"],
        "records": records,
        "source_checkpoint_manifest_sha256": manifest["source_checkpoint_manifest_sha256"],
        "switches": spec["switches"],
        "threshold_digest": spec["threshold_digest"],
        "thresholds": spec["thresholds"],
    }


def _recovered_projection(root: Path) -> dict[str, Any]:
    report = _load_object(
        root / "evidence" / "sentinel-v5-recovered-metrics" / "verified-report.json",
        label="recovered report",
    )
    receipt = _load_object(
        root / "evidence" / "sentinel-v5-recovered-metrics" / "source-rescue-receipt.json",
        label="recovered receipt",
    )
    _verify_self_hash(report, "verification_sha256", label="recovered report")
    if (
        report.get("authoritative") is not False
        or report.get("accepted_capacity_evidence") is not False
        or receipt.get("authoritative") is not False
        or receipt.get("accepted_capacity_evidence") is not False
    ):
        raise ValueError("recovered evidence flags differ")
    readiness = cast(dict[str, Any], report["readiness"])
    failed = [
        row
        for row in cast(list[dict[str, Any]], readiness["gates"])
        if row.get("passed") is False
    ]
    for name, passed, digest in cast(list[list[object]], readiness["qualifying_controls"]):
        if passed is False:
            failed.append(
                {
                    "metric": str(name),
                    "passed": False,
                    "source_sha256": str(digest),
                    "target": True,
                }
            )
    return {
        "accepted_capacity_evidence": report["accepted_capacity_evidence"],
        "arms": report["arms"],
        "authoritative": report["authoritative"],
        "failed_gates": failed,
        "first_missing_official_stage": report["first_missing_official_stage"],
        "official_chain_status": report["official_chain_status"],
        "official_predecessor_stage_manifests": report[
            "official_predecessor_stage_manifests"
        ],
        "qualifier": _RECOVERED_QUALIFIER,
        "readiness": readiness,
        "source_artifact_sha256": report["source_artifact_sha256"],
        "source_receipt_sha256": report["source_receipt_sha256"],
        "verification_sha256": report["verification_sha256"],
    }


def _trust_projection(root: Path) -> dict[str, Any]:
    test_path = root / "tests" / "trust" / "test_verifier.py"
    source_sha256 = _sha256(test_path.read_bytes())
    return {
        "checks": [
            {"check": "identity", "evidence": "AGENT_IDENTITY_MISMATCH", "status": "tested"},
            {"check": "mandate", "evidence": "MANDATE_SCOPE_VIOLATION", "status": "tested"},
            {"check": "scope", "evidence": "AMOUNT_LIMIT_EXCEEDED", "status": "tested"},
            {"check": "binding", "evidence": "CART_HASH_MISMATCH", "status": "tested"},
            {"check": "replay", "evidence": "NONCE_REPLAY", "status": "tested"},
        ],
        "implementation": "src/apar/trust/verifier.py",
        "separate_from_model_prediction": True,
        "test_evidence": "tests/trust/test_verifier.py",
        "test_evidence_sha256": source_sha256,
    }


def build_console_evidence(root: Path) -> dict[str, Any]:
    """Build one deterministic document without mutating source evidence."""
    root = root.resolve()
    threat = _load_object(
        root / "fixtures" / "threats" / "app-personalized-mule.json",
        label="APP threat card",
    )
    manifest, spec, scenarios = _verify_portable(root)
    document: dict[str, Any] = {
        "copy_boundary": {
            "evidence_seed": 404,
            "kaggle_locked_successor_run": False,
            "local_locked_attempt": "started_and_irreversibly_aborted",
            "no_candidate_manifest_chunks_or_judge_summary": True,
            "published_successful_seed_2404_result": False,
            "retry_permitted": False,
        },
        "portable": _portable_projection(manifest, spec, scenarios),
        "recovered": _recovered_projection(root),
        "scenario_context": _scenario_context(root, threat),
        "schema_version": "apar-console-evidence/1",
        "threat": _threat_projection(threat),
        "trust_proof": _trust_projection(root),
    }
    document["document_sha256"] = _sha256(_canonical(document))
    return document


def write_console_evidence(root: Path, output: Path) -> dict[str, Any]:
    """Write canonical presentation evidence and return the owned document."""
    document = build_console_evidence(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical(document))
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/public/data/console-evidence.json"),
    )
    args = parser.parse_args()
    document = write_console_evidence(args.root, args.output)
    print(
        json.dumps(
            {
                "document_sha256": document["document_sha256"],
                "output": str(args.output),
                "portable_arm": cast(dict[str, Any], document["portable"])["arm"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
