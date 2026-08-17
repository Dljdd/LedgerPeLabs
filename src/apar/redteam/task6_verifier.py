"""Canonical Task 6 raw-evidence export and independent verification.

The verifier consumes serialized public search documents and evaluator-owned traces. It
does not hold a policy capability and never invokes adaptive search.
"""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import NoReturn, cast

from apar.contracts.decisions import Action
from apar.redteam.benchmark import BenchmarkObservation, DefenderRuleSet
from apar.redteam.llm_policy import LLMAuditRecord
from apar.redteam.policies import ParameterBounds
from apar.redteam.search import (
    EvaluationContract,
    PolicyBinding,
    SearchResult,
)

_HEX = frozenset("0123456789abcdef")
_SEARCH_RESULT_FIELDS = frozenset(
    {
        "family",
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        "disclosure_profile_digest",
        "evaluation_contract_digest",
        "authority_id",
        "evaluator_capability_id",
        "evaluator_code_digest",
        "policy_capability_id",
        "policy_name",
        "policy_version",
        "policy_code_digest",
        "policy_callable_digest",
        "run_group_id",
        "result_id",
        "seed",
        "proposals",
        "trials",
        "objective_values",
        "winner",
        "proposal_budget",
        "query_budget",
        "logical_time_budget",
        "wall_time_budget_ms",
        "proposals_used",
        "queries_used",
        "logical_time_used",
        "wall_time_elapsed_ms",
        "wall_time_exhausted",
        "wall_time_overrun_ms",
    }
)
_CELL_FIELDS = frozenset(
    {
        "cell_id",
        "cell_kind",
        "family",
        "policy_name",
        "seed",
        "public_context",
        "search_result",
        "candidate_sequence",
        "evaluation_traces",
        "evaluation_trace_digest",
        "llm_audit_attempts",
        "llm_audit_digest",
        "cell_digest",
    }
)
_TRACE_CORE_FIELDS = frozenset(
    {
        "cell_id",
        "family",
        "seed",
        "candidate_id",
        "candidate_document_digest",
        "command_digest",
        "command_count",
        "command_type_counts",
        "event_digest",
        "event_count",
        "event_type_counts",
        "ledger_digest",
        "ledger_entry_count",
        "fresh_replay_succeeded",
        "ledger_conserved",
        "derived_feature_vector",
        "matched_defender_rule",
        "decision",
        "role_bound_value_components",
        "executed_role_bound_value",
        "feedback_realized_value",
    }
)
_LLM_FIELDS = frozenset(
    {
        "cell_id",
        "seed",
        "provider",
        "model_id",
        "policy_version",
        "schema_digest",
        "prompt_digest",
        "response_digest",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "call_status",
        "failure_family",
        "cache_hit",
    }
)


class EvidenceVerificationError(ValueError):
    """The raw artifact is incomplete, inconsistent, or not canonical."""


def _fail(message: str) -> NoReturn:
    raise EvidenceVerificationError(message)


def _assert_exact_json(value: object, *, path: str = "document") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_exact_json(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail(f"{path} contains a non-exact string key")
        for key, item in value.items():
            _assert_exact_json(item, path=f"{path}.{key}")
        return
    _fail(f"{path} contains a non-exact JSON value")


def _canonical_payload(value: object) -> bytes:
    _assert_exact_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one strict JSON value in the artifact's canonical byte form."""
    return _canonical_payload(value) + b"\n"


def canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def strict_json_loads(raw: bytes, *, require_canonical: bool = True) -> object:
    """Load strict JSON while rejecting duplicate keys and noncanonical encodings."""
    if type(raw) is not bytes:
        _fail("JSON input must be exact bytes")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: _fail(f"non-finite JSON constant: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceVerificationError("evidence is not valid UTF-8 JSON") from error
    _assert_exact_json(value)
    if require_canonical and canonical_json_bytes(value) != raw:
        _fail("evidence bytes are not canonical JSON")
    return value


def _exact_dict(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        _fail(f"{label} field set is not exact")
    if any(type(key) is not str for key in value):
        _fail(f"{label} keys are not exact strings")
    return cast(dict[str, object], value)


def _exact_list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{label} must be an exact list")
    return cast(list[object], value)


def _exact_text(value: object, *, label: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _fail(f"{label} must be exact text")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_hex(value: object, *, label: str) -> str:
    text = _exact_text(value, label=label)
    if len(text) != 64 or not set(text) <= _HEX:
        _fail(f"{label} must be lowercase SHA-256 text")
    return text


def _decimal(value: object, *, label: str, nonnegative: bool = False) -> Decimal:
    text = _exact_text(value, label=label)
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise EvidenceVerificationError(f"{label} is not Decimal text") from error
    if not number.is_finite() or (nonnegative and number < 0) or str(number) != text:
        _fail(f"{label} is not canonical finite Decimal text")
    return number


def _evaluation_contract_document(contract: EvaluationContract) -> dict[str, object]:
    return {
        "family": contract.family,
        "bounds_digest": contract.bounds_digest,
        "hidden_template_digest": contract.hidden_template_digest,
        "background_digest": contract.background_digest,
        "population_digest": contract.population_digest,
        "evaluator_digest": contract.evaluator_digest,
        "defender_digest": contract.defender_digest,
        "disclosure_profile": {
            "profile_id": contract.disclosure_profile.profile_id,
            "expose_realized_value": (
                contract.disclosure_profile.expose_realized_value
            ),
            "profile_digest": contract.disclosure_profile_digest,
        },
        "contract_digest": contract.contract_digest,
    }


def build_search_cell_document(
    *,
    cell_kind: str,
    result: SearchResult,
    public_bounds: ParameterBounds,
    evaluation_contract: EvaluationContract,
    policy_binding: PolicyBinding,
    defender: DefenderRuleSet,
    evaluation_traces: tuple[BenchmarkObservation, ...],
    llm_audit_records: tuple[LLMAuditRecord, ...],
) -> dict[str, object]:
    """Export one complete cell without trusting any aggregate report."""
    if cell_kind not in {"target", "negative_control"} or type(cell_kind) is not str:
        raise TypeError("cell_kind must be exact target or negative_control text")
    if type(result) is not SearchResult:
        raise TypeError("result must be an exact SearchResult")
    if type(public_bounds) is not ParameterBounds:
        raise TypeError("public_bounds must be exact")
    if type(evaluation_contract) is not EvaluationContract:
        raise TypeError("evaluation_contract must be exact")
    if type(policy_binding) is not PolicyBinding:
        raise TypeError("policy_binding must be exact")
    if type(defender) is not DefenderRuleSet:
        raise TypeError("defender must be exact")
    if type(evaluation_traces) is not tuple or any(
        type(trace) is not BenchmarkObservation for trace in evaluation_traces
    ):
        raise TypeError("evaluation_traces must be an exact tuple")
    if type(llm_audit_records) is not tuple or any(
        type(record) is not LLMAuditRecord for record in llm_audit_records
    ):
        raise TypeError("llm_audit_records must be an exact tuple")
    if len(evaluation_traces) != len(result.trials):
        raise ValueError("every trial must have exactly one evaluator-owned trace")

    search_document = result.canonical_document()
    cell_id = canonical_digest(
        {
            "cell_kind": cell_kind,
            "family": result.family,
            "policy_name": result.policy_name,
            "seed": result.seed,
        }
    )
    trace_documents: list[dict[str, object]] = []
    for index, trace in enumerate(evaluation_traces):
        trace_core = {"cell_id": cell_id, "seed": result.seed, **trace.document()}
        trace_documents.append(
            {
                "trial_index": index,
                **trace_core,
                "trace_digest": canonical_digest(trace_core),
            }
        )
    audit_documents: list[dict[str, object]] = []
    for index, record in enumerate(llm_audit_records):
        audit_core = {
            "cell_id": cell_id,
            "seed": result.seed,
            **cast(dict[str, object], record.model_dump(mode="json")),
        }
        audit_documents.append(
            {
                "attempt_index": index,
                "trial_index": index if index < len(result.trials) else None,
                **audit_core,
                "record_digest": canonical_digest(audit_core),
            }
        )
    candidate_sequence = [
        {
            "trial_index": index,
            "adaptive_vector": candidate.params.document(),
            "vector_fingerprint": candidate.fingerprint,
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "generation": candidate.generation,
        }
        for index, candidate in enumerate(result.proposals)
    ]
    cell_core: dict[str, object] = {
        "cell_id": cell_id,
        "cell_kind": cell_kind,
        "family": result.family,
        "policy_name": result.policy_name,
        "seed": result.seed,
        "public_context": {
            "public_bounds": public_bounds.document(),
            "evaluation_contract": _evaluation_contract_document(evaluation_contract),
            "evaluator_binding": {
                "capability_id": result.evaluator_capability_id,
                "code_digest": result.evaluator_code_digest,
            },
            "policy_binding": policy_binding.model_dump(mode="json"),
            "defender": {
                "version": defender.version,
                "rules": [rule.document() for rule in defender.rules],
                "defender_digest": defender.defender_digest,
            },
        },
        "search_result": {
            "document": search_document,
            "canonical_document_digest": canonical_digest(search_document),
            "process_local_issuance_seal": result.result_seal,
            "seal_scope": "process_local_nonportable_hmac",
        },
        "candidate_sequence": candidate_sequence,
        "evaluation_traces": trace_documents,
        "evaluation_trace_digest": canonical_digest(trace_documents),
        "llm_audit_attempts": audit_documents,
        "llm_audit_digest": canonical_digest(audit_documents),
    }
    return {**cell_core, "cell_digest": canonical_digest(cell_core)}


def _tagged_vector_to_raw(value: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    entries = _exact_list(value, label="tagged adaptive vector")
    raw_entries: list[dict[str, object]] = []
    tagged_entries: list[dict[str, object]] = []
    names: list[str] = []
    for raw_entry in entries:
        entry = _exact_dict(
            raw_entry,
            frozenset({"name", "value"}),
            label="tagged adaptive entry",
        )
        name = _exact_text(entry["name"], label="adaptive parameter name")
        tagged = _exact_dict(
            entry["value"],
            frozenset({"type", "value"}),
            label="tagged adaptive value",
        )
        kind = _exact_text(tagged["type"], label="adaptive value type")
        raw_value = tagged["value"]
        if kind == "integer":
            _exact_int(raw_value, label="adaptive integer", minimum=0)
        elif kind in {"decimal", "string"}:
            _exact_text(raw_value, label=f"adaptive {kind}")
            if kind == "decimal":
                _decimal(raw_value, label="adaptive decimal")
        elif kind == "string_tuple":
            tuple_value = _exact_list(raw_value, label="adaptive string tuple")
            if not tuple_value or any(type(item) is not str for item in tuple_value):
                _fail("adaptive string tuple is not exact")
        else:
            _fail("adaptive value type is undeclared")
        names.append(name)
        raw_entries.append({"name": name, "value": raw_value})
        tagged_entries.append(cast(dict[str, object], raw_entry))
    if names != sorted(set(names)):
        _fail("adaptive vector names are not unique and sorted")
    return {"entries": raw_entries}, tagged_entries


def _vector_json_mapping(value: object) -> dict[str, object]:
    vector = _exact_dict(value, frozenset({"entries"}), label="adaptive vector")
    entries = _exact_list(vector["entries"], label="adaptive vector entries")
    result: dict[str, object] = {}
    names: list[str] = []
    for raw in entries:
        entry = _exact_dict(
            raw,
            frozenset({"name", "value"}),
            label="adaptive vector entry",
        )
        name = _exact_text(entry["name"], label="adaptive vector entry name")
        names.append(name)
        result[name] = entry["value"]
    if names != sorted(set(names)):
        _fail("adaptive vector entry names must be unique and sorted")
    return result


def _untagged_value(value: object, *, label: str) -> object:
    tagged = _exact_dict(value, frozenset({"type", "value"}), label=label)
    kind = _exact_text(tagged["type"], label=f"{label} type")
    raw = tagged["value"]
    if kind == "integer":
        return _exact_int(raw, label=label)
    if kind == "decimal":
        _decimal(raw, label=label)
        return raw
    if kind == "string":
        return _exact_text(raw, label=label)
    if kind == "string_tuple":
        items = _exact_list(raw, label=label)
        if not items or any(type(item) is not str for item in items):
            _fail(f"{label} string tuple is not exact")
        return items
    _fail(f"{label} type is undeclared")


def _bounds_schema_document(value: dict[str, object]) -> dict[str, object]:
    domains = _exact_list(value["domains"], label="bounds domains")
    parameters: list[dict[str, object]] = []
    names: list[str] = []
    for raw in domains:
        domain = _exact_dict(
            raw,
            frozenset({"name", "kind", "values"}),
            label="bounds domain",
        )
        name = _exact_text(domain["name"], label="bounds domain name")
        kind = _exact_text(domain["kind"], label="bounds domain kind")
        values = _exact_list(domain["values"], label="bounds domain values")
        if not values:
            _fail("bounds domain values must not be empty")
        names.append(name)
        parameters.append(
            {
                "name": name,
                "kind": kind,
                "allowed_values": [
                    _untagged_value(item, label="bounds domain value") for item in values
                ],
            }
        )
    if names != sorted(set(names)):
        _fail("bounds domain names must be unique and sorted")
    return {
        "family": value["family"],
        "parameters": parameters,
        "required": names,
        "additional_parameters": False,
    }


def _bounds_context(value: object) -> tuple[str, dict[bytes, list[dict[str, object]]]]:
    bounds = _exact_dict(
        value,
        frozenset({"family", "defaults", "domains", "feasible_vectors"}),
        label="public bounds",
    )
    family = _exact_text(bounds["family"], label="bounds family")
    _tagged_vector_to_raw(bounds["defaults"])
    _exact_list(bounds["domains"], label="bounds domains")
    feasible = _exact_list(bounds["feasible_vectors"], label="feasible vectors")
    vectors: dict[bytes, list[dict[str, object]]] = {}
    for item in feasible:
        raw, tagged = _tagged_vector_to_raw(item)
        key = _canonical_payload(raw)
        if key in vectors:
            _fail("public bounds contain duplicate feasible vectors")
        vectors[key] = tagged
    if not vectors:
        _fail("public bounds need at least one feasible vector")
    return family, vectors


def _validate_contract(
    value: object,
    *,
    family: str,
    bounds_digest: str,
) -> dict[str, object]:
    contract = _exact_dict(
        value,
        frozenset(
            {
                "family",
                "bounds_digest",
                "hidden_template_digest",
                "background_digest",
                "population_digest",
                "evaluator_digest",
                "defender_digest",
                "disclosure_profile",
                "contract_digest",
            }
        ),
        label="evaluation contract",
    )
    if contract["family"] != family or contract["bounds_digest"] != bounds_digest:
        _fail("evaluation contract family or bounds provenance differs")
    for name in (
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        "contract_digest",
    ):
        _exact_hex(contract[name], label=f"contract {name}")
    disclosure = _exact_dict(
        contract["disclosure_profile"],
        frozenset({"profile_id", "expose_realized_value", "profile_digest"}),
        label="disclosure profile",
    )
    profile_id = _exact_text(disclosure["profile_id"], label="disclosure profile ID")
    expose = disclosure["expose_realized_value"]
    if type(expose) is not bool:
        _fail("disclosure exposure must be exact bool")
    expected_profile = canonical_digest(
        {"profile_id": profile_id, "expose_realized_value": expose}
    )
    if disclosure["profile_digest"] != expected_profile:
        _fail("disclosure profile digest differs")
    expected_contract = canonical_digest(
        {
            "family": family,
            "bounds_digest": bounds_digest,
            "hidden_template_digest": contract["hidden_template_digest"],
            "background_digest": contract["background_digest"],
            "population_digest": contract["population_digest"],
            "evaluator_digest": contract["evaluator_digest"],
            "defender_digest": contract["defender_digest"],
            "disclosure_profile_digest": expected_profile,
        }
    )
    if contract["contract_digest"] != expected_contract:
        _fail("evaluation contract digest differs")
    return contract


def _candidate_identity(
    value: object,
    feasible: dict[bytes, list[dict[str, object]]],
) -> tuple[str, str, str | None, int]:
    candidate = _exact_dict(
        value,
        frozenset({"params", "parent_id", "generation"}),
        label="candidate",
    )
    params = _exact_dict(
        candidate["params"], frozenset({"entries"}), label="candidate params"
    )
    key = _canonical_payload(params)
    tagged = feasible.get(key)
    if tagged is None:
        _fail("candidate adaptive vector is outside public bounds")
    fingerprint = canonical_digest(tagged)
    parent = candidate["parent_id"]
    if parent is not None:
        parent = _exact_hex(parent, label="candidate parent ID")
    generation = _exact_int(candidate["generation"], label="candidate generation")
    candidate_id = canonical_digest(
        {"fingerprint": fingerprint, "generation": generation, "parent_id": parent}
    )
    return candidate_id, fingerprint, parent, generation


def _feedback_objective(value: object) -> tuple[str, str, str | None, Decimal]:
    feedback = _exact_dict(
        value,
        frozenset({"action", "reason_family", "realized_value"}),
        label="feedback",
    )
    action = _exact_text(feedback["action"], label="feedback action")
    if action not in {item.value for item in Action}:
        _fail("feedback action is undeclared")
    reason = _exact_text(feedback["reason_family"], label="feedback reason")
    if (action == Action.APPROVE.value) != (reason == "approved"):
        _fail("feedback approval and reason are inconsistent")
    raw_realized = feedback["realized_value"]
    realized = Decimal(0)
    if raw_realized is not None:
        realized = _decimal(raw_realized, label="feedback realized value", nonnegative=True)
    penalty = {
        Action.APPROVE.value: Decimal(0),
        Action.CHALLENGE.value: Decimal("0.25"),
        Action.DECLINE.value: Decimal(1),
    }[action]
    return action, reason, cast(str | None, raw_realized), realized - penalty


def _validate_count_records(
    value: object,
    *,
    key_name: str,
    label: str,
) -> int:
    records = _exact_list(value, label=label)
    names: list[str] = []
    total = 0
    for raw in records:
        record = _exact_dict(
            raw,
            frozenset({key_name, "count"}),
            label=f"{label} record",
        )
        names.append(_exact_text(record[key_name], label=f"{label} name"))
        total += _exact_int(record["count"], label=f"{label} count", minimum=1)
    if names != sorted(set(names)):
        _fail(f"{label} names must be unique and sorted")
    return total


def _validate_defender(value: object, *, contract_digest: str) -> tuple[list[object], str]:
    defender = _exact_dict(
        value,
        frozenset({"version", "rules", "defender_digest"}),
        label="defender",
    )
    version = _exact_text(defender["version"], label="defender version")
    rules = _exact_list(defender["rules"], label="defender rules")
    expected = canonical_digest({"version": version, "rules": rules})
    if defender["defender_digest"] != expected or expected != contract_digest:
        _fail("defender provenance digest differs")
    return rules, expected


def _matched_rule_document(
    rules: list[object],
    *,
    family: str,
    features: dict[str, Decimal],
) -> dict[str, object] | None:
    triggered: list[dict[str, object]] = []
    for raw in rules:
        rule = _exact_dict(
            raw,
            frozenset({"family", "feature", "threshold", "action", "reason_family"}),
            label="defender rule",
        )
        feature = _exact_text(rule["feature"], label="defender feature")
        threshold = _decimal(rule["threshold"], label="defender threshold")
        action = _exact_text(rule["action"], label="defender action")
        _exact_text(rule["reason_family"], label="defender reason")
        if action not in {Action.CHALLENGE.value, Action.DECLINE.value}:
            _fail("defender rule action is not challenge or decline")
        if rule["family"] == family and feature in features and features[feature] >= threshold:
            triggered.append(rule)
    if not triggered:
        return None
    severity = {Action.CHALLENGE.value: 1, Action.DECLINE.value: 2}
    return min(
        triggered,
        key=lambda rule: (
            -severity[cast(str, rule["action"])],
            cast(str, rule["reason_family"]),
            cast(str, rule["feature"]),
        ),
    )


def _verify_trace(
    value: object,
    *,
    index: int,
    cell_id: str,
    family: str,
    seed: int,
    candidate: object,
    candidate_id: str,
    feedback: object,
    disclosure_exposes_value: bool,
    rules: list[object],
) -> None:
    trace = _exact_dict(
        value,
        _TRACE_CORE_FIELDS | frozenset({"trial_index", "trace_digest"}),
        label="evaluation trace",
    )
    if trace["trial_index"] != index:
        _fail("evaluation trace trial order differs")
    core = {
        key: trace[key]
        for key in trace
        if key not in {"trial_index", "trace_digest"}
    }
    if trace["trace_digest"] != canonical_digest(core):
        _fail("evaluation trace digest differs")
    if (
        trace["cell_id"] != cell_id
        or trace["seed"] != seed
        or trace["family"] != family
        or trace["candidate_id"] != candidate_id
    ):
        _fail("evaluation trace candidate or family differs")
    if trace["candidate_document_digest"] != canonical_digest(candidate):
        _fail("evaluation trace candidate document digest differs")
    command_count = _exact_int(trace["command_count"], label="trace command count")
    event_count = _exact_int(trace["event_count"], label="trace event count")
    ledger_count = _exact_int(trace["ledger_entry_count"], label="trace ledger count")
    if _validate_count_records(
        trace["command_type_counts"], key_name="name", label="command type counts"
    ) != command_count:
        _fail("trace command count differs from command type counts")
    if _validate_count_records(
        trace["event_type_counts"], key_name="event_type", label="event type counts"
    ) != event_count:
        _fail("trace event count differs from event type counts")
    for name in ("command_digest", "event_digest", "ledger_digest"):
        _exact_hex(trace[name], label=f"trace {name}")
    if type(trace["fresh_replay_succeeded"]) is not bool or type(
        trace["ledger_conserved"]
    ) is not bool:
        _fail("trace replay/conservation flags must be exact bools")
    feature_records = _exact_list(
        trace["derived_feature_vector"], label="derived feature vector"
    )
    features: dict[str, Decimal] = {}
    names: list[str] = []
    for raw in feature_records:
        record = _exact_dict(
            raw,
            frozenset({"name", "value"}),
            label="derived feature",
        )
        name = _exact_text(record["name"], label="derived feature name")
        names.append(name)
        features[name] = _decimal(record["value"], label="derived feature value")
    if names != sorted(set(names)):
        _fail("derived feature vector must be unique and sorted")
    decision = _exact_dict(
        trace["decision"],
        frozenset({"action", "reason_family"}),
        label="trace decision",
    )
    action = _exact_text(decision["action"], label="trace decision action")
    reason = _exact_text(decision["reason_family"], label="trace decision reason")
    public_action, public_reason, public_realized, _objective = _feedback_objective(feedback)
    if action != public_action or reason != public_reason:
        _fail("trace decision differs from public feedback")

    replay_succeeded = trace["fresh_replay_succeeded"]
    if not replay_succeeded:
        if (
            trace["ledger_conserved"] is not False
            or command_count != 0
            or event_count != 0
            or ledger_count != 0
            or features
            or trace["matched_defender_rule"] is not None
            or action != Action.DECLINE.value
            or reason != "invalid_candidate"
        ):
            _fail("failed evaluation trace is not canonical")
    else:
        if trace["ledger_conserved"] is not True:
            _fail("successful trace lacks conserved ledger evidence")
        matched = _matched_rule_document(rules, family=family, features=features)
        if trace["matched_defender_rule"] != matched:
            _fail("trace matched defender rule differs from frozen decision")
        expected_action = Action.APPROVE.value if matched is None else matched["action"]
        expected_reason = "approved" if matched is None else matched["reason_family"]
        if action != expected_action or reason != expected_reason:
            _fail("trace decision differs from frozen defender rule")

    components = _exact_list(
        trace["role_bound_value_components"], label="role-bound value components"
    )
    payment_ids: list[str] = []
    outstanding_total = Decimal(0)
    for raw in components:
        component = _exact_dict(
            raw,
            frozenset(
                {
                    "payment_id",
                    "positive_value",
                    "removed_value",
                    "outstanding_value",
                    "outstanding_minor_units",
                }
            ),
            label="role-bound value component",
        )
        payment_id = _exact_text(component["payment_id"], label="payment ID")
        payment_ids.append(payment_id)
        positive = _decimal(
            component["positive_value"], label="positive role-bound value", nonnegative=True
        )
        removed = _decimal(
            component["removed_value"], label="removed role-bound value", nonnegative=True
        )
        outstanding = _decimal(
            component["outstanding_value"],
            label="outstanding role-bound value",
            nonnegative=True,
        )
        if positive - removed != outstanding:
            _fail("role-bound value component does not conserve principal")
        minor_units = _exact_int(
            component["outstanding_minor_units"], label="outstanding minor units"
        )
        if outstanding * 100 != minor_units:
            _fail("role-bound value minor units differ")
        outstanding_total += outstanding
    if payment_ids != sorted(set(payment_ids)):
        _fail("role-bound value components must be unique and sorted")
    executed = _decimal(
        trace["executed_role_bound_value"],
        label="executed role-bound value",
        nonnegative=True,
    )
    if outstanding_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN) != executed:
        _fail("role-bound value components differ from final realized value")
    raw_feedback_value = trace["feedback_realized_value"]
    if raw_feedback_value is None:
        trace_feedback_value = None
    else:
        trace_feedback_value = _decimal(
            raw_feedback_value, label="trace feedback realized value", nonnegative=True
        )
    expected_trace_value = executed if action == Action.APPROVE.value else Decimal("0.00")
    if replay_succeeded and trace_feedback_value != expected_trace_value:
        _fail("trace feedback value differs from executed decision value")
    if not replay_succeeded and trace_feedback_value is not None:
        _fail("failed trace must expose no evaluator value")
    expected_public_value = (
        None
        if not disclosure_exposes_value
        else None if trace_feedback_value is None else str(trace_feedback_value)
    )
    if public_realized != expected_public_value:
        _fail("public feedback violates the frozen disclosure profile")


def _verify_llm_attempts(
    value: object,
    *,
    policy_name: str,
    cell_id: str,
    seed: int,
    trial_count: int,
    public_bounds: object,
    proposals: list[object],
    trials: list[object],
    expected_cache: object | None,
) -> None:
    attempts = _exact_list(value, label="LLM audit attempts")
    if policy_name != "cached_llm" and attempts:
        _fail("non-LLM cell contains LLM audit attempts")
    if policy_name == "cached_llm" and len(attempts) != trial_count:
        _fail("cached-LLM audit attempt count differs from trials")
    cache: dict[str, object] | None = None
    if expected_cache is not None:
        if type(expected_cache) is not dict or any(
            type(key) is not str for key in expected_cache
        ):
            _fail("frozen LLM cache must be an exact string-keyed object")
        cache = cast(dict[str, object], expected_cache)
    bounds = _exact_dict(
        public_bounds,
        frozenset({"family", "defaults", "domains", "feasible_vectors"}),
        label="LLM public bounds",
    )
    _bounds_family, feasible = _bounds_context(bounds)
    schema = _bounds_schema_document(bounds)
    schema_digest = canonical_digest(
        {
            "protocol": "apar-decision-only-planner-v2",
            "bounds": schema,
            "response_fields": ["generation", "params", "parent_id"],
        }
    )
    visible_history: list[dict[str, object]] = []
    for index, raw in enumerate(attempts):
        attempt = _exact_dict(
            raw,
            _LLM_FIELDS | frozenset({"attempt_index", "trial_index", "record_digest"}),
            label="LLM audit attempt",
        )
        if attempt["attempt_index"] != index or attempt["trial_index"] != index:
            _fail("LLM audit attempt is not bound to its trial")
        if attempt["cell_id"] != cell_id or attempt["seed"] != seed:
            _fail("LLM audit attempt is not bound to its cell")
        core = {
            key: attempt[key]
            for key in attempt
            if key not in {"attempt_index", "trial_index", "record_digest"}
        }
        if attempt["record_digest"] != canonical_digest(core):
            _fail("LLM audit record digest differs")
        for name in ("provider", "model_id", "policy_version", "call_status"):
            _exact_text(attempt[name], label=f"LLM {name}")
        for name in ("schema_digest", "prompt_digest", "response_digest"):
            _exact_hex(attempt[name], label=f"LLM {name}")
        for name in ("latency_ms", "input_tokens", "output_tokens"):
            _exact_int(attempt[name], label=f"LLM {name}")
        if type(attempt["cache_hit"]) is not bool:
            _fail("LLM cache_hit must be exact bool")
        success = cast(str, attempt["call_status"]).endswith("_success")
        if success != (attempt["failure_family"] is None):
            _fail("LLM status and failure family differ")
        if attempt["cache_hit"] != cast(str, attempt["call_status"]).startswith("cache_"):
            _fail("LLM status and cache evidence differ")
        if attempt["schema_digest"] != schema_digest:
            _fail("LLM schema digest differs from public bounds")
        expected_prompt = {
            "protocol": "apar-decision-only-planner-v2",
            "bounds": schema,
            "history": visible_history,
            "response_contract": {
                "required": ["generation", "params", "parent_id"],
                "additional_fields": False,
            },
        }
        if attempt["prompt_digest"] != canonical_digest(expected_prompt):
            _fail("LLM prompt digest differs from the public trial prefix")
        if cache is not None:
            cached = _exact_dict(
                cache.get(attempt["prompt_digest"]),
                frozenset(
                    {
                        "provider",
                        "model_id",
                        "policy_version",
                        "schema_digest",
                        "response_digest",
                        "transport",
                    }
                ),
                label="frozen LLM cache record",
            )
            transport = _exact_dict(
                cached["transport"],
                frozenset({"latency_ms", "input_tokens", "output_tokens", "output"}),
                label="frozen LLM cache transport",
            )
            output = _exact_dict(
                transport["output"],
                frozenset({"generation", "params", "parent_id"}),
                label="frozen LLM cache output",
            )
            proposal = _exact_dict(
                proposals[index],
                frozenset({"params", "parent_id", "generation"}),
                label="cached-LLM proposal",
            )
            proposal_params = _vector_json_mapping(proposal["params"])
            if output != {
                "generation": proposal["generation"],
                "params": proposal_params,
                "parent_id": proposal["parent_id"],
            }:
                _fail("cached LLM response is not the emitted trial proposal")
            if (
                cached["provider"] != attempt["provider"]
                or cached["model_id"] != attempt["model_id"]
                or cached["policy_version"] != attempt["policy_version"]
                or cached["schema_digest"] != attempt["schema_digest"]
                or cached["response_digest"] != attempt["response_digest"]
                or cached["response_digest"] != canonical_digest(output)
                or attempt["latency_ms"] != 0
                or attempt["input_tokens"] != transport["input_tokens"]
                or attempt["output_tokens"] != transport["output_tokens"]
                or attempt["call_status"] != "cache_success"
                or attempt["failure_family"] is not None
                or attempt["cache_hit"] is not True
            ):
                _fail("LLM audit attempt differs from its frozen cache record")
        trial = _exact_dict(
            trials[index],
            frozenset({"candidate", "feedback", "objective_value"}),
            label="cached-LLM visible trial",
        )
        proposal = _exact_dict(
            proposals[index],
            frozenset({"params", "parent_id", "generation"}),
            label="cached-LLM proposal",
        )
        feedback = _exact_dict(
            trial["feedback"],
            frozenset({"action", "reason_family", "realized_value"}),
            label="cached-LLM public feedback",
        )
        tagged = feasible.get(_canonical_payload(proposal["params"]))
        if tagged is None:
            _fail("cached-LLM proposal is outside frozen public bounds")
        visible_history.append(
            {
                "candidate_id": canonical_digest(
                    {
                        "fingerprint": canonical_digest(tagged),
                        "generation": proposal["generation"],
                        "parent_id": proposal["parent_id"],
                    }
                ),
                "parent_id": proposal["parent_id"],
                "generation": proposal["generation"],
                "params": _vector_json_mapping(proposal["params"]),
                "feedback": {
                    "action": feedback["action"],
                    "reason_family": feedback["reason_family"],
                    "realized_value": feedback["realized_value"],
                },
                "objective_value": trial["objective_value"],
            }
        )


def verify_search_cell(
    value: object,
    *,
    expected_cell_kind: str | None = None,
    expected_family: str | None = None,
    expected_policy: str | None = None,
    expected_seed: int | None = None,
    expected_budget: int | None = None,
    expected_wall_time_budget_ms: int | None = None,
    expected_llm_cache: object | None = None,
) -> dict[str, object]:
    """Recompute one cell's candidate, feedback, budget, trace, and value evidence."""
    _assert_exact_json(value, path="search_cell")
    cell = _exact_dict(value, _CELL_FIELDS, label="search cell")
    core = {key: cell[key] for key in cell if key != "cell_digest"}
    if cell["cell_digest"] != canonical_digest(core):
        _fail("search cell digest differs")
    cell_kind = _exact_text(cell["cell_kind"], label="cell kind")
    family = _exact_text(cell["family"], label="cell family")
    policy_name = _exact_text(cell["policy_name"], label="cell policy")
    seed = _exact_int(cell["seed"], label="cell seed")
    expected_cell_id = canonical_digest(
        {
            "cell_kind": cell_kind,
            "family": family,
            "policy_name": policy_name,
            "seed": seed,
        }
    )
    if cell["cell_id"] != expected_cell_id:
        _fail("search cell identity digest differs")
    for actual, expected, label in (
        (cell_kind, expected_cell_kind, "cell kind"),
        (family, expected_family, "family"),
        (policy_name, expected_policy, "policy"),
        (seed, expected_seed, "seed"),
    ):
        if expected is not None and actual != expected:
            _fail(f"search cell {label} differs from expected")

    context = _exact_dict(
        cell["public_context"],
        frozenset(
            {
                "public_bounds",
                "evaluation_contract",
                "evaluator_binding",
                "policy_binding",
                "defender",
            }
        ),
        label="public context",
    )
    bounds_family, feasible = _bounds_context(context["public_bounds"])
    if bounds_family != family:
        _fail("public bounds family differs from cell")
    bounds_digest = canonical_digest(context["public_bounds"])
    contract = _validate_contract(
        context["evaluation_contract"], family=family, bounds_digest=bounds_digest
    )
    policy = _exact_dict(
        context["policy_binding"],
        frozenset({"name", "version", "capability_id", "code_digest", "callable_digest"}),
        label="policy binding",
    )
    if policy["name"] != policy_name:
        _fail("policy binding name differs from cell")
    for name in ("capability_id", "code_digest", "callable_digest"):
        _exact_hex(policy[name], label=f"policy {name}")
    evaluator = _exact_dict(
        context["evaluator_binding"],
        frozenset({"capability_id", "code_digest"}),
        label="evaluator binding",
    )
    _exact_hex(evaluator["capability_id"], label="evaluator capability ID")
    _exact_hex(evaluator["code_digest"], label="evaluator code digest")
    rules, defender_digest = _validate_defender(
        context["defender"], contract_digest=cast(str, contract["defender_digest"])
    )

    search = _exact_dict(
        cell["search_result"],
        frozenset(
            {
                "document",
                "canonical_document_digest",
                "process_local_issuance_seal",
                "seal_scope",
            }
        ),
        label="search result evidence",
    )
    document = _exact_dict(search["document"], _SEARCH_RESULT_FIELDS, label="SearchResult")
    if search["canonical_document_digest"] != canonical_digest(document):
        _fail("SearchResult canonical document digest differs")
    _exact_hex(search["process_local_issuance_seal"], label="process-local issuance seal")
    if search["seal_scope"] != "process_local_nonportable_hmac":
        _fail("SearchResult seal scope is not explicitly nonportable")
    if (
        document["family"] != family
        or document["seed"] != seed
        or document["policy_name"] != policy_name
        or document["bounds_digest"] != bounds_digest
        or document["defender_digest"] != defender_digest
        or document["evaluation_contract_digest"] != contract["contract_digest"]
        or document["disclosure_profile_digest"]
        != cast(dict[str, object], contract["disclosure_profile"])["profile_digest"]
        or document["policy_capability_id"] != policy["capability_id"]
        or document["policy_version"] != policy["version"]
        or document["policy_code_digest"] != policy["code_digest"]
        or document["policy_callable_digest"] != policy["callable_digest"]
        or document["evaluator_capability_id"] != evaluator["capability_id"]
        or document["evaluator_code_digest"] != evaluator["code_digest"]
    ):
        _fail("SearchResult policy, defender, or evaluator provenance differs")
    for name in (
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
    ):
        if document[name] != contract[name]:
            _fail(f"SearchResult {name} provenance differs")
    for name in (
        "authority_id",
        "evaluator_capability_id",
        "evaluator_code_digest",
        "policy_capability_id",
        "policy_code_digest",
        "policy_callable_digest",
        "run_group_id",
        "result_id",
    ):
        _exact_hex(document[name], label=f"SearchResult {name}")

    proposals = _exact_list(document["proposals"], label="SearchResult proposals")
    trials = _exact_list(document["trials"], label="SearchResult trials")
    objectives = _exact_list(
        document["objective_values"], label="SearchResult objectives"
    )
    if not (len(proposals) == len(trials) == len(objectives)):
        _fail("SearchResult proposal, trial, and objective counts differ")
    candidate_ids: list[str] = []
    objective_numbers: list[Decimal] = []
    approved_count = 0
    net_value = Decimal(0)
    first_approval = len(trials) + 1
    seen: set[str] = set()
    candidate_sequence = _exact_list(
        cell["candidate_sequence"], label="candidate sequence"
    )
    if len(candidate_sequence) != len(proposals):
        _fail("candidate sequence count differs from SearchResult")
    traces = _exact_list(cell["evaluation_traces"], label="evaluation traces")
    if len(traces) != len(trials):
        _fail("evaluation trace count differs from SearchResult trials")
    if cell["evaluation_trace_digest"] != canonical_digest(traces):
        _fail("evaluation trace collection digest differs")
    disclosure = cast(dict[str, object], contract["disclosure_profile"])
    expose_value = cast(bool, disclosure["expose_realized_value"])
    for index, (proposal, raw_trial, raw_objective) in enumerate(
        zip(proposals, trials, objectives, strict=True)
    ):
        candidate_id, fingerprint, parent, generation = _candidate_identity(
            proposal, feasible
        )
        if generation != index or (index == 0 and parent is not None) or (
            index > 0 and parent not in seen
        ):
            _fail("candidate lineage is not contiguous and past-only")
        seen.add(candidate_id)
        candidate_ids.append(candidate_id)
        trial = _exact_dict(
            raw_trial,
            frozenset({"candidate", "feedback", "objective_value"}),
            label="visible trial",
        )
        if trial["candidate"] != proposal:
            _fail("visible trial candidate differs from proposal order")
        action, _reason, realized, expected_objective = _feedback_objective(
            trial["feedback"]
        )
        observed_objective = _decimal(
            trial["objective_value"], label="visible trial objective"
        )
        sequence_objective = _decimal(raw_objective, label="SearchResult objective")
        if observed_objective != expected_objective or sequence_objective != expected_objective:
            _fail("visible objective is not derived from public feedback")
        objective_numbers.append(expected_objective)
        if action == Action.APPROVE.value:
            approved_count += 1
            first_approval = min(first_approval, index + 1)
            if realized is not None:
                net_value += Decimal(realized)
        sequence = _exact_dict(
            candidate_sequence[index],
            frozenset(
                {
                    "trial_index",
                    "adaptive_vector",
                    "vector_fingerprint",
                    "candidate_id",
                    "parent_id",
                    "generation",
                }
            ),
            label="candidate sequence entry",
        )
        _raw, tagged = _tagged_vector_to_raw(sequence["adaptive_vector"])
        if (
            sequence["trial_index"] != index
            or tagged != feasible[_canonical_payload(cast(dict[str, object], proposal)["params"])]
            or sequence["vector_fingerprint"] != fingerprint
            or sequence["candidate_id"] != candidate_id
            or sequence["parent_id"] != parent
            or sequence["generation"] != generation
        ):
            _fail("candidate sequence differs from complete SearchResult proposal")
        _verify_trace(
            traces[index],
            index=index,
            cell_id=expected_cell_id,
            family=family,
            seed=seed,
            candidate=proposal,
            candidate_id=candidate_id,
            feedback=trial["feedback"],
            disclosure_exposes_value=expose_value,
            rules=rules,
        )

    winner = document["winner"]
    if not proposals:
        if winner is not None:
            _fail("empty SearchResult has a winner")
    else:
        winner_index = min(
            range(len(proposals)),
            key=lambda index: (-objective_numbers[index], candidate_ids[index]),
        )
        if winner != proposals[winner_index]:
            _fail("SearchResult winner differs from visible objectives")
    proposal_budget = _exact_int(document["proposal_budget"], label="proposal budget")
    query_budget = _exact_int(document["query_budget"], label="query budget")
    logical_budget = _exact_int(
        document["logical_time_budget"], label="logical-time budget"
    )
    proposals_used = _exact_int(document["proposals_used"], label="proposal usage")
    queries_used = _exact_int(document["queries_used"], label="query usage")
    logical_used = _exact_int(document["logical_time_used"], label="logical-time usage")
    wall_budget = _exact_int(
        document["wall_time_budget_ms"], label="wall-time budget"
    )
    wall_elapsed = _exact_int(
        document["wall_time_elapsed_ms"], label="wall-time elapsed"
    )
    wall_overrun = _exact_int(
        document["wall_time_overrun_ms"], label="wall-time overrun"
    )
    exhausted = document["wall_time_exhausted"]
    if type(exhausted) is not bool:
        _fail("wall-time exhaustion must be exact bool")
    if not (
        proposal_budget == query_budget == logical_budget
        and proposals_used == queries_used == logical_used == len(proposals)
        and len(proposals) <= proposal_budget
        and wall_overrun == max(0, wall_elapsed - wall_budget)
    ):
        _fail("SearchResult budget or actual usage accounting differs")
    if not exhausted and len(proposals) != proposal_budget:
        _fail("non-exhausted SearchResult did not consume proposal budget")
    if expected_budget is not None and proposal_budget != expected_budget:
        _fail("SearchResult proposal budget differs from expected")
    if expected_wall_time_budget_ms is not None and wall_budget != expected_wall_time_budget_ms:
        _fail("SearchResult wall-time budget differs from expected")

    attempts = _exact_list(cell["llm_audit_attempts"], label="LLM audit attempts")
    if cell["llm_audit_digest"] != canonical_digest(attempts):
        _fail("LLM audit collection digest differs")
    _verify_llm_attempts(
        attempts,
        policy_name=policy_name,
        cell_id=expected_cell_id,
        seed=seed,
        trial_count=len(trials),
        public_bounds=context["public_bounds"],
        proposals=proposals,
        trials=trials,
        expected_cache=expected_llm_cache,
    )
    with localcontext() as context_precision:
        context_precision.prec = 28
        speed = Decimal(first_approval)
    return {
        "proposal_count": len(trials),
        "approved_count": approved_count,
        "valid_yield": str(
            Decimal(0)
            if not trials
            else Decimal(approved_count) / Decimal(len(trials))
        ),
        "net_settled_value": str(net_value),
        "adaptation_speed": str(speed),
        "campaign_scale": approved_count,
        "wall_time_exhausted": exhausted,
        "wall_time_overrun_ms": wall_overrun,
        "configured_budget": proposal_budget,
        "actual_budget": proposals_used,
    }


_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "external_approval",
        "preregistration_commit",
        "preregistration_file_sha256",
        "preregistration_canonical_digest",
        "protocol",
        "execution_audit",
        "evidence",
        "summary",
        "bundle_digest",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "cells",
        "cell_count",
        "total_trial_count",
        "cached_llm_attempt_count",
        "evidence_digest",
    }
)
_EXECUTION_AUDIT_FIELDS = frozenset(
    {
        "network_transport",
        "network_call_count",
        "deadline_exhausted_cell_count",
        "wall_time_overrun_ms",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "matched_budgets",
        "confirmatory_valid",
        "criterion_met",
        "supported_family_count",
        "adaptive_claim",
        "negative_control",
        "families",
        "cached_llm_audit",
    }
)
_TARGET_POLICY_ORDER = ("fixed", "random", "adaptive", "cached_llm")


def derive_artifact_scoped_provenance(
    value: object,
    *,
    preregistered_contexts: object,
    preregistered_policy_bindings: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Rehydrate nonportable capability IDs from one frozen raw artifact."""
    _assert_exact_json(value, path="artifact-scoped result")
    _assert_exact_json(
        preregistered_contexts,
        path="artifact-scoped preregistered contexts",
    )
    _assert_exact_json(
        preregistered_policy_bindings,
        path="artifact-scoped preregistered policies",
    )
    bundle = _exact_dict(value, _BUNDLE_FIELDS, label="result bundle")
    bundle_core = {key: bundle[key] for key in bundle if key != "bundle_digest"}
    if bundle["bundle_digest"] != canonical_digest(bundle_core):
        _fail("artifact result bundle digest differs")
    evidence = _exact_dict(bundle["evidence"], _EVIDENCE_FIELDS, label="raw evidence")
    cells = _exact_list(evidence["cells"], label="raw evidence cells")
    if evidence["evidence_digest"] != canonical_digest(cells):
        _fail("artifact raw-evidence digest differs")
    contexts = _exact_dict(
        preregistered_contexts,
        frozenset({"targets", "negative_control"}),
        label="preregistered evidence contexts",
    )
    if type(contexts["targets"]) is not dict or any(
        type(family) is not str for family in cast(dict[object, object], contexts["targets"])
    ):
        _fail("preregistered target contexts must be an exact string-keyed object")
    target_contexts = cast(dict[str, object], contexts["targets"])
    negative_stable = _exact_dict(
        contexts["negative_control"],
        frozenset(
            {
                "public_bounds",
                "evaluation_contract",
                "evaluator_code_digest",
                "defender",
            }
        ),
        label="preregistered negative-control context",
    )
    negative_contract = cast(dict[str, object], negative_stable["evaluation_contract"])
    negative_family = _exact_text(
        negative_contract.get("family"),
        label="preregistered negative-control family",
    )
    expected_evaluator_contexts = {
        *(('target', family) for family in target_contexts),
        ("negative_control", negative_family),
    }
    preregistered_policies = _exact_dict(
        preregistered_policy_bindings,
        frozenset(_TARGET_POLICY_ORDER),
        label="preregistered policy bindings",
    )
    evaluator_id_sets: dict[tuple[str, str], set[str]] = {
        key: set() for key in expected_evaluator_contexts
    }
    policy_id_sets: dict[str, set[str]] = {
        name: set() for name in _TARGET_POLICY_ORDER
    }
    authority_ids: set[str] = set()
    run_group_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for raw_cell in cells:
        cell = _exact_dict(raw_cell, _CELL_FIELDS, label="raw evidence cell")
        cell_core = {key: cell[key] for key in cell if key != "cell_digest"}
        if cell["cell_digest"] != canonical_digest(cell_core):
            _fail("artifact cell digest differs")
        cell_kind = _exact_text(cell["cell_kind"], label="artifact cell kind")
        family = _exact_text(cell["family"], label="artifact cell family")
        policy_name = _exact_text(cell["policy_name"], label="artifact cell policy")
        seed = _exact_int(cell["seed"], label="artifact cell seed")
        context_key = (cell_kind, family)
        if context_key not in evaluator_id_sets or policy_name not in policy_id_sets:
            _fail("artifact cell provenance is outside preregistered contexts")
        raw_stable_context = (
            target_contexts[family]
            if cell_kind == "target"
            else negative_stable
        )
        stable_context = _exact_dict(
            raw_stable_context,
            frozenset(
                {
                    "public_bounds",
                    "evaluation_contract",
                    "evaluator_code_digest",
                    "defender",
                }
            ),
            label="preregistered evidence context",
        )
        stable_policy = _exact_dict(
            preregistered_policies[policy_name],
            frozenset(
                {
                    "version",
                    "code_digest",
                    "callable_digest",
                    "capability_id_scope",
                }
            ),
            label="preregistered policy binding",
        )
        if stable_policy["capability_id_scope"] != (
            "process_local_nonportable_not_preregistered"
        ):
            _fail("preregistered policy capability scope is not explicitly nonportable")
        public_context = _exact_dict(
            cell["public_context"],
            frozenset(
                {
                    "public_bounds",
                    "evaluation_contract",
                    "evaluator_binding",
                    "policy_binding",
                    "defender",
                }
            ),
            label="artifact public context",
        )
        evaluator_binding = _exact_dict(
            public_context["evaluator_binding"],
            frozenset({"capability_id", "code_digest"}),
            label="artifact evaluator binding",
        )
        policy_binding = _exact_dict(
            public_context["policy_binding"],
            frozenset(
                {"name", "version", "capability_id", "code_digest", "callable_digest"}
            ),
            label="artifact policy binding",
        )
        search = _exact_dict(
            cell["search_result"],
            frozenset(
                {
                    "document",
                    "canonical_document_digest",
                    "process_local_issuance_seal",
                    "seal_scope",
                }
            ),
            label="artifact search result",
        )
        document = _exact_dict(
            search["document"],
            _SEARCH_RESULT_FIELDS,
            label="artifact SearchResult",
        )
        if search["canonical_document_digest"] != canonical_digest(document):
            _fail("artifact SearchResult canonical digest differs")
        _exact_hex(
            search["process_local_issuance_seal"],
            label="artifact process-local issuance seal",
        )
        if search["seal_scope"] != "process_local_nonportable_hmac":
            _fail("artifact issuance seal is not explicitly process-local")
        evaluator_id = _exact_hex(
            evaluator_binding["capability_id"],
            label="artifact evaluator capability ID",
        )
        policy_id = _exact_hex(
            policy_binding["capability_id"],
            label="artifact policy capability ID",
        )
        authority_id = _exact_hex(
            document["authority_id"],
            label="artifact authority ID",
        )
        run_group_id = _exact_hex(
            document["run_group_id"],
            label="artifact run-group ID",
        )
        result_id = _exact_hex(
            document["result_id"],
            label="artifact result ID",
        )
        cell_id = _exact_hex(cell["cell_id"], label="artifact cell ID")
        if (
            public_context["public_bounds"] != stable_context["public_bounds"]
            or public_context["evaluation_contract"]
            != stable_context["evaluation_contract"]
            or public_context["defender"] != stable_context["defender"]
            or evaluator_binding["code_digest"]
            != stable_context["evaluator_code_digest"]
        ):
            _fail("artifact evaluator or contract provenance differs from preregistration")
        if (
            policy_binding["version"] != stable_policy["version"]
            or policy_binding["code_digest"] != stable_policy["code_digest"]
            or policy_binding["callable_digest"] != stable_policy["callable_digest"]
        ):
            _fail("artifact policy provenance differs from preregistration")
        public_contract = cast(
            dict[str, object], public_context["evaluation_contract"]
        )
        disclosure = cast(dict[str, object], public_contract["disclosure_profile"])
        if (
            document["family"] != family
            or document["seed"] != seed
            or document["policy_name"] != policy_name
            or policy_binding["name"] != policy_name
            or document["evaluator_capability_id"] != evaluator_id
            or document["policy_capability_id"] != policy_id
            or document["evaluator_code_digest"] != evaluator_binding["code_digest"]
            or document["policy_version"] != policy_binding["version"]
            or document["policy_code_digest"] != policy_binding["code_digest"]
            or document["policy_callable_digest"] != policy_binding["callable_digest"]
            or document["bounds_digest"] != public_contract["bounds_digest"]
            or document["hidden_template_digest"]
            != public_contract["hidden_template_digest"]
            or document["background_digest"] != public_contract["background_digest"]
            or document["population_digest"] != public_contract["population_digest"]
            or document["evaluator_digest"] != public_contract["evaluator_digest"]
            or document["defender_digest"] != public_contract["defender_digest"]
            or document["disclosure_profile_digest"] != disclosure["profile_digest"]
            or document["evaluation_contract_digest"]
            != public_contract["contract_digest"]
        ):
            _fail("artifact embedded and public provenance differs")
        evaluator_id_sets[context_key].add(evaluator_id)
        policy_id_sets[policy_name].add(policy_id)
        authority_ids.add(authority_id)
        run_group_ids.add(run_group_id)
        if (
            result_id == cell_id
            or result_id in artifact_ids
            or cell_id in artifact_ids
        ):
            _fail("artifact result or cell identity is not unique")
        artifact_ids.update((result_id, cell_id))

    if len(authority_ids) != 1 or len(run_group_ids) != 1:
        _fail("artifact authority or run-group identity is not unique")
    if any(len(ids) != 1 for ids in evaluator_id_sets.values()):
        _fail("artifact evaluator capability identity differs within one context")
    evaluator_ids = {
        key: next(iter(ids)) for key, ids in evaluator_id_sets.items()
    }
    if len(set(evaluator_ids.values())) != len(evaluator_ids):
        _fail("artifact evaluator capability identities are reused across contexts")
    if any(len(ids) != 1 for ids in policy_id_sets.values()):
        _fail("artifact policy capability identity differs within one policy")
    policy_ids = {name: next(iter(ids)) for name, ids in policy_id_sets.items()}
    if len(set(policy_ids.values())) != len(policy_ids):
        _fail("artifact policy capability identities are reused across policies")

    def context_document(
        preregistered: object,
        *,
        key: tuple[str, str],
    ) -> dict[str, object]:
        stable = _exact_dict(
            preregistered,
            frozenset(
                {
                    "public_bounds",
                    "evaluation_contract",
                    "evaluator_code_digest",
                    "defender",
                }
            ),
            label="preregistered evidence context",
        )
        return {
            "public_bounds": stable["public_bounds"],
            "evaluation_contract": stable["evaluation_contract"],
            "evaluator_binding": {
                "capability_id": evaluator_ids[key],
                "code_digest": stable["evaluator_code_digest"],
            },
            "defender": stable["defender"],
        }

    expected_contexts: dict[str, object] = {
        "targets": {
            family: context_document(
                preregistered,
                key=("target", family),
            )
            for family, preregistered in target_contexts.items()
        },
        "negative_control": context_document(
            negative_stable,
            key=("negative_control", negative_family),
        ),
    }
    expected_policies: dict[str, object] = {}
    for name, raw_stable in preregistered_policies.items():
        stable = cast(dict[str, object], raw_stable)
        expected_policies[name] = {
            "name": name,
            "version": stable["version"],
            "capability_id": policy_ids[name],
            "code_digest": stable["code_digest"],
            "callable_digest": stable["callable_digest"],
        }
    return expected_contexts, expected_policies


def _protocol_parts(
    value: object,
) -> tuple[
    dict[str, object],
    tuple[int, ...],
    int,
    int,
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    protocol = _exact_dict(
        value,
        frozenset(
            {
                "protocol_version",
                "experiment_id",
                "replication_kind",
                "algorithm_retuned_after_v3_3",
                "seeds",
                "budgets",
                "maximum_confirmatory_attempts",
                "policies",
                "metrics",
                "negative_control",
                "network",
                "evidence_limits",
                "uncertainty",
                "fairness",
                "stopping_rule",
                "approval_boundary",
            }
        ),
        label="Task 6 protocol",
    )
    seed_values = _exact_list(protocol["seeds"], label="protocol seeds")
    seeds = tuple(_exact_int(seed, label="protocol seed") for seed in seed_values)
    if not seeds or len(set(seeds)) != len(seeds):
        _fail("protocol seeds must be non-empty and unique")
    budgets = _exact_dict(
        protocol["budgets"],
        frozenset({"proposal", "query", "logical_time", "wall_time_ms"}),
        label="protocol budgets",
    )
    budget = _exact_int(budgets["proposal"], label="protocol proposal budget", minimum=1)
    if (
        budgets["query"] != budget
        or budgets["logical_time"] != budget
        or type(budgets["wall_time_ms"]) is not int
        or budgets["wall_time_ms"] < 1
    ):
        _fail("protocol discrete or wall-time budgets are not exact and matched")
    wall_budget = budgets["wall_time_ms"]
    policies = _exact_dict(
        protocol["policies"],
        frozenset(_TARGET_POLICY_ORDER),
        label="protocol policies",
    )
    if any(type(version) is not str or not version for version in policies.values()):
        _fail("protocol policy versions must be exact text")
    raw_metrics = protocol["metrics"]
    if type(raw_metrics) is not dict or any(type(key) is not str for key in raw_metrics):
        _fail("protocol metrics must be an exact string-keyed object")
    metrics = cast(dict[str, dict[str, object]], raw_metrics)
    if not metrics:
        _fail("protocol needs target family metrics")
    for family, raw in metrics.items():
        details = _exact_dict(
            raw,
            frozenset({"primary_outcome", "minimum_delta", "no_metric_switching"}),
            label=f"protocol metric {family}",
        )
        if details["primary_outcome"] not in {"net_settled_value_rate", "valid_yield"}:
            _fail("protocol primary outcome is unsupported")
        _decimal(details["minimum_delta"], label="protocol minimum delta", nonnegative=True)
        if details["no_metric_switching"] is not True:
            _fail("protocol permits metric switching")
    negative = _exact_dict(
        protocol["negative_control"],
        frozenset(
            {
                "family",
                "policy_cells",
                "primary_outcome",
                "minimum_delta",
                "expected_observed_delta",
                "support_expected",
                "included_in_supported_family_count",
                "same_seeds_and_budgets_as_targets",
            }
        ),
        label="negative-control protocol",
    )
    negative_policies = _exact_list(
        negative["policy_cells"], label="negative-control policies"
    )
    if negative_policies != ["fixed", "random", "adaptive"]:
        _fail("negative-control policy cells changed")
    if (
        negative["primary_outcome"] != "valid_yield"
        or negative["expected_observed_delta"] != "0"
        or negative["support_expected"] is not False
        or negative["included_in_supported_family_count"] is not False
        or negative["same_seeds_and_budgets_as_targets"] is not True
    ):
        _fail("negative-control protocol changed")
    limits = _exact_dict(
        protocol["evidence_limits"],
        frozenset(
            {
                "expected_cell_count",
                "trials_per_complete_cell",
                "maximum_total_trials",
                "maximum_cached_llm_attempts",
                "maximum_bundle_bytes",
                "lossless",
            }
        ),
        label="evidence limits",
    )
    for name in (
        "expected_cell_count",
        "trials_per_complete_cell",
        "maximum_total_trials",
        "maximum_cached_llm_attempts",
        "maximum_bundle_bytes",
    ):
        _exact_int(limits[name], label=f"evidence limit {name}", minimum=1)
    if limits["lossless"] is not True or limits["trials_per_complete_cell"] != budget:
        _fail("evidence protocol is not complete and lossless")
    return protocol, seeds, budget, wall_budget, metrics, negative, limits


def _expected_cell_specs(
    *,
    seeds: tuple[int, ...],
    metrics: dict[str, dict[str, object]],
    negative: dict[str, object],
) -> tuple[tuple[str, str, str, int], ...]:
    target = tuple(
        ("target", family, policy, seed)
        for family in sorted(metrics)
        for policy in _TARGET_POLICY_ORDER
        for seed in seeds
    )
    negative_family = _exact_text(negative["family"], label="negative-control family")
    controls = tuple(
        ("negative_control", negative_family, cast(str, policy), seed)
        for policy in cast(list[object], negative["policy_cells"])
        for seed in seeds
    )
    return target + controls


def _aggregate_cell_metrics(cells: list[dict[str, object]]) -> dict[str, object]:
    if not cells:
        _fail("policy aggregate has no raw cells")
    proposal_count = sum(cast(int, cell["proposal_count"]) for cell in cells)
    approved_count = sum(cast(int, cell["approved_count"]) for cell in cells)
    net_value = sum(
        (_decimal(cell["net_settled_value"], label="cell net value") for cell in cells),
        Decimal(0),
    )
    with localcontext() as precision:
        precision.prec = 28
        valid_yield = (
            Decimal(0)
            if proposal_count == 0
            else Decimal(approved_count) / Decimal(proposal_count)
        )
        speed = sum(
            (_decimal(cell["adaptation_speed"], label="cell speed") for cell in cells),
            Decimal(0),
        ) / Decimal(len(cells))
    return {
        "proposal_count": proposal_count,
        "approved_count": approved_count,
        "valid_yield": str(valid_yield),
        "net_settled_value": str(net_value),
        "adaptation_speed": str(speed),
        "campaign_scale": approved_count,
    }


def _primary_value(metrics: dict[str, object], outcome: str) -> Decimal:
    if outcome == "valid_yield":
        return _decimal(metrics["valid_yield"], label="aggregate valid yield")
    return _decimal(metrics["net_settled_value"], label="aggregate net value")


def _delta(adaptive: Decimal, random: Decimal, outcome: str) -> Decimal:
    if outcome == "valid_yield":
        return adaptive - random
    if random == 0:
        return Decimal(0) if adaptive == 0 else Decimal(1)
    with localcontext() as precision:
        precision.prec = 28
        return (adaptive - random) / random


def _uncertainty(
    adaptive_cells: list[dict[str, object]],
    random_cells: list[dict[str, object]],
    *,
    outcome: str,
) -> dict[str, object]:
    deltas = [
        _delta(_primary_value(adaptive, outcome), _primary_value(random, outcome), outcome)
        for adaptive, random in zip(adaptive_cells, random_cells, strict=True)
    ]
    with localcontext() as precision:
        precision.prec = 28
        mean = sum(deltas, Decimal(0)) / Decimal(len(deltas))
        references = sorted(
            sum(
                (
                    delta if mask & (1 << index) else -delta
                    for index, delta in enumerate(deltas)
                ),
                Decimal(0),
            )
            / Decimal(len(deltas))
            for mask in range(2 ** len(deltas))
        )
    lower = references[int(Decimal("0.025") * Decimal(len(references) - 1))]
    upper = references[int(Decimal("0.975") * Decimal(len(references) - 1))]
    return {
        "method": "exact_paired_sign_resampling_reference_interval",
        "role": "descriptive_only",
        "per_seed_deltas": [str(delta) for delta in deltas],
        "mean_per_seed_delta": str(mean),
        "reference_interval_95": [str(lower), str(upper)],
    }


def _derive_bundle_documents(
    cells: list[object],
    *,
    protocol: dict[str, object],
    seeds: tuple[int, ...],
    budget: int,
    wall_budget: int,
    metrics_protocol: dict[str, dict[str, object]],
    negative_protocol: dict[str, object],
    expected_contexts: object,
    expected_policy_bindings: object,
    expected_llm_cache: object,
    network_call_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    contexts = _exact_dict(
        expected_contexts,
        frozenset({"targets", "negative_control"}),
        label="frozen evidence contexts",
    )
    target_contexts = contexts["targets"]
    if type(target_contexts) is not dict or set(target_contexts) != set(metrics_protocol):
        _fail("frozen target evidence contexts differ from protocol families")
    policies = _exact_dict(
        expected_policy_bindings,
        frozenset(_TARGET_POLICY_ORDER),
        label="frozen policy bindings",
    )
    specs = _expected_cell_specs(
        seeds=seeds,
        metrics=metrics_protocol,
        negative=negative_protocol,
    )
    if len(cells) != len(specs):
        _fail("raw evidence cell count differs from the preregistered design")
    verified: dict[tuple[str, str, str, int], dict[str, object]] = {}
    authority_ids: set[object] = set()
    run_group_ids: set[object] = set()
    result_ids: set[object] = set()
    cell_ids: set[object] = set()
    exhausted_count = 0
    overrun_total = 0
    cached_attempts: list[dict[str, object]] = []
    for raw_cell, spec in zip(cells, specs, strict=True):
        kind, family, policy_name, seed = spec
        cell = _exact_dict(raw_cell, _CELL_FIELDS, label="raw evidence cell")
        expected_family_context = (
            cast(dict[str, object], target_contexts)[family]
            if kind == "target"
            else contexts["negative_control"]
        )
        if type(expected_family_context) is not dict:
            _fail("frozen family context must be an exact object")
        expected_context = {
            **cast(dict[str, object], expected_family_context),
            "policy_binding": policies[policy_name],
        }
        if cell["public_context"] != expected_context:
            _fail("raw cell public context differs from preregistered provenance")
        cell_metrics = verify_search_cell(
            cell,
            expected_cell_kind=kind,
            expected_family=family,
            expected_policy=policy_name,
            expected_seed=seed,
            expected_budget=budget,
            expected_wall_time_budget_ms=wall_budget,
            expected_llm_cache=expected_llm_cache,
        )
        if spec in verified:
            _fail("raw evidence contains a duplicate cell")
        verified[spec] = cell_metrics
        search = cast(dict[str, object], cell["search_result"])
        result = cast(dict[str, object], search["document"])
        authority_ids.add(result["authority_id"])
        run_group_ids.add(result["run_group_id"])
        result_ids.add(result["result_id"])
        cell_ids.add(cell["cell_id"])
        exhausted_count += cast(bool, cell_metrics["wall_time_exhausted"])
        overrun_total += cast(int, cell_metrics["wall_time_overrun_ms"])
        if policy_name == "cached_llm":
            cached_attempts.append(
                {
                    "cell_id": cell["cell_id"],
                    "attempts": cell["llm_audit_attempts"],
                }
            )
    if (
        len(authority_ids) != 1
        or len(run_group_ids) != 1
        or len(result_ids) != len(cells)
        or len(cell_ids) != len(cells)
    ):
        _fail("raw cells do not share one authority/run group or unique result identities")
    if type(network_call_count) is not int or network_call_count < 0:
        _fail("network call count must be an exact non-negative integer")
    execution_audit = {
        "network_transport": "disabled_assertion_client",
        "network_call_count": network_call_count,
        "deadline_exhausted_cell_count": exhausted_count,
        "wall_time_overrun_ms": overrun_total,
    }
    matched_budgets = exhausted_count == 0 and overrun_total == 0 and all(
        cell["configured_budget"] == budget and cell["actual_budget"] == budget
        for cell in verified.values()
    )
    family_summaries: dict[str, object] = {}
    raw_supported_count = 0
    total_adaptive_net = Decimal(0)
    total_random_net = Decimal(0)
    for family in sorted(metrics_protocol):
        policy_cells: dict[str, list[dict[str, object]]] = {
            policy: [verified[("target", family, policy, seed)] for seed in seeds]
            for policy in _TARGET_POLICY_ORDER
        }
        aggregates = {
            policy: _aggregate_cell_metrics(policy_cells[policy])
            for policy in _TARGET_POLICY_ORDER
        }
        metric_protocol = metrics_protocol[family]
        outcome = cast(str, metric_protocol["primary_outcome"])
        minimum = _decimal(
            metric_protocol["minimum_delta"], label="family minimum delta", nonnegative=True
        )
        observed = _delta(
            _primary_value(aggregates["adaptive"], outcome),
            _primary_value(aggregates["random"], outcome),
            outcome,
        )
        supported = matched_budgets and observed >= minimum
        raw_supported_count += supported
        total_adaptive_net += _decimal(
            aggregates["adaptive"]["net_settled_value"], label="adaptive net value"
        )
        total_random_net += _decimal(
            aggregates["random"]["net_settled_value"], label="random net value"
        )
        family_summaries[family] = {
            "primary_outcome": outcome,
            "minimum_delta": str(minimum),
            "observed_delta": str(observed),
            "supported": supported,
            **aggregates,
            "uncertainty": _uncertainty(
                policy_cells["adaptive"], policy_cells["random"], outcome=outcome
            ),
        }
    negative_family = cast(str, negative_protocol["family"])
    negative_cells = {
        policy: [
            verified[("negative_control", negative_family, policy, seed)] for seed in seeds
        ]
        for policy in ("fixed", "random", "adaptive")
    }
    negative_aggregates = {
        policy: _aggregate_cell_metrics(cells_for_policy)
        for policy, cells_for_policy in negative_cells.items()
    }
    negative_random_yield = _decimal(
        negative_aggregates["random"]["valid_yield"], label="negative random yield"
    )
    negative_adaptive_yield = _decimal(
        negative_aggregates["adaptive"]["valid_yield"], label="negative adaptive yield"
    )
    negative_delta = negative_adaptive_yield - negative_random_yield
    negative_supported = matched_budgets and negative_delta >= _decimal(
        negative_protocol["minimum_delta"], label="negative minimum delta", nonnegative=True
    )
    negative_summary = {
        "family": negative_family,
        "primary_outcome": "valid_yield",
        "minimum_delta": cast(str, negative_protocol["minimum_delta"]),
        "included_in_supported_family_count": False,
        "matched_budgets": matched_budgets,
        "network_call_count": network_call_count,
        "random_valid_yield": str(negative_random_yield),
        "adaptive_valid_yield": str(negative_adaptive_yield),
        "observed_valid_yield_delta": str(negative_delta),
        "supported": negative_supported,
    }
    target_claim = total_adaptive_net > total_random_net
    confirmatory_valid = (
        matched_budgets
        and network_call_count == 0
        and negative_delta == 0
        and negative_supported is False
    )
    supported_count = raw_supported_count if confirmatory_valid else 0
    criterion = (
        confirmatory_valid
        and supported_count == len(metrics_protocol)
        and target_claim
    )
    if not confirmatory_valid:
        for raw_family in family_summaries.values():
            cast(dict[str, object], raw_family)["supported"] = False
    summary: dict[str, object] = {
        "matched_budgets": matched_budgets,
        "confirmatory_valid": confirmatory_valid,
        "criterion_met": criterion,
        "supported_family_count": supported_count,
        "adaptive_claim": "supported" if criterion else "not_supported",
        "negative_control": negative_summary,
        "families": family_summaries,
        "cached_llm_audit": {
            "attempt_count": sum(
                len(cast(list[object], record["attempts"])) for record in cached_attempts
            ),
            "cache_success_count": sum(
                attempt["call_status"] == "cache_success"
                for record in cached_attempts
                for attempt in cast(list[dict[str, object]], record["attempts"])
            ),
            "network_call_count": network_call_count,
            "audit_digest": canonical_digest(cached_attempts),
        },
    }
    return execution_audit, summary


def build_result_bundle_document(
    *,
    protocol: dict[str, object],
    cells: list[dict[str, object]],
    expected_contexts: dict[str, object],
    expected_policy_bindings: dict[str, object],
    expected_llm_cache: dict[str, object],
    external_approval: dict[str, str],
    preregistration_canonical_digest: str,
    network_call_count: int,
) -> dict[str, object]:
    """Build and independently verify the canonical raw-evidence result document."""
    _assert_exact_json(protocol, path="result.protocol")
    checked_protocol, seeds, budget, wall_budget, metrics, negative, limits = (
        _protocol_parts(protocol)
    )
    execution_audit, summary = _derive_bundle_documents(
        cast(list[object], cells),
        protocol=checked_protocol,
        seeds=seeds,
        budget=budget,
        wall_budget=wall_budget,
        metrics_protocol=metrics,
        negative_protocol=negative,
        expected_contexts=expected_contexts,
        expected_policy_bindings=expected_policy_bindings,
        expected_llm_cache=expected_llm_cache,
        network_call_count=network_call_count,
    )
    approval = _exact_dict(
        external_approval,
        frozenset({"approved_freeze_commit", "approved_prereg_sha256"}),
        label="external approval",
    )
    commit = _exact_text(approval["approved_freeze_commit"], label="approved commit")
    if len(commit) != 40 or not set(commit) <= _HEX:
        _fail("approved commit must be a full lowercase Git object ID")
    prereg_sha = _exact_hex(
        approval["approved_prereg_sha256"], label="approved preregistration SHA-256"
    )
    prereg_digest = _exact_hex(
        preregistration_canonical_digest, label="preregistration canonical digest"
    )
    evidence = {
        "cells": cells,
        "cell_count": len(cells),
        "total_trial_count": sum(
            len(_exact_list(cell["evaluation_traces"], label="cell traces"))
            for cell in cells
        ),
        "cached_llm_attempt_count": sum(
            len(_exact_list(cell["llm_audit_attempts"], label="cell audits"))
            for cell in cells
        ),
        "evidence_digest": canonical_digest(cells),
    }
    core: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "executed_evidence_replication",
        "external_approval": external_approval,
        "preregistration_commit": commit,
        "preregistration_file_sha256": prereg_sha,
        "preregistration_canonical_digest": prereg_digest,
        "protocol": protocol,
        "execution_audit": execution_audit,
        "evidence": evidence,
        "summary": summary,
    }
    document = {**core, "bundle_digest": canonical_digest(core)}
    maximum_bytes = cast(int, limits["maximum_bundle_bytes"])
    if len(canonical_json_bytes(document)) > maximum_bytes:
        _fail("complete raw evidence bundle exceeds its stable preregistered size bound")
    return document


def verify_result_bundle(
    value: object,
    *,
    expected_protocol: dict[str, object],
    expected_contexts: dict[str, object],
    expected_policy_bindings: dict[str, object],
    expected_llm_cache: dict[str, object],
    expected_external_approval: dict[str, str] | None = None,
    expected_preregistration_canonical_digest: str | None = None,
) -> dict[str, object]:
    """Recompute the complete Task 6 decision from raw cells, never policy search."""
    _assert_exact_json(value, path="result_bundle")
    bundle = _exact_dict(value, _BUNDLE_FIELDS, label="result bundle")
    core = {key: bundle[key] for key in bundle if key != "bundle_digest"}
    if bundle["bundle_digest"] != canonical_digest(core):
        _fail("result bundle digest differs")
    if bundle["schema_version"] != "1.0.0" or bundle["status"] != (
        "executed_evidence_replication"
    ):
        _fail("result bundle schema or status differs")
    if bundle["protocol"] != expected_protocol:
        _fail("result protocol differs from preregistration")
    protocol, seeds, budget, wall_budget, metrics, negative, limits = _protocol_parts(
        bundle["protocol"]
    )
    approval = _exact_dict(
        bundle["external_approval"],
        frozenset({"approved_freeze_commit", "approved_prereg_sha256"}),
        label="result external approval",
    )
    commit = _exact_text(approval["approved_freeze_commit"], label="approved commit")
    if len(commit) != 40 or not set(commit) <= _HEX:
        _fail("result approved commit is not a full Git object ID")
    prereg_sha = _exact_hex(
        approval["approved_prereg_sha256"], label="approved preregistration SHA-256"
    )
    prereg_digest = _exact_hex(
        bundle["preregistration_canonical_digest"],
        label="result preregistration canonical digest",
    )
    if (
        bundle["preregistration_commit"] != commit
        or bundle["preregistration_file_sha256"] != prereg_sha
    ):
        _fail("result preregistration approval binding differs")
    if expected_external_approval is not None and approval != expected_external_approval:
        _fail("result external approval differs from expected approval")
    if (
        expected_preregistration_canonical_digest is not None
        and prereg_digest != expected_preregistration_canonical_digest
    ):
        _fail("result preregistration canonical digest differs")
    evidence = _exact_dict(bundle["evidence"], _EVIDENCE_FIELDS, label="raw evidence")
    cells = _exact_list(evidence["cells"], label="raw evidence cells")
    if evidence["evidence_digest"] != canonical_digest(cells):
        _fail("raw evidence collection digest differs")
    cell_count = _exact_int(evidence["cell_count"], label="raw evidence cell count")
    trial_count = _exact_int(evidence["total_trial_count"], label="raw evidence trial count")
    attempt_count = _exact_int(
        evidence["cached_llm_attempt_count"], label="raw evidence LLM attempt count"
    )
    observed_trials = sum(
        len(_exact_list(cast(dict[str, object], cell)["evaluation_traces"], label="cell traces"))
        for cell in cells
    )
    observed_attempts = sum(
        len(_exact_list(cast(dict[str, object], cell)["llm_audit_attempts"], label="cell audits"))
        for cell in cells
    )
    if (
        cell_count != len(cells)
        or trial_count != observed_trials
        or attempt_count != observed_attempts
        or cell_count != limits["expected_cell_count"]
        or trial_count > cast(int, limits["maximum_total_trials"])
        or attempt_count > cast(int, limits["maximum_cached_llm_attempts"])
        or trial_count != cell_count * budget
    ):
        _fail("raw evidence cell/trial/LLM counts differ from preregistration")
    if len(canonical_json_bytes(bundle)) > cast(int, limits["maximum_bundle_bytes"]):
        _fail("raw evidence bundle exceeds its preregistered size bound")
    execution = _exact_dict(
        bundle["execution_audit"],
        _EXECUTION_AUDIT_FIELDS,
        label="execution audit",
    )
    network_calls = _exact_int(
        execution["network_call_count"], label="execution network calls"
    )
    derived_execution, derived_summary = _derive_bundle_documents(
        cells,
        protocol=protocol,
        seeds=seeds,
        budget=budget,
        wall_budget=wall_budget,
        metrics_protocol=metrics,
        negative_protocol=negative,
        expected_contexts=expected_contexts,
        expected_policy_bindings=expected_policy_bindings,
        expected_llm_cache=expected_llm_cache,
        network_call_count=network_calls,
    )
    if execution != derived_execution:
        _fail("execution audit differs from raw cell deadlines or network evidence")
    summary = _exact_dict(bundle["summary"], _SUMMARY_FIELDS, label="result summary")
    if summary != derived_summary:
        _fail("result summary/aggregate/claim differs from raw evidence")
    return derived_summary


__all__ = [
    "EvidenceVerificationError",
    "build_result_bundle_document",
    "build_search_cell_document",
    "canonical_digest",
    "canonical_json_bytes",
    "derive_artifact_scoped_provenance",
    "strict_json_loads",
    "verify_result_bundle",
    "verify_search_cell",
]
