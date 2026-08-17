"""Strict public-only JSON transport for the disposable policy worker."""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from typing import cast

from apar.contracts.decisions import Action
from apar.contracts.scenarios import FeedbackField
from apar.redteam.policies import (
    AdaptiveParameter,
    AdaptiveVector,
    AttackCandidate,
    DomainKind,
    Feedback,
    ParameterBounds,
    ParameterDomain,
    VisibleTrial,
    reconstruct_bounds,
    reconstruct_history,
    validate_candidate_lineage,
)


class WireContractError(ValueError):
    """The worker request or response is not an exact public wire document."""


def _strict_tree(value: object, *, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise WireContractError(f"{label} contains a non-finite number")
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _strict_tree(item, label=label)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise WireContractError(f"{label} keys must be exact strings")
            _strict_tree(item, label=label)
        return
    raise WireContractError(f"{label} contains a non-JSON value")


def canonical_json_bytes(value: object) -> bytes:
    """Return canonical strict JSON without interpreter extensions."""
    _strict_tree(value, label="wire document")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def strict_json_loads(raw: bytes) -> object:
    """Reject duplicate keys, non-finite values, non-UTF-8, and noncanonical bytes."""
    if type(raw) is not bytes:
        raise WireContractError("wire input must be exact bytes")

    def pairs(pairs_value: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs_value:
            if key in result:
                raise WireContractError(f"duplicate wire key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                WireContractError(f"non-finite wire value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WireContractError("wire input is not strict UTF-8 JSON") from error
    _strict_tree(value, label="wire document")
    if canonical_json_bytes(value) != raw:
        raise WireContractError("wire input is not canonical JSON")
    return value


def _object(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise WireContractError(f"{label} field set is not exact")
    return cast(dict[str, object], value)


def _list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise WireContractError(f"{label} must be an exact list")
    return cast(list[object], value)


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise WireContractError(f"{label} must be exact non-empty text")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise WireContractError(f"{label} must be an exact non-negative integer")
    return value


def _decode_tagged(value: object) -> object:
    tagged = _object(value, {"type", "value"}, label="adaptive value")
    kind = _text(tagged["type"], label="adaptive value type")
    raw = tagged["value"]
    if kind == "decimal":
        text = _text(raw, label="adaptive Decimal")
        try:
            number = Decimal(text)
        except InvalidOperation as error:
            raise WireContractError("adaptive Decimal is invalid") from error
        if not number.is_finite() or str(number) != text:
            raise WireContractError("adaptive Decimal is not canonical")
        return number
    if kind == "integer" and type(raw) is int:
        return raw
    if kind == "string" and type(raw) is str and raw:
        return raw
    if kind == "string_tuple":
        items = _list(raw, label="adaptive string tuple")
        if not items or any(type(item) is not str for item in items):
            raise WireContractError("adaptive string tuple is invalid")
        return tuple(cast(list[str], items))
    raise WireContractError("adaptive tagged value type is invalid")


def _vector_from_wire(value: object) -> AdaptiveVector:
    entries = _list(value, label="adaptive vector")
    decoded: list[AdaptiveParameter] = []
    for entry_value in entries:
        entry = _object(entry_value, {"name", "value"}, label="adaptive parameter")
        decoded.append(
            AdaptiveParameter(
                name=_text(entry["name"], label="adaptive parameter name"),
                value=_decode_tagged(entry["value"]),
            )
        )
    return AdaptiveVector(entries=tuple(decoded))


def bounds_to_wire(bounds: ParameterBounds) -> dict[str, object]:
    """Serialize the exact public bounds document and no evaluator-owned state."""
    checked = reconstruct_bounds(bounds)
    return checked.document()


def bounds_from_wire(value: object) -> ParameterBounds:
    """Reconstruct sealed public bounds from the strict tagged document."""
    document = _object(
        value,
        {"family", "defaults", "domains", "feasible_vectors"},
        label="parameter bounds",
    )
    domains: list[ParameterDomain] = []
    for domain_value in _list(document["domains"], label="parameter domains"):
        domain = _object(
            domain_value,
            {"name", "kind", "values"},
            label="parameter domain",
        )
        try:
            kind = DomainKind(_text(domain["kind"], label="domain kind"))
        except ValueError as error:
            raise WireContractError("domain kind is undeclared") from error
        domains.append(
            ParameterDomain(
                name=_text(domain["name"], label="domain name"),
                kind=kind,
                values=tuple(
                    _decode_tagged(item)
                    for item in _list(domain["values"], label="domain values")
                ),
            )
        )
    return ParameterBounds(
        family=_text(document["family"], label="bounds family"),
        defaults=_vector_from_wire(document["defaults"]),
        domains=tuple(domains),
        feasible_vectors=tuple(
            _vector_from_wire(item)
            for item in _list(document["feasible_vectors"], label="feasible vectors")
        ),
    )


def candidate_to_wire(candidate: AttackCandidate) -> dict[str, object]:
    candidate.assert_pristine()
    return {
        "candidate_id": candidate.candidate_id,
        "generation": candidate.generation,
        "params": candidate.params.document(),
        "parent_id": candidate.parent_id,
    }


def candidate_from_wire(
    value: object,
    *,
    history: tuple[VisibleTrial, ...],
    bounds: ParameterBounds,
) -> AttackCandidate:
    document = _object(
        value,
        {"candidate_id", "generation", "params", "parent_id"},
        label="attack candidate",
    )
    parent = document["parent_id"]
    if parent is not None and type(parent) is not str:
        raise WireContractError("candidate parent_id must be text or null")
    candidate = AttackCandidate(
        params=_vector_from_wire(document["params"]),
        parent_id=parent,
        generation=_integer(document["generation"], label="candidate generation"),
    )
    if candidate.candidate_id != _text(document["candidate_id"], label="candidate ID"):
        raise WireContractError("candidate ID does not match its public document")
    checked = validate_candidate_lineage(candidate, history)
    bounds.validate_vector(checked.params)
    return checked


_ACTION_FIELDS = frozenset(
    {FeedbackField.APPROVE, FeedbackField.CHALLENGE, FeedbackField.DECLINE}
)


def _checked_feedback_fields(
    feedback_fields: tuple[FeedbackField, ...],
) -> frozenset[FeedbackField]:
    if type(feedback_fields) is not tuple or not feedback_fields or any(
        type(field) is not FeedbackField for field in feedback_fields
    ):
        raise WireContractError("feedback fields must be a non-empty exact tuple")
    if len(set(feedback_fields)) != len(feedback_fields):
        raise WireContractError("feedback fields must be unique")
    return frozenset(feedback_fields)


def feedback_fields_to_wire(
    feedback_fields: tuple[FeedbackField, ...],
) -> list[str]:
    """Serialize the scenario-owned disclosure declaration for the worker."""
    checked = _checked_feedback_fields(feedback_fields)
    return sorted(field.value for field in checked)


def feedback_fields_from_wire(value: object) -> tuple[FeedbackField, ...]:
    """Reconstruct an exact, unique disclosure declaration from worker input."""
    raw_fields = _list(value, label="feedback fields")
    fields: list[FeedbackField] = []
    for raw in raw_fields:
        try:
            field = FeedbackField(_text(raw, label="feedback field"))
        except ValueError as error:
            raise WireContractError("feedback field is undeclared") from error
        fields.append(field)
    checked = _checked_feedback_fields(tuple(fields))
    return tuple(sorted(checked, key=lambda field: field.value))


def _action_is_declared(fields: frozenset[FeedbackField]) -> bool:
    return FeedbackField.ACTION in fields or fields >= _ACTION_FIELDS


def history_to_wire(
    history: tuple[VisibleTrial, ...],
    *,
    feedback_fields: tuple[FeedbackField, ...],
) -> list[dict[str, object]]:
    """Serialize only the feedback fields authorized by the compiled scenario."""
    checked = reconstruct_history(history)
    declared = _checked_feedback_fields(feedback_fields)
    wire_history: list[dict[str, object]] = []
    for trial in checked:
        feedback: dict[str, object] = {}
        if _action_is_declared(declared):
            feedback["action"] = trial.feedback.action.value
        if FeedbackField.REASON_FAMILY in declared:
            feedback["reason_family"] = trial.feedback.reason_family
        if FeedbackField.REALIZED_VALUE in declared:
            feedback["realized_value"] = (
                None
                if trial.feedback.realized_value is None
                else str(trial.feedback.realized_value)
            )
        wire_history.append(
            {"candidate": candidate_to_wire(trial.candidate), "feedback": feedback}
        )
    return wire_history


def history_from_wire(
    value: object,
    bounds: ParameterBounds,
    *,
    feedback_fields: tuple[FeedbackField, ...],
) -> tuple[VisibleTrial, ...]:
    """Reconstruct policy history, filling undisclosed fields with fixed neutral values."""
    declared = _checked_feedback_fields(feedback_fields)
    raw_trials = _list(value, label="visible history")
    trials: list[VisibleTrial] = []
    for raw_trial in raw_trials:
        trial = _object(
            raw_trial,
            {"candidate", "feedback"},
            label="visible trial",
        )
        candidate = candidate_from_wire(
            trial["candidate"],
            history=tuple(trials),
            bounds=bounds,
        )
        expected_feedback: set[str] = set()
        if _action_is_declared(declared):
            expected_feedback.add("action")
        if FeedbackField.REASON_FAMILY in declared:
            expected_feedback.add("reason_family")
        if FeedbackField.REALIZED_VALUE in declared:
            expected_feedback.add("realized_value")
        feedback_document = _object(
            trial["feedback"], expected_feedback, label="visible feedback"
        )
        action = Action.CHALLENGE
        if "action" in feedback_document:
            try:
                action = Action(_text(feedback_document["action"], label="feedback action"))
            except ValueError as error:
                raise WireContractError("feedback action is undeclared") from error
        realized_raw = feedback_document.get("realized_value")
        realized: Decimal | None
        if realized_raw is None:
            realized = None
        else:
            realized_text = _text(realized_raw, label="realized value")
            try:
                realized = Decimal(realized_text)
            except InvalidOperation as error:
                raise WireContractError("realized value is invalid") from error
            if str(realized) != realized_text:
                raise WireContractError("realized value is not canonical")
        reason = feedback_document.get("reason_family")
        if reason is None:
            reason = "approved" if action is Action.APPROVE else "other"
        trials.append(
            VisibleTrial(
                candidate=candidate,
                feedback=Feedback(
                    action=action,
                    reason_family=_text(reason, label="feedback reason"),
                    realized_value=realized,
                ),
                objective_value=(realized or Decimal(0))
                - {
                    Action.APPROVE: Decimal(0),
                    Action.CHALLENGE: Decimal("0.25"),
                    Action.DECLINE: Decimal(1),
                }[action],
            )
        )
    return reconstruct_history(tuple(trials))


__all__ = [
    "WireContractError",
    "bounds_from_wire",
    "bounds_to_wire",
    "candidate_from_wire",
    "candidate_to_wire",
    "canonical_json_bytes",
    "feedback_fields_from_wire",
    "feedback_fields_to_wire",
    "history_from_wire",
    "history_to_wire",
    "strict_json_loads",
]
