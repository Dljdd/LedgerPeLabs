"""One-shot verified defender freeze before evaluator-only hidden release."""

from __future__ import annotations

import ast
import hashlib
import hmac
import secrets
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Never

from pydantic import ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.evaluation.defender_attestation import (
    DefenderBundleVerifier,
    VerifiedDefenderAttestation,
)
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

_OBJECT_TOKEN = object()
_MAX_SOURCE_BYTES = 2_000_000
_MAX_HIDDEN_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_RELEASES = 64
HIDDEN_CONTEXT_MEDIA_TYPE = "application/vnd.apar.hidden-evaluation-context+json"


class HiddenBoundaryError(ValueError):
    """A hidden-release caller violated freeze, authority, or store isolation."""


class HiddenImportAudit(ExternalContract):
    """Static evidence that defender packages cannot import evaluator internals."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    passed: bool
    scanned_files: tuple[str, ...]
    violations: tuple[str, ...]
    audit_digest: str

    @field_validator("scanned_files", "violations", mode="before")
    @classmethod
    def collections_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("hidden import audit collections must be exact tuples")
        return value

    @model_validator(mode="after")
    def evidence_is_canonical(self) -> HiddenImportAudit:
        if self.scanned_files != tuple(sorted(set(self.scanned_files))):
            raise ValueError("scanned hidden-boundary files must be sorted and unique")
        if self.violations != tuple(sorted(set(self.violations))):
            raise ValueError("hidden-boundary violations must be sorted and unique")
        if self.passed != (not self.violations):
            raise ValueError("hidden import audit pass state is inconsistent")
        expected = hashlib.sha256(
            canonical_json_bytes(
                self.model_dump(mode="json", exclude={"audit_digest"})
            )
        ).hexdigest()
        if self.audit_digest != expected:
            raise ValueError("hidden import audit digest is inconsistent")
        return self


class HiddenEvaluationCapability:
    """Opaque exact-identity proof of one attested defender freeze."""

    __slots__ = ("__weakref__",)

    def __new__(cls, token: object = None) -> HiddenEvaluationCapability:
        if token is not _OBJECT_TOKEN:
            raise HiddenBoundaryError("hidden capability cannot be constructed externally")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _OBJECT_TOKEN:
            raise HiddenBoundaryError("hidden capability cannot be reinitialized")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("hidden capability is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("hidden capability is immutable")

    def __copy__(self) -> Never:
        raise TypeError("hidden capability identity cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("hidden capability identity cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("hidden capability identity cannot be serialized")

    @property
    def bundle_manifest_digest(self) -> str:
        return _capability_state(self).bundle_manifest_digest

    @property
    def bundle_id(self) -> str:
        return _capability_state(self).bundle_id

    @property
    def issued_at(self) -> datetime:
        return _capability_state(self).issued_at

    def __repr__(self) -> str:
        state = _capability_state(self)
        return (
            "HiddenEvaluationCapability("
            f"bundle_manifest_digest={state.bundle_manifest_digest!r}, "
            f"bundle_id={state.bundle_id!r}, issued_at={state.issued_at.isoformat()!r})"
        )

    __str__ = __repr__


class HiddenArmEvidenceBinding(ExternalContract):
    """One exact Task 11 evaluator-input/evidence binding."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    evaluator_input_digest: str
    derivation_evidence_digest: str
    metric_report_digest: str

    @field_validator(
        "evaluator_input_digest", "derivation_evidence_digest", "metric_report_digest"
    )
    @classmethod
    def evidence_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value


class HiddenEvaluationReceipt(ExternalContract):
    """Authority-MACed proof of hidden content release and exact evaluator use."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    capability_digest: str
    defender_attestation_digest: str
    defender_top_ref_digest: str
    bundle_manifest_digest: str
    restricted_ref_digest: str
    restricted_artifact_digest: str
    canonical_content_digest: str
    evaluator_context_digest: str
    release_sequence: int
    released_at: datetime
    sealed_at: datetime
    arm_evidence: tuple[HiddenArmEvidenceBinding, ...]
    authority_mac: str
    receipt_digest: str

    @field_validator(
        "capability_digest",
        "defender_attestation_digest",
        "defender_top_ref_digest",
        "bundle_manifest_digest",
        "restricted_ref_digest",
        "restricted_artifact_digest",
        "canonical_content_digest",
        "evaluator_context_digest",
        "authority_mac",
        "receipt_digest",
    )
    @classmethod
    def receipt_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @field_validator("release_sequence", mode="before")
    @classmethod
    def sequence_is_exact_bounded_int(cls, value: object) -> object:
        if type(value) is not int or not 1 <= value <= _MAX_RELEASES:
            raise ValueError("hidden release sequence is invalid")
        return value

    @field_validator("released_at", "sealed_at")
    @classmethod
    def receipt_times_are_utc(cls, value: datetime) -> datetime:
        if type(value) is not datetime:
            raise ValueError("hidden receipt timestamps must be exact datetimes")
        return validate_utc_timestamp(value)

    @field_validator("arm_evidence", mode="before")
    @classmethod
    def arm_evidence_is_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("hidden arm evidence must be an exact tuple")
        return value

    @model_validator(mode="after")
    def receipt_is_canonical(self) -> HiddenEvaluationReceipt:
        if tuple(item.arm for item in self.arm_evidence) != (
            "rules_only",
            "gbdt_only",
            "layered_hybrid",
        ):
            raise ValueError("hidden receipt must bind all arms in canonical order")
        if self.sealed_at < self.released_at:
            raise ValueError("hidden receipt cannot precede release")
        expected = _digest_bytes(
            canonical_json_bytes(
                self.model_dump(mode="json", exclude={"receipt_digest"})
            )
        )
        if self.receipt_digest != expected:
            raise ValueError("hidden receipt digest is inconsistent")
        return self

    def to_json(self) -> bytes:
        if type(self) is not HiddenEvaluationReceipt:
            raise HiddenBoundaryError("hidden receipt must have its exact type")
        checked = HiddenEvaluationReceipt.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        payload = canonical_json_bytes(checked.model_dump(mode="json"))
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise HiddenBoundaryError("hidden receipt exceeds its resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> HiddenEvaluationReceipt:
        if type(payload) is not bytes or len(payload) > _MAX_RECEIPT_BYTES:
            raise HiddenBoundaryError("hidden receipt payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise HiddenBoundaryError("hidden receipt must be a JSON object")
            if type(document.get("arm_evidence")) is list:
                document["arm_evidence"] = tuple(document["arm_evidence"])
            receipt = cls.model_validate(document)
            if receipt.to_json() != payload:
                raise HiddenBoundaryError("hidden receipt JSON is not canonical")
            return receipt
        except (ValidationError, WireContractError) as error:
            raise HiddenBoundaryError(str(error)) from error


class HiddenReleaseRequest:
    """Opaque one-use request that contains no resolved restricted bytes."""

    __slots__ = ("__weakref__",)

    def __new__(cls, token: object = None) -> HiddenReleaseRequest:
        if token is not _OBJECT_TOKEN:
            raise HiddenBoundaryError("hidden release request cannot be constructed")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _OBJECT_TOKEN:
            raise HiddenBoundaryError("hidden release request cannot be reinitialized")

    def __copy__(self) -> Never:
        raise TypeError("hidden release request cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("hidden release request cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("hidden release request cannot be serialized")


class ResolvedHiddenEvaluation:
    """Exact canonical evaluator bytes resolved only after decisions freeze."""

    __slots__ = ("__weakref__",)

    def __new__(cls, token: object = None) -> ResolvedHiddenEvaluation:
        if token is not _OBJECT_TOKEN:
            raise HiddenBoundaryError("resolved hidden evaluation cannot be constructed")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _OBJECT_TOKEN:
            raise HiddenBoundaryError("resolved hidden evaluation cannot be reinitialized")

    @property
    def payload(self) -> bytes:
        return bytes(_resolved_state(self).payload)


@dataclass(frozen=True, slots=True)
class _CapabilityState:
    digest: str
    attestation: VerifiedDefenderAttestation
    bundle_manifest_digest: str
    bundle_id: str
    issued_at: datetime


@dataclass(slots=True)
class _AuthorityState:
    verifier: DefenderBundleVerifier
    restricted_store: ArtifactStore
    mac_key: bytes
    active_capability: HiddenEvaluationCapability | None = None
    release_sequence: int = 0
    active_receipt: HiddenEvaluationReceipt | None = None
    sealed_sequences: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _RequestState:
    authority_ref: weakref.ReferenceType[HiddenEvaluationAuthority]
    capability: HiddenEvaluationCapability
    restricted_ref: ArtifactRef
    released_at: datetime
    sequence: int
    used: bool = False


@dataclass(frozen=True, slots=True)
class _ResolvedState:
    authority_ref: weakref.ReferenceType[HiddenEvaluationAuthority]
    capability: HiddenEvaluationCapability
    restricted_ref: ArtifactRef
    released_at: datetime
    sequence: int
    payload: bytes
    canonical_content_digest: str


_AUTHORITIES: weakref.WeakKeyDictionary[HiddenEvaluationAuthority, _AuthorityState] = (
    weakref.WeakKeyDictionary()
)
_CAPABILITIES: weakref.WeakKeyDictionary[HiddenEvaluationCapability, _CapabilityState] = (
    weakref.WeakKeyDictionary()
)
_REQUESTS: weakref.WeakKeyDictionary[HiddenReleaseRequest, _RequestState] = (
    weakref.WeakKeyDictionary()
)
_RESOLVED: weakref.WeakKeyDictionary[ResolvedHiddenEvaluation, _ResolvedState] = (
    weakref.WeakKeyDictionary()
)


class HiddenEvaluationAuthority:
    """Pinned authority whose mutable state is unreachable through attributes."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        verifier: DefenderBundleVerifier,
        restricted_store: ArtifactStore,
    ) -> None:
        if self in _AUTHORITIES:
            raise HiddenBoundaryError("hidden authority is already initialized")
        if type(verifier) is not DefenderBundleVerifier:
            raise HiddenBoundaryError("hidden authority requires the exact neutral verifier")
        if type(restricted_store) is not ArtifactStore:
            raise HiddenBoundaryError("hidden authority requires an exact restricted store")
        _AUTHORITIES[self] = _AuthorityState(
            verifier=verifier,
            restricted_store=restricted_store,
            mac_key=secrets.token_bytes(32),
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("hidden evaluation authority is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("hidden evaluation authority is immutable")

    def freeze_and_issue(
        self,
        attestation: VerifiedDefenderAttestation,
        *,
        issued_at: datetime,
    ) -> HiddenEvaluationCapability:
        """Independently reverify one exact attestation and issue only once."""
        state = _authority_state(self)
        if state.active_capability is not None:
            raise HiddenBoundaryError("hidden authority is already frozen")
        if type(attestation) is not VerifiedDefenderAttestation or not state.verifier.verify(
            attestation
        ):
            raise HiddenBoundaryError(
                "hidden release requires an exact verified signed defender attestation"
            )
        if type(issued_at) is not datetime:
            raise HiddenBoundaryError("hidden capability issue time must be exact")
        try:
            validate_utc_timestamp(issued_at)
        except ValueError as error:
            raise HiddenBoundaryError("hidden capability issue time is invalid") from error
        if attestation.frozen_at > issued_at:
            raise HiddenBoundaryError("defender freeze follows hidden capability issue")
        capability = HiddenEvaluationCapability(_OBJECT_TOKEN)
        capability_digest = _digest_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "1.0.0",
                    "attestation_digest": attestation.attestation_digest,
                    "bundle_manifest_digest": attestation.bundle_manifest_digest,
                    "bundle_id": attestation.bundle_id,
                    "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
                    "nonce": secrets.token_hex(32),
                }
            )
        )
        _CAPABILITIES[capability] = _CapabilityState(
            digest=capability_digest,
            attestation=attestation,
            bundle_manifest_digest=attestation.bundle_manifest_digest,
            bundle_id=attestation.bundle_id,
            issued_at=issued_at,
        )
        state.active_capability = capability
        return capability

    def prepare_release(
        self,
        capability: HiddenEvaluationCapability | None,
        restricted_ref: ArtifactRef,
        *,
        released_at: datetime,
    ) -> HiddenReleaseRequest:
        """Bind a one-use restricted ref without reading its payload."""
        state = _authority_state(self)
        if state.active_capability is None:
            raise HiddenBoundaryError("restricted refs require a frozen defender")
        if (
            type(capability) is not HiddenEvaluationCapability
            or capability is not state.active_capability
        ):
            raise HiddenBoundaryError("hidden capability identity is invalid")
        if type(restricted_ref) is not ArtifactRef:
            raise HiddenBoundaryError("restricted ref must be exact")
        if (
            restricted_ref.media_type != HIDDEN_CONTEXT_MEDIA_TYPE
            or restricted_ref.size_bytes < 1
            or restricted_ref.size_bytes > _MAX_HIDDEN_BYTES
        ):
            raise HiddenBoundaryError("restricted ref violates hidden context media limits")
        if type(released_at) is not datetime:
            raise HiddenBoundaryError("hidden release time must be exact")
        try:
            validate_utc_timestamp(released_at)
        except ValueError as error:
            raise HiddenBoundaryError("hidden release time is invalid") from error
        if released_at < _capability_state(capability).issued_at:
            raise HiddenBoundaryError("hidden release precedes defender capability")
        if state.release_sequence >= _MAX_RELEASES:
            raise HiddenBoundaryError("hidden release cap is exhausted")
        state.release_sequence += 1
        request = HiddenReleaseRequest(_OBJECT_TOKEN)
        _REQUESTS[request] = _RequestState(
            authority_ref=weakref.ref(self),
            capability=capability,
            restricted_ref=restricted_ref,
            released_at=released_at,
            sequence=state.release_sequence,
        )
        return request

    def receipt_from_json(self, payload: bytes) -> HiddenEvaluationReceipt:
        """Return only the active exact receipt for identical canonical bytes."""
        state = _authority_state(self)
        candidate = HiddenEvaluationReceipt.from_json(payload)
        if state.active_receipt is None or candidate != state.active_receipt:
            raise HiddenBoundaryError("hidden receipt is not active for this authority")
        return state.active_receipt


def resolve_hidden_release(request: HiddenReleaseRequest) -> ResolvedHiddenEvaluation:
    """Resolve canonical restricted bytes after replay freezes all decisions."""
    if type(request) is not HiddenReleaseRequest:
        raise HiddenBoundaryError("hidden release request must be exact")
    try:
        request_state = _REQUESTS[request]
    except KeyError as error:
        raise HiddenBoundaryError("hidden release request is not authority issued") from error
    if request_state.used:
        raise HiddenBoundaryError("hidden release request is already consumed")
    authority = request_state.authority_ref()
    if authority is None:
        raise HiddenBoundaryError("hidden release authority expired")
    state = _authority_state(authority)
    if state.active_capability is not request_state.capability:
        raise HiddenBoundaryError("hidden release capability is no longer active")
    try:
        payload = state.restricted_store.read(request_state.restricted_ref)
        if type(payload) is not bytes or not payload or len(payload) > _MAX_HIDDEN_BYTES:
            raise HiddenBoundaryError("restricted hidden payload violates resource limits")
        document = strict_json_loads(payload)
        if canonical_json_bytes(document) != payload:
            raise HiddenBoundaryError("restricted hidden payload is not canonical JSON")
    except HiddenBoundaryError:
        raise
    except Exception as error:
        raise HiddenBoundaryError(
            "restricted reference is invalid for the pinned store"
        ) from error
    request_state.used = True
    resolved = ResolvedHiddenEvaluation(_OBJECT_TOKEN)
    _RESOLVED[resolved] = _ResolvedState(
        authority_ref=request_state.authority_ref,
        capability=request_state.capability,
        restricted_ref=request_state.restricted_ref,
        released_at=request_state.released_at,
        sequence=request_state.sequence,
        payload=payload,
        canonical_content_digest=_digest_bytes(payload),
    )
    return resolved


def seal_hidden_evaluation(
    resolved: ResolvedHiddenEvaluation,
    arm_evidence: tuple[HiddenArmEvidenceBinding, ...],
    *,
    sealed_at: datetime,
) -> HiddenEvaluationReceipt:
    """Seal exact Task 11 evidence digests after all three evaluations."""
    if type(resolved) is not ResolvedHiddenEvaluation or type(arm_evidence) is not tuple:
        raise HiddenBoundaryError("hidden receipt inputs must have exact types")
    resolved_state = _resolved_state(resolved)
    authority = resolved_state.authority_ref()
    if authority is None:
        raise HiddenBoundaryError("hidden release authority expired")
    state = _authority_state(authority)
    capability_state = _capability_state(resolved_state.capability)
    if state.active_capability is not resolved_state.capability:
        raise HiddenBoundaryError("hidden release capability is not active")
    if resolved_state.sequence in state.sealed_sequences:
        raise HiddenBoundaryError("hidden release is already sealed")
    evidence: list[HiddenArmEvidenceBinding] = []
    for item in arm_evidence:
        if type(item) is not HiddenArmEvidenceBinding:
            raise HiddenBoundaryError("hidden arm evidence must have exact types")
        try:
            evidence.append(
                HiddenArmEvidenceBinding.model_validate(
                    item.model_dump(mode="python", warnings=False), strict=True
                )
            )
        except ValidationError as error:
            raise HiddenBoundaryError("hidden arm evidence failed validation") from error
    evidence_tuple = tuple(evidence)
    if type(sealed_at) is not datetime:
        raise HiddenBoundaryError("hidden receipt seal time must be exact")
    try:
        validate_utc_timestamp(sealed_at)
    except ValueError as error:
        raise HiddenBoundaryError("hidden receipt seal time is invalid") from error
    fields: dict[str, object] = {
        "capability_digest": capability_state.digest,
        "defender_attestation_digest": capability_state.attestation.attestation_digest,
        "defender_top_ref_digest": capability_state.attestation.top_ref.sha256,
        "bundle_manifest_digest": capability_state.bundle_manifest_digest,
        "restricted_ref_digest": _digest_bytes(
            canonical_json_bytes(_ref_document(resolved_state.restricted_ref))
        ),
        "restricted_artifact_digest": resolved_state.restricted_ref.sha256,
        "canonical_content_digest": resolved_state.canonical_content_digest,
        "evaluator_context_digest": _evaluator_context_digest(
            resolved_state.payload
        ),
        "release_sequence": resolved_state.sequence,
        "released_at": resolved_state.released_at,
        "sealed_at": sealed_at,
        "arm_evidence": evidence_tuple,
    }
    mac_document = _receipt_wire_document(fields)
    authority_mac = hmac.new(
        state.mac_key, canonical_json_bytes(mac_document), hashlib.sha256
    ).hexdigest()
    provisional = {**fields, "authority_mac": authority_mac}
    receipt_digest = _digest_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0.0",
                **mac_document,
                "authority_mac": authority_mac,
            }
        )
    )
    receipt = HiddenEvaluationReceipt.model_validate(
        {**provisional, "receipt_digest": receipt_digest}
    )
    state.sealed_sequences.add(resolved_state.sequence)
    state.active_receipt = receipt
    return receipt


def verify_hidden_receipt(
    receipt: HiddenEvaluationReceipt,
    resolved: ResolvedHiddenEvaluation,
    arm_evidence: tuple[HiddenArmEvidenceBinding, ...],
) -> bool:
    """Require exact active receipt identity, MAC, content, and arm bindings."""
    if (
        type(receipt) is not HiddenEvaluationReceipt
        or type(resolved) is not ResolvedHiddenEvaluation
        or type(arm_evidence) is not tuple
    ):
        return False
    try:
        resolved_state = _resolved_state(resolved)
        authority = resolved_state.authority_ref()
        if authority is None:
            return False
        state = _authority_state(authority)
        if state.active_receipt is not receipt or receipt.arm_evidence != arm_evidence:
            return False
        fields = receipt.model_dump(
            mode="python",
            exclude={"schema_version", "authority_mac", "receipt_digest"},
        )
        expected_mac = hmac.new(
            state.mac_key,
            canonical_json_bytes(_receipt_wire_document(fields)),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(receipt.authority_mac, expected_mac)
    except Exception:
        return False


def audit_hidden_import_boundary(apar_source_root: Path) -> HiddenImportAudit:
    """AST-audit defense/features imports, including literal dynamic imports."""
    if not isinstance(apar_source_root, Path):
        raise HiddenBoundaryError("APAR source root must be an exact Path")
    root = apar_source_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise HiddenBoundaryError("APAR source root must be a regular directory")
    scanned: list[str] = []
    violations: list[str] = []
    for package in ("defense", "features"):
        package_root = root / package
        if not package_root.is_dir() or package_root.is_symlink():
            raise HiddenBoundaryError(f"missing defender package: {package}")
        for path in sorted(package_root.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                raise HiddenBoundaryError("hidden import audit rejects non-regular source")
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            if len(payload) > _MAX_SOURCE_BYTES:
                raise HiddenBoundaryError("hidden import audit source exceeds resource cap")
            try:
                tree = ast.parse(payload, filename=relative)
            except (SyntaxError, ValueError) as error:
                raise HiddenBoundaryError(f"hidden import audit cannot parse {relative}") from error
            scanned.append(relative)
            for line, module in _hidden_imports(tree):
                violations.append(f"{relative}:{line}:{module}")
    scanned_tuple = tuple(sorted(scanned))
    violations_tuple = tuple(sorted(violations))
    fields: dict[str, object] = {
        "passed": not violations,
        "scanned_files": scanned_tuple,
        "violations": violations_tuple,
    }
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "1.0.0",
                "passed": not violations,
                "scanned_files": list(scanned_tuple),
                "violations": list(violations_tuple),
            }
        )
    ).hexdigest()
    return HiddenImportAudit.model_validate({**fields, "audit_digest": digest})


def _hidden_imports(tree: ast.AST) -> tuple[tuple[int, str], ...]:
    found: list[tuple[int, str]] = []
    importlib_names = {"importlib"}
    builtins_names = {"builtins"}
    dynamic_import_names = {"__import__", "importlib.import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
                elif alias.name == "builtins":
                    builtins_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    dynamic_import_names.add(alias.asname or alias.name)
    dynamic_import_names.update(f"{name}.import_module" for name in importlib_names)
    dynamic_import_names.update(f"{name}.__import__" for name in builtins_names)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            targets = (
                list(node.targets)
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    alias_name = _call_name(target)
                    source_name = _call_name(value)
                    if source_name in importlib_names and alias_name not in importlib_names:
                        importlib_names.add(alias_name)
                        dynamic_import_names.add(f"{alias_name}.import_module")
                        changed = True
                    if source_name in builtins_names and alias_name not in builtins_names:
                        builtins_names.add(alias_name)
                        dynamic_import_names.add(f"{alias_name}.__import__")
                        changed = True
                    if alias_name and (
                        source_name in dynamic_import_names
                        or _is_getattr_import(value, importlib_names, builtins_names)
                    ) and alias_name not in dynamic_import_names:
                        dynamic_import_names.add(alias_name)
                        changed = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_hidden_module(alias.name):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_hidden_module(module):
                found.append((node.lineno, module))
            elif module == "apar":
                for alias in node.names:
                    if alias.name == "evaluation_hidden":
                        found.append((node.lineno, "apar.evaluation_hidden"))
            elif node.level and (
                module == "evaluation_hidden"
                or module.startswith("evaluation_hidden.")
            ):
                found.append((node.lineno, module))
            elif node.level and not module:
                for alias in node.names:
                    if alias.name == "evaluation_hidden":
                        found.append((node.lineno, "apar.evaluation_hidden"))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name not in dynamic_import_names:
                continue
            argument: ast.expr | None = node.args[0] if node.args else None
            if argument is None:
                for keyword in node.keywords:
                    if keyword.arg in {"name", "module"}:
                        argument = keyword.value
                        break
            if (
                isinstance(argument, ast.Constant)
                and type(argument.value) is str
            ):
                if _is_hidden_module(argument.value):
                    found.append((node.lineno, argument.value))
            else:
                found.append((node.lineno, "<unresolved-dynamic-import>"))
    return tuple(sorted(set(found)))


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _is_getattr_import(
    node: ast.expr,
    importlib_names: set[str],
    builtins_names: set[str],
) -> bool:
    if (
        not isinstance(node, ast.Call)
        or _call_name(node.func) != "getattr"
        or len(node.args) < 2
        or not isinstance(node.args[1], ast.Constant)
        or type(node.args[1].value) is not str
    ):
        return False
    owner = _call_name(node.args[0])
    attribute = node.args[1].value
    return (owner in importlib_names and attribute == "import_module") or (
        owner in builtins_names and attribute == "__import__"
    )


def _is_hidden_module(module: str) -> bool:
    return module == "apar.evaluation_hidden" or module.startswith(
        "apar.evaluation_hidden."
    )


def _authority_state(authority: HiddenEvaluationAuthority) -> _AuthorityState:
    if type(authority) is not HiddenEvaluationAuthority:
        raise HiddenBoundaryError("hidden authority must have its exact type")
    try:
        return _AUTHORITIES[authority]
    except KeyError as error:
        raise HiddenBoundaryError("hidden authority is not initialized") from error


def _capability_state(capability: HiddenEvaluationCapability) -> _CapabilityState:
    if type(capability) is not HiddenEvaluationCapability:
        raise HiddenBoundaryError("hidden capability must have its exact type")
    try:
        return _CAPABILITIES[capability]
    except KeyError as error:
        raise HiddenBoundaryError("hidden capability is not authority issued") from error


def _resolved_state(resolved: ResolvedHiddenEvaluation) -> _ResolvedState:
    if type(resolved) is not ResolvedHiddenEvaluation:
        raise HiddenBoundaryError("resolved hidden evaluation must have its exact type")
    try:
        return _RESOLVED[resolved]
    except KeyError as error:
        raise HiddenBoundaryError("resolved hidden evaluation is not authority issued") from error


def _ref_document(ref: ArtifactRef) -> dict[str, object]:
    return {
        "sha256": ref.sha256,
        "media_type": ref.media_type,
        "size_bytes": ref.size_bytes,
        "relative_path": ref.relative_path,
    }


def _receipt_wire_document(fields: dict[str, object]) -> dict[str, object]:
    document: dict[str, object] = {}
    for name, value in fields.items():
        if isinstance(value, datetime):
            document[name] = value.isoformat().replace("+00:00", "Z")
        elif type(value) is tuple:
            document[name] = [
                item.model_dump(mode="json")
                if type(item) is HiddenArmEvidenceBinding
                else item
                for item in value
            ]
        else:
            document[name] = value
    return document


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evaluator_context_digest(payload: bytes) -> str:
    return _digest_bytes(b"apar-hidden-evaluator-context-v1\x00" + payload)


def _validate_digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("hidden evidence digest must be lowercase SHA-256")


__all__ = [
    "HiddenBoundaryError",
    "HiddenEvaluationAuthority",
    "HiddenEvaluationCapability",
    "HiddenEvaluationReceipt",
    "HiddenArmEvidenceBinding",
    "HiddenImportAudit",
    "HiddenReleaseRequest",
    "ResolvedHiddenEvaluation",
    "audit_hidden_import_boundary",
    "resolve_hidden_release",
    "seal_hidden_evaluation",
    "verify_hidden_receipt",
]
