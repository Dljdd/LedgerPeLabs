"""One-shot verified defender freeze before evaluator-only hidden release."""

from __future__ import annotations

import ast
import hashlib
import hmac
import secrets
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, Self, cast, runtime_checkable

from pydantic import field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.runs.wire import canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

_CAPABILITY_TOKEN = object()
_MAX_CAPABILITY_BYTES = 16_384
_MAX_SOURCE_BYTES = 2_000_000


class _ManifestContract(Protocol):
    """Neutral shape exposed by the pinned verified-bundle loader."""

    bundle_id: str
    threshold_digest: str
    rollback_ref: str
    frozen_at: datetime

    @classmethod
    def model_validate(cls, obj: object, *, strict: bool | None = None) -> Self: ...

    def model_dump(
        self, *, mode: Literal["json", "python"], warnings: bool = False
    ) -> dict[str, object]: ...


class _LoadedBundleContract(Protocol):
    @property
    def manifest(self) -> _ManifestContract: ...


@runtime_checkable
class _VerifiedBundleLoader(Protocol):
    def load(self, top_ref: ArtifactRef) -> _LoadedBundleContract: ...


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


class HiddenEvaluationCapability(bytes):
    """Opaque immutable authority-MACed proof that one defender was frozen."""

    __slots__ = ()

    def __new__(
        cls, payload: bytes, token: object = None
    ) -> HiddenEvaluationCapability:
        if token is not _CAPABILITY_TOKEN or type(payload) is not bytes:
            raise HiddenBoundaryError("hidden capability cannot be constructed externally")
        _capability_document(payload)
        return bytes.__new__(cls, payload)

    def __init__(self, payload: bytes, token: object = None) -> None:
        del payload
        if token is not _CAPABILITY_TOKEN:
            raise HiddenBoundaryError("hidden capability cannot be reinitialized")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("hidden capability is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("hidden capability is immutable")

    def __copy__(self) -> HiddenEvaluationCapability:
        return self

    def __deepcopy__(self, memo: object) -> HiddenEvaluationCapability:
        del memo
        return self

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("hidden capabilities cannot cross evaluator processes")

    def to_bytes(self) -> bytes:
        """Return opaque bytes; no constructor accepts them without this authority."""
        return bytes(memoryview(self))

    @property
    def bundle_manifest_digest(self) -> str:
        return cast(str, _capability_document(self.to_bytes())["bundle_manifest_digest"])

    @property
    def bundle_id(self) -> str:
        return cast(str, _capability_document(self.to_bytes())["bundle_id"])

    @property
    def issued_at(self) -> datetime:
        value = cast(str, _capability_document(self.to_bytes())["issued_at"])
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def __repr__(self) -> str:
        return (
            "HiddenEvaluationCapability("
            f"bundle_manifest_digest={self.bundle_manifest_digest!r}, "
            f"bundle_id={self.bundle_id!r}, issued_at={self.issued_at.isoformat()!r})"
        )

    __str__ = __repr__


class HiddenEvaluationAuthority:
    """Pinned one-shot authority resolving restricted refs only after bundle load."""

    __slots__ = (
        "_active_capability_digest",
        "_bundle_publisher",
        "_mac_key",
        "_restricted_store",
        "_store_token",
    )
    _active_capability_digest: str | None
    _bundle_publisher: _VerifiedBundleLoader
    _mac_key: bytes
    _restricted_store: ArtifactStore
    _store_token: str

    def __init__(
        self,
        bundle_publisher: _VerifiedBundleLoader,
        restricted_store: ArtifactStore,
    ) -> None:
        if hasattr(self, "_mac_key"):
            raise HiddenBoundaryError("hidden authority is already initialized")
        if not isinstance(bundle_publisher, _VerifiedBundleLoader):
            raise HiddenBoundaryError("hidden authority requires a verified bundle loader")
        if type(restricted_store) is not ArtifactStore:
            raise HiddenBoundaryError("hidden authority requires an exact restricted store")
        object.__setattr__(self, "_bundle_publisher", bundle_publisher)
        object.__setattr__(self, "_restricted_store", restricted_store)
        object.__setattr__(self, "_mac_key", secrets.token_bytes(32))
        object.__setattr__(self, "_store_token", secrets.token_hex(32))
        object.__setattr__(self, "_active_capability_digest", None)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("hidden evaluation authority is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("hidden evaluation authority is immutable")

    def freeze_and_issue(
        self,
        manifest: _ManifestContract,
        top_ref: ArtifactRef,
        *,
        issued_at: datetime,
    ) -> HiddenEvaluationCapability:
        """Verify signed top ref and rollback chain, then seal a one-shot capability."""
        if self._active_capability_digest is not None:
            raise HiddenBoundaryError("hidden authority is already frozen")
        try:
            if type(top_ref) is not ArtifactRef:
                raise TypeError("top reference must be exact")
            if type(issued_at) is not datetime:
                raise TypeError("issue time must be exact")
            validate_utc_timestamp(issued_at)
            loaded = self._bundle_publisher.load(top_ref)
            loaded_manifest = loaded.manifest
            if type(manifest) is not type(loaded_manifest):
                raise TypeError("manifest must have the verified loader's exact type")
            checked_manifest = type(loaded_manifest).model_validate(
                manifest.model_dump(mode="python", warnings=False), strict=True
            )
            if loaded.manifest != checked_manifest:
                raise ValueError("top reference does not identify supplied manifest")
            if top_ref.sha256 != hashlib.sha256(
                canonical_json_bytes(checked_manifest.model_dump(mode="json"))
            ).hexdigest():
                raise ValueError("top reference is not the canonical manifest")
            if checked_manifest.frozen_at > issued_at:
                raise ValueError("defender freeze time follows hidden capability issue")
        except Exception as error:
            raise HiddenBoundaryError(
                "hidden release requires a verified signed frozen defender"
            ) from error

        document: dict[str, object] = {
            "schema_version": "1.0.0",
            "bundle_manifest_digest": top_ref.sha256,
            "bundle_id": checked_manifest.bundle_id,
            "threshold_digest": checked_manifest.threshold_digest,
            "rollback_ref": checked_manifest.rollback_ref,
            "frozen_at": checked_manifest.frozen_at.isoformat().replace("+00:00", "Z"),
            "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
            "store_token": self._store_token,
            "nonce": secrets.token_hex(32),
        }
        unsigned = canonical_json_bytes(document)
        payload = canonical_json_bytes(
            {
                **document,
                "authority_mac": hmac.new(
                    self._mac_key, unsigned, hashlib.sha256
                ).hexdigest(),
            }
        )
        capability = HiddenEvaluationCapability(payload, _CAPABILITY_TOKEN)
        object.__setattr__(
            self, "_active_capability_digest", hashlib.sha256(payload).hexdigest()
        )
        return capability

    def resolve(
        self,
        capability: HiddenEvaluationCapability | None,
        restricted_ref: ArtifactRef,
    ) -> bytes:
        """Resolve through the pinned restricted store after exact capability proof."""
        if self._active_capability_digest is None:
            raise HiddenBoundaryError("restricted refs require a frozen defender")
        try:
            if type(capability) is not HiddenEvaluationCapability:
                raise TypeError("capability must be exact")
            payload = capability.to_bytes()
            if hashlib.sha256(payload).hexdigest() != self._active_capability_digest:
                raise ValueError("capability does not match active freeze")
            document = _capability_document(payload)
            supplied_mac = cast(str, document.pop("authority_mac"))
            expected_mac = hmac.new(
                self._mac_key, canonical_json_bytes(document), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(supplied_mac, expected_mac):
                raise ValueError("capability MAC mismatch")
            if document.get("store_token") != self._store_token:
                raise ValueError("capability belongs to another store authority")
        except Exception as error:
            raise HiddenBoundaryError("hidden evaluation capability is invalid") from error
        try:
            if type(restricted_ref) is not ArtifactRef:
                raise TypeError("restricted ref must be exact")
            return self._restricted_store.read(restricted_ref)
        except Exception as error:
            raise HiddenBoundaryError(
                "restricted reference is invalid for the pinned store"
            ) from error


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
    dynamic_import_names = {"__import__", "importlib.import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    dynamic_import_names.add(
                        f"{alias.asname or alias.name}.import_module"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    dynamic_import_names.add(alias.asname or alias.name)
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
        elif isinstance(node, ast.Call) and node.args:
            name = _call_name(node.func)
            argument = node.args[0]
            if name in dynamic_import_names and isinstance(
                argument, ast.Constant
            ) and type(argument.value) is str and _is_hidden_module(argument.value):
                found.append((node.lineno, argument.value))
    return tuple(sorted(set(found)))


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return ""


def _is_hidden_module(module: str) -> bool:
    return module == "apar.evaluation_hidden" or module.startswith(
        "apar.evaluation_hidden."
    )


def _capability_document(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_CAPABILITY_BYTES:
        raise HiddenBoundaryError("hidden capability payload is invalid")
    try:
        document = strict_json_loads(payload)
    except Exception as error:
        raise HiddenBoundaryError("hidden capability payload is invalid") from error
    if type(document) is not dict:
        raise HiddenBoundaryError("hidden capability payload must be an object")
    expected = {
        "schema_version",
        "bundle_manifest_digest",
        "bundle_id",
        "threshold_digest",
        "rollback_ref",
        "frozen_at",
        "issued_at",
        "store_token",
        "nonce",
        "authority_mac",
    }
    if set(document) != expected or canonical_json_bytes(document) != payload:
        raise HiddenBoundaryError("hidden capability payload is not canonical")
    if document.get("schema_version") != "1.0.0":
        raise HiddenBoundaryError("hidden capability schema is unsupported")
    for name in (
        "bundle_manifest_digest",
        "threshold_digest",
        "store_token",
        "nonce",
        "authority_mac",
    ):
        value = document.get(name)
        if type(value) is not str or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise HiddenBoundaryError(f"hidden capability {name} is invalid")
    for name in ("frozen_at", "issued_at"):
        value = document.get(name)
        if type(value) is not str:
            raise HiddenBoundaryError("hidden capability timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            validate_utc_timestamp(parsed)
        except (TypeError, ValueError) as error:
            raise HiddenBoundaryError("hidden capability timestamp is invalid") from error
    return dict(document)


__all__ = [
    "HiddenBoundaryError",
    "HiddenEvaluationAuthority",
    "HiddenEvaluationCapability",
    "HiddenImportAudit",
    "audit_hidden_import_boundary",
]
