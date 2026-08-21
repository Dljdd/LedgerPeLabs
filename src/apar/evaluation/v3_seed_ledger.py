"""Evaluator-owned encrypted seed ledger for Defend v3.

The ledger seals evaluator-held seed material using AES-256-GCM. The public
preregistration binds only SHA-256 commitments; plaintext seeds never appear in
any manifest or public artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.v3_protocol import SeedCommitment, V3ProtocolError

_NONCE_LENGTH = 12
_KEY_LENGTH = 32
_TAG_LENGTH = 16
_MAX_PAYLOAD_BYTES = 1 << 20


class V3SeedLedgerError(V3ProtocolError):
    """The encrypted seed ledger is malformed, tampered with, or unsealable."""


class SeedLedgerEnvelope(ExternalContract):
    """Canonical sealed seed payload with authenticated encryption metadata."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_id: str
    key_id: str
    nonce_base64: str
    ciphertext_base64: str
    payload_sha256: str

    @model_validator(mode="after")
    def envelope_fields_are_exact(self) -> Self:
        import base64
        import binascii

        if len(self.key_id) != 64 or set(self.key_id) - frozenset("0123456789abcdef"):
            raise ValueError("key_id must be a lowercase SHA-256 digest")
        try:
            nonce = base64.b64decode(self.nonce_base64, validate=True)
            ciphertext = base64.b64decode(self.ciphertext_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("seed ledger base64 fields are invalid") from error
        if len(nonce) != _NONCE_LENGTH:
            raise ValueError("seed ledger nonce must be exactly 12 bytes")
        if len(ciphertext) < _TAG_LENGTH or len(ciphertext) > _MAX_PAYLOAD_BYTES + _TAG_LENGTH:
            raise ValueError("seed ledger ciphertext length is invalid")
        if self.payload_sha256 != hashlib.sha256(ciphertext).hexdigest():
            raise ValueError("seed ledger payload digest mismatch")
        return self


class SealedSeedLedger(ExternalContract):
    """A sealed ledger with public commitments and no plaintext seed material."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_id: str
    commitments: tuple[SeedCommitment, ...]
    envelope: SeedLedgerEnvelope

    @model_validator(mode="after")
    def commitments_are_complete(self) -> Self:
        if not self.commitments:
            raise ValueError("seed ledger requires at least one commitment")
        names = [item.name for item in self.commitments]
        if len(names) != len(set(names)):
            raise ValueError("seed ledger commitment names must be unique")
        return self


def generate_key() -> bytes:
    """Return a fresh 256-bit symmetric key."""
    return os.urandom(_KEY_LENGTH)


def key_identity(key: bytes) -> str:
    """Return the SHA-256 key identity for a 256-bit key."""
    if type(key) is not bytes or len(key) != _KEY_LENGTH:
        raise V3SeedLedgerError("seed ledger key must be exactly 32 bytes")
    return hashlib.sha256(key).hexdigest()


def commitment_for(name: str, seed: bytes) -> SeedCommitment:
    """Derive a public commitment from a named seed without revealing it."""
    if type(name) is not str or not name or type(seed) is not bytes or not seed:
        raise V3SeedLedgerError("seed commitment requires a nonempty name and seed")
    digest = hashlib.sha256(canonical_json_bytes({"name": name, "seed": seed.hex()})).hexdigest()
    return SeedCommitment(name=name, commitment_sha256=digest)


def seal_seed_ledger(
    *,
    protocol_id: str,
    seeds: dict[str, bytes],
    key: bytes,
    nonce: bytes | None = None,
) -> SealedSeedLedger:
    """Encrypt seed material and return a public sealed ledger."""
    if type(protocol_id) is not str or not protocol_id:
        raise V3SeedLedgerError("protocol_id must be nonempty")
    if type(seeds) is not dict or not seeds:
        raise V3SeedLedgerError("seed ledger requires at least one seed")
    if type(key) is not bytes or len(key) != _KEY_LENGTH:
        raise V3SeedLedgerError("seed ledger key must be exactly 32 bytes")
    if any(type(name) is not str or not name for name in seeds):
        raise V3SeedLedgerError("seed names must be nonempty strings")
    if any(type(seed) is not bytes or not seed for seed in seeds.values()):
        raise V3SeedLedgerError("seeds must be nonempty bytes")

    commitments = tuple(commitment_for(name, seed) for name, seed in sorted(seeds.items()))
    payload = canonical_json_bytes(
        {
            "protocol_id": protocol_id,
            "seeds": {name: seed.hex() for name, seed in sorted(seeds.items())},
        }
    )
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise V3SeedLedgerError("seed ledger payload exceeds size limit")
    nonce_bytes = nonce if nonce is not None else os.urandom(_NONCE_LENGTH)
    if type(nonce_bytes) is not bytes or len(nonce_bytes) != _NONCE_LENGTH:
        raise V3SeedLedgerError("seed ledger nonce must be exactly 12 bytes")
    ciphertext = AESGCM(key).encrypt(nonce_bytes, payload, protocol_id.encode("utf-8"))
    envelope = SeedLedgerEnvelope(
        schema_version="1.0.0",
        protocol_id=protocol_id,
        key_id=key_identity(key),
        nonce_base64=__import__("base64").b64encode(nonce_bytes).decode("ascii"),
        ciphertext_base64=__import__("base64").b64encode(ciphertext).decode("ascii"),
        payload_sha256=hashlib.sha256(ciphertext).hexdigest(),
    )
    return SealedSeedLedger(
        schema_version="1.0.0",
        protocol_id=protocol_id,
        commitments=commitments,
        envelope=envelope,
    )


def open_seed_ledger(
    ledger: SealedSeedLedger,
    *,
    key: bytes,
) -> dict[str, bytes]:
    """Decrypt and verify the sealed ledger, returning named seed bytes."""
    if type(key) is not bytes or len(key) != _KEY_LENGTH:
        raise V3SeedLedgerError("seed ledger key must be exactly 32 bytes")
    if ledger.envelope.key_id != key_identity(key):
        raise V3SeedLedgerError("seed ledger key identity mismatch")
    import base64

    ciphertext = base64.b64decode(ledger.envelope.ciphertext_base64, validate=True)
    nonce = base64.b64decode(ledger.envelope.nonce_base64, validate=True)
    try:
        payload = AESGCM(key).decrypt(nonce, ciphertext, ledger.protocol_id.encode("utf-8"))
    except InvalidTag as error:
        raise V3SeedLedgerError("seed ledger authentication failed") from error
    try:
        document = strict_json_loads(payload)
    except WireContractError as error:
        raise V3SeedLedgerError("seed ledger payload is not strict JSON") from error
    if not isinstance(document, dict):
        raise V3SeedLedgerError("seed ledger payload must be an object")
    if document.get("protocol_id") != ledger.protocol_id:
        raise V3SeedLedgerError("seed ledger protocol mismatch")
    raw_seeds = document.get("seeds")
    if not isinstance(raw_seeds, dict):
        raise V3SeedLedgerError("seed ledger payload must contain named seeds")
    result: dict[str, bytes] = {}
    for name, hex_value in raw_seeds.items():
        if type(name) is not str or type(hex_value) is not str:
            raise V3SeedLedgerError("seed ledger entries are malformed")
        try:
            seed = bytes.fromhex(hex_value)
        except ValueError as error:
            raise V3SeedLedgerError("seed ledger seed is not valid hex") from error
        if commitment_for(name, seed) not in ledger.commitments:
            raise V3SeedLedgerError("seed ledger commitment mismatch")
        result[name] = seed
    if set(result) != {item.name for item in ledger.commitments}:
        raise V3SeedLedgerError("seed ledger seeds do not match commitments")
    return result


def verify_commitments(
    ledger: SealedSeedLedger,
    *,
    expected: tuple[SeedCommitment, ...],
) -> None:
    """Verify that ledger commitments exactly match the expected public set."""
    if tuple(sorted(ledger.commitments, key=lambda item: item.name)) != tuple(
        sorted(expected, key=lambda item: item.name)
    ):
        raise V3SeedLedgerError("seed ledger commitments differ from expected set")


__all__ = [
    "SealedSeedLedger",
    "SeedLedgerEnvelope",
    "V3SeedLedgerError",
    "commitment_for",
    "generate_key",
    "key_identity",
    "open_seed_ledger",
    "seal_seed_ledger",
    "verify_commitments",
]
