"""Encrypted seed ledger integrity and fail-closed decryption tests."""

from __future__ import annotations

import pytest

from apar.evaluation.v3_seed_ledger import (
    SealedSeedLedger,
    V3SeedLedgerError,
    commitment_for,
    generate_key,
    key_identity,
    open_seed_ledger,
    seal_seed_ledger,
    verify_commitments,
)


def test_seal_and_open_roundtrip() -> None:
    key = generate_key()
    seeds = {"bootstrap": b"\x01\x02", "hidden_evaluation": b"\x03\x04"}
    ledger = seal_seed_ledger(protocol_id="apar-defend-v3", seeds=seeds, key=key)
    assert open_seed_ledger(ledger, key=key) == seeds


def test_commitment_does_not_reveal_seed() -> None:
    commitment = commitment_for("model_training", b"secret-seed")
    assert "secret" not in commitment.commitment_sha256
    assert len(commitment.commitment_sha256) == 64


def test_wrong_key_fails_closed() -> None:
    key = generate_key()
    wrong = generate_key()
    ledger = seal_seed_ledger(
        protocol_id="apar-defend-v3", seeds={"public_training": b"\x01"}, key=key
    )
    with pytest.raises(V3SeedLedgerError, match="key identity mismatch"):
        open_seed_ledger(ledger, key=wrong)


def test_tampered_ciphertext_fails_authentication() -> None:
    import base64

    key = generate_key()
    ledger = seal_seed_ledger(
        protocol_id="apar-defend-v3", seeds={"calibration_fitting": b"\x01"}, key=key
    )
    raw = bytearray(base64.b64decode(ledger.envelope.ciphertext_base64))
    raw[0] ^= 0xFF
    tampered_envelope = ledger.envelope.model_copy(
        update={
            "ciphertext_base64": base64.b64encode(bytes(raw)).decode("ascii"),
            "payload_sha256": __import__("hashlib").sha256(bytes(raw)).hexdigest(),
        }
    )
    tampered = ledger.model_copy(update={"envelope": tampered_envelope})
    with pytest.raises(V3SeedLedgerError, match="authentication failed"):
        open_seed_ledger(tampered, key=key)


def test_commitments_match_expected_set() -> None:
    key = generate_key()
    seeds = {"benign_only_control": b"\x01", "score_permutation_control": b"\x02"}
    ledger = seal_seed_ledger(protocol_id="apar-defend-v3", seeds=seeds, key=key)
    expected = tuple(commitment_for(name, seed) for name, seed in sorted(seeds.items()))
    verify_commitments(ledger, expected=expected)


def test_commitment_mismatch_is_rejected() -> None:
    key = generate_key()
    ledger = seal_seed_ledger(
        protocol_id="apar-defend-v3", seeds={"threshold_candidate_generation": b"\x01"}, key=key
    )
    wrong = (commitment_for("threshold_candidate_generation", b"\x02"),)
    with pytest.raises(V3SeedLedgerError, match="differ from expected"):
        verify_commitments(ledger, expected=wrong)


def test_nonce_uniqueness_produces_different_ciphertexts() -> None:
    key = generate_key()
    first = seal_seed_ledger(
        protocol_id="apar-defend-v3", seeds={"adversarial_efficacy_generation": b"\x01"}, key=key
    )
    second = seal_seed_ledger(
        protocol_id="apar-defend-v3", seeds={"adversarial_efficacy_generation": b"\x01"}, key=key
    )
    assert first.envelope.ciphertext_base64 != second.envelope.ciphertext_base64


def test_empty_seeds_rejected() -> None:
    key = generate_key()
    with pytest.raises(V3SeedLedgerError, match="at least one seed"):
        seal_seed_ledger(protocol_id="apar-defend-v3", seeds={}, key=key)
