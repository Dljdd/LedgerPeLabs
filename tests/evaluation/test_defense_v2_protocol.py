"""Sealed Defend v2 protocol and frozen-v1 isolation contracts."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

from apar.evaluation.v2_protocol import (
    V2ProtocolError,
    load_v2_protocol,
    verify_v1_roots,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/defense/competition-v2-profile.json"


def copy_v1_roots(destination: Path) -> Path:
    for relative in ("docs/experiments", "fixtures/defense/v1"):
        source = ROOT / relative
        target = destination / relative
        target.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)
    return destination


def test_production_strata_are_exact() -> None:
    p = load_v2_protocol(PROFILE)
    assert p.operating.transaction_count == 100_000
    assert [(s.name, s.fraud_transaction_count) for s in p.strata] == [
        ("low", 100), ("medium", 500), ("high", 1_000)
    ]


def test_v1_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root = copy_v1_roots(tmp_path)
    (root / "docs/experiments/defense-v1-result.json").write_bytes(b"changed")
    with pytest.raises(V2ProtocolError, match="frozen v1 root"):
        verify_v1_roots(root)


def test_profile_is_canonical_and_digest_bound() -> None:
    payload = PROFILE.read_bytes()
    assert payload == json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")).encode()
    assert load_v2_protocol(PROFILE).profile_sha256


def test_fixture_protocol_cannot_serialize_as_preregistration() -> None:
    from apar.evaluation.v2_protocol import V2Protocol

    fixture = V2Protocol.fixture(transaction_count=100)
    with pytest.raises(V2ProtocolError, match="fixture-only"):
        fixture.canonical_bytes()


def test_unknown_and_duplicate_strata_are_rejected(tmp_path: Path) -> None:
    document = json.loads(PROFILE.read_bytes())
    document["strata"] = [document["strata"][0], document["strata"][0], document["strata"][2]]
    document.pop("profile_sha256")
    document["profile_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    with pytest.raises(V2ProtocolError, match="strata"):
        load_v2_protocol(path)


def _rewrite_profile(tmp_path: Path, mutate: object) -> Path:
    document = json.loads(PROFILE.read_bytes())
    mutate(document)
    unsigned = dict(document)
    unsigned.pop("profile_sha256", None)
    document["profile_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "profile.json"
    path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    return path


def test_production_day_and_stratum_denominators_are_sealed(tmp_path: Path) -> None:
    path = _rewrite_profile(tmp_path, lambda d: d["operating"].update(day_count=1))
    with pytest.raises(V2ProtocolError, match="28 synthetic days"):
        load_v2_protocol(path)

    path = _rewrite_profile(tmp_path, lambda d: d["strata"][0].update(transaction_count=99_999))
    with pytest.raises(V2ProtocolError, match="stratum denominator"):
        load_v2_protocol(path)


def test_declared_v1_roots_must_match_frozen_mapping(tmp_path: Path) -> None:
    path = _rewrite_profile(
        tmp_path,
        lambda d: d["v1_roots"].update({"docs/experiments/defense-v1-result.json": "0" * 64}),
    )
    with pytest.raises(V2ProtocolError, match="frozen v1 root mapping"):
        load_v2_protocol(path)

    path = _rewrite_profile(tmp_path, lambda d: d["v1_roots"].pop("docs/experiments/defense-v1-result.json"))
    with pytest.raises(V2ProtocolError, match="frozen v1 root mapping"):
        load_v2_protocol(path)


def test_family_allocation_must_be_equal(tmp_path: Path) -> None:
    path = _rewrite_profile(
        tmp_path,
        lambda d: d["strata"][0].update(family_transaction_counts=[24, 25, 25, 26]),
    )
    with pytest.raises(V2ProtocolError, match="equal family allocation"):
        load_v2_protocol(path)


def test_fixture_rejects_non_divisible_family_count() -> None:
    from apar.evaluation.v2_protocol import PrevalenceStratum

    with pytest.raises(ValueError, match="divisible by four"):
        PrevalenceStratum.fixture(transaction_count=100, fraud_transaction_count=5)
