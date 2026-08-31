from __future__ import annotations

import pytest

from scripts.submission.model import ReleaseError
from scripts.submission.scan import scan_payloads


def test_scanner_accepts_plain_synthetic_release_content() -> None:
    """Ordinary code and explicitly synthetic fixture text should remain shippable."""
    payloads = {
        "NOTICE.md": b"Synthetic APAR demonstration.\n",
        "scenario.json": b'{"privacy_classification":"synthetic"}\n',
        "tool.py": b"#!/usr/bin/env python3\nprint('offline')\n",
    }

    scan_payloads(
        payloads,
        allowed_emails=frozenset(),
        exemptions=frozenset(),
        max_file_bytes=4096,
        max_total_bytes=8192,
    )


@pytest.mark.parametrize(
    ("path", "payload", "rule"),
    (
        ("config.json", b'{"api_key":"sk_live_abcdefghijklmnopqrstuvwxyz"}', "generic_secret"),
        ("note.txt", b"contact person@example.com", "email"),
        ("note.txt", b"source is /Users/private-user/project", "local_path"),
        ("kaggle.json", b'{"username":"judge","key":"secret"}', "sensitive_filename"),
        ("fixture.txt", b"card=4111 1111 1111 1111", "pii_pan"),
        ("fixture.txt", b"taxpayer=123-45-6789", "pii_ssn"),
        ("data.csv", b"name,email\nAlice,alice@example.com\n", "real_data_file"),
    ),
)
def test_scanner_fails_closed_on_sensitive_or_personal_content(
    path: str, payload: bytes, rule: str
) -> None:
    """Removing any named rule would admit a common release disclosure class."""
    with pytest.raises(ReleaseError, match=rule):
        scan_payloads(
            {path: payload},
            allowed_emails=frozenset(),
            exemptions=frozenset(),
            max_file_bytes=4096,
            max_total_bytes=8192,
        )


def test_scanner_allows_only_exact_email_and_finding_exemptions() -> None:
    """An exemption for one hash-bound model must not suppress the same rule elsewhere."""
    scan_payloads(
        {
            "NOTICE.md": b"Contact release@example.org\n",
            "model.json": b'{"number":"4111111111111111"}',
        },
        allowed_emails=frozenset({"release@example.org"}),
        exemptions=frozenset({("model.json", "pii_pan")}),
        max_file_bytes=4096,
        max_total_bytes=8192,
    )
    with pytest.raises(ReleaseError, match="pii_pan"):
        scan_payloads(
            {"other.json": b'{"number":"4111111111111111"}'},
            allowed_emails=frozenset({"release@example.org"}),
            exemptions=frozenset({("model.json", "pii_pan")}),
            max_file_bytes=4096,
            max_total_bytes=8192,
        )


def test_scanner_rejects_large_and_binary_payloads() -> None:
    """Large or opaque payloads could conceal datasets or credentials from text scanning."""
    with pytest.raises(ReleaseError, match="large_file"):
        scan_payloads(
            {"large.txt": b"x" * 4097},
            allowed_emails=frozenset(),
            exemptions=frozenset(),
            max_file_bytes=4096,
            max_total_bytes=8192,
        )
    with pytest.raises(ReleaseError, match="binary_file"):
        scan_payloads(
            {"opaque.bin": b"safe\x00hidden"},
            allowed_emails=frozenset(),
            exemptions=frozenset(),
            max_file_bytes=4096,
            max_total_bytes=8192,
        )
