"""Narrow, fail-closed secret, privacy, path, and payload scanner."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from scripts.submission.model import ReleaseError, ScanReport

_EMAIL = re.compile(
    rb"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    rb"(?![A-Za-z0-9.-])"
)
_LOCAL_PATH = re.compile(
    rb"(?:/Users/[A-Za-z0-9._-]+(?:/[^\s\"']*)?|/home/[A-Za-z0-9._-]+(?:/[^\s\"']*)?|"
    rb"/private/(?:tmp|var/folders)/[^\s\"']+|[A-Za-z]:\\Users\\[^\s\"']+)"
)
_SSN = re.compile(rb"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PAN_CANDIDATE = re.compile(rb"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PRIVATE_KEY = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
_KNOWN_TOKEN = re.compile(
    rb"(?:AKIA[0-9A-Z]{16}|gh[opsu]_[A-Za-z0-9]{24,}|xox[baprs]-[A-Za-z0-9-]{16,}|"
    rb"sk_(?:live|test)_[A-Za-z0-9]{16,})"
)
_GENERIC_SECRET = re.compile(
    rb"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|secret)"
    rb"\s*[\"']?\s*[:=]\s*[\"'][^\"'\r\n]{12,}[\"']"
)
_KAGGLE_CREDENTIAL = re.compile(
    rb"(?is)[\"']username[\"']\s*:\s*[\"'][^\"']+[\"'].{0,256}"
    rb"[\"']key[\"']\s*:\s*[\"'][^\"']+[\"']"
)
_REAL_DATA_SUFFIXES = frozenset({".csv", ".db", ".parquet", ".sqlite", ".sqlite3", ".xls", ".xlsx"})
_SENSITIVE_NAMES = frozenset(
    {".env", "credentials", "credentials.json", "kaggle.json", "secrets.json"}
)


def _luhn(candidate: bytes) -> bool:
    digits = [int(character) for character in candidate.decode("ascii") if character.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = digit
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _exempt(path: str, rule: str, exemptions: frozenset[tuple[str, str]]) -> bool:
    return (path, rule) in exemptions


def scan_payloads(
    payloads: dict[str, bytes],
    *,
    allowed_emails: frozenset[str],
    exemptions: frozenset[tuple[str, str]],
    max_file_bytes: int,
    max_total_bytes: int,
) -> ScanReport:
    """Reject any non-exempt sensitive finding across the exact archive payload map."""
    findings: list[str] = []
    applied_exemptions = 0
    total_bytes = sum(len(payload) for payload in payloads.values())
    if total_bytes > max_total_bytes:
        findings.append(f"total_size:{total_bytes}>{max_total_bytes}")
    for path, payload in sorted(payloads.items()):
        rules: set[str] = set()
        pure_path = PurePosixPath(path)
        if len(payload) > max_file_bytes:
            rules.add("large_file")
        if pure_path.name.lower() in _SENSITIVE_NAMES or pure_path.suffix.lower() == ".ipynb":
            rules.add("sensitive_filename")
        if pure_path.suffix.lower() in _REAL_DATA_SUFFIXES:
            rules.add("real_data_file")
        if b"\x00" in payload:
            rules.add("binary_file")
        else:
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError:
                rules.add("binary_file")
        if _LOCAL_PATH.search(payload):
            rules.add("local_path")
        if _PRIVATE_KEY.search(payload) or _KNOWN_TOKEN.search(payload):
            rules.add("known_secret")
        if _GENERIC_SECRET.search(payload):
            rules.add("generic_secret")
        if _KAGGLE_CREDENTIAL.search(payload):
            rules.add("kaggle_credentials")
        if _SSN.search(payload):
            rules.add("pii_ssn")
        if any(_luhn(match.group()) for match in _PAN_CANDIDATE.finditer(payload)):
            rules.add("pii_pan")
        for match in _EMAIL.finditer(payload):
            email = match.group().decode("ascii")
            if email not in allowed_emails:
                rules.add("email")
        for rule in sorted(rules):
            if _exempt(path, rule, exemptions):
                applied_exemptions += 1
            else:
                findings.append(f"{rule}:{path}")
    if findings:
        raise ReleaseError("submission scan failed: " + ", ".join(findings))
    return ScanReport(
        exemption_count=applied_exemptions,
        files_scanned=len(payloads),
        total_bytes=total_bytes,
    )
