"""Independently verify a Sentinel v5 development result artifact (fail-closed)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_VALID_STATUSES = {"development_ready", "development_not_ready", "invalid_corpus"}
_FORBIDDEN_CLAIMS = {
    "winner", "production_ready", "competition_validated", "confirmatory_supported",
}


def _fail(message: str) -> int:
    print(f"INVALID: {message}", file=sys.stderr)
    return 1


def verify(document: dict) -> int:
    status = document.get("status", "")

    if status == "smoke":
        print("VERIFIED: smoke evidence; not production-ready")
        return 0

    if status not in _VALID_STATUSES:
        return _fail(f"status '{status}' is not recognized")

    serialized = str(document).lower()
    for claim in _FORBIDDEN_CLAIMS:
        if f'"{claim}"' in serialized:
            return _fail(f"forbidden claim detected: {claim}")

    fidelity_status = document.get("fidelity_status", "")
    if status == "development_ready" and fidelity_status != "pass":
        return _fail("ready verdict requires fidelity_status='pass'")

    arms = document.get("arms", {})
    if status != "invalid_corpus" and not isinstance(arms, dict):
        return _fail("arms must be a mapping")

    if status == "development_ready" and not arms:
        return _fail("ready verdict requires at least one evaluated arm")

    mandatory_fields = [
        "recall",
        "false_decline_rate",
        "challenge_rate",
        "captured_value_fraction",
        "support_total",
        "support_fraud",
        "support_legitimate",
    ]
    latency_fields = ["p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    for arm_name, arm_data in arms.items():
        if not isinstance(arm_data, dict):
            return _fail(f"arm '{arm_name}' must be a mapping")
        for field in mandatory_fields:
            if field not in arm_data or arm_data[field] is None:
                return _fail(f"arm '{arm_name}' missing mandatory metric: {field}")
            value = arm_data[field]
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return _fail(f"arm '{arm_name}' has non-finite {field}")
        if status == "development_ready":
            for field in latency_fields:
                if arm_data.get(field) is None:
                    return _fail(
                        f"ready verdict requires non-null {field} in arm '{arm_name}'"
                    )

    if document.get("sealed_evaluation_allowed") is True:
        return _fail("sealed_evaluation_allowed must be false")

    print(f"VERIFIED: {status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args()
    try:
        raw = args.result_path.read_bytes()
        if b"Infinity" in raw or b"NaN" in raw:
            return _fail("non-finite JSON literal (NaN/Infinity) detected")
        document = json.loads(raw)
    except (json.JSONDecodeError, OSError) as error:
        return _fail(f"cannot parse result file: {error}")
    if not isinstance(document, dict):
        return _fail("result must be a JSON object")
    return verify(document)


if __name__ == "__main__":
    raise SystemExit(main())
