"""Minimal evaluator-signed source fixture for the isolated Defend worker."""

from __future__ import annotations

import base64
import os
import time


def evaluate(
    inputs: object, config: dict[str, object]
) -> bytes:
    """Return sealed evaluator evidence while recording the isolated process identity."""
    if type(inputs).__name__ != "VerifiedEvaluationInputs" or type(config) is not dict:
        raise TypeError("fixture worker inputs are not exact")
    if set(config) != {"delay_seconds", "marker_path", "request_base64"}:
        raise ValueError("fixture worker configuration differs")
    marker_path = config["marker_path"]
    delay_seconds = config["delay_seconds"]
    request_base64 = config["request_base64"]
    if (
        type(marker_path) is not str
        or type(delay_seconds) is not float
        or type(request_base64) is not str
    ):
        raise TypeError("fixture worker configuration is invalid")
    descriptor = os.open(
        marker_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    finally:
        os.close(descriptor)
    if delay_seconds:
        time.sleep(delay_seconds)
    return base64.b64decode(request_base64, validate=True)
