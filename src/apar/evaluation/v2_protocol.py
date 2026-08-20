"""Evaluator-only compatibility import for the isolated public v2 protocol.

Defender code must import :mod:`apar.v2_protocol`; the preexecution boundary
rejects this legacy evaluator package path because importing it executes the
historical evaluator package initializer.
"""

from apar.v2_protocol import (
    FIXTURE_CAMPAIGN_INJECTION_SEED,
    FIXTURE_OPERATING_POPULATION_SEED,
    OperatingPopulationProfile,
    PrevalenceStratum,
    SeedCommitment,
    V2Budget,
    V2Protocol,
    V2ProtocolError,
    load_v2_protocol,
    verify_v1_roots,
)

__all__ = [
    "FIXTURE_CAMPAIGN_INJECTION_SEED",
    "FIXTURE_OPERATING_POPULATION_SEED",
    "OperatingPopulationProfile",
    "PrevalenceStratum",
    "SeedCommitment",
    "V2Budget",
    "V2Protocol",
    "V2ProtocolError",
    "load_v2_protocol",
    "verify_v1_roots",
]
