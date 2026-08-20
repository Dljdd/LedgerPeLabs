# Task 1 report: sealed Defend v2 protocol

## RED/GREEN evidence

- RED: `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_protocol.py -q` could not run because this checkout has no `.venv`; the equivalent `python -m pytest ...` then failed at collection with `ModuleNotFoundError: apar.evaluation.v2_protocol`.
- GREEN: `python -m pytest tests/evaluation/test_defense_v2_protocol.py -q` passed: 5 tests.
- Frozen-v1 command was attempted as specified, but collection is blocked by the environment missing `pyarrow` (`ModuleNotFoundError: pyarrow`). No v1 source or fixture was changed.

## Files changed

- `config/defense/competition-v2-profile.json`: canonical production profile, exact 100,000 denominator and low/medium/high fraud counts, budgets, seed commitments, profile digest, and frozen v1 roots.
- `src/apar/evaluation/v2_protocol.py`: closed Pydantic contracts, canonical/digest-bound loader, fixture-only serialization guard, and hard-coded v1 root verification.
- `tests/evaluation/test_defense_v2_protocol.py`: profile exactness, canonical digest, fixture isolation, duplicate-strata rejection, and fail-closed v1 mismatch tests.

## Self-review

- Unknown fields are rejected by `ExternalContract`; strata names are a closed `Literal`; duplicate/missing strata fail validation.
- Production profiles enforce the exact denominator and fraud counts. Fixture profiles are explicit and cannot be serialized or loaded as preregistration input.
- JSON loading uses strict canonical bytes and SHA-256; no pickle or evaluator-hidden imports are used.
- Frozen v1 verification reads only the named hard-coded roots and fails closed on missing or changed bytes.

## Concerns

- The requested frozen-v1 test command cannot complete in this environment because `pyarrow` is not installed. The new protocol tests pass independently.

## Review round 1/5 fixes

- RED: three regression tests initially failed for mutable production day/stratum denominators, substituted/omitted v1 root mappings, and unequal family allocations.
- GREEN: `python -m pytest tests/evaluation/test_defense_v2_protocol.py -q` passed: 8 tests.
- Production validation now requires 28 synthetic days, 100,000 transactions in every stratum, exact equality with the hard-coded v1 root mapping, and equal four-family allocations.
- The frozen-v1 environment concern remains unchanged: `pyarrow` is unavailable for collection of the frozen suite.

## Review round 2/5 fixes

- RED: the new fixture regression initially exposed that non-divisible fraud counts reached Pydantic allocation validation with an indirect error.
- GREEN: `python -m pytest tests/evaluation/test_defense_v2_protocol.py -q` passed: 9 tests.
- `PrevalenceStratum.fixture()` now explicitly rejects fraud counts not divisible by four with a clear `ValueError`, while production validation remains unchanged.
