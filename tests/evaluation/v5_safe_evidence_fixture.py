"""Process-local cache for the expensive real seed-404 evidence fixture."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from scripts.build_defense_v5_safe_evidence import build_safe_evidence

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def safe_v5_evidence_bytes() -> bytes:
    """Build the real safe evidence once per Python test process."""
    retained = os.environ.get("APAR_V5_SAFE_EVIDENCE_FIXTURE")
    if retained:
        return Path(retained).read_bytes()
    return build_safe_evidence(ROOT)
