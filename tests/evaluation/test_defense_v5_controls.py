"""Legacy descriptive control API removal."""

from __future__ import annotations

from apar.evaluation import v5_evaluation
from apar.evaluation.v5_controls import execute_v5_controls


def test_descriptive_control_entry_point_is_retired() -> None:
    """Reintroducing a string-only control result must fail this boundary test."""
    assert not hasattr(v5_evaluation, "run_v5_controls")
    assert callable(execute_v5_controls)
