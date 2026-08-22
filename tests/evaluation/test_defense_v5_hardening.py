"""One-round adaptive hardening tests for Defend v5."""

from __future__ import annotations

from apar.evaluation.v5_hardening import V5HardeningResult, run_v5_adaptive_hardening
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
PROTOCOL = load_v5_development_protocol(ROOT / "config/defense/defense-v5-development.json")


class TestHardening:
    def test_returns_result(self) -> None:
        corpus = build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE)
        result = run_v5_adaptive_hardening(protocol=PROTOCOL, corpus=corpus)
        assert isinstance(result, V5HardeningResult)
        assert result.status in ("completed", "no_delta", "not_ready")

    def test_holdout_separate_from_hardening(self) -> None:
        result = run_v5_adaptive_hardening(
            protocol=PROTOCOL,
            corpus=build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE),
        )
        if result.hardening_campaign_ids and result.holdout_campaign_ids:
            assert not (result.hardening_campaign_ids & result.holdout_campaign_ids)

    def test_singleton_search_yields_no_claim(self) -> None:
        result = run_v5_adaptive_hardening(
            protocol=PROTOCOL,
            corpus=build_v5_corpus(PROTOCOL, profile=V5Profile.SMOKE),
        )
        hardening_count = len(result.hardening_campaign_ids or frozenset())
        assert not (result.adaptive_advantage_claimed and hardening_count <= 1)
