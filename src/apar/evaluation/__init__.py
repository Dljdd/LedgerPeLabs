"""Verified development evaluation corpus contracts."""

from apar.evaluation.contracts import (
    CorpusManifest,
    CorpusProfile,
    EvaluationTruthRow,
    FrozenCorpus,
)
from apar.evaluation.corpus import CorpusVerificationError, assemble_verified_corpus
from apar.evaluation.regimes import (
    DerivedRegimeManifest,
    RegimeKind,
    RegimeSpec,
    derive_regime,
    frozen_corpus_digest,
)
from apar.evaluation.splits import (
    EntityCohort,
    EvaluationSplit,
    SplitConfig,
    make_evaluation_split,
    make_leave_one_family_out,
)

__all__ = [
    "CorpusManifest",
    "CorpusProfile",
    "CorpusVerificationError",
    "DerivedRegimeManifest",
    "EntityCohort",
    "EvaluationSplit",
    "EvaluationTruthRow",
    "FrozenCorpus",
    "RegimeKind",
    "RegimeSpec",
    "SplitConfig",
    "assemble_verified_corpus",
    "derive_regime",
    "frozen_corpus_digest",
    "make_evaluation_split",
    "make_leave_one_family_out",
]
