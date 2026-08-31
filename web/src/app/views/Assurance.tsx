import { useState } from "react";

import { Icon } from "../Icon";
import { shortHash, titleCase } from "../format";
import type { ConsoleEvidence, TraceMode, VerifiedTrace } from "../types";

export function Assurance({ evidence, trace, traceMode }: { evidence: ConsoleEvidence; trace: VerifiedTrace; traceMode: TraceMode }) {
  const [selectedLineage, setSelectedLineage] = useState(0);
  const lineage: [string, string, string][] = [
    ["Portable bundle manifest", evidence.portable.bundle_manifest_sha256, "Verified input"],
    ["Stage 30 source checkpoint", evidence.portable.source_checkpoint_manifest_sha256, "Accepted source"],
    ["Arm specification", evidence.portable.arm_spec_sha256, "ensemble_with_graph"],
    [traceMode === "live_local_scorer" ? "Live local scorer trace" : "Committed fallback trace", trace.trace_sha256, "Replay verified"],
    ["Console evidence document", evidence.document_sha256, "Local projection"],
    ["Recovered diagnostic verification", evidence.recovered.verification_sha256, "Non-authoritative"],
  ];
  const selectedArtifact = lineage[selectedLineage] ?? lineage[0];
  return (
    <div className="page">
      <header className="page-header split-header">
        <div><p className="eyebrow">06 · Assurance</p><h1>Evidence before assertion.</h1><p>Every promotion claim is scoped by lineage, verification status, and an explicit human gate.</p></div>
        <div className="assurance-score"><span className="score-ring"><strong>6</strong><small>bound artifacts</small></span><span><strong>Local evidence chain</strong><small>Offline · deterministic reset</small></span></div>
      </header>

      <section className="assurance-grid">
        <article className="lineage-panel">
          <div className="panel-head"><div><p className="eyebrow">Evidence lineage</p><h2>Bound artifacts</h2></div><span className="pill pill-good">Inputs verified</span></div>
          <ol className="lineage-list" aria-label="Evidence lineage artifacts">
            {lineage.map(([label, hash, status], index) => <li className={selectedLineage === index ? "is-selected" : ""} key={label}><button aria-label={`Inspect ${label} lineage`} aria-pressed={selectedLineage === index} onClick={() => setSelectedLineage(index)} type="button"><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{label}</strong><code title={hash}>{shortHash(hash, 12)}</code></div><small>{status}</small></button></li>)}
          </ol>
          {selectedArtifact ? <div className="lineage-focus" aria-label="Selected lineage artifact" aria-live="polite" key={selectedArtifact[0]} role="status"><span>{selectedArtifact[0]}</span><code>{selectedArtifact[1]}</code><small>{selectedArtifact[2]}</small></div> : null}
        </article>
        <aside className="promotion-panel">
          <p className="eyebrow">Human promotion gate</p><div className="promotion-state"><Icon name="warning" size={25} /><span><strong>Promotion blocked</strong><small>{evidence.recovered.readiness.status.replace("_", " ")}</small></span></div>
          <p>No automated path can promote the diagnostic full_sentinel arm. Model-risk review and explicit human authorization remain required after policy refinement and a complete evidence chain.</p>
          <button aria-disabled="true" className="button button-disabled" disabled type="button">Human approval required</button>
          <div className="gate-meta"><span>Official chain</span><strong>{titleCase(evidence.recovered.official_chain_status)}</strong><span>First missing stage</span><code>{evidence.recovered.first_missing_official_stage}</code></div>
        </aside>
      </section>

      <section className="trust-panel" aria-label="Agentic integrity proof">
        <div className="trust-copy">
          <p className="eyebrow">Separate proof point · agentic integrity</p><h2>TrustVerifier rejects invalid authority.</h2>
          <p>Identity, mandate, scope, cart binding, and replay checks are genuine tested controls. This is a deterministic authorization proof—not evidence of graph-model performance.</p>
          <div className="source-chip"><span>TEST SOURCE</span><code title={evidence.trust_proof.test_evidence_sha256}>{evidence.trust_proof.test_evidence}</code></div>
        </div>
        <div className="trust-checks">
          {evidence.trust_proof.checks.map((check, index) => <article key={check.check}><span className="trust-step">{String(index + 1).padStart(2, "0")}</span><div><h3>{titleCase(check.check)}</h3><code>{check.evidence}</code></div><span className="pill pill-good"><Icon name="check" size={13} /> Tested</span></article>)}
        </div>
      </section>

      <section className="boundary-grid">
        <article className="seed-boundary">
          <p className="eyebrow">Seed execution record</p><h2>Exact boundary</h2>
          <p>The portable demo and recovered Kaggle metrics use seed 404 only; no Kaggle locked-successor/seed-2404 chain was run.</p>
          <p>An earlier local locked-development attempt started and irreversibly aborted. No candidate manifest, chunks, judge summary, or successful seed-2404 result was published, and retry is not permitted.</p>
        </article>
        <article className="limitations">
          <p className="eyebrow">Known limitations</p><h2>What this console does not claim</h2>
          <ul>
            <li><Icon name="warning" size={16} /> No production or real Mastercard data</li>
            <li><Icon name="warning" size={16} /> No official Stage 70 metrics claim</li>
            <li><Icon name="warning" size={16} /> Fixed-trace latency is environment-specific</li>
            <li><Icon name="warning" size={16} /> Twelve cases are curated synthetic replay checks</li>
            <li><Icon name="warning" size={16} /> Analyst-time benefit remains evidence pending</li>
          </ul>
        </article>
      </section>

      <section className="final-statement"><span className="statement-mark" aria-hidden="true">A</span><div><p className="eyebrow">Assurance statement</p><h2>Portable predictions are <code>ensemble_with_graph</code> only.</h2><p>Recovered four-arm diagnostics remain non-authoritative and do not convert the live demo into a complete hybrid claim.</p></div></section>
    </div>
  );
}
