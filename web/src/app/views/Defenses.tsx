import { Icon } from "../Icon";
import { formatMetric, shortHash, titleCase } from "../format";
import type { ConsoleEvidence } from "../types";

const armDescriptions: Record<string, string> = {
  rules_only: "Deterministic policy signals and reasonable controls.",
  ensemble_no_graph: "Calibrated member voting without relationship features.",
  ensemble_with_graph: "Calibrated member voting with bound graph features.",
  full_sentinel: "Graph ensemble plus deterministic routing architecture.",
};

export function Defenses({ evidence }: { evidence: ConsoleEvidence }) {
  const metrics = ["f1", "recall", "precision", "false_decline_rate", "challenge_rate", "p95_latency_ms"];
  return (
    <div className="page">
      <header className="page-header split-header">
        <div><p className="eyebrow">05 · Defense posture</p><h1>Four arms. One honest champion.</h1><p>The portable demonstration is the accepted Stage 30 <strong>ensemble_with_graph</strong> arm—not the complete hybrid.</p></div>
        <div className="champion-badge"><span>DEMO CHAMPION</span><strong>ensemble_with_graph</strong><small>Portable · hash-bound · demo only</small></div>
      </header>

      <section className="architecture-lanes" aria-label="Defense architecture">
        {evidence.recovered.arms.map((arm, index) => {
          const isChampion = arm.arm === "ensemble_with_graph";
          const isFull = arm.arm === "full_sentinel";
          return (
            <article className={`${isChampion ? "is-champion" : ""} ${isFull ? "is-not-ready" : ""}`} key={arm.arm}>
              <div className="arm-top"><span>{String(index + 1).padStart(2, "0")}</span>{isChampion ? <span className="pill pill-accent">Live portable</span> : isFull ? <span className="pill pill-critical">Not ready</span> : <span className="pill">Diagnostic arm</span>}</div>
              <h2>{arm.arm}</h2><p>{armDescriptions[arm.arm]}</p>
              <div className="arm-flow" aria-hidden="true"><i /><b>→</b><i /><b>→</b><i className={index > 1 ? "active" : ""} />{isFull ? <><b>→</b><i className="failed" /></> : null}</div>
              <code title={arm.deterministic_result_sha256}>{shortHash(arm.deterministic_result_sha256, 8)}</code>
            </article>
          );
        })}
      </section>

      <section className="metrics-panel">
        <div className="panel-head metrics-head">
          <div><p className="eyebrow">Bound comparison evidence</p><h2>Recovered four-arm diagnostics</h2></div>
          <div className="qualifier"><Icon name="warning" /><span><strong>{evidence.recovered.qualifier}</strong><small>authoritative=false · accepted_capacity_evidence=false</small></span></div>
        </div>
        <div className="table-scroll">
          <table className="metrics-table">
            <thead><tr><th>Arm</th>{metrics.map((metric) => <th key={metric}>{titleCase(metric).replace("P95 Latency Ms", "P95 latency")}</th>)}</tr></thead>
            <tbody>{evidence.recovered.arms.map((arm) => <tr className={arm.arm === "ensemble_with_graph" ? "is-champion" : arm.arm === "full_sentinel" ? "is-failed" : ""} key={arm.arm}><th scope="row"><span>{arm.arm}</span>{arm.arm === "ensemble_with_graph" ? <small>portable champion</small> : arm.arm === "full_sentinel" ? <small>diagnostic only</small> : null}</th>{metrics.map((metric) => <td className="numeric" key={metric}>{formatMetric(arm.aggregate[metric] ?? 0, metric)}</td>)}</tr>)}</tbody>
          </table>
        </div>
        <p className="table-caption">Curated synthetic replay diagnostics recovered from machine-readable artifacts. These are not production estimates and the official evidence chain is incomplete at {evidence.recovered.first_missing_official_stage}.</p>
      </section>

      <section className="two-column defense-conclusion">
        <div className="panel failed-gates">
          <div className="panel-head"><div><p className="eyebrow">Full Sentinel diagnostic</p><h2>Promotion gates failed</h2></div><span className="pill pill-critical">{evidence.recovered.readiness.status.replace("_", " ")}</span></div>
          {evidence.recovered.failed_gates.map((gate) => <div className="gate-row" key={gate.metric}><span className="gate-x" aria-hidden="true">×</span><span><strong>{titleCase(gate.metric)}</strong><small>{typeof gate.point === "number" ? `Observed ${formatMetric(gate.point, gate.metric)} · ` : ""}Target {typeof gate.target === "number" ? formatMetric(gate.target, gate.metric) : String(gate.target)}</small></span><span>FAIL</span></div>)}
        </div>
        <div className="panel conclusion-card">
          <p className="eyebrow">Current conclusion</p>
          <h2>Graph ensemble usable.<br />Full-hybrid routing not ready.</h2>
          <p>The graph ensemble is the currently usable competition model. Deterministic full-hybrid routing requires policy refinement before it can be considered for promotion.</p>
          <div className="conclusion-status"><Icon name="check" /><span><strong>Champion claim is bounded</strong><small>Portable prediction and recovered comparison evidence remain separately labeled.</small></span></div>
        </div>
      </section>
    </div>
  );
}
