import { Icon } from "../Icon";
import { RouteLink } from "../Shell";
import { formatMoney, formatPercent } from "../format";
import type { ConsoleEvidence, RoutePath, VerifiedTrace } from "../types";

interface OverviewProps {
  evidence: ConsoleEvidence;
  navigate: (path: RoutePath) => void;
  trace: VerifiedTrace;
}

function traceTone(action: string): string {
  if (action === "decline_hold") return "critical";
  if (action === "review_hold" || action === "challenge") return "amber";
  return "good";
}

function TraceFootprint({ evidence, trace }: { evidence: ConsoleEvidence; trace: VerifiedTrace }) {
  const challenge = evidence.portable.thresholds.model_challenge;
  const review = evidence.portable.thresholds.model_review;
  const decline = evidence.portable.thresholds.model_decline;
  const thresholdsBound = [challenge, review, decline].every((value) => Number.isFinite(value));

  if (!thresholdsBound || challenge === undefined || review === undefined || decline === undefined || trace.traces.length === 0) {
    return <div className="trace-footprint-unavailable"><strong>Trace visualization unavailable</strong><span>Bound thresholds or replay events are missing.</span></div>;
  }

  const width = 360;
  const plotLeft = 8;
  const plotRight = 352;
  const plotTop = 8;
  const plotBottom = 58;
  const xFor = (index: number) => trace.traces.length === 1 ? width / 2 : plotLeft + (index / (trace.traces.length - 1)) * (plotRight - plotLeft);
  const yFor = (probability: number) => plotBottom - probability * (plotBottom - plotTop);
  const points = trace.traces.map((record, index) => `${xFor(index)},${yFor(record.calibrated_probability)}`).join(" ");
  const counts = trace.traces.reduce<Record<string, number>>((summary, record) => {
    summary[record.final_action] = (summary[record.final_action] ?? 0) + 1;
    return summary;
  }, {});
  const summary = `${trace.traces.length} ordered calibrated decisions: ${counts.decline_hold ?? 0} decline hold, ${counts.review_hold ?? 0} review hold, ${counts.challenge ?? 0} challenge, and ${counts.approve ?? 0} approve.`;

  return (
    <div className="trace-footprint" aria-label={summary} role="img">
      <div className="trace-footprint-head"><span>Curated decision footprint</span><span>{trace.traces.length} events</span></div>
      <svg aria-hidden="true" viewBox={`0 0 ${width} 68`}>
        <line className="trace-threshold-line is-decline" vectorEffect="non-scaling-stroke" x1={plotLeft} x2={plotRight} y1={yFor(decline)} y2={yFor(decline)} />
        <line className="trace-threshold-line is-review" vectorEffect="non-scaling-stroke" x1={plotLeft} x2={plotRight} y1={yFor(review)} y2={yFor(review)} />
        <line className="trace-threshold-line is-challenge" vectorEffect="non-scaling-stroke" x1={plotLeft} x2={plotRight} y1={yFor(challenge)} y2={yFor(challenge)} />
        <polyline className="trace-risk-line" points={points} vectorEffect="non-scaling-stroke" />
        {trace.traces.map((record, index) => (
          <circle className={`trace-risk-point is-${traceTone(record.final_action)}`} cx={xFor(index)} cy={yFor(record.calibrated_probability)} key={record.event_id} r="3.5" />
        ))}
      </svg>
      <div className="trace-threshold-labels">
        <span>Challenge {formatPercent(challenge, 1)}</span>
        <span>Review {formatPercent(review, 1)}</span>
        <span>Decline {formatPercent(decline, 1)}</span>
      </div>
    </div>
  );
}

export function Overview({ evidence, navigate, trace }: OverviewProps) {
  const rails = [...new Set(trace.traces.map((record) => record.presentation_ground_truth.rail))];
  return (
    <div className="page page-overview">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Threat posture · selected golden path</p>
          <h1>Adaptive payment<br />assurance.</h1>
          <p className="hero-lede">
            Detect the campaign—not only the transaction—and intervene before authorized funds become irrecoverable.
          </p>
          <div className="hero-actions">
            <RouteLink className="button button-primary" navigate={navigate} path="/replay">
              Run verified replay <Icon name="arrow" />
            </RouteLink>
            <RouteLink className="text-link" navigate={navigate} path="/scenario">
              Inspect scenario controls <Icon name="chevron" size={16} />
            </RouteLink>
          </div>
        </div>
        <div className="hero-signal" aria-label="Selected threat summary">
          <div className="signal-header">
            <span className="signal-kicker">THREAT / APP–MULE</span>
            <span className="pill pill-critical">High-risk path</span>
          </div>
          <p className="signal-title">{evidence.threat.title}</p>
          <TraceFootprint evidence={evidence} trace={trace} />
          <div className="signal-flow" aria-label="Campaign progression">
            <span>Persuasion</span><i /><span>Authorized transfer</span><i /><span>Mule dispersion</span>
          </div>
          <dl className="signal-stats">
            <div><dt>Attempted value</dt><dd>{formatMoney(evidence.scenario_context.value_total)}</dd></div>
            <div><dt>Campaign payments</dt><dd>{evidence.scenario_context.payment_count}</dd></div>
            <div><dt>Evidence class</dt><dd>Synthetic only</dd></div>
          </dl>
        </div>
      </section>

      <section className="metric-ribbon" aria-label="Bound evidence summary">
        <div><span className="metric-value">{trace.traces.length}</span><span>curated replay events</span></div>
        <div><span className="metric-value">{rails.length}</span><span>payment contexts</span></div>
        <div><span className="metric-value">{evidence.scenario_context.graph.nodes.length}</span><span>linked entities</span></div>
        <div><span className="metric-value">{evidence.threat.confidence.toFixed(2)}</span><span>threat-source confidence</span></div>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><p className="eyebrow">Capability delta</p><h2>What changes with generative fraud</h2></div>
          <p>Bounded changes to personalization and iteration speed; no claim of autonomous settlement access.</p>
        </div>
        <div className="delta-grid">
          <article className="delta-card baseline">
            <span className="card-index">01 / BASELINE</span>
            <h3>Conventional APP scam</h3>
            <p>Static scripts, broad targeting, manual iteration, and authorized push payment initiation by the victim.</p>
          </article>
          <div className="delta-arrow" aria-hidden="true"><Icon name="arrow" size={22} /></div>
          <article className="delta-card adaptive">
            <span className="card-index">02 / BOUNDED DELTA</span>
            <h3>Adaptive persuasion</h3>
            <ul className="check-list">
              <li><Icon name="check" size={16} /> Synthetic personalization</li>
              <li><Icon name="check" size={16} /> Faster message iteration</li>
              <li><Icon name="check" size={16} /> Campaign-scale feedback</li>
            </ul>
          </article>
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading compact">
          <div><p className="eyebrow">Rail context</p><h2>One assurance layer, three contexts</h2></div>
        </div>
        <div className="rail-grid">
          {[
            { name: "Card", copy: "CNP testing and synthetic refund context", key: "card" },
            { name: "A2A", copy: "Selected APP scam and mule golden path", key: "a2a" },
            { name: "Agentic", copy: "Intent and mandate-integrity context", key: "agentic" },
          ].map(({ name, copy, key }) => (
            <article className={key === "a2a" ? "rail-card selected" : "rail-card"} key={key}>
              <span className="rail-glyph" aria-hidden="true">{name.slice(0, 1)}</span>
              <div><h3>{name}</h3><p>{copy}</p></div>
              <span className={key === "a2a" ? "pill pill-accent" : "pill"}>{key === "a2a" ? "Golden path" : "Curated context"}</span>
            </article>
          ))}
        </div>
      </section>

      <section className="claim-boundary">
        <div><span className="eyebrow">Evidence boundary</span><h2>Facts remain separate from APAR inference.</h2></div>
        <div className="claim-list">
          {evidence.threat.evidence.map((item) => (
            <div key={item.evidence_id}>
              <span className={item.is_project_inference ? "pill pill-amber" : "pill pill-good"}>
                {item.is_project_inference ? "APAR inference" : `Grade ${item.quality_grade} source`}
              </span>
              <p>{item.claim}</p><small>{item.publisher}</small>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
