import { Icon } from "../Icon";
import { RouteLink } from "../Shell";
import { formatMoney } from "../format";
import type { ConsoleEvidence, RoutePath, VerifiedTrace } from "../types";

interface OverviewProps {
  evidence: ConsoleEvidence;
  navigate: (path: RoutePath) => void;
  trace: VerifiedTrace;
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
