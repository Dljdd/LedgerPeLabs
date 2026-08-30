import { Icon } from "../Icon";
import { RouteLink } from "../Shell";
import { formatMoney, shortHash, titleCase } from "../format";
import type { ConsoleEvidence, RoutePath, VerifiedTrace } from "../types";

export function Scenario({ evidence, navigate, trace }: { evidence: ConsoleEvidence; navigate: (path: RoutePath) => void; trace: VerifiedTrace }) {
  const counts = trace.traces.reduce<Record<string, number>>((total, record) => {
    const rail = record.presentation_ground_truth.rail;
    total[rail] = (total[rail] ?? 0) + 1;
    return total;
  }, {});
  const config = evidence.threat.default_config;

  return (
    <div className="page">
      <header className="page-header split-header">
        <div><p className="eyebrow">02 · Scenario control</p><h1>Bounded campaign replay</h1></div>
        <div className="header-note"><span className="pill pill-good">Synthetic only</span><p>No real people, accounts, credentials, or payment instructions.</p></div>
      </header>

      <section className="scenario-hero">
        <div className="scenario-intro">
          <span className="overline">Selected threat / {evidence.threat.threat_id}</span>
          <h2>{evidence.threat.title}</h2>
          <p>A controlled network-view simulation of personalized persuasion, authorized transfer, mule ingress, layering, and cash-out.</p>
          <RouteLink className="button button-primary" navigate={navigate} path="/replay">Start verified replay <Icon name="play" /></RouteLink>
        </div>
        <div className="motif-card">
          <span className="eyebrow">Campaign motif</span>
          <div className="motif-code">{evidence.scenario_context.motif_signature}</div>
          <div className="motif-visual" aria-label="fan in to mule, layer, fan out, cash out">
            <span className="node-stack"><i /><i /><i /></span><b>→</b><span className="node-risk" /><b>→</b><span className="node-risk small" /><b>→</b><span className="node-stack reverse"><i /><i /></span>
          </div>
          <dl className="mini-stats">
            <div><dt>Value</dt><dd>{formatMoney(evidence.scenario_context.value_total)}</dd></div>
            <div><dt>Ledger</dt><dd>{evidence.scenario_context.ledger_conserved ? "Conserved" : "Failed"}</dd></div>
          </dl>
        </div>
      </section>

      <section className="two-column">
        <div className="panel">
          <div className="panel-head"><div><p className="eyebrow">Campaign constraints</p><h2>Deterministic inputs</h2></div><span className="pill">Read only</span></div>
          <dl className="definition-table">
            <div><dt>Generator seed</dt><dd>{config.seed}</dd></div>
            <div><dt>Duration</dt><dd>{config.duration_hours} hours</dd></div>
            <div><dt>Query budget</dt><dd>{config.query_budget}</dd></div>
            <div><dt>Attacker mode</dt><dd>{titleCase(config.attacker_mode)}</dd></div>
            <div><dt>Viewpoint</dt><dd>{titleCase(config.viewpoint)}</dd></div>
            <div><dt>Event ordering</dt><dd>{titleCase(config.event_ordering)}</dd></div>
          </dl>
          <div className="hash-line"><span>Schedule SHA-256</span><code title={evidence.scenario_context.schedule_sha256}>{shortHash(evidence.scenario_context.schedule_sha256)}</code></div>
        </div>

        <div className="panel">
          <div className="panel-head"><div><p className="eyebrow">Curated model context</p><h2>12-case portable replay</h2></div><span className="pill pill-amber">Demo only</span></div>
          <div className="rail-distribution">
            {Object.entries(counts).map(([rail, count]) => (
              <div key={rail}><span><b>{titleCase(rail)}</b><small>{count} events</small></span><meter max={trace.traces.length} value={count}>{count} of {trace.traces.length}</meter></div>
            ))}
          </div>
          <p className="panel-footnote">These curated cases exercise card, A2A, and agentic contexts. They are replay checks—not production prevalence or performance estimates.</p>
        </div>
      </section>

      <section className="stage-row" aria-label="Campaign stages">
        {config.campaign_stages.map((stage, index) => (
          <article key={stage.stage_id}>
            <span>{String(index + 1).padStart(2, "0")}</span><div><h3>{titleCase(stage.stage_id)}</h3><p>{stage.description}</p></div>
          </article>
        ))}
      </section>

      <section className="boundary-banner">
        <Icon name="warning" />
        <div><strong>Safety boundary is enforced.</strong><p>Execution is limited to the committed portable scorer and hash-bound synthetic inputs. No training or locked experiment is available from this console.</p></div>
      </section>
    </div>
  );
}
