import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";

import { formatMoney, formatPercent, shortHash, titleCase } from "../format";
import type { ConsoleEvidence, GraphEdge, TraceMode, VerifiedTrace } from "../types";
import { useReducedMotion } from "../useReducedMotion";

function actionTone(action: string): string {
  if (action === "decline_hold") return "critical";
  if (action === "review_hold" || action === "challenge") return "amber";
  return "good";
}

function DecisionRuler({ action, probability, thresholds }: { action: string; probability: number; thresholds: Record<string, number> }) {
  const challenge = thresholds.model_challenge;
  const review = thresholds.model_review;
  const decline = thresholds.model_decline;
  if (typeof challenge !== "number"
    || typeof review !== "number"
    || typeof decline !== "number"
    || !Number.isFinite(challenge)
    || !Number.isFinite(review)
    || !Number.isFinite(decline)
    || challenge < 0
    || challenge > review
    || review > decline
    || decline > 1) {
    return <div className="decision-ruler-unavailable"><strong>{formatPercent(probability, 1)}</strong><span>Threshold evidence unavailable</span></div>;
  }

  const style = {
    "--risk": Math.max(0, Math.min(1, probability)),
    "--challenge-threshold": `${challenge * 100}%`,
    "--review-threshold": `${review * 100}%`,
    "--decline-threshold": `${decline * 100}%`,
    gridTemplateColumns: `${challenge}fr ${review - challenge}fr ${decline - review}fr`,
  } as CSSProperties;
  const label = `Calibrated risk ${formatPercent(probability, 1)} with final action ${titleCase(action)}. Bound action thresholds: challenge ${formatPercent(challenge, 1)}, review ${formatPercent(review, 1)}, decline ${formatPercent(decline, 1)}.`;

  return (
    <div className={`decision-ruler is-${actionTone(action)}`} aria-label={label} role="img">
      <div className="risk-readout" key={`${probability}-${action}`}>
        <strong>{formatPercent(probability, 1)}</strong>
        <span><small>Calibrated probability</small><b>{titleCase(action)}</b></span>
      </div>
      <div className="decision-scale" style={style} aria-hidden="true">
        <span className="decision-band band-approve">Approve</span>
        <span className="decision-band band-challenge">Challenge</span>
        <span className="decision-band band-review">Review</span>
        <i className="decision-observed" />
        <i className="threshold-marker marker-challenge" />
        <i className="threshold-marker marker-review" />
        <i className="threshold-marker marker-decline" />
      </div>
      <div className="threshold-labels">
        <span>Challenge <b>{formatPercent(challenge, 1)}</b></span>
        <span>Review <b>{formatPercent(review, 1)}</b></span>
        <span>Decline <b>{formatPercent(decline, 1)}</b></span>
      </div>
    </div>
  );
}

function CampaignPlaybackGraph({ evidence, selectedEdge }: { evidence: ConsoleEvidence; selectedEdge: number }) {
  const { edges, nodes } = evidence.scenario_context.graph;
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const activeEdge = edges[selectedEdge] ?? edges[0];
  const activeSource = activeEdge ? byId.get(activeEdge.source) : undefined;
  const activeTarget = activeEdge ? byId.get(activeEdge.target) : undefined;
  const visitedNodeIds = new Set(
    edges
      .slice(0, selectedEdge + 1)
      .flatMap((edge) => [edge.source, edge.target]),
  );
  return (
    <svg className="replay-campaign-graph" aria-label={`${nodes.length} genuine scenario entities and ${edges.length} ordered payment edges; edges are revealed through payment ${selectedEdge + 1}`} role="img" viewBox="20 20 650 670">
      <defs>
        <marker id="replay-payment-arrow" markerHeight="6" markerWidth="6" orient="auto" refX="5" refY="3"><path d="M0,0 L0,6 L6,3 z" /></marker>
      </defs>
      <g className="replay-graph-edges">
        {edges.map((edge, index) => {
          const source = byId.get(edge.source);
          const target = byId.get(edge.target);
          if (!source || !target) return null;
          const distance = Math.hypot(target.x - source.x, target.y - source.y);
          const targetInset = distance === 0 ? 0 : 22 / distance;
          return (
            <line
              className={`${index === selectedEdge ? "is-selected" : ""} ${index <= selectedEdge ? "is-revealed" : "is-concealed"}`}
              key={edge.payment_id}
              markerEnd="url(#replay-payment-arrow)"
              x1={source.x}
              x2={target.x - (target.x - source.x) * targetInset}
              y1={source.y}
              y2={target.y - (target.y - source.y) * targetInset}
            >
              <title>{`${source.label} to ${target.label}, ${formatMoney(edge.amount, edge.currency)}, ${titleCase(edge.stage)}`}</title>
            </line>
          );
        })}
      </g>
      {activeEdge && activeSource && activeTarget ? (
        <circle
          aria-hidden="true"
          className="replay-value-packet"
          key={activeEdge.payment_id}
          r="6"
          style={{
            "--packet-source-x": `${activeSource.x}px`,
            "--packet-source-y": `${activeSource.y}px`,
            "--packet-target-x": `${activeTarget.x}px`,
            "--packet-target-y": `${activeTarget.y}px`,
          } as CSSProperties}
        />
      ) : null}
      <g className="replay-graph-nodes">
        {nodes.map((node) => {
          const nodeClasses = [
            visitedNodeIds.has(node.id) ? "is-visited" : "",
            node.id === activeEdge?.source ? "is-active-source" : "",
            node.id === activeEdge?.target ? "is-active-target" : "",
          ].filter(Boolean).join(" ");
          return (
            <g className={nodeClasses} key={node.id} transform={`translate(${node.x} ${node.y})`}>
              <circle className="node-halo" r="24" />
              <circle className="node-body" r="17" />
              <circle className="node-core" r="4" />
              <text x="27" y="-2">{node.label}</text>
              <text className="node-meta" x="27" y="12">{node.role.toUpperCase()} / {node.country}</text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

function scenarioParties(evidence: ConsoleEvidence, edge: GraphEdge) {
  const nodes = evidence.scenario_context.graph.nodes;
  return {
    source: nodes.find((node) => node.id === edge.source)?.label ?? shortHash(edge.source, 8),
    target: nodes.find((node) => node.id === edge.target)?.label ?? shortHash(edge.target, 8),
  };
}

function synchronizedTraceIndex(campaignIndex: number, campaignLength: number, traceLength: number) {
  if (campaignLength <= 1 || traceLength <= 1) return 0;
  // Presentation progress only; the repository does not bind scenario edges to trace rows.
  const progress = campaignIndex / (campaignLength - 1);
  return Math.round(progress * (traceLength - 1));
}

export function Replay({ evidence, trace, traceMode }: { evidence: ConsoleEvidence; trace: VerifiedTrace; traceMode: TraceMode }) {
  const [current, setCurrent] = useState(0);
  const [campaignStep, setCampaignStep] = useState(0);
  const [campaignPlaying, setCampaignPlaying] = useState(false);
  const record = trace.traces[current] ?? trace.traces[0];
  const inputRecord = evidence.portable.records[current] ?? evidence.portable.records[0];
  const campaignEdges = evidence.scenario_context.graph.edges;
  const campaignEdge = campaignEdges[campaignStep] ?? campaignEdges[0];
  const total = trace.traces.length;
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    setCurrent(synchronizedTraceIndex(campaignStep, campaignEdges.length, total));
  }, [campaignEdges.length, campaignStep, total]);

  useEffect(() => {
    if (!campaignPlaying || reducedMotion) return;
    const timer = window.setInterval(() => {
      setCampaignStep((index) => {
        if (index >= campaignEdges.length - 1) {
          setCampaignPlaying(false);
          return index;
        }
        return index + 1;
      });
    }, 900);
    return () => window.clearInterval(timer);
  }, [campaignEdges.length, campaignPlaying, reducedMotion]);

  useEffect(() => {
    if (reducedMotion) setCampaignPlaying(false);
  }, [reducedMotion]);

  const actionCounts = useMemo(() => trace.traces.reduce<Record<string, number>>((counts, item) => {
    counts[item.final_action] = (counts[item.final_action] ?? 0) + 1;
    return counts;
  }, {}), [trace.traces]);

  if (!record || !inputRecord || !campaignEdge) {
    return <div className="empty-state"><h1>No verified replay events</h1><p>The console stopped safely because the bound replay evidence is empty.</p></div>;
  }

  const parties = scenarioParties(evidence, campaignEdge);
  const resetStreams = () => {
    setCampaignPlaying(false);
    setCampaignStep(0);
    setCurrent(0);
  };
  const toggleCampaign = () => {
    if (reducedMotion) {
      setCampaignPlaying(false);
      const nextCampaignStep = campaignStep >= campaignEdges.length - 1 ? 0 : campaignStep + 1;
      setCampaignStep(nextCampaignStep);
      setCurrent(synchronizedTraceIndex(nextCampaignStep, campaignEdges.length, total));
      return;
    }
    if (!campaignPlaying) {
      const nextCampaignStep = campaignStep === campaignEdges.length - 1 ? 0 : campaignStep;
      setCampaignStep(nextCampaignStep);
      setCurrent(synchronizedTraceIndex(nextCampaignStep, campaignEdges.length, total));
    }
    setCampaignPlaying((value) => !value);
  };
  const campaignControlLabel = reducedMotion
    ? campaignStep === campaignEdges.length - 1 ? "Restart both streams" : "Step both streams"
    : campaignPlaying ? "Pause both streams" : campaignStep === campaignEdges.length - 1 ? "Replay both streams" : "Play both streams";

  return (
    <div className="page replay-page">
      <header className="page-header split-header replay-header">
        <div><p className="eyebrow">03 · Detection narrative</p><h1>Verified decision replay</h1><p>Advance both evidence streams through one presentation control, then inspect each independently.</p></div>
        <div className="live-arm"><span className="status-dot" aria-hidden="true" /><span><small>{traceMode === "live_local_scorer" ? "LIVE LOCAL SCORER" : "HASH-BOUND VERIFIED FALLBACK"}</small><strong>ensemble_with_graph</strong></span></div>
      </header>

      <section className="replay-narrative" aria-label="Campaign and detection narrative">
        <div className="campaign-playback">
          <header className="campaign-playback-head">
            <div><span>PRESENTATION PLAYBACK / CAMPAIGN {String(campaignStep + 1).padStart(2, "0")}/{String(campaignEdges.length).padStart(2, "0")} · PORTABLE {String(current + 1).padStart(2, "0")}/{String(total).padStart(2, "0")}</span><h2>Advance campaign and model evidence together</h2></div>
            <div className="campaign-transport">
              <button aria-pressed={reducedMotion ? undefined : campaignPlaying} className="campaign-play-button" onClick={toggleCampaign} type="button"><i aria-hidden="true" />{campaignControlLabel}</button>
              <button aria-label="Reset both streams" className="campaign-reset-button" onClick={resetStreams} type="button"><span aria-hidden="true">↺</span>Reset both</button>
            </div>
          </header>
          <CampaignPlaybackGraph evidence={evidence} selectedEdge={campaignStep} />
          <div className="campaign-progress"><i style={{ transform: `scaleX(${(campaignStep + 1) / campaignEdges.length})` }} /><span>{formatMoney(campaignEdge.cumulative_attempted_value, campaignEdge.currency)} cumulative attempted</span></div>
          <div className="campaign-tape-scroll">
            <ol className="campaign-tape" aria-label="Ordered campaign payments">
              {campaignEdges.map((edge, index) => (
                <li key={edge.payment_id}>
                  <button aria-current={campaignStep === index ? "step" : undefined} onClick={() => { setCampaignPlaying(false); setCampaignStep(index); setCurrent(synchronizedTraceIndex(index, campaignEdges.length, total)); }} type="button">
                    <span>{String(index + 1).padStart(2, "0")}</span><i aria-hidden="true" /><b>{formatMoney(edge.amount, edge.currency)}</b><small>{titleCase(edge.stage)}</small>
                  </button>
                </li>
              ))}
            </ol>
          </div>
          <footer className="campaign-evidence-foot">
            <span>{evidence.scenario_context.graph.nodes.length} entities</span>
            <span>{campaignEdges.length} genuine graph edges</span>
            <span>Synthetic only</span>
            <code title={evidence.scenario_context.graph.graph_sha256}>graph {shortHash(evidence.scenario_context.graph.graph_sha256, 8)}</code>
          </footer>
        </div>

        <aside className="replay-inspector">
          <section className="scenario-payment" aria-live="polite">
            <div className="scenario-payment-state" key={campaignEdge.payment_id}>
              <header><span>SCENARIO PAYMENT {String(campaignStep + 1).padStart(2, "0")}</span><b>Genuine graph edge</b></header>
              <h2>{titleCase(campaignEdge.stage)}</h2>
              <strong>{formatMoney(campaignEdge.amount, campaignEdge.currency)}</strong>
              <div className="scenario-route"><b>{parties.source}</b><i aria-hidden="true">→</i><b>{parties.target}</b></div>
              <dl><div><dt>Cumulative attempted</dt><dd>{formatMoney(campaignEdge.cumulative_attempted_value, campaignEdge.currency)}</dd></div><div><dt>Event time</dt><dd>{campaignEdge.event_time.slice(11, 19)} UTC</dd></div></dl>
            </div>
          </section>

          <section className="portable-response" aria-label="Model evidence" role="region">
            <div className="portable-response-state" key={record.event_id}>
              <header><span>PORTABLE REPLAY / EVENT {String(current + 1).padStart(2, "0")}</span><code>{record.arm}</code></header>
              <DecisionRuler action={record.final_action} probability={record.calibrated_probability} thresholds={evidence.portable.thresholds} />
              <dl className="portable-facts">
                <div><dt>Event</dt><dd>{shortHash(record.event_id, 12)}</dd></div>
                <div><dt>{traceMode === "live_local_scorer" ? "Local scorer latency" : "Fixed-trace latency"}</dt><dd>{record.latency_ms.toFixed(3)} ms</dd></div>
                <div><dt>Reason</dt><dd>{record.reason_codes.join(", ")}</dd></div>
                <div><dt>Feature vector</dt><dd>{inputRecord.model_input.features.length} bound features</dd></div>
              </dl>
            </div>
            <div className="portable-status">
              <span aria-live="polite">Event {String(current + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}</span>
              <small>Manual trace inspection below</small>
            </div>
          </section>

          <div className="trace-selector" aria-label="Select independent portable trace event" role="group">
            {trace.traces.map((item, index) => (
              <button aria-label={`Portable event ${index + 1}, ${formatPercent(item.calibrated_probability, 1)}, ${titleCase(item.final_action)}`} aria-pressed={current === index} className={`is-${actionTone(item.final_action)}`} key={item.event_id} onClick={() => setCurrent(index)} type="button"><span>{String(index + 1).padStart(2, "0")}</span><i aria-hidden="true" /><b>{formatPercent(item.calibrated_probability, 1)}</b></button>
            ))}
          </div>

          <p className="stream-boundary"><b>Scenario graph</b><span>with</span><b>portable replay</b><small>Shared presentation control · independent evidence streams. No payment-to-trace record mapping asserted.</small></p>
        </aside>
      </section>

      <section className="replay-postscript">
        <div className="replay-explanation">
          <div><p className="eyebrow">Intervention explanation</p><h2>From calibrated probability to action.</h2><p>The threshold ruler is bound to the selected portable event. It visualizes the model response only; campaign edges and post-event truth remain separate.</p></div>
          <div className="reason-box"><span>Reason evidence</span>{record.reason_codes.map((reason) => <code key={reason}>{reason}</code>)}</div>
          <p className="technical-note">{traceMode === "live_local_scorer" ? "Latency was observed during this local scoring run and is not a production latency estimate." : "Latency is preserved from this committed fallback trace and is not a production latency estimate."}</p>
        </div>
        <section className="truth-panel replay-truth" aria-label="Post-event truth" role="region">
          <div className="truth-head"><p className="eyebrow">Truth · selected portable event</p><span className="truth-lock">WITHHELD FROM MODEL</span></div>
          <h2>{titleCase(record.presentation_ground_truth.family)}</h2>
          <dl>
            <div><dt>Observed label</dt><dd>{record.presentation_ground_truth.label === 1 ? "Positive fraud case" : "Legitimate case"}</dd></div>
            <div><dt>Rail</dt><dd>{record.presentation_ground_truth.rail.toUpperCase()}</dd></div>
            <div><dt>Amount</dt><dd>{formatMoney(record.presentation_ground_truth.amount, record.presentation_ground_truth.currency)}</dd></div>
          </dl>
        </section>
      </section>

      <section className="event-ledger">
        <div className="panel-head"><div><p className="eyebrow">Ordered portable trace</p><h2>Curated event ledger</h2></div><div className="action-summary">{Object.entries(actionCounts).map(([action, count]) => <span key={action}>{count} {titleCase(action)}</span>)}</div></div>
        <div className="table-scroll">
          <table>
            <thead><tr><th scope="col">#</th><th scope="col">Event</th><th scope="col">Input context</th><th scope="col">Calibrated risk</th><th scope="col">Action</th><th className="truth-column" scope="col">Post-event truth</th></tr></thead>
            <tbody>
              {trace.traces.map((item, index) => (
                <tr className={index === current ? "is-current" : ""} key={item.event_id}>
                  <td><button aria-label={`Focus event ${index + 1}`} onClick={() => setCurrent(index)} type="button">{String(index + 1).padStart(2, "0")}</button></td>
                  <td><code>{item.event_id.slice(0, 8)}</code></td>
                  <td>{item.presentation_ground_truth.rail.toUpperCase()} · {formatMoney(item.presentation_ground_truth.amount)}</td>
                  <td className="numeric">{formatPercent(item.calibrated_probability, 1)}</td>
                  <td><span className={`action-label action-${actionTone(item.final_action)}`}><i aria-hidden="true" />{titleCase(item.final_action)}</span></td>
                  <td className="truth-column">{titleCase(item.presentation_ground_truth.family)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
