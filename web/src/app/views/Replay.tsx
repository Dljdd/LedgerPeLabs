import { useEffect, useMemo, useState } from "react";

import { Icon } from "../Icon";
import { formatMoney, formatPercent, shortHash, titleCase } from "../format";
import type { ConsoleEvidence, TraceMode, VerifiedTrace } from "../types";

function actionTone(action: string): string {
  if (action === "decline_hold") return "critical";
  if (action === "review_hold" || action === "challenge") return "amber";
  return "good";
}

export function Replay({ evidence, trace, traceMode }: { evidence: ConsoleEvidence; trace: VerifiedTrace; traceMode: TraceMode }) {
  const [current, setCurrent] = useState(0);
  const [playing, setPlaying] = useState(false);
  const record = trace.traces[current] ?? trace.traces[0];
  const inputRecord = evidence.portable.records[current] ?? evidence.portable.records[0];
  const total = trace.traces.length;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setCurrent((index) => {
        if (index >= total - 1) {
          setPlaying(false);
          return index;
        }
        return index + 1;
      });
    }, 850);
    return () => window.clearInterval(timer);
  }, [playing, total]);

  const actionCounts = useMemo(() => trace.traces.reduce<Record<string, number>>((counts, item) => {
    counts[item.final_action] = (counts[item.final_action] ?? 0) + 1;
    return counts;
  }, {}), [trace.traces]);

  if (!record || !inputRecord) {
    return <div className="empty-state"><h1>No verified replay events</h1><p>The console stopped safely because the trace is empty.</p></div>;
  }

  const stepForward = () => setCurrent((index) => Math.min(index + 1, total - 1));
  const reset = () => { setPlaying(false); setCurrent(0); };

  return (
    <div className="page replay-page">
      <header className="page-header split-header replay-header">
        <div><p className="eyebrow">03 · Decision surface</p><h1>Verified decision replay</h1><p>Accepted checkpoint evidence, scored by the local portable graph ensemble.</p></div>
        <div className="live-arm"><span className="status-dot" aria-hidden="true" /><span><small>{traceMode === "live_local_scorer" ? "LIVE LOCAL SCORER" : "HASH-BOUND VERIFIED FALLBACK"}</small><strong>ensemble_with_graph</strong></span></div>
      </header>

      <section className="replay-controller" aria-label="Replay controls">
        <div className="control-buttons">
          <button className="button button-primary compact-button" onClick={() => setPlaying((value) => !value)} type="button">
            <Icon name={playing ? "pause" : "play"} /> {playing ? "Pause replay" : current === total - 1 ? "Replay complete" : "Run replay"}
          </button>
          <button className="icon-button" disabled={current === total - 1} onClick={stepForward} title="Step forward" type="button"><Icon name="step" /><span className="sr-only">Step forward</span></button>
          <button className="icon-button" onClick={reset} title="Reset replay" type="button"><Icon name="reset" /><span className="sr-only">Reset replay</span></button>
        </div>
        <div className="replay-progress">
          <div><span aria-live="polite">Event {String(current + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}</span><span>{playing ? "Advancing" : current === total - 1 ? "Complete" : "Ready"}</span></div>
          <progress max={total} value={current + 1}>{current + 1} of {total}</progress>
        </div>
          <div className="trace-bind"><span>Trace</span><code title={trace.trace_sha256}>{shortHash(trace.trace_sha256, 8)}</code><span className="pill pill-good">{traceMode === "live_local_scorer" ? "Scored locally" : "Verified fallback"}</span></div>
      </section>

      <section className="decision-grid">
        <article className="decision-main" aria-label="Model evidence" role="region">
          <div className="panel-head"><div><p className="eyebrow">Model evidence</p><h2>Decision at event time</h2></div><span className={`pill pill-${actionTone(record.final_action)}`}>{titleCase(record.final_action)}</span></div>
          <div className="decision-score">
            <div className="probability-ring" style={{ "--score": `${record.calibrated_probability * 100}%` } as React.CSSProperties}>
              <span><strong>{formatPercent(record.calibrated_probability, 1)}</strong><small>calibrated risk</small></span>
            </div>
            <dl>
              <div><dt>Final action</dt><dd>{titleCase(record.final_action)}</dd></div>
              <div><dt>Fixed-trace latency</dt><dd>{record.latency_ms.toFixed(3)} ms</dd></div>
              <div><dt>Member disagreement</dt><dd>{record.disagreement.toFixed(4)}</dd></div>
              <div><dt>Feature vector</dt><dd>{inputRecord.model_input.features.length} bound features</dd></div>
            </dl>
          </div>
          <div className="reason-box"><span>Reason evidence</span>{record.reason_codes.map((reason) => <code key={reason}>{reason}</code>)}</div>
          <p className="technical-note">Latency is preserved from this committed fallback trace and is not a production latency estimate.</p>
        </article>

        <article className="truth-panel" aria-label="Post-event truth" role="region">
          <div className="truth-head"><p className="eyebrow">Truth · post-event only</p><span className="truth-lock">WITHHELD FROM MODEL</span></div>
          <h2>{titleCase(record.presentation_ground_truth.family)}</h2>
          <dl>
            <div><dt>Observed label</dt><dd>{record.presentation_ground_truth.label === 1 ? "Positive fraud case" : "Legitimate case"}</dd></div>
            <div><dt>Rail</dt><dd>{record.presentation_ground_truth.rail.toUpperCase()}</dd></div>
            <div><dt>Amount</dt><dd>{formatMoney(record.presentation_ground_truth.amount, record.presentation_ground_truth.currency)}</dd></div>
          </dl>
          <div className="separation-note"><Icon name="check" /><p><strong>Structural separation passed</strong>Truth is attached only after the model decision for presentation scoring.</p></div>
        </article>
      </section>

      <section className="event-ledger">
        <div className="panel-head"><div><p className="eyebrow">Ordered trace</p><h2>Curated event ledger</h2></div><div className="action-summary">{Object.entries(actionCounts).map(([action, count]) => <span key={action}>{count} {titleCase(action)}</span>)}</div></div>
        <div className="table-scroll">
          <table>
            <thead><tr><th scope="col">#</th><th scope="col">Event</th><th scope="col">Input context</th><th scope="col">Calibrated risk</th><th scope="col">Action</th><th className="truth-column" scope="col">Post-event truth</th></tr></thead>
            <tbody>
              {trace.traces.map((item, index) => (
                <tr className={index === current ? "is-current" : ""} key={item.event_id}>
                  <td><button aria-label={`Focus event ${index + 1}`} onClick={() => { setPlaying(false); setCurrent(index); }} type="button">{String(index + 1).padStart(2, "0")}</button></td>
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
