import { useMemo, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";

import { formatMoney, formatPercent, shortHash, titleCase } from "../format";
import type { ConsoleEvidence, GraphNode, VerifiedTrace } from "../types";

function Graph({ evidence, selected, onSelect }: { evidence: ConsoleEvidence; selected: string | null; onSelect: (node: GraphNode) => void }) {
  const { edges, nodes } = evidence.scenario_context.graph;
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const edgeAmounts = edges.map((edge) => Number(edge.amount));
  const minAmount = Math.min(...edgeAmounts);
  const maxAmount = Math.max(...edgeAmounts);
  const edgeWidth = (amount: number) => maxAmount === minAmount ? 2 : 1.2 + ((amount - minAmount) / (maxAmount - minAmount)) * 2.8;
  const selectFromKeyboard = (event: KeyboardEvent<SVGGElement>, node: GraphNode) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(node);
    }
  };

  return (
    <svg aria-label={`Campaign entity graph with ${nodes.length} linked entities and ${edges.length} directional payment edges; edge weight represents payment amount`} className={`campaign-graph ${selected ? "has-selection" : ""}`} role="img" viewBox="20 20 650 660">
      <defs>
        <marker id="payment-arrow" markerHeight="6" markerUnits="strokeWidth" markerWidth="6" orient="auto" refX="5" refY="3" viewBox="0 0 6 6"><path d="M0 0L6 3L0 6Z" /></marker>
        <marker id="payment-arrow-focused" markerHeight="6" markerUnits="strokeWidth" markerWidth="6" orient="auto" refX="5" refY="3" viewBox="0 0 6 6"><path d="M0 0L6 3L0 6Z" /></marker>
      </defs>
      <g className="graph-edges">
        {edges.map((edge) => {
          const source = byId.get(edge.source);
          const target = byId.get(edge.target);
          if (!source || !target) return null;
          const isFocused = selected === source.id || selected === target.id;
          const isDimmed = selected !== null && !isFocused;
          const distance = Math.hypot(target.x - source.x, target.y - source.y);
          const targetInset = distance === 0 ? 0 : 22 / distance;
          const x2 = target.x - (target.x - source.x) * targetInset;
          const y2 = target.y - (target.y - source.y) * targetInset;
          return (
            <line
              className={`${isFocused ? "is-focused" : ""} ${isDimmed ? "is-dimmed" : ""}`}
              key={edge.payment_id}
              markerEnd={`url(#${isFocused ? "payment-arrow-focused" : "payment-arrow"})`}
              style={{ "--edge-width": edgeWidth(Number(edge.amount)) } as CSSProperties}
              vectorEffect="non-scaling-stroke"
              x1={source.x}
              x2={x2}
              y1={source.y}
              y2={y2}
            >
              <title>{`${edge.payment_id}: ${source.label} to ${target.label}, ${formatMoney(edge.amount, edge.currency)}, ${titleCase(edge.stage)}`}</title>
            </line>
          );
        })}
      </g>
      <g className="graph-nodes">
        {nodes.map((node) => (
          <g className={`${node.illicit ? "is-illicit" : ""} ${selected === node.id ? "is-selected" : ""}`} key={node.id} onClick={() => onSelect(node)} role="button" tabIndex={0} onKeyDown={(event) => selectFromKeyboard(event, node)} aria-label={`${node.label}, ${node.role}, ${node.country}${node.illicit ? ", known illicit in synthetic truth" : ""}`}>
            <circle className="node-halo" cx={node.x} cy={node.y} r="25" />
            <circle className="node-body" cx={node.x} cy={node.y} r="17" />
            <circle className="node-core" cx={node.x} cy={node.y} r="5" />
            <text x={node.x + 25} y={node.y - 3}>{node.label}</text>
            <text className="node-meta" x={node.x + 25} y={node.y + 13}>{node.role.toUpperCase()} · {node.country}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

export function Investigation({ evidence, trace }: { evidence: ConsoleEvidence; trace: VerifiedTrace }) {
  const [selected, setSelected] = useState<GraphNode | null>(() => evidence.scenario_context.graph.nodes.find((node) => node.illicit) ?? null);
  const edges = evidence.scenario_context.graph.edges;
  const firstAppAlert = trace.traces.find((record) => record.presentation_ground_truth.family === "app_scam_mule" && record.final_action !== "approve");
  const firstAppAlertIndex = firstAppAlert ? trace.traces.indexOf(firstAppAlert) : -1;
  const linkedEdges = useMemo(() => selected ? edges.filter((edge) => edge.source === selected.id || edge.target === selected.id) : [], [edges, selected]);
  const maxLinkedAmount = Math.max(0, ...linkedEdges.map((edge) => Number(edge.amount)));
  const illicitCount = evidence.scenario_context.graph.nodes.filter((node) => node.illicit).length;

  return (
    <div className="page">
      <header className="page-header split-header">
        <div><p className="eyebrow">04 · Investigation</p><h1>Campaign-level evidence</h1><p>Follow value progression across linked actors instead of opening isolated transaction alerts.</p></div>
        <div className="case-id"><span>CASE GROUP</span><code title={evidence.scenario_context.case_grouping.case_id}>{shortHash(evidence.scenario_context.case_grouping.case_id, 16)}</code><small>Generated campaign ID · deterministic grouping</small></div>
      </header>

      <section className="investigation-layout">
        <article className="graph-panel">
          <div className="panel-head"><div><p className="eyebrow">Entity graph</p><h2>APP–mule value path</h2></div><div className="graph-legend"><span><i className="legend-neutral" />Entity</span><span><i className="legend-risk" />Synthetic illicit truth</span></div></div>
          <Graph evidence={evidence} onSelect={setSelected} selected={selected?.id ?? null} />
          <div className="graph-foot"><span>{evidence.scenario_context.graph.nodes.length} entities</span><span>{edges.length} payment edges</span><span>{illicitCount} synthetic illicit nodes</span><span>Arrow = payment direction</span><span>Weight = amount</span><code title={evidence.scenario_context.graph.graph_sha256}>graph {shortHash(evidence.scenario_context.graph.graph_sha256, 8)}</code></div>
        </article>

        <aside className="entity-inspector" aria-label="Selected entity details">
          <div className="panel-head"><div><p className="eyebrow">Focused entity</p><h2>{selected?.label ?? "Select a node"}</h2></div>{selected?.illicit ? <span className="pill pill-critical">Illicit truth</span> : <span className="pill">Context node</span>}</div>
          {selected ? <div className="entity-selection" key={selected.id}>
              <dl className="definition-table compact-definitions">
                <div><dt>Role</dt><dd>{titleCase(selected.role)}</dd></div>
                <div><dt>Country</dt><dd>{selected.country}</dd></div>
                <div><dt>Linked payments</dt><dd>{linkedEdges.length}</dd></div>
                <div><dt>Account</dt><dd><code>{shortHash(selected.account_id, 12)}</code></dd></div>
              </dl>
              <div className="linked-events"><span className="eyebrow">Connected value</span>{linkedEdges.map((edge) => <div key={edge.payment_id}><span><b>{titleCase(edge.stage)}</b><small>{edge.event_time.slice(11, 19)} UTC</small></span><strong>{formatMoney(edge.amount, edge.currency)}</strong><i aria-hidden="true" className="linked-value-bar" style={{ "--linked-value": maxLinkedAmount > 0 ? Number(edge.amount) / maxLinkedAmount : 0 } as CSSProperties} /></div>)}</div>
            </div> : null}
        </aside>
      </section>

      <section className="investigation-strip">
        <article><span className="card-index">First curated APP intervention</span><strong>{firstAppAlert ? titleCase(firstAppAlert.final_action) : "Evidence pending"}</strong><p>{firstAppAlert ? `${formatMoney(firstAppAlert.presentation_ground_truth.amount)} · event ${String(firstAppAlertIndex + 1).padStart(2, "0")} · ${formatPercent(firstAppAlert.calibrated_probability, 1)} calibrated` : "No bound APP intervention"}</p></article>
        <article><span className="card-index">CUMULATIVE ATTEMPTED VALUE</span><strong>{formatMoney(evidence.scenario_context.value_total)}</strong><p>{evidence.scenario_context.payment_count} ordered campaign payments</p></article>
        <article><span className="card-index">ANALYST TIME ESTIMATE</span><strong>Evidence pending</strong><p>No placeholder productivity claim</p></article>
        <article><span className="card-index">CASE GROUPING</span><strong>1 campaign case</strong><p>{evidence.scenario_context.case_grouping.event_count} grouped events</p></article>
      </section>

      <section className="value-ledger panel">
        <div className="panel-head"><div><p className="eyebrow">Value progression</p><h2>Ordered campaign payments</h2></div><span className="pill pill-good">Ledger conserved</span></div>
        <div className="table-scroll"><table><thead><tr><th>#</th><th>Time</th><th>Stage</th><th>Source → target</th><th>Amount</th><th>Cumulative</th></tr></thead><tbody>{edges.map((edge, index) => <tr key={edge.payment_id}><td>{String(index + 1).padStart(2, "0")}</td><td className="numeric">{edge.event_time.slice(11, 19)}</td><td>{titleCase(edge.stage)}</td><td><code>{edge.source.slice(0, 6)} → {edge.target.slice(0, 6)}</code></td><td className="numeric">{formatMoney(edge.amount, edge.currency)}</td><td className="numeric">{formatMoney(edge.cumulative_attempted_value, edge.currency)}</td></tr>)}</tbody></table></div>
      </section>
    </div>
  );
}
