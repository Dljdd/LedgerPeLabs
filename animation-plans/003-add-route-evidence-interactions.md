# 003 — Add route-specific evidence interactions

- **Status**: DONE
- **Commit**: b37433c
- **Severity**: MEDIUM
- **Category**: Missed opportunities, physicality, cohesion
- **Estimated scope**: 7 files, about 280 lines

## Problem

The console has strong evidence-bound data but most routes present their main
visual as a static illustration. This makes Overview, Scenario, Defenses, and
Assurance feel like documents rather than a connected investigation product.
Investigation already supports selection, but its connected-value evidence is
still a text-only list.

Overview renders all 12 genuine trace values but has no event focus:

```tsx
/* web/src/app/views/Overview.tsx:43-58 — current */
<div className="trace-footprint" aria-label={summary} role="img">
  <div className="trace-footprint-head"><span>Curated decision footprint</span><span>{trace.traces.length} events</span></div>
  <svg aria-hidden="true" viewBox={`0 0 ${width} 68`}>
    ...
    {trace.traces.map((record, index) => (
      <circle className={`trace-risk-point is-${traceTone(record.final_action)}`} cx={xFor(index)} cy={yFor(record.calibrated_probability)} key={record.event_id} r="3.5" />
    ))}
  </svg>
  <div className="trace-threshold-labels">...</div>
</div>
```

Scenario's repository-bound motif and three configured campaign stages are
visually disconnected:

```tsx
/* web/src/app/views/Scenario.tsx:28-37 and 66-71 — current */
<div className="motif-card">
  <span className="eyebrow">Campaign motif</span>
  <div className="motif-code">{evidence.scenario_context.motif_signature}</div>
  <div className="motif-visual" aria-label="fan in to mule, layer, fan out, cash out">
    <span className="node-stack"><i /><i /><i /></span><b>→</b><span className="node-risk" /><b>→</b><span className="node-risk small" /><b>→</b><span className="node-stack reverse"><i /><i /></span>
  </div>
  ...
</div>

<section className="stage-row" aria-label="Campaign stages">
  {config.campaign_stages.map((stage, index) => (
    <article key={stage.stage_id}>...</article>
  ))}
</section>
```

Defenses has four architecture lanes but no focus state that lets a judge
compare them deliberately:

```tsx
/* web/src/app/views/Defenses.tsx:21-33 — current */
<section className="architecture-lanes" aria-label="Defense architecture">
  {evidence.recovered.arms.map((arm, index) => {
    ...
    return (
      <article className={`${isChampion ? "is-champion" : ""} ${isFull ? "is-not-ready" : ""}`} key={arm.arm}>
        ...
      </article>
    );
  })}
</section>
```

Assurance shortens hashes and only exposes full values through title tooltips,
which are poor for keyboard and mobile inspection:

```tsx
/* web/src/app/views/Assurance.tsx:22-27 — current */
<article className="lineage-panel">
  ...
  <ol className="lineage-list">
    {lineage.map(([label, hash, status], index) => <li key={label}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{label}</strong><code title={hash}>{shortHash(hash, 12)}</code></div><small>{status}</small></li>)}
  </ol>
</article>
```

## Target

Add one restrained, user-driven, repository-bound visual interaction to every
non-Replay route. No route entrance, automatic carousel, decorative loop, or
new numeric claim is allowed.

### Overview — trace focus

Keep a dedicated element with `role="img"` and the existing summary so the
current accessibility contract remains. Add a separate 12-button selector and
one active readout containing only:

- `Event NN`
- `formatPercent(calibrated_probability, 1)`
- `titleCase(final_action)`

Do not show ground-truth family, label, amount, or rail in this model-focused
overview visual. Add a vertical SVG cursor driven by the selected event's
existing `xFor(index)` coordinate. Move the cursor with a transform-only 240ms
`var(--ease-in-out)` transition. Highlight the selected point with
`transform: scale(1.45)` using `transform-box: fill-box` and
`transform-origin: center`; other point state changes use 180ms opacity and
transform transitions. The 12 selectors must expose event number,
probability, and action in their accessible names.

### Scenario — campaign-stage focus

Initialize the selected configured campaign stage to index 0. Each of the
three current stage cards becomes keyboard/click selectable and uses
`aria-pressed`. Do not autoplay them. Apply `is-active` to the selected card
and `data-stage={selectedStage}` to the existing motif card. Give the existing
motif elements semantic classes so the selected high-level stage emphasizes:

- Stage 0, `persuasion`: the first cluster.
- Stage 1, `transfer`: the first risk node and its incoming connector.
- Stage 2, `mule_dispersion`: the remaining risk node, fan-out cluster, and
  outgoing connectors.

This is a product illustration of the bound motif, not a new model result. Add
visible copy `Configured stage NN · <stage id>` near the motif. Use only 220ms
opacity and transform transitions; inactive motif elements remain visible at
opacity `.3` so content is never hidden.

### Investigation — connected-value bars

For the already-selected node, calculate the maximum amount among
`linkedEdges`. Add an aria-hidden bar to each current connected-value row using
`Number(edge.amount) / maxLinkedAmount`. Render it as a full-width pseudo track
whose fill is `transform: scaleX(var(--linked-value))` from the left. The
amount, time, and stage remain unchanged and visible. On entity change, the
existing keyed `.entity-selection` adds `translateY(4%)` to its starting style
and returns to zero over 200ms `var(--ease-out)`. The selected graph halo uses
`transform: scale(.88)` to `scale(1)` over 240ms `var(--ease-in-out)` with
`transform-box: fill-box` and `transform-origin: center`.

### Defenses — deliberate arm focus

Initialize focused arm to `ensemble_with_graph`. Each lane retains its article
semantics and gets a real child button with an accessible name such as
`Focus ensemble_with_graph architecture`. Do not create a full-card invisible
overlay. The button is visible, compact, and says `Inspect arm`. Clicking,
focusing, or fine-pointer hovering a lane changes the focused arm; persistent
selection remains after pointer leave. Add `is-focused` to the selected
article and `aria-pressed` to its button. The current `.arm-flow` boxes remain
the only topology and must not gain invented component labels. Its squares
transition by opacity and transform only, with a 40ms visual stagger based on
their existing DOM order; the transition must not delay text or interaction.
Below the lanes, render a keyed `architecture-focus` readout containing only
the selected arm name, the existing `armDescriptions[arm]`, its existing
status label, and its full `deterministic_result_sha256`.

### Assurance — inspectable lineage

Initialize selected lineage index to 0. Replace each visual lineage row with a
real button inside the list item. Preserve the row's number, label, short hash,
and status. Use `aria-pressed` and `is-selected`. Below the list render a keyed
`lineage-focus` with the selected label, full hash, and existing status. The
full hash must wrap and remain selectable. Animate only a 200ms opacity and
`translateY(4%)` transition on this occasional selection. Under the
TrustVerifier checks, add a static connector line and step-index emphasis; do
not animate check success or imply that verification is running live.

## Repo conventions to follow

- Easing tokens live at `web/src/styles.css:20-21`; reuse
  `--ease-out: cubic-bezier(.23, 1, .32, 1)` and
  `--ease-in-out: cubic-bezier(.77, 0, .175, 1)`.
- Button press feedback at `web/src/styles.css:88-89` uses `scale(.97)` at
  140ms and should remain.
- The Investigation graph already provides the correct keyboard pattern for
  SVG nodes at `web/src/app/views/Investigation.tsx:14-18`.
- Hover behavior belongs inside the existing
  `@media (hover: hover) and (pointer: fine)` query at
  `web/src/styles.css:515-523`.
- Dense analytical panels stay opaque; no translucent cards, glow, gradient,
  hover lift, rounded floating surfaces, or hidden-on-load content.

## Steps

1. In `web/src/app/views/Overview.tsx`, add selected trace-event state, the
   transform-driven SVG cursor, active-point class, selector buttons, and
   model-only active readout. Preserve a separate `role="img"` summary element.
2. In `web/src/app/views/Scenario.tsx`, add selected configured-stage state,
   semantic motif classes, `data-stage`, visible configured-stage readout, and
   accessible selection buttons inside the current three stage cards.
3. In `web/src/app/views/Investigation.tsx`, derive the maximum linked amount,
   render transform-scaled aria-hidden bars, and retain all existing values and
   node keyboard behavior.
4. In `web/src/app/views/Defenses.tsx`, add focused-arm state, visible focus
   buttons, persistent fine-pointer focus, `is-focused`, and the exact bounded
   architecture readout described above.
5. In `web/src/app/views/Assurance.tsx`, add lineage selection buttons and the
   full-hash readout. Keep TrustVerifier separate from the graph-model claim.
6. Add responsive styles to `web/src/styles.css`. At 560px, every new selector
   and readout must fit within the viewport; horizontal trace controls may
   scroll within their own labeled container.
7. Add unit tests in `web/src/app/App.test.tsx` covering Overview event focus,
   Scenario stage selection, Defenses default champion focus and alternate arm
   selection, Assurance full-hash selection, and Investigation value bars.
8. Extend `web/e2e/golden-path.spec.ts` with keyboard activation for at least
   the Overview trace selector and Assurance lineage selector. Keep Axe checks
   on Overview and Replay and add Assurance if it introduces no false positive.

## Boundaries

- Do NOT change or synthesize any model probability, action, latency, reason,
  graph edge, payment value, timeline value, metric, hash, status, or badge.
- Recovered metrics must visibly retain `authoritative=false` and
  `accepted_capacity_evidence=false`.
- `full_sentinel` remains diagnostic/not ready. Live portable predictions
  remain `ensemble_with_graph` only.
- Do NOT imply that a visual selection reruns scoring or verification.
- Do NOT place post-event truth inside a model-input/output visualization.
- Do NOT add dependencies, canvas, WebGL, network calls, remote assets, ambient
  loops, entrance animations, or new evidence files.
- If a displayed value cannot be obtained from the existing `evidence` or
  `trace` props, show no value rather than inventing one.

## Verification

- **Mechanical**: from `web/`, run `npm run typecheck`, `npm run lint`,
  `npm test -- --run`, `npm run build`, and `npm run e2e`. All must pass.
- **Feel check**: focus and click each new interaction at normal speed and at
  10% DevTools playback. Selection must respond immediately; movement settles
  in 240ms or less and never delays evidence text.
- Navigate all six routes at 1440×1000, 820×1180, and 390×844. Confirm no
  clipping, horizontal page overflow, overlapping controls, or unreadable
  hashes. Internal horizontal scrolling is allowed only for the labeled trace
  selector and existing tables/tapes.
- Use keyboard only to select an Overview event, Scenario stage, Defense arm,
  Investigation node, and Assurance lineage artifact. Visible focus must remain
  clear at each step.
- Toggle reduced motion and confirm all values remain visible, selection is
  immediate, and the only retained animation is brief opacity/color feedback.
- **Done when**: every route has a useful evidence-bound visual interaction,
  all existing evidence assertions remain unchanged, and Axe reports zero
  violations on Overview, Replay, and Assurance.
