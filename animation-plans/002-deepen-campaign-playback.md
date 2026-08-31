# 002 — Deepen campaign playback motion

- **Status**: DONE
- **Commit**: b37433c
- **Severity**: HIGH
- **Category**: Purpose and frequency, physicality, accessibility
- **Estimated scope**: 3 files, about 150 lines

## Problem

Replay advances the selected campaign edge every 900ms, but the graph only
changes line paint. The source and target entities remain visually identical,
there is no visible transfer between them, and the evidence inspector replaces
its values without spatial continuity:

```tsx
/* web/src/app/views/Replay.tsx:126-138 — current */
useEffect(() => {
  if (!campaignPlaying) return;
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
}, [campaignEdges.length, campaignPlaying]);
```

```tsx
/* web/src/app/views/Replay.tsx:94-102 — current */
<g className="replay-graph-nodes">
  {nodes.map((node) => (
    <g key={node.id} transform={`translate(${node.x} ${node.y})`}>
      <circle className="node-body" r="17" />
      <circle className="node-core" r="4" />
      <text x="27" y="-2">{node.label}</text>
      <text className="node-meta" x="27" y="12">{node.role.toUpperCase()} / {node.country}</text>
    </g>
  ))}
</g>
```

```css
/* web/src/styles.css:301-308 — current */
.replay-graph-edges line { stroke: #675f54; stroke-width: 1.1; opacity: .78; vector-effect: non-scaling-stroke; transition: opacity 220ms var(--ease-out), stroke 180ms var(--ease-out), stroke-width 180ms var(--ease-out); }
.replay-graph-edges line.is-selected { stroke: var(--orange); stroke-width: 2.5; opacity: 1; }
.replay-graph-edges line.is-concealed { opacity: .07; }
.replay-campaign-graph marker path { fill: #8c8377; }
.replay-graph-nodes .node-body { fill: #100f0c; stroke: #aaa092; stroke-width: 1.1; vector-effect: non-scaling-stroke; }
.replay-graph-nodes .node-core { fill: var(--text-soft); }
```

Reduced-motion mode only shortens CSS transitions. It does not change the
JavaScript autoplay behavior, so pressing Play still produces a continuous
sequence of state changes:

```css
/* web/src/styles.css:622-632 — current */
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .replay-progress-fill,
  .decision-observed,
  .risk-readout,
  .campaign-progress > i,
  .replay-graph-edges line,
  .graph-edges line,
  .graph-nodes .node-body,
  .graph-nodes .node-halo,
  .entity-selection { transition-duration: .01ms; }
}
```

## Target

Keep the genuine ten-edge order and 900ms interval unchanged. Add one
explanatory transfer marker that moves from the selected edge's real source
coordinates to its real target coordinates. Animate only `transform` and
`opacity`; the marker runs for 720ms, leaving 180ms of rest before the next
edge. Use the existing strong on-screen movement curve:

```css
@keyframes replay-value-transfer {
  0% {
    opacity: 0;
    transform: translate(var(--packet-source-x), var(--packet-source-y)) scale(.9);
  }
  18% { opacity: 1; }
  72% { opacity: 1; }
  100% {
    opacity: 0;
    transform: translate(var(--packet-target-x), var(--packet-target-y)) scale(1);
  }
}

.replay-value-packet {
  fill: var(--orange-hot);
  stroke: #100f0c;
  stroke-width: 3;
  vector-effect: non-scaling-stroke;
  animation: replay-value-transfer 720ms var(--ease-in-out) both;
  pointer-events: none;
}
```

Remount the marker by keying it with the genuine `payment_id`. Add graph-node
classes derived only from `edges.slice(0, selectedEdge + 1)`:

- `is-visited` for any endpoint already encountered.
- `is-active-source` for the selected edge source.
- `is-active-target` for the selected edge target.

Add a `node-halo` circle before each node body. Use opacity and transform for
focus; never animate SVG coordinates or radius:

```css
.replay-graph-nodes g { opacity: .28; transition: opacity 220ms var(--ease-out); }
.replay-graph-nodes g.is-visited { opacity: .72; }
.replay-graph-nodes g.is-active-source,
.replay-graph-nodes g.is-active-target { opacity: 1; }
.replay-graph-nodes .node-halo {
  fill: none;
  stroke: var(--orange);
  stroke-width: 1.5;
  opacity: 0;
  transform: scale(.9);
  transform-box: fill-box;
  transform-origin: center;
  transition: opacity 220ms var(--ease-out), transform 240ms var(--ease-in-out);
}
.replay-graph-nodes .is-active-target .node-halo {
  opacity: .9;
  transform: scale(1);
}
```

Wrap only the changing campaign-payment values in a child keyed by
`campaignEdge.payment_id`. Wrap only the changing portable decision and facts
in a child keyed by `record.event_id`; keep the portable control buttons
outside that keyed child so keyboard focus is never discarded. Both state
wrappers use a 200ms entrance transition:

```css
.scenario-payment-state,
.portable-response-state {
  opacity: 1;
  transform: translateY(0);
  transition: opacity 200ms var(--ease-out), transform 200ms var(--ease-out);
}
@starting-style {
  .scenario-payment-state,
  .portable-response-state {
    opacity: .58;
    transform: translateY(4%);
  }
}
```

Use `window.matchMedia("(prefers-reduced-motion: reduce)")` in Replay through a
small reusable `web/src/app/useReducedMotion.ts` hook. It must subscribe with
`addEventListener("change", ...)`, clean up the listener, and tolerate tests
where `matchMedia` is unavailable. In reduced-motion mode the campaign control
becomes an explicit one-edge **Step campaign** action and must never start the
900ms interval. At the final edge its label is **Reset campaign**. Normal mode
keeps Play/Pause and labels the final stopped state **Replay campaign**.

Under the existing reduced-motion query, disable the transfer animation and
position the marker at its target without movement. Remove only translate/scale
movement from the keyed wrappers while retaining their 160–200ms opacity
feedback:

```css
@media (prefers-reduced-motion: reduce) {
  .replay-value-packet {
    animation: none;
    opacity: 1;
    transform: translate(var(--packet-target-x), var(--packet-target-y));
  }
  .scenario-payment-state,
  .portable-response-state {
    transform: none;
    transition: opacity 160ms var(--ease-out);
  }
  .replay-graph-nodes .node-halo { transform: none; }
}
```

## Repo conventions to follow

- Easing tokens are at `web/src/styles.css:20-21`; reuse
  `--ease-out: cubic-bezier(.23, 1, .32, 1)` and
  `--ease-in-out: cubic-bezier(.77, 0, .175, 1)`.
- The transform-only campaign progress implementation at
  `web/src/styles.css:309-312` is the performance exemplar.
- `CampaignPlaybackGraph` already binds source/target coordinates and payment
  IDs from the repository graph at `web/src/app/views/Replay.tsx:64-104`.
- `aria-live="polite"` on the scenario payment and portable event counter is
  part of the accessibility contract.

## Steps

1. Create `web/src/app/useReducedMotion.ts` with the defensive matchMedia hook
   described above.
2. In `web/src/app/views/Replay.tsx`, derive the active edge endpoints and
   visited node IDs; render node classes, a `node-halo`, and the keyed transfer
   marker using only the bound node coordinates and payment ID.
3. Integrate `useReducedMotion`. Preserve the current 900ms normal-mode timer;
   implement deterministic one-edge Step/Reset behavior for reduced motion.
4. Add the keyed scenario and portable state wrappers. Do not key or remount
   the portable control buttons.
5. Add the exact motion styles above to `web/src/styles.css`. A subtle opacity
   pulse on the existing play-state dot is allowed only while
   `aria-pressed="true"`; it must be disabled under reduced motion.
6. Add tests in `web/src/app/App.test.tsx` for normal Play/Pause labels and for
   reduced-motion Step/Reset behavior. Restore any matchMedia mocks after each
   test.
7. Update `web/e2e/golden-path.spec.ts` only when accessible names change; keep
   the existing golden path intent.

## Boundaries

- Do NOT change graph nodes, edges, coordinates, event ordering, payment
  values, cumulative values, timestamps, probability values, actions, reasons,
  thresholds, latency, hashes, or evidence files.
- Do NOT imply a mapping between a scenario payment and a portable replay row.
- Do NOT relabel the portable arm; it remains `ensemble_with_graph` only.
- Do NOT add route entrances, ambient loops outside active replay, bounce,
  motion dependencies, canvas, WebGL, or network calls.
- Truth remains structurally separate from model input/output.
- If the cited code has drifted materially, STOP and report instead of
  improvising.

## Verification

- **Mechanical**: from `web/`, run `npm run typecheck`, `npm run lint`,
  `npm test -- --run`, `npm run build`, and `npm run e2e`. All must pass.
- **Feel check**: in normal motion, play the campaign and confirm each bound
  edge gets one 720ms source-to-target transfer followed by 180ms rest; the
  active target is unmistakable, and pausing stops future steps immediately.
- At 10% DevTools playback speed, confirm only transform and opacity move for
  the transfer marker and keyed inspector content.
- Switch rapidly among tape entries and portable trace events; confirm content
  always reflects the selected IDs and the portable control retains focus.
- Toggle reduced motion; confirm the button says Step campaign, one press moves
  exactly one genuine edge, no interval starts, and opacity/state color still
  communicates the change.
- **Done when**: all ten genuine payments can be played, paused, selected, and
  reset deterministically, with no claim-boundary or accessibility regression.
