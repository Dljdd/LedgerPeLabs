# 004 — Consolidate motion accessibility

- **Status**: DONE
- **Commit**: b37433c
- **Severity**: MEDIUM
- **Category**: Accessibility, cohesion and tokens
- **Estimated scope**: 2 files, about 65 lines

## Problem

The stylesheet has strong easing tokens but repeats literal duration values.
The current reduced-motion block collapses graph paint feedback as well as
movement, even though short color and opacity transitions help users understand
selection:

```css
/* web/src/styles.css:20-21 — current */
--ease-out: cubic-bezier(.23, 1, .32, 1);
--ease-in-out: cubic-bezier(.77, 0, .175, 1);
```

```css
/* web/src/styles.css:622-632 — current */
.replay-progress-fill,
.decision-observed,
.risk-readout,
.campaign-progress > i,
.replay-graph-edges line,
.graph-edges line,
.graph-nodes .node-body,
.graph-nodes .node-halo,
.entity-selection { transition-duration: .01ms; }
```

Plans 002 and 003 add several related 140–240ms transitions. Without a shared
duration scale the console will drift as it evolves.

## Target

Add exact motion duration tokens beside the existing easing tokens:

```css
--duration-press: 140ms;
--duration-fast: 160ms;
--duration-standard: 220ms;
--duration-progress: 240ms;
```

Replace literal 140ms, 160ms, 220ms, and 240ms durations only in interactive
motion rules touched by plans 001–003. Do not mechanically rewrite unrelated
CSS or change any duration's value.

Rewrite reduced-motion handling by property purpose:

- Disable transform movement for progress fills, transfer packets, node halos,
  and keyed content.
- Disable the active replay pulse.
- Retain 160ms `var(--ease-out)` opacity, color, background-color, border-color,
  fill, and stroke selection feedback.
- Keep button press transforms disabled.
- Keep `html { scroll-behavior: auto; }`.
- Do not use a global `* { transition-duration: .01ms !important; }` reset.

The JavaScript reduced-motion behavior specified in plan 002 is mandatory and
must remain the authority for disabling campaign autoplay.

## Repo conventions to follow

- All design tokens live in the single `:root` block in
  `web/src/styles.css:1-28`.
- The existing scoped reduced-motion query at `web/src/styles.css:622-636` is
  the correct location; extend it rather than creating parallel queries.
- `web/src/app/useReducedMotion.ts` from plan 002 owns JavaScript branching;
  CSS owns visual movement suppression.

## Steps

1. Add the four exact duration tokens to `:root` after the easing tokens.
2. Replace matching literals only in motion rules added or touched by plans
   001–003. Preserve every effective duration.
3. Refactor the existing reduced-motion query to disable positional motion and
   replay looping while retaining short state-identification feedback.
4. Add a Playwright reduced-motion test in `web/e2e/golden-path.spec.ts`: call
   `page.emulateMedia({ reducedMotion: "reduce" })`, navigate to Replay, confirm
   the control says `Step campaign`, press it once, confirm campaign payment 02,
   wait at least 1100ms, and confirm it remains on payment 02.
5. Add a CSS assertion or browser evaluation confirming the transfer packet has
   `animation-name: none` under reduced motion.

## Boundaries

- Do NOT change colors, easing curves, layout, DOM order, evidence values,
  replay intervals, or accessible names beyond the plan 002 reduced-motion
  control labels.
- Do NOT introduce `transition: all`, `ease-in`, `scale(0)`, global animation
  resets, or `!important`.
- Do NOT suppress focus indicators or color/opacity selection feedback.
- Plans 002 and 003 must be implemented before this plan.

## Verification

- **Mechanical**: from `web/`, run `npm run typecheck`, `npm run lint`,
  `npm test -- --run`, `npm run build`, and `npm run e2e`. All must pass.
- **Feel check**: toggle reduced motion while Replay is open. Confirm autoplay
  cannot start, transfer position changes do not animate, and active edge/node,
  selector, and inspector feedback remains visually clear.
- In the Animations panel at 10% playback, normal motion must use only the four
  token durations for interactive transitions touched in this pass.
- **Done when**: normal motion remains crisp and explanatory, reduced motion
  removes travel without removing state comprehension, and no route changes its
  evidence content.
