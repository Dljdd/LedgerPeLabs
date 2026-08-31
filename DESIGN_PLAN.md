# Design Implementation Plan: APAR Assurance Console

## Summary

- **Scope:** Six-route judge-facing assurance console
- **Target:** `web/src/app` and `web/src/styles.css`
- **Winning direction:** Variant F campaign narrative with Variant B Editorial Casefile treatment
- **Key improvements:** graph-led campaign playback, persistent detection inspector, independent evidence selectors, serif narrative hierarchy, warm ink surfaces, and reduced card/pill repetition

## Files Changed

- [x] `web/src/app/views/Replay.tsx` — graph-led campaign and portable-response narrative
- [x] `web/src/styles.css` — Editorial Casefile tokens, typography, surfaces, responsive behavior, and state motion
- [x] `DESIGN_MEMORY.md` — durable product design rules

## Implementation

1. Apply an offline-safe editorial serif stack to narrative headings and decisive values.
2. Retain sans-serif body copy and monospaced labels, hashes, timing, probabilities, and evidence identifiers.
3. Present the 10 genuine scenario payment edges as an ordered campaign playback.
4. Keep the 12 verified portable events independently selectable and visibly label the lack of record-level mapping.
5. Show calibrated probability, bound thresholds, final action, latency, feature count, and reason evidence for the selected portable event.
6. Move post-event truth into a separate examination panel marked as withheld from the model.
7. Carry warm dossier surfaces, fine rulework, and semantic orange/red/amber/green states through every console route.
8. Preserve offline behavior, deterministic reset, keyboard operation, reduced motion, and responsive breakpoints.

## Component Contract

### Replay

- **Props:** `evidence: ConsoleEvidence`, `trace: VerifiedTrace`, `traceMode: TraceMode`
- **State:** selected scenario edge, campaign playback state, and independently selected portable trace event
- **Events:** campaign play/pause, scenario-edge focus, portable-event focus, step-forward, and reset
- **Evidence rule:** selected scenario and portable records never imply a payment-to-trace mapping

## Required UI States

- **Loading:** retained at the application loader boundary
- **Empty/degraded:** replay stops safely when bound scenario or trace evidence is absent
- **Disabled:** portable step-forward control disables at the final trace event
- **Selected:** scenario and portable selectors expose `aria-current` or `aria-pressed`
- **Playing:** campaign progress and graph-edge reveal are the only continuous motion
- **Reduced motion:** state transitions collapse under `prefers-reduced-motion`

## Accessibility Checklist

- [x] Semantic buttons, headings, regions, lists, tables, and SVG descriptions
- [x] Keyboard-accessible campaign and portable controls
- [x] Visible focus treatment
- [x] Non-color state labels and icons
- [x] Desktop and mobile Playwright journey
- [x] Automated WCAG accessibility scan
- [x] Viewport overflow audit at supported breakpoints

## Verification Checklist

- [x] TypeScript check
- [x] ESLint
- [x] Focused unit tests
- [x] Production build after Design Lab cleanup
- [x] Existing portable-model replay verification
- [x] Final screenshots
- [x] Repository hygiene scans and diff validation

## Design Tokens

- **Display:** `Iowan Old Style`, `Palatino Linotype`, Palatino, Georgia, serif
- **Body:** `Avenir Next`, Avenir, `Helvetica Neue`, system sans-serif
- **Evidence:** `SFMono-Regular`, Consolas, `Liberation Mono`, monospace
- **Ink:** `#0a0907`
- **Paper:** `#f1e9dc`
- **Dossier surface:** `#14110d`
- **Rule:** `#3f382e`
- **Mastercard-adjacent orange:** `#e8793d`
