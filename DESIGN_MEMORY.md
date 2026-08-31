# APAR Design Memory

## Brand Tone

- Institutional, forensic, precise, restrained, and memorable.
- Prefer an evidence dossier or examination-room character over a generic analytics dashboard.
- Avoid repeated floating cards, colored top-edge decoration, ordinary metadata pills, glow, decorative grids, fake application chrome, and hover lift.

## Layout and Spacing

- Use comfortable-compact density: dense evidence can remain visible when its hierarchy is explicit.
- Organize complex tasks as a focus surface plus a persistent evidence inspector.
- Use continuous rulework, ledger rows, timelines, and split panes before isolated card grids.
- Keep analytical surfaces opaque. Translucency is limited to sticky navigation where it clarifies hierarchy.
- Corners remain square; elevation comes from contrast and rule weight rather than shadows.

## Typography

- Use the offline-safe editorial serif stack for page narratives, campaign stages, decisive amounts, probabilities, and conclusions.
- Use the humanist/system sans stack for body copy, navigation, controls, and explanatory text.
- Use the monospaced stack for hashes, evidence labels, time, latency, probability tables, thresholds, and machine-readable identifiers.
- Preserve tabular numerals for all metrics and financial values.

## Color

- Build on deep warm ink rather than cool charcoal.
- Use warm paper text and muted taupe metadata.
- Orange indicates focus and selected progress, red indicates decline/failure, amber indicates caution or post-event truth, and green indicates verified/passed state.
- Every semantic state must also include text, shape, or iconography; color is never the only cue.

## Interaction Patterns

- Motion is reserved for campaign playback, graph focus, calibrated progress, and meaningful state changes.
- Selected ordered steps use `aria-current`; independent selectors use `aria-pressed`.
- Preserve visible reset and step-forward controls for deterministic demonstrations.
- Do not animate content into existence or hide judge-critical evidence behind hover.
- Respect `prefers-reduced-motion` and keep touch targets at least 44 CSS pixels where practical.

## Evidence Presentation

- Always call live portable predictions `ensemble_with_graph`.
- Scenario graph records and portable trace records are independent evidence streams unless an artifact provides an explicit mapping.
- Post-event truth must be structurally separate and marked as withheld from the model.
- Recovered diagnostics retain `authoritative=false`, `accepted_capacity_evidence=false`, and their non-authoritative qualifier.
- Evidence-pending states replace unsupported comparison numbers or productivity claims.

## Accessibility

- Maintain WCAG 2.2 AA contrast, visible focus, semantic regions, table headers, screen-reader labels, and keyboard operation.
- Mobile layouts stack the focus surface and inspector without horizontal page overflow; dense tapes and tables may scroll inside explicitly bounded containers.

## Repository Conventions

- Frontend remains isolated under `web/` and uses React, TypeScript, Vite, and local CSS tokens.
- No runtime network dependencies, remote fonts, telemetry, credentials, or external analytics.
- Source every displayed model value from parsed console evidence or the verified trace.
