# 001 — Tighten replay and interaction motion

- **Status**: TODO
- **Commit**: 1beb91a
- **Severity**: MEDIUM
- **Category**: Performance, accessibility, purpose and frequency
- **Estimated scope**: 2 files, about 45 lines

## Problem

The replay progress indicator animates a layout property for 500ms, which is
slower than the 300ms UI budget and forces width work during a frequently used
step interaction:

```css
/* web/src/styles.css:212 — current */
progress::-webkit-progress-value { background: var(--orange); transition: width 500ms var(--ease-in-out); }
```

The primary navigation and controls also apply hover states without checking
whether the device has a precise hover-capable pointer:

```css
/* web/src/styles.css:56, 91, 95, 206 and 374 — current */
.nav-link:hover { color: var(--text); background: #11151a; }
.button-primary:hover { background: var(--orange-hot); }
.text-link:hover { color: var(--orange-hot); }
.icon-button:hover:not(:disabled) { background: #252b32; }
.round-link:hover { background: var(--surface-2); }
```

Finally, reduced-motion mode removes every transition, including color feedback
that helps users understand state:

```css
/* web/src/styles.css:463-466 — current */
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
```

## Target

Use a transform-only visual progress fill while retaining the native
`<progress>` element for accessibility. The fill must transition in 240ms with
the existing `--ease-in-out: cubic-bezier(.77, 0, .175, 1)` token and use a left
transform origin:

```css
.replay-progress-track {
  height: 4px;
  overflow: hidden;
  background: #2c3138;
}

.replay-progress-fill {
  display: block;
  width: 100%;
  height: 100%;
  background: var(--orange);
  transform: scaleX(var(--replay-progress));
  transform-origin: left center;
  transition: transform 240ms var(--ease-in-out);
}
```

Render the visual track beside a screen-reader-only native progress element:

```tsx
<div className="replay-progress-track" aria-hidden="true">
  <span
    className="replay-progress-fill"
    style={{ "--replay-progress": (current + 1) / total } as React.CSSProperties}
  />
</div>
<progress className="sr-only" max={total} value={current + 1}>
  {current + 1} of {total}
</progress>
```

Wrap the five hover rules in this exact device query:

```css
@media (hover: hover) and (pointer: fine) {
  .nav-link:hover { color: var(--text); background: #11151a; }
  .button-primary:hover { background: var(--orange-hot); }
  .text-link:hover { color: var(--orange-hot); }
  .icon-button:hover:not(:disabled) { background: #252b32; }
  .round-link:hover { background: var(--surface-2); }
}
```

Reduced-motion mode must remove the progress movement and press scaling while
retaining short color feedback:

```css
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .replay-progress-fill { transition-duration: .01ms; }
  .button:active:not(:disabled),
  .icon-button:active:not(:disabled),
  .round-link:active { transform: none; }
}
```

## Repo conventions to follow

- Easing tokens already live at the top of `web/src/styles.css`; reuse
  `--ease-in-out` rather than introducing another curve.
- Press feedback already uses `transform: scale(.97)` and 140ms transitions at
  `web/src/styles.css:88-89`; do not change that normal-motion behavior.
- The native progress label and `aria-live` event count in
  `web/src/app/views/Replay.tsx:61-64` are part of the accessibility contract.

## Steps

1. In `web/src/app/views/Replay.tsx`, import the `CSSProperties` type from React.
2. Replace only the visible `<progress>` rendering at line 63 with the visual
   track and screen-reader-only native progress shown above.
3. In `web/src/styles.css`, remove the three native progress presentation rules
   at lines 210-212 and add the exact transform-based rules under
   `.replay-progress`.
4. Move the five existing hover declarations into the exact pointer query shown
   above without changing their colors.
5. Replace the global reduced-motion reset with the scoped rules shown above.
6. Do not add a motion dependency; this interaction is deterministic CSS.

## Boundaries

- Do NOT change replay timing, event ordering, probability values, actions, or
  trace state.
- Do NOT modify graph focus transitions; they already communicate a meaningful
  state change using paint-only stroke and fill properties.
- Do NOT add route entrance animations, staggered content, spring motion, or
  decorative movement.
- Do NOT change evidence copy or data files.
- If these code locations drift from the excerpts, stop and report instead of
  improvising.

## Verification

- **Mechanical**: from `web/`, run the repository-bundled Node 24 runtime for
  TypeScript, ESLint, Vitest, Vite build, then run
  `APAR_PYTHON=../.venv/bin/python npm run e2e`. All commands must pass.
- **Feel check**: open Replay, step rapidly through events, and confirm the fill
  retargets smoothly in 240ms without restarting or delaying event content.
- In Chrome DevTools, set animation playback to 10% and confirm the progress
  scales from its left edge without width/layout changes.
- Emulate touch and confirm navigation/control hover colors do not stick after
  taps.
- Toggle `prefers-reduced-motion`; confirm progress movement and press scaling
  are removed while color feedback remains.
- **Done when**: Playwright still announces Event 01–12 correctly, rapid stepping
  remains responsive, no layout property animates, and both pointer modes behave
  deterministically.
