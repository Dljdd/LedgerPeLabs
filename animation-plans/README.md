# Console motion plans

| Plan | Title | Severity | Status |
| --- | --- | --- | --- |
| [001](001-tighten-console-motion.md) | Tighten replay and interaction motion | MEDIUM | DONE |
| [002](002-deepen-campaign-playback.md) | Deepen campaign playback motion | HIGH | DONE |
| [003](003-add-route-evidence-interactions.md) | Add route-specific evidence interactions | MEDIUM | DONE |
| [004](004-consolidate-motion-accessibility.md) | Consolidate motion accessibility | MEDIUM | DONE |

## Recommended execution order

1. Plan 001 is complete and remains the baseline.
2. Execute plan 002 to deepen the bound campaign progression and add the
   JavaScript reduced-motion branch.
3. Execute plan 003 after plan 002 so route-specific state wrappers reuse the
   same motion conventions.
4. Execute plan 004 last to consolidate tokens and verify the final
   reduced-motion behavior across the combined implementation.

Plans 002–004 intentionally avoid route entrances, decorative ambient loops,
bounce, `transition: all`, `ease-in`, and `scale(0)`. Every new visual state is
bound to existing repository evidence or a clearly labeled static product
illustration.
