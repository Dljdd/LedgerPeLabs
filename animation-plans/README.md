# Console motion plans

| Plan | Title | Severity | Status |
| --- | --- | --- | --- |
| [001](001-tighten-console-motion.md) | Tighten replay and interaction motion | MEDIUM | TODO |

## Recommended execution order

1. Execute plan 001 as a single bounded change. It has no dependency on other
   plans and must not alter the evidence or replay state contracts.

The current motion already avoids route entrances, decorative loops, bounce,
`transition: all`, `ease-in`, and `scale(0)`. No additional animation work is
recommended for this institutional console.
