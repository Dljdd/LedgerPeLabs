# APAR console motion audit

Audit date: 2026-08-30

Scope: `web/src/styles.css`, the replay controller, the investigation graph, and route navigation.

## Verdict

The console uses motion only for replay progression, graph focus, native focus/press feedback, and route-navigation state. There are no entrance animations, looping decorative effects, hover lifts, scale-from-zero transitions, `transition: all` declarations, or content-hidden initial states.

## Timing and easing

- Standard state transitions use `cubic-bezier(.23, 1, .32, 1)` at 180–220 ms.
- Replay progress uses `cubic-bezier(.77, 0, .175, 1)` at 500 ms because it represents ordered state progression.
- Press feedback uses `scale(.97)` at 140 ms.
- Reduced-motion mode collapses transition and animation durations and disables smooth scrolling.

## Findings

1. Replay progress width is a meaningful state transition and should remain.
2. Graph edge/node focus transitions are meaningful investigation-state feedback and should remain.
3. Navigation, button, and focus feedback is brief and property-specific.
4. No additional animation would improve comprehension; adding ambient motion would weaken the institutional assurance posture.

## Follow-up

Retain the current motion system. Future changes should keep truth/model separation visible at rest and must not delay the presentation of evidence.
