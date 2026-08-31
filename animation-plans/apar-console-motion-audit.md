# APAR console motion audit

Audit date: 2026-08-30

Scope: `web/src/styles.css`, the replay controller, the investigation graph, and route navigation.

## Verdict

The original restrained motion baseline was retained and expanded after the
approved route-visual pass. Motion remains limited to replay progression,
user-selected evidence focus, native focus/press feedback, and meaningful
state continuity. There are no route entrance animations, ambient decorative
loops, hover lifts, scale-from-zero transitions, `transition: all`
declarations, or content-hidden initial states.

## Timing and easing

- Standard state transitions use `cubic-bezier(.23, 1, .32, 1)` at 180–220 ms.
- Replay and evidence progress use `cubic-bezier(.77, 0, .175, 1)` at 240 ms;
  the explanatory campaign transfer uses the same curve for 720 ms within the
  existing 900 ms event interval.
- Press feedback uses `scale(.97)` at 140 ms.
- Reduced-motion mode disables positional travel and campaign autoplay while
  retaining short opacity and color feedback.

## Findings

1. Replay progress, active endpoints, and the transform-only transfer marker
   explain genuine ordered campaign movement and should remain.
2. Graph edge/node focus and route-specific selection transitions are
   meaningful investigation-state feedback and should remain.
3. Navigation, button, selector, and focus feedback is brief and
   property-specific.
4. Overview trace, Scenario stage, Investigation value, Defenses arm, and
   Assurance lineage interactions remain user-driven and evidence-bound.
5. Additional ambient motion would weaken the institutional assurance posture.

## Follow-up

Retain the current motion system. Future changes should keep truth/model
separation visible at rest, preserve explicit reduced-motion stepping, and
must not delay the presentation of evidence.
