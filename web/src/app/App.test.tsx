import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import rawEvidence from "../../public/data/console-evidence.json";
import rawTrace from "../../public/data/verified-trace.json";
import { App } from "./App";
import { parseEvidence, parseTrace } from "./evidence";

const originalMatchMedia = typeof window.matchMedia === "function"
  ? window.matchMedia.bind(window)
  : undefined;

function mockReducedMotion(matches: boolean) {
  let currentMatches = matches;
  let changeListener: ((event: MediaQueryListEvent) => void) | undefined;
  const mediaQuery = {
    addEventListener: vi.fn((_type: string, listener: (event: MediaQueryListEvent) => void) => {
      changeListener = listener;
    }),
    dispatchEvent: vi.fn(),
    get matches() {
      return currentMatches;
    },
    media: "(prefers-reduced-motion: reduce)",
    onchange: null,
    removeEventListener: vi.fn((_type: string, listener: (event: MediaQueryListEvent) => void) => {
      if (changeListener === listener) changeListener = undefined;
    }),
  } as unknown as MediaQueryList;
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn().mockReturnValue(mediaQuery),
  });
  return {
    setMatches(nextMatches: boolean) {
      currentMatches = nextMatches;
      changeListener?.({ matches: nextMatches } as MediaQueryListEvent);
    },
  };
}

afterEach(() => {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: originalMatchMedia,
  });
});

describe("console evidence boundary", () => {
  it("rejects a portable arm mislabeled as full_sentinel", () => {
    const malformed = structuredClone(rawEvidence);
    malformed.portable.arm = "full_sentinel";

    expect(() => parseEvidence(malformed)).toThrow(
      "portable arm must be ensemble_with_graph",
    );
  });

  it("rejects recovered metrics without the required qualifier", () => {
    const malformed = structuredClone(rawEvidence);
    malformed.recovered.qualifier = "Recovered metrics";

    expect(() => parseEvidence(malformed)).toThrow(
      "recovered evidence qualifier differs",
    );
  });

  it("rejects a fixed trace that is not replay verified", () => {
    const malformed = structuredClone(rawTrace);
    malformed.replay_verified = false;

    expect(() => parseTrace(malformed, parseEvidence(rawEvidence))).toThrow(
      "fixed trace is not replay verified",
    );
  });
});

describe("native route boundary", () => {
  afterEach(() => history.replaceState({}, "", "/"));

  it("redirects an unknown route to overview", () => {
    history.replaceState({}, "", "/unknown");

    render(
      <App
        evidence={parseEvidence(rawEvidence)}
        trace={parseTrace(rawTrace, parseEvidence(rawEvidence))}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /adaptive payment assurance/i }),
    ).toBeVisible();
    expect(location.pathname).toBe("/overview");
  });

  it("moves through the replay and restores the canonical first event", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /replay/i }));
    expect(screen.getByRole("heading", { name: /verified decision replay/i })).toBeVisible();
    expect(screen.getByText("Event 01 / 12")).toBeVisible();
    expect(screen.getByRole("region", { name: /model evidence/i })).toBeVisible();
    expect(screen.getByRole("region", { name: /post-event truth/i })).toBeVisible();
    expect(screen.getByRole("img", { name: /calibrated risk 100.0%.*challenge 10.0%.*review 50.4%.*decline 100.0%/i })).toBeVisible();
    expect(screen.getByRole("img", { name: /14 genuine scenario entities and 10 ordered payment edges/i })).toBeVisible();
    expect(screen.getByText(/no payment-to-trace record mapping asserted/i)).toBeVisible();

    await user.click(screen.getByRole("button", { name: /step forward/i }));
    expect(screen.getByText("Event 02 / 12")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /reset replay/i }));
    expect(screen.getByText("Event 01 / 12")).toBeVisible();
  });

  it("keeps scenario playback independent from portable event selection", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /replay/i }));
    const campaign = within(screen.getByRole("list", { name: /ordered campaign payments/i })).getAllByRole("button");
    await user.click(campaign[4]!);

    expect(campaign[4]).toHaveAttribute("aria-current", "step");
    expect(screen.getByText("Event 01 / 12")).toBeVisible();

    const portable = within(screen.getByRole("group", { name: /select independent portable trace event/i })).getAllByRole("button");
    await user.click(portable[1]!);
    expect(portable[1]).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Event 02 / 12")).toBeVisible();
  });

  it("exposes normal campaign play, pause, and replay controls", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /replay/i }));
    await user.click(screen.getByRole("button", { name: "Play campaign" }));
    expect(screen.getByRole("button", { name: "Pause campaign" })).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "Pause campaign" }));
    expect(screen.getByRole("button", { name: "Play campaign" })).toHaveAttribute("aria-pressed", "false");

    const payments = within(screen.getByRole("list", { name: /ordered campaign payments/i })).getAllByRole("button");
    await user.click(payments.at(-1)!);
    expect(screen.getByRole("button", { name: "Replay campaign" })).toBeVisible();
  });

  it("steps and resets campaign playback when reduced motion is requested", async () => {
    mockReducedMotion(true);
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /replay/i }));
    const stepCampaign = screen.getByRole("button", { name: "Step campaign" });
    expect(stepCampaign).not.toHaveAttribute("aria-pressed");
    await user.click(stepCampaign);
    expect(screen.getByText("SCENARIO PAYMENT 02")).toBeVisible();

    const payments = within(screen.getByRole("list", { name: /ordered campaign payments/i })).getAllByRole("button");
    await user.click(payments.at(-1)!);
    const resetCampaign = screen.getByRole("button", { name: "Reset campaign" });
    expect(resetCampaign).not.toHaveAttribute("aria-pressed");
    await user.click(resetCampaign);
    expect(screen.getByText("SCENARIO PAYMENT 01")).toBeVisible();
  });

  it("cancels active campaign playback when reduced motion turns on", async () => {
    const motionPreference = mockReducedMotion(false);
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /replay/i }));
    await user.click(screen.getByRole("button", { name: "Play campaign" }));
    expect(screen.getByRole("button", { name: "Pause campaign" })).toHaveAttribute("aria-pressed", "true");

    act(() => motionPreference.setMatches(true));
    expect(screen.getByRole("button", { name: "Step campaign" })).not.toHaveAttribute("aria-pressed");

    act(() => motionPreference.setMatches(false));
    expect(screen.getByRole("button", { name: "Play campaign" })).toHaveAttribute("aria-pressed", "false");
  });

  it("keeps recovered metrics and integrity proof in their own claim lanes", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    const primary = screen.getByRole("navigation", { name: "Primary" });
    await user.click(within(primary).getByRole("link", { name: /defenses/i }));
    expect(screen.getAllByText("ensemble_with_graph").length).toBeGreaterThan(0);
    expect(screen.getByText("Recovered diagnostic evidence — non-authoritative")).toBeVisible();

    await user.click(within(primary).getByRole("link", { name: /assurance/i }));
    expect(screen.getByText(/no Kaggle locked-successor\/seed-2404 chain was run/i)).toBeVisible();
    expect(screen.getByRole("region", { name: /agentic integrity proof/i })).toBeVisible();
  });

  it("binds the investigation first alert to a curated APP replay record", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /investigation/i }));

    expect(screen.getByText("First curated APP intervention")).toBeVisible();
    expect(screen.getByText(/event 02 · 100.0% calibrated/i)).toBeVisible();
    expect(screen.getByRole("img", { name: /14 linked entities and 10 directional payment edges; edge weight represents payment amount/i })).toBeVisible();
    expect(screen.getByText("Arrow = payment direction")).toBeVisible();
    expect(screen.getByText("Weight = amount")).toBeVisible();
  });

  it("renders the overview footprint from the ordered verified trace", () => {
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    expect(screen.getByRole("img", { name: /12 ordered calibrated decisions: 6 decline hold, 2 review hold, 1 challenge, and 3 approve/i })).toBeVisible();
  });

  it("focuses a model-only overview trace event", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    const selector = screen.getByRole("group", { name: /select curated trace event/i });
    const eventFour = within(selector).getByRole("button", { name: /event 4, 70.5%, review hold/i });
    await user.click(eventFour);

    expect(eventFour).toHaveAttribute("aria-pressed", "true");
    const readout = screen.getByRole("status", { name: /focused trace event/i });
    expect(within(readout).getByText("Event 04")).toBeVisible();
    expect(within(readout).getByText("70.5%")).toBeVisible();
    expect(within(readout).getByText("Review Hold")).toBeVisible();
  });

  it("selects configured campaign stages without autoplay", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /scenario/i }));
    const stages = screen.getByRole("region", { name: "Campaign stages" });
    const transfer = within(stages).getByRole("button", { name: /focus campaign stage 2 transfer/i });
    await user.click(transfer);

    expect(transfer).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelector(".motif-card")).toHaveAttribute("data-stage", "1");
    expect(screen.getByText("Configured stage 02 · transfer")).toBeVisible();
  });

  it("focuses defense arms with the graph ensemble selected by default", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /defenses/i }));
    const architecture = screen.getByRole("region", { name: "Defense architecture" });
    const champion = within(architecture).getByRole("button", { name: "Focus ensemble_with_graph architecture" });
    expect(champion).toHaveAttribute("aria-pressed", "true");

    const rules = within(architecture).getByRole("button", { name: "Focus rules_only architecture" });
    await user.click(rules);
    expect(rules).toHaveAttribute("aria-pressed", "true");
    const focus = screen.getByRole("status", { name: "Focused architecture" });
    expect(within(focus).getByText("rules_only")).toBeVisible();
    expect(within(focus).getByText(evidence.recovered.arms[0]!.deterministic_result_sha256)).toBeVisible();
  });

  it("reveals full lineage hashes through keyboard-ready controls", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /assurance/i }));
    const lineage = screen.getByRole("list", { name: "Evidence lineage artifacts" });
    const checkpoint = within(lineage).getByRole("button", { name: "Inspect Stage 30 source checkpoint lineage" });
    await user.click(checkpoint);

    expect(checkpoint).toHaveAttribute("aria-pressed", "true");
    const focus = screen.getByRole("status", { name: "Selected lineage artifact" });
    expect(within(focus).getByText(evidence.portable.source_checkpoint_manifest_sha256)).toBeVisible();
  });

  it("renders connected-value bars from the selected entity's bound edges", async () => {
    const user = userEvent.setup();
    const evidence = parseEvidence(rawEvidence);
    render(<App evidence={evidence} trace={parseTrace(rawTrace, evidence)} />);

    await user.click(within(screen.getByRole("navigation", { name: "Primary" })).getByRole("link", { name: /investigation/i }));
    const inspector = screen.getByRole("complementary", { name: /selected entity details/i });
    const linkedRows = inspector.querySelectorAll(".linked-events > div");
    const valueBars = inspector.querySelectorAll(".linked-value-bar");

    expect(valueBars.length).toBe(linkedRows.length);
    expect(valueBars.length).toBeGreaterThan(0);
    expect((valueBars[0] as HTMLElement).style.getPropertyValue("--linked-value")).toMatch(/^(0(?:\.\d+)?|1)$/);
  });
});
