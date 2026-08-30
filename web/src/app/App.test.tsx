import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import rawEvidence from "../../public/data/console-evidence.json";
import rawTrace from "../../public/data/verified-trace.json";
import { App } from "./App";
import { parseEvidence, parseTrace } from "./evidence";

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

    await user.click(screen.getByRole("button", { name: /step forward/i }));
    expect(screen.getByText("Event 02 / 12")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /reset replay/i }));
    expect(screen.getByText("Event 01 / 12")).toBeVisible();
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
});
