import { describe, expect, it, vi } from "vitest";

import rawEvidence from "../../public/data/console-evidence.json";
import rawTrace from "../../public/data/verified-trace.json";
import { loadConsoleData } from "./loader";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json" } });
}

describe("scoring worker fallback", () => {
  it("uses the verified fixed trace when the local worker is unavailable", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(jsonResponse(rawEvidence))
      .mockResolvedValueOnce(jsonResponse({ status: "scorer_unavailable" }, 503))
      .mockResolvedValueOnce(jsonResponse(rawTrace));

    const loaded = await loadConsoleData(fetcher);

    expect(loaded.traceMode).toBe("hash_bound_verified_fallback");
    expect(loaded.trace.trace_sha256).toBe(rawTrace.trace_sha256);
  });

  it("uses a replay-verified local scorer response when available", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(jsonResponse(rawEvidence))
      .mockResolvedValueOnce(jsonResponse(rawTrace));

    const loaded = await loadConsoleData(fetcher);

    expect(loaded.traceMode).toBe("live_local_scorer");
  });
});
