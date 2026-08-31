import { parseEvidence, parseTrace } from "./evidence";
import type { ConsoleEvidence, TraceMode, VerifiedTrace } from "./types";

type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export interface LoadedConsoleData {
  evidence: ConsoleEvidence;
  trace: VerifiedTrace;
  traceMode: TraceMode;
}

async function readJson(response: Response, label: string): Promise<unknown> {
  if (!response.ok) throw new Error(`${label} request failed: ${response.status}`);
  return response.json() as Promise<unknown>;
}

export async function loadConsoleData(fetcher: Fetcher = fetch): Promise<LoadedConsoleData> {
  const evidence = parseEvidence(
    await readJson(await fetcher("/data/console-evidence.json"), "evidence"),
  );
  try {
    const liveResponse = await fetcher("/api/score", {
      method: "POST",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(5000),
    });
    return {
      evidence,
      trace: parseTrace(await readJson(liveResponse, "local scorer"), evidence),
      traceMode: "live_local_scorer",
    };
  } catch {
    const fallback = await readJson(
      await fetcher("/data/verified-trace.json"),
      "verified fallback trace",
    );
    return {
      evidence,
      trace: parseTrace(fallback, evidence),
      traceMode: "hash_bound_verified_fallback",
    };
  }
}
