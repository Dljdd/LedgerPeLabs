import type { ConsoleEvidence, VerifiedTrace } from "./types";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function parseEvidence(value: unknown): ConsoleEvidence {
  if (!isObject(value) || value.schema_version !== "apar-console-evidence/1") {
    throw new Error("unsupported console evidence schema");
  }
  if (!isObject(value.portable)) {
    throw new Error("portable evidence is missing");
  }
  if (value.portable.arm !== "ensemble_with_graph") {
    throw new Error("portable arm must be ensemble_with_graph");
  }
  if (value.portable.authoritative !== false || value.portable.accepted_capacity_evidence !== false) {
    throw new Error("portable evidence boundary differs");
  }
  if (!isObject(value.recovered)) {
    throw new Error("recovered evidence is missing");
  }
  if (value.recovered.qualifier !== "Recovered diagnostic evidence — non-authoritative") {
    throw new Error("recovered evidence qualifier differs");
  }
  if (value.recovered.authoritative !== false || value.recovered.accepted_capacity_evidence !== false) {
    throw new Error("recovered evidence boundary differs");
  }
  return value as unknown as ConsoleEvidence;
}

export function parseTrace(value: unknown, evidence: ConsoleEvidence): VerifiedTrace {
  if (!isObject(value) || value.schema_version !== "apar-sentinel-v5-portable-demo-trace/1") {
    throw new Error("unsupported fixed trace schema");
  }
  if (value.replay_verified !== true) {
    throw new Error("fixed trace is not replay verified");
  }
  if (value.bundle_manifest_sha256 !== evidence.portable.bundle_manifest_sha256) {
    throw new Error("fixed trace bundle binding differs");
  }
  if (!Array.isArray(value.traces) || value.traces.length !== evidence.portable.records.length) {
    throw new Error("fixed trace event count differs");
  }
  for (const [index, record] of value.traces.entries()) {
    if (!isObject(record) || record.arm !== "ensemble_with_graph") {
      throw new Error(`fixed trace arm differs at event ${index + 1}`);
    }
    const accepted = evidence.portable.records[index];
    if (
      !accepted ||
      record.event_id !== accepted.event_id ||
      record.calibrated_probability !== accepted.accepted_checkpoint_evidence.probability ||
      record.final_action !== accepted.accepted_checkpoint_evidence.action
    ) {
      throw new Error(`fixed trace checkpoint evidence differs at event ${index + 1}`);
    }
  }
  return value as unknown as VerifiedTrace;
}
