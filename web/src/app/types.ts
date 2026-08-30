export type RoutePath =
  | "/overview"
  | "/scenario"
  | "/replay"
  | "/investigation"
  | "/defenses"
  | "/assurance";

export type TraceMode = "live_local_scorer" | "hash_bound_verified_fallback";

export interface ConsoleEvidence {
  schema_version: "apar-console-evidence/1";
  document_sha256: string;
  threat: {
    title: string;
    threat_id: string;
    family: string;
    rails: string[];
    channels: string[];
    safety_class: string;
    confidence: number;
    status: string;
    implementation_status: string;
    attacker_objective: string;
    genai_capability: {
      iteration_speed: boolean;
      personalization: boolean;
    };
    default_config: {
      query_budget: number;
      duration_hours: number;
      seed: number;
      event_ordering: string;
      viewpoint: string;
      attacker_mode: string;
      export_level: string;
      campaign_stages: { stage_id: string; description: string }[];
    };
    evidence: ThreatEvidence[];
  };
  scenario_context: {
    campaign_id: string;
    family: string;
    seed: number;
    synthetic: boolean;
    payment_count: number;
    value_total: string;
    settled_value: string;
    ledger_conserved: boolean;
    motif_signature: string;
    schedule_sha256: string;
    case_grouping: {
      case_id: string;
      event_count: number;
      estimated_analyst_minutes: { status: string };
    };
    graph: {
      graph_sha256: string;
      nodes: GraphNode[];
      edges: GraphEdge[];
    };
  };
  portable: {
    arm: "ensemble_with_graph";
    authoritative: boolean;
    accepted_capacity_evidence: boolean;
    demo_only: boolean;
    bundle_manifest_sha256: string;
    source_checkpoint_manifest_sha256: string;
    arm_spec_sha256: string;
    threshold_digest: string;
    thresholds: Record<string, number>;
    feature_names: string[];
    records: PortableRecord[];
  };
  recovered: {
    qualifier: "Recovered diagnostic evidence — non-authoritative";
    authoritative: false;
    accepted_capacity_evidence: false;
    official_chain_status: string;
    first_missing_official_stage: string;
    readiness: {
      status: string;
      evaluated_arm: string;
      readiness_sha256: string;
      gates: Gate[];
      qualifying_controls: [string, boolean, string][];
    };
    arms: RecoveredArm[];
    failed_gates: Gate[];
    source_artifact_sha256: string;
    source_receipt_sha256: string;
    verification_sha256: string;
  };
  trust_proof: {
    implementation: string;
    test_evidence: string;
    test_evidence_sha256: string;
    separate_from_model_prediction: boolean;
    checks: TrustCheck[];
  };
  copy_boundary: {
    evidence_seed: number;
    kaggle_locked_successor_run: boolean;
    local_locked_attempt: string;
    no_candidate_manifest_chunks_or_judge_summary: boolean;
    published_successful_seed_2404_result: boolean;
    retry_permitted: boolean;
  };
}

export interface VerifiedTrace {
  schema_version: "apar-sentinel-v5-portable-demo-trace/1";
  bundle_manifest_sha256: string;
  model_load_ms: number;
  scoring_wall_ms: number;
  replay_verified: true;
  trace_sha256: string;
  traces: TraceRecord[];
}

export interface TraceRecord {
  event_id: string;
  arm: "ensemble_with_graph";
  calibrated_probability: number;
  final_action: string;
  model_action: string;
  latency_ms: number;
  disagreement: number;
  reason_codes: string[];
  raw_member_scores: number[];
  calibrated_member_scores: number[];
  replay_probability_abs_error: number;
  presentation_ground_truth: PortableRecord["post_event_truth"];
}

export interface GraphNode {
  id: string;
  account_id: string;
  label: string;
  role: string;
  country: string;
  illicit: boolean;
  x: number;
  y: number;
}

export interface GraphEdge {
  payment_id: string;
  source: string;
  target: string;
  source_account: string;
  target_account: string;
  amount: string;
  currency: string;
  event_time: string;
  stage: string;
  cumulative_attempted_value: string;
}

export interface PortableRecord {
  event_id: string;
  model_input: { features: number[] };
  accepted_checkpoint_evidence: {
    probability: number;
    action: string;
    probability_action: string;
    model_raw_scores: number[];
    model_calibrated_scores: number[];
    rule_components: string[];
    rule_score: number | null;
    novelty_score: number | null;
    trust_routed: boolean;
  };
  post_event_truth: {
    label: number;
    family: string;
    rail: string;
    amount: number;
    currency: string;
  };
}

export interface Gate {
  metric: string;
  passed: boolean;
  target: number | boolean | null;
  point?: number;
  lower?: number | null;
  upper?: number | null;
  gate_sha256?: string;
  source_sha256: string;
}

export interface RecoveredArm {
  arm: string;
  aggregate: Record<string, number>;
  complete_metrics_sha256: string;
  deterministic_result_sha256: string;
  receipt_sha256: string;
  support_sha256: string;
}

export interface TrustCheck {
  check: string;
  status: string;
  evidence: string;
}

export interface ThreatEvidence {
  evidence_id: string;
  publisher: string;
  claim: string;
  direct_source_url: string;
  quality_grade: string;
  is_project_inference: boolean;
}
