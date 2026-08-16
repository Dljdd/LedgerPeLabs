# Data and API contracts

## 1. Contract principles

- Every payload is versioned.
- Every identifier is synthetic or pseudonymous.
- Event, ingestion, availability, and decision time are distinct.
- Rail-specific fields are not forced into one universal model.
- Missing, unavailable, delayed, and inapplicable are different states.
- Online features carry provenance to source event IDs.
- Candidate models cannot read labels, campaign IDs, scenario IDs, or generator metadata.

## 2. Common event envelope

```json
{
  "schema_version": "1.0.0",
  "event_id": "evt_01...",
  "event_type": "payment.authorization.requested",
  "rail": "card",
  "viewpoint": "network_with_partner_enrichment",
  "event_time": "2026-08-16T10:10:00Z",
  "ingested_at": "2026-08-16T10:10:00.025Z",
  "available_at": "2026-08-16T10:10:00.025Z",
  "decision_at": "2026-08-16T10:10:00.040Z",
  "trace_id": "trc_01...",
  "party_refs": {},
  "payment": {},
  "rail_data": {},
  "lineage": {
    "synthetic": true,
    "scenario_run_id": "run_01..."
  },
  "privacy": {
    "classification": "synthetic",
    "retention_class": "competition_demo"
  }
}
```

`scenario_run_id` is retained for audit and simulator operation but is prohibited from model features.

## 3. Card authorization contract

Required or conditionally required fields:

- Authorization and transaction identifiers.
- Pseudonymous account or token reference.
- Issuer and acquirer references.
- Merchant and merchant-category reference.
- Amount and currency.
- Merchant, issuer, and transaction countries where authorized.
- Card-present or CNP indicator.
- Entry mode.
- Token and cryptogram status.
- Authentication or 3DS result where available.
- Response outcome and reason availability time.
- Clearing, settlement, dispute, chargeback, and recovery references.

Device, IP, session, email, shipping, and identity attributes must be labeled as optional partner enrichment, not assumed network-native data.

## 4. A2A contract

- Payment and instruction identifiers.
- Sending and receiving institution references.
- Debtor and creditor account pseudonyms.
- Amount and currency.
- Initiation channel.
- Beneficiary creation and age where available.
- Name-match result and availability where authorized.
- Warning, step-up, or confirmation state.
- Authorization and settlement state.
- Scam report, freeze, recovery, and cash-out references.
- Purpose or category only when available and permitted.

## 5. Agentic-commerce request contract

```json
{
  "request_id": "apr_01...",
  "agent": {
    "agent_id": "agt_01...",
    "key_id": "key_01...",
    "signature": "base64..."
  },
  "mandate": {
    "mandate_id": "mnd_01...",
    "version": 3,
    "user_ref": "usr_01...",
    "merchant_refs": ["mer_01..."],
    "beneficiary_refs": [],
    "max_amount": "150.00",
    "currency": "USD",
    "category_scope": ["TRAVEL"],
    "issued_at": "2026-08-16T09:00:00Z",
    "expires_at": "2026-08-16T12:00:00Z"
  },
  "intent": {
    "merchant_ref": "mer_01...",
    "cart_hash": "sha256...",
    "payment_intent_hash": "sha256...",
    "amount": "120.00",
    "currency": "USD"
  },
  "credential": {
    "token_ref": "tok_01...",
    "scope": "single_merchant_single_use"
  },
  "consent_ref": "cns_01...",
  "nonce": "nonce_01...",
  "issued_at": "2026-08-16T10:09:55Z",
  "expires_at": "2026-08-16T10:14:55Z"
}
```

Competition code may use simplified local signatures, but the semantics and rejection behavior must remain explicit.

## 6. Feature contract

```yaml
feature_name: beneficiary_fan_in_24h
version: 1.0.0
applicable_rails: [a2a]
owner: feature_state
source_event_types:
  - payment.authorization.requested
entity_key: beneficiary_ref
window: 24h
aggregation: unique debtor_account_ref
availability_rule: source.available_at < current.decision_at
missing_behavior: null_with_indicator
freshness_sla_ms: 250
online: true
privacy_purpose: campaign_detection
forbidden_sources:
  - is_fraud
  - campaign_id
  - scenario_id
  - generator_branch
```

## 7. Scoring request and response

### Request

```json
{
  "request_id": "scr_01...",
  "event": {},
  "requested_models": ["champion", "challenger"],
  "decision_context": {
    "decision_owner": "issuer",
    "allowed_actions": ["approve", "challenge", "decline"]
  }
}
```

### Response

```json
{
  "request_id": "scr_01...",
  "trace_id": "trc_01...",
  "model_version": "risk-0.4.0",
  "policy_version": "policy-0.3.0",
  "integrity": {
    "status": "pass",
    "reason_codes": []
  },
  "risk": {
    "score": 0.084,
    "calibrated_band": "medium"
  },
  "recommended_action": "challenge",
  "decision_owner": "issuer",
  "reason_codes": ["BENEFICIARY_RECENTLY_ADDED"],
  "evidence_refs": ["evt_01..."],
  "feature_state": "fresh",
  "fallback_used": false,
  "latency_ms": 24
}
```

## 8. Scenario contract

```yaml
scenario_id: app-mule-personalized-v1
version: 1.0.0
threat_card_ref: threat-app-genai-004@2
rail: a2a
viewpoint: network_with_bank_enrichment
genai_capability:
  personalization: true
  translation: true
  adaptive_planning: true
attacker:
  objective: expected_net_settled_value
  query_budget: 40
  feedback: [approve, challenge, decline, realized_value]
economics:
  acquisition_cost: configured
  mule_commission: configured
  recovery_probability: configured
lifecycle:
  label_delay_days: configured
hidden_validity:
  profile: hidden-oracle-a
safety:
  synthetic_only: true
  export_level: sanitized
```

## 9. Evaluation report contract

The report shall contain:

- Run, scenario, generator, model, policy, and evaluator hashes.
- Dataset partitions and sample counts.
- Fraud prevalence and value distribution.
- Per-family and per-segment metrics.
- Operational action rates and budgets.
- Calibration and latency.
- Leakage and metamorphic tests.
- Adaptive-search ablation.
- Hidden-evaluation results.
- Failed gates and limitations.
- Human reviewer identity, decision, and timestamp.

## 10. Validation and compatibility

- Unknown major schema versions are rejected.
- Unknown optional fields are ignored but retained in raw lineage.
- Missing required fields are rejected with a stable reason.
- Rail-inapplicable fields cannot become online features.
- Decimal money types are used for lifecycle and reconciliation.
- Event IDs are globally unique inside a run.
- All timestamps include timezone or use UTC.

## 11. Availability matrix template

| Field or feature | Rail | Source owner | Native or enriched | Available by decision? | Freshness | Missing behavior |
|---|---|---|---|---|---|---|
| Amount and currency | All | Payment message | Native | Yes | Request-time | Reject if missing |
| Device ID | Card/A2A | Partner | Enriched | Sometimes | Declared | Null plus indicator |
| Beneficiary age | A2A | Sending bank | Enriched | Sometimes | Declared | Null plus indicator |
| Chargeback label | Card | Issuer/acquirer | Delayed | No | Days or months | Evaluation/training only |
| Campaign ID | Synthetic range | Simulator | Audit only | Prohibited | N/A | Never a feature |

Every implemented feature must have a completed row.

