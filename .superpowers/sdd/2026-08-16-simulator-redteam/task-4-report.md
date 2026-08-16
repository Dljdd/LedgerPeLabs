# Task 4 report: agentic trust verifier and rail

## Outcome

Implemented a deterministic, fail-closed agentic payment trust plane and an engine-native
agentic rail. A request is admitted only after the ordered identity, Ed25519 signature,
approved-mandate, amount, currency, payee, cart, time-validity, nonce, and receipt-chain
checks pass. The risk scorer is unreachable on integrity failure, and ledger accounts are
bound to the signed mandate user and signed request payee before any value can move.

## RED evidence

The first focused run was captured before production modules existed:

```text
.venv/bin/python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q
ERROR tests/trust/test_verifier.py - ModuleNotFoundError: No module named 'apar.trust'
ERROR tests/simulator/test_agentic_rail.py - ModuleNotFoundError: No module named
'apar.simulator.rails.agentic'
2 errors during collection
```

Subsequent TDD cycles also demonstrated the intended failures before their fixes:

- Two future-issued request/mandate cases were incorrectly allowed before time-validity
  admission was added.
- A well-shaped state record for an unregistered agent was accepted before state ownership
  was bound to the configured identity registry.
- Two ledger-account substitution cases were accepted before command accounts were bound
  to signed request claims.
- The payment actor identifier could be used as the debit account before the debit was bound
  specifically to the mandate's user reference.
- A syntactically valid but unreproducible receipt-state record was accepted before state
  records retained canonical request/signature digests and replayed the receipt derivation.

## GREEN evidence

```text
.venv/bin/python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q
41 passed
```

The focused suite covers:

- Real Ed25519 verification using `cryptography`, including unknown identity, malformed
  signature, and invalid signature cases.
- Mandate escalation, amount, currency, payee, cart, expiry boundary, future issuance,
  nonce replay, and broken receipt-chain rejection.
- Exact check-order precedence, including expiry before replay and replay before receipt
  chain.
- Immutable canonical mandate, request, and receipt records; exact scalar, Decimal, digest,
  UUID, and UTC boundaries; subclass and non-finite rejection.
- Failed-check non-mutation, state round-trip, unknown-agent state rejection, duplicate nonce
  rejection, reproducible receipt derivation, and broken state-chain rejection with
  `AGENTIC_STATE_CORRUPT`.
- Integrity failure before scorer callback, event emission, or ledger posting; scorer called
  exactly once after a pass; approved and risk-declined paths; replay denial; deterministic
  events and receipts; forged/unknown commands; signed payer/payee-to-ledger binding; and
  value conservation.

## Contract extensions

`ReasonCode` now includes only the exact integrity values emitted by this task:

- `AGENT_IDENTITY_MISMATCH`
- `SIGNATURE_INVALID`
- `AMOUNT_LIMIT_EXCEEDED`
- `CURRENCY_MISMATCH`
- `PAYEE_BINDING_MISMATCH`
- `CART_HASH_MISMATCH`
- `MANDATE_EXPIRED`
- `NONCE_REPLAY`
- `RECEIPT_CHAIN_BROKEN`

No `EventKind` extension was needed. The agentic rail reuses `AUTHORIZATION` and
`AUTHORIZATION_DECLINED` and records integrity/action evidence in `rail_data`.

## Design and tradeoffs

- Only public Ed25519 keys enter production configuration. Test private keys are fixed,
  synthetic fixtures and are never stored in receipts, events, or run state.
- Receipt hashes are deterministic SHA-256 audit commitments over the canonical signed
  request, signature digest, and prior receipt. They intentionally look like receipts but
  are not signatures, payment tokens, credentials, or claims of external attestation.
- Nonce and receipt history is a closed, tuple-ordered, engine-owned state value. It detects
  malformed or accidentally rewritten state and enforces replay/chain semantics. As with the
  other rail histories, it is not a security boundary against a trusted in-process engine
  owner that can replace all state and code.
- A single `MANDATE_EXPIRED` time-validity code covers both not-yet-valid and expired mandate
  or request windows. This preserves the approved exact reason-code surface and deterministic
  check order.
- The initial prototype mandate bound one payee and one cart exactly. Fix round 1 below expands
  that boundary to explicit merchant, category/product, credential, consent, authentication,
  payment-intent, and outcome bindings without hidden permissive defaults.
- A passing integrity check consumes its nonce even when the downstream scorer challenges or
  declines. Integrity failures never consume a nonce or revise receipt state.

## Files

- `src/apar/trust/__init__.py`
- `src/apar/trust/verifier.py`
- `src/apar/simulator/rails/agentic.py`
- `src/apar/simulator/rails/__init__.py`
- `src/apar/contracts/decisions.py`
- `tests/trust/test_verifier.py`
- `tests/simulator/test_agentic_rail.py`

`validation_spike/` remains unchanged.

## Fix round 1/5

### Review findings addressed

1. Expanded the approved agentic contract rather than deferring it. The immutable mandate
   and request now bind merchant separately from payee/beneficiary, permitted category and
   product scopes, payment-intent hash, synthetic credential reference and scope, consent,
   authentication/step-up requirement, and the execution outcome receipt. These fields are
   public so Task 5 can construct every declared agentic-intent attack through normal command
   interfaces.
2. Split verification into non-consuming `preview` and final `commit`. The adapter revokes a
   preview if the scorer raises or returns anything other than an exact `Action`; nonce and
   receipt state remain unchanged and a later retry succeeds. A preview issued by another
   verifier or abandoned after scoring failure cannot be committed.
3. Added exact request-within-mandate temporal nesting: mandate issuance may equal request
   creation, request expiry may equal mandate expiry, and either request boundary outside the
   mandate fails with `MANDATE_TIME_SCOPE_VIOLATION` before runtime expiry/replay checks.
4. Added `AUTHENTICATION_CHALLENGE` for `Action.CHALLENGE`; approval and decline retain their
   existing event kinds. Event evidence now carries the committed receipt outcome.
5. Failure receipts now bind the canonical request digest and signature digest as well as
   reason, prior receipt, and rejected outcome, preventing same-ID/reason collisions across
   distinct failed requests.
6. Corrected the state-corruption test to submit a fully shaped seven-field record. It reaches
   the receipt-reproduction check and asserts that exact failure cause.

### Additional exact reason codes

- `MERCHANT_BINDING_MISMATCH`
- `CATEGORY_SCOPE_VIOLATION`
- `PRODUCT_SCOPE_VIOLATION`
- `PAYMENT_INTENT_HASH_MISMATCH`
- `CREDENTIAL_BINDING_MISMATCH`
- `TOKEN_SCOPE_VIOLATION`
- `CONSENT_BINDING_MISMATCH`
- `AUTHENTICATION_REQUIRED`
- `MANDATE_TIME_SCOPE_VIOLATION`
- `EXECUTION_RECEIPT_MISMATCH`

### RED evidence

- New contract tests initially failed collection because authentication and receipt-outcome
  types did not exist.
- A preview produced by one verifier was accepted by a different verifier before issuance
  ownership was recorded.
- An abandoned preview captured during a scorer exception remained committable before explicit
  preview revocation.
- The reviewed expansion tests failed before production fields and checks were implemented;
  the focused suite then exposed the intended future-mandate reason-precedence adjustment.

### GREEN evidence

```text
.venv/bin/python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q
63 passed

.venv/bin/python -m pytest -q
451 passed

.venv/bin/ruff check src tests
All checks passed!

.venv/bin/mypy --strict src
Success: no issues found in 33 source files

.venv/bin/mypy src
Success: no issues found in 33 source files

.venv/bin/python scripts/verify_g0.py
G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store
```

`git diff --check` passed and `validation_spike/` is unchanged from `8d6a897`.

### Tradeoffs

- Category and product scopes are non-empty, unique, sorted tuples. This intentionally keeps
  the public attack surface deterministic instead of accepting arbitrary container semantics.
- Credential identifiers and scopes are synthetic references only. No private key, payment
  token, PAN, or usable credential enters the receipt, event, or run state.
- Preview issuance is verifier-instance-local and callback-sequential. This matches the
  deterministic single-thread simulator; a distributed implementation would replace it with
  a transactional nonce/receipt service.
- Standalone `verify` retains its original replay behavior by performing preview plus a
  `VERIFIED` commit. The rail commits the exact downstream `APPROVE`, `CHALLENGE`, or `DECLINE`
  outcome instead.

## Fix round 2/5

### Review findings addressed

1. Receipt commitments and persisted eight-field state records now bind the exact agent ID,
   nonce, authentication-evidence reference, canonical request digest, signature digest,
   prior receipt, and outcome. State loading recomputes that commitment, rejects nonce or
   agent relabeling, duplicate nonce/evidence consumption, broken per-agent chains, and
   malformed records, while legitimate interleaved multi-agent chains round-trip.
2. Rail execution now uses verifier `preview`, `prepare_commit`, `projected_state`, and
   `apply_commit` boundaries. The event, projected closed state, and final truthful receipt
   are constructed before effects; an approval posts value before the outcome is committed.
   Overdraft or posting failure discards the prepared plan and leaves verifier records,
   ledger entries, engine events, and entity state unchanged. Fresh-engine retries through
   the same direct verifier succeed. Challenge and decline commit truthful non-executed
   outcomes without a ledger post.
3. Mandates now bind trusted user/actor and beneficiary/counterparty entity UUIDs in addition
   to account, merchant, and payee identifiers. The verifier rejects graph-identity
   substitution before scoring or value/state effects. Events expose the verified UUIDs as
   actor/counterparty and the account/merchant/payee/entity bindings in `party_refs`.
4. The agent no longer self-attests step-up success. The signed request carries only a
   synthetic evidence reference; a separately configured immutable
   `AuthenticationEvidence` registry binds the evidence to agent, user, mandate, nonce,
   payment-intent hash, request ID, exact verified outcome, and closed UTC validity window.
   Missing, substituted, unbound, expired, or replayed evidence fails with a stable reason.
   Mandates that require no step-up need no registry record, while their signed correlation
   reference remains receipt-bound and single-use.
5. Finalization takes an exact UTC `now`, rejects authentication/request/mandate expiry and
   time reversal after preview, and consumes every commit attempt once. A new preview attempt
   revokes prior ephemeral preview/plan work even when the new request rejects; scorer errors,
   malformed scorer values, posting failures, and explicit plan discards likewise leave no
   retained ephemeral transaction.

### Exact contract changes

New emitted integrity reasons:

- `AUTHORITY_IDENTITY_MISMATCH`
- `AUTHENTICATION_EVIDENCE_MISSING`
- `AUTHENTICATION_EVIDENCE_MISMATCH`
- `AUTHENTICATION_EVIDENCE_EXPIRED`
- `AUTHENTICATION_EVIDENCE_REPLAY`

The obsolete self-attested `AuthenticationState` and its no-longer-emitted
`AUTHENTICATION_REQUIRED` reason were removed. No round-2 `EventKind` extension was needed;
the semantic `AUTHENTICATION_CHALLENGE` kind added in round 1 remains the exact challenge
event.

### RED evidence

The first round-2 focused run, before production edits, failed collection because the new
trusted evidence contracts did not exist:

```text
ImportError: cannot import name 'AuthenticationEvidence' from 'apar.trust.verifier'
ImportError: cannot import name 'AuthenticationEvidence' from 'apar.trust.verifier'
2 errors during collection
```

Subsequent focused RED cycles demonstrated the concrete defects before each fix:

- existing receipt commitments accepted nonce/agent relabeling because those fields were not
  in the digest;
- an approved outcome was committed before an overdraft/posting exception;
- arbitrary signed actor/counterparty UUIDs reached the event graph;
- self-attested authentication had no independently owned evidence;
- a tampered or foreign preview and expired finalization needed single-use handling;
- a rejected new preview left an earlier preview committable;
- malformed commit attempts left their preview reusable;
- commit timestamps earlier than preview issuance were accepted; and
- no-step-up histories could not reload without a registry entry before the persisted-state
  rule was corrected.

### GREEN evidence

```text
.venv/bin/python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q
80 passed

.venv/bin/python -m pytest -q
468 passed

.venv/bin/ruff check src tests
All checks passed!

.venv/bin/mypy --strict src
Success: no issues found in 33 source files

.venv/bin/mypy src
Success: no issues found in 33 source files

.venv/bin/python scripts/verify_g0.py
G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store
```

`git diff --check` passed and `validation_spike/` is unchanged from `1024f01`.

### Tradeoffs and files

- The prepare/apply transaction is intentionally verifier-instance-local and single-threaded,
  matching the simulator engine. The adapter applies a verifier-issued plan only after a
  successful ledger post; its remaining state write and exact-plan append use already
  validated closed values and cannot fail in the normal context implementation. A real
  distributed payment service would use a durable database/ledger transaction or saga.
- Authentication evidence and credential fields are synthetic identifiers/digests only. No
  usable credential, authenticator secret, PAN, or external payment token is exported.
- One verifier owns at most one ephemeral preview or prepared plan. This avoids abandoned
  preview retention and makes the public two-phase API deterministic; concurrent production
  execution would require transaction IDs and isolated persistent locks.
- Files changed in this round: `src/apar/contracts/decisions.py`,
  `src/apar/trust/verifier.py`, `src/apar/trust/__init__.py`,
  `src/apar/simulator/rails/agentic.py`, `tests/trust/test_verifier.py`, and
  `tests/simulator/test_agentic_rail.py`.

## Fix round 3/5

### Review findings addressed

1. No-step-up requests now use an exact `None` authentication-evidence representation.
   Any string reference—including one colliding with the trusted evidence registry—is
   rejected at the canonical request boundary. Their persisted receipt record uses an empty
   internal lineage sentinel, which remains receipt-bound but is excluded from evidence
   replay tracking and duplicate-evidence checks. Multiple no-step-up records therefore do
   not consume or poison trusted step-up evidence, including across agents.
2. Every public `preview`, `prepare_commit`, and `commit` attempt revokes all prior ephemeral
   previews/plans before inspecting any request, receipt, outcome, timestamp, or subclass.
   Public apply/discard attempts likewise revoke a pending plan before validating the caller
   candidate. The public methods delegate to private validated helpers so standalone
   `commit` and the adapter retain their successful single-thread path without making stale
   capabilities reusable.
3. `TrustCommitPlan` is now a frozen, identity-only capability with value equality disabled.
   It owns a canonically reconstructed exact `IntegrityReceipt` and exposes no state record,
   preview hash, nonce, or caller-controlled commit fields. The verifier separately retains
   a private canonical receipt and state record. Projection and apply require the exact
   issued plan instance; apply writes and returns only the private retained values. Forged,
   copied, subclassed, cross-verifier, mutated, duplicate, or malformed plans cannot commit.

### RED evidence

The first focused run after adding the reviewed exploit tests produced 23 failures:

```text
23 failed, 62 passed
```

The failures reproduced all three findings:

- `None` was rejected for no-step-up requests while arbitrary and registry-colliding strings
  were accepted and written into evidence-shaped state;
- malformed request, receipt, timestamp, and subclass attempts left an existing preview
  reusable at the public preview/prepare/commit boundaries;
- value-equal reconstructed and cross-verifier plans were accepted, a caller-mutated public
  receipt was returned as committed, and malformed apply attempts left the real plan usable.

Additional RED cycles showed that the public agentic command fingerprint rejected the new
exact `None` representation and that a mutable empty-string subclass could enter restored
no-step-up state before exact scalar validation was added.

### GREEN evidence

```text
.venv/bin/python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q
112 passed

.venv/bin/python -m pytest -q
500 passed

.venv/bin/ruff check src tests
All checks passed!

.venv/bin/mypy --strict src
Success: no issues found in 33 source files

.venv/bin/mypy src
Success: no issues found in 33 source files

.venv/bin/python scripts/verify_g0.py
G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store
```

`git diff --check` passed and `validation_spike/` is unchanged from `cdc4c96`.

### Coverage and tradeoffs

- Coverage includes no-step-up arbitrary/colliding references, cross-agent poisoning,
  legitimate step-up after no-step-up activity, public command/engine round-trip, exact empty
  state scalar ownership, malformed objects and subclasses at every public transaction
  boundary, copied and reconstructed plans, cross-verifier reuse, caller mutation, malformed
  apply, duplicate apply, discard identity, and valid state round-trip.
- The public plan intentionally remains inspectable for its synthetic final receipt so the
  adapter can construct a prevalidated event before posting. That receipt is a detached
  canonical copy; the verifier never trusts it for apply. The actual record and committed
  receipt remain private and verifier-owned.
- The identity capability is deliberately process-local and single-threaded, matching the
  engine boundary. Distributed execution would replace Python object identity with a durable
  transaction handle guarded by a database/ledger transaction.
- Round-3 files: `src/apar/trust/verifier.py`,
  `src/apar/simulator/rails/agentic.py`, `tests/trust/test_verifier.py`, and
  `tests/simulator/test_agentic_rail.py`.

## Fix round 4/5

### Review findings addressed

1. `discard_preview` now snapshots the pending preview and revokes both ephemeral slots at
   method entry, before inspecting the caller argument. It then requires the exact issued
   preview object. `None`, arbitrary objects, wrong exact receipts, receipt subclasses,
   equality-equivalent copies, consumed previews, and repeated discard attempts all fail
   without leaving either a preview or prepared plan reusable. A valid issued preview still
   discards successfully once.
2. `AgenticPaymentCommand` now requires an exact `AgentPaymentRequest` before validating
   account text or dereferencing signed request fields. `None`, arbitrary objects, and
   request subclasses therefore fail with the stable boundary `TypeError` and cannot leak an
   `AttributeError`. The other Task 4 public constructors were checked for the same ordering
   error; none dereference a constructor argument before their applicable exact-type gate.

### RED evidence

The targeted test run before production edits reproduced both review findings:

```text
11 failed, 1 passed, 112 deselected
```

Every malformed discard path either left the pending preview committable, left the prepared
plan applicable, or silently accepted a wrong/copy receipt. `None` and arbitrary command
requests leaked `AttributeError`; the already-defensive request-subclass case was the single
passing control.

### GREEN evidence

```text
.venv/bin/python -m pytest tests/trust/test_verifier.py \
  tests/simulator/test_agentic_rail.py -q \
  -k 'preview_discard or validates_request_before or request_subclass_before'
13 passed, 112 deselected

.venv/bin/python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q
125 passed

.venv/bin/python -m pytest -q
513 passed

.venv/bin/ruff check src tests
All checks passed!

.venv/bin/mypy --strict src
Success: no issues found in 33 source files

.venv/bin/mypy src
Success: no issues found in 33 source files

.venv/bin/python scripts/verify_g0.py
G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store
```

`git diff --check` passed and `validation_spike/` is unchanged from `492c736`.

### Self-review and tradeoff

- Preview discard now mirrors commit-plan capability semantics: object identity, not value
  equality, proves possession of the live verifier-issued capability. This is deliberately
  process-local and single-threaded, consistent with the simulator trust boundary.
- Revocation happens even when the discard argument is malformed or stale. This makes a
  caller mistake fail closed at the cost of requiring a fresh preview for retry, which is the
  intended ephemeral-capability policy.
- The patch changes only the two reviewed public boundaries, their regressions, and this
  report. Trust ordering, persistent state, receipt derivation, ledger behavior, event
  emission, and outcome semantics are unchanged.
