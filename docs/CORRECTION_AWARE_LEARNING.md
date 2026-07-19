# Correction-aware shadow learning

## Release status and positioning

This branch adds an **integrated Hermes Agent shadow-evaluation substrate**. It is not a standalone model-training package and it does not update model weights, prompts, memories, skills, or policies. Hermes persists structural execution traces, automatically captures narrowly authorized foreground-pytest outcomes, accepts an explicit taxonomy-only correction command, provides a local operator control plane for lesson inspection/retraction, and exposes count-only recurrence metrics to the model. Every generated lesson remains inactive.

The capability is correction-aware because corrected attempts can be linked to the original failed evidence and accepted only with a later strict verifier pass. It is shadow learning because the resulting matches, hint text, and lesson states are reports only:

- `shadow_only` is `true`;
- `activation_allowed` is `false`;
- `active_lesson_count` is always `0`; and
- `prompt_modified` is `false`.

These declarations describe this implementation boundary; they are not a claim that arbitrary data in a Hermes database is safe.

## Task frame and non-goals

Contrastive inspection accepts operator-supplied `task_text`. A deterministic keyword map projects that transient text into one or more allowlisted categories:

`capability_audit`, `capability_extension_ideation`, `project_ideation`, `duplicate_risk`, `cross_session_recall`, `system_state_audit`, `debugging`, `user_correction`, `verification`, and `deployment`.

Matching then uses category-to-tool and category-to-taxonomy maps, safe structural episode fields, and effective Outcome Events. The packet exports the categories, not the task text. Empty or unmatched text produces no categories and therefore no relevant matches or hints.

This task framing is deliberately small. It is **not** semantic intent recognition, a general user-prose correction detector, a prompt-injection defense, or a security classifier. Keywords can miss paraphrases and can classify quoted/adversarial words. The operator must interpret the report.

Non-goals for this release include:

- active adaptation or automatic application of a hint;
- prompt assembly, prompt mutation, or retrieval injected into a model call;
- model fine-tuning, online RL, or weight updates;
- deriving corrections, retractions, or replacements from free-form user prose (the exact `/correct <taxonomy>` control command is structural, not prose interpretation);
- background verification or background-review capture;
- parsing arbitrary test runners, shell pipelines, CI logs, or pytest output text;
- treating a successful command exit as proof beyond the narrow direct-pytest contract.

## Vocabulary map

| Term | Meaning in this release |
|---|---|
| **trace** | One Hermes turn, identified by an opaque `trace_id`, with structural status, timing, and call counts. |
| **span** | A nested structural operation such as `turn`, `model_call`, `tool_call`, `subagent`, `final_answer`, or `turn_error`. |
| **Outcome Event** | An immutable fact attached to a trace: an allowlisted event type, source, polarity, confidence, optional taxonomy, timestamp, optional digest identifier, and optional `supersedes_event_id`. |
| **producer** | A versioned structural authority allowed to turn transient runtime facts into a restricted Outcome Event or relation. It is not the event's `source`. |
| **source** | The claimed evidence role stored on an Outcome Event: `user`, `verifier`, `evaluator`, `operator`, or `system`. |
| **taxonomy** | One allowlisted failure/lesson code: `duplicate_existing_capability`, `wrong_live_checkout`, `wrong_scope`, `inspect_before_edit`, `prerequisite_not_checked`, `premature_completion`, `verification_missing`, `unsupported_claim`, `stale_context`, or `repeated_tool_failure`. |
| **root** | A distinct trace carrying effective, high-confidence (`>= 0.9`), negative, user-sourced evidence for a taxonomy. Multiple events on one trace count as one root. |
| **strict corrected pair** | A positive, high-confidence user `user_correction` that supersedes a different trace's effective, negative, high-confidence user event with the same taxonomy, plus the latest qualifying verifier result at or after the correction being `verification_passed`. |
| **effective event** | An event that survives the finite-DAG suppression rules described below. Historical queries still include every immutable row. |
| **contrastive match** | A privacy-minimized positive, negative, or corrected episode selected for an operator's task frame. Corrected episodes can also be positive when ready, above the reward threshold, and strictly verified. |
| **hint** | Fixed template text keyed by taxonomy. It is emitted only in a shadow packet and is never applied automatically. |
| **lesson** | A deterministic pseudonymous ID plus taxonomy lifecycle counts/state in the canonical recurrence report. It is never active in this release. |
| **observed / candidate / activation-eligible** | Lifecycle report states at one root, two roots, or all eligibility gates respectively. “Activation-eligible” is evidence sufficiency, not permission to activate. |

Outcome event types are also allowlisted: `user_correction`, `user_rejection`, `user_confirmation`, `verification_passed`, `verification_failed`, `assumption_invalidated`, `wrong_scope`, `duplicate_proposal`, `unsupported_claim`, `premature_completion`, `rollback_required`, and `repeated_tool_failure`.

## Structural producer model

### Runtime producer

The only automatic runtime producer is `foreground_pytest.v1`. After a terminal tool returns, Hermes transiently inspects its arguments and structured result. It emits only `verification_passed` or `verification_failed` candidates and fixes their persisted fields to:

- source `verifier`;
- polarity `positive` for pass or `negative` for fail;
- confidence `1.0`;
- no taxonomy, evidence digest, or supersession.

Recognition is fail-closed. The tool must be `terminal`, `background` must be omitted or exactly `false`, and the command must directly invoke `pytest`, `py.test`, or `python -m pytest` (including supported `python3`/`python3.x` executable names). Any shell metacharacter (semicolon, ampersand, pipe, redirection, backtick, dollar sign, newline, carriage return, or NUL), wrapper, malformed shape, or non-execution option such as collection/setup/help/fixtures causes no event. Only integer exit code `0` (pass) or `1` (test failure) is accepted; booleans and pytest exit codes `2` through `5` are ignored. Command text and result content are discarded after classification.

Capture is skipped on interruption and for background-review origins/agents. Background terminal jobs are not captured, and later `process` polling is not a producer. There is no asynchronous verifier watching jobs after the turn.

### Authorization matrix

“Runtime enforced” below means the normal agent path can create the row and validates the producer-fixed semantics. Low-level Python functions remain available to trusted repository code and fixtures; possession of local database/code execution is therefore inside the trust boundary.

| Write path | Intended caller | Allowed output | Enforcement |
|---|---|---|---|
| `classify_foreground_pytest` → `append_producer_outcome_event` | Normal Hermes runtime | Strict pytest pass/fail verifier events only | **Runtime enforced** by frozen candidate structure, exact producer `foreground_pytest.v1`, deterministic ID, fixed fields, trace existence, and append-only DB constraints. |
| `/correct <taxonomy>` → `record_session_correction` | Authorized CLI/gateway user | One negative `user_correction` on the latest completed assistant trace in the same session | Exact one-token allowlist, deterministic idempotency, fixed source/polarity/confidence, no free-form text, no model call, and no prompt/history write. |
| `/lessons list|inspect|retract` | Local CLI operator only | Bounded allowlisted lesson state/evidence handles or one atomic structural retraction | `cli_only`; absent from gateway commands and model tools. Retraction accepts only an opaque event ID and atomically appends an operator event plus relation. |
| `shadow_recurrence_counts` | Model tool | Global count-only recurrence/verifier metrics and fixed safety flags | No arguments, no arbitrary DB path, read-only SQLite, no taxonomy/lesson/evidence/linkage identifiers, and no write/activation operation. |
| `append_outcome_event` | Trusted fixture/operator/internal code | Any allowlisted event/source/polarity/taxonomy combination | Validation and append-only constraints, but **not runtime producer authorization**. It is an internal construction API and must not be exposed as proof that arbitrary claimed `source="user"` or `source="verifier"` is authenticated. |
| `append_outcome_event_relation` | Trusted structural operator/internal code | `corrects`, `retracts`, or `replaces` | Exact producer `structural_operator.v1` is checked in Python and by a SQLite trigger. Endpoints must exist and cycles are rejected. |
| `retract_outcome_event` | Trusted structural operator/internal code | A `retracts` relation | Same producer requirement, plus the relation's source event must itself have `source="operator"`. No reason/prose argument exists. |
| Direct SQLite access | Local database administrator | Potentially arbitrary legacy/corrupt data | Outside the application authorization boundary. CHECK/append-only/producer triggers reduce mistakes but do not defend against an administrator altering schema or files. Readers fail closed on unknown taxonomy and cyclic suppressor components where applicable. |

For `corrects` and `replaces`, the structural producer is required, but the current API does not additionally require an operator-sourced source event. That is a meaningful limitation, not an implied authorization rule.

## Deterministic idempotency and bounded retry queue

A runtime event ID is `shadow:` plus SHA-256 of a length-framed tuple `(trace_id, tool_call_id, producer)`. Command, result text, and outcome are intentionally absent. Replaying the same structural call returns the existing event only when every candidate-derived and producer-fixed immutable field matches. A contradictory replay collides and fails closed rather than creating two facts. The first write assigns `created_at`; retries return that timestamp unchanged. Concurrent authorized retries re-read and perform the same full comparison.

Runtime candidates are held in an insertion-ordered, in-memory map keyed by event ID. The trace is persisted before its candidate Outcome Events, satisfying trace-existence validation. Failed appends remain queued and retain their original trace attribution across later turns. Each trace-persistence opportunity retries the pending snapshot; failures never fail the user's turn and diagnostics contain only stage, error class, pending count, and attempt count.

The pending map is bounded at 256 entries. At capacity the oldest pending entry is dropped and a count-only capacity warning is logged. Conflicting candidates are removed and their ID is quarantined. The quarantine is also bounded operationally: once it already contains 256 IDs and another conflict occurs, automatic capture is disabled and pending candidates are cleared. Queues and quarantines are process memory only—there is no durable retry journal, retry count, exponential backoff, or guarantee that evidence survives process exit.

Relations use a deterministic UUID5 derived from source ID, target ID, kind, and producer when no relation ID is supplied. An exact retry returns the immutable existing relation; a conflicting duplicate fails.

## Correction, retraction, and replacement semantics

Outcome Events and relations are append-only. Historical counts include every event. Effective readers register a common resolver over `retracts` and `replaces` edges:

- an event is effective when it has no **effective** incoming suppressor;
- a suppressed event can become effective again when its suppressor is itself effectively suppressed;
- deterministic topological traversal resolves finite acyclic chains; and
- cyclic legacy/corrupt components fail closed as ineffective.

The three relation names are not interchangeable:

| Mechanism | Current effect |
|---|---|
| `corrects` relation | Structural/audit linkage only. It does **not** suppress the target in the effective resolver and is not the mechanism used to form strict corrected pairs. |
| `retracts` relation | Suppresses the target while the source event is effective. The source event must be operator-sourced. |
| `replaces` relation | Has the same target-suppression behavior as `retracts` in current effective queries; the different kind preserves control-plane intent. |
| `supersedes_event_id` on an Outcome Event | Links correction evidence to the original event for strict corrected-pair queries and participates in cycle prevention for new relations. By itself it does **not** make the superseded event ineffective. |

Thus, a valid strict corrected pair requires the `supersedes_event_id` field and strict evidence checks; a `corrects` relation alone is insufficient.

The strict verifier result is the latest qualifying verifier pass/fail by `(created_at, event_id)` at or after the correction. A later failure downgrades a pair; a still later pass recovers it. Retraction/replacement of any required effective event removes that support.

## Recurrence policy

`shadow_lesson_lifecycle.v1` calculates state per taxonomy from trusted roots:

| State | Requirement |
|---|---|
| `absent` (`lifecycle_state=0`) | zero effective roots |
| `observed` (`1`) | at least one effective root |
| `candidate` (`2`) | at least two effective roots |
| `activation_eligible` (`3`) | at least three effective roots, roots linked to at least two distinct non-null trace sessions, and at least one valid strict corrected pair |

All three eligibility conditions are required. Session diversity is calculated only when the `traces` table supplies session linkage. Evaluation aggregates all evidence before applying the output lesson limit, so row ordering/noise cannot starve older roots. The database is opened read-only with `PRAGMA query_only=ON`; a missing database returns an empty report without creating a file. The limit is positive and capped at 200.

Eligibility never changes `active=false`, `activation_allowed=false`, or `prompt_modified=false`.

## Privacy threat model and packet contracts

### Protected against

The reporting boundaries are designed to prevent accidental export of stored prompts, user messages, tool arguments/results, stdout/stderr, provider payloads, span metadata, evidence digests, and raw trace/session/event/relation identifiers. Recurrence reads only allowlisted Outcome Event fields plus trace-to-session linkage; it does not read raw user text, prompts, tool payloads, or span metadata. Contrastive retrieval uses privacy-minimized TrainingEpisode structure, allowlisted tool/outcome/taxonomy names, reward/readiness, and pseudonymous episode IDs.

The synthetic evaluation plants ten distinct canaries under raw-looking metadata keys, verifies construction-owned canary counts, and requires zero exported canaries and zero forbidden raw-content keys. This demonstrates fixture coverage; it is not a general information-flow proof.

### Not protected against

- A local administrator or trusted internal caller forging structurally valid events.
- Sensitive information encoded into allowlisted names, counts, timing, rewards, taxonomy selection, or linkability patterns.
- Cross-report correlation: episode hashes and deterministic lesson UUID5 values are intentionally linkable.
- Dictionary attacks against `user_message_hash` for low-entropy messages. The hash is persisted but is not used for matching.
- A future caller violating the tracing metadata contract. `_safe_metadata` bounds and serializes metadata; it is not a user-prose/secret detector.
- Inference from operator-supplied task categories or count changes.

### Canonical internal recurrence packet

`evaluate_shadow_recurrence` returns `shadow_recurrence_report.v1`, including policy thresholds, historical/effective event and verifier counts, per-taxonomy root/state aggregates, and a bounded `lessons` list. Each lesson contains a deterministic pseudonymous `lesson_id`, taxonomy, historical/effective event/root counts, session count, strict-pair count, lifecycle state, and immutable shadow safety flags. Store identifiers and evidence digests are not exported.

### Contrastive packet

`retrieve_contrastive_episodes` returns `contrastive_episode_retrieval.v1`: allowlisted task categories; bounded positive, negative, and corrected matches; fixed-template shadow hints; episode count; privacy declarations; latency; and shadow flags. Matches contain a truncated SHA-256 pseudonym, score/reward, matched structural features, safe tool names, allowlisted outcome/taxonomy names, and event count. Raw task text is used only for category classification and is not returned. `contrastive_replay_eval.v1` converts this to counts, recurrence/taxonomy aggregates, privacy findings, coverage provenance, latency, and `ok`.

### Public model count-only recurrence packet

The model-visible `shadow_recurrence_counts` tool has an empty argument schema and always reads the active profile database. It deliberately removes lessons, lesson IDs, taxonomies, trace/session/event/relation IDs, and digests. Its data object contains only:

- `schema_version`, `policy_version`;
- `shadow_only`, `activation_allowed`, `prompt_modified`;
- `historical_event_count`, `effective_event_count`, `candidate_lesson_count`, `activation_eligible_lesson_count`, `active_lesson_count`;
- `captured_verifier_counts` limited to `historical_pass`, `historical_fail`, `effective_pass`, `effective_fail`; and
- privacy booleans limited to `raw_content_exported`, `raw_user_text_read`, `raw_prompt_read`, and `raw_tool_payloads_read`.

The CLI eval's optional real-database `shadow_recurrence` section is independently projected to count-only `shadow_recurrence_eval.v1` and omits verifier/privacy subobjects. Synthetic fixture sections contain additional test counters but no store linkage.

## User and operator usage

Record a correction immediately after an assistant answer:

```text
/correct wrong_scope
```

The payload must be exactly one taxonomy code from the vocabulary table. Hermes resolves the latest persisted trace with a completed `final_answer` span in the same session, writes one deterministic negative correction, and returns an acknowledgement. The command itself, arbitrary prose, prompt text, and assistant output are not stored in the Outcome Event and are never sent to the model. Repeating the same taxonomy for the same trace is idempotent. With no eligible prior trace, the command fails closed.

Local operators can inspect and retract structural evidence without exposing that control plane to gateways or the model:

```text
/lessons list
/lessons inspect lesson:<opaque-id>
/lessons retract explicit-correction:<opaque-id>
```

`list` and `inspect` return only allowlisted taxonomy/state/count fields and opaque evidence handles; they never return prompts, user text, tool payloads, span metadata, evidence digests, or trace/session IDs. `retract` accepts one exact event handle and atomically appends an operator control event plus `retracts` relation. Historical evidence remains immutable. This is a local-process operator boundary, not protection against a host administrator with code or SQLite access.

The model-visible `shadow_recurrence_counts` tool needs no arguments and has no write, inspection, export, task-text, DB-path, activation, or retraction operation. The broader `training_episodes` tool remains available only through the explicitly enabled `self_improvement` toolset for offline/operator evaluation; it is no longer a core/default model tool.

For operator-only contrastive inspection with that opt-in toolset:

```json
{
  "operation": "contrastive_replay_eval",
  "db_path": "~/.hermes/state.db",
  "task_text": "audit whether Hermes learns from mistakes and uses prior failed episodes",
  "limit": 100,
  "min_reward": 1.0,
  "max_negative_reward": 0.0
}
```

`max_negative_reward` must be finite and strictly less than `min_reward`. Ordinary operator calls do not possess planted-canary provenance, so `contrastive_replay_eval.ok`/`privacy_eval_valid` are expected to remain false even when no raw-content finding is observed. Use the synthetic CLI fixture for the validated privacy evaluation.

Run the offline evaluation without touching the active Hermes database:

```bash
python scripts/evaluate_self_improvement_capabilities.py
```

Add `--db-path ~/.hermes/state.db` to append bounded, read-only replay/contrastive/recurrence sections for a real database. `--output-json PATH` is the only mode that writes a requested report artifact; temporary fixture databases, code, and JSONL are otherwise created under `TemporaryDirectory` and removed.

## Temporary-database evaluation and current measurements

A fresh default CLI run from this release worktree completed successfully (process exit `0`). Latencies are local and non-contractual; this run measured:

| Fixture metric | Result |
|---|---:|
| trace episodes / ready for training | 2 / 1 |
| trace conversion + JSONL export | 4 ms, 5,885 bytes |
| replay rows passed / failed | 1 / 1 (the intentional failed episode makes `fixture_replay_eval.ok=false`) |
| contrastive positive / negative / corrected matches | 1 / 1 / 1 |
| generated / unsupported hints | 1 / 0 |
| contrastive recurrence | `duplicate_existing_capability: 2` |
| contrastive retrieval latency | 1 ms |
| privacy canaries covered / findings | 10 / 0 |
| recurrence historical / effective events | 9 / 8 |
| recurrence candidate / eligible / active lessons | 1 / 1 / 0 |
| classified / authorized persisted / idempotent retry | 2 / 1 / 1 |
| authorization / idempotency / parity / privacy findings | 0 / 0 / 0 / 0 |
| observed / candidate / eligible / downgrade / recovery transitions | 1 / 1 / 1 / 1 / 1 |
| strict verified pairs / effective retractions | 1 / 1 |

Both `fixture_contrastive_replay_eval.ok` and `fixture_shadow_recurrence_eval.ok` were `true`. Semantic-code and trace-to-RL measurements remain in the shared harness but are not claims about correction activation.

The synthetic recurrence database is separate from the trace-to-RL fixture database. It creates four manual high-confidence user-negative roots in four sessions, a correction/replacement trace, an operator trace, direct pytest pass/fail classifications, later fail/pass transitions, and one structural retraction. These manual user events are trusted fixture construction through the internal API—not evidence that runtime user prose is detected or authenticated.

## Drift and parity audit

The release checks two execution-path parity concerns:

1. sequential and concurrent tool execution preserve terminal results and queue equivalent structural candidates; and
2. the synthetic evaluator compares sequential and thread-pool classification of pass/fail inputs (`parity_finding_count=0` in the fresh run).

All recurrence consumers call the same canonical evaluator, while the tool and CLI apply separate count-only projections. Tests assert that these projections remain bounded and do not leak linkage rows. Schema versions intentionally differ (`shadow_recurrence_report.v1` internally versus `shadow_recurrence_eval.v1` in the CLI fixture/real-DB projection), so consumers must key on the reported version rather than assume identical shapes.

Before changing this subsystem, audit these duplicated/versioned surfaces together:

- outcome/taxonomy/source allowlists and SQLite CHECK constraints;
- task categories, keyword maps, category-to-tool/taxonomy maps, and hint templates;
- runtime direct-pytest grammar and producer-fixed persistence fields;
- strict-pair SQL in outcome retrieval and recurrence evaluation;
- effective-event resolver use by summaries, pairs, contrastive retrieval, and recurrence;
- canonical, tool count-only, and CLI count-only packet schemas; and
- sequential/concurrent capture plus privacy canary coverage.

A zero parity finding covers only the fixture inputs. It does not prove provider, platform, shell, pytest-plugin, or distributed-worker parity.

## Tracing prerequisite repair

Automatic verifier capture depends on a persisted trace: `append_outcome_event` rejects a missing referenced trace when the `traces` table exists. This release therefore wires structural tracing into the real agent turn instead of relying on fixture-only traces. A turn recorder starts before model/tool work, stores only a user-message SHA-256 rather than text, records nested model/tool/subagent/final-answer structure, closes unfinished spans as `abandoned`, and atomically records the completed trace and full span set. Re-persisting a trace replaces its complete span set rather than merging a partial set.

The persistence order is trace first, queued Outcome Events second, recorder marked persisted last. Trace failure prevents event append and leaves candidates retryable. Event append failure does not erase the successfully stored trace or fail the user turn. This prerequisite requires a session database and session persistence; ephemeral `persist_session=False` flows do not durably contribute evidence.

## Limitations and gates before active adaptation

This release intentionally stops before activation. At minimum, an active-adaptation release still needs:

1. **Authenticated evidence producers.** The exact taxonomy-only correction command is now a reviewed runtime producer; broader correction/rejection verifiers and stronger identity binding still require explicit producer designs. Do not expose the permissive fixture API as authentication.
2. **Prose and injection design.** If user prose is ever interpreted, define provenance, quotation/forwarding handling, prompt-injection resistance, confidence calibration, consent, and appeal/retraction workflows. No such detector exists today.
3. **Durable delivery.** Replace or supplement the process-local lossy queue with transactional/outbox semantics, bounded durable retention, observability, and recovery tests.
4. **Activation governance.** Add an independently authorized approval/promotion path, rollback/kill switch, version pinning, audit trail, rejection state, expiry, and conflict resolution. “Activation-eligible” alone must never activate.
5. **Prompt safety and measurement.** Specify how a lesson could enter prompts, cap influence and token cost, isolate untrusted text, and prove prompt/non-prompt A/B behavior. Current code has no prompt wiring.
6. **Privacy review.** Threat-model linkage/timing/count leakage, hash dictionary attacks, retention/deletion, access control, database migration, and real-world canary coverage. Replace contractual metadata discipline with enforceable schemas where needed.
7. **Verifier breadth and quality.** Define trustworthy non-pytest verification, flaky-test handling, test selection/coverage, retries, contradictory evidence, and background/distributed execution semantics.
8. **Drift/parity gates.** Add schema compatibility tests across upgrades and representative platforms/providers/tool paths; preserve aggregate-before-limit and effective-event parity.
9. **Operational evaluation.** Establish precision/recall, false-activation, recurrence reduction, latency/storage budgets, and longitudinal cross-session metrics on consented data.
10. **Trust-boundary hardening.** Separate fixture/operator APIs from production capabilities and enforce producer authorization at a boundary stronger than an importable Python function/local SQLite file.

Until those gates are implemented and reviewed, treat every match, hint, and eligible lesson as diagnostic output only.

## Reproduction from a clean tree

From the repository root, with project dependencies installed, run the focused release suite:

```bash
python -m pytest -q \
  tests/agent/test_contrastive_episode_retrieval.py \
  tests/agent/test_correction_learning_adapter.py \
  tests/agent/test_outcome_events.py \
  tests/agent/test_outcome_event_relations.py \
  tests/agent/test_shadow_outcome_capture.py \
  tests/agent/test_shadow_recurrence.py \
  tests/agent/test_tracing.py \
  tests/agent/test_training_episodes.py \
  tests/run_agent/test_shadow_outcome_integration.py \
  tests/cli/test_correction_commands.py \
  tests/test_evaluate_self_improvement_capabilities.py \
  tests/test_trace_persistence.py \
  tests/tools/test_self_improvement_tool.py
```

Run the fresh temporary-database evaluation:

```bash
python scripts/evaluate_self_improvement_capabilities.py
```

Optionally exercise the active database read-only in addition to the fixture:

```bash
python scripts/evaluate_self_improvement_capabilities.py \
  --db-path ~/.hermes/state.db \
  --limit 100
```

There is no repository-wide Markdown linter or local-link checker configured. A dependency-free sanity check for the links added by this release is:

```bash
python - <<'PY'
from pathlib import Path
import re
for source in (Path("README.md"), Path("docs/CORRECTION_AWARE_LEARNING.md")):
    text = source.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" not in target and not target.startswith("#"):
            path = (source.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"broken local link in {source}: {target}"
print("local Markdown links: ok")
PY
```
