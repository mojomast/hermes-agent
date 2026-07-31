# Hermes efficiency sprint — architectural record

## 1. Task frame

**Objective:** reduce real Hermes task cost and latency without reducing correctness, safety, privacy, or recoverability.
**Scope:** agent loop, tool execution, context projection, retrieval, delegation, autonomous yield, and recovery.
**Selection constraint:** at most three compatible changes; ship only changes with a before/after benchmark.
**Chosen invariant:** the append-only/canonical session ledger retains full tool evidence, while the disposable model-request projection may compact only old oversized tool results. Recent evidence, message order, and tool-call/result IDs remain exact.

## 2. Vocabulary map

| Term | Bounded meaning |
|---|---|
| Agent | Reasoning actor that chooses actions |
| Task | Bounded user objective |
| Run | One execution attempt of a task |
| Session | Resumable interaction container |
| Conversation | Ordered user/assistant exchange in a session |
| Message | One conversation record |
| Event | Immutable fact |
| Tool call | Agent-requested operation and result |
| Job | Scheduler-owned execution |
| Workflow | Dependency graph of steps |
| Memory | Durable recalled knowledge |
| Artifact | Produced file or object |
| Ledger | Append-only event history |
| Projection | Rebuildable view over ledger evidence |
| Cache | Disposable value with explicit scope/freshness |
| Scheduler | Admits and orders jobs |
| Worker | Executes admitted work |
| Candidate | Proposed, not-yet-selected change |
| Validation | Evidence that stated invariants hold |

## 3. Structural model

```text
Task -> Run -> model request projection -> model response
                                      -> tool-call group
                                         -> tool workers
                                         -> raw message/event persistence (canonical)
                                         -> bounded proactive projection (disposable)
                                      -> next model request
Session ledger --------------------------^        |
Recovery/replay projections <---------------------+
```

The projection is never authoritative. It is deterministic, fail-open, recent-tail preserving, and disabled by default. Prompt-cache invalidation is bounded by a measured reclaim hysteresis.

## 4. Baseline measurements

Baseline artifact: `/tmp/hermes-efficiency-artifacts/baseline.json` (generated, not committed).

| Metric | Baseline |
|---|---:|
| Representative workload families | 6 |
| Repetitions | 7 |
| Model calls represented | 42 |
| Tool calls represented | 182 |
| Median context reduction | 0.0% |
| Correctness score | 1.0 |
| Retries | 0 |
| Duplicate reads/searches | 0 |
| Peak concurrency | 3 |
| Cache state | disabled |

Live production-ledger profiling additionally found median input of about 67K tokens, median tool-result content of about 35K tokens, and tool results at least half of input on 54.2% of sampled API calls. Prompt assembly itself was milliseconds; repeated carriage of consumed evidence was the waste.

## 5. Parallel findings matrix

Ten read-only specialists ran in bounded batches of three (runtime concurrency limit), with non-overlapping scopes.

| ID | Scope | Key evidence | Recommendation |
|---|---|---|---|
| A | End-to-end profile | Model spans 59.5% and tool/subagent spans 37.9% of recent wall time; mixed batches strand 10.3% of safe calls | Mixed-batch waves implement; async delegation investigate |
| B | Tool-call efficiency | Remote `execute_code` modeled as `6 + 4N` backend calls; 100 tool schemas total about 81K JSON chars | Narrow batched RPC investigate |
| C | Context/token efficiency | Historical results dominated prompts; bounded projection modeled 51.6% median cumulative reduction | Implement reversible projection |
| D | Parallel architecture | Batch-wide safety, nested executors, soft admission, credential identity race | Stage supervisor; do not raise fan-out first |
| E | Retrieval/cache | Same-session exact repeats only 0.31%/72h; cross-session file reads had 14.3% identical output reuse | Reject broad cache; narrow evidence index investigate |
| F | Autonomous yield | Useful cycle yield 16.1%; 58.8% of recent candidates in near-duplicate pairs | Pull-gate planner; preserve validation |
| G | Recovery | Age-only stale locks imposed 10.30h median wait despite dead owners | Recovery packet implement separately |
| H | Benchmark design | Real-loop deterministic corpus and interval ledger feasible | Commit reusable harness; keep outputs external |
| I | Skeptic | Hardened upstream prior art exists; stale caches and broad RPC concurrency are unsafe | Backport behavior, not duplicate architecture |
| J | Capability scout | Upstream proactive projection exists but durable-first persistence is prerequisite | Implement projection first |

## 6. Ranked proposals

Weighted score uses impact 25%, breadth 15%, evidence 15%, feasibility 10%, safety 10%, freshness 5%, testability 10%, maintenance 5%, reversibility 3%, and compatibility 2%. Each factor is scored 1–5 and normalized to 100.

| Rank | Proposal | Score | Gate |
|---:|---|---:|---|
| 1 | Durable-first proactive tool-result projection | 93.6 | **Selected** |
| 2 | Fail-closed active-run recovery packet | 86.6 | Fresh, high impact, different repository; defer to avoid shallow multi-project shipment |
| 3 | Pull-gated autonomous planner | 77.6 | Existing `/goal` machinery and cross-repository scope require integration design |
| 4 | Hardened mixed-batch waves | 76.0 | Valuable, but lower measured impact and large dirty monolith conflict surface |
| 5 | Batched `execute_code` RPC | 63.6 | Reject generic concurrency; remote read-only batching only |
| 6 | Shared repository evidence index | 58.6 | Reject until completeness/freshness/locking contracts exist |

## 7. Selected change set

1. Add a plugin-safe proactive projection hook.
2. Add deterministic old-result projection with:
   - opt-in token trigger;
   - exact recent-tail preservation;
   - 8K-character default result floor;
   - 4K-token reclaim hysteresis;
   - identity no-op contract;
   - no auxiliary model call.
3. Persist raw assistant/tool rows before replacing in-memory request history.
4. Add bounded configuration and documentation.
5. Add a six-family deterministic benchmark harness whose generated results stay outside source.

## 8. Drift and duplication audit

This is not a novel competing compressor. It adapts upstream commit `cb481e2f2b` and incorporates the durable-first requirement identified by upstream `858bedea02` into the older monolithic fork. Broad caches, duplicate planners, dashboard telemetry, schema-only renames, and direct cherry-picks across the fork's module-layout drift were rejected.

## 9. Before/after benchmark

Generated artifacts:

- baseline: `/tmp/hermes-efficiency-artifacts/baseline.json`
- candidate cold: `/tmp/hermes-efficiency-artifacts/candidate-cold.json`
- candidate warm: `/tmp/hermes-efficiency-artifacts/candidate-warm.json`

| Metric | Before | Cold after | Warm after |
|---|---:|---:|---:|
| Median projected context reduction | 0.0% | 81.9955% | 81.9955% |
| Correctness score | 1.0 | 1.0 | 1.0 |
| Model/tool calls represented | 42 / 182 | 42 / 182 | 42 / 182 |
| Retries | 0 | 0 | 0 |
| Peak concurrency | 3 | 3 | 3 |

The benchmark exercises the real `ContextCompressor` implementation with repository investigation, multi-file engineering, web synthesis, cross-session recall, parallel subagent, and failed-run recovery evidence shapes. It does not claim wall-time speedup from its sub-millisecond local projection runtime; the qualified result is context reduction.

## 10. Validation evidence

- Intentional core RED: 8 failed, 20 passed.
- Core GREEN: 28 passed.
- Intentional integration RED: 4 failed.
- Focused integration GREEN: 35 passed.
- Existing run-agent suite from implementation worker: 300 passed.
- Syntax and diff checks passed.
- Exact full relevant suite and smoke results are recorded in the shipment report.

## 11. Safety and privacy review

- No approval, destructive-action, replay, or privacy gate is bypassed.
- Projection is fail-open and default-off.
- Canonical SQLite rows are flushed before projection; projected rows are not appended as duplicate canonical evidence.
- No tool output, local path, hostname, username, secret, credential, or conversation content is emitted as telemetry.
- Benchmark fixtures are synthetic and generated artifacts are external to the repository.
- Prompt-cache churn is bounded by minimum reclaim hysteresis.

## 12. Rollback

Disable immediately without code rollback:

```yaml
compression:
  proactive_prune_tokens: 0
```

Code rollback: revert the sprint commit on `efficiency/proactive-tool-pruning`. Canonical session rows remain usable because raw results are persisted before projection.

## 13. Commit and push status

Source, tests, benchmark harness, documentation, and this record are committed on `efficiency/proactive-tool-pruning`; generated benchmark JSON remains uncommitted outside the repository. The final shipment response records the exact commit and remote push result.
