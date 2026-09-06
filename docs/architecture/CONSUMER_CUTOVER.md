# Nexus Learning Canonical Consumer Cutover & State Root Specification

## 1. Overview

This document specifies the consumer inventory, migration classifications, single-writer rules, and explicit state-root contracts for the canonical learning capabilities extracted from `Nexus-new` into `James3014/nexus-learning`.

The canonical authority of learning and experience contracts resides strictly in `nexus-learning`. Evolving dual implementations across repos is forbidden.

---

## 2. Canonical Capability Consumer Inventory

| Canonical Module (`nexus-learning`) | Legacy Equivalent (`Nexus-new`) | Current Direct Callers (`Nexus-new`) | Async / Event Callers | Existing Tests | Classification | Migration Action |
| --- | --- | --- | --- | --- | --- | --- |
| `nexus_learning.contracts` | `nexus.contracts.learning_experience` | `nexus/app/research_flow_service.py`<br>`nexus/contracts/s2t_export.py`<br>`nexus/engine/learning_policy_store.py`<br>`nexus/engine/pipeline_crystal.py`<br>`scripts/ops/nexus_runtime_lifecycle_acceptance.py`<br>`scripts/ops/export_s2t_agent_lightning.py` | Event hooks in flow service, pipeline crystal transitions | `tests/contracts/test_learning_experience.py`<br>`tests/contracts/test_s2t_export.py` | `ADAPTER_REQUIRED` | In `Nexus-new`, replace implementation in `nexus/contracts/learning_experience.py` with thin re-exports/forwarding adapter importing `nexus_learning.contracts`. |
| `nexus_learning.episode_projection` | `nexus.learning.learning_episode_projection` | `nexus/learning/learning_closure_effectiveness.py`<br>`nexus/app/research_flow_service.py` | Research flow projection dispatch | `tests/learning/test_learning_episode_projection.py` | `ADAPTER_REQUIRED` | Forward calls from legacy module to `nexus_learning.episode_projection`. Direct callers migrate import to `nexus_learning.episode_projection`. |
| `nexus_learning.closure_effectiveness` | `nexus.learning.learning_closure_effectiveness` | `nexus/core/router.py`<br>`nexus/research/learn_mode.py` | Post-run learning closure evaluation | `tests/learning/test_learning_closure_effectiveness.py` | `MIGRATE_TO_NEXUS_LEARNING` | Migrate callers directly to `nexus_learning.closure_effectiveness` or forward through legacy adapter. |
| `nexus_learning.effectiveness_measurement` | `nexus.learning.learning_effectiveness_measurement` | Scorer utilities / offline analysis | Metric aggregation pipelines | `tests/learning/test_learning_effectiveness_measurement.py` | `MIGRATE_TO_NEXUS_LEARNING` | Consume purely from `nexus_learning.effectiveness_measurement`. |
| `nexus_learning.coverage_contract` | `nexus.learning.learning_coverage_contract` | `nexus/learning/learning_coverage_probes.py` | None | `tests/learning/test_learning_coverage_contract.py` | `MIGRATE_TO_NEXUS_LEARNING` | Consume directly from `nexus_learning.coverage_contract`. |
| `nexus_learning.coverage_probes` | `nexus.learning.learning_coverage_probes` | Audit tools / diagnostic scripts | None | `tests/learning/test_learning_coverage_probes.py` | `MIGRATE_TO_NEXUS_LEARNING` | Consume directly from `nexus_learning.coverage_probes`. |
| `nexus_learning.outcome_memory` | `nexus.learning.outcome_memory` | `nexus/core/router.py`<br>`nexus/core/capability_executor_registry.py`<br>`nexus/app/research_flow_service.py`<br>`scripts/ops/build_antigravity_closure_ledger.py` | Asynchronous worker writeback via `save_episode_and_tune` | `tests/learning/test_outcome_memory_worker_write.py`<br>`tests/learning/test_outcome_memory.py` | `ADAPTER_REQUIRED` | Make `nexus.learning.outcome_memory.OutcomeMemoryManager` a forwarding facade calling `nexus_learning.outcome_memory.OutcomeMemoryManager` with explicit `project_root`. |
| `nexus_learning.retrieval_audit` | `nexus.learning.retrieval_audit` | `nexus/learning/knowledge_index.py` | None | `tests/learning/test_retrieval_audit.py` | `ADAPTER_REQUIRED` | Forward `log_retrieval_audit` and `AuditEntry` to `nexus_learning.retrieval_audit`. |

---

## 3. Explicit State Root Contract (`LearningStateRoot`)

Persistent learning stores must not depend implicitly on the process's current working directory (`cwd`).

### Contract Invariants
1. **Explicit Root Binding**: All persistent read/write operations require an explicit `project_root` or an explicitly configured `NEXUS_LEARNING_STATE_ROOT` environment variable.
2. **Fail-Closed on Underspecified State**: If no `project_root` or environment variable is supplied, `resolve_learning_state_root` raises `ValueError`, refusing silent fallback to `Path.cwd()`.
3. **Local Dev Exception**: Process cwd fallback is permissible only if explicitly opted-in via `allow_dev_cwd_fallback=True` for offline script/dev usage. Production integration paths MUST NOT pass this flag.
4. **Resolved Standard Hierarchy**:
   - `root / ".nexus" / "memory" / "outcome_history.jsonl"` (Outcome records)
   - `root / ".nexus" / "memory" / "dynamic_learning_policy.json"` (Dynamic autotuned policy)
   - `root / ".nexus" / "memory" / "learning_episodes.jsonl"` (Canonical learning episodes)
   - `root / ".nexus" / "audit" / "retrieval_log.jsonl"` (Retrieval audit log)

---

## 4. Single-Writer Rule

To prevent dual-write corruption or state divergence when both legacy and standalone packages coexist during migration:

1. **Single Writer**: For any given `.nexus` state tree, only `nexus_learning` is the authoritative writer.
2. **No Dual Writes**: Legacy code in `Nexus-new` must not write independently to the same state files with a distinct schema or format.
3. **Schema Compatibility**: Legacy readers or thin forwarding adapters may read the state files only if adhering strictly to `nexus.learning_episode.v1` and `nexus_outcome_memory_episode.v1`.
4. **Locking & Atomicity**: Atomic file appending with advisory file locks (`fcntl.flock` with thread-level fallback) is enforced on all episode additions.
5. **No Historical Bulk Renaming**: Historical data in `.nexus/memory` must be preserved as-is without backfilling or altering historical episode identities.

---

## 5. Python 3.10 vs 3.11 Runtime Decision

- `nexus-learning` requires `Python >= 3.11` to utilize modern typing, UTC timezone utilities, and built-in tomllib.
- Legacy `Nexus-new` environments declare support for `Python >= 3.10`.
- **Decision**: `nexus-learning` will NOT lower its requirement to Python 3.10.
  - Legacy consumers in `Nexus-new` will be upgraded to Python 3.11 in CI and supported deployment profiles (already standard across modern workflows).
  - Any remaining legacy-only modules that cannot upgrade to Python 3.11 immediately remain isolated in `Nexus-new` under `LEGACY_ONLY` classification until their runtimes are upgraded.

---

## 6. Advisory Authority Boundary

Learning outputs (e.g. `promoted_capabilities`, `penalized_capabilities`, and `dynamic_learning_policy.json`) are strictly **advisory policy artifacts**.

- Learning outputs CANNOT mutate `CapabilityPlanner` decisions.
- Learning outputs CANNOT select models or workers.
- Learning outputs CANNOT modify `Workforce` admission.
- Learning outputs CANNOT authorize PR approvals, merges, releases, or production deployments.

---

## 7. Required Cross-Repo Follow-Up for `Nexus-new`

Once this standalone release is stabilized:
1. In `Nexus-new`, add dependency on `nexus-learning>=0.1.0`.
2. Convert `nexus/contracts/learning_experience.py` to forward to `nexus_learning.contracts`.
3. Convert `nexus/learning/outcome_memory.py` to forward to `nexus_learning.outcome_memory`.
4. Convert `nexus/learning/learning_episode_projection.py` to forward to `nexus_learning.episode_projection`.
5. Convert `nexus/learning/learning_closure_effectiveness.py` to forward to `nexus_learning.closure_effectiveness`.
6. Ensure all callers provide explicit `project_root` into `OutcomeMemoryManager` and `RetrievalAuditLogger`.
