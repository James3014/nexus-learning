# Nexus Learning Architecture Boundary (v2)

## Why v2

v1 split by authority (pure contracts only) and left the learning loop unclosed: as of 2026-10-08 Nexus-new recorded 91 outcome episodes, 40 retrieved, 0 applied, 0 with observed uplift. v2 splits by learning-loop stage ownership; the authority invariants below are a property every stage must keep, not a package boundary.

## Learning-loop stage ownership

| Stage | Owner | Modules / contract | Notes |
|---|---|---|---|
| Storage | nexus-learning | `outcome_memory`, `closure_effectiveness`, `episode_projection`, `state_root`, `retrieval_audit` | explicit state root, single writer, content-addressed episode ids. |
| Reflection | nexus-learning | `lessons` (Phase 1, implemented): canonical `nexus.learning_lesson.v1` schema, lesson store, reflector that distills episodes into lessons through an injected judge callable | package stays I/O-free except the lesson store; no provider dependency. |
| Retrieval | nexus-runtime (read-only port) | `LearningReadPort` in nexus-runtime; nexus-learning provides deterministic retrieval helpers only | consumers must record `retrieved_lesson_ids` and `applied_lesson_ids` on the episode. |
| Measurement | nexus-learning | `effectiveness_measurement` (`paired_memory_uplift`, `replay_scorecard`, `compare_workflows_at_required_quality`), `coverage_contract`, `coverage_probes` | memory-on vs memory-off evidence is the only admissible uplift evidence. |
| Adoption | nexus-learning | `adoption` (store for adoption and rollback artifacts), `contracts` (recommendation, validation, adoption, rollback), `experiment_integrity` | recommendation != authority; adoption artifacts are advisory overlays read by consumers. |

## Frozen research-governance modules

The following modules are research-campaign governance contracts, not learning-loop stages. They have no consumer outside this repository. They are frozen: no new features, bug fixes only, and they will move to a dedicated research-governance owner when one exists.

- `nexus_learning.campaign_closeout`
- `nexus_learning.campaign_gates`
- `nexus_learning.experiment_preflight`
- `nexus_learning.experiment_run_identity`
- `nexus_learning.necessary_condition_stop`
- `nexus_learning.research_frontier`
- `nexus_learning.workflow_friction`

Note: `experiment_integrity` is NOT frozen because `contracts.build_learning_policy_recommendation` binds recommendations to it.

Each frozen module carries `LEARNING_BOUNDARY_STATUS = "FROZEN_RESEARCH_GOVERNANCE"`, enforced by `tests/test_boundary_v2.py`.

## Authority invariants

- Learning recommendations are NOT execution authority.
- No direct route mutation allowed.
- No direct CapabilityPlanner mutation allowed.
- No workforce admission mutation allowed.
- No model self-promotion allowed.

## Consumer pins

| Consumer | Requirement |
|---|---|
| Nexus-new | must pin the same nexus-learning revision as nexus-runtime |
| nexus-runtime | must pin the same nexus-learning revision as Nexus-new |

Pin drift between consumers is a Phase 0 defect.

## Definition of "helpful"

- at least 50 non-mock episodes;
- `applied_lesson_ids` non-empty on real episodes;
- one memory-on vs memory-off scorecard with positive paired uplift;
- one governed adoption artifact produced from that scorecard.
