# Nexus Learning Migration Matrix

| Path | Direct Callers | Tests | Authority Source | Current Role | External Dependencies | Classification | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nexus/learning/learning_episode_projection.py` | local_heal, learning_closure | `test_learning_episode_projection.py` | Episode Lifecycle | Canonical Episode Projection | Standard library only | `CANONICAL_ADAPT` | Canonical episode normalization and content-addressed identity. Migrated to `nexus_learning.episode_projection`. |
| `nexus/learning/learning_closure_effectiveness.py` | local_heal, runtime_learning | `test_learning_closure_effectiveness.py` | Episode Lifecycle | Learning closure & evaluation | contracts.learning_experience | `CANONICAL_ADAPT` | Evaluates learning closure effectiveness post-execution without routing authority. Migrated to `nexus_learning.closure_effectiveness`. |
| `nexus/learning/learning_effectiveness_measurement.py` | learning_scorer | `test_learning_effectiveness_measurement.py` | Effectiveness Metric | Statistical effectiveness metrics | Standard library only | `CANONICAL_ADAPT` | Pure statistical evaluation of learning uplifts. Migrated to `nexus_learning.effectiveness_measurement`. |
| `nexus/learning/learning_coverage_contract.py` | coverage_probes | `test_learning_coverage_contract.py` | Coverage Taxonomy | Phase & capability coverage contract | contracts.learning_experience | `CANONICAL_ADAPT` | Definitional taxonomy of capability coverage. Migrated to `nexus_learning.coverage_contract`. |
| `nexus/learning/learning_coverage_probes.py` | audit tools | `test_learning_coverage_probes.py` | Coverage Inspection | Coverage verification probe | coverage_contract | `CANONICAL_ADAPT` | Probes capability execution coverage. Migrated to `nexus_learning.coverage_probes`. |
| `nexus/learning/outcome_memory.py` | local_heal, orchestrator | `test_outcome_memory_worker_write.py` | Outcome Storage | Stores verified episode outcomes | Standard library only | `CANONICAL_ADAPT` | Outcome storage and replay memory without route decision authority. Migrated to `nexus_learning.outcome_memory`. |
| `nexus/learning/retrieval_audit.py` | audit services | `test_retrieval_audit.py` | Audit Trail | Retrieval integrity & audit logs | Standard library only | `CANONICAL_ADAPT` | Audits retrieval provenance and receipt chains. Migrated to `nexus_learning.retrieval_audit`. |
| `nexus/contracts/learning_experience.py` | learning, local_heal | `test_learning_experience.py` | Contract Authority | Core Learning Experience Schemas | Standard library only | `CANONICAL_ADAPT` | Canonical schema and validator for episode/recommendation/validation/adoption/rollback. Migrated to `nexus_learning.contracts`. |
| `nexus/learning/cycle_analyzer.py` | services/refactor_engine | legacy tests | Refactor tool | Cycle dependency analysis | AST, networkx | `LEAVE_IN_NEXUS_NEW` | Architecture refactoring tool, not canonical learning lifecycle. |
| `nexus/learning/disk_janitor.py` | maintenance | test_disk_janitor | Maintenance | Disk cleanup for learning cache | shutil, os | `LEAVE_IN_NEXUS_NEW` | Maintenance janitor for legacy cache directories. |
| `nexus/learning/disk_policy.py` | maintenance | none | Maintenance | Disk quota policy | json | `LEAVE_IN_NEXUS_NEW` | Disk quota policy for legacy cache. |
| `nexus/learning/embedding_cache.py` | legacy retrieval | test_embedding_cache | Cache | Local vector embedding cache | sqlite3, numpy | `LEAVE_IN_NEXUS_NEW` | Experimental embedding cache; outside thin learning authority. |
| `nexus/learning/eternal_memory.py` | legacy memory | test_memory | Memory Lab | Long-term memory store | sqlite3 | `LAB_ONLY` | Experimental eternal memory storage. |
| `nexus/learning/external_skill_loader.py` | commands | test_skill | Skill loader | Loads external skill markdown | yaml, pathlib | `LEAVE_IN_NEXUS_NEW` | Skill packaging loader; non-core authority. |
| `nexus/learning/fair_skill_candidate_pool.py` | skill_fit | tests/skill | Skill Lab | Multi-armed bandit candidate pool | random | `LAB_ONLY` | Experimental bandit routing logic; prohibited in canonical learning. |
| `nexus/learning/federated_rag.py` | none | none | RAG Lab | Experimental federated retrieval | lancedb | `LAB_ONLY` | RAG experiment; prohibited from canonical core. |
| `nexus/learning/governance_mutants.py` | test hooks | test_governance | Test fixture | Synthetic governance mutations | copy | `LEAVE_IN_NEXUS_NEW` | Governance testing tool. |
| `nexus/learning/knowledge_index.py` | guards | test_knowledge | Index | Knowledge snippet inverted index | sqlite3 | `LEAVE_IN_NEXUS_NEW` | Keyword knowledge index for legacy bot. |
| `nexus/learning/latent_predictor_v20.py` | router hooks | test_predictor | Predictive model | Latent outcome predictor | math, torch | `LAB_ONLY` | Neural predictor for routing; prohibited in canonical learning. |
| `nexus/learning/lewm_predictor.py` | router hooks | test_lewm | Predictive model | Weight predictor | math | `LAB_ONLY` | Predictive routing model; prohibited in canonical learning. |
| `nexus/learning/metrics.py` | refactor_engine | test_metrics | Metrics | Code complexity metrics | radon/ast | `LEAVE_IN_NEXUS_NEW` | General code metrics calculator. |
| `nexus/learning/router_nas_tuner.py` | legacy router | test_nas | NAS Lab | Neural architecture search tuner | torch, optuna | `LAB_ONLY` | Directly mutates router architectures; strictly forbidden. |
| `nexus/learning/search_strategies.py` | retrieval | test_search | Search | Search strategies for skills | typing | `LEAVE_IN_NEXUS_NEW` | Skill search strategy donors. |
| `nexus/learning/sf2_bounded_probe.py` | diagnostic | test_probe | Probe | State-feedback probe | json | `LEAVE_IN_NEXUS_NEW` | Probe diagnostic tool. |
| `nexus/learning/shared_playbook.py` | app/mount | test_playbook | Playbook | Markdown playbook parser | yaml | `LEAVE_IN_NEXUS_NEW` | Static playbook loader. |
| `nexus/learning/skill_artifact.py` | skill registry | test_skill | Skill storage | Skill artifact serializer | json | `LEAVE_IN_NEXUS_NEW` | Skill artifact packaging. |
| `nexus/learning/skill_catalog.py` | skill registry | test_skill | Skill catalog | Dynamic skill catalogue | json | `LEAVE_IN_NEXUS_NEW` | Skill registry donor. |
| `nexus/learning/skill_discovery_lane.py` | skill discovery| test_skill | Skill Lab | Automatic skill discovery | re | `LAB_ONLY` | Experimental discovery lane. |
| `nexus/learning/skill_exchange.py` | p2p | test_skill | Skill Lab | P2P skill exchange protocol | aiohttp | `LAB_ONLY` | Experimental exchange. |
| `nexus/learning/skill_fit_ablation.py` | ablation | test_ablation | Ablation Lab | Skill ablation testing | json | `LAB_ONLY` | Skill fitting experiment. |
| `nexus/learning/skill_fit_ablation_core.py`| interceptor | test_interceptor| Ablation Lab | Ablation execution matrix | json | `LAB_ONLY` | Skill fitting experiment. |
| `nexus/learning/skill_fit_candidate_index.py`| candidate | test_candidate | Skill Lab | Candidate indexing | json | `LAB_ONLY` | Skill fitting experiment. |
| `nexus/learning/skill_fit_closure.py` | closure | test_closure | Skill Lab | Skill fitting closure loop | json | `LAB_ONLY` | Skill fitting experiment. |
| `nexus/learning/skill_fit_followup.py` | followup | test_followup | Skill Lab | Skill followup loop | json | `LAB_ONLY` | Skill fitting experiment. |
| `nexus/learning/skill_fit_promotion.py` | promotion | test_promotion | Skill Lab | Self-promotion of skills | json | `LAB_ONLY` | Prohibited self-promotion machinery. |
| `nexus/learning/skill_fit_status.py` | status | test_status | Skill Lab | Skill status queries | json | `LAB_ONLY` | Skill fitting experiment. |
| `nexus/learning/skill_inventory_roots.py`| roots | test_roots | Skill catalog | Skill file tree resolver | pathlib | `LEAVE_IN_NEXUS_NEW` | Path resolution helper. |
| `nexus/learning/skill_lifecycle.py` | lifecycle | test_lifecycle | Skill Lab | Skill activation/deactivation | json | `LAB_ONLY` | Mutates skill activation state. |
| `nexus/learning/skill_memory_index.py` | s2t_strict | test_skill_mem | S2T Memory | S2T skill execution record | sqlite3 | `LEAVE_IN_NEXUS_NEW` | S2T-specific memory index; donor in Nexus-new. |
| `nexus/learning/skill_registry.py` | tools/skills | test_registry | Skill registry | Central registry of skills | yaml, pathlib | `LEAVE_IN_NEXUS_NEW` | Legacy skill registry; donor in Nexus-new. |
| `nexus/learning/skill_route_taxonomy.py` | router | test_taxonomy | Taxonomy | Route to skill map | json | `LEAVE_IN_NEXUS_NEW` | Router-coupled taxonomy. |
| `nexus/learning/skill_scanner.py` | scanner | test_scanner | Skill scanner | AST scanner for skill declarations | ast | `LEAVE_IN_NEXUS_NEW` | Codebase scanner. |
| `nexus/learning/skill_schema.py` | schema | test_schema | Skill schema | Frontmatter schemas | pydantic | `LEAVE_IN_NEXUS_NEW` | Skill frontmatter schema. |
| `nexus/learning/skill_store.py` | store | test_store | Skill storage | Skill disk storage | pathlib | `LEAVE_IN_NEXUS_NEW` | Storage backend for legacy skills. |
| `nexus/learning/sota_searcher.py` | search | test_sota | SOTA Lab | Automated SOTA search | requests | `LAB_ONLY` | Web search experiment. |
| `nexus/learning/vector_cache.py` | vector | test_vector | Vector Lab | Memory-mapped vector cache | numpy | `LAB_ONLY` | Vector search experiment. |
| `nexus/learning/zero_trust_v2_behavior.py` | zero_trust | test_zt | Zero Trust Lab | Behavior profiling | json | `LAB_ONLY` | Experimental behavioral gate. |
| `nexus/learning/zero_trust_v2_behavior_adapter.py` | zero_trust | test_zt | Zero Trust Lab | Behavior adapter | json | `LAB_ONLY` | Experimental behavioral gate. |
| `nexus/learning/zero_trust_v2_clean_slate.py` | zero_trust | test_zt | Zero Trust Lab | Clean slate reset | json | `LAB_ONLY` | Experimental sandbox reset. |
| `nexus/learning/zero_trust_v2_physical_runner.py` | zero_trust | test_zt | Zero Trust Lab | Physical execution runner | subprocess | `LAB_ONLY` | Execution runner; belongs in runtime/lab. |
| `nexus/learning/zero_trust_v2_physical_sandbox.py` | zero_trust | test_zt | Zero Trust Lab | Physical sandbox isolation | docker/chroot | `LAB_ONLY` | Sandbox isolation; belongs in runtime/lab. |
| `nexus/learning/zero_trust_v2_promotion.py` | zero_trust | test_zt | Zero Trust Lab | Model/skill promotion | json | `LAB_ONLY` | Prohibited self-promotion authority. |
| `nexus/learning/zero_trust_v2_receipts.py` | zero_trust | test_zt | Zero Trust Lab | Receipt format | json | `LAB_ONLY` | Experimental receipts. |
| `nexus/learning/zero_trust_v2_sandbox.py` | zero_trust | test_zt | Zero Trust Lab | Mock sandbox | json | `LAB_ONLY` | Experimental sandbox. |
| `nexus/learning/zero_trust_v2_skill_gate.py` | zero_trust | test_zt | Zero Trust Lab | Skill gate enforcement | json | `LAB_ONLY` | Experimental skill gate. |
