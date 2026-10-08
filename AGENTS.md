# Nexus Learning Agent Guidelines

nexus-learning owns:
- Learning-loop Storage: episode projection, outcome memory, retrieval audit, explicit state roots
- Learning-loop Reflection: canonical lesson schema and lesson store (Phase 1)
- Learning-loop Measurement: closure effectiveness, paired memory uplift, coverage contract and probes
- Learning-loop Adoption: evidence-bounded recommendation, independent validation, governed adoption and rollback contracts
- Frozen research-governance contracts (see docs/architecture/BOUNDARY.md): bug fixes only

nexus-learning does NOT own:
- lesson retrieval at execution time (nexus-runtime LearningReadPort)
- vector indexes or provider calls
- route selection
- CapabilityPlanner mutation
- worker/model selection
- Workforce admission mutation
- model promotion
- recommendation approval
- merge / release authority
- production / public claims

Ordinary bounded engineering does not require Task Cards.

Escalate only on:
- public protocol change
- authority semantic change
- security boundary change
- cross-repo contract break
- release / production claim


## Nexus Core issue-bound completion evidence

- This repository is enrolled in the standalone `nexus-certify` Golden Path through `.nexus-core/config.toml`.
- For mutation work tracked by a repository-local GitHub Issue, run `nexus-certify issue-init --issue <N>` before relying on Issue-bound completion evidence, and run `nexus-certify issue-check --issue <N>` before claiming engineering completion.
- This binding is Evidence Trust + Completion only. It does not select the execution lane, route, worker/model, Candidate acceptance, merge, release, deployment, or production authority.
- DIRECT work remains transport-neutral. A Core mutation session is not required solely because repository files are being changed.
