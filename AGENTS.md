# Nexus Learning Agent Guidelines

nexus-learning owns:
- Learning episode projection and normalization
- Learning closure effectiveness measurement
- Coverage contract and probes
- Evidence-bounded policy recommendation lifecycle
- Independent recommendation validation schemas
- Governed adoption and rollback contracts
- Outcome memory storage and retrieval audit

nexus-learning does NOT own:
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
