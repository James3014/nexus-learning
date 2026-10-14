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
- For mutation work tracked by a repository-local GitHub Issue, run `nexus-certify issue-init --issue <N> --evidence-universe trusted-config` before relying on Issue-bound completion evidence, and run `nexus-certify issue-check --issue <N> --evidence-universe trusted-config --require-trusted-config` before claiming engineering completion (the same mode as the CI gate; the Issue needs no `NEXUS_CORE_EVIDENCE_UNIVERSE` marker).
- This binding is Evidence Trust + Completion only. It does not select the execution lane, route, worker/model, Candidate acceptance, merge, release, deployment, or production authority.
- DIRECT work remains transport-neutral. A Core mutation session is not required solely because repository files are being changed.

## Handoff

`HANDOFF.md` at the repository root is the shared progress log for every agent (Claude, Codex, others) and every device. It records state only; it grants no authority and never overrides this file, Issues, or Task Cards.

- **Read first.** At session start read `HANDOFF.md`. If its recorded HEAD is behind, run `git log <recorded-HEAD>..HEAD` and correct stale entries before acting on them.
- **Update at milestones.** Update it when a meaningful step finishes or before the session ends — not every turn. Overwrite stale state; do not append history.
- **Format.** The header line holds date, author agent, and current short HEAD. `▶ 下一步` has exactly one item. Every todo carries an Issue number (`#N`); items without one go under `待確認`. Record why, blockers, and next — not changed-file lists (git already has those).
- **Size cap: 150 lines.** Move finished or expired content to `docs/archive/HANDOFF-YYYY-MM.md`.
- **Read back.** After writing, re-read `HANDOFF.md` to confirm the change actually landed.
- **No secrets.** Never put tokens, passwords, credentials, or internal hostnames in it.
