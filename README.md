# Nexus Learning

Nexus Learning is the standalone evidence-bounded learning and adaptation system extracted from Nexus-new.

The package is the single owner for the Storage, Reflection, Measurement and Adoption stages of the learning loop. Retrieval is a read-only port in nexus-runtime, not part of this package. Research-governance modules are frozen (bug fixes only) and have no consumer outside this repository. See [Architecture Boundary (v2)](docs/architecture/BOUNDARY.md) for stage ownership and invariants.

## Lifecycle Boundary

```text
verified outcome evidence
        ↓
learning episode
        ↓
evaluation / effectiveness
        ↓
bounded recommendation
        ↓
independent validation
        ↓
external authority may adopt
```

**Recommendation != Authority**

Learning never self-promotes models, mutates CapabilityPlanner, alters workforce admission, or approves changes.

Experiment run identity, cohort preflight/staged stop gates, and campaign closeout evidence are documented in [Experiment Evidence Contracts](docs/architecture/EXPERIMENT_EVIDENCE_CONTRACTS.md).

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check nexus_learning tests
```
