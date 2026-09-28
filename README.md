# Nexus Learning

Nexus Learning is the standalone evidence-bounded learning and adaptation system extracted from Nexus-new.

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
