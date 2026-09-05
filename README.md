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

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check nexus_learning tests
```
