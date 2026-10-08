"""BOUNDARY v2 enforcement: frozen research-governance modules and classification."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG_DIR = REPO / "nexus_learning"
DOC = REPO / "docs" / "architecture" / "BOUNDARY.md"
MARKER_NAME = "LEARNING_BOUNDARY_STATUS"
MARKER_VALUE = "FROZEN_RESEARCH_GOVERNANCE"

FROZEN = {
    "campaign_closeout",
    "campaign_gates",
    "experiment_preflight",
    "experiment_run_identity",
    "necessary_condition_stop",
    "research_frontier",
    "workflow_friction",
}
CORE = {
    "contracts",
    "outcome_memory",
    "closure_effectiveness",
    "episode_projection",
    "effectiveness_measurement",
    "coverage_contract",
    "coverage_probes",
    "retrieval_audit",
    "state_root",
    "experiment_integrity",
    "lessons",
    "adoption",
}

_DOC_BULLET = re.compile(r"^- `nexus_learning\.([a-z_]+)`$")


def _imported_package_modules(path: Path) -> set[str]:
    """Return nexus_learning submodule names imported by the file at ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "nexus_learning":
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] == "nexus_learning":
                if len(parts) >= 2:
                    found.add(parts[1])
                else:
                    # ``from nexus_learning import campaign_gates`` form
                    found.update(alias.name for alias in node.names)
    return found


def test_frozen_modules_carry_marker() -> None:
    for name in sorted(FROZEN):
        module = importlib.import_module(f"nexus_learning.{name}")
        assert getattr(module, MARKER_NAME, None) == MARKER_VALUE, name


def test_core_modules_have_no_marker() -> None:
    for name in sorted(CORE):
        module = importlib.import_module(f"nexus_learning.{name}")
        assert not hasattr(module, MARKER_NAME), name


def test_doc_lists_exactly_frozen_modules() -> None:
    listed = {
        match.group(1)
        for line in DOC.read_text(encoding="utf-8").splitlines()
        if (match := _DOC_BULLET.match(line))
    }
    assert listed == FROZEN


def test_core_never_imports_frozen() -> None:
    violations = {
        name: sorted(_imported_package_modules(PKG_DIR / f"{name}.py") & FROZEN)
        for name in sorted(CORE)
    }
    violations = {name: deps for name, deps in violations.items() if deps}
    assert not violations, violations


def test_every_package_module_is_classified() -> None:
    modules = {path.stem for path in PKG_DIR.glob("*.py") if path.name != "__init__.py"}
    unclassified = modules - CORE - FROZEN
    assert not unclassified, f"classify in CORE or FROZEN: {sorted(unclassified)}"
    assert CORE.isdisjoint(FROZEN)
