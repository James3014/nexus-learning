from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = (
    "nexus",
    "product",
    "scripts",
    "runtimes",
    "nexus_core",
    "nexus_runtime",
    "repository_intelligence",
    "nexus_open_swe_runtime",
)
FORBIDDEN_KEYWORDS = (
    "CapabilityPlanner",
    "workforce_admission",
    "worker_registry",
    "open_swe",
    "nexus_open_swe_runtime",
)


def _scan(text: str, filename: str) -> list[str]:
    violations = []
    tree = ast.parse(text, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_ROOTS:
                    violations.append(f"{filename}:{node.lineno}: import {alias.name}")
                for kw in FORBIDDEN_KEYWORDS:
                    if kw.lower() in alias.name.lower():
                        violations.append(
                            f"{filename}:{node.lineno}: import {alias.name} (matches {kw})"
                        )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_ROOTS:
                    violations.append(f"{filename}:{node.lineno}: from {node.module} import ...")
                for kw in FORBIDDEN_KEYWORDS:
                    if kw.lower() in node.module.lower():
                        violations.append(
                            f"{filename}:{node.lineno}: from {node.module} (matches {kw})"
                        )
    return violations


def test_no_runtime_or_planner_authority():
    repo_root = Path(__file__).resolve().parent.parent
    pkg_dir = repo_root / "nexus_learning"
    assert pkg_dir.is_dir()

    violations = []
    for py_file in pkg_dir.rglob("*.py"):
        violations.extend(_scan(py_file.read_text(encoding="utf-8"), str(py_file)))

    assert not violations, (
        "Found forbidden runtime/authority dependencies in nexus_learning:\n"
        + "\n".join(violations)
    )


def test_guard_blocks_cross_repo_runtime_roots():
    snippets = {
        "import nexus_core.evidence": "import nexus_core.evidence\n",
        "from nexus_core import kernel": "from nexus_core import kernel\n",
        "import nexus_runtime.kernel": "import nexus_runtime.kernel\n",
        "from nexus_runtime.planning import x": "from nexus_runtime.planning import x\n",
        "import repository_intelligence.impact": "import repository_intelligence.impact\n",
        "from repository_intelligence import impact": "from repository_intelligence import impact\n",
        "import nexus_open_swe_runtime": "import nexus_open_swe_runtime\n",
    }
    missed = [label for label, text in snippets.items() if not _scan(text, "<snippet>")]
    assert not missed, f"guard does not flag: {missed}"
