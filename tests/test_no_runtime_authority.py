from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = ("nexus", "product", "scripts", "runtimes")
FORBIDDEN_KEYWORDS = (
    "CapabilityPlanner",
    "workforce_admission",
    "worker_registry",
    "open_swe",
    "nexus_open_swe_runtime",
)


def test_no_runtime_or_planner_authority():
    repo_root = Path(__file__).resolve().parent.parent
    pkg_dir = repo_root / "nexus_learning"
    assert pkg_dir.is_dir()

    violations = []
    for py_file in pkg_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_ROOTS:
                        violations.append(f"{py_file}:{node.lineno}: import {alias.name}")
                    for kw in FORBIDDEN_KEYWORDS:
                        if kw.lower() in alias.name.lower():
                            violations.append(
                                f"{py_file}:{node.lineno}: import {alias.name} (matches {kw})"
                            )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in FORBIDDEN_ROOTS:
                        violations.append(f"{py_file}:{node.lineno}: from {node.module} import ...")
                    for kw in FORBIDDEN_KEYWORDS:
                        if kw.lower() in node.module.lower():
                            violations.append(
                                f"{py_file}:{node.lineno}: from {node.module} (matches {kw})"
                            )

    assert not violations, (
        "Found forbidden runtime/authority dependencies in nexus_learning:\n"
        + "\n".join(violations)
    )
