"""Test that built standalone nexus-learning wheel installs and executes cleanly."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_installed_wheel_smoke_in_isolated_venv(tmp_path: Path):
    repo_root = Path(__file__).parent.parent.resolve()
    dist_dir = repo_root / "dist"
    wheels = sorted(dist_dir.glob("nexus_learning-*.whl"))
    if not wheels:
        pytest.fail("No nexus-learning wheel found in dist/. Run 'uv build' before running this test.")

    target_wheel = wheels[-1]

    # Create isolated venv
    venv_dir = tmp_path / "smoke_venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_pip = venv_dir / "bin" / "pip"
    venv_python = venv_dir / "bin" / "python"

    # Install the wheel
    subprocess.run([str(venv_pip), "install", str(target_wheel)], check=True)

    # Run smoke probe outside the repo checkout
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    probe_script = outside_dir / "probe.py"
    probe_script.write_text(
        """\
import os
import sys
from pathlib import Path

import nexus_learning
from nexus_learning.contracts import build_nexus_learning_episode
from nexus_learning.state_root import LearningStateRoot, resolve_learning_state_root

pkg_path = Path(nexus_learning.__file__).resolve()
print("PKG_FILE:", pkg_path)
assert "site-packages" in str(pkg_path), f"Not in site-packages: {pkg_path}"

episode = build_nexus_learning_episode(
    task_id="probe-task",
    terminal_outcome="SUCCEEDED",
    terminal_evidence={"verifier_status": "PASS"},
)
assert episode["episode_id"].startswith("lep:")

root = resolve_learning_state_root("/tmp/probe_project")
assert isinstance(root, LearningStateRoot)
print("PROBE_OK")
"""
    )

    res_probe = subprocess.run(
        [str(venv_python), str(probe_script)],
        cwd=str(outside_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PROBE_OK" in res_probe.stdout
    assert str(repo_root) not in res_probe.stdout
