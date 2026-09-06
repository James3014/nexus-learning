"""Smoke test for installed standalone nexus-learning wheel."""

import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    outside = Path(tempfile.gettempdir()).resolve()
    os.chdir(outside)

    import nexus_learning
    from nexus_learning.contracts import build_nexus_learning_episode
    from nexus_learning.state_root import LearningStateRoot, resolve_learning_state_root

    pkg_file = Path(nexus_learning.__file__).resolve()
    print(f"[smoke] Loaded nexus_learning from: {pkg_file}")

    if "site-packages" not in str(pkg_file):
        print(f"[smoke] ERROR: nexus_learning was not imported from site-packages: {pkg_file}", file=sys.stderr)
        return 1

    episode = build_nexus_learning_episode(
        task_id="smoke-task",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
    )
    if not episode.get("episode_id"):
        print("[smoke] ERROR: Failed to build valid learning episode", file=sys.stderr)
        return 1

    root = resolve_learning_state_root("/tmp/test_smoke")
    if not isinstance(root, LearningStateRoot):
        print("[smoke] ERROR: Failed to resolve LearningStateRoot", file=sys.stderr)
        return 1

    print("[smoke] SUCCESS: nexus-learning smoke verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
