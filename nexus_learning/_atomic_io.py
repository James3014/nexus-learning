"""Atomic text replacement for learning state files (shared I/O helper)."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def _atomic_write_text(path: Path, text: str) -> Path:
    """Replace ``path`` with ``text`` via a same-directory temp file and ``os.replace``.

    If the write or replace fails, the prior file is left intact and the temp file is removed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return target
