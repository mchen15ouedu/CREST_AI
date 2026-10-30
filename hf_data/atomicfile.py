"""Atomic writes into the shared on-disk stores (forcing PQF, param clips, obs).

A writer renders into a temp file beside the target and os.replace()s it on,
so a concurrent reader (EF5, another run) sees the old file or the new one,
never half of one. The temp name is unique per CALL, not per process: the
dashboard server writes from threads and the fleet from worker processes, and
nested basins clip the same hour into one store - with a shared name one
writer renamed the other's half-written file away.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager


@contextmanager
def atomic_path(path: str):
    """Yield a fresh temp path beside `path`. On a clean exit it replaces
    `path`; on any error the temp file is deleted and the error re-raised."""
    d, base = os.path.split(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=base + ".", suffix=".tmp")
    os.close(fd)                    # the writer (pyarrow, GDAL) opens the path itself
    os.chmod(tmp, 0o644)            # mkstemp's 0600 would carry onto the store file
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
