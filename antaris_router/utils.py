"""Shared utilities for antaris-router."""

import json
import os
import tempfile


def atomic_write_json(path: str, data, indent: int = 2) -> None:
    """Write JSON atomically using tmp file + os.replace.
    
    Prevents torn/partial writes from crashes or interrupted I/O.
    Does NOT prevent lost updates from concurrent writers.
    """
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        # Best-effort directory fsync for crash-consistent rename on POSIX
        try:
            dir_fd = os.open(dir_path, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
