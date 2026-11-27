"""Publish a staged directory without overwriting any existing directory entry."""

import ctypes
import errno
import os
from pathlib import Path
import sys


def publish_directory(stage: Path, destination: Path) -> None:
    """Atomically move a complete directory on the same filesystem, or fail."""
    if os.name == "nt":
        stage.rename(destination)  # Windows rename always refuses an existing target.
        return

    paths = (os.fsencode(stage), os.fsencode(destination))
    if sys.platform == "linux":
        symbol = "renameat2"
        # AT_FDCWD for both paths, RENAME_NOREPLACE.
        arguments = (-100, paths[0], -100, paths[1], 1)
        types = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    elif sys.platform == "darwin":
        symbol = "renamex_np"
        arguments = (*paths, 4)  # RENAME_EXCL
        types = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    else:
        raise OSError(errno.ENOTSUP, "Exclusive directory rename unavailable", destination)

    rename = getattr(ctypes.CDLL(None, use_errno=True), symbol, None)
    if rename is None:
        raise OSError(errno.ENOTSUP, f"{symbol} unavailable", destination)
    rename.argtypes = types
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)
