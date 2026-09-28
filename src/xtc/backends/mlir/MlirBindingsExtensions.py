#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Optional XTC extensions to the MLIR Python bindings."""

from __future__ import annotations

import importlib
import logging
import threading
from types import ModuleType

logger = logging.getLogger(__name__)

# Optional XTC extensions to the MLIR Python bindings: the module to import
# mapped to the pass-pipeline entries it contributes.
_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "mlir.xtc_transform": (
        "func.func(reduce-extract-slices)",
        "scf-add-vector-acc",
    ),
}

# Reverse map from a contributed pass to its providing module, so a pass can be
# gated without the caller naming its extension.
_PASS_OWNER: dict[str, str] = {}
for _module, _module_passes in _EXTENSIONS.items():
    for _pass in _module_passes:
        assert _pass not in _PASS_OWNER, f"pass {_pass!r} declared by two extensions"
        _PASS_OWNER[_pass] = _module

# Extension modules that imported successfully, resolved once at import time.
_loaded: dict[str, ModuleType] = {}
for _module in _EXTENSIONS:
    try:
        _loaded[_module] = importlib.import_module(_module)
    except ImportError as _exc:
        logger.debug("MLIR binding extension %r unavailable: %s", _module, _exc)


def passes(pass_names: list[str]) -> list[str]:
    """Return the subset of ``pass_names`` whose extension is available."""
    for pass_name in pass_names:
        if pass_name not in _PASS_OWNER:
            raise KeyError(f"unknown extension pass: {pass_name!r}")
    return [p for p in pass_names if _PASS_OWNER[p] in _loaded]


def module(module_name: str, warning: str | None = None) -> ModuleType | None:
    """Return the loaded extension module, or None if unavailable.

    If ``warning`` is given and the module is missing, log
    ``"<module> module not installed, <warning>"`` once per process.
    """
    if module_name not in _EXTENSIONS:
        raise KeyError(f"unknown extension module: {module_name!r}")
    loaded = _loaded.get(module_name)
    if loaded is None and warning is not None:
        _warn_once(module_name, warning)
    return loaded


# (module, warning) pairs already warned about, shared by compile threads.
_warned: set[tuple[str, str]] = set()
_warned_lock = threading.Lock()


def _warn_once(module_name: str, warning: str) -> None:
    with _warned_lock:
        if (module_name, warning) in _warned:
            return
        _warned.add((module_name, warning))
    logger.warning("%s module not installed, %s", module_name, warning)
