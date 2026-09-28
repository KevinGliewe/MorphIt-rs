"""MorphIt: approximate triangle meshes and robots with spheres.

The optimizer is the Rust port of MorphIt (https://github.com/HIRO-group/MorphIt-1)
with results in the same JSON schema. Long computations release the GIL.

    import morphit_rs as mi

    mesh = mi.Mesh.load("bunny.obj")
    config = mi.Config.preset("MorphIt-B", num_spheres=64, seed=42)
    result = mi.pack(mesh, config)
    result.centers, result.radii          # NumPy arrays
    result.save("bunny.json")

`morphit_rs.compat` mirrors the original Python API (`get_config`,
`update_config_from_dict`, `MorphIt`, `train_morphit`).
"""

from __future__ import annotations

import atexit
import logging as _logging
import threading as _threading

from ._native import (
    BusyError,
    CancelledError,
    Config,
    ConfigError,
    Mesh,
    MeshError,
    MorphItError,
    MorphItIOError,
    PackResult,
    RobotError,
    RobotPackage,
    Session,
    StateError,
    StepInfo,
    aggregate_overall,
    devices,
    evaluate_packing,
    example_objects,
    example_robots,
    link_quality,
    object_mjcf,
    object_urdf,
    pack,
    presets,
    set_num_threads,
    spheres_from_object_urdf,
    version,
)
from . import _native

__version__ = version()

__all__ = [
    "BusyError",
    "CancelledError",
    "Config",
    "ConfigError",
    "Mesh",
    "MeshError",
    "MorphItError",
    "MorphItIOError",
    "PackResult",
    "RobotError",
    "RobotPackage",
    "Session",
    "StateError",
    "StepInfo",
    "aggregate_overall",
    "devices",
    "enable_logging",
    "evaluate_packing",
    "example_objects",
    "example_robots",
    "link_quality",
    "object_mjcf",
    "object_urdf",
    "pack",
    "presets",
    "set_num_threads",
    "spheres_from_object_urdf",
    "version",
]

_log_thread: _threading.Thread | None = None
_log_stop = _threading.Event()


def _flush_logs() -> None:
    for level, target, message in _native._drain_logs():
        _logging.getLogger("morphit_rs." + target.replace("::", ".")).log(level, message)


def _pump(interval: float) -> None:
    while not _log_stop.wait(interval):
        _flush_logs()


def enable_logging(level: str = "INFO", interval: float = 0.2) -> None:
    """Forward the library's log messages (mesh preparation, GPU fallback,
    pruning, ...) to Python's `logging`, under the `morphit_rs` logger.

    Records at `level` and above are collected; a daemon thread hands them to
    `logging` every `interval` seconds (and once more at exit). Configure
    handlers as usual, e.g. `logging.basicConfig(level=logging.INFO)`.
    """
    global _log_thread
    _native._install_logging(level)
    if _log_thread is None:
        _log_thread = _threading.Thread(target=_pump, args=(interval,), name="morphit-logs", daemon=True)
        _log_thread.start()
        atexit.register(_flush_logs)
