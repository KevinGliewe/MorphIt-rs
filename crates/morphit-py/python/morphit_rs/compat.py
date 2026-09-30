"""The original MorphIt Python API on top of the Rust optimizer.

Scripts written for the Python MorphIt (https://github.com/HIRO-group/MorphIt-1)
switch by changing their imports::

    # from config import get_config, update_config_from_dict
    # from morphit import MorphIt
    # from training import train_morphit
    from morphit_rs.compat import get_config, update_config_from_dict, MorphIt, train_morphit

    config = get_config("MorphIt-B")
    config = update_config_from_dict(config, {"model.num_spheres": 20, "model.mesh_path": "link0.obj"})
    model = MorphIt(config)
    tracker = train_morphit(model, iteration_callback=lambda i, model, loss_info: ...)
    model.save_results()

Covered: `get_config`, `update_config_from_dict`, attribute access on the
config (`config.model.num_spheres = 20`), `MorphIt` (mesh, samples,
`centers`/`radii`/`masses`, `mesh_prep_report`, `save_results`,
`get_sphere_statistics`, `train`), `train_morphit` with `iteration_callback`
and the same `loss_info` keys, and `ConvergenceTracker` (`metrics`, `save`,
`analyze_convergence`, `plot_training_metrics`).

Differences:

- `centers`, `radii`, `masses` and the other arrays are torch tensors when torch
  is installed, NumPy arrays otherwise. They are read-only snapshots; assigning
  them is not supported, and the model is not an `nn.Module`.
- `model.device` defaults to `"auto"` (a GPU for large problems) instead of
  CUDA-or-CPU; `"cuda"`, `"cuda:N"` and `"mps"` are accepted.
- PyVista visualization (`pv_init`, `pv_render`, ...) is not available; use
  MorphIt Studio or the web viewer.
- The `results/evolution/*.json` logs are not written.
- Random streams differ from PyTorch's: results agree statistically, not bit
  for bit, with the Python code for the same seed (and bit for bit with every
  other MorphIt-rs front end).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ._native import Config, ConfigError, Mesh, Session, presets

__all__ = [
    "ConvergenceTracker",
    "MeshPrepReport",
    "MorphIt",
    "MorphItConfig",
    "get_config",
    "train_morphit",
    "update_config_from_dict",
]

_SECTIONS = ("model", "training", "visualization")
_TOP_LEVEL = ("results_dir", "output_filename", "random_seed")

LOSS_NAMES = (
    "coverage_loss",
    "overlap_penalty",
    "boundary_penalty",
    "surface_loss",
    "containment_loss",
    "sqem_loss",
    "hausdorff_loss",
    "mesh_containment_loss",
    "mass_loss",
    "com_loss",
    "inertia_loss",
    "flatness_loss",
)


def _torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415

        return torch
    except ImportError:
        return None


def _out(a: Any) -> Any:
    """A torch tensor when torch is installed (scripts call `.detach()`), else NumPy."""
    torch = _torch()
    a = np.ascontiguousarray(a)
    return torch.from_numpy(a) if torch is not None else a


class _Section:
    """One config section (`model`, `training`, `visualization`) with attribute access."""

    _name: str
    _fields: Tuple[str, ...]

    def __init__(self, name: str, values: Dict[str, Any]):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_fields", tuple(values))
        for k, v in values.items():
            object.__setattr__(self, k, v)

    def __setattr__(self, key: str, value: Any) -> None:
        if key not in self._fields:
            raise AttributeError(f"{self._name} has no parameter {key!r}")
        object.__setattr__(self, key, value)

    def __getattr__(self, key: str) -> Any:
        # Only called for missing attributes; the parameters are set in __init__.
        raise AttributeError(f"{self._name} has no parameter {key!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self._fields}

    def __repr__(self) -> str:
        return f"{self._name.capitalize()}Config({', '.join(f'{k}={getattr(self, k)!r}' for k in self._fields)})"


class MorphItConfig:
    """The original `MorphItConfig`: `model`, `training`, `visualization`,
    `results_dir`, `output_filename`, `random_seed`. Defaults come from the Rust
    optimizer's configuration (the same fields and defaults as Python's)."""

    model: _Section
    training: _Section
    visualization: _Section
    results_dir: str
    output_filename: str
    random_seed: Optional[int]

    def __init__(self, preset: str = "MorphIt-B"):
        d = Config(preset).to_dict()
        for s in _SECTIONS:
            object.__setattr__(self, s, _Section(s, d[s]))
        for k in _TOP_LEVEL:
            object.__setattr__(self, k, d[k])

    def __setattr__(self, key: str, value: Any) -> None:
        if key in _SECTIONS:
            if not isinstance(value, _Section):
                raise AttributeError(f"cannot replace the {key} section")
        elif key not in _TOP_LEVEL:
            raise AttributeError(f"MorphItConfig has no attribute {key!r}")
        object.__setattr__(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {s: getattr(self, s).to_dict() for s in _SECTIONS}
        d.update({k: getattr(self, k) for k in _TOP_LEVEL})
        return d

    def to_native(self) -> Config:
        """The equivalent `morphit_rs.Config` (raises `ValueError` for invalid values)."""
        c = Config.from_dict(self.to_dict())
        c.validate()
        return c

    def __repr__(self) -> str:
        return f"MorphItConfig({self.to_dict()!r})"


def get_config(loss_config: str = "MorphIt-B") -> MorphItConfig:
    """Defaults with the loss weights of a preset (MorphIt-V, -S, -B, -Obj, -Obj-mass)."""
    try:
        return MorphItConfig(loss_config)
    except ConfigError:
        raise ValueError(f"Unknown loss config: {loss_config}. Available: {presets()}") from None


def update_config_from_dict(config: MorphItConfig, updates: Dict[str, Any]) -> MorphItConfig:
    """Apply `{"section.param": value}` or top-level `{"random_seed": 0}` updates
    in place (in order) and return the same config. Unknown keys raise `ValueError`."""
    for key, value in updates.items():
        if "." in key:
            section, param = key.split(".", 1)
            if section not in _SECTIONS:
                raise ValueError(f"Unknown config section: {section}")
            sec = getattr(config, section)
            if param not in sec._fields:
                raise ValueError(f"Unknown parameter '{param}' in section '{section}'")
            setattr(sec, param, value)
        elif key in _TOP_LEVEL:
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return config


class MeshPrepReport:
    """What mesh preparation did: `action`, `reason`, body counts, face counts,
    volumes and `warnings` as attributes; `to_dict()` for the JSON form."""

    def __init__(self, d: Dict[str, Any]):
        self._d = dict(d)
        for k, v in self._d.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._d)

    def __repr__(self) -> str:
        return f"MeshPrepReport(action={self._d.get('action')!r}, reason={self._d.get('reason')!r})"


class MorphIt:
    """The original `MorphIt` model: loads and prepares the mesh, draws the
    samples and places the initial spheres on construction."""

    def __init__(self, config: Optional[MorphItConfig] = None):
        self.config = config if config is not None else get_config()
        self._build()

    def _build(self) -> None:
        native = self.config.to_native()
        self._built_config = native.to_dict()
        self._session = Session(Mesh.load(self.config.model.mesh_path), native)
        s = self._session
        self.device = s.device
        self.num_spheres = self.config.model.num_spheres
        self.mesh_path = self.config.model.mesh_path
        self.query_mesh = s.mesh
        self.mesh_prep_report = MeshPrepReport(s.mesh_prep)
        self.density = float(self.config.model.density)
        self.mesh_volume = float(self.query_mesh.volume)
        self.mesh_mass = self.mesh_volume * self.density
        self.mesh_com = _out(np.asarray(self.query_mesh.center_mass, dtype=np.float64))
        self.mesh_inertia = _out(np.asarray(self.query_mesh.inertia) * self.density)
        inside, surface = s.samples()
        self.inside_samples = _out(inside)
        self.surface_samples = _out(surface)

    @property
    def session(self) -> Session:
        """The underlying `morphit_rs.Session`."""
        return self._session

    @property
    def centers(self) -> Any:
        return _out(self._session.centers)

    @property
    def radii(self) -> Any:
        return _out(self._session.radii)

    @property
    def masses(self) -> Any:
        return _out(self._session.masses)

    def save_results(self, filename: Optional[str] = None) -> None:
        """Write `results_dir/filename` (default `output_filename`) in the Python JSON schema."""
        path = Path(self.config.results_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / (filename or self.config.output_filename)
        with open(filepath, "w") as f:
            json.dump(self._session.result().to_dict(), f, indent=4)
        print(f"Results saved to: {filepath}")

    def get_sphere_statistics(self) -> Dict[str, Any]:
        radii = np.asarray(self._session.radii)
        centers = np.asarray(self._session.centers)
        total = float(np.sum(4.0 / 3.0 * np.pi * radii**3))
        return {
            "num_spheres": int(len(radii)),
            "radius_stats": {
                "min": float(radii.min()),
                "max": float(radii.max()),
                "mean": float(radii.mean()),
                "std": float(radii.std()),
            },
            "total_sphere_volume": total,
            "mesh_volume": self.mesh_volume,
            "volume_ratio": total / self.mesh_volume,
            "center_bounds": {"min": centers.min(axis=0).tolist(), "max": centers.max(axis=0).tolist()},
        }

    def train(self, config_updates: Optional[Dict[str, Any]] = None) -> "ConvergenceTracker":
        return train_morphit(self, config_updates)

    def _no_viz(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "PyVista visualization is not part of morphit_rs; view results in MorphIt Studio "
            "(https://kevingliewe.github.io/MorphIt-rs/) or the web viewer"
        )

    pv_init = pv_render = pv_close = pv_screenshot = _no_viz
    initialize_render_thread = stop_render_thread = _no_viz


class ConvergenceTracker:
    """Training history in the original layout (`metrics`), with `save()` to
    `results/training_logs/{model_name}_training_log.json`."""

    def __init__(self, model_name: str, save_dir: str = "results/training_logs"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.metrics: Dict[str, Any] = {
            "total_loss": [],
            "component_losses": {name: [] for name in LOSS_NAMES},
            "gradient_info": {"position_grad_mag": [], "radius_grad_mag": []},
            "sphere_stats": {"num_spheres": [], "min_radius": [], "max_radius": [], "mean_radius": []},
            "iterations": [],
            "time_per_iteration": [],
            "density_control_events": [],
        }

    def update(
        self, iteration: int, loss_dict: Dict[str, Any], model: MorphIt, grad_info: Dict[str, float], time_taken: float
    ) -> None:
        m = self.metrics
        m["iterations"].append(iteration)
        m["total_loss"].append(loss_dict["total"])
        for key, value in loss_dict["components"].items():
            m["component_losses"][key].append(value)
        for key, value in grad_info.items():
            m["gradient_info"][key].append(value)
        radii = np.asarray(model.session.radii)
        m["sphere_stats"]["num_spheres"].append(int(len(radii)))
        m["sphere_stats"]["min_radius"].append(float(radii.min()))
        m["sphere_stats"]["max_radius"].append(float(radii.max()))
        m["sphere_stats"]["mean_radius"].append(float(radii.mean()))
        m["time_per_iteration"].append(time_taken)

    def record_density_control(self, iteration: int, spheres_added: int, spheres_removed: int) -> None:
        self.metrics["density_control_events"].append(
            {"iteration": iteration, "spheres_added": spheres_added, "spheres_removed": spheres_removed}
        )

    def save(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        filename = self.save_dir / f"{self.model_name}_training_log.json"
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Saved training metrics to {filename}")
        return filename

    def analyze_convergence(self, window_size: int = 20, threshold: float = 0.01) -> Dict[str, Any]:
        losses = self.metrics["total_loss"]
        if len(losses) < window_size:
            return {"converged": False, "reason": "Not enough data points"}
        recent = losses[-window_size:]
        loss_change = abs(recent[0] - recent[-1]) / max(abs(recent[0]), 1e-5)
        g = self.metrics["gradient_info"]
        pos = g["position_grad_mag"][-window_size:]
        rad = g["radius_grad_mag"][-window_size:]
        grad_small = sum(x < 1e-4 for x in pos) > window_size // 2 and sum(x < 1e-4 for x in rad) > window_size // 2
        return {"converged": loss_change < threshold or grad_small, "loss_change": loss_change, "grad_small": grad_small}

    def plot_training_metrics(self, save_fig: bool = True) -> Any:
        """Loss, components, gradients and sphere statistics (needs matplotlib)."""
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("plot_training_metrics needs matplotlib: pip install morphit-rs[plot]") from e
        m = self.metrics
        it = m["iterations"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].semilogy(it, m["total_loss"])
        axes[0, 0].set_title("Total loss")
        for name, values in m["component_losses"].items():
            if any(v != 0 for v in values):
                axes[0, 1].semilogy(it, np.abs(values), label=name)
        axes[0, 1].set_title("Weighted loss components")
        axes[0, 1].legend(fontsize=7)
        for name, values in m["gradient_info"].items():
            axes[1, 0].semilogy(it, values, label=name)
        axes[1, 0].set_title("Gradient magnitudes")
        axes[1, 0].legend(fontsize=7)
        for name in ("min_radius", "mean_radius", "max_radius"):
            axes[1, 1].plot(it, m["sphere_stats"][name], label=name)
        axes[1, 1].set_title("Radii")
        axes[1, 1].legend(fontsize=7)
        for ax in axes.flat:
            ax.set_xlabel("iteration")
        fig.tight_layout()
        if save_fig:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_dir / f"{self.model_name}_training_metrics.png", dpi=120)
        return fig


def train_morphit(
    model: MorphIt,
    config: Optional[Dict[str, Any]] = None,
    iteration_callback: Optional[Callable[[int, MorphIt, Dict[str, Any]], Any]] = None,
) -> ConvergenceTracker:
    """Train `model` and return its `ConvergenceTracker`.

    `config` holds dotted-key updates for `model.config`. `iteration_callback(
    iteration, model, loss_info)` runs after every step with `loss_info` keys
    `total_loss`, `weighted_losses`, `raw_losses`, `grad_info` and `iter_time`;
    exceptions it raises are reported as warnings and do not stop training.
    """
    if config is not None:
        update_config_from_dict(model.config, config)
    if model.config.to_native().to_dict() != model._built_config:
        # Training parameters changed after construction: rebuild (same seed -> same start).
        model._build()
    session = model.session
    tracker = ConvergenceTracker(Path(model.mesh_path).stem)
    verbose = int(model.config.training.verbose_frequency or 0)
    print("\n=== Starting MorphIt Training ===")
    for step in session:
        grad_info = {"position_grad_mag": step.position_grad_mag, "radius_grad_mag": step.radius_grad_mag}
        loss_info: Dict[str, Any] = {
            "total_loss": step.total_loss,
            "weighted_losses": step.weighted_losses,
            "raw_losses": step.raw_losses,
            "grad_info": grad_info,
            "iter_time": step.seconds,
        }
        tracker.update(
            step.iteration,
            {"total": step.total_loss, "components": step.weighted_losses},
            model,
            grad_info,
            step.seconds,
        )
        if iteration_callback is not None:
            try:
                iteration_callback(step.iteration, model, loss_info)
            except Exception as e:  # noqa: BLE001 - the original swallows callback errors too
                warnings.warn(f"iteration_callback raised {e!r}", RuntimeWarning, stacklevel=2)
        if verbose and step.iteration % verbose == 0:
            print(f"\n[Iter {step.iteration}] Time: {step.seconds:.4f}s")
            print(f"Total Loss: {step.total_loss:.6f}")
            print(f"Spheres: {step.num_spheres}")
        dc = step.density_control
        if dc is not None:
            tracker.record_density_control(step.iteration, dc["added"], dc["removed"])
        if step.converged:
            print("\n=== Training Converged ===")
            print(f"Stopping at iteration {step.iteration}")
    removed = session.finalize()
    if removed:
        print(f"\n[final prune] removed {removed} sphere(s) whose centers escaped the mesh "
              f"or that lay inside another sphere")
    model.num_spheres = session.num_spheres
    print("\n=== Training Complete ===")
    print(f"Density control operations: {len(tracker.metrics['density_control_events'])}")
    print(f"Final sphere count: {model.num_spheres}")
    print("=" * 26)
    return tracker
