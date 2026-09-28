"""The original MorphIt API (ported from its test_morphit.py where it applies)."""

import json
import shutil

import numpy as np
import pytest

from morphit_rs.compat import (
    ConvergenceTracker,
    MorphIt,
    MorphItConfig,
    get_config,
    train_morphit,
    update_config_from_dict,
)

from conftest import CORE_FIXTURES


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    shutil.copy(CORE_FIXTURES / "link0.obj", tmp_path / "link0.obj")
    shutil.copy(CORE_FIXTURES / "mesh_prep" / "two_overlapping.obj", tmp_path / "boxes.obj")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def small(config: MorphItConfig, mesh="link0.obj") -> MorphItConfig:
    return update_config_from_dict(
        config,
        {
            "model.mesh_path": mesh,
            "model.num_spheres": 6,
            "model.num_inside_samples": 800,
            "model.num_surface_samples": 800,
            "model.device": "cpu",
            "training.iterations": 20,
            "training.verbose_frequency": 1000,
            "random_seed": 0,
        },
    )


def test_configuration():
    for name in ("MorphIt-V", "MorphIt-S", "MorphIt-B"):
        assert isinstance(get_config(name), MorphItConfig)
    c = get_config("MorphIt-B")
    same = update_config_from_dict(c, {"model.num_spheres": 10, "training.iterations": 50, "results_dir": "x"})
    assert same is c
    assert (c.model.num_spheres, c.training.iterations, c.results_dir) == (10, 50, "x")
    c.model.num_spheres = 11  # attribute access as in the original
    assert c.to_dict()["model"]["num_spheres"] == 11
    with pytest.raises(ValueError):
        update_config_from_dict(c, {"model.bogus": 1})
    with pytest.raises(ValueError):
        update_config_from_dict(c, {"nosection.x": 1})
    with pytest.raises(ValueError):
        get_config("MorphIt-X")
    with pytest.raises(AttributeError):
        c.model.bogus = 1


def test_model_creation(workdir):
    model = MorphIt(small(get_config()))
    for attr in ("centers", "radii", "query_mesh", "inside_samples", "surface_samples"):
        assert hasattr(model, attr)
    assert model.num_spheres == 6
    assert np.asarray(model.centers).shape == (6, 3)
    stats = model.get_sphere_statistics()
    assert set(stats) == {"num_spheres", "radius_stats", "total_sphere_volume", "mesh_volume", "volume_ratio", "center_bounds"}
    with pytest.raises(NotImplementedError):
        model.pv_init()


def test_mesh_prep(workdir):
    model = MorphIt(small(get_config(), "boxes.obj"))
    assert model.mesh_prep_report.action == "unioned"
    model.save_results()
    saved = json.loads((workdir / "results" / "output" / "morphit_results.json").read_text())
    assert saved["mesh_prep"]["action"] == "unioned"
    off = MorphIt(update_config_from_dict(small(get_config(), "boxes.obj"), {"model.union_overlapping_bodies": False}))
    assert off.mesh_prep_report.action == "disabled"


def test_minimal_training(workdir):
    model = MorphIt(small(get_config()))
    infos = []
    tracker = train_morphit(model, iteration_callback=lambda i, m, info: infos.append((i, info)))
    assert isinstance(tracker, ConvergenceTracker)
    assert len(infos) == 20 and infos[0][0] == 0
    assert set(infos[0][1]) == {"total_loss", "weighted_losses", "raw_losses", "grad_info", "iter_time"}
    assert set(infos[0][1]["grad_info"]) == {"position_grad_mag", "radius_grad_mag"}
    m = tracker.metrics
    assert len(m["iterations"]) == len(m["total_loss"]) == 20
    assert len(m["component_losses"]["coverage_loss"]) == 20
    log = tracker.save()
    assert log.resolve() == (workdir / "results" / "training_logs" / "link0_training_log.json").resolve()
    model.save_results("test_results.json")
    data = json.loads((workdir / "results" / "output" / "test_results.json").read_text())
    assert list(data) == ["centers", "radii", "masses", "mesh_path", "num_spheres", "per_sphere_mass", "mesh_prep", "config"]
    assert data["num_spheres"] == model.num_spheres


def test_callback_errors_do_not_stop_training(workdir):
    model = MorphIt(small(get_config()))

    def bad(i, m, info):
        raise RuntimeError("boom")

    with pytest.warns(RuntimeWarning):
        tracker = train_morphit(model, iteration_callback=bad)
    assert len(tracker.metrics["iterations"]) == 20


def test_train_applies_updates_and_is_deterministic(workdir):
    a = MorphIt(small(get_config()))
    a.train({"training.iterations": 12})
    b = MorphIt(update_config_from_dict(small(get_config()), {"training.iterations": 12}))
    train_morphit(b)
    assert np.array_equal(np.asarray(a.centers), np.asarray(b.centers))


def test_torch_tensors_when_available(workdir):
    torch = pytest.importorskip("torch")
    model = MorphIt(small(get_config()))
    assert isinstance(model.centers, torch.Tensor)
    assert model.centers.detach().cpu().numpy().shape == (6, 3)
