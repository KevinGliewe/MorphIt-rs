import json

import pytest

import morphit_rs as mi


def test_presets():
    assert mi.presets() == ["MorphIt-V", "MorphIt-S", "MorphIt-B", "MorphIt-Obj", "MorphIt-Obj-mass"]
    assert mi.Config.presets() == mi.presets()
    v = mi.Config("MorphIt-V")
    assert v["training.coverage_weight"] == 5000
    assert mi.Config() == mi.Config.preset("MorphIt-B")
    with pytest.raises(mi.ConfigError):
        mi.Config("MorphIt-X")


def test_preset_shortcuts():
    c = mi.Config.preset("MorphIt-S", num_spheres=12, iterations=40, seed=3, device="cpu")
    assert c["model.num_spheres"] == 12
    assert c["training.iterations"] == 40
    assert c["random_seed"] == 3
    assert c["model.device"] == "cpu"


def test_dotted_get_set():
    c = mi.Config()
    c["training.center_lr"] = 0.001
    c["model.num_spheres"] = 32.0  # an integral float is accepted for an int field
    c["random_seed"] = None
    assert c["training.center_lr"] == 0.001
    assert c["model.num_spheres"] == 32
    assert c["random_seed"] is None
    with pytest.raises(mi.ConfigError) as e:
        c["training.nope"] = 1
    assert e.value.key == "training.nope"
    assert isinstance(e.value, ValueError)
    with pytest.raises(mi.ConfigError):
        c["model.num_spheres"] = "many"


def test_update_is_all_or_nothing():
    c = mi.Config()
    with pytest.raises(mi.ConfigError):
        c.update({"model.num_spheres": 5, "model.nope": 1})
    assert c["model.num_spheres"] == mi.Config()["model.num_spheres"]
    c.update({"model.num_spheres": 5, "training.iterations": 10})
    assert (c["model.num_spheres"], c["training.iterations"]) == (5, 10)


def test_round_trips():
    c = mi.Config.preset(num_spheres=9, seed=1)
    d = c.to_dict()
    assert set(d) >= {"model", "training", "visualization", "results_dir", "output_filename", "random_seed"}
    assert mi.Config.from_dict(d) == c
    assert mi.Config.from_json(c.to_json()) == c
    assert json.loads(c.to_json())["model"]["num_spheres"] == 9
    partial = mi.Config.from_dict({"model": {"num_spheres": 3}})
    assert partial["model.num_spheres"] == 3
    cp = c.copy()
    cp["model.num_spheres"] = 1
    assert c["model.num_spheres"] == 9


def test_validate():
    c = mi.Config()
    c["model.num_spheres"] = 0
    with pytest.raises(mi.ConfigError):
        c.validate()
