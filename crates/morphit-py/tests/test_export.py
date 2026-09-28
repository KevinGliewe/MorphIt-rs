import shutil
import subprocess

import numpy as np
import pytest

import morphit_rs as mi

from conftest import REPO, ROBOT_FIXTURES, load_json


def test_object_models_match_python():
    cases = load_json(ROBOT_FIXTURES / "py_object_model.json")
    for name, c in cases.items():
        if c["rgba"] != [0.2, 0.6, 1.0, 1.0]:
            continue  # only the default color is expressible as #rrggbb exactly
        for anchored, key in ((False, ""), (True, "_anchored")):
            urdf, centroid = mi.object_urdf(c["centers"], c["radii"], name=c["robot_name"], anchored=anchored)
            assert urdf == c["urdf" + key], name
            assert list(centroid) == c["centroid"], name
            mjcf, _ = mi.object_mjcf(c["centers"], c["radii"], name=c["robot_name"], anchored=anchored)
            assert mjcf == c["mjcf" + key], name


def test_spheres_from_urdf_round_trip():
    centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.05]])
    radii = np.array([0.02, 0.03])
    urdf, _ = mi.object_urdf(centers, radii, decimals=9)
    c, r = mi.spheres_from_object_urdf(urdf)
    assert np.allclose(c, centers) and np.allclose(r, radii)
    with pytest.raises(ValueError):
        mi.object_urdf(centers, radii, color="blue")


def test_evaluate_packing(link0, small_config):
    r = mi.pack(link0, small_config)
    q = mi.evaluate_packing(link0, r, surface_samples=2000, volume_samples=2000)
    assert q["actual_n"] == r.num_spheres
    assert 0.0 < q["r_in"] <= 1.0
    with pytest.raises(ValueError):
        mi.evaluate_packing(link0, r, nonsense=1)


def test_link_quality_and_aggregate(link0, small_config):
    r = mi.pack(link0, small_config)
    q = mi.link_quality(link0, "link0", 0, r.centers, r.radii)
    assert q["link_name"] == "link0" and q["num_spheres"] == r.num_spheres
    overall = mi.aggregate_overall([q, q])
    assert overall["num_spheres"] == 2 * r.num_spheres


@pytest.mark.skipif(
    shutil.which("morphit") is None and not (REPO / "target" / "release" / "morphit.exe").exists()
    and not (REPO / "target" / "release" / "morphit").exists(),
    reason="the morphit CLI is not built",
)
def test_same_result_as_cli(tmp_path, link0_path):
    cli = shutil.which("morphit") or str(
        next(p for p in (REPO / "target" / "release" / "morphit.exe", REPO / "target" / "release" / "morphit") if p.exists())
    )
    out = tmp_path / "cli.json"
    subprocess.run(
        [cli, "pack", str(link0_path), "-p", "MorphIt-B", "-n", "8", "--iterations", "30", "--seed", "7",
         "--device", "cpu", "--set", "model.num_inside_samples=1000", "--set", "model.num_surface_samples=1000",
         "-o", str(out)],
        check=True,
        capture_output=True,
    )
    cli_result = mi.PackResult.load(out)
    c = mi.Config.preset("MorphIt-B", num_spheres=8, iterations=30, seed=7, device="cpu")
    c["model.num_inside_samples"] = 1000
    c["model.num_surface_samples"] = 1000
    py_result = mi.pack(mi.Mesh.load(link0_path), c)
    assert np.array_equal(py_result.centers, cli_result.centers)
    assert np.array_equal(py_result.radii, cli_result.radii)


def test_misc():
    assert mi.version() == mi.__version__
    assert any(e["default"] for e in mi.example_objects())
    assert any(r["name"] == "kinova" for r in mi.example_robots())
    for d in mi.devices():
        assert {"index", "name", "backend", "kind", "software"} <= set(d)
