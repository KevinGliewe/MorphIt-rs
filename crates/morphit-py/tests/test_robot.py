import io
import zipfile

import pytest

import morphit_rs as mi

from conftest import EXAMPLES

KINOVA = EXAMPLES / "kinova_description"


@pytest.fixture(scope="module")
def pkg():
    return mi.RobotPackage.from_folder(KINOVA)


def test_inspect(pkg):
    assert pkg.urdfs()
    report = pkg.inspect()
    items = [c for c in report["collisions"] if c["action"] == "pack"]
    assert items and all(c["mesh_path"] for c in items)
    poses = pkg.link_poses()
    assert all(len(p["rotation"]) == 9 and len(p["translation"]) == 3 for p in poses.values())
    assert pkg.mesh(items[0]["mesh_path"]).volume > 0


def test_pack_link_and_assemble(pkg):
    report = pkg.inspect()
    items = [c for c in report["collisions"] if c["action"] == "pack"][:2]
    pkg.clear_link_results()
    for item in items:
        s = pkg.pack_link(item, num_spheres=4, iterations=10, seed=0, device="cpu")
        assert s.run() == "completed"
        pkg.set_link_result(item["link_name"], item["collision_index"], s.result())
    assert len(pkg.link_results()) == len(items)
    urdf, stats = pkg.assemble(base_color="#3399ff", color_variation=0.5)
    assert "<sphere" in urdf
    assert stats["mesh_collisions_replaced"] == len(items)


def test_pack_all_with_callback_stop(pkg):
    calls = []

    def cb(link, index, step):
        calls.append((link, index, step.iteration))
        return len(calls) < 15  # stop in the second link

    results = pkg.pack_all(num_spheres=3, iterations=10, seed=0, device="cpu", callback=cb)
    assert len(results) == 1
    (link, index), r = next(iter(results.items()))
    assert r.num_spheres >= 1 and calls[0][:2] == (link, index)


def test_invalid_params(pkg):
    item = next(c for c in pkg.inspect()["collisions"] if c["action"] == "pack")
    with pytest.raises(mi.RobotError):
        pkg.pack_link(item, variant="MorphIt-Obj")  # the web API allows V, S and B only
    with pytest.raises(mi.RobotError):
        pkg.pack_link(item, num_spheres=0)


def test_from_zip_and_files():
    buf = io.BytesIO()
    files = {}
    with zipfile.ZipFile(buf, "w") as z:
        for p in KINOVA.rglob("*"):
            if p.is_file():
                rel = p.relative_to(KINOVA).as_posix()
                z.write(p, rel)
                files[rel] = p.read_bytes()
    a = mi.RobotPackage.from_zip(buf.getvalue())
    b = mi.RobotPackage.from_files(files)
    assert a.files() == b.files() == mi.RobotPackage.from_folder(KINOVA).files()
