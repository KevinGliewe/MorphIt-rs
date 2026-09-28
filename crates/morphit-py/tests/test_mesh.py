import numpy as np
import pytest

import morphit_rs as mi

from conftest import CORE_FIXTURES, load_json


def unit_box(offset=(0.0, 0.0, 0.0)):
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float) + offset
    f = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ]
    )
    return v, f


def test_load_and_properties(link0):
    assert link0.vertices.shape[1] == 3 and link0.vertices.dtype == np.float64
    assert link0.faces.shape == (200, 3)
    assert link0.volume == pytest.approx(0.0029964693753239025, rel=1e-12)
    lo, hi = link0.bounds
    assert all(a < b for a, b in zip(lo, hi))
    assert link0.inertia.shape == (3, 3)
    info = link0.info()
    assert info["faces"] == 200 and info["source_path"].endswith("link0.obj")
    assert "obj" in mi.Mesh.supported_extensions()


def test_errors(tmp_path):
    with pytest.raises(mi.MorphItIOError) as e:
        mi.Mesh.load(tmp_path / "missing.obj")
    assert isinstance(e.value, OSError)
    with pytest.raises(mi.MeshError):
        mi.Mesh.from_bytes(b"not a mesh", "obj")
    with pytest.raises(ValueError):
        mi.Mesh.from_arrays(np.zeros((3, 2)), [[0, 1, 2]])


def test_from_arrays_round_trip():
    v, f = unit_box()
    m = mi.Mesh.from_arrays(v, f)
    assert np.array_equal(m.vertices, v)
    assert np.array_equal(m.faces, f)
    assert m.volume == pytest.approx(1.0)
    inside = m.contains([[0.3, 0.4, 0.45], [2.0, 0.4, 0.45]])
    assert inside.tolist() == [True, False]
    # Plain lists work too.
    assert mi.Mesh.from_arrays(v.tolist(), f.tolist()).volume == pytest.approx(1.0)


def test_from_bytes(link0_path):
    m = mi.Mesh.from_bytes(link0_path.read_bytes(), "obj", "link0.obj")
    assert m.faces.shape == (200, 3)


def test_mesh_prep_matches_python():
    expected = load_json(CORE_FIXTURES / "py_mesh_prep.json")
    for name, case in expected.items():
        m = mi.Mesh.load(CORE_FIXTURES / case["file"])
        report = m.prep_report()
        want = case["report"]
        assert report["action"] == want["action"], name
        assert report["n_bodies"] == want["n_bodies"], name
        assert report["volume_after"] == pytest.approx(want["volume_after"], rel=1e-9), name


def test_union_and_hull():
    m = mi.Mesh.load(CORE_FIXTURES / "mesh_prep" / "two_overlapping.obj")
    assert m.prep_report()["action"] == "unioned"
    assert m.prep_report(union=False)["action"] == "disabled"
    u = m.prepared()
    assert u.volume < m.volume  # the overlap is counted once
    h = m.prepared(convex_hull=True)
    rep = m.prep_report(convex_hull=True)
    assert rep["convex_hull"] is True and rep["n_hulled"] >= 1
    assert h.volume >= u.volume - 1e-12


def test_save_obj_and_stl(tmp_path, link0):
    obj, stl = tmp_path / "a.obj", tmp_path / "a.stl"
    link0.save(obj)
    link0.save(stl)
    assert mi.Mesh.load(obj).volume == pytest.approx(link0.volume, rel=1e-12)
    assert mi.Mesh.load(stl).volume == pytest.approx(link0.volume, rel=1e-5)
    assert link0.to_obj().startswith("#")
    assert isinstance(link0.to_stl(), bytes)
