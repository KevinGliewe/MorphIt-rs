import os
import threading
import time

import numpy as np
import pytest

import morphit_rs as mi


def test_pack_is_deterministic(link0, small_config):
    a = mi.pack(link0, small_config)
    b = mi.pack(link0, small_config)
    assert np.array_equal(a.centers, b.centers)
    assert np.array_equal(a.radii, b.radii)
    assert a.centers.shape == (a.num_spheres, 3) and len(a) == a.num_spheres


def test_session_iteration_equals_pack(link0, small_config):
    s = mi.Session(link0, small_config)
    assert s.state == "running" and s.iteration == 0
    steps = list(s)
    assert [st.iteration for st in steps] == list(range(30))
    assert steps[-1].done and s.is_done
    assert set(steps[0].weighted_losses) == set(steps[0].raw_losses)
    assert "coverage_loss" in steps[0].weighted_losses
    s.finalize()
    assert s.state == "finalized"
    r = s.result()
    p = mi.pack(link0, small_config)
    assert np.array_equal(r.centers, p.centers) and np.array_equal(r.radii, p.radii)
    with pytest.raises(mi.StateError):
        s.step()


def test_run_with_callback_and_stop(link0, small_config):
    s = mi.Session(link0, small_config)
    seen = []
    assert s.run(lambda st: seen.append(st.iteration), every=10) == "completed"
    assert seen == [0, 10, 20, 29]
    s2 = mi.Session(link0, small_config)
    assert s2.run(lambda st: st.iteration < 4) == "cancelled"
    assert s2.iteration == 5 and s2.state == "running"
    assert s2.run() == "completed"  # a cancelled session can be resumed
    assert s2.state == "finalized"


def test_snapshot_reads_and_cancel_from_another_thread(link0, small_config):
    small_config["training.iterations"] = 100_000
    s = mi.Session(link0, small_config)
    outcome = []
    t = threading.Thread(target=lambda: outcome.append(s.run()))
    t.start()
    deadline = time.time() + 30
    while s.iteration < 5 and time.time() < deadline:
        time.sleep(0.01)
    # Reads never block while another thread runs the session.
    assert s.is_running
    assert s.centers.shape == (8, 3) and s.radii.shape == (8,)
    with pytest.raises(mi.BusyError):
        s.step()
    with pytest.raises(mi.BusyError):
        s.finalize()
    s.cancel()
    t.join(30)
    assert outcome == ["cancelled"]
    assert not s.is_running and s.state == "running"


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs two cores")
def test_gil_is_released(link0, small_config):
    small_config["training.iterations"] = 150
    mi.pack(link0, small_config)  # warm up the thread pool

    def one():
        mi.pack(link0, small_config)

    t0 = time.perf_counter()
    one()
    one()
    serial = time.perf_counter() - t0
    # While a pack runs, a pure-Python thread keeps making progress.
    counter = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter[0] += 1

    th = threading.Thread(target=spin)
    th.start()
    t0 = time.perf_counter()
    one()
    stop.set()
    th.join()
    assert counter[0] > 1000, "Python thread starved: the GIL was held during pack()"
    assert serial > 0


def test_accessors(link0, small_config):
    s = mi.Session(link0, small_config)
    inside, surface = s.samples()
    assert inside.shape == (1000, 3) and surface.shape == (1000, 3)
    assert link0.contains(inside).all()
    assert s.device == "cpu" and s.gpu_error is None
    # The session records the mesh's source path.
    small_config["model.mesh_path"] = link0.source_path
    assert s.config == small_config
    assert s.mesh_prep["action"] == "unchanged"
    assert s.init_info["seed"] == 7
    assert s.last_step is None
    s.step()
    assert s.last_step.iteration == 0
    hist = s.history()
    assert isinstance(hist, dict) and hist
    assert "Session(" in repr(s)


def test_result_io(tmp_path, link0, small_config):
    r = mi.pack(link0, small_config)
    p = tmp_path / "out" / "r.json"
    r.save(p)
    back = mi.PackResult.load(p)
    assert np.array_equal(back.centers, r.centers)
    small_config["model.mesh_path"] = link0.source_path
    assert back.config == small_config
    d = r.to_dict()
    assert list(d) == ["centers", "radii", "masses", "mesh_path", "num_spheres", "per_sphere_mass", "mesh_prep", "config"]
    assert mi.PackResult.from_json(r.to_json()).num_spheres == r.num_spheres
