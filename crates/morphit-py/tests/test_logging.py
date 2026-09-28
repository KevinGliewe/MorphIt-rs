import logging

import morphit_rs as mi

from conftest import CORE_FIXTURES


def test_library_logs_reach_python_logging(caplog):
    mi.enable_logging("INFO")
    mi._flush_logs()  # drop anything queued by earlier tests
    caplog.set_level(logging.INFO, logger="morphit_rs")
    mesh = mi.Mesh.load(CORE_FIXTURES / "mesh_prep" / "two_overlapping.obj")
    mesh.prepared()
    mi._flush_logs()
    msgs = [r.getMessage() for r in caplog.records if r.name.startswith("morphit_rs.")]
    assert any("unioned 2 overlapping bodies" in m for m in msgs), msgs
