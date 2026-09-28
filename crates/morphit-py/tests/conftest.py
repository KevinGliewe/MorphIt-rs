import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CORE_FIXTURES = REPO / "crates" / "morphit" / "tests" / "fixtures"
ROBOT_FIXTURES = REPO / "crates" / "morphit-robot" / "tests" / "fixtures"
EXAMPLES = REPO / "web" / "examples"


@pytest.fixture
def link0_path() -> Path:
    return CORE_FIXTURES / "link0.obj"


@pytest.fixture
def link0(link0_path):
    import morphit_rs as mi

    return mi.Mesh.load(link0_path)


@pytest.fixture
def small_config():
    """Fast settings for tests: few spheres, samples and iterations, fixed seed."""
    import morphit_rs as mi

    c = mi.Config.preset("MorphIt-B", num_spheres=8, iterations=30, seed=7, device="cpu")
    c["model.num_inside_samples"] = 1000
    c["model.num_surface_samples"] = 1000
    return c


def load_json(path: Path):
    return json.loads(path.read_text())
