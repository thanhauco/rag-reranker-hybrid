from pathlib import Path

import pytest

from hybrid_rag import build_pipeline, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["bm25", "hybrid_heuristic", "hybrid_weighted_dynamic"])
def test_offline_configs_build_and_run(sample, name):
    pipe = build_pipeline(load_config(CONFIGS / f"{name}.yaml")).index(sample.corpus)
    assert pipe.search("too many clients already").results[0].id == "d02"
