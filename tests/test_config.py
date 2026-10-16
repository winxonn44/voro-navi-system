import pytest

from voronav.config import MapSourceConfig, load_config, load_q2_config


def test_load_shipped_configs():
    for name in ("debug", "bench_quick", "bench_q1"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.bench.sizes


def test_unknown_key_fails_loudly(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bench:\n  sizez: [64]\n")
    with pytest.raises(ValueError, match="sizez"):
        load_config(p)


def test_bad_backend_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bench:\n  backends: [cuda_magic]\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_load_shipped_q2_configs():
    for name in ("debug_q2", "bench_q2_quick", "bench_q2"):
        cfg = load_q2_config(f"configs/{name}.yaml")
        assert cfg.q2.maps and cfg.q2.planners
        assert all(isinstance(m, MapSourceConfig) for m in cfg.q2.maps)
    quick = load_q2_config("configs/bench_q2_quick.yaml")
    # the everyday config must not depend on downloaded data
    assert all(not m.kind.endswith("_dir") for m in quick.q2.maps)


@pytest.mark.parametrize("body, match", [
    ("q2:\n  querys_per_map: 3\n", "querys_per_map"),
    ("q2:\n  maps:\n    - {kind: corridor, sizes: [64], gap: 8, gapp: 3}\n", "gapp"),
    ("roadmap:\n  backend: cuda_magic\n", "backend"),
    ("q2:\n  planners: [voronoi, dstar]\n", "dstar"),
    ("q2:\n  lambdas: [-1]\n", "lambdas"),
    ("q2:\n  rrt_every: 0\n", "rrt_every"),
    ("q2:\n  min_dist_frac: 1.5\n", "min_dist_frac"),
    ("q2:\n  maps: []\n", "at least one"),
    ("q2:\n  maps:\n    - {kind: lava_lake, sizes: [64]}\n", "kind"),
    ("q2:\n  maps:\n    - {kind: corridor, sizes: [64]}\n", "gap"),
    ("q2:\n  maps:\n    - {kind: corridor, sizes: [64], gap: 8, seeds: [1]}\n", "seeds"),
    ("q2:\n  maps:\n    - {kind: random_obstacles, n_shapes: 5}\n", "sizes"),
    ("q2:\n  maps:\n    - {kind: movingai_dir}\n", "dir"),
    ("q2:\n  maps:\n    - {kind: image_dir, dir: x, sizes: [64]}\n", "sizes"),
    ("roadmap:\n  bridge_lam: -1\n", "bridge_lam"),
    ("q2:\n  voronoi_variants:\n    - {name: a, join: teleport}\n", "join"),
    ("q2:\n  voronoi_variants:\n    - {name: a, join_lamb: 4}\n", "join_lamb"),
    ("q2:\n  voronoi_variants:\n    - {name: a, roadmap_lam: -2}\n", "roadmap_lam"),
    ("q2:\n  voronoi_variants:\n    - {name: a}\n    - {name: a, join: best}\n", "unique"),
    ("q2:\n  voronoi_variants: []\n", "non-empty"),
    ("q2:\n  smoothing: [bezier]\n", "bezier"),
    ("q2:\n  smoothing: [safe, safe]\n", "duplicates"),
])
def test_bad_q2_config_rejected(tmp_path, body, match):
    p = tmp_path / "bad.yaml"
    p.write_text(body)
    with pytest.raises(ValueError, match=match):
        load_q2_config(p)


def test_q2_option_names_match_the_planners():
    from voronav import config
    from voronav.planning import planners, smoothing

    assert config.JOIN_MODES == planners.JOIN_MODES
    assert config.SMOOTH_MODES == smoothing.SMOOTH_MODES
    default = load_q2_config("configs/bench_q2.yaml").q2.voronoi_variants[0]
    # the first variant is the original planner, so old and new runs compare
    assert (default.join, default.join_lam, default.roadmap_lam) == ("nearest", 0.0, 0.0)
