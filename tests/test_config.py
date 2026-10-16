import pytest

from voronav.config import load_config


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
