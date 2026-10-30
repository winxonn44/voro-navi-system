import pytest

torch = pytest.importorskip("torch")

from voronav.devices import DISABLE_MPS_ENV, mps_enabled
from voronav.voronoi.jfa_torch import available_devices


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_env_switch_hides_mps(monkeypatch, value):
    monkeypatch.setenv(DISABLE_MPS_ENV, value)
    assert not mps_enabled()
    assert "mps" not in available_devices()


@pytest.mark.parametrize("value", ["", "0"])
def test_env_switch_off_follows_torch(monkeypatch, value):
    monkeypatch.setenv(DISABLE_MPS_ENV, value)
    assert mps_enabled() == torch.backends.mps.is_available()
