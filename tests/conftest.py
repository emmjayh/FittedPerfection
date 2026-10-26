import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty project root: FP_HOME points here and no machine config leaks in."""
    monkeypatch.setenv('FP_HOME', str(tmp_path))
    for name in ('FP_BLENDER', 'FP_BODIES', 'MODEL_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def garment(tmp_path):
    path = tmp_path/'garment.zip'
    path.write_bytes(b'stand-in garment bytes')
    return path
