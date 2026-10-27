"""One read-only fit configuration with per-section hashes; pinned models."""
import pytest

from fittedperfection import models, settings


def test_settings_cannot_be_changed_at_runtime():
    with pytest.raises(TypeError):
        settings.SETTINGS['fit']['transfer_neighbours'] = 8
    with pytest.raises(TypeError):
        settings.SETTINGS['fit']['girth_range'][0] = 0.5


def test_spec_values():
    fit = settings.section('fit')
    assert fit['transfer_neighbours'] == 32 and tuple(fit['leg_length_range']) == (0.85, 1.6)
    assert settings.section('bind')['hem_fraction']['knee'] == 0.60


def test_section_hashes_are_stable_and_distinct():
    hashes = {name: settings.section_sha256(name) for name in settings.SECTIONS}
    assert hashes == {name: settings.section_sha256(name) for name in settings.SECTIONS}
    assert len(set(hashes.values())) == len(hashes)


def test_model_pins_and_installed_decoder():
    assert models.pin('sam')['model'] == 'sam-3.1'
    assert models.pin('wonder')['model'] == 'muse-wonder-1.0'
    assert models.check_decoder() == '0.1.0'
