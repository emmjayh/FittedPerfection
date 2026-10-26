"""Intake treats archives as data and picks the plain geometry export."""
import zipfile

import pytest

from fittedperfection.intake import IntakeError, choose_mesh, unpack


def archive(path, **files):
    with zipfile.ZipFile(path, 'w') as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    return path


def test_unpacks_flat_known_files(tmp_path):
    zip_path = archive(tmp_path/'a.zip', **{'base.obj': 'v 0 0 0', 'texture_diffuse.png': 'png'})
    assert unpack(zip_path, tmp_path/'out') == ['base.obj', 'texture_diffuse.png']


@pytest.mark.parametrize('name', ['../evil.obj', 'sub/base.obj', 'run.py', '.hidden.png', 'C:evil.obj'])
def test_refuses_anything_unexpected(tmp_path, name):
    zip_path = archive(tmp_path/'a.zip', **{name: 'x', 'texture_diffuse.png': 'png'})
    with pytest.raises(IntakeError):
        unpack(zip_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_mesh_choice_prefers_plain_obj_then_fbx_never_basic():
    assert choose_mesh(['base_basic_pbr.fbx', 'base.fbx', 'base_basic_shaded.fbx']) == 'base.fbx'
    assert choose_mesh(['lod.fbx', 'base.obj']) == 'base.obj'
    assert choose_mesh(['lod_basic_pbr.fbx', 'lod.fbx']) == 'lod.fbx'
    with pytest.raises(IntakeError):
        choose_mesh(['base_basic_pbr.fbx', 'texture_diffuse.png'])
