"""The four target bodies: checks on the real files, and the never-replace rule."""
import json

import numpy as np
import pytest

from fittedperfection import bodies, config
from fittedperfection.gltf import GLBError, read_skinned_mesh
from fittedperfection.hashing import file_sha256

REAL = config.PACKAGE.parent/'assets'/'bodies'
needs_real_bodies = pytest.mark.skipif(not (REAL/'bodies.json').is_file(), reason='bodies not made yet')


@needs_real_bodies
def test_real_bodies_share_topology_and_skeleton_and_pass_checks():
    measures = bodies.check(REAL)                     # raises on any failed check
    assert set(measures) == set(config.BODIES)
    assert len({facts['vertices'] for facts in measures.values()}) == 1


@needs_real_bodies
def test_real_bodies_hit_their_target_heights():
    record = json.loads((REAL/'bodies.json').read_text(encoding='utf-8'))
    for body, entry in record['bodies'].items():
        target = bodies.TARGET_HEIGHT_M[body]
        if target is not None:
            assert abs(entry['measures']['height_m'] - target) <= bodies.HEIGHT_TOLERANCE_M
    assert record['bodies'][config.DONOR_BODY]['macros']['height'] == 0.5


@needs_real_bodies
def test_record_matches_the_files():
    record = json.loads((REAL/'bodies.json').read_text(encoding='utf-8'))
    assert record['mpfb']['version'] == '2.0.18' and record['rig'] == 'default'
    for body, entry in record['bodies'].items():
        assert entry['sha256'] == file_sha256(REAL/body/'BODY.glb')
        assert entry['sex'] == config.BODIES[body]


@needs_real_bodies
def test_skeleton_reads_as_expected():
    mesh = read_skinned_mesh(REAL/config.DONOR_BODY/'BODY.glb')
    assert mesh.bone_names[0] == 'root' and mesh.bone_parents[0] == -1
    assert (mesh.bone_parents[1:] >= 0).all()
    left, right = mesh.bone_names.index('upperarm01.L'), mesh.bone_names.index('upperarm01.R')
    assert mesh.bone_heads[left][0] > 0 > mesh.bone_heads[right][0]      # the body's left is +X
    assert np.allclose(mesh.weights.sum(axis=1), 1, atol=1e-3)


def test_reader_refuses_a_mesh_without_a_skin(tmp_path):
    import trimesh
    path = tmp_path/'plain.glb'
    trimesh.creation.box().export(path)
    with pytest.raises(GLBError, match='skinned mesh'):
        read_skinned_mesh(path)


def write_record(folder, **record):
    folder.mkdir(parents=True)
    (folder/'bodies.json').write_text(json.dumps(record), encoding='utf-8')


def test_existing_bodies_are_never_overwritten(home):
    write_record(home/'assets/bodies', status='draft_for_review', reviewed=False)
    (home/'config.json').write_text(json.dumps({'bodies': 'assets/bodies'}))
    with pytest.raises(bodies.BodyError, match='never overwritten'):
        bodies.make()


def test_an_unreviewed_draft_is_set_aside_not_deleted(home):
    write_record(home/'assets/bodies', status='draft_for_review', reviewed=False)
    bodies._set_aside_draft(home/'assets/bodies')
    assert (home/'assets/bodies.stale-1/bodies.json').is_file() and not (home/'assets/bodies').exists()


def test_reviewed_bodies_are_never_replaced(home):
    write_record(home/'assets/bodies', status='reviewed', reviewed=True)
    with pytest.raises(bodies.BodyError, match='never replaced'):
        bodies._set_aside_draft(home/'assets/bodies')
    assert (home/'assets/bodies/bodies.json').is_file()
