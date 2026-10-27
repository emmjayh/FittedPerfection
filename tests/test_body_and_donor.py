"""Body model (regions, skinning, landmarks), the donor (Step A) and the GLB writer."""
import json
from pathlib import Path

import numpy as np
import pytest

from fittedperfection import body, donor, meshes, settings
from fittedperfection.glbwrite import write_mesh
from fittedperfection.gltf import read_textured_mesh

REAL = Path(__file__).resolve().parents[1]/'assets'/'bodies'/'bodies.json'
needs_bodies = pytest.mark.skipif(not REAL.is_file(), reason='bodies not made')


@needs_bodies
def test_regions_are_symmetric_and_cover_every_vertex():
    b = body.load('body_01_medium')
    counts = {r: int((b.vertex_region == r).sum()) for r in ('arm.left', 'arm.right', 'leg.left', 'leg.right', 'trunk')}
    assert counts['arm.left'] == counts['arm.right'] and abs(counts['leg.left'] - counts['leg.right']) <= 2
    assert sum(counts.values()) == len(b.positions)


@needs_bodies
def test_skinning_round_trip():
    b = body.load('body_01_medium')
    bones = len(b.bone_names)
    turn = donor.about(b.joint('upperarm01.L'), donor.rotation_between(np.array([1.0, 0, -1]), np.array([1.0, 0, 0])))
    transforms = np.repeat(np.eye(4)[None], bones, axis=0)
    transforms[b.bone_region == 'arm.left'] = turn
    posed = body.skin(b, transforms)
    matrices = body.vertex_matrices(b, transforms)
    back = np.einsum('vij,vj->vi', np.linalg.inv(matrices)[:, :3, :3], posed) + np.linalg.inv(matrices)[:, :3, 3]
    assert np.abs(back - b.positions).max() < 1e-6
    assert np.abs(posed[b.vertex_region == 'trunk'] - b.positions[b.vertex_region == 'trunk']).max() < 0.2


@needs_bodies
def test_body_landmarks_are_sane():
    b = body.load('body_01_medium')
    marks = body.landmarks(b)
    assert marks['shoulders']['centre'][2] > marks['crotch_height_m'] + 0.4
    assert abs(marks['legs']['left']['ankle_end'][2] - marks['floor_m']) < 0.05
    upper, lower = b.joint('upperarm01.L'), b.joint('lowerarm01.L')
    assert abs(np.degrees(np.arcsin((upper[2] - lower[2]) / np.linalg.norm(lower - upper))) - 60) < 2


@needs_bodies
@pytest.mark.parametrize('run', ['moonshorts', 'blackpantsgreentop'])
def test_donor_matches_shoulders_stands_on_the_floor_and_follows_the_sleeves(run):
    root = Path(__file__).resolve().parents[1]/'runs'/run
    if not (root/'bind'/'binding.json').is_file():
        pytest.skip('run not prepared')
    binding = json.loads((root/'bind'/'binding.json').read_text(encoding='utf-8'))
    labels = np.load(root/'labels'/'labels.npz')['labels'].astype(str)
    cage = read_textured_mesh(root/'full'/'CAGE.glb')
    _, areas = meshes.face_normals(cage.positions, cage.triangles)
    built = donor.build(binding, labels, cage.positions[cage.triangles].mean(axis=1), areas, settings.section('fit'))
    record = built['record']
    low, high = settings.section('fit')['leg_length_range']
    assert low <= record['leg_scale'] <= high
    assert abs(built['rest'][:, 2].min()) < 0.03                      # standing on the floor
    b = built['body']
    wrist = built['posed'][np.argmin(np.linalg.norm(b.positions - b.joint('wrist.L'), axis=1))]
    shoulder = np.array(binding['shoulders']['left'])
    cuff = np.array(binding['sleeves']['left']['cuff_end'])
    direction = (wrist - shoulder) / np.linalg.norm(wrist - shoulder)
    assert np.dot(direction, (cuff - shoulder) / np.linalg.norm(cuff - shoulder)) > 0.95


def test_glb_writer_round_trips(tmp_path):
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1.0]])
    write_mesh(tmp_path/'t.glb', positions, [[0, 1, 2]], uvs=np.array([[0, 0], [1, 0], [0, 1.0]]))
    mesh = read_textured_mesh(tmp_path/'t.glb')
    assert np.allclose(mesh.positions, positions, atol=1e-6) and mesh.triangles.tolist() == [[0, 1, 2]]
