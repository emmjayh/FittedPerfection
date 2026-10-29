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


@needs_bodies
def test_body_shoulder_top_sits_on_the_trunk_just_above_the_arm_roots():
    marks = body.landmarks(body.load('body_01_medium'))
    roots, top = marks['shoulders']['centre'][2], marks['shoulder_top']['centre'][2]
    assert marks['shoulder_top']['method'].startswith('shoulder_top')
    assert 0.0 < top - roots < 0.06                       # not the neck or the head


def _built(run):
    root = Path(__file__).resolve().parents[1]/'runs'/run
    if not (root/'bind'/'binding.json').is_file():
        pytest.skip('run not prepared')
    binding = json.loads((root/'bind'/'binding.json').read_text(encoding='utf-8'))
    labels = np.load(root/'labels'/'labels.npz')['labels'].astype(str)
    cage = read_textured_mesh(root/'full'/'CAGE.glb')
    _, areas = meshes.face_normals(cage.positions, cage.triangles)
    return binding, donor.build(binding, labels, cage.positions[cage.triangles].mean(axis=1), areas,
                                settings.section('fit'))


@needs_bodies
@pytest.mark.parametrize('run', ['moonshorts', 'hallowewenbaddie'])
def test_donor_legs_aim_over_the_rear_of_the_sole(run):
    binding, built = _built(run)
    b = built['body']
    for side, joint in (('left', 'foot.L'), ('right', 'foot.R')):
        foot = binding['feet'][side]
        target = np.array(foot['heel']) + donor.ANKLE_ALONG_SOLE * (np.array(foot['toe']) - np.array(foot['heel']))
        ankle = built['posed'][np.argmin(np.linalg.norm(b.positions - b.joint(joint), axis=1))]
        assert np.linalg.norm((ankle - target)[:2]) < 0.06            # the ankle stands over the shoe, not beside it


@needs_bodies
@pytest.mark.parametrize('run', ['baddietanknoarms', 'blackpantsgreentop'])
def test_donor_neck_only_lengthens_within_its_range(run):
    _, built = _built(run)
    low, high = settings.section('fit')['neck_length_range']
    assert low == 1.0 and low <= built['record']['neck_scale'] <= high


@needs_bodies
@pytest.mark.parametrize('run', ['blackshirtsuspenderjeans', 'blackpantsgreentop'])
def test_donor_wrists_meet_the_cuffs(run):
    binding, built = _built(run)
    root = Path(__file__).resolve().parents[1]/'runs'/run
    labels = np.load(root/'labels'/'labels.npz')['labels'].astype(str)
    cage = read_textured_mesh(root/'full'/'CAGE.glb')
    centres = cage.positions[cage.triangles].mean(axis=1)
    _, areas = meshes.face_normals(cage.positions, cage.triangles)
    b = built['body']
    for side, s in (('left', 'L'), ('right', 'R')):
        cuff = labels == f'cuff.{side}'
        if cuff.sum() < donor.MIN_CUFF_FACES:
            continue
        wrist = donor.apply(built['posed_bones'][b.bone_names.index(f'wrist.{s}')], b.joint(f'wrist.{s}'))
        assert np.linalg.norm(wrist - np.average(centres[cuff], axis=0, weights=areas[cuff])) < 0.05
        assert isinstance(built['record']['arm_scale'][side], dict)


@needs_bodies
def test_donor_arms_keep_their_length_without_a_cuff():
    _, built = _built('hallowewenbaddie')
    assert built['record']['arm_scale'] == {'left': 'kept (no cuff)', 'right': 'kept (no cuff)'}


def ring_points(half_width, half_depth, centre_y=0.0, heights=np.arange(0.0, 0.3, 0.01), count=72, gap=None):
    """Points on elliptical rings along z; `gap` drops the back sector (y > 0, |x| < |y|) over a height range."""
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    out = []
    for z in heights:
        ring = np.column_stack([half_width * np.cos(angle), centre_y + half_depth * np.sin(angle), np.full(count, z)])
        if gap is not None and gap[0] <= z < gap[1]:
            rel = ring[:, :2] - [0, centre_y]
            ring = ring[~((rel[:, 1] > 0) & (np.abs(rel[:, 0]) < np.abs(rel[:, 1])))]
        out.append(ring)
    return np.vstack(out)


TRUNK_CONFIG = {'girth_range': [0.85, 1.15], 'trunk_offset_max_m': 0.06}


def fitted_trunk(cloth, donor_points=None):
    points = ring_points(0.15, 0.10) if donor_points is None else donor_points
    move, move_rest, offset_at, summary = donor.trunk_girth(points, points.copy(), np.ones(len(points), bool), cloth,
                                                            TRUNK_CONFIG, 0.03)
    assert np.allclose(move, move_rest)                                # the same move in both shapes
    move[:, 1] += offset_at(points[:, 2])
    return points + move, summary


def test_trunk_depth_and_width_fit_the_cloth_separately():
    out, summary = fitted_trunk(ring_points(0.14, 0.11))               # narrower and deeper than the donor
    assert np.ptp(out[:, 1]) / 0.20 > 1.03 and np.ptp(out[:, 0]) / 0.30 < 0.95
    assert abs(out[:, 1].mean()) < 0.003                               # centred: nothing to move


def test_trunk_slices_move_front_to_back_to_centre_in_the_cloth():
    out, summary = fitted_trunk(ring_points(0.15, 0.10, centre_y=0.03))
    assert abs(out[:, 1].mean() - 0.03 * 0.97) < 0.005 and summary['offset_cm'][0] > 2.5


def test_a_slice_with_no_back_cloth_takes_depth_and_offset_from_its_neighbours():
    out, summary = fitted_trunk(ring_points(0.15, 0.12, centre_y=0.02, gap=(0.12, 0.18)))
    assert summary['filled_from_neighbours']['depth'] >= 1
    opening = (out[:, 2] >= 0.12) & (out[:, 2] < 0.18)
    assert abs(np.ptp(out[opening, 1]) - np.ptp(out[~opening, 1])) < 0.01
    assert abs(out[opening, 1].mean() - out[~opening, 1].mean()) < 0.005


@needs_bodies
@pytest.mark.parametrize('run', ['hallowewenbaddie', 'moonshorts', 'baddietanknoarms', 'blackpantsgreentop'])
def test_donor_back_sits_inside_the_cloth(run):
    root = Path(__file__).resolve().parents[1]/'runs'/run
    if not (root/'bind'/'binding.json').is_file():
        pytest.skip('run not prepared')
    binding = json.loads((root/'bind'/'binding.json').read_text(encoding='utf-8'))
    labels = np.load(root/'labels'/'labels.npz')['labels'].astype(str)
    cage = read_textured_mesh(root/'full'/'CAGE.glb')
    centres = cage.positions[cage.triangles].mean(axis=1)
    _, areas = meshes.face_normals(cage.positions, cage.triangles)
    built = donor.build(binding, labels, centres, areas, settings.section('fit'))
    rest, trunk = built['rest'], built['body'].vertex_region == 'trunk'
    cloth = np.isin(labels, ('torso', 'pants', 'waistband'))
    crotch, top = binding['crotch']['height_m'], binding['shoulders']['centre'][2]
    gaps = []
    for z in np.arange(crotch + 0.1, top - 0.1, 0.03):              # the centre strip of the back, slice by slice
        d = trunk & (rest[:, 2] >= z) & (rest[:, 2] < z + 0.03) & (np.abs(rest[:, 0]) < 0.06)
        k = cloth & (centres[:, 2] >= z) & (centres[:, 2] < z + 0.03) & (np.abs(centres[:, 0]) < 0.06)
        back = k & (centres[:, 1] > rest[d, 1].mean()) if d.sum() >= 4 else k & False
        if back.sum() >= 2:
            gaps.append(np.percentile(centres[back, 1], 10) - rest[d, 1].max())
    assert len(gaps) >= 5 and np.median(gaps) >= -0.005
