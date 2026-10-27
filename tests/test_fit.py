"""Fit Steps B-C: label regions, rigid groups, seam blending, un-pose, transfer, and twins on a real run."""
from pathlib import Path

import numpy as np
import pytest

from fittedperfection import body, donor, fit, meshes, settings

REAL = Path(__file__).resolve().parents[1]/'assets'/'bodies'/'bodies.json'
needs_bodies = pytest.mark.skipif(not REAL.is_file(), reason='bodies not made')


def test_label_regions():
    assert fit.label_region('sleeve.left') == 'arm.left' and fit.label_region('cuff.right') == 'arm.right'
    assert fit.label_region('footwear.left') == 'leg.left' and fit.label_region('pant_leg.right') == 'leg.right'
    assert fit.label_region('pants') == 'hips' and fit.label_region('hanging') == 'hanging'
    assert fit.label_region('torso') == 'trunk' and fit.label_region('hardware') == 'trunk'


def test_rigid_fit_recovers_a_turn_and_clamps_it():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 3))
    turn = donor.rotation_between_axis_angle(np.array([0, 0, 1.0]), np.radians(40))
    moved = source @ turn.T + [0.1, 0.2, 0.3]
    rotation, shift, angle = fit.rigid_fit(source, moved)
    assert abs(angle - 40) < 1e-6 and np.allclose(source @ rotation.T + shift, moved)
    rotation, _, angle = fit.rigid_fit(source, moved, max_rotation_degrees=25)
    assert abs(angle - 40) < 1e-6                                       # the field's turn is recorded
    assert abs(np.degrees(np.arccos((np.trace(rotation) - 1) / 2)) - 25) < 1e-6


def _strip():
    """Left shoe, right shoe sharing one point with it, a buckle on the left shoe, and a lone piece of hardware."""
    triangles = np.array([[0, 1, 2], [0, 2, 12],            # footwear.left
                          [2, 3, 4], [3, 4, 13],            # footwear.right (shares point 2 with the left)
                          [1, 5, 6], [5, 6, 7],             # hardware touching the left shoe at point 1
                          [8, 9, 10], [9, 10, 11]])         # hardware on its own
    labels = np.array(['footwear.left'] * 2 + ['footwear.right'] * 2 + ['hardware'] * 4)
    positions = np.random.default_rng(2).normal(size=(14, 3))
    return positions, triangles, labels


def test_rigid_groups_merge_hardware_into_its_shoe_and_never_merge_shoes():
    positions, triangles, labels = _strip()
    groups = {g['name'] + str(i): set(g['vertices'].tolist())
              for i, g in enumerate(fit.rigid_groups(labels, triangles, meshes.face_graph(positions, triangles)))}
    assert sorted(groups.values(), key=min) == [{0, 1, 2, 5, 6, 7, 12}, {3, 4, 13}, {8, 9, 10, 11}]
    names = {frozenset(v): k.rstrip('0123456789') for k, v in groups.items()}
    assert names[frozenset({0, 1, 2, 5, 6, 7, 12})] == 'shoe.left' and names[frozenset({3, 4, 13})] == 'shoe.right'


def test_place_rigid_keeps_the_exact_shape():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(20, 3))
    moved = source + rng.normal(scale=0.05, size=source.shape) + [0, 0, 1]
    out, record = fit.place_rigid(source, moved, [{'name': 'hardware', 'vertices': np.arange(10)}])
    d0 = np.linalg.norm(source[:10, None] - source[None, :10], axis=2)
    d1 = np.linalg.norm(out[:10, None] - out[None, :10], axis=2)
    assert np.allclose(d0, d1) and np.allclose(out[10:], moved[10:]) and record[0]['points'] == 10


def test_seam_blend_is_smooth_inside_the_band_and_fixed_outside():
    count = 21                                              # a chain of points: region a, then region b
    triangles = np.array([[i, i + 1, i + 1] for i in range(count - 1)])
    adjacency = fit.point_adjacency(triangles, count)
    regions = np.array(['a'] * 10 + ['b'] * 11, dtype=object)
    band = fit.seam_band(adjacency, regions, 3)
    assert band[6:14].all() and not band[:5].any() and not band[16:].any()
    values = np.where(regions == 'a', 0.0, 1.0)
    out = fit.blend_across_seams(values, adjacency, band, 500).ravel()
    assert np.allclose(out[~band], values[~band])
    assert np.all(np.diff(out) >= -1e-9) and np.diff(out).max() < 0.5    # a ramp, not a step


@needs_bodies
def test_unpose_undoes_a_known_pose():
    b = body.load('body_01_medium')
    pose = np.repeat(np.eye(4)[None], len(b.bone_names), axis=0)
    pose[b.bone_region == 'arm.left'] = donor.about(b.joint('upperarm01.L'),
                                                    donor.rotation_between(np.array([1.0, 0, -1]), np.array([1.0, 0, 0])))
    posed = body.skin(b, pose)
    built = {'body': b, 'pose': pose, 'posed': posed, 'crotch_posed': body.landmarks(b)['crotch_height_m']}
    arm = np.flatnonzero(b.vertex_region == 'arm.left')
    back = fit.unpose(posed[arm], b.vertex_region[arm], built, settings.section('fit'))
    error = np.linalg.norm(back - b.positions[arm], axis=1)
    assert np.median(error) < 0.005 and error.max() < 0.03


@needs_bodies
def test_transfer_moves_donor_surface_points_onto_the_target_surface():
    source, target = body.load('body_01_medium'), body.load('body_04_broad_muscular')
    crotch = body.landmarks(source)['crotch_height_m']
    chosen = np.arange(0, len(source.positions), 7)
    moved = fit.transfer(source.positions[chosen], source.vertex_region[chosen], source.positions, target.positions,
                         source, crotch, settings.section('fit'))
    error = np.linalg.norm(moved - target.positions[chosen], axis=1)
    assert np.median(error) < 0.01 and np.percentile(error, 99) < 0.04


@pytest.mark.parametrize('run', ['moonshorts', 'baddietanknoarms'])
def test_twins_stay_together_on_every_body(run):
    root = Path(__file__).resolve().parents[1]/'runs'/run
    if not (root/'fit'/'body_01_medium'/'cage-fit.npz').is_file():
        pytest.skip('run not prepared')
    from fittedperfection.gltf import read_textured_mesh
    cage = read_textured_mesh(root/'full'/'CAGE.glb')
    welded = meshes.weld(cage.positions)
    for folder in (root/'fit').iterdir():
        if folder.is_dir():
            positions = np.load(folder/'cage-fit.npz')['positions']
            first = np.unique(welded, return_index=True)[1]
            assert np.abs(positions - positions[first][welded]).max() == 0
