"""Steps F-G: binding the full resolution to the cage, rebuilding it, skin weights, and the skinned GLB writer."""
from pathlib import Path

import numpy as np
import pytest

from scipy.spatial import cKDTree

from fittedperfection import config, donor, fullres, wrap
from fittedperfection.glbwrite import write_mesh
from fittedperfection.gltf import gltf_to_canonical, read_skinned_mesh

REAL = Path(__file__).resolve().parents[1]/'assets'/'bodies'/'bodies.json'


def grid(n, size=1.0, z=0.0):
    s = np.linspace(0, size, n)
    x, y = np.meshgrid(s, s, indexing='ij')
    points = np.column_stack([x.ravel(), y.ravel(), np.full(n * n, z)])
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            triangles += [[a, a + n, a + 1], [a + 1, a + n, a + n + 1]]
    return points, np.array(triangles)


def turn(degrees, axis=(0, 0, 1)):
    return donor.rotation_between_axis_angle(np.array(axis, dtype=float), np.radians(degrees))


def setup(fine_offset=0.01):
    cage, cage_triangles = grid(6)
    labels = np.array(['torso'] * len(cage_triangles))
    fine, _ = grid(25)
    fine[:, :2] = fine[:, :2] * 0.9 + 0.05
    fine[:, 2] = fine_offset + 0.003 * np.sin(fine[:, 0] * 20)
    binding = fullres.bind(fine, np.array(['torso'] * len(fine)), cage, cage_triangles, labels)
    return cage, cage_triangles, fine, binding


def rebuild(cage, cage_triangles, fine, binding, fitted, rigid=()):
    rotation = fullres.rotation_field(cage, fitted, cage_triangles, 3)
    scale = fullres.scale_field(cage, fitted, cage_triangles, (0.5, 2.0))
    return fullres.rebuild(binding, fine, cage, fitted, cage_triangles, rotation, scale, list(rigid))


def test_a_rigidly_moved_cage_carries_the_detail_exactly():
    cage, cage_triangles, fine, binding = setup()
    r = turn(35, (1, 1, 0))
    out, _ = rebuild(cage, cage_triangles, fine, binding, cage @ r.T + [0.3, -0.2, 0.5])
    assert np.abs(out - (fine @ r.T + [0.3, -0.2, 0.5])).max() < 1e-9


def test_a_bent_cage_keeps_the_offsets():
    cage, cage_triangles, fine, binding = setup()
    fitted = cage.copy()
    right = fitted[:, 0] > 0.5
    angle = np.radians(30)
    fitted[right, 2] = (fitted[right, 0] - 0.5) * np.sin(angle)
    fitted[right, 0] = 0.5 + (fitted[right, 0] - 0.5) * np.cos(angle)
    out, _ = rebuild(cage, cage_triangles, fine, binding, fitted)
    _, _, _, d = wrap.closest_on_mesh(out, fitted, cage_triangles, np.arange(len(cage_triangles)),
                                      cKDTree(fitted[cage_triangles].mean(axis=1)))
    assert np.abs(d - np.linalg.norm(binding['offset'], axis=1)).max() < 0.004


def test_points_bind_to_their_own_label_even_when_another_layer_is_nearer():
    boot, boot_triangles = grid(4, z=0.0)
    trouser, trouser_triangles = grid(4, z=0.02)
    cage = np.vstack([boot, trouser])
    triangles = np.vstack([boot_triangles, trouser_triangles + len(boot)])
    labels = np.array(['footwear.left'] * len(boot_triangles) + ['pant_leg.left'] * len(trouser_triangles))
    point = np.array([[0.5, 0.5, 0.008]])                                  # nearer the boot, but trouser cloth
    binding = fullres.bind(point, np.array(['pant_leg.left']), cage, triangles, labels)
    assert labels[binding['face'][0]] == 'pant_leg.left'
    binding = fullres.bind(point, np.array(['footwear.left']), cage, triangles, labels)
    assert labels[binding['face'][0]] == 'footwear.left'


def test_rigid_groups_come_out_as_exact_similarity_copies():
    cage, cage_triangles, fine, binding = setup()
    group = np.flatnonzero((cage[:, 0] <= 0.41) & (cage[:, 1] <= 0.41))
    fitted = cage + np.c_[np.zeros(len(cage)), np.zeros(len(cage)), 0.2 * cage[:, 1] ** 2]    # a gentle curl
    r = turn(20)
    fitted[group] = 0.8 * cage[group] @ r.T + [0.1, 0.2, 0.3]
    out, which = rebuild(cage, cage_triangles, fine, binding, fitted, rigid=[group])
    inside = which == 0
    assert inside.any()
    assert np.abs(out[inside] - (0.8 * fine[inside] @ r.T + [0.1, 0.2, 0.3])).max() < 1e-9


def test_top_influences_keep_four_normalised_and_share_by_owner():
    joints = np.array([[0, 1, 2, 3, 4, 5], [1, 1, 2, 2, 3, 3], [0, 1, 2, 3, 4, 5]])
    weights = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 1, 1, 1, 1, 1.0], [6, 5, 4, 3, 2, 1.0]])
    j, w = fullres.top_influences(np.array([0, 1, 2]), joints, weights, 3, 4)
    assert np.allclose(w.sum(axis=1), 1) and (w > 0).sum(axis=1).max() <= 4
    assert set(j[0]) == {2, 3, 4, 5} and set(j[2]) == {0, 1, 2, 3}
    shared_j, shared_w = fullres.top_influences(np.array([7, 7, 9]), joints, weights, 3, 4)
    assert np.array_equal(shared_j[0], shared_j[1]) and np.allclose(shared_w[0], shared_w[1])


def test_a_point_on_a_skin_vertex_takes_that_vertex_weights():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    skin = wrap.Skin(positions, triangles)
    body_joints = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    body_weights = np.array([[0.7, 0.3], [1.0, 0.0], [0.5, 0.5], [0.9, 0.1]])
    regions = [('all', np.arange(2), np.arange(2))]
    j, w = fullres.skin_weights(np.array([[0, 0, 0.01], [1, 0, 0.01]]), regions, skin, body_joints, body_weights)
    assert dict(zip(j[0], w[0]))[0] == pytest.approx(0.7) and dict(zip(j[0], w[0]))[1] == pytest.approx(0.3)
    assert dict(zip(j[1], w[1]))[1] == pytest.approx(1.0)


def test_twins_take_one_label():
    welded = np.array([0, 1, 2, 0, 3])                                  # vertex 3 is a UV-split twin of vertex 0
    triangles = np.array([[0, 1, 2], [3, 4, 1]])
    labels = fullres.vertex_labels(np.array(['torso', 'sleeve.left']), triangles, welded)
    assert len(labels) == 4 and labels[0] in ('torso', 'sleeve.left')


@pytest.mark.skipif(not REAL.is_file(), reason='bodies not made')
def test_skinned_writer_round_trips_on_the_body_skeleton(tmp_path):
    from fittedperfection import body
    b = body.load('body_01_medium')
    j, w = fullres.top_influences(np.arange(len(b.positions)), b.joints, b.weights, len(b.positions), 4)
    write_mesh(tmp_path/'s.glb', b.positions, b.triangles,
               skin={'joints': j, 'weights': w, 'skeleton': config.body_file('body_01_medium'), 'root_offset_m': 0.04})
    m = read_skinned_mesh(tmp_path/'s.glb')
    assert np.abs(gltf_to_canonical(m.positions) - b.positions).max() < 1e-6
    assert (m.joints == j).all() and np.abs(m.weights - w).max() < 1e-6
    assert m.bone_names == b.bone_names and (m.bone_parents == b.parents).all()
