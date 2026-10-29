"""Step D wrap: closest points, looseness kept, relief kept, layers in order, contact, anchors, free cloth."""
import numpy as np
import pytest

from fittedperfection import settings, wrap

CONFIG = dict(settings.section('fit'))


def tube(radius, z_low, z_high, around=48, rings=31, bump=None):
    """An open tube along z (points ring by ring) and its triangles; bump=(ring, segment, height) pushes a patch out."""
    angle = np.linspace(0, 2 * np.pi, around, endpoint=False)
    z = np.linspace(z_low, z_high, rings)
    r = np.full((rings, around), float(radius))
    if bump:
        ring, segment, height = bump
        r[ring - 2:ring + 3, segment - 2:segment + 3] += height
    points = np.stack([r * np.cos(angle)[None], r * np.sin(angle)[None], np.repeat(z[:, None], around, 1)], -1).reshape(-1, 3)
    triangles = []
    for k in range(rings - 1):
        for s in range(around):
            a, b = k * around + s, k * around + (s + 1) % around
            triangles += [[a, b, a + around], [b, b + around, a + around]]
    return points, np.array(triangles)


BODY = wrap.Skin(*tube(0.1, -0.2, 1.2, around=96, rings=71))
HEIGHT = 1.0
EASE = CONFIG['contact_ease_of_height'] * HEIGHT


def radius(points):
    return np.linalg.norm(points[:, :2], axis=1)


def run(points, triangles, keep=None, rigid=()):
    keep = np.full(len(points), CONFIG['wrap_keep']) if keep is None else keep
    everything = [('all', np.arange(len(points)), np.arange(len(BODY.triangles)))]
    return wrap.solve(points, triangles, keep, BODY, everything, np.array(rigid, dtype=np.int64), CONFIG, HEIGHT)


def test_closest_point_on_triangles_matches_brute_force():
    rng = np.random.default_rng(4)
    a, b, c = (rng.normal(size=(200, 3)) for _ in range(3))
    p = rng.normal(size=(200, 3)) * 2
    q, bary = wrap.closest_on_triangles(p, a, b, c)
    s, t = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120))
    keep = s + t <= 1
    s, t = s[keep], t[keep]
    samples = a[:, None] + (b - a)[:, None] * s[None, :, None] + (c - a)[:, None] * t[None, :, None]
    brute = np.linalg.norm(samples - p[:, None], axis=2).min(axis=1)
    assert np.all(np.linalg.norm(q - p, axis=1) <= brute + 1e-9)
    assert np.allclose(np.einsum('pk,pkj->pj', bary, np.stack([a, b, c], 1)), q)


def test_loose_cloth_wraps_in_and_never_goes_inside():
    points, triangles = tube(0.15, 0.3, 0.7)
    out, record = run(points, triangles)
    middle = radius(out).reshape(31, 48)[8:23]
    assert middle.max() < 0.13                                  # 5 cm of looseness comes in most of the way
    assert radius(out).min() > 0.1 + 0.5 * EASE and record['inside_at_end'] == 0


def test_cloth_standing_well_off_the_body_hangs_and_keeps_its_width():
    widths = {}
    for cloth in (0.15, 0.2, 0.3):                              # 5, 10 and 20 cm off the body
        points, triangles = tube(cloth, 0.3, 0.7)
        widths[cloth] = np.median(radius(run(points, triangles)[0]).reshape(31, 48)[8:23]) - 0.1
    assert widths[0.3] > 0.18                                   # a wide trouser leg stays wide
    assert widths[0.15] < widths[0.2] < widths[0.3]             # and looser cloth always ends looser
    assert widths[0.2] < 0.09                                   # in between it still comes in part of the way


def test_relief_keeps_its_height():
    points, triangles = tube(0.2, 0.3, 0.7, bump=(15, 10, 0.03))
    out = radius(run(points, triangles)[0]).reshape(31, 48)
    assert out[15, 10] - np.median(out[15]) > 0.02               # a 3 cm pocket stays a pocket


def test_layers_keep_their_order():
    inner, t_inner = tube(0.17, 0.3, 0.7)
    outer, t_outer = tube(0.23, 0.3, 0.7)
    out, _ = run(np.vstack([inner, outer]), np.vstack([t_inner, t_outer + len(inner)]))
    assert np.all(radius(out[len(inner):]) > radius(out[:len(inner)]) + 0.005)


def test_cloth_inside_the_body_comes_out_to_ease():
    points, triangles = tube(0.09, 0.3, 0.7)
    out, record = run(points, triangles)
    assert radius(out).min() > 0.1 + 0.5 * EASE and record['contact_points'] > 0


def test_rigid_anchors_do_not_move():
    points, triangles = tube(0.2, 0.3, 0.7)
    anchored = np.arange(48 * 14, 48 * 16)
    out, _ = run(points, triangles, rigid=anchored)
    assert np.abs(out[anchored] - points[anchored]).max() < 1e-3


def test_free_hanging_cloth_keeps_its_looseness():
    labels = np.array(['skirt'] * 4 + ['torso'] * 4)
    triangles = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3], [4, 5, 6], [5, 6, 7], [4, 6, 7], [4, 5, 7]])
    keep = wrap.point_keep(labels, triangles, 8, 0.3)
    assert keep[:4].tolist() == [1.0] * 4 and keep[4:].tolist() == [0.3] * 4
    points, triangles = tube(0.2, 0.3, 0.7)
    out, _ = run(points, triangles, keep=np.ones(len(points)))
    assert np.abs(radius(out) - 0.2).max() < 0.01


@pytest.mark.parametrize('name', ['wrap_keep', 'wrap_bulk_radius_m', 'wrap_hanging_from_m', 'wrap_hanging_full_m',
                                  'contact_rounds', 'arap_iterations'])
def test_wrap_settings_exist(name):
    assert name in CONFIG


def shear(source, moved, triangles):
    a, b, c = (source[triangles[:, k]] for k in range(3))
    A, B, C = (moved[triangles[:, k]] for k in range(3))
    e1, e2 = b - a, c - a
    n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True)
    u = e1 / np.linalg.norm(e1, axis=1, keepdims=True); v = np.cross(n, u)
    S = np.stack([np.stack([(e1 * u).sum(1), (e1 * v).sum(1)], 1), np.stack([(e2 * u).sum(1), (e2 * v).sum(1)], 1)], 2)
    J = np.stack([B - A, C - A], 2) @ np.linalg.inv(S)
    sv = np.linalg.svd(J, compute_uv=False)
    return sv[:, 0] / sv[:, 1]


def test_a_printed_patch_keeps_its_shape_while_the_tube_wraps():
    points, triangles = tube(0.2, 0.3, 0.7)
    grid = np.arange(len(points)).reshape(31, 48)
    patch = grid[12:19, 4:12].ravel()
    in_patch = np.isin(triangles, patch).all(axis=1)
    everything = [('all', np.arange(len(points)), np.arange(len(BODY.triangles)))]
    keep = np.full(len(points), CONFIG['wrap_keep'])
    plain, _ = wrap.solve(points, triangles, keep, BODY, everything, np.zeros(0, int), CONFIG, HEIGHT)
    printed, _ = wrap.solve(points, triangles, keep, BODY, everything, np.zeros(0, int), CONFIG, HEIGHT, printed=patch)
    assert np.median(shear(points, printed, triangles[in_patch])) < np.median(shear(points, plain, triangles[in_patch]))
    far = np.isin(triangles, grid[:, 30:40].ravel()).all(axis=1)              # plain fabric away from the print
    assert np.abs(radius(printed)[grid[:, 30:40].ravel()] - radius(plain)[grid[:, 30:40].ravel()]).max() < 0.01
