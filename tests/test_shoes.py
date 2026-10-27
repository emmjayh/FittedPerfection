"""Step E shoes: frames, sole thickness, placement on a foot, shafts, ghost removal."""
import numpy as np

from fittedperfection import shoes


def block(length, width, z_low, z_high, origin=(0.0, 0.0), step=0.005, yaw_degrees=0.0):
    """Points filling a box whose length runs to the front (-Y) from a heel at `origin`."""
    x, y, z = np.meshgrid(np.arange(-width / 2, width / 2 + 1e-9, step), np.arange(0, -length - 1e-9, -step),
                          np.arange(z_low, z_high + 1e-9, step), indexing='ij')
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    a = np.radians(yaw_degrees)
    turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return points @ turn.T + [origin[0], origin[1], 0]


def shoe():
    """A sole 3 cm thick and 12 cm wide under an upper 9 cm wide: a 30 cm shoe."""
    return np.vstack([block(0.30, 0.12, 0.0, 0.0249), block(0.30, 0.09, 0.03, 0.10)])


def test_frame_finds_heel_length_and_front():
    f = shoes.frame(block(0.25, 0.10, 0.0, 0.05, origin=(0.3, 0.1)))
    assert np.allclose(f['origin'], [0.3, 0.1, 0.0], atol=0.006)
    assert abs(f['length'] - 0.25) < 0.006 and abs(f['width'] - 0.10) < 0.006
    assert f['rotation'][:, 0] @ [0, -1, 0] > 0.99                     # forward is the front (-Y)


def test_sole_thickness_is_the_height_of_the_sole_edge():
    thickness, found = shoes.sole_thickness(shoe(), 0.08)
    assert found and abs(thickness - 0.03) < 0.006


def test_sole_thickness_falls_back_without_an_edge():
    thickness, found = shoes.sole_thickness(block(0.30, 0.10, 0.0, 0.10), 0.08)
    assert not found and thickness == shoes.DEFAULT_SOLE_M


def test_placement_scales_to_the_foot_and_puts_the_footbed_on_the_sole():
    donor_foot = block(0.25, 0.10, 0.0, 0.06)
    target_foot = block(0.20, 0.08, 0.0, 0.05, origin=(0.3, 0.1), yaw_degrees=10)
    points = shoe()
    move, record = shoes.placement(points, donor_foot, target_foot, 0.03, 0.25)
    assert abs(record['scale'] - 0.8) < 0.03
    moved = move(points)
    pick = np.random.default_rng(5).choice(len(points), 200, replace=False)
    before = np.linalg.norm(points[pick, None] - points[None, pick], axis=2)
    after = np.linalg.norm(moved[pick, None] - moved[None, pick], axis=2)
    assert np.allclose(after, record['scale'] * before, atol=1e-9)        # same shape, scaled
    assert abs(moved[:, 2].min() + record['scale'] * 0.03) < 1e-9          # the sole goes under the floor
    heel = moved[np.argmin(np.linalg.norm(points - [0, 0, 0], axis=1))]
    assert np.linalg.norm(heel[:2] - [0.3, 0.1]) < 0.02
    assert abs(record['shoe_to_foot_fitted'] - record['shoe_to_foot_source']) < 0.05


def test_a_wide_foot_raises_the_scale():
    donor_foot = block(0.25, 0.10, 0.0, 0.06)
    wide = block(0.20, 0.11, 0.0, 0.05)
    _, record = shoes.placement(shoe(), donor_foot, wide, 0.03, 0.25)
    assert abs(record['scale'] - 1.1) < 0.06


def test_scale_follows_the_foot_the_shoe_was_made_for():
    donor_foot = block(0.20, 0.08, 0.0, 0.06)                 # the donor's foot is small; the shoe was made for 25 cm
    target_foot = block(0.25, 0.10, 0.0, 0.05)
    _, record = shoes.placement(shoe(), donor_foot, target_foot, 0.03, 0.25)
    assert abs(record['scale'] - 1.0) < 0.04 and abs(record['shoe_to_foot_source'] - 1.2) < 0.03


def test_shaft_is_what_lies_above_the_ankle_line():
    points = np.array([[0, 0, 0.02], [0, 0, 0.09], [0, 0, 0.3], [0, 0, 0.5]])
    assert shoes.shaft(np.arange(4), points, 0.1).tolist() == [2, 3]


def heightfield(length, width, top, step=0.005):
    """A closed-enough shoe surface for the height map: a top sheet over the footprint and a bottom sheet."""
    xs, ys = np.arange(-width / 2, width / 2 + 1e-9, step), np.arange(0, -length - 1e-9, -step)
    x, y = np.meshgrid(xs, ys, indexing='ij')
    grid = np.column_stack([x.ravel(), y.ravel()])
    points = np.vstack([np.column_stack([grid, np.full(len(grid), top)]), np.column_stack([grid, np.zeros(len(grid))])])
    nx, ny = len(xs), len(ys)
    triangles = []
    for sheet in (0, nx * ny):
        for i in range(nx - 1):
            for j in range(ny - 1):
                a = sheet + i * ny + j
                triangles += [[a, a + ny, a + 1], [a + 1, a + ny, a + ny + 1]]
    return points, np.array(triangles)


def test_ghost_removal_hides_what_the_shoe_covers_and_keeps_the_shin():
    points, triangles = heightfield(0.30, 0.12, 0.10)
    tiny = np.array([[0, 0.004, 0], [0.004, 0, 0], [0, 0, 0.004]])
    centres = {'foot inside': [0, -0.1, 0.03], 'toe through the sole': [0, -0.25, -0.01],
               'shin above': [0, -0.05, 0.25], 'beside the shoe': [0.2, -0.1, 0.03]}
    body_points = np.vstack([tiny + c for c in centres.values()])
    body_triangles = np.arange(len(body_points)).reshape(-1, 3)
    hidden = set(shoes.hidden_faces(body_points, body_triangles, points, triangles, 0.005).tolist())
    assert hidden == {0, 1}
