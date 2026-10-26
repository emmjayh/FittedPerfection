"""Camera plan: calibrated frames look where they should, everything stays in frame."""
import numpy as np

from fittedperfection import cameras

FIGURE = np.array([[-0.8, -0.3, 0.0], [0.8, 0.3, 1.9], [0.2, -0.3, 0.05]])     # tip of the toes at the front


def test_front_left_back_right_frames_face_the_right_way():
    plan = {c['name']: c for c in cameras.plan(FIGURE, 72, 1024)}
    frames = cameras.calibrated_frames(72)
    assert frames == {'front': 0, 'left': 18, 'back': 36, 'right': 54}
    assert np.allclose(plan['orbit-000']['forward'], [0, 1, 0])        # front: looking from -Y towards +Y
    assert np.allclose(plan['orbit-018']['forward'], [-1, 0, 0])       # left: looking from +X (own left)
    assert np.allclose(plan['orbit-036']['forward'], [0, -1, 0])
    assert np.allclose(plan['orbit-054']['forward'], [1, 0, 0])
    assert np.allclose(plan['orbit-000']['right'], [1, 0, 0])          # own left appears on the image's right


def test_everything_projects_inside_every_image_and_up_is_up():
    for camera in cameras.plan(FIGURE, 72, 1024):
        x, y, _ = cameras.project(camera, FIGURE)
        assert (x > 0).all() and (x < 1024).all() and (y > 0).all() and (y < 1024).all(), camera['name']
    front = cameras.plan(FIGURE, 72, 1024)[0]
    _, y, depth = cameras.project(front, np.array([[0, 0, 1.9], [0, 0, 0.0], [0, -0.3, 1.0], [0, 0.3, 1.0]]))
    assert y[0] < y[1]                       # the top of the garment is near the top of the image
    assert depth[2] > depth[3]               # the front is nearer to the front camera
