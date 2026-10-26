"""The views stage's face-ID check: right pixels pass, mislabelled pixels are caught."""
import numpy as np

from fittedperfection import cameras
from fittedperfection.views import distance_to_named_triangle

POSITIONS = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.0],      # triangle 0, facing the front
                      [-0.5, 0.0, 1.2], [0.5, 0.0, 1.2], [0.0, 0.0, 1.8]])     # triangle 1, above it
TRIANGLES = np.array([[0, 1, 2], [3, 4, 5]])
CAMERA = cameras.plan(POSITIONS, 4, 100)[0]          # the front camera, 100 px square


def pixel_of(point):
    x, y, _ = cameras.project(CAMERA, np.array([point]))
    return int(y[0]), int(x[0])


def test_pixels_inside_their_triangle_are_at_distance_zero():
    ids = np.full((100, 100), -1)
    ids[pixel_of([0.0, 0.0, 0.3])] = 0
    ids[pixel_of([0.0, 0.0, 1.4])] = 1
    assert distance_to_named_triangle(CAMERA, ids, POSITIONS, TRIANGLES).max() == 0.0


def test_a_pixel_naming_the_wrong_triangle_is_far_from_it():
    ids = np.full((100, 100), -1)
    ids[pixel_of([0.0, 0.0, 1.4])] = 0                # inside triangle 1 but labelled 0
    assert distance_to_named_triangle(CAMERA, ids, POSITIONS, TRIANGLES).max() > 5.0
