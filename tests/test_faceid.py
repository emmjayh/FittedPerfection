"""The face-ID code round-trips and survives small colour rounding."""
import numpy as np
import pytest

from fittedperfection import faceid


def test_round_trip_for_every_index():
    colours = faceid.encode(40_000)
    image = colours.reshape(200, 200, 3)
    assert np.array_equal(faceid.decode(image).ravel(), np.arange(40_000))


def test_black_is_background():
    assert faceid.decode(np.zeros((2, 2, 3), np.uint8)).tolist() == [[-1, -1], [-1, -1]]


@pytest.mark.parametrize('shift', [-2, -1, 1, 2])
def test_small_rounding_does_not_change_the_index(shift):
    colours = faceid.encode(68_920).astype(np.int64) + shift
    assert np.array_equal(faceid.decode(np.clip(colours, 0, 255).astype(np.uint8)), np.arange(68_920))


def test_too_many_triangles_is_refused():
    with pytest.raises(ValueError):
        faceid.encode(faceid.LIMIT + 1)
