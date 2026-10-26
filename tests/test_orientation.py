"""Facing from the feet: found for every direction, refused when unclear."""
import numpy as np
import pytest

from fittedperfection.orientation import OrientationUnclear, facing, turn_about_z

rng = np.random.default_rng(7)


def leg(centre_x, toe=(0.0, -1.0), foot=True, height=1.8):
    """A shin cylinder plus a foot that reaches 0.18 m towards the toes and 0.06 m back."""
    angle = rng.uniform(0, 2 * np.pi, 3000)
    z = rng.uniform(0.0, height * 0.5, 3000)
    shin = np.stack([centre_x + 0.05 * np.cos(angle), 0.05 * np.sin(angle), z], axis=1)
    if not foot:
        return shin
    toe = np.array(toe)
    along = rng.uniform(-0.06, 0.18, 2000)
    across = rng.uniform(-0.045, 0.045, 2000)
    side = np.array([-toe[1], toe[0]])
    xy = np.array([centre_x, 0.0]) + along[:, None] * toe + across[:, None] * side
    return np.vstack([shin, np.column_stack([xy, rng.uniform(0.0, 0.05, 2000)])])


def figure(toe=(0.0, -1.0)):
    head = np.array([[0.0, 0.0, 1.8]])
    return np.vstack([leg(-0.12, toe), leg(0.12, toe), head])


@pytest.mark.parametrize('name, turn', [('-Y', 0), ('+X', -90), ('+Y', 180), ('-X', 90)])
def test_toes_are_found_in_every_direction(name, turn):
    points = turn_about_z(figure(), -turn)       # a figure facing -Y, turned so its toes point at `name`
    result = facing(points)
    assert result['toe_direction'] == name and result['turn_about_z_degrees'] == turn
    turned = turn_about_z(points, result['turn_about_z_degrees'])
    assert facing(turned)['toe_direction'] == '-Y'


def test_clear_feet_are_confirmed():
    assert facing(figure())['status'] == 'confirmed'


def test_cloth_trailing_behind_the_heels_makes_it_provisional():
    points = figure()
    trail = []
    for x in (-0.12, 0.12):                      # hem cloth on the floor reaching 0.15 m behind each heel
        trail.append(np.column_stack([x + rng.uniform(-0.04, 0.04, 800), rng.uniform(0.0, 0.15, 800),
                                      rng.uniform(0.0, 0.03, 800)]))
    result = facing(np.vstack([points, *trail]))
    assert result['status'] == 'provisional' and result['toe_direction'] == '-Y'
    assert 1.05 <= min(result['evidence']['toe_over_heel_reach']) < 1.4


def test_high_heels_touching_the_floor_in_two_places_are_one_foot_each():
    shins = np.vstack([leg(-0.12, foot=False), leg(0.12, foot=False), [[0, 0, 1.8]]])
    pieces = []
    for x in (-0.12, 0.12):
        pad = np.column_stack([x + rng.uniform(-0.045, 0.045, 1500), rng.uniform(-0.18, -0.07, 1500),
                               rng.uniform(0, 0.02, 1500)])                     # ball of the foot, at the front
        tip = np.column_stack([x + rng.uniform(-0.012, 0.012, 300), rng.uniform(0.05, 0.07, 300),
                               rng.uniform(0, 0.02, 300)])                      # separate heel tip behind
        pieces += [pad, tip]
    result = facing(np.vstack([shins, *pieces]))
    assert result['toe_direction'] == '-Y' and result['status'] == 'confirmed'


def test_wide_hollow_trouser_legs_nearly_touching_are_still_two_legs():
    points = [figure()]
    for x in (-0.19, 0.19):                       # 0.36 m wide hollow legs, 2 cm apart at the middle
        angle = rng.uniform(0, 2 * np.pi, 6000)
        points.append(np.column_stack([x + 0.18 * np.cos(angle), 0.18 * np.sin(angle), rng.uniform(0.3, 0.5, 6000)]))
    assert facing(np.vstack(points))['toe_direction'] == '-Y'


def test_a_long_skirt_hiding_the_feet_is_refused():
    angle = rng.uniform(0, 2 * np.pi, 8000)
    radius = rng.uniform(0.0, 0.35, 8000)
    skirt = np.stack([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0, 0.4, 8000)], axis=1)
    with pytest.raises(OrientationUnclear, match='not separate'):
        facing(np.vstack([skirt, [[0, 0, 1.8]]]))


def test_round_feet_with_no_clear_toe_are_refused():
    points = np.vstack([leg(-0.12, foot=False), leg(0.12, foot=False), [[0, 0, 1.8]]])
    disc = []
    for x in (-0.12, 0.12):
        angle = rng.uniform(0, 2 * np.pi, 2000)
        radius = rng.uniform(0, 0.09, 2000)
        disc.append(np.stack([x + radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0, 0.05, 2000)], axis=1))
    with pytest.raises(OrientationUnclear, match='too similar|different ways'):
        facing(np.vstack([points, *disc]))
