"""Bind decisions: facing confirmation and refusal, the crotch prior, gap-tolerant sections."""
import numpy as np
import pytest

from fittedperfection import bind, settings

CONFIG = settings.section('bind')


def feet(direction=(0.0, -1.0), ratio=2.5):
    return {side: {'toe_direction': [direction[0], direction[1], 0.0], 'toe_over_heel_reach': ratio}
            for side in ('left', 'right')}


def test_clear_shoes_confirm_a_provisional_facing():
    assert bind.check_facing(feet(), 'provisional', CONFIG)['status'] == 'confirmed by footwear'


def test_weak_shoe_evidence_refuses_a_provisional_facing():
    with pytest.raises(bind.BindError, match='do not clearly confirm'):
        bind.check_facing(feet(ratio=1.12), 'provisional', CONFIG)


def test_shoes_pointing_away_from_the_front_refuse():
    with pytest.raises(bind.BindError, match='do not point to the front'):
        bind.check_facing(feet(direction=(0.0, 1.0)), 'confirmed', CONFIG)


def test_no_shoes_keeps_a_confirmed_facing_but_refuses_a_provisional_one():
    assert bind.check_facing({}, 'confirmed', CONFIG)['status'] == 'confirmed'
    with pytest.raises(bind.BindError, match='no footwear'):
        bind.check_facing({}, 'provisional', CONFIG)


def test_crotch_uses_the_measurement_inside_the_prior_band_and_the_edge_outside():
    shoulder = 1.6                                       # prior 0.944 m, band +- 0.112 m
    assert bind.crotch(0.95, shoulder, CONFIG)['source'].startswith('measured')
    outside = bind.crotch(1.2, shoulder, CONFIG)
    assert outside['source'].startswith('prior band edge') and np.isclose(outside['height_m'], 0.944 + 0.112, atol=1e-3)
    assert bind.crotch(None, shoulder, CONFIG)['source'].startswith('prior (no branching')


def test_sections_skip_gaps():
    points = np.array([[0, 0, z] for z in (0.0, 0.01, 0.2, 0.21)], dtype=float)
    chain = bind.sections(points, np.ones(4), np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.05)
    assert len(chain) == 2 and np.isclose(chain[0][2], 0.005) and np.isclose(chain[1][2], 0.205)


INTAKE_FACING = {'toe_direction': '-Y', 'turn_about_z_degrees': 0, 'status': 'provisional'}


def owner(answer, turn=0):
    return {'facing': answer, 'by': 'owner', 'shown': {'toe_direction': '-Y', 'turn_about_z_degrees': turn}}


def test_weak_shoes_ask_the_owner_and_only_a_matching_answer_decides():
    weak = feet(ratio=1.12)
    with pytest.raises(bind.FacingUnconfirmed, match='confirm-facing'):
        bind.check_facing(weak, 'provisional', CONFIG, INTAKE_FACING, None)
    record = bind.check_facing(weak, 'provisional', CONFIG, INTAKE_FACING, owner('front'))
    assert record['status'] == 'confirmed by the owner' and record['confirmation']['facing'] == 'front'
    with pytest.raises(bind.FacingRejected, match='shows the back'):
        bind.check_facing(weak, 'provisional', CONFIG, INTAKE_FACING, owner('back'))
    with pytest.raises(bind.FacingUnconfirmed):                       # answered about a different turn
        bind.check_facing(weak, 'provisional', CONFIG, INTAKE_FACING, owner('front', turn=180))
    assert bind.check_facing({}, 'provisional', CONFIG, INTAKE_FACING, owner('front'))['status'] == \
        'confirmed by the owner'


def test_an_answer_never_overrides_shoes_that_contradict_the_front():
    with pytest.raises(bind.BindError, match='do not point to the front'):
        bind.check_facing(feet(direction=(0.0, 1.0)), 'provisional', CONFIG, INTAKE_FACING, owner('front'))
    assert bind.check_facing(feet(), 'provisional', CONFIG, INTAKE_FACING, None)['status'] == 'confirmed by footwear'
