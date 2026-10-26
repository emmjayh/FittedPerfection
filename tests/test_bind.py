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
