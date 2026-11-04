"""Display underwear (changes for this build): bands from each body's own measures, never part of a fit."""
import json
from pathlib import Path

import pytest

from fittedperfection import body, config, underwear

REAL = Path(__file__).resolve().parents[1]/'assets'/'bodies'/'bodies.json'
needs_bodies = pytest.mark.skipif(not REAL.is_file(), reason='bodies not made')


@needs_bodies
@pytest.mark.parametrize('body_id', list(config.BODIES))
def test_bands_are_ordered_and_the_top_is_for_female_bodies_only(body_id, tmp_path):
    record = underwear.paint(body_id, cache=tmp_path)
    b = record['bands']
    assert b['briefs_trunk'] < b['briefs_leg'] < b['briefs_top']
    assert record['top'] == (config.BODIES[body_id] == 'female')
    if record['top']:
        assert b['briefs_top'] < b['top_low'] < b['top_high']
    else:
        assert 'top_low' not in b
    count = len(body.load(body_id).positions)
    assert len(record['vertex_arm']) == len(record['vertex_leg']) == count


@needs_bodies
def test_paint_is_deterministic_and_cached_by_the_body_file(tmp_path):
    first = underwear.paint('body_01_medium', cache=tmp_path/'a')
    again = underwear.paint('body_01_medium', cache=tmp_path/'b')
    assert first == again
    files = list((tmp_path/'a').glob('*.json'))
    assert len(files) == 1 and json.loads(files[0].read_text(encoding='utf-8')) == first
    assert underwear.paint('body_01_medium', cache=tmp_path/'a') == first          # read back, not recomputed
    assert underwear._cache_key('body_01_medium') != underwear._cache_key('body_02_tall_slender')


def test_the_rule_paints_briefs_and_top_but_never_arms_or_thighs():
    record = {'arm_limit': 0.5, 'leg_limit': 0.5, 'top': True,
              'bands': {'briefs_trunk': 0.70, 'briefs_leg': 0.74, 'briefs_top': 0.90, 'top_low': 1.1, 'top_high': 1.25}}
    assert underwear.painted(record, 0.80, arm=0.0, leg=0.0)
    assert underwear.painted(record, 0.72, arm=0.0, leg=0.0)              # the crotch, on the trunk
    assert not underwear.painted(record, 0.72, arm=0.0, leg=1.0)          # the thigh at the same height
    assert not underwear.painted(record, 0.80, arm=0.9, leg=0.0)          # a hand at the hips
    assert underwear.painted(record, 1.2, arm=0.0, leg=0.0)
    assert not underwear.painted(record, 1.0, arm=0.0, leg=0.0)           # the belly between
    assert not underwear.painted({**record, 'top': False}, 1.2, arm=0.0, leg=0.0)


def test_the_fit_never_imports_it():
    package = Path(underwear.__file__).parent
    for name in ('fit.py', 'wrap.py', 'donor.py', 'shoes.py', 'fullres.py', 'settings.py', 'label.py', 'bind.py'):
        assert 'underwear' not in (package/name).read_text(encoding='utf-8')
