"""Stage 7 review: penetration, coverage, test poses, Wonder findings (spend-guarded) and their parsing."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial import cKDTree

from fittedperfection import config, review, wonder, wrap

REAL = Path(__file__).resolve().parents[1]/'assets'/'bodies'/'bodies.json'


def cylinder(radius, z_low, z_high, around=64, rings=41):
    angle = np.linspace(0, 2 * np.pi, around, endpoint=False)
    z = np.linspace(z_low, z_high, rings)
    points = np.stack([radius * np.cos(angle)[None].repeat(rings, 0), radius * np.sin(angle)[None].repeat(rings, 0),
                       np.repeat(z[:, None], around, 1)], -1).reshape(-1, 3)
    triangles = []
    for k in range(rings - 1):
        for s in range(around):
            a, b = k * around + s, k * around + (s + 1) % around
            triangles += [[a, b, a + around], [b, b + around, a + around]]
    return points, np.array(triangles)


BODY = wrap.Skin(*cylinder(0.1, 0.0, 1.0))


def test_penetration_counts_cloth_inside_the_body():
    inside, _ = cylinder(0.09, 0.3, 0.7, rings=11)
    outside, _ = cylinder(0.12, 0.3, 0.7, rings=11)
    record = review.penetration(np.vstack([inside, outside]), BODY)
    assert abs(record['inside_share'] - 0.5) < 0.01 and record['clearance_cm']['p50'] > -1.5


def test_coverage_finds_cloth_along_the_normal_and_misses_a_hole():
    sleeve, _ = cylinder(0.12, 0.3, 0.6)
    points = BODY.positions
    hit = review.covered(points, BODY.vertex_normals, cKDTree(sleeve), sleeve, 0.06, 0.5)
    z = points[:, 2]
    assert hit[(z > 0.35) & (z < 0.55)].all() and not hit[z > 0.8].any()


@pytest.mark.skipif(not REAL.is_file(), reason='bodies not made')
def test_test_poses_move_skin_and_cloth_by_the_same_weights():
    from fittedperfection import body as bodies
    b = bodies.load('body_01_medium')
    turn = review.bone_turn(b, b.positions, 'lowerarm01', 60, -1.0)
    assert np.abs(review.lbs(turn, b.joints, b.weights, b.positions) - bodies.skin(b, turn)).max() < 1e-9
    wrist = np.argmin(np.linalg.norm(b.positions - b.joint('wrist.L'), axis=1))
    moved = bodies.skin(b, turn)[wrist] - b.positions[wrist]
    assert moved[1] < -0.05                                                  # the forearm comes forward
    knee = review.bone_turn(b, b.positions, 'lowerleg01', 30, 1.0)
    ankle = np.argmin(np.linalg.norm(b.positions - b.joint('foot.L'), axis=1))
    assert (bodies.skin(b, knee)[ankle] - b.positions[ankle])[1] > 0.05     # the shin goes back


def test_findings_are_parsed_with_or_without_code_fences():
    assert review.parse_findings('```json\n{"findings": [], "overall": "ok"}\n```')['overall'] == 'ok'
    assert review.parse_findings('not json')['unparsed'] == 'not json'


def sheets(folder):
    for view in ('front', 'back'):
        Image.new('RGB', (400, 200), (10 if view == 'front' else 20, 30, 40)).save(folder/f'sheet-{view}.png')


def test_wonder_sends_nothing_without_a_spending_allowance(tmp_path, monkeypatch):
    sheets(tmp_path)
    monkeypatch.setattr(config, 'runs_folder', lambda: tmp_path)
    monkeypatch.setattr(config, 'spend_allowance_usd', lambda: 0.0)
    monkeypatch.setattr(review.secrets, 'api_key', lambda: 'test-key')
    monkeypatch.setattr(wonder.WonderClient, '_call', lambda *a, **k: pytest.fail('a request was sent'))
    record = review.wonder_review(tmp_path, np.array(['torso', 'sleeve.left', 'skin']))
    assert record['status'] == 'skipped (no spending allowed)' and record['parts'] == ['sleeve', 'torso']


def test_wonder_review_sends_one_composite_and_keeps_the_findings(tmp_path, monkeypatch):
    sheets(tmp_path)
    sent = []

    def fake_call(self, method, path, body, content_type):
        sent.append(json.loads(body))
        return {'id': 'r1', 'status': 'completed', 'output': [{'content': [
            {'type': 'output_text', 'text': '{"findings": [{"body": 2, "view": "front", "where": "left knee", '
                                            '"problem": "skin_showing", "what": "gap"}], "overall": "one gap"}'}]}]}

    monkeypatch.setattr(config, 'runs_folder', lambda: tmp_path)
    monkeypatch.setattr(config, 'spend_allowance_usd', lambda: 1.0)
    monkeypatch.setattr(review.secrets, 'api_key', lambda: 'test-key')
    monkeypatch.setattr(wonder.WonderClient, '_call', fake_call)
    record = review.wonder_review(tmp_path, np.array(['torso']))
    assert record['status'] == 'reviewed' and record['findings']['findings'][0]['problem'] == 'skin_showing'
    content = sent[0]['input'][0]['content']
    assert sum(part['type'] == 'input_image' for part in content) == 1 and 'torso' in content[0]['text']
    assert sent[0]['reasoning']['effort'] == 'minimal'
    again = review.wonder_review(tmp_path, np.array(['torso']))           # cached: no second request
    assert len(sent) == 1 and again['call']['cached']
