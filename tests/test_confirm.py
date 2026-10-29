"""Owner confirmations: kept by garment, written only on purpose, an earlier answer set aside."""
import json

import pytest

from fittedperfection import confirm


def intake_folder(tmp_path, status='provisional'):
    folder = tmp_path/'run'/'full'
    folder.mkdir(parents=True)
    (folder/'intake.json').write_text(json.dumps({'facing': {
        'toe_direction': '-Y', 'turn_about_z_degrees': 0, 'status': status,
        'evidence': {'toe_over_heel_reach': [1.11, 1.07]}}}))
    for name in confirm.RENDERS:
        (folder/name).write_bytes(b'png ' + name.encode())
    return folder


def test_an_answer_is_kept_by_garment_with_what_was_shown(home, tmp_path):
    folder = intake_folder(tmp_path)
    assert confirm.read('abc') is None and confirm.file_sha256('abc') is None
    confirm.write_facing('abc', 'front', folder)
    record = confirm.read('abc')
    assert record['facing'] == 'front' and record['by'] == 'owner'
    assert record['shown']['toe_over_heel_reach'] == [1.11, 1.07] and set(record['shown']['renders']) == set(confirm.RENDERS)
    assert confirm.matches(record, {'toe_direction': '-Y', 'turn_about_z_degrees': 0})
    assert not confirm.matches(record, {'toe_direction': '-Y', 'turn_about_z_degrees': 90})
    assert confirm.path_for('abc').parent == home/'confirmations'


def test_a_new_answer_sets_the_old_one_aside(home, tmp_path):
    folder = intake_folder(tmp_path)
    confirm.write_facing('abc', 'back', folder)
    confirm.write_facing('abc', 'front', folder)
    assert confirm.read('abc')['facing'] == 'front'
    assert json.loads((home/'confirmations'/'abc.json.replaced-1').read_text())['facing'] == 'back'


def test_only_a_provisional_facing_and_a_known_answer_are_accepted(home, tmp_path):
    with pytest.raises(confirm.ConfirmError, match='answer'):
        confirm.write_facing('abc', 'maybe', intake_folder(tmp_path))
    with pytest.raises(confirm.ConfirmError, match='nothing to confirm'):
        confirm.write_facing('def', 'front', intake_folder(tmp_path/'other', status='confirmed'))
