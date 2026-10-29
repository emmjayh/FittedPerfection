"""Command line: outfits by name, the held-out test fits, and a smoke run that stops at intake."""
import json
import zipfile

import pytest

from fittedperfection.cli import main
from fittedperfection.hashing import file_sha256


@pytest.fixture
def project(home, tmp_path):
    for split, name in (('dev', 'moon'), ('test-fits', 'cloak')):
        archive = home/'outfits'/split/f'{name}.zip'
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'w') as bundle:      # a texture but no mesh: intake refuses it
            bundle.writestr('texture_diffuse.png', f'{name} texture')
    manifest = {'outfits': {
        name: {'split': split, 'file': f'outfits/{split}/{name}.zip',
               'sha256': file_sha256(home/'outfits'/split/f'{name}.zip'), 'bytes': 11}
        for split, name in (('dev', 'moon'), ('test-fits', 'cloak'))}}
    (home/'outfits/manifest.json').write_text(json.dumps(manifest))
    blender = tmp_path/'blender.exe'
    blender.write_bytes(b'stand-in')
    (home/'config.json').write_text(json.dumps({'blender': str(blender), 'bodies': 'assets/bodies'}))
    return home


def test_smoke_run_stops_at_intake(project, capsys):
    assert main(['run', 'moon']) == 1
    assert 'no OBJ or FBX' in capsys.readouterr().err
    state = json.loads((project/'runs/moon/run.json').read_text())
    assert state['status'] == 'stopped_at_intake' and (project/'runs/moon/live.html').is_file()
    assert main(['status', 'moon']) == 0
    assert 'stopped_at_intake' in capsys.readouterr().out


def test_test_fits_need_the_flag(project, capsys):
    assert main(['run', 'cloak']) == 1
    assert 'held out' in capsys.readouterr().err
    assert not (project/'runs/cloak').exists()
    assert main(['run', 'cloak', '--test-fit']) == 1          # allowed; stops at intake
    assert (project/'runs/cloak/run.json').is_file()


def test_changed_archive_is_refused(project, capsys):
    (project/'outfits/dev/moon.zip').write_bytes(b'tampered')
    assert main(['run', 'moon']) == 1
    assert 'no longer matches' in capsys.readouterr().err


def test_outfits_listing(project, capsys):
    assert main(['outfits']) == 0
    out = capsys.readouterr().out
    assert 'dev' in out and 'moon' in out and 'test-fits' in out


def test_confirm_facing_records_the_answer_for_the_runs_garment(home, capsys):
    folder = home/'runs'/'baggy'
    (folder/'full').mkdir(parents=True)
    (folder/'run.json').write_text(json.dumps({'garment_sha256': 'f00d'}))
    (folder/'full'/'intake.json').write_text(json.dumps({'facing': {
        'toe_direction': '-Y', 'turn_about_z_degrees': 0, 'status': 'provisional',
        'evidence': {'toe_over_heel_reach': [1.11, 1.07]}}}))
    assert main(['confirm-facing', 'baggy', '--front']) == 0
    assert json.loads((home/'confirmations'/'f00d.json').read_text())['facing'] == 'front'
    assert 'review-front.png' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(['confirm-facing', 'baggy'])                             # an answer is required
