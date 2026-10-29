"""The app layer: uploads, the held-out guard, jobs with spend approval, and the write protection."""
import hashlib
import http.client
import io
import json
import sys
import textwrap
import threading
import time
import zipfile

import pytest

from fittedperfection import app, config, confirm, viewer


def archive_bytes(names=('base.obj', 'texture_diffuse.png')):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        for name in names:
            bundle.writestr(name, b'v 0 0 0\n' if name.endswith('.obj') else b'png')
    return buffer.getvalue()


FACING_PIPELINE = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    args = sys.argv[1:]
    name = args[args.index('--name') + 1]
    folder = Path(os.environ['FP_TEST_RUNS'])/name
    (folder/'full').mkdir(parents=True, exist_ok=True)
    (folder/'full'/'intake.json').write_text(json.dumps({'facing': {'toe_direction': '-Y', 'turn_about_z_degrees': 0,
        'status': 'provisional', 'evidence': {'toe_over_heel_reach': [1.11, 1.07]}}}))
    for view in ('review-front.png', 'review-side.png'):
        (folder/'full'/view).write_bytes(b'png')
    done = {'status': 'complete', 'seconds': 0.1}
    stages = {n: done for n in ('intake', 'views', 'vision', 'label')}
    answers = list(Path(os.environ['FP_TEST_CONFIRMATIONS']).glob('*.json'))
    if not answers:
        stages['bind'] = {'status': 'stopped', 'reason': 'FacingUnconfirmed: the shoes do not clearly confirm it'}
        state = {'status': 'stopped_at_bind', 'stages': stages}
    elif json.loads(answers[0].read_text())['facing'] == 'back':
        stages['bind'] = {'status': 'stopped', 'reason': 'FacingRejected: you said the front view shows the back'}
        state = {'status': 'stopped_at_bind', 'stages': stages}
    else:
        stages.update({n: done for n in ('bind', 'fit', 'review')})
        state = {'status': 'complete', 'stages': stages}
    (folder/'run.json').write_text(json.dumps(state))
''')


FAKE_PIPELINE = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    args = sys.argv[1:]
    name, allowance = args[args.index('--name') + 1], float(args[args.index('--allow-spend') + 1])
    folder = Path(os.environ['FP_TEST_RUNS'])/name
    folder.mkdir(parents=True, exist_ok=True)
    done = {'status': 'complete', 'seconds': 0.1}
    time.sleep(float(os.environ.get('FP_TEST_SLEEP', '0')))
    if allowance < 0.4:
        stages = {'intake': done, 'views': done, 'vision': {'status': 'stopped', 'needs_usd': 0.4,
                  'reason': 'SpendRefused: vision needs 12 new SAM requests, about $0.40; allowed $0.00.'}}
        state = {'status': 'stopped_at_vision', 'stages': stages}
    else:
        stages = {n: done for n in ('intake', 'views', 'vision', 'label', 'bind', 'fit', 'review')}
        state = {'status': 'complete', 'stages': stages, 'allowance': allowance}
    (folder/'run.json').write_text(json.dumps(state))
''')


@pytest.fixture
def place(tmp_path, monkeypatch):
    runs, uploads = tmp_path/'runs', tmp_path/'uploads'
    runs.mkdir()
    monkeypatch.setattr(config, 'runs_folder', lambda: runs)
    monkeypatch.setattr(app, 'uploads_folder', lambda: uploads)
    monkeypatch.setattr(config, 'outfits', lambda: {})
    monkeypatch.setenv('FP_TEST_RUNS', str(runs))
    script = tmp_path/'fake_pipeline.py'
    script.write_text(FAKE_PIPELINE)
    return {'runs': runs, 'uploads': uploads, 'command': [sys.executable, str(script)]}


def wait_until_done(jobs, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = jobs.status()
        if status['state'] != 'running':
            return status
        time.sleep(0.1)
    raise AssertionError('the job did not finish')


def store(data, name='My Outfit.zip'):
    return app.store_upload(io.BytesIO(data).read, len(data), name)


def test_an_archive_is_kept_under_its_hash_after_checking_its_listing(place):
    data = archive_bytes()
    answer = store(data)
    assert answer['upload'] == hashlib.sha256(data).hexdigest() and answer['name'] == 'my-outfit'
    assert (place['uploads']/f'{answer["upload"]}.zip').read_bytes() == data


@pytest.mark.parametrize('data', [b'not a zip at all', archive_bytes(('nested/base.obj', 'texture_diffuse.png')),
                                  archive_bytes(('base.obj', 'texture_diffuse.png', 'run.exe')),
                                  archive_bytes(('base.obj',))])
def test_anything_but_a_garment_archive_is_refused(place, data):
    with pytest.raises(app.AppError):
        store(data)
    assert not any(place['uploads'].glob('*.zip'))


def test_an_oversize_upload_is_refused_before_reading(place, monkeypatch):
    monkeypatch.setattr(app, 'MAX_UPLOAD_BYTES', 10)
    with pytest.raises(app.AppError):
        store(archive_bytes())


def test_a_held_out_outfit_needs_the_test_fit_tick(place, monkeypatch):
    data = archive_bytes()
    monkeypatch.setattr(config, 'outfits', lambda: {'secret': {'split': 'test-fits', 'sha256': hashlib.sha256(data)
                                                               .hexdigest(), 'path': place['uploads']/'x.zip'}})
    answer = store(data)
    assert answer['held_out'] == 'secret'
    jobs = app.Jobs(place['command'])
    with pytest.raises(app.AppError, match='test fit'):
        jobs.start({'upload': answer['upload'], 'name': 'try'})


def test_a_new_outfit_starts_with_nothing_allowed_asks_then_reuses_the_run(place):
    answer = store(archive_bytes())
    jobs = app.Jobs(place['command'])
    jobs.start({'upload': answer['upload'], 'name': 'Fresh Fit'})
    status = wait_until_done(jobs)
    assert status['state'] == 'needs_spend' and status['needs_usd'] == 0.4 and status['needs_requests'] == 12
    assert status['allowance_usd'] == 0.0 and status['name'] == 'fresh-fit'
    jobs.approve(wonder=True)
    status = wait_until_done(jobs)
    run = json.loads((place['runs']/'fresh-fit'/'run.json').read_text())
    assert status['state'] == 'complete' and 0.45 <= run['allowance'] <= 0.46     # rounded up, never down


def test_one_job_at_a_time_and_cancel(place, monkeypatch):
    monkeypatch.setenv('FP_TEST_SLEEP', '30')
    answer = store(archive_bytes())
    jobs = app.Jobs(place['command'])
    jobs.start({'upload': answer['upload'], 'name': 'slow'})
    with pytest.raises(app.AppError, match='already running'):
        jobs.start({'upload': answer['upload'], 'name': 'other'})
    assert jobs.cancel()['state'] == 'cancelled'
    with pytest.raises(app.AppError):
        jobs.approve()


@pytest.fixture
def server(place, monkeypatch):
    monkeypatch.setattr(viewer, 'JOBS', app.Jobs(place['command']))
    httpd = viewer.ViewerServer(('127.0.0.1', 0), viewer.ViewerHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=b'', headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    return response.status, json.loads(response.read() or b'null')


def test_writes_only_from_the_viewer_page(server):
    data = archive_bytes()
    page = {'Origin': f'http://127.0.0.1:{server}', 'X-FittedPerfection': '1', 'X-Filename': 'a.zip'}
    assert request(server, 'POST', '/api/uploads', data, {**page, 'Origin': 'http://evil.example'})[0] == 403
    assert request(server, 'POST', '/api/uploads', data, {k: v for k, v in page.items()
                                                          if k != 'X-FittedPerfection'})[0] == 403
    code, answer = request(server, 'POST', '/api/uploads', data, page)
    assert code == 200 and answer['upload'] == hashlib.sha256(data).hexdigest()
    assert request(server, 'GET', '/api/jobs/current')[1] == {'state': 'idle'}


@pytest.fixture
def facing_place(place, tmp_path, monkeypatch):
    answers = tmp_path/'confirmations'
    monkeypatch.setattr(confirm, 'folder', lambda: answers)
    monkeypatch.setenv('FP_TEST_CONFIRMATIONS', str(answers))
    answers.mkdir()
    script = tmp_path/'facing_pipeline.py'
    script.write_text(FACING_PIPELINE)
    return {**place, 'answers': answers, 'command': [sys.executable, str(script)]}


def test_an_unclear_facing_asks_the_owner_and_their_answer_restarts_the_run(facing_place):
    answer = store(archive_bytes())
    jobs = app.Jobs(facing_place['command'])
    with pytest.raises(app.AppError, match='nothing is waiting'):
        jobs.confirm_facing('front')
    jobs.start({'upload': answer['upload'], 'name': 'baggy'})
    status = wait_until_done(jobs)
    assert status['state'] == 'needs_facing' and status['facing']['renders'] == [
        '/files/runs/baggy/full/review-front.png', '/files/runs/baggy/full/review-side.png']
    with pytest.raises(app.AppError, match='answer'):
        jobs.confirm_facing('sideways')
    assert not list(facing_place['answers'].iterdir())
    jobs.confirm_facing('front')
    status = wait_until_done(jobs)
    assert status['state'] == 'complete'
    saved = json.loads((facing_place['answers']/f"{answer['upload']}.json").read_text())
    assert saved['facing'] == 'front' and saved['by'] == 'owner'


def test_a_back_answer_is_recorded_and_the_run_stops_saying_so(facing_place):
    answer = store(archive_bytes())
    jobs = app.Jobs(facing_place['command'])
    jobs.start({'upload': answer['upload'], 'name': 'baggy'})
    wait_until_done(jobs)
    jobs.confirm_facing('back')
    status = wait_until_done(jobs)
    assert status['state'] == 'stopped' and status['facing'] is None
    assert any('FacingRejected' in (s.get('reason') or '') for s in status['stages'])
