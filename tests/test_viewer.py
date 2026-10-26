"""Viewer catalog and file serving limits."""
import json
import threading
import urllib.error
import urllib.request
import pytest

from fittedperfection import viewer


def put(path, data=b'x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def project(home):
    runs = home/'runs'
    for folder in ('fit', 'fit.stale-1', 'fit.stale-2', 'fit.partial'):
        put(runs/'moon'/folder/'body_01_medium'/'DRAFT.glb')
    put(runs/'moon/fit/body_01_medium/fit.report.json', b'{}')
    put(runs/'moon/run.json', json.dumps({'garment_sha256': 'abc'}).encode())
    put(runs/'moon/analysis/views/front.png')
    put(runs/'moon-again/fit/body_03_short_heavy/DRAFT.glb')
    put(runs/'moon-again/run.json', json.dumps({'garment_sha256': 'abc'}).encode())
    put(runs/'other/run.json', json.dumps({'garment_sha256': 'def'}).encode())
    put(runs/'other/fit/not_a_body/DRAFT.glb')
    put(home/'assets/bodies/body_01_medium/BODY.glb', b'body')
    put(home/'secret.json', b'{}')
    (home/'viewer').mkdir()
    (home/'config.json').write_text(json.dumps({'bodies': 'assets/bodies'}))
    return home


def test_catalog_groups_runs_by_garment_and_orders_versions(project):
    listing = viewer.catalog()
    moon = next(o for o in listing['outfits'] if o['garment_sha256'] == 'abc')
    assert moon['runs'] == ['moon', 'moon-again']
    assert [v['id'] for v in moon['versions']] == ['moon/fit', 'moon/fit.stale-2', 'moon/fit.stale-1', 'moon-again/fit']
    assert moon['versions'][0]['reports'] == {'body_01_medium': '/files/runs/moon/fit/body_01_medium/fit.report.json'}
    assert moon['source'] == {'front': '/files/runs/moon/analysis/views/front.png'}
    other = next(o for o in listing['outfits'] if o['garment_sha256'] == 'def')
    assert other['versions'] == []
    assert listing['bodies'] == {'body_01_medium': {'sex': 'female', 'url': '/files/bodies/body_01_medium/BODY.glb'}}


def test_server_stays_inside_its_folders(project):
    server = viewer.ViewerServer(('127.0.0.1', 0), viewer.ViewerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        assert urllib.request.urlopen(base+'/files/bodies/body_01_medium/BODY.glb').read() == b'body'
        assert json.loads(urllib.request.urlopen(base+'/api/catalog').read())['body_order'][0] == 'body_01_medium'
        for bad in ('/files/runs/../secret.json', '/files/runs/%2e%2e/secret.json', '/files/elsewhere/x.glb',
                    '/files/runs/moon/run.json.tmp'):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base+bad)
    finally:
        server.shutdown()
        server.server_close()



@pytest.mark.parametrize('holder_address', ['127.0.0.1', '0.0.0.0'])
def test_a_port_held_by_another_program_is_refused(project, holder_address):
    import socket
    holder = socket.socket()
    holder.bind((holder_address, 0))
    holder.listen()
    try:
        with pytest.raises(OSError):
            viewer.ViewerServer(('127.0.0.1', holder.getsockname()[1]), viewer.ViewerHandler).server_close()
    finally:
        holder.close()


def test_serve_explains_a_busy_port(project, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError('another program is using port 1')
    monkeypatch.setattr(viewer, 'ViewerServer', refuse)
    with pytest.raises(RuntimeError, match='already in use'):
        viewer.serve(1)
