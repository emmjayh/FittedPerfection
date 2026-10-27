"""Local server for the 3D review viewer (spec §5.7, human review).

`python -m fittedperfection view` serves viewer/ plus two read-only folders, on this
computer only (127.0.0.1):

    /files/runs/...     the project's runs/
    /files/bodies/...   the four target bodies

/api/catalog lists outfits (runs grouped by garment hash), every fit version
in them (fit/ and each fit.stale-N/), and the four bodies.

The app layer (fittedperfection/app.py): POST /api/uploads (a dropped archive), /api/jobs (start a fit),
/api/jobs/approve (allow the spending it asked for), /api/jobs/cancel; GET /api/jobs/current and /api/outfits.
Writes are accepted only from this viewer's own page (its Origin plus a custom header).
"""
import json
import mimetypes
import re
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import app, config

JOBS = app.Jobs()
ALLOWED = {'.glb', '.png', '.jpg', '.jpeg', '.json', '.html'}
SOURCE_VIEWS = ('front', 'left', 'back', 'right')
VERSION_FOLDER = re.compile(r'^fit(\.stale-(\d+))?$')


def shared_folders():
    folders = {'runs': config.runs_folder()}
    try:
        folders['bodies'] = config.bodies_folder()
    except config.ConfigError:
        pass
    return {name: path for name, path in folders.items() if path.is_dir()}


def link(share, base, path):
    return f'/files/{share}/' + Path(path).relative_to(base).as_posix()


def fit_versions(run_folder, runs):
    """fit/ first, then fit.stale-N/ newest first; only folders holding at least one DRAFT.glb."""
    versions = []
    for folder in run_folder.iterdir():
        match = VERSION_FOLDER.match(folder.name)
        if not match or not folder.is_dir():
            continue
        drafts = {body: link('runs', runs, folder/body/'DRAFT.glb')
                  for body in config.BODIES if (folder/body/'DRAFT.glb').is_file()}
        if not drafts:
            continue
        reports = {body: link('runs', runs, folder/body/'fit.report.json')
                   for body in drafts if (folder/body/'fit.report.json').is_file()}
        # the deliverable when there is one: the full-resolution outfit on the shown body (raised, covered faces
        # hidden); older versions only have the cage draft and the plain body
        outfits = {body: link('runs', runs, folder/body/'OUTFIT.glb') for body in drafts
                   if (folder/body/'OUTFIT.glb').is_file()}
        shown = {body: link('runs', runs, folder/body/'BODY.glb') for body in drafts
                 if (folder/body/'BODY.glb').is_file()}
        newest = max((folder/body/'DRAFT.glb').stat().st_mtime for body in drafts)
        stale = int(match.group(2)) if match.group(2) else 0
        review = run_folder/'review'
        audits = {body: link('runs', runs, review/body/'audit.json') for body in drafts
                  if stale == 0 and (review/body/'audit.json').is_file()}
        wonder = link('runs', runs, review/'wonder.json') if stale == 0 and (review/'wonder.json').is_file() else None
        versions.append({'id': f'{run_folder.name}/{folder.name}', 'run': run_folder.name, 'folder': folder.name,
                         'drafts': drafts, 'outfits': outfits, 'shown_bodies': shown, 'reports': reports,
                         'audits': audits, 'wonder': wonder, 'time': newest, '_order': (stale != 0, -stale)})
    versions.sort(key=lambda version: version.pop('_order'))
    return versions


def catalog():
    shared = shared_folders()
    outfits = {}
    runs = shared.get('runs')
    if runs:
        for run_folder in sorted(path for path in runs.iterdir() if (path/'run.json').is_file()):
            state = json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
            key = state.get('garment_sha256') or run_folder.name
            outfit = outfits.setdefault(key, {'garment_sha256': key, 'runs': [], 'versions': [], 'source': {}})
            outfit['runs'].append(run_folder.name)
            outfit['versions'] += fit_versions(run_folder, runs)
            for view in SOURCE_VIEWS:
                image = run_folder/'analysis'/'views'/f'{view}.png'
                if image.is_file():
                    outfit['source'].setdefault(view, link('runs', runs, image))
    bodies = {}
    if 'bodies' in shared:
        base = shared['bodies']
        bodies = {body: {'sex': sex, 'url': link('bodies', base, base/body/'BODY.glb')}
                  for body, sex in config.BODIES.items() if (base/body/'BODY.glb').is_file()}
    return {'outfits': sorted(outfits.values(), key=lambda o: o['runs'][0]), 'bodies': bodies,
            'body_order': list(config.BODIES)}


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(config.root()/'viewer'), **kwargs)

    def log_message(self, format, *args):
        pass

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_shared(self, share, relative):
        base = shared_folders().get(share)
        if base is None:
            return self.send_error(404)
        target = (base/relative).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file() or target.suffix.lower() not in ALLOWED:
            return self.send_error(404)
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(target.stat().st_size))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        with open(target, 'rb') as stream:
            while block := stream.read(1 << 20):
                self.wfile.write(block)

    def _json(self, value, code=200):
        body = json.dumps(value).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _from_this_page(self):
        """Writes only from this viewer's own page: its Origin and a custom header other sites cannot send
        without a CORS preflight, which this server never grants."""
        port = self.server.server_address[1]
        origin = self.headers.get('Origin', '')
        return self.headers.get('X-FittedPerfection') == '1' and \
            origin in (f'http://127.0.0.1:{port}', f'http://localhost:{port}')

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        if not self._from_this_page():
            return self._json({'error': 'refused: not from this viewer page'}, 403)
        length = int(self.headers.get('Content-Length') or 0)
        try:
            if path == '/api/uploads':
                return self._json(app.store_upload(self.rfile.read, length, self.headers.get('X-Filename', '')))
            request = json.loads(self.rfile.read(min(length, 10_000)) or b'{}') if length else {}
            if path == '/api/jobs':
                return self._json(JOBS.start(request))
            if path == '/api/jobs/approve':
                return self._json(JOBS.approve(wonder=bool(request.get('wonder'))))
            if path == '/api/jobs/cancel':
                return self._json(JOBS.cancel())
        except app.AppError as refused:
            return self._json({'error': str(refused)}, 400)
        except (ValueError, OSError) as failed:
            return self._json({'error': f'{type(failed).__name__}: {failed}'}, 400)
        return self._json({'error': 'unknown request'}, 404)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == '/api/catalog':
            return self._send(json.dumps(catalog()).encode('utf-8'), 'application/json')
        if path == '/api/jobs/current':
            return self._json(JOBS.status())
        if path == '/api/outfits':
            return self._json(app.listed_outfits())
        if path.startswith('/files/'):
            share, _, relative = path[len('/files/'):].partition('/')
            return self._send_shared(share, relative)
        return super().do_GET()


def _exclusive(sock):
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)


def _port_taken_anywhere(port):
    """True if any program holds the port on all IPv4 or IPv6 addresses. Windows still lets
    127.0.0.1 bind beside such a program, and browsers asking for localhost then reach it."""
    for family, address in ((socket.AF_INET, '0.0.0.0'), (socket.AF_INET6, '::')):
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue                                  # no IPv6 on this machine
        try:
            _exclusive(probe)
            probe.bind((address, port))
        except OSError:
            return True
        finally:
            probe.close()
    return False


class ViewerServer(ThreadingHTTPServer):
    """Serves only when it really owns its port, so the browser can't reach another program."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        port = self.server_address[1]
        if port and _port_taken_anywhere(port):
            raise OSError(f'another program is using port {port}')
        _exclusive(self.socket)
        super().server_bind()


def serve(port):
    mimetypes.add_type('text/javascript', '.js')
    mimetypes.add_type('model/gltf-binary', '.glb')
    try:
        server = ViewerServer(('127.0.0.1', port), ViewerHandler)
    except OSError as error:
        raise RuntimeError(f'port {port} is already in use ({error}); try --port with another number')
    print(f'[fp] viewer: http://127.0.0.1:{port}/  (Ctrl+C stops it)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
