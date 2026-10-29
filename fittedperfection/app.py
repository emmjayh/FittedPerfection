"""The app layer: fit a new outfit from the viewer (drop an archive, see progress, confirm spending, view results).

Nothing here fits anything. An upload is checked (its listing only, never unpacked) and kept by hash; a job is
the ordinary command, `python -m fittedperfection run ARCHIVE --name NAME --allow-spend X`, in its own process,
so the fit keeps its seal and stage records exactly as from a terminal. One job at a time.

Facing: when bind cannot tell which way an outfit faces (FacingUnconfirmed), the page shows intake's renders and
asks the owner; the answer is recorded (confirm.py) and the same run restarts from bind.

Spending: a job always starts with nothing allowed. Vision then either finds every SAM reply cached (free) or
stops before sending anything, recording what it would need (`needs_usd` in run.json). The page shows that,
and only an explicit approval restarts the same run with that allowance (finished stages are reused).
"""
import hashlib
import json
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

from . import config, confirm, intake
from .stages import NAMES

MAX_UPLOAD_BYTES = 500_000_000
WONDER_ALLOWANCE_USD = 0.05          # one Muse Wonder fit review, rounded up from the client's estimate
PICTURES = {'views': 'analysis/views/front.png', 'label': 'labels/review.png', 'review': 'review/sheet-front.png'}
LOG_LINES = 40


class AppError(ValueError):
    """A request the app refuses, with a message for the page."""


def uploads_folder():
    return config.root()/'uploads'


def clean_name(text):
    name = re.sub(r'[^a-z0-9_-]+', '-', str(text).lower()).strip('-_')[:60]
    if not name:
        raise AppError('choose a run name (letters, digits, - or _)')
    return name


def held_out(sha):
    """The manifest name of a held-out test-fit outfit with these bytes, else None."""
    for name, entry in config.outfits().items():
        if entry['split'] == 'test-fits' and entry['sha256'] == sha:
            return name
    return None


def store_upload(read, length, filename):
    """Keep a dropped archive under its hash after checking it is a garment archive intake would accept."""
    if length <= 0 or length > MAX_UPLOAD_BYTES:
        raise AppError(f'the archive must be between 1 byte and {MAX_UPLOAD_BYTES // 1_000_000} MB')
    folder = uploads_folder()
    folder.mkdir(exist_ok=True)
    temporary = folder/f'incoming-{threading.get_ident()}.part'
    digest, remaining = hashlib.sha256(), length
    with open(temporary, 'wb') as stream:
        while remaining:
            block = read(min(remaining, 1 << 20))
            if not block:
                break
            digest.update(block)
            stream.write(block)
            remaining -= len(block)
    if remaining:
        temporary.unlink()
        raise AppError('the upload was cut short')
    try:
        with zipfile.ZipFile(temporary) as bundle:
            intake.check_members(bundle)
    except (zipfile.BadZipFile, intake.IntakeError) as refused:
        temporary.unlink()
        raise AppError(f'not a garment archive: {refused}') from None
    sha = digest.hexdigest()
    target = folder/f'{sha}.zip'
    if target.exists():
        temporary.unlink()
    else:
        temporary.replace(target)
    stem = Path(filename or 'outfit').stem
    return {'upload': sha, 'bytes': length, 'name': clean_name(stem) if re.search(r'[a-z0-9]', stem.lower())
            else 'outfit', 'held_out': held_out(sha)}


def archive_for(request):
    """(archive path, garment sha) for a job request naming an upload or a listed outfit."""
    if request.get('upload'):
        sha = str(request['upload'])
        if not re.fullmatch(r'[0-9a-f]{64}', sha) or not (uploads_folder()/f'{sha}.zip').is_file():
            raise AppError('unknown upload; drop the archive again')
        return uploads_folder()/f'{sha}.zip', sha
    listed = config.outfits()
    entry = listed.get(str(request.get('outfit', '')))
    if entry is None:
        raise AppError('choose an archive or a listed outfit')
    return entry['path'], entry['sha256']


class Jobs:
    """The one job the app runs at a time, and what the page needs to follow it."""

    def __init__(self, command=None):
        self._command = command or [sys.executable, '-m', 'fittedperfection']
        self._lock = threading.Lock()
        self.job = None                                            # {'name', 'archive', 'process', 'allowance', ...}

    def start(self, request):
        archive, sha = archive_for(request)
        name = clean_name(request.get('name') or Path(archive).stem)
        if held_out(sha) and not request.get('test_fit'):
            raise AppError(f'this is the held-out test outfit {held_out(sha)!r}; tick "test fit" to run it on purpose')
        allowance = float(request.get('allow_spend_usd') or 0.0)
        if allowance < 0 or allowance > 5:
            raise AppError('the allowance must be between $0 and $5')
        with self._lock:
            if self.job and self.job['process'].poll() is None:
                raise AppError(f'a fit is already running ({self.job["name"]}); wait for it or cancel it')
            if self.job:
                self.job['stream'].close()
            log = config.runs_folder()/'_app'/f'{name}.log'          # not in the run folder: the runner owns it
            command = list(self._command) + ['run', str(archive), '--name', name, '--allow-spend', f'{allowance:.4f}']
            if held_out(sha):
                command.append('--test-fit')
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = open(log, 'a', encoding='utf-8')
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(config.root()))
            self.job = {'name': name, 'archive': str(archive), 'sha': sha, 'process': process, 'stream': stream,
                        'log': log, 'allowance': allowance, 'test_fit': bool(held_out(sha)),
                        'wonder': bool(request.get('wonder'))}
        return self.status()

    def approve(self, wonder=False):
        """Restart the stopped job with what vision said it needs (plus a Wonder review if asked)."""
        status = self.status()
        if status['state'] != 'needs_spend':
            raise AppError('nothing is waiting for approval')
        allowance = round(status['needs_usd'] + (WONDER_ALLOWANCE_USD if wonder else 0.0) + 0.005, 2)
        job = self.job
        return self.start({'upload': job['sha'] if Path(job['archive']).parent == uploads_folder() else None,
                           'outfit': self._listed_name(job['sha']), 'name': job['name'], 'test_fit': job['test_fit'],
                           'allow_spend_usd': allowance, 'wonder': wonder})

    def confirm_facing(self, answer):
        """Record the owner's answer about the facing, then restart the stopped job (bind onward reruns)."""
        status = self.status()
        if status['state'] != 'needs_facing':
            raise AppError('nothing is waiting for a facing answer')
        job = self.job
        try:
            confirm.write_facing(job['sha'], answer, config.runs_folder()/job['name']/'full')
        except confirm.ConfirmError as refused:
            raise AppError(str(refused)) from None
        return self.start({'upload': job['sha'] if Path(job['archive']).parent == uploads_folder() else None,
                           'outfit': self._listed_name(job['sha']), 'name': job['name'], 'test_fit': job['test_fit'],
                           'allow_spend_usd': job['allowance'], 'wonder': job['wonder']})

    @staticmethod
    def _listed_name(sha):
        return next((n for n, e in config.outfits().items() if e['sha256'] == sha), None)

    def cancel(self):
        with self._lock:
            if not self.job or self.job['process'].poll() is not None:
                raise AppError('no fit is running')
            self.job['process'].terminate()
            self.job['process'].wait(timeout=30)
            self.job['cancelled'] = True
        return self.status()

    def status(self):
        """State, per-stage progress, the last log lines and stage pictures, for the page."""
        job = self.job
        if job is None:
            return {'state': 'idle'}
        folder = config.runs_folder()/job['name']
        state_file = folder/'run.json'
        run = json.loads(state_file.read_text(encoding='utf-8')) if state_file.is_file() else {'stages': {}}
        running = job['process'].poll() is None
        if running:
            job['stream'].flush()
        elif not job['stream'].closed:
            job['stream'].close()                                  # the job ended: let go of its log
        stages = [{'name': n, **{k: v for k, v in run['stages'].get(n, {}).items()
                                 if k in ('status', 'seconds', 'reason', 'why', 'started', 'needs_usd')}}
                  for n in NAMES]
        stopped = next((s for s in stages if s.get('status') == 'stopped'), None)
        if running:
            state = 'running'
        elif job.get('cancelled'):
            state = 'cancelled'
        elif run.get('status') == 'complete':
            state = 'complete'
        elif stopped and stopped.get('needs_usd') is not None:
            state = 'needs_spend'
        elif stopped and stopped.get('reason', '').startswith('FacingUnconfirmed'):
            state = 'needs_facing'
        else:
            state = 'stopped'
        requests = re.search(r'needs (\d+) new SAM requests', (stopped or {}).get('reason', ''))
        log = job['log'].read_text(encoding='utf-8', errors='replace').splitlines()[-LOG_LINES:] \
            if job['log'].is_file() else []
        pictures = {stage: f'/files/runs/{job["name"]}/{path}' for stage, path in PICTURES.items()
                    if (folder/path).is_file()}
        facing = None
        if state == 'needs_facing':
            facing = {'renders': [f'/files/runs/{job["name"]}/full/{n}' for n in confirm.RENDERS
                                  if (folder/'full'/n).is_file()],
                      'toe_over_heel_reach': run['stages'].get('intake', {}).get('details', {}).get('toe_over_heel_reach')}
        return {'state': state, 'name': job['name'], 'allowance_usd': job['allowance'], 'stages': stages,
                'facing': facing,
                'needs_usd': (stopped or {}).get('needs_usd'),
                'needs_requests': int(requests.group(1)) if requests else None,
                'wonder_usd': WONDER_ALLOWANCE_USD, 'held_out': job['test_fit'], 'log': log, 'pictures': pictures}


def listed_outfits():
    """Outfits from the manifest the page can start without a drop (held-out ones marked)."""
    return [{'name': n, 'split': e['split'], 'present': e['path'].is_file()} for n, e in config.outfits().items()]
