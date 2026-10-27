"""Run folders and the stage runner (spec §4, §6).

runs/<name>/ holds one garment's pipeline. Every stage writes one folder with
a stage.json listing the hashes of everything it read:

    code      the stage's own module files
    settings  the settings sections it uses
    models    the model pins it uses
    tools     external programs (path, size, modification time)
    garment   the garment archive, for stages that read it
    bodies    the four target bodies, for stages that read them
    upstream  the output hashes of the stages it reads from

Rules:
- A stage is reused when that record is unchanged and its folder still holds
  exactly what it wrote.
- Otherwise its folder is renamed to <folder>.stale-N (nothing is deleted or
  overwritten) and the stage runs into <folder>.partial, which becomes
  <folder> only when the stage finishes.
- Later stages compare upstream *output* hashes, so an earlier stage that
  reruns and writes the same bytes leaves them cached.
- rerun_from sets that stage and all later ones aside first.
- fresh refuses garment bytes any earlier run has seen, and writes a
  provenance card.
"""
import datetime
import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import VERSION, config, live, models, seal, settings
from .hashing import file_sha256, folder_sha256, json_sha256

RECORD = 'stage.json'
LEDGER = 'ledger.jsonl'


@dataclass(frozen=True)
class Stage:
    name: str
    folder: str                 # relative to the run folder
    action: Callable            # action(ctx, out_folder) -> dict of details for the record
    needs: tuple = ()           # stages whose outputs it reads
    code: tuple = ()            # its module files, relative to the fittedperfection package
    settings: tuple = ()        # settings sections it reads
    models: tuple = ()          # model pins it uses
    tools: tuple = ()           # external programs: 'blender'
    garment: bool = False
    bodies: bool = False
    sealed: bool = False        # run with the network and other programs shut off


class RunError(RuntimeError):
    pass


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def set_aside(path, kind='stale'):
    """Rename path to path.<kind>-N using the first free N; return the new path."""
    n = 1
    while path.with_name(f'{path.name}.{kind}-{n}').exists():
        n += 1
    target = path.with_name(f'{path.name}.{kind}-{n}')
    path.rename(target)
    return target


def outputs_sha256(folder):
    return folder_sha256(folder, skip=(RECORD,))


def tool_identity(name):
    path = {'blender': config.blender}[name]()
    info = path.stat()
    return {'path': str(path), 'bytes': info.st_size, 'modified_ns': info.st_mtime_ns}


def code_identity():
    """Which code is doing this run. Called outside the seal (it may ask git)."""
    identity = {'fittedperfection': VERSION, 'python': platform.python_version()}
    record = config.frozen_record()
    if record:
        identity.update(frozen=record['id'], commit=record['commit'], uncommitted_changes=False)
    else:
        identity['frozen'] = None
        try:
            def git(*args):
                return subprocess.run(['git', *args], cwd=config.PACKAGE.parent, capture_output=True,
                                      text=True, check=True).stdout.strip()
            identity['commit'] = git('rev-parse', 'HEAD')
            identity['uncommitted_changes'] = bool(git('status', '--porcelain', '--', 'fittedperfection'))
        except (OSError, subprocess.CalledProcessError):
            identity.update(commit=None, uncommitted_changes=None)
    for module in ('numpy', 'scipy', 'trimesh', 'PIL'):
        try:
            identity[module] = __import__(module).__version__
        except ImportError:
            identity[module] = None
    return identity


def seen_before(runs_folder):
    """{garment sha256: [where]} from every run.json and the append-only ledger."""
    seen = {}
    runs_folder = Path(runs_folder)
    for path in sorted(runs_folder.glob('*/run.json')):
        sha = json.loads(path.read_text(encoding='utf-8')).get('garment_sha256')
        if sha:
            seen.setdefault(sha, []).append(str(path.parent))
    ledger = runs_folder/LEDGER
    if ledger.is_file():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            if line.strip():
                entry = json.loads(line)
                seen.setdefault(entry['garment_sha256'], []).append(entry['run'])
    return seen


class Context:
    """What a stage action is given."""

    def __init__(self, run, garment):
        self.run = run
        self.garment = garment

    @property
    def run_folder(self):
        return self.run.folder

    def output_of(self, stage_name):
        return self.run.folder/self.run.stages[stage_name].folder

    def call(self, command, log_name):
        """Run an external program, logging to logs/<log_name>. Refused while sealed."""
        log = self.run.folder/'logs'/log_name
        log.parent.mkdir(exist_ok=True)
        with open(log, 'w', encoding='utf-8') as stream:
            result = subprocess.run([str(part) for part in command], stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RunError(f'{Path(str(command[0])).name} exited with {result.returncode}; see {log}')


class Run:
    def __init__(self, folder, pipeline):
        self.folder = Path(folder)
        self.stages = {stage.name: stage for stage in pipeline}
        self.order = [stage.name for stage in pipeline]
        self.state_file = self.folder/'run.json'
        self.state = json.loads(self.state_file.read_text(encoding='utf-8')) if self.state_file.is_file() else None

    def save(self):
        temporary = self.folder/'run.json.tmp'
        temporary.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
        temporary.replace(self.state_file)
        live.write(self.folder, self.state, self.order)

    def _note_aside(self, stage_name, path, why):
        self.state.setdefault('set_aside', []).append(
            {'stage': stage_name, 'folder': path.relative_to(self.folder).as_posix(), 'why': why, 'time': now()})

    def input_record(self, stage):
        record = {
            'code': {name: file_sha256(config.PACKAGE/name) for name in stage.code},
            'settings': {name: settings.section_sha256(name) for name in stage.settings},
            'models': {name: models.pin(name) for name in stage.models},
            'tools': {name: tool_identity(name) for name in stage.tools},
            'upstream': {name: self.state['stages'][name]['outputs_sha256'] for name in stage.needs},
        }
        if stage.garment:
            record['garment_sha256'] = self.state['garment_sha256']
        if stage.bodies:
            record['bodies'] = {body: file_sha256(config.body_file(body)) for body in config.BODIES}
        return record

    def _why_rerun(self, stage, inputs_sha):
        """None when the stage can be reused, else the reason it must run."""
        previous = self.state['stages'].get(stage.name)
        if not previous:
            return 'not run yet'
        if previous.get('status') != 'complete':
            return f'last attempt {previous.get("status")}'
        if previous.get('inputs_sha256') != inputs_sha:
            return 'inputs changed'
        folder = self.folder/stage.folder
        if not folder.is_dir():
            return 'output folder missing'
        if outputs_sha256(folder) != previous.get('outputs_sha256'):
            return 'outputs changed on disk'
        return None

    def run_stage(self, stage, ctx):
        for name in stage.needs:
            if self.state['stages'].get(name, {}).get('status') != 'complete':
                raise RunError(f'{stage.name} needs {name}, which is not complete')
        inputs = self.input_record(stage)
        inputs_sha = json_sha256(inputs)
        why = self._why_rerun(stage, inputs_sha)
        if why is None:
            print(f'[fp] {stage.name}: reused', flush=True)
            return self.state['stages'][stage.name]

        folder = self.folder/stage.folder
        if folder.exists():
            self._note_aside(stage.name, set_aside(folder), why)
        partial = folder.with_name(folder.name+'.partial')
        if partial.exists():
            self._note_aside(stage.name, set_aside(partial, 'failed'), 'left by an interrupted attempt')
        partial.mkdir(parents=True)

        print(f'[fp] {stage.name}: running ({why})', flush=True)
        started, clock = now(), time.perf_counter()
        self.state['stages'][stage.name] = {'status': 'running', 'started': started, 'why': why}
        self.save()
        try:
            if stage.sealed:
                with seal.sealed():
                    details = stage.action(ctx, partial) or {}
            else:
                details = stage.action(ctx, partial) or {}
        except Exception as error:
            self.state['stages'][stage.name] = {
                'status': 'stopped', 'started': started, 'why': why,
                'seconds': round(time.perf_counter()-clock, 2), 'reason': f'{type(error).__name__}: {error}',
                **({'needs_usd': round(float(error.estimate_usd), 4)}
                   if getattr(error, 'estimate_usd', None) is not None else {}),
                'partial': partial.relative_to(self.folder).as_posix()}
            self.state['status'] = 'stopped_at_'+stage.name
            self.save()
            raise
        seconds = round(time.perf_counter()-clock, 2)
        record = {'stage': stage.name, 'inputs': inputs, 'inputs_sha256': inputs_sha, 'sealed': stage.sealed,
                  'started': started, 'seconds': seconds, 'details': details}
        (partial/RECORD).write_text(json.dumps(record, indent=2), encoding='utf-8')
        partial.rename(folder)
        self.state['stages'][stage.name] = {
            'status': 'complete', 'started': started, 'seconds': seconds, 'why': why,
            'inputs_sha256': inputs_sha, 'outputs_sha256': outputs_sha256(folder), 'details': details}
        self.save()
        print(f'[fp] {stage.name}: complete in {seconds}s', flush=True)
        return self.state['stages'][stage.name]

    def execute(self, garment, name, rerun_from=None, stop_after=None, fresh=False):
        for stage_name in (rerun_from, stop_after):
            if stage_name and stage_name not in self.stages:
                raise RunError(f'unknown stage {stage_name!r}; stages: {", ".join(self.order)}')
        garment = Path(garment).resolve()
        if not garment.is_file():
            raise RunError(f'garment archive not found: {garment}')
        sha = file_sha256(garment)
        runs_folder = self.folder.parent

        if fresh:
            earlier = seen_before(runs_folder).get(sha)
            if earlier or self.folder.exists():
                where = sorted(set(earlier or [str(self.folder)]))
                raise RunError('not a first fit: these garment bytes were already run in ' + ', '.join(where))
        if self.state is None:
            if self.folder.exists():
                raise RunError(f'{self.folder} exists but has no run.json; refusing to write into it')
            self.folder.mkdir(parents=True)
            self.state = {'name': name, 'garment': str(garment), 'garment_sha256': sha, 'created': now(),
                          'status': 'new', 'stages': {}, 'fit_reviewed': False, 'game_ready': False}
            with open(runs_folder/LEDGER, 'a', encoding='utf-8') as ledger:
                ledger.write(json.dumps({'garment_sha256': sha, 'run': str(self.folder), 'time': now()}) + '\n')
        elif self.state['garment_sha256'] != sha:
            raise RunError(f'run {self.folder.name} belongs to other garment bytes; choose a new run name')

        self.state['code'] = code_identity()
        if fresh:
            self.state['provenance'] = {
                'garment file': garment.name,
                'garment sha256': sha,
                'earlier runs of these bytes': f'none (checked {len(list(runs_folder.glob("*/run.json")))} runs '
                                               'and the ledger)',
                'started': now(),
                'machine': socket.gethostname(),
                'fittedperfection': VERSION,
                'commit': self.state['code'].get('commit'),
                'frozen copy': self.state['code'].get('frozen') or 'no (development code)',
                'part labels': models.PINS['sam']['model'],
                'reviewer': models.PINS['wonder']['model'],
                'fit': 'local maths only; network and other programs shut off while fitting',
            }
            self._write_provenance()
        if rerun_from:
            for stage_name in self.order[self.order.index(rerun_from):]:
                folder = self.folder/self.stages[stage_name].folder
                if folder.exists():
                    self._note_aside(stage_name, set_aside(folder), f'--rerun-from {rerun_from}')
                self.state['stages'].pop(stage_name, None)
        self.state['status'] = 'running'
        self.save()

        ctx = Context(self, garment)
        for stage_name in self.order:
            self.run_stage(self.stages[stage_name], ctx)
            if stage_name == stop_after:
                self.state['status'] = 'prepared_through_'+stage_name
                break
        else:
            self.state['status'] = 'complete'
        if fresh:
            self.state['provenance']['finished'] = now()
            self._write_provenance()
        self.save()
        return self.state

    def _write_provenance(self):
        (self.folder/'provenance.json').write_text(json.dumps(self.state['provenance'], indent=2), encoding='utf-8')
