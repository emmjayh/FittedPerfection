"""Frozen copies of the code for long batches (spec §3).

`python -m fittedperfection freeze` exports the `fittedperfection` package as committed at HEAD (never
uncommitted edits) into frozen/<commit12>/, with a record of every file's
hash. The copy runs through its own launcher, checks its files each time it
starts and refuses to run if any changed, so editing the working code cannot
disturb a batch halfway. Runs still go to the project's runs/ folder.
"""
import datetime
import io
import json
import subprocess
import tarfile
from pathlib import Path

from . import config
from .hashing import folder_manifest

LAUNCHER = 'run_frozen.py'
LAUNCHER_SOURCE = '''"""Runs this frozen FittedPerfection copy: python run_frozen.py run OUTFIT ..."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fittedperfection  # noqa: E402

if Path(fittedperfection.__file__).resolve().parent != HERE/'fittedperfection':
    sys.exit('frozen copy is shadowed by ' + fittedperfection.__file__)

from fittedperfection.cli import main  # noqa: E402

sys.exit(main())
'''


class FreezeError(RuntimeError):
    pass


def _git(repo, *args):
    return subprocess.run(['git', *args], cwd=repo, capture_output=True, check=True).stdout


def code_files(folder):
    return {path: sha for path, sha in folder_manifest(folder).items() if '__pycache__' not in path}


def freeze(repo=None):
    repo = Path(repo or config.PACKAGE.parent).resolve()
    if config.frozen_record():
        raise FreezeError('this is already a frozen copy; freeze from the project folder')
    if _git(repo, 'status', '--porcelain', '--', 'fittedperfection').strip():
        raise FreezeError('fittedperfection/ has uncommitted changes; commit first (frozen copies come from HEAD)')
    commit = _git(repo, 'rev-parse', 'HEAD').decode().strip()
    target = repo/'frozen'/commit[:12]
    if target.exists():
        verify(target)
        return target
    building = target.with_name(target.name+'.partial')
    building.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(_git(repo, 'archive', '--format=tar', 'HEAD', 'fittedperfection'))) as archive:
        archive.extractall(building, filter='data')
    (building/LAUNCHER).write_text(LAUNCHER_SOURCE, encoding='utf-8')
    record = {
        'id': commit[:12],
        'commit': commit,
        'root': str(repo),
        'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'files': code_files(building/'fittedperfection'),
    }
    (building/'FROZEN.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    building.rename(target)
    return target


def verify(folder=None):
    """Raise FreezeError if a frozen copy's files differ from its record."""
    folder = Path(folder) if folder else config.PACKAGE.parent
    record = json.loads((folder/'FROZEN.json').read_text(encoding='utf-8'))
    actual = code_files(folder/'fittedperfection')
    if actual != record['files']:
        changed = sorted(set(actual) ^ set(record['files']) | {p for p in actual if actual[p] != record['files'].get(p)})
        raise FreezeError(f'frozen copy {record["id"]} was changed ({", ".join(changed)}); refusing to run')
    return record
