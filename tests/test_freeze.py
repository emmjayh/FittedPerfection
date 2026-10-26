"""Frozen copies come from HEAD, refuse uncommitted code, check themselves and refuse edits."""
import subprocess
import sys

import pytest

from fittedperfection.config import PACKAGE
from fittedperfection.freeze import LAUNCHER, FreezeError, freeze, verify


def git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path/'project'
    for source in PACKAGE.rglob('*.py'):
        target = repo/'fittedperfection'/source.relative_to(PACKAGE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    git(repo, 'init', '-q', '-b', 'main')
    git(repo, 'add', '.')
    git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-q', '-m', 'code')
    return repo


def launch(folder, *args, cwd):
    return subprocess.run([sys.executable, str(folder/LAUNCHER), *args], capture_output=True, text=True, cwd=cwd)


def test_freeze_exports_head_and_runs(repo):
    folder = freeze(repo)
    record = verify(folder)
    assert record['root'] == str(repo.resolve()) and 'run.py' in record['files']
    assert freeze(repo) == folder                      # one frozen copy per commit
    result = launch(folder, '--help', cwd=repo)
    assert result.returncode == 0 and 'FittedPerfection' in result.stdout


def test_uncommitted_code_cannot_be_frozen(repo):
    (repo/'fittedperfection/settings.py').write_text('# edited\n', encoding='utf-8')
    with pytest.raises(FreezeError, match='uncommitted'):
        freeze(repo)


def test_an_edited_frozen_copy_refuses_to_run(repo):
    folder = freeze(repo)
    (folder/'fittedperfection/settings.py').write_text('# edited\n', encoding='utf-8')
    with pytest.raises(FreezeError, match='settings.py'):
        verify(folder)
    result = launch(folder, 'outfits', cwd=repo)
    assert result.returncode == 2 and 'refusing to run' in result.stderr
