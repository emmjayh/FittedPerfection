"""The Meta Model API key, from secrets.env (git-ignored) or the environment.

Only the vision and review stages ask for it. It is refused while the fit is
sealed and never written into run records, reports or logs.
"""
import os

from . import config, seal

FILE = 'secrets.env'
KEY = 'MODEL_API_KEY'


def _from_file():
    path = config.root()/FILE
    found = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                name, value = line.split('=', 1)
                found[name.strip()] = value.strip().strip('"\'')
    return found


def api_key():
    if seal.is_sealed():
        raise seal.SealBroken(f'sealed: {KEY} refused')
    value = os.environ.get(KEY) or _from_file().get(KEY)
    if not value:
        raise ValueError(f'{KEY} is not set: put it in {FILE} (template: secrets.example.env)')
    return value
