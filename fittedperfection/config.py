"""Where things are: the project root, machine settings, target bodies and test outfits.

The root is the folder holding the `fittedperfection` package, or the root recorded by a
frozen copy (see freeze.py), so frozen runs read and write the same runs/.
FP_HOME overrides it (tests use this).
"""
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
FROZEN_RECORD = PACKAGE.parent/'FROZEN.json'

# The four fixed target bodies (spec §1, changes section): id -> sex.
BODIES = {
    'body_01_medium': 'female',
    'body_02_tall_slender': 'female',
    'body_03_short_heavy': 'male',
    'body_04_broad_muscular': 'male',
}
DONOR_BODY = 'body_01_medium'


class ConfigError(ValueError):
    pass


def frozen_record():
    """The frozen copy's record when running from one, else None."""
    if FROZEN_RECORD.is_file():
        return json.loads(FROZEN_RECORD.read_text(encoding='utf-8'))
    return None


def root():
    if os.environ.get('FP_HOME'):
        return Path(os.environ['FP_HOME']).resolve()
    record = frozen_record()
    return Path(record['root']) if record else PACKAGE.parent


def machine():
    """config.json (git-ignored) with FP_BLENDER / FP_BODIES overriding it."""
    path = root()/'config.json'
    values = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    for key in ('blender', 'bodies'):
        if os.environ.get(f'FP_{key.upper()}'):
            values[key] = os.environ[f'FP_{key.upper()}']
    return values


def _path(key):
    value = machine().get(key)
    if not value:
        raise ConfigError(f'"{key}" is missing: copy config.example.json to config.json')
    path = Path(value)
    return path if path.is_absolute() else root()/path


def blender():
    return _path('blender')


def bodies_folder():
    return _path('bodies')


def body_file(body_id):
    if body_id not in BODIES:
        raise ConfigError(f'unknown body {body_id!r}')
    return bodies_folder()/body_id/'BODY.glb'


def runs_folder():
    return root()/'runs'


def outfits():
    """Test outfits from outfits/manifest.json: name -> {split, path, sha256, bytes}."""
    path = root()/'outfits'/'manifest.json'
    if not path.is_file():
        return {}
    listed = json.loads(path.read_text(encoding='utf-8'))['outfits']
    return {name: {**entry, 'path': root()/entry['file']} for name, entry in listed.items()}


def spend_allowance_usd():
    """How much this command may spend on paid API calls (FP_ALLOW_SPEND_USD, set by --allow-spend). It is 0 by
    default, so only cached answers are used unless the owner explicitly allows spending."""
    try:
        return max(0.0, float(os.environ.get('FP_ALLOW_SPEND_USD', '0') or 0))
    except ValueError:
        return 0.0
