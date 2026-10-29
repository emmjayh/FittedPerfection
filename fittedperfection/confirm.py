"""Owner confirmations: answers only a person gives, kept by garment (spec §5.1, changes for this build).

When an outfit's facing is provisional at intake and its footwear cannot confirm it, bind stops and asks the
owner. The answer is kept as confirmations/<garment sha256>.json in the project root, so it belongs to the
garment's bytes, not to a run name:

    {"facing": "front" | "back", "by": "owner", "time": ..., "shown": {intake's facing record, render hashes}}

"front" means the front view shows the outfit's front. Only the app's buttons and the `confirm-facing`
command write it; no stage ever does. A new answer sets the old file aside (never overwritten).
"""
import hashlib
import json
from datetime import datetime

from . import config

ANSWERS = ('front', 'back')
RENDERS = ('review-front.png', 'review-side.png')       # intake's views of the outfit as it was found


class ConfirmError(ValueError):
    pass


def folder():
    return config.root()/'confirmations'


def path_for(garment_sha):
    return folder()/f'{garment_sha}.json'


def read(garment_sha):
    """The owner's confirmation for this garment, or None."""
    path = path_for(garment_sha)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def file_sha256(garment_sha):
    """Hash of the confirmation file (for a stage's input record), or None when there is none."""
    path = path_for(garment_sha)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def shown(intake_folder):
    """What the owner is shown and answers about: intake's facing record and its two renders' hashes."""
    record = json.loads((intake_folder/'intake.json').read_text(encoding='utf-8'))['facing']
    return {'toe_direction': record['toe_direction'], 'turn_about_z_degrees': record['turn_about_z_degrees'],
            'status': record['status'], 'toe_over_heel_reach': record['evidence']['toe_over_heel_reach'],
            'renders': {name: hashlib.sha256((intake_folder/name).read_bytes()).hexdigest()
                        for name in RENDERS if (intake_folder/name).is_file()}}


def matches(confirmation, intake_facing):
    """Whether a confirmation answers this intake's facing (same toe direction and turn)."""
    seen = (confirmation or {}).get('shown') or {}
    return seen.get('toe_direction') == intake_facing['toe_direction'] and \
        seen.get('turn_about_z_degrees') == intake_facing['turn_about_z_degrees']


def write_facing(garment_sha, answer, intake_folder):
    """Record the owner's answer for this garment; an earlier answer is set aside, not overwritten."""
    if answer not in ANSWERS:
        raise ConfirmError(f'the answer must be one of {ANSWERS}')
    record = json.loads((intake_folder/'intake.json').read_text(encoding='utf-8'))['facing']
    if record['status'] != 'provisional':
        raise ConfirmError(f'the facing was {record["status"]} at intake; there is nothing to confirm')
    path = path_for(garment_sha)
    path.parent.mkdir(exist_ok=True)
    if path.is_file():
        n = 1
        while path.with_name(f'{path.name}.replaced-{n}').exists():
            n += 1
        path.rename(path.with_name(f'{path.name}.replaced-{n}'))
    confirmation = {'facing': answer, 'by': 'owner', 'time': datetime.now().astimezone().isoformat(timespec='seconds'),
                    'shown': shown(intake_folder)}
    path.write_text(json.dumps(confirmation, indent=1), encoding='utf-8', newline='\n')
    return confirmation
