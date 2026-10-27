"""The four fixed target bodies (spec §1): made once with MPFB, checked, reviewed, then never overwritten.

`python -m fittedperfection bodies make` builds every body in its own Blender
process (tools/make_bodies.py), renders front and side views for review
(tools/render_bodies.py), checks the results and writes assets/bodies/ with
bodies.json recording every input. It refuses if assets/bodies/ already exists.
"""
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import config
from .gltf import read_skinned_mesh
from .hashing import file_sha256

MPFB = {'version': '2.0.18', 'zip': 'add-on-mpfb-v2.0.18.zip',
        'zip_sha256': 'c075b3e9b94b8e5556a5f03f4653602d02ea306de08126e8f55ed6dee28a2dad',
        'source': 'https://extensions.blender.org/add-ons/mpfb/', 'licence': 'GPL-3.0-or-later'}
RIG = 'default'
ARM_DROP_DEG = 60.0          # upper arms, degrees below horizontal: a relaxed A-pose (MPFB's own is ~47.5)

_SHARED = {'age': 0.5, 'cupsize': 0.5, 'firmness': 0.5, 'race': {'asian': 0.33, 'caucasian': 0.33, 'african': 0.33}}
# Build per body. MPFB's 'height' value is solved to hit TARGET_HEIGHT_M (None: keep 0.5,
# MPFB's default), because the slider's scale depends on the other values.
MACROS = {
    'body_01_medium':         {'gender': 0.0, 'height': 0.5, 'weight': 0.5, 'muscle': 0.5, 'proportions': 0.5,  **_SHARED},
    'body_02_tall_slender':   {'gender': 0.0, 'height': 0.5, 'weight': 0.2, 'muscle': 0.4, 'proportions': 0.65, **_SHARED},
    'body_03_short_heavy':    {'gender': 1.0, 'height': 0.5, 'weight': 1.0, 'muscle': 0.5, 'proportions': 0.35, **_SHARED},
    'body_04_broad_muscular': {'gender': 1.0, 'height': 0.5, 'weight': 0.7, 'muscle': 1.0, 'proportions': 0.7,  **_SHARED},
}
TARGET_HEIGHT_M = {
    'body_01_medium': None,
    'body_02_tall_slender': 1.78,
    'body_03_short_heavy': 1.65,
    'body_04_broad_muscular': 1.85,
}
HEIGHT_TOLERANCE_M = 0.005
assert set(MACROS) == set(TARGET_HEIGHT_M) == set(config.BODIES)

TOOLS = config.PACKAGE.parent/'tools'
MAKE_SCRIPT = TOOLS/'make_bodies.py'
RENDER_SCRIPT = TOOLS/'render_bodies.py'


class BodyError(RuntimeError):
    pass


def blender_environment():
    environment = dict(os.environ)
    environment['BLENDER_USER_RESOURCES'] = str(config.root()/'.blender')
    return environment


def _blender(script, arguments, log):
    command = [str(config.blender()), '--background', '--python-exit-code', '1', '--python', str(script), '--',
               *map(str, arguments)]
    with open(log, 'w', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=blender_environment())
    if result.returncode:
        raise BodyError(f'Blender failed running {script.name}; see {log}')


def measure(mesh):
    """Facts about one body, from its GLB alone."""
    names = mesh.bone_names
    head = lambda bone: mesh.bone_heads[names.index(bone)]
    upper, lower = head('upperarm01.L'), head('lowerarm01.L')
    arm = lower - upper                                   # glTF: +Y is up
    arm_drop = float(np.degrees(np.arcsin(-arm[1] / np.linalg.norm(arm))))
    lowest, highest = mesh.positions.min(axis=0), mesh.positions.max(axis=0)
    return {
        'height_m': round(float(highest[1] - lowest[1]), 4),
        'lowest_y_m': round(float(lowest[1]), 5),
        'width_m': round(float(highest[0] - lowest[0]), 4),
        'depth_m': round(float(highest[2] - lowest[2]), 4),
        'upper_arm_below_horizontal_deg': round(arm_drop, 1),
        'toes_ahead_of_heels': bool(head('toe1-1.L')[2] > head('foot.L')[2]),
        'vertices': int(len(mesh.positions)),
        'triangles': int(len(mesh.triangles)),
        'bones': len(names),
        'max_influences': int((mesh.weights > 0).sum(axis=1).max()),
        'weight_sum_range': [round(float(mesh.weights.sum(axis=1).min()), 6),
                             round(float(mesh.weights.sum(axis=1).max()), 6)],
    }


def check(folder):
    """Shared topology and skeleton across all four, plus each body's own sanity. Returns per-body measures."""
    meshes = {body: read_skinned_mesh(Path(folder)/body/'BODY.glb') for body in config.BODIES}
    reference = meshes[config.DONOR_BODY]
    problems = []
    for body, mesh in meshes.items():
        if mesh.positions.shape != reference.positions.shape or not np.array_equal(mesh.triangles, reference.triangles):
            problems.append(f'{body}: mesh topology differs from {config.DONOR_BODY}')
        if mesh.bone_names != reference.bone_names or not np.array_equal(mesh.bone_parents, reference.bone_parents):
            problems.append(f'{body}: skeleton differs from {config.DONOR_BODY}')
        sums = mesh.weights.sum(axis=1)
        if np.abs(sums - 1).max() > 1e-3:
            problems.append(f'{body}: skin weights do not sum to 1 (range {sums.min():.4f}..{sums.max():.4f})')
    measures = {body: measure(mesh) for body, mesh in meshes.items()}
    for body, facts in measures.items():
        if abs(facts['lowest_y_m']) > 0.002:
            problems.append(f'{body}: feet not on the floor (lowest point {facts["lowest_y_m"]} m)')
        if not facts['toes_ahead_of_heels']:
            problems.append(f'{body}: does not face +Z')
        if abs(facts['upper_arm_below_horizontal_deg'] - ARM_DROP_DEG) > 2.0:
            problems.append(f'{body}: arms not in an A-pose ({facts["upper_arm_below_horizontal_deg"]} deg)')
    if problems:
        raise BodyError('; '.join(problems))
    return measures


def _build(body, macros, folder, logs, tag):
    folder.mkdir(parents=True, exist_ok=True)
    _blender(MAKE_SCRIPT, ['--body-id', body, '--macros', json.dumps(macros), '--out', folder/'BODY.glb',
                           '--report', folder/'make.json', '--arm-drop', ARM_DROP_DEG], logs/f'make-{body}-{tag}.log')
    positions = read_skinned_mesh(folder/'BODY.glb').positions
    return float(positions[:, 1].max() - positions[:, 1].min())


def _solve_height(body, macros, target, folder, logs):
    """Bisect MPFB's height value (0..1, taller with larger values) until the body is target +- tolerance."""
    tries, low, high = [], 0.0, 1.0
    for attempt in range(14):
        value = round((low + high) / 2, 5)
        height = _build(body, {**macros, 'height': value}, folder, logs, f'try{attempt}')
        tries.append({'height_value': value, 'height_m': round(height, 4)})
        if abs(height - target) <= HEIGHT_TOLERANCE_M:
            return value, tries
        low, high = (value, high) if height < target else (low, value)
    raise BodyError(f'{body}: could not reach {target} m (tries: {tries})')


def _set_aside_draft(target):
    """An unreviewed draft may be set aside (never deleted); reviewed bodies may not be replaced."""
    record = json.loads((target/'bodies.json').read_text(encoding='utf-8')) if (target/'bodies.json').is_file() else {}
    if record.get('reviewed') or record.get('status') != 'draft_for_review':
        raise BodyError(f'{target} holds reviewed bodies; they are never replaced')
    n = 1
    while target.with_name(f'{target.name}.stale-{n}').exists():
        n += 1
    target.rename(target.with_name(f'{target.name}.stale-{n}'))


def make(replace_draft=False):
    target = config.bodies_folder()
    if target.exists():
        if not replace_draft:
            raise BodyError(f'{target} already exists; bodies are never overwritten (an unreviewed draft can be '
                            'set aside with --replace-draft)')
        _set_aside_draft(target)
    zip_path = config.root()/'downloads'/MPFB['zip']
    if not zip_path.is_file() or file_sha256(zip_path) != MPFB['zip_sha256']:
        raise BodyError(f'{zip_path} is missing or does not match the pinned MPFB hash')
    building = target.with_name(target.name+'.partial')
    if building.exists():
        shutil.rmtree(building)
    logs = building/'logs'
    logs.mkdir(parents=True)

    macros_used, height_tries = {}, {}
    for body, macros in MACROS.items():
        print(f'[fp] making {body}', flush=True)
        if TARGET_HEIGHT_M[body] is None:
            macros_used[body] = macros
            _build(body, macros, building/body, logs, 'final')
        else:
            value, height_tries[body] = _solve_height(body, macros, TARGET_HEIGHT_M[body], building/body, logs)
            macros_used[body] = {**macros, 'height': value}
    measures = check(building)
    print('[fp] rendering review images', flush=True)
    (building/'review').mkdir()
    _blender(RENDER_SCRIPT, ['--bodies', building, '--out', building/'review', *config.BODIES],
             building/'logs'/'render.log')

    blender_report = json.loads((building/config.DONOR_BODY/'make.json').read_text(encoding='utf-8'))
    record = {
        'made': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'status': 'draft_for_review',
        'reviewed': False,
        'mpfb': MPFB,
        'rig': RIG,
        'arm_drop_deg': ARM_DROP_DEG,
        'blender': blender_report['blender'],
        'scripts': {path.name: file_sha256(path) for path in (MAKE_SCRIPT, RENDER_SCRIPT)},
        'bodies': {body: {'sex': config.BODIES[body], 'macros': macros_used[body],
                          'target_height_m': TARGET_HEIGHT_M[body], 'height_tries': height_tries.get(body, []),
                          'measures': measures[body], 'sha256': file_sha256(building/body/'BODY.glb')}
                   for body in config.BODIES},
        'bone_names': blender_report['bones'],
    }
    (building/'bodies.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    building.rename(target)
    return target, record
