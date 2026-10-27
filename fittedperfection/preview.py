"""Review renders for fit work (Blender, outside the seal). Not a pipeline stage: it writes to runs/_review/.

    python -m fittedperfection preview NAME --donor     the donor (Step A) inside the outfit's cage
    python -m fittedperfection preview NAME --fit       the source, then every body wearing its outfit
    python -m fittedperfection preview NAME --pose      every body and outfit in the test pose
"""
import json
import subprocess

import numpy as np

from . import config, donor, meshes, seal, settings
from .glbwrite import write_mesh
from .gltf import read_textured_mesh

SCRIPT = config.PACKAGE/'blender'/'preview.py'


def donor_inputs(run_folder):
    binding = json.loads((run_folder/'bind'/'binding.json').read_text(encoding='utf-8'))
    labels = np.load(run_folder/'labels'/'labels.npz')['labels'].astype(str)
    cage = read_textured_mesh(run_folder/'full'/'CAGE.glb')
    centres = cage.positions[cage.triangles].mean(axis=1)
    _, areas = meshes.face_normals(cage.positions, cage.triangles)
    return binding, labels, centres, areas


def render(items, out, name, pose=False):
    command = [str(config.blender()), '--background', '--factory-startup', '--python-exit-code', '1', '--python',
               str(SCRIPT), '--', '--out', str(out), '--name', name] + (['--pose'] if pose else [])
    for item in items:
        command += ['--item', item]
    log = out/f'{name}.log'
    with open(log, 'w', encoding='utf-8') as stream:
        if subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode:
            raise RuntimeError(f'preview render failed; see {log}')


def donor_preview(name):
    run_folder = config.runs_folder()/name
    out = config.runs_folder()/'_review'/f'donor-{name}'
    out.mkdir(parents=True, exist_ok=True)
    with seal.sealed():                              # built exactly as the sealed fit will build it
        built = donor.build(*donor_inputs(run_folder), settings.section('fit'))
    write_mesh(out/'donor.glb', built['posed'], built['body'].triangles, colour=(0.82, 0.63, 0.51, 1.0), name='donor')
    (out/'donor.json').write_text(json.dumps(built['record'], indent=1), encoding='utf-8')
    render([f'{run_folder/"full"/"CAGE.glb"}|texture', f'{out/"donor.glb"}|skin'], out, name)
    return out, built['record']


def fit_preview(name, spacing=1.3, pose=False):
    """The source, then each body (BODY.glb: raised, covered faces hidden) wearing its full-resolution OUTFIT.glb
    (or the cage DRAFT.glb before Step G existed). pose: elbows and knees bent (spec §5.7 test pose)."""
    run_folder = config.runs_folder()/name
    out = config.runs_folder()/'_review'/(f'pose-{name}' if pose else f'fit-{name}')
    out.mkdir(parents=True, exist_ok=True)
    items = [] if pose else [f'{run_folder/"full"/"CAGE.glb"}|texture|0']
    for i, body in enumerate(config.BODIES, start=1):
        folder = run_folder/'fit'/body
        shown, outfit = folder/'BODY.glb', folder/'OUTFIT.glb'
        items += [f'{shown if shown.is_file() else config.body_file(body)}|skin|{i * spacing}',
                  f'{outfit if outfit.is_file() else folder/"DRAFT.glb"}|texture|{i * spacing}']
    render(items, out, name, pose)
    return out
