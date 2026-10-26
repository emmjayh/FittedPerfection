"""Stage 2, views (spec §5.2): calibrated renders of the cage, face-ID images and the turntable video.

Our own cameras (fittedperfection/cameras.py) drive Blender (fittedperfection/
blender/views.py). Every run then checks its own face-ID images: every pixel must
lie inside the projected outline of the triangle it names. That proves both the
camera numbers and the triangle indexing the label stage relies on, and unlike a
centre-of-pixels test it is not fooled by triangles that are only partly visible.
"""
import json
import shutil

import numpy as np
from PIL import Image

from . import cameras, config, faceid, settings
from .gltf import read_textured_mesh

BLENDER_SCRIPT = config.PACKAGE/'blender'/'views.py'
CODE = ('views.py', 'cameras.py', 'faceid.py', 'gltf.py', 'blender/views.py')
MIN_PIXELS_ON_THEIR_TRIANGLE = 0.999        # fraction within 1 px of the triangle they name


class ViewsError(RuntimeError):
    pass


def distance_to_named_triangle(camera, ids, positions, triangles):
    """For every foreground pixel: its distance in pixels from the projected triangle its ID names (0 inside)."""
    ys, xs = np.nonzero(ids >= 0)
    px, py, _ = cameras.project(camera, positions)
    corners = np.stack([px, py], axis=1)[triangles[ids[ys, xs]]]
    pixel = np.stack([xs + 0.5, ys + 0.5], axis=1)
    inside = np.ones(len(pixel), dtype=bool)
    nearest = np.full(len(pixel), np.inf)
    for i in range(3):
        a, b, c = corners[:, i], corners[:, (i + 1) % 3], corners[:, (i + 2) % 3]
        edge, to_pixel = b - a, pixel - a
        cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        inside &= cross(edge, to_pixel) * cross(edge, c - a) >= 0
        along = np.clip((to_pixel * edge).sum(axis=1) / np.maximum((edge * edge).sum(axis=1), 1e-12), 0, 1)
        nearest = np.minimum(nearest, np.hypot(*(to_pixel - along[:, None] * edge).T))
    return np.where(inside, 0.0, nearest)


def check_face_ids(camera, image_path, positions, triangles):
    """How well this face-ID image agrees with the cameras and the cage's triangles."""
    rgb = np.asarray(Image.open(image_path).convert('RGB'))
    ids = faceid.decode(rgb)
    foreground = ids >= 0
    if (ids[foreground] >= len(triangles)).any():
        raise ViewsError(f'{image_path.name}: face IDs beyond the cage\'s {len(triangles)} triangles')
    residual = (rgb[foreground].astype(np.int64) - faceid.OFFSET) % faceid.STEP
    residual = np.minimum(residual, faceid.STEP - residual)
    distance = distance_to_named_triangle(camera, ids, positions, triangles)
    return {'coverage': round(float(foreground.mean()), 4),
            'triangles_seen': int(len(np.unique(ids[foreground]))),
            'pixels_on_their_triangle': round(float(np.mean(distance <= 1.0)), 5) if distance.size else 1.0,
            'max_pixel_distance': round(float(distance.max()), 2) if distance.size else 0.0,
            'max_byte_rounding': int(residual.max()) if residual.size else 0}


def run(ctx, out):
    view_settings = settings.section('views')
    cage_path = ctx.output_of('intake')/'CAGE.glb'
    cage = read_textured_mesh(cage_path)
    plan = cameras.plan(cage.positions, view_settings['orbit_frames'], view_settings['image_size'])
    (out/'cameras.json').write_text(json.dumps(plan, indent=1), encoding='utf-8', newline='\n')
    ctx.call([config.blender(), '--background', '--factory-startup', '--python-exit-code', '1',
              '--python', BLENDER_SCRIPT, '--', '--cage', cage_path, '--cameras', out/'cameras.json',
              '--fps', view_settings['fps'], '--out', out], 'views-blender.log')
    blender = json.loads((out/'blender.json').read_text(encoding='utf-8'))
    if blender['triangles'] != len(cage.triangles):
        raise ViewsError(f'Blender saw {blender["triangles"]} triangles, the cage has {len(cage.triangles)}')
    if blender['video_frames'] != view_settings['orbit_frames']:
        raise ViewsError(f'orbit.mp4 has {blender["video_frames"]} frames, expected {view_settings["orbit_frames"]}')

    checks = {camera['name']: check_face_ids(camera, out/'faceid'/f'{camera["name"]}.png', cage.positions,
                                             cage.triangles) for camera in plan}
    worst = min(checks.items(), key=lambda item: item[1]['pixels_on_their_triangle'])
    if worst[1]['pixels_on_their_triangle'] < MIN_PIXELS_ON_THEIR_TRIANGLE:
        raise ViewsError(f'face IDs disagree with the cameras in {worst[0]}: only '
                         f'{worst[1]["pixels_on_their_triangle"]:.2%} of pixels lie on the triangle they name')
    if max(check['max_byte_rounding'] for check in checks.values()) > 2:
        raise ViewsError('face-ID colours were rounded by more than 2; the code is no longer safe')

    calibrated = cameras.calibrated_frames(view_settings['orbit_frames'])
    for name, index in calibrated.items():
        shutil.copyfile(out/'color'/f'orbit-{index:03d}.png', out/f'{name}.png')
    record = {'cage_triangles': len(cage.triangles), 'image_size': view_settings['image_size'],
              'orbit_frames': view_settings['orbit_frames'], 'fps': view_settings['fps'],
              'calibrated': {name: f'orbit-{index:03d}' for name, index in calibrated.items()},
              'stills': ['top', 'bottom'], 'video': 'orbit.mp4', 'video_bytes': blender['video_bytes'],
              'face_id_checks': checks}
    (out/'views.json').write_text(json.dumps(record, indent=1), encoding='utf-8', newline='\n')
    return {'frames': len(plan), 'worst_pixels_on_their_triangle': worst[1]['pixels_on_their_triangle'],
            'mean_triangles_seen': int(np.mean([c['triangles_seen'] for c in checks.values()])),
            'video_bytes': blender['video_bytes']}
