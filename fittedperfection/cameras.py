"""The calibrated orthographic cameras of the views stage (spec §5.2), and their projection.

Canonical frame: Z up, garment front -Y, its own left +X. The turntable circles
the vertical axis. Frame 0 looks at the front from -Y and the angle grows
towards the garment's left, so frames 0, n/4, n/2 and 3n/4 are front, left, back
and right. Top and bottom are stills looking straight down and up. Pure numpy;
the same numbers drive Blender's cameras and the label stage's projection.
"""
import numpy as np

CALIBRATED = ('front', 'left', 'back', 'right')


def _camera(name, centre, forward, up, scale, size):
    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    return {'name': name, 'centre': [float(c) for c in centre], 'forward': forward.round(12).tolist(),
            'up': up.round(12).tolist(), 'right': right.round(12).tolist(), 'scale': float(scale), 'size': int(size)}


def plan(points, orbit_frames, size, margin=1.06):
    """Every camera for one cage: the turntable frames, then top and bottom."""
    points = np.asarray(points, dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    centre = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, (low[2] + high[2]) / 2])
    radius = float(np.sqrt(((points[:, :2] - centre[:2]) ** 2).sum(axis=1)).max())
    orbit_scale = max(2 * radius, high[2] - low[2]) * margin
    cameras = []
    for i in range(orbit_frames):
        angle = 2 * np.pi * i / orbit_frames
        towards_camera = np.array([np.sin(angle), -np.cos(angle), 0.0])
        cameras.append(_camera(f'orbit-{i:03d}', centre, -towards_camera, (0, 0, 1), orbit_scale, size))
    flat_scale = max(high[0] - low[0], high[1] - low[1]) * margin
    cameras.append(_camera('top', centre, (0, 0, -1), (0, 1, 0), flat_scale, size))
    cameras.append(_camera('bottom', centre, (0, 0, 1), (0, -1, 0), flat_scale, size))
    return cameras


def calibrated_frames(orbit_frames):
    """Turntable frame index for front, left, back and right."""
    if orbit_frames % 4:
        raise ValueError('the turntable frame count must be a multiple of 4')
    return {name: i * orbit_frames // 4 for i, name in enumerate(CALIBRATED)}


def project(camera, points):
    """Points -> (pixel x, pixel y, depth towards the camera). Pixel (0, 0) is the top-left corner."""
    offset = np.asarray(points, dtype=np.float64) - np.array(camera['centre'])
    size, scale = camera['size'], camera['scale']
    x = offset @ np.array(camera['right']) / scale * size + size / 2
    y = size / 2 - offset @ np.array(camera['up']) / scale * size
    depth = -(offset @ np.array(camera['forward']))
    return x, y, depth
