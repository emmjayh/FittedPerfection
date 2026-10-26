"""Which way a garment faces, read from its feet (spec §5.1: from evidence, never guessed).

Pure numpy, so Blender's bundled Python can run it during intake as well.

Points are in Blender's frame after import (Z up). Just above the floor each
foot or shoe reaches much further towards the toes than towards the heel,
measured from its own leg's centre on the lower shin (18-26% of height). That
band sits above most boot toe boxes, which would otherwise pull the centre
towards the toes and hide the difference. The result is the turn about Z that
puts the toes at -Y, the canonical front, with a status:

    confirmed    both feet agree and toe reach beats heel reach by min_ratio
    provisional  both feet agree but by less (e.g. trousers pooling over the shoes
                 reach back as far as the toes reach forward). The bind stage must
                 confirm it from the footwear labels or refuse the outfit.

It refuses outright when there are no two separate feet (e.g. a long dress hides
them), when the feet disagree, or when toe and heel reach are about equal.
"""
import numpy as np

DIRECTIONS = {'+X': (1.0, 0.0), '-X': (-1.0, 0.0), '+Y': (0.0, 1.0), '-Y': (0.0, -1.0)}
OPPOSITE = {'+X': '-X', '-X': '+X', '+Y': '-Y', '-Y': '+Y'}
TURN_TO_FRONT = {'-Y': 0, '+X': -90, '+Y': 180, '-X': 90}      # degrees about +Z that send the toes to -Y


class OrientationUnclear(ValueError):
    pass


def _closest_distance(a, b, limit):
    """Smallest distance between 2D point sets a and b, or `limit` if nothing is closer than that."""
    cells = {}
    for point, key in zip(b, map(tuple, np.floor(b / limit).astype(np.int64))):
        cells.setdefault(key, []).append(point)
    cells = {key: np.array(value) for key, value in cells.items()}
    best = limit
    for point, (x, y) in zip(a, np.floor(a / limit).astype(np.int64)):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near = cells.get((x + dx, y + dy))
                if near is not None:
                    best = min(best, float(np.sqrt(((near - point) ** 2).sum(axis=1)).min()))
    return best


def _two_means(points, rounds=25):
    """Deterministic 2-means on 2D points: start from the two mutually farthest-apart points."""
    centre = points.mean(axis=0)
    first = points[np.argmax(((points - centre) ** 2).sum(axis=1))]
    second = points[np.argmax(((points - first) ** 2).sum(axis=1))]
    centres = np.stack([first, second])
    for _ in range(rounds):
        nearest = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
        if min((nearest == 0).sum(), (nearest == 1).sum()) == 0:
            break
        updated = np.stack([points[nearest == k].mean(axis=0) for k in (0, 1)])
        if np.allclose(updated, centres):
            break
        centres = updated
    return centres


def facing(points, foot_band=0.03, leg_band=(0.18, 0.26), min_leg_gap=0.05, min_leg_clearance=0.01,
           min_ratio=1.4, provisional_min_ratio=1.05, min_points=20):
    """Return the toe direction, the turn about Z that makes it -Y, a status and the evidence; raise if unclear."""
    points = np.asarray(points, dtype=np.float64)
    low, high = points[:, 2].min(), points[:, 2].max()
    height = high - low
    relative = (points[:, 2] - low) / height
    legs = points[(relative >= leg_band[0]) & (relative <= leg_band[1])][:, :2]
    feet = points[relative <= foot_band][:, :2]
    if len(legs) < 2 * min_points or len(feet) < 2 * min_points:
        raise OrientationUnclear('too few points near the floor to find the feet')
    # Left and right legs on the lower shin. They must be genuinely separate there: at least
    # min_leg_clearance apart everywhere. The two halves of a long skirt touch; two baggy
    # trouser legs, however close, do not.
    centres = _two_means(legs)
    leg_owner = np.argmin(((legs[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    shins = [legs[leg_owner == k] for k in (0, 1)]
    if min(len(shin) for shin in shins) < min_points:
        raise OrientationUnclear('only one leg found on the lower shin')
    centres = np.stack([shin.mean(axis=0) for shin in shins])
    gap = float(np.linalg.norm(centres[0] - centres[1]))
    clearance = _closest_distance(shins[0], shins[1], limit=max(min_leg_clearance, 1e-3) * 4)
    if gap < min_leg_gap or clearance < min_leg_clearance:
        raise OrientationUnclear(f'the legs are not separate on the lower shin (closest {clearance:.3f} m); '
                                 'something such as a long skirt hides them')

    # Everything at the floor (sole, separate heel tip, cloth around it) belongs to the nearest
    # leg; each foot's reach is measured from its own leg's centre.
    foot_owner = np.argmin(((feet[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    reaches = []
    for k in (0, 1):
        foot = feet[foot_owner == k]
        if len(foot) < min_points:
            raise OrientationUnclear('only one foot found at the floor')
        reaches.append({name: float(((foot - centres[k]) @ np.array(d)).max()) for name, d in DIRECTIONS.items()})

    toes = [max(reach, key=reach.get) for reach in reaches]
    if toes[0] != toes[1]:
        raise OrientationUnclear(f'the two feet point different ways ({toes[0]} and {toes[1]})')
    toe = toes[0]
    ratios = [reach[toe] / max(reach[OPPOSITE[toe]], 1e-6) for reach in reaches]
    if min(ratios) < provisional_min_ratio:
        raise OrientationUnclear(f'toe and heel reach are too similar (ratios {ratios[0]:.2f}, {ratios[1]:.2f}; '
                                 f'need {provisional_min_ratio} to continue provisionally, {min_ratio} to confirm)')
    return {
        'toe_direction': toe,
        'turn_about_z_degrees': TURN_TO_FRONT[toe],
        'status': 'confirmed' if min(ratios) >= min_ratio else 'provisional',
        'evidence': {
            'height_m': round(float(height), 4),
            'legs_apart_m': round(gap, 4),
            'legs_closest_m': round(clearance, 4),
            'toe_over_heel_reach': [round(r, 2) for r in ratios],
            'reach_m': [{name: round(value, 4) for name, value in reach.items()} for reach in reaches],
        },
    }


def turn_about_z(points, degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(points) @ rotation.T
