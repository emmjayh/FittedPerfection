"""Step E (spec §5.6, §7.1): shoes, without warping the body.

1. Measure each shoe once on the source: its footprint frame and the sole thickness at the
   ball (the height of the sole's edge, where the cross-section narrows from outsole to upper).
   The foot it was made for is taken as a share of the source figure's height (about 15%);
   the donor's foot (Step A, posed into the outfit) gives where that foot sits in the shoe.
2. Place it on each body: a uniform scale so it keeps its margin around that body's foot,
   and a rigid move mapping the donor's foot frame onto the target's, with the footbed on
   the target's sole (the shoe's bottom goes under the floor). Pitch is kept: heels stay heels.
3. Footwear above the ankle (boot shafts) is cloth for the wrap, not part of the rigid shoe.
4. The whole fit is raised by the sole thickness so the shoes stand on the floor; the body is
   not changed, only offset (root_offset_m).
5. Ghost removal: body faces inside or poking through a shoe are listed as hidden, the way
   games hide skin under clothing, so a flat foot in a heel is simply not drawn.
numpy/scipy only (sealed).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

from . import body as bodies

SIDES = {'left': 'foot.L', 'right': 'foot.R'}
BALL = (0.6, 0.8)                 # where along the footprint (from the heel) the ball of the foot is
LIP_STEP_M = 0.005                # height step of the cross-sections searched for the sole's edge
LIP_MIN_DROP_M = 0.005            # the sole's edge must narrow the cross-section by at least this
DEFAULT_SOLE_M = 0.015            # sole thickness when no edge is found


def foot_vertices(body, side):
    """Indices of the body's foot vertices (mostly weighted to the foot bone and its children)."""
    bones = bodies._descendants(body.parents, body.bone_names.index(SIDES[side]))
    weight = (body.weights * np.isin(body.joints, list(bones))).sum(axis=1)
    return np.flatnonzero(weight >= 0.5)


def foot_points(body, positions, side):
    """The body's foot (vertices mostly weighted to the foot bone and its children)."""
    return positions[foot_vertices(body, side)]


def frame(points):
    """A footprint frame: origin at the heel on the sole, rotation with columns (forward, left, up),
    and the footprint's length and width. Forward is the long axis pointing to the front (-Y)."""
    xy = points[:, :2]
    centre = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    forward = vt[0] if vt[0] @ [0, -1] > 0 else -vt[0]
    left = np.array([-forward[1], forward[0]])
    along, across = (xy - centre) @ forward, (xy - centre) @ left
    origin = np.r_[centre + along.min() * forward, points[:, 2].min()]
    rotation = np.column_stack([np.r_[forward, 0], np.r_[left, 0], [0, 0, 1.0]])
    return {'origin': origin, 'rotation': rotation, 'length': float(np.ptp(along)), 'width': float(np.ptp(across))}


def sole_thickness(points, search):
    """Height of the sole's edge at the ball: the cross-section narrows sharply where the outsole ends and the
    upper begins. Returns (thickness, found)."""
    f = frame(points)
    local = (points - f['origin']) @ f['rotation']
    ball = (local[:, 0] > BALL[0] * f['length']) & (local[:, 0] < BALL[1] * f['length'])
    heights = np.arange(0.0, search, LIP_STEP_M)
    widths = []
    for z in heights:
        layer = ball & (local[:, 2] >= z) & (local[:, 2] < z + LIP_STEP_M)
        widths.append(np.ptp(local[layer, 1]) if layer.sum() >= 4 else np.nan)
    widths = np.array(widths)
    drops = widths[:-1] - widths[1:]
    if not np.isfinite(drops).any() or np.nanmax(drops) < LIP_MIN_DROP_M:
        return DEFAULT_SOLE_M, False
    edge = int(np.nanargmax(drops))
    return float(heights[edge] + LIP_STEP_M), True


def placement(shoe_points, donor_foot, target_foot, thickness, source_foot_length):
    """The similarity (scale s, then a rigid move) taking the shoe from the donor's foot onto the target's foot,
    footbed on the target's sole. The scale keeps the shoe's margin around the foot: target foot over the foot
    the shoe was made for (source_foot_length, with the donor's foot proportions for its width); the donor's
    foot gives where that foot sits in the shoe. Returns a function of points and a record."""
    d, t = frame(donor_foot), frame(target_foot)
    source_width = d['width'] * source_foot_length / d['length']
    scale = max(t['length'] / source_foot_length, t['width'] / source_width)
    linear = scale * t['rotation'] @ d['rotation'].T
    shift = t['origin'] - linear @ d['origin']
    shift[2] += t['origin'][2] - scale * thickness - (shoe_points @ linear.T + shift)[:, 2].min()   # bottom exact

    def move(points):
        return points @ linear.T + shift

    shoe = frame(shoe_points)
    return move, {'scale': round(float(scale), 4), 'sole_m': round(float(thickness), 4),
                  'sole_placed_m': round(float(scale * thickness), 4),
                  'shoe_to_foot_source': round(shoe['length'] / source_foot_length, 3),
                  'shoe_to_foot_fitted': round(shoe['length'] * scale / t['length'], 3)}


def shaft(shoe_vertices, points, ankle_height):
    """Shoe points above the ankle line (boot shafts): cloth for the wrap rather than part of the rigid shoe."""
    return shoe_vertices[points[shoe_vertices, 2] > ankle_height]


def join(points, cloth, footwear, blend):
    """Cloth that meets a shoe (a stocking ending inside it, a trouser cuff over it): the cloth points within
    `blend` of the footwear, each one's nearest footwear point, and a weight from 1 (touching) to 0 (at `blend`)."""
    cloth = np.setdiff1d(cloth, footwear)
    if not len(footwear) or not len(cloth):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    distance, nearest = cKDTree(points[footwear]).query(points[cloth])
    near = distance < blend
    weight = 1.0 - distance[near] / blend
    return cloth[near], footwear[nearest[near]], weight * weight * (3 - 2 * weight)


def hidden_faces(body_positions, body_triangles, shoe_points, shoe_triangles, grid):
    """Body faces inside a shoe or poking through it: within the shoe's footprint and below its upper
    surface there (a height map of the shoe on a `grid` raster, interior filled)."""
    samples = np.vstack([shoe_points, shoe_points[shoe_triangles].mean(axis=1)])
    low = samples[:, :2].min(axis=0) - 2 * grid
    cells = np.floor((samples[:, :2] - low) / grid).astype(int)
    size = cells.max(axis=0) + 3
    top = np.full(size, -np.inf)
    np.maximum.at(top, (cells[:, 0], cells[:, 1]), samples[:, 2])
    covered = ndimage.binary_fill_holes(ndimage.binary_dilation(np.isfinite(top), iterations=2))
    reach = max(3, int(round(0.04 / grid)))                      # interior cells take the nearby upper's height
    top = np.where(covered, ndimage.maximum_filter(np.where(np.isfinite(top), top, -1e9), size=2 * reach + 1), -np.inf)
    centres = body_positions[body_triangles].mean(axis=1)
    c = np.floor((centres[:, :2] - low) / grid).astype(int)
    inside = (c >= 0).all(axis=1) & (c[:, 0] < size[0]) & (c[:, 1] < size[1])
    hidden = np.zeros(len(centres), dtype=bool)
    hidden[inside] = centres[inside, 2] < top[c[inside, 0], c[inside, 1]]
    return np.flatnonzero(hidden)
