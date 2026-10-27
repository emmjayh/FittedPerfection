"""Step A (spec §5.6): reshape the medium body into a stand-in ("donor") that sits inside the source outfit.

Everything is built as one world transform per bone, so the donor stays a skinned
copy of body_01_medium (same mesh and rig) and its rest pose and posed shape come
from the same numbers:
1. uniform scale about the shoulder centre so the shoulder-to-crotch span matches,
   then a move onto the garment's shoulder centre;
2. leg length: the legs scale vertically about the crotch so crotch-to-floor matches
   (clamped), and the figure stands on the floor; hems are never used;
3. the trunk is centred horizontally inside the torso and trouser cloth;
4. pose: each arm turns about its own shoulder joint to point at the cuff, each leg
   about its hip to point at the ankle (straight, as no knee is used);
5. girth: each region's sections scale radially to sit just inside the innermost
   cloth, capped. Applied per vertex, blended by the region weights so regions join
   smoothly, identically in the rest pose and the posed shape.
Landmarks on both sides come from the same bind functions (fittedperfection/body.py).
"""
import numpy as np

from . import body as bodies

GARMENT_PARTS = {
    'trunk': ('torso', 'pants', 'waistband'),
    'arm.left': ('sleeve.left', 'cuff.left'), 'arm.right': ('sleeve.right', 'cuff.right'),
    'leg.left': ('pant_leg.left',), 'leg.right': ('pant_leg.right',),
}


def translation(offset):
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def about(point, linear):
    """4x4 applying the 3x3 `linear` about `point`."""
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    return translation(point) @ matrix @ translation(-np.asarray(point))


def rotation_between(a, b):
    """Smallest rotation turning direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * cross + (1 - c) * cross @ cross


def apply(matrix, point):
    return matrix[:3, :3] @ point + matrix[:3, 3]


def build(binding, labels, garment_centres, garment_areas, config, body_id='body_01_medium'):
    """The donor: rest positions, posed positions, per-bone pose transforms, and a record of every choice."""
    body = bodies.load(body_id)
    marks = bodies.landmarks(body)
    bones = len(body.bone_names)
    record = {'body': body_id}

    # 1. scale on the shoulder centre by the shoulder-to-crotch span, then move onto the garment's shoulder centre
    body_shoulder = np.array(marks['shoulders']['centre'])
    garment_shoulder = np.array(binding['shoulders']['centre'])
    body_span = body_shoulder[2] - marks['crotch_height_m']
    garment_span = garment_shoulder[2] - binding['crotch']['height_m']
    scale = garment_span / body_span
    shape = translation(garment_shoulder) @ np.diag([scale, scale, scale, 1.0]) @ translation(-body_shoulder)
    record['scale'] = round(float(scale), 4)

    # 2. leg length: crotch-to-floor matches (clamped); then stand on the floor
    crotch_now = apply(shape, np.array([0, 0, marks['crotch_height_m']]))[2]
    floor_now = apply(shape, np.array([0, 0, marks['floor_m']]))[2]
    wanted = binding['crotch']['height_m']
    low, high = config['leg_length_range']
    leg_scale = float(np.clip(wanted / (crotch_now - floor_now), low, high))
    record['leg_scale'] = round(leg_scale, 4)
    record['leg_scale_clamped'] = bool(not np.isclose(leg_scale, wanted / (crotch_now - floor_now)))
    legs = about(np.array([0, 0, crotch_now]), np.diag([1.0, 1.0, leg_scale]))
    per_bone = np.repeat(shape[None], bones, axis=0)
    for region in ('leg.left', 'leg.right'):
        per_bone[body.bone_region == region] = legs @ shape
    rest0 = bodies.skin(body, per_bone)
    stand = translation([0, 0, -rest0[:, 2].min()])

    # 3. centre the trunk horizontally inside the torso/trouser cloth
    trunk_cloth = np.isin(labels, ('torso', 'pants', 'waistband')) & \
        (garment_centres[:, 2] > wanted + 0.1) & (garment_centres[:, 2] < garment_shoulder[2] - 0.1)
    trunk_body = (body.vertex_region == 'trunk') & (rest0[:, 2] - rest0[:, 2].min() > wanted + 0.1) & \
        (rest0[:, 2] - rest0[:, 2].min() < garment_shoulder[2] - 0.1)
    shift = np.zeros(3)
    if trunk_cloth.sum() > 20 and trunk_body.sum() > 20:
        shift[:2] = np.average(garment_centres[trunk_cloth, :2], axis=0, weights=garment_areas[trunk_cloth]) - \
            rest0[trunk_body, :2].mean(axis=0)
    centre = translation(shift)
    record['trunk_shift_m'] = np.round(shift, 4).tolist()
    rest_bones = np.einsum('ij,bjk->bik', centre @ stand, per_bone)
    rest = bodies.skin(body, rest_bones)

    # 4. pose: arms about their shoulder joints to the cuff, legs about their hips to the ankle
    pose = np.repeat(np.eye(4)[None], bones, axis=0)
    record['pose'] = {}
    joints = {name: apply(rest_bones[body.bone_names.index(name)], body.joint(name)) for name in
              ('upperarm01.L', 'upperarm01.R', 'wrist.L', 'wrist.R', 'upperleg01.L', 'upperleg01.R', 'foot.L', 'foot.R')}
    for region, side, start, end, target in (
            ('arm.left', 'left', 'upperarm01.L', 'wrist.L', (binding['sleeves'].get('left') or {}).get('cuff_end')),
            ('arm.right', 'right', 'upperarm01.R', 'wrist.R', (binding['sleeves'].get('right') or {}).get('cuff_end')),
            ('leg.left', 'left', 'upperleg01.L', 'foot.L', binding['legs']['left']['ankle_end']),
            ('leg.right', 'right', 'upperleg01.R', 'foot.R', binding['legs']['right']['ankle_end'])):
        if target is None:
            record['pose'][region] = 'kept (no sleeve to follow)'
            continue
        pivot = joints[start]
        turn = rotation_between(joints[end] - pivot, np.asarray(target) - pivot)
        pose[body.bone_region == region] = about(pivot, turn)
        record['pose'][region] = round(float(np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1)))), 2)
    posed_bones = np.einsum('bij,bjk->bik', pose, rest_bones)
    posed = bodies.skin(body, posed_bones)

    # 5. girth, measured in the posed donor against the cloth, applied to both shapes
    rest, posed, record['girth'] = girth(body, rest, posed, pose, labels, garment_centres, config)
    return {'body': body, 'rest': rest, 'posed': posed, 'pose': pose, 'rest_bones': rest_bones,
            'posed_bones': posed_bones, 'record': record}


def region_axes(body, posed, pose):
    """Per region: a point and a direction its sections are taken along (posed space)."""
    axes = {'trunk': (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))}
    for region, start, end in (('arm.left', 'upperarm01.L', 'wrist.L'), ('arm.right', 'upperarm01.R', 'wrist.R'),
                               ('leg.left', 'upperleg01.L', 'foot.L'), ('leg.right', 'upperleg01.R', 'foot.R')):
        inside = body.vertex_region == region
        points = posed[inside]
        centre = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
        axes[region] = (centre, vt[0] / np.linalg.norm(vt[0]))
    return axes


def girth(body, rest, posed, pose, labels, garment_centres, config, step=0.03):
    low, high = config['girth_range']
    axes = region_axes(body, posed, pose)
    move_posed = np.zeros_like(posed)
    move_rest = np.zeros_like(rest)
    summary = {}
    for region, parts in GARMENT_PARTS.items():
        origin, axis = axes[region]
        weight = body.region_weight[region]
        cloth = np.isin(labels, parts)
        t_body = (posed - origin) @ axis
        t_cloth = (garment_centres - origin) @ axis
        inside = body.vertex_region == region
        factors, centres_posed, centres_rest, starts = [], [], [], []
        for start in np.arange(t_body[inside].min(), t_body[inside].max(), step):
            section = inside & (t_body >= start) & (t_body < start + step)
            if section.sum() < 6:
                continue
            c = posed[section].mean(axis=0)
            radial = posed[section] - c - np.outer((posed[section] - c) @ axis, axis)
            r_body = np.percentile(np.linalg.norm(radial, axis=1), 90)
            ring = cloth & (t_cloth >= start) & (t_cloth < start + step)
            factor = 1.0
            if ring.sum() >= 15:
                around = garment_centres[ring] - c - np.outer((garment_centres[ring] - c) @ axis, axis)
                distance = np.linalg.norm(around, axis=1)
                angles = np.arctan2(around @ np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0]),
                                    around @ np.cross(axis, np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0])))
                covered = len(np.unique(((angles + np.pi) / (2 * np.pi) * 12).astype(int) % 12)) / 12
                if covered >= 0.6 and r_body > 1e-4:
                    factor = float(np.clip(0.97 * np.percentile(distance, 10) / r_body, low, high))
            factors.append(factor)
            centres_posed.append(c)
            centres_rest.append(rest[section].mean(axis=0))
            starts.append(start)
        if not factors:
            continue
        factors = np.array(factors)
        smoothed = np.array([np.median(factors[max(0, i - 1):i + 2]) for i in range(len(factors))])
        index = np.clip(np.searchsorted(np.array(starts), t_body, side='right') - 1, 0, len(starts) - 1)
        f = smoothed[index][:, None]
        cp, cr = np.array(centres_posed)[index], np.array(centres_rest)[index]
        along_p = np.outer((posed - cp) @ axis, axis)
        move_posed += weight[:, None] * ((f - 1) * (posed - cp - along_p))
        # the same radial factor in the rest pose, along the region's rest direction
        rest_axis = np.linalg.inv(pose[np.argmax(body.bone_region == region)][:3, :3]) @ axis if region != 'trunk' else axis
        along_r = np.outer((rest - cr) @ rest_axis, rest_axis)
        move_rest += weight[:, None] * ((f - 1) * (rest - cr - along_r))
        summary[region] = {'sections': len(factors), 'min': round(float(smoothed.min()), 3),
                           'max': round(float(smoothed.max()), 3),
                           'clamped': int(np.sum(np.isclose(smoothed, low) | np.isclose(smoothed, high)))}
    return rest + move_rest, posed + move_posed, summary
