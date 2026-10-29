"""Step A (spec §5.6): reshape the medium body into a stand-in ("donor") that sits inside the source outfit.

Everything is built as one world transform per bone, so the donor stays a skinned
copy of body_01_medium (same mesh and rig) and its rest pose and posed shape come
from the same numbers:
1. uniform scale about the shoulder centre so the shoulder-to-crotch span matches,
   then a move onto the garment's shoulder centre;
2. leg length: the legs scale vertically about the crotch so crotch-to-floor matches
   (clamped), and the figure stands on the floor; hems are never used;
   then the neck lengthens (clamped) so the head starts at the top of the neck cloth;
3. (no global centring: the trunk is centred front to back slice by slice in step 5);
4. pose: each arm turns about its own shoulder joint to point at the cuff, each leg
   about its hip to point at the ankle (straight, as no knee is used);
5. girth: each region's sections scale to sit just inside the innermost cloth, capped.
   Limbs scale radially. Trunk slices fit depth and width separately and move front to
   back so their front and back margins match (a busty dress with a thin back is deep in
   front and shallow behind; one radial factor left the donor's back outside the cloth).
   Applied per vertex, blended by the region weights so regions join smoothly,
   identically in the rest pose and the posed shape.
Landmarks on both sides come from the same bind functions (fittedperfection/body.py).
"""
import numpy as np

from . import body as bodies

MIN_CUFF_FACES = 20              # a sleeve end counts as a cuff (the wrist) with at least this many cuff faces
NECK_RADIUS_M = 0.12             # neck cloth: torso/collar this close to the centre line, above the shoulders
ANKLE_ALONG_SOLE = 0.25          # the ankle stands over this share of the sole, from the heel
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


def rotation_between_axis_angle(axis, angle):
    """Rotation of `angle` radians about the unit `axis` (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def apply(matrix, point):
    return matrix[:3, :3] @ point + matrix[:3, 3]


def build(binding, labels, garment_centres, garment_areas, config, body_id='body_01_medium'):
    """The donor: rest positions, posed positions, per-bone pose transforms, and a record of every choice."""
    body = bodies.load(body_id)
    marks = bodies.landmarks(body)
    bones = len(body.bone_names)
    record = {'body': body_id}

    # 1. scale on the shoulder centre by the shoulder-to-crotch span, then move onto the garment's shoulder centre;
    #    the body's shoulders are measured the way bind measured the garment's (sleeve roots or shoulder_top)
    sleeveless = binding['shoulders']['method'].startswith('shoulder_top')
    body_shoulder = np.array(marks['shoulder_top' if sleeveless else 'shoulders']['centre'])
    record['shoulders_measured_by'] = marks['shoulder_top' if sleeveless else 'shoulders']['method']
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

    # 2b. neck length: generator figures are headless with long neck stubs, and their neck cloth runs to the top of
    #     the stub. Mapped as it is, a neckline lands on the donor's chin. The neck lengthens (never shortens,
    #     clamped) so the top of the neck cloth (torso, collar and a hood's neck near the centre line) sits at a set
    #     share of the donor's neck, from its base to the skull.
    record['neck_scale'] = 1.0
    neck_cloth = np.isin(labels, ('torso', 'collar', 'hood')) & (garment_centres[:, 2] > garment_shoulder[2]) &         (np.linalg.norm(garment_centres[:, :2] - garment_shoulder[:2], axis=1) < NECK_RADIUS_M)
    if neck_cloth.sum() >= 20:
        neck_top = float(np.percentile(garment_centres[neck_cloth, 2], 98))
        base = apply(stand @ shape, body.joint('neck01'))
        head = apply(stand @ shape, body.joint('head'))
        low, high = config['neck_length_range']
        share = config['neck_top_share']
        neck_scale = float(np.clip((neck_top - base[2]) / (share * (head[2] - base[2])), low, high))
        record.update(neck_scale=round(neck_scale, 4), neck_top_m=round(neck_top, 4))
        neck = bodies._descendants(body.parents, body.bone_names.index('neck01'))
        above = bodies._descendants(body.parents, body.bone_names.index('head'))
        stretch_neck = np.linalg.inv(stand) @ about(base, np.diag([1.0, 1.0, neck_scale])) @ stand
        lift = translation([0, 0, (neck_scale - 1.0) * (head[2] - base[2])])
        for bone in neck:
            per_bone[bone] = (lift if bone in above else stretch_neck) @ per_bone[bone]
        rest0 = bodies.skin(body, per_bone)

    # 3. no global centring. Front to back, each trunk slice is centred inside the cloth in step 5 (an average over
    #    the whole trunk put the donor's back outside a busty dress); side to side, step 1 already put the donor on
    #    the garment's shoulder centre.
    rest_bones = np.einsum('ij,bjk->bik', stand, per_bone)
    rest = bodies.skin(body, rest_bones)

    # 3b. arm length: a cuff sits at the wrist, so where a sleeve has a labelled cuff the donor's arm (upper arm,
    #     forearm, hand) stretches along itself about the shoulder joint until its wrist is as far from that joint
    #     as the cuff's centre (clamped). Generator figures' arms vary a lot; without this the sleeve maps too far
    #     down or up every body's arm. Applied to the rest shape like leg length, so un-pose and transfer carry the
    #     sleeve elbow to elbow and wrist to wrist. Without a cuff the arm keeps its length (a plain sleeve end
    #     cannot tell a long sleeve from a three-quarter one).
    record['arm_scale'] = {}
    for side, s in (('left', 'L'), ('right', 'R')):
        cuff = labels == f'cuff.{side}'
        if cuff.sum() < MIN_CUFF_FACES:
            record['arm_scale'][side] = 'kept (no cuff)'
            continue
        shoulder_bone, wrist_bone = body.bone_names.index(f'upperarm01.{s}'), body.bone_names.index(f'wrist.{s}')
        joint = apply(rest_bones[shoulder_bone], body.joint(f'upperarm01.{s}'))
        wrist = apply(rest_bones[wrist_bone], body.joint(f'wrist.{s}'))
        cuff_centre = np.average(garment_centres[cuff], axis=0, weights=garment_areas[cuff])
        wanted_length = np.linalg.norm(cuff_centre - joint)
        low, high = config['arm_length_range']
        arm_scale = float(np.clip(wanted_length / np.linalg.norm(wrist - joint), low, high))
        record['arm_scale'][side] = {'scale': round(arm_scale, 4), 'clamped': bool(
            not np.isclose(arm_scale, wanted_length / np.linalg.norm(wrist - joint)))}
        axis = (wrist - joint) / np.linalg.norm(wrist - joint)
        stretch_arm = about(joint, np.eye(3) + (arm_scale - 1.0) * np.outer(axis, axis))
        for bone in bodies._descendants(body.parents, shoulder_bone):
            rest_bones[bone] = stretch_arm @ rest_bones[bone]
    rest = bodies.skin(body, rest_bones)

    # 4. pose: arms about their shoulder joints to the cuff, legs about their hips towards the ankle, which sits
    #    over the rear quarter of the sole bind measured (heel to toe). The leg centreline's last point is at the
    #    front of the shoe, and a tall boot's shaft top is near the knee, so neither stands in for the ankle.
    def ankle(side):
        foot = (binding.get('feet') or {}).get(side)
        if foot and foot.get('heel') is not None and foot.get('toe') is not None:
            heel, toe = np.asarray(foot['heel']), np.asarray(foot['toe'])
            return heel + ANKLE_ALONG_SOLE * (toe - heel)
        return binding['legs'][side]['ankle_end']

    pose = np.repeat(np.eye(4)[None], bones, axis=0)
    record['pose'] = {}
    joints = {name: apply(rest_bones[body.bone_names.index(name)], body.joint(name)) for name in
              ('upperarm01.L', 'upperarm01.R', 'wrist.L', 'wrist.R', 'upperleg01.L', 'upperleg01.R', 'foot.L', 'foot.R')}
    for region, side, start, end, target in (
            ('arm.left', 'left', 'upperarm01.L', 'wrist.L', (binding['sleeves'].get('left') or {}).get('cuff_end')),
            ('arm.right', 'right', 'upperarm01.R', 'wrist.R', (binding['sleeves'].get('right') or {}).get('cuff_end')),
            ('leg.left', 'left', 'upperleg01.L', 'foot.L', ankle('left')),
            ('leg.right', 'right', 'upperleg01.R', 'foot.R', ankle('right'))):
        if target is None:
            record['pose'][region] = 'kept (no sleeve to follow)'
            continue
        pivot = joints[start]
        turn = rotation_between(joints[end] - pivot, np.asarray(target) - pivot)
        pose[body.bone_region == region] = about(pivot, turn)
        record['pose'][region] = round(float(np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1)))), 2)

    # 4b. elbows follow the sleeve, as legs go hip -> knee -> ankle: a relaxed arm bends at the elbow and sleeves
    #     sag forward, so a rigidly turned arm leaves its elbow at the back edge of the sleeve. The upper arm turns
    #     to the sleeve's centreline at the donor's upper-arm length, then the forearm turns from there to the
    #     cuff (its centre where one is labelled, else the sleeve's end). A sleeve shorter than the upper arm
    #     leaves the arm as it is.
    record['elbow'] = {}
    for side, s in (('left', 'L'), ('right', 'R')):
        sleeve = binding['sleeves'].get(side)
        if not sleeve or len(sleeve.get('points') or []) < 3:
            continue
        index = {n: body.bone_names.index(f'{n}.{s}') for n in ('upperarm01', 'lowerarm01', 'wrist')}
        placed = lambda name: apply(pose[index[name]] @ rest_bones[index[name]], body.joint(f'{name}.{s}'))
        joint, elbow = placed('upperarm01'), placed('lowerarm01')
        points = np.array(sleeve['points'])
        points = points[np.argsort((points - joint) @ (np.asarray(sleeve['cuff_end']) - joint))]
        reach = np.linalg.norm(points - joint, axis=1)
        upper_length = np.linalg.norm(elbow - joint)
        beyond = np.flatnonzero(reach >= upper_length)
        if not len(beyond) or beyond[0] == 0:
            record['elbow'][side] = 'kept (sleeve shorter than the upper arm)'
            continue
        k = beyond[0]
        t = (upper_length - reach[k - 1]) / max(reach[k] - reach[k - 1], 1e-9)
        elbow_target = points[k - 1] + t * (points[k] - points[k - 1])
        upper_turn = rotation_between(elbow - joint, elbow_target - joint)
        arm = list(bodies._descendants(body.parents, index['upperarm01']))
        pose[arm] = np.einsum('ij,bjk->bik', about(joint, upper_turn), pose[arm])
        elbow, wrist = placed('lowerarm01'), placed('wrist')
        cuff = labels == f'cuff.{side}'
        wrist_target = np.average(garment_centres[cuff], axis=0, weights=garment_areas[cuff]) \
            if cuff.sum() >= MIN_CUFF_FACES else np.asarray(sleeve['cuff_end'])
        fore_turn = rotation_between(wrist - elbow, wrist_target - elbow)
        forearm = list(bodies._descendants(body.parents, index['lowerarm01']))
        pose[forearm] = np.einsum('ij,bjk->bik', about(elbow, fore_turn), pose[forearm])
        angle = lambda r: round(float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))), 2)
        record['elbow'][side] = {'upper_arm_turn_deg': angle(upper_turn), 'forearm_turn_deg': angle(fore_turn)}
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
        if region == 'trunk':
            moved_posed, moved_rest, offset_at, summary[region] = trunk_girth(
                posed, rest, body.vertex_region == region, garment_centres[cloth], config, step)
            move_posed += weight[:, None] * moved_posed
            move_rest += weight[:, None] * moved_rest
            # the front-to-back offset carries the whole body, so limbs stay attached: the trunk by height (the
            # legs take its lowest slice, the head its top one), each arm by the slice at its shoulder
            shift = np.zeros(len(posed))
            for part, part_weight in body.region_weight.items():
                heights = np.full(len(posed), posed[body.vertex_region == part, 2].max()) \
                    if part.startswith('arm.') else posed[:, 2]
                shift += part_weight * offset_at(heights)
            move_posed[:, 1] += shift
            move_rest[:, 1] += shift
            continue
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


SECTOR_MIN_FACES = 4             # a trunk slice's front, back or sides count as covered with this many cloth faces
SECTOR_HALF_ANGLE_DEG = 20       # front, back and sides are measured within this angle of -y, +y and +-x


def sectors(offsets):
    """Masks of offsets (from a slice centre, in xy) within the narrow front (-y), back (+y) and side (+-x) sectors."""
    angle = np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0]))
    off = lambda a: np.abs((angle - a + 180) % 360 - 180)
    return {'front': off(-90) <= SECTOR_HALF_ANGLE_DEG, 'back': off(90) <= SECTOR_HALF_ANGLE_DEG,
            'sides': (off(0) <= SECTOR_HALF_ANGLE_DEG) | (off(180) <= SECTOR_HALF_ANGLE_DEG)}


def trunk_girth(posed, rest, inside, cloth_centres, config, step):
    """Trunk girth, slice by slice along z: depth (y) and width (x) scale separately to sit just inside the
    innermost cloth in front, behind and at the sides, and the slice moves front to back so its front and back
    margins match. A side with no cloth (a back opening, a keyhole) gives nothing: depth and offset come from the
    nearest slices covered front and back, width from slices covered at the sides, and with none at all the
    radial factor stands in. Returns the scaling moves (posed, rest; unweighted), the front-to-back offset as a
    function of height, and a summary."""
    low, high = config['girth_range']
    limit = config['trunk_offset_max_m']
    z = posed[:, 2]
    starts, centres_posed, centres_rest, depth, width, offset, radial = [], [], [], [], [], [], []
    for start in np.arange(z[inside].min(), z[inside].max(), step):
        section = inside & (z >= start) & (z < start + step)
        if section.sum() < 6:
            continue
        c = posed[section].mean(axis=0)
        rel = posed[section] - c
        ring = (cloth_centres[:, 2] >= start) & (cloth_centres[:, 2] < start + step)
        around = cloth_centres[ring] - c
        body_sector, cloth_sector = sectors(rel), sectors(around)
        # distance from the slice centre in each narrow sector: the donor's outermost surface, the innermost cloth
        reach = {name: (np.percentile(np.linalg.norm(rel[body_sector[name], :2], axis=1), 90) if body_sector[name].sum() >= 3
                        else np.nan,
                        0.97 * np.percentile(np.linalg.norm(around[cloth_sector[name], :2], axis=1), 10)
                        if cloth_sector[name].sum() >= SECTOR_MIN_FACES else np.nan)
                 for name in ('front', 'back', 'sides')}
        (front_d, cloth_front), (back_d, cloth_back), (side_d, cloth_side) = \
            reach['front'], reach['back'], reach['sides']
        d, w, o, r = np.nan, np.nan, np.nan, 1.0
        if not np.isnan([front_d, back_d, cloth_front, cloth_back]).any():
            d = float(np.clip((cloth_front + cloth_back) / (front_d + back_d), low, high))
            o = float(np.clip(((cloth_back - d * back_d) - (cloth_front - d * front_d)) / 2, -limit, limit))
        if not np.isnan([side_d, cloth_side]).any():
            w = float(np.clip(cloth_side / side_d, low, high))
        if len(around) >= 15:                                          # the radial factor, as for the limbs
            r_body = max(np.percentile(np.linalg.norm(rel[:, :2], axis=1), 90), 1e-4)
            r = float(np.clip(0.97 * np.percentile(np.linalg.norm(around[:, :2], axis=1), 10) / r_body, low, high))
        starts.append(start)
        centres_posed.append(c)
        centres_rest.append(rest[section].mean(axis=0))
        depth.append(d)
        width.append(w)
        offset.append(o)
        radial.append(r)
    if not starts:
        return np.zeros_like(posed), np.zeros_like(rest), {}
    starts, radial = np.array(starts), np.array(radial)

    def filled(values, fallback):
        values = np.array(values, dtype=float)
        known = ~np.isnan(values)
        if not known.any():
            return np.asarray(fallback, dtype=float), len(values)
        return np.interp(starts, starts[known], values[known]), int((~known).sum())

    def median3(values):
        return np.array([np.median(values[max(0, i - 1):i + 2]) for i in range(len(values))])

    depth, filled_depth = filled(depth, radial)
    offset, _ = filled(offset, np.zeros(len(starts)))
    width, filled_width = filled(width, radial)
    depth, width, offset = median3(depth), median3(width), median3(offset)
    index = np.clip(np.searchsorted(starts, z, side='right') - 1, 0, len(starts) - 1)
    moves = []
    for points, centres in ((posed, np.array(centres_posed)), (rest, np.array(centres_rest))):
        rel = points - centres[index]
        move = np.zeros_like(points)
        move[:, 0] = (width[index] - 1) * rel[:, 0]
        move[:, 1] = (depth[index] - 1) * rel[:, 1]
        moves.append(move)

    def offset_at(heights):
        return offset[np.clip(np.searchsorted(starts, heights, side='right') - 1, 0, len(starts) - 1)]

    summary = {'sections': len(starts), 'depth': [round(float(depth.min()), 3), round(float(depth.max()), 3)],
               'width': [round(float(width.min()), 3), round(float(width.max()), 3)],
               'offset_cm': [round(float(offset.min()) * 100, 1), round(float(offset.max()) * 100, 1)],
               'filled_from_neighbours': {'depth': filled_depth, 'width': filled_width},
               'clamped': int(np.sum(np.isclose(depth, low) | np.isclose(depth, high)) +
                              np.sum(np.isclose(width, low) | np.isclose(width, high)))}
    return moves[0], moves[1], offset_at, summary
