"""Stage 6, fit (spec §5.6), Steps A-C so far. Sealed: numpy/scipy only, no network, no other programs.

A. donor: body_01_medium reshaped and posed inside the outfit (fittedperfection/donor.py).
B. un-pose: every cage point moves into the donor's rest pose by a soft blend of the
   inverse pose of its nearest same-region donor vertices. Near label seams the blended
   transforms are interpolated harmonically, so the field changes smoothly from one
   region to the next. Rigid groups (hardware, whole shoes) move with one transform each.
C. transfer: donor and targets share one mesh, so each body vertex has an exact move
   (target - donor). Each cage point moves by the Gaussian-weighted average move of its
   k nearest same-region donor vertices, sigma = max(2 d1, 0.03 height). No per-anchor
   rotations. Rigid groups are placed as exact copies (rigid fit to the field, rotation
   clamped). UV-split twins share one result, so seams cannot open.

D. wrap (fittedperfection/wrap.py): one coordinated solve that brings the cloth close to
   each body's skin (keeping its relief and part of its looseness), keeps the local shape
   of the Step C result, holds contact outside the skin plus ease, and anchors rigid groups.

E. shoes (fittedperfection/shoes.py), placed before D: each shoe scaled to the body's foot and
   moved onto it with its footbed on the sole; only the part below the ankle is a rigid anchor
   (boot shafts are wrapped). After D the fit is raised by the sole (root_offset_m) and the body
   faces the shoes cover are listed as hidden (ghost removal); BODY.glb is the body as shown.

G. full resolution (fittedperfection/fullres.py): GARMENT.glb bound to the source cage by label,
   rebuilt on each fitted cage with a smooth rotation field (rigid groups as exact copies).
F. skinning: weights from each body's rig, top 4, rigid groups shared. OUTFIT.glb is the
   full-resolution outfit skinned to the body's skeleton, BODY.glb the body as shown (covered
   faces hidden); both carry the root offset. DRAFT.glb stays as the cage preview. Every output is a preview draft: fit_reviewed
and game_ready are false.
"""
import json

import numpy as np
from scipy.spatial import cKDTree

from . import body as bodies, config, donor, fullres, meshes, settings, shoes, wrap
from .glbwrite import write_mesh
from .gltf import read_textured_mesh

CODE = ('fit.py', 'wrap.py', 'shoes.py', 'fullres.py', 'donor.py', 'body.py', 'bind.py', 'label.py', 'meshes.py', 'gltf.py', 'glbwrite.py')
REGION_OF = {'sleeve': 'arm', 'cuff': 'arm', 'pant_leg': 'leg', 'footwear': 'leg', 'pants': 'hips',
             'hanging': 'hanging'}
RIGID_CLASSES = ('hardware',)


def label_region(name):
    """Garment label -> body region: sleeve/cuff -> arm.side, legwear/footwear -> leg.side, pants -> hips,
    hanging -> hanging (led by the pelvis), everything else (torso, collar, hood, waistband, attachments,
    hardware) -> trunk."""
    base, _, side = name.partition('.')
    kind = REGION_OF.get(base, 'trunk')
    return f'{kind}.{side}' if kind in ('arm', 'leg') and side else kind


def point_regions(labels, triangles, count):
    """Region of every cage vertex: the most common region of the faces around it (skin and 'none' left blank;
    the fit gives those the region of the nearest donor vertex)."""
    face_region = np.array([label_region(n) if n not in ('skin', 'none') else '' for n in labels], dtype=object)
    votes = {}
    for corner in range(3):
        for vertex, region in zip(triangles[:, corner], face_region):
            if region:
                votes.setdefault(vertex, {}).setdefault(region, 0)
                votes[vertex][region] += 1
    out = np.array([''] * count, dtype=object)
    for vertex, tally in votes.items():
        out[vertex] = max(tally, key=tally.get)
    return out


def donor_region_sets(body, positions, crotch, band=(0.15, 0.10), overlap=0.05):
    """Donor vertex indices per region. A region includes every vertex with at least `overlap` of its skin
    weight in that region, so neighbouring regions share the body's own blend zones (shoulders, hips) and
    garment seams that do not sit exactly on the body's boundaries still move smoothly. Left and right
    never mix. hips = trunk + legs around the crotch, so trouser seats do not split."""
    sets = {r: np.flatnonzero(body.region_weight[r] >= overlap)
            for r in ('trunk', 'arm.left', 'arm.right', 'leg.left', 'leg.right')}
    near = (positions[:, 2] > crotch - band[0]) & (positions[:, 2] < crotch + band[1])
    sets['hips'] = np.flatnonzero(near & ((body.region_weight['trunk'] >= overlap) |
                                          (body.region_weight['leg.left'] >= overlap) |
                                          (body.region_weight['leg.right'] >= overlap)))
    # Hanging cloth (tails, dangling straps) follows the pelvis as one piece, not whichever thigh is nearer.
    sets['hanging'] = np.flatnonzero(body.region_weight['trunk'] >= 0.5)
    sets[''] = np.arange(len(positions))                                    # skin / unknown: any region
    return sets


def gaussian_blend(points, regions, anchors, sets, k, sigma_of_nearest, sigma_floor):
    """Per point: (indices of its k nearest same-region anchors, normalised Gaussian weights)."""
    count = len(points)
    index = np.zeros((count, k), dtype=np.int64)
    weight = np.zeros((count, k))
    for region in np.unique(regions):
        chosen = np.flatnonzero(regions == region)
        members = sets.get(region, sets[''])
        kk = min(k, len(members))
        distance, nearest = cKDTree(anchors[members]).query(points[chosen], k=kk)
        distance, nearest = distance.reshape(len(chosen), kk), nearest.reshape(len(chosen), kk)
        sigma = np.maximum(sigma_of_nearest * distance[:, :1], sigma_floor)
        w = np.exp(-0.5 * (distance / sigma) ** 2)
        index[chosen, :kk], weight[chosen, :kk] = members[nearest], w / w.sum(axis=1, keepdims=True)
    return index, weight


def rigid_fit(source, moved, max_rotation_degrees=None):
    """Rotation R and translation t moving `source` onto `moved` (least squares, no scaling),
    with the rotation angle optionally clamped."""
    a, b = source.mean(axis=0), moved.mean(axis=0)
    u, _, vt = np.linalg.svd((source - a).T @ (moved - b))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1, 1, d]) @ u.T
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    if max_rotation_degrees is not None and angle > max_rotation_degrees:
        axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
        axis /= np.linalg.norm(axis)
        rotation = donor.rotation_between_axis_angle(axis, np.radians(max_rotation_degrees))
    return rotation, b - rotation @ a, float(angle)


def rigid_groups(labels, triangles, graph):
    """Pieces that must keep their exact shape (spec §5.6 C): each connected hardware piece, and each whole
    shoe (boot shafts get their own treatment in the shoes step); pieces that touch merge into one."""
    vertex_sets = []
    piece, count = meshes.parts(graph, np.isin(labels, RIGID_CLASSES))
    for k in range(count):
        vertex_sets.append(('hardware', np.unique(triangles[piece == k])))
    for side in ('left', 'right'):
        vertex_sets.append((f'shoe.{side}', np.unique(triangles[labels == f'footwear.{side}'])))
    # A hardware piece touching a shoe is part of that shoe (a buckle on a strap): two separate rigid fits
    # would tear the faces joining them. It joins only the shoe it shares most points with, and the two
    # shoes never merge, even where the source mesh joins them.
    shoes = [[name, vertices] for name, vertices in vertex_sets if name.startswith('shoe') and len(vertices) >= 4]
    merged = []
    for name, vertices in vertex_sets:
        if name.startswith('shoe') or len(vertices) < 4:
            continue
        shared = [np.intersect1d(shoe[1], vertices).size for shoe in shoes]
        if shared and max(shared) > 0:
            best = int(np.argmax(shared))
            shoes[best][1] = np.union1d(shoes[best][1], vertices)
        else:
            merged.append((name, vertices))
    for shoe_a, shoe_b in ((0, 1),) if len(shoes) == 2 else ():
        common = np.intersect1d(shoes[shoe_a][1], shoes[shoe_b][1])         # joined in the source: split the seam
        shoes[shoe_b][1] = np.setdiff1d(shoes[shoe_b][1], common)
    merged += [(name, vertices) for name, vertices in shoes]
    return [{'name': name, 'vertices': vertices} for name, vertices in merged]


def point_adjacency(triangles, count):
    """Row-normalised vertex adjacency of the welded cage (for blending fields across label seams)."""
    from scipy.sparse import coo_matrix, diags
    i = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    j = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    matrix = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(count, count)).tocsr()
    matrix.data[:] = 1.0
    return diags(1.0 / np.maximum(np.asarray(matrix.sum(axis=1)).ravel(), 1)) @ matrix


def seam_band(adjacency, regions, rings):
    """Points within `rings` edges of a point of another region: where the field must change smoothly."""
    codes = np.unique(regions, return_inverse=True)[1].astype(float)
    neighbour_mean = adjacency @ codes
    band = np.abs(neighbour_mean - codes) > 1e-9
    for _ in range(rings):
        band = band | ((adjacency @ band.astype(float)) > 0)
    return band


def blend_across_seams(values, adjacency, band, iterations):
    """Harmonic blend of a per-point field inside the seam band, with the values outside held fixed.
    The field (un-pose transforms, transfer moves) then changes smoothly from one label region to the next
    instead of jumping where a sleeve label meets a bodice label."""
    values = values.reshape(len(values), -1).copy()
    inside = np.flatnonzero(band)
    for _ in range(iterations):
        values[inside] = (adjacency @ values)[inside]
    return values


def unpose(points, regions, built, config_fit, adjacency=None, band=None):
    """Step B: posed (source) cage points -> the donor's rest pose."""
    body = built['body']
    inverse = np.linalg.inv(bodies.vertex_matrices(body, built['pose']))
    sets = donor_region_sets(body, built['posed'], built['crotch_posed'])
    height = float(built['posed'][:, 2].max() - built['posed'][:, 2].min())
    index, weight = gaussian_blend(points, regions, built['posed'], sets, config_fit['unpose_neighbours'],
                                   config_fit['transfer_sigma_of_nearest'], config_fit['transfer_sigma_of_height'] * height)
    blended = np.einsum('pk,pkij->pij', weight, inverse[index])
    if band is not None and band.any():
        blended = blend_across_seams(blended, adjacency, band, config_fit['seam_blend_iterations']).reshape(-1, 4, 4)
    return np.einsum('pij,pj->pi', blended[:, :3, :3], points) + blended[:, :3, 3]


def transfer(points, regions, donor_rest, target_rest, body, crotch, config_fit, adjacency=None, band=None):
    """Step C: donor-rest cage points -> a target body, by the smooth field of exact body vertex moves."""
    sets = donor_region_sets(body, donor_rest, crotch)
    height = float(donor_rest[:, 2].max() - donor_rest[:, 2].min())
    index, weight = gaussian_blend(points, regions, donor_rest, sets, config_fit['transfer_neighbours'],
                                   config_fit['transfer_sigma_of_nearest'], config_fit['transfer_sigma_of_height'] * height)
    moves = np.einsum('pk,pkj->pj', weight, (target_rest - donor_rest)[index])
    if band is not None and band.any():
        moves = blend_across_seams(moves, adjacency, band, config_fit['seam_blend_iterations'])
    return points + moves


def place_rigid(source, moved, groups, max_rotation=None):
    """Replace each rigid group's moved points by an exact copy of its source shape fitted to them."""
    out = moved.copy()
    record = []
    for group in groups:
        v = group['vertices']
        rotation, shift, angle = rigid_fit(source[v], moved[v], max_rotation)
        out[v] = source[v] @ rotation.T + shift
        record.append({'name': group['name'], 'points': int(len(v)), 'field_rotation_deg': round(angle, 2)})
    return out, record


def stretch(source, fitted, triangles, keep):
    """Faces whose longest edge grew more than 3x and 10x versus the source (spec §5.7), among faces kept."""
    def edges(p):
        a, b, c = p[triangles[:, 0]], p[triangles[:, 1]], p[triangles[:, 2]]
        return np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], 1)
    ratio = (edges(fitted) / np.maximum(edges(source), 1e-9)).max(axis=1)[keep]
    return {'faces': int(keep.sum()), 'over_3x': int((ratio > 3).sum()), 'over_10x': int((ratio > 10).sum()),
            'p99_ratio': round(float(np.percentile(ratio, 99)), 3)}


def skin_regions(target, regions, weights=None):
    """(region, point indices, skin faces) for the wrap: each cage point measures against the skin of its own
    region (arm, leg, trunk), so baggy inner legs never measure against the other leg. Trouser seats (hips)
    measure against the trunk and both legs; free-hanging and unknown points against the whole skin.
    weights (Step F, the points' positions): hanging cloth above the knees takes its weights from the
    pelvis-led skin, so tails follow the hips rather than whichever thigh is nearest; below the knees
    (straps at the ankle) it follows the nearest skin like everything else."""
    crotch = bodies.landmarks(target)['crotch_height_m']
    sets = donor_region_sets(target, target.positions, crotch)
    sets['hips'] = np.unique(np.concatenate([sets['trunk'], sets['leg.left'], sets['leg.right']]))
    if weights is not None:
        knee = min(target.joint('lowerleg01.L')[2], target.joint('lowerleg01.R')[2])
        regions = np.where((regions == 'hanging') & (weights[:, 2] < knee), '', regions).astype(object)
    out = []
    for region in np.unique(regions):
        chosen = np.flatnonzero(regions == region)
        if region == '' or region not in sets or (region == 'hanging' and weights is None):
            out.append(('all', chosen, np.arange(len(target.triangles))))
            continue
        member = np.zeros(len(target.positions), dtype=bool)
        member[sets[region]] = True
        out.append((region, chosen, np.flatnonzero(member[target.triangles].any(axis=1))))
    return out


def bare_skin(target):
    """Skin faces that are bare by nature (head, hands): cloth near them is not wrapped onto them."""
    bones = set()
    for root in ('head', 'wrist.L', 'wrist.R'):
        bones |= bodies._descendants(target.parents, target.bone_names.index(root))
    weight = (target.weights * np.isin(target.joints, list(bones))).sum(axis=1)
    return weight[target.triangles].mean(axis=1) >= 0.5


def run(ctx, out):
    config_fit = settings.section('fit')
    cage = read_textured_mesh(ctx.output_of('intake')/'CAGE.glb')
    labels = np.load(ctx.output_of('label')/'labels.npz')['labels'].astype(str)
    binding = json.loads((ctx.output_of('bind')/'binding.json').read_text(encoding='utf-8'))
    centres = cage.positions[cage.triangles].mean(axis=1)
    _, areas = meshes.face_normals(cage.positions, cage.triangles)

    # one result per welded vertex, so UV-split twins can never separate
    welded = meshes.weld(cage.positions)
    first = np.unique(welded, return_index=True)[1]
    unique_points = cage.positions[first]
    unique_triangles = welded[cage.triangles]
    regions = point_regions(labels, unique_triangles, len(unique_points))
    graph = meshes.face_graph(cage.positions, cage.triangles)

    built = donor.build(binding, labels, centres, areas, config_fit)
    body = built['body']
    # Points with no garment region (mannequin skin, unknown) take the region of the donor vertex nearest
    # them, so neck skin belongs to the trunk and never follows a raised arm.
    blank = regions == ''
    if blank.any():
        nearest = cKDTree(built['posed']).query(unique_points[blank])[1]
        regions[blank] = body.vertex_region[nearest]
    built['crotch_posed'] = binding['crotch']['height_m']
    donor_crotch = float(bodies.landmarks(body, built['rest'])['crotch_height_m'])
    groups = rigid_groups(labels, unique_triangles, graph)
    adjacency = point_adjacency(unique_triangles, len(unique_points))
    band = seam_band(adjacency, regions, config_fit['seam_blend_rings'])

    rest_points = unpose(unique_points, regions, built, config_fit, adjacency, band)
    rest_points, rigid_unpose = place_rigid(unique_points, rest_points, groups)

    # Step E, measured once: each shoe's sole thickness, the donor's foot inside it, and the shaft above the ankle
    # (cloth for the wrap); only the shoe below the ankle line is a rigid anchor
    shoe_parts = {}
    for group in groups:
        if not group['name'].startswith('shoe.'):
            continue
        side = group['name'].split('.')[1]
        bone = body.bone_names.index(shoes.SIDES[side])
        donor_foot = shoes.foot_points(body, built['posed'], side)
        ankle = donor.apply(built['posed_bones'][bone], body.joint(shoes.SIDES[side]))[2] - donor_foot[:, 2].min()
        thickness, found = shoes.sole_thickness(unique_points[group['vertices']], config_fit['shoe_sole_search_m'])
        # the ankle inside the shoe stands on the sole; only a real boot shaft (well above it) is cloth
        line = unique_points[group['vertices'], 2].min() + thickness + ankle + config_fit['shoe_shaft_above_ankle_m']
        tall = unique_points[group['vertices'], 2].max() - line >= config_fit['shoe_shaft_min_m']
        shaft = shoes.shaft(group['vertices'], unique_points, line) if tall else np.zeros(0, dtype=np.int64)
        shoe_parts[side] = {'vertices': group['vertices'], 'thickness': thickness, 'edge_found': found,
                            'shaft': shaft, 'line': line, 'donor_foot': donor_foot,
                            'faces': unique_triangles[labels == f'footwear.{side}']}
    # a pair stands on one sole: the thicker measured edge (or the default when neither was found)
    measured = [p['thickness'] for p in shoe_parts.values() if p['edge_found']]
    for part in shoe_parts.values():
        part['thickness'] = max(measured) if measured else part['thickness']
    keep = labels != 'skin'
    # printed designs (label stage) are kept stiff in the wrap
    label_file = np.load(ctx.output_of('label')/'labels.npz')
    printed = np.unique(unique_triangles[label_file['prints']]) if 'prints' in label_file.files         else np.zeros(0, dtype=np.int64)
    looseness_kept = wrap.point_keep(labels, unique_triangles, len(unique_points), config_fit['wrap_keep'])
    rigid = np.unique(np.concatenate([g['vertices'] for g in groups])) if groups else np.zeros(0, dtype=np.int64)
    rigid = np.setdiff1d(rigid, np.concatenate([p['shaft'] for p in shoe_parts.values()]) if shoe_parts else [])
    cloth = np.unique(unique_triangles[keep & ~np.char.startswith(labels, 'footwear')])

    # Step G1, once: bind the full-resolution outfit to the source cage (one result per welded vertex: twins agree)
    garment = read_textured_mesh(ctx.output_of('intake')/'GARMENT.glb')
    full_welded = meshes.weld(garment.positions)
    full_points = garment.positions[np.unique(full_welded, return_index=True)[1]]
    full_face_labels = np.load(ctx.output_of('label')/'fullres-labels.npz')['labels'].astype(str)  # label stage
    full_binding = fullres.bind(full_points, fullres.vertex_labels(full_face_labels, garment.triangles, full_welded),
                                unique_points, unique_triangles, labels)
    full_regions = regions[unique_triangles[full_binding['face'], 0]]
    full_keep = full_face_labels != 'skin'
    shafts = np.concatenate([p['shaft'] for p in shoe_parts.values()]) if shoe_parts else np.zeros(0, dtype=np.int64)
    rigid_parts = [np.setdiff1d(g['vertices'], shafts) for g in groups]
    summary = {}
    for target in config.BODIES:
        folder = out/target
        folder.mkdir()
        target_body = bodies.load(target)
        target_rest = target_body.positions
        fitted = transfer(rest_points, regions, built['rest'], target_rest, body, donor_crotch, config_fit,
                          adjacency, band)
        field_placed = fitted.copy()                                  # Step C before rigid copies (boot shafts)
        fitted, rigid_transfer = place_rigid(rest_points, fitted, groups, config_fit['rigid_max_rotation_degrees'])
        shoe_record = {}
        for side, part in shoe_parts.items():                         # Step E: each shoe onto this body's foot
            move, shoe_record[side] = shoes.placement(unique_points[part['vertices']], part['donor_foot'],
                                                      shoes.foot_points(target_body, target_rest, side),
                                                      part['thickness'],
                                                      config_fit['foot_length_of_height'] * binding['height_m'])
            # A boot shaft is cloth on the leg: just above the ankle it follows the shoe, higher up it eases into
            # the leg's own placement (Step C), so a knee-high boot stays knee-high instead of shrinking with the
            # foot and uncovering what its rim hid.
            shaft = part['shaft']
            ease_in = np.clip((unique_points[shaft, 2] - part['line']) / config_fit['shoe_shaft_blend_m'], 0.0, 1.0)
            ease_in = ease_in * ease_in * (3 - 2 * ease_in)
            fitted[part['vertices']] = move(unique_points[part['vertices']])
            fitted[shaft] = (1 - ease_in)[:, None] * fitted[shaft] + ease_in[:, None] * field_placed[shaft]
            shoe_record[side].update(edge_found=part['edge_found'], shaft_points=int(len(part['shaft'])))
        skin = wrap.Skin(target_rest, target_body.triangles, bare_skin(target_body))
        before = {'offsets': wrap.offsets(fitted, skin, cloth),
                  'stretch': stretch(cage.positions, fitted[welded], cage.triangles, keep)}
        transferred = fitted
        fitted, wrap_record = wrap.solve(fitted, unique_triangles, looseness_kept, skin,
                                         skin_regions(target_body, regions), rigid, config_fit, target_body.height,
                                         lambda field: blend_across_seams(field, adjacency, band,
                                                                          config_fit['seam_blend_iterations']),
                                         printed)
        wrap_record['before'] = before
        wrap_record['after'] = {'offsets': wrap.offsets(fitted, skin, cloth)}
        # raise everything by the sole so the shoes stand on the floor (the body is offset, never changed), then
        # hide the body faces the shoes cover or that poke through them
        lift = max([-float(fitted[p['vertices'], 2].min()) for p in shoe_parts.values()], default=0.0)
        fitted = fitted + [0.0, 0.0, lift]
        body_lifted = target_rest + [0.0, 0.0, lift]
        hidden = np.zeros(0, dtype=np.int64)
        for side, part in shoe_parts.items():
            own = np.unique(part['faces'])
            covered = shoes.hidden_faces(body_lifted, target_body.triangles, fitted[own],
                                         np.searchsorted(own, part['faces']), config_fit['ghost_grid_m'])
            shoe_record[side]['hidden_body_faces'] = int(len(covered))
            hidden = np.union1d(hidden, covered)
        np.save(folder/'hidden_body_faces.npy', hidden)
        skeleton = config.body_file(target)
        body_joints, body_weights = fullres.top_influences(np.arange(len(target_rest)), target_body.joints,
                                                           target_body.weights, len(target_rest),
                                                           config_fit['max_influences'])
        write_mesh(folder/'BODY.glb', target_rest, np.delete(target_body.triangles, hidden, axis=0),
                   colour=(0.82, 0.63, 0.51, 1.0), name=f'{target}-shown',
                   skin={'joints': body_joints, 'weights': body_weights, 'skeleton': skeleton, 'root_offset_m': lift})

        # Steps G2 and F: rebuild the full resolution on the fitted cage, then skin it to this body's rig (weights
        # measured in the body's own frame; the root offset carries the lift)
        rotation = fullres.rotation_field(unique_points, fitted, unique_triangles,
                                          config_fit['rotation_smoothing_iterations'])
        local_scale = fullres.scale_field(unique_points, fitted, unique_triangles, config_fit['fullres_scale_range'])
        full_fit, full_group = fullres.rebuild(full_binding, full_points, unique_points, fitted, unique_triangles,
                                               rotation, local_scale, rigid_parts)
        full_fit -= [0.0, 0.0, lift]
        joints, weights = fullres.skin_weights(full_fit, skin_regions(target_body, full_regions, weights=full_fit), skin,
                                               target_body.joints, target_body.weights, full_group,
                                               config_fit['max_influences'])
        write_mesh(folder/'OUTFIT.glb', full_fit[full_welded], garment.triangles[full_keep], uvs=garment.uvs,
                   texture_png=garment.texture_png, name=f'{target}-outfit',
                   skin={'joints': joints[full_welded], 'weights': weights[full_welded], 'skeleton': skeleton,
                         'root_offset_m': lift})
        full_record = {'vertices': int(len(garment.positions)), 'faces_kept': int(full_keep.sum()),
                       'binding_distance_cm': {'p50': round(float(np.percentile(full_binding['distance'], 50)) * 100, 2),
                                               'p99': round(float(np.percentile(full_binding['distance'], 99)) * 100, 2)},
                       'stretch_vs_source': stretch(garment.positions, full_fit[full_welded], garment.triangles,
                                                    full_keep),
                       'influences_used_mean': round(float((weights > 1e-6).sum(axis=1).mean()), 2)}
        positions = fitted[welded]
        stretched = stretch(cage.positions, positions, cage.triangles, keep)
        np.savez_compressed(folder/'cage-fit.npz', positions=positions.astype(np.float32),
                            transferred_positions=transferred[welded].astype(np.float32),
                            rest_positions=rest_points[welded].astype(np.float32), regions=regions[welded].astype(str))
        write_mesh(folder/'DRAFT.glb', positions, cage.triangles[keep], uvs=cage.uvs, texture_png=cage.texture_png,
                   name=f'{target}-draft')
        report = {'status': 'preview_steps_A-G', 'body': target, 'fit_reviewed': False, 'game_ready': False,
                  'note': 'OUTFIT.glb is the full-resolution outfit skinned to this body; not reviewed',
                  'root_offset_m': round(float(lift), 4), 'printed_points': int(len(printed)), 'shoes': shoe_record, 'full_resolution': full_record,
                  'hidden_body_faces': int(len(hidden)),
                  'settings': json.loads(json.dumps(dict(config_fit), default=list)),
                  'donor': built['record'], 'rigid_groups': {'unpose': rigid_unpose, 'transfer': rigid_transfer},
                  'wrap': wrap_record, 'stretch_vs_source': stretched, 'gates': {}}
        (folder/'fit.report.json').write_text(json.dumps(report, indent=1, default=float), encoding='utf-8', newline='\n')
        summary[target] = {'stretch': stretched, 'offsets': wrap_record['after']['offsets']}
    return {'donor': {k: built['record'][k] for k in ('scale', 'leg_scale', 'pose')}, 'stretch': summary,
            'rigid_groups': [g['name'] for g in groups]}
