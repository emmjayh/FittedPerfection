"""Stage 5, bind (spec §5.5): measure where the body must be inside this outfit.

From the cage and its labels (canonical frame: Z up, front -Y, own left +X):
- sleeve centrelines: the largest connected sleeve piece per side (fragments under
  3% of the torso area ignored; a cuff counts only if it touches that piece's end),
  section centroids along its main axis, tolerant of gaps;
- leg centrelines: per side, legwear + bare leg + footwear below the crotch, section
  centroids from the crotch to the floor;
- shoulders: the sleeve roots; a sleeveless top uses shoulder_top landmarks;
- crotch: the branch height found while labelling, checked against the prior
  0.59 x shoulder height +- 0.07; the prior is used (and said so) when it is missing
  or outside;
- feet: from the footwear only; toe direction from the sole's long axis, signed by
  toe reach vs heel reach from the lower-shin leg centre. This confirms a provisional
  intake facing; when it cannot, the owner's recorded answer does (confirm.py), or bind
  stops and asks;
- knees: the narrowest leg section near the expected knee height, accepted only when
  that minimum is clear and both legs agree (cuffs and frills can fake one); otherwise
  none. The method is recorded;
- legwear length class (from SAM's length phrases) -> hem fraction.
"""
import json

import numpy as np
from PIL import Image, ImageDraw

from . import cameras, confirm, meshes, settings
from .gltf import read_textured_mesh

CODE = ('bind.py', 'meshes.py', 'cameras.py', 'gltf.py', 'confirm.py')


class BindError(ValueError):
    pass


class FacingUnconfirmed(BindError):
    """The facing is provisional and nothing confirms it: the owner is asked."""


class FacingRejected(BindError):
    """The owner said the front view shows the outfit's back."""


def _r(value, places=4):
    return None if value is None else (np.round(np.asarray(value, dtype=float), places).tolist())


def principal_axis(points, weights):
    centre = np.average(points, axis=0, weights=weights)
    covariance = np.cov((points - centre).T, aweights=weights)
    values, vectors = np.linalg.eigh(covariance)
    return centre, vectors[:, np.argmax(values)]


def sections(points, weights, origin, axis, step):
    """Weighted centroids of slabs of `step` along `axis` from `origin`; empty slabs are skipped (gaps)."""
    t = (points - origin) @ axis
    out = []
    for start in np.arange(t.min(), t.max(), step):
        inside = (t >= start) & (t < start + step)
        if weights[inside].sum() > 0:
            out.append(np.average(points[inside], axis=0, weights=weights[inside]))
    return np.array(out)


def sleeve_chain(side, labels, centres, areas, graph, torso_area, config):
    sleeve = labels == f'sleeve.{side}'
    if not sleeve.any():
        return None
    piece, count = meshes.parts(graph, sleeve)
    sizes = np.bincount(piece[sleeve], weights=areas[sleeve])
    largest = int(np.argmax(sizes))
    if sizes[largest] < config['min_sleeve_fragment_of_torso'] * torso_area:
        return None
    body = piece == largest
    cuff = labels == f'cuff.{side}'
    if cuff.any():                                       # a cuff counts only if it touches this sleeve
        a, b = graph.pairs[:, 0], graph.pairs[:, 1]
        cuff_piece, cuff_count = meshes.parts(graph, cuff)
        touching = set(cuff_piece[a[body[b] & cuff[a]]]) | set(cuff_piece[b[body[a] & cuff[b]]])
        for k in touching - {-1}:
            body |= cuff_piece == k
    points, weights = centres[body], areas[body]
    origin, axis = principal_axis(points, weights)
    if axis[0] * (1 if side == 'left' else -1) < 0:      # point outwards from the body
        axis = -axis
    chain = sections(points, weights, origin, axis, config['section_step'])
    return {'points': _r(chain), 'root': _r(chain[0]), 'cuff_end': _r(chain[-1]),
            'axis': _r(axis), 'length_m': round(float(np.linalg.norm(chain[-1] - chain[0])), 4),
            'faces': int(body.sum()), 'fragments_ignored': int(count - 1)}


def leg_chain(side, labels, centres, areas, crotch, config):
    sign = 1 if side == 'left' else -1
    evidence = np.isin(labels, (f'pant_leg.{side}', f'footwear.{side}')) | \
        ((labels == 'skin') & (np.sign(centres[:, 0]) == sign))
    evidence &= centres[:, 2] < crotch
    if evidence.sum() < 20:
        return None
    points, weights = centres[evidence], areas[evidence]
    chain = sections(points, weights, np.array([0.0, 0.0, crotch]), np.array([0.0, 0.0, -1.0]), config['section_step'])
    return {'points': _r(chain), 'hip': _r(chain[0]), 'ankle_end': _r(chain[-1]), 'faces': int(evidence.sum())}


def knee(side, labels, centres, areas, crotch, height, config):
    """Narrowest leg-tube section near the expected knee height, if that minimum is clear."""
    sign = 1 if side == 'left' else -1
    tube = (np.isin(labels, (f'pant_leg.{side}',)) | ((labels == 'skin') & (np.sign(centres[:, 0]) == sign)))
    expected = config['knee_of_height'] * height
    window = config['knee_window_of_height'] * height
    radii = []
    for z in np.arange(expected - window, expected + window, config['section_step']):
        slab = tube & (centres[:, 2] >= z) & (centres[:, 2] < z + config['section_step'])
        if slab.sum() >= 8:
            middle = np.average(centres[slab, :2], axis=0, weights=areas[slab])
            radii.append((z + config['section_step'] / 2, float(np.median(np.linalg.norm(centres[slab, :2] - middle, axis=1)))))
    if len(radii) < 5:
        return {'height_m': None, 'method': 'none: too little leg tube near the expected knee height'}
    heights, values = np.array(radii).T
    k = int(np.argmin(values))
    neighbours = np.r_[values[:max(k - 1, 0)], values[k + 2:]]
    if k in (0, len(values) - 1) or neighbours.size == 0 or values[k] > (1 - config['knee_min_narrowing']) * np.median(neighbours):
        return {'height_m': None, 'method': 'none: no clear narrowest section near the expected knee height'}
    return {'height_m': round(float(heights[k]), 4), 'method': 'narrowest leg section near the expected knee height',
            'radius_m': round(float(values[k]), 4)}


def foot(side, labels, centres, areas, height, config):
    """Heel, toe and toe direction from this side's footwear only. The direction is the sole's long axis,
    signed by toe reach vs heel reach measured from the leg's centre on the lower shin (as at intake, but
    with only shoe-labelled faces, so trouser hems trailing on the floor do not count)."""
    sign = 1 if side == 'left' else -1
    shoe = labels == f'footwear.{side}'
    if shoe.sum() < 20:
        return None
    low = centres[shoe, 2].min()
    sole = shoe & (centres[:, 2] < low + config['sole_band_of_height'] * height)
    shin = (np.sign(centres[:, 0]) == sign) & (centres[:, 2] > 0.18 * height) & (centres[:, 2] < 0.26 * height) &         np.isin(labels, (f'pant_leg.{side}', f'footwear.{side}', 'pants', 'skin'))
    if sole.sum() < 8 or shin.sum() < 8:
        return None
    leg_centre = np.average(centres[shin, :2], axis=0, weights=areas[shin])
    _, axis = principal_axis(centres[sole, :2], areas[sole])
    reach = (centres[sole, :2] - leg_centre) @ axis
    forward, back = reach.max(), -reach.min()
    if forward < back:
        axis, forward, back = -axis, back, forward
    along = (centres[sole, :2] - leg_centre) @ axis
    toe, heel = centres[sole][np.argmax(along)], centres[sole][np.argmin(along)]
    ankle = np.r_[leg_centre, centres[shoe, 2].max()]
    return {'toe_direction': _r(np.r_[axis, 0.0]), 'toe_over_heel_reach': round(float(forward / max(back, 1e-6)), 2),
            'ankle': _r(ankle), 'toe': _r(toe), 'heel': _r(heel), 'sole_length_m': round(float(forward + back), 4),
            'faces': int(shoe.sum())}


def owner_answer(why, intake_facing, confirmation, evidence):
    """A provisional facing the footwear cannot confirm: the owner's recorded answer decides, or bind stops and
    asks (FacingUnconfirmed). The fitter never guesses."""
    if confirmation and intake_facing and confirm.matches(confirmation, intake_facing):
        if confirmation['facing'] == 'back':
            raise FacingRejected('you said the front view shows the back; turning an outfit round is not supported '
                                 'yet; refusing')
        return {'status': 'confirmed by the owner', 'confirmation': confirmation, **evidence}
    raise FacingUnconfirmed(f'{why}; refusing until the owner confirms the front (the app asks, or '
                            '`python -m fittedperfection confirm-facing NAME --front`)')


def check_facing(feet, intake_status, config, intake_facing=None, confirmation=None):
    """Toes must point to -Y. A provisional intake facing needs clear shoe evidence here, or the owner's
    confirmation (confirm.py), or is refused."""
    if not feet:
        if intake_status == 'provisional':
            return owner_answer('the facing was provisional at intake and there is no footwear to confirm it',
                                intake_facing, confirmation, {'evidence': 'no footwear'})
        return {'status': intake_status, 'evidence': 'no footwear; facing as found at intake'}
    dots = {side: float(np.dot(f['toe_direction'][:2], [0.0, -1.0])) for side, f in feet.items()}
    ratios = {side: f['toe_over_heel_reach'] for side, f in feet.items()}
    if min(dots.values()) < config['toe_forward_min_dot']:
        raise BindError(f'the shoes do not point to the front ({dots}); the facing is ambiguous, refusing')
    if min(ratios.values()) < config['toe_over_heel_min']:
        if intake_status == 'provisional':
            return owner_answer(f'the shoes do not clearly confirm the provisional facing (toe/heel {ratios})',
                                intake_facing, confirmation, {'shoe_toe_forward': dots, 'shoe_toe_over_heel': ratios})
        return {'status': 'confirmed at intake', 'shoe_toe_forward': dots, 'shoe_toe_over_heel': ratios}
    return {'status': 'confirmed by footwear' if intake_status == 'provisional' else 'confirmed at intake and by footwear',
            'shoe_toe_forward': dots, 'shoe_toe_over_heel': ratios}


def shoulders(sleeves, labels, centres, areas, height):
    torso = labels == 'torso'
    if sleeves.get('left') and sleeves.get('right'):
        left, right = np.array(sleeves['left']['root']), np.array(sleeves['right']['root'])
        return {'left': _r(left), 'right': _r(right), 'centre': _r((left + right) / 2), 'method': 'sleeve roots'}
    chest = torso & (centres[:, 2] > 0.65 * height) & (centres[:, 2] < 0.75 * height)
    if chest.sum() < 20:
        raise BindError('no torso around chest height to place the shoulders')
    half_width = np.percentile(np.abs(centres[chest, 0]), 95)
    # Sleeveless: shoulder_top is the top of the torso cloth over each shoulder, looked for well away from the
    # centre (55-90% of the chest half-width) so a high neckline is not taken for a shoulder.
    points = {}
    for side, sign in (('left', 1), ('right', -1)):
        band = torso & (sign * centres[:, 0] > 0.55 * half_width) & (sign * centres[:, 0] < 0.9 * half_width) &             (centres[:, 2] > 0.6 * height)
        if band.sum() < 5:
            raise BindError(f'no torso over the {side} shoulder to place shoulder_top')
        top = np.percentile(centres[band, 2], 95)
        near = band & (centres[:, 2] > top - 0.03)
        points[side] = np.array([sign * 0.75 * half_width, np.average(centres[near, 1], weights=areas[near]), top])
    return {'left': _r(points['left']), 'right': _r(points['right']),
            'centre': _r((points['left'] + points['right']) / 2), 'method': 'shoulder_top (sleeveless)'}


def crotch(label_crotch, shoulder_height, config):
    prior = config['crotch_prior_of_shoulder_height'] * shoulder_height
    tolerance = config['crotch_prior_tolerance'] * shoulder_height
    if label_crotch is None:
        return {'height_m': round(prior, 4), 'source': 'prior (no branching legs found)', 'prior_m': round(prior, 4)}
    if abs(label_crotch - prior) <= tolerance:
        return {'height_m': round(label_crotch, 4), 'source': 'measured where the legs branch', 'prior_m': round(prior, 4)}
    clamped = prior + np.sign(label_crotch - prior) * tolerance
    return {'height_m': round(float(clamped), 4), 'measured_m': round(label_crotch, 4), 'prior_m': round(prior, 4),
            'source': 'prior band edge (the measured branch is outside the prior band)'}


def review_image(views, record, out_path):
    tiles = []
    camera_list = json.loads((views/'cameras.json').read_text(encoding='utf-8'))
    by_name = {c['name']: c for c in camera_list}
    calibrated = json.loads((views/'views.json').read_text(encoding='utf-8'))['calibrated']
    for view in ('front', 'left'):
        camera = by_name[calibrated[view]]
        image = Image.open(views/'color'/f'{calibrated[view]}.png').convert('RGB')
        image = Image.blend(image, Image.new('RGB', image.size, (20, 20, 20)), 0.45)
        draw = ImageDraw.Draw(image)

        def line(points, colour, width=4):
            if points is None or len(points) < 2:
                return
            x, y, _ = cameras.project(camera, np.array(points))
            draw.line(list(zip(x, y)), fill=colour, width=width)

        def dot(point, colour, radius=7):
            if point is None:
                return
            x, y, _ = cameras.project(camera, np.array([point]))
            draw.ellipse((x[0] - radius, y[0] - radius, x[0] + radius, y[0] + radius), fill=colour)

        for side in ('left', 'right'):
            sleeve = record['sleeves'].get(side)
            line(sleeve and sleeve['points'], (80, 160, 255))
            leg = record['legs'].get(side)
            line(leg and leg['points'], (120, 230, 90))
            foot_record = record['feet'].get(side)
            if foot_record:
                line([foot_record['heel'], foot_record['toe']], (255, 170, 40), 5)
                dot(foot_record['ankle'], (255, 170, 40))
            knee_record = record['knees'].get(side) or {}
            if knee_record.get('height_m') and leg:
                points = np.array(leg['points'])
                dot(points[np.argmin(np.abs(points[:, 2] - knee_record['height_m']))], (255, 80, 80))
        dot(record['shoulders']['left'], (255, 255, 255)); dot(record['shoulders']['right'], (255, 255, 255))
        dot(record['shoulders']['centre'], (255, 255, 0), 5)
        _, y, _ = cameras.project(camera, np.array([[0, 0, record['crotch']['height_m']]]))
        draw.line((0, y[0], image.width, y[0]), fill=(255, 0, 255), width=2)
        draw.text((10, 10), f'{view}: sleeves blue, legs green, feet orange, knees red, shoulders white, crotch magenta',
                  fill=(240, 240, 240))
        tiles.append(image.resize((620, 620)))
    sheet = Image.new('RGB', (1240, 620))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * 620, 0))
    sheet.save(out_path)


def run(ctx, out):
    config = settings.section('bind')
    mesh = read_textured_mesh(ctx.output_of('intake')/'CAGE.glb')
    intake_record = json.loads((ctx.output_of('intake')/'intake.json').read_text(encoding='utf-8'))
    label_folder = ctx.output_of('label')
    labels = np.load(label_folder/'labels.npz')['labels'].astype(str)
    label_record = json.loads((label_folder/'labels.json').read_text(encoding='utf-8'))
    centres = mesh.positions[mesh.triangles].mean(axis=1)
    _, areas = meshes.face_normals(mesh.positions, mesh.triangles)
    graph = meshes.face_graph(mesh.positions, mesh.triangles)
    height = float(mesh.positions[:, 2].max() - mesh.positions[:, 2].min())
    torso_area = float(areas[labels == 'torso'].sum())

    sleeves = {side: sleeve_chain(side, labels, centres, areas, graph, torso_area, config) for side in ('left', 'right')}
    shoulder = shoulders(sleeves, labels, centres, areas, height)
    shoulder_height = float(shoulder['centre'][2])
    crotch_record = crotch(label_record['crotch_height_m'], shoulder_height, config)
    legs = {side: leg_chain(side, labels, centres, areas, crotch_record['height_m'], config) for side in ('left', 'right')}
    if not (legs['left'] and legs['right']):
        raise BindError('could not find both legs below the crotch')
    feet = {side: f for side in ('left', 'right') if (f := foot(side, labels, centres, areas, height, config))}
    facing = check_facing(feet, intake_record['facing']['status'], config, intake_record['facing'],
                          confirm.read(ctx.run.state['garment_sha256']))
    knees = {side: knee(side, labels, centres, areas, crotch_record['height_m'], height, config)
             for side in ('left', 'right')}
    found = [k['height_m'] for k in knees.values()]
    if None in found or abs(found[0] - found[1]) > config['knee_max_side_difference']:
        if any(h is not None for h in found):              # one leg alone, or legs that disagree: not a knee
            reason = f'none: the legs do not agree ({found[0]} vs {found[1]} m)'
            knees = {side: {'height_m': None, 'method': reason, 'single_leg_reading': k} for side, k in knees.items()}
    length = label_record.get('legwear_length')
    record = {
        'frame': {'up': '+Z', 'front': '-Y', 'own_left': '+X'}, 'height_m': round(height, 4),
        'sleeves': sleeves, 'shoulders': shoulder, 'shoulder_height_m': round(shoulder_height, 4),
        'crotch': crotch_record, 'legs': legs, 'feet': feet, 'facing': facing, 'knees': knees,
        'legwear_length': length, 'hem_fraction': config['hem_fraction'][length] if length else None,
        'sleeveless': not (sleeves['left'] and sleeves['right']),
    }
    (out/'binding.json').write_text(json.dumps(record, indent=1), encoding='utf-8', newline='\n')
    review_image(ctx.output_of('views'), record, out/'review.png')
    return {'sleeves': {s: (v['length_m'] if v else None) for s, v in sleeves.items()},
            'shoulders': shoulder['method'], 'crotch': crotch_record, 'facing': facing['status'],
            'knees': {s: k['height_m'] for s, k in knees.items()}, 'legwear_length': length}
