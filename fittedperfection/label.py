"""Stage 4, label (spec §5.4): SAM's 2D masks -> one part label per cage triangle.

1. Lift + specificity: per view, garment masks are painted largest first, so where
   they overlap the smaller (more specific) one wins each pixel. Skin masks form a
   separate layer and never erase garment evidence. The face-ID image says which
   triangle each pixel shows.
2. Fuse: per triangle and class, the share of all its visible pixels (over every
   view) that the class won, weighted by mask confidence. A view that sees the
   triangle without the label counts against it.
3. Smooth: one sparse solve per class over the face graph; links are weakened
   across source seams and creases, so labels spread along fabric, fill hidden
   triangles (and seen ones no mask touched), and stop at real boundaries.
4. Rules, each reporting what it changed: branch split (trousers below the crotch
   are legs), open shell (torso cloth below the hips that no longer wraps the body
   hangs), host (attachments take the label of what they are sewn onto), skin check
   (skin only where SAM and the texture colour agree), host for unidentified pieces
   (trims and lace with no vocabulary word, and skin rejected by colour with no garment
   evidence, take the label they are sewn onto), covered leg skin, coherent
   limbs, and finally sides (+X is the garment's own left).
"""
import io
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from . import faceid, meshes, settings, vocabulary
from .gltf import read_textured_mesh

CODE = ('label.py', 'meshes.py', 'faceid.py', 'gltf.py', 'vocabulary.py')
CLASSES = tuple(vocabulary.CLASSES)
SIDED = ('sleeve', 'cuff', 'pant_leg', 'footwear')
LIMBS = ('sleeve', 'cuff', 'pant_leg')
# Structural parts are never swallowed as islands by a limb (shoes inside trouser hems stay shoes).
KEEP_AS_PARTS = ('footwear', 'hardware', 'cuff', 'collar', 'hood', 'waistband', 'none')
COLOURS = {'torso': (230, 70, 70), 'sleeve': (70, 140, 240), 'cuff': (40, 220, 220), 'collar': (250, 210, 40),
           'hood': (160, 90, 220), 'waistband': (250, 140, 30), 'pants': (60, 190, 90), 'pant_leg': (150, 230, 90),
           'skirt': (240, 110, 200), 'footwear': (140, 90, 50), 'hardware': (255, 255, 255),
           'attachment': (255, 60, 160), 'hanging': (120, 120, 255), 'skin': (255, 200, 160), 'none': (60, 60, 60)}


# -- 1-2. lift, specificity, fusion --------------------------------------------------------------

SKIN = CLASSES.index('skin')


def paint_view(records, rasters, size):
    """Per pixel for one view: the winning garment class and its confidence, and the skin cover.

    Garment classes compete: masks are painted largest first, so the smaller (more specific) one wins.
    Skin is a different kind of evidence (it is confirmed by colour later), so it is kept in its own
    layer and never erases garment evidence: tights found over a "bare leg" stay tights.
    """
    winner = np.full((size, size), -1, dtype=np.int64)
    confidence = np.zeros((size, size))
    skin = np.zeros((size, size))
    painted = [(int(rasters[r['raster']].sum()), r) for r in records if r['class'] in CLASSES]
    for _, record in sorted(painted, key=lambda item: -item[0]):
        left, top, right, bottom = record['bounds']
        mask = rasters[record['raster']] > 0
        weight = record['confidence'] if record['confidence'] is not None else 1.0
        if record['class'] == 'skin':
            region = skin[top:bottom, left:right]
            region[mask] = np.maximum(region[mask], weight)
        else:
            winner[top:bottom, left:right][mask] = CLASSES.index(record['class'])
            confidence[top:bottom, left:right][mask] = weight
    return winner, confidence, skin


def fuse(view_images, records_by_view, rasters, face_count):
    """Per face: pixels seen over all views; pixels any mask covered; per garment class the
    confidence-weighted pixels it won; for skin the confidence-weighted pixels any skin mask covered."""
    seen = np.zeros(face_count)
    covered = np.zeros(face_count)
    won = np.zeros((face_count, len(CLASSES)))
    for view, ids in view_images.items():
        foreground = ids >= 0
        faces = ids[foreground]
        seen += np.bincount(faces, minlength=face_count)
        winner, confidence, skin = paint_view(records_by_view.get(view, []), rasters, ids.shape[0])
        labelled = winner[foreground] >= 0
        key = faces[labelled] * len(CLASSES) + winner[foreground][labelled]
        won += np.bincount(key, weights=confidence[foreground][labelled],
                           minlength=face_count * len(CLASSES)).reshape(face_count, len(CLASSES))
        won[:, SKIN] += np.bincount(faces, weights=skin[foreground], minlength=face_count)
        any_mask = (winner[foreground] >= 0) | (skin[foreground] > 0)
        covered += np.bincount(faces[any_mask], minlength=face_count)
    evidence = won / np.maximum(seen, 1)[:, None]
    return evidence, seen, covered


# -- 3. smoothing -------------------------------------------------------------------------------

def smooth(evidence, covered, graph, strength, seam_weight, crease_weight, crease_degrees, half_weight_pixels):
    """Solve (A + strength * L) X = A E per class; L links touching faces.

    A trusts a face by how many of its pixels some mask covered. A face that is seen but that no mask
    touches says nothing about what it is, so like a hidden face it is filled from its neighbours.
    """
    n = graph.face_count
    weight = np.ones(len(graph.pairs))
    weight[graph.seam] *= seam_weight
    weight[graph.angle_deg > crease_degrees] *= crease_weight
    i, j = graph.pairs[:, 0], graph.pairs[:, 1]
    adjacency = coo_matrix((np.r_[weight, weight], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    trust = covered / (covered + half_weight_pixels)
    system = (diags(trust + 1e-6) + strength * laplacian).tocsc()
    solver = splu(system)
    return np.column_stack([solver.solve(trust * evidence[:, c]) for c in range(evidence.shape[1])])


# -- 4. rules -----------------------------------------------------------------------------------

def legs_apart_below(z, centres, areas, graph):
    """True when the garment surface below height z is two separate, comparable pieces (two legs), judged
    by which faces share edges, so legs that hang close together still count as two."""
    below = centres[:, 2] < z
    piece, count = meshes.parts(graph, below)
    if count < 2:
        return False
    sizes = np.sort(np.bincount(piece[below], weights=areas[below]))[::-1]
    return sizes[1] >= 0.3 * sizes[0] and sizes[1] >= 0.02 * areas.sum()


def crotch_height(centres, areas, graph, height, band=(0.30, 0.65), step=0.01):
    """Where the garment's two legs join (its branch height), from the whole surface, so it does not depend
    on the labels: the highest height in the band where a crotch can be (spec §5.5: about 0.59 x shoulder
    height) below which the surface is still two legs, provided it is seen joining above. None when there
    are no two legs in the band (a long dress) or they never join in it. Bind measures the crotch itself."""
    floor = centres[:, 2].min()
    heights = np.arange(floor + band[0] * height, floor + band[1] * height, step)
    apart = [legs_apart_below(z, centres, areas, graph) for z in heights]
    if not any(apart):
        return None
    last = max(z for z, separate in zip(heights, apart) if separate)
    joins = any(not separate for z, separate in zip(heights, apart) if z > last)
    return float(last) if joins else None


def branch_split(labels, centres, crotch):
    if crotch is None:
        return 0
    change = (labels == 'pants') & (centres[:, 2] < crotch)
    labels[change] = 'pant_leg'
    return int(change.sum())


def open_shell(labels, centres, crotch, height, coverage_needed, band=0.03, bins=24):
    """Torso cloth below the hips that no longer wraps around the body hangs (coat tails)."""
    hips = crotch if crotch is not None else 0.45 * height
    torso = (labels == 'torso') & (centres[:, 2] < hips)
    changed = 0
    for z in np.arange(centres[torso, 2].min() if torso.any() else 0, hips, band):
        slab = torso & (centres[:, 2] >= z) & (centres[:, 2] < z + band)
        if slab.sum() < 5:
            continue
        everything = (centres[:, 2] >= z) & (centres[:, 2] < z + band) & (labels != 'none')
        middle = centres[everything, :2].mean(axis=0)
        offset = centres[slab, :2] - middle
        occupied = np.unique(((np.arctan2(offset[:, 1], offset[:, 0]) + np.pi) / (2 * np.pi) * bins).astype(int) % bins)
        if len(occupied) / bins < coverage_needed:
            labels[slab] = 'hanging'
            changed += int(slab.sum())
    return changed


def boundary_vote(graph, labels, inside, exclude=()):
    """Most common label just outside the faces in `inside` (by shared edges), or None."""
    a, b = graph.pairs[:, 0], graph.pairs[:, 1]
    crossing = np.r_[b[inside[a] & ~inside[b]], a[inside[b] & ~inside[a]]]
    outside = [label for label in labels[crossing] if label not in exclude]
    if not outside:
        return None, 0.0
    names, counts = np.unique(outside, return_counts=True)
    return names[np.argmax(counts)], counts.max() / counts.sum()


def host_attachments(labels, graph):
    piece, count = meshes.parts(graph, labels == 'attachment')
    changed = 0
    for k in range(count):
        inside = piece == k
        host, _ = boundary_vote(graph, labels, inside, exclude=('attachment', 'none', 'hardware', 'skin'))
        if host is not None:
            labels[inside] = host
            changed += int(inside.sum())
    return changed


def host_unlabelled(labels, graph, areas, largest_share):
    """Unidentified pieces (trims, lace, ornaments the vocabulary has no word for) are attachments too:
    each takes the label it is sewn onto. A large unidentified piece stays 'none' and is reported."""
    piece, count = meshes.parts(graph, labels == 'none')
    total = areas.sum()
    changed, kept = 0, []
    for k in range(count):
        inside = piece == k
        if areas[inside].sum() > largest_share * total:
            kept.append(round(float(areas[inside].sum() / total), 4))
            continue
        # Never skin: a trim sewn onto bare skin is still clothing. With only skin around it, it is a
        # garment piece of its own (like straps worn on the bare back).
        host, _ = boundary_vote(graph, labels, inside, exclude=('none', 'skin'))
        labels[inside] = host if host is not None else 'attachment'
        changed += int(inside.sum())
    return changed, kept


def face_colours(mesh):
    """Texture colour per triangle: the mean of samples at its three corners and its centre."""
    if mesh.texture_png is None:
        return None
    texture = np.asarray(Image.open(io.BytesIO(mesh.texture_png)).convert('RGB')).astype(np.float64)
    h, w = texture.shape[:2]
    corners = mesh.uvs[mesh.triangles]                                      # (T, 3, 2)
    samples = np.concatenate([corners, corners.mean(axis=1, keepdims=True)], axis=1) % 1.0
    x = np.clip((samples[..., 0] * w).astype(int), 0, w - 1)
    y = np.clip((samples[..., 1] * h).astype(int), 0, h - 1)
    return texture[y, x].mean(axis=1)


def skin_check(labels, scores, colours, centres, height, min_distance, spread_factor, confident_score):
    """Skin needs SAM and the texture colour to agree; clearly different colours revert to their best garment.

    The reference tone comes from confident skin on the upper body (shoulders, chest, back) that no
    garment mask covers, since legs
    may be wearing stockings. The limit adapts to how much that reference skin itself varies (its
    median distance from the tone, which mislabelled faces in the reference cannot inflate).
    """
    skin = labels == 'skin'
    if colours is None or not skin.any():
        return 0, None, None
    garment_scores = np.delete(scores, SKIN, axis=1)
    confident = skin & (scores[:, SKIN] >= confident_score) & (garment_scores.max(axis=1) < 0.5 * confident_score)
    reference = confident & (centres[:, 2] > 0.55 * height)
    if reference.sum() < 20:
        reference = confident if confident.sum() >= 20 else skin
    tone = np.median(colours[reference], axis=0)
    spread = np.median(np.linalg.norm(colours[reference] - tone, axis=1))     # robust to mislabelled faces
    limit = max(min_distance, spread_factor * spread)
    differs = skin & (np.linalg.norm(colours - tone, axis=1) > limit)
    garment = scores.copy()
    garment[:, CLASSES.index('skin')] = -np.inf
    best = np.array(CLASSES)[np.argmax(garment, axis=1)]
    labels[differs] = np.where(garment[differs].max(axis=1) > 0.05, best[differs], 'none')
    return int(differs.sum()), [round(float(c)) for c in tone], round(float(limit), 1)


def covered_leg_skin(labels, centres, areas, crotch, height, needed, band=0.04):
    top = crotch if crotch is not None else 0.45 * height
    changed = 0
    for side in (1, -1):
        leg = (np.sign(centres[:, 0]) == side) & (centres[:, 2] < top)
        for z in np.arange(0, top, band):
            slab = leg & (centres[:, 2] >= z) & (centres[:, 2] < z + band)
            covered = slab & np.isin(labels, ('pant_leg', 'footwear'))
            if not slab.any() or areas[covered].sum() < needed * areas[slab & (labels != 'none')].sum():
                continue
            skin = slab & (labels == 'skin')
            if skin.any():
                names, weights = np.unique(labels[covered], return_inverse=True)
                labels[skin] = names[np.argmax(np.bincount(weights, weights=areas[covered]))]
                changed += int(skin.sum())
    return changed


def coherent_limbs(labels, graph, areas, island_fraction):
    """Small islands of another label inside a limb take the limb's label."""
    changed = 0
    for name in np.unique(labels):
        if name in KEEP_AS_PARTS or name in LIMBS:
            continue
        piece, count = meshes.parts(graph, labels == name)
        for k in range(count):
            inside = piece == k
            host, share = boundary_vote(graph, labels, inside)
            if host in LIMBS and share >= 0.8 and areas[inside].sum() < island_fraction * areas[labels == host].sum():
                labels[inside] = host
                changed += int(inside.sum())
    return changed


def sides(labels, centres, graph, areas, both_sides_share=0.2):
    """Sided classes become .left (+X, the garment's own left) or .right, one side per connected piece.

    A piece that clearly spans both sides (two shoes joined through the mesh, say) is split at the
    middle instead.
    """
    changed = 0
    for name in SIDED:
        piece, count = meshes.parts(graph, labels == name)
        for k in range(count):
            inside = piece == k
            left_share = areas[inside & (centres[:, 0] >= 0)].sum() / max(areas[inside].sum(), 1e-12)
            if min(left_share, 1 - left_share) > both_sides_share:
                labels[inside] = np.where(centres[inside, 0] >= 0, f'{name}.left', f'{name}.right')
            else:
                labels[inside] = f'{name}.left' if left_share >= 0.5 else f'{name}.right'
            changed += int(inside.sum())
    return changed


def length_class(records, rasters):
    """Which length phrase SAM matched most strongly (confidence-weighted mask area over the turntable)."""
    totals = {}
    for record in records:
        if record['length'] and record['view'].startswith('orbit'):
            weight = record['confidence'] if record['confidence'] is not None else 1.0
            totals[record['length']] = totals.get(record['length'], 0.0) + weight * float(rasters[record['raster']].sum())
    return (max(totals, key=totals.get) if totals else None), {k: round(v) for k, v in totals.items()}


# -- review image ---------------------------------------------------------------------------------

def review_image(labels, margin, view_images, views):
    names = ('front', 'left', 'back', 'right')
    colours = {name: COLOURS[name.split('.')[0]] for name in np.unique(labels)}
    side_tint = {name: (0.75 if name.endswith('.right') else 1.0) for name in colours}
    palette = np.array([[c * side_tint[n] for c in colours[n]] for n in labels], dtype=np.float64)
    tile = 420
    sheet = Image.new('RGB', (tile * 4 + 190, tile * 2 + 60), (25, 28, 32))
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        ids = view_images[views[name]]
        for row, colour_of in enumerate((palette, None)):
            image = np.full(ids.shape + (3,), 25.0)
            visible = ids >= 0
            if colour_of is not None:
                image[visible] = colour_of[ids[visible]]
            else:                                                 # how sure: dark = clear winner, red = close call
                m = np.clip(margin[ids[visible]] / 0.3, 0, 1)[:, None]
                image[visible] = m * np.array([70, 70, 80]) + (1 - m) * np.array([255, 60, 40])
            sheet.paste(Image.fromarray(image.astype(np.uint8)).resize((tile, tile)), (i * tile, row * (tile + 30) + 30))
        draw.text((i * tile + 6, 8), f'{name}: labels', fill=(230, 230, 230))
        draw.text((i * tile + 6, tile + 38), f'{name}: close calls in red', fill=(230, 230, 230))
    for k, name in enumerate(sorted(colours)):
        draw.rectangle((tile * 4 + 12, 30 + k * 24, tile * 4 + 30, 46 + k * 24), fill=tuple(int(c) for c in
                                                                                          palette[labels == name][0]))
        draw.text((tile * 4 + 38, 32 + k * 24), name, fill=(230, 230, 230))
    return sheet


# -- the stage ----------------------------------------------------------------------------------

def run(ctx, out):
    config = settings.section('label')
    mesh = read_textured_mesh(ctx.output_of('intake')/'CAGE.glb')
    views_folder, vision_folder = ctx.output_of('views'), ctx.output_of('vision')
    cameras_list = json.loads((views_folder/'cameras.json').read_text(encoding='utf-8'))
    view_record = json.loads((views_folder/'views.json').read_text(encoding='utf-8'))
    vision = json.loads((vision_folder/'vision.json').read_text(encoding='utf-8'))
    rasters = np.load(vision_folder/'masks.npz')
    records_by_view = {}
    for record in vision['records']:
        records_by_view.setdefault(record['view'], []).append(record)
    view_images = {camera['name']: faceid.decode(np.asarray(Image.open(views_folder/'faceid'/f'{camera["name"]}.png')
                                                            .convert('RGB'))) for camera in cameras_list}
    face_count = len(mesh.triangles)
    centres = mesh.positions[mesh.triangles].mean(axis=1)
    _, areas = meshes.face_normals(mesh.positions, mesh.triangles)
    height = float(mesh.positions[:, 2].max() - mesh.positions[:, 2].min())
    graph = meshes.face_graph(mesh.positions, mesh.triangles)

    evidence, seen, covered = fuse(view_images, records_by_view, rasters, face_count)
    scores = smooth(evidence, covered, graph, config['smoothing'], config['seam_weight'], config['crease_weight'],
                    config['crease_degrees'], config['half_trust_pixels'])
    order = np.sort(scores, axis=1)
    margin = order[:, -1] - order[:, -2]
    labels = np.array(CLASSES, dtype=object)[np.argmax(scores, axis=1)]
    labels[order[:, -1] < config['min_score']] = 'none'
    smoothed = labels.copy()

    changes = {}
    crotch = crotch_height(centres, areas, graph, height)
    changes['branch_split'] = branch_split(labels, centres, crotch)
    changes['open_shell'] = open_shell(labels, centres, crotch, height, config['shell_coverage'])
    changes['host_attachments'] = host_attachments(labels, graph)
    colours = face_colours(mesh)
    changes['skin_check'], skin_tone, skin_limit = skin_check(
        labels, scores, colours, centres, height, config['skin_colour_min_distance'],
        config['skin_colour_spread_factor'], config['confident_skin_score'])
    changes['host_unlabelled'], large_unlabelled = host_unlabelled(labels, graph, areas,
                                                                   config['largest_unlabelled_share'])
    changes['covered_leg_skin'] = covered_leg_skin(labels, centres, areas, crotch, height, config['leg_covered_share'])
    changes['coherent_limbs'] = coherent_limbs(labels, graph, areas, config['limb_island_fraction'])
    changes['sides'] = sides(labels, centres, graph, areas)
    length, length_scores = length_class(vision['records'], rasters)

    labels = labels.astype(str)
    np.savez_compressed(out/'labels.npz', labels=labels, smoothed=smoothed.astype(str), scores=scores.astype(np.float32),
                        evidence=evidence.astype(np.float32), seen=seen.astype(np.float32), covered=covered.astype(np.float32), margin=margin.astype(np.float32))
    names, counts = np.unique(labels, return_counts=True)
    area_share = {name: round(float(areas[labels == name].sum() / areas.sum()), 4) for name in names}
    review_image(labels, margin, view_images, {name: frame for name, frame in view_record['calibrated'].items()}
                 ).save(out/'review.png')
    summary = {'classes': list(CLASSES), 'faces': face_count, 'labels': {n: int(c) for n, c in zip(names, counts)},
               'area_share': area_share, 'rule_changes': changes,
               'crotch_height_m': None if crotch is None else round(crotch, 4),
               'legwear_length': length, 'length_scores': length_scores, 'skin_tone_rgb': skin_tone,
               'skin_colour_limit': skin_limit,
               'never_seen_faces': int((seen == 0).sum()), 'large_unlabelled_pieces': large_unlabelled,
               'close_calls': round(float((margin < 0.1).mean()), 4)}
    (out/'labels.json').write_text(json.dumps(summary, indent=1), encoding='utf-8', newline='\n')
    return {k: summary[k] for k in ('labels', 'rule_changes', 'crotch_height_m', 'legwear_length', 'close_calls')}
