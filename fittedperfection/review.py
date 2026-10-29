"""Stage 7, audit and review (spec §5.7). Diagnostics and a second opinion for the human, never acceptance:
fit_reviewed and game_ready stay false whatever this finds.

R1 audit (numpy, per body, from OUTFIT.glb):
   penetration: share of sampled cloth inside the body and clearance percentiles (shoes and hardware left
     out: feet inside shoes are hidden by design);
   coverage: body skin points with cloth along their outward normal, against the donor inside the source,
     so lost coverage (holes, missing stockings) shows per region;
   synthetic poses: body and outfit posed together by their weights (elbows 30/60, knees 15/30), share of
     cloth inside the body in each;
   silhouette: width profile against the source, normalised by height (uncalibrated: reported, never a gate);
   stretch: copied from the fit report.
R2 renders (Blender): the source and each body from the front, three-quarter, side and back, with sheets.
R3 Muse Wonder: one composite image (front and back rows), structured findings, cached, spend-guarded:
   with no spending allowed and nothing cached, nothing is sent.
"""
import json
import re

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from . import body as bodies, config, donor, fit, fullres, meshes, models, secrets, settings, wonder, wrap
from .gltf import gltf_to_canonical, read_skinned_mesh, read_textured_mesh
from .samapi import SamApiError, SpendRefused

CODE = ('review.py', 'blender/review.py', 'blender/preview.py', 'wonder.py', 'wrap.py', 'fullres.py', 'fit.py',
        'body.py', 'donor.py', 'gltf.py')
SCRIPT = config.PACKAGE/'blender'/'review.py'
VIEWS = ('front', 'three_quarter', 'side', 'back')
LEFT_OUT = ('footwear', 'hardware', 'skin')
# body region -> the cloth regions (fit.label_region) that can cover it
OWN_CLOTH = {'arm.left': ('arm.left',), 'arm.right': ('arm.right',), 'leg.left': ('leg.left', 'hips'),
             'leg.right': ('leg.right', 'hips'), 'trunk': ('trunk', 'hips', 'hanging')}


def poses(config_audit):
    """Test poses from settings (spec §5.7): name -> (bone, degrees)."""
    angles = config_audit['synthetic_poses']
    return {**{f'elbows_{d}': ('lowerarm01', d) for d in angles['elbow_degrees']},
            **{f'knees_{d}': ('lowerleg01', d) for d in angles['knee_degrees']}}


# ---------------------------------------------------------------- R1 audit

def penetration(points, skin):
    h = skin.heights(points)[0]
    p1, p5, p50 = np.percentile(h, [1, 5, 50]) * 100
    return {'inside_share': round(float((h < 0).mean()), 4), 'clearance_cm': {'p1': round(float(p1), 2),
            'p5': round(float(p5), 2), 'p50': round(float(p50), 2)}}


def covered(points, normals, cloth_tree, cloth, reach, min_cos, k=16):
    """Per body point: is there cloth along its outward normal within `reach`?"""
    distance, nearest = cloth_tree.query(points, k=k, distance_upper_bound=reach)
    hit = np.zeros(len(points), dtype=bool)
    for column in range(k):
        ok = np.isfinite(distance[:, column])
        if not ok.any():
            continue
        towards = cloth[np.minimum(nearest[ok, column], len(cloth) - 1)] - points[ok]
        cos = np.einsum('ij,ij->i', towards, normals[ok]) / np.maximum(np.linalg.norm(towards, axis=1), 1e-12)
        hit[np.flatnonzero(ok)[cos >= min_cos]] = True
    return hit


def bone_turn(body, positions, bone, degrees, front_sign):
    """Per-bone transforms turning `bone` (both sides) and its children about its joint, about the bone's own
    X axis, signed so the child end moves towards front_sign * Y (elbows forward: -1, knees back: +1)."""
    transforms = np.repeat(np.eye(4)[None], len(body.bone_names), axis=0)
    for side in ('L', 'R'):
        index = body.bone_names.index(f'{bone}.{side}')
        joint, axis = body.rest[index][:3, 3], body.rest[index][:3, 0] / np.linalg.norm(body.rest[index][:3, 0])
        children = list(bodies._descendants(body.parents, index))
        tip = max(children, key=lambda c: np.linalg.norm(body.rest[c][:3, 3] - joint))
        tip_point = body.rest[tip][:3, 3]
        for sign in (1.0, -1.0):
            turn = donor.about(joint, donor.rotation_between_axis_angle(axis, sign * np.radians(degrees)))
            if (donor.apply(turn, tip_point) - tip_point)[1] * front_sign > 0:
                break
        transforms[children] = turn
    return transforms


def lbs(transforms, joints, weights, positions):
    blended = np.einsum('vk,vkij->vij', weights, transforms[joints])
    return np.einsum('vij,vj->vi', blended[:, :3, :3], positions) + blended[:, :3, 3]


def silhouette(source, fitted, bins=20):
    """Mean relative difference of front-view width profiles, heights normalised (uncalibrated)."""
    def profile(points):
        z = (points[:, 2] - points[:, 2].min()) / np.ptp(points[:, 2])
        height = np.ptp(points[:, 2])
        widths = np.array([np.ptp(points[(z >= a) & (z < a + 1 / bins), 0]) if ((z >= a) & (z < a + 1 / bins)).sum()
                           > 3 else np.nan for a in np.arange(0, 1, 1 / bins)])
        return widths / height
    a, b = profile(source), profile(fitted)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
    return {'mean_width_change': round(float(np.mean(np.abs(b[ok] - a[ok]) / a[ok])), 3),
            'note': 'uncalibrated: reported only, never a gate (spec §5.7)'}


def audit(run_folder, target, garment, vertex_labels, donor_posed, config_audit, rng_seed=0):
    folder = run_folder/'fit'/target
    outfit = read_skinned_mesh(folder/'OUTFIT.glb')
    points = gltf_to_canonical(outfit.positions)
    b = bodies.load(target)
    skin = wrap.Skin(b.positions, b.triangles)
    cloth = ~np.isin(np.char.partition(vertex_labels.astype(str), '.')[:, 0], LEFT_OUT)
    sample = np.random.default_rng(rng_seed).choice(np.flatnonzero(cloth), min(config_audit['penetration_samples'],
                                                                               int(cloth.sum())), replace=False)
    record = {'body': target, 'rest': penetration(points[sample], skin)}

    hidden = np.load(folder/'hidden_body_faces.npy')
    shown = np.ones(len(b.positions), dtype=bool)
    shown[np.unique(b.triangles[hidden])] = False
    bare = np.zeros(len(b.positions), dtype=bool)
    for root in ('head', 'wrist.L', 'wrist.R'):
        bones = bodies._descendants(b.parents, b.bone_names.index(root))
        bare |= (b.weights * np.isin(b.joints, list(bones))).sum(axis=1) >= 0.5
    check = shown & ~bare
    reach, min_cos = config_audit['coverage_reach_m'], config_audit['coverage_min_cos']
    donor_skin = wrap.Skin(donor_posed, b.triangles)
    cloth_region = np.array([fit.label_region(n) for n in vertex_labels], dtype=object)
    regions = b.vertex_region[check]
    record['coverage'] = {}
    for region in np.unique(regions):
        r = regions == region
        # only the region's own cloth covers it (a sleeveless donor's arm beside a cargo pocket is not covered),
        # and never the generator's mannequin skin
        fabric = (vertex_labels != 'skin') & np.isin(cloth_region, OWN_CLOTH.get(region, (region,)))
        where = np.flatnonzero(check)[r]
        now = np.zeros(len(r), dtype=bool)
        before = np.zeros(len(r), dtype=bool)
        if fabric.any():
            now[r] = covered(b.positions[where], skin.vertex_normals[where], cKDTree(points[fabric]), points[fabric],
                             reach, min_cos, k=48)
            before[r] = covered(donor_posed[where], donor_skin.vertex_normals[where], cKDTree(garment[fabric]),
                                garment[fabric], reach, min_cos, k=48)
        source = int((before & r).sum())
        record['coverage'][str(region)] = {
            'covered_in_source': round(source / max(int(r.sum()), 1), 3),
            'covered_now': round(int((now & r).sum()) / max(int(r.sum()), 1), 3),
            'lost_share': round(int((before & ~now & r).sum()) / max(source, 1), 3)}

    record['poses'] = {}
    for name, (bone, degrees) in poses(config_audit).items():
        transforms = bone_turn(b, b.positions, bone, degrees, -1.0 if bone == 'lowerarm01' else 1.0)
        posed_body = bodies.skin(b, transforms)
        posed_cloth = lbs(transforms, outfit.joints[sample], outfit.weights[sample], points[sample])
        record['poses'][name] = penetration(posed_cloth, wrap.Skin(posed_body, b.triangles))
    record['silhouette'] = silhouette(garment[cloth], points[cloth])
    report = json.loads((folder/'fit.report.json').read_text(encoding='utf-8'))
    record['stretch'] = {'cage': report['stretch_vs_source'],
                         'full_resolution': report.get('full_resolution', {}).get('stretch_vs_source')}
    return record


# ---------------------------------------------------------------- R2 renders and sheets

def sheet(paths, out, height=600):
    images = [Image.open(p).convert('RGB') for p in paths]
    images = [i.resize((max(1, int(i.width * height / i.height)), height)) for i in images]
    canvas = Image.new('RGB', (sum(i.width for i in images), height), (40, 40, 40))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    canvas.save(out)
    return canvas


def stack(images, out):
    width = max(i.width for i in images)
    canvas = Image.new('RGB', (width, sum(i.height for i in images)), (40, 40, 40))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(out)


def renders(ctx, run_folder, out):
    folder = out/'renders'
    folder.mkdir()
    items = [f'source|{run_folder/"full"/"GARMENT.glb"}|texture']
    for target in config.BODIES:
        fit = run_folder/'fit'/target
        items += [f'{target}|{fit/"BODY.glb"}|skin', f'{target}|{fit/"OUTFIT.glb"}|texture']
    command = [config.blender(), '--background', '--factory-startup', '--python-exit-code', '1', '--python', SCRIPT,
               '--', '--out', folder, '--height', 900]
    for item in items:
        command += ['--item', item]
    ctx.call(command, 'review-blender.log')
    groups = ['source', *config.BODIES]
    rows = [sheet([folder/f'{g}-{view}.png' for g in groups], out/f'sheet-{view}.png') for view in VIEWS]
    stack(rows, out/'sheet.png')


# ---------------------------------------------------------------- R3 Muse Wonder

def parse_findings(text):
    cleaned = re.sub(r'^```(?:json)?|```$', '', (text or '').strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        return {'unparsed': text}


def wonder_review(out, labels):
    composite = out/'wonder-input.png'
    rows = [Image.open(out/f'sheet-{view}.png').convert('RGB') for view in ('front', 'back')]
    width = 1600
    rows = [r.resize((width, int(r.height * width / r.width))) for r in rows]
    stack(rows, composite)
    names = sorted({str(n).split('.')[0] for n in np.unique(labels)} - {'none', 'skin'})
    prompt = wonder.FIT_PROMPT.format(parts=', '.join(names))
    record = {'model': models.pin('wonder')['model'], 'image': composite.name, 'parts': names,
              'fit_reviewed': False, 'note': 'a checklist for the human, never a verdict (spec §5.7)'}
    try:
        client = wonder.WonderClient(config.runs_folder()/'_cache'/'wonder', record['model'], models.API_BASE,
                                     api_key=secrets.api_key(), budget_usd=config.spend_allowance_usd())
        text, summary = client.review(prompt, [composite], max_output_tokens=3000, effort='minimal')
    except SpendRefused as refused:
        return {**record, 'status': 'skipped (no spending allowed)', 'detail': str(refused)}
    except (SamApiError, ValueError, OSError) as failed:
        return {**record, 'status': 'not reviewed', 'detail': f'{type(failed).__name__}: {failed}'}
    return {**record, 'status': 'reviewed', 'call': summary, 'findings': parse_findings(text)}


# ---------------------------------------------------------------- the stage

def run(ctx, out):
    config_audit = settings.section('audit')
    run_folder = ctx.run_folder
    cage = read_textured_mesh(ctx.output_of('intake')/'CAGE.glb')
    garment = read_textured_mesh(ctx.output_of('intake')/'GARMENT.glb')
    labels = np.load(ctx.output_of('label')/'labels.npz')['labels'].astype(str)
    face_labels = np.load(ctx.output_of('label')/'fullres-labels.npz')['labels'].astype(str)
    welded = meshes.weld(garment.positions)
    vertex_labels = fullres.vertex_labels(face_labels, garment.triangles, welded)[welded]

    binding = json.loads((ctx.output_of('bind')/'binding.json').read_text(encoding='utf-8'))
    centres = cage.positions[cage.triangles].mean(axis=1)
    _, areas = meshes.face_normals(cage.positions, cage.triangles)
    donor_posed = donor.build(binding, labels, centres, areas, settings.section('fit'))['posed']

    summary = {}
    for target in config.BODIES:
        record = audit(run_folder, target, garment.positions, vertex_labels, donor_posed, config_audit)
        (out/target).mkdir()
        (out/target/'audit.json').write_text(json.dumps(record, indent=1), encoding='utf-8', newline='\n')
        summary[target] = {'inside_rest': record['rest']['inside_share'],
                           'inside_poses': {k: v['inside_share'] for k, v in record['poses'].items()},
                           'lost_coverage': {k: v['lost_share'] for k, v in record['coverage'].items()}}
    renders(ctx, run_folder, out)
    review = wonder_review(out, labels)
    (out/'wonder.json').write_text(json.dumps(review, indent=1), encoding='utf-8', newline='\n')
    (out/'audit.json').write_text(json.dumps({'fit_reviewed': False, 'game_ready': False, 'bodies': summary},
                                             indent=1), encoding='utf-8', newline='\n')
    return {'audit': summary, 'wonder': review['status']}
