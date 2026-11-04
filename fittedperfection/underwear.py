"""Display underwear for the target bodies (changes for this build): never part of a fit.

Wherever a body is shown (the 3D viewer, the Blender preview, turntable and review renders) it wears plain grey
underwear painted on its own surface: high briefs on every body and a bandeau top on the female bodies. Nothing
moves: the viewer colours by height in a shader and Blender cuts the faces along the band heights and paints
them. The fit never imports this module.

The bands come from each body's own measures, so they sit right on every build:
    briefs: from the crotch (legs: 1% of height above it, trunk: 2% below) up to the root joint + 4.5%;
    top (female bodies): the breast joint - 5.5% to + 3.5%.
The briefs stop at the crotch, so they never cover the thighs, where a fit's real gaps must still show. Arm
vertices (arm weight >= 0.5) are never painted, so hands resting at the hips stay bare.

A record is cached by the body file's sha256 and this module's own bytes, in runs/_cache/underwear/.
"""
import hashlib
import json
from pathlib import Path

from . import body as bodies, config
from .hashing import file_sha256

COLOUR = (0.20, 0.22, 0.26)            # linear RGB, a plain dark grey that reads as clothing, not skin
ARM_LIMIT = 0.5                        # vertices pulled at least this much by arm bones are never painted
LEG_LIMIT = 0.5                        # ... and by leg bones use the leg edge of the briefs
BRIEFS_TOP = 0.045                     # fractions of the body's height
BRIEFS_LEG = 0.01
BRIEFS_TRUNK = -0.02
TOP_LOW, TOP_HIGH = -0.055, 0.035


def bands(body_id):
    """Band heights (canonical z, metres, the body's rest pose) and the per-vertex arm and leg weights."""
    b = bodies.load(body_id)
    height = b.height
    crotch = bodies.landmarks(b)['crotch_height_m']
    result = {'briefs_top': b.joint('root')[2] + BRIEFS_TOP * height,
              'briefs_leg': crotch + BRIEFS_LEG * height,
              'briefs_trunk': crotch + BRIEFS_TRUNK * height}
    top = config.BODIES[body_id] == 'female'
    if top:
        breast = b.joint('breast.L')[2]
        result.update(top_low=breast + TOP_LOW * height, top_high=breast + TOP_HIGH * height)
    arm = b.region_weight['arm.left'] + b.region_weight['arm.right']
    leg = b.region_weight['leg.left'] + b.region_weight['leg.right']
    return {name: round(float(value), 5) for name, value in result.items()}, top, arm, leg


def _cache_key(body_id):
    own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(f'{file_sha256(config.body_file(body_id))}:{own}'.encode()).hexdigest()


def paint(body_id, cache=None):
    """The paint record for one body: bands, whether it has a top, the colour, and per-vertex arm and leg
    weights (in the body file's vertex order, which every shown BODY.glb keeps)."""
    folder = Path(cache) if cache is not None else config.runs_folder()/'_cache'/'underwear'
    path = folder/f'{_cache_key(body_id)}.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    heights, top, arm, leg = bands(body_id)
    record = {'body': body_id, 'bands': heights, 'top': top, 'colour': list(COLOUR),
              'arm_limit': ARM_LIMIT, 'leg_limit': LEG_LIMIT,
              'arm_roots': [bodies.LIMB_ROOTS['arm.left'], bodies.LIMB_ROOTS['arm.right']],
              'leg_roots': [bodies.LIMB_ROOTS['leg.left'], bodies.LIMB_ROOTS['leg.right']],
              'vertex_arm': [round(float(w), 3) for w in arm], 'vertex_leg': [round(float(w), 3) for w in leg],
              'note': 'display only (changes for this build); never part of a fit'}
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding='utf-8')
    return record


def painted(record, z, arm, leg):
    """Whether a point at rest height z with these arm and leg weights is painted (the rule both renderers use)."""
    if arm >= record['arm_limit']:
        return False
    b = record['bands']
    low = b['briefs_leg'] if leg > record['leg_limit'] else b['briefs_trunk']
    if low < z < b['briefs_top']:
        return True
    return bool(record['top']) and b['top_low'] < z < b['top_high']


def write_all(path, cache=None):
    """One JSON file holding every body's record (body id -> record), for the Blender scripts."""
    path = Path(path)
    path.write_text(json.dumps({body_id: paint(body_id, cache) for body_id in config.BODIES}), encoding='utf-8')
    return path
