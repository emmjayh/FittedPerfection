"""The one fit configuration (spec §3).

No profiles, candidates or opt-in flags: a value here is changed for
everyone, after the regression set has been reviewed, or not at all. Each
stage declares which sections it reads; their hashes go into the stage's
input record, so editing a section reruns exactly the stages that use it.
"""
import json
from types import MappingProxyType

from .hashing import json_sha256

_VALUES = {
    'intake': {
        'cage_faces': 40_000,
        # Canonical frame (spec §5.1): Z up, garment forward -Y, garment's own left +X.
        'frame': {'up': [0, 0, 1], 'forward': [0, -1, 0], 'left': [1, 0, 0]},
        # Facing from the feet (fittedperfection/orientation.py). Bands are fractions of height;
        # below min_ratio (but above provisional_min_ratio) the facing is provisional and bind
        # must confirm it from the footwear labels.
        'orientation': {'foot_band': 0.03, 'leg_band': [0.18, 0.26], 'min_leg_gap': 0.05, 'min_leg_clearance': 0.01,
                        'min_ratio': 1.4, 'provisional_min_ratio': 1.05},
    },
    'views': {
        # Turntable of orbit_frames around the vertical axis (frames 0, n/4, n/2, 3n/4 are front, left,
        # back, right) plus top and bottom stills, square images; the turntable also as an MP4 at fps.
        'image_size': 1024,
        'orbit_frames': 72,
        'fps': 24,
    },
    'vision': {
        'vocabulary_version': 5,
        'mask_encoding': 'one_bit',
        'include_confidence': True,
    },
    'label': {
        'crease_degrees': 40.0,
    },
    'bind': {
        'min_sleeve_fragment_of_torso': 0.03,
        'crotch_prior_of_shoulder_height': 0.59,
        'crotch_prior_tolerance': 0.07,
        'hem_fraction': {'shorts': 0.30, 'knee': 0.60, 'cropped': 0.78, 'full': 1.0},
    },
    'fit': {
        'leg_length_range': [0.85, 1.2],
        'girth_range': [0.85, 1.15],
        'transfer_neighbours': 32,
        'transfer_sigma_of_nearest': 2.0,
        'transfer_sigma_of_height': 0.03,
        'rigid_max_rotation_degrees': 25.0,
        'contact_ease_of_height': 0.006,
        'max_influences': 4,
    },
    'audit': {
        'stretch_limits': [3.0, 10.0],
        'synthetic_poses': {'elbow_degrees': [30, 60], 'knee_degrees': [15, 30]},
    },
}


def _read_only(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_read_only(item) for item in value)
    return value


SETTINGS = _read_only(json.loads(json.dumps(_VALUES)))
SECTIONS = tuple(SETTINGS)


def section(name):
    return SETTINGS[name]


def section_sha256(name):
    return json_sha256(_VALUES[name])
