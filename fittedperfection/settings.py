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
        # Smoothing over the face graph: links across source seams and creases sharper than
        # crease_degrees are weakened; faces seen with half_trust_pixels pixels are half trusted.
        'smoothing': 1.0,
        'seam_weight': 0.1,
        'crease_weight': 0.3,
        'crease_degrees': 40.0,
        'half_trust_pixels': 20.0,
        'min_score': 0.15,                 # below this best score a face stays 'none'
        'shell_coverage': 0.7,             # torso slices below the hips wrapping less than this hang
        # Skin colour check: the limit is spread_factor x the median distance of the upper-body
        # reference skin from its median tone, and never below min_distance (RGB units).
        'skin_colour_min_distance': 25.0,
        'skin_colour_spread_factor': 4.0,
        'confident_skin_score': 0.5,
        'leg_covered_share': 0.6,          # a leg band this covered by legwear/footwear turns its skin into it
        'limb_island_fraction': 0.05,      # islands smaller than this share of a limb take the limb's label
        'largest_unlabelled_share': 0.05,  # unidentified pieces bigger than this share stay 'none'
    },
    'bind': {
        'min_sleeve_fragment_of_torso': 0.03,
        'crotch_prior_of_shoulder_height': 0.59,
        'crotch_prior_tolerance': 0.07,
        'hem_fraction': {'shorts': 0.30, 'knee': 0.60, 'cropped': 0.78, 'full': 1.0},
        'section_step': 0.03,              # metres between centreline sections
        'knee_of_height': 0.28,            # expected knee height as a share of the garment height
        'knee_window_of_height': 0.06,
        'knee_min_narrowing': 0.05,        # the knee section must be this much narrower than its surroundings
        'knee_max_side_difference': 0.04,  # both legs' knees must agree within this (metres)
        'sole_band_of_height': 0.02,
        'toe_forward_min_dot': 0.7,        # toes must point within ~45 degrees of the front
        'toe_over_heel_min': 1.4,          # shoe-only toe/heel reach needed to confirm a provisional facing
    },
    'fit': {
        'leg_length_range': [0.85, 1.6],   # spec 0.85-1.2; raised to what bind's crotch band allows (see SPEC changes)
        'girth_range': [0.85, 1.15],
        'neck_length_range': [1.0, 1.6],   # donor neck may lengthen to the garment's neckline, never shorten
        'transfer_neighbours': 32,
        'unpose_neighbours': 16,
        'seam_blend_rings': 6,             # how far either side of a label seam the field is blended
        'seam_blend_iterations': 60,
        'transfer_sigma_of_nearest': 2.0,
        'transfer_sigma_of_height': 0.03,
        'rigid_max_rotation_degrees': 25.0,
        'contact_ease_of_height': 0.006,
        'wrap_keep': 0.3,                  # share of the cloth's overall looseness kept by the wrap (relief kept whole)
        'wrap_bulk_radius_m': 0.06,        # looseness is the height above the skin smoothed over this radius
        'wrap_target_weight': 0.5,
        'contact_weight': 5.0,
        'contact_rounds': 4,
        'arap_iterations': 8,
        'foot_length_of_height': 0.15,     # the foot a source shoe was made for, as a share of the figure's height
        'shoe_shaft_above_ankle_m': 0.03,  # footwear higher than this above the ankle is a shaft (wrapped, not rigid)
        'shoe_shaft_min_m': 0.15,          # ...when the footwear rises at least this far above that line (a boot)
        'shoe_sole_search_m': 0.08,        # the sole's edge is looked for this high from the bottom of the shoe
        'ghost_grid_m': 0.005,             # raster of the shoe height map used to hide covered body faces
        'rotation_smoothing_iterations': 3,  # smoothing of the per-vertex rotation field carrying full-res detail
        'fullres_scale_range': [0.5, 2.0],   # local scale applied to full-res offsets, clamped
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
