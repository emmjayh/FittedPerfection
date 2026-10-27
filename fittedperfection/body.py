"""The target bodies as the fit sees them: mesh, rig, regions, skinning and landmarks.

Everything is in the canonical frame (Z up, front -Y, own left +X). Regions come
exactly from the skin weights: a vertex belongs to the arm or leg whose bones
pull it most, otherwise to the trunk. Landmarks are measured with the same
functions bind uses on garments (the body's regions standing in for garment
labels), so body and garment landmarks mean the same thing.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import bind, config, label, meshes, settings
from .gltf import gltf_to_canonical, read_skinned_mesh

LIMB_ROOTS = {'arm.left': 'upperarm01.L', 'arm.right': 'upperarm01.R',
              'leg.left': 'upperleg01.L', 'leg.right': 'upperleg01.R'}
STAND_IN_LABEL = {'arm.left': 'sleeve.left', 'arm.right': 'sleeve.right',
                  'leg.left': 'pant_leg.left', 'leg.right': 'pant_leg.right', 'trunk': 'torso'}
_TO_CANONICAL = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


@dataclass
class Body:
    id: str
    positions: np.ndarray        # (V, 3) rest pose, canonical
    triangles: np.ndarray        # (T, 3)
    joints: np.ndarray           # (V, K) bone indices
    weights: np.ndarray          # (V, K)
    bone_names: list
    parents: np.ndarray
    rest: np.ndarray             # (B, 4, 4) rest-pose world matrices, canonical
    bone_region: np.ndarray      # (B,) region of each bone
    vertex_region: np.ndarray    # (V,) region with the largest total weight
    region_weight: dict = field(default_factory=dict)   # region -> (V,) total weight of its bones

    def joint(self, name):
        return self.rest[self.bone_names.index(name)][:3, 3].copy()

    @property
    def height(self):
        return float(self.positions[:, 2].max() - self.positions[:, 2].min())


def _descendants(parents, root):
    found, frontier = set(), [root]
    while frontier:
        bone = frontier.pop()
        found.add(bone)
        frontier += [i for i, p in enumerate(parents) if p == bone]
    return found


@lru_cache(maxsize=None)
def load(body_id):
    mesh = read_skinned_mesh(config.body_file(body_id))
    names = mesh.bone_names
    bone_region = np.array(['trunk'] * len(names), dtype=object)
    for region, root in LIMB_ROOTS.items():
        for bone in _descendants(mesh.bone_parents, names.index(root)):
            bone_region[bone] = region
    regions = ('trunk',) + tuple(LIMB_ROOTS)
    per_vertex_region = bone_region[mesh.joints]                          # (V, K)
    region_weight = {r: (mesh.weights * (per_vertex_region == r)).sum(axis=1) for r in regions}
    stacked = np.stack([region_weight[r] for r in regions], axis=1)
    rest = np.einsum('ij,bjk,lk->bil', _TO_CANONICAL, mesh.bone_matrices, _TO_CANONICAL)
    return Body(id=body_id, positions=gltf_to_canonical(mesh.positions), triangles=mesh.triangles,
                joints=mesh.joints, weights=mesh.weights, bone_names=names, parents=mesh.bone_parents, rest=rest,
                bone_region=bone_region, vertex_region=np.array(regions, dtype=object)[np.argmax(stacked, axis=1)],
                region_weight=region_weight)


def skin(body, transforms, positions=None):
    """Linear-blend skinning: each vertex moves by the weighted blend of its bones' world transforms (B, 4, 4)."""
    points = body.positions if positions is None else positions
    blended = np.einsum('vk,vkij->vij', body.weights, transforms[body.joints])     # (V, 4, 4)
    return np.einsum('vij,vj->vi', blended[:, :3, :3], points) + blended[:, :3, 3]


def vertex_matrices(body, transforms):
    """Each vertex's blended 4x4 transform (for moving nearby points the way the body moves)."""
    return np.einsum('vk,vkij->vij', body.weights, transforms[body.joints])


def face_regions(body):
    votes = body.vertex_region[body.triangles]
    out = votes[:, 0].copy()
    out[(votes[:, 1] == votes[:, 2]) & (votes[:, 0] != votes[:, 1])] = votes[:, 1][(votes[:, 1] == votes[:, 2]) &
                                                                                (votes[:, 0] != votes[:, 1])]
    return out


def landmarks(body, positions=None):
    """Body landmarks measured with bind's own garment functions (regions stand in for labels)."""
    positions = body.positions if positions is None else positions
    labels = np.array([STAND_IN_LABEL[r] for r in face_regions(body)], dtype=object).astype(str)
    centres = positions[body.triangles].mean(axis=1)
    _, areas = meshes.face_normals(positions, body.triangles)
    graph = meshes.face_graph(positions, body.triangles)
    height = float(positions[:, 2].max() - positions[:, 2].min())
    config_bind = settings.section('bind')
    torso_area = float(areas[labels == 'torso'].sum())
    arms = {side: bind.sleeve_chain(side, labels, centres, areas, graph, torso_area, config_bind)
            for side in ('left', 'right')}
    shoulders = bind.shoulders(arms, labels, centres, areas, height)
    # The sleeveless measure too, for garments bind measured that way: the top of the trunk over each
    # shoulder. The neck and head are not cloth-bearing torso there, so they are left out.
    neck = _descendants(body.parents, body.bone_names.index('neck01'))
    head_weight = (body.weights * np.isin(body.joints, list(neck))).sum(axis=1)
    no_head = labels.copy()
    no_head[head_weight[body.triangles].mean(axis=1) >= 0.5] = 'none'
    shoulder_top = bind.shoulders({}, no_head, centres, areas, height)
    crotch = label.crotch_height(centres, areas, graph, height)
    if crotch is None:
        raise ValueError(f'{body.id}: could not find where the legs branch')
    legs = {side: bind.leg_chain(side, labels, centres, areas, crotch, config_bind) for side in ('left', 'right')}
    return {'arms': arms, 'shoulders': shoulders, 'shoulder_top': shoulder_top, 'crotch_height_m': crotch, 'legs': legs,
            'height_m': height, 'floor_m': float(positions[:, 2].min())}
