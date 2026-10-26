"""Triangle-mesh helpers for labelling and binding: welding, face adjacency, seams, creases, parts.

glTF splits vertices wherever UVs or normals jump, so faces are first welded by
position to find which faces really touch. An edge whose two faces used
different (split) vertices is a seam in the source mesh: a UV or shading break.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class FaceGraph:
    pairs: np.ndarray          # (E, 2) face indices sharing an edge
    seam: np.ndarray           # (E,) True where the shared edge is split in the source (UV/shading seam)
    angle_deg: np.ndarray      # (E,) angle between the two face normals
    face_count: int


def weld(positions, decimals=6):
    """Welded vertex id for every vertex (same position -> same id)."""
    _, ids = np.unique(np.round(positions, decimals), axis=0, return_inverse=True)
    return ids.ravel()


def face_normals(positions, triangles):
    a, b, c = (positions[triangles[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    length = np.linalg.norm(cross, axis=1, keepdims=True)
    return cross / np.maximum(length, 1e-20), 0.5 * length.ravel()


def face_graph(positions, triangles):
    welded = weld(positions)
    normals, _ = face_normals(positions, triangles)
    corners = [(0, 1), (1, 2), (2, 0)]
    ends = np.concatenate([welded[triangles[:, [i, j]]] for i, j in corners])
    raw = np.concatenate([triangles[:, [i, j]] for i, j in corners])
    faces = np.tile(np.arange(len(triangles)), 3)
    order = np.argsort(ends, axis=1)
    ends = np.take_along_axis(ends, order, axis=1)
    raw = np.take_along_axis(raw, order, axis=1)
    keys = ends[:, 0].astype(np.int64) * (welded.max() + 1) + ends[:, 1]
    sort = np.argsort(keys, kind='stable')
    keys, raw, faces = keys[sort], raw[sort], faces[sort]
    pairs, seams = [], []
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    stops = np.r_[starts[1:], len(keys)]
    for start, stop in zip(starts[stops - starts >= 2], stops[stops - starts >= 2]):
        for i in range(start, stop):                     # every pair, so non-manifold edges connect too
            for j in range(i + 1, stop):
                if faces[i] != faces[j]:
                    pairs.append((faces[i], faces[j]))
                    seams.append(not np.array_equal(raw[i], raw[j]))
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    cosine = np.clip((normals[pairs[:, 0]] * normals[pairs[:, 1]]).sum(axis=1), -1, 1)
    return FaceGraph(pairs=pairs, seam=np.array(seams, dtype=bool), angle_deg=np.degrees(np.arccos(cosine)),
                     face_count=len(triangles))


def parts(graph, mask):
    """Connected pieces among the faces where mask is True: (part id per face or -1, number of parts)."""
    keep = mask[graph.pairs[:, 0]] & mask[graph.pairs[:, 1]]
    edges = graph.pairs[keep]
    n = graph.face_count
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    count, labels = connected_components(adjacency, directed=False)
    labels = np.where(mask, labels, -1)
    _, labels = np.unique(labels, return_inverse=True)          # renumber 0.. over the masked faces
    labels = labels.reshape(-1) - (1 if (~mask).any() else 0)
    return np.where(mask, labels, -1), int(labels.max() + 1) if mask.any() else 0
