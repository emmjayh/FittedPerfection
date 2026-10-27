"""Steps F and G (spec §5.6): the full-resolution outfit, rebuilt on each fitted cage and skinned to each body.

G1 (once per outfit): every full-resolution face takes the label of its nearest cage face, and every
   welded full-resolution vertex its majority label (UV-split twins always agree). Each vertex binds
   to its closest cage triangle of the same label (barycentric coordinates plus the offset from that
   point), so a trouser hem never catches the boot under it.
G2 (per body): a smooth per-vertex rotation field and a local scale from source cage to fitted cage
   carry each offset: vertex = barycentric point on the fitted triangle + rotation * scale * offset.
   Rigid groups use their exact source-to-fitted similarity, so they stay exact copies.
F  (per body): weights from the body's own rig, blended from its closest skin triangle in the
   vertex's region, top 4 normalised. Rigid groups share one weight set.
numpy/scipy only (sealed).
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree

from . import meshes, wrap

FALLBACK_MARGIN_M = 0.03          # a same-label face may be this much further than the nearest face of any label


def vertex_labels(face_labels, triangles, welded):
    """Majority label of the faces around each welded vertex."""
    names, codes = np.unique(face_labels, return_inverse=True)
    count = welded.max() + 1
    votes = np.zeros((count, len(names)), dtype=np.int32)
    for corner in range(3):
        np.add.at(votes, (welded[triangles[:, corner]], codes), 1)
    return names[np.argmax(votes, axis=1)]


def transfer_labels(points, triangles, cage_points, cage_triangles, cage_labels):
    """Label of each full-resolution face: that of the nearest cage face (by centre)."""
    nearest = cKDTree(cage_points[cage_triangles].mean(axis=1)).query(points[triangles].mean(axis=1))[1]
    return cage_labels[nearest]


def bind(points, labels, cage_points, cage_triangles, cage_labels):
    """Per point: cage triangle, barycentric coordinates and offset (point minus its closest point on that
    triangle), preferring the closest triangle of the point's own label."""
    every = np.arange(len(cage_triangles))
    centres = cage_points[cage_triangles].mean(axis=1)
    q, bary, face, distance = wrap.closest_on_mesh(points, cage_points, cage_triangles, every, cKDTree(centres))
    for name in np.unique(labels):
        chosen = np.flatnonzero(labels == name)
        own = np.flatnonzero(cage_labels == name)
        if not len(own):
            continue
        q_own, bary_own, face_own, d_own = wrap.closest_on_mesh(points[chosen], cage_points, cage_triangles, own,
                                                                cKDTree(centres[own]))
        use = d_own <= distance[chosen] + FALLBACK_MARGIN_M
        sel = chosen[use]
        q[sel], bary[sel], face[sel], distance[sel] = q_own[use], bary_own[use], face_own[use], d_own[use]
    return {'face': face, 'bary': bary, 'offset': points - q, 'distance': distance}


def polar(matrices):
    """Nearest rotations to a stack of 3x3 matrices."""
    u, _, vt = np.linalg.svd(matrices)
    rotation = u @ vt
    flip = np.linalg.det(rotation) < 0
    if flip.any():
        u[flip, :, 2] *= -1
        rotation[flip] = u[flip] @ vt[flip]
    return rotation


def rotation_field(source, fitted, triangles, iterations):
    """Per cage vertex, the rotation best mapping its source edges to its fitted edges, smoothed over neighbours."""
    count = len(source)
    edges = wrap.edges_of(triangles)
    rotation = wrap.best_rotations(source, fitted, edges, count)
    i, j = edges[:, 0], edges[:, 1]
    adjacency = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(count, count)).tocsr()
    average = diags(1.0 / np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)) @ adjacency
    for _ in range(iterations):
        rotation = polar(0.5 * rotation + 0.5 * (average @ rotation.reshape(count, 9)).reshape(count, 3, 3))
    return rotation


def scale_field(source, fitted, triangles, scale_range):
    """Per cage vertex, the local scale: square root of the area ratio of the faces around it, clamped."""
    _, area_source = meshes.face_normals(source, triangles)
    _, area_fitted = meshes.face_normals(fitted, triangles)
    total_source, total_fitted = np.zeros(len(source)), np.zeros(len(source))
    for corner in range(3):
        np.add.at(total_source, triangles[:, corner], area_source)
        np.add.at(total_fitted, triangles[:, corner], area_fitted)
    return np.clip(np.sqrt(total_fitted / np.maximum(total_source, 1e-20)), *scale_range)


def similarity(source, moved):
    """Scale s, rotation R and shift t with moved ~ s R source + t (least squares)."""
    a, b = source.mean(axis=0), moved.mean(axis=0)
    x, y = source - a, moved - b
    u, sigma, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1, 1, d]) @ u.T
    scale = (sigma * [1, 1, d]).sum() / max((x * x).sum(), 1e-20)
    return scale, rotation, b - scale * rotation @ a


def rebuild(binding, source_points, cage_source, cage_fitted, cage_triangles, rotation, scale, rigid):
    """Full-resolution points on the fitted cage. rigid: list of cage vertex index arrays, each one rigid group."""
    corners = cage_triangles[binding['face']]
    bary = binding['bary']
    base = np.einsum('pk,pkj->pj', bary, cage_fitted[corners])
    blended = polar(np.einsum('pk,pkij->pij', bary, rotation[corners]))
    local = np.einsum('pk,pk->p', bary, scale[corners])
    out = base + local[:, None] * np.einsum('pij,pj->pi', blended, binding['offset'])
    group = rigid_face_groups(cage_triangles, rigid, len(cage_source))[binding['face']]
    for g, vertices in enumerate(rigid):
        chosen = group == g
        if chosen.any():
            s, r, t = similarity(cage_source[vertices], cage_fitted[vertices])
            out[chosen] = s * source_points[chosen] @ r.T + t
    return out, group


def rigid_face_groups(triangles, rigid, count):
    """Rigid group of each cage face (all three corners in the group), or -1."""
    owner = np.full(count, -1)
    for g, vertices in enumerate(rigid):
        owner[vertices] = g
    corners = owner[triangles]
    return np.where((corners[:, 0] >= 0) & (corners == corners[:, :1]).all(axis=1), corners[:, 0], -1)


def skin_weights(points, faces_of_point, skin, body_joints, body_weights, groups=None, influences=4):
    """Per point: the top `influences` joints and normalised weights, blended from the closest skin triangle of
    its region. groups (optional, -1 for none): points of one group share the group's summed weights."""
    count = len(points)
    face, bary = np.zeros(count, dtype=np.int64), np.zeros((count, 3))
    for name, chosen, faces in faces_of_point:
        _, bary[chosen], face[chosen], _ = skin.closest(points[chosen], faces, name)
    corners = skin.triangles[face]                                            # (P, 3)
    joints = body_joints[corners].reshape(count, -1)                          # (P, 3K)
    weights = (body_weights[corners] * bary[:, :, None]).reshape(count, -1)
    owner = np.arange(count) if groups is None else np.where(groups >= 0, count + groups, np.arange(count))
    return top_influences(owner, joints, weights, count, influences)


def top_influences(owner, joints, weights, count, influences):
    """Sum weights per (owner, joint), keep each owner's largest `influences`, normalise; rows read their owner."""
    bones = int(joints.max()) + 1
    rows = np.repeat(owner, joints.shape[1])
    keys = rows.astype(np.int64) * bones + joints.ravel()
    unique, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=weights.ravel())
    key_owner, key_joint = unique // bones, unique % bones
    order = np.lexsort((-sums, key_owner))
    key_owner, key_joint, sums = key_owner[order], key_joint[order], sums[order]
    start = np.r_[0, np.flatnonzero(np.diff(key_owner)) + 1]
    rank = np.arange(len(key_owner)) - np.repeat(start, np.diff(np.r_[start, len(key_owner)]))
    keep = rank < influences
    owners = np.unique(key_owner)
    slot = np.searchsorted(owners, key_owner[keep])
    out_joints = np.zeros((len(owners), influences), dtype=np.int64)
    out_weights = np.zeros((len(owners), influences))
    out_joints[slot, rank[keep]] = key_joint[keep]
    out_weights[slot, rank[keep]] = sums[keep]
    out_weights /= np.maximum(out_weights.sum(axis=1, keepdims=True), 1e-20)
    row = np.searchsorted(owners, owner)
    return out_joints[row], out_weights[row]
