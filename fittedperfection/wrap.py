"""Step D (spec §5.6): one coordinated cloth solve that wraps the cloth close to the skin and keeps its shape.

After Step C every cage point has a height above the target skin, h (signed, along the
direction to its closest skin point). h splits into
  bulk:   h smoothed over the cloth surface (how loose the garment hangs), and
  relief: h - bulk (folds, pockets, ruffles, cuffs: the design's surface shape).
The wrapped height keeps all the relief and only part of the looseness:
  h' = max(ease + keep * max(bulk - ease, 0) + relief, ease)
Shrinking proportionally keeps layers in order; no target is closer to the skin than ease, so
cloth found inside the body (not relief) comes out.
Skirts, hanging cloth and unknown parts keep their bulk (keep = 1), and so does cloth whose closest
skin is bare by nature (head, hands): a high collar stays a collar, not a face mask.

One ARAP-style system then balances, in the same solve: the local shape of the Step C result
(edges and rotations), the wrap targets, contact springs for points and triangles closer to the
skin than ease (a face centre that is too close pushes its corners out; active set renewed each
round), and hard anchors for rigid groups. No repair passes.
numpy/scipy only (sealed).
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

from . import meshes

KEEP_BULK = ('skirt', 'hanging', 'none')     # cloth that hangs free keeps its looseness (contact only)
HARD = 1e4                                   # weight of a hard anchor relative to the shape term
CANDIDATES = 16                              # skin triangles examined per point


def closest_on_triangles(p, a, b, c):
    """Closest point on each triangle (a, b, c) to each p (Ericson), and its barycentric coordinates."""
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        total = va + vb + vc
        v, w = vb / total, vc / total                                   # inside the face
        rules = [((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), None, 'bc', (d4 - d3) / ((d4 - d3) + (d5 - d6))),
                 ((vb <= 0) & (d2 >= 0) & (d6 <= 0), None, 'ac', d2 / (d2 - d6)),
                 ((d6 >= 0) & (d5 <= d6), (0.0, 1.0), None, None),
                 ((vc <= 0) & (d1 >= 0) & (d3 <= 0), None, 'ab', d1 / (d1 - d3)),
                 ((d3 >= 0) & (d4 <= d3), (1.0, 0.0), None, None),
                 ((d1 <= 0) & (d2 <= 0), (0.0, 0.0), None, None)]
        for mask, corner, edge, t in rules:                             # later rules take priority (Ericson's order)
            if corner is not None:
                v, w = np.where(mask, corner[0], v), np.where(mask, corner[1], w)
            elif edge == 'bc':
                v, w = np.where(mask, 1 - t, v), np.where(mask, t, w)
            elif edge == 'ac':
                v, w = np.where(mask, 0.0, v), np.where(mask, t, w)
            else:
                v, w = np.where(mask, t, v), np.where(mask, 0.0, w)
    v, w = np.nan_to_num(v), np.nan_to_num(w)
    return a + ab * v[:, None] + ac * w[:, None], np.column_stack([1 - v - w, v, w])


def closest_on_mesh(points, positions, triangles, faces, tree, k=CANDIDATES):
    """Closest point on the given faces of a mesh (tree: a cKDTree of those faces' centres), checking the k
    nearest by centre exactly: point, barycentric coordinates, face index, distance."""
    k = min(k, len(faces))
    _, nearest = tree.query(points, k=k)
    nearest = faces[nearest.reshape(len(points), k)]
    best_q, best_bary, best_face = None, None, None
    best_d = np.full(len(points), np.inf)
    for column in range(k):
        tri = triangles[nearest[:, column]]
        q, bary = closest_on_triangles(points, *(positions[tri[:, i]] for i in range(3)))
        d = np.linalg.norm(points - q, axis=1)
        better = d < best_d
        if best_q is None:
            best_q, best_bary, best_face = q, bary, nearest[:, column].copy()
        best_q[better], best_bary[better], best_face[better] = q[better], bary[better], nearest[better, column]
        best_d = np.minimum(best_d, d)
    return best_q, best_bary, best_face, best_d


class Skin:
    """A body surface for closest-point queries, optionally restricted to the faces of a region."""

    def __init__(self, positions, triangles, bare=None):
        self.positions, self.triangles = positions, triangles
        self.bare = np.zeros(len(triangles), dtype=bool) if bare is None else bare     # head and hands
        normals, areas = meshes.face_normals(positions, triangles)
        vertex_normals = np.zeros_like(positions)
        for corner in range(3):
            np.add.at(vertex_normals, triangles[:, corner], normals * areas[:, None])
        self.vertex_normals = vertex_normals / np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12)
        self.centres = positions[triangles].mean(axis=1)
        self._trees = {}

    def closest(self, points, faces=None, key='all'):
        """Closest point on the skin (only `faces`, cached under `key`): point, barycentric coordinates, face, distance."""
        faces = np.arange(len(self.triangles)) if faces is None else faces
        if key not in self._trees:
            self._trees[key] = cKDTree(self.centres[faces])
        return closest_on_mesh(points, self.positions, self.triangles, faces, self._trees[key])

    def heights(self, points, faces=None, key='all'):
        """Signed height of each point above the skin (only `faces`, cached under `key`), its closest skin
        point, the outward unit direction and the skin face it lies on."""
        best_q, best_bary, best_face, best_d = self.closest(points, faces, key)
        tri = self.triangles[best_face]
        normal = np.einsum('pk,pkj->pj', best_bary, self.vertex_normals[tri])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        offset = points - best_q
        sign = np.where(np.einsum('ij,ij->i', offset, normal) < 0, -1.0, 1.0)
        direction = np.where(best_d[:, None] > 1e-6, sign[:, None] * offset / np.maximum(best_d[:, None], 1e-12), normal)
        return sign * best_d, best_q, direction, best_face


def point_keep(labels, triangles, count, keep):
    """Per point: the share of its looseness it keeps (`keep`, or 1 for free-hanging cloth)."""
    free = np.isin(np.char.partition(labels.astype(str), '.')[:, 0], KEEP_BULK).astype(float)
    votes, total = np.zeros(count), np.zeros(count)
    for corner in range(3):
        np.add.at(votes, triangles[:, corner], free)
        np.add.at(total, triangles[:, corner], 1.0)
    return np.where(votes / np.maximum(total, 1) > 0.5, 1.0, keep)


def edges_of(triangles):
    pairs = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0)


def smooth(values, edges, count, radius, edge_length):
    """Diffuse a per-point value over the surface to about `radius` (uniform neighbour averaging)."""
    i, j = edges[:, 0], edges[:, 1]
    adjacency = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(count, count)).tocsr()
    average = diags(1.0 / np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)) @ adjacency
    iterations = int(np.ceil((radius / max(edge_length, 1e-6)) ** 2))
    out = values.copy()
    for _ in range(iterations):
        out = 0.5 * out + 0.5 * (average @ out)                 # lazy averaging: no oscillation on odd cycles
    return out


def best_rotations(reference, current, edges, count):
    """Per point, the rotation that best maps its reference edges onto its current edges (ARAP local step)."""
    i, j = edges[:, 0], edges[:, 1]
    r, x = reference[i] - reference[j], current[i] - current[j]
    outer = np.einsum('ea,eb->eab', r, x)
    covariance = np.zeros((count, 3, 3))
    np.add.at(covariance, i, outer)
    np.add.at(covariance, j, outer)
    u, _, vt = np.linalg.svd(covariance)
    rotation = np.einsum('pji,pkj->pik', vt, u)                    # V U^T
    flip = np.linalg.det(rotation) < 0
    if flip.any():
        vt[flip, 2] *= -1
        rotation[flip] = np.einsum('pji,pkj->pik', vt[flip], u[flip])
    return rotation


def solve(start, triangles, keep, skin, faces_of_point, rigid, config, height, blend=None, printed=None):
    """Step D on one body. start: Step C points (also the shape reference); each point measures its height
    against its own region's skin.
    rigid: indices of hard-anchored points. faces_of_point: (region name, point indices, skin faces).
    blend: optional function that smooths a per-point field across label seams (regions measure against
    different skin, so their wrap moves must meet smoothly). printed: indices of points in printed designs; edges
    between them are `print_stiffness` times stiffer, so logos and text keep their shape while plain fabric
    takes up the wrap. Returns the solved points and a record."""
    count = len(start)
    edges = edges_of(triangles)
    edge_length = float(np.median(np.linalg.norm(start[edges[:, 0]] - start[edges[:, 1]], axis=1)))
    ease = config['contact_ease_of_height'] * height

    group = np.zeros(count, dtype=np.int64)
    for g, (_, chosen, _) in enumerate(faces_of_point):
        group[chosen] = g
    face_group = group[triangles[:, 0]]

    def measure(points, which):
        h, q, u, face = np.zeros(len(points)), np.zeros((len(points), 3)), np.zeros((len(points), 3)),             np.zeros(len(points), dtype=np.int64)
        for g, (name, _, faces) in enumerate(faces_of_point):
            chosen = np.flatnonzero(which == g)
            if len(chosen):
                h[chosen], q[chosen], u[chosen], face[chosen] = skin.heights(points[chosen], faces, name)
        return h, q, u, face

    def contact(points):
        """Per point: how far it must move out (and which way) so that it and every face around it clear ease."""
        h_p, _, u_p, _ = measure(points, group)
        h_f, _, u_f, _ = measure(points[triangles].mean(axis=1), face_group)
        deficit = np.r_[ease - h_p, np.tile(ease - h_f, 3)]
        direction = np.vstack([u_p, u_f, u_f, u_f])
        owner = np.r_[np.arange(count), triangles[:, 0], triangles[:, 1], triangles[:, 2]]
        order = np.argsort(deficit)                                       # the largest need is written last
        need, way = np.zeros(count), np.zeros((count, 3))
        need[owner[order]], way[owner[order]] = deficit[order], direction[order]
        return need, way, h_p

    h, q, u, face = measure(start, group)
    keep = np.where(skin.bare[face], 1.0, keep)
    bulk = smooth(h, edges, count, config['wrap_bulk_radius_m'], edge_length)
    keep = smooth(keep, edges, count, config['wrap_bulk_radius_m'], edge_length)   # a skirt eases into its bodice
    # never a target inside ease: cloth found inside the body is not relief to keep
    wrapped = np.maximum(ease + keep * np.maximum(bulk - ease, 0.0) + (h - bulk), ease)
    move = u * (wrapped - h)[:, None]
    if blend is not None:
        move = blend(move).reshape(count, 3)
    targets = start + move

    i, j = edges[:, 0], edges[:, 1]
    stiffness = np.ones(len(i))
    if printed is not None and len(printed):
        in_print = np.zeros(count, dtype=bool)
        in_print[printed] = True
        stiffness[in_print[i] & in_print[j]] = config['print_stiffness']
    laplacian = coo_matrix((np.r_[stiffness, stiffness], (np.r_[i, j], np.r_[j, i])), shape=(count, count)).tocsr()
    laplacian = diags(np.asarray(laplacian.sum(axis=1)).ravel()) - laplacian
    anchored = np.zeros(count, dtype=bool)
    anchored[rigid] = True
    w_target = np.where(anchored, 0.0, config['wrap_target_weight'])
    reference = start
    current = start.copy()
    active = np.zeros(count, dtype=bool)
    contact_goal = np.zeros((count, 3))
    for _ in range(config['contact_rounds']):
        need, way, _ = contact(current)
        active |= (need > 0) & ~anchored
        contact_goal[active] = current[active] + way[active] * np.maximum(need[active], 0.0)[:, None]
        w_contact = np.where(active, config['contact_weight'], 0.0)
        w_anchor = np.where(anchored, HARD, 0.0)
        weight = w_target + w_contact + w_anchor
        goal = (w_target[:, None] * targets + w_contact[:, None] * contact_goal + w_anchor[:, None] * start)
        factor = factorized((laplacian + diags(weight)).tocsc())
        for _ in range(config['arap_iterations']):
            rotation = best_rotations(reference, current, edges, count)
            d = reference[i] - reference[j]
            term = stiffness[:, None] * 0.5 * np.einsum('eab,eb->ea', rotation[i] + rotation[j], d)
            rhs = np.zeros((count, 3))
            np.add.at(rhs, i, term)
            np.add.at(rhs, j, -term)
            rhs += goal
            current = np.column_stack([factor(rhs[:, axis]) for axis in range(3)])
    _, _, h_end = contact(current)
    record = {'ease_m': round(ease, 4), 'keep': float(config['wrap_keep']), 'contact_points': int(active.sum()),
              'below_ease_at_end': int(((h_end < 0.5 * ease) & ~anchored).sum()),
              'inside_at_end': int(((h_end < 0) & ~anchored).sum())}
    return current, record


def offsets(points, skin, keep_points):
    """Offset statistics of the given points against the whole skin (for the report)."""
    h = skin.heights(points[keep_points])[0]
    p10, p50, p90 = np.percentile(h, [10, 50, 90])
    return {'inside_share': round(float((h < 0).mean()), 4), 'p10_cm': round(float(p10) * 100, 2),
            'median_cm': round(float(p50) * 100, 2), 'p90_cm': round(float(p90) * 100, 2)}
