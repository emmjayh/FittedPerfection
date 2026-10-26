"""Read skinned meshes from .glb files with numpy only (no Blender).

Covers what FittedPerfection needs from a body or a fitted draft: positions,
triangles, skin joints and weights, and the skeleton (bone names, parents,
rest-pose world matrices). glTF is +Y up, metres; bodies face +Z.
"""
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
_NORMALIZED_MAX = {np.uint8: 255.0, np.uint16: 65535.0, np.int8: 127.0, np.int16: 32767.0}


class GLBError(ValueError):
    pass


@dataclass
class SkinnedMesh:
    positions: np.ndarray        # (V, 3) float64, bind pose
    triangles: np.ndarray        # (T, 3) int64
    joints: np.ndarray           # (V, K) int64, indices into bone_names
    weights: np.ndarray          # (V, K) float64
    bone_names: list
    bone_parents: np.ndarray     # (B,) int64, -1 for roots
    bone_matrices: np.ndarray    # (B, 4, 4) float64, rest-pose world matrices

    @property
    def bone_heads(self):
        return self.bone_matrices[:, :3, 3]


def read_glb(path):
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from('<4sII', data, 0)
    if magic != b'glTF' or version != 2 or length != len(data):
        raise GLBError(f'{path}: not a glTF 2.0 binary')
    offset, document, binary = 12, None, b''
    while offset < length:
        size, kind = struct.unpack_from('<II', data, offset)
        chunk = data[offset + 8: offset + 8 + size]
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + size
    if document is None:
        raise GLBError(f'{path}: no JSON chunk')
    return document, binary


def _accessor(document, binary, index):
    accessor = document['accessors'][index]
    view = document['bufferViews'][accessor['bufferView']]
    dtype = np.dtype(_COMPONENTS[accessor['componentType']])
    width = _WIDTH[accessor['type']]
    count = accessor['count']
    start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    stride = view.get('byteStride') or dtype.itemsize * width
    rows = np.ndarray((count, width), dtype=dtype, buffer=binary, offset=start, strides=(stride, dtype.itemsize))
    values = rows.astype(np.float64 if accessor.get('normalized') or dtype == np.float32 else np.int64)
    if accessor.get('normalized'):
        values /= _NORMALIZED_MAX[dtype.type]
    return values


def _local_matrix(node):
    if 'matrix' in node:
        return np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
    x, y, z, w = node.get('rotation', (0.0, 0.0, 0.0, 1.0))
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array(node.get('scale', (1.0, 1.0, 1.0)))
    matrix[:3, 3] = node.get('translation', (0.0, 0.0, 0.0))
    return matrix


def _world_matrices(document):
    nodes = document.get('nodes', [])
    parent = {child: index for index, node in enumerate(nodes) for child in node.get('children', ())}
    world = {}

    def resolve(index):
        if index not in world:
            local = _local_matrix(nodes[index])
            world[index] = resolve(parent[index]) @ local if index in parent else local
        return world[index]

    for index in range(len(nodes)):
        resolve(index)
    return world, parent


def read_skinned_mesh(path):
    """The single skinned mesh in a .glb (bodies and drafts each hold exactly one)."""
    document, binary = read_glb(path)
    skinned = [node for node in document.get('nodes', []) if 'mesh' in node and 'skin' in node]
    if len(skinned) != 1:
        raise GLBError(f'{path}: expected one skinned mesh, found {len(skinned)}')
    node = skinned[0]
    primitives = document['meshes'][node['mesh']]['primitives']
    if len(primitives) != 1:
        raise GLBError(f'{path}: expected one primitive, found {len(primitives)}')
    attributes = primitives[0]['attributes']
    positions = _accessor(document, binary, attributes['POSITION'])
    triangles = _accessor(document, binary, primitives[0]['indices']).reshape(-1, 3)
    sets = sorted(int(key.split('_')[1]) for key in attributes if key.startswith('JOINTS_'))
    joints = np.hstack([_accessor(document, binary, attributes[f'JOINTS_{i}']) for i in sets]).astype(np.int64)
    weights = np.hstack([_accessor(document, binary, attributes[f'WEIGHTS_{i}']) for i in sets])

    skin = document['skins'][node['skin']]
    joint_nodes = skin['joints']
    world, parent = _world_matrices(document)
    position_in_skin = {node_index: i for i, node_index in enumerate(joint_nodes)}
    parents = np.array([position_in_skin.get(parent.get(n), -1) for n in joint_nodes], dtype=np.int64)
    return SkinnedMesh(
        positions=positions,
        triangles=triangles,
        joints=joints,
        weights=weights,
        bone_names=[document['nodes'][n].get('name', f'node{n}') for n in joint_nodes],
        bone_parents=parents,
        bone_matrices=np.stack([world[n] for n in joint_nodes]),
    )


@dataclass
class TexturedMesh:
    positions: np.ndarray        # (V, 3) float64, canonical frame: Z up, front -Y, own left +X
    triangles: np.ndarray        # (T, 3) int64
    uvs: np.ndarray              # (V, 2) float64, glTF convention (v down from the top of the image)
    texture_png: bytes | None    # the base-colour image, if embedded


def gltf_to_canonical(points):
    """glTF (+Y up, front +Z) -> the canonical Blender frame (+Z up, front -Y): (x, y, z) -> (x, -z, y)."""
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])


def read_textured_mesh(path):
    """A plain (unskinned) single-primitive mesh, such as an intake GARMENT.glb or CAGE.glb."""
    document, binary = read_glb(path)
    nodes = [node for node in document.get('nodes', []) if 'mesh' in node]
    if len(nodes) != 1:
        raise GLBError(f'{path}: expected one mesh, found {len(nodes)}')
    world, _ = _world_matrices(document)
    matrix = world[document['nodes'].index(nodes[0])]
    primitives = document['meshes'][nodes[0]['mesh']]['primitives']
    if len(primitives) != 1:
        raise GLBError(f'{path}: expected one primitive, found {len(primitives)}')
    primitive = primitives[0]
    attributes = primitive['attributes']
    positions = _accessor(document, binary, attributes['POSITION'])
    positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
    texture = None
    material = document.get('materials', [{}])[primitive['material']] if 'material' in primitive else {}
    colour = material.get('pbrMetallicRoughness', {}).get('baseColorTexture')
    if colour is not None:
        image = document['images'][document['textures'][colour['index']]['source']]
        view = document['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        texture = bytes(binary[start:start + view['byteLength']])
    return TexturedMesh(
        positions=gltf_to_canonical(positions),
        triangles=_accessor(document, binary, primitive['indices']).reshape(-1, 3),
        uvs=_accessor(document, binary, attributes['TEXCOORD_0']) if 'TEXCOORD_0' in attributes else
        np.zeros((len(positions), 2)),
        texture_png=texture,
    )
