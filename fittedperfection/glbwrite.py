"""Write simple .glb files with numpy only (the fit is sealed, so it cannot ask Blender).

One mesh, one primitive: positions given in the canonical frame (converted to glTF's
+Y up, front +Z), triangles, optional UVs with an embedded PNG base-colour texture,
or a flat base colour.
"""
import json
import struct
from pathlib import Path

import numpy as np


def canonical_to_gltf(points):
    """Canonical (+Z up, front -Y) -> glTF (+Y up, front +Z): (x, y, z) -> (x, z, -y)."""
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])


def _pad(data, filler=b'\x00'):
    return data + filler * ((4 - len(data) % 4) % 4)


def add_skin(document, accessors, attributes, add, skin, name):
    """Skin the mesh to a body's skeleton: its joint nodes are copied unchanged (same indices), the mesh node and
    the rig root follow, and a root node on top carries the lift (glTF +Y) so the figure stands on its soles."""
    from .gltf import _accessor, read_glb                      # reading only; avoids a cycle at import time
    body, body_binary = read_glb(skin['skeleton'])
    body_skin = body['skins'][0]
    old = list(body_skin['joints'])
    new_index = {node: i for i, node in enumerate(old)}         # joint nodes are written in skin order
    joints = list(range(len(old)))
    inverse = _accessor(body, body_binary, body_skin['inverseBindMatrices']).astype(np.float32)
    accessors.append({'bufferView': add(np.asarray(skin['joints'], dtype=np.uint16).tobytes(), 34962),
                      'componentType': 5123, 'count': len(skin['joints']), 'type': 'VEC4'})
    attributes['JOINTS_0'] = len(accessors) - 1
    accessors.append({'bufferView': add(np.asarray(skin['weights'], dtype=np.float32).tobytes(), 34962),
                      'componentType': 5126, 'count': len(skin['weights']), 'type': 'VEC4'})
    attributes['WEIGHTS_0'] = len(accessors) - 1
    accessors.append({'bufferView': add(inverse.tobytes()), 'componentType': 5126, 'count': len(inverse), 'type': 'MAT4'})
    nodes = []
    for node in old:
        copy = {k: v for k, v in body['nodes'][node].items() if k not in ('children', 'mesh', 'skin')}
        children = [new_index[c] for c in body['nodes'][node].get('children', ()) if c in new_index]
        nodes.append({**copy, **({'children': children} if children else {})})
    mesh_node, rig_node, root_node = len(nodes), len(nodes) + 1, len(nodes) + 2
    has_parent = {c for n in nodes for c in n.get('children', ())}
    roots = [j for j in joints if j not in has_parent]
    nodes += [{'mesh': 0, 'name': name, 'skin': 0},
              {'name': f'{name}.rig', 'children': roots + [mesh_node]},
              {'name': f'{name}.root_offset', 'translation': [0.0, float(skin.get('root_offset_m', 0.0)), 0.0],
               'children': [rig_node]}]
    document.update(nodes=nodes, scenes=[{'nodes': [root_node]}],
                    skins=[{'name': f'{name}.skin', 'joints': joints, 'inverseBindMatrices': len(accessors) - 1,
                            'skeleton': roots[0]}])


def write_mesh(path, positions, triangles, uvs=None, texture_png=None, colour=(0.8, 0.8, 0.8, 1.0), name='mesh',
               skin=None):
    """skin (optional): {'joints': (V, 4) indices into the skeleton's joints, 'weights': (V, 4), 'skeleton': a
    body .glb whose joint nodes and inverse bind matrices are copied, 'root_offset_m': lift of the whole rig}."""
    positions = canonical_to_gltf(positions).astype(np.float32)
    triangles = np.asarray(triangles, dtype=np.uint32)
    chunks, views, accessors = [], [], []

    def add(data, target=None):
        offset = sum(len(c) for c in chunks)
        chunks.append(_pad(data))
        views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(data), **({'target': target} if target else {})})
        return len(views) - 1

    accessors.append({'bufferView': add(positions.tobytes(), 34962), 'componentType': 5126, 'count': len(positions),
                      'type': 'VEC3', 'min': positions.min(axis=0).tolist(), 'max': positions.max(axis=0).tolist()})
    accessors.append({'bufferView': add(triangles.ravel().tobytes(), 34963), 'componentType': 5125,
                      'count': int(triangles.size), 'type': 'SCALAR'})
    attributes = {'POSITION': 0}
    material = {'name': name, 'pbrMetallicRoughness': {'baseColorFactor': list(colour), 'metallicFactor': 0.0,
                                                       'roughnessFactor': 0.9}, 'doubleSided': True}
    document = {'asset': {'version': '2.0', 'generator': 'FittedPerfection glbwrite'}, 'scene': 0,
                'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0, 'name': name}]}
    if uvs is not None:
        accessors.append({'bufferView': add(np.asarray(uvs, dtype=np.float32).tobytes(), 34962), 'componentType': 5126,
                          'count': len(uvs), 'type': 'VEC2'})
        attributes['TEXCOORD_0'] = len(accessors) - 1
    if texture_png is not None:
        image_view = add(texture_png)
        document['images'] = [{'bufferView': image_view, 'mimeType': 'image/png'}]
        document['samplers'] = [{'magFilter': 9729, 'minFilter': 9987}]
        document['textures'] = [{'sampler': 0, 'source': 0}]
        material['pbrMetallicRoughness']['baseColorTexture'] = {'index': 0}
        material['pbrMetallicRoughness']['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
    if skin is not None:
        add_skin(document, accessors, attributes, add, skin, name)
    binary = b''.join(chunks)
    document.update(accessors=accessors, bufferViews=views, buffers=[{'byteLength': len(binary)}],
                    materials=[material],
                    meshes=[{'name': name, 'primitives': [{'attributes': attributes, 'indices': 1, 'material': 0}]}])
    text = _pad(json.dumps(document, separators=(',', ':')).encode('utf-8'), b' ')
    total = 12 + 8 + len(text) + 8 + len(binary)
    Path(path).write_bytes(struct.pack('<4sII', b'glTF', 2, total) + struct.pack('<I4s', len(text), b'JSON') + text +
                           struct.pack('<I4s', len(binary), b'BIN\x00') + binary)
