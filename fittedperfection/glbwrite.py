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


def write_mesh(path, positions, triangles, uvs=None, texture_png=None, colour=(0.8, 0.8, 0.8, 1.0), name='mesh'):
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
    binary = b''.join(chunks)
    document.update(accessors=accessors, bufferViews=views, buffers=[{'byteLength': len(binary)}],
                    materials=[material],
                    meshes=[{'name': name, 'primitives': [{'attributes': attributes, 'indices': 1, 'material': 0}]}])
    text = _pad(json.dumps(document, separators=(',', ':')).encode('utf-8'), b' ')
    total = 12 + 8 + len(text) + 8 + len(binary)
    Path(path).write_bytes(struct.pack('<4sII', b'glTF', 2, total) + struct.pack('<I4s', len(text), b'JSON') + text +
                           struct.pack('<I4s', len(binary), b'BIN\x00') + binary)
