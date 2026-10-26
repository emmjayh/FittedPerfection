"""Intake inside Blender (spec §5.1). Run by fittedperfection/intake.py, not imported.

    blender --background --factory-startup --python-exit-code 1 --python fittedperfection/blender/intake.py -- \
        --mesh SOURCE.obj|fbx --texture texture_diffuse.png --settings JSON --out FOLDER

Imports and joins the generator mesh, finds which way it faces from its feet
(fittedperfection.orientation), turns it to the canonical frame (Z up, front -Y,
own left +X) standing on the floor and centred, then writes:
    GARMENT.glb   every source face and UV, with the diffuse texture (the deliverable)
    CAGE.glb      the same garment decimated for labelling and fitting maths
    intake.json   counts, the turn and translation applied, and the facing evidence
    review-front.png / review-side.png   full (left) and cage (right) side by side
If the facing is unclear it writes intake.json with the reason and exits with an error.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fittedperfection.orientation import OrientationUnclear, facing  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--texture', required=True)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--out', required=True)
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def select_only(objects, active=None):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active or objects[0]


def import_joined(path):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    if path.suffix.lower() == '.obj':
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.fbx(filepath=str(path))
    for obj in [o for o in bpy.data.objects if o.type != 'MESH']:
        bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes:
        raise RuntimeError(f'{path.name} holds no mesh')
    meshes, dropped = keep_most_detailed(meshes)
    select_only(meshes)
    if len(meshes) > 1:
        bpy.ops.object.join()
    garment = bpy.context.view_layer.objects.active
    select_only([garment])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    garment.name = garment.data.name = 'garment'
    return garment, dropped


def world_box(obj):
    corners = np.array([obj.matrix_world @ Vector(c) for c in obj.bound_box])
    return corners.min(axis=0), corners.max(axis=0)


def keep_most_detailed(meshes):
    """Generators can ship several levels of detail of one outfit in one file (model_LOD0..LOD4),
    stacked in the same place. Joining them would stack copies, so only the most detailed is kept:
    by name where levels are named, otherwise among meshes that occupy the same space."""
    named = {}
    for obj in meshes:
        match = re.search(r'_?LOD(\d+)$', obj.name, flags=re.IGNORECASE)
        if match:
            named.setdefault(re.sub(r'_?LOD\d+$', '', obj.name, flags=re.IGNORECASE), []).append((int(match[1]), obj))
    dropped = []
    for _, levels in named.items():
        if len(levels) > 1:
            levels.sort(key=lambda item: item[0])
            dropped += [obj for _, obj in levels[1:]]
    kept = [obj for obj in meshes if obj not in dropped]
    by_detail = sorted(kept, key=lambda obj: -len(obj.data.polygons))
    final = []
    for obj in by_detail:                         # unnamed copies: same box as a more detailed mesh
        low, high = world_box(obj)
        duplicate = False
        for other in final:
            other_low, other_high = world_box(other)
            overlap = np.clip(np.minimum(high, other_high) - np.maximum(low, other_low), 0, None).prod()
            union = (high - low).prod() + (other_high - other_low).prod() - overlap
            if union > 0 and overlap / union > 0.97:
                duplicate = True
                break
        (dropped if duplicate else final).append(obj)
    names = [obj.name for obj in dropped]
    for obj in dropped:
        bpy.data.objects.remove(obj, do_unlink=True)
    return final, names


def vertices(obj):
    coordinates = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get('co', coordinates)
    return coordinates.reshape(-1, 3)


def triangle_count(obj):
    return sum(len(polygon.vertices) - 2 for polygon in obj.data.polygons)


def textured_material(texture_path):
    material = bpy.data.materials.new('garment')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    shader = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
    image_node = nodes.new('ShaderNodeTexImage')
    image_node.image = bpy.data.images.load(str(texture_path))
    image_node.image.pack()
    material.node_tree.links.new(image_node.outputs['Color'], shader.inputs['Base Color'])
    return material


def export(obj, path):
    select_only([obj])
    bpy.ops.export_scene.gltf(filepath=str(path), export_format='GLB', use_selection=True, export_yup=True,
                              export_materials='EXPORT', export_image_format='AUTO', export_normals=True,
                              export_texcoords=True, export_animations=False, export_skins=False,
                              export_morph=False, export_apply=False)


def render_side_by_side(full, cage, out):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'TEXTURE'
    scene.render.image_settings.file_format = 'PNG'
    size = vertices(full).max(axis=0) - vertices(full).min(axis=0)
    cage.location.x = max(size[0], size[1]) * 1.3
    camera = bpy.data.objects.new('review', bpy.data.cameras.new('review'))
    camera.data.type = 'ORTHO'
    scene.collection.objects.link(camera)
    scene.camera = camera
    for view in ('front', 'side'):
        # The camera always looks from -Y towards +Y, at the front. For the side view each copy is
        # turned -90 degrees about Z in place, which brings its own left side (+X) round to face -Y.
        for obj in (full, cage):
            obj.rotation_euler.z = 0.0 if view == 'front' else -math.pi / 2
        bpy.context.view_layer.update()
        world = np.vstack([vertices(obj) @ np.array(obj.matrix_world)[:3, :3].T + np.array(obj.matrix_world)[:3, 3]
                           for obj in (full, cage)])
        low, high = world.min(axis=0), world.max(axis=0)
        width, height = (high[0] - low[0]) * 1.06, (high[2] - low[2]) * 1.06
        camera.data.ortho_scale = max(width, height)
        camera.location = ((low[0] + high[0]) / 2, low[1] - 10, (low[2] + high[2]) / 2)
        camera.rotation_euler = (math.pi / 2, 0, 0)
        scene.render.resolution_x = int(1600 * width / max(width, height))
        scene.render.resolution_y = int(1600 * height / max(width, height))
        scene.render.filepath = str(out/f'review-{view}.png')
        bpy.ops.render.render(write_still=True)
    for obj in (full, cage):
        obj.rotation_euler.z = 0.0
        obj.location.x = 0.0


def main():
    args = arguments()
    settings = json.loads(args.settings)
    out = Path(args.out)
    report = {'blender': bpy.app.version_string, 'source_mesh': Path(args.mesh).name}
    garment, dropped = import_joined(Path(args.mesh))
    report['dropped_duplicate_meshes'] = dropped
    points = vertices(garment)
    extent = points.max(axis=0) - points.min(axis=0)
    report.update(source_vertices=len(points), source_triangles=triangle_count(garment),
                  imported_extent_m=[round(float(e), 4) for e in extent])
    # Standing: far taller than deep. (Not "taller than wide": sleeves can span nearly the height.)
    if extent[2] < 2.0 * min(extent[0], extent[1]):
        report.update(status='refused', reason=f'not standing upright after import (extent {extent.round(3)})')
        (out/'intake.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')
        raise RuntimeError(report['reason'])
    try:
        found = facing(points, **settings['orientation'])
    except OrientationUnclear as error:
        report.update(status='refused', reason=f'facing unclear: {error}')
        (out/'intake.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')
        raise

    garment.rotation_euler.z = math.radians(found['turn_about_z_degrees'])
    select_only([garment])
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    turned = vertices(garment)
    low, high = turned.min(axis=0), turned.max(axis=0)
    shift = np.array([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
    garment.location = tuple(shift)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    material = textured_material(Path(args.texture))
    garment.data.materials.clear()
    garment.data.materials.append(material)
    export(garment, out/'GARMENT.glb')

    cage = garment.copy()
    cage.data = garment.data.copy()
    cage.name = cage.data.name = 'cage'
    bpy.context.scene.collection.objects.link(cage)
    decimate = cage.modifiers.new('decimate', 'DECIMATE')
    decimate.decimate_type = 'COLLAPSE'
    decimate.ratio = min(1.0, settings['cage_faces'] / triangle_count(garment))
    decimate.use_collapse_triangulate = True
    select_only([cage])
    bpy.ops.object.modifier_apply(modifier=decimate.name)
    export(cage, out/'CAGE.glb')

    render_side_by_side(garment, cage, out)
    report.update(status='complete', facing=found, translation_m=[round(float(s), 5) for s in shift],
                  frame={'up': '+Z', 'front': '-Y', 'own_left': '+X',
                         'glb': 'glTF +Y up; the garment faces +Z; its own left is +X'},
                  full_triangles=triangle_count(garment), cage_triangles=triangle_count(cage),
                  cage_vertices=len(cage.data.vertices))
    (out/'intake.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')


main()
