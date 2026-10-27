"""Review renders of GLBs together, inside Blender (outside the seal). Run by `python -m fittedperfection preview`.

    blender --background --factory-startup --python-exit-code 1 --python fittedperfection/blender/preview.py -- \
        --out FOLDER --name NAME --item "PATH|STYLE|OFFSET_X" --item ...

Styles: texture (as authored), skin (flat skin tone), cloth (flat grey). Writes
NAME-front.png and NAME-side.png, plus NAME-xray-front.png and NAME-xray-side.png
where everything is see-through so what is inside shows. Orthographic, framed on
everything shown; front looks from -Y, side from the garment's own left (+X).
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

COLOURS = {'skin': (0.82, 0.63, 0.51, 1.0), 'cloth': (0.6, 0.62, 0.66, 1.0)}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--item', action='append', required=True)
    parser.add_argument('--pose', action='store_true')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


# Audit test pose (spec §5.7): elbows 60 degrees, knees 30 degrees, about each bone's own X axis.
TEST_POSE = {'lowerarm01.L': 60, 'lowerarm01.R': 60, 'lowerleg01.L': 30, 'lowerleg01.R': 30}


def apply_pose():
    for armature in [o for o in bpy.data.objects if o.type == 'ARMATURE']:
        for bone, degrees in TEST_POSE.items():
            pose_bone = armature.pose.bones.get(bone)
            if pose_bone is not None:
                pose_bone.rotation_mode = 'XYZ'
                pose_bone.rotation_euler = (math.radians(degrees), 0.0, 0.0)


def no_metadata(render):
    for name in dir(render):
        if name.startswith('use_stamp'):
            setattr(render, name, False)


def parse_item(item):
    """PATH|STYLE|OFFSET_X ('|' because Windows paths contain ':')."""
    parts = item.split('|')
    return parts[0], (parts[1] if len(parts) > 1 else 'texture'), float(parts[2]) if len(parts) > 2 else 0.0


def import_item(path, style, offset, axis):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    added = [obj for obj in bpy.data.objects if obj not in before]
    for obj in added:
        if obj.parent is None:
            obj.location[axis] += offset if axis == 0 else -offset
    meshes = [obj for obj in added if obj.type == 'MESH' and not obj.name.startswith('Icosphere')
              and (obj.parent is None or obj.parent.type == 'ARMATURE' or obj.parent.type == 'EMPTY')]
    if style in COLOURS:
        material = bpy.data.materials.new(style)
        material.diffuse_color = COLOURS[style]
        for obj in meshes:
            obj.data.materials.clear()
            obj.data.materials.append(material)
    return meshes


def bounds(meshes):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        points += [evaluated.matrix_world @ v.co for v in mesh.vertices]
        evaluated.to_mesh_clear()
    low = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    high = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return low, high


def main():
    args = arguments()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    no_metadata(scene.render)
    scene.render.engine = 'BLENDER_WORKBENCH'
    shading = scene.display.shading
    shading.light, shading.color_type = 'STUDIO', 'TEXTURE'
    shading.show_backface_culling = False
    scene.render.image_settings.file_format = 'PNG'
    camera = bpy.data.objects.new('camera', bpy.data.cameras.new('camera'))
    camera.data.type = 'ORTHO'
    scene.collection.objects.link(camera)
    scene.camera = camera
    out = Path(args.out)
    # Figures stand in a row across the front view and in a row along the depth for the side view,
    # so each camera sees them next to each other rather than behind one another.
    for view, axis in (('front', 0), ('side', 1)):
        for obj in [o for o in bpy.data.objects if o.type != 'CAMERA']:
            bpy.data.objects.remove(obj, do_unlink=True)
        meshes = []
        for item in args.item:
            path, style, offset = parse_item(item)
            meshes += import_item(path, style, offset, axis)
        if args.pose:
            apply_pose()
        low, high = bounds(meshes)
        centre = (low + high) / 2
        width = (high.x - low.x) if view == 'front' else (high.y - low.y)
        tall = high.z - low.z
        camera.data.ortho_scale = max(width, tall) * 1.06
        if view == 'front':
            camera.location, camera.rotation_euler = (centre.x, low.y - 10, centre.z), (math.pi / 2, 0, 0)
        else:
            camera.location, camera.rotation_euler = (high.x + 10, centre.y, centre.z), (math.pi / 2, 0, math.pi / 2)
        scale = 1200 / max(width, tall)
        scene.render.resolution_x, scene.render.resolution_y = max(64, int(width * scale * 1.06)), int(tall * scale * 1.06)
        for xray in (False, True):
            shading.show_xray, shading.xray_alpha = xray, 0.45
            scene.render.filepath = str(out/f'{args.name}{"-xray" if xray else ""}-{view}.png')
            bpy.ops.render.render(write_still=True)


main()
