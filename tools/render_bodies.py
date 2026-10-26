"""Front and side review renders of the target bodies, inside Blender (build step 2).

    blender --background --python-exit-code 1 --python tools/render_bodies.py -- \
        --bodies FOLDER --out REVIEW_FOLDER body_01_medium body_02_tall_slender ...

Writes <body>-front.png and <body>-side.png per body, plus lineup-front.png and
lineup-side.png with all of them side by side. Flat skin tone, orthographic
cameras. After glTF import the bodies stand on Blender's Z-up floor, facing -Y.
"""
import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SKIN = (0.82, 0.63, 0.51)
GAP = 1.25
PIXELS_PER_METRE = 520


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bodies', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('ids', nargs='+')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def empty_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def set_up_look():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.color_type = 'SINGLE'
    shading.single_color = SKIN
    shading.show_cavity = True
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new('World')
    scene.render.image_settings.file_format = 'PNG'


def import_body(path, offset):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    added = [obj for obj in bpy.data.objects if obj not in before]
    for obj in added:
        if obj.parent is None:
            obj.location += Vector(offset)
    # Only the skinned body: the importer also adds a hidden bone-display shape ("Icosphere").
    return [obj for obj in added if obj.type == 'MESH' and obj.parent is not None and obj.parent.type == 'ARMATURE']


def bounds(meshes):
    """From the evaluated (skinned) vertices: bound_box of a skinned mesh can sit away from what renders."""
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


def render(meshes, view, path):
    low, high = bounds(meshes)
    centre = (low + high) / 2
    size = high - low
    width_m = (size.x if view == 'front' else size.y) * 1.15
    height_m = size.z * 1.08
    camera_data = bpy.data.cameras.new('review')
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = max(width_m, height_m)
    camera = bpy.data.objects.new('review', camera_data)
    bpy.context.scene.collection.objects.link(camera)
    if view == 'front':
        camera.location = (centre.x, low.y - 5, centre.z)
        camera.rotation_euler = (1.5707963, 0, 0)
    else:                                            # from the body's left (+X), looking at it side-on
        camera.location = (high.x + 5, centre.y, centre.z)
        camera.rotation_euler = (1.5707963, 0, 1.5707963)
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = max(64, int(width_m * PIXELS_PER_METRE))
    scene.render.resolution_y = max(64, int(height_m * PIXELS_PER_METRE))
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)


def main():
    args = arguments()
    folder, out = Path(args.bodies), Path(args.out)
    set_up_look()
    for body in args.ids:
        empty_scene()
        meshes = import_body(folder/body/'BODY.glb', (0, 0, 0))
        for view in ('front', 'side'):
            render(meshes, view, out/f'{body}-{view}.png')
    # Lineups: in a row across the front view, and in a row along the depth axis for the side
    # view, so each camera sees all of them next to each other rather than behind one another.
    for view, axis in (('front', 0), ('side', 1)):
        empty_scene()
        meshes = []
        for i, body in enumerate(args.ids):
            offset = [0.0, 0.0, 0.0]
            offset[axis] = (i - (len(args.ids) - 1) / 2) * GAP * (1 if axis == 0 else -1)
            meshes += import_body(folder/body/'BODY.glb', offset)
        render(meshes, view, out/f'lineup-{view}.png')


main()
