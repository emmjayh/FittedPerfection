"""Human-review renders (spec §5.7), inside Blender: each group (the source, then each body wearing its outfit)
from the front, three-quarter, side and back. Run by the review stage.

    blender --background --factory-startup --python-exit-code 1 --python fittedperfection/blender/review.py -- \
        --out FOLDER --height 900 --item "GROUP|PATH|STYLE" --item ...

Writes FOLDER/<group>-<view>.png. Styles as in preview.py (texture, skin, cloth). Orthographic, framed on the
group; the body is a flat skin tone so cloth cutting into it shows.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview import bounds, import_item, no_metadata                    # noqa: E402  (Blender's own python)

# camera direction (from the figure towards the camera) per view; front looks from -Y, side from own left (+X)
VIEWS = {'front': (0.0, -1.0), 'three_quarter': (math.sqrt(0.5), -math.sqrt(0.5)), 'side': (1.0, 0.0),
         'back': (0.0, 1.0)}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--item', action='append', required=True)
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def main():
    args = arguments()
    groups = {}
    for item in args.item:
        group, path, style = item.split('|')
        groups.setdefault(group, []).append((path, style))
    scene = bpy.context.scene
    no_metadata(scene.render)
    scene.render.engine = 'BLENDER_WORKBENCH'
    shading = scene.display.shading
    shading.light, shading.color_type, shading.show_backface_culling = 'STUDIO', 'TEXTURE', False
    scene.render.image_settings.file_format = 'PNG'
    out = Path(args.out)
    for group, items in groups.items():
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        camera = bpy.data.objects.new('camera', bpy.data.cameras.new('camera'))
        camera.data.type = 'ORTHO'
        scene.collection.objects.link(camera)
        scene.camera = camera
        meshes = []
        for path, style in items:
            meshes += import_item(path, style, 0.0, 0)
        low, high = bounds(meshes)
        centre, tall = (low + high) / 2, high.z - low.z
        reach = max(high.x - low.x, high.y - low.y)
        for view, (dx, dy) in VIEWS.items():
            direction = Vector((dx, dy, 0.0))
            camera.location = centre + direction * 10
            camera.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
            camera.data.ortho_scale = max(reach, tall) * 1.06
            scene.render.resolution_y = args.height
            scene.render.resolution_x = max(64, int(args.height * max(reach, 0.3 * tall) / max(reach, tall)))
            scene.render.filepath = str(out/f'{group}-{view}.png')
            bpy.ops.render.render(write_still=True)


main()
