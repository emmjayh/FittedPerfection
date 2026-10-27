"""Turntable video of fits, inside Blender: each group (the source, then each body wearing its outfit) spins in
place, side by side, one full turn, rendered to PNG frames and joined into an H.264 MP4. Run by
`python -m fittedperfection preview NAME --turntable`.

    blender --background --factory-startup --python-exit-code 1 --python fittedperfection/blender/turntable.py -- \
        --out VIDEO.mp4 --frames FOLDER --title TEXT --item "GROUP|PATH|STYLE" --item ...
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview import bounds, import_item, no_metadata                    # noqa: E402  (Blender's own python)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--title', default='')
    parser.add_argument('--item', action='append', required=True)
    parser.add_argument('--count', type=int, default=120)          # frames per turn (5 s at 24 fps)
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=720)
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def encode(frames, fps, width, height, path):
    """PNG frames -> H.264 MP4 through Blender's sequencer and built-in FFmpeg."""
    video = bpy.data.scenes.new('video')
    video.sequence_editor_create()
    editor = video.sequence_editor
    strips = editor.strips if hasattr(editor, 'strips') else editor.sequences
    strip = strips.new_image('turn', str(frames[0]), 1, 1)
    for frame in frames[1:]:
        strip.elements.append(frame.name)
    video.frame_start, video.frame_end = 1, len(frames)
    no_metadata(video.render)
    video.render.fps = fps
    video.render.resolution_x, video.render.resolution_y = width, height
    video.render.resolution_percentage = 100
    video.view_settings.view_transform = 'Standard'
    settings = video.render.image_settings
    if hasattr(settings, 'media_type'):
        settings.media_type = 'VIDEO'
    settings.file_format = 'FFMPEG'
    video.render.ffmpeg.format = 'MPEG4'
    video.render.ffmpeg.codec = 'H264'
    video.render.ffmpeg.constant_rate_factor = 'HIGH'
    video.render.ffmpeg.ffmpeg_preset = 'GOOD'
    video.render.filepath = str(path)
    bpy.ops.render.render(animation=True, scene=video.name)
    produced = sorted(path.parent.glob(path.stem + '*' + path.suffix))
    if not path.is_file() and len(produced) == 1:
        produced[0].rename(path)
    if not path.is_file():
        raise RuntimeError(f'the video was not written ({produced})')


def main():
    args = arguments()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    no_metadata(scene.render)
    scene.render.engine = 'BLENDER_WORKBENCH'
    shading = scene.display.shading
    shading.light, shading.color_type, shading.show_backface_culling = 'STUDIO', 'TEXTURE', False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.fps = 24

    groups = {}
    for item in args.item:
        group, path, style = item.split('|')
        groups.setdefault(group, []).append((path, style))
    # each group spins about its own vertical axis; groups stand in a row, spaced by their widest turn
    pivots, reach, x, gap = [], [], 0.0, 0.25
    for name, items in groups.items():
        before = set(bpy.data.objects)
        meshes = []
        for path, style in items:
            meshes += import_item(path, style, 0.0, 0)
        low, high = bounds(meshes)
        radius = max((Vector((p.x, p.y)) - Vector(((low.x + high.x) / 2, (low.y + high.y) / 2))).length
                     for p in (low, high, Vector((low.x, high.y, 0)), Vector((high.x, low.y, 0))))
        pivot = bpy.data.objects.new(f'turn-{name}', None)
        scene.collection.objects.link(pivot)
        pivot.location = ((low.x + high.x) / 2, (low.y + high.y) / 2, 0.0)
        for obj in set(bpy.data.objects) - before:
            if obj.parent is None and obj is not pivot:
                world = obj.matrix_world.copy()
                obj.parent = pivot
                obj.matrix_world = world
        x += radius
        pivot.location.x, pivot.location.y = x, 0.0
        x += radius + gap
        pivots.append(pivot)
        reach.append((low.z, high.z))
    for pivot in pivots:
        pivot.rotation_euler = (0, 0, 0)
        pivot.keyframe_insert('rotation_euler', index=2, frame=1)
        pivot.rotation_euler = (0, 0, 2 * math.pi)
        pivot.keyframe_insert('rotation_euler', index=2, frame=args.count + 1)
        action = pivot.animation_data.action
        curves = action.fcurves if hasattr(action, 'fcurves') else [c for layer in action.layers for strip in
                                                                      layer.strips for bag in strip.channelbags
                                                                      for c in bag.fcurves]
        for curve in curves:
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'

    bottom, top = min(r[0] for r in reach), max(r[1] for r in reach)
    width, tall = x - gap, top - bottom
    camera = bpy.data.objects.new('camera', bpy.data.cameras.new('camera'))
    camera.data.type = 'ORTHO'
    scene.collection.objects.link(camera)
    scene.camera = camera
    aspect = args.width / args.height
    camera.data.ortho_scale = max(width, tall * 1.18 * aspect) * 1.04
    camera.location = (width / 2, -20.0, bottom + tall * 0.55)
    camera.rotation_euler = (math.pi / 2, 0, 0)
    if args.title:
        curve = bpy.data.curves.new('title', 'FONT')
        curve.body = args.title
        text = bpy.data.objects.new('title', curve)
        scene.collection.objects.link(text)
        size = camera.data.ortho_scale / aspect * 0.045
        curve.size = size
        text.rotation_euler = (math.pi / 2, 0, 0)
        text.location = (width / 2 - camera.data.ortho_scale / 2 + size, -10.0,
                         camera.location.z + camera.data.ortho_scale / aspect / 2 - size * 1.8)
        material = bpy.data.materials.new('title')
        material.diffuse_color = (0.85, 0.88, 0.92, 1.0)
        text.data.materials.append(material)

    frames = Path(args.frames)
    frames.mkdir(parents=True, exist_ok=True)
    scene.frame_start, scene.frame_end = 1, args.count
    scene.render.filepath = str(frames/'frame_')
    bpy.ops.render.render(animation=True)
    encode(sorted(frames.glob('frame_*.png')), 24, args.width, args.height, Path(args.out))


main()
