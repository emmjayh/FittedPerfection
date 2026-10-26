"""Views inside Blender (spec §5.2). Run by fittedperfection/views.py, not imported.

    blender --background --factory-startup --python-exit-code 1 --python fittedperfection/blender/views.py -- \
        --cage CAGE.glb --cameras cameras.json --fps 24 --out FOLDER

For every camera in cameras.json (made by fittedperfection.cameras) it renders
    color/<name>.png    the textured cage, studio light, anti-aliased (what SAM sees)
    faceid/<name>.png   each visible triangle's index in the face-ID code (fittedperfection.faceid),
                        flat colour, no anti-aliasing, so every pixel is exactly one triangle
and then encodes the turntable colour frames as orbit.mp4 (H.264) with Blender's
own encoder. Triangle indices are those of CAGE.glb as stored, which the views
stage checks by projecting triangle centres.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fittedperfection import faceid  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cage', required=True)
    parser.add_argument('--cameras', required=True)
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--out', required=True)
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def import_cage(path):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=False)
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if len(meshes) != 1:
        raise RuntimeError(f'expected one mesh in {path}, found {len(meshes)}')
    cage = meshes[0]
    if any(len(polygon.vertices) != 3 for polygon in cage.data.polygons):
        raise RuntimeError('the cage must be all triangles')
    return cage


def add_face_ids(cage):
    """A per-corner colour attribute holding each triangle's face-ID colour (exact sRGB bytes)."""
    mesh = cage.data
    colours = faceid.encode(len(mesh.polygons)).astype(np.float64) / 255.0
    corner_colours = np.ones((len(mesh.loops), 4))
    loop_starts = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get('loop_start', loop_starts)
    for corner in range(3):
        corner_colours[loop_starts + corner, :3] = colours
    attribute = mesh.color_attributes.new('faceid', 'BYTE_COLOR', 'CORNER')
    attribute.data.foreach_set('color_srgb', corner_colours.ravel())
    mesh.color_attributes.active_color = attribute
    mesh.color_attributes.render_color_index = mesh.color_attributes.find('faceid')


def no_metadata(render):
    """Blender stamps the date, time, host and more into every PNG and video it writes. Identical
    renders must give identical files (for reuse and for the SAM cache), so all stamps go off."""
    for name in dir(render):
        if name.startswith('use_stamp'):
            setattr(render, name, False)


def set_up_scene(size):
    scene = bpy.context.scene
    no_metadata(scene.render)
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = scene.world or bpy.data.worlds.new('world')
    shading = scene.display.shading
    shading.show_cavity = shading.show_shadows = shading.show_object_outline = False
    shading.show_specular_highlight = False
    shading.show_backface_culling = False
    shading.show_xray = False
    camera = bpy.data.objects.new('camera', bpy.data.cameras.new('camera'))
    camera.data.type = 'ORTHO'
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def place(camera, spec, reach):
    right, up, forward = (np.array(spec[key]) for key in ('right', 'up', 'forward'))
    location = np.array(spec['centre']) - forward * reach
    rotation = np.column_stack([right, up, -forward])        # Blender cameras look down their -Z with +Y up
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, location
    camera.matrix_world = Matrix(matrix.tolist())
    camera.data.ortho_scale = spec['scale']
    camera.data.clip_start = 0.01
    camera.data.clip_end = reach * 2


def render(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def encode_video(frames, fps, size, path):
    """Turntable PNGs -> H.264 MP4 through Blender's sequencer and built-in FFmpeg."""
    video = bpy.data.scenes.new('video')
    video.sequence_editor_create()
    editor = video.sequence_editor
    strips = editor.strips if hasattr(editor, 'strips') else editor.sequences   # renamed in Blender 4.4
    strip = strips.new_image('orbit', str(frames[0]), 1, 1)
    for frame in frames[1:]:
        strip.elements.append(frame.name)
    video.frame_start, video.frame_end = 1, len(frames)
    no_metadata(video.render)
    video.render.fps = fps
    video.render.resolution_x = video.render.resolution_y = size
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
    if path.is_file():
        return path
    if len(produced) == 1:
        produced[0].rename(path)
        return path
    raise RuntimeError(f'the video was not written ({produced})')


def main():
    args = arguments()
    out = Path(args.out)
    specs = json.loads(Path(args.cameras).read_text(encoding='utf-8'))
    cage = import_cage(Path(args.cage))
    add_face_ids(cage)
    size = specs[0]['size']
    scene, camera = set_up_scene(size)
    extent = np.array([list(corner) for corner in cage.bound_box])
    reach = float(np.linalg.norm(extent.max(axis=0) - extent.min(axis=0))) + 5.0
    (out/'color').mkdir(parents=True, exist_ok=True)
    (out/'faceid').mkdir(exist_ok=True)
    shading = scene.display.shading
    for spec in specs:
        place(camera, spec, reach)
        shading.light, shading.color_type = 'STUDIO', 'TEXTURE'
        scene.display.render_aa = '8'
        scene.world.color = (0.18, 0.18, 0.18)
        render(scene, out/'color'/f'{spec["name"]}.png')
        shading.light, shading.color_type = 'FLAT', 'VERTEX'
        scene.display.render_aa = 'OFF'
        scene.world.color = (0.0, 0.0, 0.0)
        render(scene, out/'faceid'/f'{spec["name"]}.png')
    orbit = sorted((out/'color').glob('orbit-*.png'))
    video = encode_video(orbit, args.fps, size, out/'orbit.mp4')
    clip = bpy.data.movieclips.load(str(video))
    report = {'blender': bpy.app.version_string, 'triangles': len(cage.data.polygons), 'frames': len(specs),
              'video_frames': clip.frame_duration, 'video_bytes': video.stat().st_size}
    (out/'blender.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')


main()
