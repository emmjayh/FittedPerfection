"""Make one target body with MPFB, inside Blender (build step 2).

    blender --background --python-exit-code 1 --python tools/make_bodies.py -- \
        --body-id body_01_medium --macros '{"gender": 0.0, ...}' --out BODY.glb --report body.json

Run by `python -m fittedperfection bodies make`, one fresh Blender process per body.
Blender must see MPFB: the command points BLENDER_USER_RESOURCES at the project's
.blender/ folder, where MPFB is installed.

Order matters. MPFB fits its rig to helper geometry inside the base mesh, so:
macro values -> bake the shape -> stand it on the floor -> add the default rig and
weights -> lower the arms to the chosen angle and make that the rest pose -> delete the
helper geometry -> triangulate with fixed diagonals -> export.
"""
import argparse
import json
import sys

import math

import bmesh
import bpy
from mathutils import Matrix, Vector


def mpfb(module_suffix, name):
    """Find an MPFB service: as an extension its package path is not fixed (MPFB's own samples do this)."""
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith(module_suffix) and hasattr(module, name):
            return getattr(module, name)
    raise RuntimeError(f'MPFB is not loaded ({module_suffix}); is BLENDER_USER_RESOURCES set to .blender/?')


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--body-id', required=True)
    parser.add_argument('--macros', required=True, help='JSON macro values for MPFB create_human')
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--arm-drop', type=float, required=True, help='upper arms, degrees below horizontal')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def empty_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def stand_on_floor(basemesh):
    """Move the mesh so the lowest vertex of the visible body (not the helpers) is at z = 0."""
    body = basemesh.vertex_groups['body'].index
    lowest = min((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices
                 if any(g.group == body and g.weight > 0 for g in v.groups))
    basemesh.location.z -= lowest
    bpy.ops.object.select_all(action='DESELECT')
    basemesh.select_set(True)
    bpy.context.view_layer.objects.active = basemesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def lower_arms(armature, basemesh, degrees_below_horizontal):
    """Rotate each upper arm about its shoulder joint, down in the arm's own vertical plane, to the given
    angle below horizontal; then make that pose the mesh shape and the rig's rest pose (weights unchanged)."""
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    before = {}
    for side in ('L', 'R'):
        upper, lower = armature.pose.bones[f'upperarm01.{side}'], armature.pose.bones[f'lowerarm01.{side}']
        head, elbow = armature.matrix_world @ upper.head, armature.matrix_world @ lower.head
        direction = (elbow - head).normalized()
        current = math.degrees(math.asin(max(-1.0, min(1.0, -direction.z))))
        before[side] = round(current, 2)
        axis = direction.cross(Vector((0.0, 0.0, -1.0))).normalized()          # turning about it lowers the arm
        turn = Matrix.Translation(upper.head) @ Matrix.Rotation(math.radians(degrees_below_horizontal - current), 4,
                                                                 axis) @ Matrix.Translation(-upper.head)
        upper.matrix = turn @ upper.matrix
        bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    basemesh.select_set(True)
    bpy.context.view_layer.objects.active = basemesh
    modifier = next(m for m in basemesh.modifiers if m.type == 'ARMATURE')
    bpy.ops.object.modifier_apply(modifier=modifier.name)                     # the posed shape becomes the mesh
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply(selected=False)                               # and the pose becomes the rest pose
    bpy.ops.object.mode_set(mode='OBJECT')
    restored = basemesh.modifiers.new('Armature', 'ARMATURE')
    restored.object = armature
    return before


def delete_helpers(basemesh, bone_names):
    """Delete every vertex outside MPFB's 'body' group, then every vertex group that is not a bone."""
    body = basemesh.vertex_groups['body'].index
    keep = {v.index for v in basemesh.data.vertices if any(g.group == body and g.weight > 0 for g in v.groups)}
    mesh = bmesh.new()
    mesh.from_mesh(basemesh.data)
    bmesh.ops.delete(mesh, geom=[v for v in mesh.verts if v.index not in keep], context='VERTS')
    # Split quads along a fixed diagonal. The exporter would otherwise pick the shorter one,
    # which depends on the body's shape and breaks the shared topology between bodies.
    bmesh.ops.triangulate(mesh, faces=mesh.faces[:], quad_method='FIXED', ngon_method='EAR_CLIP')
    mesh.to_mesh(basemesh.data)
    mesh.free()
    for group in list(basemesh.vertex_groups):
        if group.name not in bone_names:
            basemesh.vertex_groups.remove(group)
    for modifier in list(basemesh.modifiers):
        if modifier.type != 'ARMATURE':
            basemesh.modifiers.remove(modifier)
    basemesh.data.materials.clear()


def main():
    args = arguments()
    macros = json.loads(args.macros)
    HumanService = mpfb('services.humanservice', 'HumanService')
    TargetService = mpfb('services.targetservice', 'TargetService')

    empty_scene()
    basemesh = HumanService.create_human(mask_helpers=False, detailed_helpers=True, extra_vertex_groups=True,
                                         feet_on_ground=False, scale=0.1, macro_detail_dict=macros)
    TargetService.bake_targets(basemesh)
    stand_on_floor(basemesh)
    armature = HumanService.add_builtin_rig(basemesh, 'default', import_weights=True)
    if armature is None:
        raise RuntimeError('MPFB could not add the default rig')
    arms_before = lower_arms(armature, basemesh, args.arm_drop)
    bone_names = [bone.name for bone in armature.data.bones]
    delete_helpers(basemesh, set(bone_names))
    basemesh.name = basemesh.data.name = args.body_id
    armature.name = armature.data.name = args.body_id + '.rig'

    bpy.ops.object.select_all(action='DESELECT')
    for obj in (armature, basemesh):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.gltf(filepath=args.out, export_format='GLB', use_selection=True, export_yup=True,
                              export_skins=True, export_all_influences=True, export_animations=False,
                              export_morph=False, export_materials='NONE', export_normals=True,
                              export_texcoords=True, export_apply=False)

    manifest = sys.modules[HumanService.__module__.rsplit('.services', 1)[0]]
    report = {
        'body': args.body_id,
        'macros': macros,
        'blender': bpy.app.version_string,
        'mpfb_module': manifest.__name__,
        'vertices': len(basemesh.data.vertices),
        'faces': len(basemesh.data.polygons),
        'bones': bone_names,
        'arm_drop_before_deg': arms_before,
        'arm_drop_deg': args.arm_drop,
    }
    with open(args.report, 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)


main()
