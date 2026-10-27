"""Stage 1, intake (spec §5.1): generator archive -> full/GARMENT.glb + full/CAGE.glb in the canonical frame.

The archive is treated as data: only flat file names with known extensions
are unpacked, into this stage's own folder. Blender (fittedperfection/blender/
intake.py) imports, orients from the feet and exports; only the files actually
used are kept under full/source/.
"""
import json
import shutil
import zipfile
from pathlib import Path

from . import config, settings

ACCEPTED = {'.obj', '.mtl', '.fbx', '.png', '.jpg', '.jpeg'}
MAX_UNPACKED_BYTES = 4 << 30
TEXTURE = 'texture_diffuse.png'
BLENDER_SCRIPT = config.PACKAGE/'blender'/'intake.py'
CODE = ('intake.py', 'orientation.py', 'blender/intake.py')


class IntakeError(ValueError):
    pass


def check_members(bundle):
    """The archive's entries, refusing anything but flat, known file types within the size limit, and requiring
    a garment mesh and its texture. Reads the listing only; nothing is unpacked."""
    members = bundle.infolist()
    total = 0
    for member in members:
        name = member.filename
        if member.is_dir() or '/' in name or '\\' in name or name.startswith('.') or ':' in name:
            raise IntakeError(f'unexpected entry in the archive: {name!r}')
        if Path(name).suffix.lower() not in ACCEPTED:
            raise IntakeError(f'unexpected file type in the archive: {name!r}')
        total += member.file_size
    if total > MAX_UNPACKED_BYTES:
        raise IntakeError(f'archive unpacks to {total / 1e9:.1f} GB, more than allowed')
    names = [m.filename for m in members]
    choose_mesh(names)
    if TEXTURE not in names:
        raise IntakeError(f'the archive has no {TEXTURE}')
    return members


def unpack(archive, folder):
    """Unpack only flat, known file types; refuse anything else in the archive."""
    with zipfile.ZipFile(archive) as bundle:
        members = check_members(bundle)
        folder.mkdir(parents=True)
        for member in members:
            with bundle.open(member) as source, open(folder/member.filename, 'wb') as target:
                shutil.copyfileobj(source, target)
    return sorted(p.name for p in folder.iterdir())


def choose_mesh(names):
    """The plain geometry export: base/lod OBJ before FBX; never the *_basic_* shaded copies."""
    candidates = [n for n in names if Path(n).suffix.lower() in ('.obj', '.fbx') and '_basic_' not in n]
    if not candidates:
        raise IntakeError('no OBJ or FBX garment mesh in the archive')
    rank = lambda n: (Path(n).suffix.lower() != '.obj', Path(n).stem.lower() not in ('base', 'lod'), n)
    return sorted(candidates, key=rank)[0]


def run(ctx, out):
    source = out/'source'
    names = unpack(ctx.garment, source)
    mesh = choose_mesh(names)
    if TEXTURE not in names:
        raise IntakeError(f'the archive has no {TEXTURE}')
    keep = {mesh, TEXTURE} | {n for n in names if n.lower().endswith('.mtl')}
    for name in set(names) - keep:
        (source/name).unlink()

    stage_settings = {'cage_faces': settings.section('intake')['cage_faces'],
                      'orientation': json.loads(json.dumps(dict(settings.section('intake')['orientation']),
                                                           default=list))}
    try:
        ctx.call([config.blender(), '--background', '--factory-startup', '--python-exit-code', '1',
                  '--python', BLENDER_SCRIPT, '--', '--mesh', source/mesh, '--texture', source/TEXTURE,
                  '--settings', json.dumps(stage_settings), '--out', out], 'intake-blender.log')
    except RuntimeError:
        report = out/'intake.json'
        if report.is_file():
            reason = json.loads(report.read_text(encoding='utf-8')).get('reason')
            if reason:
                raise IntakeError(reason) from None
        raise
    report = json.loads((out/'intake.json').read_text(encoding='utf-8'))
    return {'mesh': mesh, 'source_triangles': report['source_triangles'], 'cage_triangles': report['cage_triangles'],
            'toe_direction': report['facing']['toe_direction'],
            'facing_status': report['facing']['status'],
            'dropped_duplicate_meshes': report.get('dropped_duplicate_meshes', []),
            'turned_degrees': report['facing']['turn_about_z_degrees'],
            'toe_over_heel_reach': report['facing']['evidence']['toe_over_heel_reach']}
