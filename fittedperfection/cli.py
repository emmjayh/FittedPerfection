"""Command line: python -m fittedperfection run | status | outfits | freeze | view."""
import argparse
import json
import os
import sys
from pathlib import Path

from . import VERSION, config


def _resolve_outfit(args):
    """(archive path, default run name) for a manifest name or an archive path."""
    listed = config.outfits()
    if args.outfit in listed:
        entry = listed[args.outfit]
        if entry['split'] == 'test-fits' and not args.test_fit:
            raise ValueError(f'{args.outfit} is held out for test fits; add --test-fit to run it on purpose')
        from .hashing import file_sha256
        if file_sha256(entry['path']) != entry['sha256']:
            raise ValueError(f'{entry["path"]} no longer matches its hash in outfits/manifest.json')
        return entry['path'], args.outfit
    path = Path(args.outfit)
    if not path.is_file():
        raise ValueError(f'{args.outfit!r} is neither a listed outfit nor an archive file '
                         f'(listed: {", ".join(listed) or "none"})')
    return path, path.stem


def cmd_run(args):
    from .run import Run
    from .stages import PIPELINE
    os.environ['FP_ALLOW_SPEND_USD'] = str(args.allow_spend)
    archive, default_name = _resolve_outfit(args)
    name = args.name or default_name
    folder = config.runs_folder()/name
    config.runs_folder().mkdir(exist_ok=True)
    state = Run(folder, PIPELINE).execute(archive, name, rerun_from=args.rerun_from,
                                          stop_after=args.stop_after, fresh=args.fresh)
    print(f'[fp] {state["status"]} - {folder/"live.html"}')
    return 0


def cmd_status(args):
    from .stages import NAMES
    state = json.loads((config.runs_folder()/args.name/'run.json').read_text(encoding='utf-8'))
    print(f'{state["name"]}: {state["status"]}  (garment {state["garment_sha256"][:12]})')
    for name in NAMES:
        record = state['stages'].get(name, {})
        print(f'  {name:<7} {record.get("status", "waiting"):<9} {record.get("reason", "")}')
    return 0


def cmd_outfits(args):
    for name, entry in config.outfits().items():
        present = 'ok' if entry['path'].is_file() else 'MISSING'
        print(f'{entry["split"]:<10} {name:<26} {entry["bytes"] / 1e6:7.1f} MB  {present}')
    return 0


def cmd_freeze(args):
    from .freeze import LAUNCHER, freeze
    folder = freeze()
    print(f'[fp] frozen copy: {folder}\n       run it with: python {folder/LAUNCHER} run OUTFIT')
    return 0


def cmd_bodies_make(args):
    from .bodies import make
    folder, record = make(replace_draft=args.replace_draft)
    for body, entry in record['bodies'].items():
        facts = entry['measures']
        print(f'  {body:<24} {entry["sex"]:<7} {facts["height_m"]:.3f} m tall, arms {facts["upper_arm_below_horizontal_deg"]} deg down')
    print(f'[fp] bodies written to {folder} - review {folder/"review"} before committing')
    return 0


def cmd_view(args):
    from .viewer import serve
    serve(args.port)
    return 0


def build_parser():
    from .stages import NAMES
    parser = argparse.ArgumentParser(prog='fittedperfection', description=f'FittedPerfection {VERSION}')
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('run', help='run the pipeline for one outfit')
    run.add_argument('outfit', help='a name from outfits/manifest.json, or a generator archive (.zip)')
    run.add_argument('--name', help='run folder name under runs/ (default: the outfit name)')
    run.add_argument('--fresh', action='store_true', help='stage mode: refuse bytes seen before, write provenance')
    run.add_argument('--rerun-from', choices=NAMES, help='set this stage and later ones aside, then rerun them')
    run.add_argument('--stop-after', choices=NAMES, help='stop after this stage')
    run.add_argument('--test-fit', action='store_true', help='allow a held-out test-fits outfit')
    run.add_argument('--allow-spend', type=float, default=0.0, metavar='USD',
                     help='allow up to this much on paid API calls (default 0: cached answers only)')
    run.set_defaults(handler=cmd_run)

    status = commands.add_parser('status', help='show one run')
    status.add_argument('name')
    status.set_defaults(handler=cmd_status)

    commands.add_parser('outfits', help='list the test outfits').set_defaults(handler=cmd_outfits)
    commands.add_parser('freeze', help='make a frozen copy of the committed code').set_defaults(handler=cmd_freeze)

    bodies = commands.add_parser('bodies', help='the four target bodies')
    bodies_commands = bodies.add_subparsers(dest='bodies_command', required=True)
    make = bodies_commands.add_parser('make', help='make them with MPFB (refuses if they exist)')
    make.add_argument('--replace-draft', action='store_true',
                      help='set an unreviewed draft aside first (reviewed bodies are never replaced)')
    make.set_defaults(handler=cmd_bodies_make)

    view = commands.add_parser('view', help='open the 3D review viewer on this computer')
    view.add_argument('--port', type=int, default=8765)
    view.set_defaults(handler=cmd_view)
    return parser


def main(argv=None):
    if config.frozen_record():
        from .freeze import FreezeError, verify
        try:
            verify()
        except FreezeError as error:
            print(f'[fp] {error}', file=sys.stderr)
            return 2
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, NotImplementedError, OSError) as error:
        print(f'[fp] {type(error).__name__}: {error}', file=sys.stderr)
        return 1
