"""Stage 3, vision (spec §5.3): SAM 3.1 masks for every vocabulary phrase, on every view.

Per phrase: one video request on the turntable (orbit.mp4), so each part found
keeps one object id all the way round, plus one image request each for the top
and bottom stills. Network on (this stage is never sealed); every answer comes
from, or goes into, the permanent cache in fittedperfection.samapi, so a rerun
of unchanged views makes no API calls and yields identical masks.

Output: masks.npz (one raster per record, box-local) and vision.json (an index of
records: phrase, class, view, object id, confidence, bounds; plus every request's
cache key and usage, and the cost estimate).
"""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format, video_segmentation_format

from . import config, models, secrets, settings, vocabulary
from .samapi import SamClient

CODE = ('vision.py', 'samapi.py', 'vocabulary.py')
PRICE_PER_IMAGE = 2.50 / 1000
PRICE_PER_VIDEO_FRAME = 0.20 / 1000
MIN_BILLED_VIDEO_FRAMES = 50
WORKERS = 8


class VisionError(RuntimeError):
    pass


def parse(kind, text):
    """Segmentation text -> list of (frame index or None, object id, confidence, bounds, raster)."""
    parser = (video_segmentation_format() if kind == 'video' else image_segmentation_format()).create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    problems = [d for d in result.diagnostics if getattr(d, 'severity', 'error') == 'error']
    if problems:
        raise VisionError(f'SAM output could not be read: {problems[:3]}')
    parsed = []
    for record in result.records:
        if record.kind != 'mask':
            continue
        raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
        raster = raster.reshape(record.mask.height, record.mask.width)
        bounds = [int(record.bounds.left), int(record.bounds.top), int(record.bounds.right), int(record.bounds.bottom)]
        frame = record.frame.frame_index if record.frame is not None else None
        parsed.append((frame, str(record.object_id), record.confidence, bounds, raster))
    return parsed


def image_content_sha(path):
    """What an image shows: a hash of its decoded pixels, ignoring any file metadata."""
    pixels = np.asarray(Image.open(path).convert('RGB'))
    return hashlib.sha256(repr(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def video_content_sha(views, orbit_frames, fps):
    """What the turntable video shows: its frames' decoded pixels in order, and the frame rate."""
    digest = hashlib.sha256(f'turntable fps={fps}'.encode())
    for i in range(orbit_frames):
        digest.update(image_content_sha(views/'color'/f'orbit-{i:03d}.png').encode())
    return digest.hexdigest()


def slug(phrase):
    return re.sub(r'[^a-z0-9]+', '-', phrase.lower()).strip('-')


def run(ctx, out):
    vision_settings = settings.section('vision')
    if vision_settings['vocabulary_version'] != vocabulary.VERSION:
        raise VisionError(f'settings ask for vocabulary v{vision_settings["vocabulary_version"]}, '
                          f'code has v{vocabulary.VERSION}')
    models.check_decoder()
    views = ctx.output_of('views')
    record = json.loads((views/'views.json').read_text(encoding='utf-8'))
    orbit_frames = record['orbit_frames']
    pin = models.pin('sam')
    sam = SamClient(config.runs_folder()/'_cache'/'sam', pin['model'], models.API_BASE, api_key=secrets.api_key())
    options = {'mask_encoding': vision_settings['mask_encoding'],
               'include_confidence': 'true' if vision_settings['include_confidence'] else 'false'}

    jobs = []
    for phrase in vocabulary.phrases():
        jobs.append((phrase, 'video', views/record['video'], None))
        for still in record['stills']:
            jobs.append((phrase, 'image', views/'color'/f'{still}.png', still))
    if any(kind == 'video' for _, kind, _, _ in jobs):
        sam.upload(views/record['video'])                   # once, before the parallel requests

    content = {views/record['video']: video_content_sha(views, orbit_frames, record['fps'])}
    content.update({views/'color'/f'{still}.png': image_content_sha(views/'color'/f'{still}.png')
                    for still in record['stills']})

    def request(job):
        phrase, kind, media, still = job
        text, info = sam.segment(phrase, media, kind, options, content_sha=content[media])
        return job, text, info

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = list(pool.map(request, jobs))

    rasters, index, requests = {}, [], []
    images_processed = video_frames = 0
    for (phrase, kind, media, still), text, info in answers:
        usage = info.get('usage') or {}
        if not info['cached']:
            images_processed += int(usage.get('images_processed', 0) or 0)
            video_frames += int(usage.get('video_frames_processed', 0) or 0)
        requests.append({'phrase': phrase, 'kind': kind, 'view': still or 'orbit', 'key': info['key'],
                         'cached': info['cached'], 'response_id': info.get('response_id'), 'usage': usage})
        for frame, object_id, confidence, bounds, raster in parse(kind, text):
            if kind == 'video':
                if frame is None or not 0 <= frame < orbit_frames:
                    raise VisionError(f'{phrase}: frame {frame} outside the turntable')
                view = f'orbit-{frame:03d}'
            else:
                view = still
            name = f'r{len(index):06d}'
            rasters[name] = raster
            index.append({'raster': name, 'phrase': phrase, 'class': vocabulary.PHRASE_CLASS.get(phrase),
                          'length': vocabulary.LENGTHS.get(phrase), 'view': view, 'object': object_id,
                          'confidence': None if confidence is None else round(float(confidence), 4),
                          'bounds': bounds, 'shape': list(raster.shape)})
    np.savez_compressed(out/'masks.npz', **rasters)

    fresh = [r for r in requests if not r['cached']]
    cost = images_processed * PRICE_PER_IMAGE + video_frames * PRICE_PER_VIDEO_FRAME
    summary = {'model': pin['model'], 'decoder': pin['decoder'], 'vocabulary_version': vocabulary.VERSION,
               'options': options, 'phrases': len(vocabulary.phrases()), 'requests': len(requests),
               'new_requests': len(fresh), 'records': len(index),
               'billed': {'images': images_processed, 'video_frames': video_frames,
                          'estimated_usd': round(cost, 4)}}
    (out/'vision.json').write_text(json.dumps({**summary, 'requests_made': requests, 'records': index}, indent=1),
                                   encoding='utf-8', newline='\n')
    found = {}
    for entry in index:
        found[entry['phrase']] = found.get(entry['phrase'], 0) + 1
    return {**summary, 'phrases_with_masks': len(found)}
