"""Muse Wonder reviews (spec §4a): a second opinion for the human, never a decision.

Wonder only reads images and returns findings; it never edits a label, mesh or
setting and never accepts anything. Every call is cached under a hash of model,
prompt and image bytes, and the stored record keeps those hashes with the reply
(spec §5.7). Standard library only; reuses the SAM client's HTTP plumbing.
"""
import base64
import datetime
import hashlib
import json
from pathlib import Path

from .hashing import json_sha256
from .samapi import SamClient

SKIN_PROMPT = (
    'Left: renders of a 3D outfit on a mannequin. Right: the same views, where every surface our software '
    'labelled as bare mannequin skin is tinted orange. Check only the skin labels. Is anything tinted orange '
    'actually clothing (fabric, stockings, straps, lace), or is any bare skin left untinted? Reply with JSON '
    'only: {"skin_marked_but_clothing": [{"view": "front|back", "where": "...", "what": "..."}], '
    '"bare_skin_not_marked": [{"view": "...", "where": "..."}], "overall": "one sentence"}. '
    'Empty lists if all is right.')


class WonderClient(SamClient):
    def review(self, prompt, image_paths, max_output_tokens=600, effort='low'):
        """Findings text for one prompt and some PNG images, with the cache record (no call if cached)."""
        images = [Path(p) for p in image_paths]
        image_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in images]
        definition = {'model': self.model, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'image_sha256': image_hashes, 'max_output_tokens': max_output_tokens, 'effort': effort}
        key = json_sha256(definition)
        cached = self.cache/'responses'/f'{key}.json'
        if cached.is_file():
            record = json.loads(cached.read_text(encoding='utf-8'))
            return record['text'], {**record['summary'], 'cached': True, 'key': key}
        content = [{'type': 'input_text', 'text': prompt}]
        for path in images:
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            content.append({'type': 'input_image', 'image_url': f'data:image/png;base64,{encoded}'})
        request = {'model': self.model, 'stream': False, 'max_output_tokens': max_output_tokens,
                   'reasoning': {'effort': effort}, 'input': [{'role': 'user', 'content': content}]}
        response = self._call('POST', '/responses', json.dumps(request).encode('utf-8'), 'application/json')
        text = ''.join(part.get('text', '') for item in response.get('output', [])
                       for part in item.get('content', []) or [] if part.get('type') == 'output_text')
        summary = {'response_id': response.get('id'), 'status': response.get('status'), 'usage': response.get('usage'),
                   'model': response.get('model', self.model),
                   'received': datetime.datetime.now().astimezone().isoformat(timespec='seconds')}
        stored = {'definition': definition, 'prompt': prompt, 'images': [p.name for p in images], 'text': text,
                  'summary': summary}
        cached.write_text(json.dumps(stored, indent=1), encoding='utf-8')
        return text, {**summary, 'cached': False, 'key': key}
