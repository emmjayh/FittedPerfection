"""SAM 3.1 over the Meta Model API, with a permanent response cache (spec §4a, changes section).

Requests go to POST {base}/responses (OpenAI-compatible Responses format), one
phrase per request, non-streaming. Videos are uploaded once through the Files
API and referenced by file id. Results vary slightly between calls, so every
response is stored under a hash of everything that defines the request (model,
kind, phrase, media bytes, options). A request seen before is answered from the
cache with no network use at all, which keeps reruns identical and free.
Standard library only.
"""
import base64
import datetime
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .hashing import file_sha256, json_sha256

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class SamApiError(RuntimeError):
    pass


class SpendRefused(SamApiError):
    """A paid call would go over what the owner allowed (default: nothing)."""


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def urllib_transport(method, url, headers, body, timeout):
    """(status, body bytes) for one HTTP request; the only function here that touches the network."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class SamClient:
    def __init__(self, cache_folder, model, base_url, api_key=None, transport=urllib_transport,
                 timeout=900, retries=4, pause=2.0, budget_usd=0.0):
        self.cache = Path(cache_folder)
        (self.cache/'responses').mkdir(parents=True, exist_ok=True)
        self.model, self.base_url = model, base_url.rstrip('/')
        self.api_key, self.transport = api_key, transport
        self.timeout, self.retries, self.pause = timeout, retries, pause
        self.network_calls = 0
        self.budget_usd, self.spent_usd = budget_usd, 0.0
        self._lock = threading.Lock()

    def charge(self, estimate_usd, what):
        """Reserve an estimated cost before a paid call, or refuse if it would exceed the allowance."""
        with self._lock:
            if self.spent_usd + estimate_usd > self.budget_usd + 1e-9:
                raise SpendRefused(f'{what} would cost about ${estimate_usd:.4f}; allowed ${self.budget_usd:.2f},'
                                   f' already ${self.spent_usd:.4f}. Nothing was sent.')
            self.spent_usd += estimate_usd

    # -- HTTP -------------------------------------------------------------------

    def _call(self, method, path, body, content_type):
        if not self.api_key:
            raise SamApiError('no API key; set MODEL_API_KEY in secrets.env')
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': content_type,
                   'Accept': 'application/json'}
        for attempt in range(self.retries + 1):
            self.network_calls += 1
            status, payload = self.transport(method, self.base_url + path, headers, body, self.timeout)
            if status < 300:
                return json.loads(payload)
            if status not in RETRY_STATUS or attempt == self.retries:
                raise SamApiError(f'{method} {path} failed with HTTP {status}: {payload[:500]!r}')
            time.sleep(self.pause * 2 ** attempt)

    # -- files ------------------------------------------------------------------

    def upload(self, path):
        """File id for a video, uploading it only once per content hash."""
        path = Path(path)
        sha = file_sha256(path)
        record_path = self.cache/'files'/f'{sha}.json'
        if record_path.is_file():
            return json.loads(record_path.read_text(encoding='utf-8'))['id'], sha
        boundary = uuid.uuid4().hex
        body = b''.join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f'Content-Type: video/mp4\r\n\r\n'.encode(), path.read_bytes(), f'\r\n--{boundary}--\r\n'.encode()])
        uploaded = self._call('POST', '/files', body, f'multipart/form-data; boundary={boundary}')
        record_path.parent.mkdir(exist_ok=True)
        record_path.write_text(json.dumps({'id': uploaded['id'], 'sha256': sha, 'name': path.name,
                                           'uploaded': _now(), 'response': uploaded}, indent=1), encoding='utf-8')
        return uploaded['id'], sha

    # -- segmentation -------------------------------------------------------------

    def is_cached(self, phrase, kind, options, media_sha):
        definition = {'model': self.model, 'kind': kind, 'phrase': phrase, 'media_sha256': media_sha,
                      'options': dict(options)}
        return (self.cache/'responses'/f'{json_sha256(definition)}.json').is_file()

    def segment(self, phrase, media_path, kind, options, content_sha=None, estimate_usd=None):
        """Segmentation text for one phrase on one image or video, plus the cache record.

        content_sha identifies what the media shows (e.g. a hash of its decoded pixels). When given, it
        keys the cache instead of the file bytes, so a re-encoded file with different metadata but the
        same pictures is still answered from the cache.
        """
        if kind not in ('image', 'video'):
            raise ValueError(kind)
        media_path = Path(media_path)
        media_sha = content_sha or file_sha256(media_path)
        definition = {'model': self.model, 'kind': kind, 'phrase': phrase, 'media_sha256': media_sha,
                      'options': dict(options)}
        key = json_sha256(definition)
        cached = self.cache/'responses'/f'{key}.json'
        if cached.is_file():
            record = json.loads(cached.read_text(encoding='utf-8'))
            return record['text'], {**record['summary'], 'cached': True, 'key': key}

        self.charge(estimate_usd if estimate_usd is not None else (0.01 if kind == 'video' else 0.0025),
                    f'SAM {kind} request "{phrase}"')
        if kind == 'video':
            file_id, _ = self.upload(media_path)
            media = {'type': 'input_video', 'file_id': file_id}
        else:
            encoded = base64.b64encode(media_path.read_bytes()).decode('ascii')
            media = {'type': 'input_image', 'image_url': f'data:image/png;base64,{encoded}'}
        request = {'model': self.model, 'stream': False,
                   'metadata': {name: str(value) for name, value in options.items()},
                   'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': phrase}, media]}]}
        response = self._call('POST', '/responses', json.dumps(request).encode('utf-8'), 'application/json')
        text = ''.join(part.get('text', '') for item in response.get('output', [])
                       for part in item.get('content', []) or [] if part.get('type') == 'output_text')
        summary = {'response_id': response.get('id'), 'status': response.get('status'),
                   'usage': response.get('usage'), 'received': _now(), 'model': response.get('model', self.model)}
        stored = {'definition': definition, 'text': text, 'summary': summary,
                  'response': {k: v for k, v in response.items() if k != 'output'}}
        temporary = cached.with_suffix('.tmp')
        temporary.write_text(json.dumps(stored, indent=1), encoding='utf-8')
        temporary.replace(cached)
        return text, {**summary, 'cached': False, 'key': key}
