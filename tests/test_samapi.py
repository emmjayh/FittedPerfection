"""SAM client: request shape, retries, upload once, and cached answers that never touch the network."""
import json

import pytest

from fittedperfection import seal
from fittedperfection.samapi import SamApiError, SamClient


class FakeApi:
    def __init__(self, failures=0):
        self.calls, self.failures = [], failures

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers, body))
        if self.failures:
            self.failures -= 1
            return 503, b'busy'
        if url.endswith('/files'):
            return 200, json.dumps({'id': 'file-1', 'status': 'uploaded'}).encode()
        request = json.loads(body)
        text = '<0f>1<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|>' if request['input'][0]['content'][0]['text'] == 'boot' else ''
        return 200, json.dumps({'id': 'resp-1', 'status': 'completed', 'usage': {'images': 1},
                                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}).encode()


@pytest.fixture
def media(tmp_path):
    (tmp_path/'frame.png').write_bytes(b'png bytes')
    (tmp_path/'orbit.mp4').write_bytes(b'mp4 bytes')
    return tmp_path


def client(tmp_path, api):
    return SamClient(tmp_path/'cache', 'sam-3.1', 'https://api.example/v1', api_key='k', transport=api, pause=0)


def test_image_request_shape_and_text(media, tmp_path):
    api = FakeApi()
    text, info = client(tmp_path, api).segment('boot', media/'frame.png', 'image', {'mask_encoding': 'one_bit'})
    assert text.startswith('<0f>1') and info['cached'] is False
    method, url, headers, body = api.calls[0]
    request = json.loads(body)
    assert url == 'https://api.example/v1/responses' and headers['Authorization'] == 'Bearer k'
    assert request['model'] == 'sam-3.1' and request['stream'] is False
    assert request['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,')
    assert request['metadata'] == {'mask_encoding': 'one_bit'}


def test_cached_answer_makes_no_network_call_even_when_sealed(media, tmp_path):
    first = client(tmp_path, FakeApi())
    first.segment('boot', media/'frame.png', 'image', {})
    api = FakeApi()
    again = client(tmp_path, api)
    with seal.sealed():                                    # any network use would raise here
        text, info = again.segment('boot', media/'frame.png', 'image', {})
    assert info['cached'] is True and api.calls == [] and text.startswith('<0f>')


def test_video_is_uploaded_once_and_referenced_by_id(media, tmp_path):
    api = FakeApi()
    sam = client(tmp_path, api)
    for phrase in ('boot', 'sleeve'):
        sam.segment(phrase, media/'orbit.mp4', 'video', {})
    uploads = [call for call in api.calls if call[1].endswith('/files')]
    assert len(uploads) == 1
    request = json.loads(api.calls[-1][3])
    assert request['input'][0]['content'][1] == {'type': 'input_video', 'file_id': 'file-1'}


def test_empty_result_is_valid(media, tmp_path):
    text, _ = client(tmp_path, FakeApi()).segment('cape', media/'frame.png', 'image', {})
    assert text == ''


def test_busy_server_is_retried_then_errors(media, tmp_path):
    assert client(tmp_path, FakeApi(failures=2)).segment('boot', media/'frame.png', 'image', {})[0]
    with pytest.raises(SamApiError, match='503'):
        client(tmp_path, FakeApi(failures=10)).segment('shoe', media/'frame.png', 'image', {})


def test_missing_key_is_refused(media, tmp_path):
    sam = SamClient(tmp_path/'cache', 'sam-3.1', 'https://api.example/v1', api_key=None, transport=FakeApi())
    with pytest.raises(SamApiError, match='MODEL_API_KEY'):
        sam.segment('boot', media/'frame.png', 'image', {})
