"""Reading SAM 3.1 output: a real response (the boots on moonshorts' front view) parses into placed masks."""
from pathlib import Path

from fittedperfection import vocabulary
from fittedperfection.vision import parse, slug

REAL = (Path(__file__).parent/'data'/'sam-boot-front.txt').read_text(encoding='utf-8')


def test_real_image_response_parses_into_box_local_masks():
    records = parse('image', REAL)
    assert len(records) == 2
    for frame, object_id, confidence, (left, top, right, bottom), raster in records:
        assert frame is None and 0.8 < confidence <= 1
        assert raster.shape == (bottom - top, right - left)       # box-local, half-open bounds
        assert 0.3 < raster.mean() < 0.95
    assert {r[1] for r in records} == {'0', '1'}


def test_empty_output_means_no_matches():
    assert parse('image', '') == [] and parse('video', '') == []


def test_vocabulary_is_v5_and_complete():
    assert vocabulary.VERSION == 5 and len(vocabulary.phrases()) == 42
    assert vocabulary.PHRASE_CLASS['sneaker'] == 'footwear' and vocabulary.LENGTHS['capri pants'] == 'cropped'
    assert not set(vocabulary.REJECTED) & set(vocabulary.phrases())
    assert slug('knee-length shorts') == 'knee-length-shorts'
