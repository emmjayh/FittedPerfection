"""The two Meta models FittedPerfection calls, pinned (spec §4a).

Both run on the Meta Model API, never inside the fit. A stage that uses one
records its pin, so changing a pin reruns that stage and everything after it.
"""
from importlib import metadata
from types import MappingProxyType

API_BASE = 'https://api.meta.ai/v1'

PINS = MappingProxyType({
    'sam': MappingProxyType({
        'model': 'sam-3.1',
        'endpoint': API_BASE + '/responses',
        'decoder': 'meta-sam-parser==0.1.0',
        'job': 'part segmentation from the fixed vocabulary (evidence only)',
    }),
    'wonder': MappingProxyType({
        'model': 'muse-wonder-1.0',
        'endpoint': API_BASE + '/responses',
        'job': 'visual review of labels and fits (findings only, never acceptance)',
    }),
})


class PinMismatch(RuntimeError):
    pass


def pin(key):
    return dict(PINS[key])


def check_decoder():
    """The installed mask decoder must be the pinned version."""
    name, version = PINS['sam']['decoder'].split('==')
    installed = metadata.version(name)
    if installed != version:
        raise PinMismatch(f'{name} {installed} is installed; FittedPerfection is pinned to {version}')
    return installed
