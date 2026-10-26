"""The fixed SAM 3.1 vocabulary, version 5 (spec §5.3). No per-outfit prompts.

Each phrase is sent on its own (SAM 3.1 takes one concept per request) and its
masks count as evidence for one part class. The length phrases only decide a
legwear length class for bind. A change here is a new version and reruns vision.
"""
from types import MappingProxyType

VERSION = 5

CLASSES = MappingProxyType({
    'torso': ('shirt', 'jacket', 'top', 'hoodie', 'vest', 'coat', 'sweater', 'dress'),
    'sleeve': ('sleeve',),
    'cuff': ('sleeve cuff',),
    'collar': ('collar',),
    'hood': ('hood',),
    'waistband': ('waistband', 'belt'),
    'pants': ('pants', 'shorts', 'jeans'),
    'pant_leg': ('pant leg', 'sock', 'stocking', 'tights', 'leggings'),
    'skirt': ('skirt',),
    'footwear': ('boot', 'shoe', 'sneaker'),
    'hardware': ('buckle', 'metal clasp', 'chain'),
    'attachment': ('pocket', 'strap', 'zipper'),
    'hanging': ('coat tail', 'cape'),
    'skin': ('skin', 'bare skin', 'bare leg', 'bare back', 'bare shoulders'),
})

# Evidence only: which of these matches best sets the legwear length class (and hem fraction in bind).
LENGTHS = MappingProxyType({
    'shorts': 'shorts',
    'knee-length shorts': 'knee',
    'capri pants': 'cropped',
    'long pants': 'full',
})

# Tried and rejected: they over-select. Kept here so they cannot creep back in.
REJECTED = ('mannequin', 'bare arm', 'exposed skin')

PHRASE_CLASS = MappingProxyType({phrase: name for name, phrases in CLASSES.items() for phrase in phrases})


def phrases():
    """Every distinct phrase to send, in a fixed order: class phrases first, then length-only ones."""
    ordered = list(PHRASE_CLASS)
    ordered += [phrase for phrase in LENGTHS if phrase not in PHRASE_CLASS]
    return tuple(ordered)


def check():
    if set(REJECTED) & set(phrases()):
        raise ValueError(f'rejected phrases in the vocabulary: {set(REJECTED) & set(phrases())}')
    if len(set(PHRASE_CLASS)) != sum(len(p) for p in CLASSES.values()):
        raise ValueError('a phrase is listed under two classes')
    return True


check()
