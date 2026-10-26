"""The seven pipeline stages in order (spec §4).

Each stage is declared here once: its folder, what it reads and the code that
implements it. A stage that is not built yet stops the run and names the
build step that adds it; its `code` entry is filled in when it is built.
"""
from .run import Stage


def not_built(step):
    def action(ctx, out):
        raise NotImplementedError(f'not built yet (build step {step})')
    return action


PIPELINE = (
    Stage('intake', 'full', not_built(3), garment=True, settings=('intake',), tools=('blender',)),
    Stage('views', 'analysis/views', not_built(3), needs=('intake',), settings=('views',),
          tools=('blender',)),
    Stage('vision', 'analysis/vision', not_built(3), needs=('views',), settings=('vision',),
          models=('sam',)),
    Stage('label', 'labels', not_built(3), needs=('intake', 'views', 'vision'), settings=('label',)),
    Stage('bind', 'bind', not_built(3), needs=('intake', 'label'), settings=('bind',)),
    Stage('fit', 'fit', not_built(4), needs=('intake', 'label', 'bind'), settings=('fit',), bodies=True,
          sealed=True),
    Stage('review', 'review', not_built(8), needs=('intake', 'fit'), settings=('audit',),
          models=('wonder',), bodies=True, tools=('blender',)),
)

NAMES = tuple(stage.name for stage in PIPELINE)
