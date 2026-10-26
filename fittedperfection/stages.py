"""The seven pipeline stages in order (spec §4).

Each stage is declared here once: its folder, what it reads and the code that
implements it. A stage that is not built yet stops the run and names the
build step that adds it; its `code` entry is filled in when it is built.
"""
from . import intake, label, views, vision
from .run import Stage


def not_built(step):
    def action(ctx, out):
        raise NotImplementedError(f'not built yet (build step {step})')
    return action


PIPELINE = (
    Stage('intake', 'full', intake.run, code=intake.CODE, garment=True, settings=('intake',),
          tools=('blender',)),
    Stage('views', 'analysis/views', views.run, code=views.CODE, needs=('intake',), settings=('views',),
          tools=('blender',)),
    Stage('vision', 'analysis/vision', vision.run, code=vision.CODE, needs=('views',), settings=('vision',),
          models=('sam',)),
    Stage('label', 'labels', label.run, code=label.CODE, needs=('intake', 'views', 'vision'),
          settings=('label',)),
    Stage('bind', 'bind', not_built(3), needs=('intake', 'label'), settings=('bind',)),
    Stage('fit', 'fit', not_built(4), needs=('intake', 'label', 'bind'), settings=('fit',), bodies=True,
          sealed=True),
    Stage('review', 'review', not_built(8), needs=('intake', 'fit'), settings=('audit',),
          models=('wonder',), bodies=True, tools=('blender',)),
)

NAMES = tuple(stage.name for stage in PIPELINE)
