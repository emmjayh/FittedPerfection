"""live.html for the stage demo (spec §4), rewritten whenever run.json changes.

One card per stage, left to right; the provenance card in stage mode; each
stage's picture once it exists. It reloads itself every two seconds until
the run ends, and works opened straight from disk.
"""
import html
import time
from pathlib import Path

PICTURES = (
    ('analysis/views/front.png', 'The generated outfit'),
    ('labels/review.png', 'Parts found by SAM 3.1'),
    ('review/comparison.png', 'On the four bodies'),
)
LOOK = {'complete': 'done', 'running': 'busy', 'stopped': 'fail'}

STYLE = """
:root{color-scheme:dark}
body{margin:0;padding:28px 32px;background:#101418;color:#dfe6ee;font:15px/1.45 'Segoe UI',system-ui,sans-serif}
h1{font-size:24px;margin:0;display:inline-block;margin-right:14px}
.state{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#93a1b0}
p{color:#93a1b0;margin:6px 0 22px}
.stages{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:26px}
.stage{background:#1a2027;border-radius:10px;border-top:4px solid #3a444f;padding:10px 12px}
.stage b{display:block}.stage span{color:#93a1b0;font-size:12px;word-break:break-word}
.done{border-color:#4cc38a}.busy{border-color:#f0b429}.fail{border-color:#ef5b5b}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px;font-size:13px;margin:0 0 26px}
dt{color:#93a1b0}dd{margin:0;font-family:Consolas,monospace}
figure{display:inline-block;margin:0 18px 18px 0;vertical-align:top}
img{max-width:min(900px,100%);max-height:520px;border-radius:8px;background:#1a2027}
figcaption{color:#93a1b0;font-size:13px}
"""


def _stage_card(name, record):
    status = record.get('status', 'waiting')
    if status == 'stopped':
        note = record.get('reason', '')
    elif 'seconds' in record:
        note = f'{status} · {record["seconds"]:.1f} s'
    else:
        note = status
    return f'<div class="stage {LOOK.get(status, "")}"><b>{html.escape(name)}</b><span>{html.escape(note)}</span></div>'


def write(folder, state, order):
    folder = Path(folder)
    status = str(state.get('status', ''))
    running = status in ('new', 'running')
    cards = ''.join(_stage_card(name, state.get('stages', {}).get(name, {})) for name in order)
    provenance = ''.join(f'<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>'
                         for k, v in (state.get('provenance') or {}).items())
    stamp = int(time.time())
    pictures = ''.join(f'<figure><img src="{src}?v={stamp}" alt=""><figcaption>{html.escape(caption)}</figcaption>'
                       f'</figure>' for src, caption in PICTURES if (folder/src).is_file())
    name = html.escape(state.get('name', folder.name))
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        + ('<meta http-equiv="refresh" content="2">' if running else '')
        + f'<title>{name} · FittedPerfection</title><style>{STYLE}</style></head><body>'
        + f'<h1>{name}</h1><span class="state">{html.escape(status)}</span>'
        + '<p>SAM 3.1 finds the outfit\'s parts. The fit itself is plain local maths with the network and '
          'other programs shut off. Nothing here is accepted until a person has reviewed it.</p>'
        + f'<section class="stages">{cards}</section>'
        + (f'<dl>{provenance}</dl>' if provenance else '')
        + f'<section>{pictures}</section></body></html>\n'
    )
    temporary = folder/'live.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(folder/'live.html')
