// "Fit a new outfit": drop an archive (or pick a listed one), follow the run stage by stage, approve spending
// only when asked, and show the result in the viewer when it completes (fittedperfection/app.py).

const el = (id) => document.getElementById(id);
const HEADERS = { 'X-FittedPerfection': '1' };            // the server refuses writes without it
let chosen = null;                                         // {upload} or {outfit}
let polling = null;

async function post(path, body, extra = {}) {
  const response = await fetch(path, { method: 'POST', headers: { ...HEADERS, ...extra }, body });
  const answer = await response.json();
  if (!response.ok) throw new Error(answer.error || response.statusText);
  return answer;
}

function note(text) { el('job-status').textContent = text || ''; }

async function upload(file) {
  if (!file.name.toLowerCase().endsWith('.zip')) { el('drop-note').textContent = 'Only .zip archives.'; return; }
  el('drop-note').textContent = `Checking ${file.name} (${(file.size / 1e6).toFixed(1)} MB)…`;
  try {
    const answer = await post('/api/uploads', file, { 'Content-Type': 'application/zip', 'X-Filename': file.name });
    chosen = { upload: answer.upload };
    el('pick-listed').value = '';
    el('run-name').value = answer.name;
    el('test-fit').checked = false;
    el('drop-note').textContent = `${file.name} is ready.` +
      (answer.held_out ? ` It is the held-out test outfit "${answer.held_out}": tick "test fit" to run it.` : '');
  } catch (error) {
    chosen = null;
    el('drop-note').textContent = error.message;
  }
}

function describe(stage) {
  if (stage.status === 'running') {
    const since = stage.started ? Math.round((Date.now() - Date.parse(stage.started)) / 1000) : 0;
    return `running ${since}s`;
  }
  if (stage.status === 'complete') return `done ${stage.seconds?.toFixed(1) ?? ''}s`;
  if (stage.status === 'stopped') return stage.needs_usd != null ? 'waiting for your OK' : 'stopped';
  return 'waiting';
}

function show(status) {
  el('job').hidden = status.state === 'idle';
  if (status.state === 'idle') return;
  el('job-stages').replaceChildren(...status.stages.map((stage) => {
    const item = document.createElement('li');
    item.className = stage.status || '';
    item.innerHTML = `<b></b><span></span>`;
    item.querySelector('b').textContent = stage.name;
    item.querySelector('span').textContent = describe(stage);
    if (stage.status === 'stopped' && stage.reason) item.title = stage.reason;
    return item;
  }));
  el('job-pictures').replaceChildren(...Object.entries(status.pictures || {}).map(([stage, url]) => {
    const image = document.createElement('img');
    image.src = `${url}?v=${stage}-${status.stages.find((s) => s.name === stage)?.seconds ?? 0}`;
    image.alt = `${stage} picture`;
    return image;
  }));
  el('job-log').textContent = (status.log || []).join('\n');
  el('cancel-fit').hidden = status.state !== 'running';
  el('spend').hidden = status.state !== 'needs_spend';
  if (status.state === 'needs_spend') {
    const requests = status.needs_requests != null ? `${status.needs_requests} SAM requests` : 'paid requests';
    el('spend-text').textContent = `Labelling "${status.name}" needs ${requests}, about $${status.needs_usd.toFixed(2)}.` +
      ' Nothing has been sent. Approve to go ahead.';
    el('wonder-usd').textContent = status.wonder_usd.toFixed(2);
  }
  const words = { running: `Fitting "${status.name}"…`, needs_spend: 'Waiting for your approval.',
    complete: `"${status.name}" is fitted. It is shown in the viewer.`, cancelled: 'Cancelled.',
    stopped: `Stopped: ${status.stages.find((s) => s.status === 'stopped')?.reason || 'see details'}` };
  note(words[status.state] || status.state);
}

async function poll(onComplete) {
  const status = await (await fetch('/api/jobs/current')).json();
  show(status);
  if (status.state === 'running') {
    polling = setTimeout(() => poll(onComplete), 1000);
  } else {
    polling = null;
    if (status.state === 'complete') onComplete(status.name);
  }
}

function follow(onComplete) {
  clearTimeout(polling);
  poll(onComplete).catch((error) => note(`Lost track of the fit: ${error.message}`));
}

export async function initApp({ onComplete }) {
  const listed = await (await fetch('/api/outfits')).json();
  const select = el('pick-listed');
  select.replaceChildren(new Option('– none –', ''), ...listed.filter((o) => o.present).map((o) =>
    new Option(o.split === 'test-fits' ? `${o.name} (held out)` : o.name, o.name)));
  select.addEventListener('change', () => {
    chosen = select.value ? { outfit: select.value } : null;
    if (select.value) {
      el('run-name').value = select.value;
      el('drop-note').textContent = '';
    }
  });

  const drop = el('drop');
  drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    drop.classList.remove('over');
    if (event.dataTransfer.files[0]) upload(event.dataTransfer.files[0]);
  });
  el('pick-file').addEventListener('change', (event) => { if (event.target.files[0]) upload(event.target.files[0]); });

  el('start-fit').addEventListener('click', async () => {
    if (!chosen) { note('Drop an archive or choose a listed outfit first.'); return; }
    try {
      show(await post('/api/jobs', JSON.stringify({ ...chosen, name: el('run-name').value,
        test_fit: el('test-fit').checked, allow_spend_usd: 0 }), { 'Content-Type': 'application/json' }));
      follow(onComplete);
    } catch (error) { note(error.message); }
  });
  el('spend-approve').addEventListener('click', async () => {
    try {
      show(await post('/api/jobs/approve', JSON.stringify({ wonder: el('spend-wonder').checked }),
        { 'Content-Type': 'application/json' }));
      follow(onComplete);
    } catch (error) { note(error.message); }
  });
  el('spend-decline').addEventListener('click', () => { el('spend').hidden = true; note('Not approved. Nothing was sent.'); });
  el('cancel-fit').addEventListener('click', async () => {
    try { show(await post('/api/jobs/cancel', '{}', { 'Content-Type': 'application/json' })); }
    catch (error) { note(error.message); }
  });
  follow(onComplete);                                       // pick up a fit already running after a reload
}
