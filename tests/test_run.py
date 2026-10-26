"""The stage runner: records, reuse, setting aside, rerun-from, stop-after, fresh mode, sealing."""
import json
import socket
import subprocess

import pytest

from fittedperfection import seal
from fittedperfection.run import Run, RunError, Stage, seen_before

ran = []


def writes(name, text):
    def action(ctx, out):
        ran.append(name)
        (out/'out.txt').write_text(text(ctx) if callable(text) else text, encoding='utf-8')
        return {'stage': name}
    return action


def appends(name, upstream):
    return writes(name, lambda ctx: (ctx.output_of(upstream)/'out.txt').read_text(encoding='utf-8') + '>' + name)


def pipeline(first='a'):
    return (Stage('a', 'a', writes('a', first), garment=True, settings=('fit',)),
            Stage('b', 'deep/b', appends('b', 'a'), needs=('a',)),
            Stage('c', 'c', appends('c', 'b'), needs=('b',)))


@pytest.fixture
def runs(home):
    ran.clear()
    folder = home/'runs'
    folder.mkdir()
    return folder


def state_of(folder):
    return json.loads((folder/'run.json').read_text(encoding='utf-8'))


def test_first_run_writes_each_stage_with_its_record(runs, garment):
    state = Run(runs/'r', pipeline()).execute(garment, 'r')
    assert state['status'] == 'complete' and ran == ['a', 'b', 'c']
    assert (runs/'r/c/out.txt').read_text() == 'a>b>c'
    record = json.loads((runs/'r/deep/b/stage.json').read_text())
    assert record['inputs']['upstream'] == {'a': state['stages']['a']['outputs_sha256']}
    assert 'fit' in json.loads((runs/'r/a/stage.json').read_text())['inputs']['settings']
    assert state['fit_reviewed'] is False and state['game_ready'] is False
    assert (runs/'r/live.html').is_file() and (runs/'ledger.jsonl').is_file()


def test_unchanged_stages_are_reused(runs, garment):
    Run(runs/'r', pipeline()).execute(garment, 'r')
    ran.clear()
    Run(runs/'r', pipeline()).execute(garment, 'r')
    assert ran == []


def test_rerun_from_sets_later_stages_aside(runs, garment):
    Run(runs/'r', pipeline()).execute(garment, 'r')
    ran.clear()
    state = Run(runs/'r', pipeline()).execute(garment, 'r', rerun_from='b')
    assert ran == ['b', 'c']
    assert (runs/'r/deep/b.stale-1/out.txt').is_file() and (runs/'r/c.stale-1/out.txt').is_file()
    assert [entry['why'] for entry in state['set_aside']] == ['--rerun-from b', '--rerun-from b']


def test_identical_rerun_keeps_later_stages(runs, garment):
    Run(runs/'r', pipeline()).execute(garment, 'r')
    ran.clear()
    (runs/'r/a/out.txt').write_text('edited by hand')
    Run(runs/'r', pipeline()).execute(garment, 'r')
    assert ran == ['a']                       # a rewrites the same bytes, so b and c stay
    assert (runs/'r/a.stale-1/out.txt').read_text() == 'edited by hand'


def test_changed_output_reruns_everything_after_it(runs, garment):
    Run(runs/'r', pipeline('first')).execute(garment, 'r')
    ran.clear()
    state = state_of(runs/'r')
    state['stages']['a']['inputs_sha256'] = 'pretend the code changed'
    (runs/'r/run.json').write_text(json.dumps(state))
    Run(runs/'r', pipeline('second')).execute(garment, 'r')
    assert ran == ['a', 'b', 'c']
    assert (runs/'r/c/out.txt').read_text() == 'second>b>c'
    assert (runs/'r/c.stale-1/out.txt').read_text() == 'first>b>c'


def test_a_stopped_stage_keeps_its_partial_output(runs, garment):
    def fails(ctx, out):
        (out/'half.txt').write_text('half done')
        raise ValueError('toe direction is ambiguous')
    with pytest.raises(ValueError):
        Run(runs/'r', (Stage('a', 'a', writes('a', 'a')), Stage('b', 'b', fails, needs=('a',)))).execute(garment, 'r')
    state = state_of(runs/'r')
    assert state['status'] == 'stopped_at_b' and 'ambiguous' in state['stages']['b']['reason']
    assert (runs/'r/b.partial/half.txt').is_file() and not (runs/'r/b').exists()
    Run(runs/'r', (Stage('a', 'a', writes('a', 'a')), Stage('b', 'b', writes('b', 'b'), needs=('a',)))).execute(garment, 'r')
    assert (runs/'r/b.partial.failed-1/half.txt').is_file() and (runs/'r/b/out.txt').is_file()


def test_stop_after(runs, garment):
    state = Run(runs/'r', pipeline()).execute(garment, 'r', stop_after='b')
    assert state['status'] == 'prepared_through_b' and ran == ['a', 'b']


def test_a_run_name_belongs_to_one_garment(runs, garment, tmp_path):
    Run(runs/'r', pipeline()).execute(garment, 'r')
    other = tmp_path/'other.zip'
    other.write_bytes(b'other bytes')
    with pytest.raises(RunError, match='other garment bytes'):
        Run(runs/'r', pipeline()).execute(other, 'r')


def test_never_writes_into_a_foreign_folder(runs, garment):
    (runs/'r').mkdir()
    with pytest.raises(RunError, match='refusing'):
        Run(runs/'r', pipeline()).execute(garment, 'r')


def test_fresh_refuses_seen_bytes_and_writes_provenance(runs, garment, tmp_path):
    state = Run(runs/'first', pipeline()).execute(garment, 'first', fresh=True)
    card = json.loads((runs/'first/provenance.json').read_text())
    assert card['garment sha256'] == state['garment_sha256'] and 'finished' in card
    with pytest.raises(RunError, match='not a first fit'):
        Run(runs/'second', pipeline()).execute(garment, 'second', fresh=True)
    assert not (runs/'second').exists()
    (runs/'first').rename(tmp_path/'moved-away')        # the ledger still remembers
    assert state['garment_sha256'] in seen_before(runs)


def test_fresh_card_exists_even_when_a_stage_stops(runs, garment):
    def fails(ctx, out):
        raise NotImplementedError('not built yet')
    with pytest.raises(NotImplementedError):
        Run(runs/'f', (Stage('a', 'a', fails),)).execute(garment, 'f', fresh=True)
    assert json.loads((runs/'f/provenance.json').read_text())['garment file'] == 'garment.zip'


def test_sealed_stage_cannot_use_network_or_programs(runs, garment):
    def network(ctx, out):
        socket.socket()

    def program(ctx, out):
        subprocess.run(['git', '--version'])

    for action in (network, program):
        with pytest.raises(seal.SealBroken):
            Run(runs/action.__name__, (Stage('fit', 'fit', action, sealed=True),)).execute(garment, action.__name__)
    assert not seal.is_sealed()
    socket.socket().close()


def test_unknown_stage_names_are_refused(runs, garment):
    with pytest.raises(RunError, match='unknown stage'):
        Run(runs/'r', pipeline()).execute(garment, 'r', rerun_from='nope')
