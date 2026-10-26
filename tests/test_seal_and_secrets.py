"""The seal covers every thread; the API key comes from secrets.env or the environment and never while sealed."""
import socket
import threading

import pytest

from fittedperfection import seal, secrets


def test_seal_covers_other_threads():
    caught = []

    def worker():
        try:
            socket.socket()
        except seal.SealBroken as error:
            caught.append(error)

    with seal.sealed():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert len(caught) == 1 and not seal.is_sealed()


def test_nested_seals_open_only_when_all_close():
    with seal.sealed():
        with seal.sealed():
            pass
        assert seal.is_sealed()
    assert not seal.is_sealed()


def test_key_from_file_and_environment_wins(home, monkeypatch):
    (home/'secrets.env').write_text('# comment\nMODEL_API_KEY="from-file"\n', encoding='utf-8')
    assert secrets.api_key() == 'from-file'
    monkeypatch.setenv('MODEL_API_KEY', 'from-env')
    assert secrets.api_key() == 'from-env'


def test_missing_or_blank_key_is_an_error(home):
    with pytest.raises(ValueError, match='secrets.env'):
        secrets.api_key()
    (home/'secrets.env').write_text('MODEL_API_KEY=\n', encoding='utf-8')
    with pytest.raises(ValueError):
        secrets.api_key()


def test_key_refused_while_sealed(home):
    (home/'secrets.env').write_text('MODEL_API_KEY=k\n', encoding='utf-8')
    with seal.sealed(), pytest.raises(seal.SealBroken):
        secrets.api_key()
