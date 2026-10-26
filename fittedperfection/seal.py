"""Seal the process while the fit runs: no network, no other programs (spec §2.1).

`sealed()` closes the whole process, every thread, for as long as any sealed
block is open. An audit hook (the interpreter-wide interception point) then
refuses whole families of events: all socket use, URL and protocol clients,
and every way of starting a program. Compiled code that calls the operating
system directly is out of any audit hook's reach, so the fit itself is kept
to numpy, scipy and FittedPerfection's own modules.
"""
import sys
import threading
from contextlib import contextmanager

REFUSED = (
    'socket.', 'urllib.', 'http.client.', 'ftplib.', 'smtplib.', 'poplib.', 'imaplib.', 'nntplib.',
    'webbrowser.', 'subprocess.', 'os.system', 'os.spawn', 'os.posix_spawn', 'os.exec', 'os.fork',
    'os.startfile', '_winapi.CreateProcess', 'pty.',
)


class SealBroken(RuntimeError):
    """Sealed code tried to reach the network or start a program."""


_lock = threading.Lock()
_open_blocks = 0


def is_sealed():
    return _open_blocks > 0


def _audit(event, args):
    if _open_blocks and event.startswith(REFUSED):
        raise SealBroken(f'sealed: {event} refused')


sys.addaudithook(_audit)


@contextmanager
def sealed():
    global _open_blocks
    with _lock:
        _open_blocks += 1
    try:
        yield
    finally:
        with _lock:
            _open_blocks -= 1
