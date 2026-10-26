"""SHA-256 of files, folders and JSON values."""
import hashlib
import json
from pathlib import Path

_BLOCK = 4 << 20


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value):
    """Order-independent hash of anything json.dumps can write."""
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def folder_manifest(folder, skip=()):
    """{posix relative path: sha256} for every file below folder, except names in skip."""
    folder = Path(folder)
    manifest = {}
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder).as_posix()
        if path.is_file() and relative not in skip:
            manifest[relative] = file_sha256(path)
    return manifest


def folder_sha256(folder, skip=()):
    return json_sha256(folder_manifest(folder, skip))
