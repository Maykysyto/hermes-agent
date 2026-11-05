"""Durable, profile-local originals for Hermes memory/skill proposals (stdlib only)."""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time


def _identity(subsystem: str, pid: str) -> None:
    if subsystem not in ('memory', 'skills'):
        raise ValueError('invalid subsystem')
    if not isinstance(pid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', pid):
        raise ValueError('invalid proposal id')


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _private_dir(path: Path) -> None:
    if not path.exists():
        if not path.parent.exists():
            _private_dir(path.parent)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _sync_dir(path.parent)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f'not a directory: {path}')
    path.chmod(0o700)


def _write(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f'not a regular file: {path}')
        return stream.read()


def _record(raw: bytes, subsystem: str, pid: str | None = None) -> dict:
    value = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('proposal must be a JSON object')
    _identity(subsystem, value.get('id'))
    if value.get('subsystem') != subsystem or (pid is not None and value['id'] != pid):
        raise ValueError('proposal identity mismatch')
    return value


def _archive(home: Path, subsystem: str, pid: str, raw: bytes) -> Path:
    _identity(subsystem, pid)
    home = Path(home).resolve()
    digest = hashlib.sha256(raw).hexdigest()
    parent = home / 'write-archive' / subsystem / pid
    _private_dir(parent)
    target = parent / digest
    identity = dict(profile_home=str(home), subsystem=subsystem, id=pid, sha256=digest)
    if not target.exists():
        temporary = Path(tempfile.mkdtemp(prefix='.capture-', dir=parent))
        try:
            _write(temporary / 'original.json', raw)
            _write(temporary / 'receipt.json', json.dumps(dict(identity, captured_at=time.time()),
                                                        ensure_ascii=False).encode('utf-8'))
            _sync_dir(temporary)
            try:
                os.rename(temporary, target)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    # Also sync on idempotent retries: a previous caller may have failed its final fsync.
    _sync_dir(parent)
    if target.is_symlink() or not target.is_dir():
        raise ValueError('invalid archive directory')
    if _read(target / 'original.json') != raw:
        raise ValueError('archive bytes are corrupt; refusing to replace evidence')
    receipt = json.loads(_read(target / 'receipt.json'))
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in identity.items()) or not receipt.get('captured_at'):
        raise ValueError('archive receipt is corrupt; refusing to replace evidence')
    return target / 'original.json'


def archive_record(home: Path, subsystem: str, raw: bytes) -> Path:
    """Capture valid native record bytes without normalizing or overwriting versions."""
    record = _record(raw, subsystem)
    return _archive(home, subsystem, record['id'], raw)


def stage_record(home: Path, subsystem: str, record: dict) -> dict:
    """Serialize once; archive durably before atomically publishing pending."""
    raw = json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8') + b'\n'
    _record(raw, subsystem)
    archive_record(home, subsystem, raw)
    parent = Path(home).resolve() / 'pending' / subsystem
    _private_dir(parent)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # link publishes atomically, refusing any existing pending record.
        pending = parent / (record['id'] + '.json')
        os.link(temporary, pending)
        try:
            _sync_dir(parent)
        except OSError:
            pending.unlink()
            _sync_dir(parent)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    return record


def archive_existing(home: Path, subsystem: str, pending_id: str) -> bool:
    """Capture current pending bytes; malformed records remain pending and raise."""
    _identity(subsystem, pending_id)
    path = Path(home).resolve() / 'pending' / subsystem / (pending_id + '.json')
    try:
        raw = _read(path)
    except FileNotFoundError:
        return False
    _archive(home, subsystem, pending_id, raw)
    _record(raw, subsystem, pending_id)
    return True
