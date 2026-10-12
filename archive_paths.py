"""Resolve A100 storage paths and old absolute symlinks inside the NAS mount."""
import os
from functools import lru_cache
from pathlib import Path

NAS = Path(os.environ.get('JEPA_NAS_ROOT', '/data1/liuyeqiang/nas/A100'))
PREFIXES = (
    ('/data1/LWR', NAS / 'data1/LWR'),
    ('/data3/testdata', NAS / 'testdata'),
    ('/data2/usr_for_deadline', NAS / 'usr_for_deadline'),
)


def rebase_archive_path(path):
    value = str(path)
    for old, new in PREFIXES:
        if value == old or value.startswith(old + '/'):
            return Path(str(new) + value[len(old):])
    return Path(value)


@lru_cache(maxsize=32768)
def resolve_archive_path(path):
    """Follow each archived symlink after rebasing its original storage root."""
    path = rebase_archive_path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.parent == path:
        return path
    parent = resolve_archive_path(str(path.parent))
    candidate = parent / path.name
    if candidate.is_symlink():
        target = Path(os.readlink(candidate))
        if not target.is_absolute():
            target = candidate.parent / target
        return resolve_archive_path(str(target))
    return candidate
