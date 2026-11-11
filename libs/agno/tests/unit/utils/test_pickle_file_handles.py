import gc
import warnings
from pathlib import Path

from agno.utils.pickle import pickle_object_to_file, unpickle_object_from_file


class _TrackedPath(type(Path())):
    """Path subclass that records every file object opened via .open()."""

    opened: list = []

    def open(self, *args, **kwargs):
        f = super().open(*args, **kwargs)
        _TrackedPath.opened.append(f)
        return f


def test_pickle_object_to_file_closes_handle(tmp_path):
    _TrackedPath.opened = []
    path = _TrackedPath(tmp_path / "obj.pkl")
    pickle_object_to_file({"a": 1}, path)
    assert len(_TrackedPath.opened) == 1
    assert all(f.closed for f in _TrackedPath.opened), "file handle leaked in pickle_object_to_file"


def test_unpickle_object_from_file_closes_handle(tmp_path):
    plain = tmp_path / "obj.pkl"
    pickle_object_to_file({"a": 1}, plain)
    _TrackedPath.opened = []
    path = _TrackedPath(plain)
    assert unpickle_object_from_file(path) == {"a": 1}
    assert len(_TrackedPath.opened) == 1
    assert all(f.closed for f in _TrackedPath.opened), "file handle leaked in unpickle_object_from_file"
