import json
import os

import pytest

from xunit_rag.utils import atomic_json


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing violations")
@pytest.mark.parametrize("persistent", [False, True])
def test_busy_atomic_checkpoint_preserves_valid_destination(tmp_path, monkeypatch, persistent):
    path = tmp_path / "checkpoint.json"
    atomic_json(path, {"state": "previous_complete"})
    replace = os.replace
    attempts = []

    def busy_replace(source, destination):
        attempts.append(1)
        assert json.loads(path.read_text(encoding="utf-8")) == {"state": "previous_complete"}
        if persistent or len(attempts) < 3:
            error = PermissionError("temporary sharing violation")
            error.winerror = 32
            raise error
        return replace(source, destination)

    monkeypatch.setattr("xunit_rag.utils.os.replace", busy_replace)
    monkeypatch.setattr("xunit_rag.utils.time.sleep", lambda _: None)
    if persistent:
        with pytest.raises(PermissionError):
            atomic_json(path, {"state": "new_complete"})
        assert len(attempts) == 6
        assert json.loads(path.read_text(encoding="utf-8"))["state"] == "previous_complete"
        temp = list(tmp_path.glob("*.tmp"))
        assert len(temp) == 1 and json.loads(temp[0].read_text(encoding="utf-8"))["state"] == "new_complete"
    else:
        atomic_json(path, {"state": "new_complete"})
        assert len(attempts) == 3
        assert json.loads(path.read_text(encoding="utf-8"))["state"] == "new_complete"
        assert not list(tmp_path.glob("*.tmp"))
