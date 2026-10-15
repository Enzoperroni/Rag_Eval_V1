import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else canonical(value)).encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(value, encoding="utf-8")
    # Antivirus/indexers can briefly deny a Windows rename after the write handle is closed.
    # Retry only the atomic replacement; keep the old destination and the completed temp on failure.
    for attempt in range(6):
        try:
            os.replace(temp, path)
            return
        except OSError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def write_jsonl(path, values):
    atomic_text(
        path, "".join(canonical(v.model_dump() if hasattr(v, "model_dump") else v) + "\n" for v in values)
    )


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def utc_now():
    return datetime.now(UTC).isoformat()


class Trace:
    def __init__(self, path, run_id):
        self.path, self.run_id = Path(path), run_id
        self.started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **payload):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                canonical({"run_id": self.run_id, "time": utc_now(), "event": event, **payload}) + "\n"
            )


@contextmanager
def timed(trace, event, **payload):
    start = time.monotonic()
    try:
        yield
    finally:
        trace.emit(event, seconds=time.monotonic() - start, **payload)
