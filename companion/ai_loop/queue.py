import json
from pathlib import Path
from collections import deque
from typing import Any, Optional, Deque

__all__ = ["JobQueue", "push", "pop", "peek", "size", "clear"]

class JobQueue:
    """A minimal JSONL-backed FIFO queue for tests."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._q: Deque[Any] = deque()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._q.append(json.loads(line))
                    except Exception:
                        pass

    def _persist_all(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for item in self._q:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def add_job(self, job: Any) -> None:
        """Alias used in tests; appends a job dict to the JSONL queue."""
        self._q.append(job)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    def push(self, item: Any) -> None:
        self.add_job(item)

    def pop(self) -> Optional[Any]:
        if not self._q:
            return None
        item = self._q.popleft()
        self._persist_all()
        return item

    def peek(self) -> Optional[Any]:
        return self._q[0] if self._q else None

    def size(self) -> int:
        return len(self._q)

    def clear(self) -> None:
        self._q.clear()
        self._persist_all()

# In-memory helper queue
_tmp_q: Deque[Any] = deque()

def push(item: Any) -> None:
    _tmp_q.append(item)

def pop() -> Optional[Any]:
    return _tmp_q.popleft() if _tmp_q else None

def peek() -> Optional[Any]:
    return _tmp_q[0] if _tmp_q else None

def size() -> int:
    return len(_tmp_q)

def clear() -> None:
    _tmp_q.clear()
