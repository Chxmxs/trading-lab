from collections import deque
from typing import Any, Optional

__all__ = ["push", "pop", "peek", "size", "clear"]

_q: "deque[Any]" = deque()

def push(item: Any) -> None:
    _q.append(item)

def pop() -> Optional[Any]:
    return _q.popleft() if _q else None

def peek() -> Optional[Any]:
    return _q[0] if _q else None

def size() -> int:
    return len(_q)

def clear() -> None:
    _q.clear()
