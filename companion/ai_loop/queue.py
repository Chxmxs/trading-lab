# -*- coding: utf-8 -*-
"""
queue.py — minimal JSONL-backed job queue for the AI loop.

Each line in queue.jsonl is a JSON object:
  {"run_id":"...", "priority":"failed_first|low_oos|low_trades|normal", "added_ts":"ISO-UTC"}

Priority ordering (lowest score pops first):
  failed_first (0) < low_oos (1) < low_trades (2) < normal (5)

Public API:
  - JobQueue()
  - add_item(item: dict) -> None
  - pop_next() -> dict|None
  - reprioritize(run_id: str, new_priority: str) -> bool
  - snapshot() -> dict
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from companion.logging_config import configure_logging

log = configure_logging(__name__)

AI_LOOP_DIR = Path("companion/ai_loop")
AI_LOOP_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_PATH = AI_LOOP_DIR / "queue.jsonl"

PRIORITY_SCORE = {
    "failed_first": 0,
    "low_oos": 1,
    "low_trades": 2,
    "normal": 5,
}

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class JobQueue:
    """JSONL-backed queue with simple priority + FIFO per priority."""

    def __init__(self, path: Path = QUEUE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            # create empty file
            self.path.write_text("", encoding="utf-8")

    def _load_all(self) -> List[Dict]:
        if not self.path.exists():
            return []
        items: List[Dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except Exception as e:
                    log.error("Skipping corrupt queue line: %s (%s)", line[:120], e)
        return items

    def _save_all(self, items: List[Dict]) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def add_item(self, item: Dict) -> None:
        """Add a job. Ensures required fields and avoids dupes by run_id+priority."""
        run_id = item.get("run_id")
        if not run_id:
            raise ValueError("item.run_id is required")
        priority = item.get("priority", "failed_first")
        if priority not in PRIORITY_SCORE:
            priority = "normal"
        item = {
            "run_id": run_id,
            "priority": priority,
            "added_ts": item.get("added_ts", _utcnow_iso()),
            **{k: v for k, v in item.items() if k not in ("run_id", "priority", "added_ts")},
        }
        items = self._load_all()
        if any(x.get("run_id") == run_id and x.get("priority") == priority for x in items):
            log.info("Queue already contains run_id=%s priority=%s", run_id, priority)
            return
        items.append(item)
        self._save_all(items)
        log.info("Queued run_id=%s priority=%s", run_id, priority)

    def _sort_key(self, obj: Dict):
        score = PRIORITY_SCORE.get(obj.get("priority", "normal"), 5)
        # earlier added_ts dequeues earlier within same priority
        ts = obj.get("added_ts") or _utcnow_iso()
        return (score, ts)

    def pop_next(self) -> Optional[Dict]:
        """Pop the next job by priority then FIFO; returns the job or None if empty."""
        items = self._load_all()
        if not items:
            return None
        items_sorted = sorted(items, key=self._sort_key)
        next_item = items_sorted.pop(0)
        # remove the specific instance we popped
        removed = False
        new_items = []
        for it in items:
            if (not removed and it.get("run_id") == next_item.get("run_id")
                and it.get("priority") == next_item.get("priority")
                and it.get("added_ts") == next_item.get("added_ts")):
                removed = True
                continue
            new_items.append(it)
        self._save_all(new_items)
        log.info("Dequeued run_id=%s priority=%s", next_item.get("run_id"), next_item.get("priority"))
        return next_item

    def reprioritize(self, run_id: str, new_priority: str) -> bool:
        if new_priority not in PRIORITY_SCORE:
            new_priority = "normal"
        items = self._load_all()
        changed = False
        for it in items:
            if it.get("run_id") == run_id:
                it["priority"] = new_priority
                changed = True
        if changed:
            self._save_all(items)
            log.info("Reprioritized run_id=%s -> %s", run_id, new_priority)
        return changed

    def snapshot(self) -> Dict:
        items = self._load_all()
        counts = {}
        for it in items:
            p = it.get("priority", "normal")
            counts[p] = counts.get(p, 0) + 1
        return {"path": str(self.path), "count": len(items), "by_priority": counts, "items": items}
