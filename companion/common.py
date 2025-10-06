# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime, timezone

def utc_now_str() -> str:
    """UTC timestamp string for folder/run naming. Format: YYYY-MM-DDTHHMMSS"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
