# -*- coding: utf-8 -*-
import json
from pathlib import Path

from companion.explorer.context_builder import build_data_map
from companion.explorer import build_prompt_context, enrich_prompt_with_context

def test_build_prompt_context_smoke(tmp_path, monkeypatch):
    # Build the real map (ok if empty)
    m = build_data_map()
    ctx = build_prompt_context()
    assert "available_symbols" in ctx
    assert "timeframes_by_symbol" in ctx
    # Ensure enrich adds DATA CONTEXT block
    prompt = enrich_prompt_with_context("Base")
    assert "## DATA CONTEXT" in prompt
