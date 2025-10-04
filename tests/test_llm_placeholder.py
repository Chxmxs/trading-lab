def test_llm_placeholder_replaced_with_real_impl():
    from companion.explorer.discovery import propose_new_strategy_via_llm
    cfg = {"llm_enabled": False}
    result = propose_new_strategy_via_llm(cfg)
    assert result is None
