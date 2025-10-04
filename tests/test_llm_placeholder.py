def test_llm_placeholder():
    from companion.explorer.discovery import propose_new_strategy_via_llm
    import pytest
    with pytest.raises(NotImplementedError):
        propose_new_strategy_via_llm(None, "")
