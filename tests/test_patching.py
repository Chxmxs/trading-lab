import pandas as pd
from companion.patch_registry import apply_all_patches

def test_timestamp_patch_basic(tmp_path):
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "open": [1, 2],
        "close": [2, 3]
    })
    patched = apply_all_patches(df)
    assert "timestamp" not in patched.columns
    assert isinstance(patched.index, pd.DatetimeIndex)
