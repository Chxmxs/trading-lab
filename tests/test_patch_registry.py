import pandas as pd
from companion.patch_registry import apply_all_patches

def test_apply_all_patches_creates_timestamp_index(tmp_path):
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "open": [1, 2],
        "close": [2, 3]
    })
    out = apply_all_patches(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
