import pandas as pd
import pytest

def test_purged_kfold_import_and_split():
    """
    Ensures PurgedKFold from mlfinlab.cross_validation.cross_validation works
    and splits an indexed dataframe without error.
    """
    try:
        from mlfinlab.cross_validation.cross_validation import PurgedKFold
    except ImportError:
        pytest.skip("mlfinlab not installed in this env")

    df = pd.DataFrame({"x": range(100)}, index=pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"))
    pkf = PurgedKFold(n_splits=5, samples_info_sets=df.index, embargo_td=2)
    splits = list(pkf.split(None))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
