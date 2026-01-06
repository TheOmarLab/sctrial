import pandas as pd

from sctrial.utils import ensure_unique_index


def test_ensure_unique_index():
    df = pd.DataFrame({"val": [1, 3, 10]}, index=["A", "A", "B"])

    # Test mean aggregation
    df_mean = ensure_unique_index(df, agg="mean")
    assert df_mean.loc["A", "val"] == 2.0
    assert len(df_mean) == 2

    # Test sum aggregation
    df_sum = ensure_unique_index(df, agg="sum")
    assert df_sum.loc["A", "val"] == 4
