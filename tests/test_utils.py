import pandas as pd
import pytest
from sctrial.utils import ensure_unique_index

def safe_filename(s: str, maxlen: int = 180) -> str:
    s = str(s)
    # Ensure we handle common greek letters often found in cell names
    s = s.replace("γ", "gamma").replace("δ", "delta").replace("α", "alpha").replace("β", "beta")
    s = re.sub(r"\s+", "_", s.strip())

def test_ensure_unique_index():
    df = pd.DataFrame({"val": [1, 3, 10]}, index=["A", "A", "B"])
    
    # Test mean aggregation
    df_mean = ensure_unique_index(df, agg="mean")
    assert df_mean.loc["A", "val"] == 2.0
    assert len(df_mean) == 2
    
    # Test sum aggregation
    df_sum = ensure_unique_index(df, agg="sum")
    assert df_sum.loc["A", "val"] == 4