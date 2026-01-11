import sctrial as st


def test_e_value_rr_basic():
    e, e_ci = st.e_value_rr(2.0, ci_lower=1.5, ci_upper=2.5)
    assert e > 1.0
    assert e_ci is not None
