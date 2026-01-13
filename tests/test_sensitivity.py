import sctrial as st


def test_e_value_rr_basic():
    e, e_ci = st.e_value_rr(2.0, ci_lower=1.5, ci_upper=2.5)
    assert e > 1.0
    assert e_ci is not None


def test_e_value_rr_invalid_ci_order():
    try:
        st.e_value_rr(2.0, ci_lower=2.5, ci_upper=1.5)
        assert False, "Expected ValueError for ci_lower > ci_upper"
    except ValueError:
        pass


def test_e_value_rr_estimate_outside_ci():
    try:
        st.e_value_rr(3.0, ci_lower=1.5, ci_upper=2.5)
        assert False, "Expected ValueError for estimate outside CI bounds"
    except ValueError:
        pass
