Mixed Effects Models
====================

Mixed effects models can be a useful sensitivity analysis when fixed-effects DiD may
be unstable or over-parameterized. This tutorial compares fixed-effects and mixed-effects
DiD results.

1. Run mixed-effects DiD
------------------------

.. code-block:: python

   import sctrial as st

   res_mixed = st.did_table_mixed(
       adata,
       features=["sig_IFN_Response", "sig_Cytotoxicity"],
       design=design,
       visits=("Pre", "Post"),
       aggregate="participant_visit",
   )
   print(res_mixed.head())

2. Compare fixed vs mixed effects
---------------------------------

.. code-block:: python

   comparison = st.compare_fixed_vs_mixed(
       adata,
       features=["sig_IFN_Response"],
       design=design,
       visits=("Pre", "Post"),
       aggregate="participant_visit",
   )
   print(comparison)

Interpretation
--------------

- Fixed effects control for all time-invariant participant differences.
- Mixed effects estimate participant variability via random intercepts.
- If fixed- and mixed-effects results diverge strongly, report both as a sensitivity
  analysis and discuss model assumptions.
