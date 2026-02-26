Cross-Validation and Robustness
===============================

This tutorial shows how to assess the stability of DiD estimates using cross-validation.
These diagnostics help identify influential participants and quantify estimate robustness.

1. Fit baseline DiD
-------------------

.. code-block:: python

   import sctrial as st

   res = st.did_table(
       adata,
       features=["sig_IFN_Response", "sig_Cytotoxicity"],
       design=design,
       visits=("Pre", "Post"),
       aggregate="participant_visit",
       use_bootstrap=True,
       n_boot=499,
   )

2. Leave-one-out cross-validation (LOO)
-----------------------------------------------

.. code-block:: python

   loo = st.loo_cv_did(
       adata,
       feature="sig_IFN_Response",
       design=design,
       visits=("Pre", "Post"),
       aggregate="participant_visit",
       standardize=True,
   )
   print(loo.head())

   # Summarize influence
   summary = st.cv_summary(loo)
   print(summary)

3. K-fold cross-validation
--------------------------

.. code-block:: python

   kfold = st.kfold_cv_did(
       adata,
       feature="sig_IFN_Response",
       design=design,
       visits=("Pre", "Post"),
       k=5,
       aggregate="participant_visit",
       standardize=True,
       seed=42,
   )
   print(kfold.head())

4. Influence diagnostics
------------------------

.. code-block:: python

   influence = st.influence_diagnostics(kfold)
   print(influence.head())

Interpretation
--------------

- Large swings in beta_DiD when a participant is removed indicate sensitivity.
- Consider reporting robust results (median effect) or trimming highly influential
  participants as a sensitivity analysis.
