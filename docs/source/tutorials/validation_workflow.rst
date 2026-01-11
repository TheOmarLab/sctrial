Validation Workflow
===================

This tutorial shows how to validate trial data before running inference. The goal is to
catch common problems early: missing columns, unpaired participants, and features that
are not present in the dataset.

1. Define the trial design
--------------------------

.. code-block:: python

   import sctrial as st

   design = st.TrialDesign(
       participant_col="participant_id",
       visit_col="visit",
       arm_col="arm",
       arm_treated="Treated",
       arm_control="Control",
       celltype_col="cell_type",
   )

2. Validate the AnnData object
------------------------------

.. code-block:: python

   # Raises an error if required columns are missing
   st.validate_adata(adata, design)

   # Validate a set of features (genes or obs columns)
   valid, missing = st.validate_features(adata, ["IFNG", "CD3D", "sig_IFN"], allow_missing=True)
   print("Valid features:", valid)
   print("Missing features:", missing)

3. Run a full diagnostic report
-------------------------------

.. code-block:: python

   report = st.diagnose_trial_data(adata, design, verbose=True)
   if report["warnings"]:
       print("Warnings:")
       for w in report["warnings"]:
           print("-", w)

4. Pairing checks
-----------------

Pairing is critical for longitudinal analyses. Use the diagnostic output to confirm
that enough participants have data at both visits. For detailed pairing inspection:

.. code-block:: python

   pairing = (
       adata.obs
       .groupby(["participant_id", "visit"])
       .size()
       .reset_index(name="n_cells")
   )
   paired = pairing.pivot(index="participant_id", columns="visit", values="n_cells")
   paired = paired.dropna()
   print(f"Paired participants: {len(paired)}")

Next steps
----------

Once validation passes, proceed to DiD or within-arm comparisons. For robust inference
under small samples, enable bootstrap inference in `did_table`.
