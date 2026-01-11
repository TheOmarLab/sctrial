Multi-Timepoint Analysis
========================

When trials include 3+ visits, `sctrial` provides tools for event-study and trend
analyses beyond two-point DiD.

1. Event-study DiD (visit-by-visit effects)
-------------------------------------------

.. code-block:: python

   import sctrial as st

   res_event = st.event_study_did(
       adata,
       features=["sig_IFN_Response"],
       design=design,
       baseline_visit="V1",
       visits=["V1", "V2", "V3"],
   )
   print(res_event.head())

2. Trend interaction tests
--------------------------

.. code-block:: python

   res_trend = st.trend_interaction(
       adata,
       features=["sig_IFN_Response"],
       design=design,
       visits=["V1", "V2", "V3", "V4"],
   )
   print(res_trend.head())

3. Parallel trends diagnostics (pre-treatment)
----------------------------------------------

.. code-block:: python

   res_parallel = st.test_parallel_trends(
       adata,
       features=["sig_IFN_Response"],
       design=design,
       pre_visits=["V1", "V2"],
   )
   print(res_parallel.head())

Interpretation
--------------

- Event-study results show how treatment effects evolve over time relative to baseline.
- Trend interaction tests whether arms diverge in slopes across multiple visits.
- Parallel trends tests should be run on pre-treatment visits only.
