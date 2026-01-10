Installation
============

Stable release (recommended)
----------------------------

.. code-block:: bash

   pip install sctrial

Optional dependencies
---------------------

.. code-block:: bash

   # plotting + GSEA
   pip install "sctrial[plots,gsea]"

   # AUCell scoring (pySCENIC)
   pip install "sctrial[aucell]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/TheOmarLab/sctrial.git
   cd sctrial
   pip install .

Development setup
-----------------

.. code-block:: bash

   pip install -e ".[dev,plots,gsea,aucell]"
   pre-commit install

Notes
-----

- **Python**: 3.9+ recommended.
- **Runtimes**: CPU‑only is sufficient for most workflows.
- **GSEA**: requires network access to fetch gene set libraries (or provide local GMT files).
