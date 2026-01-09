Installation
============

You can install `sctrial` from source:

.. code-block:: bash

   git clone https://github.com/TheOmarLab/sctrial.git
   cd sctrial
   pip install .

Or with optional dependencies:

.. code-block:: bash

   pip install ".[plots,gsea]"

Development
-----------

To set up a development environment:

.. code-block:: bash

   pip install -e ".[dev,plots,gsea]"
   pre-commit install
