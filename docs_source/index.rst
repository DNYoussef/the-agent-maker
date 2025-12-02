Agent Forge V2 Documentation
============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   architecture
   api/modules
   testing
   deployment

Welcome to Agent Forge V2
-------------------------

Agent Forge V2 is an 8-phase AI agent creation pipeline that builds small, efficient models from scratch.

Key Features
-----------

* **Local-First**: Runs entirely on consumer hardware (GTX 1660+, 6GB+ VRAM)
* **Small Models**: 25M parameter TRM × Titans-MAG architecture
* **Production-Ready**: 100% NASA POT10 compliant
* **Comprehensive Testing**: 47 tests, ≥90% coverage
* **Real-Time Monitoring**: Streamlit dashboard

Quick Start
----------

.. code-block:: bash

   # Install dependencies
   pip install -r requirements-dev.txt
   pip install -e .

   # Launch dashboard
   streamlit run src/ui/app.py

   # Run tests
   pytest tests/ -v --cov=src

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
