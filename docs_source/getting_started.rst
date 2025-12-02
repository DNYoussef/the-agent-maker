Getting Started
===============

Installation
-----------

Prerequisites
^^^^^^^^^^^^

* Python 3.10 or higher
* CUDA-capable GPU (GTX 1660 or better, 6GB+ VRAM) - optional but recommended
* 16GB+ system RAM
* 50GB disk space

Basic Installation
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Clone repository
   git clone https://github.com/agent-forge/agent-forge-v2.git
   cd agent-forge-v2

   # Install dependencies
   pip install -r requirements-dev.txt
   pip install -e .

   # Install pre-commit hooks (optional)
   pre-commit install

Quick Start
----------

Launch Dashboard
^^^^^^^^^^^^^^^

.. code-block:: bash

   streamlit run src/ui/app.py

Open browser to http://localhost:8501

Run Tests
^^^^^^^^

.. code-block:: bash

   # All tests with coverage
   pytest tests/ -v --cov=src --cov-report=html

   # Quick test runner
   python scripts/run_tests.py

   # NASA POT10 check
   python .github/hooks/nasa_pot10_check.py src/**/*.py

Basic Usage
----------

Model Registry
^^^^^^^^^^^^^

.. code-block:: python

   from cross_phase.storage.model_registry import ModelRegistry

   # Create registry
   registry = ModelRegistry()

   # Create session
   registry.create_session("session_001", {
       "pipeline": "agent-forge-v2",
       "description": "First training run"
   })

   # Register model
   model_id = registry.register_model(
       model_id="model_001",
       session_id="session_001",
       phase="phase1",
       model_path="/path/to/model.pt",
       params=25_000_000,
       size_mb=95.4,
       metrics={"loss": 2.34}
   )

MuGrokfast Optimizer
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cross_phase.mugrokfast.optimizer import create_optimizer_from_phase

   # Create optimizer for Phase 1
   optimizer = create_optimizer_from_phase(model, phase_num=1)

   # Training loop
   for epoch in range(10):
       for batch in dataloader:
           optimizer.zero_grad()
           loss = model(batch)
           loss.backward()
           optimizer.step()

Prompt Baking
^^^^^^^^^^^^

.. code-block:: python

   from cross_phase.prompt_baking.baker import bake_prompt

   prompt = "You are a step-by-step reasoning assistant..."

   baked_model = bake_prompt(
       model=model,
       prompt=prompt,
       tokenizer=tokenizer,
       calibration_data=dataset
   )

Pipeline Orchestrator
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cross_phase.orchestrator.pipeline import PipelineOrchestrator
   import yaml

   # Load configuration
   with open("config/pipeline_config.yaml") as f:
       config = yaml.safe_load(f)

   # Run full pipeline
   with PipelineOrchestrator(config) as pipeline:
       results = pipeline.run_full_pipeline()

Next Steps
---------

* Read the :doc:`architecture` guide
* Explore the :doc:`api/modules` documentation
* Learn about :doc:`testing` practices
* Review :doc:`deployment` options
