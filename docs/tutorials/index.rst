Tutorials
=========

Step-by-step guides for common simulation tasks.

.. toctree::
   :maxdepth: 2
   :caption: Beginner

   quantum_basics
   nbody_simulation
   coherence_evolution

.. toctree::
   :maxdepth: 2
   :caption: Intermediate

   custom_integrators
   multiverse_analysis
   visualization_guide

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   gpu_acceleration
   extending_framework
   performance_optimization

Getting Started
---------------

If you're new to the framework, we recommend following the tutorials in order:

1. **Quantum Basics** - Learn to create and manipulate quantum states
2. **N-Body Simulation** - Simulate gravitational dynamics
3. **Coherence Evolution** - Model universe coherence through cosmic epochs

Each tutorial includes:

- Conceptual background
- Code examples with explanations
- Exercises to test your understanding
- Links to related API documentation

Prerequisites
-------------

Before starting the tutorials, ensure you have:

- Python 3.9 or later
- The ``cosmic-unified-sim`` package installed
- Basic familiarity with NumPy
- (Optional) Matplotlib for visualization

.. code-block:: bash

   pip install cosmic-unified-sim[all]

Jupyter Notebooks
-----------------

Interactive versions of these tutorials are available as Jupyter notebooks
in the ``examples/`` directory:

.. code-block:: bash

   cd examples/
   jupyter notebook

Available notebooks:

- ``01_quantum_basics.ipynb``
- ``02_nbody_simulation.ipynb``
- ``03_coherence_evolution.ipynb``

