Unified Cosmological Simulation
================================

.. image:: https://badge.fury.io/py/cosmic-unified-sim.svg
   :target: https://pypi.org/project/cosmic-unified-sim/

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

A comprehensive Python library for cosmological simulations combining quantum mechanics,
N-body dynamics, coherence evolution, matter genesis, and holographic analysis.

Quick Start
-----------

.. code-block:: python

   from sim import QuantumFabric, NBodySimulator, CoherenceModel

   # Quantum simulation
   qf = QuantumFabric(num_qubits=3)
   qf.apply_entanglement_operator([(0, 1), (1, 2)])
   print(f"Entanglement entropy: {qf.get_entanglement_entropy():.4f}")

   # N-body simulation
   from sim.cosmic import SystemPresets
   presets = SystemPresets()
   bodies = presets.create_earth_moon_system()
   sim = NBodySimulator(bodies)
   times, states = sim.simulate(t_span=(0, 30*24*3600), n_points=1000)

Installation
------------

.. code-block:: bash

   pip install cosmic-unified-sim

   # With GPU support (NVIDIA)
   pip install cosmic-unified-sim[gpu-cuda]

Features
--------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - **quantum**
     - Multi-qubit systems, entanglement, emergent laws, observer decoherence
   * - **cosmic**
     - N-body gravitational simulations, orbital mechanics, presets
   * - **coherence**
     - Universe coherence evolution, information theory, predictions
   * - **genesis**
     - Parametric resonance, leptogenesis, quantum particle creation
   * - **holographic**
     - k-alpha analysis, information capacity, cosmological models

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/quantum_basics
   tutorials/nbody_simulation
   tutorials/coherence_evolution
   tutorials/custom_integrators
   tutorials/visualization_guide
   tutorials/multiverse_analysis
   tutorials/gpu_acceleration
   tutorials/extending_framework
   tutorials/performance_optimization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/quantum
   api/cosmic
   api/coherence
   api/genesis
   api/holographic
   api/core

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Links
-----

* `PyPI Package <https://pypi.org/project/cosmic-unified-sim/>`_
* `GitHub Repository <https://github.com/xtimon/unified-sim>`_
* `Issue Tracker <https://github.com/xtimon/unified-sim/issues>`_

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
