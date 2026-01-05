Quick Start Guide
=================

This guide covers basic usage of each module in the unified simulation framework.

Quantum Simulation
------------------

Create and manipulate quantum systems:

.. code-block:: python

   from sim.quantum import QuantumFabric, EmergentLaws, HUMAN_OBSERVER

   # Create 3-qubit system (starts in |000⟩ state)
   qf = QuantumFabric(num_qubits=3)
   print(f"Initial state: {qf.get_state_info()}")

   # Apply Hadamard to create superposition
   qf.apply_hadamard(0)  # First qubit now in (|0⟩ + |1⟩)/√2

   # Create Bell states via entanglement
   qf.apply_entanglement_operator([(0, 1), (1, 2)])
   print(f"Entanglement entropy: {qf.get_entanglement_entropy():.4f}")

   # Measure a qubit
   result = qf.measure(0)
   print(f"Measured qubit 0: |{result}⟩")

   # Emergent physics
   particles = EmergentLaws.simulate_particle_creation(vacuum_energy=0.2)
   energy = EmergentLaws.landauer_principle(bits_erased=1e6, temperature=300)
   print(f"Landauer energy: {energy:.2e} J")

N-Body Simulation
-----------------

Simulate gravitational dynamics:

.. code-block:: python

   from sim.cosmic import NBodySimulator, SystemPresets

   # Create Earth-Moon system from presets
   presets = SystemPresets()
   bodies = presets.create_earth_moon_system()

   # Create simulator
   sim = NBodySimulator(bodies)

   # Run simulation for 30 days
   times, states = sim.simulate(
       t_span=(0, 30 * 24 * 3600),  # 30 days in seconds
       n_points=1000
   )

   # Analyze results
   print(f"Total energy: {sim.get_total_energy():.4e} J")
   print(f"Center of mass: {sim.get_center_of_mass()}")

   # Check energy conservation
   initial_e, change = sim.get_energy_conservation()
   print(f"Energy change: {change*100:.6f}%")

Custom Bodies
^^^^^^^^^^^^^

Create custom celestial bodies:

.. code-block:: python

   from sim.cosmic import Body, NBodySimulator
   import numpy as np

   # Create a binary star system
   star1 = Body(
       name="Star A",
       mass=2e30,  # 1 solar mass
       position=np.array([-1e11, 0, 0]),
       velocity=np.array([0, -15000, 0])
   )

   star2 = Body(
       name="Star B",
       mass=1.5e30,
       position=np.array([1e11, 0, 0]),
       velocity=np.array([0, 20000, 0])
   )

   sim = NBodySimulator([star1, star2])
   times, states = sim.simulate(t_span=(0, 365.25*24*3600), n_points=2000)

Coherence Evolution
-------------------

Model universe coherence:

.. code-block:: python

   from sim.coherence import CoherenceModel
   from sim.constants import UNIVERSE_STAGES

   model = CoherenceModel()

   # Evolve through 12 universe stages
   K, C, Total = model.evolve(N=12, alpha=0.66)

   # Display results
   for i, stage in enumerate(UNIVERSE_STAGES):
       print(f"{stage}: K = {K[i]:.4f}")

   print(f"Growth factor: {K[-1]/K[0]:.2f}x")

   # Information analysis
   info = model.information_content(K)
   print(f"Shannon entropy: {info['entropy']:.4f} bits")

   # Predict future evolution
   K_future, stages = model.predict_future(current_stage=12, total_stages=20)

Matter Genesis
--------------

Simulate early universe particle creation:

.. code-block:: python

   from sim.genesis import (
       ParametricResonance,
       LeptogenesisModel,
       MatterGenesisSimulation
   )

   # Parametric resonance (preheating)
   pr = ParametricResonance(inflaton_mass=1e13, coupling=1e-7)
   rate = pr.particle_production_rate(phi_amplitude=1e16, k=1.0)
   print(f"Production rate: {rate:.2e}")

   # Leptogenesis
   lepto = LeptogenesisModel(M=1e10, Yukawa=1e-6, CP_violation=1e-6)
   result = lepto.solve_leptogenesis()
   print(f"Baryon asymmetry η_B: {result['eta_B']:.2e}")

   # Full matter genesis simulation
   sim = MatterGenesisSimulation()
   history = sim.evolve_universe(total_time=1000, dt=1.0)
   summary = sim.get_summary(history)
   print(f"Final composition: {summary['composition']}")

Holographic Analysis
--------------------

Analyze cosmological data:

.. code-block:: python

   from sim.holographic import HolographicAnalysis
   from sim.constants import CosmologicalConstants

   analysis = HolographicAnalysis()
   results = analysis.analyze_all_models()

   print(f"Mean k: {results['mean_k']:.6f}")
   print(f"k/α ≈ {results['mean_k_over_alpha']:.1f}")  # ≈ 66

   # Compare formulas
   comparison = analysis.formula_comparison()
   print(f"Best formula: {comparison['best_formula']}")

Command Line Interface
----------------------

The ``sim`` command provides access to all modules:

.. code-block:: bash

   # Show information
   sim info
   sim info --gpu

   # Quantum simulation
   sim quantum --qubits 5 --entangle

   # N-body simulation
   sim cosmic --system solar --days 365

   # Coherence evolution
   sim coherence --stages 24 --alpha 0.66

   # Holographic analysis
   sim holographic --report

   # Configuration management
   sim config --generate my_config.yaml

Configuration
-------------

Create and use configuration files:

.. code-block:: python

   from sim.core.config import SimulationConfig

   # Create custom config
   config = SimulationConfig()
   config.quantum.num_qubits = 5
   config.cosmic.integrator = "yoshida4"
   config.save("my_config.yaml")

   # Load config
   config = SimulationConfig.from_file("my_config.yaml")

Visualization
-------------

Plot simulation results:

.. code-block:: python

   from sim.visualization import (
       plot_trajectories_3d,
       plot_coherence_evolution,
       plot_quantum_state,
       animate_simulation
   )

   # 3D trajectory plot
   fig = plot_trajectories_3d(bodies, title="Earth-Moon System")

   # Coherence evolution
   fig = plot_coherence_evolution(K, stages=UNIVERSE_STAGES)

   # Quantum state distribution
   probs = qf.get_probability_distribution()
   fig = plot_quantum_state(probs)

   # Create animation
   anim = animate_simulation(bodies, save_path='orbit.gif')

