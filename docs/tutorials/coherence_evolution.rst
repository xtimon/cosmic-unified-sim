Coherence Evolution Tutorial
============================

This tutorial covers universe coherence modeling with the ``sim.coherence`` module.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

The coherence model describes how information and order evolve through
cosmic epochs. This tutorial covers:

- Understanding the coherence model
- Simulating universe evolution
- Information analysis
- Multiverse scenarios

The Coherence Model
-------------------

Basic Concepts
^^^^^^^^^^^^^^

**Coherence (K)** represents the degree of order or correlation in the universe:

- K = 0: Complete disorder (maximum entropy)
- K = 1: Perfect order (minimum entropy)

**Complexity (C)** represents information-processing capacity:

.. math::

   C = \sqrt{K(1-K)}

Complexity is maximized at K = 0.5 (edge of chaos).

Basic Evolution
^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.coherence import CoherenceModel
   from sim.constants import UNIVERSE_STAGES

   # Create model
   model = CoherenceModel()

   # Evolve through 12 cosmic stages
   K, C, Total = model.evolve(N=12, alpha=0.66)

   print("Stage                   K        C       Total")
   print("-" * 50)
   for i, stage in enumerate(UNIVERSE_STAGES):
       print(f"{stage:20s} {K[i]:8.5f} {C[i]:8.5f} {Total[i]:8.5f}")

Understanding the Evolution
---------------------------

The Alpha Parameter
^^^^^^^^^^^^^^^^^^^

The coherence coupling constant α ≈ 0.66 controls evolution rate:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   model = CoherenceModel()

   # Compare different alpha values
   alphas = [0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
   plt.figure(figsize=(10, 6))

   for alpha in alphas:
       K, C, Total = model.evolve(N=12, alpha=alpha)
       plt.plot(range(13), K, label=f"α = {alpha}")

   plt.xlabel("Cosmic Stage")
   plt.ylabel("Coherence K")
   plt.legend()
   plt.title("Coherence Evolution for Different α")
   plt.show()

Why α ≈ 2/3?
^^^^^^^^^^^^

.. code-block:: python

   # The "natural" alpha
   alpha_natural = 2/3
   print(f"α = 2/3 = {alpha_natural:.6f}")

   # This represents spatial dimensions / spacetime dimensions
   # 3 spatial / (3 spatial + 1 temporal) = 3/4? No...
   # Actually: relates to degrees of freedom ratio

Growth Factor
^^^^^^^^^^^^^

.. code-block:: python

   K, C, Total = model.evolve(N=12, alpha=0.66)

   growth = K[-1] / K[0]
   print(f"Initial coherence: {K[0]:.6f}")
   print(f"Final coherence: {K[-1]:.6f}")
   print(f"Growth factor: {growth:.2f}x")

Information Analysis
--------------------

Entropy and Information
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Analyze information content
   info = model.information_content(K)

   print("Information Analysis:")
   print(f"  Shannon entropy: {info['entropy']:.4f} bits")
   print(f"  Complexity measure: {info['complexity']:.4f}")
   print(f"  Information capacity: {info['capacity']:.4e} bits")

Mutual Information
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Information shared between consecutive stages
   mi = model.mutual_information(K, C)
   print(f"Mutual information: {mi:.4f} bits")

   # Per-stage analysis
   for i in range(len(K) - 1):
       mi_i = model.mutual_information_pair(K[i], K[i+1])
       print(f"  Stage {i} → {i+1}: {mi_i:.4f} bits")

Future Predictions
------------------

Extend the model beyond current observations:

.. code-block:: python

   # Predict 12 more stages into the future
   K_future, future_stages = model.predict_future(
       current_stage=12,
       total_stages=24
   )

   print("\nFuture Predictions:")
   print("Stage    Coherence    Description")
   print("-" * 45)
   for i, k in enumerate(K_future):
       stage_num = 12 + i
       print(f"{stage_num:5d}    {k:9.6f}    Stage {stage_num}")

   # Asymptotic behavior
   K_inf = K_future[-1]
   print(f"\nAsymptotic coherence: K∞ ≈ {K_inf:.6f}")

Multiverse Simulation
---------------------

The ``UniverseSimulator`` class enables statistical analysis:

.. code-block:: python

   from sim.coherence import UniverseSimulator

   simulator = UniverseSimulator()

   # Simulate 1000 universes with varying parameters
   universes = simulator.multiverse_simulation(
       n_universes=1000,
       alpha_range=(0.5, 0.8)
   )

   print(f"Simulated {len(universes)} universes")

Statistical Analysis
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Analyze the multiverse
   stats = simulator.statistical_analysis(universes)

   print("\nMultiverse Statistics:")
   print(f"  Mean final coherence: {stats['mean_final_coherence']:.4f}")
   print(f"  Std final coherence: {stats['std_final_coherence']:.4f}")
   print(f"  Coherent fraction: {stats['coherent_fraction']*100:.1f}%")
   print(f"  Mean growth factor: {stats['mean_growth_factor']:.2f}")

Fine-Tuning Analysis
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # How sensitive is coherence to alpha?
   import numpy as np

   alphas = np.linspace(0.5, 0.8, 100)
   final_K = []

   for alpha in alphas:
       K, _, _ = model.evolve(N=12, alpha=alpha)
       final_K.append(K[-1])

   # Find optimal alpha
   optimal_idx = np.argmax(final_K)
   print(f"Optimal α: {alphas[optimal_idx]:.4f}")
   print(f"Maximum K: {final_K[optimal_idx]:.6f}")

Connection to Physics
---------------------

Cosmological Parameters
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.constants import CosmologicalConstants

   cosmo = CosmologicalConstants()

   # Compare with observed parameters
   print("Cosmological Parameters:")
   print(f"  Hubble constant: {cosmo.H0} km/s/Mpc")
   print(f"  Matter density Ω_m: {cosmo.omega_m}")
   print(f"  Dark energy Ω_Λ: {cosmo.omega_lambda}")
   print(f"  Baryon density Ω_b: {cosmo.omega_b}")

Information Bounds
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Holographic bound on information
   from sim.constants import PhysicalConstants

   pc = PhysicalConstants()

   # Universe horizon
   horizon_radius = 4.4e26  # meters (observable universe)

   # Planck area
   l_p = pc.PLANCK_LENGTH
   A_planck = l_p**2

   # Maximum bits (holographic bound)
   horizon_area = 4 * np.pi * horizon_radius**2
   max_bits = horizon_area / (4 * A_planck)
   print(f"Maximum information: {max_bits:.2e} bits")

Custom Models
-------------

Extend the coherence model:

.. code-block:: python

   class CustomCoherenceModel(CoherenceModel):
       """Coherence model with modified evolution."""

       def evolve(self, N=12, alpha=0.66, beta=0.1):
           """
           Modified evolution with additional parameter.

           Args:
               N: Number of stages
               alpha: Primary coupling
               beta: Secondary coupling (new)
           """
           K = np.zeros(N + 1)
           C = np.zeros(N + 1)
           K[0] = 0.01  # Initial coherence

           for i in range(N):
               C[i] = np.sqrt(K[i] * (1 - K[i]))
               # Modified evolution equation
               dK = alpha * C[i] * (1 - K[i]) - beta * K[i]**2
               K[i + 1] = K[i] + dK
               K[i + 1] = np.clip(K[i + 1], 0, 1)

           C[N] = np.sqrt(K[N] * (1 - K[N]))
           Total = K + C

           return K, C, Total

   # Use custom model
   custom = CustomCoherenceModel()
   K, C, Total = custom.evolve(N=12, alpha=0.66, beta=0.05)

Visualization
-------------

.. code-block:: python

   from sim.visualization import SimPlotter

   plotter = SimPlotter()

   # Evolution plot
   fig = plotter.plot_coherence_evolution(
       K, C, Total,
       stages=UNIVERSE_STAGES,
       title="Universe Coherence Evolution"
   )

   # Phase space
   fig = plotter.plot_coherence_phase_space(K, C)

   # Multiverse histogram
   fig = plotter.plot_multiverse_distribution(universes)

Exercises
---------

1. **Critical Alpha**: Find the critical value of α below which coherence
   decreases over time.

2. **Phase Transitions**: Identify which cosmic stages show the largest
   changes in coherence.

3. **Alternative Evolution**: Implement an evolution equation based on
   logistic growth.

Next Steps
----------

- :doc:`multiverse_analysis` - Advanced multiverse statistics
- :doc:`/api/coherence` - Full API reference

