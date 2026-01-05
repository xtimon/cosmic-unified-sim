Multiverse Analysis
===================

Statistical analysis of universe ensembles and parameter sensitivity.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

The multiverse analysis tools allow you to:

- Simulate many universes with varying parameters
- Perform statistical analysis on outcomes
- Study parameter sensitivity and fine-tuning
- Identify "selection effects" favoring certain configurations

Basic Multiverse Simulation
---------------------------

.. code-block:: python

   from sim.coherence import UniverseSimulator
   import numpy as np

   simulator = UniverseSimulator()

   # Simulate 10,000 universes with random alpha
   universes = simulator.multiverse_simulation(
       n_universes=10000,
       alpha_range=(0.3, 0.9)
   )

   print(f"Simulated {len(universes)} universes")

Statistical Analysis
--------------------

Basic Statistics
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Analyze results
   stats = simulator.statistical_analysis(universes)

   print("Multiverse Statistics:")
   print(f"  Mean final coherence: {stats['mean_final_coherence']:.4f}")
   print(f"  Std final coherence: {stats['std_final_coherence']:.4f}")
   print(f"  Median: {stats['median_final_coherence']:.4f}")
   print(f"  Coherent fraction: {stats['coherent_fraction']*100:.1f}%")

Distribution Analysis
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   from scipy import stats as sp_stats

   final_K = np.array([u["final_coherence"] for u in universes])

   # Fit distributions
   normal_params = sp_stats.norm.fit(final_K)
   beta_params = sp_stats.beta.fit(final_K)

   # Plot
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.hist(final_K, bins=100, density=True, alpha=0.7, label='Observed')

   x = np.linspace(0, 1, 1000)
   ax.plot(x, sp_stats.norm.pdf(x, *normal_params), 'r-', label='Normal fit')
   ax.plot(x, sp_stats.beta.pdf(x, *beta_params), 'g-', label='Beta fit')

   ax.legend()
   ax.set_xlabel("Final Coherence")
   ax.set_ylabel("Density")
   plt.show()

Parameter Sensitivity
---------------------

Alpha Sensitivity
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # How sensitive is the outcome to alpha?
   alphas = np.array([u["alpha"] for u in universes])
   final_K = np.array([u["final_coherence"] for u in universes])

   # Correlation
   correlation = np.corrcoef(alphas, final_K)[0, 1]
   print(f"Correlation(α, K_final): {correlation:.4f}")

   # Scatter plot
   fig, ax = plt.subplots()
   ax.scatter(alphas, final_K, alpha=0.1, s=1)
   ax.set_xlabel("α")
   ax.set_ylabel("Final Coherence")

   # Trend line
   z = np.polyfit(alphas, final_K, 2)
   p = np.poly1d(z)
   alpha_line = np.linspace(alphas.min(), alphas.max(), 100)
   ax.plot(alpha_line, p(alpha_line), 'r-', linewidth=2)
   plt.show()

Critical Points
^^^^^^^^^^^^^^^

.. code-block:: python

   # Find critical alpha (transition point)
   from scipy.optimize import minimize_scalar

   def coherence_at_alpha(alpha):
       model = CoherenceModel()
       K, _, _ = model.evolve(N=12, alpha=alpha)
       return K[-1]

   # Find alpha that maximizes coherence
   result = minimize_scalar(
       lambda a: -coherence_at_alpha(a),
       bounds=(0.3, 0.9),
       method='bounded'
   )

   optimal_alpha = result.x
   print(f"Optimal α: {optimal_alpha:.4f}")
   print(f"Max coherence: {coherence_at_alpha(optimal_alpha):.4f}")

Fine-Tuning Analysis
--------------------

How "fine-tuned" is our universe?

.. code-block:: python

   # Our universe's alpha
   our_alpha = 0.66
   our_K = coherence_at_alpha(our_alpha)

   # What fraction of universes have higher coherence?
   better_fraction = np.mean(final_K > our_K)
   print(f"Fraction with K > {our_K:.4f}: {better_fraction*100:.2f}%")

   # How much could alpha vary while maintaining "habitability"?
   threshold = 0.5  # Minimum coherence for habitability

   habitable = [u for u in universes if u["final_coherence"] > threshold]
   habitable_alphas = [u["alpha"] for u in habitable]

   print(f"\nHabitability analysis (K > {threshold}):")
   print(f"  Habitable fraction: {len(habitable)/len(universes)*100:.1f}%")
   print(f"  α range: [{min(habitable_alphas):.3f}, {max(habitable_alphas):.3f}]")
   print(f"  Width: Δα = {max(habitable_alphas) - min(habitable_alphas):.3f}")

Multi-Parameter Analysis
------------------------

Vary multiple parameters simultaneously:

.. code-block:: python

   from itertools import product

   # Parameter grid
   alpha_values = np.linspace(0.4, 0.8, 20)
   initial_K_values = np.linspace(0.001, 0.1, 20)

   results = np.zeros((len(alpha_values), len(initial_K_values)))

   model = CoherenceModel()
   for i, alpha in enumerate(alpha_values):
       for j, K0 in enumerate(initial_K_values):
           K = np.zeros(13)
           K[0] = K0
           for n in range(12):
               C = np.sqrt(K[n] * (1 - K[n]))
               K[n+1] = K[n] + alpha * C * (1 - K[n])
           results[i, j] = K[-1]

   # Plot heatmap
   fig, ax = plt.subplots(figsize=(10, 8))
   im = ax.imshow(results, origin='lower', aspect='auto',
                  extent=[initial_K_values[0], initial_K_values[-1],
                          alpha_values[0], alpha_values[-1]])
   plt.colorbar(im, label='Final Coherence')
   ax.set_xlabel('Initial K')
   ax.set_ylabel('α')
   ax.set_title('Coherence Sensitivity')
   plt.show()

Anthropic Selection
-------------------

Model observer selection effects:

.. code-block:: python

   def habitability_function(K, C):
       """
       Probability of observers given coherence and complexity.
       Observers require both order (K) and information processing (C).
       """
       # Require minimum coherence for stable structures
       if K < 0.3:
           return 0
       # Require minimum complexity for information processing
       if C < 0.2:
           return 0
       # Peaked around optimal values
       return np.exp(-((K - 0.7)**2 + (C - 0.4)**2) / 0.1)

   # Apply selection to multiverse
   weights = []
   for u in universes:
       K = u["final_coherence"]
       C = np.sqrt(K * (1 - K))
       weights.append(habitability_function(K, C))

   weights = np.array(weights)
   weights /= weights.sum()  # Normalize

   # Selected distribution
   selected_K = np.random.choice(final_K, size=10000, p=weights)

   fig, ax = plt.subplots()
   ax.hist(final_K, bins=50, density=True, alpha=0.5, label='Prior')
   ax.hist(selected_K, bins=50, density=True, alpha=0.5, label='Selected')
   ax.legend()
   ax.set_xlabel('Final Coherence')
   ax.set_title('Anthropic Selection Effect')
   plt.show()

   print(f"Prior mean K: {np.mean(final_K):.4f}")
   print(f"Selected mean K: {np.mean(selected_K):.4f}")

Bootstrap Confidence Intervals
------------------------------

.. code-block:: python

   from scipy.stats import bootstrap

   # Bootstrap CI for mean coherence
   final_K_data = (np.array([u["final_coherence"] for u in universes]),)

   result = bootstrap(
       final_K_data,
       np.mean,
       n_resamples=10000,
       confidence_level=0.95
   )

   print(f"Mean: {np.mean(final_K):.4f}")
   print(f"95% CI: [{result.confidence_interval.low:.4f}, "
         f"{result.confidence_interval.high:.4f}]")

Next Steps
----------

- :doc:`/api/coherence` - Full API reference
- Research papers on multiverse cosmology

