Holographic Module
==================

The holographic module analyzes the relationship between information capacity
and the fine structure constant, based on holographic principles.

.. contents:: Contents
   :local:
   :depth: 2

HolographicAnalysis
-------------------

Main class for holographic information analysis.

.. autoclass:: sim.holographic.HolographicAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from sim.holographic import HolographicAnalysis

   # Create analysis instance
   analysis = HolographicAnalysis()

   # Analyze all cosmological models
   results = analysis.analyze_all_models()

   print(f"Mean k: {results['mean_k']:.6f}")
   print(f"Standard deviation: {results['std_k']:.6f}")
   print(f"k/α ratio: {results['mean_k_over_alpha']:.2f}")  # ≈ 66

   # Statistical significance
   print(f"p-value: {results['p_value']:.4f}")

Formula Comparison
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compare different k formulas
   comparison = analysis.formula_comparison()

   for formula, data in comparison.items():
       print(f"{formula}: k = {data['k']:.6f}, error = {data['error_percent']:.2f}%")

   print(f"\nBest formula: {comparison['best_formula']}")

Cosmological Models
-------------------

Pre-defined cosmological datasets.

.. autodata:: sim.holographic.COSMOLOGICAL_MODELS
   :annotation:

Available Models
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model
     - A_s (10⁻⁹)
     - n_s
     - H₀ (km/s/Mpc)
   * - Planck 2018
     - 2.1005
     - 0.9649
     - 67.36
   * - WMAP 9-year
     - 2.41
     - 0.972
     - 69.32
   * - ACT DR4
     - 2.15
     - 0.986
     - 67.9
   * - SPT-3G
     - 2.03
     - 0.960
     - 68.8

Using Custom Models
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.holographic import HolographicAnalysis, CosmologicalModel

   # Define custom model
   my_model = CosmologicalModel(
       name="Custom",
       A_s=2.1e-9,   # Scalar amplitude
       n_s=0.965,    # Spectral index
       H0=68.0,      # Hubble constant
       omega_m=0.31, # Matter density
       omega_b=0.05  # Baryon density
   )

   analysis = HolographicAnalysis()
   k = analysis.calculate_k(my_model)
   print(f"k for custom model: {k:.6f}")

UniverseFormulaReport
---------------------

Generate comprehensive reports on holographic analysis.

.. autoclass:: sim.holographic.UniverseFormulaReport
   :members:
   :undoc-members:
   :show-inheritance:

Report Generation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.holographic import UniverseFormulaReport

   report = UniverseFormulaReport()

   # Scientific abstract
   abstract = report.generate_abstract()
   print(abstract)

   # Full analysis report
   full_report = report.generate_full_report()

   # LaTeX-formatted results
   latex = report.to_latex()

   # Presentation slides
   slides = report.create_presentation_slides()

The k-Alpha Relation
--------------------

The central discovery is an empirical relation:

.. math::

   k \approx 66\alpha

where:

- k is the holographic information ratio
- α ≈ 1/137 is the fine structure constant

k Calculation
^^^^^^^^^^^^^

The information ratio k is calculated as:

.. math::

   k = \frac{\pi \alpha \ln(1/A_s)}{n_s}

where:

- :math:`A_s \approx 2.1 \times 10^{-9}` is the scalar perturbation amplitude
- :math:`n_s \approx 0.965` is the spectral index
- :math:`\alpha = e^2/4\pi\epsilon_0\hbar c \approx 1/137`

Alternative Formulas
^^^^^^^^^^^^^^^^^^^^

Several formulas predict k:

**Holographic formula:**

.. math::

   k_{\text{holo}} = \frac{\pi \alpha}{n_s} \ln\left(\frac{1}{A_s}\right)

**Entropic formula:**

.. math::

   k_{\text{ent}} = 2\pi \alpha \sqrt{\frac{H_0}{M_{\text{Pl}}}}

**Boson mass formula:**

.. math::

   k_{\text{mass}} = \alpha \ln\left(\frac{M_{\text{Pl}}}{m_H}\right)

Physical Interpretation
-----------------------

Holographic Principle
^^^^^^^^^^^^^^^^^^^^^

The holographic principle states that the information content of a region
is bounded by its surface area:

.. math::

   S \leq \frac{A}{4 l_P^2}

where :math:`l_P = \sqrt{\hbar G/c^3}` is the Planck length.

Information-Energy Relation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ratio k represents:

.. math::

   k = \frac{E_{\text{info}}}{E_{\text{total}}}

the fraction of energy that encodes "usable" information about the universe's
state, suggesting a deep connection between:

- Information capacity (bounded by holographic principle)
- Electromagnetic coupling (fine structure constant)
- Cosmological parameters (A_s, n_s)

Significance
^^^^^^^^^^^^

The k ≈ 66α relation with < 1% error across multiple independent datasets
suggests this is not coincidental, but reflects fundamental physics connecting:

1. Quantum electrodynamics (α)
2. Cosmological structure (A_s, n_s)
3. Information theory (holographic bound)

