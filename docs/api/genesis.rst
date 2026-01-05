Genesis Module
==============

The genesis module simulates matter creation in the early universe,
including parametric resonance, leptogenesis, and quantum particle creation.

.. contents:: Contents
   :local:
   :depth: 2

ParametricResonance
-------------------

Simulates post-inflation reheating via parametric resonance.

.. autoclass:: sim.genesis.ParametricResonance
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from sim.genesis import ParametricResonance

   # Create resonance model
   pr = ParametricResonance(
       inflaton_mass=1e13,  # GeV
       coupling=1e-7,       # Yukawa coupling
       expansion_rate=1e10  # Hubble parameter
   )

   # Calculate instability bands
   bands = pr.instability_bands()
   print(f"First resonance band: k ∈ [{bands[0][0]:.2e}, {bands[0][1]:.2e}]")

   # Particle production rate
   rate = pr.particle_production_rate(
       phi_amplitude=1e16,  # Inflaton amplitude
       k=1.0                # Momentum mode
   )
   print(f"Production rate: {rate:.2e}")

   # Floquet analysis
   floquet = pr.floquet_analysis(q=1.0, A=0.5)
   print(f"Floquet exponent: {floquet['mu']:.4f}")

   # Time evolution
   result = pr.solve_mathieu(t_span=(0, 100), q=1.0, A=0.5)

LeptogenesisModel
-----------------

Models baryogenesis via leptogenesis mechanism.

.. autoclass:: sim.genesis.LeptogenesisModel
   :members:
   :undoc-members:
   :show-inheritance:

Baryon Asymmetry Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.genesis import LeptogenesisModel

   # Standard leptogenesis parameters
   lepto = LeptogenesisModel(
       M=1e10,           # Heavy neutrino mass (GeV)
       Yukawa=1e-6,      # Yukawa coupling
       CP_violation=1e-6 # CP asymmetry parameter
   )

   # Solve Boltzmann equations
   result = lepto.solve_leptogenesis()

   print(f"Final lepton asymmetry: {result['Y_L']:.2e}")
   print(f"Baryon asymmetry η_B: {result['eta_B']:.2e}")
   print(f"Observed η_B ≈ 6e-10: match = {abs(result['eta_B'] - 6e-10) < 1e-9}")

   # Parameter scan
   scan = lepto.parameter_scan(
       M_range=(1e8, 1e14),
       eps_range=(1e-8, 1e-4)
   )

   # Find parameters matching observation
   viable = scan[scan['eta_B'] > 1e-10]

QuantumCreation
---------------

Models quantum particle creation from vacuum fluctuations.

.. autoclass:: sim.genesis.QuantumCreation
   :members:
   :undoc-members:
   :show-inheritance:

Bogoliubov Coefficients
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.genesis import QuantumCreation

   qc = QuantumCreation(field_mass=1e-5)

   # Calculate Bogoliubov coefficients
   alpha, beta = qc.bogoliubov_coefficients(
       k=1.0,          # Momentum
       t_span=(0, 100) # Time range
   )

   # Particle number
   n_k = abs(beta)**2
   print(f"Created particles: n_k = {n_k:.4f}")

   # Spectrum analysis
   spectrum = qc.particle_spectrum(
       k_range=(0.1, 10),
       n_k=100,
       t_span=(0, 100)
   )

   # Total particle density
   n_total = spectrum['n_total']
   print(f"Total particle density: {n_total:.4e}")

MatterGenesisSimulation
-----------------------

Full simulation of matter creation in the early universe.

.. autoclass:: sim.genesis.MatterGenesisSimulation
   :members:
   :undoc-members:
   :show-inheritance:

Complete Simulation
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.genesis import MatterGenesisSimulation

   # Initialize simulation
   sim = MatterGenesisSimulation(
       volume_size=1.0,     # Hubble volume
       initial_temp=1e15    # GeV
   )

   # Evolve through reheating
   history = sim.evolve_universe(
       total_time=1000,  # Hubble times
       dt=1.0
   )

   # Get summary
   summary = sim.get_summary(history)
   print(f"Final temperature: {summary['final_temp']:.2e} GeV")
   print(f"Final composition: {summary['composition']}")
   print(f"Baryon-to-photon ratio: {summary['eta']:.2e}")

   # Phase transitions
   for pt in summary['phase_transitions']:
       print(f"  {pt['name']} at T = {pt['temperature']:.2e} GeV")

Physical Background
-------------------

Parametric Resonance (Preheating)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After inflation, the inflaton field φ oscillates around its minimum,
creating particles via parametric resonance. The evolution is governed
by the Mathieu equation:

.. math::

   \ddot{\chi}_k + (A_k + 2q \cos 2z) \chi_k = 0

where:

- :math:`A_k = k^2/m_\phi^2 + 2q`
- :math:`q = g^2 \Phi^2 / 4m_\phi^2` (resonance parameter)
- :math:`z = m_\phi t` (dimensionless time)

Particle production is exponential in the instability bands:

.. math::

   n_k \sim e^{2\mu_k t}

where μₖ is the Floquet exponent.

Leptogenesis
^^^^^^^^^^^^

The baryon asymmetry is generated through:

1. **Heavy neutrino decay**: :math:`N \rightarrow l + H` with CP violation
2. **Sphaleron processes**: Convert lepton asymmetry to baryon asymmetry

The final baryon-to-photon ratio:

.. math::

   \eta_B = \frac{n_B - n_{\bar{B}}}{n_\gamma} \approx 10^{-2} \epsilon \kappa

where ε is the CP asymmetry and κ is the washout factor.

Quantum Particle Creation
^^^^^^^^^^^^^^^^^^^^^^^^^

In an expanding universe, the vacuum state |0⟩ᵢₙ at early times
differs from the vacuum |0⟩ₒᵤₜ at late times:

.. math::

   |0\rangle_{\text{in}} = \prod_k \frac{1}{|\alpha_k|} \sum_{n=0}^{\infty} 
   \left(\frac{\beta_k^*}{\alpha_k^*}\right)^n |n_k, n_{-k}\rangle_{\text{out}}

The Bogoliubov coefficient β determines particle creation:

.. math::

   \langle n_k \rangle = |\beta_k|^2

