Cosmic Module
=============

The cosmic module provides N-body gravitational simulation with multiple
integration methods and preset celestial systems.

.. contents:: Contents
   :local:
   :depth: 2

NBodySimulator
--------------

Main class for gravitational N-body simulations.

.. autoclass:: sim.cosmic.NBodySimulator
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from sim.cosmic import NBodySimulator, SystemPresets

   # Create Earth-Moon system
   presets = SystemPresets()
   bodies = presets.create_earth_moon_system()

   # Initialize simulator
   sim = NBodySimulator(bodies)

   # Run 30-day simulation
   times, states = sim.simulate(
       t_span=(0, 30 * 24 * 3600),
       n_points=1000,
       rtol=1e-10
   )

   # Analyze results
   print(f"Total energy: {sim.get_total_energy():.4e} J")
   print(f"Center of mass: {sim.get_center_of_mass()}")

   # Energy conservation check
   initial_e, change = sim.get_energy_conservation()
   print(f"Relative energy change: {change:.2e}")

Body
----

Class representing a celestial body.

.. autoclass:: sim.cosmic.Body
   :members:
   :undoc-members:
   :show-inheritance:

Creating Custom Bodies
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.cosmic import Body
   import numpy as np

   # Create the Sun
   sun = Body(
       name="Sun",
       mass=1.989e30,  # kg
       position=np.array([0.0, 0.0, 0.0]),
       velocity=np.array([0.0, 0.0, 0.0]),
       radius=6.96e8  # m
   )

   # Create Earth
   earth = Body(
       name="Earth",
       mass=5.972e24,
       position=np.array([1.496e11, 0.0, 0.0]),  # 1 AU
       velocity=np.array([0.0, 29783.0, 0.0])  # orbital velocity
   )

System Presets
--------------

Pre-configured celestial systems.

.. autoclass:: sim.cosmic.SystemPresets
   :members:
   :undoc-members:
   :show-inheritance:

Available Presets
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.cosmic import SystemPresets

   presets = SystemPresets()

   # Earth-Moon system (2 bodies)
   bodies = presets.create_earth_moon_system()

   # Inner solar system (Sun + 4 planets)
   bodies = presets.create_solar_system(include_outer_planets=False)

   # Full solar system (Sun + 8 planets)
   bodies = presets.create_solar_system(include_outer_planets=True)

   # Binary star system
   bodies = presets.create_binary_star_system(
       m1=2e30, m2=1.5e30,
       separation=2e11,
       eccentricity=0.3
   )

   # Classic three-body problem
   bodies = presets.create_three_body_problem()

Integrators
-----------

Numerical integration methods for N-body dynamics.

.. automodule:: sim.cosmic.integrators
   :members:
   :undoc-members:
   :show-inheritance:

Available Integrators
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Name
     - Order
     - Description
   * - ``rk45``
     - 4-5
     - Adaptive Runge-Kutta (default, good for general use)
   * - ``verlet``
     - 2
     - Velocity Verlet (symplectic, energy-conserving)
   * - ``leapfrog``
     - 2
     - Leapfrog method (symplectic)
   * - ``yoshida4``
     - 4
     - 4th-order Yoshida (symplectic, high accuracy)
   * - ``forest-ruth``
     - 4
     - Forest-Ruth integrator (symplectic)

Using Custom Integrators
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.cosmic import NBodySimulator, SystemPresets
   from sim.cosmic.integrators import get_integrator, Yoshida4Integrator

   # Via string name
   sim = NBodySimulator(bodies, integrator="yoshida4")

   # Via integrator object
   integrator = Yoshida4Integrator()
   sim = NBodySimulator(bodies, integrator=integrator)

   # Custom step simulation
   from sim.cosmic.integrators import IntegratorState

   state = IntegratorState(
       positions=positions,
       velocities=velocities,
       masses=masses,
       time=0.0
   )

   for _ in range(1000):
       state = integrator.step(state, dt=3600, acceleration_func=accel)

Cosmic Calculator
-----------------

Utility class for orbital mechanics calculations.

.. autoclass:: sim.cosmic.CosmicCalculator
   :members:
   :undoc-members:
   :show-inheritance:

Example Calculations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.cosmic import CosmicCalculator

   calc = CosmicCalculator()

   # Orbital velocity
   v = calc.orbital_velocity(
       M=1.989e30,  # Central mass (Sun)
       r=1.496e11   # Distance (1 AU)
   )  # ≈ 29.78 km/s

   # Orbital period
   T = calc.orbital_period(
       M=1.989e30,
       r=1.496e11
   )  # ≈ 365.25 days

   # Escape velocity
   v_esc = calc.escape_velocity(M=5.972e24, r=6.371e6)  # From Earth surface

   # Hill sphere radius
   r_hill = calc.hill_radius(m=5.972e24, M=1.989e30, a=1.496e11)

Physical Background
-------------------

Equations of Motion
^^^^^^^^^^^^^^^^^^^

The N-body problem is governed by Newton's law of gravitation:

.. math::

   \ddot{\mathbf{r}}_i = G \sum_{j \neq i} \frac{m_j (\mathbf{r}_j - \mathbf{r}_i)}{|\mathbf{r}_j - \mathbf{r}_i|^3}

where :math:`G = 6.674 \times 10^{-11}` m³/(kg·s²).

Energy Conservation
^^^^^^^^^^^^^^^^^^^

Total energy is conserved in the N-body problem:

.. math::

   E = T + V = \sum_i \frac{1}{2} m_i v_i^2 - G \sum_{i < j} \frac{m_i m_j}{r_{ij}}

Symplectic integrators preserve this conservation property exactly (up to floating-point precision).

Angular Momentum
^^^^^^^^^^^^^^^^

Total angular momentum is also conserved:

.. math::

   \mathbf{L} = \sum_i m_i \mathbf{r}_i \times \mathbf{v}_i

