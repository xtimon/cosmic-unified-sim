N-Body Simulation Tutorial
==========================

This tutorial covers gravitational N-body simulation with the ``sim.cosmic`` module.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

N-body simulation models the gravitational interaction between multiple bodies.
This tutorial covers:

- Creating celestial bodies
- Using preset systems
- Running simulations
- Analyzing results
- Choosing integrators

Your First Simulation
---------------------

Let's simulate the Earth-Moon system:

.. code-block:: python

   from sim.cosmic import NBodySimulator, SystemPresets

   # Create Earth-Moon system from presets
   presets = SystemPresets()
   bodies = presets.create_earth_moon_system()

   print(f"Created {len(bodies)} bodies:")
   for body in bodies:
       print(f"  {body.name}: mass = {body.mass:.2e} kg")

   # Create simulator
   sim = NBodySimulator(bodies)

   # Run for 30 days
   times, states = sim.simulate(
       t_span=(0, 30 * 24 * 3600),  # seconds
       n_points=1000
   )

   print(f"\nSimulation complete:")
   print(f"  Time steps: {len(times)}")
   print(f"  Duration: {times[-1] / (24*3600):.1f} days")

Analyzing Results
-----------------

Energy Conservation
^^^^^^^^^^^^^^^^^^^

A good simulation conserves total energy:

.. code-block:: python

   # Check energy conservation
   initial_energy, relative_change = sim.get_energy_conservation()

   print(f"Initial energy: {initial_energy:.4e} J")
   print(f"Relative change: {relative_change:.2e}")
   print(f"Energy conserved to {abs(relative_change)*100:.4f}%")

For the default RK45 integrator, expect ~0.001% drift over 30 days.

Orbital Analysis
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get bodies after simulation
   earth = sim.get_body("Earth")
   moon = sim.get_body("Moon")

   # Calculate orbital distance over time
   import numpy as np

   trajectory_earth = earth.get_trajectory_array()
   trajectory_moon = moon.get_trajectory_array()

   distances = np.linalg.norm(trajectory_moon - trajectory_earth, axis=1)

   print(f"Orbital distance:")
   print(f"  Min: {distances.min()/1e6:.0f} km")
   print(f"  Max: {distances.max()/1e6:.0f} km")
   print(f"  Mean: {distances.mean()/1e6:.0f} km")

Center of Mass
^^^^^^^^^^^^^^

.. code-block:: python

   # Center of mass (should remain stationary)
   com = sim.get_center_of_mass()
   print(f"Center of mass: {com}")

   # Total momentum (should be conserved)
   momentum = sim.get_total_momentum()
   print(f"Total momentum: {momentum}")

Creating Custom Bodies
----------------------

Define your own celestial systems:

.. code-block:: python

   from sim.cosmic import Body, NBodySimulator
   import numpy as np

   # Create a binary star system
   star1 = Body(
       name="Star A",
       mass=2e30,  # 1 solar mass
       position=np.array([-1e11, 0, 0]),  # m
       velocity=np.array([0, -15000, 0]),  # m/s
       radius=7e8  # optional
   )

   star2 = Body(
       name="Star B",
       mass=1.5e30,  # 0.75 solar mass
       position=np.array([1.33e11, 0, 0]),
       velocity=np.array([0, 20000, 0])
   )

   # Optional: add a planet
   planet = Body(
       name="Planet",
       mass=6e24,
       position=np.array([5e11, 0, 0]),
       velocity=np.array([0, 25000, 0])
   )

   # Create simulator
   sim = NBodySimulator([star1, star2, planet])

   # Simulate for 1 year
   times, states = sim.simulate(
       t_span=(0, 365.25 * 24 * 3600),
       n_points=2000
   )

Preset Systems
--------------

The ``SystemPresets`` class provides common configurations:

.. code-block:: python

   from sim.cosmic import SystemPresets

   presets = SystemPresets()

   # Earth-Moon (2 bodies)
   bodies = presets.create_earth_moon_system()

   # Inner Solar System (Sun + Mercury, Venus, Earth, Mars)
   bodies = presets.create_solar_system(include_outer_planets=False)

   # Full Solar System (Sun + all 8 planets)
   bodies = presets.create_solar_system(include_outer_planets=True)

   # Binary star with custom parameters
   bodies = presets.create_binary_star_system(
       m1=2e30,         # Mass of star 1
       m2=1.5e30,       # Mass of star 2
       separation=2e11,  # Initial separation
       eccentricity=0.3  # Orbital eccentricity
   )

   # Classic three-body problem (figure-8 orbit)
   bodies = presets.create_three_body_problem()

Choosing Integrators
--------------------

Different integrators suit different problems:

RK45 (Default)
^^^^^^^^^^^^^^

Adaptive step size, good general-purpose choice:

.. code-block:: python

   sim = NBodySimulator(bodies)  # Uses RK45 by default
   times, states = sim.simulate(
       t_span=(0, 1e7),
       n_points=1000,
       rtol=1e-10,  # Relative tolerance
       atol=1e-12   # Absolute tolerance
   )

Symplectic Integrators
^^^^^^^^^^^^^^^^^^^^^^

Preserve energy exactly—better for long simulations:

.. code-block:: python

   # Verlet (2nd order, fast)
   sim = NBodySimulator(bodies, integrator="verlet")

   # Leapfrog (2nd order)
   sim = NBodySimulator(bodies, integrator="leapfrog")

   # Yoshida (4th order, very accurate)
   sim = NBodySimulator(bodies, integrator="yoshida4")

   # Forest-Ruth (4th order)
   sim = NBodySimulator(bodies, integrator="forest-ruth")

Comparison
^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   integrators = ["rk45", "verlet", "yoshida4"]
   energy_errors = {}

   for name in integrators:
       sim = NBodySimulator(bodies.copy(), integrator=name)
       sim.simulate(t_span=(0, 365.25*24*3600), n_points=1000)
       _, error = sim.get_energy_conservation()
       energy_errors[name] = abs(error)
       print(f"{name}: {error:.2e}")

Long-Duration Simulations
-------------------------

For simulations spanning years or more:

.. code-block:: python

   from sim.core.checkpoint import CheckpointManager
   from sim.core.progress import ProgressTracker

   # Setup
   manager = CheckpointManager("./checkpoints")
   sim = NBodySimulator(bodies, integrator="yoshida4")

   # Simulate 100 years with checkpoints
   total_time = 100 * 365.25 * 24 * 3600
   chunk_time = 365.25 * 24 * 3600  # 1 year chunks

   with ProgressTracker(100, desc="Simulating") as pbar:
       t = 0
       while t < total_time:
           sim.simulate(t_span=(t, t + chunk_time), n_points=100)

           # Save checkpoint
           manager.save(
               state=sim.get_state(),
               name="solar_system",
               simulation_type="cosmic",
               step=int(t / chunk_time)
           )

           t += chunk_time
           pbar.update()

   # Keep only last 5 checkpoints
   manager.cleanup("solar_system", keep_last=5)

Visualization
-------------

Plot orbits in 3D:

.. code-block:: python

   from sim.visualization import SimPlotter

   plotter = SimPlotter()

   # 3D trajectory plot
   fig = plotter.plot_nbody_3d(
       bodies=sim.bodies,
       title="Earth-Moon System",
       center_on_com=True
   )

   # 2D projection
   fig = plotter.plot_nbody_2d(
       bodies=sim.bodies,
       projection="xy"
   )

   # Animation
   anim = plotter.animate_nbody(
       bodies=sim.bodies,
       interval=50,  # ms between frames
       save_path="orbit.gif"
   )

Using CosmicCalculator
----------------------

Utility calculations:

.. code-block:: python

   from sim.cosmic import CosmicCalculator

   calc = CosmicCalculator()

   # Orbital velocity at 1 AU from Sun
   v = calc.orbital_velocity(M=1.989e30, r=1.496e11)
   print(f"Earth's orbital velocity: {v/1000:.2f} km/s")

   # Orbital period
   T = calc.orbital_period(M=1.989e30, r=1.496e11)
   print(f"Earth's orbital period: {T/(24*3600):.2f} days")

   # Escape velocity from Earth's surface
   v_esc = calc.escape_velocity(M=5.972e24, r=6.371e6)
   print(f"Earth escape velocity: {v_esc/1000:.2f} km/s")

   # Hill sphere (Moon's stability region)
   r_hill = calc.hill_radius(m=7.342e22, M=5.972e24, a=3.844e8)
   print(f"Moon's Hill sphere: {r_hill/1000:.0f} km")

Exercises
---------

1. **Solar System**: Simulate the full solar system for 1 year. Compare
   Earth's position with known orbital parameters.

2. **Three-Body Problem**: Run the figure-8 three-body solution. Verify
   that all bodies follow the same path.

3. **Asteroid Impact**: Add a small asteroid on collision course with
   Earth. Calculate time to impact.

Next Steps
----------

- :doc:`coherence_evolution` - Model universe coherence
- :doc:`custom_integrators` - Create your own integrators
- :doc:`/api/cosmic` - Full API reference

