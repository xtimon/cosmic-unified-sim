Visualization Guide
===================

Create publication-quality plots and animations from simulation results.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

The visualization module provides tools for:

- 2D and 3D trajectory plots
- Coherence evolution diagrams
- Quantum state visualizations
- Animated simulations

Basic Setup
-----------

.. code-block:: python

   from sim.visualization import SimPlotter
   import matplotlib.pyplot as plt

   # Create plotter with style
   plotter = SimPlotter(style="dark_background", figsize=(12, 8))

N-Body Trajectories
-------------------

3D Plots
^^^^^^^^

.. code-block:: python

   from sim.cosmic import NBodySimulator, SystemPresets

   # Run simulation
   presets = SystemPresets()
   bodies = presets.create_earth_moon_system()
   sim = NBodySimulator(bodies)
   sim.simulate(t_span=(0, 30*24*3600), n_points=1000)

   # Create 3D plot
   fig = plotter.plot_nbody_3d(
       bodies=sim.bodies,
       title="Earth-Moon System (30 days)",
       center_on_com=True,
       show_velocities=False
   )
   plt.show()

2D Projections
^^^^^^^^^^^^^^

.. code-block:: python

   # XY projection
   fig = plotter.plot_nbody_2d(
       bodies=sim.bodies,
       projection="xy",
       title="Earth-Moon System (XY Plane)"
   )

   # Multiple projections
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   for ax, proj in zip(axes, ["xy", "xz", "yz"]):
       plotter.plot_nbody_2d(bodies=sim.bodies, projection=proj, ax=ax)
   plt.tight_layout()
   plt.show()

Orbital Elements
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Plot orbital distance over time
   import numpy as np

   earth = sim.get_body("Earth")
   moon = sim.get_body("Moon")

   traj_earth = earth.get_trajectory_array()
   traj_moon = moon.get_trajectory_array()

   distances = np.linalg.norm(traj_moon - traj_earth, axis=1)

   fig, ax = plt.subplots()
   ax.plot(sim.times / (24*3600), distances / 1e6)
   ax.set_xlabel("Time (days)")
   ax.set_ylabel("Distance (1000 km)")
   ax.set_title("Earth-Moon Distance")
   plt.show()

Coherence Evolution
-------------------

Standard Plot
^^^^^^^^^^^^^

.. code-block:: python

   from sim.coherence import CoherenceModel
   from sim.constants import UNIVERSE_STAGES

   model = CoherenceModel()
   K, C, Total = model.evolve(N=12, alpha=0.66)

   fig = plotter.plot_coherence_evolution(
       K, C, Total,
       stages=UNIVERSE_STAGES,
       title="Universe Coherence Evolution"
   )
   plt.show()

Phase Space
^^^^^^^^^^^

.. code-block:: python

   # K vs C phase diagram
   fig, ax = plt.subplots()
   ax.plot(K, C, 'b-o', markersize=8)

   # Add stage labels
   for i, stage in enumerate(UNIVERSE_STAGES):
       ax.annotate(f"{i}", (K[i], C[i]), fontsize=8)

   ax.set_xlabel("Coherence K")
   ax.set_ylabel("Complexity C")
   ax.set_title("Coherence-Complexity Phase Space")
   plt.show()

Multiverse Distribution
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.coherence import UniverseSimulator

   simulator = UniverseSimulator()
   universes = simulator.multiverse_simulation(n_universes=1000)

   # Histogram of final coherence
   final_K = [u["final_coherence"] for u in universes]

   fig, ax = plt.subplots()
   ax.hist(final_K, bins=50, edgecolor='black', alpha=0.7)
   ax.axvline(np.mean(final_K), color='red', linestyle='--',
              label=f'Mean: {np.mean(final_K):.4f}')
   ax.set_xlabel("Final Coherence")
   ax.set_ylabel("Count")
   ax.legend()
   plt.show()

Quantum States
--------------

Probability Distribution
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.quantum import QuantumFabric

   qf = QuantumFabric(num_qubits=3)
   qf.apply_hadamard(0)
   qf.apply_cnot(0, 1)
   qf.apply_cnot(1, 2)

   probs = np.abs(qf.state)**2

   fig, ax = plt.subplots()
   states = [f"|{i:03b}⟩" for i in range(8)]
   ax.bar(states, probs)
   ax.set_xlabel("Basis State")
   ax.set_ylabel("Probability")
   ax.set_title("GHZ State Distribution")
   plt.show()

Bloch Sphere
^^^^^^^^^^^^

.. code-block:: python

   # Single qubit on Bloch sphere
   from mpl_toolkits.mplot3d import Axes3D

   def bloch_coords(state):
       """Get Bloch sphere coordinates from 2-state vector."""
       alpha, beta = state[0], state[1]
       x = 2 * np.real(np.conj(alpha) * beta)
       y = 2 * np.imag(np.conj(alpha) * beta)
       z = np.abs(alpha)**2 - np.abs(beta)**2
       return x, y, z

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Draw sphere
   u = np.linspace(0, 2*np.pi, 50)
   v = np.linspace(0, np.pi, 50)
   xs = np.outer(np.cos(u), np.sin(v))
   ys = np.outer(np.sin(u), np.sin(v))
   zs = np.outer(np.ones(50), np.cos(v))
   ax.plot_surface(xs, ys, zs, alpha=0.1)

   # Plot state
   qf = QuantumFabric(num_qubits=1)
   qf.apply_hadamard(0)
   x, y, z = bloch_coords(qf.state)
   ax.scatter([x], [y], [z], s=100, c='red')

   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   plt.show()

Animations
----------

N-Body Animation
^^^^^^^^^^^^^^^^

.. code-block:: python

   from matplotlib.animation import FuncAnimation

   # Prepare data
   bodies = sim.bodies
   n_frames = len(bodies[0].trajectory)

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Initialize plot elements
   lines = [ax.plot([], [], [], '-')[0] for _ in bodies]
   points = [ax.plot([], [], [], 'o', markersize=8)[0] for _ in bodies]

   def init():
       for line, point in zip(lines, points):
           line.set_data([], [])
           line.set_3d_properties([])
           point.set_data([], [])
           point.set_3d_properties([])
       return lines + points

   def animate(frame):
       for i, body in enumerate(bodies):
           traj = body.trajectory[:frame+1]
           if len(traj) > 0:
               xs, ys, zs = zip(*traj)
               lines[i].set_data(xs, ys)
               lines[i].set_3d_properties(zs)
               points[i].set_data([xs[-1]], [ys[-1]])
               points[i].set_3d_properties([zs[-1]])
       return lines + points

   anim = FuncAnimation(
       fig, animate, init_func=init,
       frames=n_frames, interval=50, blit=True
   )

   # Save
   anim.save('orbit.gif', writer='pillow', fps=20)

Saving Figures
--------------

.. code-block:: python

   # High-resolution PNG
   fig.savefig("plot.png", dpi=300, bbox_inches='tight')

   # Vector format (PDF)
   fig.savefig("plot.pdf", bbox_inches='tight')

   # SVG for web
   fig.savefig("plot.svg", bbox_inches='tight')

Custom Styles
-------------

.. code-block:: python

   # Create custom style
   custom_style = {
       'figure.facecolor': '#1a1a2e',
       'axes.facecolor': '#16213e',
       'axes.edgecolor': '#e94560',
       'text.color': '#ffffff',
       'axes.labelcolor': '#ffffff',
       'xtick.color': '#ffffff',
       'ytick.color': '#ffffff',
   }

   with plt.rc_context(custom_style):
       plotter.plot_nbody_3d(bodies)
       plt.show()

Next Steps
----------

- :doc:`/api/core` - SimPlotter API reference
- Matplotlib gallery for more plot types

