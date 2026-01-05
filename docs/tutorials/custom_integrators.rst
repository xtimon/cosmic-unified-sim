Custom Integrators
==================

Learn how to create custom numerical integrators for N-body simulations.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

The cosmic module provides several built-in integrators, but you may need
custom integrators for specific problems. This tutorial shows how to create
and use your own.

Integrator Architecture
-----------------------

All integrators inherit from ``IntegratorBase``:

.. code-block:: python

   from abc import ABC, abstractmethod
   from dataclasses import dataclass
   from typing import Callable
   import numpy as np

   @dataclass
   class IntegratorState:
       """State container for integrators."""
       positions: np.ndarray   # Shape: (n_bodies, 3)
       velocities: np.ndarray  # Shape: (n_bodies, 3)
       masses: np.ndarray      # Shape: (n_bodies,)
       time: float

   class IntegratorBase(ABC):
       """Base class for all integrators."""

       name: str = "base"
       order: int = 0

       @abstractmethod
       def step(
           self,
           state: IntegratorState,
           dt: float,
           acceleration: Callable,
       ) -> IntegratorState:
           """Perform single integration step."""
           pass

Creating a Simple Integrator
----------------------------

Euler Method
^^^^^^^^^^^^

The simplest integrator (1st order):

.. code-block:: python

   class EulerIntegrator(IntegratorBase):
       """Simple Euler integrator (1st order)."""

       name = "euler"
       order = 1

       def step(
           self,
           state: IntegratorState,
           dt: float,
           acceleration: Callable,
       ) -> IntegratorState:
           """Euler step: x' = x + v*dt, v' = v + a*dt."""
           a = acceleration(state.positions, state.masses)

           new_positions = state.positions + state.velocities * dt
           new_velocities = state.velocities + a * dt

           return IntegratorState(
               positions=new_positions,
               velocities=new_velocities,
               masses=state.masses,
               time=state.time + dt,
           )

Testing the Integrator
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Test on harmonic oscillator: a = -x
   def harmonic_acceleration(positions, masses):
       return -positions

   state = IntegratorState(
       positions=np.array([[1.0, 0, 0]]),
       velocities=np.array([[0.0, 0, 0]]),
       masses=np.array([1.0]),
       time=0.0,
   )

   integrator = EulerIntegrator()
   dt = 0.01

   # Integrate for one period (2π)
   for _ in range(int(2 * np.pi / dt)):
       state = integrator.step(state, dt, harmonic_acceleration)

   # Check: should return near initial position
   print(f"Final position: {state.positions[0, 0]:.4f}")  # ≈ 1.0

Symplectic Integrators
----------------------

Symplectic integrators preserve the Hamiltonian structure and conserve
energy over long timescales.

Störmer-Verlet
^^^^^^^^^^^^^^

.. code-block:: python

   class StormerVerletIntegrator(IntegratorBase):
       """Störmer-Verlet integrator (2nd order, symplectic)."""

       name = "stormer_verlet"
       order = 2

       def step(
           self,
           state: IntegratorState,
           dt: float,
           acceleration: Callable,
       ) -> IntegratorState:
           # Half-step velocity
           a = acceleration(state.positions, state.masses)
           v_half = state.velocities + 0.5 * a * dt

           # Full-step position
           new_positions = state.positions + v_half * dt

           # Complete velocity with new acceleration
           a_new = acceleration(new_positions, state.masses)
           new_velocities = v_half + 0.5 * a_new * dt

           return IntegratorState(
               positions=new_positions,
               velocities=new_velocities,
               masses=state.masses,
               time=state.time + dt,
           )

Higher-Order Methods
--------------------

Yoshida 6th Order
^^^^^^^^^^^^^^^^^

Higher-order symplectic integrators use composition:

.. code-block:: python

   class Yoshida6Integrator(IntegratorBase):
       """6th-order Yoshida integrator."""

       name = "yoshida6"
       order = 6

       # Yoshida coefficients
       w1 = -1.17767998417887
       w2 = 0.235573213359357
       w3 = 0.784513610477560
       w0 = 1 - 2 * (w1 + w2 + w3)

       def __init__(self):
           # Symmetric coefficient sequence
           self.d = [self.w3, self.w2, self.w1, self.w0,
                     self.w1, self.w2, self.w3]
           self.c = [self.w3/2, (self.w3+self.w2)/2, (self.w2+self.w1)/2,
                     (self.w1+self.w0)/2, (self.w0+self.w1)/2,
                     (self.w1+self.w2)/2, (self.w2+self.w3)/2, self.w3/2]

       def step(self, state, dt, acceleration):
           pos = state.positions.copy()
           vel = state.velocities.copy()

           # Apply substeps
           for i in range(7):
               pos += self.c[i] * dt * vel
               a = acceleration(pos, state.masses)
               vel += self.d[i] * dt * a

           pos += self.c[7] * dt * vel

           return IntegratorState(
               positions=pos,
               velocities=vel,
               masses=state.masses,
               time=state.time + dt,
           )

Registering Your Integrator
---------------------------

Add to the integrator factory:

.. code-block:: python

   # In sim/cosmic/integrators.py

   INTEGRATORS = {
       "rk45": RK45Integrator,
       "verlet": VerletIntegrator,
       "leapfrog": LeapfrogIntegrator,
       "yoshida4": Yoshida4Integrator,
       "forest-ruth": ForestRuthIntegrator,
       "euler": EulerIntegrator,        # Add your integrator
       "yoshida6": Yoshida6Integrator,
   }

   def get_integrator(name: str) -> IntegratorBase:
       if name not in INTEGRATORS:
           raise ValueError(f"Unknown integrator: {name}")
       return INTEGRATORS[name]()

Using in Simulations
--------------------

.. code-block:: python

   from sim.cosmic import NBodySimulator, SystemPresets

   presets = SystemPresets()
   bodies = presets.create_solar_system()

   # Use custom integrator by name
   sim = NBodySimulator(bodies, integrator="yoshida6")

   # Or pass instance directly
   my_integrator = Yoshida6Integrator()
   sim = NBodySimulator(bodies, integrator=my_integrator)

Performance Comparison
----------------------

.. code-block:: python

   import time

   integrators = ["euler", "verlet", "yoshida4", "yoshida6"]
   results = {}

   for name in integrators:
       sim = NBodySimulator(bodies.copy(), integrator=name)

       start = time.time()
       sim.simulate(t_span=(0, 365.25*24*3600), n_points=1000)
       elapsed = time.time() - start

       _, energy_error = sim.get_energy_conservation()

       results[name] = {
           "time": elapsed,
           "energy_error": abs(energy_error),
       }

   print("Integrator     Time (s)    Energy Error")
   print("-" * 45)
   for name, data in results.items():
       print(f"{name:14s} {data['time']:8.2f}    {data['energy_error']:.2e}")

Next Steps
----------

- :doc:`performance_optimization` - Optimize your simulations
- :doc:`/api/cosmic` - Full API reference

