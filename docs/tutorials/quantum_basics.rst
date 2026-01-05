Quantum Basics Tutorial
=======================

This tutorial introduces quantum simulation with the ``sim.quantum`` module.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

Quantum computers operate on qubits—two-level quantum systems that can exist
in superpositions of |0⟩ and |1⟩ states. This tutorial covers:

- Creating multi-qubit systems
- Applying quantum gates
- Measuring entanglement
- Simulating emergent physical phenomena

Creating a Quantum System
-------------------------

The ``QuantumFabric`` class represents a multi-qubit quantum system:

.. code-block:: python

   from sim.quantum import QuantumFabric

   # Create a 3-qubit system
   # Initial state: |000⟩ (all qubits in |0⟩)
   qf = QuantumFabric(num_qubits=3)

   # Check the initial state
   print(f"Number of qubits: {qf.num_qubits}")
   print(f"State dimension: {qf.dim}")  # 2³ = 8
   print(f"Initial state info: {qf.get_state_info()}")

The state vector has 2ⁿ complex amplitudes for n qubits:

.. code-block:: python

   # Access the state vector
   state = qf.state
   print(f"State vector shape: {state.shape}")  # (8,)
   print(f"|000⟩ amplitude: {state[0]}")  # 1.0+0j
   print(f"|001⟩ amplitude: {state[1]}")  # 0j

Applying Quantum Gates
----------------------

Single-Qubit Gates
^^^^^^^^^^^^^^^^^^

Apply gates to individual qubits:

.. code-block:: python

   # Hadamard gate creates superposition
   qf = QuantumFabric(num_qubits=3)
   qf.apply_hadamard(0)  # Apply to first qubit

   # Now in state (|000⟩ + |100⟩)/√2
   print(f"|000⟩ probability: {abs(qf.state[0])**2:.4f}")  # 0.5
   print(f"|100⟩ probability: {abs(qf.state[4])**2:.4f}")  # 0.5

Other single-qubit gates:

.. code-block:: python

   # Pauli gates
   qf.apply_pauli_x(0)  # Bit flip: |0⟩ ↔ |1⟩
   qf.apply_pauli_y(0)  # Bit + phase flip
   qf.apply_pauli_z(0)  # Phase flip: |1⟩ → -|1⟩

   # Phase gate
   qf.apply_phase(0, phi=np.pi/4)  # |1⟩ → e^(iφ)|1⟩

Two-Qubit Gates
^^^^^^^^^^^^^^^

Create entanglement with two-qubit gates:

.. code-block:: python

   qf = QuantumFabric(num_qubits=2)

   # Start with |00⟩, apply Hadamard to first qubit
   qf.apply_hadamard(0)  # (|00⟩ + |10⟩)/√2

   # Apply CNOT with control=0, target=1
   qf.apply_cnot(control=0, target=1)

   # Now in Bell state: (|00⟩ + |11⟩)/√2
   print("Bell state:")
   print(f"  |00⟩: {abs(qf.state[0])**2:.4f}")  # 0.5
   print(f"  |01⟩: {abs(qf.state[1])**2:.4f}")  # 0.0
   print(f"  |10⟩: {abs(qf.state[2])**2:.4f}")  # 0.0
   print(f"  |11⟩: {abs(qf.state[3])**2:.4f}")  # 0.5

Entanglement Operators
^^^^^^^^^^^^^^^^^^^^^^

Apply entanglement across multiple pairs:

.. code-block:: python

   qf = QuantumFabric(num_qubits=4)

   # Entangle pairs: (0,1), (1,2), (2,3)
   qf.apply_entanglement_operator([(0, 1), (1, 2), (2, 3)])

   # All qubits are now entangled
   entropy = qf.get_entanglement_entropy()
   print(f"Entanglement entropy: {entropy:.4f}")

Measuring Entanglement
----------------------

Entanglement Entropy
^^^^^^^^^^^^^^^^^^^^

The von Neumann entropy quantifies entanglement:

.. code-block:: python

   qf = QuantumFabric(num_qubits=4)

   # Product state (no entanglement)
   entropy_product = qf.get_entanglement_entropy()
   print(f"Product state entropy: {entropy_product:.4f}")  # ≈ 0

   # Create entanglement
   qf.apply_hadamard(0)
   qf.apply_cnot(0, 1)
   qf.apply_cnot(1, 2)
   qf.apply_cnot(2, 3)

   entropy_entangled = qf.get_entanglement_entropy()
   print(f"Entangled state entropy: {entropy_entangled:.4f}")  # > 0

Bipartite Entanglement
^^^^^^^^^^^^^^^^^^^^^^

Measure entanglement between specific partitions:

.. code-block:: python

   qf = QuantumFabric(num_qubits=4)
   qf.apply_entanglement_operator([(0, 1), (2, 3)])

   # Entropy of first two qubits vs last two
   entropy_12 = qf.get_entanglement_entropy(partition=[0, 1])
   entropy_34 = qf.get_entanglement_entropy(partition=[2, 3])

   print(f"Entropy (qubits 0,1 vs 2,3): {entropy_12:.4f}")
   print(f"Entropy (qubits 2,3 vs 0,1): {entropy_34:.4f}")

Quantum Measurement
-------------------

Measurement collapses the quantum state:

.. code-block:: python

   qf = QuantumFabric(num_qubits=2)
   qf.apply_hadamard(0)
   qf.apply_cnot(0, 1)  # Bell state

   # Before measurement
   print("Before measurement:")
   print(f"  |00⟩: {abs(qf.state[0])**2:.4f}")
   print(f"  |11⟩: {abs(qf.state[3])**2:.4f}")

   # Measure qubit 0
   result = qf.measure(0)
   print(f"\nMeasured qubit 0: |{result}⟩")

   # After measurement, state collapses
   print("\nAfter measurement:")
   for i in range(4):
       if abs(qf.state[i])**2 > 0.01:
           print(f"  |{i:02b}⟩: {abs(qf.state[i])**2:.4f}")

Measurement Statistics
^^^^^^^^^^^^^^^^^^^^^^

Run multiple measurements to see probability distribution:

.. code-block:: python

   from collections import Counter

   results = []
   for _ in range(1000):
       qf = QuantumFabric(num_qubits=2)
       qf.apply_hadamard(0)
       qf.apply_cnot(0, 1)
       results.append(qf.measure(0))

   counts = Counter(results)
   print(f"|0⟩ occurrences: {counts[0]} ({counts[0]/10:.1f}%)")
   print(f"|1⟩ occurrences: {counts[1]} ({counts[1]/10:.1f}%)")

Emergent Physics
----------------

The ``EmergentLaws`` class simulates physical phenomena:

Particle Creation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.quantum import EmergentLaws

   # Simulate vacuum fluctuations creating particles
   particles = EmergentLaws.simulate_particle_creation(
       vacuum_energy=0.2,
       observation_time=1.0
   )

   print(f"Created {len(particles)} particle-antiparticle pairs")
   for p in particles[:3]:
       print(f"  Type: {p.type}, Energy: {p.energy:.4f}")

Landauer's Principle
^^^^^^^^^^^^^^^^^^^^

The minimum energy to erase information:

.. code-block:: python

   # Erasing 1 million bits at room temperature
   energy = EmergentLaws.landauer_principle(
       bits_erased=1e6,
       temperature=300  # Kelvin
   )

   print(f"Minimum energy to erase 10⁶ bits: {energy:.2e} J")
   print(f"That's about {energy/1.6e-19:.0f} eV")

Exercises
---------

1. **GHZ State**: Create a 3-qubit GHZ state (|000⟩ + |111⟩)/√2 and verify
   its entanglement entropy.

2. **Quantum Teleportation**: Implement the quantum teleportation protocol
   using two Bell pairs.

3. **Decoherence**: Use the ``Observer`` class to simulate how measurement
   affects quantum superposition.

Solutions
^^^^^^^^^

**Exercise 1: GHZ State**

.. code-block:: python

   qf = QuantumFabric(num_qubits=3)
   qf.apply_hadamard(0)
   qf.apply_cnot(0, 1)
   qf.apply_cnot(1, 2)

   # Verify: should have |000⟩ and |111⟩ with equal probability
   print(f"|000⟩: {abs(qf.state[0])**2:.4f}")  # 0.5
   print(f"|111⟩: {abs(qf.state[7])**2:.4f}")  # 0.5
   print(f"Entropy: {qf.get_entanglement_entropy():.4f}")  # log₂(2) = 1

Next Steps
----------

- :doc:`nbody_simulation` - Simulate gravitational dynamics
- :doc:`/api/quantum` - Full API reference

