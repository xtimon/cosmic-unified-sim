Quantum Module
==============

The quantum module provides tools for simulating multi-qubit quantum systems,
including entanglement, measurement, and emergent physical laws.

.. contents:: Contents
   :local:
   :depth: 2

QuantumFabric
-------------

The main class for quantum system simulation.

.. autoclass:: sim.quantum.QuantumFabric
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
^^^^^^^^^^^^^

Creating a quantum system:

.. code-block:: python

   from sim.quantum import QuantumFabric

   # Create 3-qubit system (initializes to |000⟩)
   qf = QuantumFabric(num_qubits=3)

   # Apply Hadamard to first qubit
   qf.apply_hadamard(0)

   # Create entanglement between qubits
   qf.apply_entanglement_operator([(0, 1), (1, 2)])

   # Measure entanglement entropy
   entropy = qf.get_entanglement_entropy()
   print(f"Entanglement entropy: {entropy:.4f}")

   # Perform measurement
   result = qf.measure(0)
   print(f"Measured |{result}⟩")

Quantum Operators
-----------------

Efficient quantum gate operations using sparse matrices and caching.

.. automodule:: sim.quantum.operators
   :members:
   :undoc-members:
   :show-inheritance:

Pauli Gates
^^^^^^^^^^^

.. code-block:: python

   from sim.quantum.operators import pauli_x, pauli_y, pauli_z, hadamard

   X = pauli_x()  # Bit flip
   Y = pauli_y()  # Bit + phase flip
   Z = pauli_z()  # Phase flip
   H = hadamard() # Superposition

Multi-Qubit Gates
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.quantum.operators import build_cnot_sparse, build_swap_sparse

   # CNOT gate on 3-qubit system, control=0, target=2
   cnot = build_cnot_sparse(control=0, target=2, num_qubits=3)

   # SWAP gate
   swap = build_swap_sparse(qubit1=0, qubit2=1, num_qubits=3)

Emergent Laws
-------------

Simulation of emergent physical phenomena from quantum fluctuations.

.. autoclass:: sim.quantum.EmergentLaws
   :members:
   :undoc-members:
   :show-inheritance:

Example: Particle Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.quantum import EmergentLaws

   # Simulate particle-antiparticle creation from vacuum
   particles = EmergentLaws.simulate_particle_creation(
       vacuum_energy=0.2,
       observation_time=1.0
   )
   print(f"Created {len(particles)} particles")

   # Landauer's principle: minimum energy to erase information
   energy = EmergentLaws.landauer_principle(
       bits_erased=1e6,
       temperature=300  # Kelvin
   )
   print(f"Minimum energy: {energy:.2e} J")

Observer
--------

Observer models with different decoherence properties.

.. autoclass:: sim.quantum.Observer
   :members:
   :undoc-members:
   :show-inheritance:

Predefined Observers
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.quantum import HUMAN_OBSERVER, LIGO_OBSERVER

   # Human observer (slow decoherence)
   human = HUMAN_OBSERVER
   print(f"Human decoherence time: {human.decoherence_time}")

   # LIGO detector (fast, sensitive)
   ligo = LIGO_OBSERVER
   print(f"LIGO sensitivity: {ligo.sensitivity}")

Mathematical Background
-----------------------

State Representation
^^^^^^^^^^^^^^^^^^^^

Quantum states are represented as complex vectors in a :math:`2^n`-dimensional
Hilbert space, where :math:`n` is the number of qubits:

.. math::

   |\psi\rangle = \sum_{i=0}^{2^n-1} c_i |i\rangle

where :math:`c_i \in \mathbb{C}` and :math:`\sum_i |c_i|^2 = 1`.

Entanglement Entropy
^^^^^^^^^^^^^^^^^^^^

The von Neumann entanglement entropy is calculated as:

.. math::

   S = -\text{Tr}(\rho_A \log_2 \rho_A)

where :math:`\rho_A` is the reduced density matrix obtained by tracing out
the complement of subsystem A.

Gate Operations
^^^^^^^^^^^^^^^

Common quantum gates used:

- **Hadamard**: :math:`H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}`

- **Pauli-X**: :math:`X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`

- **CNOT**: :math:`\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X`

