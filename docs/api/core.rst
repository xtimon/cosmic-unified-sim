Core Module
===========

The core module provides shared infrastructure: configuration, checkpointing,
logging, progress tracking, I/O utilities, and GPU acceleration.

.. contents:: Contents
   :local:
   :depth: 2

Configuration
-------------

.. automodule:: sim.core.config
   :members:
   :undoc-members:
   :show-inheritance:

SimulationConfig
^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.config import SimulationConfig

   # Create default config
   config = SimulationConfig()

   # Modify settings
   config.quantum.num_qubits = 5
   config.cosmic.integrator = "yoshida4"
   config.cosmic.rtol = 1e-12
   config.logging.level = "DEBUG"

   # Validate
   config.validate()

   # Save to file
   config.save("my_config.yaml")

   # Load from file
   config = SimulationConfig.from_file("my_config.yaml")

Global Config
^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.config import get_config, set_config, load_config

   # Get global config
   config = get_config()

   # Set global config
   set_config(my_config)

   # Load and set from file
   load_config("simulation.yaml")

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Configuration can also be set via environment variables:

.. code-block:: bash

   export SIM_QUANTUM_QUBITS=5
   export SIM_COSMIC_INTEGRATOR=yoshida4
   export SIM_LOG_LEVEL=DEBUG

.. code-block:: python

   from sim.core.config import config_from_env
   config = config_from_env()

Checkpointing
-------------

Save and restore simulation state for long-running computations.

.. automodule:: sim.core.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from sim.core.checkpoint import CheckpointManager

   # Create manager
   manager = CheckpointManager("./checkpoints")

   # Save checkpoint
   state = {
       "positions": positions,
       "velocities": velocities,
       "time": current_time
   }
   path = manager.save(
       state=state,
       name="nbody_simulation",
       simulation_type="cosmic",
       step=1000,
       compress=True
   )

   # Load checkpoint
   data = manager.load(path)
   restored_state = data["state"]
   metadata = data["metadata"]

   # Load latest
   latest = manager.load_latest("nbody_simulation")

   # List checkpoints
   checkpoints = manager.list_checkpoints("nbody_simulation")

   # Cleanup old checkpoints
   manager.cleanup("nbody_simulation", keep_last=5)

Convenience Functions
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.checkpoint import (
       save_checkpoint, load_checkpoint, load_latest_checkpoint
   )

   # Quick save
   path = save_checkpoint(state, "my_sim", "quantum", step=100)

   # Quick load
   data = load_checkpoint(path)
   data = load_latest_checkpoint("my_sim")

Progress Tracking
-----------------

Progress bars and callbacks for long-running simulations.

.. automodule:: sim.core.progress
   :members:
   :undoc-members:
   :show-inheritance:

ProgressTracker
^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.progress import ProgressTracker

   # Basic usage with context manager
   with ProgressTracker(1000, desc="Simulating") as pbar:
       for i in range(1000):
           # do work
           pbar.update()
           pbar.set_postfix(energy=f"{energy:.2e}")

   # With callback
   def my_callback(info):
       print(f"Progress: {info.current}/{info.total}, ETA: {info.eta:.1f}s")

   tracker = ProgressTracker(1000, callback=my_callback)

Progress Iterator
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.progress import progress_iter

   # Wrap any iterable
   for item in progress_iter(data, desc="Processing"):
       process(item)

   # With list comprehension
   results = [f(x) for x in progress_iter(inputs, desc="Computing")]

Logging
-------

Centralized logging with colored output and context management.

.. automodule:: sim.core.logging
   :members:
   :undoc-members:
   :show-inheritance:

Setup and Usage
^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.logging import setup_logging, get_logger

   # Setup logging
   setup_logging(level="DEBUG", log_file="simulation.log")

   # Get module logger
   logger = get_logger("my_module")
   logger.info("Starting simulation")
   logger.debug("Debug info: x = %s", x)
   logger.warning("Energy drift detected")

Context Managers
^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.logging import silence, verbose

   # Silence all logging
   with silence():
       run_noisy_computation()

   # Enable verbose output
   with verbose():
       run_detailed_analysis()

Exceptions
----------

Custom exception hierarchy for clear error handling.

.. automodule:: sim.core.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Exception Hierarchy
^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   SimulationError (base)
   ├── QuantumError
   │   ├── InvalidQuantumStateError
   │   └── InvalidQubitIndexError
   ├── CosmicError
   │   ├── NoBodiesError
   │   └── CollisionError
   ├── GPUError
   │   ├── NoGPUError
   │   └── GPUMemoryError
   ├── ConfigurationError
   └── CheckpointError

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.exceptions import (
       SimulationError, QuantumError, InvalidQubitIndexError
   )

   try:
       qf.apply_hadamard(10)  # Invalid qubit index
   except InvalidQubitIndexError as e:
       print(f"Error: {e}")
   except QuantumError as e:
       print(f"Quantum error: {e}")
   except SimulationError as e:
       print(f"Simulation error: {e}")

GPU Acceleration
----------------

Automatic GPU backend selection and unified array operations.

.. automodule:: sim.core.gpu
   :members:
   :undoc-members:
   :show-inheritance:

Backend Selection
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.gpu import GPUBackend, get_array_module

   # Automatic selection
   backend = GPUBackend()  # Chooses best available

   # Force specific backend
   backend = GPUBackend(preferred="cuda")
   backend = GPUBackend(preferred="opencl")
   backend = GPUBackend(preferred="cpu")

   # Get array module (numpy-like API)
   xp = get_array_module()

   # Use like numpy
   a = xp.array([1, 2, 3])
   b = xp.zeros((100, 100))
   c = xp.dot(a, a)

GPU Information
^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.gpu import gpu_info, is_gpu_available

   # Check availability
   if is_gpu_available():
       info = gpu_info()
       print(f"Backend: {info['backend']}")
       print(f"Device: {info['device_name']}")
       print(f"Memory: {info['memory_gb']:.1f} GB")

I/O Utilities
-------------

Save and load simulations in various formats.

.. automodule:: sim.core.io
   :members:
   :undoc-members:
   :show-inheritance:

SimulationResult
^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.base import SimulationResult

   # Create result
   result = SimulationResult(
       name="my_simulation",
       times=times,
       states={"positions": pos, "velocities": vel},
       parameters={"n_bodies": 10, "integrator": "yoshida4"},
       metadata={"author": "John Doe"}
   )

   # Access data
   print(f"Duration: {result.duration}")
   print(f"Steps: {result.n_steps}")
   pos_final = result.get_state("positions", -1)

   # Serialize
   data = result.to_dict()
   restored = SimulationResult.from_dict(data)

File I/O
^^^^^^^^

.. code-block:: python

   from sim.core.io import SimulationIO

   io = SimulationIO()

   # Save to various formats
   io.save_json(result, "simulation.json")
   io.save_hdf5(result, "simulation.h5")
   io.save_csv(result, "simulation.csv")

   # Load
   result = io.load_json("simulation.json")
   result = io.load_hdf5("simulation.h5")

