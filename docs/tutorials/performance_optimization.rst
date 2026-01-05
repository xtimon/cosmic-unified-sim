Performance Optimization
========================

Tips and techniques for faster simulations.

.. contents:: Contents
   :local:
   :depth: 2

Profiling
---------

Before optimizing, profile to find bottlenecks:

.. code-block:: python

   import cProfile
   import pstats
   from io import StringIO

   # Profile simulation
   profiler = cProfile.Profile()
   profiler.enable()

   sim.run(duration=1000)

   profiler.disable()

   # Analyze results
   stream = StringIO()
   stats = pstats.Stats(profiler, stream=stream)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   print(stream.getvalue())

Line Profiling
^^^^^^^^^^^^^^

.. code-block:: bash

   pip install line_profiler

.. code-block:: python

   from line_profiler import LineProfiler

   lp = LineProfiler()
   lp.add_function(sim._step)

   lp.enable()
   sim.run(duration=100)
   lp.disable()

   lp.print_stats()

Memory Profiling
^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install memory_profiler

.. code-block:: python

   from memory_profiler import profile

   @profile
   def run_simulation():
       sim = MySimulation()
       return sim.run(duration=1000)

   result = run_simulation()

NumPy Optimization
------------------

Vectorization
^^^^^^^^^^^^^

Replace loops with vectorized operations:

.. code-block:: python

   # Slow: Python loop
   def slow_pairwise_distance(positions):
       n = len(positions)
       distances = np.zeros((n, n))
       for i in range(n):
           for j in range(n):
               distances[i, j] = np.linalg.norm(
                   positions[i] - positions[j]
               )
       return distances

   # Fast: Vectorized
   def fast_pairwise_distance(positions):
       diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
       return np.linalg.norm(diff, axis=2)

   # Benchmark
   positions = np.random.randn(100, 3)
   %timeit slow_pairwise_distance(positions)  # ~50 ms
   %timeit fast_pairwise_distance(positions)  # ~1 ms

Memory Layout
^^^^^^^^^^^^^

Use contiguous arrays:

.. code-block:: python

   # Ensure C-contiguous
   arr = np.ascontiguousarray(arr)

   # Check memory layout
   print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
   print(f"F-contiguous: {arr.flags['F_CONTIGUOUS']}")

   # Choose appropriate layout for operations
   # Row operations: C-contiguous (default)
   # Column operations: F-contiguous
   arr_fortran = np.asfortranarray(arr)

Data Types
^^^^^^^^^^

Use appropriate precision:

.. code-block:: python

   # Double precision (default, 64-bit)
   arr64 = np.array([1, 2, 3], dtype=np.float64)

   # Single precision (32-bit, 2x faster for many operations)
   arr32 = np.array([1, 2, 3], dtype=np.float32)

   # For integer indices
   idx = np.array([0, 1, 2], dtype=np.int32)

Numba JIT Compilation
---------------------

Compile hot functions:

.. code-block:: python

   from numba import jit, prange
   import numpy as np

   # Basic JIT
   @jit(nopython=True)
   def compute_forces(positions, masses, G):
       n = len(masses)
       forces = np.zeros_like(positions)

       for i in range(n):
           for j in range(n):
               if i != j:
                   r = positions[j] - positions[i]
                   dist = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
                   forces[i] += G * masses[j] * r / dist**3

       return forces

   # Parallel JIT
   @jit(nopython=True, parallel=True)
   def compute_forces_parallel(positions, masses, G):
       n = len(masses)
       forces = np.zeros_like(positions)

       for i in prange(n):  # Parallel loop
           for j in range(n):
               if i != j:
                   r = positions[j] - positions[i]
                   dist = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
                   forces[i] += G * masses[j] * r / dist**3

       return forces

Caching
-------

Cache expensive computations:

.. code-block:: python

   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def expensive_computation(n: int) -> float:
       # Cache results for repeated calls
       return sum(i**2 for i in range(n))

   # For numpy arrays (not hashable), use custom caching
   class ArrayCache:
       def __init__(self, maxsize=100):
           self.cache = {}
           self.maxsize = maxsize

       def get(self, key, default=None):
           return self.cache.get(key, default)

       def set(self, key, value):
           if len(self.cache) >= self.maxsize:
               # Remove oldest
               self.cache.pop(next(iter(self.cache)))
           self.cache[key] = value

Sparse Matrices
---------------

For quantum operations:

.. code-block:: python

   from scipy import sparse

   # Dense matrix (slow for large systems)
   n = 1000
   H_dense = np.zeros((n, n))
   # Fill matrix...

   # Sparse matrix (fast for sparse patterns)
   H_sparse = sparse.lil_matrix((n, n))
   # Fill matrix...
   H_sparse = H_sparse.tocsr()  # Convert to efficient format

   # Matrix-vector multiplication
   v = np.random.randn(n)
   result = H_sparse @ v  # Fast

Batch Processing
----------------

Process data in batches:

.. code-block:: python

   def process_in_batches(data, batch_size=1000):
       results = []
       for i in range(0, len(data), batch_size):
           batch = data[i:i + batch_size]
           result = process_batch(batch)
           results.append(result)
       return np.concatenate(results)

Algorithmic Improvements
------------------------

N-Body: Barnes-Hut
^^^^^^^^^^^^^^^^^^

O(N²) → O(N log N):

.. code-block:: python

   # Direct O(N²) - for small N
   def direct_forces(positions, masses):
       n = len(masses)
       forces = np.zeros_like(positions)
       for i in range(n):
           for j in range(i+1, n):
               # Compute pairwise force
               ...
       return forces

   # Barnes-Hut O(N log N) - for large N
   # Uses octree to approximate distant bodies
   class BarnesHutTree:
       def __init__(self, positions, masses, theta=0.5):
           self.theta = theta  # Opening angle
           self.root = self._build_tree(positions, masses)

       def compute_forces(self, positions):
           # Traverse tree, use approximation when
           # size/distance < theta
           ...

Quantum: Sparse Gates
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use cached sparse operators
   from sim.quantum.operators import build_cnot_sparse, OperatorCache

   cache = OperatorCache()

   # First call: builds and caches
   cnot = build_cnot_sparse(0, 1, 10)  # 2^10 x 2^10 matrix

   # Subsequent calls: returns cached
   cnot = build_cnot_sparse(0, 1, 10)  # Instant

Checkpointing for Long Runs
---------------------------

Save progress periodically:

.. code-block:: python

   from sim.core.checkpoint import CheckpointManager
   from sim.core.progress import ProgressTracker

   manager = CheckpointManager("./checkpoints")

   # Resume from checkpoint if available
   latest = manager.load_latest("my_simulation")
   if latest:
       state = latest["state"]
       start_step = latest["metadata"]["step"]
   else:
       state = initial_state()
       start_step = 0

   # Run with checkpoints
   total_steps = 100000
   checkpoint_interval = 1000

   with ProgressTracker(total_steps - start_step) as pbar:
       for step in range(start_step, total_steps):
           state = advance(state)

           if step % checkpoint_interval == 0:
               manager.save(state, "my_simulation", "type", step)
               manager.cleanup("my_simulation", keep_last=5)

           pbar.update()

Parallel Processing
-------------------

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   import multiprocessing as mp

   def run_single_simulation(params):
       sim = MySimulation(**params)
       return sim.run(duration=100)

   # Run multiple simulations in parallel
   param_list = [{"param1": i} for i in range(100)]

   with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
       results = list(executor.map(run_single_simulation, param_list))

Benchmarking
------------

.. code-block:: python

   import time
   from contextlib import contextmanager

   @contextmanager
   def timer(name=""):
       start = time.perf_counter()
       yield
       elapsed = time.perf_counter() - start
       print(f"{name}: {elapsed:.3f}s")

   # Usage
   with timer("Simulation"):
       result = sim.run(duration=1000)

   with timer("Analysis"):
       analysis = analyze(result)

Summary
-------

**Quick wins:**

1. Profile first
2. Vectorize NumPy operations
3. Use appropriate dtypes
4. Enable GPU if available

**For large simulations:**

5. Use sparse matrices
6. Implement caching
7. Consider Numba JIT
8. Use better algorithms (Barnes-Hut, etc.)

**For long runs:**

9. Checkpoint regularly
10. Process in batches
11. Use parallel processing

Next Steps
----------

- :doc:`gpu_acceleration` - GPU-specific optimization
- NumPy documentation on performance

