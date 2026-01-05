GPU Acceleration
================

Accelerate simulations using CUDA, OpenCL, or Vulkan backends.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The framework supports multiple GPU backends:

- **CUDA** (via CuPy) - NVIDIA GPUs
- **OpenCL** (via PyOpenCL) - Cross-platform
- **Vulkan** (via VulkPy) - Modern graphics API
- **CPU** (NumPy fallback) - Always available

Checking GPU Availability
-------------------------

.. code-block:: python

   from sim.core.gpu import GPUBackend, is_gpu_available, gpu_info

   # Check if any GPU is available
   if is_gpu_available():
       info = gpu_info()
       print(f"GPU Backend: {info['backend']}")
       print(f"Device: {info['device_name']}")
       print(f"Memory: {info['memory_gb']:.1f} GB")
   else:
       print("No GPU available, using CPU")

   # Or via CLI
   # $ sim info --gpu

Selecting a Backend
-------------------

Automatic Selection
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.gpu import GPUBackend

   # Automatically selects best available
   backend = GPUBackend()
   print(f"Using: {backend.name}")

Manual Selection
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Force specific backend
   backend = GPUBackend(preferred="cuda")
   backend = GPUBackend(preferred="opencl")
   backend = GPUBackend(preferred="cpu")

   # If preferred not available, falls back to next best

Using GPU Arrays
----------------

Basic Operations
^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.core.gpu import get_array_module

   xp = get_array_module()  # numpy-like API

   # Create arrays (on GPU if available)
   a = xp.array([1, 2, 3, 4, 5])
   b = xp.zeros((100, 100))
   c = xp.random.randn(1000)

   # Operations work like NumPy
   result = xp.dot(a, a)
   matrix = xp.eye(100) + b
   fft = xp.fft.fft(c)

Data Transfer
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from sim.core.gpu import get_array_module, to_numpy, to_gpu

   xp = get_array_module()

   # NumPy to GPU
   cpu_array = np.random.randn(1000, 1000)
   gpu_array = to_gpu(cpu_array)

   # GPU to NumPy
   result_cpu = to_numpy(gpu_array)

GPU-Accelerated Simulations
---------------------------

Quantum Simulation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.quantum import QuantumFabric
   from sim.core.config import get_config

   # Enable GPU via config
   config = get_config()
   config.quantum.use_gpu = True

   # Create quantum fabric (uses GPU if available)
   qf = QuantumFabric(num_qubits=10)  # 2^10 = 1024 states

   # Large state vectors benefit from GPU
   qf.apply_hadamard(0)
   qf.apply_entanglement_operator([(i, i+1) for i in range(9)])

   entropy = qf.get_entanglement_entropy()
   print(f"Entanglement entropy: {entropy:.4f}")

N-Body Simulation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sim.cosmic import NBodySimulator, SystemPresets

   # For many bodies, GPU acceleration helps
   bodies = generate_galaxy(n_stars=10000)  # Custom function

   sim = NBodySimulator(bodies, use_gpu=True)
   times, states = sim.simulate(t_span=(0, 1e9), n_points=1000)

Memory Management
-----------------

For large simulations, manage GPU memory:

.. code-block:: python

   from sim.core.gpu import GPUBackend

   backend = GPUBackend()

   # Check memory
   free, total = backend.memory_info()
   print(f"GPU Memory: {free/1e9:.1f} / {total/1e9:.1f} GB")

   # Clear cache
   backend.clear_cache()

   # Memory pool (CuPy)
   if backend.name == "cuda":
       import cupy as cp
       mempool = cp.get_default_memory_pool()
       mempool.set_limit(size=4 * 1024**3)  # 4 GB limit

Benchmarking
------------

Compare CPU vs GPU performance:

.. code-block:: python

   import time
   import numpy as np
   from sim.core.gpu import GPUBackend, get_array_module

   sizes = [100, 500, 1000, 2000, 5000]
   results = {"cpu": [], "gpu": []}

   for n in sizes:
       # CPU
       a_cpu = np.random.randn(n, n)
       b_cpu = np.random.randn(n, n)
       start = time.time()
       for _ in range(10):
           c_cpu = np.dot(a_cpu, b_cpu)
       results["cpu"].append((time.time() - start) / 10)

       # GPU
       xp = get_array_module()
       a_gpu = xp.array(a_cpu)
       b_gpu = xp.array(b_cpu)
       xp.dot(a_gpu, b_gpu)  # Warmup
       start = time.time()
       for _ in range(10):
           c_gpu = xp.dot(a_gpu, b_gpu)
       if hasattr(xp, 'cuda'):
           xp.cuda.Stream.null.synchronize()
       results["gpu"].append((time.time() - start) / 10)

   print("Matrix size  CPU (ms)   GPU (ms)   Speedup")
   for i, n in enumerate(sizes):
       speedup = results["cpu"][i] / results["gpu"][i]
       print(f"{n:5d}x{n:<5d} {results['cpu'][i]*1000:8.2f}  "
             f"{results['gpu'][i]*1000:8.2f}   {speedup:.1f}x")

Installation
------------

CUDA (NVIDIA)
^^^^^^^^^^^^^

.. code-block:: bash

   pip install cosmic-unified-sim[gpu-cuda]

   # Requires CUDA toolkit
   # Check: nvcc --version

OpenCL (Cross-platform)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install cosmic-unified-sim[gpu-opencl]

   # Requires OpenCL drivers for your GPU

Vulkan
^^^^^^

.. code-block:: bash

   pip install cosmic-unified-sim[gpu-vulkan]

   # Requires Vulkan SDK

Troubleshooting
---------------

**GPU not detected:**

.. code-block:: python

   # Check CUDA
   import subprocess
   result = subprocess.run(['nvidia-smi'], capture_output=True)
   print(result.stdout.decode())

**Out of memory:**

- Reduce batch size
- Use memory pooling
- Process in chunks

**Slower than expected:**

- Include data transfer time
- Use larger batch sizes
- Avoid CPU-GPU synchronization

Next Steps
----------

- :doc:`performance_optimization` - General optimization tips
- :doc:`/api/core` - GPU module API reference

