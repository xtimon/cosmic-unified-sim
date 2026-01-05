Installation
============

Requirements
------------

* Python 3.9 or later
* NumPy >= 1.21.0
* SciPy >= 1.7.0
* Matplotlib >= 3.5.0
* Pandas >= 1.3.0

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

   pip install cosmic-unified-sim

Or install from source:

.. code-block:: bash

   git clone https://github.com/xtimon/unified-sim.git
   cd unified-sim
   pip install -e .

Installation Options
--------------------

With development tools:

.. code-block:: bash

   pip install -e ".[dev]"

With documentation tools:

.. code-block:: bash

   pip install -e ".[docs]"

With all extras:

.. code-block:: bash

   pip install -e ".[all]"

GPU Acceleration
----------------

NVIDIA CUDA (via CuPy)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -e ".[gpu-cuda]"

Requires CUDA toolkit installed on your system.

AMD/NVIDIA/Intel (via OpenCL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -e ".[gpu-opencl]"

Vulkan
^^^^^^

.. code-block:: bash

   pip install -e ".[gpu-vulkan]"

Docker
------

Build and run with Docker:

.. code-block:: bash

   # CPU version
   docker build -t unified-sim .
   docker run -it unified-sim sim info

   # GPU version (requires nvidia-docker)
   docker build --target gpu -t unified-sim:gpu .
   docker run --gpus all -it unified-sim:gpu sim info

Docker Compose:

.. code-block:: bash

   # Development environment with Jupyter
   docker-compose up dev

   # Run tests
   docker-compose up test

Verification
------------

Verify your installation:

.. code-block:: python

   import sim
   print(f"Version: {sim.__version__}")

   # Test quantum module
   from sim.quantum import QuantumFabric
   qf = QuantumFabric(num_qubits=2)
   print("Quantum module: OK")

   # Test cosmic module
   from sim.cosmic import NBodySimulator
   print("Cosmic module: OK")

Or from command line:

.. code-block:: bash

   sim info
   sim info --gpu  # Check GPU availability

