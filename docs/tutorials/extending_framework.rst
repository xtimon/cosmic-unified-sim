Extending the Framework
=======================

Add new physics modules, custom simulations, and integrations.

.. contents:: Contents
   :local:
   :depth: 2

Architecture Overview
---------------------

The framework follows a modular architecture:

.. code-block:: text

   sim/
   ├── core/           # Shared infrastructure
   │   ├── base.py     # SimulationResult, SimulationBase
   │   ├── config.py   # Configuration system
   │   └── ...
   ├── quantum/        # Quantum mechanics
   ├── cosmic/         # N-body dynamics
   ├── coherence/      # Coherence evolution
   ├── genesis/        # Matter creation
   └── holographic/    # Information analysis

Creating a New Module
---------------------

Basic Structure
^^^^^^^^^^^^^^^

.. code-block:: python

   # sim/mymodule/__init__.py
   from .simulation import MySimulation
   from .models import MyModel

   __all__ = ["MySimulation", "MyModel"]

Simulation Class
^^^^^^^^^^^^^^^^

Inherit from ``SimulationBase``:

.. code-block:: python

   # sim/mymodule/simulation.py
   from dataclasses import dataclass
   from typing import Dict, Optional, Tuple
   import numpy as np

   from sim.core.base import SimulationBase, SimulationResult
   from sim.core.exceptions import SimulationError
   from sim.core.logging import get_logger

   logger = get_logger("mymodule")

   @dataclass
   class MySimulationConfig:
       """Configuration for MySimulation."""
       param1: float = 1.0
       param2: int = 100

   class MySimulation(SimulationBase):
       """
       My custom simulation.

       Examples:
           >>> sim = MySimulation(param1=2.0)
           >>> result = sim.run(duration=100)
       """

       def __init__(
           self,
           param1: float = 1.0,
           param2: int = 100,
           config: Optional[MySimulationConfig] = None,
       ):
           super().__init__()
           self.config = config or MySimulationConfig(param1, param2)
           self.state = self._initialize_state()
           logger.info(f"MySimulation initialized with param1={param1}")

       def _initialize_state(self) -> np.ndarray:
           """Initialize simulation state."""
           return np.zeros(self.config.param2)

       def run(
           self,
           duration: float,
           dt: float = 0.1,
       ) -> SimulationResult:
           """
           Run the simulation.

           Args:
               duration: Total simulation time
               dt: Time step

           Returns:
               SimulationResult with times and states
           """
           n_steps = int(duration / dt)
           times = np.linspace(0, duration, n_steps)
           states = np.zeros((n_steps, len(self.state)))

           for i, t in enumerate(times):
               self.state = self._step(self.state, dt)
               states[i] = self.state

           return SimulationResult(
               name="my_simulation",
               times=times,
               states={"data": states},
               parameters=self.config.__dict__,
           )

       def _step(self, state: np.ndarray, dt: float) -> np.ndarray:
           """Single simulation step."""
           # Your physics here
           return state + dt * self._derivative(state)

       def _derivative(self, state: np.ndarray) -> np.ndarray:
           """Calculate state derivative."""
           return -self.config.param1 * state

Adding Configuration
--------------------

Integrate with the configuration system:

.. code-block:: python

   # sim/core/config.py (add to existing)

   @dataclass
   class MyModuleConfig:
       """Configuration for my module."""
       param1: float = 1.0
       param2: int = 100
       use_gpu: bool = False

   @dataclass
   class SimulationConfig:
       # ... existing configs ...
       mymodule: MyModuleConfig = field(default_factory=MyModuleConfig)

Custom Exceptions
-----------------

Add module-specific exceptions:

.. code-block:: python

   # sim/core/exceptions.py (add to existing)

   class MyModuleError(SimulationError):
       """Base exception for mymodule."""
       pass

   class InvalidParameterError(MyModuleError):
       """Raised when a parameter is invalid."""
       def __init__(self, param_name: str, value: Any, reason: str):
           super().__init__(
               f"Invalid {param_name}={value}: {reason}",
               {"param_name": param_name, "value": value}
           )

Adding CLI Commands
-------------------

Extend the CLI:

.. code-block:: python

   # sim/cli/main.py (add to existing)

   def cmd_mymodule(args):
       """Run my simulation."""
       from sim.mymodule import MySimulation

       sim = MySimulation(
           param1=args.param1,
           param2=args.param2,
       )

       result = sim.run(duration=args.duration)

       print(f"Simulation complete")
       print(f"Final state mean: {result.states['data'][-1].mean():.4f}")

   # Add to argument parser
   parser_my = subparsers.add_parser(
       "mymodule",
       help="Run my simulation"
   )
   parser_my.add_argument("--param1", type=float, default=1.0)
   parser_my.add_argument("--param2", type=int, default=100)
   parser_my.add_argument("--duration", type=float, default=100)
   parser_my.set_defaults(func=cmd_mymodule)

Writing Tests
-------------

.. code-block:: python

   # tests/test_mymodule.py

   import numpy as np
   import pytest
   from numpy.testing import assert_allclose

   from sim.mymodule import MySimulation

   class TestMySimulation:
       """Tests for MySimulation."""

       def test_initialization(self):
           """Test basic initialization."""
           sim = MySimulation(param1=2.0)
           assert sim.config.param1 == 2.0

       def test_run_returns_result(self):
           """Test that run returns SimulationResult."""
           sim = MySimulation()
           result = sim.run(duration=10)

           assert result.name == "my_simulation"
           assert len(result.times) > 0
           assert "data" in result.states

       def test_conservation_law(self):
           """Test that some quantity is conserved."""
           sim = MySimulation()
           result = sim.run(duration=100)

           # Check conservation
           initial = result.states["data"][0].sum()
           final = result.states["data"][-1].sum()
           assert_allclose(initial, final, rtol=1e-6)

       def test_invalid_parameter(self):
           """Test error handling for invalid parameters."""
           with pytest.raises(ValueError):
               MySimulation(param1=-1.0)

Documentation
-------------

Add API documentation:

.. code-block:: rst

   .. docs/api/mymodule.rst

   My Module
   =========

   .. automodule:: sim.mymodule
      :members:
      :undoc-members:

Update ``docs/index.rst``:

.. code-block:: rst

   .. toctree::
      :maxdepth: 2
      :caption: API Reference

      api/quantum
      api/cosmic
      api/mymodule  # Add here

Export from Package
-------------------

Update ``sim/__init__.py``:

.. code-block:: python

   from .mymodule import MySimulation

   __all__ = [
       # ... existing exports ...
       "MySimulation",
   ]

Plugin Architecture
-------------------

For external extensions, create a plugin system:

.. code-block:: python

   # sim/plugins.py

   import importlib
   from typing import Dict, Type

   _plugins: Dict[str, Type] = {}

   def register_plugin(name: str, cls: Type) -> None:
       """Register a plugin class."""
       _plugins[name] = cls

   def get_plugin(name: str) -> Type:
       """Get a registered plugin."""
       if name not in _plugins:
           raise KeyError(f"Unknown plugin: {name}")
       return _plugins[name]

   def load_plugin(module_name: str) -> None:
       """Load a plugin from a module."""
       module = importlib.import_module(module_name)
       if hasattr(module, "register"):
           module.register()

Usage:

.. code-block:: python

   # my_extension.py

   from sim.plugins import register_plugin
   from sim.core.base import SimulationBase

   class MyExtension(SimulationBase):
       ...

   def register():
       register_plugin("my_extension", MyExtension)

Next Steps
----------

- :doc:`/contributing` - Contribution guidelines
- :doc:`/api/core` - Core module API

