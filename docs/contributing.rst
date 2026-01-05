Contributing Guide
==================

Thank you for your interest in contributing to the Unified Cosmological
Simulation framework! This guide will help you get started.

.. contents:: Contents
   :local:
   :depth: 2

Getting Started
---------------

Development Setup
^^^^^^^^^^^^^^^^^

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/cosmic-unified-sim.git
      cd cosmic-unified-sim

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[all]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pip install pre-commit
      pre-commit install

Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ -v --cov=sim --cov-report=html

   # Run specific test file
   pytest tests/test_quantum.py -v

   # Run specific test
   pytest tests/test_quantum.py::TestQuantumFabric::test_hadamard -v

Code Style
----------

Formatting
^^^^^^^^^^

We use Black for code formatting and isort for import sorting:

.. code-block:: bash

   # Format code
   black sim tests

   # Sort imports
   isort sim tests

   # Check without modifying
   black --check sim tests
   isort --check-only sim tests

Linting
^^^^^^^

We use flake8 for linting:

.. code-block:: bash

   flake8 sim tests

Configuration is in ``.flake8``:

.. code-block:: ini

   [flake8]
   max-line-length = 100
   extend-ignore = E203, W503

Type Hints
^^^^^^^^^^

We use type hints throughout the codebase:

.. code-block:: python

   from typing import Dict, List, Optional, Tuple
   import numpy as np
   from numpy.typing import NDArray

   def calculate_energy(
       positions: NDArray[np.float64],
       velocities: NDArray[np.float64],
       masses: NDArray[np.float64],
   ) -> float:
       """Calculate total energy."""
       ...

Check types with mypy:

.. code-block:: bash

   mypy sim --ignore-missing-imports

Documentation
^^^^^^^^^^^^^

Use Google-style docstrings:

.. code-block:: python

   def simulate(
       self,
       t_span: Tuple[float, float],
       n_points: int = 100,
       rtol: float = 1e-8,
   ) -> Tuple[np.ndarray, np.ndarray]:
       """
       Run the N-body simulation.

       Args:
           t_span: Start and end times (t0, t1) in seconds.
           n_points: Number of output time points.
           rtol: Relative tolerance for the integrator.

       Returns:
           Tuple of (times, states) arrays.

       Raises:
           ValueError: If no bodies are in the simulation.
           SimulationError: If integration fails.

       Examples:
           >>> sim = NBodySimulator(bodies)
           >>> times, states = sim.simulate((0, 1000), n_points=100)
       """

Making Changes
--------------

Branch Naming
^^^^^^^^^^^^^

Use descriptive branch names:

- ``feature/add-new-integrator``
- ``fix/quantum-measurement-bug``
- ``docs/update-installation``
- ``refactor/gpu-backend``

Commit Messages
^^^^^^^^^^^^^^^

Follow conventional commits:

.. code-block:: text

   feat: add Yoshida 6th-order integrator

   - Implement coefficients from original paper
   - Add tests for energy conservation
   - Update documentation

   Closes #42

Prefixes:

- ``feat:`` New feature
- ``fix:`` Bug fix
- ``docs:`` Documentation only
- ``style:`` Formatting, no code change
- ``refactor:`` Code restructuring
- ``test:`` Adding tests
- ``chore:`` Maintenance tasks

Pull Requests
^^^^^^^^^^^^^

1. Create a new branch from ``develop``
2. Make your changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR against ``develop`` branch

PR Template:

.. code-block:: markdown

   ## Description
   Brief description of changes.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring

   ## Testing
   - [ ] Tests added/updated
   - [ ] All tests pass

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Changelog updated

Adding New Features
-------------------

New Module
^^^^^^^^^^

1. Create the module in ``sim/``:

   .. code-block:: text

      sim/
        newmodule/
          __init__.py
          core.py
          utils.py

2. Export in ``sim/__init__.py``:

   .. code-block:: python

      from .newmodule import NewClass

3. Add tests in ``tests/test_newmodule.py``

4. Add documentation in ``docs/api/newmodule.rst``

5. Update ``docs/index.rst`` toctree

New Integrator
^^^^^^^^^^^^^^

Add to ``sim/cosmic/integrators.py``:

.. code-block:: python

   class MyIntegrator(IntegratorBase):
       """My custom integrator."""

       name = "my_integrator"
       order = 4

       def step(
           self,
           state: IntegratorState,
           dt: float,
           acceleration: Callable,
       ) -> IntegratorState:
           """Single integration step."""
           # Implementation
           ...

Register in ``get_integrator()``:

.. code-block:: python

   INTEGRATORS = {
       ...
       "my_integrator": MyIntegrator,
   }

Testing Guidelines
------------------

Test Structure
^^^^^^^^^^^^^^

.. code-block:: python

   import pytest
   import numpy as np
   from numpy.testing import assert_allclose

   class TestMyFeature:
       """Tests for MyFeature class."""

       def test_basic_functionality(self):
           """Test basic operation."""
           result = my_function(input)
           assert result == expected

       def test_edge_cases(self):
           """Test edge cases."""
           assert my_function(0) == 0
           assert my_function(-1) == expected_negative

       def test_raises_on_invalid_input(self):
           """Test that invalid input raises exception."""
           with pytest.raises(ValueError):
               my_function(invalid_input)

       @pytest.mark.parametrize("input,expected", [
           (1, 1),
           (2, 4),
           (3, 9),
       ])
       def test_parametrized(self, input, expected):
           """Test with multiple inputs."""
           assert my_function(input) == expected

Test Coverage
^^^^^^^^^^^^^

Aim for >80% coverage on new code:

.. code-block:: bash

   pytest --cov=sim --cov-report=term-missing

Release Process
---------------

Version Bumping
^^^^^^^^^^^^^^^

1. Update version in:

   - ``pyproject.toml``
   - ``sim/__init__.py``

2. Update ``CHANGELOG.md``

3. Create release PR

4. After merge, tag release:

   .. code-block:: bash

      git tag v0.2.0
      git push origin v0.2.0

CI/CD automatically publishes to PyPI on tagged releases.

Getting Help
------------

- **Issues**: `GitHub Issues <https://github.com/xtimon/cosmic-unified-sim/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/xtimon/cosmic-unified-sim/discussions>`_
- **Email**: tisanov@yahoo.com

Thank you for contributing! 🚀

