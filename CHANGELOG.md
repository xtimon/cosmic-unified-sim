# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-05

### Added

#### Core Infrastructure
- **Custom Exceptions**: Domain-specific exception hierarchy (`SimulationError`, `QuantumError`, `CosmicError`, etc.)
- **Logging Framework**: Centralized logging with colored output, file logging support, and context managers
- **Configuration System**: YAML/JSON configuration files with validation and environment variable support
- **Checkpoint System**: Save/restore simulation state with compression and automatic cleanup
- **Progress Tracking**: tqdm-based progress bars with fallback for non-interactive environments

#### Quantum Module
- **Optimized Operators**: Sparse matrix implementation for quantum gates
- **Operator Caching**: LRU cache for frequently used gate matrices
- **New Gates**: `rotation_x`, `rotation_y`, `rotation_z`, `build_cz_sparse`, `build_swap_sparse`, `build_toffoli_sparse`
- **Utility Functions**: `fidelity`, `trace_distance`, `partial_trace`, `is_unitary`, `is_hermitian`

#### Cosmic Module
- **Symplectic Integrators**: Verlet, Leapfrog, Yoshida4, Yoshida6, Forest-Ruth
- **Adaptive Integration**: Automatic time-step adjustment based on error estimation
- **Integrator Factory**: `get_integrator()` and `list_integrators()` functions

#### DevOps
- **GitHub Actions CI**: Automated testing on Python 3.9-3.12, linting, and package building
- **Pre-commit Hooks**: Black, isort, flake8, mypy, bandit integration
- **Docker Support**: Multi-stage Dockerfile with CPU, GPU, and development targets
- **Docker Compose**: Development environment with Jupyter Lab

#### Documentation
- **Sphinx Setup**: Full documentation structure with RTD theme
- **Jupyter Notebooks**: Interactive tutorials for quantum, cosmic, and coherence modules
- **API Reference**: Auto-generated documentation from docstrings

#### Testing
- **Expanded Test Suite**: Comprehensive tests for quantum, cosmic, and core modules
- **Test Categories**: Markers for slow and GPU tests

### Changed
- **GPU Backend**: Refactored to use strategy pattern, reducing code duplication
- **CLI**: Added progress bars, validation, and configuration file support
- **pyproject.toml**: Added new dependencies (tqdm, pyyaml) and tool configurations

### Fixed
- Energy conservation check in N-body simulator no longer mutates body state
- Improved normalization handling in quantum state operations

## [Unreleased]

### Planned
- Additional integrators
- More preset systems
- Extended documentation

