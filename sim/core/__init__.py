"""
Core Module
===========

Base classes, utilities, and infrastructure for simulations.
"""

from .base import SimulationBase, SimulationResult
from .exceptions import (
    SimulationError,
    QuantumError,
    InvalidQubitIndexError,
    InvalidQuantumStateError,
    EntanglementError,
    CosmicError,
    NoBodiesError,
    InvalidBodyError,
    CollisionError,
    IntegrationError,
    CoherenceError,
    InvalidStageError,
    GenesisError,
    PhysicsViolationError,
    HolographicError,
    GPUError,
    GPUNotAvailableError,
    GPUMemoryError,
    ConfigurationError,
    CheckpointError,
    ValidationError,
)
from .gpu import (
    GPUBackend,
    get_backend,
    get_array_module,
    GPU_AVAILABLE,
    list_available_backends,
    get_gpu_info,
)
from .logging import (
    logger,
    setup_logging,
    get_logger,
    debug,
    info,
    warning,
    error,
    critical,
    silence,
    verbose,
)
from .checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    get_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
)
from .config import (
    SimulationConfig,
    QuantumConfig,
    CosmicConfig,
    CoherenceConfig,
    GenesisConfig,
    VisualizationConfig,
    OutputConfig,
    LoggingConfig,
    get_config,
    set_config,
    load_config,
    reset_config,
    generate_example_config,
)
from .progress import (
    ProgressTracker,
    ProgressCallback,
    ProgressInfo,
    progress_iter,
)

__all__ = [
    # Base classes
    "SimulationBase",
    "SimulationResult",
    # Exceptions
    "SimulationError",
    "QuantumError",
    "InvalidQubitIndexError",
    "InvalidQuantumStateError",
    "EntanglementError",
    "CosmicError",
    "NoBodiesError",
    "InvalidBodyError",
    "CollisionError",
    "IntegrationError",
    "CoherenceError",
    "InvalidStageError",
    "GenesisError",
    "PhysicsViolationError",
    "HolographicError",
    "GPUError",
    "GPUNotAvailableError",
    "GPUMemoryError",
    "ConfigurationError",
    "CheckpointError",
    "ValidationError",
    # GPU
    "GPUBackend",
    "get_backend",
    "get_array_module",
    "GPU_AVAILABLE",
    "list_available_backends",
    "get_gpu_info",
    # Logging
    "logger",
    "setup_logging",
    "get_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "silence",
    "verbose",
    # Checkpoint
    "CheckpointManager",
    "CheckpointMetadata",
    "get_checkpoint_manager",
    "save_checkpoint",
    "load_checkpoint",
    "load_latest_checkpoint",
    # Config
    "SimulationConfig",
    "QuantumConfig",
    "CosmicConfig",
    "CoherenceConfig",
    "GenesisConfig",
    "VisualizationConfig",
    "OutputConfig",
    "LoggingConfig",
    "get_config",
    "set_config",
    "load_config",
    "reset_config",
    "generate_example_config",
    # Progress
    "ProgressTracker",
    "ProgressCallback",
    "ProgressInfo",
    "progress_iter",
]
