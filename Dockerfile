# Multi-stage Dockerfile for unified-sim
# Supports CPU-only and GPU (CUDA) builds

# ==============================================================================
# Base stage with common dependencies
# ==============================================================================
FROM python:3.11-slim AS base

LABEL maintainer="Timur Isanov <tisanov@yahoo.com>"
LABEL description="Unified Cosmological Simulation Framework"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash simuser

WORKDIR /app

# ==============================================================================
# Builder stage - install dependencies
# ==============================================================================
FROM base AS builder

# Install build dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --user -r requirements.txt

# ==============================================================================
# Production stage (CPU only)
# ==============================================================================
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /root/.local /home/simuser/.local

# Copy application code
COPY --chown=simuser:simuser . .

# Install package
RUN pip install -e .

# Switch to non-root user
USER simuser

# Set PATH for user-installed packages
ENV PATH=/home/simuser/.local/bin:$PATH

# Default command
CMD ["sim", "info"]

# ==============================================================================
# Development stage
# ==============================================================================
FROM production AS development

USER root

# Install development dependencies
RUN pip install -e ".[dev]" \
    && pip install jupyter jupyterlab ipywidgets

USER simuser

# Expose Jupyter port
EXPOSE 8888

# Default to Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ==============================================================================
# GPU stage (NVIDIA CUDA)
# ==============================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS gpu

LABEL maintainer="Timur Isanov <tisanov@yahoo.com>"
LABEL description="Unified Cosmological Simulation Framework (GPU)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN useradd --create-home --shell /bin/bash simuser

WORKDIR /app

# Copy and install
COPY . .
RUN pip install --upgrade pip \
    && pip install -e ".[gpu-cuda]"

USER simuser

CMD ["sim", "info"]

# ==============================================================================
# Test stage
# ==============================================================================
FROM development AS test

USER root
RUN pip install pytest-cov
USER simuser

# Run tests on build (coverage disabled during build due to filesystem restrictions)
RUN pytest tests -v

CMD ["pytest", "tests", "-v"]

