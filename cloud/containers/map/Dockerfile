# Financial Topic Modeling - Map Phase Container
# GPU-enabled container for firm-level topic processing
# Base: CUDA 11.8 (stable on g4dn/g5 instances)
# Python 3.10 REQUIRED (intersection of cuML and sentence-transformers requirements)

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libpq-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# =============================================================================
# CRITICAL INSTALL ORDER - DO NOT REORDER
# Order: base numerical → RAPIDS → PyTorch → transformers → app deps
# =============================================================================

# Step 1: Install base numerical packages (sets numpy/pandas constraints)
RUN pip install \
    "numpy>=1.23,<2.0" \
    "pandas>=2.0,<2.2.2" \
    "scipy>=1.8.0" \
    "scikit-learn==1.2" \
    "pyarrow>=14.0.1,<15.0.0"

# Step 2: Install RAPIDS cuML 24.04 (has pandas 2.x support)
# Must be before PyTorch to avoid CUDA conflicts
RUN pip install \
    --extra-index-url https://pypi.nvidia.com \
    "cuml-cu11==24.04.*"

# Step 3: Install PyTorch with CUDA 11.8 support
# sentence-transformers >= 3.0 requires PyTorch >= 2.2
RUN pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.2

# Step 4: Install transformers ecosystem
RUN pip install \
    "transformers>=4.34.0,<4.43.0" \
    "sentence-transformers>=3.0.0"

# Step 5: Install remaining application dependencies
COPY cloud/containers/map/requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Download SpaCy model
RUN python3 -m spacy download en_core_web_sm

# Verify critical imports work
RUN python3 -c "\
import numpy as np; \
import pandas as pd; \
import cuml; \
import torch; \
import sentence_transformers; \
import bertopic; \
print(f'NumPy: {np.__version__}'); \
print(f'Pandas: {pd.__version__}'); \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print('All imports successful!')"

# Copy application code
COPY cloud/ /app/cloud/
COPY cloud/containers/map/entrypoint.py /app/entrypoint.py

# Set working directory and Python path
WORKDIR /app
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import cuml; print('OK')" || exit 1

# Run entrypoint
ENTRYPOINT ["python3", "/app/entrypoint.py"]
