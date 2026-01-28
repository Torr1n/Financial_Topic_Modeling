# Financial Topic Modeling - Map Phase Container
# GPU-enabled container for firm-level topic processing
# Base: CUDA 11.8 (stable on g4dn/g5 instances)

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

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
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
# Note: sentence-transformers >= 3.0 requires PyTorch >= 2.2
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Install cuML for GPU-accelerated UMAP/HDBSCAN
# Note: cuml-cu11 is the stable version for CUDA 11.x
RUN pip install --no-cache-dir \
    cuml-cu11==24.02.* \
    --extra-index-url https://pypi.nvidia.com

# Copy and install Python dependencies
COPY cloud/containers/map/requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download SpaCy model
RUN python3 -m spacy download en_core_web_sm

# Copy application code
# Note: We copy the entire cloud/ directory to preserve package structure
COPY cloud/ /app/cloud/
COPY cloud/containers/map/entrypoint.py /app/entrypoint.py

# Set working directory and Python path
WORKDIR /app
ENV PYTHONPATH=/app

# Health check (basic Python import test)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import cuml; print('OK')" || exit 1

# Run entrypoint
ENTRYPOINT ["python3", "/app/entrypoint.py"]
