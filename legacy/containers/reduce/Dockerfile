# Reduce Phase Container
# Aggregates firm topics into cross-firm themes using Dual-BERTopic
#
# Build: docker build -t ftm-reduce:latest -f cloud/containers/reduce/Dockerfile .
# Run (local): docker run -e LOCAL_MODE=true \
#              -e LOCAL_INPUT=/data/firm-topics -e LOCAL_OUTPUT=/data/themes.json \
#              -v $(pwd)/output:/data ftm-reduce:latest

FROM python:3.10-slim

WORKDIR /app

# System dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY cloud/containers/reduce/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy cloud source code
COPY cloud/src/ ./cloud/src/
COPY cloud/__init__.py ./cloud/
COPY cloud/config/ ./cloud/config/

# Copy entrypoint
COPY cloud/containers/reduce/entrypoint.py .

# Default command
CMD ["python", "entrypoint.py"]
