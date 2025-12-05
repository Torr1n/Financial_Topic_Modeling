# Map Phase Container
# Processes single firm transcripts into topics
#
# Build: docker build -t ftm-map:latest -f cloud/containers/map/Dockerfile .
# Run (local): docker run -e LOCAL_MODE=true -e FIRM_ID=1001 \
#              -e LOCAL_INPUT=/data/transcripts.csv -e LOCAL_OUTPUT=/data/output \
#              -v $(pwd)/data:/data ftm-map:latest

FROM python:3.10-slim

WORKDIR /app

# System dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY cloud/containers/map/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy cloud source code
COPY cloud/src/ ./cloud/src/
COPY cloud/__init__.py ./cloud/
COPY cloud/config/ ./cloud/config/

# Copy entrypoint
COPY cloud/containers/map/entrypoint.py .

# Default command
CMD ["python", "entrypoint.py"]
