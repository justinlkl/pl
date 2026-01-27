FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal build deps for packages like lightgbm when needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libomp-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal dependency manifests and upgrade pip
COPY requirements.txt requirements.txt
COPY requirements-ensemble.txt requirements-ensemble.txt

RUN python -m pip install --upgrade pip setuptools wheel

# Install core requirements by default
RUN pip install --no-cache-dir -r requirements.txt

# Optional: install ensemble extras at build time by setting BUILD_ENSEMBLE=1
ARG BUILD_ENSEMBLE=0
RUN if [ "$BUILD_ENSEMBLE" = "1" ] ; then pip install --no-cache-dir -r requirements-ensemble.txt ; fi

# Default is an interactive shell; we'll mount the repo at runtime in docker-compose.
CMD ["bash"]
