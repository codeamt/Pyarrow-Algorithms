FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    redis-server \
    libsnappy-dev \
    libzstd-dev \
    && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

# Configure Redis for testing
RUN mkdir -p /var/lib/redis && \
    chown -R redis:redis /var/lib/redis && \
    sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf