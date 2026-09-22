# Frigate Learn pipeline container (data collection + dashboard).
#
# Runs on the Frigate VM next to the `frigate` container. Only the lightweight
# pipeline runs here: collect -> build (HTTP collection into SQLite) + the local
# dashboard. VLM verification, SAM audit, training and
# Hailo deployment all run on the dedicated training machine at deploy time.
#
# Build from the repo root:
#   docker build -t frigate-learn:latest .
#
# Runtime: entrypoint starts a nightly pipeline cron (config automation
# schedule) and the dashboard (uvicorn) as the foreground process.

FROM python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends cron ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
# numpy is imported by frigate_learn.audit at CLI load time but only declared
# in the `ml` extra; the collector/dashboard don't need torch/ultralytics.
RUN pip install --no-cache-dir "numpy" ".[web]"

COPY docker-entrypoint.sh /usr/local/bin/frigate-learn-entrypoint
RUN chmod +x /usr/local/bin/frigate-learn-entrypoint

# The `web` command reloads config from the CWD (config.yaml + .env live
# here), so /config must be the working directory at runtime.
WORKDIR /config

RUN mkdir -p /data /var/log

ENTRYPOINT ["/usr/local/bin/frigate-learn-entrypoint"]