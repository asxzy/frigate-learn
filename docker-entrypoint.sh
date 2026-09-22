#!/bin/sh
# frigate-learn container entrypoint.
#
#   * applies schema migrations (idempotent; the webapp also does it on boot)
#   * installs a nightly pipeline cron (default: config automation.schedule;
#     override with $FRIGATE_LEARN_SCHEDULE) that runs the lightweight stages
#     collect -> verify -> build
#   * runs the dashboard as the foreground process so container health == dashboard
set -eu

CONFIG="${FRIGATE_LEARN_CONFIG:-/config/config.yaml}"
WEB_HOST="${FRIGATE_LEARN_WEB_HOST:-0.0.0.0}"
WEB_PORT="${FRIGATE_LEARN_WEB_PORT:-8090}"
LOG=/var/log/frigate-learn-pipeline.log

/usr/local/bin/frigate-learn --config "$CONFIG" db init

SCHEDULE="${FRIGATE_LEARN_SCHEDULE:-0 3 * * *}"
touch "$LOG"
CRON_LINE="${SCHEDULE} ( . /config/.env 2>/dev/null || true; /usr/local/bin/frigate-learn --config \"${CONFIG}\" run --keep-going >> ${LOG} 2>&1 )"
crontab -l 2>/dev/null | grep -F -- "frigate-learn --config" >/dev/null \
    || printf '%s\n' "$CRON_LINE" | crontab -
cron -f &
CRON_PID=$!

trap 'kill "$CRON_PID" 2>/dev/null || true' TERM INT

exec /usr/local/bin/frigate-learn --config "$CONFIG" web --host "$WEB_HOST" --port "$WEB_PORT"