#!/usr/bin/env bash
# Bring up Postgres + TimescaleDB for development.
# Ensures the docker daemon is running (this environment starts it on demand),
# then starts the `db` service and waits for it to be healthy.
set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  echo "[db-up] docker daemon not running; starting it..."
  sudo -n dockerd >/tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 30); do
    docker info >/dev/null 2>&1 && break
    sleep 1
  done
fi

echo "[db-up] starting db service..."
docker compose up -d db

echo "[db-up] waiting for healthy..."
for _ in $(seq 1 40); do
  status=$(docker inspect -f '{{.State.Health.Status}}' swingx-db 2>/dev/null || echo "starting")
  if [ "$status" = "healthy" ]; then
    echo "[db-up] db is healthy on localhost:5433"
    exit 0
  fi
  sleep 2
done

echo "[db-up] ERROR: db did not become healthy in time" >&2
docker compose logs db | tail -40 >&2
exit 1
