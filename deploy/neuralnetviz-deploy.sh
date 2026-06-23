#!/usr/bin/env bash
# Pull the latest image from GHCR and (re)start the container stack on the VPS.
# Installed at /usr/local/bin/neuralnetviz-deploy.sh; run by SSH or a GitHub
# Actions workflow_dispatch step. Idempotent.
#
#   neuralnetviz-deploy.sh [IMAGE_TAG]
#
# Reads the compose files + .env from $DEPLOY_DIR. Never echoes secrets.
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/var/lib/neuralnetviz}"
TAG="${1:-${IMAGE_TAG:-latest}}"

cd "$DEPLOY_DIR"

# IMAGE_TAG flows into docker-compose.vps.yml.
export IMAGE_TAG="$TAG"

compose() {
  docker compose -f docker-compose.yml -f docker-compose.vps.yml "$@"
}

echo "==> Pulling images (tag: $TAG)"
compose pull

echo "==> Starting stack"
compose up -d

echo "==> Waiting for health"
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8801/api/health >/dev/null 2>&1; then
    echo "    healthy"
    break
  fi
  sleep 3
  [ "$i" = 30 ] && { echo "    FAILED health check"; compose logs --tail=50 app; exit 1; }
done

echo "==> Pruning dangling images"
docker image prune -f >/dev/null 2>&1 || true

echo "==> Done. Running containers:"
compose ps
