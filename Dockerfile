# NeuraNetViz — containerized FastAPI + TensorFlow app, instrumented for Splunk
# Observability via the OpenTelemetry sidecar (see docker-compose.yml).
#
# Multi-stage: stage 1 builds a venv with the pinned deps (incl. the Splunk OTel
# distro, kept on the protobuf<5 line that TF 2.15 requires); stage 2 is a slim
# runtime that copies only the venv + app code + models + static assets.
#
# Python 3.11 is the newest interpreter TF 2.15 ships manylinux wheels for.

# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve and install dependencies into an isolated venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for any source wheels (h5py etc. ship binary wheels, but keep gcc
# available so a pin bump doesn't break the build silently).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the OTel auto-instrumentation bootstrap so the runtime image doesn't have
# to resolve instrumentation packages at start. opentelemetry-bootstrap detects
# installed libs (fastapi) and installs matching instrumentors.
RUN opentelemetry-bootstrap -a install

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image with just the venv and the app
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# curl for the container HEALTHCHECK; libgomp1 is TF's OpenMP runtime dep.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Keras/TF: don't try to write to a read-only home; keep logs quiet.
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOME=/home/appuser

WORKDIR /app

# Copy only what the app serves at runtime (see .dockerignore for exclusions).
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser static/ ./static/

USER appuser

EXPOSE 8801

# Liveness: the app exposes GET /api/health (app/main.py).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8801/api/health || exit 1

# Run uvicorn under the OTel auto-instrumentation agent. Traces + metrics are
# exported via OTLP/HTTP to the otel-collector sidecar (OTEL_* env from compose).
# Mirrors the legacy systemd ExecStart (deploy/neuralnetviz.service).
CMD ["opentelemetry-instrument", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8801", "--workers", "1"]
