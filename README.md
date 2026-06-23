# NeuraNetViz

Live, in-browser visualization of a convolutional neural network for AI/ML classes.
Upload an animal image, watch data flow layer-by-layer through the trained CNN, and see the
prediction emerge — with feature-map heatmaps and a recorded training-run playback.

Live demo: <https://neuralnetviz.clebervisconti.com>

## What it shows

- A small CNN trained **from scratch** on the 6 animal classes of CIFAR-10
  (bird, cat, deer, dog, frog, horse).
- An animated SVG network diagram with neon edges and glow that pulses data through each layer.
- The actual per-layer activations (top channels as 0-255 heatmaps) for the uploaded image.
- A training playback page that animates the real loss/accuracy curves captured during training.

## Architecture

```
input(32x32x3) → Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Conv2D(64) → GAP → Dense(64) → Dense(6, softmax)
```

## Run locally

### Docker (recommended — matches production)

The app ships as a container with an OpenTelemetry Collector sidecar that exports
traces + metrics to Splunk Observability Cloud.

```bash
cp .env.example .env          # fill SPLUNK_ACCESS_TOKEN (realm defaults to us1)
docker compose up --build     # app + otel-collector
open http://localhost:8801
```

### Bare Python (no telemetry sidecar)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py                # only needed for the from-scratch CNN (teaching mode)
uvicorn app.main:app --reload  # serves http://localhost:8000
```

## Deploy

**Pipeline:** GitHub (source of truth) → Docker build + test locally → GHCR via GitHub
Actions → `docker compose` on the HostGator VPS, behind OpenLiteSpeed + Cloudflare at
`neuralnetviz.clebervisconti.com`.

- **CI** — pushing to `main` builds the image and pushes
  `ghcr.io/clebervisconti/neuralnetviz:{latest,<git-sha>}` (`.github/workflows/build-and-push.yml`).
- **VPS** — the override `docker-compose.vps.yml` runs the GHCR image bound to
  `127.0.0.1:8801` (OLS proxies the subdomain to it, unchanged). Deploy with
  `deploy/neuralnetviz-deploy.sh [tag]`, which pulls + `up -d` + health-checks.
- **Telemetry** — the `otel-collector` sidecar (`deploy/otel-collector-config.yaml`)
  forwards app traces + metrics to Splunk O11y; browser RUM is in `static/index.html`.
- **Rollback** — the legacy systemd unit (`deploy/neuralnetviz.service`) is kept; stop
  the stack and `systemctl start neuralnetviz`, or redeploy a previous image tag.

This flow is codified in the `app-deployment-orchestrator` skill as **Shape E
(containerized app + OTel sidecar)**.

## Tech

- TensorFlow / Keras (model + training)
- FastAPI + Uvicorn (backend)
- Vanilla JS + SVG + Canvas (frontend; no framework, no build step)

## License

MIT
