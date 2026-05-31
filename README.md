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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py                # trains the CNN and writes models/
uvicorn app.main:app --reload  # serves http://localhost:8000
```

## Deploy

Behind a reverse proxy (OpenLiteSpeed on the HostGator VPS) with Cloudflare in front of
`neuralnetviz.clebervisconti.com`. A systemd unit (`deploy/neuralnetviz.service`)
runs the FastAPI app as a long-lived process; OLS proxies the subdomain to it.

## Tech

- TensorFlow / Keras (model + training)
- FastAPI + Uvicorn (backend)
- Vanilla JS + SVG + Canvas (frontend; no framework, no build step)

## License

MIT
