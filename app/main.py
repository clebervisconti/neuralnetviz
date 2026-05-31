"""FastAPI backend for NeuraNetViz.

Endpoints
---------
GET  /                     -> main inference visualization page
GET  /training             -> training playback page
GET  /api/architecture     -> network architecture JSON
GET  /api/training-history -> recorded training metrics
POST /api/predict          -> classify image; returns predictions + per-layer activation summaries
GET  /api/health           -> liveness probe
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STATIC = ROOT / "static"

app = FastAPI(title="NeuraNetViz", version="0.1.0")


class ShortCacheStatic(BaseHTTPMiddleware):
    """Cap static-asset cache TTL so iterating on the UI doesn't get stuck behind
    Cloudflare's default cache. 60s is long enough that hot reloads still benefit
    from edge caching, short enough that pushes go live within a minute."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=60"
        elif path in {"/", "/training"}:
            response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(ShortCacheStatic)

# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------

_model: tf.keras.Model | None = None
_activation_model: tf.keras.Model | None = None
_probe_layer_names: list[str] = []
_arch: dict | None = None


def _load() -> tuple[tf.keras.Model, tf.keras.Model, dict]:
    global _model, _activation_model, _probe_layer_names, _arch
    if _model is None:
        model_path = MODELS / "animal_cnn.keras"
        if not model_path.exists():
            raise RuntimeError(
                f"Trained model not found at {model_path}. Run `python train.py` first."
            )
        _model = tf.keras.models.load_model(model_path)
        wanted = [l for l in _model.layers if l.__class__.__name__ in {
            "Conv2D", "MaxPooling2D", "GlobalAveragePooling2D", "Dense"
        }]
        _probe_layer_names = [l.name for l in wanted]
        _activation_model = tf.keras.Model(
            inputs=_model.input,
            outputs=[l.output for l in wanted],
            name="activation_probe",
        )
        _arch = json.loads((MODELS / "architecture.json").read_text())
    return _model, _activation_model, _arch


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_bytes: bytes) -> tuple[np.ndarray, str]:
    """Return (1,32,32,3) float array and a base64 PNG of the resized input."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    buf = io.BytesIO()
    img.resize((128, 128), Image.NEAREST).save(buf, format="PNG")
    encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return arr[None, ...], encoded


def _activation_summary(act: np.ndarray) -> dict:
    """Compress an activation tensor into something light enough to ship to the browser."""
    a = np.asarray(act)
    summary: dict = {
        "shape": list(a.shape),
        "mean": float(a.mean()),
        "max": float(a.max()),
        "min": float(a.min()),
    }
    if a.ndim == 4:
        # (1, H, W, C) - emit up to 8 channel heatmaps as 0-255 grids
        a = a[0]
        h, w, c = a.shape
        # take the top-8 channels by activation energy
        energies = a.reshape(-1, c).sum(axis=0)
        top = np.argsort(-energies)[: min(8, c)]
        heatmaps = []
        for ch in top:
            plane = a[:, :, ch]
            mn, mx = float(plane.min()), float(plane.max())
            if mx - mn < 1e-9:
                norm = np.zeros_like(plane)
            else:
                norm = (plane - mn) / (mx - mn)
            heatmaps.append({
                "channel": int(ch),
                "values": (norm * 255).astype(np.uint8).flatten().tolist(),
                "h": int(h),
                "w": int(w),
            })
        summary["heatmaps"] = heatmaps
    elif a.ndim == 2:
        # (1, units) -> 1D bar chart
        summary["values"] = a[0].astype(float).tolist()
    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/architecture")
def architecture():
    _, _, arch = _load()
    return arch


@app.get("/api/training-history")
def training_history():
    path = MODELS / "training_history.json"
    if not path.exists():
        raise HTTPException(404, "training history not available; run train.py")
    return JSONResponse(json.loads(path.read_text()))


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(400, "expected an image upload")
    raw = await image.read()
    try:
        arr, preview = _preprocess(raw)
    except Exception as exc:
        raise HTTPException(400, f"could not decode image: {exc}")

    model, act_model, arch = _load()
    activations = act_model.predict(arr, verbose=0)
    # normalize to list (when there's only one output, Keras hands back a single tensor)
    if not isinstance(activations, list):
        activations = [activations]
    # final prediction is the last activation when its layer is the Dense output
    probs = model.predict(arr, verbose=0)[0]
    classes = arch["classes"]
    ranked = sorted(
        ({"label": cls, "prob": float(p)} for cls, p in zip(classes, probs)),
        key=lambda x: -x["prob"],
    )

    summaries = []
    for name, act in zip(_probe_layer_names, activations):
        summaries.append({"name": name, **_activation_summary(act)})

    return {
        "preview": preview,
        "predictions": ranked,
        "layers": summaries,
    }


# ---------------------------------------------------------------------------
# Static / pages
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/training")
def training_page():
    return FileResponse(STATIC / "training.html")
