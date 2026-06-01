"""FastAPI backend for NeuraNetViz.

The UI now only exposes the `pretrained` MobileNetV2 mode — accurate on
real-world animal photos. The `teaching` (from-scratch CIFAR-10 CNN) and
`finetuned` modes are still implemented and reachable via ?mode= for
anyone who wants to compare in the API, but they are no longer surfaced
in the page.

Endpoints
---------
GET  /                              -> main inference page
GET  /api/architecture?mode=...     -> network architecture JSON
POST /api/predict?mode=...          -> classify image; activations + predictions
GET  /api/health                    -> liveness probe
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

import tensorflow as tf

from app import imagenet_map

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STATIC = ROOT / "static"

Mode = Literal["teaching", "pretrained", "finetuned"]

app = FastAPI(title="NeuraNetViz", version="0.2.0")


class ShortCacheStatic(BaseHTTPMiddleware):
    """Cap static-asset cache TTL so iterating on the UI doesn't get stuck behind
    Cloudflare's default cache. 60s is long enough that hot reloads still benefit
    from edge caching, short enough that pushes go live within a minute."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=60"
        elif path == "/":
            response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(ShortCacheStatic)


# ---------------------------------------------------------------------------
# Model registries — one entry per mode, all lazy-loaded.
# ---------------------------------------------------------------------------

class ModelBundle:
    """Everything the request handler needs to run inference + emit a viz payload
    for a particular mode."""

    def __init__(
        self,
        model: tf.keras.Model,
        activation_model: tf.keras.Model,
        probe_layer_names: list[str],
        arch: dict,
        input_size: int,
        preprocess_fn,
        decode_fn=None,
    ):
        self.model = model
        self.activation_model = activation_model
        self.probe_layer_names = probe_layer_names
        self.arch = arch
        self.input_size = input_size
        self.preprocess_fn = preprocess_fn
        self.decode_fn = decode_fn


_bundles: dict[Mode, ModelBundle] = {}


def _build_arch(layers: list[tf.keras.layers.Layer], classes: list[str], hide_input: bool = False) -> dict:
    arch_layers = []
    seen_input = False
    for layer in layers:
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        arch_layers.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": [d if d is not None else "?" for d in out_shape],
            "params": int(layer.count_params()),
        })
    return {"classes": classes, "layers": arch_layers}


def _load_teaching() -> ModelBundle:
    """The from-scratch CIFAR-10 CNN. Inputs are 32x32 in [0,1]."""
    model_path = MODELS / "animal_cnn.keras"
    if not model_path.exists():
        raise RuntimeError(f"Trained model not found at {model_path}. Run `python train.py` first.")
    model = tf.keras.models.load_model(model_path)
    wanted = [
        l for l in model.layers
        if l.__class__.__name__ in {"Conv2D", "MaxPooling2D", "GlobalAveragePooling2D", "Dense"}
    ]
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=[l.output for l in wanted],
        name="activation_probe",
    )
    arch = json.loads((MODELS / "architecture.json").read_text())

    def preprocess(image_bytes: bytes) -> tuple[np.ndarray, str]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        buf = io.BytesIO()
        img.resize((128, 128), Image.NEAREST).save(buf, format="PNG")
        encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return arr[None, ...], encoded

    return ModelBundle(
        model=model,
        activation_model=activation_model,
        probe_layer_names=[l.name for l in wanted],
        arch=arch,
        input_size=32,
        preprocess_fn=preprocess,
    )


# Landmark layers of MobileNetV2 we expose as the "visible" architecture —
# enough to convey depth without flooding the diagram with 150 lines.
_MOBILENET_LANDMARKS = [
    "input_1",                  # input
    "Conv1",                    # stem
    "block_3_expand",           # early features
    "block_6_expand",           # mid features
    "block_10_expand",          # later features
    "block_13_expand",          # high-level features
    "Conv_1",                   # final conv
    "global_average_pooling2d", # GAP
    "predictions",              # softmax
]


def _load_pretrained() -> ModelBundle:
    """MobileNetV2 with ImageNet weights. Inputs are 224x224 preprocessed for
    MobileNet (values in [-1,1])."""
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input,
        decode_predictions,
    )

    model = MobileNetV2(weights="imagenet", include_top=True, alpha=1.0)
    # build the activation probe: only expose landmark layers that actually
    # exist in this version of MobileNetV2.
    available = {l.name for l in model.layers}
    wanted_names = [n for n in _MOBILENET_LANDMARKS if n in available]
    wanted = [model.get_layer(n) for n in wanted_names]
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=[l.output for l in wanted],
        name="mobilenet_probe",
    )

    classes = ["bird", "cat", "deer", "dog", "frog", "horse"]
    arch = _build_arch(wanted, classes)

    def preprocess(image_bytes: bytes) -> tuple[np.ndarray, str]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        arr = preprocess_input(np.asarray(img, dtype=np.float32))  # -> [-1, 1]
        buf = io.BytesIO()
        img.resize((224, 224), Image.LANCZOS).save(buf, format="PNG")
        encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return arr[None, ...], encoded

    return ModelBundle(
        model=model,
        activation_model=activation_model,
        probe_layer_names=wanted_names,
        arch=arch,
        input_size=224,
        preprocess_fn=preprocess,
        decode_fn=decode_predictions,
    )


def _load_finetuned() -> ModelBundle:
    """The locally fine-tuned MobileNetV2 (head-only then last-blocks unfreeze)
    on the 6 CIFAR-10 animal classes, 96x96 inputs. Trained by train_pretrained.py."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model_path = MODELS / "animal_mobilenet.keras"
    if not model_path.exists():
        raise RuntimeError(
            f"Fine-tuned model not found at {model_path}. Run `python train_pretrained.py` first."
        )
    model = tf.keras.models.load_model(model_path)

    # The base MobileNetV2 is nested inside `model` as a single sub-model layer
    # (Keras wraps `base(inputs)` as one layer with the base's name). Reach in to
    # grab landmark layers for the activation probe + arch viz.
    base = None
    for l in model.layers:
        if isinstance(l, tf.keras.Model) and "mobilenet" in l.name.lower():
            base = l
            break
    if base is None:
        # Fallback: treat the whole model as flat
        base = model

    landmark_candidates = [
        "Conv1", "block_3_expand", "block_6_expand",
        "block_10_expand", "block_13_expand", "Conv_1",
    ]
    available = {l.name for l in base.layers}
    wanted_in_base = [n for n in landmark_candidates if n in available]
    # Also expose the head GAP + Dense from the outer model
    head_names = [l.name for l in model.layers if l.name in {"head_gap", "output"}]

    # Build probe model that returns each landmark's activation. To get
    # intermediate activations from a nested base, we need to lift them into
    # the outer functional model.
    base_outputs = [base.get_layer(n).output for n in wanted_in_base]
    probe_base = tf.keras.Model(inputs=base.input, outputs=base_outputs, name="mobilenet_probe_inner")

    inner_acts = probe_base(model.input)
    if not isinstance(inner_acts, list):
        inner_acts = [inner_acts]
    head_acts = [model.get_layer(n).output for n in head_names]
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=inner_acts + head_acts,
        name="finetuned_probe",
    )

    arch = json.loads((MODELS / "architecture_pretrained.json").read_text())

    def preprocess(image_bytes: bytes) -> tuple[np.ndarray, str]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((96, 96))
        arr = preprocess_input(np.asarray(img, dtype=np.float32))
        buf = io.BytesIO()
        img.resize((192, 192), Image.LANCZOS).save(buf, format="PNG")
        encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return arr[None, ...], encoded

    return ModelBundle(
        model=model,
        activation_model=activation_model,
        probe_layer_names=wanted_in_base + head_names,
        arch=arch,
        input_size=96,
        preprocess_fn=preprocess,
    )


def _get_bundle(mode: Mode) -> ModelBundle:
    if mode not in _bundles:
        if mode == "teaching":
            _bundles[mode] = _load_teaching()
        elif mode == "pretrained":
            _bundles[mode] = _load_pretrained()
        elif mode == "finetuned":
            _bundles[mode] = _load_finetuned()
        else:
            raise HTTPException(400, f"unknown mode: {mode}")
    return _bundles[mode]


# ---------------------------------------------------------------------------
# Activation -> JSON payload
# ---------------------------------------------------------------------------

def _activation_summary(act: np.ndarray) -> dict:
    a = np.asarray(act)
    summary: dict = {
        "shape": list(a.shape),
        "mean": float(a.mean()),
        "max": float(a.max()),
        "min": float(a.min()),
    }
    if a.ndim == 4:
        a = a[0]
        h, w, c = a.shape
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
                # Real activation magnitude of this channel (mean over its spatial
                # plane). The `values` above are per-channel normalized for the
                # heatmap image and so always peak at 255 — useless as a node
                # score. `act` preserves the true strength so the diagram can color
                # and label each Conv node by its actual activation, like GAP does.
                "act": float(plane.mean()),
                "h": int(h),
                "w": int(w),
            })
        # Layer-wide reference so the frontend can normalize channel strengths
        # consistently (strongest channel in the layer -> full intensity).
        summary["chan_act_max"] = max((hm["act"] for hm in heatmaps), default=0.0)
        summary["heatmaps"] = heatmaps
    elif a.ndim == 2:
        summary["values"] = a[0].astype(float).tolist()
    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "modes": ["teaching", "pretrained", "finetuned"]}


@app.get("/api/architecture")
def architecture(mode: Mode = Query("teaching")):
    bundle = _get_bundle(mode)
    arch = dict(bundle.arch)
    arch["mode"] = mode
    arch["input_size"] = bundle.input_size
    return arch


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    mode: Mode = Query("teaching"),
):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(400, "expected an image upload")
    raw = await image.read()
    bundle = _get_bundle(mode)
    try:
        arr, preview = bundle.preprocess_fn(raw)
    except Exception as exc:
        raise HTTPException(400, f"could not decode image: {exc}")

    activations = bundle.activation_model.predict(arr, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]

    classes = bundle.arch["classes"]
    if mode == "pretrained":
        # Reduce 1000 ImageNet probs to our 6 animal buckets.
        probs_raw = bundle.model.predict(arr, verbose=0)
        decoded = bundle.decode_fn(probs_raw, top=30)[0]
        bucketed = imagenet_map.aggregate(decoded, classes)
        ranked = sorted(
            ({"label": cls, "prob": float(p)} for cls, p in bucketed.items()),
            key=lambda x: -x["prob"],
        )
        # also surface the raw ImageNet top-3 so the demo is transparent
        raw_top = [{"label": name, "prob": float(p)} for _w, name, p in decoded[:3]]
    else:
        probs = bundle.model.predict(arr, verbose=0)[0]
        ranked = sorted(
            ({"label": cls, "prob": float(p)} for cls, p in zip(classes, probs)),
            key=lambda x: -x["prob"],
        )
        raw_top = None

    summaries = []
    for name, act in zip(bundle.probe_layer_names, activations):
        summaries.append({"name": name, **_activation_summary(act)})

    payload = {
        "preview": preview,
        "predictions": ranked,
        "layers": summaries,
        "mode": mode,
    }
    if raw_top is not None:
        payload["imagenet_top"] = raw_top
    return payload


# ---------------------------------------------------------------------------
# Static / pages
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
