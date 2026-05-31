"""Fine-tune MobileNetV2 on the 6 CIFAR-10 animal classes.

Pipeline
--------
1. Load CIFAR-10 animal subset and upscale 32x32 -> 96x96 (MobileNetV2 needs
   at least 96 to keep its block 13 spatial dims > 0).
2. Apply MobileNetV2's official preprocess_input ([-1, 1]).
3. Phase 1: freeze MobileNetV2, train only the new 6-way head (~5 epochs).
4. Phase 2: unfreeze the last ~30 layers, fine-tune at a tiny learning rate
   (~10 epochs).

Honesty note: CIFAR-10 source images are 32x32 — when we upscale to 96 they
look very blurry compared to real photos a user might upload at inference
time. So this fine-tuned model is expected to do worse on sharp real-world
photos than the pure ImageNet-pretrained model called via app/imagenet_map.py.
The educational value is showing the transfer-learning workflow end-to-end,
not winning on accuracy.

Saves
-----
- models/animal_mobilenet.keras
- models/architecture_pretrained.json
- models/training_history_pretrained.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

ANIMAL_LABELS = {2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse"}
ANIMAL_IDXS = sorted(ANIMAL_LABELS.keys())
CLASS_NAMES = [ANIMAL_LABELS[i] for i in ANIMAL_IDXS]
INPUT_SIZE = 96  # MobileNetV2 supports {96, 128, 160, 192, 224}; 96 keeps it fast


def load_animal_subset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    tm = np.isin(y_train, ANIMAL_IDXS)
    em = np.isin(y_test, ANIMAL_IDXS)
    x_train, y_train = x_train[tm], y_train[tm]
    x_test, y_test = x_test[em], y_test[em]
    remap = {orig: new for new, orig in enumerate(ANIMAL_IDXS)}
    y_train = np.array([remap[int(y)] for y in y_train])
    y_test = np.array([remap[int(y)] for y in y_test])
    return (x_train.astype(np.float32), y_train), (x_test.astype(np.float32), y_test)


def make_dataset(x: np.ndarray, y: np.ndarray, batch: int, training: bool) -> tf.data.Dataset:
    """Build a tf.data pipeline that upscales on the GPU and applies
    MobileNetV2's preprocess_input."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.batch(batch)

    def prep(images, labels):
        # 32x32 -> 96x96
        images = tf.image.resize(images, [INPUT_SIZE, INPUT_SIZE], method="bilinear")
        if training:
            images = tf.image.random_flip_left_right(images)
            # mild brightness/contrast jitter
            images = tf.image.random_brightness(images, 8.0)
            images = tf.image.random_contrast(images, 0.9, 1.1)
            images = tf.clip_by_value(images, 0.0, 255.0)
        images = preprocess_input(images)  # -> [-1, 1]
        return images, labels

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Return (full_model, base_model). The base is kept exposed so phase 2
    can unfreeze its tail."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=0.5,  # smaller variant — faster fine-tune, ~1.5M params
    )
    base.trainable = False  # phase 1: head only

    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="input_image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.Dropout(0.3, name="head_drop")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="animal_mobilenet")
    return model, base


# Landmark layers from inside MobileNetV2 we expose in the architecture
# viz. The frontend draws these as the network diagram.
LANDMARK_NAMES = [
    "Conv1",
    "block_3_expand",
    "block_6_expand",
    "block_10_expand",
    "block_13_expand",
    "Conv_1",
]


def export_architecture(model: tf.keras.Model, base: tf.keras.Model) -> None:
    arch: list[dict] = []
    arch.append({
        "name": "input_image",
        "type": "InputLayer",
        "output_shape": ["?", INPUT_SIZE, INPUT_SIZE, 3],
        "params": 0,
    })
    available = {l.name: l for l in base.layers}
    for n in LANDMARK_NAMES:
        if n not in available:
            continue
        l = available[n]
        out_shape = l.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        arch.append({
            "name": n,
            "type": l.__class__.__name__,
            "output_shape": [d if d is not None else "?" for d in out_shape],
            "params": int(l.count_params()),
        })
    arch.append({
        "name": "head_gap",
        "type": "GlobalAveragePooling2D",
        "output_shape": ["?", base.output_shape[-1]],
        "params": 0,
    })
    arch.append({
        "name": "output",
        "type": "Dense",
        "output_shape": ["?", len(CLASS_NAMES)],
        "params": int(model.get_layer("output").count_params()),
    })
    (MODELS / "architecture_pretrained.json").write_text(
        json.dumps({"classes": CLASS_NAMES, "layers": arch}, indent=2)
    )


class HistoryDump(callbacks.Callback):
    """Capture per-batch and per-epoch metrics, tagging which phase produced
    each row so the playback page can render phase boundaries."""

    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase
        self.batches: list[dict] = []
        self.epochs: list[dict] = []
        self._epoch = 0
        self._batch_in_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._batch_in_epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self._batch_in_epoch % 5 == 0:
            self.batches.append({
                "phase": self.phase,
                "epoch": self._epoch,
                "batch": int(batch),
                "loss": float(logs.get("loss", 0.0)),
                "accuracy": float(logs.get("accuracy", 0.0)),
            })
        self._batch_in_epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append({
            "phase": self.phase,
            "epoch": int(epoch),
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
        })


def main():
    print("Loading CIFAR-10 animal subset...")
    (x_train, y_train), (x_test, y_test) = load_animal_subset()
    print(f"  train: {x_train.shape}, test: {x_test.shape}, upscaled to {INPUT_SIZE}x{INPUT_SIZE}")

    batch = int(os.environ.get("BATCH", "64"))
    train_ds = make_dataset(x_train, y_train, batch, training=True)
    test_ds = make_dataset(x_test, y_test, batch, training=False)

    model, base = build_model()
    model.summary()
    print(f"Total params: {model.count_params():,} | trainable head: {sum(np.prod(v.shape) for v in model.trainable_weights):,}")

    # --- Phase 1 — head only ------------------------------------------------
    phase1_epochs = int(os.environ.get("PHASE1_EPOCHS", "5"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n=== Phase 1: head-only, {phase1_epochs} epochs ===")
    dump1 = HistoryDump("head")
    model.fit(train_ds, validation_data=test_ds, epochs=phase1_epochs,
              callbacks=[dump1], verbose=2)

    # --- Phase 2 — fine-tune last N layers ---------------------------------
    phase2_epochs = int(os.environ.get("PHASE2_EPOCHS", "10"))
    unfreeze = int(os.environ.get("UNFREEZE_LAYERS", "30"))
    base.trainable = True
    for layer in base.layers[:-unfreeze]:
        layer.trainable = False
    # Keep all BatchNormalization layers in inference mode — fine-tuning their
    # running stats on a small dataset is a known foot-gun.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    print(f"\n=== Phase 2: fine-tune last {unfreeze} layers, {phase2_epochs} epochs ===")
    print(f"  trainable params now: {trainable:,}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    dump2 = HistoryDump("finetune")
    model.fit(train_ds, validation_data=test_ds, epochs=phase2_epochs,
              callbacks=[dump2], verbose=2)

    print("\nSaving model + metadata...")
    model.save(MODELS / "animal_mobilenet.keras")
    export_architecture(model, base)
    (MODELS / "training_history_pretrained.json").write_text(json.dumps({
        "classes": CLASS_NAMES,
        "epochs": dump1.epochs + dump2.epochs,
        "batches": dump1.batches + dump2.batches,
        "phases": [
            {"name": "head", "epochs": phase1_epochs},
            {"name": "finetune", "epochs": phase2_epochs, "unfrozen_layers": unfreeze},
        ],
    }, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
