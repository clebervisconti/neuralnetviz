"""Train a small CNN on the 6 animal classes of CIFAR-10.

Classes: bird, cat, deer, dog, frog, horse.
Saves:
  models/animal_cnn.keras   - trained weights
  models/training_history.json - per-epoch loss/accuracy for the training viz
  models/architecture.json  - layer-by-layer description for the network viz
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# CIFAR-10 label indices for animals only
ANIMAL_LABELS = {
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
}
ANIMAL_IDXS = sorted(ANIMAL_LABELS.keys())
CLASS_NAMES = [ANIMAL_LABELS[i] for i in ANIMAL_IDXS]


def load_animal_subset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    train_mask = np.isin(y_train, ANIMAL_IDXS)
    test_mask = np.isin(y_test, ANIMAL_IDXS)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # remap labels to 0..5
    remap = {orig: new for new, orig in enumerate(ANIMAL_IDXS)}
    y_train = np.array([remap[int(y)] for y in y_train])
    y_test = np.array([remap[int(y)] for y in y_test])

    return (x_train / 255.0, y_train), (x_test / 255.0, y_test)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(32, 32, 3), name="input_image")
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="output")(x)
    model = models.Model(inputs, outputs, name="animal_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_architecture(model: tf.keras.Model) -> None:
    """Dump a JSON description of every layer for the front-end network viz."""
    arch = []
    for layer in model.layers:
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        params = int(layer.count_params())
        arch.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": [d if d is not None else "?" for d in out_shape],
            "params": params,
        })
    payload = {"classes": CLASS_NAMES, "layers": arch}
    (MODELS / "architecture.json").write_text(json.dumps(payload, indent=2))


class HistoryDump(callbacks.Callback):
    """Capture per-batch loss/accuracy so the viz can replay the run smoothly."""

    def __init__(self):
        super().__init__()
        self.batches: list[dict] = []
        self.epochs: list[dict] = []
        self._epoch = 0
        self._batch_in_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._batch_in_epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        # sample every 5th batch to keep the file small
        if self._batch_in_epoch % 5 == 0:
            self.batches.append({
                "epoch": self._epoch,
                "batch": int(batch),
                "loss": float(logs.get("loss", 0.0)),
                "accuracy": float(logs.get("accuracy", 0.0)),
            })
        self._batch_in_epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append({
            "epoch": int(epoch),
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
        })


def main():
    print("Loading CIFAR-10 animal subset...")
    (x_train, y_train), (x_test, y_test) = load_animal_subset()
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    model = build_model()
    model.summary()

    dump = HistoryDump()
    epochs = int(os.environ.get("EPOCHS", "12"))
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=128,
        callbacks=[dump],
        verbose=2,
    )

    print("Saving model and metadata...")
    model.save(MODELS / "animal_cnn.keras")
    export_architecture(model)
    (MODELS / "training_history.json").write_text(json.dumps({
        "classes": CLASS_NAMES,
        "epochs": dump.epochs,
        "batches": dump.batches,
    }, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
