"""
Deepfake Detection — Training Script
Optimized for small datasets (≤ 2000 images) using:
  • Pretrained Xception (ImageNet weights)
  • Heavy data augmentation
  • Two-phase fine-tuning
  • Class-weight balancing
  • EarlyStopping to avoid overfitting
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (load_img, img_to_array,
                                                   ImageDataGenerator)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = "processed"     # output of 1_preprocess.py
MODEL_PATH    = "deepfake_model.h5"
IMG_SIZE      = (224, 224)
MAX_IMAGES    = 2000            # hard cap — 1000 real + 1000 fake
BATCH_SIZE    = 16              # safe for 8 GB RAM laptops
EPOCHS_HEAD   = 10              # phase 1: only train new head
EPOCHS_FINE   = 30              # phase 2: fine-tune top layers
LR_HEAD       = 1e-3
LR_FINE       = 1e-5
UNFREEZE_LAST = 40              # how many Xception layers to unfreeze

# ── LOAD & CAP DATASET ───────────────────────────────────────────────────────
def load_dataset(processed_dir, max_per_class=1000):
    """Load up to max_per_class images per label, apply Xception preprocessing."""
    images, labels = [], []
    for label_idx, class_name in enumerate(["real", "fake"]):
        folder = Path(processed_dir) / class_name
        files  = (list(folder.glob("*.jpg")) +
                  list(folder.glob("*.jpeg")) +
                  list(folder.glob("*.png")))

        # Shuffle before capping so we get a representative sample
        np.random.seed(42)
        np.random.shuffle(files)
        files = files[:max_per_class]

        print(f"  [{class_name}] Loading {len(files)} images …")
        for f in files:
            try:
                img = load_img(f, target_size=IMG_SIZE)
                arr = img_to_array(img)
                # Xception expects pixels in [-1, 1] (not 0–1)
                arr = preprocess_input(arr)
                images.append(arr)
                labels.append(label_idx)
            except Exception as e:
                print(f"    Skip {f.name}: {e}")

    X = np.array(images, dtype="float32")
    y = np.array(labels, dtype="float32")
    return X, y


print("=" * 50)
print("Loading dataset (max 2000 images) …")
X, y = load_dataset(PROCESSED_DIR, max_per_class=MAX_IMAGES // 2)
X, y = shuffle(X, y, random_state=42)
print(f"  Total loaded: {len(X)}  (real={int(sum(y==0))}, fake={int(sum(y==1))})")

# 80 / 20 split — stratified so both classes appear in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

# ── CLASS WEIGHTS (handles imbalance) ───────────────────────────────────────
cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train)
class_weights = {0: cw[0], 1: cw[1]}
print(f"  Class weights → real: {cw[0]:.2f}, fake: {cw[1]:.2f}")

# ── DATA AUGMENTATION (virtually expands small dataset) ─────────────────────
# Only applied to training set — test set stays clean
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest"
)

# No augmentation on test data
test_datagen = ImageDataGenerator()   # identity — just batches

train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=42)
test_gen  = test_datagen.flow(X_test,  y_test,  batch_size=BATCH_SIZE, shuffle=False)

steps_per_epoch  = max(1, len(X_train) // BATCH_SIZE)
validation_steps = max(1, len(X_test)  // BATCH_SIZE)

# ── BUILD MODEL ──────────────────────────────────────────────────────────────
print("\nBuilding Xception-based model …")

base = Xception(weights="imagenet",       # loads pretrained weights automatically
                include_top=False,         # remove ImageNet classifier head
                input_shape=(224, 224, 3))
base.trainable = False                     # freeze entire base for phase 1

x = base.output
x = GlobalAveragePooling2D()(x)           # pool spatial features → 1D vector
x = BatchNormalization()(x)               # stabilise activations
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)                       # heavy dropout for small data
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x) # 0 = real, 1 = fake

model = Model(inputs=base.input, outputs=output)

print(f"  Total params    : {model.count_params():,}")
print(f"  Trainable params: {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")

# ── PHASE 1 — Train only the new head (base frozen) ─────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_HEAD),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

callbacks_phase1 = [
    EarlyStopping(monitor="val_accuracy", patience=4,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", patience=3,
                      factor=0.5, min_lr=1e-7, verbose=1)
]

print("\n── Phase 1: Training head (base frozen) ──")
h1 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks_phase1
)

# ── PHASE 2 — Fine-tune top UNFREEZE_LAST layers of Xception ────────────────
print(f"\n── Phase 2: Unfreezing top {UNFREEZE_LAST} Xception layers ──")
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:
    layer.trainable = False

trainable_now = sum(1 for l in base.layers if l.trainable)
print(f"  Trainable Xception layers: {trainable_now} / {len(base.layers)}")

# Much lower LR for fine-tuning to avoid destroying pretrained features
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINE),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_accuracy", patience=7,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", patience=3,
                      factor=0.3, min_lr=1e-8, verbose=1)
]

h2 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks_phase2
)

# ── FINAL EVALUATION ─────────────────────────────────────────────────────────
print("\n── Final evaluation on test set ──")
results = model.evaluate(test_gen, steps=validation_steps, verbose=1)
metric_names = ["loss", "accuracy", "auc", "precision", "recall"]
for name, val in zip(metric_names, results):
    print(f"  {name:<12}: {val:.4f}")

# ── PLOT CURVES ──────────────────────────────────────────────────────────────
acc1 = h1.history.get("accuracy", [])
acc2 = h2.history.get("accuracy", [])
val1 = h1.history.get("val_accuracy", [])
val2 = h2.history.get("val_accuracy", [])

all_acc = acc1 + acc2
all_val = val1 + val2
epochs  = range(1, len(all_acc) + 1)
phase2_start = len(acc1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(epochs, all_acc, label="Train accuracy", linewidth=2)
axes[0].plot(epochs, all_val, label="Val accuracy",   linewidth=2)
axes[0].axhline(0.95, color="red", linestyle="--", linewidth=1, label="95% target")
axes[0].axvline(phase2_start, color="gray", linestyle=":", linewidth=1,
                label="Fine-tune start")
axes[0].set_title("Accuracy over epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss
loss1  = h1.history.get("loss", [])
loss2  = h2.history.get("loss", [])
vloss1 = h1.history.get("val_loss", [])
vloss2 = h2.history.get("val_loss", [])
all_loss  = loss1  + loss2
all_vloss = vloss1 + vloss2

axes[1].plot(epochs, all_loss,  label="Train loss", linewidth=2)
axes[1].plot(epochs, all_vloss, label="Val loss",   linewidth=2)
axes[1].axvline(phase2_start, color="gray", linestyle=":", linewidth=1,
                label="Fine-tune start")
axes[1].set_title("Loss over epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

print(f"\n✓ Model saved  → {MODEL_PATH}")
print(f"✓ Curves saved → training_curves.png")