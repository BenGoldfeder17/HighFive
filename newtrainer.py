import os
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# ------------------------------------------------------
# PARAMETERS & DIRECTORY SETUP
# ------------------------------------------------------
# On macOS/Linux, drop the "C:" and use a POSIX path
dataset_dir      = "/Users/Blake/Documents/GEEN1400/dataset-original"
img_height       = 150
img_width        = 150
batch_size       = 16
epochs           = 30
learning_rate    = 1e-4
validation_split = 0.3  # 30% reserved for val+test

# ------------------------------------------------------
# DATASET CREATION
# ------------------------------------------------------
# 1) Training (70%)
train_ds = image_dataset_from_directory(
    dataset_dir,
    labels="inferred",
    label_mode="int",
    validation_split=validation_split,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 2) Validation+Test (30%), to be halved
val_test_ds = image_dataset_from_directory(
    dataset_dir,
    labels="inferred",
    label_mode="int",
    validation_split=validation_split,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 3) Capture class names & count before wrapping in prefetch
class_names = train_ds.class_names
num_classes = len(class_names)
print("Detected classes:", class_names)

# 4) Split into 15% val / 15% test
val_batches = tf.data.experimental.cardinality(val_test_ds) // 2
val_ds      = val_test_ds.take(val_batches)
test_ds     = val_test_ds.skip(val_batches)

# 5) Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

print(f"Train batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Val   batches: {tf.data.experimental.cardinality(val_ds)}")
print(f"Test  batches: {tf.data.experimental.cardinality(test_ds)}")

# ------------------------------------------------------
# MODEL DEFINITION
# ------------------------------------------------------
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, 3, activation='relu'), MaxPooling2D(),
    Conv2D(64, 3, activation='relu'), MaxPooling2D(),
    Conv2D(128,3, activation='relu'), MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------
# CALLBACKS: Early Stopping & Checkpointing
# ------------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_trash_recycling_model.h5', save_best_only=True)
]

# ------------------------------------------------------
# TRAINING
# ------------------------------------------------------
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks
)

# ------------------------------------------------------
# EVALUATE
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ------------------------------------------------------
# SAVE FINAL MODEL
# ------------------------------------------------------
model.save("final_trash_recycling_model.h5")
print("Saved final model to final_trash_recycling_model.h5")

# ------------------------------------------------------
# OPTIONAL: Plot Learning Curves
# ------------------------------------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.show()
