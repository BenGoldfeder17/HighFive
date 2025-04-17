import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Activation, Flatten,
    Dense, Dropout
)

# ------------------------------------------------------
# PARAMETERS & DIRECTORY SETUP
# ------------------------------------------------------
dataset_dir = r"\Users\Blake\Documents\GEEN1400\dataset-original"
classes     = ['trash', 'recycling']
class_to_label = {cls:i for i,cls in enumerate(classes)}

img_height, img_width = 150, 150
batch_size = 16
epochs     = 30
learning_rate = 1e-4

# ------------------------------------------------------
# COLLECT IMAGE PATHS & LABELS
# ------------------------------------------------------
all_paths  = []
all_labels = []
for cls in classes:
    folder = os.path.join(dataset_dir, cls)
    if not os.path.isdir(folder):
        print(f"Warning: Missing folder {folder}")
        continue
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in {'.jpg','.jpeg','.png'}:
            continue
        all_paths.append(os.path.join(folder, fname))
        all_labels.append(class_to_label[cls])

print(f"Found {len(all_paths)} total images across {classes}")

# ------------------------------------------------------
# TRAIN/VAL/TEST SPLIT (70/15/15)
# ------------------------------------------------------
data = list(zip(all_paths, all_labels))
np.random.shuffle(data)
paths, labels = zip(*data)

n = len(paths)
n_train = int(0.7*n)
n_val   = int(0.15*n)

train_paths  = paths[:n_train]
train_labels = labels[:n_train]
val_paths    = paths[n_train:n_train+n_val]
val_labels   = labels[n_train:n_train+n_val]
test_paths   = paths[n_train+n_val:]
test_labels  = labels[n_train+n_val:]

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# ------------------------------------------------------
# DATASET CREATION
# ------------------------------------------------------
def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0
    return img, label

def make_dataset(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(load_and_preprocess,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds   = make_dataset(val_paths,   val_labels,   shuffle=False)
test_ds  = make_dataset(test_paths,  test_labels,  shuffle=False)

# ------------------------------------------------------
# MODEL DEFINITION: 3‑CHANNEL INPUT, 2‑WAY OUTPUT
# ------------------------------------------------------
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3,3)), Activation('relu'), MaxPooling2D(),
    Conv2D(64, (3,3)), Activation('relu'), MaxPooling2D(),
    Conv2D(128,(3,3)), Activation('relu'), MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------
# TRAINING
# ------------------------------------------------------
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds
)

# ------------------------------------------------------
# EVALUATE & SAVE
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

model.save("trash_recycling_model.h5")
print("Saved model to trash_recycling_model.h5")
