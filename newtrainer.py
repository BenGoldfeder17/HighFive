import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

# ------------------------------------------------------
# PARAMETERS & DIRECTORY SETUP
# ------------------------------------------------------
dataset_dir    = r"F:\dataset-original"
img_height     = 150
img_width      = 150
batch_size     = 16
epochs         = 30
learning_rate  = 1e-4

# ------------------------------------------------------
# GUARD: ensure base folder exists
# ------------------------------------------------------
if not os.path.isdir(dataset_dir):
    raise FileNotFoundError(
        f"Dataset directory not found:\n  {dataset_dir}\n"
        "Please verify the path (use raw-string r\"...\" or double backslashes)."
    )

# ------------------------------------------------------
# DISCOVER CLASSES DYNAMICALLY
# ------------------------------------------------------
classes = sorted(
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
)
if not classes:
    raise RuntimeError(f"No class sub‑folders found under '{dataset_dir}'")
print(f"Discovered classes: {classes}")

class_to_label = {cls: idx for idx, cls in enumerate(classes)}

# ------------------------------------------------------
# COLLECT IMAGE PATHS & LABELS
# ------------------------------------------------------
all_paths, all_labels = [], []
for cls in classes:
    folder = os.path.join(dataset_dir, cls)
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in {'.jpg', '.jpeg', '.png'}:
            continue
        all_paths.append(os.path.join(folder, fname))
        all_labels.append(class_to_label[cls])

if not all_paths:
    raise RuntimeError(f"No images found in sub‑folders of '{dataset_dir}'")
print(f"Found {len(all_paths)} images across {classes}")

# ------------------------------------------------------
# TRAIN/VAL/TEST SPLIT (70/15/15)
# ------------------------------------------------------
data = list(zip(all_paths, all_labels))
np.random.shuffle(data)
paths, labels = zip(*data)

n       = len(paths)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

train_paths  = paths[:n_train]
train_labels = labels[:n_train]
val_paths    = paths[n_train:n_train + n_val]
val_labels   = labels[n_train:n_train + n_val]
test_paths   = paths[n_train + n_val:]
test_labels  = labels[n_train + n_val:]

print(f"Split → Train:{len(train_paths)}, Val:{len(val_paths)}, Test:{len(test_paths)}")

# ------------------------------------------------------
# DATASET CREATION
# ------------------------------------------------------
def load_and_preprocess(path, label):
    # Read file bytes
    img_bytes = tf.io.read_file(path)
    # Decode based on file content
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    # Resize & normalize
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0
    return img, label

def make_dataset(paths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # Skip any elements that caused decode errors
    ds = ds.apply(tf.data.experimental.ignore_errors())
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds   = make_dataset(val_paths,   val_labels,   shuffle=False)
test_ds  = make_dataset(test_paths,  test_labels,  shuffle=False)

# ------------------------------------------------------
# MODEL DEFINITION: 3‑CHANNEL INPUT, N‑WAY OUTPUT
# ------------------------------------------------------
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3,3)), Activation('relu'), MaxPooling2D(),
    Conv2D(64, (3,3)), Activation('relu'), MaxPooling2D(),
    Conv2D(128,(3,3)),Activation('relu'), MaxPooling2D(),
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
# EVALUATE & SAVE (Keras native .keras format)
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

model.save("trash_recycling_model.keras")
print("Saved model to trash_recycling_model.keras")
