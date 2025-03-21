import tensorflow as tf
import keras
from keras import layers
import os
import pathlib
import subprocess
import sys

# Function to install packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except Exception as e:
        print(f"Failed to install {package}: {e}")

# List of required libraries
packages = [
    "tensorflow",
    "torch",
    "torchvision",
    "torchaudio",
    "opencv-python",
    "openvino",
    "datasets",
]

# Install each package if not already installed
for package in packages:
    try:
        __import__(package.split()[0])
    except ImportError:
        install_package(package)


# Load dataset
login(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,false, true,false)
dataset_path = "viola77data/recycling-dataset"
data_dir = pathlib.Path(dataset_path)

if not data_dir.exists():
    print(f"Error: Dataset path {dataset_path} not found.")
    sys.exit(1)

# Parameters
img_height, img_width = 180, 180
batch_size = 32

dataset = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Split dataset
num_samples = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * num_samples)
train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

# Normalize
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Define model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(dataset.class_names), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save model
model.save("recycling_model.h5")

print("Training complete. Model saved as recycling_model.h5")
