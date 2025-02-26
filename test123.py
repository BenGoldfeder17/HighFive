import time
import numpy as np
import tensorflow as tf
from picamera import PiCamera
from picamera.array import PiRGBArray
from PIL import Image

# --- Configuration ---
MODEL_PATH = 'model.h5'            # Path to your trained TensorFlow/Keras model
IMG_WIDTH, IMG_HEIGHT = 224, 224   # Expected input dimensions for the model
CLASS_NAMES = ['non_recyclable', 'recyclable']  # Modify based on your training

# --- Load the Model ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# --- Initialize the Camera ---
camera = PiCamera()
camera.resolution = (640, 480)  # Set resolution as needed
camera.framerate = 30         # Set framerate for video capture
time.sleep(2)  # Allow the camera to warm up

# Create a PiRGBArray object for capturing frames
rawCapture = PiRGBArray(camera, size=camera.resolution)

print("Starting video stream...")

try:
    # Continuously capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
        # Obtain the NumPy array representing the image
        image_array = frame.array

        # --- Process the Image ---
        # Convert the image array to a PIL Image for easier processing
        image = Image.fromarray(image_array)
        # Resize to the input dimensions expected by the model
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        # Convert back to a NumPy array and normalize pixel values to [0, 1]
        img_np = np.array(image).astype('float32') / 255.0
        # Add a batch dimension as the model expects batches of images
        img_np = np.expand_dims(img_np, axis=0)

        # --- Run Inference ---
        predictions = model.predict(img_np)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_index]
        result = CLASS_NAMES[predicted_index]

        # --- Output the Result ---
        if result == 'recyclable':
            print("Detected RECYCLABLE item (Confidence: {:.2f}%)".format(confidence * 100))
        else:
            print("Detected NON-RECYCLABLE item (Confidence: {:.2f}%)".format(confidence * 100))

        # Clear the stream for the next frame
        rawCapture.truncate(0)

except KeyboardInterrupt:
    print("Video analysis stopped by user.")

finally:
    camera.close()
