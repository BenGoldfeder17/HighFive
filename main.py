import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera import PiCamera
from tflite_runtime.interpreter import Interpreter

# jon has a big forehead
# --- Servo Motor Setup ---
# Define the GPIO pin connected to the servo
SERVO_PIN = 18

# Setup GPIO using BCM numbering and initialize PWM for the servo at 50Hz.
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz frequency
pwm.start(0)

def set_servo_angle(angle):
    """
    Converts an angle (in degrees) to the appropriate PWM duty cycle and
    moves the servo to that angle.
    """
    # Typical conversion for many servos: duty cycle = angle/18 + 2
    duty = angle / 18 + 2
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)  # Allow time for servo to move
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

# --- Object Categories Mapping ---
# Define a set of COCO class labels that you consider recyclable.
# Adjust or expand this list as needed.
recycling_items = {'bottle', 'can', 'cup', 'box'}

# --- Load TFLite Model and Labels ---
MODEL_PATH = 'detect.tflite'    # Path to your TFLite detection model
LABELS_PATH = 'labelmap.txt'      # Path to your label map file

# Initialize the TFLite interpreter.
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_labels(path):
    """
    Loads the label map from a file. The file should have one label per line.
    """
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(LABELS_PATH)

# --- Camera Setup ---
# Initialize the PiCamera.
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30

def capture_image():
    """
    Captures an image from the Pi Camera and returns it as a NumPy array.
    """
    image = np.empty((480, 640, 3), dtype=np.uint8)
    camera.capture(image, 'rgb')
    return image

# --- Image Preprocessing and Detection ---
def process_image(image):
    """
    Resizes and normalizes the captured image to match the model's input requirements.
    """
    input_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]
    height, width = input_shape[1], input_shape[2]
    img_resized = cv2.resize(image, (width, height))
    # Expand dimensions and normalize the pixel values (example normalization)
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
    input_data = (input_data - 127.5) / 127.5
    return input_data

def detect_objects(image):
    """
    Runs inference on the image and returns bounding boxes, class indices,
    confidence scores, and number of detections.
    """
    input_data = process_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Retrieve detection results from the interpreter
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]     # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])[0]   # Class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]    # Confidence scores
    num = interpreter.get_tensor(output_details[3]['index'])[0]       # Number of detections
    return boxes, classes, scores, int(num)

def determine_category(classes, scores, threshold=0.5):
    """
    Checks the detected objects. If any detection with confidence above the threshold
    is in the recycling_items list, returns 'recycling'; otherwise returns 'trash'.
    """
    category = 'trash'
    for i, score in enumerate(scores):
        if score > threshold:
            label = labels.get(int(classes[i]), 'unknown')
            print(f"Detected: {label} with confidence {score:.2f}")
            if label in recycling_items:
                category = 'recycling'
                break
    return category

# --- Main Loop ---
def main():
    try:
        while True:
            print("Capturing image...")
            image = capture_image()

            print("Performing object detection...")
            boxes, classes, scores, num = detect_objects(image)
            category = determine_category(classes, scores)

            if category == 'recycling':
                print("Item is recyclable. Adjusting servo for recycling bin.")
                set_servo_angle(90)  # Rotate servo to 90° for recycling
            else:
                print("Item is trash. Adjusting servo for trash bin.")
                set_servo_angle(0)   # Rotate servo to 0° for trash

            # Pause before the next detection cycle
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping program...")
    finally:
        pwm.stop()
        GPIO.cleanup()
        camera.close()

if __name__ == '__main__':
    main()
