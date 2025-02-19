# Creating a program that uses the Raspberry Pi AI Camera with the Pi 4 to detect which trash goes in the recycling and which goes in the trash

# Importing the necessary libraries
#fuck me in the asshole
import cv2
import numpy as np
import os
import time
import picamera
import picamera.array
import RPi.GPIO as GPIO
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from picamera import PiCamera
from picamera.array import PiRGBArray
from tflite_runtime.interpreter import Interpreter

# Setting up the GPIO pins for the servo motor
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
pwm = GPIO.PWM(18, 50)
pwm.start(0)

# Setting up the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

# Initialize the camera and the servo motor
camera.start_preview()
time.sleep(2)
pwm.ChangeDutyCycle(2)
time.sleep(1)

# Loop through the camera frames
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    input_data = np.expand_dims(image_rgb, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    prediction = labels[np.argmax(results)]
    print(prediction)

    if prediction == 'recycling':
        pwm.ChangeDutyCycle(7)  # Rotate servo to the left
        time.sleep(1)
        pwm.ChangeDutyCycle(2)  # Return to the neutral position
        time.sleep(1)
    else:
        pwm.ChangeDutyCycle(12)  # Rotate servo to the right
        time.sleep(1)
        pwm.ChangeDutyCycle(2)  # Return to the neutral position
        time.sleep(1)

    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.stop_preview()
pwm.stop()
GPIO.cleanup()
cv2.destroyAllWindows()

# End of program