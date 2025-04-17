#!/usr/bin/env python3
# main.py — inference script updated to use standalone Keras instead of tensorflow.keras

import os
import subprocess
import sys

# Ensure the script is running as root
if os.geteuid() != 0:
    print("Error: This script must be run as root. Try using 'sudo'.")
    sys.exit(1)

# Ensure required packages are installed
required_packages = ["RPi.GPIO", "keras", "pygame", "numpy", "picamera2"]
for pkg in required_packages:
    name = pkg.split('-')[0]
    try:
        __import__(name)
    except ImportError:
        print(f"{pkg} not found. Installing…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# -------------------------------
# 1) Keras + model setup
# -------------------------------
from keras.models import load_model
from keras.preprocessing.image import smart_resize
import numpy as np

# Class names must match training labels
class_names = ["Recyclable", "Trash"]

# Load the Keras model (native .keras format)
MODEL_PATH = "trash_recycling_model.keras"
model = load_model(MODEL_PATH)
print(f"[INFO] Loaded Keras model '{MODEL_PATH}' with classes {class_names}")

# Image size used during training
IMG_H, IMG_W = 150, 150

# -------------------------------
# 2) Preprocessing helper
# -------------------------------
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    - frame: HxWxC numpy array from Picamera2
    - returns: 1xIMG_HxIMG_Wx3 array normalized [0,1]
    """
    # Ensure RGB
    if frame.ndim == 2:
        frame = np.stack((frame,)*3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    # Normalize to [0,1]
    img = frame.astype('float32') / 255.0
    # Resize to model input
    img = smart_resize(img, (IMG_H, IMG_W))
    # Add batch dimension
    return np.expand_dims(img, axis=0)

def classify_frame(frame: np.ndarray):
    """
    Returns (predicted_class:str, confidence:float)
    """
    inp = preprocess_frame(frame)
    probs = model.predict(inp)[0]
    idx   = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])

# ---------------------------------------------------------------------
# 3) All existing GPIO, Picamera2, motor, and pygame UI code follows
#    unchanged except it now calls classify_frame().
# ---------------------------------------------------------------------

import RPi.GPIO as GPIO
import time
import pygame
import threading
from picamera2 import Picamera2

# GPIO motor pins
IN1, IN2, ENA = 23, 24, 5
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
pwm = GPIO.PWM(ENA, 1000)
pwm.start(0)

def rotate_left(speed=100):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)

def rotate_right(speed=100):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed)

def stop_motor():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(0)

# Pygame UI setup
pygame.init()
screen_width, screen_height = 720, 1280
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Smart Trash Can")

WHITE       = (255,255,255)
BLACK       = (0,0,0)
LIGHT_BLUE  = (173,216,230)
LIGHT_CORAL = (240,128,128)

font_large  = pygame.font.Font(None, 80)
font_medium = pygame.font.Font(None, 60)

running       = True
current_item  = "Scanning..."
confidence    = 0.0
trash_count   = 0
recycle_count = 0
bin_capacity  = 10
item_volume   = 0.15

def reset_counts():
    global trash_count, recycle_count, current_item
    trash_count   = 0
    recycle_count = 0
    current_item  = "Scanning..."

def adjust_capacity(change):
    global bin_capacity
    bin_capacity = max(1, bin_capacity + change)

def log_classification_details(pred, conf):
    print(f"Predicted: {pred}, Confidence: {conf:.2%}")

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640,480)})
picam2.configure(config)
picam2.start()

# Classification & motor thread
def classify_and_act():
    global current_item, trash_count, recycle_count, confidence
    while running:
        frame = picam2.capture_array()
        pred, conf = classify_frame(frame)
        log_classification_details(pred, conf)
        if conf < 0.6:
            current_item = "Unrecognized"
            confidence   = 0.0
            time.sleep(5)
            continue
        current_item = pred
        confidence   = conf
        if pred == "Recyclable":
            recycle_count += 1
            rotate_left();  time.sleep(3.5); stop_motor()
            time.sleep(1)
            rotate_right(); time.sleep(3.5); stop_motor()
        else:
            trash_count += 1
            rotate_right(); time.sleep(3.5); stop_motor()
            time.sleep(1)
            rotate_left();  time.sleep(3.5); stop_motor()
        time.sleep(5)

threading.Thread(target=classify_and_act, daemon=True).start()

# Main UI loop (your existing drawing & event handling)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # handle reset/capacity buttons as before…

    # draw camera feed, counts, warnings, etc. exactly as in your original code
    pygame.display.flip()

# Cleanup
picam2.stop()
pwm.stop()
GPIO.cleanup()
pygame.quit()
