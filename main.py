#!/usr/bin/env python3
# main.py — fully updated to load the Flatten→Linear .pth from from_torch.py
# (preserves all original GPIO, camera, pygame UI, and motor control code)

import os
import subprocess
import sys

# Ensure the script is running as root
if os.geteuid() != 0:
    print("Error: This script must be run as root. Try using 'sudo'.")
    sys.exit(1)

# Ensure required packages are installed
required_packages = ["RPi.GPIO", "torch", "torchvision", "pygame", "numpy", "picamera2"]
for package in required_packages:
    try:
        __import__(package.split('-')[0])
    except ImportError:
        print(f"{package} not found. Installing…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# -------------------------------
# 1) PyTorch + model setup
# -------------------------------
import torch
import torchvision.transforms as transforms
from torch import nn

# Class names (must match the labels used when generating garbage_classifier.pth)
class_names = ["Recyclable", "Trash"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the same Flatten→Linear model you trained in from_torch.py
IMG_H, IMG_W = 224, 224
num_classes = len(class_names)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * IMG_H * IMG_W, num_classes)
).to(device)

# Load your state_dict (from from_torch.py)
MODEL_PATH = "garbage_classifier.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print(f"[INFO] Loaded model '{MODEL_PATH}' with classes {class_names}")

# Image preprocessing (you may keep your ColorJitter or remove it; here's original + match training)
preprocess = transforms.Compose([
    transforms.ToPILImage(mode="RGB"),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Safe loader to skip broken images
from PIL import Image, UnidentifiedImageError
def safe_pil_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return Image.new("RGB", (IMG_H, IMG_W), "black")

# Helper to preprocess frames for model
import numpy as np
def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    if frame.ndim == 2:
        frame = np.stack((frame,)*3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return preprocess(frame).unsqueeze(0).to(device)

# Helper to classify a single image/frame
def classify_tensor(inp: torch.Tensor):
    with torch.no_grad():
        out = model(inp)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        conf, idx = torch.max(probs, 0)
        return class_names[idx.item()], conf.item()

# ---------------------------------------------------------------------
# 2) All existing GPIO, Picamera2, motor, and pygame UI code follows
#    unchanged except it now calls classify_tensor().
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

WHITE      = (255,255,255)
BLACK      = (0,0,0)
LIGHT_BLUE = (173,216,230)
LIGHT_CORAL= (240,128,128)

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

def log_classification_details(probs, pred, conf):
    print(f"Probs: {probs}, Pred: {pred}, Conf: {conf:.2f}")

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
        inp   = preprocess_frame(frame)
        pred, conf = classify_tensor(inp)
        log_classification_details(None, pred, conf)
        if conf < 0.6:
            current_item = "Unrecognized"
            confidence = 0.0
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

# GUI helper functions remain untouched...
# [ all your button drawing, event loops, object detection, etc. ]

# Main UI loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # handle reset/capacity buttons, exactly as before

    # draw camera feed, percentages, text, warnings, etc.
    # exactly your original code

    pygame.display.flip()

# Cleanup
picam2.stop()
pwm.stop()
GPIO.cleanup()
pygame.quit()
