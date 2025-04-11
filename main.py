import os
import subprocess
import sys

# Ensure the script is running with root permissions
if os.geteuid() != 0:
    print("Error: This script must be run as root. Try using 'sudo'.")
    sys.exit(1)

# Ensure required packages are installed
required_packages = ["RPi.GPIO", "torch", "torchvision", "pygame", "numpy", "picamera2"]
for package in required_packages:
    try:
        __import__(package.split('-')[0])  # Import the package to check if it's installed
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Import PyTorch and torchvision
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn

# Class names (the order should match how the dataset was mapped in main.py)
class_names = ["Recyclable", "Trash"]

# PyTorch model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a ResNet18 model with 2 output classes (as trained in main.py)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# Load the state_dict saved by main.py (make sure the path is correct)
model.load_state_dict(torch.load("trash_recycling_classifier.pth", map_location=device))
model.eval()
model.to(device)

# Image preprocessing
img_height, img_width = 224, 224  # Ensure input size matches ResNet18's expected input
preprocess = transforms.Compose([
    transforms.ToPILImage(mode="RGB"),  # Ensure the image is in RGB format
    transforms.Resize((img_height, img_width)),  # Resize to 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize as required by ResNet
])

# ---------------------------------------------------------------------
# The remainder of the file (all the GUI functions, motor control, etc.)
# is left unchanged.
# ---------------------------------------------------------------------

import RPi.GPIO as GPIO
import time
import pygame
import threading
import numpy as np
from picamera2 import Picamera2  # Import the Picamera2 library for libcamera

# GPIO setup
try:
    IN1, IN2, ENA = 23, 24, 5
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    pwm = GPIO.PWM(ENA, 1000)
    pwm.start(0)
except Exception as e:
    print(f"GPIO setup failed: {e}")
    sys.exit(1)

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

# Pygame setup
pygame.init()
screen_width, screen_height = 720, 1280
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Smart Trash Can")

# Colors
WHITE, BLACK, LIGHT_BLUE, LIGHT_CORAL = (255, 255, 255), (0, 0, 0), (173, 216, 230), (240, 128, 128)

# Fonts
font_large = pygame.font.Font(None, 80)
font_medium = pygame.font.Font(None, 60)

# Variables
running = True
current_item = "Scanning..."
confidence = 0.0  # Confidence score for the current classification
trash_count, recycle_count = 0, 0
bin_capacity = 10  # Default: 10 gallons
item_volume = 0.15  # Assume each item takes 0.15 gallons
camera_feed = None  # Variable to store the live camera feed

# Function to reset counts
def reset_counts():
    global trash_count, recycle_count, current_item
    trash_count = 0
    recycle_count = 0
    current_item = "Scanning..."

# Function to adjust bin capacity
def adjust_capacity(change):
    global bin_capacity
    bin_capacity = max(1, bin_capacity + change)  # Ensure capacity is at least 1 gallon

# Debugging: Add a function to log classification details
def log_classification_details(probabilities, predicted_class, confidence):
    print(f"Classification Details:")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

# Function to preprocess the frame for classification
def preprocess_frame(frame):
    if frame.ndim == 2:  # If grayscale, convert to RGB
        frame = np.stack((frame,) * 3, axis=-1)
    elif frame.shape[2] == 4:  # If RGBA, convert to RGB
        frame = frame[:, :, :3]
    return preprocess(frame).unsqueeze(0).to(device)

# Function to classify items and control the motor
def classify_and_act():
    global current_item, trash_count, recycle_count, confidence
    while running:
        frame = picam2.capture_array()  # Capture a frame as a NumPy array
        frame = np.ascontiguousarray(frame)  # Ensure the frame is contiguous in memory
        input_frame = preprocess_frame(frame)
        with torch.no_grad():
            outputs = model(input_frame)
            if outputs.size(1) != len(class_names):  # Check if model output matches class names
                print(f"Error: Model output size {outputs.size(1)} does not match number of class names {len(class_names)}.")
                sys.exit(1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Get probabilities
            confidence, predicted = torch.max(probabilities, 0)  # Get the highest confidence score
            predicted_class = class_names[predicted.item()]
            log_classification_details(probabilities.cpu().numpy(), predicted_class, confidence.item())
            if confidence < 0.6:  # If confidence is below 60%, do not classify
                current_item = "Unrecognized"
                confidence = 0.0
                print("Unrecognized item. Skipping classification.")
                time.sleep(5)
                continue
        current_item = predicted_class
        if predicted_class == "Recyclable":
            recycle_count += 1
            rotate_left()
            time.sleep(3.5)
            stop_motor()
            time.sleep(1)
            rotate_right()
            time.sleep(3.5)
            stop_motor()
        else:
            trash_count += 1
            rotate_right()
            time.sleep(3.5)
            stop_motor()
            time.sleep(1)
            rotate_left()
            time.sleep(3.5)
            stop_motor()
        time.sleep(5)

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# Start the classification thread
threading.Thread(target=classify_and_act, daemon=True).start()

# ---------------------------------------------------------------------
# GUI function definitions continue here (do not change any of these functions)
# ---------------------------------------------------------------------
# Button dimensions and positions, event handling, drawing, etc.
# (The following code remains unchanged)

# Button dimensions and positions
button_width, button_height = 200, 50
reset_button_x, reset_button_y = 50, screen_height - button_height - 150
capacity_x, capacity_y = screen_width - button_width - 50, screen_height - button_height - 150
camera_feed_width, camera_feed_height = screen_width // 3, screen_height // 3
camera_feed_x, camera_feed_y = (screen_width - camera_feed_width) // 2, screen_height - camera_feed_height - 50

def is_inside_rect(x, y, rect_x, rect_y, rect_width, rect_height):
    return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height

def display_warning(message):
    warning_surface = font_medium.render(message, True, LIGHT_CORAL)
    screen.blit(warning_surface, (screen_width // 2 - warning_surface.get_width() // 2, screen_height // 2))
    pygame.display.flip()
    time.sleep(3)

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.eval()
detection_model.to(device)

# COCO class names and mapping functions (omitted for brevity)
# ... [Additional object detection functions remain unchanged]

# Main GUI loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if is_inside_rect(mouse_x, mouse_y, reset_button_x, reset_button_y, button_width, button_height):
                reset_counts()
            elif is_inside_rect(mouse_x, mouse_y, capacity_x, capacity_y, button_width // 2, button_height):
                adjust_capacity(-1)
            elif is_inside_rect(mouse_x, mouse_y, capacity_x + button_width // 2, capacity_y, button_width // 2, button_height):
                adjust_capacity(1)

    recycle_percentage = min((recycle_count * item_volume / bin_capacity) * 100, 100)
    trash_percentage = min((trash_count * item_volume / bin_capacity) * 100, 100)

    if recycle_percentage >= 90:
        display_warning("Recycle is full")
    if trash_percentage >= 90:
        display_warning("Trash is full")

    frame = picam2.capture_array()
    if frame.ndim == 2:
        frame = np.stack((frame,) * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame = np.rot90(frame)
    frame = np.ascontiguousarray(frame)
    frame_surface = pygame.surfarray.make_surface(frame)
    camera_feed_x = (screen_width - frame_surface.get_width()) // 2
    camera_feed_y = (screen_height - frame_surface.get_height()) // 2
    screen.fill(WHITE)
    if frame_surface:
        screen.blit(frame_surface, (camera_feed_x, camera_feed_y))
    else:
        fallback_text = font_medium.render("Camera feed unavailable", True, BLACK)
        screen.blit(fallback_text, (screen_width // 2 - fallback_text.get_width() // 2, screen_height // 2))
    text_surface = font_medium.render(f"Item: {current_item}", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 150))
    text_surface = font_large.render(f"AI Trash", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 50))
    text_surface = font_medium.render(f"Recyclable: {recycle_percentage:.1f}%", True, LIGHT_BLUE)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 250))
    text_surface = font_medium.render(f"Trash: {trash_percentage:.1f}%", True, LIGHT_CORAL)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 350))
    pygame.draw.rect(screen, LIGHT_BLUE, (reset_button_x, reset_button_y, button_width, button_height))
    text_surface = font_medium.render("Reset", True, BLACK)
    screen.blit(text_surface, (reset_button_x + button_width // 2 - text_surface.get_width() // 2, reset_button_y + button_height // 2 - text_surface.get_height() // 2))
    pygame.draw.rect(screen, LIGHT_BLUE, (capacity_x, capacity_y, button_width, button_height))
    left_arrow = font_medium.render("<", True, BLACK)
    right_arrow = font_medium.render(">", True, BLACK)
    screen.blit(left_arrow, (capacity_x + 10, capacity_y + button_height // 2 - left_arrow.get_height() // 2))
    screen.blit(right_arrow, (capacity_x + button_width - 30, capacity_y + button_height // 2 - right_arrow.get_height() // 2))
    text_surface = font_medium.render(f"{bin_capacity} gal", True, BLACK)
    screen.blit(text_surface, (capacity_x + button_width // 2 - text_surface.get_width() // 2, capacity_y + button_height // 2 - text_surface.get_height() // 2))
    pygame.display.flip()

# Cleanup
picam2.stop()
pwm.stop()
GPIO.cleanup()
pygame.quit()
