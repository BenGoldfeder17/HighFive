import os
import subprocess
import sys
import torch
import torchvision.transforms as transforms
import numpy as np
import pygame
import time
import threading
from picamera2 import Picamera2
import RPi.GPIO as GPIO

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

# Class names
class_names = ["Recyclable", "Trash"]

# PyTorch model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture matching the .pth file
def load_pretrained_model(model_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(3 * 224 * 224, 10)  # Adjust input/output dimensions as needed
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

# Load the pretrained model
model_path = "garbage_classifier.pth"  # Replace with the actual path to your .pth file
model = load_pretrained_model(model_path)

# Image preprocessing
img_height, img_width = 224, 224
preprocess = transforms.Compose([
    transforms.ToPILImage(mode="RGB"),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
confidence = 0.0
trash_count, recycle_count = 0, 0
bin_capacity = 10
item_volume = 0.15
camera_feed = None

# Function to reset counts
def reset_counts():
    global trash_count, recycle_count, current_item
    trash_count = 0
    recycle_count = 0
    current_item = "Scanning..."

# Function to adjust bin capacity
def adjust_capacity(change):
    global bin_capacity
    bin_capacity = max(1, bin_capacity + change)

# Function to preprocess the frame for classification
def preprocess_frame(frame):
    if frame.ndim == 2:
        frame = np.stack((frame,) * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return preprocess(frame).unsqueeze(0).to(device)

# Function to classify items and control the motor
def classify_and_act():
    global current_item, trash_count, recycle_count, confidence
    while running:
        frame = picam2.capture_array()
        frame = np.ascontiguousarray(frame)
        input_frame = preprocess_frame(frame)
        with torch.no_grad():
            outputs = model(input_frame)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            predicted_class = class_names[predicted.item()]
            if confidence < 0.6:
                current_item = "Unrecognized"
                confidence = 0.0
                time.sleep(5)
                continue
        current_item = predicted_class
        if predicted_class == "Recyclable":
            recycle_count += 1
            rotate_left()
            time.sleep(3.5)
            stop_motor()
        else:
            trash_count += 1
            rotate_right()
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

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if reset_button_x <= mouse_x <= reset_button_x + button_width and reset_button_y <= mouse_y <= reset_button_y + button_height:
                reset_counts()
            elif capacity_x <= mouse_x <= capacity_x + button_width // 2 and capacity_y <= mouse_y <= capacity_y + button_height:
                adjust_capacity(-1)
            elif capacity_x + button_width // 2 <= mouse_x <= capacity_x + button_width and capacity_y <= mouse_y <= capacity_y + button_height:
                adjust_capacity(1)

    # Update display
    screen.fill(WHITE)
    pygame.display.flip()

# Cleanup
picam2.stop()
pwm.stop()
GPIO.cleanup()
pygame.quit()
