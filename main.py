import os
import subprocess
import sys

# Ensure the script is running with root permissions
if os.geteuid() != 0:
    print("Error: This script must be run as root. Try using 'sudo'.")
    sys.exit(1)

# Ensure required packages are installed
required_packages = ["RPi.GPIO", "torch", "torchvision", "pygame"]
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

# PyTorch model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.eval()
model.to(device)

# Class names (replace with your specific classes if needed)
class_names = ["Recyclable", "Trash"]

# Image preprocessing
img_height, img_width = 180, 180
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_frame(frame):
    return preprocess(frame).unsqueeze(0).to(device)

import RPi.GPIO as GPIO
import time
import cv2
import pygame
import threading

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

# Function to classify items and control the motor
def classify_and_act():
    global current_item, trash_count, recycle_count
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        input_frame = preprocess_frame(frame)
        with torch.no_grad():
            outputs = model(input_frame)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
        current_item = predicted_class
        if predicted_class == "Recyclable":
            recycle_count += 1
            rotate_left()
        else:
            trash_count += 1
            rotate_right()
        time.sleep(3)
        stop_motor()
    cap.release()

# Start the classification thread
threading.Thread(target=classify_and_act, daemon=True).start()

# Button dimensions and position
button_width, button_height = 200, 50
button_x = screen_width // 2 - button_width // 2
button_y = 600

# Function to check if a point is inside the button
def is_inside_button(x, y):
    return button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height

# Main loop
cap = cv2.VideoCapture(0)  # Initialize camera
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:  # Check for mouse click
            mouse_x, mouse_y = event.pos
            if is_inside_button(mouse_x, mouse_y):  # If click is inside the reset button
                reset_counts()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  # Decrease capacity on left arrow
                adjust_capacity(-1)
            elif event.key == pygame.K_RIGHT:  # Increase capacity on right arrow
                adjust_capacity(1)

    # Calculate percentages
    recycle_percentage = min((recycle_count * item_volume / bin_capacity) * 100, 100)
    trash_percentage = min((trash_count * item_volume / bin_capacity) * 100, 100)

    # Capture live camera feed
    ret, frame = cap.read()
    if ret:
        camera_feed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Pygame
        camera_feed = cv2.resize(camera_feed, (screen_width // 2, screen_height // 2))
        camera_feed = pygame.surfarray.make_surface(camera_feed)

    # Clear screen
    screen.fill(WHITE)

    # Render labels
    text_surface = font_large.render(f"Smart Trash Can", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 50))
    text_surface = font_medium.render(f"Item: {current_item}", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 150))
    text_surface = font_medium.render(f"Recyclable: {recycle_percentage:.1f}%", True, LIGHT_BLUE)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 250))
    text_surface = font_medium.render(f"Trash: {trash_percentage:.1f}%", True, LIGHT_CORAL)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 350))

    # Render capacity adjustment
    text_surface = font_medium.render(f"Capacity: {bin_capacity} gallons", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 450))
    left_arrow = font_medium.render("<", True, BLACK)
    right_arrow = font_medium.render(">", True, BLACK)
    screen.blit(left_arrow, (screen_width // 2 - 100, 450))
    screen.blit(right_arrow, (screen_width // 2 + 100, 450))

    # Render reset button
    pygame.draw.rect(screen, LIGHT_BLUE, (button_x, button_y, button_width, button_height))
    text_surface = font_medium.render("Reset", True, BLACK)
    screen.blit(text_surface, (button_x + button_width // 2 - text_surface.get_width() // 2, button_y + button_height // 2 - text_surface.get_height() // 2))

    # Render live camera feed
    if camera_feed:
        screen.blit(camera_feed, (screen_width // 4, screen_height // 2))

    # Update display
    pygame.display.flip()

# Cleanup
cap.release()  # Release camera
pwm.stop()
GPIO.cleanup()
pygame.quit()
