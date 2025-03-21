import tensorflow as tf
import keras
from keras import layers
import os
import pathlib
import subprocess
import sys
import cv2
import pygame
import threading
import time
import RPi.GPIO as GPIO

# Ensure required packages are installed in a virtual environment
def install_packages():
    required_packages = [
        "tensorflow", "keras", "opencv-python", "pygame", "RPi.GPIO"
    ]
    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Call the package installer
install_packages()

# Initialize Pygame
pygame.init()

# Screen dimensions and setup
screen_width, screen_height = 720, 1280
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Smart Trash Can")

# Colors
WHITE = (255, 255, 255)
LIGHT_BLUE = (173, 216, 230)
LIGHT_CORAL = (240, 128, 128)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Fonts
font_large = pygame.font.Font(None, 80)
font_medium = pygame.font.Font(None, 60)
font_small = pygame.font.Font(None, 40)

# GPIO setup for motor control
IN1 = 23
IN2 = 24
ENA = 5
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
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

# Load the trained model
model = keras.models.load_model("recycling_model.h5")

# Parameters
img_height, img_width = 180, 180
class_names = ["Recyclable", "Trash"]  # Replace with actual class names if available

# Function to preprocess frames
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (img_width, img_height))
    frame_normalized = frame_resized / 255.0
    return tf.expand_dims(frame_normalized, axis=0)

# Variables
current_item = "Scanning..."
recycle_count = 0
trash_count = 0
running = True

# Function to classify items and control the motor
def classify_and_control():
    global current_item, recycle_count, trash_count
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        sys.exit(1)

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Preprocess and classify the frame
        input_frame = preprocess_frame(frame)
        predictions = model.predict(input_frame)
        predicted_class = class_names[tf.argmax(predictions[0])]
        current_item = predicted_class

        # Update counts and control motor
        if predicted_class == "Recyclable":
            recycle_count += 1
            rotate_left()
            time.sleep(3.5)
            stop_motor()
        elif predicted_class == "Trash":
            trash_count += 1
            rotate_right()
            time.sleep(3.5)
            stop_motor()

        time.sleep(1)  # Pause before next classification

    cap.release()

# Function to render text centered on the screen
def render_text_centered(text, font, color, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(screen_width // 2, y))
    screen.blit(text_surface, text_rect)

# Start the classification thread
threading.Thread(target=classify_and_control, daemon=True).start()

# Main loop for Pygame interface
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill(WHITE)

    # Render labels
    render_text_centered(f"Smart Trash Can", font_large, BLACK, 100)
    render_text_centered(f"Item: {current_item}", font_medium, GRAY, 250)
    render_text_centered(f"Recyclable Count: {recycle_count}", font_medium, LIGHT_BLUE, 400)
    render_text_centered(f"Trash Count: {trash_count}", font_medium, LIGHT_CORAL, 500)

    # Update display
    pygame.display.flip()

# Cleanup
pygame.quit()
pwm.stop()
GPIO.cleanup()
