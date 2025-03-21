import os
import subprocess
import sys

# Ensure required packages are installed
required_packages = ["RPi.GPIO", "tensorflow", "opencv-python", "pygame"]
for package in required_packages:
    try:
        __import__(package.split('-')[0])  # Import the package to check if it's installed
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import RPi.GPIO as GPIO
import time
import tensorflow as tf
import cv2
import pygame
import threading

# GPIO setup
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

# TensorFlow model setup
model = tf.keras.models.load_model("recycling_model.h5")
class_names = ["Recyclable", "Trash"]
img_height, img_width = 180, 180

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (img_width, img_height))
    frame_normalized = frame_resized / 255.0
    return tf.expand_dims(frame_normalized, axis=0)

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

# Function to classify items and control the motor
def classify_and_act():
    global current_item, trash_count, recycle_count
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        input_frame = preprocess_frame(frame)
        predictions = model.predict(input_frame)
        predicted_class = class_names[tf.argmax(predictions[0])]
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

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate percentages
    recycle_percentage = min((recycle_count * item_volume / bin_capacity) * 100, 100)
    trash_percentage = min((trash_count * item_volume / bin_capacity) * 100, 100)

    # Clear screen
    screen.fill(WHITE)

    # Render labels
    text_surface = font_large.render(f"Smart Trash Can", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 100))
    text_surface = font_medium.render(f"Item: {current_item}", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 250))
    text_surface = font_medium.render(f"Recyclable: {recycle_percentage:.1f}%", True, LIGHT_BLUE)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 400))
    text_surface = font_medium.render(f"Trash: {trash_percentage:.1f}%", True, LIGHT_CORAL)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 500))

    # Update display
    pygame.display.flip()

# Cleanup
pwm.stop()
GPIO.cleanup()
pygame.quit()
