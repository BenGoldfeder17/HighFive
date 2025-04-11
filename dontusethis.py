import os
import subprocess
import sys
import time
import threading

# Ensure the script is running with root permissions
if os.geteuid() != 0:
    print("Error: This script must be run as root. Try using 'sudo'.")
    sys.exit(1)

# Ensure required packages are installed
required_packages = ["RPi.GPIO", "pygame", "numpy", "picamera2"]
for package in required_packages:
    try:
        __import__(package.split('-')[0])
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Import required libraries
import RPi.GPIO as GPIO
import pygame
import numpy as np
from picamera2 import Picamera2

# ----------------------------
# Motor Control Setup
# ----------------------------
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

# ----------------------------
# Pygame and Camera Setup
# ----------------------------
pygame.init()
screen_width, screen_height = 720, 1280
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Smart Trash Can")

WHITE, BLACK, LIGHT_BLUE, LIGHT_CORAL = (255,255,255), (0,0,0), (173,216,230), (240,128,128)
font_large = pygame.font.Font(None, 80)
font_medium = pygame.font.Font(None, 60)

# Variables
running = True
current_item = "Scanning..."
trash_count, recycle_count = 0, 0
bin_capacity = 10  # in gallons
item_volume = 0.15  # approximate volume per item

# Reset and capacity adjustment functions
def reset_counts():
    global trash_count, recycle_count, current_item
    trash_count = 0
    recycle_count = 0
    current_item = "Scanning..."

def adjust_capacity(change):
    global bin_capacity
    bin_capacity = max(1, bin_capacity + change)

# ----------------------------
# Camera Initialization and Background Capture
# ----------------------------
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# Allow the camera to stabilize, then capture the background frame
time.sleep(2)
background_frame = picam2.capture_array()
if background_frame.ndim == 2:  # convert grayscale to RGB
    background_frame = np.stack((background_frame,)*3, axis=-1)
elif background_frame.shape[2] == 4:  # drop alpha channel if present
    background_frame = background_frame[:, :, :3]
background_frame = np.ascontiguousarray(background_frame)

# ----------------------------
# Classification Function Using Red Detection & Foreground Extraction
# ----------------------------
def classify_and_act():
    global current_item, trash_count, recycle_count, background_frame
    while running:
        frame = picam2.capture_array()  # Capture frame as a NumPy array
        if frame.ndim == 2:
            frame = np.stack((frame,)*3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = np.ascontiguousarray(frame)
        
        # Compute absolute difference between current frame and background
        diff = np.abs(frame.astype(np.float32) - background_frame.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)
        # Create mask for pixels that differ significantly (threshold set at 20)
        mask = diff_gray > 20
        # Calculate the ratio of changed pixels
        changed_ratio = np.sum(mask) / mask.size
        if changed_ratio < 0.1:
            # Less than 10% pixels changed – no significant foreground.
            current_item = "No Movement"
            print("No significant change detected.")
            time.sleep(5)
            continue
        
        # Compute average red value over the foreground (changed pixels)
        if np.sum(mask) > 0:
            avg_red = np.mean(frame[:,:,0][mask])
        else:
            avg_red = np.mean(frame[:,:,0])
        print(f"Foreground average red: {avg_red}")
        
        # New classification rule using red channel:
        # If the average red value on foreground pixels is over 90, classify as "Recyclable"
        if avg_red > 90:
            predicted_class = "Recyclable"
        else:
            predicted_class = "Trash"
        current_item = predicted_class

        # Motor control based on classification
        if predicted_class == "Recyclable":
            recycle_count += 1
            rotate_left()
            time.sleep(3.5)
            stop_motor()
            time.sleep(1)
            rotate_right()
            time.sleep(3.5)
            stop_motor()
        elif predicted_class == "Trash":
            trash_count += 1
            rotate_right()
            time.sleep(3.5)
            stop_motor()
            time.sleep(1)
            rotate_left()
            time.sleep(3.5)
            stop_motor()

        time.sleep(5)

# Start the classification in a separate thread
threading.Thread(target=classify_and_act, daemon=True).start()

# ----------------------------
# GUI Functions and Main Loop
# ----------------------------
button_width, button_height = 200, 50
reset_button_x, reset_button_y = 50, screen_height - button_height - 150
capacity_x, capacity_y = screen_width - button_width - 50, screen_height - button_height - 150

def is_inside_rect(x, y, rect_x, rect_y, rect_width, rect_height):
    return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height

def display_warning(message):
    warning_surface = font_medium.render(message, True, LIGHT_CORAL)
    screen.blit(warning_surface, (screen_width // 2 - warning_surface.get_width() // 2, screen_height // 2))
    pygame.display.flip()
    time.sleep(3)

# Main GUI loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if is_inside_rect(mouse_x, mouse_y, reset_button_x, reset_button_y, button_width, button_height):
                reset_counts()
            elif is_inside_rect(mouse_x, mouse_y, capacity_x, capacity_y, button_width//2, button_height):
                adjust_capacity(-1)
            elif is_inside_rect(mouse_x, mouse_y, capacity_x + button_width//2, capacity_y, button_width//2, button_height):
                adjust_capacity(1)

    # Calculate fill percentages based on counts and capacity
    recycle_percentage = min((recycle_count * item_volume / bin_capacity) * 100, 100)
    trash_percentage = min((trash_count * item_volume / bin_capacity) * 100, 100)

    # Display warnings if capacities are nearly full
    if recycle_percentage >= 90:
        display_warning("Recycle is full")
    if trash_percentage >= 90:
        display_warning("Trash is full")

    # Obtain latest camera image for display
    frame = picam2.capture_array()
    if frame.ndim == 2:
        frame = np.stack((frame,)*3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame = np.rot90(frame)
    frame = np.ascontiguousarray(frame)
    frame_surface = pygame.surfarray.make_surface(frame)
    camera_feed_x = (screen_width - frame_surface.get_width()) // 2
    camera_feed_y = (screen_height - frame_surface.get_height()) // 2
    screen.fill(WHITE)
    if frame_surface is not None:
        screen.blit(frame_surface, (camera_feed_x, camera_feed_y))
    else:
        fallback_text = font_medium.render("Camera feed unavailable", True, BLACK)
        screen.blit(fallback_text, (screen_width // 2 - fallback_text.get_width() // 2, screen_height // 2))
    
    # Draw text for current classification and counts
    text_surface = font_medium.render(f"Item: {current_item}", True, BLACK)
    screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, 150))
    title_surface = font_large.render("AI Trash", True, BLACK)
    screen.blit(title_surface, (screen_width // 2 - title_surface.get_width() // 2, 50))
    recyc_surface = font_medium.render(f"Recyclable: {recycle_percentage:.1f}%", True, LIGHT_BLUE)
    screen.blit(recyc_surface, (screen_width // 2 - recyc_surface.get_width() // 2, 250))
    trash_surface = font_medium.render(f"Trash: {trash_percentage:.1f}%", True, LIGHT_CORAL)
    screen.blit(trash_surface, (screen_width // 2 - trash_surface.get_width() // 2, 350))
    
    # Draw buttons for reset and capacity control
    pygame.draw.rect(screen, LIGHT_BLUE, (reset_button_x, reset_button_y, button_width, button_height))
    reset_text = font_medium.render("Reset", True, BLACK)
    screen.blit(reset_text, (reset_button_x + button_width//2 - reset_text.get_width()//2, reset_button_y + button_height//2 - reset_text.get_height()//2))
    pygame.draw.rect(screen, LIGHT_BLUE, (capacity_x, capacity_y, button_width, button_height))
    left_arrow = font_medium.render("<", True, BLACK)
    right_arrow = font_medium.render(">", True, BLACK)
    screen.blit(left_arrow, (capacity_x + 10, capacity_y + button_height//2 - left_arrow.get_height()//2))
    screen.blit(right_arrow, (capacity_x + button_width - 30, capacity_y + button_height//2 - right_arrow.get_height()//2))
    capacity_text = font_medium.render(f"{bin_capacity} gal", True, BLACK)
    screen.blit(capacity_text, (capacity_x + button_width//2 - capacity_text.get_width()//2, capacity_y + button_height//2 - capacity_text.get_height()//2))
    
    pygame.display.flip()

# Cleanup resources
picam2.stop()
pwm.stop()
GPIO.cleanup()
pygame.quit()
