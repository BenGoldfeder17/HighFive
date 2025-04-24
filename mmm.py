#!/home/highfive/Downloads/highfive/Downloads/bin/python3.11
import os

# On a 64-bit Pi 5 the MMIO base is at 0xFE000000
os.environ["BCM2835_PERI_BASE"] = "0xFE000000"
os.environ["BCM2708_PERI_BASE"] = "0xFE000000"

import time
import threading
import tensorflow as tf
import numpy as np
import RPi.GPIO as GPIO
import pygame
from picamera2 import Picamera2

# ---------------------------------------------------------------------
# 1) Keras + model setup
# ---------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "trash_recycling_model2.keras")
model       = tf.keras.models.load_model(MODEL_PATH)

# Raw class labels from the model
class_names = ["biodegradable", "cardboard", "glass", "metal",
               "paper", "plastic", "trash"]

# Binning map: which labels go to recycling vs trash
recycle_classes = {"cardboard", "glass", "metal", "paper"}
trash_classes   = {"biodegradable", "plastic", "trash"}

IMG_H, IMG_W  = 150, 150
ITEM_WEIGHT   = 0.125   # lbs per item
DIFF_THRESH   = 5       # threshold for object presence

def preprocess_frame(frame: np.ndarray) -> tf.Tensor:
    if frame.ndim == 2:
        frame = np.stack((frame,)*3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    img = tf.convert_to_tensor(frame, dtype=tf.float32) / 255.0
    img = tf.image.resize(img, [IMG_H, IMG_W])
    return tf.expand_dims(img, axis=0)

def classify_frame(frame: np.ndarray):
    inp   = preprocess_frame(frame)
    probs = model(inp, training=False)[0].numpy()
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    label = class_names[idx] if conf >= 0.6 else "unknown"
    return label, conf

# ---------------------------------------------------------------------
# 2) GPIO & motor setup
# ---------------------------------------------------------------------
IN1, IN2, ENA = 23, 24, 5
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT); GPIO.setup(IN2, GPIO.OUT); GPIO.setup(ENA, GPIO.OUT)
pwm = GPIO.PWM(ENA, 1000); pwm.start(0)

def rotate_left(speed=100):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW); pwm.ChangeDutyCycle(speed)
def rotate_right(speed=100):
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH); pwm.ChangeDutyCycle(speed)
def stop_motor():
    pwm.ChangeDutyCycle(0); GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)

# ---------------------------------------------------------------------
# 3) Picamera2 setup + capture background
# ---------------------------------------------------------------------
picam2 = Picamera2()
cfg    = picam2.create_preview_configuration(main={"size": (640,480)})
picam2.configure(cfg)
picam2.start()
time.sleep(2)  # allow exposure to settle
background = picam2.capture_array().astype(np.int16)

# ---------------------------------------------------------------------
# 4) Shared state & classification thread
# ---------------------------------------------------------------------
running         = True
display_label   = "Waiting for Object"  # will show “Trash (Plastic)”, etc.
confidence      = 0.0
trash_weight    = 0.0
recycle_weight  = 0.0
bin_capacity    = 10.0   # gallons

def classify_and_act():
    global display_label, confidence, trash_weight, recycle_weight
    while running:
        frame    = picam2.capture_array()
        diff_val = np.mean(np.abs(frame.astype(np.int16) - background))
        if diff_val < DIFF_THRESH:
            display_label = "Waiting for Object"
            confidence    = 0.0
        else:
            # mask unchanged pixels white
            diff_map = np.mean(np.abs(frame.astype(np.int16) - background), axis=2)
            mask     = diff_map >= DIFF_THRESH
            proc     = frame.copy()
            proc[~mask] = 255

            time.sleep(2)  # wait before identifying
            raw_label, conf = classify_frame(proc)
            confidence       = conf

            # decide bin and execute motor cycle
            if raw_label in recycle_classes:
                display_label   = f"Recycle ({raw_label.capitalize()})"
                recycle_weight += ITEM_WEIGHT
                rotate_left();  time.sleep(3.5); stop_motor()
                time.sleep(1)
                rotate_right(); time.sleep(3.5); stop_motor()
            elif raw_label in trash_classes:
                display_label = f"Trash ({raw_label.capitalize()})"
                trash_weight  += ITEM_WEIGHT
                rotate_right(); time.sleep(3.5); stop_motor()
                time.sleep(1)
                rotate_left();  time.sleep(3.5); stop_motor()
            else:
                display_label = f"Unknown ({raw_label.capitalize()})"
                stop_motor()

        time.sleep(5)

threading.Thread(target=classify_and_act, daemon=True).start()

# ---------------------------------------------------------------------
# 5) Pygame UI setup
# ---------------------------------------------------------------------
pygame.init()
SCREEN_W, SCREEN_H = 720, 1280
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("AI Trash")

WHITE       = (255,255,255)
BLACK       = (0,0,0)
LIGHT_BLUE  = (173,216,230)  # recycling
LIGHT_CORAL = (240,128,128)  # trash

font_title  = pygame.font.Font(None, 100)
font_text   = pygame.font.Font(None, 60)
font_button = pygame.font.Font(None, 50)

# Button rects (slightly higher)
btn_h = 80; btn_w = 220
btn_y = SCREEN_H - btn_h - 100
reset_rect = pygame.Rect(60, btn_y, btn_w, btn_h)
cap_rect   = pygame.Rect(SCREEN_W - btn_w - 60, btn_y, btn_w, btn_h)

# ---------------------------------------------------------------------
# 6) Main UI loop
# ---------------------------------------------------------------------
try:
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                x,y = ev.pos
                if reset_rect.collidepoint(x,y):
                    trash_weight   = recycle_weight = 0.0
                    display_label  = "Waiting for Object"
                elif cap_rect.collidepoint(x,y):
                    if x < cap_rect.centerx:
                        bin_capacity = max(1.0, bin_capacity - 0.5)
                    else:
                        bin_capacity += 0.5

        screen.fill(WHITE)

        # Title
        title = font_title.render("AI Trash", True, BLACK)
        screen.blit(title, ((SCREEN_W - title.get_width())//2, 20))

        # Current classification
        cur = font_text.render(display_label, True, BLACK)
        screen.blit(cur, ((SCREEN_W - cur.get_width())//2, 140))

        # Percentages
        pct_rec = min(recycle_weight / bin_capacity, 1.0)
        pct_tr  = min(trash_weight   / bin_capacity, 1.0)
        rec = font_text.render(f"Recycling: {pct_rec:.1%}", True, LIGHT_BLUE)
        tr  = font_text.render(f"Trash:     {pct_tr:.1%}", True, LIGHT_CORAL)
        screen.blit(rec, ((SCREEN_W - rec.get_width())//2, 220))
        screen.blit(tr,  ((SCREEN_W - tr.get_width())//2, 300))

        # Camera feed
        frame = picam2.capture_array()
        if frame.ndim == 2:
            frame = np.stack((frame,)*3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:,:,:3]
        h,w  = frame.shape[:2]
        surf = pygame.image.frombuffer(frame.tobytes(), (w,h), 'RGB')
        fw   = SCREEN_W - 120
        fh   = int(fw*h / w)
        feed = pygame.transform.scale(surf, (fw, fh))
        screen.blit(feed, ((SCREEN_W - fw)//2, 380))

        # Reset button
        pygame.draw.rect(screen, LIGHT_BLUE, reset_rect)
        rs = font_button.render("Reset", True, BLACK)
        screen.blit(rs, (reset_rect.x + (btn_w - rs.get_width())//2,
                         reset_rect.y + (btn_h - rs.get_height())//2))

        # Capacity slider
        pygame.draw.rect(screen, LIGHT_BLUE, cap_rect)
        ct = font_button.render(f"< {bin_capacity:.1f} gal >", True, BLACK)
        screen.blit(ct, (cap_rect.x + (btn_w - ct.get_width())//2,
                         cap_rect.y + (btn_h - ct.get_height())//2))

        pygame.display.flip()

finally:
    picam2.stop()
    pwm.stop()
    GPIO.cleanup()
    pygame.quit()
