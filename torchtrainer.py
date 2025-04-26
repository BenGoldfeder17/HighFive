#!/usr/bin/env python3
import os
import sys
import time
import io
import threading

import numpy as np
import cv2
import RPi.GPIO as GPIO
import pygame
from picamera2 import Picamera2
from PIL import Image
from flask import Flask, send_file
from werkzeug.serving import make_server
import openai

# ─── Helper: extract foreground by background subtraction ────────────────
def extract_foreground(frame: np.ndarray,
                       background: np.ndarray,
                       diff_thresh: int = 30,
                       blur_ksize: int = 5,
                       morph_ksize: int = 5) -> np.ndarray:
    # 1) abs diff & grayscale
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 2) blur & threshold
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(blurred, diff_thresh, 255, cv2.THRESH_BINARY)
    # 3) clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 4) apply
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(frame, mask_3c)

# ─── 0) Prompt for Project‐Scoped Key & Project ID ───────────────────────
OPENAI_KEY     = input("Enter your OpenAI Project API key (sk-proj-…): ").strip()
if not OPENAI_KEY.startswith("sk-proj-"):
    print("[ERROR] Only project‐scoped keys (sk-proj-…) accepted."); sys.exit(1)
OPENAI_PROJECT = input("Enter your OpenAI project ID: ").strip()

openai.api_key = OPENAI_KEY
openai._custom_headers = {"OpenAI-Project": OPENAI_PROJECT}

# ─── 1) Class info & thresholds ─────────────────────────────────────────
CLASS_NAMES    = ["biodegradable","cardboard","glass","metal","paper","plastic","trash"]
RECYCLE_SET    = {"cardboard","glass","metal","paper"}
TRASH_SET      = {"biodegradable","plastic","trash"}

DETECT_THRESH  = 5       # raw diff threshold for motion detection
FG_DIFF_THRESH = 30      # threshold inside extract_foreground
FG_BLUR_KSIZE  = 5       # blur kernel for mask
FG_MORPH_KSIZE = 5       # morph kernel for mask
PAUSE_BEFORE   = 2       # delay before classify
ITEM_WEIGHT    = 0.125   # lbs per item
BIN_CAP        = 10.0    # gallons capacity

# ─── 2) GPIO & motor setup ──────────────────────────────────────────────
IN1, IN2, ENA = 23, 24, 5
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
pwm = GPIO.PWM(ENA, 1000)
pwm.start(0)

def rotate_left(speed=100):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)

def rotate_right(speed=100):
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed)

def stop_motor():
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)

# ─── 3) Picamera2 init & background capture ─────────────────────────────
os.environ["BCM2835_PERI_BASE"] = "0xFE000000"
os.environ["BCM2708_PERI_BASE"] = "0xFE000000"

picam2 = Picamera2()
cfg    = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(cfg)
picam2.start()
time.sleep(2)
bg = picam2.capture_array()
# ensure uint8 for cv2 functions
background = bg.astype(np.uint8)

# ─── 4) Flask server to serve masked foreground image ───────────────────
app = Flask(__name__)

@app.route("/latest.jpg")
def latest_jpg():
    frame = picam2.capture_array().astype(np.uint8)
    fg    = extract_foreground(frame, background,
                               diff_thresh=FG_DIFF_THRESH,
                               blur_ksize=FG_BLUR_KSIZE,
                               morph_ksize=FG_MORPH_KSIZE)
    buf   = io.BytesIO()
    Image.fromarray(fg).save(buf, format="JPEG", quality=70)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

class ServerThread(threading.Thread):
    def __init__(self, app, host="0.0.0.0", port=5000):
        super().__init__(daemon=True)
        self.server = make_server(host, port, app)
    def run(self):
        self.server.serve_forever()
    def shutdown(self):
        self.server.shutdown()

flask_thread = ServerThread(app)
flask_thread.start()
IMAGE_ENDPOINT = "http://<YOUR_PUBLIC_URL>/latest.jpg"
print(f"[INFO] Serving masked image at {IMAGE_ENDPOINT}")

# ─── 5) GPT-4 Vision classification via image URL ───────────────────────
def classify_with_url():
    resp = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type":"text",
                 "text":"Classify this image as exactly one of: "
                        "biodegradable, cardboard, glass, metal, paper, plastic, trash."},
                {"type":"image_url",
                 "image_url":{"url": IMAGE_ENDPOINT}}
            ]
        }],
        temperature=0
    )
    return resp.choices[0].message.content.strip().lower()

# ─── 6) Shared state & classification thread ────────────────────────────
running        = True
display_label  = "Waiting for Object"
trash_weight   = 0.0
recycle_weight = 0.0

def classify_and_act():
    global display_label, trash_weight, recycle_weight
    while running:
        frame = picam2.capture_array().astype(np.uint8)
        # detect motion on raw frame
        diff  = np.mean(np.abs(frame.astype(np.int16) - background.astype(np.int16)))
        if diff < DETECT_THRESH:
            display_label = "Waiting for Object"
        else:
            time.sleep(PAUSE_BEFORE)
            label = classify_with_url()
            print(f"[GPT] → {label}")
            if label in RECYCLE_SET:
                display_label   = f"Recycle ({label})"
                recycle_weight += ITEM_WEIGHT
                rotate_left();  time.sleep(3.5); stop_motor()
                time.sleep(1)
                rotate_right(); time.sleep(3.5); stop_motor()
            elif label in TRASH_SET:
                display_label  = f"Trash ({label})"
                trash_weight   += ITEM_WEIGHT
                rotate_right(); time.sleep(3.5); stop_motor()
                time.sleep(1)
                rotate_left();  time.sleep(3.5); stop_motor()
            else:
                display_label = f"Unknown ({label})"
                stop_motor()
        time.sleep(5)

threading.Thread(target=classify_and_act, daemon=True).start()

# ─── 7) Pygame UI loop (showing masked foreground) ──────────────────────
pygame.init()
SCREEN_W, SCREEN_H = 720, 1280
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("AI Trash")

WHITE       = (255,255,255)
BLACK       = (0,0,0)
LIGHT_BLUE  = (173,216,230)
LIGHT_CORAL = (240,128,128)

font_title  = pygame.font.Font(None, 100)
font_text   = pygame.font.Font(None, 60)
font_button = pygame.font.Font(None, 50)

btn_h, btn_w = 80, 220
btn_y        = SCREEN_H - btn_h - 100
reset_rect   = pygame.Rect(60, btn_y, btn_w, btn_h)
cap_rect     = pygame.Rect(SCREEN_W - btn_w - 60, btn_y, btn_w, btn_h)

try:
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                x,y = ev.pos
                if reset_rect.collidepoint(x,y):
                    trash_weight = recycle_weight = 0.0
                    display_label = "Waiting for Object"
                elif cap_rect.collidepoint(x,y):
                    # adjust capacity
                    new_cap = BIN_CAP + (0.5 if x>cap_rect.centerx else -0.5)
                    BIN_CAP = max(1.0, new_cap)

        screen.fill(WHITE)
        screen.blit(font_title.render("AI Trash", True, BLACK),
                    ((SCREEN_W - font_title.size("AI Trash")[0])//2, 20))
        screen.blit(font_text.render(display_label, True, BLACK),
                    ((SCREEN_W - font_text.size(display_label)[0])//2, 140))

        pct_r = min(recycle_weight / BIN_CAP, 1.0)
        pct_t = min(trash_weight   / BIN_CAP, 1.0)
        screen.blit(font_text.render(f"Recycling: {pct_r:.1%}", True, LIGHT_BLUE),
                    ((SCREEN_W - font_text.size("Recycling:")[0])//2, 220))
        screen.blit(font_text.render(f"Trash:     {pct_t:.1%}", True, LIGHT_CORAL),
                    ((SCREEN_W - font_text.size("Trash:")[0])//2, 300))

        # grab and display the masked foreground
        raw = picam2.capture_array().astype(np.uint8)
        fg  = extract_foreground(raw, background,
                                 diff_thresh=FG_DIFF_THRESH,
                                 blur_ksize=FG_BLUR_KSIZE,
                                 morph_ksize=FG_MORPH_KSIZE)
        # ensure RGB for pygame
        if fg.ndim == 2:
            fg = np.stack((fg,)*3, axis=-1)
        elif fg.shape[2] == 4:
            fg = fg[:,:,:3]

        h,w  = fg.shape[:2]
        surf = pygame.image.frombuffer(fg.tobytes(), (w,h), 'RGB')
        fw   = SCREEN_W - 120
        fh   = int(fw * h / w)
        feed = pygame.transform.scale(surf, (fw, fh))
        screen.blit(feed, ((SCREEN_W - fw)//2, 380))

        pygame.draw.rect(screen, LIGHT_BLUE, reset_rect)
        rs = font_button.render("Reset", True, BLACK)
        screen.blit(rs, (reset_rect.x + (btn_w - rs.get_width())//2,
                         reset_rect.y + (btn_h - rs.get_height())//2))

        pygame.draw.rect(screen, LIGHT_BLUE, cap_rect)
        ct = font_button.render(f"< {BIN_CAP:.1f} gal >", True, BLACK)
        screen.blit(ct, (cap_rect.x + (btn_w - ct.get_width())//2,
                         cap_rect.y + (btn_h - ct.get_height())//2))

        pygame.display.flip()

finally:
    picam2.stop()
    pwm.stop()
    GPIO.cleanup()
    pygame.quit()
    flask_thread.shutdown()
