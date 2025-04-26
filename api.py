#!/home/highfive/Downloads/highfive/Downloads/bin/python3.11
import os
import sys
import time
import threading
import base64
import io

import numpy as np
import RPi.GPIO as GPIO
import pygame
from picamera2 import Picamera2
from PIL import Image
import openai

# ─── 0) OpenAI QuickStart setup ──────────────────────────────────────────
OPENAI_KEY = input("Enter your OpenAI Project API key (must start with sk-proj-): ").strip()
if not OPENAI_KEY.startswith("sk-proj-"):
    print("[ERROR] This script only accepts project-scoped API keys (sk-proj-...).")
    sys.exit(1)

OPENAI_PROJECT = input("Enter your OpenAI project ID: ").strip()

# Set API key and project ID as custom header
openai.api_key = OPENAI_KEY
openai._custom_headers = {
    "OpenAI-Project": OPENAI_PROJECT
}

# ─── 1) Categories & bin mapping ───────────────────────────────────────────
CLASS_NAMES    = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic", "trash"]
RECYCLE_SET    = {"cardboard", "glass", "metal", "paper","recycling"}
TRASH_SET      = {"biodegradable", "plastic", "trash"}

# ─── 2) Constants ──────────────────────────────────────────────────────────
ITEM_WEIGHT    = 0.125   # lbs per detected item
DIFF_THRESH    = 5       # pixel‐difference threshold
PAUSE_BEFORE   = 2       # seconds to wait once motion detected
BIN_CAP        = 10.0    # default capacity (gallons)

# ─── 3) GPIO & motor setup ────────────────────────────────────────────────
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
    pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

# ─── 4) Camera init & background capture ────────────────────────────────────
os.environ["BCM2835_PERI_BASE"] = "0xFE000000"
os.environ["BCM2708_PERI_BASE"] = "0xFE000000"

picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"size": (640,480)})
picam2.configure(cfg)
picam2.start()
time.sleep(2)
background = picam2.capture_array().astype(np.int16)

# ─── 5) OpenAI‐based classify function (resized, JPEG, sanitized + logging) ─
def classify_with_openai(frame: np.ndarray) -> (str, float):
    global CLASS_NAMES

    # 1) shrink to 224×224 & convert to RGB
    pil = Image.fromarray(frame).resize((224,224), Image.LANCZOS)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    # 2) JPEG-compress
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=20)
    b64 = base64.b64encode(buf.getvalue()).decode()

    # 3) prompt model to reply with exactly one category word
    prompt = (
        "Classify the object on the platform image into exactly one of these words:\n"
        "biodegradable, cardboard, glass, metal, paper, plastic\n"
        "Reply with exactly one word (no punctuation).\n\n"
        "Here is the image (base64 JPEG):\n" + b64
    )
    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expert recycling classifier."},
            {"role": "user",   "content": prompt}
        ],
        max_completion_tokens=3
    )

    raw = resp.choices[0].message.content.strip().lower()
    print(f"[OpenAI] raw response → {raw}")  # log to terminal

    # 4) sanitize: match against CLASS_NAMES
    label = next((c for c in CLASS_NAMES if raw == c or c in raw), None)
    if label is None:
        print(f"[OpenAI] unexpected label “{raw}”, defaulting to “trash”")
        label = "trash"
    return label, 1.0

# ─── 6) Shared state & background thread ───────────────────────────────────
running         = True
display_label   = "Waiting for Object"
confidence      = 0.0
trash_weight    = 0.0
recycle_weight  = 0.0
bin_capacity    = BIN_CAP

def classify_and_act():
    global display_label, confidence, trash_weight, recycle_weight
    while running:
        frame    = picam2.capture_array()
        diff_val = np.mean(np.abs(frame.astype(np.int16) - background))
        if diff_val < DIFF_THRESH:
            display_label = "Waiting for Object"
            confidence    = 0.0
        else:
            time.sleep(PAUSE_BEFORE)
            raw_label, conf = classify_with_openai(frame)
            confidence       = conf

            if raw_label in RECYCLE_SET:
                display_label   = f"Recycle ({raw_label.capitalize()})"
                recycle_weight += ITEM_WEIGHT
                rotate_left();  time.sleep(4); stop_motor()
                time.sleep(1)
                rotate_right(); time.sleep(4); stop_motor()
            elif raw_label in TRASH_SET:
                display_label = f"Trash ({raw_label.capitalize()})"
                trash_weight  += ITEM_WEIGHT
                rotate_right(); time.sleep(4); stop_motor()
                time.sleep(1)
                rotate_left();  time.sleep(4); stop_motor()
            else:
                display_label = f"Unknown ({raw_label.capitalize()})"
                stop_motor()

        time.sleep(5)

threading.Thread(target=classify_and_act, daemon=True).start()

# ─── 7) Pygame UI loop ───────────────────────────────────────────────────────
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
                    trash_weight   = recycle_weight = 0.0
                    display_label  = "Waiting for Object"
                elif cap_rect.collidepoint(x,y):
                    bin_capacity += -0.5 if x < cap_rect.centerx else 0.5
                    bin_capacity = max(1.0, bin_capacity)

        screen.fill(WHITE)
        title = font_title.render("AI Trash", True, BLACK)
        screen.blit(title, ((SCREEN_W - title.get_width())//2, 20))

        cur = font_text.render(display_label, True, BLACK)
        screen.blit(cur, ((SCREEN_W - cur.get_width())//2, 140))

        pct_rec = min(recycle_weight / bin_capacity, 1.0)
        pct_tr  = min(trash_weight   / bin_capacity, 1.0)
        rec = font_text.render(f"Recycling: {pct_rec:.1%}", True, LIGHT_BLUE)
        tr  = font_text.render(f"Trash:     {pct_tr:.1%}", True, LIGHT_CORAL)
        screen.blit(rec, ((SCREEN_W - rec.get_width())//2, 220))
        screen.blit(tr,  ((SCREEN_W - tr.get_width())//2, 300))

        frame = picam2.capture_array()
        if frame.ndim == 2:
            frame = np.stack((frame,)*3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:,:,:3]
        h,w  = frame.shape[:2]
        surf = pygame.image.frombuffer(frame.tobytes(), (w,h), 'RGB')
        fw   = SCREEN_W - 120
        fh   = int(fw * h / w)
        feed = pygame.transform.scale(surf, (fw, fh))
        screen.blit(feed, ((SCREEN_W - fw)//2, 380))

        pygame.draw.rect(screen, LIGHT_BLUE, reset_rect)
        rs = font_button.render("Reset", True, BLACK)
        screen.blit(rs, (reset_rect.x + (btn_w - rs.get_width())//2,
                         reset_rect.y + (btn_h - rs.get_height())//2))

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
