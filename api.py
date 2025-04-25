#!/usr/bin/env python3
import sys
import time
import base64
import io
import threading

import numpy as np
import cv2
from PIL import Image
import openai

# ─── 0) Prompt for Project-Scoped API Key ────────────────────────────────
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

# ─── 1) Class Info ───────────────────────────────────────────────────────
CLASS_NAMES    = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic", "trash"]
RECYCLE_SET    = {"cardboard", "glass", "metal", "paper"}
TRASH_SET      = {"biodegradable", "plastic", "trash"}

ITEM_WEIGHT    = 0.125  # lbs per item
DIFF_THRESH    = 5      # motion threshold
PAUSE_BEFORE   = 2      # seconds before classifying
BIN_CAP        = 10.0   # gallons

# ─── 2) Webcam Setup ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access webcam.")
    sys.exit(1)

time.sleep(2)
ret, frame = cap.read()
if not ret:
    print("[ERROR] Failed to read from webcam.")
    sys.exit(1)
background = frame.astype(np.int16)

# ─── 3) Classify with OpenAI ─────────────────────────────────────────────
def classify_with_openai(frame: np.ndarray) -> (str, float):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert recycling classifier."},
            {"role": "user", "content":
                "Classify this image into one of: biodegradable, cardboard, glass, metal, paper, plastic, trash.\n"
                "Here is the image (base64 PNG):\n" + b64_img
            }
        ],
        temperature=0.0
    )
    label = resp.choices[0].message.content.strip().lower()
    return label, 1.0

# ─── 4) Classify Loop ────────────────────────────────────────────────────
running = True
trash_weight = 0.0
recycle_weight = 0.0
bin_capacity = BIN_CAP

def classify_loop():
    global trash_weight, recycle_weight
    while running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            continue

        diff = np.mean(np.abs(frame.astype(np.int16) - background))
        if diff < DIFF_THRESH:
            print("[INFO] No object detected.")
        else:
            print("[INFO] Object detected. Classifying...")
            time.sleep(PAUSE_BEFORE)

            try:
                label, _ = classify_with_openai(frame)
            except Exception as e:
                print(f"[ERROR] OpenAI request failed: {e}")
                continue

            if label in RECYCLE_SET:
                recycle_weight += ITEM_WEIGHT
                print(f"[RECYCLE] {label.capitalize()} → Recycle bin")
            elif label in TRASH_SET:
                trash_weight += ITEM_WEIGHT
                print(f"[TRASH] {label.capitalize()} → Trash bin")
            else:
                print(f"[UNKNOWN] {label.capitalize()} → No action taken")

            r_pct = recycle_weight / bin_capacity
            t_pct = trash_weight / bin_capacity
            print(f"  • Recycled: {recycle_weight:.2f} lbs ({r_pct:.0%})")
            print(f"  • Trash:    {trash_weight:.2f} lbs ({t_pct:.0%})")

        time.sleep(5)

# ─── 5) Start Main Loop ──────────────────────────────────────────────────
try:
    print("🚀 AI Trash Classifier (Project Key Mode) Running... Ctrl+C to stop\n")
    threading.Thread(target=classify_loop, daemon=True).start()

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Exiting.")
    running = False
    cap.release()
