import json
import os
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
REF_PATH = os.path.join(HERE, "references.json")


THRESHOLDS = {"A": 0.12, "S": 0.12, "T": 0.12, "N": 0.12, "M": 0.12, "E": 0.12}   # only the letters that need a custom value
DEFAULT_THRESHOLD = 0.15

HOLD_SECONDS = 1.0   # how long the sign must be held
GRACE = 0.25         # forgive brief tracking flickers so the timer doesn't reset

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Shared between the camera thread and the web endpoints
lock = threading.Lock()
state = {"target": None, "detected": None, "score": None,
         "correct": False, "hand_visible": False,
         "matching": False, "hold_progress": 0}
latest_jpeg = None

if os.path.exists(REF_PATH):
    with open(REF_PATH) as f:
        references = {k: np.array(v) for k, v in json.load(f).items()}
else:
    references = {}
    print("No references.json found. Record some signs first.")


def normalize(hand):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand])
    pts = pts - pts[0]
    return pts / np.linalg.norm(pts[9])


def score(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def camera_loop():
    global latest_jpeg
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    cap = cv2.VideoCapture(0)
    start = time.time()
    last_ts = -1

    hold_start = None
    last_match = 0.0
    last_target = None

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int((time.time() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            current = None
            h, w, _ = frame.shape
            for hand in result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 5, (0, 0, 255), -1)
                current = normalize(hand)

            with lock:
                target = state["target"]
                if target != last_target:       # switched letters, start over
                    hold_start = None
                    last_target = target

                matching = False
                target_score = None
                if current is not None and target in references:
                    target_score = score(current, references[target])
                    matching = target_score < THRESHOLDS.get(target, DEFAULT_THRESHOLD)

                now = time.monotonic()
                if matching:
                    last_match = now
                    if hold_start is None:
                        hold_start = now
                elif hold_start is not None and now - last_match > GRACE:
                    hold_start = None

                progress = 0.0 if hold_start is None else min((now - hold_start) / HOLD_SECONDS, 1.0)

                state.update(
                    hand_visible=current is not None,
                    detected=None,
                    score=None if target_score is None else round(target_score, 3),
                    matching=matching,
                    hold_progress=round(progress * 100),
                    correct=progress >= 1.0,
                )

            

            _, buf = cv2.imencode(".jpg", frame)
            with lock:
                latest_jpeg = buf.tobytes()


# ---------- Web server ----------
# static_folder="static" + static_url_path="" serves static/script.js at /script.js, etc.
FRONT_END = os.path.join(HERE, "..", "Front End")
app = Flask(__name__, static_folder=FRONT_END, static_url_path="")


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/signs")
def get_signs():
    return jsonify(signs=list(references.keys()))


@app.post("/target")
def set_target():
    sign = (request.get_json(silent=True) or {}).get("sign")
    if sign not in references:
        return jsonify(ok=False, error=f"Unknown sign: {sign}"), 400
    with lock:
        state["target"] = sign
    return jsonify(ok=True, target=sign)


@app.get("/status")
def get_status():
    with lock:
        return jsonify(dict(state))


def mjpeg_stream():
    while True:
        with lock:
            frame = latest_jpeg
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.get("/video")
def video():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(port=8000, threaded=True)  # leave debug off, or the camera starts twice