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

OVERLAY_ALPHA = 0.33


THRESHOLDS = {"A": 0.15, "S": 0.15, "T": 0.15, "N": 0.15, "M": 0.15, "E": 0.15, "J": 0.25, "Z": 0.25}   # only the letters that need a custom value
DEFAULT_THRESHOLD = 0.2

HOLD_SECONDS = 0.75   # how long the sign must be held
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
         "matching": False, "hold_progress": 0,
         "kind": "static", "phase": None, "hint": None}
latest_jpeg = None

if os.path.exists(REF_PATH):
    with open(REF_PATH) as f:
        references = {k: np.array(v) for k, v in json.load(f).items()}
else:
    references = {}
    print("No references.json found. Record some signs first.")

MOTION_TIPS = {"J": 20, "Z": 8}   # J follows the pinky tip, Z the index tip
MOTION_FILE = os.path.join(HERE, "motion_references.json")
MOTION_THRESHOLD = 0.3           # path match strictness (tune this)
MOVE_START = 0.4                  # how far the tip must move to begin (in hand sizes)
STILL_SPEED = 0.02                # per-frame movement below this counts as still
STILL_SECONDS = 0.3               # standing still this long ends the attempt
MAX_ATTEMPT_SECONDS = 2.5
MIN_PATH_LENGTH = 1.0             # rejects tiny wobbles
RESULT_SECONDS = 2.0              # how long "done" stays on screen
TEMPLATE_MODE = False             # set True to record new templates



FINGERS = {
    "thumb": range(1, 5),
    "index": range(5, 9),
    "middle": range(9, 13),
    "ring": range(13, 17),
    "pinky": range(17, 21),
}
HINT_MIN = 0.15   # a finger must be at least this far off to get a hint (tune this)


def worst_finger(current, ref):
    dists = np.linalg.norm(current - ref, axis=1)
    errors = {name: float(np.mean(dists[list(idx)])) for name, idx in FINGERS.items()}
    name = max(errors, key=errors.get)
    return name if errors[name] > HINT_MIN else None

if os.path.exists(MOTION_FILE):
    with open(MOTION_FILE) as f:
        motion_templates = {k: [np.array(p) for p in v] for k, v in json.load(f).items()}
else:
    motion_templates = {}

def resample(points, n=32):
    pts = np.array(points)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    if cum[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, pts[:, 0]),
                     np.interp(t, cum, pts[:, 1])], axis=1)


def normalize_path(points):
    pts = resample(points)          # always 32 evenly spaced points
    pts = pts - pts[0]              # start at the origin
    size = max(np.ptp(pts, axis=0).max(), 1e-6)
    return pts / size               # scale so the shape fits in a unit box


def path_score(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))



def normalize(hand):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand])
    pts = pts - pts[0]
    return pts / np.linalg.norm(pts[9])


def score(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


class MotionTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "waiting"
        self.trail = []
        self.anchor = None
        self.scale = 1.0
        self.start_time = 0.0
        self.last_move = 0.0
        self.result_until = 0.0
        self.correct = False


    def output(self):
        return {"phase": self.phase, "correct": self.correct}

    def update(self, target, hand, current, now, w, h):
        tip = None
        if hand is not None:
            lm = hand[MOTION_TIPS[target]]
            tip = np.array([lm.x * w, lm.y * h])      # pixels

        if self.phase == "done":
            if now > self.result_until:
                self.reset()
            return self.output()

        if self.phase == "waiting":
            pose_ok = (current is not None and target in references and
                       score(current, references[target]) < THRESHOLDS.get(target, DEFAULT_THRESHOLD))
            if not pose_ok:
                self.anchor = None
            elif self.anchor is None:
                self.anchor = tip
                self.scale = np.linalg.norm([(hand[9].x - hand[0].x) * w,
                                             (hand[9].y - hand[0].y) * h])
            elif np.linalg.norm(tip - self.anchor) > MOVE_START * self.scale:
                self.phase = "recording"
                self.start_time = self.last_move = now
                self.trail = [self.anchor / self.scale, tip / self.scale]
            return self.output()

        # recording
        if tip is not None:
            p = tip / self.scale                       # size-independent units
            if np.linalg.norm(p - self.trail[-1]) > STILL_SPEED:
                self.last_move = now
            self.trail.append(p)
        if (now - self.last_move > STILL_SECONDS or
                now - self.start_time > MAX_ATTEMPT_SECONDS):
            self.finish(target, now)
        return self.output()

    def finish(self, target, now):
        self.phase = "done"
        self.result_until = now + RESULT_SECONDS
        self.correct = False
        path = np.array(self.trail)
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        if length < MIN_PATH_LENGTH:
            return
        probe = normalize_path(path)

        if TEMPLATE_MODE:                              # save instead of scoring
            motion_templates.setdefault(target, []).append(probe)
            with open(MOTION_FILE, "w") as f:
                json.dump({k: [p.tolist() for p in v] for k, v in motion_templates.items()}, f)
            print(f"Saved template for {target}", flush=True)
            return

        if target in motion_templates:
            best = min(path_score(probe, t) for t in motion_templates[target])
            self.correct = best < MOTION_THRESHOLD

#----------------------------------------------------------------------
#----------------------CAMERA LOOP STARTS HERE-------------------------
#----------------------------------------------------------------------


motion = MotionTracker()

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
            overlay = frame.copy()          # draw on a copy, not the real frame

            current_hand = None
            for hand in result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(overlay, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(overlay, p, 5, (0, 0, 255), -1)
                current = normalize(hand)
                current = normalize(hand)
                current_hand = hand

            # blend the copy over the original
            frame = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0)

            with lock:
                target = state["target"]
                if target != last_target:
                    hold_start = None
                    motion.reset()
                    last_target = target

                if target in MOTION_TIPS:
                    out = motion.update(target, current_hand, current, time.monotonic(), w, h)
                    state.update(kind="motion", hand_visible=current is not None,
                                matching=False, hold_progress=0, score=None, detected=None, **out)
                else:


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
                        # hint=hint
                    )

                    # matching = False
                    # target_score = None
                    # hint = None
                    # if current is not None and target in references:
                    #     target_score = score(current, references[target])
                    #     matching = target_score < THRESHOLDS.get(target, DEFAULT_THRESHOLD)
                    #     if not matching:
                    #         hint = worst_finger(current, references[target])

            

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


