import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import tkinter as tk
from tkinter import messagebox
from collections import deque
import platform
import subprocess
import os

# ------------------------- CONFIG -----------------------------
BLINK_GROUP_TIME = 0.85
BLINK_DEBOUNCE = 0.18
EYE_RATIO_THRESHOLD = 0.23
SMOOTHING = 0.45
RATIO_BUFFER_SIZE = 6
CONSECUTIVE_CLOSED_REQUIRED = 2
# --------------------------------------------------------------

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
mp_face_mesh = mp.solutions.face_mesh

running = False
blink_count = 0
last_blink_time = 0.0
last_detect_time = 0.0

smoothed_x = None
smoothed_y = None
ratio_buffer = deque(maxlen=RATIO_BUFFER_SIZE)
consecutive_closed = 0

config_lock = threading.Lock()

# Light Theme
THEME = {
    'bg': '#F7F9FC',
    'panel': '#FFFFFF',
    'accent': '#3B82F6',
    'accent_text': '#FFFFFF',
    'text': '#0F172A',
    'muted': '#475569'
}

# ---- OPEN FILE USING ENTER (3 blinks) ----
def open_focused_item():
    try:
        pyautogui.press('enter')
        print("Opened focused item via Enter")
    except:
        print("Failed to trigger Enter key")

# ---- EYE RATIO FUNCTION ----
def eye_aspect_ratio(upper, lower, left_corner, right_corner, w, h):
    v = abs((upper.y - lower.y) * h)
    h_dist = abs((left_corner.x - right_corner.x) * w)
    if h_dist == 0:
        return 1.0
    return v / h_dist

# ---- ACTION BASED ON BLINK COUNT ----
def perform_action_for_blinks(count):
    print("Blink group:", count)
    if count == 1:
        pyautogui.click()
    elif count == 2:
        pyautogui.rightClick()
    elif count == 3:
        open_focused_item()

# ---- CAMERA THREAD / MAIN LOGIC ----
def camera_loop():
    global running, blink_count, last_blink_time, last_detect_time
    global smoothed_x, smoothed_y, ratio_buffer, consecutive_closed

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Camera not available")
            running = False
            return

        while running:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # ---- Cursor control ----
                try:
                    iris = lm[468]
                    cx = int(screen_w * iris.x)
                    cy = int(screen_h * iris.y)
                except:
                    cx, cy = None, None

                if cx is not None:
                    if smoothed_x is None:
                        smoothed_x, smoothed_y = cx, cy
                    else:
                        smoothed_x = int(smoothed_x * (1 - SMOOTHING) + cx * SMOOTHING)
                        smoothed_y = int(smoothed_y * (1 - SMOOTHING) + cy * SMOOTHING)

                    pyautogui.moveTo(smoothed_x, smoothed_y, _pause=False)

                # ---- Blink detection ----
                try:
                    upper = lm[145]
                    lower = lm[159]
                    left_corner = lm[33]
                    right_corner = lm[133]

                    ratio = eye_aspect_ratio(upper, lower, left_corner, right_corner, w, h)
                    ratio_buffer.append(ratio)
                    smooth_ratio = sum(ratio_buffer) / len(ratio_buffer)

                    now = time.time()
                    if smooth_ratio < EYE_RATIO_THRESHOLD:
                        consecutive_closed += 1
                    else:
                        consecutive_closed = 0

                    if consecutive_closed >= CONSECUTIVE_CLOSED_REQUIRED and (now - last_detect_time) > BLINK_DEBOUNCE:
                        blink_count += 1
                        last_detect_time = now
                        last_blink_time = now
                        consecutive_closed = 0
                        print("Blink detected ->", blink_count)

                    if blink_count > 0 and (now - last_blink_time) > BLINK_GROUP_TIME:
                        perform_action_for_blinks(blink_count)
                        blink_count = 0
                except:
                    pass

            cv2.imshow("Eye Control — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

        cam.release()
        cv2.destroyAllWindows()
        print("Camera stopped cleanly")

# ---- START CAMERA ----
def start_camera():
    global running
    if running: return
    running = True
    threading.Thread(target=camera_loop, daemon=True).start()
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")


# ---- STOP CAMERA ----
def stop_camera():
    global running
    running = False
    start_btn.config(state="normal")
    stop_btn.config(state="disabled")

# ------------------------- UI -------------------------------
root = tk.Tk()
root.title("Eye Blink Control — Smooth")
root.configure(bg=THEME['bg'])
root.geometry('400x180')
root.resizable(False, False)

frame = tk.Frame(root, bg=THEME['panel'], padx=15, pady=15)
frame.place(relx=0.5, rely=0.5, anchor='center')

start_btn = tk.Button(frame, text="Start Tracking", width=16, command=start_camera,
                      bg=THEME['accent'], fg=THEME['accent_text'], bd=0)
start_btn.grid(row=0, column=0, pady=8, padx=6)


stop_btn = tk.Button(frame, text="Stop Tracking", width=16, command=stop_camera,
                     state="disabled", bg='#E5E7EB', fg=THEME['text'], bd=0)
stop_btn.grid(row=0, column=1, pady=8, padx=6)

tk.Label(frame, text="1 blink = Left | 2 = Right | 3 = Open focused item", 
         bg=THEME['panel'], fg=THEME['muted']).grid(row=1, column=0, columnspan=2)

quit_btn = tk.Button(frame, text="Quit", width=16, command=root.quit,
                     bg="#F97316", fg="white", bd=0)
quit_btn.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()