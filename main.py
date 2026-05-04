import os
import subprocess
import cv2
import numpy as np

from hand_tracker import HandTracker
from drawing_canvas import DrawingCanvas

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("მოდელი არ მოიძებნა — ვტვირთავ...")
        subprocess.run(["curl", "-L", "-o", MODEL_PATH, MODEL_URL], check=True)
        print("მოდელი ჩამოტვირთულია.")

# ფერების პალიტრა: სახელი → BGR
PALETTE = [
    ("Red",    (0,   0,   255)),
    ("Green",  (0,   200, 0  )),
    ("Blue",   (255, 50,  50 )),
    ("Yellow", (0,   220, 220)),
    ("Purple", (200, 0,   200)),
    ("White",  (255, 255, 255)),
    ("Eraser", (50,  50,  50 )),
    ("Clear",  None            ),
]

PALETTE_H = 85   
ERASER_COLOR = (0, 0, 0)


def build_header(width):
    header = np.zeros((PALETTE_H, width, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    btn_w = width // len(PALETTE)

    for i, (name, color) in enumerate(PALETTE):
        x1, x2 = i * btn_w, (i + 1) * btn_w
        fill = color if color else (20, 20, 20)
        cv2.rectangle(header, (x1 + 4, 6), (x2 - 4, PALETTE_H - 6), fill, -1)
        label = name.upper()
        text_x = x1 + max(4, (btn_w - len(label) * 8) // 2)
        cv2.putText(header, label, (text_x, PALETTE_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0) if name in ("White", "Yellow") else (220, 220, 220),
                    1, cv2.LINE_AA)
    return header, btn_w


def highlight_selected(header, selected_idx, btn_w):
    disp = header.copy()
    x1 = selected_idx * btn_w
    x2 = x1 + btn_w
    cv2.rectangle(disp, (x1 + 2, 2), (x2 - 2, PALETTE_H - 2), (255, 255, 255), 3)
    return disp


def get_palette_index(x, btn_w):
    return min(x // btn_w, len(PALETTE) - 1)


def main():
    ensure_model()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, test_frame = cap.read()
    if not ret:
        print("კამერა ვერ გაიხსნა.")
        return

    h, w = test_frame.shape[:2]
    header_base, btn_w = build_header(w)

    canvas = DrawingCanvas(w, h)
    tracker = HandTracker(model_path=MODEL_PATH)

    draw_color = PALETTE[0][1]   # default: წითელი
    selected_idx = 0
    prev_pt = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=True)
        lms = tracker.get_landmarks(frame)

        if lms:
            fingers = tracker.fingers_up(lms)
            idx_tip = lms[8]   # index fingertip
            mid_tip = lms[12]  # middle fingertip

            two_up  = fingers[1] and fingers[2] and not fingers[3] and not fingers[4]
            one_up  = fingers[1] and not fingers[2]

            if two_up:
                # ---- Selection / hover mode ----
                prev_pt = None
                if idx_tip[1] < PALETTE_H:
                    pi = get_palette_index(idx_tip[0], btn_w)
                    name, color = PALETTE[pi]
                    if name == "Clear":
                        canvas.clear()
                    else:
                        selected_idx = pi
                        draw_color = ERASER_COLOR if name == "Eraser" else color
                # cursor
                cv2.circle(frame, idx_tip, 14, draw_color if draw_color != ERASER_COLOR else (180, 180, 180), 2)
                cv2.circle(frame, mid_tip, 14, (150, 150, 150), 1)

            elif one_up:
                # ---- Drawing mode ----
                if idx_tip[1] > PALETTE_H:
                    if draw_color == ERASER_COLOR:
                        canvas.erase(idx_tip)
                        cv2.circle(frame, idx_tip, canvas.eraser_thickness,
                                   (180, 180, 180), 2)
                    else:
                        if prev_pt:
                            canvas.draw_line(prev_pt, idx_tip, draw_color)
                        prev_pt = idx_tip
                        cv2.circle(frame, idx_tip, 8, draw_color, -1)
                else:
                    prev_pt = None
            else:
                prev_pt = None
        else:
            prev_pt = None

        # canvas-ის overlay
        frame = canvas.overlay(frame)

        # header
        header_disp = highlight_selected(header_base, selected_idx, btn_w)
        frame[:PALETTE_H, :] = header_disp

        
        cv2.putText(frame,
                    "1 finger: Draw  |  2 fingers: Select color  |  C: Clear  |  Q: Quit",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Virtual Painter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
