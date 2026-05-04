import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

TIP_IDS = [4, 8, 12, 16, 20]

_t0 = time.perf_counter()


def _ts_ms():
    return int((time.perf_counter() - _t0) * 1000)


class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", max_hands=1,
                 detection_conf=0.75, tracking_conf=0.75):
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._result = None

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._result = self._landmarker.detect_for_video(mp_image, _ts_ms())

        if draw and self._result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_lms in self._result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 0), 1)
                for pt in pts:
                    cv2.circle(frame, pt, 3, (255, 255, 255), -1)
        return frame

    def get_landmarks(self, frame, hand_index=0):
        if not self._result or not self._result.hand_landmarks:
            return []
        if hand_index >= len(self._result.hand_landmarks):
            return []
        h, w = frame.shape[:2]
        return [
            (int(lm.x * w), int(lm.y * h))
            for lm in self._result.hand_landmarks[hand_index]
        ]

    def fingers_up(self, landmarks):
        if not landmarks:
            return [0, 0, 0, 0, 0]
        fingers = []
        fingers.append(1 if landmarks[TIP_IDS[0]][0] < landmarks[TIP_IDS[0] - 1][0] else 0)
        for i in range(1, 5):
            fingers.append(1 if landmarks[TIP_IDS[i]][1] < landmarks[TIP_IDS[i] - 2][1] else 0)
        return fingers
