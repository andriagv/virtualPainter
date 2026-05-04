import cv2
import numpy as np


class DrawingCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.brush_thickness = 10
        self.eraser_thickness = 50

    def draw_line(self, p1, p2, color):
        cv2.line(self.canvas, p1, p2, color, self.brush_thickness)

    def erase(self, center):
        cv2.circle(self.canvas, center, self.eraser_thickness, (0, 0, 0), -1)

    def clear(self):
        self.canvas[:] = 0

    def overlay(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        return cv2.add(frame_bg, canvas_fg)
