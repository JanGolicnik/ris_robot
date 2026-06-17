#!/usr/bin/env python3

import cv2
import numpy as np


COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 50]), np.array([10, 255, 255])),
        (np.array([170, 100, 50]), np.array([180, 255, 255]))
    ],
    "green": [
        (np.array([50, 120, 120]), np.array([80, 255, 255]))
    ],
    "blue": [
        (np.array([90, 80, 150]), np.array([120, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 120, 120]), np.array([40, 255, 255]))
    ]
}

class LineDetector:

    def find_line(self, image, color):

        h, w = image.shape[:2]

        # only bottom part of image
        roi = image[int(h * 0.75):, :]

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        mask = None

        for lo, hi in COLOR_RANGES[color]:

            current = cv2.inRange(
                hsv,
                lo,
                hi
            )

            if mask is None:
                mask = current
            else:
                mask = cv2.bitwise_or(mask, current)

        # connect broken line pieces
        mask = cv2.dilate(
            mask,
            np.ones((3,3), np.uint8),
            iterations=1
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, None, None, None, mask

        largest = max(
            contours,
            key=cv2.contourArea
        )

        x, y, bw, bh = cv2.boundingRect(
            largest
        )

        # reject tiny garbage
        if bw < 30:
            return False, None, None, None, mask

        rect = cv2.minAreaRect(
            largest
        )

        center = rect[0]
        angle = rect[2]

        cx = int(center[0])
        cy = int(center[1]) + int(h * 0.75)

        return (
            True,
            cx,
            cy,
            angle,
            mask
        )