#!/usr/bin/env python3

import cv2
import numpy as np

COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 50]), np.array([10, 255, 255])),
        (np.array([170, 100, 50]), np.array([180, 255, 255])),
    ],
    "green": [(np.array([50, 120, 120]), np.array([80, 255, 255]))],
    "blue": [(np.array([90, 90, 150]), np.array([110, 255, 255]))],
    "yellow": [(np.array([20, 120, 120]), np.array([40, 255, 255]))],
}


class LineDetector:
    def find_line(self, image, color):

        h, w = image.shape[:2]

        # only lower part of image
        roi = image[int(h * 0.70):, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = None

        for lo, hi in COLOR_RANGES[color]:
            current = cv2.inRange(hsv, lo, hi)

            if mask is None:
                mask = current
            else:
                mask = cv2.bitwise_or(mask, current)

        # clean mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, None, None, None, mask, False, False, False

        # choose contour closest to image center
        best = None
        best_score = float("inf")

        for c in contours:

            area = cv2.contourArea(c)

            if area < 100:
                continue

            M = cv2.moments(c)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])

            score = abs(cx - mask.shape[1] // 2)

            if score < best_score:
                best_score = score
                best = c

        if best is None:
            return False, None, None, None, mask, False, False, False

        rect = cv2.minAreaRect(best)

        center = rect[0]
        angle = rect[2]

        cx = int(center[0])
        cy = int(center[1]) + int(h * 0.70)

        # junction detection
        junction_roi = mask[int(mask.shape[0] * 0.25):, :]

        third = junction_roi.shape[1] // 3

        left_pixels = cv2.countNonZero(
            junction_roi[:, :third]
        )

        center_pixels = cv2.countNonZero(
            junction_roi[:, third:2 * third]
        )

        right_pixels = cv2.countNonZero(
            junction_roi[:, 2 * third:]
        )

        left_exists = left_pixels > 200
        center_exists = center_pixels > 200
        right_exists = right_pixels > 200

        cv2.imshow("line mask", mask)

        return (
            True,
            cx,
            cy,
            angle,
            mask,
            left_exists,
            center_exists,
            right_exists,
        )