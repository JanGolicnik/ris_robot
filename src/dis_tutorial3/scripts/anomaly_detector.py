#!/usr/bin/env python3

import cv2
import numpy as np


class AnomalyDetector:
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blackhat highlights dark defects (cracks, scratches)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 35, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # blob analysis
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        num_blobs, largest_blob, total_blob_area = 0, 0, 0
        for i in range(1, num_labels):
            blob_area = stats[i, cv2.CC_STAT_AREA]
            if blob_area < 10:
                continue
            num_blobs += 1
            largest_blob = max(largest_blob, blob_area)
            total_blob_area += blob_area

        # hough circles for donut-shaped defects
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=50,
            param1=45,
            param2=26,
            minRadius=10,
            maxRadius=80,
        )

        # specular detection — suppress false positives from bright reflections
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        specular_ratio = np.sum(hsv[:, :, 2] > 220) / hsv[:, :, 2].size
        is_specular = specular_ratio > 0.6

        is_anomaly = (largest_blob > 500 or circles is not None) and not is_specular

        # debug visualizations
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # cv2.imshow("gray eq", clahe.apply(gray))

        hough_vis = image.copy()
        if circles is not None:
            for cx, cy, r in np.round(circles[0]).astype(int):
                cv2.circle(hough_vis, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(hough_vis, (cx, cy), 2, (0, 0, 255), 3)
        else:
            cv2.putText(
                hough_vis,
                "no circles",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        # cv2.imshow("hough", hough_vis)

        print(
            f"min={np.min(blackhat)} max={np.max(blackhat)} mean={np.mean(blackhat):.1f} "
            f"largest_blob={largest_blob} n_blobs={num_blobs} specular={specular_ratio:.2f} area={cv2.countNonZero(mask)}"
        )

        return is_anomaly, mask, blackhat
