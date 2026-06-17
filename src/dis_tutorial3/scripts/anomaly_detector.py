#!/usr/bin/env python3

import cv2
import numpy as np

class AnomalyDetector:

    def detect(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        #naredimo kernel za blackhat operacijo, da dobimo temne dele slike
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (31,31)
        )

        #poudari razpoke, praske...
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        #pretvori sivinsko sliko v crno belo oz binarno
        _, mask = cv2.threshold(
            blackhat,
            60,
            255,
            cv2.THRESH_BINARY
        )

        #to samo poveca bele dele, da se lazje konture najdejo
        mask = cv2.dilate(
            mask,
            np.ones((3,3), np.uint8),
            iterations=1
        )

        #poveze jih v vecjo celoto
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        largest_blob = 0

        #od vseh kontur najde najvecjo
        for i in range(1, num_labels):
            blob_area = stats[i, cv2.CC_STAT_AREA]
            largest_blob = max(largest_blob, blob_area)

        area = cv2.countNonZero(mask)

        is_anomaly = largest_blob > 500

        print(
            "min =", np.min(blackhat),
            "max =", np.max(blackhat),
            "mean =", np.mean(blackhat),
            "largest_blob =", largest_blob
        )

        print(
            "area =", area
        )

        return is_anomaly, mask, blackhat