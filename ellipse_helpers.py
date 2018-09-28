#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np


def approx_ellipse_perimeter(a, b):
    lmbd = (max(a, b) - min(a, b)) / (a + b)
    return (a + b) * np.pi * (1 + 3 * lmbd ** 2 / (10 + np.sqrt(4 - 3 * lmbd ** 2)))


def find_nearest_point_on_ellipse(semi_major, semi_minor, p):
    ########################################################################
    #                                                                      #
    #  This Code is based on https://github.com/0xfaded/ellipse_demo       #
    #  See Carl Chatfield's blog for an excellent explanation:             #
    #  https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/  #
    #  This function is licensed under an MIT-License:                     #
    #  https://github.com/0xfaded/ellipse_demo/blob/master/LICENSE         #
    #                                                                      #
    ########################################################################
    px = abs(p[0])
    py = abs(p[1])
    tx = 0.707
    ty = 0.707
    a = semi_major
    b = semi_minor
    for x in range(0, 3):
        x = a * tx
        y = b * ty
        ex = (a*a - b*b) * tx**3 / a
        ey = (b*b - a*a) * ty**3 / b
        rx = x - ex
        ry = y - ey
        qx = px - ex
        qy = py - ey
        r = np.hypot(ry, rx)
        q = np.hypot(qy, qx)
        tx = min(1, max(0, (qx * r / q + ex) / a))
        ty = min(1, max(0, (qy * r / q + ey) / b))
        t = np.hypot(ty, tx)
        tx /= t
        ty /= t
    return np.copysign(a * tx, p[0]), np.copysign(b * ty, p[1])


def rotate_point(p, pivot, angle):
    # Move to pivot
    tp = p[0] - pivot[0], p[1] - pivot[1]
    # Rotate
    pr = (tp[0] * np.cos(angle / 180 * np.pi) - tp[1] * np.sin(angle / 180 * np.pi),
          tp[0] * np.sin(angle / 180 * np.pi) + tp[1] * np.cos(angle / 180 * np.pi))
    # Move back
    return pr[0] + pivot[0], pr[1] + pivot[1]


if __name__ == "__main__":
    contour = np.array([
        [[200, 100]], [[250, 30]], [[350, 100]], [[440, 300]],
        [[420, 390]], [[300, 330]], [[200, 200]]], dtype=np.int)
    ellipse = ((320, 210), (200, 400), -30)
    import cv2
    cvs = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(cvs, (int(ellipse[0][0]-1), int(ellipse[0][1]-1)), (int(ellipse[0][0]+1), int(ellipse[0][1]+1)),
                  (20, 255, 255), 1)
    cv2.drawContours(cvs, [contour], 0, (255, 0, 0), 1)
    rotated_cnt = []
    for p in contour:
        rotated_cnt.append([rotate_point(p[0], ellipse[0], -ellipse[2]+90)])
    cv2.drawContours(cvs, [np.array(rotated_cnt, dtype=np.int)], 0, (255, 255, 0), 1)
    cv2.ellipse(cvs, (ellipse[0], ellipse[1], 90), (128, 0, 128), 1)
    cv2.ellipse(cvs, ellipse, (255, 0, 255), 1)
    for p in contour:
        tp = p[0][0] - ellipse[0][0], p[0][1] - ellipse[0][1]
        cv2.rectangle(
            cvs,
            (int(tp[0]-1), int(tp[1]-1)),
            (int(tp[0]+1), int(tp[1]+1)),
            (20, 5, 100), 1)
        rtp = rotate_point(tp, (0, 0), -ellipse[2]+90)
        cv2.rectangle(
            cvs,
            (int(rtp[0]-1), int(rtp[1]-1)),
            (int(rtp[0]+1), int(rtp[1]+1)),
            (20, 50, 200), 1)
        poe = find_nearest_point_on_ellipse(ellipse[1][1] / 2, ellipse[1][0] / 2, rtp)
        cv2.rectangle(cvs,
                      (int(poe[0] - 1), int(poe[1] - 1)),
                      (int(poe[0] + 1), int(poe[1] + 1)),
                      (128, 0, 128), 1)
        poer = rotate_point(poe, (0, 0), ellipse[2]-90)
        cv2.rectangle(cvs,
                      (int(poer[0] - 1), int(poer[1] - 1)),
                      (int(poer[0] + 1), int(poer[1] + 1)),
                      (150, 0, 30), 1)
        poert = poer[0] + ellipse[0][0], poer[1] + ellipse[0][1]
        cv2.rectangle(cvs,
                      (int(poert[0] - 1), int(poert[1] - 1)),
                      (int(poert[0] + 1), int(poert[1] + 1)),
                      (255, 255, 255), 1)
        dist = np.sqrt((p[0][0] - poert[0]) ** 2 + (p[0][1] - poert[1]) ** 2)
        print(dist)
    cv2.imshow('Ellipse test', cvs)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
