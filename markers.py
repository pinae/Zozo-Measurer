#!/usr/bin/python3
# -*- coding: utf-8 -*-
#######################################################################
#
#  This Code is based on the work of Ksasao (https://github.com/ksasao)
#  who released his code under Apache License V2.0.
#  See: https://gist.github.com/ksasao/bc9c548d5e38932f2d0d11912ba541d0
#
#######################################################################
from __future__ import division, print_function, unicode_literals
from cv2 import cvtColor, GaussianBlur, threshold, findContours, contourArea, boundingRect, arcLength
from cv2 import COLOR_BGR2GRAY, THRESH_BINARY_INV, THRESH_OTSU, RETR_TREE, CHAIN_APPROX_NONE
from cv2 import boundingRect, fitEllipse, warpAffine, getAffineTransform
from cv2 import circle, LINE_AA
from cv2 import resize, INTER_CUBIC, putText, FONT_HERSHEY_PLAIN
from ellipse_helpers import approx_ellipse_perimeter, find_nearest_point_on_ellipse, rotate_point
import numpy as np


def contour_sanity_check(contour, image_height, point_d=0.02):
    # We assume a persons height is between 1,3m and 2,2m.
    # In a reasonably sane photographed image the person is either
    # head to toe in the image or uses at least half the frame.
    # Points with ID have a diameter of 2cm. The small points at
    # the neck have a diameter of 1cm.
    x, y, w, h = boundingRect(contour)
    # Calculate a lower bound for the size of one pixel
    lb = 1.3 / image_height
    # Calculate an upper bound for the size of one pixel
    ub = 2 * 2.2 / image_height
    # Checking bounds for width and height
    if max(w, h) * ub < point_d or max(w, h) * lb > point_d:
        return False
    # The maximum area of a point is a circle
    if contourArea(contour) > np.pi * (point_d / 2 / lb) ** 2:
        return False
    # The maximum perimeter of a point is a circle
    if arcLength(contour, True) * lb > np.pi * point_d:
        return False
    # The minimum perimeter of a point is 2*d
    if arcLength(contour, True) * ub < 2 * point_d:
        return False
    # The perimeter should not be much bigger than that of the
    # minimal ellipse with the same area
    epsilon_factor = 1.5
    if arcLength(contour, True) > epsilon_factor * approx_ellipse_perimeter(w, h):
        return False
    # Calculate the average quadratic distance of the contour to a fitted ellipse
    if len(contour) < 5:
        return False
    ellipse = fitEllipse(contour)
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
        return False
    # For very flat ellipses there is no hope to detect a point id later
    if min(ellipse[1]) < 0.1 * max(ellipse[1]):
        return False
    # Check if the contour is roughly elliptical
    quad_dist = 0
    for p in contour:
        tp = p[0][0] - ellipse[0][0], p[0][1] - ellipse[0][1]
        rtp = rotate_point(tp, (0, 0), -ellipse[2] + 90)
        poe = find_nearest_point_on_ellipse(ellipse[1][1] / 2, ellipse[1][0] / 2, rtp)
        poer = rotate_point(poe, (0, 0), ellipse[2] - 90)
        poert = poer[0] + ellipse[0][0], poer[1] + ellipse[0][1]
        quad_dist += (p[0][0] - poert[0]) ** 2 + (p[0][1] - poert[1]) ** 2
    if quad_dist / len(contour) > 1.0:
        return False
    # This contour could be a point
    return True


def find_marker_ellipses(im):
    im_gray = cvtColor(im, COLOR_BGR2GRAY)
    im_blur = GaussianBlur(im_gray, (3, 3), 0)
    ret, th = threshold(im_blur, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)
    imgEdge, contours, hierarchy = findContours(th, RETR_TREE, CHAIN_APPROX_NONE)
    points = []
    origins = []
    ellipses = []

    import cv2
    im_draw = im.copy()

    id_point_candidates = []
    small_point_candidates = []
    for cnt in contours:
        if contour_sanity_check(cnt, im.shape[0], point_d=0.02):
            cv2.drawContours(im_draw, [cnt], 0, (0, 255, 255), 1)
            id_point_candidates.append(cnt)
        elif contour_sanity_check(cnt, im.shape[0], point_d=0.01):
            cv2.drawContours(im_draw, [cnt], 0, (255, 255, 0), 1)
            small_point_candidates.append(cnt)
        else:
            cv2.drawContours(im_draw, [cnt], 0, (255, 0, 255), 1)

    for i, cnt in enumerate(id_point_candidates):
        x, y, w, h = boundingRect(cnt)
        ellipse = fitEllipse(cnt)
        points.append(im_gray[y:y + h, x:x + w])
        origins.append((x, y))
        ellipses.append(ellipse)

    from detect_points import scale_preview
    im_preview, sf = scale_preview(im_draw)
    cv2.imshow('Contours', im_preview)

    return points, origins, ellipses


def unskew_point(imc, origin, ellipse):
    center_in_cut = ellipse[0][0] - origin[0], ellipse[0][1] - origin[1]
    source_points = np.float32([center_in_cut,
                                [center_in_cut[0] + np.sin(ellipse[2] / 180 * np.pi) * ellipse[1][1] / 2,
                                 center_in_cut[1] - np.cos(ellipse[2] / 180 * np.pi) * ellipse[1][1] / 2],
                                [center_in_cut[0] - np.cos(ellipse[2] / 180 * np.pi) * ellipse[1][0] / 2,
                                 center_in_cut[1] - np.sin(ellipse[2] / 180 * np.pi) * ellipse[1][0] / 2]])
    image_center = max(imc.shape) / 2, max(imc.shape) / 2
    target_points = np.float32([image_center,
                                [image_center[0] + ellipse[1][1] / 2,
                                 image_center[1]],
                                [image_center[0],
                                 image_center[1] - ellipse[1][1] / 2]])
    return warpAffine(imc, getAffineTransform(source_points, target_points), (max(imc.shape), max(imc.shape)))


def draw_small_point(shape, r):
    small_point = np.zeros(shape, dtype=np.uint8)
    circle(small_point, (int(shape[0] / 2 * 16), int(shape[1] / 2 * 16)),
           radius=int(r * 0.5 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)
    circle(small_point, (int(shape[0] / 2 * 16), int(shape[1] / 2 * 16)),
           radius=int(r * 0.1 * 16), color=0, thickness=-1, lineType=LINE_AA, shift=4)
    return small_point


def is_small_point(imc, ellipse):
    center_mask = np.zeros(imc.shape, dtype=np.uint8)
    circle(center_mask, (int(imc.shape[0] / 2 * 16), int(imc.shape[1] / 2 * 16)),
           radius=int(ellipse[1][1] * 0.09 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)
    center_sum = np.sum(imc * (center_mask / 255))
    sum_of_center_mask = np.sum(center_mask / 255)
    avg_center_color = center_sum / sum_of_center_mask
    circle_mask = draw_small_point(imc.shape, ellipse[1][1])
    circle_sum = np.sum(imc * (circle_mask / 255))
    sum_of_circle_mask = np.sum(circle_mask / 255)
    avg_circle_color = circle_sum / sum_of_circle_mask
    brightest_color = np.max(imc * (circle_mask / 255))
    darkest_color = np.min(imc * (center_mask / 255) + (255 - center_mask))
    center_darkness = 1 - (avg_center_color - darkest_color) / ((brightest_color - darkest_color) / 2)
    ring_brightness = 1 - (brightest_color - avg_circle_color) / ((brightest_color - darkest_color) / 2)
    return np.max([0, 0.5 * center_darkness + 0.5 * ring_brightness])


def get_point_id(imc, ellipse):
    mask = generate_mask(imc.shape, ellipse[1][1], 0)
    masked_sum = np.sum(imc * mask)
    sum_of_mask = np.sum(mask)
    min_sum = masked_sum / sum_of_mask
    pattern_angle = 0
    for alpha in range(1, 60):
        mask = generate_mask(imc.shape, ellipse[1][1] / 2, alpha)
        masked_sum = np.sum(imc * mask)
        sum_of_mask = np.sum(mask)
        avg_color = masked_sum / sum_of_mask
        if avg_color < min_sum:
            min_sum = avg_color
            pattern_angle = alpha
    sums = np.zeros(12, dtype=np.float32)
    for i in range(12):
        mask = generate_mask(imc.shape, ellipse[1][1] / 2, pattern_angle, bit_mask=1 << i)
        masked_sum = np.sum(imc * mask)
        sum_of_mask = np.sum(mask)
        sums[i] = masked_sum / sum_of_mask
    min_sum = np.min(sums)
    max_sum = np.max(sums)
    thresh = min_sum + (max_sum - min_sum) / 2
    id_bits = 0
    confidence = 1
    for i in range(12):
        confidence = min(confidence, 1 - (min(sums[i] - min_sum, max_sum - sums[i]) ** 2) / ((max_sum - thresh) ** 2))
        if sums[i] < thresh:
            id_bits = id_bits | 1 << i
    max_id = find_max_id_in_pattern(id_bits)
    return max_id, confidence


def generate_mask(shape, r, alpha, bit_mask=4095, draw_center=False):
    mask = np.zeros(shape, dtype=np.uint8)
    if draw_center:
        circle(mask, (int(shape[0] / 2 * 16), int(shape[1] / 2 * 16)),
               radius=int(r * 0.09 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)
    for i in range(6):
        if 1 << i & bit_mask > 0:
            circle(mask, (int((shape[0] / 2 + np.sin((alpha + 30 + i * 60) / 180 * np.pi) * 0.4 * r) * 16),
                          int((shape[1] / 2 - np.cos((alpha + 30 + i * 60) / 180 * np.pi) * 0.4 * r) * 16)),
                   radius=int(r * 0.09 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)
        if 1 << i + 6 & bit_mask > 0:
            circle(mask, (int((shape[0] / 2 + np.sin((alpha + i * 60) / 180 * np.pi) * 0.693 * r) * 16),
                          int((shape[1] / 2 - np.cos((alpha + i * 60) / 180 * np.pi) * 0.693 * r) * 16)),
                   radius=int(r * 0.09 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)
    return mask.astype(np.float32) / 255


def find_max_id_in_pattern(pattern):
    low = pattern & 63
    high = (pattern & 63 << 6) >> 6
    id_candidates = np.zeros(6, dtype=np.int)
    id_candidates[0] = pattern
    for i in range(1, 6):
        l = (low << i) % (1 << 6) + ((low << i) >> 6)
        h = (high << i) % (1 << 6) + ((high << i) >> 6)
        id_candidates[i] = l + (h << 6)
    return np.max(id_candidates)


def find_best_10_confidences(confidences):
    best_10_confidences = np.zeros(10, dtype=np.float32)
    best_10_indexes = np.zeros(10, dtype=np.uint32)
    for i in range(len(confidences)):
        if np.min(best_10_confidences) < confidences[i]:
            min_index = np.argmin(best_10_confidences)
            best_10_confidences[min_index] = confidences[i]
            best_10_indexes[min_index] = i
    return best_10_confidences, best_10_indexes


def draw_ideal_point(point_id, size=(18, 18), angle=0):
    mask = 1 - generate_mask(size, max(size)/2, angle, point_id, draw_center=True)
    img = np.zeros((*size, 3), dtype=np.uint8)
    circle(img, (int(size[0] / 2 * 16), int(size[1] / 2 * 16)), radius=int(max(size) / 2 * 16),
           color=(255, 255, 255), thickness=-1, lineType=LINE_AA, shift=4)
    img = (img * np.stack([mask, mask, mask], axis=2)).astype(np.uint8)
    return img


def collect_points(target_size, data):
    canvas = np.zeros((target_size[0] * 3 + 41, len(data) * target_size[1], 3), dtype=np.uint8)
    for i in range(len(data)):
        spi = data[i]["skewed_point"]
        w = min(int(spi.shape[1] * target_size[1] / spi.shape[0]), target_size[0])
        h = min(int(spi.shape[0] * target_size[0] / spi.shape[1]), target_size[1])
        spis = resize(spi, (w, h), interpolation=INTER_CUBIC)
        pis = resize(data[i]["unskewed_point"], target_size, interpolation=INTER_CUBIC)
        if data[i]["point_id"] == 0:  # Draw a small point
            sp = draw_small_point(target_size, target_size[0] / 2)
            ideal_point = np.stack([sp, sp, sp], axis=2)
        else:  # Draw a big point
            ideal_point = draw_ideal_point(data[i]["point_id"], size=target_size, angle=0)
        canvas[(target_size[0] - spis.shape[0]) // 2:
               (target_size[0] - spis.shape[0]) // 2 + spis.shape[0],
               i * target_size[1] + (target_size[1] - spis.shape[1]) // 2:
               i * target_size[1] + (target_size[1] - spis.shape[1]) // 2 + spis.shape[1]] = np.stack(
            [spis, spis, spis], axis=2)
        canvas[target_size[0]:
               target_size[0]+pis.shape[0],
               i * target_size[1]:
               i * target_size[1] + pis.shape[1]] = np.stack([pis, pis, pis], axis=2)
        canvas[2 * target_size[0]:
               2 * target_size[0] + ideal_point.shape[0],
               i * target_size[1]:
               i * target_size[1] + ideal_point.shape[1]] = ideal_point
        putText(canvas, str(data[i]["point_id"]), (i * target_size[1] + 4, target_size[0] * 3 + 12),
                fontFace=FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 255),
                thickness=1,
                lineType=LINE_AA)
        putText(canvas, str(int(data[i]["confidence"]*100)) + "%", (i * target_size[1] + 4, target_size[0] * 3 + 26),
                fontFace=FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, int(data[i]["confidence"] * 255), int((1 - data[i]["confidence"]) * 255)),
                thickness=1,
                lineType=LINE_AA)
        putText(canvas, "{:02.1f}mm".format(data[i]["distance"]), (i * target_size[1] + 4, target_size[0] * 3 + 38),
                fontFace=FONT_HERSHEY_PLAIN,
                fontScale=0.7,
                color=(255, 0, 255),
                thickness=1,
                lineType=LINE_AA)
    return canvas


if __name__ == "__main__":
    print("Calculating how many different markers are possible.")
    id_collection = set()
    for i in range(4095):
        points_in_inner_ring = 0
        points_in_outer_ring = 0
        for j in range(6):
            if i & 1 << j:
                points_in_inner_ring += 1
            if i & 1 << (j + 6):
                points_in_outer_ring += 1
        if 5 >= points_in_inner_ring >= 2 and 5 >= points_in_inner_ring >= 2:
            # This could be a valid id
            id_collection.add(find_max_id_in_pattern(i))
    print(len(id_collection))
