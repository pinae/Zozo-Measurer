#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from markers import find_marker_ellipses, unskew_point, get_point_id, collect_points, is_small_point
from markers import find_best_10_confidences
import cv2
import sys


def scale_preview(preview_img, max_preview_size=(1800, 960)):
    scale_fac = min(max_preview_size[1] / preview_img.shape[0], max_preview_size[0] / preview_img.shape[1], 1)
    scaled_im = cv2.resize(preview_img, (int(preview_img.shape[1] * scale_fac), int(preview_img.shape[0] * scale_fac)),
                           interpolation=cv2.INTER_CUBIC)
    return scaled_im, scale_fac


def detect_points(img):
    skewed_points, origins, ellipses = find_marker_ellipses(img)
    unskewed_points = [unskew_point(skewed_points[i], origins[i], ellipses[i])
                       for i in range(len(skewed_points))]
    point_ids = []
    confidences = []
    positions = []
    distances = []
    for i in range(len(unskewed_points)):
        positions.append(ellipses[i][0])
        confidence_for_small_point = is_small_point(unskewed_points[i], ellipses[i])
        if confidence_for_small_point > 0.3:
            point_ids.append(0)
            confidences.append(confidence_for_small_point)
        else:
            point_id, confidence = get_point_id(unskewed_points[i], ellipses[i])
            point_ids.append(point_id)
            confidences.append(confidence)
    _, best_10_indexes = find_best_10_confidences(
        [confidences[i] if point_ids[i] > 0 else 0 for i in range(len(confidences))])
    ellipses_size_sum = 0
    for i in best_10_indexes:
        ellipses_size_sum += max(ellipses[i][1])
    avg_big_point_size = ellipses_size_sum / 10
    for i in range(len(unskewed_points)):
        if point_ids[i] == 0:
            distances.append((max(ellipses[i][1]) / avg_big_point_size) / 0.0025)
        else:
            distances.append((max(ellipses[i][1]) / avg_big_point_size) / 0.005)
    raw_data = [{
        "skewed_point": skewed_points[i],
        "unskewed_point": unskewed_points[i],
        "origin": origins[i],
        "ellipse": ellipses[i],
        "confidence": confidences[i],
        "point_id": point_ids[i],
        "point_type": "small_point" if point_ids[i] == 0 else "big_point",
        "position": positions[i],
        "distance": distances[i]
    } for i in range(len(unskewed_points))]
    return point_ids, confidences, positions, distances, raw_data


if __name__ == '__main__':
    args = sys.argv
    input_file_name = args[1]
    im = cv2.imread(input_file_name)
    point_ids, confidences, positions, distances, raw_data = detect_points(im)

    p_coll_img = collect_points((64, 64), raw_data)
    if min(p_coll_img.shape[0:1]) > 0:
        # cv2.imshow('Collected Points', p_coll_img)
        cv2.imwrite("collected_points.png", p_coll_img)

    im_draw = im.copy()
    for i in range(len(point_ids)):
        cv2.ellipse(im_draw, raw_data[i]["ellipse"],
                    color=(0, int(confidences[i] * 255), int((1 - confidences[i]) * 255)),
                    thickness=3)
        cv2.line(im_draw,
                 (int(positions[i][0]-5), int(positions[i][1]-5)),
                 (int(positions[i][0]+5), int(positions[i][1]+5)),
                 color=(255, 0, 0), thickness=3)
        cv2.line(im_draw,
                 (int(positions[i][0] - 5), int(positions[i][1] + 5)),
                 (int(positions[i][0] + 5), int(positions[i][1] - 5)),
                 color=(255, 0, 0), thickness=3)
        if confidences[i] > 0.3:
            cv2.putText(im_draw, str(point_ids[i]),
                        (int(positions[i][0] + raw_data[i]["ellipse"][1][0] * 0.5),
                         int(positions[i][1] - raw_data[i]["ellipse"][1][0] * 0.2)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2,
                        color=(0, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.putText(im_draw, "{:02.2f}".format(distances[i]/100),
                        (int(positions[i][0] + raw_data[i]["ellipse"][1][0] * 0.4),
                         int(positions[i][1] + raw_data[i]["ellipse"][1][0] * 0.8)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2,
                        color=(255, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA)
    cv2.imwrite("point_positions.png", im_draw)
    im_preview, scale_factor = scale_preview(im_draw)
    cv2.imshow('Point positions', im_preview)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
