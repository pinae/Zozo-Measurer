#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from markers import find_marker_ellipses, unskew_point, get_point_id, collect_points
import cv2
import sys


def scale_preview(preview_img, max_preview_size=(1800, 960)):
    scale_fac = min(max_preview_size[1] / preview_img.shape[0], max_preview_size[0] / preview_img.shape[1], 1)
    scaled_im = cv2.resize(preview_img, (int(preview_img.shape[1] * scale_fac), int(preview_img.shape[0] * scale_fac)),
                           interpolation=cv2.INTER_CUBIC)
    return scaled_im, scale_fac


if __name__ == '__main__':
    args = sys.argv
    input_file_name = args[1]
    im = cv2.imread(input_file_name)
    skewed_points, origins, ellipses = find_marker_ellipses(im)
    unskewed_points = [unskew_point(skewed_points[i], origins[i], ellipses[i])
                       for i in range(len(skewed_points))]
    point_ids = [get_point_id(unskewed_points[i], ellipses[i]) for i in range(len(unskewed_points))]
    print(point_ids)

    p_coll_img = collect_points(skewed_points, unskewed_points, (64, 64), point_ids)
    if min(p_coll_img.shape[0:1]) > 0:
        cv2.imshow('Collected Points', p_coll_img)

    im_preview, scale_factor = scale_preview(im)
    # Show window
    #cv2.imshow('Source', im_preview)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
