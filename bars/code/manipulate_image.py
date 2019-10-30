import cv2
import numpy as np
import math
import imutils
from bars.code import image_show
from bars.code import classifier
from matplotlib.patches import Rectangle
import operator
from operator import itemgetter
import random

thr_1=250
thr_2=255


# def fix_image_rotation(img):
#     x_axis, y_axis = find_axes(img, position_restriction=True)
#     y_axis_length = y_axis[0][1] - y_axis[1][1]
#     x_diff = y_axis[0][0] - y_axis[1][0]
#     rotation = -math.degrees(x_diff / y_axis_length)
#     image = rotate_image(img, -rotation)
#     print("rot offset", rotation)
#     return image


def rotate_image(img, degrees):
    image = np.copy(img)
    rotated = imutils.rotate_bound(image, degrees,)
    return rotated


# def skeletonize(img):
#     """ Source: https://gist.github.com/jsheedy/3913ab49d344fac4d02bcc887ba4277d """
#     """ return a skeletonized version of img as a Mat object """
#     img = img.copy() # don't clobber original
#     skel = img.copy()
#
#     skel[:,:] = 0
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#
#     while True:
#         eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
#         temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
#         temp = cv2.subtract(img, temp)
#         skel = cv2.bitwise_or(skel, temp)
#         img[:,:] = eroded[:,:]
#         if cv2.countNonZero(img) == 0:
#             break
#
#     return skel


# def find_axes(img, th_1=thr_1, th_2=thr_2, position_restriction=True):
#     """ Source: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv """
#     """ Locates the longest vertical and horizontal lines """
#
#     copy = np.copy(img)
#     H, W = copy.shape[:2]
#
#     kernel = np.ones((3, 3), np.uint8)
#     dila = cv2.erode(copy, kernel, iterations=1)
#
#     edges = cv2.Canny(dila, th_1, th_2)
#
#     # cv2.imshow("e",edges)
#     # cv2.waitKey(0)
#
#     rho = 1  # distance resolution in pixels of the Hough grid
#     theta = np.pi / 180  # angular resolution in radians of the Hough grid
#     threshold = 15  # minimum number of votes (intersections in Hough grid cell)
#     min_line_length = 50  # minimum number of pixels making up a line
#     max_line_gap = 5 # 20  # maximum gap in pixels between connectable line segments
#
#     # Run Hough on edge detected image
#     # Output "lines" is an array containing endpoints of detected line segments
#     lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                             min_line_length, max_line_gap)
#
#     x_axis = [(0,0),(0,0)]
#     y_axis = [(0,0),(0,0)]
#
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             temp = [(x1,y1),(x2,y2)]
#
#             # compare length (x-axis)
#             if abs(temp[0][0] - temp[1][0]) > abs(x_axis[0][0] - x_axis[1][0]):
#                 if position_restriction and temp[0][1] > H/2:
#                     x_axis[0] = temp[0]
#                     x_axis[1] = temp[1]
#
#                 # if not position_restriction:
#                 #     x_axis[0] = temp[0]
#                 #     x_axis[1] = temp[1]
#
#             # compare height (y-axis)
#             if abs(temp[0][1] - temp[1][1]) > abs(y_axis[0][1] - y_axis[1][1]):
#                 if position_restriction and temp[0][0] < W/2:
#                     y_axis[0] = temp[0]
#                     y_axis[1] = temp[1]
#
#                 # if not position_restriction:
#                 #     y_axis[0] = temp[0]
#                 #     y_axis[1] = temp[1]
#
#             # cv2.line(img, temp[0], temp[1], (255, 0, 255), 2)
#
#     return x_axis, y_axis


def remove_thin_lines(img, th_1=thr_1, th_2=thr_2, correction=7):
    """Isolate bars by removing thin lines."""

    # Threshold image
    thresh = cv2.threshold(img, th_1, th_2, cv2.THRESH_BINARY)[1]

    # get rid of thinner lines
    kernel = np.ones((3, 3), np.uint8)
    isolated = cv2.dilate(thresh, kernel, iterations=3)

    if correction > 0:
        kernel = np.ones((correction, correction), np.uint8)
        isolated = cv2.erode(isolated, kernel, iterations=1)
        return cv2.bitwise_not(isolated)

    if correction < 0:
        kernel = np.ones((-correction, -correction), np.uint8)
        isolated = cv2.dilate(isolated, kernel, iterations=1)
        return cv2.bitwise_not(isolated)

    return cv2.bitwise_not(isolated)


def remove_thick_lines(img, th_1=thr_1, th_2=thr_2, correction=0):
    """Removes bars from graph leaving only thin lines and text"""
    # leaving white filled in shapes, on black background
    bars = remove_thin_lines(img, th_1=th_1, th_2=th_2, correction=correction)

    # overlay white thick lines on image
    no_bars = cv2.bitwise_or(img, bars)

    return no_bars


# def remove_horizontal_lines(img, kernel=np.ones((3, 3), np.uint8), iterations=1):
#     """ return a filtered version of img, without thin lines, as a Mat object """
#     skel = skeletonize(cv2.bitwise_not(img))
#
#     gray_thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)[1]
#     skel_thresh = cv2.threshold(cv2.bitwise_not(skel), 240, 255, cv2.THRESH_BINARY)[1]
#
#     diff_thresh = cv2.bitwise_xor(skel_thresh, gray_thresh)
#
#     opened = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
#
#     return closed


# def remove_grid_lines(img):
#     """ return a filtered version of img, without thin lines, as a Mat object """
#     copy = np.copy(img)
#     blank_image = np.zeros(copy.shape, np.uint8)
#     thresh = cv2.threshold(copy, 254, 255, cv2.THRESH_BINARY)[1]
#     threshold = 0
#     max_line_gap = 5
#     min_line_length = 100
#     edges = cv2.Canny(thresh, 10, 255, apertureSize=3)
#
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
#
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
#
#     no_bars = remove_thin_lines(copy)
#
#     diff = blank_image - cv2.bitwise_not(no_bars)
#     diff2 = copy - diff
#     cv2.imshow("img",img)
#     cv2.imshow("both",diff)
#     cv2.imshow("edges",edges)
#     # cv2.imshow("diff2",diff2)
#     cv2.waitKey()
#
#     return diff


def change_color(img, target_color, new_color, epsilon=5):
    copy = np.copy(img)
    mask = isolate_color(img, target_color, epsilon)

    copy[np.where((mask == [255, 255, 255]).all(axis=2))] = new_color

    return copy


def isolate_color(img, target_color, epsilon=5):

    lower_bound = np.array(target_color - epsilon, dtype="uint16")
    upper_bound = np.array(target_color + epsilon, dtype="uint16")

    # background_mask = image_show.make_three_dim(cv2.bitwise_not(cv2.inRange(img, lower_bound, upper_bound)))
    background_mask = image_show.make_three_dim(cv2.inRange(img, lower_bound, upper_bound))

    return background_mask


def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def find_corners(img, epsilon=3):

    # reduce noise while keeping edges fairly sharp. However, it is very slow compared to most filters.
    # img = cv2.bilateralFilter(img, 11, 17, 17)

    img = cv2.bilateralFilter(img, 7, 17, 17)

    # find (bgr) background color of image
    background_color = np.array(bincount_app(img))

    # change all pixels that are not equal to the background color to black
    mask = isolate_color(img, background_color, 12)

    # convert image to gray-scale from BGR
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((3, 3))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=2)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel, iterations=2)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=2)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel, iterations=1)

    # gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=2)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel, iterations=1)

    # remove noise only leaving medium to thick lines
    bars = remove_thin_lines(gray,)

    # calculate contours of the bars
    contours, hierarchy = cv2.findContours(bars, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(bars, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(bars, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    # contours, hierarchy = cv2.findContours(bars, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    corners = []
    for cnt in contours:

        # find corners
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        for p in approx:
            point = tuple(np.ndarray.tolist(p)[0])
            # cv2.rectangle(img, point, point, (0, 0, 255), 5)
            corners.append(point)

    # cv2.imshow("mask", mask)
    # cv2.imshow("bars", bars)
    # cv2.imshow("orig", img)
    # cv2.imshow("gray", gray)
    # cv2.waitKey()

    for i in range(0, len(corners)):
        for j in range(i+1, len(corners)):
            if abs(corners[i][0] - corners[j][0]) < epsilon and abs(corners[i][1] - corners[j][1]) < epsilon:
                corners.remove(corners[i])
                break

    return corners


def find_vertical_edges(corners):
    corners = sorted(corners, key=operator.itemgetter(0, 1))
    corners_2 = corners.copy()

    # start_point, end_point
    horizontal_edges = []

    epsilon = 3

    # find base axis line(x-axis for vertical bars, u-axis for horizontal bars)
    for i in range(0, len(corners)):
        start_point = corners[i]

        for current_point in corners_2:
            if current_point is not start_point and abs(current_point[1] - start_point[1]) < epsilon:
                if corners_2.__contains__(start_point):
                    horizontal_edges.append((start_point, current_point))
                    corners_2.remove(current_point)
                    if corners_2.__contains__(start_point):
                        corners_2.remove(start_point)
                    break

    return horizontal_edges


def find_x_axis(corners):

    edges = find_vertical_edges(corners)

    # total_edge_length, edge_height, left_most_x, right_most_x
    combined_edges = []

    for line in edges:
        temp = (0, -math.inf, math.inf, -math.inf)
        for l in combined_edges:
            current_height = line[0][1]
            l_height = l[1]

            if current_height == l_height:
                temp = l
                combined_edges.remove(temp)
                break

        temp = (temp[0] + abs(line[0][0] - line[1][0])), line[0][1], \
               min(min(line[0][0], line[1][0]), temp[2]), max(max(line[0][0], line[1][0]), temp[3])
        combined_edges.append(temp)

    base = max(combined_edges, key=itemgetter(0))
    base_line = (base[2], base[1]), (base[3], base[1])

    return base_line


def find_horizontal_pairs(target, points, epsilon, blacklist=None):
    pairs = []
    for point in points:
        if abs(point[1] - target[1]) < epsilon and target is not point and \
                (blacklist is None or blacklist.__contains__(point) is False):
            pairs.append(point)

    return pairs


def find_vertical_pairs(target, points, epsilon, blacklist=None):
    pairs = []
    for point in points:
        if abs(point[0] - target[0]) < epsilon and target is not point and \
                (blacklist is None or blacklist.__contains__(point) is False):
            pairs.append(point)

    return pairs


def find_bar_group(bl, br, corners, bases, corners_used, bars, epsilon, img):
    corners_used.append(bl)

    possible_tl = find_vertical_pairs(bl, corners, epsilon, corners_used)
    for tl in possible_tl:
        corners_used.append(tl)
        possible_tr = find_horizontal_pairs(tl, corners, epsilon, corners_used)
        # possible_tr = sorted(possible_tr, key=operator.itemgetter(0, 1), reverse=False)

        for tr in possible_tr:
            corners_used.append(tr)

            possible_br = find_vertical_pairs(tr, bases, epsilon, corners_used)

            temp_bar = bars.copy()
            x = tl[0]
            y = tl[1]
            w = tr[0] - x
            h = br[1] - y

            # print(x,y,w,h)
            temp_bar.append(Rectangle((x, y), w, h))

            if possible_br.__contains__(br):
                # color = (255, 0, 255)
                # cv2.rectangle(img, bl, bl, (255, 0, 0), 5)
                # cv2.rectangle(img, tl, tl, (0, 255, 0), 5)
                # cv2.rectangle(img, tr, tr, (0, 0, 255), 5)
                # cv2.rectangle(img, br, br, (255, 0, 255), 5)
                # cv2.imshow("ttttt", img)
                # cv2.waitKey()

                return temp_bar
            else:
                return find_bar_group((tr[0], bl[1]), br, corners, bases, corners_used, temp_bar, epsilon, img)

    return []


def find_vertical_bars(all_corners, x_axis, img, epsilon=5):
    bars = []

    # cv2.rectangle(img, x_axis[0], x_axis[1], (0, 255, 255), -1)
    # cv2.imshow('b2', img)
    # cv2.waitKey()

    # all the corners that make up the bars
    corners_used = []

    # find all corners that lie on the x-axis
    base_corners = find_horizontal_pairs(x_axis[0], all_corners, 5)

    # corners that do not lie on the x-axis
    corners = all_corners.copy()
    for base in base_corners:
        corners.remove(base)

    for current_base_index in range(0, len(base_corners) - 1):

        bl = base_corners[current_base_index]
        br = base_corners[current_base_index+1]

        corners_used.append(bl)
        bar_group = find_bar_group(bl, br, corners, base_corners, corners_used, [], epsilon, img)

        for bar in bar_group:
            cv2.rectangle(img, (bar.get_x(), bar.get_y()),
                          (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height()), (0, 255, 255), -1)

            bars.append(bar)
            # cv2.imshow('b2', img)
            # cv2.waitKey()

    return bars, corners_used


def get_vertical_bars(corners, img):

    # sort corners from left to right
    corners = sorted(corners, key=operator.itemgetter(0, 1))
    x_axis = find_x_axis(corners)
    bars, vertical_corners_used = find_vertical_bars(corners, x_axis, img)

    return bars, vertical_corners_used


def find_bars(corners, img):

    vertical_bars, vertical_corners_used = get_vertical_bars(corners, img)
    return vertical_bars, vertical_corners_used


def find_legend(all_corners, img, epsilon=5):
    legend = []
    corners_used = []

    return legend, corners_used


def find_features(img):
    copy = np.copy(img)
    all_corners = find_corners(copy)

    bars, bar_corners = find_bars(all_corners, copy)
    legend, legend_corners = find_legend(all_corners, copy)

    # image_show.show_rectangles(bars, copy, color=(255, 255, 0))
    # image_show.show_rectangles(bars, copy, color=(255, 0, 255))

    return bars, legend

